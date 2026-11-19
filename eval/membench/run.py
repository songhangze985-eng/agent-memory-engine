"""Run a stratified MemBench sample.

Examples:
  python -m eval.membench.run --sample 20
  LLM_API_KEY=... LLM_BASE_URL=... LLM_MODEL=Qwen2.5-7B-Instruct \\
    python -m eval.membench.run --sample 20 --model Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path

from eval.membench.catalog import (
    CATEGORIES,
    DRIVE_URL,
    LICENSE_NOTE,
    PAPER_TABLE5_QWEN_SUBDATASET1,
    PAPER_URL,
    REPO_URL,
)
from eval.membench.items import format_question, parse_timestamp, question_stem
from eval.mcq import end_with_choice
from eval.membench.load import load_sample
from eval.membench.score import (
    accuracy,
    capacity_bins,
    choice_correct,
    estimate_context_tokens,
    extract_steps,
    lexical_choice,
    mean,
    recall_at_10,
)
from memengine.engine import CognitiveMemoryEngine
from memengine.llm import FatalLLMError, MockLLMClient, build_llm_from_env
from memengine.textutil import estimate_tokens, step_ids_in
from memengine.vector import HashedTfidfEmbedder

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = Path(__file__).resolve().parent / ".cache" / "MemData"

_CHOICE_PROMPT = """Answer the question using only the memory context below.
Memory:
{context}
Question: (current time is {q_time}) {question}
Choices:
{choices}
Reply with only one letter: A, B, C, or D.
"""


def _choice_block(choices: dict) -> str:
    return "\n".join(f"{letter}. {text}" for letter, text in choices.items())


def _choose(client, mock: bool, context: str, item: dict) -> tuple[str | None, float, int]:
    prompt = _CHOICE_PROMPT.format(
        context=context or "(none)",
        q_time=item["q_time"],
        question=item["question"],
        choices=_choice_block(item["choices"]),
    )
    tokens = estimate_tokens(prompt)
    started = time.perf_counter()
    if mock:
        letter = lexical_choice(context, item["choices"])
        return letter, (time.perf_counter() - started) * 1000, tokens
    raw = client.complete(prompt)
    from eval.membench.score import parse_choice

    return parse_choice(raw), (time.perf_counter() - started) * 1000, tokens


def _retrieved_texts(retrieved: dict) -> list[str]:
    texts = []
    for fact in list(retrieved.get("hits") or []) + list(retrieved.get("outer") or []) + list(retrieved.get("inner") or []):
        text = fact.get("text") or ""
        if text and text not in texts:
            texts.append(text)
    return texts


def _eval_item(item: dict, client, mock: bool, artifacts_dir: Path | None = None) -> dict:
    engine = CognitiveMemoryEngine(db_path=":memory:", llm=MockLLMClient() if mock else client,
                                   vector=HashedTfidfEmbedder() if mock else None)
    write_ms = []
    usage_ctx = (
        client.track_usage()
        if (not mock and client is not None and hasattr(client, "track_usage"))
        else nullcontext({})
    )
    try:
        with usage_ctx as usage_delta:
            return _eval_item_body(item, engine, client, mock, write_ms, usage_delta or {}, artifacts_dir)
    finally:
        engine.store.close()


def _eval_item_body(item: dict, engine, client, mock: bool, write_ms: list, usage_delta: dict, artifacts_dir: Path | None = None) -> dict:
        write_actions: dict[str, int] = {}
        skipped_turns: list[dict] = []
        for turn in item["turns"]:
            started = time.perf_counter()
            added = engine.add(turn["text"], timestamp=turn["ts"])
            write_ms.append((time.perf_counter() - started) * 1000)
            ops = list(added.get("operations") or [])
            if not ops:
                key = "NO_FACT" if not added.get("facts") else "NO_OP_RETURNED"
                write_actions[key] = write_actions.get(key, 0) + 1
                skipped_turns.append({"reason": key, "step": turn.get("step"), "text": (turn.get("text") or "")[:500]})
            noop_ops = [op for op in ops if str(op.get("action") or "") == "NOOP"]
            if noop_ops:
                write_actions["NOOP"] = write_actions.get("NOOP", 0) + len(noop_ops)
                skipped_turns.append({"reason": "NOOP", "step": turn.get("step"), "text": (turn.get("text") or "")[:500], "ops": noop_ops})
            for op in ops:
                action = str(op.get("action") or "NOOP")
                if action == "NOOP":
                    continue
                write_actions[action] = write_actions.get(action, 0) + 1
        stem = question_stem(item)
        question = format_question(item)
        started = time.perf_counter()
        query_time = parse_timestamp(item.get("q_time"), max((turn["ts"] for turn in item["turns"]), default=0) + 1)
        retrieved = engine.retrieve(stem, timestamp=query_time)
        read_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        # Reuse the same retrieval so Recall@10 / same-head score the memory answer() saw.
        answered = engine.answer(stem, retrieved=retrieved)
        answer_ms = (time.perf_counter() - started) * 1000
        memory_texts = _retrieved_texts(retrieved)
        transcript = "\n".join(turn["text"] for turn in item["turns"])
        native = end_with_choice(question, answered.get("response") or "")
        same_letter, same_ms, same_tokens = _choose(client, mock, "\n".join(memory_texts), item)
        none_letter, none_ms, none_tokens = _choose(client, mock, "", item)
        full_letter, full_ms, full_tokens = _choose(client, mock, transcript, item)
        steps = extract_steps(memory_texts)
        ingest_tokens = estimate_tokens(transcript)
        snapshot = engine.store.active_snapshot()
        notebook_steps: list[int] = []
        notebook_preview: list[str] = []
        for block in snapshot:
            for line in block.get("lines") or []:
                plain = line.get("plain") or line.get("text") or ""
                notebook_steps.extend(step_ids_in(plain))
                notebook_steps.extend(int(step) for step in (line.get("source_step_ids") or []) if step is not None)
                if len(notebook_preview) < 16:
                    notebook_preview.append(plain[:200])
        evidence = [int(step) for step in (item.get("evidence_steps") or [])]
        in_notebook = sorted({step for step in notebook_steps if step in set(evidence)})
        row = {
            "tid": item["tid"],
            "category": item["category"],
            "subtype": item["subtype"],
            "scenario": item["scenario"],
            "n_turns": len(item["turns"]),
            "ground_truth": item["ground_truth"],
            "native_response": native[:240],
            "native_correct": choice_correct(native, item["ground_truth"]),
            "same_head_choice": same_letter,
            "same_head_correct": choice_correct(same_letter, item["ground_truth"]),
            "no_memory_choice": none_letter,
            "no_memory_correct": choice_correct(none_letter, item["ground_truth"]),
            "full_context_choice": full_letter,
            "full_context_correct": choice_correct(full_letter, item["ground_truth"]),
            "recall_at_10": recall_at_10(steps, item["evidence_steps"]),
            "evidence_steps": item["evidence_steps"],
            "retrieved_steps": steps[:10],
            "retrieved_preview": [text[:200] for text in memory_texts[:10]],
            "write_ms_mean": mean(write_ms),
            "read_ms": read_ms,
            "answer_ms": answer_ms,
            "choose_ms": same_ms + none_ms + full_ms,
            "ingest_tokens": ingest_tokens,
            "retrieved_tokens": estimate_context_tokens(memory_texts),
            "question_tokens": estimate_tokens(question),
            "prompt_tokens": same_tokens + none_tokens + full_tokens,
            "llm_prompt_tokens": int(usage_delta.get("prompt_tokens") or 0),
            "llm_completion_tokens": int(usage_delta.get("completion_tokens") or 0),
            "route": retrieved.get("route"),
            "matched_entities": list(retrieved.get("matched_entities") or []),
            "entity_fallback": bool(retrieved.get("entity_fallback")),
            "n_outer": len(retrieved.get("outer") or []),
            "n_inner": len(retrieved.get("inner") or []),
            "n_hits": len(retrieved.get("hits") or []),
            "write_actions": write_actions,
            "notebook_n_lines": sum(len(block.get("lines") or []) for block in snapshot),
            "notebook_steps": sorted(set(notebook_steps)),
            "evidence_in_notebook": in_notebook,
            "evidence_missing_from_notebook": [step for step in evidence if step not in set(in_notebook)],
            "notebook_preview": notebook_preview,
            "empty_retrieval": bool(retrieved.get("empty_retrieval")),
            "parsed_query": retrieved.get("parsed_query") or {},
        }
        if artifacts_dir is not None:
            _write_artifact(artifacts_dir, item, engine, retrieved, skipped_turns, memory_texts)
        return row



def _write_artifact(artifacts_dir: Path, item: dict, engine, retrieved: dict, skipped_turns: list, memory_texts: list[str]) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    snapshot = engine.store.active_snapshot()
    z_index = {}
    for key, facts in (engine.store.z_index() or {}).items():
        z_index[str(key)] = [
            {
                "id": fact.get("id"),
                "text": fact.get("text"),
                "zkeys": fact.get("zkeys") or fact.get("z_axis_keys") or [],
                "source_step_ids": fact.get("source_step_ids") or [],
            }
            for fact in facts
        ]
    payload = {
        "tid": item.get("tid"),
        "category": item.get("category"),
        "subtype": item.get("subtype"),
        "question_stem": question_stem(item),
        "empty_retrieval": bool(retrieved.get("empty_retrieval")),
        "parsed_query": retrieved.get("parsed_query") or {},
        "retrieved_top10": [
            {
                "id": fact.get("id"),
                "text": fact.get("text"),
                "zone": fact.get("zone"),
                "zkeys": fact.get("zkeys") or fact.get("z_axis_keys") or [],
                "source_step_ids": fact.get("source_step_ids") or [],
            }
            for fact in list(retrieved.get("hits") or [])
            + list(retrieved.get("outer") or [])
            + list(retrieved.get("inner") or [])
        ],
        "retrieved_preview": [text[:400] for text in memory_texts[:10]],
        "skipped_or_noop_turns": skipped_turns,
        "notebook_2d": snapshot,
        "z_index": z_index,
        "has_facts": engine.has_facts(),
    }
    path = artifacts_dir / f"{item.get('category')}_{item.get('tid')}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _category_table(rows: list[dict]) -> dict:
    systems = {
        "engine_native": "native_correct",
        "engine_same_head": "same_head_correct",
        "no_memory": "no_memory_correct",
        "full_context": "full_context_correct",
    }
    table = {}
    for name, field in systems.items():
        per = {}
        flags = []
        recalls = []
        writes = []
        reads = []
        for category in CATEGORIES:
            subset = [row for row in rows if row["category"] == category]
            cat_flags = [bool(row[field]) for row in subset]
            per[category] = {"accuracy": accuracy(cat_flags), "n": len(subset)}
            flags.extend(cat_flags)
        for row in rows:
            if row.get("recall_at_10") is not None and name == "engine_same_head":
                recalls.append(row["recall_at_10"])
            if row.get("write_ms_mean") is not None and name.startswith("engine"):
                writes.append(row["write_ms_mean"])
            if row.get("read_ms") is not None and name.startswith("engine"):
                reads.append(row["read_ms"])
        table[name] = {
            "accuracy": accuracy(flags),
            "n": len(flags),
            "by_category": per,
            "recall_at_10": mean(recalls) if name == "engine_same_head" else None,
            "write_ms_mean": mean(writes) if name.startswith("engine") else None,
            "read_ms_mean": mean(reads) if name.startswith("engine") else None,
        }
    return table


def _markdown(payload: dict) -> str:
    def pct(value):
        if value is None:
            return "—"
        return f"{value * 100:.1f}%"

    def num(value):
        if value is None:
            return "—"
        return f"{value:.3f}"

    lines = [
        "# MemBench 分层抽样实测",
        "",
        payload["note"],
        "",
        f"- 数据：{payload['data_source']} ，许可证说明：{payload['license_note']}",
        f"- 划分：官方 MemData 四类（participation/observation × factual/reflective），不是论文 Sub-dataset 1/2。",
        f"- 抽样：每类最多 {payload['sample_per_category']} 条，种子 {payload['seed']}，实际 {payload['n_items']} 条。",
        f"- 模式：`{payload['mode']}`，模型 `{payload['model']}`。",
        f"- 墙钟时间：{payload.get('wall_time_s') or 0:.1f}s。LLM token：prompt {payload.get('llm_prompt_tokens') if payload.get('llm_prompt_tokens') is not None else '—'}，completion {payload.get('llm_completion_tokens') if payload.get('llm_completion_tokens') is not None else '—'}。",
        "",
        "| 系统 | 参与·事实 | 参与·反思 | 观察·事实 | 观察·反思 | 总体准确率 | Recall@10 | 写入 ms/次 | 读取 ms/次 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    labels = {
        "engine_native": "引擎自然语言回答 + 选项适配",
        "engine_same_head": "引擎检索 + 同一作答头",
        "no_memory": "无记忆基线",
        "full_context": "全上下文基线",
    }
    order = ["engine_native", "engine_same_head", "no_memory", "full_context"]
    cats = [
        "participation_factual",
        "participation_reflective",
        "observation_factual",
        "observation_reflective",
    ]
    for key in order:
        row = payload["systems"][key]
        cells = [pct(row["by_category"][cat]["accuracy"]) for cat in cats]
        lines.append(
            "| {name} | {c0} | {c1} | {c2} | {c3} | {acc} | {recall} | {wt} | {rt} |".format(
                name=labels[key],
                c0=cells[0],
                c1=cells[1],
                c2=cells[2],
                c3=cells[3],
                acc=pct(row["accuracy"]),
                recall=pct(row["recall_at_10"]),
                wt=num(row["write_ms_mean"]),
                rt=num(row["read_ms_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "准确率：抽出的选项字母与 `ground_truth` 相同。Recall@10 只对检索结果计算：前 10 个命中步骤里，证据步骤被覆盖的比例。",
            "写入时间是 `add` 的平均墙钟时间，读取时间是 `search` 的平均墙钟时间，对应论文里记忆写入 / 检索的效率，不含作答模型时间。",
            "容量分箱（同一作答头，按写入 token 从少到多分成三段）见 JSON 的 `capacity_bins`。",
            "token 数为字符估计（中文 1 字约 1 token，其余约 4 字符 1 token）。本次客户端没有返回计费，`cost_usd` 为空。",
            "",
            "论文 Table 5（Qwen2.5-7B-Instruct，Sub-dataset 1）的报告值写在 JSON 的 `paper_reported_not_this_run`，不是上表。",
        ]
    )
    return "\n".join(lines) + "\n"


def _item_key(item: dict) -> str:
    return f"{item.get('category')}|{item.get('subtype')}|{item.get('scenario')}|{item.get('tid')}"


def _checkpoint_meta(sample: int, seed: int, mode: str, model: str | None) -> dict:
    return {"_checkpoint_meta": True, "contract_version": 2, "sample": sample, "seed": seed, "mode": mode, "model": model}


def _load_checkpoint(path: Path, meta: dict | None) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("_checkpoint_meta"):
                if meta and any(obj.get(key) != meta.get(key) for key in ("sample", "seed", "mode", "model")):
                    bak = path.with_name(path.name + ".bak")
                    path.replace(bak)
                    print(f"Checkpoint does not match this run, moved aside to {bak}", flush=True)
                    return {}
                continue
            done[_item_key(obj)] = obj
    return done


def _append_checkpoint(path: Path, row: dict, lock: threading.Lock) -> None:
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()


def _init_checkpoint(path: Path, meta: dict) -> dict[str, dict]:
    done = _load_checkpoint(path, meta)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8")
    return done


def _evaluate_all(items, client, mock: bool, concurrency: int, checkpoint_path: Path | None = None, checkpoint_meta: dict | None = None, artifacts_dir: Path | None = None) -> list[dict]:
    rows: list[dict | None] = [None] * len(items)
    lock = threading.Lock()
    done = _init_checkpoint(checkpoint_path, checkpoint_meta or {}) if checkpoint_path else {}
    pending: list[tuple[int, dict]] = []
    for index, item in enumerate(items):
        existing = done.get(_item_key(item))
        if existing is not None:
            rows[index] = existing
        else:
            pending.append((index, item))
    finished = len(items) - len(pending)
    if done:
        print(f"Resuming: {finished}/{len(items)} already finished, {len(pending)} remaining.", flush=True)
    fatal: list[FatalLLMError] = []
    other_errors: list[BaseException] = []

    def record(index: int, row: dict) -> None:
        nonlocal finished
        rows[index] = row
        if checkpoint_path is not None:
            _append_checkpoint(checkpoint_path, row, lock)
        finished += 1
        item = items[index]
        print(f"[{finished}/{len(items)}] {item['category']} {item['subtype']} tid={item['tid']}", flush=True)

    def consume_result(index: int, fn) -> None:
        try:
            record(index, fn())
        except FatalLLMError as exc:
            fatal.append(exc)
            print(f"Stopping: {exc}", flush=True)
        except CancelledError:
            return
        except Exception as exc:
            other_errors.append(exc)
            item = items[index]
            print(f"Item failed: {item['category']} tid={item['tid']}: {type(exc).__name__}", flush=True)

    if not pending:
        return rows  # type: ignore[return-value]

    if concurrency <= 1:
        for index, item in pending:
            if fatal:
                break
            consume_result(index, lambda item=item: _eval_item(item, client, mock, artifacts_dir))
    else:
        workers = min(concurrency, len(pending))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_eval_item, item, client, mock, artifacts_dir): index for index, item in pending
            }
            for future in as_completed(futures):
                index = futures[future]
                consume_result(index, future.result)
                if fatal:
                    for pending_future in futures:
                        pending_future.cancel()

    if fatal:
        kept = f" Checkpoint kept at {checkpoint_path}." if checkpoint_path else ""
        raise SystemExit(f"{fatal[0]} Finished {finished}/{len(items)}.{kept} Resume after fixing auth or balance.")
    if other_errors or any(row is None for row in rows):
        kept = f" Checkpoint kept at {checkpoint_path}." if checkpoint_path else ""
        detail = f"{type(other_errors[0]).__name__}: {other_errors[0]}" if other_errors else "incomplete results"
        raise SystemExit(f"Run incomplete ({detail}). Finished {finished}/{len(items)}.{kept}")
    return rows  # type: ignore[return-value]


def run(sample: int, seed: int, mode: str, model: str | None, data_dir: Path, concurrency: int = 1, checkpoint_path: Path | None = None, only_tids: list[int] | None = None, only_categories: list[str] | None = None, fresh: bool = False, artifacts_dir: Path | None = None) -> dict:
    if mode == "mock":
        client, resolved, resolved_model = MockLLMClient(), "mock", None
    elif mode == "live":
        client, resolved, resolved_model = build_llm_from_env()
        if resolved != "live":
            raise SystemExit("没有 LLM API key，不能用 live 模式。")
    else:
        client, resolved, resolved_model = build_llm_from_env()
    if model and resolved == "live":
        client.model = model
        resolved_model = model
    items = load_sample(sample, seed, data_dir)
    if only_categories:
        wanted_cats = set(only_categories)
        items = [item for item in items if item.get("category") in wanted_cats]
        missing_cats = wanted_cats - {item.get("category") for item in items}
        if missing_cats:
            raise SystemExit(f"--only-categories not in this sample/seed: {sorted(missing_cats)}")
    if only_tids:
        wanted = set(only_tids)
        items = [item for item in items if int(item["tid"]) in wanted]
        missing = wanted - {int(item["tid"]) for item in items}
        if missing:
            raise SystemExit(f"--only-tids not in this sample/seed: {sorted(missing)}")
    if fresh and checkpoint_path is not None and checkpoint_path.exists():
        bak = checkpoint_path.with_name(checkpoint_path.name + ".freshbak")
        checkpoint_path.replace(bak)
        print(f"--fresh: moved previous checkpoint to {bak}", flush=True)
    meta = _checkpoint_meta(sample, seed, resolved, resolved_model)
    if only_tids:
        meta = {**meta, "only_tids": sorted({int(tid) for tid in only_tids})}
    started = time.perf_counter()
    rows = _evaluate_all(items, client, resolved != "live", concurrency, checkpoint_path, meta, artifacts_dir)
    wall_s = time.perf_counter() - started
    prompt_tokens = sum(row["prompt_tokens"] + row["question_tokens"] + row["ingest_tokens"] for row in rows)
    usage_prompt = sum(int(row.get("llm_prompt_tokens") or 0) for row in rows)
    usage_completion = sum(int(row.get("llm_completion_tokens") or 0) for row in rows)
    if resolved == "live" and usage_prompt == 0 and usage_completion == 0:
        usage = getattr(client, "usage_totals", None) or {}
        usage_prompt = int(usage.get("prompt_tokens") or 0)
        usage_completion = int(usage.get("completion_tokens") or 0)
    note = (
        "本次为 mock 模式：作答头用确定性词面重叠，引擎答复由评测适配器映射到选项。这些数字不是记忆质量分数。"
        if resolved != "live"
        else (
            f"本次引擎、无记忆基线和全上下文基线使用同一个接口，模型 {resolved_model}，"
            f"地址 {getattr(client, 'base_url', '')}。费用未按 DeepSeek 价目折算，cost_usd 为空。"
        )
    )
    return {
        "benchmark": "MemBench",
        "paper": PAPER_URL,
        "data_source": REPO_URL + "/tree/main/MemData",
        "data_mirror": DRIVE_URL,
        "license_note": LICENSE_NOTE,
        "split": "MemData category files (participation/observation x factual/reflective). Not the paper's noise-extended Sub-dataset 1 or 2; data2test is not in the git tree.",
        "sample_per_category": sample,
        "seed": seed,
        "n_items": len(rows),
        "mode": resolved,
        "model": resolved_model,
        "requested_model": model,
        "meaningful_as_quality": resolved == "live",
        "note": note,
        "concurrency": concurrency,
        "llm_base_url": getattr(client, "base_url", None),
        "wall_time_s": wall_s,
        "tokens_estimated": prompt_tokens,
        "llm_prompt_tokens": usage_prompt if resolved == "live" else None,
        "llm_completion_tokens": usage_completion if resolved == "live" else None,
        "cost_usd": None,
        "systems": _category_table(rows),
        "capacity_bins": capacity_bins(rows),
        "paper_reported_not_this_run": PAPER_TABLE5_QWEN_SUBDATASET1,
        "items": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified MemBench sample against CognitiveMemoryEngine.")
    parser.add_argument("--sample", type=int, default=20, help="Items per top-level category (participation/observation x factual/reflective).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["auto", "mock", "live"], default="auto")
    parser.add_argument("--model", default=None, help="Model name for a live OpenAI-compatible endpoint. Ignored in mock mode.")
    parser.add_argument("--concurrency", type=int, default=1, help="How many trajectories to score at once. Live DeepSeek runs can use 4.")
    parser.add_argument("--data-dir", default=str(DEFAULT_CACHE), help="Directory of official MemData files. Missing files are downloaded.")
    parser.add_argument("--out", default=str(ROOT / "eval" / "results" / "membench" / "sample_mock.json"))
    parser.add_argument("--only-tids", default="", help="Comma-separated trajectory ids to keep from the stratified sample.")
    parser.add_argument("--only-categories", default="", help="Comma-separated MemBench category names to keep (e.g. participation_factual,participation_reflective).")
    parser.add_argument("--fresh", action="store_true", help="Ignore an existing checkpoint for --out and start a new run.")
    parser.add_argument("--artifacts-dir", default="", help="Directory for per-item diagnostic snapshots. Default: <out-parent>/artifacts")
    args = parser.parse_args()
    if args.sample < 1:
        raise SystemExit("--sample must be >= 1")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    only_tids = [int(part.strip()) for part in args.only_tids.split(",") if part.strip()]
    only_categories = [part.strip() for part in args.only_categories.split(",") if part.strip()]
    out = Path(args.out)
    checkpoint = out.with_suffix(".jsonl")
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else (out.parent / "artifacts")
    payload = run(args.sample, args.seed, args.mode, args.model, Path(args.data_dir), args.concurrency, checkpoint, only_tids or None, only_categories or None, args.fresh, artifacts_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(_markdown(payload), encoding="utf-8")
    print(_markdown(payload))


if __name__ == "__main__":
    main()
