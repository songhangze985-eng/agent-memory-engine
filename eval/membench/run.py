"""Run a stratified MemBench sample.

Examples:
  python -m eval.membench.run --sample 20
  LLM_API_KEY=... LLM_BASE_URL=... LLM_MODEL=Qwen2.5-7B-Instruct \\
    python -m eval.membench.run --sample 20 --model Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eval.membench.catalog import (
    CATEGORIES,
    DRIVE_URL,
    LICENSE_NOTE,
    PAPER_TABLE5_QWEN_SUBDATASET1,
    PAPER_URL,
    REPO_URL,
)
from eval.membench.items import format_question
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
from memengine.llm import MockLLMClient, build_llm_from_env
from memengine.textutil import estimate_tokens

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


def _eval_item(item: dict, client, mock: bool) -> dict:
    engine = CognitiveMemoryEngine(db_path=":memory:", llm=MockLLMClient() if mock else client)
    write_ms = []
    try:
        for turn in item["turns"]:
            started = time.perf_counter()
            engine.add(turn["text"], timestamp=turn["ts"])
            write_ms.append((time.perf_counter() - started) * 1000)
        question = format_question(item)
        started = time.perf_counter()
        retrieved = engine.search(question)
        read_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        answered = engine.answer(question)
        answer_ms = (time.perf_counter() - started) * 1000
        memory_texts = _retrieved_texts(retrieved)
        transcript = "\n".join(turn["text"] for turn in item["turns"])
        native = answered.get("response") or ""
        same_letter, same_ms, same_tokens = _choose(client, mock, "\n".join(memory_texts), item)
        none_letter, none_ms, none_tokens = _choose(client, mock, "", item)
        full_letter, full_ms, full_tokens = _choose(client, mock, transcript, item)
        steps = extract_steps(memory_texts)
        ingest_tokens = estimate_tokens(transcript)
        return {
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
            "write_ms_mean": mean(write_ms),
            "read_ms": read_ms,
            "answer_ms": answer_ms,
            "choose_ms": same_ms + none_ms + full_ms,
            "ingest_tokens": ingest_tokens,
            "retrieved_tokens": estimate_context_tokens(memory_texts),
            "question_tokens": estimate_tokens(question),
            "prompt_tokens": same_tokens + none_tokens + full_tokens,
            "route": retrieved.get("route"),
        }
    finally:
        engine.store.close()


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
        "",
        "| 系统 | 参与·事实 | 参与·反思 | 观察·事实 | 观察·反思 | 总体准确率 | Recall@10 | 写入 ms/次 | 读取 ms/次 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    labels = {
        "engine_native": "引擎 answer()",
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


def run(sample: int, seed: int, mode: str, model: str | None, data_dir: Path) -> dict:
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
    rows = []
    for index, item in enumerate(items, start=1):
        print(f"[{index}/{len(items)}] {item['category']} {item['subtype']} tid={item['tid']}", flush=True)
        rows.append(_eval_item(item, client, resolved != "live"))
    prompt_tokens = sum(row["prompt_tokens"] + row["question_tokens"] + row["ingest_tokens"] for row in rows)
    note = (
        "本次为 mock 模式：作答头是确定性的词面重叠选择器，引擎 answer() 走离线规则。这些数字不是记忆质量分数。"
        if resolved != "live"
        else f"本次作答调用了 OpenAI 兼容接口，模型 {resolved_model}。引擎的写入和检索仍是本地代码。"
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
        "tokens_estimated": prompt_tokens,
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
    parser.add_argument("--data-dir", default=str(DEFAULT_CACHE), help="Directory of official MemData files. Missing files are downloaded.")
    parser.add_argument("--out", default=str(ROOT / "eval" / "results" / "membench" / "sample_mock.json"))
    args = parser.parse_args()
    if args.sample < 1:
        raise SystemExit("--sample must be >= 1")
    payload = run(args.sample, args.seed, args.mode, args.model, Path(args.data_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(_markdown(payload), encoding="utf-8")
    print(_markdown(payload))


if __name__ == "__main__":
    main()
