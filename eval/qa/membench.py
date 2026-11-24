"""MemBench end-to-end MCQ QA + step-level retrieval + 6-bucket attribution runner."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from eval.evidence_ablation import Turn
from eval.membench.catalog import CATEGORIES as MEMBENCH_CATEGORIES
from eval.membench.items import parse_timestamp, question_stem
from eval.membench_static import sample_items as sample_membench_items
from eval.qa.harness import QAItem, evaluate_group_qa, summarize_qa_rows
from eval.qa.provenance import build_provenance


def _parse_membench_item(item: dict) -> tuple[list[Turn], QAItem]:
    turns: list[Turn] = []
    for idx, t in enumerate(item["turns"]):
        step_id = int(t.get("step", idx))
        ts = float(t.get("ts", idx))
        turns.append(
            Turn(
                ref=str(step_id),
                session="s0",
                role="user",
                text=str(t["text"]),
                timestamp=ts,
                has_answer=step_id in set(item.get("evidence_steps") or ()),
                ingest_session="s0",
                mention_time=ts,
                speaker="user",
            )
        )
    stamp = parse_timestamp(item.get("q_time"), max((t.timestamp for t in turns), default=0.0) + 1.0)
    ev_turns = frozenset(str(int(s)) for s in (item.get("evidence_steps") or ()))
    choices = dict(item.get("choices") or {})
    gt_letter = str(item.get("ground_truth") or "").strip().upper()[:1]
    gold_text = choices.get(gt_letter) or str(item.get("answer_text") or "")
    qa = QAItem(
        id=f"{item['category']}:{item['subtype']}:{item.get('tid')}",
        category=str(item["category"]),
        question=question_stem(item),
        gold_answer=gold_text,
        timestamp=stamp,
        question_date=str(item.get("q_time") or ""),
        evidence_turns=ev_turns,
        evidence_sessions=frozenset({"s0"}) if ev_turns else frozenset(),
        unanswerable=False,
        choices=choices,
        mcq_gold=gt_letter or None,
    )
    return turns, qa


def _worker(args):
    turns, qa, engine_kind, llm_model, judge_model, consolidate, token_budget, recall_config = args
    # MemBench trajectories are step-level trajectories; v2_session_md maps to step-level turns here
    effective_engine = "v2_turn_md" if engine_kind == "v2_session_md" else engine_kind
    return evaluate_group_qa(
        turns,
        [qa],
        engine_kind=effective_engine,
        llm_model=llm_model,
        judge_model=judge_model,
        consolidate=consolidate,
        token_budget=token_budget,
        recall_config=recall_config,
    )


def run_membench_qa(
    data_dir: Path,
    *,
    engine_kind: str,
    per_category: int = 100,
    seed: int = 42,
    workers: int = 4,
    llm_model: str | None = None,
    judge_model: str | None = None,
    consolidate: bool = False,
    token_budget: int = 2048,
    recall_config=None,
) -> dict[str, Any]:
    raw_items, missing = sample_membench_items(data_dir, seed, per_category)
    tasks = []
    for item in raw_items:
        turns, qa = _parse_membench_item(item)
        tasks.append((turns, qa, engine_kind, llm_model, judge_model, consolidate, token_budget, recall_config))

    if workers <= 1:
        nested = [_worker(t) for t in tasks]
    else:
        with ProcessPoolExecutor(workers) as pool:
            nested = list(pool.map(_worker, tasks))
    items = [r for batch in nested for r in batch]

    prov = build_provenance(
        engine=engine_kind,
        embedder="HashedTfidfEmbedder",
        reranker="CalibratedLocalReranker" if engine_kind not in {"v2_session_md", "v2_turn_md"} else "VFS_W_verified_x_W_recency",
        llm=f"llm:{llm_model}" if llm_model else "MockLLMClient/DeterministicReader",
        judge="exact-mcq-match",
        protocol="MemBench end-to-end MCQ QA + step-level retrieval + 6-bucket attribution",
        granularity="step-level (identical across all engines)",
        seed=seed,
        dataset=f"MemBench (n={len(items)}, {per_category}/cat)",
        dataset_path=data_dir,
        extra={"consolidate": consolidate, "token_budget": token_budget, "uncached_subtypes": missing},
    )
    return {
        "benchmark": "MemBench",
        "provenance": prov,
        "summary": summarize_qa_rows(items, categories=list(MEMBENCH_CATEGORIES)),
        "items": items,
    }
