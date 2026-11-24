"""LongMemEval-S end-to-end QA + turn-level retrieval + 6-bucket attribution runner."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from eval.evidence_ablation import Turn, sample_rows
from eval.longmemeval.run import category as lme_category
from eval.qa.harness import QAItem, evaluate_group_qa, summarize_qa_rows
from eval.qa.provenance import build_provenance
from agentmem_core.chrono import parse_session_clock as _stamp

LONGMEMEVAL_CATEGORIES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "knowledge-update",
    "temporal-reasoning",
    "multi-session",
    "abstention",
]


def parse_longmemeval_qa(row: dict) -> tuple[list[Turn], QAItem]:
    ordered = sorted(
        enumerate(zip(row["haystack_dates"], row["haystack_session_ids"], row["haystack_sessions"], strict=True)),
        key=lambda item: (_stamp(item[1][0]), item[0]),
    )
    turns: list[Turn] = []
    for ordinal, (date, session_id, entries) in ordered:
        stamp = _stamp(date)
        for index, item in enumerate(entries):
            role = item["role"]
            speaker = item.get("speaker") or role
            turns.append(
                Turn(
                    f"{session_id}:{index}",
                    session_id,
                    role,
                    f"{speaker}: {item['content']}",
                    stamp + index / 1000,
                    has_answer=item.get("has_answer") is True,
                    ingest_session=f"session-{ordinal}",
                    mention_time=stamp,
                    speaker=speaker,
                )
            )
    bucket = lme_category(row)
    sessions = frozenset(row.get("answer_session_ids") or ())
    evidence = frozenset(t.ref for t in turns if t.has_answer)
    latest, older = None, frozenset()
    if bucket == "knowledge-update":
        known_sessions = set(row["haystack_session_ids"])
        pairs = {
            (name[:-2] + "_1", name)
            for name in sessions & known_sessions
            if name.endswith("_2") and name[:-2] + "_1" in known_sessions
        }
        if len(pairs) == 1:
            previous, latest = pairs.pop()
            if {previous, latest} <= {t.session for t in turns if t.has_answer}:
                older = frozenset({previous})
            else:
                latest = None
    qa = QAItem(
        id=str(row["question_id"]),
        category=bucket,
        question=str(row["question"]),
        gold_answer=str(row.get("answer") or ""),
        timestamp=_stamp(row["question_date"]),
        question_date=str(row.get("question_date") or ""),
        evidence_turns=evidence,
        evidence_sessions=sessions,
        unanswerable=(bucket == "abstention"),
        older_sessions=older,
        latest_session=latest,
    )
    return turns, qa


def _worker(args):
    turns, qa, engine_kind, llm_model, judge_model, consolidate, token_budget, recall_config = args
    return evaluate_group_qa(
        turns,
        [qa],
        engine_kind=engine_kind,
        llm_model=llm_model,
        judge_model=judge_model,
        consolidate=consolidate,
        token_budget=token_budget,
        recall_config=recall_config,
    )


def run_longmemeval_qa(
    dataset_path: Path,
    *,
    engine_kind: str,
    sample: int | None = None,
    seed: int = 42,
    workers: int = 4,
    llm_model: str | None = None,
    judge_model: str | None = None,
    consolidate: bool = False,
    token_budget: int = 2048,
    recall_config=None,
) -> dict[str, Any]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    selected = sample_rows(data, sample, seed, lme_category)
    tasks = []
    for row in selected:
        turns, qa = parse_longmemeval_qa(row)
        tasks.append((turns, qa, engine_kind, llm_model, judge_model, consolidate, token_budget, recall_config))

    if workers <= 1:
        nested = [_worker(t) for t in tasks]
    else:
        with ProcessPoolExecutor(workers) as pool:
            nested = list(pool.map(_worker, tasks))
    items = [r for batch in nested for r in batch]

    granularity = (
        "turn-level (session_md projected 1-turn/hit; session-level supplementary)"
        if engine_kind == "v2_session_md"
        else "turn-level (primary); session-level supplementary"
    )
    prov = build_provenance(
        engine=engine_kind,
        embedder="HashedTfidfEmbedder",
        reranker="CalibratedLocalReranker" if engine_kind != "v2_session_md" and engine_kind != "v2_turn_md" else "VFS_W_verified_x_W_recency",
        llm=f"llm:{llm_model}" if llm_model else "MockLLMClient/DeterministicReader",
        judge=f"llm-judge:{judge_model}" if judge_model else "deterministic-v1",
        protocol="LongMemEval-S end-to-end QA + turn-level retrieval + 6-bucket attribution",
        granularity=granularity,
        seed=seed,
        dataset=f"LongMemEval-S (n={len(items)})",
        dataset_path=dataset_path,
        extra={"consolidate": consolidate, "token_budget": token_budget},
    )
    return {
        "benchmark": "LongMemEval-S",
        "provenance": prov,
        "summary": summarize_qa_rows(items, categories=LONGMEMEVAL_CATEGORIES),
        "items": items,
    }
