"""LoCoMo end-to-end QA + turn-level retrieval + 6-bucket attribution runner."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from eval.evidence_ablation import sample_rows
from eval.locomo.run import CATEGORIES as LOCOMO_CAT_MAP, parse_turns
from eval.qa.harness import QAItem, evaluate_group_qa, summarize_qa_rows
from eval.qa.provenance import build_provenance
from memengine.chrono import parse_session_clock

LOCOMO_CATEGORIES = [
    "1 multi-hop",
    "2 temporal",
    "3 open-domain",
    "4 single-hop",
    "5 adversarial",
]


def _worker(args):
    turns, qas, engine_kind, llm_model, judge_model, consolidate, token_budget, recall_config = args
    return evaluate_group_qa(
        turns,
        qas,
        engine_kind=engine_kind,
        llm_model=llm_model,
        judge_model=judge_model,
        consolidate=consolidate,
        token_budget=token_budget,
        recall_config=recall_config,
    )


def run_locomo_qa(
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
    pool_rows = [
        (ordinal, index, item)
        for ordinal, dialog in enumerate(data)
        for index, item in enumerate(dialog["qa"])
    ]
    selected = sample_rows(pool_rows, sample, seed, lambda row: LOCOMO_CAT_MAP[int(row[2]["category"])])
    by_dialog: dict[int, list[tuple[int, dict]]] = {}
    for ordinal, index, item in selected:
        by_dialog.setdefault(ordinal, []).append((index, item))

    tasks = []
    for ordinal, chosen in sorted(by_dialog.items()):
        dialog = data[ordinal]
        turns = parse_turns(dialog["conversation"])
        turn_to_session = {t.ref: t.session for t in turns}
        stamp = max(t.timestamp for t in turns) + 1.0
        qas: list[QAItem] = []
        for index, item in chosen:
            cat_id = int(item["category"])
            cat = LOCOMO_CAT_MAP[cat_id]
            date = parse_session_clock(item.get("question_date"))
            ev_turns = frozenset(map(str, item.get("evidence") or ()))
            ev_sessions = frozenset(turn_to_session[r] for r in ev_turns if r in turn_to_session)
            gold = str(item.get("answer") or item.get("adversarial_answer") or "")
            qas.append(
                QAItem(
                    id=f"{dialog['sample_id']}:{index}",
                    category=cat,
                    question=str(item["question"]),
                    gold_answer=gold,
                    timestamp=stamp if date is None else date,
                    question_date=str(item.get("question_date") or ""),
                    evidence_turns=ev_turns,
                    evidence_sessions=ev_sessions,
                    unanswerable=(cat_id == 5),
                )
            )
        tasks.append((turns, qas, engine_kind, llm_model, judge_model, consolidate, token_budget, recall_config))

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
        reranker="CalibratedLocalReranker" if engine_kind not in {"v2_session_md", "v2_turn_md"} else "VFS_W_verified_x_W_recency",
        llm=f"llm:{llm_model}" if llm_model else "MockLLMClient/DeterministicReader",
        judge=f"llm-judge:{judge_model}" if judge_model else "deterministic-v1",
        protocol="LoCoMo end-to-end QA + turn-level retrieval + 6-bucket attribution",
        granularity=granularity,
        seed=seed,
        dataset=f"LoCoMo (n={len(items)})",
        dataset_path=dataset_path,
        extra={"consolidate": consolidate, "token_budget": token_budget},
    )
    return {
        "benchmark": "LoCoMo",
        "provenance": prov,
        "summary": summarize_qa_rows(items, categories=LOCOMO_CATEGORIES),
        "items": items,
    }
