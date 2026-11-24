"""Six-bucket root-cause attribution for every failed QA item."""
from __future__ import annotations

from collections import Counter
from typing import Iterable

ATTRIBUTION_BUCKETS = (
    "extraction_miss",
    "recall_miss",
    "ranking_drop",
    "assembly_truncation",
    "reasoning_error",
    "judge_noise",
)


def classify_failure(
    *,
    correct: bool,
    unanswerable: bool,
    gold_turns: Iterable[str],
    ingested_turns: Iterable[str],
    recalled_turns: Iterable[str],
    ranked_turns: Iterable[str],
    packed_turns: Iterable[str],
    borderline_judge: bool = False,
) -> str | None:
    """Map one failed QA item onto exactly one of the 6 pipeline failure buckets.

    Returns ``None`` when ``correct`` is True.
    """
    if correct:
        return None
    if unanswerable:
        # Unanswerable / abstention / adversarial items fail when the reader fails to abstain.
        return "reasoning_error"

    gold = set(gold_turns)
    if not gold:
        return "judge_noise" if borderline_judge else "reasoning_error"

    if not (gold & set(ingested_turns)):
        return "extraction_miss"
    if not (gold & set(recalled_turns)):
        return "recall_miss"
    if not (gold & set(ranked_turns)):
        return "ranking_drop"
    if not (gold & set(packed_turns)):
        return "assembly_truncation"
    if borderline_judge:
        return "judge_noise"
    return "reasoning_error"


def summarize_attribution(rows: list[dict]) -> dict:
    """Aggregate 6-bucket attribution counts and shares over evaluated QA items."""
    wrong = [r for r in rows if not r.get("correct")]
    counts = Counter(r.get("attribution") for r in wrong if r.get("attribution"))
    total_wrong = len(wrong)
    by_bucket = {
        bucket: {
            "count": int(counts.get(bucket, 0)),
            "share_of_errors": round(counts.get(bucket, 0) / total_wrong, 4) if total_wrong else 0.0,
        }
        for bucket in ATTRIBUTION_BUCKETS
    }
    by_category: dict[str, dict[str, int]] = {}
    for row in wrong:
        cat = str(row.get("category") or "unknown")
        bucket = str(row.get("attribution") or "reasoning_error")
        by_category.setdefault(cat, {b: 0 for b in ATTRIBUTION_BUCKETS})
        by_category[cat][bucket] = by_category[cat].get(bucket, 0) + 1
    return {
        "total_questions": len(rows),
        "total_wrong": total_wrong,
        "by_bucket": by_bucket,
        "by_category": by_category,
    }
