"""Scoring helpers. Definitions are the ones used in eval/results."""

from __future__ import annotations

import math
import re


def mean(values):
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def evidence_metrics(ranked_refs, gold):
    if not gold:
        return {"recall_at_5": None, "recall_at_10": None, "mrr_at_10": None}
    hits = [set(refs) & gold for refs in ranked_refs[:10]]
    first = next((rank for rank, refs in enumerate(hits, 1) if refs), None)
    return {"recall_at_5": len(set().union(*hits[:5])) / len(gold),
            "recall_at_10": len(set().union(*hits)) / len(gold),
            "mrr_at_10": 1 / first if first else 0.0}


def retrieval_report(rows, mode):
    samples = [row[mode] for row in rows]
    negative = [row[mode] for row in rows if row.get("unanswerable")]
    answerable = [row[mode] for row in rows if not row.get("unanswerable")]
    true_empty = sum(bool(sample["empty"]) for sample in negative)
    false_empty = sum(bool(sample["empty"]) for sample in answerable)
    latency = [sample["latency_ms"] for sample in samples if sample.get("latency_ms") is not None]
    prefix = [sample.get("prefix_hit") for sample in samples if sample.get("prefix_hit") is not None]
    get = lambda key: mean(sample.get(key) for sample in samples)
    return {"n": len(rows), "answerable_n": len(answerable), "negative_n": len(negative),
            "evidence_n": sum(sample.get("recall_at_10") is not None for sample in samples),
            **{key: get(key) for key in ("recall_at_5", "recall_at_10", "mrr_at_10")},
            "update_dominance": get("update_dominates"),
            "update_n": sum(sample.get("update_dominates") is not None for sample in samples),
            "abstention_rate": true_empty / len(negative) if negative else None,
            "empty_recall_precision": true_empty / (true_empty + false_empty) if true_empty + false_empty else None,
            "true_empty": true_empty, "false_empty": false_empty,
            "false_empty_rate": false_empty / len(answerable) if answerable else None,
            **{f"avg_{key}": get(key) for key in ("injected_tokens", "prefix_tokens", "dynamic_tokens", "drilled_raw_tokens")},
            "observation_queries": sum(sample.get("observation_count", 0) > 0 for sample in samples),
            "observation_gold_queries": sum(sample.get("observation_gold_supports", 0) > 0 for sample in samples),
            "observation_only_gold_queries": sum(sample.get("observation_only_gold_supports", 0) > 0 for sample in samples),
            "prefix_hit_rate": mean(prefix), "prefix_comparisons": len(prefix),
            "p50_retrieval_ms": percentile(latency, 0.5), "p95_retrieval_ms": percentile(latency, 0.95)}


def retrieval_table(reports):
    lines = ["| Bucket | System | R@5 | R@10 | MRR@10 | Update | Abstain | Empty precision | False-empty | "
             "Injected (prefix / dynamic / raw) | Prefix reuse (N) | P50 / P95 ms |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]

    def show(value, fmt):
        return "—" if value is None else format(value, fmt)

    for bucket, label, data in reports:
        percentages = " | ".join(show(data[key], ".2%") for key in (
            "update_dominance", "abstention_rate", "empty_recall_precision", "false_empty_rate"))
        pieces = " / ".join(show(data[f"avg_{key}_tokens"], ".1f") for key in ("prefix", "dynamic", "drilled_raw"))
        latency = " / ".join(show(data[f"p{n}_retrieval_ms"], ".2f") for n in (50, 95))
        lines.append(f"| {bucket} | {label} | {show(data['recall_at_5'], '.2%')} | {show(data['recall_at_10'], '.2%')} | "
                     f"{show(data['mrr_at_10'], '.4f')} | {percentages} | {show(data['avg_injected_tokens'], '.1f')} ({pieces}) | "
                     f"{show(data['prefix_hit_rate'], '.2%')} ({data['prefix_comparisons']}) | {latency} |")
    return "\n".join(lines)


def print_retrieval_reports(reports):
    print("\n" + retrieval_table(reports))


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * p
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def answer_correct(response: str, question: dict) -> bool:
    text = response or ""
    for keyword in question.get("keywords") or []:
        if keyword not in text:
            return False
    for banned in question.get("forbidden") or []:
        if banned and banned in text:
            return False
    return True


def affirms(text: str, assertion: str) -> bool:
    """True when ``assertion`` appears without a nearby negation."""
    if not assertion or assertion not in text:
        return False
    start = 0
    while True:
        index = text.find(assertion, start)
        if index < 0:
            return False
        prefix = text[max(0, index - 8) : index]
        if not re.search(r"[不没无非]", prefix):
            return True
        start = index + len(assertion)


def trap_hallucinated(response: str, question: dict) -> bool:
    text = response or ""
    if affirms(text, question.get("false_assertion") or ""):
        return True
    for token in question.get("protected") or []:
        if token and token in text:
            return True
    return False
