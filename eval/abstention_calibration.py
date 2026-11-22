"""Replay the synthetic cohort through local recall and calibrate answerable score floors."""

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import json
from itertools import groupby
from math import floor
from pathlib import Path

from eval.abstention_cohort import SEED, build_cases
from memengine.engine import CognitiveMemoryEngine
from memengine.reranker import CalibratedLocalReranker
from memengine.retrieval.abstention import abstention_features, abstention_group
from memengine.vector import HashedTfidfEmbedder

ALPHA = 0.02
SCORES = ("ce", "support", "combined", "facet_support")


def score_case(case):
    engine = CognitiveMemoryEngine(vector=HashedTfidfEmbedder(), reranker=CalibratedLocalReranker())
    with closing(engine.store):
        sources = {}
        for turn in case["turns"]:
            turn_id = engine.retain(turn["text"], timestamp=turn["timestamp"], role=turn["role"],
                                    user_id=turn["user_id"], session_id=turn["session_id"])["turn_id"]
            sources[turn_id] = turn["ref"]
        query = case["query"]
        recall = engine.recall(query, user_id=case["user_id"], timestamp=case["timestamp"],
                               token_budget=1024, packing_mode="token_budget")
    head = recall["ranked"][:5]
    parsed = recall["parsed_query"]
    source_refs = {sources[turn] for row in head for turn in row["source_turn_ids"] if turn in sources}
    expected = [value["source_ref"] for value in case["expected_slots"].values() if value]
    return {
        "id": case["id"], "group": case["group"], "split": case["split"], "intent": case["intent"],
        "outcome": case["outcome"], "variant": case["variant"],
        "runtime_group": abstention_group(query, parsed, recall["temporal_intent"]),
        "features": abstention_features(query, head, parsed["facets"],
                                        require_entity=bool(recall["matched_entities"])),
        "empty": not head, "expected_slots": len(expected),
        "retrieved_slots": sum(ref in source_refs for ref in expected),
    }


def score_floors(rows, score):
    groups = defaultdict(dict)
    for row in rows:
        if row["outcome"] == "unsupported":
            continue
        minima = groups[row["runtime_group"]]
        value = row["features"][score]
        minima[row["group"]] = min(minima.get(row["group"], value), value)
    thresholds = {}
    for intent, group_minima in sorted(groups.items()):
        ordered = sorted(group_minima.values())
        index = floor(ALPHA * (len(ordered) + 1))
        thresholds[intent] = {"groups": len(ordered), "threshold": ordered[index - 1] if index else None}
    return thresholds


def measure(rows, score, thresholds):
    counts, rejected, empty = Counter(), Counter(), Counter()
    by_variant = defaultdict(Counter)
    for row in rows:
        outcome = row["outcome"]
        threshold = thresholds.get(row["runtime_group"], {}).get("threshold")
        reject = not row["empty"] and threshold is not None and row["features"][score] < threshold
        counts[outcome] += 1
        empty[outcome] += row["empty"]
        rejected[outcome] += reject
        by_variant[row["variant"]]["cases"] += 1
        by_variant[row["variant"]]["baseline_empty"] += row["empty"]
        by_variant[row["variant"]]["additional_rejected"] += reject
    return {outcome: {"cases": count, "baseline_empty": empty[outcome], "additional_rejected": rejected[outcome],
                      "total_rejection_rate": (empty[outcome] + rejected[outcome]) / count}
            for outcome, count in sorted(counts.items())}, dict(by_variant)


def roc_points(rows, score):
    """Descriptive pooled ROC. Production thresholds remain stratified by runtime intent."""
    totals = Counter(row["outcome"] == "unsupported" for row in rows)
    rejected = Counter()
    points = [{"score_at_or_below": None, "false_empty_rate": 0.0, "unsupported_rejection_rate": 0.0}]
    ordered = sorted(rows, key=lambda row: row["features"][score])
    for value, tied in groupby(ordered, key=lambda row: row["features"][score]):
        rejected.update(row["outcome"] == "unsupported" for row in tied)
        points.append({"score_at_or_below": value,
                       "false_empty_rate": rejected[False] / totals[False] if totals[False] else None,
                       "unsupported_rejection_rate": rejected[True] / totals[True] if totals[True] else None})
    return points


def report(rows):
    calibration = [row for row in rows if row["split"] == "calibration"]
    holdout = [row for row in rows if row["split"] == "holdout"]
    if {row["group"] for row in calibration} & {row["group"] for row in holdout}:
        raise ValueError("a source group appears in both splits")
    results = {}
    for score in SCORES:
        thresholds = score_floors(calibration, score)
        outcomes, variants = measure(holdout, score, thresholds)
        by_group = {
            intent: measure([row for row in holdout if row["runtime_group"] == intent], score, thresholds)[0]
            for intent in sorted({row["runtime_group"] for row in holdout})
        }
        answerable = [stats for summary in (outcomes, *by_group.values())
                      for label, stats in summary.items() if label != "unsupported"]
        results[score] = {"thresholds": thresholds, "holdout": outcomes, "by_variant": variants,
                          "protects_answerable_at_2pct": bool(answerable) and all(
                              stats["total_rejection_rate"] <= ALPHA for stats in answerable),
                          "by_runtime_group": by_group, "roc": roc_points(holdout, score)}
    return {"alpha": ALPHA, "backend": "hashed", "scores": results,
            "holdout_expected_slots": sum(row["expected_slots"] for row in holdout),
            "holdout_retrieved_slots": sum(row["retrieved_slots"] for row in holdout),
            "items": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=SEED, help="Seed for the grouped synthetic cohort")
    args = parser.parse_args()
    outcome = {"seed": args.seed, **report([score_case(case) for case in build_cases(seed=args.seed)])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(outcome, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: scores["holdout"] for name, scores in outcome["scores"].items()}, indent=2))


if __name__ == "__main__":
    main()
