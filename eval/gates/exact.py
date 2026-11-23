"""Exact identifier precision regression gate (pass/fail only)."""
from __future__ import annotations

from eval.exact_ident_eval import run_static


def check_exact_gate(*, seed: int = 42) -> dict:
    report = run_static(seed=seed)
    scores = report["metrics"]["Overall"]["bipolar"]
    passed = (
        scores["precision_at_1"] == 1.0
        and scores["mrr_at_10"] == 1.0
        and scores["symbol_confusion"] == 0.0
        and scores["negative_empty"] == 1.0
        and scores["procedural_top3"] == 1.0
    )
    return {
        "gate": "exact",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "cases": len(report["items"]),
        "metrics": {
            "precision_at_1": scores["precision_at_1"],
            "mrr_at_10": scores["mrr_at_10"],
            "symbol_confusion": scores["symbol_confusion"],
            "negative_empty": scores["negative_empty"],
            "procedural_top3": scores["procedural_top3"],
        },
    }
