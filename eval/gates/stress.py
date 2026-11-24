"""Stress precision, bipolar belt, and long-horizon archive regression gates (pass/fail only)."""
from __future__ import annotations

from eval.long_horizon_static import SCALES, horizon_case
from eval.stress_precision import run as run_stress_eval


def check_stress_gate(*, seed: int = 42, include_long_horizon: bool = True) -> dict:
    stress = run_stress_eval(seed=seed)
    scores = stress["overall"]["bipolar"]
    stress_ok = (
        scores["recall_at_5"] == 1.0
        and scores["recall_at_10"] == 1.0
        and scores["mrr_at_10"] == 1.0
        and scores["false_empty"] == 0
        and scores["avg_injected_tokens"] <= 180.0
    )
    horizon_scales = []
    horizon_ok = True
    if include_long_horizon:
        for turns in SCALES:
            row = horizon_case(turns, seed)
            row_ok = (
                row["taboo_recall_at_1"] == 1
                and row["taboo_hot"]
                and row["taboo_decay"] == 1.0
                and row["current_rank"] == 1
                and row["cold_wake_rate"] == 1.0
                and row["cold_top1"] == 1.0
                and row["hot_compression"] >= 0.7995
                and row["average_tokens"] <= 1926 / 7
            )
            horizon_ok = horizon_ok and row_ok
            horizon_scales.append({
                "turns": turns,
                "passed": row_ok,
                "hot_compression": round(row["hot_compression"], 4),
                "average_tokens": round(row["average_tokens"], 2),
                "cold_wake_rate": row["cold_wake_rate"],
            })

    passed = stress_ok and horizon_ok
    return {
        "gate": "stress",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "metrics": {
            "stress_recall_at_10": scores["recall_at_10"],
            "stress_mrr_at_10": scores["mrr_at_10"],
            "stress_false_empty": scores["false_empty"],
            "stress_avg_tokens": round(scores["avg_injected_tokens"], 2),
            "long_horizon_passed": horizon_ok,
            "long_horizon_scales": horizon_scales,
        },
    }
