"""Calendar & relative temporal parsing regression gate (pass/fail only)."""
from __future__ import annotations

from eval.chrono_eval import evaluate as eval_chrono


def check_chrono_gate() -> dict:
    rows = eval_chrono()
    n = len(rows)
    pass_rate = sum(row["pass"] for row in rows) / n if n else 0.0
    passed = pass_rate == 1.0
    return {
        "gate": "chrono",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "cases": n,
        "metrics": {
            "pass_rate": pass_rate,
        },
    }
