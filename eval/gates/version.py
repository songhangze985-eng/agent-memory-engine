"""Eight-class version precision regression gate (pass/fail only)."""
from __future__ import annotations

from eval.version_precision_eval import evaluate as eval_version_precision


def check_version_gate() -> dict:
    rows = eval_version_precision()
    n = len(rows)
    precision = sum(row["precision"] for row in rows) / n if n else 0.0
    grade_correct = sum(row["grade_correct"] for row in rows) / n if n else 0.0
    confusion = sum(row["confusion"] for row in rows) / n if n else 0.0
    passed = precision == 1.0 and grade_correct == 1.0 and confusion == 0.0
    return {
        "gate": "version",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "cases": n,
        "metrics": {
            "precision_at_1": precision,
            "grade_correct": grade_correct,
            "confusion": confusion,
        },
    }
