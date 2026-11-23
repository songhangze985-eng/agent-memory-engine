"""Run all deterministic pass/fail regression gates (zero composite score weighting)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eval.gates.chrono import check_chrono_gate
from eval.gates.exact import check_exact_gate
from eval.gates.stress import check_stress_gate
from eval.gates.version import check_version_gate
from eval.qa.provenance import build_provenance


def run_all_gates(*, seed: int = 42, include_long_horizon: bool = True) -> dict:
    started = time.perf_counter()
    gates = [
        check_exact_gate(seed=seed),
        check_version_gate(),
        check_chrono_gate(),
        check_stress_gate(seed=seed, include_long_horizon=include_long_horizon),
    ]
    all_passed = all(g["passed"] for g in gates)
    return {
        "suite": "regression_gates",
        "status": "PASS" if all_passed else "FAIL",
        "all_passed": all_passed,
        "elapsed_s": round(time.perf_counter() - started, 2),
        "provenance": build_provenance(
            engine="memengine",
            embedder="HashedTfidfEmbedder",
            reranker="CalibratedLocalReranker",
            llm="MockLLMClient",
            judge="exact-deterministic-gate",
            protocol="pass/fail regression gates (non-scoring)",
            granularity="item/turn-level",
            seed=seed,
            dataset="synthetic_regression_suites",
        ),
        "gates": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-long-horizon", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = run_all_gates(seed=args.seed, include_long_horizon=not args.skip_long_horizon)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("| Gate | Status | Key Metrics |")
    print("| :--- | :---: | :--- |")
    for gate in report["gates"]:
        summary = ", ".join(
            f"{k}={v}" for k, v in gate["metrics"].items() if not isinstance(v, list)
        )
        print(f"| `{gate['gate']}` | **{gate['status']}** | {summary} |")
    print(f"\nOverall Regression Gates: **{report['status']}** ({report['elapsed_s']}s)")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
