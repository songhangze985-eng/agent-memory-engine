"""CLI orchestrator for end-to-end QA + turn-level retrieval + 6-bucket attribution."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from eval.qa.harness import render_qa_markdown
from eval.qa.locomo import run_locomo_qa
from eval.qa.longmemeval import run_longmemeval_qa
from eval.qa.membench import run_membench_qa


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engines",
        nargs="+",
        default=["v1_chronicle", "v2_session_md"],
        choices=["v1_standard", "v1_chronicle", "v2_session_md", "v2_turn_md", "unified"],
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["longmemeval", "locomo", "membench"],
        choices=["longmemeval", "locomo", "membench"],
    )
    parser.add_argument("--longmemeval", type=Path, default=Path("eval/longmemeval/.cache/longmemeval_s_cleaned.json"))
    parser.add_argument("--locomo", type=Path, default=Path("eval/locomo/.cache/locomo10.json"))
    parser.add_argument("--memdata", type=Path, default=Path("eval/membench/.cache/MemData"))
    parser.add_argument("--lme-sample", type=int, default=100, help="Stratified LongMemEval-S sample (0 = full 500)")
    parser.add_argument("--locomo-sample", type=int, default=200, help="Stratified LoCoMo sample (0 = full 1,986)")
    parser.add_argument("--membench-per-cat", type=int, default=50, help="MemBench items per category (4 cats total)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 4) - 1)))
    parser.add_argument("--llm-model", type=str, default=None, help="OpenAI-compatible reader model (omit for offline deterministic)")
    parser.add_argument("--judge-model", type=str, default=None, help="OpenAI-compatible judge model (omit for deterministic-v1)")
    parser.add_argument("--consolidate", action="store_true", help="Run consolidation/dreaming before QA")
    parser.add_argument("--token-budget", type=int, default=2048)
    parser.add_argument("--out", type=Path, required=True, help="Output JSON report path")
    parser.add_argument("--md-out", type=Path, default=None, help="Optional Markdown report path")
    parser.add_argument("--viewer-out", type=Path, default=None, help="Optional per-question inspection JSONL path")
    args = parser.parse_args()

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    t0 = time.perf_counter()
    results: dict[str, dict] = {}
    md_sections: list[str] = ["# End-to-End QA & Turn-Level Retrieval Evaluation Report", ""]
    viewer_rows: list[dict] = []

    for bench in args.benchmarks:
        results[bench] = {}
        for eng in args.engines:
            if bench == "longmemeval":
                rep = run_longmemeval_qa(
                    args.longmemeval,
                    engine_kind=eng,
                    sample=None if args.lme_sample == 0 else args.lme_sample,
                    seed=args.seed,
                    workers=args.workers,
                    llm_model=args.llm_model,
                    judge_model=args.judge_model,
                    consolidate=args.consolidate,
                    token_budget=args.token_budget,
                )
            elif bench == "locomo":
                rep = run_locomo_qa(
                    args.locomo,
                    engine_kind=eng,
                    sample=None if args.locomo_sample == 0 else args.locomo_sample,
                    seed=args.seed,
                    workers=args.workers,
                    llm_model=args.llm_model,
                    judge_model=args.judge_model,
                    consolidate=args.consolidate,
                    token_budget=args.token_budget,
                )
            else:
                rep = run_membench_qa(
                    args.memdata,
                    engine_kind=eng,
                    per_category=args.membench_per_cat,
                    seed=args.seed,
                    workers=args.workers,
                    llm_model=args.llm_model,
                    judge_model=args.judge_model,
                    consolidate=args.consolidate,
                    token_budget=args.token_budget,
                )
            results[bench][eng] = {
                "benchmark": rep["benchmark"],
                "provenance": rep["provenance"],
                "summary": rep["summary"],
            }
            md_sections.append(render_qa_markdown(rep, f"{rep['benchmark']} — `{eng}`"))
            md_sections.append("")
            for item in rep["items"]:
                viewer_rows.append({"benchmark": bench, "engine": eng, **item})

    payload = {
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "seed": args.seed,
        "consolidate": args.consolidate,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = args.md_out or args.out.with_suffix(".md")
    md_path.write_text("\n".join(md_sections), encoding="utf-8")

    if args.viewer_out:
        args.viewer_out.parent.mkdir(parents=True, exist_ok=True)
        with args.viewer_out.open("w", encoding="utf-8") as fh:
            for row in viewer_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {args.out} and {md_path} in {payload['elapsed_s']}s")
    for bench, engs in results.items():
        for eng, data in engs.items():
            ov = data["summary"]["overall"]
            upd_str = "N/A" if ov["update_dominance"] is None else f"{ov['update_dominance']:.2%}"
            print(
                f"  [{bench:11s}] {eng:14s} | QA={ov['qa_accuracy']:.2%} | "
                f"Turn R@10={ov['turn_r10'] if ov['turn_r10'] is not None else 0:.2%} | "
                f"MRR@10={ov['turn_mrr10'] if ov['turn_mrr10'] is not None else 0:.4f} | "
                f"UpdDom={upd_str}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
