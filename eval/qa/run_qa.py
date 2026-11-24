"""CLI orchestrator for frozen-protocol 3-arm end-to-end QA + multi-seed intervals + 6-bucket attribution."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from eval.qa.harness import render_qa_markdown
from eval.qa.locomo import run_locomo_qa
from eval.qa.longmemeval import run_longmemeval_qa
from eval.qa.membench import run_membench_qa
from eval.qa.protocol import (
    DEFAULT_PROTOCOL_PATH,
    aggregate_multi_seed_reports,
    load_protocol,
)
from eval.qa.provenance import dir_manifest_sha256, file_sha256, render_provenance_markdown

ENGINE_CHOICES = [
    "full_context",
    "turn_rrf",
    "unified",
    "v1_standard",
    "v1_chronicle",
    "v2_session_md",
    "v2_turn_md",
]


def _verify_dataset_sha(protocol: dict[str, Any], key: str, path: Path) -> None:
    expected = ((protocol.get("datasets") or {}).get(key) or {}).get("sha256")
    if not expected or not path.exists():
        return
    actual = file_sha256(path) if path.is_file() else dir_manifest_sha256(path)
    if actual and actual != expected:
        raise ValueError(f"Dataset SHA-256 mismatch for {key} ({path}): expected {expected}, got {actual}")


def render_multi_seed_summary_markdown(agg: dict[str, Any]) -> str:
    ov = agg["overall"]
    qa = ov["qa_accuracy"]
    r10 = ov["turn_r10"]
    mrr = ov["turn_mrr10"]
    toks = ov["avg_injected_tokens"]
    p95 = ov["p95_ms"]
    lines = [
        f"## {agg['benchmark']} — `{agg['engine']}` (Multi-Seed: `{agg['seeds']}`)",
        "",
        render_provenance_markdown(agg["provenance"]),
        "",
        "| Metric | Mean | Min | Max | Std | ±95% CI |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
        f"| **QA Accuracy** | **{qa['mean']:.2%}** | {qa['min']:.2%} | {qa['max']:.2%} | {qa['std']:.2%} | ±{qa['ci95_half']:.2%} |",
    ]
    if r10 is not None:
        lines.append(
            f"| Turn R@10 | {r10['mean']:.2%} | {r10['min']:.2%} | {r10['max']:.2%} | {r10['std']:.2%} | ±{r10['ci95_half']:.2%} |"
        )
    if mrr is not None:
        lines.append(
            f"| Turn MRR@10 | {mrr['mean']:.4f} | {mrr['min']:.4f} | {mrr['max']:.4f} | {mrr['std']:.4f} | ±{mrr['ci95_half']:.4f} |"
        )
    lines.append(
        f"| Avg Injected Tokens | {toks['mean']:.1f} | {toks['min']:.1f} | {toks['max']:.1f} | {toks['std']:.1f} | ±{toks['ci95_half']:.1f} |"
    )
    lines.append(
        f"| P95 Latency (ms) | {p95['mean']:.2f} | {p95['min']:.2f} | {p95['max']:.2f} | {p95['std']:.2f} | ±{p95['ci95_half']:.2f} |"
    )
    lines.extend(["", "### Per-Category QA Accuracy (Across Seeds)", "", "| Category | N | Mean | Min | Max |", "| :--- | ---: | ---: | ---: | ---: |"])
    for cat, item in agg["by_category"].items():
        cqa = item["qa_accuracy"]
        lines.append(f"| `{cat}` | {item['n']} | **{cqa['mean']:.2%}** | {cqa['min']:.2%} | {cqa['max']:.2%} |")

    attr = agg["representative_attribution"]
    lines.extend([
        "",
        f"### Representative Error Attribution (Seed {agg['seeds'][0]}: {attr['total_wrong']} wrong / {attr['total_questions']} total)",
        "",
        "| Failure Bucket | Count | Share of Errors |",
        "| :--- | ---: | ---: |",
    ])
    for bucket, b in attr["by_bucket"].items():
        lines.append(f"| `{bucket}` | {b['count']} | {b['share_of_errors']:.2%} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH, help="Frozen protocol.yaml path")
    parser.add_argument("--engines", nargs="+", default=["full_context", "turn_rrf", "unified"], choices=ENGINE_CHOICES)
    parser.add_argument("--benchmarks", nargs="+", default=["longmemeval", "locomo"], choices=["longmemeval", "locomo", "membench"])
    parser.add_argument("--split", choices=["dev", "test", "full"], default="dev", help="Disjoint dataset split (tune on dev; final report once on test/full)")
    parser.add_argument("--longmemeval", type=Path, default=Path("eval/longmemeval/.cache/longmemeval_s_cleaned.json"))
    parser.add_argument("--locomo", type=Path, default=Path("eval/locomo/.cache/locomo10.json"))
    parser.add_argument("--memdata", type=Path, default=Path("eval/membench/.cache/MemData"))
    parser.add_argument("--lme-sample", type=int, default=60, help="Stratified LongMemEval-S sample (0 = all rows in split)")
    parser.add_argument("--locomo-sample", type=int, default=120, help="Stratified LoCoMo sample (0 = all rows in split)")
    parser.add_argument("--membench-per-cat", type=int, default=50, help="MemBench items per category")
    parser.add_argument("--seed", type=int, default=None, help="Single seed override")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Multi-seed list (default from protocol.yaml)")
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 4) - 1)))
    parser.add_argument("--llm-model", type=str, default=None, help="OpenAI-compatible reader model (omit for offline deterministic)")
    parser.add_argument("--extractor-model", type=str, default=None, help="Separate fact extractor model (explicitly recorded in provenance)")
    parser.add_argument("--judge-model", type=str, default=None, help="OpenAI-compatible judge model (omit for deterministic-v1)")
    parser.add_argument("--consolidate", action="store_true", help="Run consolidation/dreaming before QA")
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True, help="Output JSON report path")
    parser.add_argument("--md-out", type=Path, default=None, help="Optional Markdown report path")
    parser.add_argument("--viewer-out", type=Path, default=None, help="Optional per-question inspection JSONL path")
    args = parser.parse_args(argv)

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    proto_dict, proto_sha = load_protocol(args.protocol) if args.protocol.is_file() else ({}, "unfrozen")
    dev_ratio = float((proto_dict.get("splits") or {}).get("dev_ratio", 0.25))
    eff_budget = int(args.token_budget if args.token_budget is not None else (proto_dict.get("retrieval") or {}).get("token_budget", 2048))

    if args.seeds is not None:
        seeds = [int(s) for s in args.seeds]
    elif args.seed is not None:
        seeds = [int(args.seed)]
    else:
        raw_seeds = (proto_dict.get("judge") or {}).get("seeds") or [42]
        seeds = [int(s) for s in raw_seeds]

    if "longmemeval" in args.benchmarks:
        _verify_dataset_sha(proto_dict, "longmemeval_s", args.longmemeval)
    if "locomo" in args.benchmarks:
        _verify_dataset_sha(proto_dict, "locomo10", args.locomo)
    if "membench" in args.benchmarks:
        _verify_dataset_sha(proto_dict, "membench", args.memdata)

    t0 = time.perf_counter()
    results: dict[str, dict[str, Any]] = {}
    multi_seed_summary: dict[str, dict[str, Any]] = {}
    md_sections: list[str] = [
        "# Frozen-Protocol End-to-End QA Baseline Report",
        "",
        f"- **Protocol**: `{args.protocol}` (`version={proto_dict.get('protocol_version', '1.0.0')}`, `sha256={proto_sha[:16]}...`)",
        f"- **Split**: `{args.split}` (`dev_ratio={dev_ratio}`) | **Seeds**: `{seeds}` | **Arms**: `{args.engines}`",
        "",
    ]
    viewer_rows: list[dict[str, Any]] = []

    for bench in args.benchmarks:
        results[bench] = {}
        multi_seed_summary[bench] = {}
        for eng in args.engines:
            seed_reps: list[dict[str, Any]] = []
            for s in seeds:
                if bench == "longmemeval":
                    rep = run_longmemeval_qa(
                        args.longmemeval,
                        engine_kind=eng,
                        sample=None if args.lme_sample == 0 else args.lme_sample,
                        seed=s,
                        workers=args.workers,
                        llm_model=args.llm_model,
                        extractor_model=args.extractor_model,
                        judge_model=args.judge_model,
                        consolidate=args.consolidate,
                        token_budget=eff_budget,
                        split=args.split,
                        dev_ratio=dev_ratio,
                        protocol_path=args.protocol if args.protocol.is_file() else None,
                    )
                elif bench == "locomo":
                    rep = run_locomo_qa(
                        args.locomo,
                        engine_kind=eng,
                        sample=None if args.locomo_sample == 0 else args.locomo_sample,
                        seed=s,
                        workers=args.workers,
                        llm_model=args.llm_model,
                        extractor_model=args.extractor_model,
                        judge_model=args.judge_model,
                        consolidate=args.consolidate,
                        token_budget=eff_budget,
                        split=args.split,
                        dev_ratio=dev_ratio,
                        protocol_path=args.protocol if args.protocol.is_file() else None,
                    )
                else:
                    rep = run_membench_qa(
                        args.memdata,
                        engine_kind=eng,
                        per_category=args.membench_per_cat,
                        seed=s,
                        workers=args.workers,
                        llm_model=args.llm_model,
                        judge_model=args.judge_model,
                        consolidate=args.consolidate,
                        token_budget=eff_budget,
                    )
                seed_reps.append(rep)
                for item in rep["items"]:
                    viewer_rows.append({"benchmark": bench, "engine": eng, "seed": s, "split": args.split, **item})

            if len(seed_reps) == 1:
                rep0 = seed_reps[0]
                results[bench][eng] = {
                    "benchmark": rep0["benchmark"],
                    "provenance": rep0["provenance"],
                    "summary": rep0["summary"],
                }
                md_sections.append(render_qa_markdown(rep0, f"{rep0['benchmark']} — `{eng}`"))
                md_sections.append("")
            else:
                agg = aggregate_multi_seed_reports(seed_reps)
                multi_seed_summary[bench][eng] = agg
                results[bench][eng] = {
                    "benchmark": seed_reps[0]["benchmark"],
                    "provenance": agg["provenance"],
                    "summary": seed_reps[0]["summary"],
                    "multi_seed": agg,
                }
                md_sections.append(render_multi_seed_summary_markdown(agg))
                md_sections.append("")

    # Compact 3-arm comparison matrix at top of Markdown
    summary_table = [
        "## Cross-Arm Summary",
        "",
        "| Benchmark | Arm | Split | Seeds | QA Acc (Mean) | Spread (Min–Max) | Turn R@10 | Avg Tokens | P95 ms |",
        "| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for bench, engs in results.items():
        for eng, data in engs.items():
            if "multi_seed" in data:
                ms = data["multi_seed"]["overall"]
                qa = ms["qa_accuracy"]
                r10 = ms["turn_r10"]["mean"] if ms["turn_r10"] else None
                toks = ms["avg_injected_tokens"]["mean"]
                p95 = ms["p95_ms"]["mean"]
                spread = f"{qa['min']:.2%}–{qa['max']:.2%}"
                qa_str = f"**{qa['mean']:.2%}**"
            else:
                ov = data["summary"]["overall"]
                r10 = ov["turn_r10"]
                toks = ov["avg_injected_tokens"]
                p95 = ov["p95_ms"]
                spread = "single-seed"
                qa_str = f"**{ov['qa_accuracy']:.2%}**"
            r10_str = "—" if r10 is None else f"{r10:.2%}"
            summary_table.append(
                f"| `{bench}` | `{eng}` | `{args.split}` | `{seeds}` | {qa_str} | {spread} | {r10_str} | {toks:.1f} | {p95:.2f} |"
            )
    summary_table.append("")
    md_sections = [*md_sections[:5], *summary_table, *md_sections[5:]]

    payload = {
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "protocol_version": proto_dict.get("protocol_version", "1.0.0"),
        "protocol_sha256": proto_sha,
        "split": args.split,
        "seeds": seeds,
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
    for line in summary_table[2:]:
        print(" ", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
