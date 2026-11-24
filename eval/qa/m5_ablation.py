"""M5 End-to-End QA & Turn-Level Retrieval Ablation Matrix for UnifiedMemoryEngine.

Empirically tests each ranking/reasoning module so components that do not improve
end-to-end QA accuracy or turn-level recall can be permanently disabled or removed.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from eval.qa.locomo import run_locomo_qa
from eval.qa.longmemeval import run_longmemeval_qa
from eval.qa.membench import run_membench_qa
from eval.qa.provenance import render_provenance_markdown
from memengine.core.plan import RecallConfig

ABLATION_VARIANTS: list[tuple[str, str, RecallConfig, bool]] = [
    (
        "full_m4",
        "All features ON + dreaming consolidation",
        RecallConfig(
            enable_bipolar=True,
            enable_recency_rerank=True,
            enable_episode_fts=True,
            enable_entity_graph=True,
            enable_slot_supersede=True,
            enable_graph_enum=True,
            enable_deterministic_calc=True,
        ),
        True,
    ),
    (
        "no_bipolar",
        "Disable Bipolar Asymptotic Ring & belt pruning",
        RecallConfig(enable_bipolar=False),
        True,
    ),
    (
        "no_recency_tiebreak",
        "Disable W_verified * W_recency tie-breaker",
        RecallConfig(enable_recency_rerank=False),
        True,
    ),
    (
        "no_slot_supersede",
        "Disable M2 slot-level supersede & cross-session update promotion",
        RecallConfig(enable_slot_supersede=False),
        True,
    ),
    (
        "no_graph_enum_calc",
        "Disable M3 entity-graph enumeration, coverage packing & calculator",
        RecallConfig(enable_graph_enum=False, enable_deterministic_calc=False),
        True,
    ),
    (
        "no_dreaming",
        "Disable M4 dreaming / reflective belief consolidation",
        RecallConfig(),
        False,
    ),
]


def _delta_str(val: float | None, base: float | None, *, pct: bool = True) -> str:
    if val is None or base is None:
        return "—"
    diff = val - base
    if abs(diff) < 1e-6:
        return "0.00" + (" pp" if pct else "")
    if pct:
        return f"{diff * 100:+.2f} pp"
    return f"{diff:+.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--longmemeval", type=Path, default=Path("eval/longmemeval/.cache/longmemeval_s_cleaned.json"))
    parser.add_argument("--locomo", type=Path, default=Path("eval/locomo/.cache/locomo10.json"))
    parser.add_argument("--memdata", type=Path, default=Path("eval/membench/.cache/MemData"))
    parser.add_argument("--lme-sample", type=int, default=100)
    parser.add_argument("--locomo-sample", type=int, default=200)
    parser.add_argument("--membench-per-cat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 4) - 1)))
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--judge-model", type=str, default=None)
    parser.add_argument("--out", type=Path, default=Path("eval/results/m5_ablation.json"))
    parser.add_argument("--md-out", type=Path, default=Path("eval/results/m5_ablation_report.md"))
    args = parser.parse_args()

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    t0 = time.perf_counter()

    variants_out: dict[str, dict] = {}
    provenance_sample: dict | None = None

    for name, desc, cfg, use_consolidate in ABLATION_VARIANTS:
        vt0 = time.perf_counter()
        lme = run_longmemeval_qa(
            args.longmemeval,
            engine_kind="unified",
            sample=None if args.lme_sample == 0 else args.lme_sample,
            seed=args.seed,
            workers=args.workers,
            llm_model=args.llm_model,
            judge_model=args.judge_model,
            consolidate=use_consolidate,
            recall_config=cfg,
        )
        loc = run_locomo_qa(
            args.locomo,
            engine_kind="unified",
            sample=None if args.locomo_sample == 0 else args.locomo_sample,
            seed=args.seed,
            workers=args.workers,
            llm_model=args.llm_model,
            judge_model=args.judge_model,
            consolidate=use_consolidate,
            recall_config=cfg,
        )
        mem = run_membench_qa(
            args.memdata,
            engine_kind="unified",
            per_category=args.membench_per_cat,
            seed=args.seed,
            workers=args.workers,
            llm_model=args.llm_model,
            judge_model=args.judge_model,
            consolidate=use_consolidate,
            recall_config=cfg,
        )
        if provenance_sample is None:
            provenance_sample = lme["provenance"]
        variants_out[name] = {
            "description": desc,
            "config": asdict(cfg),
            "consolidate": use_consolidate,
            "elapsed_s": round(time.perf_counter() - vt0, 2),
            "longmemeval": {"provenance": lme["provenance"], "summary": lme["summary"]},
            "locomo": {"provenance": loc["provenance"], "summary": loc["summary"]},
            "membench": {"provenance": mem["provenance"], "summary": mem["summary"]},
        }
        print(
            f"  [{name:20s}] "
            f"LME QA={lme['summary']['overall']['qa_accuracy']:.2%} R@10={lme['summary']['overall']['turn_r10']:.2%} | "
            f"LoCoMo QA={loc['summary']['overall']['qa_accuracy']:.2%} R@10={loc['summary']['overall']['turn_r10']:.2%} | "
            f"MemBench QA={mem['summary']['overall']['qa_accuracy']:.2%} MRR={mem['summary']['overall']['turn_mrr10']:.4f} "
            f"({variants_out[name]['elapsed_s']}s)",
            flush=True,
        )

    base = variants_out["full_m4"]
    payload = {
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "seed": args.seed,
        "cohort": {
            "longmemeval_n": args.lme_sample,
            "locomo_n": args.locomo_sample,
            "membench_n": args.membench_per_cat * 4,
        },
        "variants": variants_out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Render Markdown report with exact deltas vs full_m4
    md: list[str] = [
        "# M5 End-to-End Ablation Report (`UnifiedMemoryEngine`)",
        "",
        "Frozen 500Q cohort (`LongMemEval-S 100Q` + `LoCoMo 200Q` + `MemBench 200Q`, `seed=42`).",
        "Each row disables **one** component relative to `full_m4`. Negative delta (`-`) means removing the component hurts performance (KEEP); `0.00` or positive (`+`) means removing it does not hurt (DISABLE/REMOVE by default).",
        "",
    ]
    if provenance_sample:
        md.append(render_provenance_markdown(provenance_sample))
        md.append("")

    md.extend([
        "## Cross-Benchmark End-to-End QA & Turn-Level Retrieval Matrix",
        "",
        "| Variant | LME QA (Δ) | LME Turn R@10 (Δ) | LME UpdDom (Δ) | LoCoMo QA (Δ) | LoCoMo Turn R@10 (Δ) | MemBench QA (Δ) | MemBench MRR@10 (Δ) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    b_lme = base["longmemeval"]["summary"]["overall"]
    b_loc = base["locomo"]["summary"]["overall"]
    b_mem = base["membench"]["summary"]["overall"]

    for name, _, _, _ in ABLATION_VARIANTS:
        v = variants_out[name]
        lme_o = v["longmemeval"]["summary"]["overall"]
        loc_o = v["locomo"]["summary"]["overall"]
        mem_o = v["membench"]["summary"]["overall"]
        if name == "full_m4":
            md.append(
                f"| **`{name}`** | **{lme_o['qa_accuracy']:.2%}** | **{lme_o['turn_r10']:.2%}** | **{lme_o['update_dominance']:.2%}** | "
                f"**{loc_o['qa_accuracy']:.2%}** | **{loc_o['turn_r10']:.2%}** | "
                f"**{mem_o['qa_accuracy']:.2%}** | **{mem_o['turn_mrr10']:.4f}** |"
            )
        else:
            md.append(
                f"| `{name}` | {lme_o['qa_accuracy']:.2%} ({_delta_str(lme_o['qa_accuracy'], b_lme['qa_accuracy'])}) | "
                f"{lme_o['turn_r10']:.2%} ({_delta_str(lme_o['turn_r10'], b_lme['turn_r10'])}) | "
                f"{lme_o['update_dominance']:.2%} ({_delta_str(lme_o['update_dominance'], b_lme['update_dominance'])}) | "
                f"{loc_o['qa_accuracy']:.2%} ({_delta_str(loc_o['qa_accuracy'], b_loc['qa_accuracy'])}) | "
                f"{loc_o['turn_r10']:.2%} ({_delta_str(loc_o['turn_r10'], b_loc['turn_r10'])}) | "
                f"{mem_o['qa_accuracy']:.2%} ({_delta_str(mem_o['qa_accuracy'], b_mem['qa_accuracy'])}) | "
                f"{mem_o['turn_mrr10']:.4f} ({_delta_str(mem_o['turn_mrr10'], b_mem['turn_mrr10'], pct=False)}) |"
            )

    md.extend([
        "",
        "## Targeted Category Breakdown",
        "",
        "| Variant | LME `knowledge-update` QA | LME `temporal-reasoning` QA | LME `multi-session` QA | LoCoMo `1 multi-hop` QA | LoCoMo `3 open-domain` QA | MemBench `obs_reflective` QA |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for name, _, _, _ in ABLATION_VARIANTS:
        v = variants_out[name]
        lme_c = v["longmemeval"]["summary"]["by_category"]
        loc_c = v["locomo"]["summary"]["by_category"]
        mem_c = v["membench"]["summary"]["by_category"]
        md.append(
            f"| `{name}` | {lme_c['knowledge-update']['qa_accuracy']:.2%} | "
            f"{lme_c['temporal-reasoning']['qa_accuracy']:.2%} | "
            f"{lme_c['multi-session']['qa_accuracy']:.2%} | "
            f"{loc_c['1 multi-hop']['qa_accuracy']:.2%} | "
            f"{loc_c['3 open-domain']['qa_accuracy']:.2%} | "
            f"{mem_c['observation_reflective']['qa_accuracy']:.2%} |"
        )

    args.md_out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out} and {args.md_out} in {payload['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
