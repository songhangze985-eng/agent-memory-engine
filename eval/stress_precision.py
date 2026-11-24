"""Paired evidence retrieval on synthetic long dialogues and out-of-store traps."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import random

from eval.evidence_ablation import Question, Turn, measure_pair
from eval.metrics import mean, print_retrieval_reports, retrieval_report
from eval.paired import LABELS, TOP_K, TOKEN_BUDGET, add_backend_argument, cli_backend, evaluate_pair, make_engine
from eval.stress_corpus import stress_dialogue
from agentmem_core.distance import consolidation_count, compute_bipolar_radius, is_noise
from agentmem_core.llm import build_llm_from_env


def run(sample=20, seed=42, mode="mock", backend="hashed"):
    items = []
    rng = random.Random(seed)
    for case_index in rng.sample(range(20), sample):
        case = stress_dialogue(case_index)
        engine = make_engine(mode, backend=backend)
        with closing(engine.store):
            raw_ids = {}
            for index, (kind, text) in enumerate(case.turns):
                added = engine.add(f"[step {index}] {text}", timestamp=index)
                raw_ids[added["turn_id"]] = Turn(str(index), "default", "user", text, index)
                if kind == "taboo":
                    gold = frozenset({str(index)})
            stamp = len(case.turns) + 1
            engine.consolidate(timestamp=stamp)
            questions = [Question(f"{case_index + 1}:answerable", "taboo", case.positive, stamp, gold),
                         Question(f"{case_index + 1}:negative", case.negative_kind, case.negative, stamp, unanswerable=True)]
            rng.shuffle(questions)
            previous = None
            for question in questions:
                pair = evaluate_pair(engine, question.text, question.timestamp)
                row = {"id": question.id, "category": question.category, "unanswerable": question.unanswerable,
                       "evidence": sorted(question.evidence), **measure_pair(engine, question, pair, raw_ids, previous)}
                previous = row["bipolar"]["prefix_version"]
                circle = pair.retrieved["poison_circle"]
                if circle and not question.unanswerable:
                    maximum = max((consolidation_count(f) for f in pair.candidates if not is_noise(f)), default=1)
                    belt = {f["id"] for f in pair.candidates if
                            circle["radius_r"] < compute_bipolar_radius(f, maximum) < circle["radius_out"]}
                    full = [item for item in pair.context["items"] if item["kind"] == "fact"]
                    for label, subset in (("bipolar", full), ("rrf", pair.baseline)):
                        row[label]["belt_compression"] = (1 - len(belt & {f["id"] for f in subset}) / len(belt)) if belt else None
                items.append(row)
    overall = {label: retrieval_report(items, label) for label in LABELS}
    for label in LABELS:
        overall[label]["belt_compression"] = mean(row[label].get("belt_compression") for row in items)
    buckets = {category: {label: retrieval_report([row for row in items if row["category"] == category], label) for label in LABELS}
               for category in sorted({row["category"] for row in items})}
    return {"benchmark": "Stress + Precision", "dataset_source": "eval/stress_corpus.py (synthetic; not PrecisionMemBench)",
            "mode": mode, "backend": backend, "seed": seed, "sample_cases": sample, "top_k": TOP_K, "token_budget": TOKEN_BUDGET,
            "overall": overall, "by_category": buckets, "items": items}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=20, help="Number of paired dialogues (one positive, one negative each)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("mock", "live"), default="mock")
    add_backend_argument(parser)
    parser.add_argument("--out", type=Path, default=Path("stress-precision.json"))
    args = parser.parse_args()
    if not 1 <= args.sample <= 20:
        parser.error("--sample must be between 1 and 20")
    model = None
    if args.mode == "live":
        _, resolved, model = build_llm_from_env()
        if resolved != "live":
            parser.error("live mode requires LLM_API_KEY or OPENAI_API_KEY")
    backend = cli_backend(parser, args.backend)
    report = {**run(args.sample, args.seed, args.mode, backend), "model": model}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['benchmark']}: {len(report['items'])} questions; mode={args.mode}; backend={backend}; seed={args.seed}; source={report['dataset_source']}")
    groups = {"Overall": report["overall"], **report["by_category"]}
    print_retrieval_reports((key, label, metrics[mode]) for key, metrics in groups.items() for mode, label in LABELS.items())
    print("\nAnswerable belt compression:", json.dumps({LABELS[mode]: report["overall"][mode]["belt_compression"] for mode in LABELS}))
    print(f"Raw JSON: {args.out.resolve()}")
    if report["overall"]["bipolar"]["false_empty_rate"] > 0.01:
        parser.exit(1, "Acceptance failed: Full System false-empty rate exceeded 1%.\n")


if __name__ == "__main__":
    main()
