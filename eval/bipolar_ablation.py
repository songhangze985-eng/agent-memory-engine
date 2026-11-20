"""Paired offline MemBench and long-dialogue ablations on shared RRF candidates."""

import argparse
import json
from contextlib import closing
from pathlib import Path
from eval.evidence_ablation import measure_pair
from eval.membench.catalog import CATEGORIES
from eval.membench.items import parse_timestamp, question_stem
from eval.membench.load import load_sample
from eval.membench.run import _choose, _item_key, evidence_turn, paired_question
from eval.membench.score import choice_correct, extract_steps, mean, recall_at_10
from eval.metrics import print_retrieval_reports, retrieval_report
from eval.paired import LABELS, TOKEN_BUDGET, TOP_K, add_backend_argument, cli_backend, evaluate_pair, make_engine, paired_retrieve, attach_consolidated_context
from eval.stress_corpus import stress_dialogue
from memengine.distance import compute_bipolar_radius, consolidation_count, is_noise
from memengine.textutil import estimate_tokens


def measure(item, facts):
    texts = list(dict.fromkeys(fact["text"] for fact in facts))
    context = "\n".join(texts)
    predicted, _, _ = _choose(None, True, context, item)
    steps = list(dict.fromkeys(extract_steps(texts)))[:10]
    return {"choice": predicted, "correct": choice_correct(predicted, item["ground_truth"]),
            "recall_at_10": recall_at_10(steps, item["evidence_steps"]), "retrieved_steps": steps,
            "context_tokens": estimate_tokens(context), "facts": len(texts)}


def compare(item, backend="hashed", consolidate=False, *, compare_consolidation=False):
    engine = make_engine("mock", allow_casual=True, backend=backend)
    with closing(engine.store):
        raw_ids = {}
        for index, turn in enumerate(item["turns"]):
            added = engine.add(turn["text"], timestamp=turn["ts"])
            raw_ids[added["turn_id"]] = evidence_turn(turn, index)
        stamp = parse_timestamp(item.get("q_time"), max((turn["ts"] for turn in item["turns"]), default=0) + 1)
        stem = question_stem(item)
        pair = evaluate_pair(engine, stem, stamp)
        question = paired_question(item, stamp)
        with_profile = None
        if consolidate or compare_consolidation:
            engine.consolidate(timestamp=stamp)
            with_profile = attach_consolidated_context(engine, pair, stem)
            if consolidate:
                pair = with_profile
        retrieved, baseline, candidates = pair.retrieved, pair.baseline, pair.candidates
        return {"id": _item_key(item), "tid": item["tid"], "category": item["category"],
                "ground_truth": item["ground_truth"], "evidence_steps": item["evidence_steps"],
                "route": retrieved["route"], "n_outer": retrieved["outer_triggered_count"],
                "n_outer_exception": sum(fact.get("isolated") or fact.get("is_conflict_or_override")
                                         or fact.get("historical") for fact in retrieved["outer"]),
                "n_belt_compressed": retrieved["belt_compressed_count"], "candidates": len(candidates),
                "poison": measure(item, retrieved["ranked"]), "rrf": measure(item, baseline),
                "paired": measure_pair(engine, question, pair, raw_ids),
                **({"consolidated_pair": measure_pair(engine, question, with_profile, raw_ids)}
                   if compare_consolidation else {})}


def summary(rows):
    outcomes = {mode: {"correct": sum(row[mode]["correct"] for row in rows),
                       "accuracy": mean([row[mode]["correct"] for row in rows]),
                       "recall_at_10": mean([row[mode]["recall_at_10"] for row in rows
                                             if row[mode]["recall_at_10"] is not None]),
                       "mean_context_tokens": mean([row[mode]["context_tokens"] for row in rows]),
                       "mean_facts": mean([row[mode]["facts"] for row in rows]),
                       "empty_context": sum(row[mode]["facts"] == 0 for row in rows)}
                for mode in ("poison", "rrf")}
    return {"n": len(rows), "n_labeled_evidence": sum(bool(row["evidence_steps"]) for row in rows),
            "n_with_penetration": sum(row["n_outer"] > 0 for row in rows),
            "outer_triggered_count": sum(row["n_outer"] for row in rows),
            "outer_explicit_exception_count": sum(row["n_outer_exception"] for row in rows),
            "belt_compressed_count": sum(row["n_belt_compressed"] for row in rows),
            "poison_only_correct": sum(row["poison"]["correct"] and not row["rrf"]["correct"] for row in rows),
            "rrf_only_correct": sum(row["rrf"]["correct"] and not row["poison"]["correct"] for row in rows), **outcomes}


def membench(data_dir, reference, phase_a=False, cohort_only=False, backend="hashed", consolidate=False, *, offline=False):
    seeds = [42, 43, 44, 45, 46]
    results = {}
    cohorts = {}
    for seed in seeds:
        items = load_sample(5, seed, data_dir, offline=True) if offline else load_sample(5, seed, data_dir)
        assert len(items) == 20
        keys = [_item_key(item) for item in items]
        assert len(keys) == len(set(keys))
        cohorts[seed] = keys
        for item, key in zip(items, keys):
            if key not in results:
                results[key] = compare(item, backend, consolidate)
        print(f"seed {seed}: 20 samples, {len(results)} unique total", flush=True)
    rows = list(results.values())
    primary = [results[key] for key in cohorts[42]]
    assert len(rows) == 99
    if reference:
        previous = json.loads(reference.read_text())
        assert {int(seed): keys for seed, keys in previous["cohorts"].items()} == cohorts
        for row, logged in zip(rows, previous["items"]):
            assert row["id"] == logged["id"] and row["evidence_steps"] == logged["evidence_steps"]
            if not (phase_a or cohort_only):
                assert row["rrf"] == logged["rrf"], row["id"]
        print(f"Verified the same 99 question IDs and evidence annotations against {reference}.")
        if phase_a:
            print("Before Phase A, exact same cohort / two-channel RRF:", json.dumps(previous["all_unique"]["rrf"]))
    paired = {key: retrieval_report([item["paired"] for item in rows], key) for key in LABELS}
    report = {"benchmark": "MemBench", "dataset_source": "https://github.com/import-myself/Membench/tree/main/MemData",
              "top_k": TOP_K, "token_budget": TOKEN_BUDGET, "mode": "mock", "backend": backend, "seeds": seeds,
              "consolidated": consolidate, "prefix_budget": "separate" if consolidate else "included",
              "seed_42": summary(primary), "all_unique": summary(rows),
              "paired_retrieval": paired,
              "by_category": {category: summary([row for row in rows if row["category"] == category]) for category in CATEGORIES},
              "paired_by_category": {category: {key: retrieval_report(
                  [row["paired"] for row in rows if row["category"] == category], key) for key in LABELS}
                  for category in CATEGORIES},
              "cohorts": cohorts, "items": rows}
    for label in ("seed_42", "all_unique", "by_category"):
        print(label, json.dumps(report[label], ensure_ascii=False))
    print(f"MemBench: {len(rows)} unique questions; mode=mock; backend={backend}; seeds={seeds}; source={report['dataset_source']}")
    print_retrieval_reports(("Overall", label, paired[key]) for key, label in LABELS.items())
    return report


def stress_case(index, backend="hashed"):
    engine = make_engine("mock", allow_casual=True, backend=backend)
    with closing(engine.store):
        case = stress_dialogue(index)
        background_ids = set()
        for index_in_chat, (kind, text) in enumerate(case.turns):
            result = engine.add(f"[step {index_in_chat}] {text}", timestamp=index_in_chat)
            if index_in_chat == 0:
                habit_id = result["operations"][0]["line_id"]
            elif kind == "background" and result["operations"][0]["action"] == "NOOP":
                background_ids.add(result["operations"][0]["target_line_id"])
            elif kind == "taboo":
                exception_id = result["operations"][0]["line_id"]
        tick = len(case.turns)
        assert engine.store.get_line(exception_id)["isolated"]
        assert consolidation_count(engine.store.get_line(habit_id)) == 6 + index % 5
        retrieved, baseline, candidates = paired_retrieve(engine, case.positive, tick + 1)
        c_max = max(consolidation_count(fact) for fact in candidates if not is_noise(fact))
        ring = retrieved["poison_circle"]
        belt = {fact["id"] for fact in candidates
                if ring["radius_r"] < compute_bipolar_radius(fact, c_max) < ring["radius_out"]}
        assert exception_id in {fact["id"] for fact in candidates}
        assert len(background_ids) == 14 + index % 5 and background_ids <= belt
        assert engine.store.get_line(habit_id)["hit_count"] == 0

        def metrics(facts):
            ids = [fact["id"] for fact in facts]
            rank = ids.index(exception_id) + 1 if exception_id in ids else None
            return {"rank": rank, "recall_at_3": int(rank is not None and rank <= 3),
                    "recall_at_5": int(rank is not None and rank <= 5), "mrr": 1 / rank if rank else 0.0,
                    "belt_compression": 1 - len(belt.intersection(ids)) / len(belt),
                    "tokens": estimate_tokens("\n".join(fact["text"] for fact in facts))}

        return {"case": index + 1, "turns": tick, "habit_count": c_max, "belt_candidates": len(belt),
                "mid_frequency_candidates": len(belt.intersection(background_ids)),
                "bipolar": metrics(retrieved["ranked"]), "rrf": metrics(baseline),
                "belt_pruned": retrieved["belt_pruned_count"],
                "cliff_dropped": retrieved["gate_stats"]["cliff_dropped_count"],
                "empty_recall": retrieved["empty_recall"],
                "outer_triggered_count": retrieved["outer_triggered_count"],
                "exception_triggered": any(fact["id"] == exception_id for fact in retrieved["outer"])}


def stress(backend="hashed"):
    rows = [stress_case(index, backend) for index in range(20)]
    keys = ("recall_at_3", "recall_at_5", "mrr", "belt_compression", "tokens")
    aggregate = {mode: {key: mean([row[mode][key] for row in rows]) for key in keys} for mode in ("bipolar", "rrf")}
    print("| Case | Turns | Belt pool | Exception rank (bipolar / RRF) | Belt compression (bipolar / RRF) |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(f"| {row['case']} | {row['turns']} | {row['belt_candidates']} | "
              f"{row['bipolar']['rank'] or '—'} / {row['rrf']['rank'] or '—'} | "
              f"{row['bipolar']['belt_compression']:.1%} / {row['rrf']['belt_compression']:.1%} |")
    print("\n| Mode | Recall@3 | Recall@5 | MRR@10 | Belt compression | Avg tokens |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for mode, values in aggregate.items():
        print(f"| {mode} | {values['recall_at_3']:.1%} | {values['recall_at_5']:.1%} | "
              f"{values['mrr']:.4f} | {values['belt_compression']:.1%} | {values['tokens']:.1f} |")
    outer = sum(row["outer_triggered_count"] for row in rows)
    exceptions = sum(row["exception_triggered"] for row in rows)
    print(f"\nOuter selections: {outer}; explicit one-off exceptions selected: {exceptions}/{len(rows)}")
    print(f"Belt candidates removed by relative pruning: {sum(row['belt_pruned'] for row in rows)}")
    print(f"Candidates dropped at cliffs: {sum(row['cliff_dropped'] for row in rows)}; "
          f"empty recalls: {sum(row['empty_recall'] for row in rows)}/{len(rows)}")
    return {"n": len(rows), "backend": backend, "outer_triggered_count": outer, "exception_triggered_count": exceptions,
            "aggregate": aggregate, "items": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("membench", "stress"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--reference", type=Path, help="Previous 99-item report, to verify the same samples and RRF baseline.")
    parser.add_argument("--phase-a", action="store_true", help="Compare the same cohort without asserting the new four-channel RRF equals the old two-channel baseline.")
    parser.add_argument("--cohort-only", action="store_true", help="Verify question IDs and gold evidence against --reference, allowing changed ranking.")
    add_backend_argument(parser)
    parser.add_argument("--offline", action="store_true", help="Require local files and hashed retrieval")
    parser.add_argument("--consolidate", action="store_true", help="Measure MemBench observations and the additional cached prefix")
    args = parser.parse_args()
    if args.mode == "membench" and not args.data_dir:
        parser.error("--data-dir is required in membench mode")
    if args.cohort_only and not args.reference:
        parser.error("--cohort-only requires --reference")
    if args.offline and args.backend != "hashed":
        parser.error("--offline requires the hashed backend")
    backend = cli_backend(parser, args.backend)
    report = membench(args.data_dir, args.reference, args.phase_a, args.cohort_only, backend, args.consolidate, offline=args.offline) if args.mode == "membench" else stress(backend)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.mode == "membench":
        overall = report["all_unique"]
        reference = json.loads(args.reference.read_text())["all_unique"]["rrf"] if args.phase_a and args.reference else overall["rrf"]
        checks = {"empty_context_zero": overall["poison"]["empty_context"] == 0,
                  "recall_not_below_reference_rrf": overall["poison"]["recall_at_10"] >= reference["recall_at_10"],
                  "outer_triggered": overall["outer_triggered_count"] > 0}
        print("Acceptance:", json.dumps(checks))
        if not all(checks.values()):
            raise SystemExit(1)
    else:
        bipolar, rrf = (report["aggregate"][mode] for mode in ("bipolar", "rrf"))
        checks = {"recall_at_3_and_5_equal_one": bipolar["recall_at_3"] == bipolar["recall_at_5"] == 1,
                  "mrr_leads_rrf": bipolar["mrr"] > rrf["mrr"],
                  "belt_compression_leads_rrf": bipolar["belt_compression"] > rrf["belt_compression"],
                  "fewer_tokens_than_rrf": bipolar["tokens"] < rrf["tokens"]}
        print("Acceptance:", json.dumps(checks))
        if not all(checks.values()):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
