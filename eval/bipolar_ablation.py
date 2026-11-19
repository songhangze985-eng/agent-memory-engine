"""Paired offline MemBench and long-dialogue ablations on shared RRF candidates."""

import argparse
import json
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from eval.membench.catalog import CATEGORIES
from eval.membench.items import parse_timestamp, question_stem
from eval.membench.load import load_sample
from eval.membench.run import _choose, _item_key
from eval.membench.score import choice_correct, extract_steps, mean, recall_at_10
from memengine import engine as engine_module
from memengine.distance import compute_bipolar_radius, consolidation_count, is_noise
from memengine.engine import CognitiveMemoryEngine, _fit_context
from memengine.llm import MockLLMClient
from memengine.textutil import estimate_tokens
from memengine.vector import HashedTfidfEmbedder


TOP_K = 10
TOKEN_BUDGET = 2048


def measure(item, facts):
    texts = list(dict.fromkeys(fact["text"] for fact in facts))
    context = "\n".join(texts)
    predicted, _, _ = _choose(None, True, context, item)
    steps = list(dict.fromkeys(extract_steps(texts)))[:10]
    return {"choice": predicted, "correct": choice_correct(predicted, item["ground_truth"]),
            "recall_at_10": recall_at_10(steps, item["evidence_steps"]), "retrieved_steps": steps,
            "context_tokens": estimate_tokens(context), "facts": len(texts)}


def paired_retrieve(engine, query, stamp):
    fused = []
    original = engine_module.weighted_rrf

    def capture(*args, **kwargs):
        candidates = original(*args, **kwargs)
        fused.append(candidates)
        return candidates

    with patch.object(engine_module, "weighted_rrf", capture):
        retrieved = engine.retrieve(query, timestamp=stamp, read_only=True,
                                    top_k=TOP_K, token_budget=TOKEN_BUDGET)
    candidates = fused[0] if fused else []
    assert len(fused) == (retrieved["route"] == "TASK_POISON_CIRCLE")
    assert len({fact["id"] for fact in candidates}) == len(candidates)
    assert all(left["rrf_score"] >= right["rrf_score"] for left, right in zip(candidates, candidates[1:]))
    baseline = _fit_context(candidates, TOP_K, TOKEN_BUDGET)
    for facts in (retrieved["ranked"], baseline):
        assert len(facts) <= TOP_K
        assert estimate_tokens("\n".join(fact["text"] for fact in facts)) <= TOKEN_BUDGET
        assert bool(facts) == bool(candidates)
    return retrieved, baseline, candidates


def compare(item):
    engine = CognitiveMemoryEngine(llm=MockLLMClient(), vector=HashedTfidfEmbedder())
    with closing(engine.store):
        for turn in item["turns"]:
            engine.add(turn["text"], timestamp=turn["ts"])
        stamp = parse_timestamp(item.get("q_time"), max((turn["ts"] for turn in item["turns"]), default=0) + 1)
        retrieved, baseline, candidates = paired_retrieve(engine, question_stem(item), stamp)
        return {"id": _item_key(item), "tid": item["tid"], "category": item["category"],
                "ground_truth": item["ground_truth"], "evidence_steps": item["evidence_steps"],
                "route": retrieved["route"], "n_outer": retrieved["outer_triggered_count"],
                "n_outer_exception": sum(fact.get("isolated") or fact.get("is_conflict_or_override")
                                         or fact.get("historical") for fact in retrieved["outer"]),
                "n_belt_compressed": retrieved["belt_compressed_count"], "candidates": len(candidates),
                "poison": measure(item, retrieved["ranked"]), "rrf": measure(item, baseline)}


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


def membench(data_dir, reference, phase_a=False):
    seeds = [42, 43, 44, 45, 46]
    results = {}
    cohorts = {}
    for seed in seeds:
        items = load_sample(5, seed, data_dir)
        assert len(items) == 20
        keys = [_item_key(item) for item in items]
        assert len(keys) == len(set(keys))
        cohorts[seed] = keys
        for item, key in zip(items, keys):
            if key not in results:
                results[key] = compare(item)
        print(f"seed {seed}: 20 samples, {len(results)} unique total", flush=True)
    rows = list(results.values())
    primary = [results[key] for key in cohorts[42]]
    assert len(rows) == 99
    if reference:
        previous = json.loads(reference.read_text())
        assert {int(seed): keys for seed, keys in previous["cohorts"].items()} == cohorts
        for row, logged in zip(rows, previous["items"]):
            assert row["id"] == logged["id"]
            if not phase_a:
                assert row["rrf"] == logged["rrf"], row["id"]
        if phase_a:
            print("Before Phase A, exact same cohort / two-channel RRF:", json.dumps(previous["all_unique"]["rrf"]))
    report = {"top_k": TOP_K, "token_budget": TOKEN_BUDGET, "mode": "mock", "seeds": seeds,
              "seed_42": summary(primary), "all_unique": summary(rows),
              "by_category": {category: summary([row for row in rows if row["category"] == category]) for category in CATEGORIES},
              "cohorts": cohorts, "items": rows}
    for label in ("seed_42", "all_unique", "by_category"):
        print(label, json.dumps(report[label], ensure_ascii=False))
    return report


def stress_case(index):
    engine = CognitiveMemoryEngine(llm=MockLLMClient(), vector=HashedTfidfEmbedder())
    with closing(engine.store):
        person = ["老周", "张敏", "李楠", "王宁", "苏晚"][index % 5]
        food = ["花生", "牛奶", "海鲜", "芝麻", "杏仁"][index // 5]
        tick = 0

        def add(text):
            nonlocal tick
            result = engine.add(f"[step {tick}] {text}", timestamp=tick)
            tick += 1
            return result

        habit = f"{person}每周三下午三点开周会，周会餐饮安排照常。"
        habit_id = add(habit)["operations"][0]["line_id"]
        for _ in range(5 + index % 5):
            add(habit)
        background_ids = set()
        for note in range(14 + index % 5):
            marker = "甲乙丙丁戊己庚辛壬癸"[note // 10] + "甲乙丙丁戊己庚辛壬癸"[note % 10]
            text = f"{person}周会餐饮方案{marker}的预算为{1000 + note}元，周会餐饮安排计划讨论与通知下达。"
            for _ in range(2):
                output = add(text)
            assert output["operations"][0]["action"] == "NOOP"
            background_ids.add(output["operations"][0]["target_line_id"])
        for note in range(12):
            add(f"通用资讯栏目报道第{note:02d}期城市公共服务进展与社区节庆活动。")
        exception = add(f"{person}对{food}过敏。")
        exception_id = exception["operations"][0]["line_id"]
        assert engine.store.get_line(exception_id)["isolated"]
        assert consolidation_count(engine.store.get_line(habit_id)) == 6 + index % 5
        queries = (f"{person}周会餐饮安排有什么需要注意的例外？", f"请确认{person}周会餐饮的特别情况。",
                   f"{person}周会餐饮方案有哪些限制？", f"安排周会餐饮时，{person}有什么饮食限制？")
        query = queries[index // 5]
        retrieved, baseline, candidates = paired_retrieve(engine, query, tick + 1)
        c_max = max(consolidation_count(fact) for fact in candidates if not is_noise(fact))
        ring = retrieved["poison_circle"]
        belt = {fact["id"] for fact in candidates
                if ring["radius_r"] < compute_bipolar_radius(fact, c_max) < ring["radius_out"]}
        assert exception_id in {fact["id"] for fact in candidates}
        assert background_ids <= belt and engine.store.get_line(habit_id)["hit_count"] == 0

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
                "outer_triggered_count": retrieved["outer_triggered_count"],
                "exception_triggered": any(fact["id"] == exception_id for fact in retrieved["outer"])}


def stress():
    rows = [stress_case(index) for index in range(20)]
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
    return {"n": len(rows), "outer_triggered_count": outer, "exception_triggered_count": exceptions,
            "aggregate": aggregate, "items": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("membench", "stress"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--reference", type=Path, help="Previous 99-item report, to verify the same samples and RRF baseline.")
    parser.add_argument("--phase-a", action="store_true", help="Compare the same cohort without asserting the new four-channel RRF equals the old two-channel baseline.")
    args = parser.parse_args()
    if args.mode == "membench" and not args.data_dir:
        parser.error("--data-dir is required in membench mode")
    report = membench(args.data_dir, args.reference, args.phase_a) if args.mode == "membench" else stress()
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
