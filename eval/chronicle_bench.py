"""Pair equally budgeted single-query and graph recall on official data."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
import hashlib
from pathlib import Path
from statistics import mean

from eval.evidence_ablation import OFFICIAL_SHA256, measure, read_dataset, retain_source_turn
from eval.locomo.run import groups as locomo_groups
from eval.longmemeval.run import groups as longmem_groups
from eval.metrics import retrieval_report
from eval.paired import make_engine
from eval.static_report import number, table, write_result

MODES = ("single_query", "reconstructive")
TOKEN_BUDGET = 2048


def evaluate(group):
    turns, questions = group
    engine = make_engine("mock")
    with closing(engine.store):
        raw_ids = {}
        for ordinal, turn in enumerate(turns):
            record = retain_source_turn(engine, ordinal, turn)
            raw_ids[record["turn_id"]] = turn
        rows = []
        for index, question in enumerate(questions):
            row = {"id": question.id, "category": question.category, "unanswerable": question.unanswerable,
                   "evidence": sorted(question.evidence)}
            for mode in MODES[::(-1 if index % 2 else 1)]:
                context = engine.build_context(question.text, reference_time=question.timestamp, mode=mode,
                                               token_budget=TOKEN_BUDGET, packing_mode="token_budget")
                facts = [item for item in context["items"] if item["kind"] == "fact"]
                archive = {}
                if question.latest_session:
                    ids = {fact["id"] for fact in facts if fact["status"] == "active" and fact["revision"]}
                    for revision in engine.store.historical_versions(ids, scope=("default", "default", None)):
                        for fact in facts:
                            if fact["root_fact_id"] == revision["root_fact_id"]:
                                archive.setdefault(fact["id"], []).append(revision)
                scores = measure(facts, turns, question, archive, raw_ids=raw_ids)
                all_refs = set().union(*(set(x) for x in scores["ranked_turns"]))
                gold = question.evidence if not question.unanswerable else ()
                row[mode] = {**scores, "budget_recall": len(all_refs.intersection(gold))/len(gold) if gold else None,
                             "injected_tokens": context["token_breakdown"]["total"],
                             "latency_ms": context["retrieval_ms"], "selected_facts": len(facts),
                             "linked_facts": sum(bool(f.get("activation")) for f in facts)}
            rows.append(row)
        return rows


def summarize(rows, mode):
    metrics = retrieval_report(rows, mode)
    budget = [row[mode]["budget_recall"] for row in rows if row[mode]["budget_recall"] is not None]
    metrics.update(budget_recall=mean(budget) if budget else None,
                   selected_facts=mean(row[mode]["selected_facts"] for row in rows),
                   linked_facts=mean(row[mode]["linked_facts"] for row in rows))
    return metrics


def render(report):
    rows = []
    for name, modes in report["by_category"].items():
        for mode, metrics in modes.items():
            rows.append((name, mode, metrics["n"], number(metrics["recall_at_10"], ".2%"),
                         number(metrics["budget_recall"], ".2%"), number(metrics["mrr_at_10"], ".4f"),
                         number(metrics["avg_injected_tokens"], ".1f"), number(metrics["p95_retrieval_ms"], ".2f"),
                         number(metrics["selected_facts"], ".1f"), number(metrics["linked_facts"], ".1f"),
                         number(metrics["update_dominance"], ".2%")))
    return table(("类别", "模式", "N", "Recall@10", "Recall@Budget", "MRR@10", "Token", "P95 ms", "事实数", "图事实", "更新支配"), rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("locomo", "longmemeval"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sample", type=int)
    args = parser.parse_args()
    name, groups = ("LoCoMo", locomo_groups) if args.benchmark == "locomo" else ("LongMemEval-S", longmem_groups)
    run(name, groups, args.data_dir, args.out, args.sample, args.seed, args.workers)


def run(name, groups, dataset, out, sample, seed, workers):
    checksum = hashlib.sha256()
    with dataset.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    digest = checksum.hexdigest()
    if digest != OFFICIAL_SHA256[name]:
        raise ValueError("official dataset checksum mismatch")
    corpus = groups(read_dataset(dataset), sample, seed)
    items = []
    with ProcessPoolExecutor(workers) as pool:
        for batch in pool.map(evaluate, corpus):
            items.extend(batch)
            if name == "LoCoMo" or len(items) % 25 == 0:
                print(f"{name}: {len(items)} paired questions complete", flush=True)
    by_category = {"Overall": items, **{cat: [row for row in items if row["category"] == cat]
                                       for cat in sorted({row["category"] for row in items})}}
    report = {"benchmark": name, "sha256": digest, "seed": seed, "mode": "mock", "backend": "hashed",
              "token_budget": TOKEN_BUDGET, "items": items,
              "by_category": {name: {mode: summarize(rows, mode) for mode in MODES} for name, rows in by_category.items()}}
    write_result(report, out, render(report))


if __name__ == "__main__":
    main()
