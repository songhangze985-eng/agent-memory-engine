"""Paired, zero-API LoCoMo multi-hop evaluation on one ledger per conversation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from pathlib import Path
from random import Random

from eval.evidence_ablation import measure, read_dataset
from eval.locomo.run import groups
from eval.metrics import retrieval_report
from eval.paired import make_engine
from eval.static_report import number, table, write_result

MODES = ("single_query", "reconstructive")


def evaluate(group):
    turns, questions = group
    engine = make_engine("mock")
    with closing(engine.store):
        raw_ids = {}
        for i, turn in enumerate(turns):
            row = engine.add(f"[step {i}] {turn.text}", timestamp=turn.timestamp, session_id=turn.session, role=turn.role)
            raw_ids[row["turn_id"]] = turn
        rows = []
        for i, question in enumerate(questions):
            row = {"id": question.id, "unanswerable": False, "evidence": sorted(question.evidence)}
            for mode in MODES[::(-1 if i % 2 else 1)]:
                context = engine.build_context(question.text, timestamp=question.timestamp, mode=mode, token_budget=2048)
                facts = [item for item in context["items"] if item["kind"] == "fact"]
                row[mode] = {**measure(facts, turns, question, {}, raw_ids=raw_ids),
                             "latency_ms": context["retrieval_ms"], "injected_tokens": context["token_breakdown"]["total"],
                             "reconstructed": context["reconstructed"], "bridge_entities": context["bridge_entities"]}
            rows.append(row)
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cohorts = [(turns, [q for q in questions if q.category == "1 multi-hop"])
               for turns, questions in groups(read_dataset(args.data_dir), None, args.seed)]
    Random(args.seed).shuffle(cohorts)
    with ProcessPoolExecutor(args.workers) as pool:
        items = [row for batch in pool.map(evaluate, cohorts) for row in batch]
    overall = {mode: retrieval_report(items, mode) for mode in MODES}
    report = {"benchmark": "LoCoMo Multi-hop", "mode": "mock", "backend": "hashed",
              "seed": args.seed, "overall": overall, "items": items}
    metrics = (("recall_at_10", ".2%"), ("mrr_at_10", ".4f"), ("avg_injected_tokens", ".1f"),
               ("p50_retrieval_ms", ".3f"), ("p95_retrieval_ms", ".3f"))
    write_result(report, args.out, table(("配置", "题数", "R@10", "MRR@10", "平均 Token", "P50 ms", "P95 ms", "二跳触发"),
        [(mode, len(items), *(number(overall[mode][key], fmt) for key, fmt in metrics),
          sum(row[mode]["reconstructed"] for row in items)) for mode in MODES]))


if __name__ == "__main__":
    main()
