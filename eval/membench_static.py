"""Evaluate 100 cached MemBench items per category with a shared fact retrieval."""
import argparse
import json
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from eval.bipolar_ablation import compare
from eval.membench.catalog import CATEGORIES
from eval.membench.items import normalize_trajectory
from eval.membench.load import _iter_scenarios
from eval.metrics import retrieval_report
from eval.static_report import official_table, write_result


def _run(item):
    return compare(item, compare_consolidation=True)


def sample_items(data_dir, seed, per_category):
    rng = random.Random(seed)
    items, missing = [], []
    for category, files in CATEGORIES.items():
        reservoir, seen = [], 0
        for relative in files:
            path = data_dir / relative
            if not path.is_file():
                missing.append(relative)
                continue
            for scenario, trajectories in _iter_scenarios(json.loads(path.read_text())):
                for raw in trajectories:
                    if not isinstance(raw, dict) or "QA" not in raw:
                        continue
                    seen += 1
                    slot = seen - 1 if seen <= per_category else rng.randrange(seen)
                    if slot < per_category:
                        candidate = (raw, path.stem, scenario)
                        if slot == len(reservoir):
                            reservoir.append(candidate)
                        else:
                            reservoir[slot] = candidate
        items.extend(normalize_trajectory(raw, category=category, subtype=subtype, scenario=scenario)
                     for raw, subtype, scenario in reservoir)
    counts = Counter(item["category"] for item in items)
    if any(counts[category] != per_category for category in CATEGORIES):
        raise ValueError(f"requested {per_category} per category, selected {counts}")
    return items, missing


def run(data_dir, seed, workers, per_category):
    items, missing = sample_items(data_dir, seed, per_category)
    if workers == 1:
        rows = list(map(_run, items))
    else:
        with ProcessPoolExecutor(workers) as pool:
            rows = list(pool.map(_run, items))

    def metrics(rows, key):
        pairs = [row[key] for row in rows]
        return {mode: retrieval_report(pairs, mode) for mode in ("bipolar", "rrf")}

    result = {"benchmark": "MemBench", "backend": "hashed", "mode": "mock", "seed": seed, "items": rows, "uncached_subtypes": missing}
    for label, key in (("default", "paired"), ("consolidated", "consolidated_pair")):
        result[label] = {"overall": metrics(rows, key),
                         "by_category": {cat: metrics([row for row in rows if row["category"] == cat], key)
                                         for cat in CATEGORIES}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--per-category", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.data_dir, args.seed, args.workers, args.per_category)
    write_result(report, args.out, "\n\n".join(f"## {key}\n\n{official_table(report[key])}"
                                                  for key in ("default", "consolidated")))


if __name__ == "__main__":
    main()
