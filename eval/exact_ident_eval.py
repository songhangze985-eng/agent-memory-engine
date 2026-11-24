"""Paired synthetic technical queries with adjacent identifiers and missing versions."""

import argparse
import json
import random
from contextlib import closing
from pathlib import Path

from eval.paired import LABELS, add_backend_argument, cli_backend, evaluate_pair, make_engine
from agentmem_core.identifiers import extract_identifiers

_GROUPS = {
    "version": ("v3.10", "v3.11", "v3.12", "v3.11.1", "3.11.2", "v3.1", "v2.4.1-rc3",
                "v2.4.1-rc4", "v2.4.1+build.2", "v2.4.1", "v10.12", "v10.2"),
    "code_symbol": ("CognitiveMemoryEngine", "CognitiveMemoryEngineV2", "collapse_and_scan_z_axis",
                    "collapse_and_scan_x_axis", "memengine/store.py", "memengine/store_v2.py"),
    "error_code": ("ERR_403", "ERR_404", "HTTP_502", "HTTP_503"),
    "project_token": ("agent-memory-engine", "agent-memory-indexer", "memory-search-kit", "memory-search-tools"),
    "quoted_name": ("《青梧发布手册》", "《青梧回滚手册》", '"Atlas Blue Release"', '"Atlas Blue Recovery"'),
}
_ABSENT = ("v3.13", "v3.11.9", "v3.09", "v3.1.1", "v2.4.2", "v2.4.1-rc5",
           "v2.4.1+build.9", "v10.3", "v10.12.1", "v10.21")


def _norm(literal):
    return extract_identifiers(literal)[0].ident_norm


def _score(facts, gold, version):
    ids = [fact["id"] for fact in facts]
    rank = ids.index(gold) + 1 if gold in ids else None
    versions = {item.ident_norm for item in extract_identifiers(facts[0]["plain"])
                if item.ident_type == "version"} if facts else set()
    return {"precision_at_1": rank == 1 if gold else None, "reciprocal_rank": 1 / rank if rank else 0.0,
            "version_confusion": bool(versions - {version} and version not in versions) if version else None,
            "empty": not facts, "ranking": ids}


def run(seed=42, backend="hashed"):
    rng = random.Random(seed)
    corpus = [(kind, literal) for kind, literals in _GROUPS.items() for literal in literals]
    rng.shuffle(corpus)
    engine = make_engine("mock", backend=backend)
    rows = []
    with closing(engine.store), engine.store.transaction():
        block = engine.store.create_block("Technical release records", 0)
        ids = {}
        for ordinal, (kind, literal) in enumerate(corpus, 1):
            body = ("For the release configuration, debug workflow, compatibility and recovery procedure, "
                    f"consult {literal}. Verified release flag is option {ordinal}.")
            if ordinal % 3 == 0:
                body = f"{literal} verified option {ordinal}."
            turn_id = engine.store.append_turn(body, ordinal)
            ids[kind, literal] = engine.store.insert_fact(block_id=block, formatted=body, plain=body,
                                                           ingest_time=ordinal, source_turn_ids=[turn_id])
        for kind, literal in [*corpus, *(("missing_version", literal) for literal in _ABSENT)]:
            gold = ids.get((kind, literal))
            version = _norm(literal) if kind in {"version", "missing_version"} else None
            query = f"What is the release configuration and debug recovery procedure for {literal}?"
            pair = evaluate_pair(engine, query, len(corpus) + 1)
            rows.append({"type": kind, "identifier": literal, "query": query, "gold_id": gold,
                         "bipolar": _score(pair.retrieved["ranked"], gold, version),
                         "rrf": _score(pair.baseline, gold, version)})

    def mean(values):
        return sum(values) / len(values) if values else None

    def summarize(subset, mode):
        found = [row[mode] for row in subset]
        positives = [row[mode] for row in subset if row["gold_id"]]
        return {"n": len(found), "precision_at_1": mean([row["precision_at_1"] for row in positives]),
                "mrr_at_10": mean([row["reciprocal_rank"] for row in positives]),
                "version_confusion_rate": mean([row["version_confusion"] for row in found if row["version_confusion"] is not None]),
                "missing_version_empty_rate": mean([row[mode]["empty"] for row in subset if not row["gold_id"]])}

    groups = {"Overall": rows, **{kind: [row for row in rows if row["type"] == kind]
                                 for kind in (*_GROUPS, "missing_version")}}
    return {"dataset_source": "eval/exact_ident_eval.py (synthetic)", "mode": "mock", "backend": backend,
            "seed": seed, "metrics": {group: {mode: summarize(subset, mode) for mode in LABELS}
                                      for group, subset in groups.items()}, "items": rows}


def static_catalog():
    versions = ("v3.10", "v3.11", "v3.11.1", "v3.12-rc1", "v3.12", "v3.12.1")
    groups = {
        "version": (*versions, *(f"v4.{i}.1" for i in range(9))),
        "error": ("ERR_403", "ERR_404", "ECONNRESET", *(f"ERR_SYNC_{i:02}" for i in range(12))),
        "symbol_path": ("CognitiveMemoryEngine", "memengine/store.py", "memengine/storage.py",
                        *(f"src/worker_{i:02}.py" for i in range(12))),
        "project": ("agent-memory-engine", *(f"agent-memory-worker-{i:02}" for i in range(14))),
        "config": tuple(f"MEMORY_LIMIT_{i:02}" for i in range(15)),
        "procedural": tuple(f"RECOVERY_FLOW_{i:02}" for i in range(15)),
    }
    absent = [("missing_version", f"v4.{i}.2") for i in range(15)]
    absent += [("missing_symbol", f"ERR_SYNC_{i:02}") for i in range(20, 35)]
    return [(kind, literal) for kind, literals in groups.items() for literal in literals], absent


def run_static(seed=42, *, evaluate=None):
    from eval.metrics import mean

    engine = make_engine("mock")
    catalog, absent = static_catalog()
    random.Random(seed).shuffle(catalog)
    rows = []
    with closing(engine.store), engine.store.transaction():
        ids = {}
        for ordinal, (kind, literal) in enumerate(catalog, 1):
            body = f"{literal} uses recovery checkpoint {ordinal} and the verified transaction workflow."
            turn = engine.store.append_turn(body, ordinal)
            ids[kind, literal] = engine.remember(body, source_turn_ids=[turn], timestamp=ordinal,
                                                memory_type="procedural" if kind == "procedural" else "fact",
                                                trigger=f"Troubleshoot {literal}" if kind == "procedural" else "")
        if evaluate is not None:
            return evaluate(engine, [(kind, literal, ids.get((kind, literal))) for kind, literal in catalog + absent])
        for kind, literal in catalog + absent:
            query = f"`{literal}`" if kind == "missing_symbol" else f"Which configuration or recovery procedure applies to {literal}?"
            pair = evaluate_pair(engine, query, len(catalog) + 1)
            gold = ids.get((kind, literal))
            entry = {"type": kind, "identifier": literal, "gold_id": gold, "query": query}
            for mode, facts in (("bipolar", pair.retrieved["ranked"]), ("rrf", pair.baseline)):
                score = _score(facts, gold, _norm(literal) if "version" in kind else None)
                symbols = {item.ident_norm for item in extract_identifiers(facts[0]["plain"])} if facts else set()
                score.update(symbol_confusion=bool(symbols and _norm(literal) not in symbols),
                             procedural_top3=any(fact["id"] == gold and fact["memory_type"] == "procedural" for fact in facts[:3])
                             if kind == "procedural" else None)
                entry[mode] = score
            rows.append(entry)

    def summarize(subset, mode):
        positives = [r[mode] for r in subset if r["gold_id"]]
        negatives = [r[mode] for r in subset if not r["gold_id"]]
        return {"n": len(subset), "precision_at_1": mean(r["precision_at_1"] for r in positives),
                "mrr_at_10": mean(r["reciprocal_rank"] for r in positives),
                "symbol_confusion": mean(r[mode]["symbol_confusion"] for r in subset),
                "negative_empty": mean(r["empty"] for r in negatives),
                "procedural_top3": mean(r[mode]["procedural_top3"] for r in subset)}

    subsets = {"Overall": rows, **{kind: [r for r in rows if r["type"] == kind] for kind in sorted({r["type"] for r in rows})}}
    return {"benchmark": "Exact Ident Static", "seed": seed, "backend": "hashed", "items": rows,
            "metrics": {group: {mode: summarize(subset, mode) for mode in LABELS} for group, subset in subsets.items()}}


def static_table(report):
    from eval.static_report import number, table

    metrics = ("precision_at_1", "mrr_at_10", "symbol_confusion", "negative_empty", "procedural_top3")
    return table(("类别", "配置", "题数", "Precision@1", "MRR@10", "串台率", "缺失空返", "Procedural Top-3"),
                 ((group, LABELS[mode], values["n"], *(number(values[key], ".4f" if key == "mrr_at_10" else ".2%")
                    for key in metrics)) for group, sides in report["metrics"].items() for mode, values in sides.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("exact-ident-ablation.json"))
    add_backend_argument(parser)
    parser.add_argument("--legacy", action="store_true", help="Run the original frozen 30+10 cohort")
    args = parser.parse_args()
    if not args.legacy:
        from eval.static_report import write_result
        if args.backend != "hashed":
            parser.error("the static 120-question suite is hashed-only")
        report = run_static(args.seed)
        write_result(report, args.out, static_table(report))
        full = report["metrics"]["Overall"]["bipolar"]
        if (full["precision_at_1"], full["mrr_at_10"], full["symbol_confusion"],
                full["negative_empty"], full["procedural_top3"]) != (1, 1, 0, 1, 1):
            parser.exit(1, "Static acceptance failed: expected exact ranking, no symbol confusion, all missing symbols empty, and procedural hits.\n")
        return
    backend = cli_backend(parser, args.backend)
    report = run(args.seed, backend)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exact identifiers: {len(report['items'])} questions; mode=mock; backend={backend}; seed={args.seed}")
    print("| Group | System | n | Precision@1 | MRR@10 | Version confusion | Missing-version empty |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for group, metrics in report["metrics"].items():
        for mode, values in metrics.items():
            fields = ["—" if value is None else f"{value:.4f}" for key, value in values.items() if key != "n"]
            print(f"| {group} | {LABELS[mode]} | {values['n']} | " + " | ".join(fields) + " |")
    print(f"Raw JSON: {args.out.resolve()}")
    overall = report["metrics"]["Overall"]
    if (overall["bipolar"]["precision_at_1"] <= overall["rrf"]["precision_at_1"]
            or overall["bipolar"]["version_confusion_rate"] != 0
            or overall["bipolar"]["missing_version_empty_rate"] != 1):
        parser.exit(1, "Acceptance failed: expected better exact ranking, no version confusion, and all missing versions empty.\n")


if __name__ == "__main__":
    main()
