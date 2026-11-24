"""Measure the deployed evidence funnel and diagnostic, uncapped RRF candidates."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, closing
from functools import partial
import hashlib
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from eval.evidence_ablation import OFFICIAL_SHA256, fact_turns, measure, read_dataset, retain_source_turn, sample_rows
from eval.locomo.run import groups as locomo_groups
from eval.longmemeval.run import groups as longmem_groups
from eval.membench.items import parse_timestamp, question_stem
from eval.membench.load import load_sample
from eval.membench.run import _item_key, evidence_turn, paired_question
from eval.membench_static import sample_items
from eval.metrics import evidence_metrics, mean, percentile
from eval.paired import evaluate_pair, make_engine
from eval.static_report import number, table, write_result
from agentmem_core.distance import BELT
from agentmem_core.gate import precision_gate
from agentmem_core.linguistics import stem_terms
from agentmem_core.retrieval import operators
from agentmem_core.temporal import competition_terms
from agentmem_core.vector import weighted_rrf


ARMS = ("semantic", "lexical", "entity_graph", "temporal", "reconstructive")
INTERVENTIONS = (*ARMS, "single_query")
WEIGHTS = dict(zip(ARMS, (.40, .35, .15, .25, .12)))
SCOPE = ("default", "default", None)


def refs(facts, raw_ids):
    return [{turn.ref for turn in fact_turns(row, (), raw_ids)} for row in facts]


def ordered_union(*channels):
    return [{"id": ident} for ident in dict.fromkeys(row["id"] for channel in channels for row in channel)]


def coverage(ranked, gold, k=None):
    return len(set().union(*ranked[:k]) & gold) / len(gold) if gold else None


def recall_options(question, rich, mode):
    return dict(mode=mode, reference_time=question.timestamp, packing_mode="token_budget") if rich else {}


def probe(engine, question, *, rich, mode):
    """Raise the per-arm depth only in this observation pass."""
    hits, fused, gates = defaultdict(list), [], []
    original_depth = engine.per_channel_k

    def capture(name, call):
        def wrapped(*args, **kwargs):
            result = call(*args, **kwargs)
            hits[name].append(result)
            return result
        return wrapped

    def fuse(*args, **kwargs):
        ranked = weighted_rrf(*args, **kwargs)
        fused.append(ranked)
        return ranked

    def gate(ranked, **kwargs):
        kept, stats = precision_gate(ranked, **kwargs)
        gates.append((ranked, kept))
        return kept, stats

    with ExitStack() as stack:
        for target, name, arm in (
            (engine, "search_vector", "semantic"),
            (engine.store, "search_lexical", "lexical"),
            (engine.store, "search_entity_inverted", "entity_graph"),
            (engine.store, "search_temporal_range", "temporal"),
            (engine.store, "occurrence_candidates", "occurrence"),
        ):
            stack.enter_context(patch.object(target, name, capture(arm, getattr(target, name))))
        stack.enter_context(patch.object(operators, "spreading_activation",
                                         capture("activation", operators.spreading_activation)))
        stack.enter_context(patch("agentmem_core.engine.weighted_rrf", fuse))
        stack.enter_context(patch("agentmem_core.engine.precision_gate", gate))
        stack.callback(setattr, engine, "per_channel_k", original_depth)
        engine.per_channel_k = 100
        if engine.reranker.is_calibrated:
            stack.enter_context(patch.object(engine, "reranker", engine.local_reranker))
        engine.retrieve(question.text, timestamp=question.timestamp, top_k=10, token_budget=2048,
                        read_only=True, **recall_options(question, rich, mode))
    channels = {arm: hits[arm][0] if hits[arm] else [] for arm in ARMS}
    activation = hits["activation"][0] if hits["activation"] else []
    occurrence = hits["occurrence"][0] if hits["occurrence"] else []
    if activation:
        channels["entity_graph"] = weighted_rrf(channels["entity_graph"], activation, w_vec=.08, w_hash=.46)
    if occurrence:
        channels["temporal"] = weighted_rrf(channels["temporal"], occurrence, w_vec=.14, w_hash=.37)
    channels["reconstructive"] = ordered_union(*hits["entity_graph"][1:], *hits["lexical"][1:])
    return channels, fused[-1] if fused else [], gates[-1] if gates else ([], [])


def ablate_recall(engine, question, gold, raw_ids, mode):
    interventions = {
        "semantic": [(engine, "search_vector", [])],
        "lexical": [(engine.store, "search_lexical", [])],
        "entity_graph": [(engine.store, "search_entity_inverted", []), (operators, "spreading_activation", [])],
        "temporal": [(engine.store, "occurrence_candidates", []), (engine.store, "search_temporal_range", [])],
        "reconstructive": [(operators, "external_entities", []), (operators, "bridge_feedback", ""),
                           (operators, "discovered_windows", ())],
        "single_query": [],
    }
    result = {}
    for arm, methods in interventions.items():
        with ExitStack() as stack:
            for target, method, value in methods:
                stack.enter_context(patch.object(target, method, return_value=value))
            query_mode = "single_query" if arm == "single_query" else mode
            bundle = engine.build_context(question.text, token_budget=2048, include_short_term=False,
                                          **recall_options(question, True, query_mode))
        rows = [row for row in bundle["items"] if row["kind"] == "fact"]
        result[arm] = evidence_metrics(refs(rows, raw_ids), gold)
    return result


def row_for(engine, question, raw_ids, stored, *, rich, mode, ablate):
    if rich:
        start = perf_counter()
        context = engine.build_context(question.text, token_budget=2048, include_short_term=False,
                                       **recall_options(question, rich, mode))
        elapsed = (perf_counter() - start) * 1000
    else:
        pair = evaluate_pair(engine, question.text, question.timestamp)
        context, elapsed = pair.context, pair.full_ms
    selected = [row for row in context["items"] if row["kind"] == "fact"]
    ranked = refs(selected, raw_ids)
    mapped_raw = stored["raw_refs"]
    gold = question.evidence if not question.unanswerable else frozenset()
    arms, candidates, (pre_gate, kept) = probe(engine, question, rich=rich, mode=mode)
    candidate_refs = refs(candidates, raw_ids)
    before = {row["id"]: mentioned for row, mentioned in zip(pre_gate, refs(pre_gate, raw_ids))}
    kept_ids = {row["id"] for row in kept}
    graph_gold = {row["id"] for row in pre_gate if row.get("graph_depth") and before[row["id"]] & gold}
    graph_belt = graph_gold & {row["id"] for row in pre_gate if row["zone"] == BELT}

    def channel_refs(rows):
        return [stored["by_fact"].get(row["id"], set()) for row in rows]

    def merged(exclude=None):
        return weighted_rrf([], [], extra_channels=tuple((WEIGHTS[arm], arms[arm]) for arm in ARMS if arm != exclude))

    top10 = set().union(*ranked[:10])
    top50 = set().union(*candidate_refs[:50])
    missing = gold - top10
    update = None
    if question.latest_session:
        active = {fact["id"]: fact["root_fact_id"] for fact in selected[:10] if fact["status"] == "active" and fact["revision"]}
        history = engine.store.historical_versions(active, scope=SCOPE)
        archive = {ident: [row for row in history if row["root_fact_id"] == root] for ident, root in active.items()}
        update = measure(selected[:10], (), question, archive, raw_ids=raw_ids).get("update_dominates")
    missed_lexical = [stored["text"].get(ref, "") for ref in missing - top50 if ref in mapped_raw]
    q_lexical = competition_terms(question.text)
    q_stems = stem_terms(question.text) & {stem for term in q_lexical for stem in stem_terms(term)}
    spelling_gap = sum(not (q_lexical & competition_terms(text)) and bool(q_stems & stem_terms(text))
                       for text in missed_lexical)
    no_word_overlap = sum(not (q_lexical & competition_terms(text) or q_stems & stem_terms(text))
                          for text in missed_lexical)
    union = merged()
    return {
        "id": question.id, "category": question.category, "evidence": sorted(question.evidence),
        "update_dominates": update,
        "negative": question.unanswerable, "unanswerable": question.unanswerable,
        "gold_n": len(gold), "raw_store": len(gold & mapped_raw) / len(gold) if gold else None,
        "fact_store": len(gold & stored["fact_refs"]) / len(gold) if gold else None,
        "candidate_depth": len(candidates),
        **{f"r{limit}": coverage(candidate_refs, gold, limit) for limit in (20, 50, 100)},
        **evidence_metrics(ranked, gold),
        "budget_recall": coverage([*ranked, {raw_ids[key].ref for key in context["evidence_turn_ids"] if key in raw_ids}], gold),
        "a": len(missing - top50) / len(gold) if gold else None,
        "b": len(missing & top50) / len(gold) if gold else None,
        "packing_loss": len(set().union(*refs(kept[:10], raw_ids)) & missing) / len(gold) if gold else None,
        "tokens": context["token_breakdown"]["total"], "latency_ms": elapsed, "empty": not ranked,
        "arm50": {arm: coverage(channel_refs(rows), gold, 50) for arm, rows in arms.items()},
        "minus_arm50": {arm: coverage(channel_refs(merged(arm)), gold, 50) for arm in ARMS},
        "fused_arm50": coverage(channel_refs(union), gold, 50),
        "graph_gold": len(graph_gold), "graph_belt": len(graph_belt), "graph_belt_dropped": len(graph_belt - kept_ids),
        "lexical_diagnosed_n": len(missed_lexical), "stem_only": spelling_gap, "no_word_overlap": no_word_overlap,
        "without": ablate_recall(engine, question, gold, raw_ids, mode) if ablate and gold else {},
    }


def inspect_store(engine, raw_ids):
    users, agents, session = SCOPE
    sources = engine.store.get_turn_window(list(raw_ids), window=0, by_id=True,
                                          user_id=users, agent_id=agents, session_id=session)
    facts = engine.store.snapshot(scope=SCOPE, include_superseded=True)
    mapped = refs(facts, raw_ids)
    return {"by_fact": {row["id"]: ids for row, ids in zip(facts, mapped)}, "fact_refs": set().union(*mapped),
            "raw_refs": {raw_ids[row["turn_id"]].ref for row in sources},
            "text": {raw_ids[row["turn_id"]].ref: row["text"] for row in sources}}


def dialogue_group(group, *, rich, backend, mode, ablate):
    turns, questions = group
    engine = make_engine("mock", backend=backend)
    with closing(engine.store):
        sources = {}
        for index, turn in enumerate(turns):
            added = retain_source_turn(engine, index, turn, chronicle=rich)
            sources[added["turn_id"]] = turn
        stored = inspect_store(engine, sources)
        return [row_for(engine, question, sources, stored, rich=rich, mode=mode, ablate=ablate) for question in questions]


def mem_item(item, *, rich, backend, mode, ablate):
    engine = make_engine("mock", allow_casual=True, backend=backend)
    with closing(engine.store):
        sources = {}
        for index, turn in enumerate(item["turns"]):
            added = engine.retain(turn["text"], timestamp=turn["ts"])
            sources[added["turn_id"]] = evidence_turn(turn, index)
        stamp = parse_timestamp(item.get("q_time"), max((turn["ts"] for turn in item["turns"]), default=0) + 1)
        question = paired_question(item, stamp)
        row = row_for(engine, question, sources, inspect_store(engine, sources), rich=rich, mode=mode, ablate=ablate)
        return [{**row, "id": _item_key(item), "category": item["category"], "evidence_steps": item["evidence_steps"]}]


def summarize(items):
    buckets = {"Overall": items, **{name: [row for row in items if row["category"] == name]
                                    for name in sorted({row["category"] for row in items})}}
    summary = {}
    keys = ("raw_store", "fact_store", "r100", "r50", "r20", "recall_at_10", "mrr_at_10",
            "budget_recall", "a", "b", "packing_loss", "tokens", "fused_arm50")
    for name, rows in buckets.items():
        stats = {key: mean(row[key] for row in rows) for key in keys}
        stats.update(n=len(rows), labeled=sum(bool(row["gold_n"]) for row in rows),
                     candidate_depth=mean(row["candidate_depth"] for row in rows),
                     p95_ms=percentile([row["latency_ms"] for row in rows], .95),
                     negative_empty=sum(row["empty"] for row in rows if row["negative"]),
                     negative_n=sum(row["negative"] for row in rows),
                     arms={arm: {"alone": mean(row["arm50"][arm] for row in rows),
                                 "without": mean(row["minus_arm50"][arm] for row in rows)} for arm in ARMS})
        if any(row.get("without") for row in rows):
            stats["leave_one_out"] = {arm: {key: mean(row.get("without", {}).get(arm, {}).get(key) for row in rows)
                                            for key in ("recall_at_10", "mrr_at_10")} for arm in INTERVENTIONS}
        stats["update_dominance"] = mean(row.get("update_dominates") for row in rows)
        for key in ("graph_gold", "graph_belt", "graph_belt_dropped", "lexical_diagnosed_n", "stem_only", "no_word_overlap"):
            stats[key] = sum(row[key] for row in rows)
        summary[name] = stats
    return summary


def render(report):
    stats = report["summary"]
    keys = ("raw_store", "fact_store", "r100", "r50", "r20", "recall_at_10", "mrr_at_10",
            "budget_recall", "a", "b", "packing_loss", "tokens", "p95_ms")
    formats = (".2%",) * 6 + (".4f",) + (".2%",) * 4 + (".1f", ".2f")
    first = table(("题型", "N", "金标题", "Raw", "Fact", "候选@100", "候选@50", "候选@20", "R@10", "MRR@10",
                   "预算覆盖", "A 缺候选", "B 候选未装配", "Top-10 装填损失", "Token", "P95 ms"),
                  ((name, row["n"], row["labeled"], *(number(row[key], fmt) for key, fmt in zip(keys, formats)))
                   for name, row in stats.items()))
    arms = table(("题型", "检索路", "独立 R@50", "移除此路 R@50", "四路+桥接 R@50", "边际百分点"),
                 ((name, arm, number(scores["alone"], ".2%"), number(scores["without"], ".2%"),
                   number(row["fused_arm50"], ".2%"),
                   number(100 * (row["fused_arm50"] - scores["without"]), "+.2f") if row["labeled"] else "—")
                  for name, row in stats.items() for arm, scores in row["arms"].items()))
    details = table(("题型", "图金标候选", "图金标夹层", "夹层裁掉", "可核查漏召证据", "仅词干共现",
                     "无字面或词干共现", "负样本空返"),
                    ((name, *(row[key] for key in ("graph_gold", "graph_belt", "graph_belt_dropped",
                                                    "lexical_diagnosed_n", "stem_only", "no_word_overlap")),
                      f'{row["negative_empty"]}/{row["negative_n"]}') for name, row in stats.items()))
    output = (f"## {report['benchmark']} · {report['ingest']} · {report['backend']} · {report['query_mode']}"
              f"\n\n{first}\n\n{arms}\n\n{details}")
    if stats["Overall"].get("leave_one_out"):
        ablations = table(("题型", "移除检索路", "R@10", "R@10 边际百分点", "MRR@10", "MRR@10 边际"),
                          ((bucket, arm, number(part["recall_at_10"], ".2%"),
                            number(100 * (values["recall_at_10"] - part["recall_at_10"]), "+.2f"),
                            number(part["mrr_at_10"], ".4f"),
                            number(values["mrr_at_10"] - part["mrr_at_10"], "+.4f"))
                           for bucket, values in stats.items() for arm, part in values.get("leave_one_out", {}).items()))
        output += "\n\n" + ablations
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("locomo", "longmemeval", "membench", "membench99"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sample", type=int)
    parser.add_argument("--ingest", choices=("frozen", "chronicle"), default="frozen")
    parser.add_argument("--query-mode", choices=("single_query", "reconstructive"))
    parser.add_argument("--ablate", action="store_true", help="Measure final context with each channel separately disabled")
    parser.add_argument("--backend", choices=("hashed", "local-neural"), default="hashed")
    args = parser.parse_args()
    seed = 42
    if args.benchmark.startswith("membench"):
        if args.benchmark == "membench99":
            by_id = {}
            for value in range(42, 47):
                for item in load_sample(5, value, args.data, offline=True):
                    by_id.setdefault(_item_key(item), item)
            data = list(by_id.values())
            if len(data) != 99:
                raise ValueError("the frozen 99-question cohort was not available")
        else:
            data, _ = sample_items(args.data, seed, 100)
        if args.sample:
            data = sample_rows(data, args.sample, seed, lambda item: item["category"])
        run, digest = mem_item, None
    else:
        name, groups = ("LoCoMo", locomo_groups) if args.benchmark == "locomo" else ("LongMemEval-S", longmem_groups)
        checksum = hashlib.sha256()
        with args.data.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                checksum.update(chunk)
        digest = checksum.hexdigest()
        if digest != OFFICIAL_SHA256[name]:
            raise ValueError(f"{name} official snapshot does not match")
        data, run = groups(read_dataset(args.data), args.sample, seed), dialogue_group
    query_mode = args.query_mode or ("reconstructive" if args.ingest == "chronicle" else "single_query")
    if args.ingest == "frozen" and query_mode != "single_query":
        parser.error("frozen samples replay production single-query recall")
    if args.ablate and args.ingest != "chronicle":
        parser.error("--ablate requires --ingest chronicle")
    worker = partial(run, rich=args.ingest == "chronicle", backend=args.backend, mode=query_mode, ablate=args.ablate)
    items = []

    def collect(batches):
        for batch in batches:
            items.extend(batch)
            if args.benchmark == "locomo" or len(items) % 25 == 0:
                print(f"{args.benchmark}: {len(items)} diagnosed", flush=True)

    if args.workers == 1:
        collect(map(worker, data))
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            collect(pool.map(worker, data))
    report = {"benchmark": args.benchmark, "ingest": args.ingest, "backend": args.backend, "query_mode": query_mode,
              "seed": seed, "sha256": digest, "items": items, "summary": summarize(items)}
    write_result(report, args.out, render(report))


if __name__ == "__main__":
    main()
