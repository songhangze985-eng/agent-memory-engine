"""Cumulative six-layer retrieval ablation over shared, offline fact ledgers."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing, ExitStack
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from eval.evidence_ablation import Question, Turn, measure_pair, read_dataset
from eval.exact_ident_eval import run_static
from eval.locomo.run import groups as locomo_groups
from eval.longmemeval.run import groups as longmem_groups
from eval.membench.items import parse_timestamp
from eval.membench.load import load_sample
from eval.membench.run import _item_key, evidence_turn, paired_question
from eval.metrics import retrieval_report
from eval.paired import Pair, TOKEN_BUDGET, TOP_K, attach_consolidated_context, evaluate_pair, make_engine
from eval.static_report import number, table, write_result
from eval.stress_corpus import stress_dialogue
from memengine.assembler import assemble_context
from memengine.engine import _fit_context, baseline_rrf
from memengine.identifiers import eligible_identifiers
from memengine.queryparse import parse_query_rules
from memengine.reranker import anchored_relevance
from memengine.temporal import parse_temporal_intent

LAYERS = ("Pure RRF", "Exact Identifier Pinning", "Turn-Pair Anchoring", "Bipolar Ring & Belt",
          "Bi-temporal Competing Recency", "Compact Consolidation")


class RRFScorer:
    is_calibrated = False

    def score(self, query, candidates):
        return [min(1.0, fact["rrf_norm"]) for fact in candidates]


def _rank_only(engine, query, stamp, stage):
    started = perf_counter()
    parsed = parse_query_rules(query)
    temporal = parse_temporal_intent(query, stamp)
    if temporal.is_historical:
        parsed["query_type"] = "TIME_TRAVEL"
    ranked = engine._recall(query, parsed, temporal, ("default", "default", None), ())
    if stage == 0:
        ranked = baseline_rrf(ranked)
    else:
        ranked, _ = eligible_identifiers(ranked, query, temporal.is_historical)
        bonus = (dict(zip((fact["id"] for fact in ranked), anchored_relevance(query, ranked)))
                 if stage == 2 and any(fact.get("contextual_query") for fact in ranked) else {})
        ranked.sort(key=lambda fact: (-(min(1.0, fact["rrf_norm"]) + 0.35 * fact["exact_hit_ratio"]
                                        + bonus.get(fact["id"], 0.0))
                                      * (0.30 if fact["version_mismatch"] else 1.0), fact["id"]))
    facts = _fit_context([{**fact, "ce_score": min(1.0, fact["rrf_norm"])} for fact in ranked], TOP_K, TOKEN_BUDGET)
    result = {"ranked": facts, "empty_recall": not facts, "gate_stats": {"tau_abs": .12},
              "parsed_query": parsed, "matched_entities": [],
              "scope": {"user_id": "default", "agent_id": "default", "session_id": None}}
    context = assemble_context(engine.store, result, prefix="", token_budget=TOKEN_BUDGET,
                               drilldown_top_n=2, drilldown_window=0, query=query)
    return Pair(result, [], ranked, context, "", (perf_counter() - started) * 1000, 0)


def evaluate_group(engine, questions, raw_ids):
    """Run layers before mutating the single ingest snapshot with consolidation."""
    outputs = [{"id": q.id, "unanswerable": q.unanswerable} for q in questions]
    through_time = []
    for stage in range(5):
        with ExitStack() as stack:
            stack.enter_context(patch.object(engine.store, "search_entity_inverted", return_value=[]))
            if stage < 4:
                stack.enter_context(patch.object(engine.store, "search_temporal", return_value=[]))
                stack.enter_context(patch("memengine.engine.temporal_competition", return_value={}))
            if stage < 2:
                stack.enter_context(patch.object(engine.store, "search_turn_anchors", return_value=([], [])))
            if stage == 0:
                stack.enter_context(patch("memengine.engine.extract_identifiers", return_value=[]))
            if stage == 3:
                stack.enter_context(patch.object(engine, "superseded_weight", 1.0))
            if stage >= 3:
                stack.enter_context(patch.object(engine, "reranker", RRFScorer()))
            for row, q in zip(outputs, questions):
                pair = (_rank_only(engine, q.text, q.timestamp, stage) if stage < 3
                        else evaluate_pair(engine, q.text, q.timestamp))
                row[str(stage)] = measure_pair(engine, q, pair, raw_ids)["bipolar"]
                if stage == 4:
                    through_time.append(pair)
    engine.consolidate(timestamp=max((q.timestamp for q in questions), default=0))
    previous = None
    for row, q, pair in zip(outputs, questions, through_time):
        attached = attach_consolidated_context(engine, pair, q.text)
        row["5"] = measure_pair(engine, q, attached, raw_ids, previous)["bipolar"]
        previous = row["5"]["prefix_version"]
    return outputs


def _official(group):
    turns, questions = group
    engine = make_engine("mock")
    with closing(engine.store):
        raw_ids = {}
        for index, turn in enumerate(turns):
            stored = engine.add(f"[step {index}] {turn.text}", timestamp=turn.timestamp,
                                session_id=turn.ingest_session or turn.session, role=turn.role)
            raw_ids[stored["turn_id"]] = turn
        return evaluate_group(engine, questions, raw_ids)


def _mem(item):
    engine = make_engine("mock")
    with closing(engine.store):
        ids = {engine.add(turn["text"], timestamp=turn["ts"])["turn_id"]: evidence_turn(turn, i)
               for i, turn in enumerate(item["turns"])}
        stamp = parse_timestamp(item.get("q_time"), max(t["ts"] for t in item["turns"]) + 1)
        return evaluate_group(engine, [paired_question(item, stamp)], ids)


def _exact(engine, catalog):
    facts = engine.store.facts_by_ids([gold for _, _, gold in catalog if gold])
    raw_ids = {turn: Turn(fact["id"], "default", "user", fact["plain"], 1.0)
               for fact in facts for turn in fact["source_turn_ids"]}
    questions = [Question(literal, kind, f"`{literal}`" if kind == "missing_symbol"
                          else f"Which configuration or recovery procedure applies to {literal}?", 100, frozenset({gold}) if gold else frozenset(),
                          unanswerable=gold is None) for kind, literal, gold in catalog]
    return evaluate_group(engine, questions, raw_ids)


def _stress(index):
    case = stress_dialogue(index)
    turns = [Turn(str(i), "default", "user", text, i) for i, (_, text) in enumerate(case.turns)]
    evidence = {str(i) for i, (kind, _) in enumerate(case.turns) if kind == "taboo"}
    return _official((turns, [Question(str(index), "taboo", case.positive, len(turns) + 1, frozenset(evidence))]))


def run(locomo, longmem, memdata, seed, workers):
    if not locomo.is_file() or not longmem.is_file() or not memdata.is_dir():
        raise FileNotFoundError("matrix requires local datasets")
    mem_items = {}
    for offset in range(5):
        for item in load_sample(5, seed + offset, memdata, offline=True):
            mem_items.setdefault(_item_key(item), item)
    groups = {"LoCoMo": (_official, locomo_groups(read_dataset(locomo), 100, seed)),
              "LongMemEval-S": (_official, longmem_groups(read_dataset(longmem), 100, seed)),
              "MemBench": (_mem, mem_items.values()), "High frequency": (_stress, range(20))}
    rows = {"Exact Ident": run_static(seed, evaluate=_exact)}
    with ProcessPoolExecutor(workers) as pool:
        for group, (worker, jobs) in groups.items():
            rows[group] = [row for batch in pool.map(worker, jobs) for row in batch]
            print(f"{group}: {len(rows[group])} verified shared-cohort questions", flush=True)
    return {"benchmark": "Ablation Matrix", "seed": seed, "mode": "mock", "backend": "hashed", "items": rows,
            "layers": {group: [retrieval_report(items, str(i)) for i in range(len(LAYERS))] for group, items in rows.items()}}


def render(report):
    fields = ("recall_at_10", "mrr_at_10", "update_dominance", "avg_injected_tokens", "p95_retrieval_ms")
    rows = []
    useful = [False] * len(LAYERS)
    for group, layers in report["layers"].items():
        prev = {}
        for i, layer in enumerate(layers):
            change = [layer[key] - prev[key] if layer[key] is not None and prev.get(key) is not None else None for key in fields]
            if i > 0:
                useful[i] |= any(delta is not None and (delta > 0 if k < 3 else delta < 0) for k, delta in enumerate(change[:4]))
            rows.append((group, layer["n"], f"L{i} {LAYERS[i]}", *(number(layer[key]) for key in fields),
                         " / ".join(number(value, "+.4f") for value in change)))
            prev = layer
    detail = ("\n\nL0–L5 共用一份事实账本和金标。实体倒排全程关闭。L4 加入事件时间通道与竞争时序。"
              "各层共用装配器及归一化 RRF 相关性分。L5 冻结 L4 事实并单列静态预算。"
              "P95 含对应层实际检索和装配调用，分位数受本机负载影响。")
    for i, has_benefit in enumerate(useful[1:], 1):
        if not has_benefit:
            detail += f"\n\nL{i} 在这些样本中未改善证据指标或总 Token。"
            if i == 5:
                detail += "默认保持关闭，只有需要稳定档案或额外 Observation 时才显式蒸馏。"
    return table(("基准", "N", "层", "R@10", "MRR@10", "UpdateDom", "Token", "P95 ms",
                  "相邻层 Δ R@10 / MRR / UpdateDom / Token / P95"), rows) + detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--longmem", type=Path, required=True)
    parser.add_argument("--memdata", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.locomo, args.longmem, args.memdata, args.seed, args.workers)
    write_result(report, args.out, render(report))


if __name__ == "__main__":
    main()
