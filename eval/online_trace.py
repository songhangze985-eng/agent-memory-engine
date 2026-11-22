"""Trace the actual LongMemEval-S recall and context assembly in one call."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, closing
from dataclasses import replace
from functools import partial
import hashlib
from itertools import product
from pathlib import Path
import platform
from time import perf_counter
from unittest.mock import patch

from eval.evidence_ablation import OFFICIAL_SHA256, fact_turns, read_dataset, retain_source_turn
from eval.longmemeval.run import category, groups
from eval.metrics import evidence_metrics, mean, percentile
from eval.packing_replay import replay, replay_table, summarize_replays
from eval.paired import TOKEN_BUDGET, make_engine
from eval.static_report import number, table, write_result
from memengine.assembler import assemble_context
from memengine.engine import _fit_context
from memengine.neural_onnx import OnnxCrossEncoderReranker
from memengine.retrieval.rerank_budget import PAIR_TOKENS


BENCHMARK = "LongMemEval-S"
STAGES = ("fused", "ce", "reranked", "gate_input", "gate_kept", "packing_input", "packed", "context")
FUSION_ABLATIONS = {
    "baseline": {},
    "quotas_only": {"channel_quotas": True},
    "weights_only": {"semantic_weighting": True},
    "graph_off_only": {"entity_graph": False},
    "combined": {"channel_quotas": True, "semantic_weighting": True, "entity_graph": False},
}
FUSION_VARIANTS = {
    **FUSION_ABLATIONS,
    "score_quantiles": {"score_quantiles": True},
    "bounded_score_quantiles": {"score_quantiles": True, "preserve_candidate_at_50": True},
}


def observe_onnx(session, batches):
    run = session.run

    def measured(output_names, input_feed, *args, **kwargs):
        start = perf_counter()
        result = run(output_names, input_feed, *args, **kwargs)
        forward_ms = (perf_counter() - start) * 1000
        mask = input_feed["attention_mask"]
        input_tokens = int(mask.sum())
        batches.append({"pairs": mask.shape[0], "input_tokens": input_tokens,
                        "padding_tokens": mask.size - input_tokens, "forward_ms": forward_ms})
        return result

    return patch.object(session, "run", measured)


def diagnose(pass_trace, context, sources, gold):
    evidence = {row["id"]: [turn.ref for turn in fact_turns(row, (), sources)]
                for row in context if row["kind"] == "fact"}
    ranks = {}
    for stage in STAGES:
        if stage == "ce":
            ids = sorted(pass_trace["fused"], key=pass_trace["ce_scores"].get, reverse=True)
        elif stage == "context":
            ids = list(evidence)
        else:
            ids = pass_trace.get(stage, [])
        first = {}
        for rank, ident in enumerate(ids, 1):
            refs = evidence[ident] if stage == "context" else pass_trace["sources"].get(ident, ())
            for ref in refs:
                if ref in gold:
                    first.setdefault(ref, (rank, ident))
        ranks[stage] = first

    details = []
    for ref in sorted(gold):
        stages = {stage: hit[ref][0] if ref in hit else None for stage, hit in ranks.items()}
        if stages["context"] is not None:
            outcome = "hit" if stages["context"] <= 10 else "below_context_top10"
        else:
            outcome = "missing_from_" + next(stage for stage in STAGES if stages[stage] is None)
        same_session_above = None
        if ref in ranks["reranked"]:
            rank, ident = ranks["reranked"][ref]
            session = pass_trace["sessions"][ident]
            same_session_above = sum(pass_trace["sessions"][other] == session
                                     for other in pass_trace["reranked"][:rank - 1])
        details.append({"ref": ref, "ranks": stages, "outcome": outcome,
                        "same_session_above_reranked": same_session_above})
    return details


def run_group(group, *, backend, mode, packing_replay=False, fusion_ablation=False, fusion_variant="baseline", ce_pairs=None):
    turns, questions = group
    engine = make_engine("mock", backend=backend, onnx_wide_pairs=ce_pairs)
    rows = []
    with closing(engine.store):
        sources = {}
        for index, turn in enumerate(turns):
            result = retain_source_turn(engine, index, turn)
            sources[result["turn_id"]] = turn
        # Include candidate IDs that never reach the final context in the provenance map.
        fact_sources = {fact["id"]: sorted({turn.ref for turn in fact_turns(fact, (), sources)})
                        for fact in engine.store.snapshot(scope=("default", "default", None),
                                                          include_cold=True, include_superseded=True)}
        session = None
        if backend == "local-neural":
            if not isinstance(engine.reranker, OnnxCrossEncoderReranker):
                raise RuntimeError("local-neural tracing requires the ONNX reranker")
            session = engine.reranker._encoder.model.model
        recall = engine._recall
        variants = FUSION_ABLATIONS if fusion_ablation else (fusion_variant,)
        for question, variant in product(questions, variants):
            passes, batches = [], []
            with ExitStack() as stack:
                if parameters := FUSION_VARIANTS[variant]:
                    def ablated_recall(*args, plan=None, **kwargs):
                        return recall(*args, plan=replace(plan, **parameters) if plan else None, **kwargs)

                    stack.enter_context(patch.object(engine, "_recall", ablated_recall))
                if session is not None:
                    stack.enter_context(observe_onnx(session, batches))
                if packing_replay:
                    packer = stack.enter_context(patch("memengine.engine._fit_context", wraps=_fit_context))
                    assembler = stack.enter_context(patch("memengine.engine.assemble_context", wraps=assemble_context))
                start = perf_counter()
                bundle = engine.build_context(question.text, reference_time=question.timestamp, mode=mode,
                                              token_budget=TOKEN_BUDGET, include_short_term=False, _trace=passes)
                elapsed = (perf_counter() - start) * 1000
                batch_size = engine.reranker.batch_size if session is not None else None
            scored_ids = [p["ce_scored_ids"] for p in passes] if ce_pairs is not None else None
            expected_pairs = sum(map(len, scored_ids)) if scored_ids is not None else sum(len(p["fused"]) for p in passes)
            if session is not None and sum(batch["pairs"] for batch in batches) != expected_pairs:
                raise RuntimeError("ONNX batch trace did not account for all scored pairs")
            if ce_pairs is not None and (
                expected_pairs > ce_pairs or any(
                    batch["input_tokens"] + batch["padding_tokens"] > PAIR_TOKENS * batch["pairs"] for batch in batches)
            ):
                raise RuntimeError("ONNX input exceeded the CE budget")
            gold = question.evidence if not question.unanswerable else frozenset()
            gold_sessions = question.evidence_sessions if not question.unanswerable else frozenset()
            items = bundle["items"]
            mapped = [fact_turns(row, (), sources) for row in items if row["kind"] == "fact"]
            metrics = evidence_metrics([{turn.ref for turn in entry} for entry in mapped], gold)
            session_coverage = evidence_metrics([{turn.session for turn in entry} for entry in mapped], gold_sessions)
            pass_trace = passes[-1] if passes else {"fused": [], "ce_scores": {}}
            details = diagnose({**pass_trace, "sources": fact_sources}, items, sources, gold)
            replay_scores = None
            if packing_replay and passes:
                replay_scores = replay(bundle, packer.call_args.args, assembler.call_args, pass_trace["ce_scores"],
                                       fact_sources, sources, gold, gold_sessions)
            options = session.get_session_options() if session is not None else None
            raw_turn_ids = bundle["evidence_turn_ids"]
            injected_refs = {turn.ref for entry in mapped for turn in entry}
            injected_refs.update(sources[ident].ref for ident in raw_turn_ids if ident in sources)
            row = {"id": question.id, "category": question.category, **metrics,
                         "session_coverage_at_10": session_coverage["recall_at_10"],
                         "budget_recall": len(injected_refs & gold) / len(gold) if gold else None,
                         "top10_distinct_sessions": len({turn.session for entry in mapped[:10] for turn in entry}),
                         "candidate_at_50": mean(row["ranks"]["fused"] is not None and row["ranks"]["fused"] <= 50
                                                 for row in details),
                         "gold": details, "passes": passes, "ce_batches": batches,
                         "ce_batch_size": batch_size,
                         "ce_threads": ({"intra_op": options.intra_op_num_threads,
                                         "inter_op": options.inter_op_num_threads} if options is not None else None),
                         "ce_providers": session.get_providers() if session is not None else [],
                         "context_fact_ids": [row["id"] for row in items if row["kind"] == "fact"],
                         "context_raw_turn_ids": raw_turn_ids, "tokens": bundle["token_breakdown"]["total"],
                         "retrieval_ms": bundle["retrieval_ms"], "latency_ms": elapsed}
            if packing_replay:
                row["packing_replay"] = replay_scores
            if fusion_ablation or fusion_variant != "baseline":
                row["fusion_variant"] = variant
            if ce_pairs is not None:
                row["ce_scored_ids"] = scored_ids
            rows.append(row)
    return rows


def summarize(rows):
    outcomes = Counter(gold["outcome"] for row in rows for gold in row["gold"])
    missing_top50 = [gold for row in rows for gold in row["gold"]
                     if gold["outcome"] != "hit" and (gold["ranks"]["fused"] or 51) <= 50]
    top50_losses = Counter(gold["outcome"] for gold in missing_top50)
    fields = ("candidate_at_50", "recall_at_10", "mrr_at_10", "session_coverage_at_10", "budget_recall")
    batches = [batch for row in rows for batch in row["ce_batches"]]
    pairs = sum(batch["pairs"] for batch in batches)
    input_tokens = sum(batch["input_tokens"] for batch in batches)
    padding_tokens = sum(batch["padding_tokens"] for batch in batches)
    replays = summarize_replays(rows)
    return {"n": len(rows), **{key: mean(row[key] for row in rows) for key in fields},
            "p50_ms": percentile([row["latency_ms"] for row in rows], .50),
            "p95_ms": percentile([row["latency_ms"] for row in rows], .95),
            "rerank_p95_ms": percentile([sum(p["rerank_ms"] for p in row["passes"]) for row in rows], .95),
            "candidate_p95_ms": percentile([sum(p["candidate_ms"] for p in row["passes"]) for row in rows], .95),
            "ce_forward_p95_ms": percentile([sum(batch["forward_ms"] for batch in row["ce_batches"])
                                             for row in rows], .95) if pairs else None,
            "ce_pairs_mean": pairs / len(rows) if pairs else None,
            "ce_tokens_per_pair": input_tokens / pairs if pairs else None,
            "ce_padding_fraction": padding_tokens / (input_tokens + padding_tokens) if pairs else None,
            "gold_outcomes": dict(outcomes), "fused_top50_misses": dict(top50_losses),
            "fused_top50_misses_outside_ce_top10": sum((gold["ranks"]["ce"] or 11) > 10 for gold in missing_top50),
            **({"packing_replay": replays} if replays else {})}


def summarize_categories(rows):
    categories = sorted({row["category"] for row in rows})
    return {"Overall": summarize(rows), **{name: summarize([row for row in rows if row["category"] == name])
                                          for name in categories}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sample", type=int)
    parser.add_argument("--category", action="append")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--backend", choices=("hashed", "local-neural"), default="hashed")
    parser.add_argument("--ce-pairs", type=int, help="Cap local ONNX pairs per question and apply a 250-token pair limit")
    parser.add_argument("--packing-replay", action="store_true",
                        help="Replay the same budget and admitted candidates in score and gold-first order")
    fusion = parser.add_mutually_exclusive_group()
    fusion.add_argument("--fusion-ablation", action="store_true",
                        help="Compare channel quotas, weights and graph removal on the same ingested histories")
    fusion.add_argument("--fusion-variant", choices=FUSION_VARIANTS, default="baseline")
    parser.add_argument("--query-mode", choices=("single_query", "reconstructive"), default="reconstructive")
    args = parser.parse_args()
    sha = hashlib.sha256()
    with args.data.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    digest = sha.hexdigest()
    if digest != OFFICIAL_SHA256[BENCHMARK]:
        raise ValueError(f"{BENCHMARK} official snapshot does not match")
    data = [row for row in read_dataset(args.data) if not args.category or category(row) in args.category]
    if not data or args.workers < 1 or args.sample is not None and args.sample < 1:
        parser.error("provide at least one question and a positive number of workers")
    if args.ce_pairs is not None and (args.ce_pairs < 1 or args.backend != "local-neural"):
        parser.error("--ce-pairs requires a positive limit and the local-neural backend")
    worker = partial(run_group, backend=args.backend, mode=args.query_mode, packing_replay=args.packing_replay,
                     fusion_ablation=args.fusion_ablation, fusion_variant=args.fusion_variant, ce_pairs=args.ce_pairs)
    questions = groups(data, args.sample, 42)
    rows = []
    traced = 0
    with ExitStack() as stack:
        executor = stack.enter_context(ProcessPoolExecutor(args.workers)) if args.workers > 1 else None
        batches = executor.map(worker, questions) if executor else map(worker, questions)
        for batch in batches:
            rows.extend(batch)
            traced += len(batch) // (len(FUSION_ABLATIONS) if args.fusion_ablation else 1)
            if traced % 10 == 0:
                print(f"{traced} questions traced", flush=True)
    summary = summarize_categories([row for row in rows if row.get("fusion_variant", "baseline") == args.fusion_variant])
    report = {"benchmark": BENCHMARK, "sha256": digest, "backend": args.backend, "mode": args.query_mode,
              "workers": args.workers, "seed": 42, "platform": platform.platform(), "summary": summary, "items": rows}
    if args.ce_pairs is not None:
        report["ce_budget"] = {"pairs": args.ce_pairs, "tokens_per_pair": PAIR_TOKENS}
    body = table(("Category", "N", "Candidate@50", "R@10", "Session coverage@10", "P50 ms", "P95 ms", "Rerank P95 ms"),
                 ((name, row["n"], *(number(row[key], ".2%") for key in
                                     ("candidate_at_50", "recall_at_10", "session_coverage_at_10")),
                   *(number(row[key], ".1f") for key in ("p50_ms", "p95_ms", "rerank_p95_ms")))
                  for name, row in summary.items()))
    if args.packing_replay:
        body += "\n\n" + replay_table(summary)
    if args.fusion_ablation:
        ablation = {variant: summarize_categories([row for row in rows if row["fusion_variant"] == variant])
                    for variant in FUSION_ABLATIONS}
        report["fusion_ablation"] = ablation
        body += "\n\n" + table(("Variant", "Category", "N", "Candidate@50", "R@10", "Session coverage@10"),
                               ((variant, category, scores["n"], number(scores["candidate_at_50"], ".2%"),
                                 number(scores["recall_at_10"], ".2%"), number(scores["session_coverage_at_10"], ".2%"))
                                for variant, categories in ablation.items() for category, scores in categories.items()))
    write_result(report, args.out, body)


if __name__ == "__main__":
    main()
