"""Reorder a captured shortlist under the original retrieval and context budgets."""
from collections import Counter

from eval.evidence_ablation import fact_turns
from eval.metrics import evidence_metrics, mean
from eval.static_report import number, table
from memengine.assembler import assemble_context
from memengine.engine import _fit_context
from memengine.textutil import estimate_tokens


def measure_context(context, sources, gold, gold_sessions):
    mapped = [fact_turns(row, (), sources) for row in context["items"] if row["kind"] == "fact"]
    turns = [{turn.ref for turn in refs} for refs in mapped]
    sessions = [{turn.session for turn in refs} for refs in mapped]
    injected = set().union(*turns)
    injected.update(sources[ident].ref for ident in context["evidence_turn_ids"] if ident in sources)
    return {**evidence_metrics(turns, gold),
            "session_coverage_at_10": evidence_metrics(sessions, gold_sessions)["recall_at_10"],
            "budget_recall": len(injected & gold) / len(gold) if gold else None,
            "tokens": context["token_breakdown"]["total"]}


def oracle_order(pool, fact_sources, gold):
    remaining = list(pool)
    ordered, uncovered = [], set(gold)
    lengths = {fact["id"]: estimate_tokens(fact["text"]) for fact in pool}
    while uncovered and remaining:
        choice = max(remaining, key=lambda fact: (
            len(uncovered.intersection(fact_sources.get(fact["id"], ()))),
            -lengths[fact["id"]]))
        refs = uncovered.intersection(fact_sources.get(choice["id"], ()))
        if not refs:
            break
        ordered.append(choice)
        remaining.remove(choice)
        uncovered -= refs
    return ordered + remaining


def replay(baseline, packing, assembly, ce_scores, fact_sources, sources, gold, gold_sessions):
    pool, top_k, packing_budget = packing
    store, result = assembly.args
    orders = {
        "baseline": pool,
        "ce_order": sorted(pool, key=lambda fact: ce_scores[fact["id"]], reverse=True),
        "gold_first": oracle_order(pool, fact_sources, gold),
    }
    scored = {}
    for name, rows in orders.items():
        ranked = _fit_context(rows, top_k, packing_budget)
        if name == "baseline" and ranked != baseline["ranked"]:
            raise RuntimeError("captured budget inputs could not reproduce the deployed ranking")
        context = assemble_context(store, {**result, "ranked": ranked, "empty_recall": not ranked}, **assembly.kwargs)
        if name == "baseline" and any(context[key] != baseline[key] for key in context):
            raise RuntimeError("captured budget inputs could not reproduce the deployed context")
        scored[name] = measure_context(context, sources, gold, gold_sessions)
    admitted = {ref for fact in pool for ref in fact_sources.get(fact["id"], ())}
    return {"admitted_gold_coverage": len(admitted & gold) / len(gold) if gold else None, **scored}


def summarize_replays(rows):
    replays = [row["packing_replay"] for row in rows if row.get("packing_replay")]
    if not replays:
        return {}
    result = {"admitted_gold_coverage": mean(row["admitted_gold_coverage"] for row in replays)}
    for label in ("baseline", "ce_order", "gold_first"):
        scores = [row[label] for row in replays]
        result[label] = {key: mean(score[key] for score in scores) for key in scores[0]}
        changes = Counter("improved" if new > old else "regressed" if new < old else "tied"
                          for row in replays
                          if (old := row["baseline"]["recall_at_10"]) is not None
                          and (new := row[label]["recall_at_10"]) is not None)
        result[label]["paired"] = {key: changes[key] for key in ("improved", "regressed", "tied")}
    return result


def replay_table(summary):
    return table(("Category", "Ordering", "R@10", "MRR@10", "Session coverage@10", "Budget recall", "Tokens", "Won / lost"),
                 ((category, label, number(row["recall_at_10"], ".2%"), number(row["mrr_at_10"], ".4f"),
                   number(row["session_coverage_at_10"], ".2%"), number(row["budget_recall"], ".2%"),
                   number(row["tokens"], ".1f"), f'{row["paired"]["improved"]} / {row["paired"]["regressed"]}')
                  for category, values in summary.items() for label, row in values.get("packing_replay", {}).items()
                  if label != "admitted_gold_coverage"))
