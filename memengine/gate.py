"""Precision gate over an already sorted list of scored facts."""

from memengine.distance import BELT


def precision_gate(ranked, *, tau_abs, allow_empty_recall, cliff_ratio, belt_prune_ratio, min_keep,
                   is_calibrated=True, top_k=10):
    max_ce = max((fact["ce_score"] for fact in ranked), default=0.0)
    stats = {"candidate_count": len(ranked), "max_ce_score": max_ce, "tau_abs": tau_abs,
             "cliff_dropped_count": 0, "belt_pruned_count": 0,
             "empty_reason": "no_candidates" if not ranked else None}
    if not ranked:
        return [], stats

    pinned = [fact for fact in ranked if fact.get("exact_ident_hit")]
    signals = any(fact.get("lexical_overlap") or fact.get("entity_anchor") or fact.get("vector_score")
                  or fact.get("temporal_match") for fact in ranked)
    if allow_empty_recall and not pinned:
        if is_calibrated and max_ce < tau_abs:
            stats["empty_reason"] = "below_absolute_threshold"
            return [], stats
        if not is_calibrated and not signals:
            stats["empty_reason"] = "no_relevance_signals"
            return [], stats

    keep = len(ranked)
    confident = 0
    if is_calibrated:
        for index, fact in enumerate(ranked):
            if (index and confident >= min_keep
                    and fact["final_score"] < cliff_ratio * ranked[index - 1]["final_score"]):
                keep = index
                break
            confident += fact["ce_score"] >= tau_abs or bool(fact.get("exact_ident_hit"))
    shortlist = [fact for index, fact in enumerate(ranked) if index < keep or fact.get("exact_ident_hit")]
    stats["cliff_dropped_count"] = len(ranked) - len(shortlist)
    threshold = belt_prune_ratio * max(fact["final_score"] for fact in ranked)
    selected = [fact for fact in shortlist if fact["zone"] != BELT or fact["final_score"] >= threshold
                or fact.get("exact_ident_hit") or fact.get("protected_bridge")]
    if not is_calibrated and (len(selected) < min(min_keep, len(shortlist))
                              or (len(ranked) <= top_k and not any(fact["zone"] != BELT for fact in selected))):
        selected = shortlist
    stats["belt_pruned_count"] = len(shortlist) - len(selected)
    return selected, stats
