"""Observable support signals for calibrating a pre-assembly abstention gate."""

import re

from agentmem_core.reranker import evidence_pair_scores
from agentmem_core.retrieval.coverage import supported_facet

_DATE_QUESTION = re.compile(r"\b(?:when|what date|which date)\b|什么时候|哪天|何时|几月几日", re.I)


def abstention_group(query, parsed, temporal):
    if parsed["facets"]:
        return "multi-facet"
    if parsed["preference_query"]:
        return "preference"
    if parsed["query_type"] == "TIME_TRAVEL" or _DATE_QUESTION.search(query):
        return "temporal"
    if temporal.get("order_hint") == "latest":
        return "current-state"
    return "factual"


def abstention_features(query, candidates, facets, *, require_entity=False):
    head = candidates[:5]
    evidence = [row for row in head if row["status"] == "active" or row.get("state_tag") == "[历史快照]"]
    if not evidence:
        return dict.fromkeys(("ce", "support", "slot_coverage", "margin", "combined", "facet_support"), 0.0)
    scores = sorted((row["ce_score"] for row in evidence), reverse=True)
    measured = evidence_pair_scores(query, evidence)
    support = max(row["ce_score"] * (coverage + clause) / 2
                  for row, (coverage, clause) in zip(evidence, measured))
    targets = facets or [query]
    slot_coverage = sum(max(part for part, _ in evidence_pair_scores(target, evidence))
                        for target in targets) / len(targets)
    margin = scores[0] - scores[1] if len(scores) > 1 else scores[0]
    facet_support = 1.0 if facets and supported_facet(facets, evidence, require_entity=require_entity) else support
    return {"ce": scores[0], "support": support, "slot_coverage": slot_coverage, "margin": margin,
            "combined": support * slot_coverage * (0.9 + 0.1 * margin), "facet_support": facet_support}
