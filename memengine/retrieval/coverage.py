"""Bounded, provenance-aware coverage for coordinated questions."""

import re
from collections import Counter

from memengine.lifecycle import normalized_content
from memengine.linguistics import stem, strip_speaker_prefix
from memengine.temporal import competition_terms
from memengine.textutil import strip_step_tags, token_units

_LIST_INTRO = re.compile(r"\b(?:tell me about|say about|details about|cover|include|both)\s+|关于|分别(?:是|提到(?:了)?(?:哪些|什么)?)?", re.I)
_SEPARATOR = re.compile(r"[,，;；、]\s*(?:and\s+)?|\s+(?:and|as well as)\s+|以及", re.I)
_QUESTION = re.compile(r"^(?:what|which|when|where|who|why|how)\b|^(?:什么|哪|何时|为何|怎么|多久)", re.I)
_AGGREGATE = re.compile(
    r"\b(?:what|which|where|who|how many)\b.{1,90}\b(?:both|have|has|types|kinds|different|various|done|visited)\b|"
    r"\b(?:activities|events|cities|countries|books|places|hobbies|reasons)\b|哪些|分别|一共", re.I)
_PRONOUNS = {"i", "me", "my", "he", "she", "his", "her", "they", "their", "we", "our"}


def _facet_terms(text):
    return {stem(term) if term.isascii() and term.isalpha() else term for term in competition_terms(text) - _PRONOUNS}


def _evidence_terms(fact):
    return _facet_terms(fact["text"]) | _facet_terms(fact.get("speaker") or "")


def explicit_facets(query):
    intro = _LIST_INTRO.search(query)
    body = query[intro.end():] if intro else query
    parts = [part.strip(" .!?。！？") for part in _SEPARATOR.split(body)]
    if not 2 <= len(parts) <= 4 or not (intro or all(_QUESTION.match(part) for part in parts)):
        return []
    terms = [_facet_terms(part) for part in parts]
    return parts if all(terms) and len({frozenset(group) for group in terms}) == len(parts) else []


def _facet_source(fact, require_entity):
    return (fact["status"] == "active" and bool(fact["source_turn_ids"]) and fact["role"] in {"user", "tool"}
            and (not require_entity or fact.get("entity_anchor")) and not fact["plain"].rstrip().endswith(("?", "？")))


def cover_evidence(facts, facets, query, *, aggregate, require_entity, tau, top_k, token_budget):
    if not facts or not (facets or aggregate and _AGGREGATE.search(query)):
        return facts
    wanted = [_facet_terms(facet) for facet in facets]
    coverage = {}
    seen = set()
    primary, duplicates = [], []
    for fact in facts:
        if fact["status"] != "active" or not fact["source_turn_ids"] or fact.get("ce_score", 0) < tau:
            primary.append(fact)
            continue
        body = strip_speaker_prefix(strip_step_tags(fact["plain"]).strip(), fact.get("speaker"))
        key = fact.get("canonical_entity_id") or fact.get("speaker"), normalized_content(body)
        if key in seen:
            duplicates.append(fact)
            continue
        seen.add(key)
        primary.append(fact)
        if wanted and _facet_source(fact, require_entity):
            visible = _evidence_terms(fact)
            coverage[fact["id"]] = {index for index, terms in enumerate(wanted) if terms <= visible}

    covered = set()
    session_counts = Counter()
    chosen = []
    spent = 0
    while len(chosen) < min(top_k, len(wanted)):
        viable = (
            (index, fact, spent + bool(spent) + token_units(fact["text"]))
            for index, fact in enumerate(primary)
            if coverage.get(fact["id"], set()) - covered
        )
        eligible = ((index, fact, cost) for index, fact, cost in viable if cost <= 4 * token_budget)
        best = max(eligible, key=lambda item: (
            len(coverage[item[1]["id"]] - covered), item[1]["final_score"],
            -session_counts[item[1]["session_id"]], -item[0]), default=None)
        if best is None:
            break
        index, fact, spent = best
        covered.update(coverage[fact["id"]])
        session_counts[fact["session_id"]] += 1
        chosen.append({**primary.pop(index), "coverage_priority": len(chosen)})

    return [*chosen, *primary, *duplicates]


def packed_facet_coverage(facets, items, *, require_entity):
    if not facets:
        return {"facets": [], "covered_at_10": 0}
    top = [row for row in items if row["kind"] == "fact"][:10]
    evidence = [
        (row["evidence_id"], _evidence_terms(row)) for row in top
        if _facet_source(row, require_entity)
    ]
    matches = []
    for facet in facets:
        wanted = _facet_terms(facet)
        matches.append({"facet": facet, "evidence_ids": [key for key, terms in evidence if wanted <= terms]})
    return {"facets": matches, "covered_at_10": sum(bool(row["evidence_ids"]) for row in matches)}
