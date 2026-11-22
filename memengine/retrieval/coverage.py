"""Bounded, provenance-aware coverage for coordinated questions."""

import re
from collections import Counter

from memengine.lifecycle import normalized_content
from memengine.linguistics import stem, strip_speaker_prefix
from memengine.temporal import competition_terms
from memengine.textutil import strip_step_tags, token_units
from memengine.verification import claims_from_text

_LIST_INTRO = re.compile(r"\b(?:tell me about|say about|details about|cover|include|both)\s+|关于|分别(?:是|提到(?:了)?(?:哪些|什么)?)?", re.I)
_SEPARATOR = re.compile(r"[,，;；、]\s*(?:and\s+)?|\s+(?:and|as well as)\s+|以及", re.I)
_QUESTION = re.compile(r"^(?:what|which|when|where|who|why|how)\b|^(?:什么|哪|何时|为何|怎么|多久)", re.I)
_WHICH = re.compile(r"^which\s+", re.I)
_AUXILIARY = re.compile(r"\b(?:did|does|do|has|have|had|is|are|was|were|can|could|will|would|should)\b", re.I)
_AGGREGATE = re.compile(
    r"\b(?:what|which|where|who|how many)\b.{1,90}\b(?:both|have|has|types|kinds|different|various|done|visited)\b|"
    r"\b(?:activities|events|cities|countries|books|places|hobbies|reasons)\b|哪些|分别|一共", re.I)
_EVIDENCE_HEADER = re.compile(r"^\[\d{4}-\d{2}-\d{2}(?: \| [^\]]+)?\]\s*")
_NEGATED = re.compile(r"\b(?:not|never)\b|n['’]t\b|从未|不曾|没有", re.I)
_PRONOUNS = {"i", "me", "my", "he", "she", "his", "her", "they", "their", "we", "our"}
_UNCERTAIN = re.compile(
    r"\b(?:unknown|not sure|not recorded|not been established|(?:don['’]t|do not) (?:know|remember))\b|"
    r"不清楚|不知道|不确定|未记录|尚未确认", re.I)


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
    if not intro and all(_WHICH.match(part) for part in parts):
        predicate = _AUXILIARY.search(parts[-1])
        if predicate:
            suffix = parts[-1][predicate.start():]
            parts[:-1] = [part if _AUXILIARY.search(part) else f"{part} {suffix}" for part in parts[:-1]]
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


def answered_facets(facets, claims, evidence, matched_entities):
    """Attribute each answered facet to a verified claim and visible citation."""
    subjects = {normalized_content(key.partition(":")[2] or key) for key in matched_entities}
    answered = {facet: [] for facet in facets}
    targets = {facet: _facet_terms(facet) for facet in facets}
    claim_citations = {}
    for index, claim in enumerate(claims):
        if _UNCERTAIN.search(claim["text"]):
            continue
        terms = _facet_terms(claim["text"])
        for key in claim["evidence_ids"]:
            row = evidence[key]
            speaker = row.get("speaker") or ""
            if subjects and not (row.get("entity_anchor") or normalized_content(speaker) in subjects):
                continue
            claim_terms = terms | _facet_terms(speaker)
            evidence_terms = _evidence_terms(row)
            for facet, wanted in targets.items():
                if wanted <= claim_terms and wanted <= evidence_terms and terms - wanted:
                    if key not in answered[facet]:
                        answered[facet].append(key)
                    keys = claim_citations.setdefault(index, [])
                    if key not in keys:
                        keys.append(key)
    return ({facet: keys for facet, keys in answered.items() if keys},
            [{**claims[index], "evidence_ids": keys} for index, keys in claim_citations.items()])


def supported_facet(facets, candidates, *, require_entity):
    """Check if a source-backed statement visibly supplies an explicit facet."""
    targets = [_facet_terms(facet) for facet in facets]
    for fact in candidates:
        if not _facet_source(fact, require_entity):
            continue
        speaker = fact.get("speaker") or ""
        text = _EVIDENCE_HEADER.sub("", strip_step_tags(fact["text"]).strip())
        if speaker:
            text = text.removeprefix(f"[{speaker}]").lstrip()
        text = strip_speaker_prefix(text.removeprefix(fact.get("state_tag", "")).lstrip(), speaker)
        speaker_terms = _facet_terms(speaker)
        for clause in claims_from_text(text):
            if _UNCERTAIN.search(clause) or _NEGATED.search(clause):
                continue
            body = _facet_terms(clause)
            terms = body | speaker_terms
            if any(target <= terms and body - target - speaker_terms for target in targets):
                return True
    return False
