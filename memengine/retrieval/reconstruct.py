"""A bounded second recall driven by identifiers found in first-pass evidence."""
import re
from collections import Counter

from memengine.identifiers import compares_versions, extract_identifiers
from memengine.temporal import competition_terms
from memengine.textutil import normalize_entity

_MULTI = re.compile(r"推荐|两次|分别|之间|关系|关联|跨|\b(?:recommend\w*|relationship|both|between|connect\w*)\b", re.I)


def bridge_keys(query, facts):
    terms = competition_terms(query)
    if not facts or not terms:
        return ()
    if not _MULTI.search(query) and any(terms <= competition_terms(row["plain"]) for row in facts[:3]):
        return ()
    identifiers = extract_identifiers(query)
    if any(item.ident_type == "version" for item in identifiers) and not compares_versions(query):
        return ()
    known = {item.ident_norm for item in identifiers}
    evidence = facts[:3]
    occurrences = Counter(normalize_entity(key) for fact in evidence for key in set(fact["entity_z_keys"]))
    found = {}
    for fact in evidence:
        speakers = set(re.findall(r"(?:^|\n)([A-Z][a-z]{1,25}):", fact["plain"]))
        keys = [key for key in fact["entity_z_keys"] if not key.startswith(("person:", "人物:"))
                or key.partition(":")[2] in speakers or occurrences[normalize_entity(key)] >= 2]
        keys += [item.ident for item in extract_identifiers(fact["plain"])
                 if item.ident_type in {"code_symbol", "error_code", "project_token"}
                 and (not item.ident.isalpha() or not item.ident.isupper())]
        for key in keys:
            norm = normalize_entity(key)
            if len(norm) >= 2 and norm not in known and norm not in query.casefold():
                found.setdefault(norm, key)
    return tuple(found.values())[:3]


def merge_passes(primary, secondary):
    pool = {row["id"]: row for row in primary}
    extra = []
    for row in secondary:
        if row["id"] in pool:
            continue
        extra.append({**row, "version_match": False, "version_mismatch": False,
                      "exact_hit_ratio": 0.0, "exact_ident_hit": 0})
    return [*primary[:30], *extra[:10], *primary[30:40-len(extra[:10])]][:40]


def select_bridged(query, primary, ranked, bridges):
    """Keep the original head and admit a bounded number of supported new facts."""
    keys = {normalize_entity(key) for key in bridges}
    existing = {fact["id"] for fact in primary}
    seen_terms = set().union(*(competition_terms(fact["plain"]) for fact in primary[:3]))
    missing = competition_terms(query) - seen_terms
    seeks_version = bool(re.search(r"版本|\bversion\b", query, re.I))
    additions = []
    for fact in ranked:
        if fact["id"] in existing:
            continue
        entities = {normalize_entity(key) for key in fact["entity_z_keys"]} | fact["identifiers"].keys()
        terms = competition_terms(fact["plain"])
        if keys.intersection(entities) and (missing.intersection(terms) or
                (seeks_version and "version" in fact["identifiers"].values())):
            additions.append(fact)
    head = min(7, max(1, len(primary) - 1))
    return [*primary[:head], *additions[:1], *primary[head:]]
