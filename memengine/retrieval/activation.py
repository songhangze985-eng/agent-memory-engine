"""Bounded graph recall and query evidence features."""
from __future__ import annotations

from collections import Counter
import re
from dataclasses import dataclass
from math import exp, sqrt

from memengine.chrono import DAY, TemporalSpan, interval_score, temporal_query
from memengine.linguistics import stem_terms
from memengine.textutil import normalize_entity
from memengine.vector import _QUERY_STOP


@dataclass(frozen=True)
class RecallPlan:
    windows: tuple = ()
    versions: tuple = ()
    graph: bool = False
    wide: bool = False
    channel_quotas: bool = False
    semantic_weighting: bool = False
    entity_graph: bool = True


_EDGE = {"entity": .78, "temporal": .55, "semantic": .85, "causal": 1.0}
_REFERENCE_EVENT = re.compile(r"\b(before|after)\s+(?:(?:the|my|their|his|her)\s+)?"
                              r"([a-z]+(?:ing|tion)|move|wedding)\b", re.I)


def spreading_activation(store, seeds, *, scope, filters, include_cold=False, top_n=45):
    """Two sparse frontiers with symmetric degree normalization."""
    hits = {}
    initial = {row["id"]: (1 / (1 + i * .065), True, False) for i, row in enumerate(seeds[:20])}
    frontier = initial
    for hop in (1, 2):
        scores = {}
        for edge in store.adjacent_links(frontier, scope=scope, filters=filters, include_cold=include_cold):
            target = edge["target"]
            if target in initial:
                continue
            incoming, same_speaker, content_link = frontier[edge["seed"]]
            score = (.65 * incoming * edge["weight"] * _EDGE[edge["link_type"]]
                     / sqrt(max(1, edge["degree"]) * max(1, edge["target_degree"])))
            same_speaker = same_speaker and bool(edge["shared_speaker"])
            content_link = content_link or edge["link_type"] in {"semantic", "causal"}
            path = score, same_speaker, content_link
            scores[target] = max(scores.get(target, (0, False, False)), path)
            if target not in hits or hits[target]["activation"] < score:
                hits[target] = {"id": target, "activation": score, "graph_depth": hop,
                                "speaker_bridge": same_speaker and content_link}
        frontier = dict(sorted(scores.items(), key=lambda pair: (-pair[1][0], pair[0]))[:28])
    return sorted(hits.values(), key=lambda row: (-row["activation"], row["id"]))[:top_n]


def query_overlap(query, fact):
    terms = stem_terms(query) - _QUERY_STOP - {"when", "what", "which", "how", "many", "much", "where", "why", "who"}
    return len(terms & stem_terms(fact.get("normalized_text") or fact["plain"])) / max(1, len(terms))


def calendar_relevance(query, windows, fact):
    start, end = fact.get("occurrence_start"), fact.get("occurrence_end")
    if start is None and end is None:
        mention = fact.get("mention_time") if fact.get("speaker") else None
        return .08 if mention is not None and any(interval_score(mention, mention, window) for window in windows) else 0
    if not windows:
        return .18 if temporal_query(query) else 0
    score = 0.0
    for window in windows:
        overlap = interval_score(start, end, window)
        if overlap:
            score = max(score, overlap)
        elif start is not None and end is not None:
            deltas = [max(0, start-window.end) if window.end is not None else 0,
                      max(0, window.start-end) if window.start is not None else 0]
            score = max(score, .25 * exp(-max(deltas)/(30*DAY)))
    return score


def discovered_windows(query, facts):
    """Use an explicit first-pass occurrence as a second-pass anchor."""
    dates = [fact for fact in facts if fact.get("time_precision") in {"day", "exact"}
             and fact.get("occurrence_start") is not None and fact.get("occurrence_end") is not None]
    if reference := _REFERENCE_EVENT.search(query):
        event = stem_terms(reference[2])
        anchor = next((fact for fact in dates if event and event <= stem_terms(fact["plain"])), None)
        if anchor:
            if reference[1].casefold() == "before":
                return (TemporalSpan(None, anchor["occurrence_start"] - 1, "day", reference[0]),)
            return (TemporalSpan(anchor["occurrence_end"] + 1, None, "day", reference[0]),)
        return ()
    if not temporal_query(query):
        return ()
    intervals = dict.fromkeys((fact["occurrence_start"], fact["occurrence_end"]) for fact in dates[:3])
    return tuple(TemporalSpan(start, end, "day", "") for start, end in intervals)


def external_entities(query, facts):
    query = query.casefold()
    result = {}
    for fact in facts[:5]:
        for key in fact["entity_z_keys"]:
            name = normalize_entity(key)
            if name and name not in query and name not in {"user", "assistant", "agent", "it", "image"}:
                if key.startswith("entity:") or len(name.split()) > 1 or any(c.isdigit() for c in name):
                    result.setdefault(name, key)
    return list(result.values())[:6]


_QUERY_ASSOCIATIONS = (
    ({"city", "country", "place", "visit", "vacation"}, ("travel", "trip", "visit", "fly", "tour")),
    ({"activitie", "activity", "hobbie", "hobby", "partake"}, ("practice", "play", "enjoy", "join", "started")),
    ({"event", "participate", "attend", "celebration"}, ("attend", "volunteer", "event", "fundraiser", "community")),
    ({"book", "read", "author"}, ("book", "read", "novel", "author", "library")),
    ({"career", "research", "study", "training"}, ("learn", "study", "degree", "training", "work")),
)


def bridge_feedback(query, facts):
    """Build a compact follow-up query from consensus among ranked seeds."""
    mentioned = stem_terms(query)
    excluded = mentioned | _QUERY_STOP | {"image", "really", "think", "say", "look", "make", "like",
                                                "good", "great", "today", "time", "thing", "user", "assistant"}
    support, weight = Counter(), Counter()
    for i, fact in enumerate(facts[:8]):
        words = {term for term in stem_terms(fact["plain"]) - excluded if len(term) >= 4}
        support.update(words)
        for term in words:
            weight[term] += 1/(1+i*.25)
    repeated = [word for word in sorted(weight, key=lambda word: (-weight[word], word)) if support[word] >= 2]
    entities = [key.partition(":")[2] for key in external_entities(query, facts)]
    related = [word for triggers, terms in _QUERY_ASSOCIATIONS if triggers & mentioned for word in terms]
    return " ".join(dict.fromkeys([*entities[:3], *related[:7], *repeated[:9]]))


def diversify_episodes(query, facts):
    """Retain the ranked head and spread the remaining aggregate evidence across episodes."""
    aggregate = re.search(r"\b(?:what|which|where|who|how many)\b.{1,90}\b(?:both|have|has|types|kinds|different|various|done|visited)\b|"
                          r"\b(?:activities|events|cities|countries|books|places|hobbies|reasons)\b|哪些|分别|一共", query, re.I)
    if len(facts) < 8 or not aggregate:
        return facts
    head, remaining = list(facts[:5]), list(facts[5:])
    counts = Counter(row["session_id"] for row in head)
    while remaining:
        chosen = max(range(len(remaining)), key=lambda i: (
            remaining[i]["final_score"] / (1 + .15 * counts[remaining[i]["session_id"]]), -i))
        fact = remaining.pop(chosen)
        head.append(fact)
        counts[fact["session_id"]] += 1
    return head
