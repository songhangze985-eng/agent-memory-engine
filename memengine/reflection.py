"""Read-only, bounded evidence verification over the memory ledger."""
import re

from memengine.chrono import TemporalSpan, date_iso, days_between, parse_session_clock
from memengine.identifiers import compares_versions
from memengine.temporal import competition_terms
from memengine.versioning import extract_versions, version_order
from memengine.verification import admissible_evidence

_GAP = re.compile(r"\b(?:how long|how many|time elapsed|between|both|relationship|and)\b|相隔|多久|分别|之间|以及", re.I)
_DISTANCE = re.compile(r"\b(?:how (?:long|many (?:days|weeks|months|years))|time elapsed)\b|相隔|间隔|多少天", re.I)
_SPLIT = re.compile(r"\s+\b(?:and|versus|vs)\b\s+|以及|和|与", re.I)


def _present(row):
    return {key: row.get(key) for key in ("id", "speaker", "mention_time", "occurrence_start",
                                          "occurrence_end", "time_precision", "supersedes_id")}


def checked_reflection(engine, query, *, scope, task_id, reference_time, token_budget):
    owner = dict(zip(("user_id", "agent_id", "session_id"), scope))
    query_options = {**owner, "reference_time": reference_time, "token_budget": token_budget,
                     "packing_mode": "token_budget", "task_id": task_id}
    first = engine.build_context(query, mode="single_query", **query_options)
    evidence = {row["id"]: row for row in first["items"] if row["kind"] == "fact"}
    count = 1
    compare = compares_versions(query)
    terms = competition_terms(query)
    coverage = set().union(*(competition_terms(row["plain"]) for row in list(evidence.values())[:3]))
    if (_GAP.search(query) or compare) and (len(evidence) < 2 or terms-coverage or _DISTANCE.search(query)):
        parts = [part.strip() for part in _SPLIT.split(query, maxsplit=1) if len(part.strip()) >= 4]
        # A full-query graph pass preserves the question while resolving a missing link.
        queries = [query] if len(parts) < 2 else parts
        for part in queries[:2]:
            result = engine.build_context(part, mode="reconstructive", **query_options)
            count += 1
            for row in result["items"]:
                if row["kind"] == "fact":
                    evidence.setdefault(row["id"], row)
    bundle = (engine.build_context(query, **query_options, retrieved={**first, "ranked": list(evidence.values())})
              if count > 1 else first)
    eligible = admissible_evidence(bundle)
    ranked = [row for row in eligible.values() if row["kind"] in {"fact", "observation"}]
    direct = None
    cited = ranked
    with_occurrence = [row for row in ranked if row.get("occurrence_start") is not None]
    if with_occurrence and re.search(r"\bhow long ago\b|距今多久", query, re.I):
        event = with_occurrence[0]
        now = parse_session_clock(reference_time)
        if now is not None and event["time_precision"] in ("day", "exact") and event["occurrence_start"] <= now:
            elapsed = days_between(TemporalSpan(event["occurrence_start"], event["occurrence_end"], "day", ""),
                                   TemporalSpan(now, now, "exact", ""))
            direct = f"{elapsed} 天" if "距今" in query else f"{elapsed} days"
            cited = [event]
    if direct is None and _DISTANCE.search(query) and len(with_occurrence) >= 2:
        # Only exact/day assertions support a definite difference in calendar days.
        point = []
        dates = set()
        for row in with_occurrence:
            day = date_iso(row["occurrence_start"])
            if row["time_precision"] in ("day", "exact") and day not in dates:
                point.append(row)
                dates.add(day)
        best_pair = []
        if len(point) >= 2:
            parts = _SPLIT.split(query, maxsplit=1)
            if len(parts) == 2:
                ordered = [sorted(point, key=lambda row: -len(competition_terms(part) & competition_terms(row["plain"])))
                           for part in parts]
                left = ordered[0][0]
                right = next((row for row in ordered[1] if row["id"] != left["id"]), None)
                if right and all(competition_terms(part) & competition_terms(row["plain"])
                                 for part, row in zip(parts, (left, right))):
                    best_pair = [left, right]
            if best_pair:
                spans = [TemporalSpan(row["occurrence_start"], row["occurrence_end"], row["time_precision"], "") for row in best_pair]
                direct = f"{days_between(*spans)} 天" if re.search(r"[\u3400-\u9fff]", query) else f"{days_between(*spans)} days"
                cited = best_pair
    if direct is None and compare:
        versions = [(version, row) for row in ranked for version in extract_versions(row["plain"])
                    if version.qualifier_op == "=="]
        families = {version.product_family for version, _ in versions} - {"default_pkg"}
        if len(versions) >= 2 and len(families) <= 1:
            older, newer = min(versions, key=lambda x: version_order(x[0])), max(versions, key=lambda x: version_order(x[0]))
            if older[0] != newer[0]:
                direct = f"{older[0].literal} → {newer[0].literal}"
                cited = list({row["id"]: row for row in (older[1], newer[1])}.values())
    if direct is None:
        answer = engine._answer_from_context(query, bundle)
        direct = answer["response"]
        claims = answer["claims"]
        citations = dict.fromkeys(key for claim in claims for key in claim["evidence_ids"])
        cited = [eligible[key] for key in citations]
        provisional = answer.get("provisional_item_ids", [])
    else:
        claims = [{"text": direct, "evidence_ids": [row["evidence_id"] for row in cited]}]
        provisional = []
    facts = [row for row in cited if row["kind"] in {"fact", "observation"}]
    return {"answer": direct, "cited_memory_ids": [row["id"] for row in facts],
            "cited_turn_ids": sorted({turn for row in cited for turn in row.get("source_turn_ids", [])}
                                     | {row["turn_id"] for row in cited if "turn_id" in row}),
            "temporal_chain": [_present(row) for row in sorted(facts, key=lambda row: (
                row["occurrence_start"] if row.get("occurrence_start") is not None else row.get("mention_time", 0), row["id"]))],
            "rounds_used": count, "claims": claims, "abstained": not claims, "provisional_item_ids": provisional}


def recent_observation_evidence(store, facts, scope, as_of, observations=None):
    entities = {row.get("canonical_entity_id") for row in facts} - {None}
    if not entities:
        return []
    if observations is None:
        observations = store.list_observations(user_id=scope[0], agent_id=scope[1], session_id=scope[2])
    stale = {row["canonical_entity_id"]: max((int(i[1:]) for i in row["derived_from_ids"] if i[1:].isdigit()), default=0)
             for row in observations if row.get("is_stale") and row.get("canonical_entity_id") in entities}
    if not stale:
        return []
    from memengine.storage.base import RecallFilter
    extra = store.recent_entity_facts(stale, scope=scope, after=min(stale.values()),
                                     filters=RecallFilter(as_of=as_of))
    return [row for row in extra if int(row["id"][1:]) > stale[row["canonical_entity_id"]]
            and (as_of is None or row["mention_time"] <= as_of)]
