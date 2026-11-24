"""Stable system prefix and token-bounded, per-query evidence assembly."""

from datetime import datetime, timezone
from hashlib import sha256
import re
from typing import TypedDict

from agentmem_core.reflection import recent_observation_evidence
from agentmem_core.retrieval.coverage import packed_facet_coverage
from agentmem_core.lifecycle import CURRENT_STATE_SLOT_PREFIXES, LEGACY_PREFERENCE_SUMMARY, normalized_content
from agentmem_core.temporal import competition_terms
from agentmem_core.textutil import estimate_tokens, strip_step_tags, token_units

_REFERENCE = re.compile(r"\b(?:it|that|this|they|he|she|there|the former|the latter)\b|这个|那个|这件|那件|他|她|它|上述", re.I)


class ContextBundle(TypedDict):
    system_prefix: str
    prefix_version: str
    dynamic_memory: str
    empty_recall: bool
    items: list[dict]
    evidence_map: dict[str, dict]
    evidence_turn_ids: list[int]
    token_breakdown: dict[str, int]
    gate_stats: dict
    lexical_facet_coverage: dict


def render_prefix(snapshot):
    sections = []
    if snapshot["core_profile"]:
        sections.append("# Core Profile\n" + snapshot["core_profile"])
    if snapshot["procedural_rules"]:
        sections.append("# Procedural Rules\n" + "\n".join(f"- {rule}" for rule in snapshot["procedural_rules"]))
    return "\n\n".join(sections)


def _fact_label(fact):
    domain = fact.get("epistemic_scope")
    if domain == "opinion":
        return f"[观点 置信度 {fact['confidence']:.2f}] " + (fact.get("state_tag") or "")
    if domain == "experience":
        return "[Agent历史动作] " + (fact.get("state_tag") or "")
    if fact.get("expires_at") is not None:
        return fact.get("state_tag", "")
    date = datetime.fromtimestamp(fact["event_time_start"], timezone.utc)
    if fact.get("temporal_role"):
        role = "最新" if fact["temporal_role"] == "latest" else "较早"
        return f"[{date:%Y-%m | }{role}]"
    state = "[较早]" if fact.get("state_tag") == "[历史已变更]" else fact.get("state_tag", "")
    return f"{state} [event_time={date:%Y-%m-%d}]".strip() if state else ""


def _excerpt(text, query_terms, budget):
    matches = list(re.finditer(r"\S+", text))
    if estimate_tokens(text) <= budget:
        return text
    if not matches or budget < 1:
        return ""
    pivot = max(matches, key=lambda match: len(query_terms & competition_terms(match[0]))).start()
    low, high = 0, len(text)
    while low < high:
        width = (low + high + 1) // 2
        start = max(0, min(len(text) - width, pivot - width // 3))
        if estimate_tokens(text[start:start + width]) <= budget:
            low = width
        else:
            high = width - 1
    start = max(0, min(len(text) - low, pivot - low // 3))
    return text[start:start + low].strip()


def _task_and_session(result, workspace, recent, budget, query):
    """Pack separate task and session quotas before allocating the long-term share."""
    candidates = []
    if workspace and workspace["status"] == "active":
        goal = {"kind": "task_goal", "text": workspace["goal"]}
        work = [{"kind": "working_item", "text": row["content"], **row} for row in workspace["items"]
                if row["verification_status"] != "rejected"]
        for item in work:
            if not item.get("support_current", True):
                item["verification_status"] = "unverified"
        candidates.append(("working_memory", "Active Task Working Memory", [goal, *work], min(512, budget // 4)))
    fact_text = {normalized_content(strip_step_tags(fact["text"])) for fact in result["ranked"]}
    turns = [{"kind": "session_turn", **row} for row in reversed(recent)
             if normalized_content(row["text"]) not in fact_text]
    if turns:
        candidates.append(("short_term", "Recent Session Buffer", turns, min(384, budget // 5)))

    entries = []
    terms = competition_terms(query)
    for section, heading, rows, limit in candidates:
        spent = 0
        for row in rows:
            status = row.get("verification_status", "").upper()
            tag = f"[{status}] " if status else f"[{row['role']}] " if "role" in row else "Goal: "
            label = f"## {heading}\n{tag}" if not spent else tag
            allowance = min(96, limit - spent - estimate_tokens(label) - 1)
            text = _excerpt(strip_step_tags(row["text"]), terms, allowance)
            if text:
                snippet = label + text
                entries.append((section, {**row, "text": text}, snippet))
                spent += estimate_tokens(snippet) + 1
    return entries


def _evidence_map(items):
    evidence = {}
    for item in items:
        if item["kind"] in {"fact", "observation"}:
            key = f"fact:{item['id']}"
        elif item["kind"] in {"raw_turn", "session_turn"}:
            key = f"turn:{item['turn_id']}"
        elif item["kind"] == "working_item":
            key = f"item:{item['item_id']}"
        else:
            continue
        item["evidence_id"] = key
        evidence[key] = item
    return evidence


def assemble_context(store, result, *, prefix, token_budget, drilldown_top_n, drilldown_window, query="",
                     workspace=None, recent=()) -> ContextBundle:
    tokens = {"system_prefix": estimate_tokens(prefix), "observations": 0, "facts": 0, "drilldown": 0}
    remaining = token_budget - tokens["system_prefix"]
    tail = _task_and_session(result, workspace, recent, remaining, query) if workspace or recent else []
    reserved = sum(estimate_tokens(snippet) + 1 for _, _, snippet in tail)
    remaining -= reserved
    items = []
    evidence_turn_ids = []
    snippets = []
    dynamic_units = dynamic_tokens = 0

    def append(section, item, snippet):
        nonlocal dynamic_units, dynamic_tokens
        if tail and not snippets and section in {"facts", "observations", "drilldown"}:
            snippet = "## Long-Term Evidence\n" + snippet
        candidate_units = dynamic_units + int(bool(dynamic_units)) + token_units(snippet)
        candidate_tokens = (candidate_units + 3) // 4
        if candidate_tokens > remaining:
            return False
        tokens[section] += candidate_tokens - dynamic_tokens
        dynamic_units, dynamic_tokens = candidate_units, candidate_tokens
        snippets.append(snippet)
        items.append(item)
        return True

    historical = result["parsed_query"]["query_type"] == "TIME_TRAVEL"
    facts = [fact for fact in result["ranked"] if not fact.get("is_stale")
             and (historical or fact["status"] == "active"
                  or not (fact.get("attribute_slot") or "").startswith(CURRENT_STATE_SLOT_PREFIXES))]
    if facts:
        wide = result.get("packing_mode") == "token_budget"
        scope = result["scope"]
        domains = result.get("epistemic_scopes")
        observations = store.list_observations(**scope, include_stale=True) if domains is None or "observation" in domains else []
        fresh = recent_observation_evidence(store, facts, (scope["user_id"], scope["agent_id"], scope["session_id"]),
                                            result.get("validity_as_of"), observations)
        observations = [item for item in observations if not item.get("is_stale")
                        and not item["content"].startswith(LEGACY_PREFERENCE_SUMMARY)]
        if domains is not None:
            fresh = [fact for fact in fresh if fact["epistemic_scope"] in domains]
        fact_ids = {fact["id"] for fact in facts}
        if fresh:
            facts = [*(fact for fact in facts if fact["memory_type"] == "procedural"),
                     *(fact for fact in fresh if fact["id"] not in fact_ids),
                     *(fact for fact in facts if fact["memory_type"] != "procedural")]
        if wide:
            # Preserve the measured head and make room for a few corroborated links.
            bridges = [fact for fact in facts[10:] if fact.get("protected_bridge") or fact.get("speaker_bridge")
                       and fact.get("graph_depth") == 2 and fact.get("entity_anchor")][:3]
            bridge_ids = {fact["id"] for fact in bridges}
            facts = [*facts[:10], *bridges, *(fact for fact in facts[10:] if fact["id"] not in bridge_ids)]
        if any("coverage_priority" in fact or fact.get("preference_priority") for fact in facts):
            facts.sort(key=lambda fact: (fact.get("coverage_priority", float("inf")), not fact.get("preference_priority")))
        tau = result["gate_stats"].get("tau_abs")
        if tau is None:
            tau = .12
        eligible_anchor = any(fact.get("context_anchor_turn_id") or fact.get("speaker_bridge")
                              for fact in facts[:drilldown_top_n] if fact.get("ce_score", 0) >= tau)
        reserve = min(144, remaining // 12) if wide and len(facts) > 10 and eligible_anchor else 0
        remaining -= reserve
        bodies = [strip_step_tags(fact["text"]).removeprefix(fact.get("state_tag", "")).strip() for fact in facts]
        compact = "\n".join(bodies)
        # Reserve every atom before spending tokens on labels or raw evidence.
        overhead = max(0, remaining - estimate_tokens(compact))
        selected, deferred = [], []
        for fact, body in zip(facts, bodies):
            label = _fact_label(fact)
            if fact.get("is_taboo"):
                label = f"[禁忌] {label}".rstrip()
            if fact.get("preference_priority") and fact.get("explicit_preference"):
                label = f"[偏好] {label}".rstrip()
            if label and estimate_tokens(label + " ") <= overhead:
                snippet = f"{label} {body}"
                overhead -= estimate_tokens(label + " ")
            else:
                snippet = body
            if body:
                if append("facts", {**fact, "kind": "fact"}, snippet):
                    selected.append(fact)
                elif reserve:
                    deferred.append((fact, body))

        fact_ids = {fact["id"] for fact in selected}
        source_turn_ids = {turn for fact in selected for turn in fact["source_turn_ids"]}
        seen = {body.casefold() for body in bodies}
        prefix_lines = {normalized_content(line.removeprefix("- ")) for line in prefix.splitlines() if line.strip()}
        observation_budget = 48
        observations.sort(key=lambda item: (-len(fact_ids.intersection(item["derived_from_fact_ids"])),
                                            -item["confidence"], item["id"]))
        for item in observations:
            references, turns = item["derived_from_fact_ids"], item["source_turn_ids"]
            if references and not fact_ids.intersection(references):
                continue
            if not references and turns and not source_turn_ids.intersection(turns):
                continue
            text = item["content"].strip()
            if (text and text.casefold() not in seen and normalized_content(text) not in prefix_lines
                    and tokens["observations"] + estimate_tokens("[Observation] " + text) + 1 <= observation_budget):
                if append("observations", {"kind": "observation", "text": text, **item}, f"[Observation] {text}"):
                    seen.add(text.casefold())

        remaining += reserve
        # Wide packing uses the reserved capacity only for cited, non-redundant source text.
        room = reserve if reserve else max(0, int(remaining * 0.75) - dynamic_tokens)
        capacity = min(400, int(token_budget * 0.25), room)
        anchor_facts = [fact for fact in selected[:drilldown_top_n]
                        if capacity and (fact.get("exact_ident_hit")
                                         or fact.get("ce_score", 0) >= tau)]
        # The latest source is the revision's own utterance. Earlier inherited sources stay available to tools.
        anchors = {max(fact["source_turn_ids"]) for fact in anchor_facts if fact["source_turn_ids"]}
        antecedents = {fact["context_anchor_turn_id"] for fact in anchor_facts if fact.get("context_anchor_turn_id")}
        direct = store.get_turn_window(sorted(anchors | antecedents), window=0, by_id=True, **scope) if anchors else []
        expand = [turn["turn_id"] for turn in direct if turn["turn_id"] in anchors
                  and (len(strip_step_tags(turn["text"])) < 40 or _REFERENCE.search(turn["text"]))]
        neighbors = store.get_turn_window(expand, window=drilldown_window or 1, by_id=True, **scope) if expand else []
        raw_by_id = {turn["turn_id"]: turn for turn in (*direct, *neighbors)}
        blocked = store.inactive_temporary_sources(raw_by_id.keys(), scope=(scope["user_id"], scope["agent_id"], scope["session_id"]),
                                                   as_of=result.get("validity_as_of"), include_superseded=bool(workspace or recent))
        for turn_id in blocked:
            raw_by_id.pop(turn_id, None)
        fact_terms = set().union(*(competition_terms(body) for body in bodies))
        query_terms = competition_terms(query)
        for turn_id in sorted(raw_by_id, key=lambda item: (item not in anchors, item)):
            turn = raw_by_id[turn_id]
            text = strip_step_tags(turn["text"]).strip()
            terms = competition_terms(text)
            if not terms or text.casefold() in seen or len(terms & fact_terms) / len(terms) > 0.80:
                continue
            if turn_id not in anchors and query_terms and not terms.intersection(query_terms):
                continue
            label = f"[Raw {turn_id}] "
            allowance = min(100, capacity - tokens["drilldown"], remaining - dynamic_tokens)
            allowance -= estimate_tokens(label) + 1
            excerpt = _excerpt(text, query_terms, allowance)
            if excerpt and append("drilldown", {**turn, "text": excerpt, "kind": "raw_turn"}, label + excerpt):
                evidence_turn_ids.append(turn["turn_id"])
                seen.add(text.casefold())
        for fact, body in deferred:
            append("facts", {**fact, "kind": "fact"}, body)

    remaining += reserved
    for section, item, snippet in tail:
        if item["kind"] == "session_turn" and item["turn_id"] in evidence_turn_ids:
            continue
        tokens.setdefault(section, 0)
        append(section, item, snippet)

    stats = result["gate_stats"]
    dynamic = "\n".join(snippets)
    if not dynamic and not result["empty_recall"]:
        stats = {**stats, "empty_reason": "budget"}
    total = sum(tokens.values())
    evidence = _evidence_map(items)
    facets = () if historical else result["parsed_query"].get("facets", ())
    return {"system_prefix": prefix, "prefix_version": sha256(prefix.encode("utf-8")).hexdigest()[:12],
            "dynamic_memory": dynamic, "empty_recall": not dynamic, "items": items,
            "evidence_map": evidence,
            "evidence_turn_ids": evidence_turn_ids,
            "token_breakdown": {**tokens, "dynamic_memory": dynamic_tokens, "total": total,
                                "prefix_tokens": tokens["system_prefix"], "facts_tokens": tokens["facts"],
                                "drilldown_tokens": tokens["drilldown"], "total_injected_tokens": total},
            "gate_stats": stats,
            "lexical_facet_coverage": packed_facet_coverage(facets, items, require_entity=bool(result["matched_entities"]))}
