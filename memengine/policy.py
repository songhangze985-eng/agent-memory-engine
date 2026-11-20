"""Deterministic policies used by the offline mock and as parse fallback.

These implement the three prompt contracts without calling a model:
origin placement, five-way notebook layout, and read-only System 2 selection.
"""

from __future__ import annotations

import re

from memengine.distance import MIN_REL, is_noise, is_override
from memengine.notebook import format_fact_line
from memengine.store import key_type
from memengine.textutil import char_bigrams, jaccard, normalize_entity, parse_before_ordinal

REFUSAL = "根据现有记忆，没有足够的直接关联可以确认这一点。"

def _specific_zkeys(keys) -> set[str]:
    """Identity keys used for overwrite and reachability; categories are too coarse."""
    return {normalize_entity(str(key)) for key in (keys or []) if str(key) and key_type(str(key)) == "entity"}


def _noop() -> dict:
    return {
        "action": "NOOP",
        "target_block_id": "",
        "target_line_id": "",
        "formatted_line": "",
        "z_axis_keys": [],
        "surprise_score": 0.0,
    }


def decide_origin(current_origin: dict | None, notebook_index: list, matched_z: list, query: str) -> dict:
    best = None
    best_score = -1.0
    query_grams = char_bigrams(query)
    for block in notebook_index or []:
        preview = f"{block.get('topic', '')} {block.get('preview', '')}"
        score = jaccard(query_grams, char_bigrams(preview))
        if score > best_score:
            best_score = score
            best = block
    if best is None:
        x_topic = "未归类"
        y_sub = "总览"
    else:
        x_topic = best.get("topic") or "未归类"
        y_sub = best.get("preview_attr") or best.get("block_id") or "总览"
    anchor = (query or "").strip()[:80]
    multi = bool(re.search(r"之前|最早|原本|以前|上次|第[0-9一二两三四五六七八九十]+次|对比", query or ""))
    radius = 0.35 if multi else 0.25

    def overlap(entity: str) -> float:
        name = entity.split(":", 1)[-1]
        return jaccard(char_bigrams(name), query_grams)

    order = sorted(matched_z or [], key=lambda entity: (overlap(entity), entity))
    shifted = True
    if current_origin and current_origin.get("x_topic") == x_topic and current_origin.get("decision_anchor") == anchor:
        shifted = False
    return {
        "origin_shifted": shifted,
        "new_origin": {
            "x_topic": x_topic,
            "y_subtopic": y_sub,
            "decision_anchor": anchor or "未命名决策",
        },
        "poison_circle": {
            "radius_r": radius,
            "z_axis_scan_order_far_to_near": order,
        },
    }


def _find_match(flat, fact):
    attribute = fact.get("attribute") or ""
    if not attribute:
        return None
    keys = _specific_zkeys(fact.get("zkeys") or [])
    if not keys:
        return None
    matched = []
    for block, line in flat:
        if (line.get("attribute") or "") != attribute:
            continue
        if fact.get("delete") and fact.get("value") and str(line.get("value")) != str(fact["value"]):
            continue
        line_keys = _specific_zkeys(line.get("zkeys") or [])
        identities = {normalize_entity(key) for key in fact.get("zkeys", []) if key.startswith(("人物:", "person:"))}
        if (identities or keys).intersection(line_keys):
            if not fact.get("delete"):
                return block, line
            matched.append((block, line))
    return matched[0] if len(matched) == 1 else None


def decide_layout(candidate_blocks: list, facts: list, timestamp) -> dict:
    blocks = [dict(block, lines=[dict(line) for line in block.get("lines") or []]) for block in (candidate_blocks or [])]
    flat = [(block, line) for block in blocks for line in block["lines"]]
    operations = []
    for fact in facts or []:
        if fact.get("is_greeting"):
            operations.append(_noop())
            continue
        match = _find_match(flat, fact)
        if fact.get("delete"):
            operations.append({**_noop(), "action": "DELETE_LINE", "target_line_id": match[1]["id"]} if match else _noop())
            if match:
                flat.remove(match)
            continue
        if match is not None:
            block, line = match
            if str(line.get("value")) == str(fact.get("value")):
                operations.append({**_noop(), "target_line_id": line["id"]})
                continue
            operations.append(
                {
                    "action": "OVERWRITE_SAME_LINE",
                    "target_block_id": block["id"],
                    "target_line_id": line["id"],
                    "formatted_line": format_fact_line(timestamp, fact["text"]),
                    "z_axis_keys": list(fact.get("zkeys") or []),
                    "surprise_score": 0.86,
                    "attribute": fact.get("attribute"),
                    "value": fact.get("value"),
                    "topic": block["topic"],
                    "is_conflict_or_override": True,
                }
            )
            line["value"] = fact.get("value")
            line["text"] = fact["text"]
            line["zkeys"] = list(fact.get("zkeys") or [])
            continue
        topic = fact.get("topic") or "杂项"
        block = next((item for item in blocks if item.get("topic") == topic), None)
        if fact.get("isolated"):
            action = "INSERT_ISOLATED_LINE"
            target = f"NEW:ISO:{len(operations)}:{topic}"
            surprise = 0.90
            block = None
        elif block is not None:
            action = "APPEND_ADJACENT_LINE"
            target = block["id"]
            surprise = 0.25
        else:
            action = "APPEND_ADJACENT_LINE"
            target = f"NEW:{topic}"
            surprise = 0.25
            created = {"id": target, "topic": topic, "lines": []}
            blocks.append(created)
            block = created
        op = {
            "action": action,
            "target_block_id": target,
            "target_line_id": "",
            "formatted_line": format_fact_line(timestamp, fact["text"]),
            "z_axis_keys": list(fact.get("zkeys") or []),
            "surprise_score": surprise,
            "attribute": fact.get("attribute"),
            "value": fact.get("value"),
            "topic": topic,
            "operation_id": f"pending-{len(operations) + 1}",
            "is_conflict_or_override": bool(fact.get("is_conflict_or_override")),
        }
        operations.append(op)
        if block is not None and action == "APPEND_ADJACENT_LINE":
            block["lines"].append(
                {
                    "id": f"pending-{len(operations)}",
                    "text": fact["text"],
                    "attribute": fact.get("attribute") or "",
                    "value": fact.get("value") or "",
                    "zkeys": list(fact.get("zkeys") or []),
                }
            )
            flat.append((block, block["lines"][-1]))
    return {"write_operations": operations}


def _guess_attribute(query: str, facts: list[dict]) -> str:
    found = []
    for fact in facts:
        attribute = fact.get("attribute") or ""
        if attribute and attribute in query and attribute not in found:
            found.append(attribute)
    if found:
        return max(found, key=len)
    if "周会" in query and any((fact.get("attribute") == "周会时间") for fact in facts):
        return "周会时间"
    if "密码" in query:
        for fact in facts:
            if "密码" in (fact.get("attribute") or ""):
                return fact["attribute"]
    if "代号" in query and any(fact.get("attribute") == "代号" for fact in facts):
        return "代号"
    if "钥匙" in query or "印章" in query or "档案" in query:
        if any(fact.get("attribute") == "存放位置" for fact in facts):
            return "存放位置"
    return ""


def travel_revision(query: str):
    index = parse_before_ordinal(query)
    if index is not None:
        return index
    if re.search(r"最早|原本|以前|\b(?:originally|original|first)\b", query or "", re.I):
        return 0
    if re.search(r"之前|上次|\b(?:last time|before|earlier|previously|used to)\b", query or "", re.I):
        return "previous"
    return None


def _dedupe(facts: list[dict]) -> list[dict]:
    kept = []
    seen = set()
    for fact in facts:
        marker = fact.get("text") or fact.get("id")
        if marker in seen:
            continue
        seen.add(marker)
        kept.append(fact)
    return kept


def _select_revision(facts: list[dict], query: str, target) -> list[dict]:
    attribute = _guess_attribute(query, facts)
    pool = [fact for fact in facts if fact.get("attribute") == attribute] if attribute else list(facts)
    if not pool:
        pool = list(facts)
    if target == "previous":
        max_rev = max(int(fact.get("revision") or 0) for fact in pool)
        target_rev = max(0, max_rev - 1)
    else:
        target_rev = int(target)
    chosen = [fact for fact in pool if int(fact.get("revision") or 0) == target_rev]
    if not chosen:
        lowers = [fact for fact in pool if int(fact.get("revision") or 0) <= target_rev]
        band = lowers or pool
        best = max(int(fact.get("revision") or 0) for fact in band) if lowers else min(int(fact.get("revision") or 0) for fact in band)
        if not lowers:
            best = min(int(fact.get("revision") or 0) for fact in pool)
            band = pool
        chosen = [fact for fact in band if int(fact.get("revision") or 0) == best]
    return _dedupe(chosen)


def plan_answer(facts: list, query: str, query_type="FACTUAL", matched_entities=(), temporal_intent=None) -> dict:
    """Read-only. Does not accept a store and does not return write operations."""
    kept, rejected = [], []
    local = {normalize_entity(key) for key in matched_entities}
    related_events = {normalize_entity(key) for fact in facts if fact.get("entity_anchor")
                      for key in fact.get("entity_z_keys") or () if key.startswith(("事件:", "event:"))}
    for fact in facts:
        weak_exception = ((is_override(fact) or fact.get("isolated")) and not fact.get("historical")
                          and not fact.get("entity_anchor") and fact.get("vector_score") is not None
                          and (fact["vector_score"] <= MIN_REL or not local))
        if is_noise(fact) or weak_exception:
            rejected.append(fact["id"])
        else:
            kept.append(fact)
    historical = [fact for fact in kept if fact.get("historical")]
    travel = travel_revision(query)
    timed = [fact for fact in kept if fact.get("temporal_match")] if query_type == "TIME_TRAVEL" else []
    intent = temporal_intent or {}
    if timed and intent.get("order_hint") in {"earliest", "latest"}:
        per_chain = {}
        for fact in timed:
            per_chain.setdefault(fact.get("root_fact_id") or fact["id"], fact)
        chosen = _dedupe(list(per_chain.values()))
    elif timed and any(intent.get(key) is not None for key in ("start_ts", "end_ts", "rev_index")):
        chosen = _dedupe(timed)
    elif isinstance(travel, int) and historical:
        attribute = _guess_attribute(query, facts)
        scoped = [fact for fact in facts if fact.get("attribute") == attribute] if attribute else list(facts)
        latest = max((int(fact.get("revision") or 0) for fact in scoped), default=-1)
        current = [fact for fact in scoped if not fact.get("historical") and int(fact.get("revision") or 0) == latest]
        # Before the Nth change is revision N-1. Past the newest stored revision, that is the current row.
        if latest >= 0 and travel >= latest and current:
            chosen = _dedupe(current)
            chosen_ids = {fact.get("id") for fact in chosen}
            rejected = [item for item in rejected if item not in chosen_ids]
        else:
            chosen = _select_revision(historical, query, travel)
    elif timed:
        chosen = _dedupe(timed)
    elif historical and (travel is not None or query_type == "TIME_TRAVEL"):
        chosen = _dedupe(historical)
    else:
        chosen = _dedupe([fact for fact in kept if not fact.get("historical")])
    if query_type != "REFLECTIVE":
        attribute = _guess_attribute(query, chosen)
        if attribute:
            chosen = [fact for fact in chosen if fact.get("attribute") == attribute]
        anchored = [fact for fact in chosen if fact.get("entity_anchor")]
        if anchored:
            chosen = anchored
        elif local and related_events:
            relevant = [fact for fact in chosen if related_events.intersection(
                normalize_entity(key) for key in fact.get("entity_z_keys") or ())]
            if relevant:
                chosen = relevant
    parts = []
    for fact in chosen:
        text = (fact.get("text") or "").strip().rstrip("。")
        if text and text not in parts:
            parts.append(text)
    synthesis = "。".join(parts)
    return {
        "evidence_synthesis": synthesis,
        "anti_intuitive_override": any(fact.get("historical") or fact.get("is_conflict_or_override")
                                       or fact.get("isolated") for fact in chosen),
        "rejected_noise_ids": rejected,
        "final_answer": synthesis + "。" if synthesis else REFUSAL,
    }

