"""Deterministic policies used by the offline mock and as parse fallback.

These implement the three prompt contracts without calling a model:
origin placement, four-way notebook layout, and read-only system-2 selection.
"""

from __future__ import annotations

import re

from memengine.notebook import format_fact_line
from memengine.textutil import char_bigrams, jaccard, parse_cn_number

REFUSAL = "根据现有记忆，没有足够的直接关联可以确认这一点。"

_ORDINAL = re.compile(r"第([0-9]+|[一二两三四五六七八九十])次")


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
    radius = 0.62 if multi else 0.45

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
            "anti_intuitive_weight": 0.90,
            "intuitive_weight": 0.40,
            "z_axis_scan_order_far_to_near": order,
        },
    }


def _find_match(flat, fact):
    attribute = fact.get("attribute") or ""
    if not attribute:
        return None
    keys = set(fact.get("zkeys") or [])
    if not keys:
        return None
    for block, line in flat:
        if (line.get("attribute") or "") != attribute:
            continue
        if keys.intersection(line.get("zkeys") or []):
            return block, line
    return None


def decide_layout(candidate_blocks: list, facts: list, timestamp) -> dict:
    blocks = [dict(block, lines=[dict(line) for line in block.get("lines") or []]) for block in (candidate_blocks or [])]
    flat = [(block, line) for block in blocks for line in block["lines"]]
    operations = []
    for fact in facts or []:
        if fact.get("is_greeting"):
            operations.append(_noop())
            continue
        match = _find_match(flat, fact)
        if match is not None:
            block, line = match
            if str(line.get("value")) == str(fact.get("value")):
                operations.append(_noop())
                continue
            operations.append(
                {
                    "action": "OVERWRITE_SAME_LINE",
                    "target_block_id": block["id"],
                    "target_line_id": line["id"],
                    "formatted_line": format_fact_line(timestamp, fact["text"]),
                    "z_axis_keys": list(fact.get("zkeys") or []),
                    "surprise_score": 0.86,
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
            surprise = 0.28
        else:
            action = "APPEND_ADJACENT_LINE"
            target = f"NEW:{topic}"
            surprise = 0.28
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


def _reachable(facts: list[dict], query: str) -> tuple[list[dict], list[str]]:
    seeds = []
    for fact in facts:
        for key in fact.get("zkeys") or []:
            name = key.split(":", 1)[-1]
            if name and name in query:
                seeds.append(fact)
                break
    frontier = set()
    for fact in seeds:
        frontier.update(fact.get("zkeys") or [])
    reachable_ids: set[str] = set()
    changed = True
    while changed:
        changed = False
        for fact in facts:
            if fact["id"] in reachable_ids:
                continue
            if frontier.intersection(fact.get("zkeys") or []):
                reachable_ids.add(fact["id"])
                before = len(frontier)
                frontier.update(fact.get("zkeys") or [])
                if len(frontier) != before:
                    changed = True
    kept = [fact for fact in facts if fact["id"] in reachable_ids]
    rejected = [fact["id"] for fact in facts if fact["id"] not in reachable_ids]
    return kept, rejected


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


def _travel_target(query: str, facts: list[dict]):
    match = _ORDINAL.search(query or "")
    if match and "之前" in query:
        return parse_cn_number(match.group(1)) - 1
    if re.search(r"最早|原本|以前", query or ""):
        return 0
    if re.search(r"之前|上次", query or ""):
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


def _better(candidate: dict, incumbent: dict) -> bool:
    rev_c = int(candidate.get("revision") or 0)
    rev_i = int(incumbent.get("revision") or 0)
    if rev_c != rev_i:
        return rev_c > rev_i
    outer_c = candidate.get("zone") == "OUTER_ANTI_INTUITIVE"
    outer_i = incumbent.get("zone") == "OUTER_ANTI_INTUITIVE"
    if outer_c != outer_i:
        return outer_c
    return float(candidate.get("updated_at") or 0) >= float(incumbent.get("updated_at") or 0)


def _latest_per_attribute(facts: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    order: list[str] = []
    for fact in facts:
        key = fact.get("attribute") or fact["id"]
        if key not in best:
            order.append(key)
            best[key] = fact
        elif _better(fact, best[key]):
            best[key] = fact
    return [best[key] for key in order]


def _refuse(rejected: list[str]) -> dict:
    return {
        "reasoning_trace": {
            "anti_intuitive_overrides": [],
            "rejected_noise_ids": list(rejected),
            "verification_status": "PARTIAL_INFO",
        },
        "final_response": REFUSAL,
    }


def plan_answer(origin: dict | None, outer: list, inner: list, query: str) -> dict:
    """Read-only. Does not accept a store and does not return write operations."""
    del origin  # the caller already placed facts relative to the origin
    tagged = []
    for fact in outer or []:
        item = dict(fact)
        item["zone"] = item.get("zone") or "OUTER_ANTI_INTUITIVE"
        tagged.append(item)
    for fact in inner or []:
        item = dict(fact)
        item["zone"] = item.get("zone") or "INNER_INTUITIVE"
        tagged.append(item)
    kept, rejected = _reachable(tagged, query or "")
    if not kept:
        return _refuse(rejected)
    travel = _travel_target(query, kept)
    overrides: list[str] = []
    if travel is not None:
        chosen = _select_revision(kept, query, travel)
        status = "VERIFIED" if chosen else "PARTIAL_INFO"
        if chosen:
            overrides.append("按时间回溯采用修改前的快照，不用当前最新值")
    else:
        live = [fact for fact in kept if not fact.get("historical")] or kept
        chosen = _latest_per_attribute(live)
        if any(fact.get("zone") == "OUTER_ANTI_INTUITIVE" for fact in chosen):
            overrides.append("采用圈外反直觉事实，覆盖同属性的圈内直觉结论")
        outer_attrs = {fact.get("attribute") for fact in chosen if fact.get("zone") == "OUTER_ANTI_INTUITIVE"}
        inner_attrs = {fact.get("attribute") for fact in live if fact.get("zone") == "INNER_INTUITIVE"}
        status = "CONFLICT_RESOLVED" if outer_attrs & inner_attrs else "VERIFIED"
    parts = []
    for fact in chosen:
        text = (fact.get("text") or "").strip().rstrip("。")
        if text and text not in parts:
            parts.append(text)
    if not parts:
        return _refuse(rejected)
    return {
        "reasoning_trace": {
            "anti_intuitive_overrides": overrides,
            "rejected_noise_ids": rejected,
            "verification_status": status,
        },
        "final_response": "。".join(parts) + "。",
    }

