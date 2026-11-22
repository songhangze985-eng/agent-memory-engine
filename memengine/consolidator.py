"""Explicit, deterministic distillation of supported facts and tool successes."""

from collections import defaultdict
import json
import re

from memengine.assembler import _excerpt, render_prefix
from memengine.chrono import date_iso
from memengine.distance import is_noise
from memengine.lifecycle import content_hash, normalized_content
from memengine.notebook import format_fact_line
from memengine.textutil import estimate_tokens, normalize_entity, strip_step_tags

_PREFERENCE = re.compile(r"(?:偏好|更喜欢|喜欢|常用)\s*([^，。；.!?]{1,40})|\b(?:prefer|really like|always use)\s+([^,.!?;]{1,55})", re.I)
_NEGATION = re.compile(r"不|没|未曾|\b(?:not|never|no|don't|doesn't|didn't)\b", re.I)


def write_memory(store, text, *, scope, now, memory_type, source_turn_ids, derived_from_ids=(), tool_call_ids=(),
                 trigger_text="", successful=(), confidence=1.0, is_constraint=False, zkeys=(), offline=False,
                 epistemic_scope=None, valid_from=None, expires_at=None, chronicle=None):
    body = f"When {trigger_text}: {text}" if trigger_text else text
    epistemic_scope = epistemic_scope or (memory_type if memory_type != "fact" else "world")
    with store.transaction():
        existing = store.content_matches(content_hash(body), scope, memory_type=memory_type)
        existing = [row for row in existing if row["epistemic_scope"] == epistemic_scope
                    and row["valid_from"] == valid_from and row["expires_at"] == expires_at]
        if not chronicle and len(existing) == 1 and existing[0]["is_constraint"] >= int(is_constraint):
            if memory_type == "observation" and (
                    existing[0]["is_stale"] or set(existing[0]["derived_from_ids"]) - set(derived_from_ids)):
                return store.supersede_fact(existing[0]["id"], plain=body, formatted=format_fact_line(now, body),
                                            scope=scope, ingest_time=now, source_turn_ids=source_turn_ids,
                                            chronicle={"mention_time": now, "derived_from_ids": list(derived_from_ids)})
            store.merge_derived_sources(existing[0], derived_from_ids, source_turn_ids, tool_call_ids, successful=successful)
            if epistemic_scope == "opinion":
                store.update_opinion_confidence(existing[0]["id"], scope, confidence)
            return existing[0]["id"]
        block = store.create_block({"fact": "记忆", "observation": "观察", "procedural": "执行规则"}[memory_type], now, scope=scope)
        return store.insert_fact(block_id=block, formatted=format_fact_line(now, body), plain=body, scope=scope,
                                 ingest_time=now, memory_type=memory_type, source_turn_ids=source_turn_ids,
                                 derived_from_ids=derived_from_ids, confidence=confidence, is_constraint=is_constraint,
                                 trigger_text=trigger_text, tool_call_ids=tool_call_ids, success_count=len(successful),
                                 zkeys=zkeys, offline=offline, epistemic_scope=epistemic_scope,
                                 valid_from=valid_from, expires_at=expires_at, chronicle=chronicle)


def refresh_profile(store, scope, now):
    facts = store.snapshot(scope=scope, include_cold=False, include_derived=True)
    observations = sorted(store.list_observations(user_id=scope[0], agent_id=scope[1]),
                          key=lambda fact: (-len(fact["source_turn_ids"]), -fact["confidence"], fact["plain"]))
    rules = sorted((fact for fact in facts if fact["memory_type"] == "procedural"),
                   key=lambda fact: (-fact["success_count"], fact["plain"]))
    constraints = sorted({fact["plain"] for fact in facts if fact["memory_type"] == "fact" and fact["is_constraint"] and fact["expires_at"] is None
                          and fact["epistemic_scope"] == "world"})
    # Stable order and text, with counts used only for choosing the rule set.
    profile, selected_rules = [], []
    stable = [fact["content"] for fact in observations if fact["confidence"] >= 0.8 and not fact.get("canonical_entity_id") and not fact.get("is_stale")
              and (len(fact["source_turn_ids"]) >= 3 or fact["content"].startswith("多轮明确表达偏好"))]
    seen = set()
    for target, choices in ((profile, constraints), (selected_rules, [fact["plain"] for fact in rules]),
                            (profile, stable)):
        for line in choices:
            clean = " ".join(strip_step_tags(line).split())
            key = normalized_content(clean)
            if key in seen or clean in target:
                continue
            seen.add(key)
            target.append(clean)
            prefix = render_prefix({"core_profile": "\n".join(profile), "procedural_rules": sorted(selected_rules)})
            if estimate_tokens(prefix) > 360:
                target.pop()
    return store.set_core_snapshot("\n".join(profile), sorted(selected_rules), user_id=scope[0], agent_id=scope[1], timestamp=now)


def consolidation_key(scope):
    return "consolidation:" + json.dumps(scope[:2], ensure_ascii=False)


def consolidate(store, *, user_id, agent_id, now, min_new_facts, force):
    owner = (user_id, agent_id, None)
    scope = (user_id, agent_id, "default")
    key = consolidation_key(owner)
    with store.transaction():
        cursor = json.loads(store.get_meta(key) or '{"fact":0,"tool":0}')
        where = "user_id=? AND agent_id=? AND status='active' AND memory_type='fact'"
        fresh = store.conn.execute(f"SELECT count(*) FROM facts WHERE {where} AND cast(substr(id,2) AS INTEGER)>?",
                                   (user_id, agent_id, cursor["fact"])).fetchone()[0]
        if not force and fresh < min_new_facts:
            return {"skipped": True, "new_facts": fresh, "observation_ids": [], "procedural_ids": []}
        seq = int(store.get_meta("seq_l") or 0)
        facts = [fact for fact in store.snapshot(scope=owner, include_cold=False)
                 if fact["source_turn_ids"] and not is_noise(fact) and fact["expires_at"] is None
                 and fact["epistemic_scope"] == "world"]
        tools = store.tool_calls(owner, after=cursor["tool"])
        groups = defaultdict(list)
        for fact in facts:
            for tag in fact["entity_z_keys"] or fact["category_z_keys"]:
                groups[f"多轮记录涉及「{normalize_entity(tag)}」。"].append(fact)
            if not _NEGATION.search(fact["plain"]) and (preference := _PREFERENCE.search(fact["plain"])):
                value = normalized_content(preference[1] or preference[2])
                if value:
                    groups[f"多轮明确表达偏好「{value}」。"].append(fact)
        observations, procedures = [], []
        for summary, group in sorted(groups.items()):
            refs = list(dict.fromkeys(fact["id"] for fact in group))
            turns = sorted({turn_id for fact in group for turn_id in fact["source_turn_ids"]})
            if len(refs) < 2 or len(turns) < 2:
                continue
            observations.append(write_memory(store, summary, scope=scope, now=now,
                                               memory_type="observation", source_turn_ids=turns, derived_from_ids=refs,
                                               confidence=0.85, offline=True))
        entity_groups = defaultdict(list)
        history = store.snapshot(scope=owner, include_cold=True, include_superseded=True)
        active_roots = {fact["root_fact_id"] for fact in history if fact["status"] == "active"}
        for fact in history:
            if (fact.get("canonical_entity_id") and fact["epistemic_scope"] in ("world", "experience")
                    and fact["expires_at"] is None and fact["root_fact_id"] in active_roots):
                entity_groups[fact["canonical_entity_id"]].append(fact)
        existing = {row["canonical_entity_id"]: row for row in store.list_observations(user_id=user_id, agent_id=agent_id, include_stale=True)
                    if row.get("canonical_entity_id")}
        for entity, group in sorted(entity_groups.items()):
            group = sorted(group, key=lambda row: (row.get("occurrence_start") or row["mention_time"], row["id"]))[-5:]
            if len(group) < 2:
                continue
            chain = " → ".join(f"[{date_iso(row.get('occurrence_start') or row['mention_time'])}] "
                               f"{_excerpt(strip_step_tags(row['plain']), set(), 30)}" for row in group)
            refs = [row["id"] for row in group]
            turns = sorted({t for row in group for t in row["source_turn_ids"]})
            details = {"canonical_entity_id": entity, "mention_time": now, "derived_from_ids": refs}
            old = existing.get(entity)
            if old and old["plain"] == chain and not old["is_stale"]:
                continue
            if old:
                fact_id = store.supersede_fact(old["id"], plain=chain, formatted=chain, source_turn_ids=turns,
                                               scope=scope, ingest_time=now, chronicle=details)
            else:
                block = store.create_block("实体档案", now, scope=scope)
                fact_id = store.insert_fact(block_id=block, plain=chain, formatted=chain, source_turn_ids=turns,
                                            scope=scope, ingest_time=now, memory_type="observation", confidence=.85,
                                            chronicle=details)
            if fact_id:
                observations.append(fact_id)
        for tool in tools:
            if not tool["success"] or tool.get("task_id") is not None:
                continue
            supports = [fact for fact in facts if fact["id"] in tool["derived_from_ids"] or
                        set(fact["source_turn_ids"]).intersection(tool["source_turn_ids"])]
            refs = [fact["id"] for fact in supports]
            turns = sorted({*tool["source_turn_ids"], *(turn for fact in supports for turn in fact["source_turn_ids"])})
            if not refs or not turns:
                continue
            procedures.append(write_memory(store, tool["technique"], scope=scope, now=now, memory_type="procedural",
                                             trigger_text=tool["trigger_text"], source_turn_ids=turns, derived_from_ids=refs,
                                             tool_call_ids=[tool["tool_call_id"]], successful=[tool["tool_call_id"]], offline=True))
        version = refresh_profile(store, owner, now)
        store.set_meta(key, json.dumps({"fact": seq, "tool": tools[-1]["seq"] if tools else cursor["tool"]}))
        return {"skipped": False, "new_facts": fresh, "observation_ids": observations,
                "procedural_ids": list(dict.fromkeys(procedures)), "prefix_version": version}
