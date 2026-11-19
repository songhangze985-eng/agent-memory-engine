"""Scoped 2D notebook, poison-circle retrieval and read-only System 2."""

from __future__ import annotations

import json
import math
import re
import time
from html import escape
from typing import Callable

from memengine.categories import CATEGORY_ENUM, GENERAL_CATEGORY, attach_dual_zkeys
from memengine.distance import EPSILON, INNER, MIN_REL, OUTER, score_fact, semantic_origin_text, unit_interval
from memengine.embeddings import OpenAICompatibleEmbedder
from memengine.extract import extract_facts, is_greeting
from memengine.llm import FatalLLMError, MockLLMClient, loads_json_object
from memengine.notebook import ACTIONS, apply_write_op, format_fact_line, line_body, render_block
from memengine.policy import decide_layout, decide_origin, plan_answer, travel_revision
from memengine.prompts import render_prompt
from memengine.queryparse import normalize_parsed_query, parse_query_rules
from memengine.router import CASUAL, TASK, entity_names, match_entities, route_query
from memengine.store import DEFAULT_SCOPE, Store, key_type
from memengine.textutil import char_bigrams, estimate_tokens, jaccard, step_ids_in
from memengine.timeline import TimelineBook
from memengine.vector import HashedTfidfEmbedder, weighted_rrf
from memengine.zaxis import collapse_and_scan_z_axis, expand_entities, order_far_to_near

NEWS_TOPIC = "时事杂讯"
NEWS_KEY = "话题:时事杂讯"
_RECALL_CUE = re.compile(r"记得|回忆|偏好|爱看|喜欢|\b(?:remember|recall|prefer|like)\b", re.I)
_QUESTION_CUE = re.compile(r"[?？]|啥|什么|哪|\b(?:what|which|who|when|where|why|how)\b", re.I)
_DELETE_CUE = re.compile(r"忘掉|忘记|删除|撤销|取消|不再|已经不|\b(?:forget|remove|delete|cancel|no longer|not anymore)\b", re.I)


class CognitiveMemoryEngine:
    def __init__(self, db_path: str = ":memory:", llm=None, delta_t_buffer: float = 10.0,
                 epsilon: float = EPSILON, enable_timeline_buffer: bool = True,
                 enable_casual_bypass: bool = True, clock: Callable[[], float] | None = None,
                 vector=None, *, categories: list[str] | None = None):
        self.epsilon = float(epsilon)
        if not 0 < self.epsilon <= 1:
            raise ValueError("epsilon must be in (0, 1]")
        self.store = Store(db_path)
        self.llm = llm if llm is not None else MockLLMClient()
        self.vector = vector if vector is not None else OpenAICompatibleEmbedder.from_env() or HashedTfidfEmbedder()
        self.enable_casual_bypass = bool(enable_casual_bypass)
        self.categories = tuple(dict.fromkeys(name.strip() for name in (categories or []) if name.strip()
                                              and name.strip() not in [*CATEGORY_ENUM, GENERAL_CATEGORY]))
        self.timeline = TimelineBook(delta_t_buffer, enabled=enable_timeline_buffer)
        self.timeline.layer1 = self.store.load_timeline(1)
        self.timeline.layer2 = self.store.load_timeline(2)
        self.timeline.events = int(self.store.get_meta("timeline_events") or 0)
        self.origin, self.radius = self._control_state(DEFAULT_SCOPE)
        self.pending: list[dict] = []
        self.fallback_count = 0
        self._base_clock = clock
        self._clock = clock

    def _now(self) -> float:
        return float(self._clock()) if self._clock is not None else time.time()

    def set_time(self, timestamp: float) -> None:
        stamp = float(timestamp)
        self._clock = lambda: stamp

    def reset_clock(self) -> None:
        self._clock = self._base_clock

    @staticmethod
    def _control_key(scope):
        return "control:" + json.dumps(scope, ensure_ascii=False)

    def _control_state(self, scope):
        raw = self.store.get_meta(self._control_key(scope))
        if raw is not None:
            data = json.loads(raw)
            return data["origin"], data["radius"]
        if scope == DEFAULT_SCOPE:
            legacy = self.store.get_meta("origin")
            radius = unit_interval(self.store.get_meta("radius"), 0.45)
            return (json.loads(legacy) if legacy else None), max(self.epsilon, radius)
        return None, max(self.epsilon, 0.45)

    def _record_timeline(self, layer, payload, now, scope, event_time=None):
        position, last = self.store.last_timeline_node(layer, scope=scope)
        tail = [last] if last else []
        self.timeline.buffer(tail, payload, now)
        if last is None or len(tail) > 1:
            position += 1
        node = tail[-1]
        self.store.save_timeline_node(layer, position, node, scope=scope, event_time=event_time)
        if scope == DEFAULT_SCOPE:
            events = int(self.store.get_meta("timeline_events") or 0) + 1
            self.store.set_meta("timeline_events", str(events))
            return layer, position, node, events
        return None

    def _refresh_timeline(self, change):
        if change is None:
            return
        layer, position, node, events = change
        nodes = self.timeline.layer1 if layer == 1 else self.timeline.layer2
        if events != self.timeline.events + 1 or position > len(nodes):
            self.timeline.layer1 = self.store.load_timeline(1)
            self.timeline.layer2 = self.store.load_timeline(2)
        else:
            nodes[position:position + 1] = [node]
        self.timeline.events = events

    @property
    def _allowed_categories(self):
        return [*CATEGORY_ENUM, *self.categories, GENERAL_CATEGORY]

    def add(self, text: str, timestamp: float | None = None, sync: bool = True, *, user_id="default",
            agent_id="default", session_id="default", role="user", event_time=None) -> dict:
        now = float(timestamp) if timestamp is not None else self._now()
        scope = (user_id, agent_id, session_id)
        facts = extract_facts(text or "") if isinstance(self.llm, MockLLMClient) else []
        job = {"text": text or "", "ts": now, "scope": scope, "role": role, "facts": facts,
               "event_time": float(event_time) if event_time is not None else now}
        if not sync:
            self.pending.append(job)
            return {"facts": facts, "operations": [], "queued": True}
        self.drain(user_id=user_id, agent_id=agent_id)
        return self._apply_job(job)

    def drain(self, *, user_id=None, agent_id=None) -> list[dict]:
        applied = []
        for job in tuple(self.pending):
            if ((user_id is not None and job["scope"][0] != user_id)
                    or (agent_id is not None and job["scope"][1] != agent_id)):
                continue
            applied.extend(self._apply_job(job)["operations"])
            self.pending.remove(job)
        return applied

    def _candidate_blocks(self, text, scope, facts):
        blocks = self.store.blocks_for_prompt(scope=scope)
        graph_keys = {key for block in blocks for line in block["lines"] for key in line["entity_z_keys"]}
        explicit = set(match_entities(text, graph_keys))
        explicit.update(key for fact in facts for key in fact.get("zkeys", []) if key_type(key) == "entity")
        topics = {fact.get("topic") for fact in facts}
        query_grams = char_bigrams(text)
        ranked = sorted(blocks, key=lambda block: (block["topic"] in topics, jaccard(
            query_grams, char_bigrams(block["topic"] + " " + " ".join(line["plain"] for line in block["lines"][-3:])))), reverse=True)
        candidates = []
        for rank, block in enumerate(ranked):
            lines = block["lines"]
            pinned = {line["id"] for line in lines if explicit.intersection(line["entity_z_keys"])}
            if rank >= 8 and not pinned:
                continue
            recent = {line["id"] for line in lines[-5:]} if rank < 8 else set()
            fields = ("id", "text", "formatted", "attribute", "value", "zkeys", "isolated")
            candidates.append({"id": block["id"], "topic": block["topic"], "lines": [
                {field: line[field] for field in fields} for line in lines if line["id"] in pinned | recent]})
        return candidates

    def _apply_job(self, job) -> dict:
        text, now, scope = job["text"], job["ts"], job["scope"]
        facts = job["facts"]
        step_ids = step_ids_in(text)
        greeting = is_greeting(text)
        candidates = [] if greeting else self._candidate_blocks(text, scope, facts)
        change = None
        if greeting:
            ops = []
        elif isinstance(self.llm, MockLLMClient):
            ops = decide_layout(candidates, facts, now)["write_operations"]
            if not facts and len(text.strip()) >= 16:
                existing = next((block for block in candidates if block["topic"] == NEWS_TOPIC), None)
                ops = [{"action": "APPEND_ADJACENT_LINE", "target_block_id": existing["id"] if existing else "NEW",
                        "topic": NEWS_TOPIC, "formatted_line": format_fact_line(now, text), "z_axis_keys": [NEWS_KEY]}]
        else:
            prompt = render_prompt("layout", step_id=str(step_ids[0]) if step_ids else "",
                                   candidate_2d_blocks=candidates, raw_turn_text=escape(text), role=escape(job["role"]),
                                   timestamp=now, categories=self._allowed_categories)
            def fallback_layout():
                nonlocal facts
                facts = extract_facts(text)
                return decide_layout(candidates, facts, now)

            data, fallback = self._call_json(prompt, fallback_layout, _valid_operations)
            ops = data["write_operations"]
        allowed_blocks = {block["id"]: block for block in candidates}
        allowed_lines = {line["id"]: (block, line) for block in candidates for line in block["lines"]}
        alias, applied = {}, []
        with self.store.transaction():
            self.store.append_ingest(text, now, scope=scope, role=job["role"], event_time=job["event_time"])
            for op in ops:
                action = op["action"]
                target = op.get("target_line_id") or ""
                block = op.get("target_block_id") or "NEW"
                if action in {"OVERWRITE_SAME_LINE", "DELETE_LINE"} and target not in allowed_lines and target not in alias:
                    continue
                if action == "DELETE_LINE" and not _DELETE_CUE.search(text):
                    continue
                if action in {"APPEND_ADJACENT_LINE", "INSERT_ISOLATED_LINE"} and block not in allowed_blocks and block != "NEW" and not block.startswith("NEW:"):
                    continue
                plain = line_body(op.get("formatted_line") or "")
                target_block, target_line = allowed_lines.get(target, ({}, {}))
                topic = target_block.get("topic") or allowed_blocks.get(block, {}).get("topic") or op.get("topic") or GENERAL_CATEGORY
                attribute = target_line.get("attribute") or op.get("attribute") or ""
                if action == "OVERWRITE_SAME_LINE" and target_line.get("attribute") and op.get("attribute") not in (None, "", attribute):
                    continue
                zkeys = [NEWS_KEY] if topic == NEWS_TOPIC else attach_dual_zkeys(
                    op.get("z_axis_keys"), plain, op.get("entity_z_keys"), op.get("category_z_keys"), categories=self.categories)
                operation = {**op, "z_axis_keys": zkeys, "topic": topic, "attribute": attribute}
                line_id = apply_write_op(self.store, operation, topic=topic, attribute=attribute,
                                         value=op.get("value") or "", plain=plain, timestamp=now, alias=alias,
                                         scope=scope, role=job["role"], event_time=job["event_time"], source_step_ids=step_ids)
                if line_id:
                    if re.fullmatch(r"pending-\d+", op.get("operation_id") or ""):
                        alias[op["operation_id"]] = line_id
                    applied.append({**operation, "line_id": line_id})
                elif action == "NOOP":
                    applied.append(op)
            writes = [op for op in applied if op.get("line_id")]
            if writes:
                topic = next((op.get("topic") for op in reversed(writes) if op.get("topic")), "撤销")
                change = self._record_timeline(2, {"topic": topic, "operations": len(writes),
                                                  "is_conflict": any(op.get("is_conflict_or_override") for op in writes)},
                                               now, scope, job["event_time"])
        self._refresh_timeline(change)
        if not isinstance(self.llm, MockLLMClient) and not greeting and not fallback:
            facts = [{"text": line_body(op["formatted_line"]), "topic": op["topic"], "attribute": op["attribute"],
                      "value": op.get("value") or "", "zkeys": op["z_axis_keys"],
                      "isolated": op["action"] == "INSERT_ISOLATED_LINE"}
                     for op in applied if op["action"] not in {"NOOP", "DELETE_LINE"}]
        return {"facts": facts, "operations": applied, "queued": False}

    def delete(self, line_id, *, hard=False, user_id="default", agent_id="default", timestamp=None) -> bool:
        now = float(timestamp) if timestamp is not None else self._now()
        scope = (user_id, agent_id, None)
        change = None
        with self.store.transaction():
            line = self.store.get_line(line_id, scope=scope, include_deleted=hard)
            deleted = line is not None and self.store.delete_line(line_id, timestamp=now, hard=hard, scope=scope)
            if deleted:
                change = self._record_timeline(2, {"topic": "撤销", "operations": 1}, now, (*scope[:2], line["session_id"]))
        self._refresh_timeline(change)
        if deleted and hard and hasattr(self.vector, "forget"):
            self.vector.forget(line_id)
        return deleted

    def search(self, query: str, timestamp: float | None = None, *, flush_pending: bool = True, **kwargs) -> dict:
        return self.retrieve(query, timestamp=timestamp, flush_pending=flush_pending, **kwargs)

    def retrieve(self, query: str, *, user_id="default", agent_id="default", session_id=None,
                 top_k=10, token_budget=2048, timestamp=None, read_only=False, flush_pending=True) -> dict:
        started = time.perf_counter()
        now = float(timestamp) if timestamp is not None else self._now()
        scope = (user_id, agent_id, session_id)
        control_scope = (user_id, agent_id, "default" if session_id is None else session_id)
        origin, radius = self._control_state(control_scope)
        direct = []

        def result(route, parsed, *, outer=None, inner=None, scan=None, fallback=False, weights=None):
            outer, inner = outer or [], inner or []
            context = "\n".join(fact["text"] for fact in outer + inner)
            history_tokens = self.store.history_tokens(scope)
            return {"route": route, "hits": [], "outer": outer, "inner": inner, "origin": origin, "radius": radius,
                    "matched_entities": direct, "entity_fallback": fallback, "empty_retrieval": not (outer or inner),
                    "parsed_query": parsed, "scan_far_to_near": scan or [], "context": context,
                    "retrieval_ms": (time.perf_counter() - started) * 1000, "context_tokens": estimate_tokens(context),
                    "history_tokens": history_tokens, "poison_circle": weights,
                    "scope": {"user_id": user_id, "agent_id": agent_id, "session_id": session_id}}

        if (self.enable_casual_bypass and is_greeting(query) and not (_RECALL_CUE.search(query) or _QUESTION_CUE.search(query))
                and not self.store.greeting_keys(query, scope=scope)):
            return result(CASUAL, parse_query_rules(query, categories=self.categories))

        if self.pending and flush_pending and not read_only:
            self.drain(user_id=user_id, agent_id=agent_id)
        snapshot = self.store.snapshot(scope=scope)
        index = self.store.z_index(snapshot=snapshot)
        direct = match_entities(query, (key for key in index if key_type(key) == "entity"))

        if direct:
            hints = list(dict.fromkeys([*direct, *expand_entities(index, direct)]))[:15]
            preview = _bucket_hits(index, direct)[:10]
        else:
            preview = [fact for fact in self.search_vector(query, top_n=10, snapshot=snapshot)
                       if fact.get("vector_score", 0) > 0 and fact.get("topic") != NEWS_TOPIC]
            hints = list(dict.fromkeys(key for fact in preview for key in fact["entity_z_keys"]))[:15]
        notebook_index = [{"block_id": fact["block_id"], "topic": fact["topic"],
                           "preview": fact["plain"][:180], "preview_attr": fact["attribute"]} for fact in preview[:8]]

        def fallback_controller():
            parsed = parse_query_rules(query, categories=self.categories)
            decision = decide_origin(origin, notebook_index, hints, query)
            if parsed["query_type"] == "TIME_TRAVEL":
                decision["poison_circle"]["radius_r"] = 0.62
            return {**parsed, **decision}

        if isinstance(self.llm, MockLLMClient):
            decision = fallback_controller()
        else:
            prompt = render_prompt("controller", categories=self._allowed_categories, current_origin_json=origin or {},
                                   notebook_2d_index=notebook_index, matched_z_entities=hints, query_stem=json.dumps(query, ensure_ascii=False))
            decision, _ = self._call_json(prompt, fallback_controller, _valid_controller)
        parsed = normalize_parsed_query(decision, query, categories=self.categories)
        travel = parsed["query_type"] == "TIME_TRAVEL"
        if (self.enable_casual_bypass and not direct and not parsed["entities"] and not parsed["categories"]
                and parsed["query_type"] == "FACTUAL" and not _RECALL_CUE.search(query)
                and route_query(query, ()) == CASUAL):
            return result(CASUAL, parsed)

        circle = decision["poison_circle"]
        origin = {field: decision["new_origin"][field] for field in ("x_topic", "y_subtopic", "decision_anchor")}
        radius = max(self.epsilon, unit_interval(circle.get("radius_r"), 0.45))
        intuitive = unit_interval(circle.get("intuitive_weight"), 0.40)
        anti = unit_interval(circle.get("anti_intuitive_weight"), 0.90)
        seeds = _resolve_entities(index, [*direct, *parsed["entities"]])
        expanded = expand_entities(index, seeds, hops=2)
        expanded_set = set(expanded)
        hashed = self.search_z_hash(seeds, parsed["categories"], snapshot=snapshot, index=index,
                                    expanded=expanded)
        semantic = parsed["semantic_vector_query"]
        facts = snapshot
        if travel:
            seed_ids = {fact["id"] for fact in hashed}
            seed_ids.update(fact["id"] for fact in preview if fact.get("vector_score", 0) > MIN_REL)
            facts = [*snapshot, *self.store.historical_versions(seed_ids, scope=scope)]
        searches = [semantic, semantic_origin_text(origin)]
        if hasattr(self.vector, "search_many"):
            vec, radial = self.vector.search_many(searches, facts, len(facts))
        else:
            vec, radial = [self.vector.search(text, facts, top_n=len(facts)) for text in searches]
        sim = {fact["id"]: unit_interval(fact.get("vector_score")) for fact in vec}
        origin_sim = {fact["id"]: unit_interval(fact.get("vector_score")) for fact in radial}
        history = [fact for fact in facts if fact.get("historical")]
        if history:
            target = travel_revision(query)
            if target == "previous":
                latest = {fact["id"]: fact["revision"] for fact in snapshot}
                history = [fact for fact in history if fact["revision"] == latest[fact["line_id"]] - 1]
            elif isinstance(target, int):
                history = [fact for fact in history if fact["revision"] == target]
        eligible_history = {fact["id"] for fact in history}
        ranked_hash = sorted([*hashed, *history], key=lambda fact: -max(sim.get(fact["id"], 0), origin_sim.get(fact["id"], 0)))
        merged = weighted_rrf([fact for fact in vec if (not fact.get("historical") or fact["id"] in eligible_history)
                               and sim.get(fact["id"], 0) > 0][:30], ranked_hash)
        buckets = {}
        for fact in merged:
            keys = fact["entity_z_keys"]
            semantic_similarity = sim.get(fact["id"], 0.0)
            similarity = max(semantic_similarity, origin_sim.get(fact["id"], 0.0))
            tagged = {**fact, "raw_dist": 1.0 - similarity, "d_origin": 1.0 - similarity,
                      "vector_score": semantic_similarity, "entity_anchor": bool(expanded_set.intersection(keys))}
            for key in keys or fact["category_z_keys"][:1] or ["未分桶"]:
                buckets.setdefault(key, []).append(tagged)
        scan = order_far_to_near(circle.get("z_axis_scan_order_far_to_near", []), list(dict.fromkeys([*expanded, *buckets])))
        collapsed = collapse_and_scan_z_axis(buckets, scan, radius, self.epsilon)
        outer, inner = _allocate(collapsed, now, intuitive, anti, top_k, token_budget, travel)
        if not read_only:
            with self.store.transaction():
                self.store.set_meta(self._control_key(control_scope), json.dumps({"origin": origin, "radius": radius}, ensure_ascii=False))
                change = self._record_timeline(1, {"origin": origin, "topic": origin["x_topic"],
                                                  "origin_shifted": decision["origin_shifted"]}, now, control_scope)
            self._refresh_timeline(change)
            if control_scope == DEFAULT_SCOPE:
                self.origin, self.radius = origin, radius
        return result(TASK, parsed, outer=outer, inner=inner, scan=scan, fallback=not direct and bool(hints),
                      weights={"radius_r": radius, "intuitive_weight": intuitive, "anti_intuitive_weight": anti,
                               "z_axis_scan_order_far_to_near": scan})

    def search_z_hash(self, entities, categories, query_stem="", *, snapshot=None, index=None, expanded=None):
        if index is None:
            index = self.store.z_index(snapshot=snapshot)
        if expanded is None:
            seeds = _resolve_entities(index, entities or [])
            if query_stem:
                seeds += match_entities(query_stem, (key for key in index if key_type(key) == "entity"))
            expanded = expand_entities(index, seeds, hops=2)
        return _bucket_hits(index, expanded or [key for key in categories or [] if key_type(key) == "category"])

    def search_vector(self, query_text: str, top_n: int = 30, *, snapshot=None) -> list[dict]:
        return self.vector.search(query_text, self.store.snapshot() if snapshot is None else snapshot, top_n=top_n)

    def has_facts(self) -> bool:
        return bool(self.store.snapshot())

    def get_recent_facts(self, limit: int = 10, *, snapshot=None) -> list[dict]:
        facts = self.store.snapshot() if snapshot is None else snapshot
        return sorted(facts, key=lambda fact: (-fact["updated_at"], fact["id"]))[:max(0, limit)]

    def answer(self, query: str, timestamp: float | None = None, retrieved: dict | None = None, *,
               user_id="default", agent_id="default") -> dict:
        if retrieved is None:
            retrieved = self.retrieve(query, user_id=user_id, agent_id=agent_id, timestamp=timestamp, read_only=True)
        elif retrieved.get("scope") and (retrieved["scope"]["user_id"], retrieved["scope"]["agent_id"]) != (user_id, agent_id):
            raise ValueError("retrieved evidence belongs to a different user or agent")
        if retrieved["route"] == CASUAL:
            return {**retrieved, "response": "", "reasoning": None, "llm_fallback": False}
        outer, inner = retrieved["outer"], retrieved["inner"]
        query_type = retrieved["parsed_query"]["query_type"]
        memories = [{key: fact.get(key) for key in ("id", "text", "zone", "z_bucket", "revision", "historical",
                                                    "source_step_ids", "attribute", "entity_z_keys", "topic",
                                                    "event_time", "updated_at", "is_conflict_or_override", "entity_anchor")}
                    for fact in outer + inner]
        prompt = render_prompt("reason", query_type=query_type, retrieved_top_k_memories=memories,
                               user_question=json.dumps(query, ensure_ascii=False))
        data, used_fallback = self._call_json(prompt, lambda: plan_answer(retrieved["origin"], outer, inner, query, query_type), _valid_reason)
        evidence_ids = {fact["id"] for fact in memories}
        safe = {"evidence_synthesis": data["evidence_synthesis"], "anti_intuitive_override": data["anti_intuitive_override"],
                "rejected_noise_ids": [item for item in data["rejected_noise_ids"] if item in evidence_ids]}
        return {**retrieved, "response": data["final_answer"], "reasoning": safe, "llm_fallback": used_fallback}

    def timeline_stats(self) -> dict:
        return {"events": self.timeline.events, "nodes": self.timeline.nodes, "layer1": len(self.timeline.layer1),
                "layer2": len(self.timeline.layer2), "reduction": self.timeline.reduction()}

    def render(self) -> str:
        return "\n\n".join(f"# {block['topic']}\n{render_block(block['lines'])}" for block in self.store.blocks_for_prompt())

    def _call_json(self, prompt, fallback, validator):
        try:
            data = loads_json_object(self.llm.complete(prompt))
            if validator(data):
                return data, False
        except FatalLLMError:
            raise
        except Exception:
            pass
        self.fallback_count += 1
        return fallback(), True


def _valid_operations(data):
    operations = data.get("write_operations")
    string_fields = ("target_block_id", "target_line_id", "formatted_line", "topic", "attribute", "value", "operation_id")
    list_fields = ("z_axis_keys", "entity_z_keys", "category_z_keys")
    return isinstance(operations, list) and all(
        isinstance(op, dict) and isinstance(op.get("action"), str) and op["action"] in ACTIONS
        and all(isinstance(op.get(field, ""), str) for field in string_fields)
        and all(_string_list(op.get(field, [])) for field in list_fields)
        and isinstance(op.get("is_conflict_or_override", False), bool)
        and (op["action"] in {"DELETE_LINE", "NOOP"} or bool(line_body(op.get("formatted_line", ""))))
        for op in operations)


def _string_list(value):
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _valid_controller(data):
    origin, circle = data.get("new_origin"), data.get("poison_circle")
    return (isinstance(origin, dict) and isinstance(circle, dict) and isinstance(data.get("origin_shifted"), bool)
            and all(isinstance(origin.get(key), str) and origin[key].strip() for key in ("x_topic", "y_subtopic", "decision_anchor"))
            and isinstance(data.get("query_type"), str) and data["query_type"] in {"FACTUAL", "REFLECTIVE", "TIME_TRAVEL"}
            and _string_list(data.get("entities")) and _string_list(data.get("categories"))
            and isinstance(data.get("semantic_vector_query"), str)
            and isinstance(circle.get("radius_r"), (int, float))
            and _string_list(circle.get("z_axis_scan_order_far_to_near", [])))


def _valid_reason(data):
    return (isinstance(data.get("final_answer"), str) and isinstance(data.get("evidence_synthesis"), str)
            and isinstance(data.get("anti_intuitive_override"), bool) and _string_list(data.get("rejected_noise_ids")))


def _resolve_entities(index, seeds):
    names = {entity_names(key).casefold() for key in seeds if key_type(key) == "entity"}
    return [key for key in index if key_type(key) == "entity" and entity_names(key).casefold() in names]


def _bucket_hits(index, keys):
    return list({fact["id"]: fact for key in keys for fact in index.get(key, [])}.values())


def _allocate(collapsed, now, intuitive, anti, top_k, token_budget, travel):
    band = {OUTER: [], INNER: []}
    for bucket in collapsed:
        for fact in bucket["facts"]:
            if fact["zone"] in band:
                band[fact["zone"]].append({**fact, "z_bucket": bucket["z_bucket"],
                                          "score": score_fact(fact, now, intuitive, anti)})
    for facts in band.values():
        facts.sort(key=lambda fact: -fact["score"])
    relaxed = travel or any(fact.get("is_conflict_or_override") for fact in band[OUTER])
    outer_cap = math.ceil(max(0, top_k) * (0.6 if relaxed else 0.4))
    reserve = ""
    if top_k > 1 and band[OUTER] and band[INNER]:
        smallest = min((fact["text"] for fact in band[OUTER]), key=lambda text: (estimate_tokens(text), len(text)))
        reserve = next((fact["text"] for fact in band[INNER]
                        if estimate_tokens(f"{smallest}\n{fact['text']}") <= token_budget), "")
    outer, inner = [], []
    context = ""
    for zone, chosen, limit in ((OUTER, outer, outer_cap), (INNER, inner, top_k)):
        for fact in band[zone]:
            if len(chosen) >= limit or len(outer) + len(inner) >= top_k:
                break
            candidate = f"{context}\n{fact['text']}" if context else fact["text"]
            with_reserve = f"{candidate}\n{reserve}" if zone == OUTER and reserve else candidate
            if estimate_tokens(with_reserve) <= token_budget:
                chosen.append(fact)
                context = candidate
    return outer, inner
