"""Scoped 2D notebook, poison-circle retrieval and read-only System 2."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from html import escape
from typing import Callable

from memengine.categories import GENERAL_CATEGORY
from memengine.distance import (BELT, EPSILON, INNER, NEWS_KEY, NEWS_TOPIC, OUTER,
                                consolidation_count, inner_boundary, is_noise, score_bipolar, unit_interval)
from memengine.embeddings import OpenAICompatibleEmbedder
from memengine.extract import extract_facts, is_greeting
from memengine.llm import FatalLLMError, MockLLMClient, loads_json_object
from memengine.notebook import ACTIONS, apply_write_op, format_fact_line, line_body, render_block
from memengine.policy import decide_layout, decide_origin, plan_answer
from memengine.prompts import render_prompt
from memengine.queryparse import parse_query_rules
from memengine.router import CASUAL, TASK, match_entities, route_query
from memengine.store import DEFAULT_SCOPE, Store, key_type
from memengine.temporal import absolute_event_text, parse_temporal_intent
from memengine.textutil import char_bigrams, estimate_tokens, jaccard, normalize_entity, step_ids_in
from memengine.timeline import TimelineBook
from memengine.vector import HashedTfidfEmbedder, weighted_rrf
from memengine.zaxis import collapse_and_scan_z_axis, order_far_to_near

_RECALL_CUE = re.compile(r"记得|回忆|偏好|爱看|喜欢|\b(?:remember|recall|prefer|like)\b", re.I)
_QUESTION_CUE = re.compile(r"[?？]|啥|什么|哪|\b(?:what|which|who|when|where|why|how)\b", re.I)
_DELETE_CUE = re.compile(r"忘掉|忘记|删除|撤销|取消|不再|已经不|\b(?:forget|remove|delete|cancel|no longer|not anymore)\b", re.I)


class CognitiveMemoryEngine:
    def __init__(self, db_path: str = ":memory:", llm=None, delta_t_buffer: float = 10.0,
                 epsilon: float = EPSILON, enable_timeline_buffer: bool = True,
                 enable_casual_bypass: bool = True, clock: Callable[[], float] | None = None,
                 vector=None, *, categories: list[str] | None = None, bipolar_lambda: float = 0.50,
                 belt_prune_ratio: float = 0.35, min_keep: int = 2):
        self.epsilon = float(epsilon)
        if not 0 < self.epsilon <= 1:
            raise ValueError("epsilon must be in (0, 1]")
        self.bipolar_lambda = float(bipolar_lambda)
        if not 0 <= self.bipolar_lambda <= 1:
            raise ValueError("bipolar_lambda must be in [0, 1]")
        self.belt_prune_ratio = float(belt_prune_ratio)
        self.min_keep = min_keep
        _validate_belt_config(self.belt_prune_ratio, min_keep)
        self.llm = llm if llm is not None else MockLLMClient()
        self.vector = vector if vector is not None else OpenAICompatibleEmbedder.from_env() or HashedTfidfEmbedder()
        self.store = Store(db_path, embedder=self.vector)
        self.enable_casual_bypass = bool(enable_casual_bypass)
        self.categories = tuple(dict.fromkeys(name.strip() for name in (categories or []) if name.strip()))
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
            return data["origin"], inner_boundary(unit_interval(data["radius"], 0.25))
        if scope == DEFAULT_SCOPE:
            legacy = self.store.get_meta("origin")
            radius = unit_interval(self.store.get_meta("radius"), 0.25)
            return (json.loads(legacy) if legacy else None), inner_boundary(radius)
        return None, 0.25

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

    def add(self, text: str, timestamp: float | None = None, sync: bool = True, *, user_id="default",
            agent_id="default", session_id="default", role="user", event_time=None) -> dict:
        now = float(timestamp) if timestamp is not None else self._now()
        scope = (user_id, agent_id, session_id)
        turn_id = self.store.append_turn(text or "", now, scope=scope, role=role, event_ts=event_time)
        facts = extract_facts(text or "") if isinstance(self.llm, MockLLMClient) else []
        job = {"text": text or "", "ts": now, "scope": scope, "role": role, "facts": facts,
               "turn_id": turn_id, "event_time": float(event_time) if event_time is not None else None}
        if not sync:
            self.pending.append(job)
            return {"facts": facts, "operations": [], "queued": True, "turn_id": turn_id}
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
        explicit = {normalize_entity(key) for key in match_entities(text, graph_keys)}
        explicit.update(normalize_entity(key) for fact in facts for key in fact.get("zkeys", []) if key_type(key) == "entity")
        topics = {fact.get("topic") for fact in facts}
        query_grams = char_bigrams(text)
        ranked = sorted(blocks, key=lambda block: (block["topic"] in topics, jaccard(
            query_grams, char_bigrams(block["topic"] + " " + " ".join(line["plain"] for line in block["lines"][-3:])))), reverse=True)
        candidates = []
        for rank, block in enumerate(ranked):
            lines = block["lines"]
            pinned = {line["id"] for line in lines if explicit.intersection(map(normalize_entity, line["entity_z_keys"]))}
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
            context = self.store.get_turn_window([job["turn_id"]], window=3, user_id=scope[0], agent_id=scope[1])
            prompt = render_prompt("layout", step_id=str(step_ids[0]) if step_ids else "",
                                   candidate_2d_blocks=candidates, raw_turn_text=escape(text), role=escape(job["role"]),
                                   timestamp=now, categories=list(self.categories),
                                   current_date_iso=datetime.fromtimestamp(job["event_time"] if job["event_time"] is not None else now, timezone.utc).date().isoformat(),
                                   recent_turns=[{"role": row["role"], "text": escape(row["text"])} for row in context
                                                 if row["turn_id"] < job["turn_id"]])
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
                if action == "NOOP":
                    if target in allowed_lines or target in alias:
                        self.store.merge_line_mentions(alias.get(target, target), step_ids, scope=scope, turn_id=job["turn_id"])
                    applied.append(op)
                    continue
                plain = line_body(op.get("formatted_line") or "")
                target_block, target_line = allowed_lines.get(target, ({}, {}))
                topic = target_block.get("topic") or allowed_blocks.get(block, {}).get("topic") or op.get("topic") or GENERAL_CATEGORY
                attribute = target_line.get("attribute") or op.get("attribute") or ""
                if action == "OVERWRITE_SAME_LINE" and target_line.get("attribute") and op.get("attribute") not in (None, "", attribute):
                    continue
                zkeys = [NEWS_KEY] if topic == NEWS_TOPIC else _write_keys(op)
                reference_time = job["event_time"] if job["event_time"] is not None else now
                rewritten, inferred_time = absolute_event_text(plain, reference_time)
                effective_time = _iso_timestamp(op.get("event_time_iso"))
                if effective_time is None:
                    effective_time = inferred_time if inferred_time is not None else reference_time
                operation = {**op, "z_axis_keys": zkeys, "topic": topic, "attribute": attribute}
                if rewritten != plain:
                    operation["formatted_line"] = format_fact_line(now, rewritten)
                line_id = apply_write_op(self.store, operation, topic=topic, attribute=attribute,
                                         value=op.get("value") or "", plain=plain, timestamp=now, alias=alias,
                                         scope=scope, role=job["role"], event_time=effective_time, source_step_ids=step_ids,
                                         source_turn_id=job["turn_id"])
                if line_id:
                    if action == "OVERWRITE_SAME_LINE":
                        alias[target] = line_id
                    if re.fullmatch(r"pending-\d+", op.get("operation_id") or ""):
                        alias[op["operation_id"]] = line_id
                    applied.append({**operation, "line_id": line_id})
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
        return {"facts": facts, "operations": applied, "queued": False, "turn_id": job["turn_id"]}

    def delete(self, line_id, *, hard=False, user_id="default", agent_id="default", timestamp=None) -> bool:
        now = float(timestamp) if timestamp is not None else self._now()
        scope = (user_id, agent_id, None)
        change = None
        with self.store.transaction():
            line = self.store.get_line(line_id, scope=scope, include_deleted=hard)
            deleted = line is not None and self.store.delete_fact(line_id, ingest_time=now, hard=hard, scope=scope)
            if deleted:
                change = self._record_timeline(2, {"topic": "撤销", "operations": 1}, now, (*scope[:2], line["session_id"]))
        self._refresh_timeline(change)
        if deleted and hard and hasattr(self.vector, "forget"):
            self.vector.forget(line_id)
        return deleted

    def search(self, query: str, timestamp: float | None = None, *, flush_pending: bool = True, **kwargs) -> dict:
        return self.retrieve(query, timestamp=timestamp, flush_pending=flush_pending, **kwargs)

    def retrieve(self, query: str, *, user_id="default", agent_id="default", session_id=None,
                 top_k=10, token_budget=2048, timestamp=None, read_only=False, flush_pending=True,
                 belt_prune_ratio: float | None = None, min_keep: int | None = None,
                 include_raw_turns: bool = False) -> dict:
        prune_ratio = self.belt_prune_ratio if belt_prune_ratio is None else float(belt_prune_ratio)
        keep_floor = self.min_keep if min_keep is None else min_keep
        _validate_belt_config(prune_ratio, keep_floor)
        started = time.perf_counter()
        now = float(timestamp) if timestamp is not None else self._now()
        scope = (user_id, agent_id, session_id)
        control_scope = (user_id, agent_id, "default" if session_id is None else session_id)
        origin, radius = self._control_state(control_scope)
        direct = []
        temporal = parse_temporal_intent(query, now)

        def result(route, parsed, *, ranked=(), belt_total=0, belt_pruned=0, scan=None, fallback=False, weights=None):
            outer = [fact for fact in ranked if fact["zone"] == OUTER]
            inner = [fact for fact in ranked if fact["zone"] == INNER]
            belt = [fact for fact in ranked if fact["zone"] == BELT]
            context = "\n".join(fact["text"] for fact in ranked)
            history_tokens = self.store.history_tokens(scope)
            drilldown = self.store.get_turn_window(
                [turn_id for fact in ranked[:3] for turn_id in fact.get("source_turn_ids", [])],
                user_id=user_id, agent_id=agent_id) if include_raw_turns else []
            return {"route": route, "hits": [], "ranked": list(ranked), "outer": outer, "inner": inner,
                    "belt": belt, "origin": origin, "radius": radius,
                    "drilldown_turns": drilldown, "temporal_intent": vars(temporal),
                    "outer_triggered_count": len(outer), "inner_count": len(inner), "belt_count": len(belt),
                    "belt_compressed_count": belt_total - len(belt),
                    "belt_pruned_count": belt_pruned,
                    "matched_entities": direct, "entity_fallback": fallback, "empty_retrieval": not ranked,
                    "parsed_query": parsed, "scan_far_to_near": scan or [], "context": context,
                    "retrieval_ms": (time.perf_counter() - started) * 1000, "context_tokens": estimate_tokens(context),
                    "history_tokens": history_tokens, "poison_circle": weights,
                    "scope": {"user_id": user_id, "agent_id": agent_id, "session_id": session_id}}

        if (self.enable_casual_bypass and is_greeting(query) and not (_RECALL_CUE.search(query) or _QUESTION_CUE.search(query))
                and not match_entities(query, self.store.entity_keys(scope=scope))):
            return result(CASUAL, parse_query_rules(query, categories=self.categories))

        if self.pending and flush_pending and not read_only:
            self.drain(user_id=user_id, agent_id=agent_id)
        parsed = parse_query_rules(query, categories=self.categories)
        if temporal.is_historical:
            parsed["query_type"] = "TIME_TRAVEL"
        with self.store.read_snapshot():
            keyed = self.store.entity_keys(scope=scope)
            direct = match_entities(query, keyed)
            # Category words such as 城市 are not task intent. The rule router still
            # requires an entity hit or a task cue before leaving the casual bypass.
            if (self.enable_casual_bypass and not direct and not parsed["entities"]
                    and parsed["query_type"] == "FACTUAL" and not _RECALL_CUE.search(query)
                    and route_query(query, ()) == CASUAL):
                return result(CASUAL, parsed)
            merged = self._recall(query, parsed, temporal, scope, direct)
        hints = list(dict.fromkeys(key for fact in merged[:10] for key in fact["entity_z_keys"]))[:15]
        preview = [{"block_id": fact["block_id"], "topic": fact["topic"],
                    "preview": fact["plain"][:180], "preview_attr": fact["attribute"]} for fact in merged[:8]]
        decision = decide_origin(origin, preview, direct or hints, query)
        circle = decision["poison_circle"]
        origin = decision["new_origin"]
        radius = 0.35 if temporal.is_historical else inner_boundary(circle["radius_r"])
        c_max = max((consolidation_count(fact) for fact in merged if not is_noise(fact)), default=1)
        rrf_max = max((fact["rrf_score"] for fact in merged), default=0.0)
        rrf_order = {fact["id"]: rank for rank, fact in enumerate(merged)}
        buckets = {}
        for fact in merged:
            keys = fact["entity_z_keys"]
            tagged = score_bipolar(fact, c_max=c_max, rrf_max=rrf_max,
                                    radius=radius, bipolar_lambda=self.bipolar_lambda, epsilon=self.epsilon)
            for key in keys or fact["category_z_keys"][:1] or ["未分桶"]:
                buckets.setdefault(key, []).append(tagged)
        scan = order_far_to_near(circle.get("z_axis_scan_order_far_to_near", []), list(dict.fromkeys([*direct, *buckets])))
        collapsed = collapse_and_scan_z_axis(buckets, scan)
        ranked = [{**fact, "z_bucket": bucket["z_bucket"]} for bucket in collapsed for fact in bucket["facts"]]
        ranked.sort(key=lambda fact: (-fact["bipolar_score"], rrf_order[fact["id"]]))
        pruned = _prune_belt(ranked, prune_ratio, keep_floor)
        selected = _fit_context(pruned, top_k, token_budget)
        if not read_only:
            with self.store.transaction():
                self.store.set_meta(self._control_key(control_scope), json.dumps({"origin": origin, "radius": radius}, ensure_ascii=False))
                line_ids = [fact["id"] for fact in selected]
                if line_ids:
                    self.store.increment_hit_counts(line_ids)
            if control_scope == DEFAULT_SCOPE:
                self.origin, self.radius = origin, radius
        return result(TASK, parsed, ranked=selected, belt_total=sum(fact["zone"] == BELT for fact in ranked),
                      belt_pruned=len(ranked) - len(pruned),
                      scan=scan, fallback=not direct and bool(hints),
                      weights={"radius_r": radius, "radius_out": 1.0 - radius, "bipolar_lambda": self.bipolar_lambda,
                               "belt_prune_ratio": prune_ratio, "min_keep": keep_floor,
                               "z_axis_scan_order_far_to_near": scan})

    def _recall(self, query, parsed, temporal, scope, direct):
        user_id, agent_id, session_id = scope
        options = dict(user_id=user_id, agent_id=agent_id, session_id=session_id, ids_only=True)
        dense = self.search_vector(parsed["semantic_vector_query"], top_n=40, **options)
        lexical = self.store.search_bm25(query, **options)
        entities = direct or parsed["entities"]
        inverted = self.store.search_entity_inverted(entities, **options)
        timed = []
        previous = (temporal.is_historical and temporal.order_hint == "latest"
                    and temporal.start_ts is None and temporal.end_ts is None)
        if temporal.is_historical or temporal.start_ts is not None or temporal.end_ts is not None:
            timed = self.store.search_temporal(start_ts=temporal.start_ts, end_ts=temporal.end_ts,
                                               order_hint=temporal.order_hint, rev_index=temporal.rev_index,
                                               entities={normalize_entity(key) for key in direct}, previous_only=previous, **options)
        channels = (dense, lexical, inverted, timed)
        ids = {hit["id"] for channel in channels for hit in channel}
        details = {fact["id"]: fact for fact in self.store.facts_by_ids(ids, scope=scope)}
        time_ids = {hit["id"] for hit in timed}
        inverted_ids = {hit["id"] for hit in inverted}
        dense_scores = {hit["id"]: hit["vector_score"] for hit in dense}
        entity_names = {normalize_entity(key) for key in direct}
        for fact in details.values():
            anchor = bool(entity_names.intersection(normalize_entity(key) for key in fact["entity_z_keys"]))
            fact.update(vector_score=dense_scores.get(fact["id"], 0.0), entity_anchor=anchor or fact["id"] in inverted_ids,
                        temporal_match=fact["id"] in time_ids)
            if fact["historical"] and not temporal.is_historical:
                fact["rrf_factor"] = 0.35
                fact["text"] = "[历史已变更] " + fact["text"]
            elif time_ids and (previous or temporal.start_ts is not None or temporal.end_ts is not None or temporal.rev_index is not None):
                if not fact["temporal_match"]:
                    fact["rrf_factor"] = 0.35

        def hydrate(channel):
            return [{**details[hit["id"]], **hit} for hit in channel]

        return weighted_rrf(hydrate(dense), hydrate(inverted), w_vec=0.40, w_hash=0.15,
                            extra_channels=((0.35, hydrate(lexical)), (0.25, hydrate(timed))))

    def search_z_hash(self, entities, query_stem="", *, user_id="default", agent_id="default", session_id="default"):
        scope = (user_id, agent_id, session_id)
        seeds = list(entities or [])
        if query_stem:
            seeds.extend(match_entities(query_stem, self.store.entity_keys(scope=scope)))
        return self.store.search_entity_inverted(seeds, include_superseded=False, user_id=user_id,
                                                 agent_id=agent_id, session_id=session_id)

    def search_vector(self, query_text: str, top_n: int = 40, *, snapshot=None, user_id="default", agent_id="default",
                      session_id=None, ids_only=False) -> list[dict]:
        if hasattr(self.vector, "encode"):
            return self.store.search_vector(query_text, top_n=top_n, ids_only=ids_only, user_id=user_id,
                                            agent_id=agent_id, session_id=session_id)
        facts = self.store.snapshot(scope=(user_id, agent_id, session_id), include_superseded=True) if snapshot is None else snapshot
        hits = [fact for fact in self.vector.search(query_text, facts, top_n=top_n) if fact.get("vector_score", 0) > 0]
        return [{"id": fact["id"], "vector_score": fact["vector_score"]} for fact in hits] if ids_only else hits

    def has_facts(self, *, user_id="default", agent_id="default", session_id="default") -> bool:
        return bool(self.store.snapshot(scope=(user_id, agent_id, session_id)))

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
        ranked = retrieved.get("ranked")
        if ranked is None:
            ranked = [fact for key in ("hits", "outer", "inner", "belt") for fact in retrieved.get(key) or []]
        query_type = retrieved["parsed_query"]["query_type"]
        memories = [{key: fact.get(key) for key in ("id", "text", "zone", "z_bucket", "revision", "historical",
                                                    "source_step_ids", "attribute", "entity_z_keys", "topic",
                                                    "event_time", "updated_at", "is_conflict_or_override", "entity_anchor",
                                                    "isolated", "vector_score", "zkeys", "temporal_match")}
                    for fact in ranked]
        entities = retrieved.get("matched_entities", [])
        prompt = render_prompt("reason", query_type=query_type, retrieved_top_k_memories=memories, matched_entities=entities,
                               user_question=json.dumps(query, ensure_ascii=False))
        data, used_fallback = self._call_json(prompt, lambda: plan_answer(ranked, query, query_type, entities), _valid_reason)
        evidence_ids = {fact["id"] for fact in memories}
        safe = {"evidence_synthesis": data["evidence_synthesis"], "anti_intuitive_override": data["anti_intuitive_override"],
                "rejected_noise_ids": [item for item in data["rejected_noise_ids"] if item in evidence_ids]}
        return {**retrieved, "response": data["final_answer"], "reasoning": safe, "llm_fallback": used_fallback}

    def timeline_stats(self, *, user_id="default", agent_id="default", session_id="default") -> dict:
        scope = (user_id, agent_id, session_id)
        layer1 = self.store.load_timeline(1, scope=scope)
        layer2 = self.store.load_timeline(2, scope=scope)
        events = sum(node.get("event_count", 1) for node in (*layer1, *layer2))
        nodes = len(layer1) + len(layer2)
        return {"events": events, "nodes": nodes, "layer1": len(layer1), "layer2": len(layer2),
                "reduction": None if events <= 0 else 1.0 - nodes / events}

    def render(self, *, user_id="default", agent_id="default", session_id="default") -> str:
        blocks = self.store.blocks_for_prompt(scope=(user_id, agent_id, session_id))
        return "\n\n".join(f"# {block['topic']}\n{render_block(block['lines'])}" for block in blocks)

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
    string_fields = ("target_block_id", "target_line_id", "formatted_line", "topic", "attribute", "value", "operation_id", "event_time_iso")
    list_fields = ("z_axis_keys", "entity_z_keys", "category_z_keys", "tags")
    return isinstance(operations, list) and all(
        isinstance(op, dict) and isinstance(op.get("action"), str) and op["action"] in ACTIONS
        and all(isinstance(op.get(field, ""), str) for field in string_fields)
        and all(_string_list(op.get(field, [])) for field in list_fields)
        and isinstance(op.get("is_conflict_or_override", False), bool)
        and (op["action"] in {"DELETE_LINE", "NOOP"} or bool(line_body(op.get("formatted_line", ""))))
        for op in operations)


def _string_list(value):
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _valid_reason(data):
    return (isinstance(data.get("final_answer"), str) and isinstance(data.get("evidence_synthesis"), str)
            and isinstance(data.get("anti_intuitive_override"), bool) and _string_list(data.get("rejected_noise_ids")))


def _write_keys(op):
    keys = [*(op.get("z_axis_keys") or ()), *(op.get("entity_z_keys") or ())]
    tags = [*(op.get("category_z_keys") or ()), *(op.get("tags") or ())]
    keys.extend(tag if key_type(tag) == "tag" else f"类目:{tag}" for tag in tags if tag.strip())
    return list(dict.fromkeys(key.strip() for key in keys if key.strip()))


def _iso_timestamp(value):
    if not value:
        return None
    try:
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (date if date.tzinfo else date.replace(tzinfo=timezone.utc)).timestamp()


def _validate_belt_config(ratio, min_keep):
    if not 0 <= ratio <= 1:
        raise ValueError("belt_prune_ratio must be in [0, 1]")
    if type(min_keep) is not int or min_keep < 1:
        raise ValueError("min_keep must be a positive integer")


def _prune_belt(ranked, ratio, min_keep):
    if not ranked:
        return []
    threshold = ratio * ranked[0]["bipolar_score"]
    kept = {fact["id"] for fact in ranked if fact["zone"] != BELT or fact["bipolar_score"] >= threshold}
    if len(kept) < min_keep:
        for fact in ranked:
            kept.add(fact["id"])
            if len(kept) >= min_keep:
                break
    return [fact for fact in ranked if fact["id"] in kept]


def _fit_context(ranked, top_k, token_budget):
    selected = []
    context = ""
    for fact in ranked:
        if len(selected) >= top_k:
            break
        candidate = f"{context}\n{fact['text']}" if context else fact["text"]
        if estimate_tokens(candidate) <= token_budget:
            selected.append(fact)
            context = candidate
    if not selected and ranked and top_k > 0 and token_budget > 0:
        first = ranked[0]
        low, high = 1, len(first["text"])
        while low < high:
            midpoint = (low + high + 1) // 2
            if estimate_tokens(first["text"][:midpoint]) <= token_budget:
                low = midpoint
            else:
                high = midpoint - 1
        selected.append({**first, "text": first["text"][:low], "context_truncated": True})
    return selected
