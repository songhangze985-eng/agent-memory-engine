"""Scoped 2D notebook, poison-circle retrieval and read-only System 2."""

from __future__ import annotations

import json
import re
import time
from dataclasses import replace
from datetime import datetime, timezone
from html import escape
from http.client import HTTPException
from typing import Callable, Literal

from memengine.assembler import _excerpt, assemble_context, render_prefix
from memengine.chrono import date_iso, parse_session_clock, resolve_temporal_spans
from memengine.categories import GENERAL_CATEGORY
from memengine.distance import (BELT, EPSILON, INNER, NEWS_KEY, NEWS_TOPIC, OUTER,
                                consolidation_count, inner_boundary, is_noise, score_bipolar, unit_interval)
from memengine.embeddings import OpenAICompatibleEmbedder
from memengine.extract import extract_facts, is_greeting
from memengine.gate import precision_gate
from memengine.identifiers import compares_versions, eligible_identifiers, extract_identifiers
from memengine.lifecycle import changes_state, content_hash, decay_factor, matching_attributes
from memengine.llm import FatalLLMError, MockLLMClient, loads_json_object
from memengine.notebook import ACTIONS, apply_write_op, format_fact_line, line_body, render_block
from memengine.policy import REFUSAL, decide_layout, decide_origin, plan_answer
from memengine.prompts import render_prompt
from memengine.queryparse import parse_query_rules
from memengine.reranker import (
    CalibratedLocalReranker, OpenAICompatibleReranker, Reranker, anchored_relevance, checked_scores,
    evidence_pair_scores,
)
from memengine.router import CASUAL, TASK, match_entities, route_query
from memengine.storage.base import DEFAULT_SCOPE, RecallFilter, SCOPES
from memengine.storage.factory import open_store
from memengine.retrieval.activation import RecallPlan, calendar_relevance, diversify_episodes, query_overlap
from memengine.retrieval.operators import recall
from memengine.versioning import extract_versions
from memengine.retrieval.reconstruct import bridge_keys, merge_passes, select_bridged
from memengine.foresight import interval
from memengine.store import key_type
from memengine.temporal import absolute_event_text, competition_terms, parse_temporal_intent, temporal_competition
from memengine.textutil import char_bigrams, estimate_tokens, jaccard, normalize_entity, step_ids_in, strip_step_tags, token_units, tokenize_for_fts
from memengine.timeline import TimelineBook
from memengine.vector import HashedTfidfEmbedder, weighted_rrf
from memengine.verification import admissible_evidence, assess_claims, evidence_matches_query, fact_is_current
from memengine.working_memory import TaskWorkspace
from memengine.zaxis import collapse_and_scan_z_axis, order_far_to_near

_RECALL_CUE = re.compile(r"记得|回忆|偏好|爱看|喜欢|\b(?:remember|recall|prefer|like)\b", re.I)
_QUESTION_CUE = re.compile(r"[?？]|啥|什么|哪|\b(?:what|which|who|when|where|why|how)\b", re.I)
_DELETE_CUE = re.compile(r"忘掉|忘记|删除|撤销|取消|不再|已经不|\b(?:forget|remove|delete|cancel|no longer|not anymore)\b", re.I)


class CognitiveMemoryEngine(TaskWorkspace):
    def __init__(self, db_path: str = ":memory:", llm=None, delta_t_buffer: float = 10.0,
                 epsilon: float = EPSILON, enable_timeline_buffer: bool = True,
                 enable_casual_bypass: bool = True, clock: Callable[[], float] | None = None,
                 vector=None, *, categories: list[str] | None = None, bipolar_lambda: float = 0.55,
                 belt_prune_ratio: float = 0.35, min_keep: int = 2, per_channel_k: int = 35,
                 reranker: Reranker | None = None, tau_abs: float | None = None, cliff_ratio: float = 0.45,
                 allow_empty_recall: bool = True, superseded_weight: float = 0.35, working_window: int = 6, db_url=None):
        self.epsilon = float(epsilon)
        if not 0 < self.epsilon <= 1:
            raise ValueError("epsilon must be in (0, 1]")
        self.bipolar_lambda = float(bipolar_lambda)
        if not 0 <= self.bipolar_lambda <= 1:
            raise ValueError("bipolar_lambda must be in [0, 1]")
        self.belt_prune_ratio = float(belt_prune_ratio)
        self.min_keep = min_keep
        _validate_belt_config(self.belt_prune_ratio, min_keep)
        if type(per_channel_k) is not int or per_channel_k < 1:
            raise ValueError("per_channel_k must be a positive integer")
        for name, value in (("tau_abs", tau_abs), ("cliff_ratio", cliff_ratio), ("superseded_weight", superseded_weight)):
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        self.per_channel_k = per_channel_k
        self.local_reranker = CalibratedLocalReranker()
        self.reranker = reranker if reranker is not None else OpenAICompatibleReranker.from_env() or self.local_reranker
        self.tau_abs = tau_abs
        self.cliff_ratio = cliff_ratio
        self.allow_empty_recall = allow_empty_recall
        self.superseded_weight = superseded_weight
        self.llm = llm if llm is not None else MockLLMClient()
        self.vector = vector if vector is not None else OpenAICompatibleEmbedder.from_env() or HashedTfidfEmbedder()
        self.store = open_store(db_url, path=db_path, embedder=self.vector)
        self.enable_casual_bypass = bool(enable_casual_bypass)
        self.categories = tuple(dict.fromkeys(name.strip() for name in (categories or []) if name.strip()))
        self.timeline = TimelineBook(delta_t_buffer, enabled=enable_timeline_buffer)
        self.timeline.layer1 = self.store.load_timeline(1)
        self.timeline.layer2 = self.store.load_timeline(2)
        self.timeline.events = int(self.store.get_meta("timeline_events") or 0)
        self.origin, self.radius = self._control_state(DEFAULT_SCOPE)
        self.pending: list[dict] = []
        if type(working_window) is not int or working_window < 1:
            raise ValueError("working_window must be a positive integer")
        self.working_window = working_window
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

    def _control_state(self, scope, metadata=None):
        get = self.store.get_meta if metadata is None else metadata.get
        raw = get(self._control_key(scope))
        if raw is not None:
            data = json.loads(raw)
            return data["origin"], inner_boundary(unit_interval(data["radius"], 0.25))
        if scope == DEFAULT_SCOPE:
            legacy = get("origin")
            radius = unit_interval(get("radius"), 0.25)
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

    def retain(self, text=None, *, buffered_turn_id=None, **options):
        if buffered_turn_id is None:
            return self.add(text, **options)
        return self._retain_buffered(buffered_turn_id, text, **options)

    def _retain_buffered(self, turn_id, text, *, user_id="default", agent_id="default", session_id="default"):
        if type(turn_id) is not int or turn_id < 1 or not session_id:
            raise ValueError("a buffered turn requires a positive ID and session")
        scope = (user_id, agent_id, session_id)
        with self.store.transaction():
            rows = self.store.get_turn_window([turn_id], 0, user_id, agent_id, session_id=session_id)
            if not rows or (text is not None and rows[0]["text"] != text):
                raise ValueError("buffered turn not found in this session")
            source = rows[0]
            if source["extraction_state"] == "done":
                return {"facts": [], "operations": [], "queued": False, "turn_id": turn_id}
            if source["extraction_state"] != "buffered":
                raise ValueError("the turn is not buffered")
            text = source["text"]
            mention = source["mention_time"]
            details = {"mention_time": source["ingest_ts"] if mention is None else mention,
                       "speaker": source["speaker"], "origin_turn": turn_id, "raw": text}
            return self._apply_job({"scope": scope, "turn_id": turn_id, "text": text, "role": source["role"],
                                   "ts": source["ingest_ts"], "event_time": source["event_ts"],
                                   "facts": extract_facts(text) if isinstance(self.llm, MockLLMClient) else [],
                                   "chronicle": details if mention is not None or source["speaker"] else None})

    def recall(self, query, *, mutate_stats=False, **options):
        options.setdefault("version_precision", True)
        options.setdefault("read_only", not mutate_stats)
        if options.get("token_budget") is not None and options.get("top_k") is None and options.get("packing_mode") is None:
            options["packing_mode"] = "token_budget"
        return self.retrieve(query, mutate_stats=mutate_stats, **options)

    def reflect(self, query, *, user_id="default", agent_id="default", session_id=None, task_id=None,
                timestamp=None, reference_time=None, token_budget=2048):
        from memengine.reflection import checked_reflection

        if not isinstance(query, str) or not query.strip():
            raise ValueError("reflect requires a nonempty query")
        return checked_reflection(self, query, scope=(user_id, agent_id, session_id), task_id=task_id,
                                  reference_time=reference_time if reference_time is not None else timestamp,
                                  token_budget=token_budget)

    def maintain(self, *, user_id="default", agent_id="default", min_new_facts=8, force=False, archive=True, timestamp=None):
        owner = dict(user_id=user_id, agent_id=agent_id, timestamp=timestamp)
        return {"consolidation": self.consolidate(**owner, min_new_facts=min_new_facts, force=force),
                **(self.decay_and_archive(**owner) if archive else {})}

    def add(self, text: str, timestamp: float | None = None, sync: bool = True, *, user_id="default",
            agent_id="default", session_id="default", role="user", event_time=None, trigger="per_turn", ttl_seconds=None, expires_at=None, valid_from=None,
            mention_time=None, speaker=None) -> dict:
        if trigger not in {"per_turn", "session_end"}:
            raise ValueError("trigger must be per_turn or session_end")
        mention = parse_session_clock(mention_time)
        now = float(timestamp) if timestamp is not None else mention if mention is not None else self._now()
        chronicle = mention is not None or speaker is not None
        if speaker is not None and (not isinstance(speaker, str) or not speaker.strip() or len(speaker) > 128):
            raise ValueError("speaker must be a nonempty name of at most 128 characters")
        scope = (user_id, agent_id, session_id)
        validity = interval((text or "") if role == "user" else "", now, ttl_seconds=ttl_seconds, expires_at=expires_at, valid_from=valid_from)
        if trigger == "session_end" and (ttl_seconds is not None or expires_at is not None):
            raise ValueError("explicit validity requires per-turn retain")
        turn_id = self.store.append_turn(text or "", now, scope=scope, role=role, event_ts=event_time,
                                         extraction_state="batch" if trigger == "session_end" else "pending",
                                         mention_time=(now if mention is None else mention) if chronicle else None, speaker=speaker)
        if trigger == "session_end":
            return {"facts": [], "operations": [], "queued": True, "turn_id": turn_id}
        facts = extract_facts(text or "") if isinstance(self.llm, MockLLMClient) else []
        job = {"text": text or "", "ts": now, "scope": scope, "role": role, "facts": facts,
               "turn_id": turn_id, "event_time": float(event_time) if event_time is not None else None, "validity": validity,
               "chronicle": {"mention_time": now if mention is None else mention, "speaker": speaker,
                             "origin_turn": turn_id, "raw": text} if chronicle else None}
        if not sync:
            self.pending.append(job)
            return {"facts": facts, "operations": [], "queued": True, "turn_id": turn_id}
        self.drain(user_id=user_id, agent_id=agent_id)
        return self._apply_job(job)

    def buffer_turn(self, text, *, user_id="default", agent_id="default", session_id="default", role="user",
                    timestamp=None, event_time=None, mention_time=None, speaker=None):
        scope = (user_id, agent_id, session_id)
        if any(not isinstance(part, str) or not part.strip() for part in scope):
            raise ValueError("buffer_turn requires a user, agent and session")
        mention = parse_session_clock(mention_time)
        stamp = (self._now() if mention is None else mention) if timestamp is None else float(timestamp)
        return self.store.append_turn(text or "", stamp, scope=scope, role=role,
                                      event_ts=event_time, mention_time=mention, speaker=speaker,
                                      extraction_state="buffered")

    def working_memory(self, session_id="default", *, user_id="default", agent_id="default"):
        return self.get_recent_turns(session_id, self.working_window, user_id=user_id, agent_id=agent_id)

    def get_recent_turns(self, session_id="default", limit=6, *, user_id="default", agent_id="default"):
        scope = (user_id, agent_id, session_id)
        if any(not isinstance(part, str) or not part.strip() for part in scope):
            raise ValueError("recent turns require a user, agent and session")
        if type(limit) is not int or limit < 0:
            raise ValueError("limit must be a nonnegative integer")
        return self.store.session_turns(scope, limit=limit)

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
        blocks = self.store.blocks_for_prompt(scope=scope, snapshot=[row for row in self.store.snapshot(scope=scope)
                                                                if row["expires_at"] is None and row["epistemic_scope"] == "world"])
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
        with self.store.reuse_write_vectors():
            return self._write_turn(job)

    def _deduplicate(self, job):
        text = strip_step_tags(job["text"])
        if (not text or is_greeting(text) or changes_state(text)
                or re.match(r"(?:他|她|它|那部|这个|\b(?:he|she|it|that)\b)", text, re.I)):
            return None
        scope, turn_id = job["scope"], job["turn_id"]
        event = job["event_time"]
        text, inferred = absolute_event_text(text, job["ts"] if event is None else event)
        event = inferred if inferred is not None else event
        exact = self.store.content_matches(content_hash(text), scope)
        known = job["facts"] or extract_facts(text)

        def compatible(fact):
            if fact["expires_at"] is not None or fact["epistemic_scope"] != "world" or fact["role"] != job["role"] or (event is not None and fact["event_time_start"] != event):
                return False
            return (matching_attributes(text, fact["plain"]) and (not known or len(known) == 1 and
                    (not known[0].get("attribute") or known[0]["attribute"].startswith("turn:") or
                     (known[0]["attribute"], known[0]["value"]) == (fact["attribute"], fact["value"]))))

        matches = [fact for fact in exact if fact["expires_at"] is None and fact["epistemic_scope"] == "world"
                   and fact["role"] == job["role"] and (event is None or fact["event_time_start"] == event)]
        level = "hash"
        if not matches:
            hits = self.store.search_vector(text, user_id=scope[0], agent_id=scope[1], session_id=scope[2],
                                            include_superseded=False, top_n=3, memory_type="fact")
            matches = [fact for fact in hits if fact["vector_score"] >= 0.92 and compatible(fact)]
            level = "semantic"
        if len(matches) != 1:
            return None
        with self.store.transaction():
            if turn_id not in self.store.pending_turn_ids([turn_id], scope):
                return []
            target = matches[0]["id"]
            if not self.store.merge_line_mentions(target, step_ids_in(job["text"]), scope=scope, turn_id=turn_id,
                                                  accessed_at=job["ts"], exact_version=True):
                return None
            change = self._record_timeline(2, {"topic": matches[0]["topic"], "operations": 1}, job["ts"], scope, event)
            self.store.finish_turns([turn_id])
        self._refresh_timeline(change)
        return [{"action": "NOOP", "target_line_id": target, "dedup_level": level}]

    def _write_turn(self, job):
        text, now, scope = job["text"], job["ts"], job["scope"]
        facts = job["facts"]
        if job["turn_id"] not in self.store.pending_turn_ids([job["turn_id"]], scope):
            return {"facts": [], "operations": [], "queued": False, "turn_id": job["turn_id"]}
        duplicate = self._deduplicate(job) if not job.get("validity", (None, None))[1] and not job.get("chronicle") else None
        if duplicate is not None:
            return {"facts": facts, "operations": duplicate, "queued": False, "turn_id": job["turn_id"]}
        step_ids = step_ids_in(text)
        greeting = is_greeting(text)
        offline = isinstance(self.llm, MockLLMClient) or job.get("offline", False)
        candidates = [] if greeting else self._candidate_blocks(text, scope, facts)
        if greeting:
            ops = []
        elif offline:
            ops = decide_layout(candidates, facts, now)["write_operations"]
            if not facts and len(text.strip()) >= 16:
                existing = next((block for block in candidates if block["topic"] == NEWS_TOPIC), None)
                ops = [{"action": "APPEND_ADJACENT_LINE", "target_block_id": existing["id"] if existing else "NEW",
                        "topic": NEWS_TOPIC, "formatted_line": format_fact_line(now, text), "z_axis_keys": [NEWS_KEY]}]
        else:
            context = self.store.session_turns(scope, limit=self.working_window, before=job["turn_id"])
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
        if job.get("validity", (None, None))[1] is not None and not any(
                op["action"] not in {"NOOP", "DELETE_LINE"} for op in ops):
            keys = [key for fact in facts for key in fact.get("zkeys", [])]
            keys.extend(match_entities(text, self.store.entity_keys(scope=scope)))
            ops = [{"action": "TEMPORARY_OVERRIDE", "target_block_id": "NEW", "topic": GENERAL_CATEGORY,
                    "formatted_line": format_fact_line(now, text), "z_axis_keys": list(dict.fromkeys(keys))}]
        applied = self._apply_operations(job, candidates, ops)
        if not offline and not greeting and not fallback:
            facts = [{"text": line_body(op["formatted_line"]), "topic": op["topic"], "attribute": op["attribute"],
                      "value": op.get("value") or "", "zkeys": op["z_axis_keys"],
                      "isolated": op["action"] == "INSERT_ISOLATED_LINE"}
                     for op in applied if op["action"] not in {"NOOP", "DELETE_LINE"}]
        return {"facts": facts, "operations": applied, "queued": False, "turn_id": job["turn_id"]}

    def _apply_operations(self, job, candidates, ops, *, batch=None):
        now, scope = job["ts"], job["scope"]
        batch = batch if batch is not None else {job["turn_id"]: job}
        allowed_blocks = {block["id"]: block for block in candidates}
        allowed_lines = {line["id"]: (block, line) for block in candidates for line in block["lines"]}
        alias, applied = {}, []
        change = None
        with self.store.transaction():
            pending = self.store.pending_turn_ids(batch, scope)
            if not pending:
                return []
            for op in ops:
                sources = op.get("source_turn_ids", [job["turn_id"]])
                if not set(sources) <= pending:
                    continue
                source = batch[max(sources)]
                text = "\n".join(batch[turn_id]["text"] for turn_id in sources)
                step_ids = step_ids_in(text)
                action = op["action"]
                target = op.get("target_line_id") or ""
                block = op.get("target_block_id") or "NEW"
                if action in {"OVERWRITE_SAME_LINE", "DELETE_LINE"} and target not in allowed_lines and target not in alias:
                    continue
                if action == "DELETE_LINE" and not _DELETE_CUE.search(text):
                    continue
                if action in {"APPEND_ADJACENT_LINE", "INSERT_ISOLATED_LINE", "TEMPORARY_OVERRIDE"} and block not in allowed_blocks and block != "NEW" and not block.startswith("NEW:"):
                    continue
                validity = source["validity"] if "validity" in source else interval(text if source["role"] == "user" else "", source["ts"])
                if action == "DELETE_LINE" and validity[1] is not None:
                    continue
                if action == "NOOP" and validity[1] is None:
                    if target in allowed_lines or target in alias:
                        self.store.merge_line_mentions(alias.get(target, target), step_ids, scope=scope, turn_ids=sources,
                                                       accessed_at=now, exact_version=True)
                    applied.append(op)
                    continue
                plain = line_body(op.get("formatted_line") or "")
                target_block, target_line = allowed_lines.get(target, ({}, {}))
                topic = target_block.get("topic") or allowed_blocks.get(block, {}).get("topic") or op.get("topic") or GENERAL_CATEGORY
                attribute = target_line.get("attribute") or op.get("attribute") or ""
                if action == "OVERWRITE_SAME_LINE" and target_line.get("attribute") and op.get("attribute") not in (None, "", attribute):
                    continue
                zkeys = [NEWS_KEY] if topic == NEWS_TOPIC else _write_keys(op)
                reference_time = source["event_time"] if source["event_time"] is not None else source["ts"]
                chronicle = source.get("chronicle")
                rewritten, inferred_time = ((plain, None) if chronicle else absolute_event_text(plain, reference_time))
                effective_time = _iso_timestamp(op.get("event_time_iso"))
                if effective_time is None:
                    effective_time = inferred_time if inferred_time is not None else reference_time
                if action == "TEMPORARY_OVERRIDE" and validity[1] is None:
                    continue
                if validity[1] is not None and action != "DELETE_LINE":
                    action = "TEMPORARY_OVERRIDE"
                    block = target_block.get("id", block)
                operation = {**op, "action": action, "target_block_id": block, "z_axis_keys": zkeys,
                             "topic": topic, "attribute": attribute}
                if rewritten != plain:
                    operation["formatted_line"] = format_fact_line(now, rewritten)
                line_id = apply_write_op(self.store, operation, topic=topic, attribute=attribute,
                                         value=op.get("value") or "", plain=plain, timestamp=now, alias=alias,
                                         scope=scope, role=source["role"], event_time=effective_time, source_step_ids=step_ids,
                                         source_turn_ids=sources, valid_from=validity[0], expires_at=validity[1],
                                         epistemic_scope=_epistemic_scope(source, plain), chronicle=chronicle)
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
            self.store.finish_turns(pending)
        self._refresh_timeline(change)
        return applied

    def flush_session(self, session_id, *, user_id="default", agent_id="default", timestamp=None):
        scope = (user_id, agent_id, session_id)
        rows = self.store.session_turns(scope, pending=True)
        operations = []
        if not rows:
            return {"turn_ids": [], "operations": []}
        now = self._now() if timestamp is None else float(timestamp)
        jobs = {row["turn_id"]: {"text": row["text"], "ts": row["ingest_ts"], "event_time": row["event_ts"],
                               "scope": scope, "role": row["role"], "turn_id": row["turn_id"], "facts": [],
                               "chronicle": {"mention_time": row["mention_time"] if row["mention_time"] is not None else row["event_ts"],
                                             "speaker": row["speaker"], "origin_turn": row["turn_id"], "raw": row["text"]}
                               if row["mention_time"] is not None or row["speaker"] else None}
                for row in rows}
        unresolved = jobs
        data = None
        if not isinstance(self.llm, MockLLMClient):
            unresolved = {}
            for turn_id, job in jobs.items():
                with self.store.reuse_write_vectors():
                    duplicate = self._deduplicate(job) if not job.get("validity", (None, None))[1] and not job.get("chronicle") else None
                if duplicate is None:
                    unresolved[turn_id] = job
                else:
                    operations.extend(duplicate)
            if unresolved:
                last = next(reversed(unresolved.values()))
                candidates = self._candidate_blocks("\n".join(job["text"] for job in unresolved.values()), scope, [])
                prompt = render_prompt("layout", step_id="", candidate_2d_blocks=candidates, raw_turn_text="", role="session",
                                       timestamp=now, categories=list(self.categories),
                                       current_date_iso=datetime.fromtimestamp(last["event_time"], timezone.utc).date().isoformat(),
                                       batch_turns=[{"turn_id": turn_id, "role": escape(job["role"]), "text": escape(job["text"]),
                                                     "needs_extraction": turn_id in unresolved,
                                                     "event_date": datetime.fromtimestamp(job["event_time"], timezone.utc).date().isoformat()}
                                                    for turn_id, job in jobs.items()],
                                       recent_turns=[{"role": row["role"], "text": escape(row["text"])} for row in
                                                     self.store.session_turns(scope, limit=self.working_window, before=min(jobs))])
                data, _ = self._call_json(prompt, lambda: None, lambda payload: _valid_operations(payload) and all(
                    op.get("source_turn_ids") and set(op["source_turn_ids"]) <= unresolved.keys()
                    for op in payload["write_operations"]))
                if data is not None:
                    ordered = sorted(data["write_operations"], key=lambda op: max(op["source_turn_ids"]))
                    operations.extend(self._apply_operations({**last, "ts": now}, candidates, ordered, batch=unresolved))
        if data is None and unresolved:
            from memengine.lifecycle import resolve_subject

            prior = self.store.session_turns(scope, limit=1, before=min(unresolved))
            previous = prior[-1]["text"] if prior else ""
            for job in unresolved.values():
                resolved = resolve_subject(job["text"], previous, extract_facts)
                operations.extend(self._apply_job({**job, "text": resolved, "facts": extract_facts(resolved), "offline": True})["operations"])
                previous = resolved
        self.pending[:] = [job for job in self.pending if job["turn_id"] not in jobs]
        return {"turn_ids": list(jobs), "operations": operations}

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
                 top_k=None, token_budget=None, timestamp=None, read_only=False, flush_pending=True,
                 belt_prune_ratio: float | None = None, min_keep: int | None = None,
                 include_raw_turns: bool = False, include_cold: bool = False, filters: RecallFilter | None = None,
                 scopes: list[str] | None = None, mode: Literal["single_query", "reconstructive"] = "single_query",
                 reference_time=None, packing_mode=None, version_precision=False, mutate_stats=False) -> dict:
        if mode not in ("single_query", "reconstructive"):
            raise ValueError("invalid recall mode")
        if scopes is not None and (isinstance(scopes, str) or not set(scopes) <= SCOPES):
            raise ValueError("invalid epistemic scopes")
        if packing_mode not in (None, "token_budget", "top_k"):
            raise ValueError("invalid packing mode")
        wide = packing_mode == "token_budget" or packing_mode is None and token_budget is not None and top_k is None
        graph = mode == "reconstructive" and (reference_time is not None or packing_mode == "token_budget")
        budget = 2048 if token_budget is None else token_budget
        limit = (60 if wide else 10) if top_k is None else top_k
        if type(budget) is not int or type(limit) is not int or budget < 0 or limit < 0:
            raise ValueError("token_budget and top_k must be nonnegative integers")
        anchor = parse_session_clock(reference_time)
        if anchor is not None:
            timestamp = anchor
        versions = tuple(extract_versions(query)) if version_precision or wide or anchor is not None else ()
        windows = tuple(span for span in resolve_temporal_spans(query, anchor)
                        if span.start is not None or span.end is not None) if anchor is not None else ()
        plan = RecallPlan(windows, versions, graph, wide) if wide or graph or windows or versions else None
        args = dict(user_id=user_id, agent_id=agent_id, session_id=session_id, top_k=limit, plan=plan,
                    token_budget=budget, timestamp=timestamp, flush_pending=flush_pending,
                    belt_prune_ratio=belt_prune_ratio, min_keep=min_keep, include_cold=include_cold,
                    filters=filters, scopes=scopes, include_raw_turns=include_raw_turns,
                    mutate_stats=mutate_stats and not read_only)
        if mode == "single_query" or graph:
            return self._retrieve_single(query, read_only=read_only, **args)
        started = time.perf_counter()
        if self.pending and flush_pending and not read_only:
            self.drain(user_id=user_id, agent_id=agent_id)
        passes = []
        result = self._retrieve_single(query, read_only=True, _passes=passes, **args)
        bridges = bridge_keys(query, result["ranked"])
        if bridges:
            result = self._retrieve_single(query, read_only=read_only, _passes=passes, _bridges=bridges,
                                           _primary=result["ranked"], **args)
        elif args["mutate_stats"] and result["route"] == TASK:
            control_scope = (user_id, agent_id, "default" if session_id is None else session_id)
            self._record_recall(control_scope, result["origin"], result["radius"], result["ranked"],
                                float(timestamp) if timestamp is not None else self._now())
        result["retrieval_ms"] = (time.perf_counter() - started) * 1000
        return result

    def _retrieve_single(self, query, *, user_id, agent_id, session_id, top_k, token_budget, timestamp, read_only,
                         flush_pending, belt_prune_ratio, min_keep, include_raw_turns, include_cold, filters, scopes, plan,
                         mutate_stats=False, _passes=None, _bridges=(), _primary=()):
        prune_ratio = self.belt_prune_ratio if belt_prune_ratio is None else float(belt_prune_ratio)
        keep_floor = self.min_keep if min_keep is None else min_keep
        _validate_belt_config(prune_ratio, keep_floor)
        started = time.perf_counter()
        now = float(timestamp) if timestamp is not None else self._now()
        scope = (user_id, agent_id, session_id)
        control_scope = (user_id, agent_id, "default" if session_id is None else session_id)
        metadata, cached_history, cached_maintenance = self.store.recall_metadata(
            scope, self._control_key(control_scope), legacy=control_scope == DEFAULT_SCOPE,
        ) if plan else (None, None, None)
        origin, radius = self._control_state(control_scope, metadata)
        direct = []
        temporal = parse_temporal_intent(query, now)

        recall_filter = replace(filters or RecallFilter(), scopes=tuple(scopes) if scopes is not None else (filters.scopes if filters else None),
                                as_of=None if temporal.is_historical else now,
                                historical_window=(temporal.start_ts, temporal.end_ts) if temporal.is_historical else None)
        def result(route, parsed, *, ranked=(), belt_total=0, scan=None, fallback=False, weights=None, gate_stats=None):
            outer = [fact for fact in ranked if fact["zone"] == OUTER]
            inner = [fact for fact in ranked if fact["zone"] == INNER]
            belt = [fact for fact in ranked if fact["zone"] == BELT]
            context = "\n".join(fact["text"] for fact in ranked)
            history_tokens = self.store.history_tokens(scope) if cached_history is None else cached_history
            drilldown = self.store.get_turn_window(
                [turn_id for fact in ranked[:3] for turn_id in fact.get("source_turn_ids", [])],
                user_id=user_id, agent_id=agent_id, session_id=session_id) if include_raw_turns else []
            blocked = self.store.inactive_temporary_sources({row["turn_id"] for row in drilldown}, scope=scope,
                                                            as_of=recall_filter.as_of)
            drilldown = [row for row in drilldown if row["turn_id"] not in blocked]
            stats = {"candidate_count": 0, "max_ce_score": 0.0, "tau_abs": None, "cliff_dropped_count": 0,
                     "belt_pruned_count": 0, "empty_reason": "casual"} if gate_stats is None else gate_stats
            if not ranked and stats["empty_reason"] is None:
                stats = {**stats, "empty_reason": "budget"}
            return {"route": route, "hits": [], "ranked": list(ranked), "outer": outer, "inner": inner,
                    "belt": belt, "origin": origin, "radius": radius,
                    "drilldown_turns": drilldown, "temporal_intent": vars(temporal),
                    "outer_triggered_count": len(outer), "inner_count": len(inner), "belt_count": len(belt),
                    "belt_compressed_count": belt_total - len(belt),
                    "belt_pruned_count": stats["belt_pruned_count"], "gate_stats": stats,
                    "matched_entities": direct, "entity_fallback": fallback,
                    "reconstructed": bool(_bridges or any(row.get("activation") for row in ranked)),
                    "bridge_entities": list(dict.fromkeys([*_bridges, *(key for row in ranked for key in row.get("bridge_keys", ())) ])),
                    "packing_mode": "token_budget" if plan and plan.wide else "top_k",
                    "epistemic_scopes": recall_filter.scopes, "validity_as_of": recall_filter.as_of,
                    "empty_retrieval": not ranked, "empty_recall": not ranked,
                    "parsed_query": parsed, "scan_far_to_near": scan or [], "context": context,
                    "retrieval_ms": (time.perf_counter() - started) * 1000, "context_tokens": estimate_tokens(context),
                    "history_tokens": history_tokens, "poison_circle": weights,
                    "scope": {"user_id": user_id, "agent_id": agent_id, "session_id": session_id}}

        if (filters is None and scopes is None and self.enable_casual_bypass and is_greeting(query) and not (_RECALL_CUE.search(query) or _QUESTION_CUE.search(query))
                and not match_entities(query, self.store.entity_keys(scope=scope, query=query))):
            return result(CASUAL, parse_query_rules(query, categories=self.categories))

        if self.pending and flush_pending and not read_only:
            self.drain(user_id=user_id, agent_id=agent_id)
        parsed = parse_query_rules(query, categories=self.categories)
        if temporal.is_historical:
            parsed["query_type"] = "TIME_TRAVEL"
        with self.store.read_snapshot():
            search_cold = include_cold or temporal.is_historical
            if plan:
                keyed, aliases = self.store.recall_entity_keys(query, scope=scope, include_cold=search_cold)
            else:
                keyed = self.store.entity_keys(scope=scope, include_cold=search_cold, query=query)
            direct = match_entities(query, keyed)
            if plan:
                direct = list(dict.fromkeys([*direct, *aliases]))
            # Categories describe a topic. Require an entity or task cue to leave casual routing.
            if (filters is None and scopes is None and self.enable_casual_bypass and not include_cold
                    and not direct and not parsed["entities"]
                    and parsed["query_type"] == "FACTUAL" and not _RECALL_CUE.search(query)
                    and route_query(query, ()) == CASUAL):
                return result(CASUAL, parsed)
            if _bridges:
                bridge_query = " ".join(key.partition(":")[2] or key for key in _bridges)
                followup = self._recall(bridge_query, parse_query_rules(bridge_query), temporal, scope, _bridges,
                                       include_cold=include_cold, filters=recall_filter)
                merged = merge_passes(_passes[0], followup)
            else:
                merged = (self._recall(query, parsed, temporal, scope, direct, include_cold=include_cold,
                                       filters=recall_filter, plan=plan) if plan else
                          self._recall(query, parsed, temporal, scope, direct, include_cold=include_cold, filters=recall_filter))
                if _passes is not None:
                    _passes.append(merged)
            maintenance = (json.loads(self.store.get_meta(self.store.maintenance_key(scope)) or "{}")
                           if cached_maintenance is None else cached_maintenance)
            historical_query = parsed["query_type"] == "TIME_TRAVEL"
            sessions = self.store.session_orders(scope, {fact["session_id"] for fact in merged[:30]})
        hints = list(dict.fromkeys(key for fact in merged[:10] for key in fact["entity_z_keys"]))[:15]
        preview = [{"block_id": fact["block_id"], "topic": fact["topic"],
                    "preview": fact["plain"][:180], "preview_attr": fact["attribute"]} for fact in merged[:8]]
        decision = decide_origin(origin, preview, direct or hints, query)
        circle = decision["poison_circle"]
        origin = decision["new_origin"]
        radius = 0.35 if historical_query else inner_boundary(circle["radius_r"])
        c_max = max((consolidation_count(fact) for fact in merged if not is_noise(fact)), default=1)
        rrf_order = {fact["id"]: rank for rank, fact in enumerate(merged)}
        ce_scores, backend = self._rerank(query, merged)
        blended = bool(plan and plan.wide and backend.is_calibrated)
        if blended:
            local = self.local_reranker.score(query, merged)
            ce_scores = [(neural + lexical) / 2 for neural, lexical in zip(ce_scores, local)]
        roles = {} if historical_query else temporal_competition(merged, sessions, query, ce_scores)
        contextual = any(fact.get("contextual_query") for fact in merged)
        use_context_score = contextual and not backend.is_calibrated
        anchor_bonus = anchored_relevance(query, merged) if use_context_score else ()
        pair_scores = evidence_pair_scores(query, merged) if plan and plan.wide and not backend.is_calibrated else ()
        buckets = {}
        temporary_terms = competition_terms(query) if any(f.get("expires_at") is not None for f in merged) else set()
        for index, (fact, ce_score) in enumerate(zip(merged, ce_scores)):
            keys = fact["entity_z_keys"]
            role = roles.get(fact["id"])
            status_weight = self.superseded_weight if fact["status"] == "superseded" and not historical_query else 1.0
            temporal_weight = 1.12 if role == "latest" else 0.65 if role == "older" and status_weight == 1.0 else 1.0
            tagged = score_bipolar({**fact, "ce_score": ce_score}, c_max=c_max, radius=radius,
                                    bipolar_lambda=self.bipolar_lambda, epsilon=self.epsilon,
                                    status_weight=status_weight,
                                    decay=1.0 if historical_query or (include_cold and fact["tier"] == "cold") else decay_factor(
                                        fact, maintenance.get("as_of"), maintenance.get("half_life_days", 30)))
            score = tagged["final_score"]
            if use_context_score:
                score += anchor_bonus[index]
            score = score * temporal_weight + 0.35 * fact["exact_hit_ratio"]
            if plan and plan.wide:
                overlap = query_overlap(query, fact)
                coverage, clause = pair_scores[index] if pair_scores else (overlap, 0)
                quality = .22 * coverage + .10 * clause + .20 * fact["rrf_norm"]
                quality += .30 * calendar_relevance(query, plan.windows, fact)
                if fact.get("speaker") and re.search(r"(?<!\w)" + re.escape(fact["speaker"]) + r"(?!\w)",
                                                      query, re.I):
                    quality += .08 * coverage
                if fact.get("activation"):
                    quality += min(.15, 1.5 * fact["activation"]) * min(1, .25 + overlap)
                    if fact.get("speaker_bridge") and fact["graph_depth"] == 2 and fact["entity_anchor"] and overlap > 0:
                        quality += min(.09, overlap * .2)
                        tagged["protected_bridge"] = tagged["zone"] == BELT
                score += quality * status_weight * temporal_weight
            if plan and plan.versions:
                score = (score + fact.get("version_boost", 0)) * fact.get("version_multiplier", 1)
            elif fact["version_mismatch"]:
                score *= .30
            if fact.get("expires_at") is not None:
                deadline = datetime.fromtimestamp(fact["expires_at"], timezone.utc)
                tagged["state_tag"] = f"[临时生效至 {deadline:%Y-%m-%dT%H:%MZ}]"
                if fact["entity_anchor"] or temporary_terms.intersection(competition_terms(fact["plain"])):
                    tagged["temporary_priority"] = True
                    score *= 1.12
            tagged.update(score=score, final_score=score, bipolar_score=score, temporal_role=role, temporal_weight=temporal_weight)
            if role == "older" and not tagged.get("state_tag"):
                tagged["state_tag"] = "[可能已过时/较早记录]"
            for key in keys or fact["category_z_keys"][:1] or ["未分桶"]:
                buckets.setdefault(key, []).append(tagged)
        scan = order_far_to_near(circle.get("z_axis_scan_order_far_to_near", []), list(dict.fromkeys([*direct, *buckets])))
        collapsed = collapse_and_scan_z_axis(buckets, scan)
        ranked = [{**fact, "z_bucket": bucket["z_bucket"]} for bucket in collapsed for fact in bucket["facts"]]
        time_direction = 1 if temporal.order_hint == "earliest" else -1
        ranked.sort(key=lambda fact: (-fact.get("version_confidence", 0),
                                     fact.get("version_prerelease", False) if fact.get("version_grade") == "CONTAINED_SPECIFIC" else False,
                                     -fact["exact_hit_ratio"], not fact.get("temporary_priority"), -fact["final_score"],
                                     fact["memory_type"] != "procedural", rrf_order[fact["id"]]))
        tau_abs = self.tau_abs if self.tau_abs is not None else (0.18 if backend.is_calibrated else 0.12)
        mismatches = sum(fact["version_mismatch"] for fact in ranked)
        ranked, identifier_mode = eligible_identifiers(ranked, query, historical_query, self.allow_empty_recall)
        strict_version = identifier_mode == "version"
        if plan and plan.versions and not historical_query and not compares_versions(query):
            if any(fact.get("version_grade") for fact in ranked):
                ranked = [fact for fact in ranked if fact.get("version_confidence", 0) > 0]
            elif self.allow_empty_recall:
                ranked = []
        exact_query = any(fact["exact_ident_hit"] for fact in ranked)
        recurrent_belt = sum(fact["zone"] == BELT and consolidation_count(fact) > 1 and not is_noise(fact)
                             for fact in ranked) >= 3 and c_max >= 4
        evidence_floor = not plan and not backend.is_calibrated and not exact_query and not strict_version and not recurrent_belt and not contextual and not _bridges and not any(f.get("temporary_priority") for f in ranked)
        floor = _fit_context(baseline_rrf(merged), top_k, token_budget) if evidence_floor else []
        wide_evidence = (not historical_query and evidence_floor and isinstance(backend, CalibratedLocalReranker)
                         and len(floor) >= 4
                         and any(estimate_tokens(fact["text"]) > 160 for fact in ranked[:3]))
        if wide_evidence:
            ranked = self.local_reranker.rank_long_evidence(ranked, floor)
        kept, gate_stats = precision_gate(ranked, tau_abs=tau_abs, allow_empty_recall=self.allow_empty_recall,
                                          cliff_ratio=self.cliff_ratio, belt_prune_ratio=0 if evidence_floor else prune_ratio,
                                          min_keep=max(keep_floor, min(10, top_k)) if blended else keep_floor,
                                          is_calibrated=backend.is_calibrated and not blended, top_k=top_k)
        if strict_version and not ranked:
            gate_stats["empty_reason"] = "no_exact_version"
        elif identifier_mode == "symbol" and not ranked:
            gate_stats["empty_reason"] = "no_exact_identifier"
        if evidence_floor and kept and not wide_evidence:
            floor_ids = {fact["id"] for fact in floor}
            kept = [fact for fact in kept if fact["id"] in floor_ids]
        if historical_query and not (plan and plan.windows):
            def temporal_order(fact):
                matched = fact["temporal_match"] and fact["ce_score"] >= tau_abs
                return (not matched,
                        time_direction * fact["event_time_start"] if matched and temporal.order_hint else 0,
                        -fact["final_score"],
                        time_direction * fact["event_time_start"] if matched else 0,
                        rrf_order[fact["id"]])

            kept.sort(key=temporal_order)
        elif evidence_floor and not wide_evidence and isinstance(backend, CalibratedLocalReranker):
            kept = backend.stabilize_shortlist(kept, floor if kept else [])
        gate_stats.update(reranker=type(backend).__name__, is_calibrated=backend.is_calibrated,
                          reranker_fallback=backend is not self.reranker, rrf_evidence_floor=evidence_floor and not wide_evidence,
                          wide_evidence=wide_evidence, blended=blended, version_mismatch_count=mismatches)
        if _bridges:
            kept = select_bridged(query, _primary, kept, _bridges)
        if plan and plan.wide:
            useful = [fact for fact in kept if fact["final_score"] >= .045 or fact.get("exact_ident_hit")]
            if plan.graph and not plan.versions and not plan.windows:
                useful = diversify_episodes(query, useful)
            kept = [_compact_evidence(query, fact) for fact in useful]
        selected = _fit_context(kept, top_k, token_budget)
        if mutate_stats and not read_only:
            self._record_recall(control_scope, origin, radius, selected, now)
        return result(TASK, parsed, ranked=selected, belt_total=sum(fact["zone"] == BELT for fact in ranked),
                      gate_stats=gate_stats,
                      scan=scan, fallback=not direct and bool(hints),
                      weights={"radius_r": radius, "radius_out": 1.0 - radius, "bipolar_lambda": self.bipolar_lambda,
                               "belt_prune_ratio": prune_ratio, "min_keep": keep_floor,
                               "z_axis_scan_order_far_to_near": scan})

    def _record_recall(self, scope, origin, radius, ranked, now):
        with self.store.transaction():
            self.store.set_meta(self._control_key(scope),
                                json.dumps({"origin": origin, "radius": radius}, ensure_ascii=False))
            if ranked:
                self.store.increment_hit_counts([fact["id"] for fact in ranked], now)
        if scope == DEFAULT_SCOPE:
            self.origin, self.radius = origin, radius

    def _recall(self, query, parsed, temporal, scope, direct, *, include_cold=False, filters=None, plan=None):
        return recall(self.store, 55 if plan and plan.wide and self.per_channel_k == 35 else self.per_channel_k, query, parsed, temporal, scope, direct,
                      vector_search=self.search_vector, fuse=weighted_rrf, include_cold=include_cold, filters=filters, plan=plan)

    def _rerank(self, query, candidates):
        if not candidates:
            return [], self.reranker
        try:
            return checked_scores(self.reranker.score(query, candidates), len(candidates)), self.reranker
        except (OSError, HTTPException, ValueError, KeyError, TypeError, AttributeError):
            if not isinstance(self.reranker, OpenAICompatibleReranker):
                raise
            return self.local_reranker.score(query, candidates), self.local_reranker

    def search_z_hash(self, entities, categories, query_stem="", *, snapshot=None, index=None, expanded=None):
        seeds = list(entities or [])
        if query_stem:
            seeds.extend(match_entities(query_stem, self.store.entity_keys()))
        return self.store.search_entity_inverted(seeds, include_superseded=False)

    def search_vector(self, query_text: str, top_n: int = 40, *, snapshot=None, user_id="default", agent_id="default",
                      session_id=None, ids_only=False, include_cold=False, filters=None) -> list[dict]:
        if hasattr(self.vector, "encode") or not getattr(self.store, "supports_local_vector", True):
            return self.store.search_vector_topk(query_text, top_n=top_n, ids_only=ids_only, user_id=user_id,
                                            agent_id=agent_id, session_id=session_id, include_cold=include_cold, filters=filters)
        facts = self.store.snapshot(scope=(user_id, agent_id, session_id), include_superseded=True,
                                    include_cold=include_cold, include_derived=True) if snapshot is None else snapshot
        facts = [fact for fact in facts if (filters.matches(fact) if filters is not None else fact["memory_type"] != "observation")]
        hits = [fact for fact in self.vector.search(query_text, facts, top_n=top_n) if fact.get("vector_score", 0) > 0]
        return [{"id": fact["id"], "vector_score": fact["vector_score"]} for fact in hits] if ids_only else hits

    def has_facts(self) -> bool:
        return bool(self.store.snapshot())

    def get_recent_facts(self, limit: int = 10, *, snapshot=None) -> list[dict]:
        facts = self.store.snapshot() if snapshot is None else snapshot
        return sorted(facts, key=lambda fact: (-fact["updated_at"], fact["id"]))[:max(0, limit)]

    def decay_and_archive(self, *, user_id="default", agent_id="default", timestamp=None,
                          half_life_days=30.0, cold_inactive_days=90.0, cold_superseded_days=30.0):
        from math import isfinite

        if (not isfinite(half_life_days) or half_life_days <= 0
                or any(not isfinite(days) or days < 0 for days in (cold_inactive_days, cold_superseded_days))):
            raise ValueError("half-life must be positive and archive ages nonnegative")
        now = self._now() if timestamp is None else float(timestamp)
        archived = self.store.archive_cold((user_id, agent_id, None), now=now, half_life_days=half_life_days,
                                           inactive_days=cold_inactive_days, superseded_days=cold_superseded_days)
        return {"archived_ids": archived}

    def record_tool_call(self, tool_call_id, *, trigger, technique, success, source_turn_ids=(), derived_from_ids=(),
                         user_id="default", agent_id="default", session_id="default", timestamp=None,
                         task_id=None, evidence=None):
        if not all(isinstance(value, str) and value.strip() for value in (tool_call_id, trigger, technique)) or type(success) is not bool:
            raise ValueError("tool call requires an ID, trigger, technique, and boolean success")
        owner = (user_id, agent_id, session_id)
        with self.store.transaction():
            self.store.record_tool_call(tool_call_id, trigger.strip(), technique.strip(), scope=owner,
                                        success=success, source_turn_ids=list(source_turn_ids), derived_from_ids=list(derived_from_ids),
                                        timestamp=self._now() if timestamp is None else float(timestamp), task_id=task_id)
            if task_id is not None:
                return self.record_task_evidence(task_id, evidence if evidence is not None else technique, "tool_evidence",
                                                 [f"tool_call:{tool_call_id}"], user_id=user_id, agent_id=agent_id,
                                                 session_id=session_id)

    def remember(self, text, *, memory_type="fact", source_turn_ids=(), tool_call_id=None, trigger="",
                 is_constraint=False, zkeys=(), user_id="default", agent_id="default", session_id="default", timestamp=None,
                 epistemic_scope=None, confidence=1.0, ttl_seconds=None, expires_at=None, valid_from=None,
                 mention_time=None, speaker=None, _promotion_task=None):
        from memengine.consolidator import write_memory

        if memory_type not in {"fact", "procedural"} or not isinstance(text, str) or not text.strip():
            raise ValueError("remember requires nonempty text and fact or procedural memory_type")
        scope = (user_id, agent_id, session_id)
        turns = list(source_turn_ids)
        calls = self.store.tool_calls(scope, call_id=tool_call_id) if tool_call_id else []
        if tool_call_id and not calls:
            raise ValueError("tool_call_id must refer to a registered tool invocation in this session")
        if calls and calls[0]["task_id"] is not None and calls[0]["task_id"] != _promotion_task:
            raise ValueError("task tool evidence must be promoted through commit_task")
        if not turns and not calls:
            raise ValueError("remember requires real source_turn_ids or a registered tool_call_id")
        turns = sorted({*turns, *(turn for call in calls for turn in call["source_turn_ids"])})
        self.store._validate_turn_ids(turns, scope)
        trigger_text = trigger.strip() or (calls[0]["trigger_text"] if calls and memory_type == "procedural" else "")
        if memory_type == "procedural" and not trigger_text:
            raise ValueError("procedural memory requires a trigger condition")
        mention = parse_session_clock(mention_time)
        timestamp = (mention if mention is not None else self._now()) if timestamp is None else float(timestamp)
        if speaker is not None and (not isinstance(speaker, str) or not speaker.strip() or len(speaker) > 128):
            raise ValueError("speaker must be a nonempty name of at most 128 characters")
        details = {"mention_time": timestamp if mention is None else mention, "speaker": speaker,
                   "origin_turn": max(turns, default=None), "raw": text} if mention is not None or speaker else None
        if epistemic_scope is None and details and memory_type == "fact":
            epistemic_scope = _epistemic_scope({"role": "user", "chronicle": details}, text)
        start, end = interval(text, timestamp, ttl_seconds=ttl_seconds, expires_at=expires_at, valid_from=valid_from)
        if epistemic_scope is None and calls and memory_type == "fact":
            epistemic_scope = "experience"
        return write_memory(self.store, text.strip(), scope=scope, now=timestamp, memory_type=memory_type,
                            source_turn_ids=turns, derived_from_ids=[ref for call in calls for ref in call["derived_from_ids"]],
                            trigger_text=trigger_text, is_constraint=is_constraint, zkeys=zkeys,
                            epistemic_scope=epistemic_scope, confidence=confidence, valid_from=start, expires_at=end, chronicle=details,
                            tool_call_ids=[call["tool_call_id"] for call in calls],
                            successful=[call["tool_call_id"] for call in calls if call["success"]])

    def forget(self, target=None, *, query=None, hard=False, user_id="default", agent_id="default", timestamp=None):
        from memengine.consolidator import consolidation_key, refresh_profile

        if (target is None) == (query is None):
            raise ValueError("provide one memory ID or one query")
        scope = (user_id, agent_id, None)
        now = self._now() if timestamp is None else float(timestamp)
        if query is not None:
            exact = self.store.content_matches(content_hash(query), scope, memory_type=None, include_cold=True)
            if len(exact) > 1:
                return False
            if exact:
                target = exact[0]["id"]
            elif not hard:
                ranked = self.retrieve(query, user_id=user_id, agent_id=agent_id, timestamp=now,
                                       read_only=True, include_cold=True, top_k=40, min_keep=40, belt_prune_ratio=0)["ranked"]
                active = sorted((fact for fact in ranked if fact["status"] == "active"), key=lambda fact: -fact["ce_score"])
                if not active or active[0]["ce_score"] < 0.75 or len(active) > 1 and active[0]["ce_score"] - active[1]["ce_score"] < 0.15:
                    return False
                target = active[0]["id"]
            else:
                return False
        with self.store.transaction():
            deleted = self.delete(target, hard=hard, user_id=user_id, agent_id=agent_id, timestamp=now)
            if deleted and self.store.get_meta(consolidation_key(scope)) is not None:
                refresh_profile(self.store, scope, now)
        return deleted

    def consolidate(self, user_id="default", agent_id="default", min_new_facts=8, *, force=False, timestamp=None):
        from memengine.consolidator import consolidate

        if type(min_new_facts) is not int or min_new_facts < 0:
            raise ValueError("min_new_facts must be a nonnegative integer")
        return consolidate(self.store, user_id=user_id, agent_id=agent_id, min_new_facts=min_new_facts, force=force,
                           now=self._now() if timestamp is None else float(timestamp))

    def build_context(self, query: str, *, user_id="default", agent_id="default", session_id=None,
                      token_budget=None, drilldown_top_n=2, drilldown_window=0,
                      mode: Literal["single_query", "reconstructive"] = "single_query",
                      scopes=None, filters=None, include_cold=False, top_k=None, timestamp=None, reference_time=None,
                      packing_mode=None, task_id=None, include_short_term=True, retrieved=None) -> dict:
        explicit_budget = token_budget is not None
        token_budget = 4096 if token_budget is None else token_budget
        for name, value in (("token_budget", token_budget), ("drilldown_top_n", drilldown_top_n),
                            ("drilldown_window", drilldown_window)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        prefix = (render_prefix(self.store.get_core_snapshot(user_id=user_id, agent_id=agent_id))
                  if scopes is None and (filters is None or filters.scopes is None) else "")
        prefix_tokens = estimate_tokens(prefix)
        if prefix_tokens > 360:
            raise ValueError("core snapshot exceeds the 360-token prefix limit")
        remaining = token_budget - prefix_tokens
        if remaining < 0:
            raise ValueError("token_budget cannot fit the stable system prefix")
        as_of = parse_session_clock(reference_time if reference_time is not None else timestamp)
        as_of = self._now() if as_of is None else as_of
        workspace_scope = (user_id, agent_id, "default" if session_id is None else session_id)
        if retrieved is not None:
            owner = {"user_id": user_id, "agent_id": agent_id, "session_id": session_id}
            if retrieved.get("scope") != owner:
                raise ValueError("retrieved evidence belongs to a different user, agent or session")
            ranking = retrieved.get("ranked", [])
            current = {row["id"]: row for row in self.store.facts_by_ids(
                [row["id"] for row in ranking], scope=(user_id, agent_id, session_id))}
            metadata = ("entity_anchor", "zone", "final_score", "vector_score", "ce_score", "temporal_match",
                        "exact_ident_hit", "temporal_role", "context_anchor_turn_id")
            result = {**retrieved, "ranked": [{**current[row["id"]], **{k: row[k] for k in metadata if k in row}}
                                              for row in ranking if row["id"] in current]}
            if timestamp is None and reference_time is None and result["validity_as_of"] is not None:
                as_of = result["validity_as_of"]
        else:
            result = self.retrieve(query, user_id=user_id, agent_id=agent_id, session_id=session_id,
                                   token_budget=remaining, read_only=True, mutate_stats=False, mode=mode, scopes=scopes,
                                   filters=filters, include_cold=include_cold, top_k=top_k, timestamp=as_of,
                                   reference_time=reference_time, packing_mode=packing_mode or
                                   ("token_budget" if explicit_budget and top_k is None else "top_k"))
        workspace = self.get_task_workspace(task_id, user_id=user_id, agent_id=agent_id,
                                            session_id=workspace_scope[2], limit=32) if task_id is not None else None
        recent = (self.store.session_turns(workspace_scope, limit=self.working_window, as_of=as_of, for_context=True)
                  if include_short_term and result["route"] != CASUAL else [])
        bundle = assemble_context(self.store, result, prefix=prefix, token_budget=token_budget,
                                  drilldown_top_n=drilldown_top_n, drilldown_window=drilldown_window, query=query,
                                  workspace=workspace, recent=recent)
        return {**result, **bundle, "scope": result["scope"], "workspace_session_id": workspace_scope[2],
                "task_id": task_id, "task_status": workspace["status"] if workspace else None, "validity_as_of": as_of}

    def as_tools(self, user_id="default", agent_id="default", session_id=None):
        from memengine.tools import MemoryTools

        return MemoryTools(self, user_id, agent_id, session_id)

    def dispatch_tool(self, name, arguments, *, user_id="default", agent_id="default", session_id=None, tool_call_id=None):
        return self.as_tools(user_id, agent_id, session_id).dispatch_tool(name, arguments, tool_call_id=tool_call_id)

    def answer(self, query: str, timestamp: float | None = None, retrieved: dict | None = None, *,
               user_id="default", agent_id="default", session_id=None, task_id=None, reference_time=None,
               token_budget=None, include_short_term=True) -> dict:
        bundle = self.build_context(query, user_id=user_id, agent_id=agent_id, session_id=session_id, task_id=task_id,
                                    timestamp=timestamp, reference_time=reference_time, token_budget=token_budget,
                                    include_short_term=include_short_term, retrieved=retrieved)
        return self._answer_from_context(query, bundle)

    def _answer_from_context(self, query, bundle):
        if bundle["route"] == CASUAL:
            return {**bundle, "response": "", "reasoning": None, "llm_fallback": False, "claims": [], "abstained": False}
        refusal = ("没有找到足够的证据。" if re.search(r"[\u3400-\u9fff]", query) else "Insufficient evidence to answer.")
        if bundle["empty_recall"]:
            response = "没有找到相关记忆。" if re.search(r"[\u3400-\u9fff]", query) else "No relevant memory found."
            return {**bundle, "response": response, "reasoning": None, "llm_fallback": False, "claims": [], "abstained": True}

        query_type = bundle["parsed_query"]["query_type"]
        eligible = admissible_evidence(bundle)
        facts = [row for row in eligible.values() if row["kind"] == "fact"]
        additional = [row for row in eligible.values() if row["kind"] != "fact" and evidence_matches_query(query, row)]
        temporal = bundle["temporal_intent"]
        # A before-Nth or plain previous query already names the revision. Keep the recalled
        # facts when the only shared term is the attribute, so the ordinal selection can run.
        revision_ask = query_type == "TIME_TRAVEL" and (
            temporal.get("rev_index") is not None or (
                temporal.get("order_hint") in {"latest", "earliest"}
                and temporal.get("start_ts") is None and temporal.get("end_ts") is None))
        # Preserve the notebook selection rules for fact synthesis. Fresh session and task evidence
        # can answer directly when long-term recall lacks a relevant fact.
        fallback_rows = facts if facts and (revision_ask or any(evidence_matches_query(query, row) for row in facts)) else additional[:1]
        if fallback_rows and fallback_rows[0]["kind"] != "fact":
            fallback_rows = [{**row, "id": row["evidence_id"]} for row in fallback_rows]
        entities = bundle["matched_entities"]
        memories = [{key: row[key] for key in ("id", "evidence_id", "text", "attribute", "entity_z_keys", "topic",
                                               "historical", "revision", "root_fact_id", "entity_anchor",
                                               "temporal_match", "vector_score", "zkeys", "is_conflict_or_override",
                                               "isolated") if key in row} for row in fallback_rows]
        offline = isinstance(self.llm, MockLLMClient)
        visible = {key: {"text": row["text"], "kind": row["kind"],
                         "status": row.get("verification_status") or row.get("role")}
                   for key, row in bundle["evidence_map"].items()
                   if key in eligible or row["kind"] == "working_item"}
        goal = next((row["text"] for row in bundle["items"] if row["kind"] == "task_goal"), "")
        prompt = render_prompt("reason", query_type=query_type, retrieved_top_k_memories=memories if offline else [],
                               matched_entities=entities,
                               temporal_intent=temporal, user_question=json.dumps(query, ensure_ascii=False),
                               context_bundle=json.dumps(goal, ensure_ascii=False),
                               citation_index=visible if not offline else {})
        if bundle["system_prefix"]:
            prompt = bundle["system_prefix"] + "\n\n" + prompt
        data, used_fallback = self._call_json(prompt,
            lambda: plan_answer(fallback_rows, query, query_type, entities, temporal), _valid_reason)
        cited = ({key for claim in data["claims"] for key in claim["evidence_ids"]} if "claims" in data else eligible.keys())
        fact_ids = [row["id"] for key, row in eligible.items() if key in cited and row["kind"] in {"fact", "observation"}]
        scope = bundle["scope"]
        states = self.store.evidence_fact_states(fact_ids, scope=(scope["user_id"], scope["agent_id"], scope["session_id"]))
        for ident in fact_ids:
            if ident not in states or not fact_is_current(states[ident], bundle["validity_as_of"], query_type == "TIME_TRAVEL"):
                eligible.pop(f"fact:{ident}", None)
        aliases = {}

        def aliases_for(row):
            ident = row.get("canonical_entity_id")
            if ident and ident not in aliases:
                aliases[ident] = self.store.unique_entity_aliases(ident, (scope["user_id"], scope["agent_id"]))
            return aliases.get(ident, ())

        approved, unresolved = assess_claims(data, eligible, alias_lookup=aliases_for, claim_time=bundle["validity_as_of"])
        task_id = bundle["task_id"]
        provisional = []
        if bundle["task_status"] == "active":
            owner = {**bundle["scope"], "session_id": bundle["workspace_session_id"]}
            refusals = {REFUSAL.rstrip("。.!"), "没有找到相关记忆", "没有找到足够的证据", "No relevant memory found"}
            for claim in unresolved:
                if claim["text"].rstrip("。.!") not in refusals:
                    refs = [key for key in claim["evidence_ids"]
                            if eligible[key]["session_id"] == bundle["workspace_session_id"]]
                    provisional.append(self.record_provisional_claim(task_id, claim["text"], refs, **owner))
        text = " ".join(row["text"] for row in approved)
        response = (data["final_answer"] if approved and not unresolved and "claims" not in data else text) or refusal
        safe = {"evidence_synthesis": text, "anti_intuitive_override": bool(approved and data["anti_intuitive_override"]),
                "rejected_noise_ids": [item for item in data["rejected_noise_ids"] if f"fact:{item}" in eligible]}
        return {**bundle, "response": response, "reasoning": safe, "llm_fallback": used_fallback,
                "claims": approved, "provisional_item_ids": provisional, "abstained": not approved}

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



def _epistemic_scope(source, plain):
    if source["role"] == "tool":
        return "experience"
    details = source.get("chronicle")
    if not details:
        return "world"
    if (details.get("speaker") or "").casefold() in {"assistant", "agent"}:
        return "experience"
    if re.search(r"\b(?:in my opinion|I believe|I think|perhaps)\b|我觉得|我认为", plain, re.I):
        return "opinion"
    return "world"


def _compact_evidence(query, fact):
    body = strip_step_tags(fact["text"])
    speaker = fact.get("speaker")
    occurrence = date_iso(fact.get("occurrence_start"))
    date = occurrence or (date_iso(fact.get("mention_time")) if speaker else None)
    prefix = f"[{' | '.join(part for part in (date, speaker) if part)}] " if date or speaker else ""
    if speaker:
        body = re.sub(r"^" + re.escape(speaker) + r":\s*", "", body, count=1, flags=re.I)
    body = _excerpt(body, competition_terms(query), 76 - estimate_tokens(prefix))
    return {**fact, "text": prefix + body}


def _valid_operations(data):
    operations = data.get("write_operations")
    string_fields = ("target_block_id", "target_line_id", "formatted_line", "topic", "attribute", "value", "operation_id", "event_time_iso")
    list_fields = ("z_axis_keys", "entity_z_keys", "category_z_keys", "tags")
    return isinstance(operations, list) and all(
        isinstance(op, dict) and isinstance(op.get("action"), str) and op["action"] in ACTIONS
        and all(isinstance(op.get(field, ""), str) for field in string_fields)
        and all(_string_list(op.get(field, [])) for field in list_fields)
        and ("source_turn_ids" not in op or isinstance(op["source_turn_ids"], list) and op["source_turn_ids"]
             and all(type(turn_id) is int and turn_id > 0 for turn_id in op["source_turn_ids"]))
        and isinstance(op.get("is_conflict_or_override", False), bool)
        and (op["action"] in {"DELETE_LINE", "NOOP"} or bool(line_body(op.get("formatted_line", ""))))
        for op in operations)


def _string_list(value):
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _valid_reason(data):
    claims = data.get("claims", [])
    return (isinstance(data.get("final_answer"), str) and isinstance(data.get("evidence_synthesis"), str)
            and isinstance(data.get("anti_intuitive_override"), bool) and _string_list(data.get("rejected_noise_ids"))
            and isinstance(claims, list) and len(claims) <= 32 and all(
                isinstance(claim, dict) and isinstance(claim.get("text"), str) and claim["text"].strip()
                and _string_list(claim.get("evidence_ids")) for claim in claims))


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


def baseline_rrf(candidates):
    return sorted((fact for fact in candidates if fact.get("baseline_rrf_score", fact["rrf_score"]) > 0),
                  key=lambda fact: (-fact.get("baseline_rrf_score", fact["rrf_score"]), fact["id"]))


def _fit_context(ranked, top_k, token_budget):
    selected = []
    used = 0
    budget = 4 * token_budget
    for fact in ranked:
        if len(selected) >= top_k:
            break
        candidate = used + bool(used) + token_units(fact["text"])
        if candidate <= budget:
            selected.append(fact)
            used = candidate
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
