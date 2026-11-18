"""Cognitive memory engine: add / search / answer over the five-step pipeline."""

from __future__ import annotations

import time
from typing import Callable

from memengine.distance import asymptotic_distance, compute_raw_dist
from memengine.extract import english_entity_keys, extract_facts, is_greeting
from memengine.llm import MockLLMClient, loads_json_object
from memengine.mcq import end_with_choice
from memengine.notebook import apply_write_op, format_fact_line, render_block
from memengine.policy import decide_layout, decide_origin, plan_answer
from memengine.prompts import render_prompt
from memengine.router import CASUAL, TASK, entity_names, match_entities, route_query
from memengine.store import Store
from memengine.textutil import char_bigrams, estimate_tokens, jaccard
from memengine.timeline import TimelineBook
from memengine.zaxis import collapse_and_scan_z_axis, expand_entities, order_far_to_near

CASUAL_REPLY = "好的，我在。需要我帮你处理具体事情时直接说就行。"
NEWS_TOPIC = "时事杂讯"
NEWS_KEY = "话题:时事杂讯"
_NOISE_MIN = 16


class CognitiveMemoryEngine:
    def __init__(
        self,
        db_path: str = ":memory:",
        llm=None,
        delta_t_buffer: float = 10.0,
        epsilon: float = 0.05,
        enable_timeline_buffer: bool = True,
        enable_casual_bypass: bool = True,
        clock: Callable[[], float] | None = None,
    ):
        self.store = Store(db_path)
        self.llm = llm if llm is not None else MockLLMClient()
        self.epsilon = float(epsilon)
        self.enable_casual_bypass = bool(enable_casual_bypass)
        self.timeline = TimelineBook(delta_t_buffer, enabled=enable_timeline_buffer)
        self.timeline.layer1 = self.store.load_timeline(1)
        self.timeline.layer2 = self.store.load_timeline(2)
        self.timeline.events = int(self.store.get_meta("timeline_events") or "0")
        self.origin = self._load_origin()
        stored_radius = self.store.get_meta("radius")
        self.radius = float(stored_radius) if stored_radius is not None else 0.45
        self.pending: list[dict] = []
        self.fallback_count = 0
        self._clock = clock

    def _now(self) -> float:
        if self._clock is not None:
            return float(self._clock())
        return time.time()

    def set_time(self, timestamp: float) -> None:
        stamp = float(timestamp)

        def _fixed() -> float:
            return stamp

        self._clock = _fixed

    def _load_origin(self) -> dict | None:
        raw = self.store.get_meta("origin")
        if not raw:
            return None
        import json

        return json.loads(raw)

    def _save_control_state(self) -> None:
        import json

        self.store.set_meta("origin", json.dumps(self.origin or {}, ensure_ascii=False))
        self.store.set_meta("radius", str(self.radius))
        self.store.replace_timeline(1, self.timeline.layer1)
        self.store.replace_timeline(2, self.timeline.layer2)
        self.store.set_meta("timeline_events", str(self.timeline.events))

    def add(self, text: str, timestamp: float | None = None, sync: bool = True) -> dict:
        """Ingest one turn. Writes run inline unless ``sync`` is false."""
        if timestamp is not None:
            self.set_time(timestamp)
        now = self._now()
        self.store.append_ingest(text or "", now)
        facts = extract_facts(text or "")
        if not facts and not self._using_mock() and not is_greeting(text or ""):
            facts = self._llm_extract(text or "")
        job = {"text": text or "", "facts": facts, "ts": now, "noise": False, "ops": []}
        if facts:
            candidates = self._candidate_blocks(facts)
            prompt = render_prompt(
                "layout",
                candidate_2d_blocks=candidates,
                extracted_facts=facts,
                timestamp=str(now),
            )
            data, _used_fallback = self._call_json(
                prompt,
                lambda: decide_layout(candidates, facts, now),
                required=("write_operations",),
            )
            job["ops"] = list(data.get("write_operations") or [])
        elif not is_greeting(text or "") and len((text or "").strip()) >= _NOISE_MIN:
            job["noise"] = True
        if sync:
            applied = self._apply_job(job)
            self.store.commit()
            return {"facts": facts, "operations": applied, "queued": False}
        self.pending.append(job)
        self.store.commit()
        return {"facts": facts, "operations": [], "queued": True}

    def drain(self) -> list[dict]:
        applied: list[dict] = []
        while self.pending:
            applied.extend(self._apply_job(self.pending.pop(0)))
        self.store.commit()
        return applied

    def search(self, query: str, timestamp: float | None = None) -> dict:
        """Route and retrieve. Task path moves the origin; it does not run system 2."""
        if timestamp is not None:
            self.set_time(timestamp)
        started = time.perf_counter()
        index = self.store.z_index()
        route = route_query(query, index.keys()) if self.enable_casual_bypass else TASK
        history_tokens = estimate_tokens(self.store.history_text())
        if route == CASUAL:
            hits = self._hash_lookup(query, index)
            elapsed = (time.perf_counter() - started) * 1000
            return {
                "route": CASUAL,
                "hits": hits,
                "outer": [],
                "inner": [],
                "origin": self.origin,
                "radius": self.radius,
                "matched_entities": [],
                "scan_far_to_near": [],
                "retrieval_ms": elapsed,
                "context_tokens": estimate_tokens("\n".join(hit["text"] for hit in hits)),
                "history_tokens": history_tokens,
            }
        direct = match_entities(query, index.keys())
        expanded = expand_entities(index, direct, hops=2)
        scan_keys = sorted(set(direct) | set(expanded))
        notebook_index = self._notebook_index()
        origin_prompt = render_prompt(
            "origin",
            current_origin_json=self.origin or {},
            notebook_2d_index=notebook_index,
            matched_z_entities=scan_keys,
            user_query=query,
        )
        decision, _used_fallback = self._call_json(
            origin_prompt,
            lambda: decide_origin(self.origin, notebook_index, scan_keys, query),
            required=("new_origin", "poison_circle"),
        )
        if not _valid_origin(decision):
            self.fallback_count += 1
            decision = decide_origin(self.origin, notebook_index, scan_keys, query)
        new_origin = decision["new_origin"]
        circle = decision["poison_circle"]
        radius = float(circle.get("radius_r", 0.45))
        self.radius = min(1.0, max(self.epsilon, radius))
        self.origin = {
            "x_topic": str(new_origin.get("x_topic") or "未归类"),
            "y_subtopic": str(new_origin.get("y_subtopic") or "总览"),
            "decision_anchor": str(new_origin.get("decision_anchor") or query[:80]),
        }
        now = self._now()
        self.timeline.record(1, {"origin": self.origin}, now)
        far_to_near = order_far_to_near(circle.get("z_axis_scan_order_far_to_near") or [], scan_keys)
        projected = {}
        for key in far_to_near:
            projected[key] = []
            for fact in index.get(key, []):
                item = dict(fact)
                item["raw_dist"] = compute_raw_dist(item, self.origin)
                projected[key].append(item)
        # The spec function reverses its input. Pass near-to-far so the scan is far-to-near.
        collapsed = collapse_and_scan_z_axis(
            projected,
            list(reversed(far_to_near)),
            self.radius,
            self.epsilon,
        )
        outer, inner = _split_zones(collapsed)
        outer = _rank_by_overlap(outer, query)
        inner = _rank_by_overlap(inner, query)
        if _is_time_travel(query):
            line_ids = []
            seen = set()
            for key in far_to_near:
                for fact in index.get(key, []):
                    if fact["id"] not in seen:
                        seen.add(fact["id"])
                        line_ids.append(fact["id"])
            for version in self.store.historical_versions(line_ids):
                version["raw_dist"] = max(0.8, float(version.get("raw_dist") or 0.8))
                version["d_asymp"] = asymptotic_distance(version["raw_dist"], self.epsilon)
                version["zone"] = "OUTER_ANTI_INTUITIVE"
                outer.append(version)
        context = _context_blob(outer, inner)
        self._save_control_state()
        self.store.commit()
        elapsed = (time.perf_counter() - started) * 1000
        return {
            "route": TASK,
            "hits": [],
            "outer": outer,
            "inner": inner,
            "origin": self.origin,
            "radius": self.radius,
            "matched_entities": direct,
            "scan_far_to_near": far_to_near,
            "retrieval_ms": elapsed,
            "context_tokens": estimate_tokens(context),
            "history_tokens": history_tokens,
            "context": context,
        }

    def answer(self, query: str, timestamp: float | None = None) -> dict:
        """Full reply. System 2 is read-only: its JSON cannot change memory or the radius."""
        retrieved = self.search(query, timestamp=timestamp)
        if retrieved["route"] == CASUAL:
            if retrieved["hits"]:
                response = "。".join(hit["text"] for hit in retrieved["hits"][:3])
            else:
                response = CASUAL_REPLY
            return {**retrieved, "response": response, "reasoning": None, "llm_fallback": False}
        prompt = render_prompt(
            "reason",
            new_origin_json=retrieved["origin"] or {},
            outer_anti_intuitive_facts=retrieved["outer"],
            inner_intuitive_facts=retrieved["inner"],
            user_query=query,
        )
        fallback = lambda: plan_answer(
            retrieved["origin"],
            retrieved["outer"],
            retrieved["inner"],
            query,
        )
        data, used_fallback = self._call_json(
            prompt,
            fallback,
            required=("final_response", "reasoning_trace"),
        )
        trace = data.get("reasoning_trace") or {}
        if not isinstance(trace, dict):
            trace = {}
        safe_trace = {
            "anti_intuitive_overrides": list(trace.get("anti_intuitive_overrides") or []),
            "rejected_noise_ids": list(trace.get("rejected_noise_ids") or []),
            "verification_status": str(trace.get("verification_status") or "PARTIAL_INFO"),
        }
        public = {key: retrieved[key] for key in retrieved if key != "context"}
        reply = end_with_choice(query, str(data.get("final_response") or ""))
        return {
            **public,
            "response": reply,
            "reasoning": safe_trace,
            "llm_fallback": used_fallback,
        }

    def _using_mock(self) -> bool:
        return isinstance(self.llm, MockLLMClient)

    def _llm_extract(self, text: str) -> list[dict]:
        """Minimal fact list for Prompt 3. Used only with a live client."""
        prompt = (
            "Extract objective facts from the utterance. "
            'Return JSON {"facts":[{"text":"...","zkeys":["person:Name","entity:token"],'
            '"attribute":"...","value":"...","topic":"dialogue","isolated":true}]}. '
            'If there is no objective fact, return {"facts":[]}.\n'
            f"Utterance:\n{text[:4000]}"
        )
        try:
            data = loads_json_object(self.llm.complete(prompt))
        except Exception:
            return []
        facts = []
        for item in data.get("facts") or []:
            if not isinstance(item, dict) or not str(item.get("text") or "").strip():
                continue
            zkeys = [str(key) for key in (item.get("zkeys") or []) if key]
            if not zkeys:
                zkeys = english_entity_keys(str(item.get("text")))
            facts.append(
                {
                    "text": str(item.get("text")).strip(),
                    "zkeys": zkeys,
                    "attribute": str(item.get("attribute") or "fact"),
                    "value": str(item.get("value") or ""),
                    "topic": str(item.get("topic") or "dialogue"),
                    "isolated": bool(item.get("isolated", True)),
                    "is_greeting": False,
                }
            )
        return facts

    def timeline_stats(self) -> dict:
        reduction = self.timeline.reduction()
        return {
            "events": self.timeline.events,
            "nodes": self.timeline.nodes,
            "layer1": len(self.timeline.layer1),
            "layer2": len(self.timeline.layer2),
            "reduction": reduction,
        }

    def render(self) -> str:
        chunks = []
        for block in self.store.blocks_for_prompt():
            body = render_block(
                [{"formatted": line["formatted"], "isolated": line["isolated"]} for line in block["lines"]]
            )
            if body:
                chunks.append(f"# {block['topic']}\n{body}")
        return "\n\n".join(chunks)

    def _apply_job(self, job: dict) -> list[dict]:
        applied: list[dict] = []
        alias: dict[str, str] = {}
        if job.get("noise"):
            existing = self.store.block_by_topic(NEWS_TOPIC)
            op = {
                "action": "APPEND_ADJACENT_LINE",
                "target_block_id": existing["id"] if existing else "NEW",
                "target_line_id": "",
                "formatted_line": format_fact_line(job["ts"], job["text"].strip()),
                "z_axis_keys": [NEWS_KEY],
                "surprise_score": 0.12,
            }
            apply_write_op(
                self.store,
                op,
                topic=NEWS_TOPIC,
                attribute="",
                value="",
                plain=job["text"].strip(),
                timestamp=job["ts"],
                alias=alias,
            )
            applied.append(op)
        else:
            facts = job.get("facts") or []
            ops = job.get("ops") or []
            for index, fact in enumerate(facts):
                op = ops[index] if index < len(ops) else {"action": "NOOP"}
                if not isinstance(op, dict):
                    op = {"action": "NOOP"}
                action = op.get("action")
                if action not in {"OVERWRITE_SAME_LINE", "APPEND_ADJACENT_LINE", "INSERT_ISOLATED_LINE", "NOOP"}:
                    op = {**op, "action": "NOOP"}
                apply_write_op(
                    self.store,
                    op,
                    topic=fact.get("topic") or "杂项",
                    attribute=fact.get("attribute") or "",
                    value=fact.get("value") or "",
                    plain=fact.get("text") or "",
                    timestamp=job["ts"],
                    alias=alias,
                )
                applied.append(op)
        if any(op.get("action") != "NOOP" for op in applied):
            inner, outer = self._zone_counts()
            self.timeline.record(2, {"in_cnt": inner, "out_cnt": outer}, job["ts"])
            self._save_control_state()
        return applied

    def _zone_counts(self) -> tuple[int, int]:
        seen = set()
        inner = outer = 0
        for facts in self.store.z_index().values():
            for fact in facts:
                if fact["id"] in seen:
                    continue
                seen.add(fact["id"])
                raw = compute_raw_dist(fact, self.origin)
                distance = asymptotic_distance(raw, self.epsilon)
                if distance <= (self.radius or 0.45):
                    inner += 1
                else:
                    outer += 1
        return inner, outer

    def _candidate_blocks(self, facts: list[dict]) -> list[dict]:
        blocks = self.store.blocks_for_prompt()
        if not blocks:
            return []
        fact_keys = set()
        topics = set()
        fact_text = []
        for fact in facts:
            topics.add(fact.get("topic") or "")
            fact_keys.update(fact.get("zkeys") or [])
            fact_text.append(fact.get("text") or "")
        blob = char_bigrams(" ".join(fact_text))

        def pinned(block: dict) -> bool:
            if block["topic"] in topics:
                return True
            for line in block["lines"]:
                if not line.get("attribute"):
                    continue
                if any(line["attribute"] == fact.get("attribute") and fact_keys.intersection(line.get("zkeys") or []) for fact in facts):
                    return True
            return False

        pinned_blocks = [block for block in blocks if pinned(block)]
        ranked = sorted(
            blocks,
            key=lambda block: jaccard(
                blob,
                char_bigrams(block["topic"] + " " + " ".join(line["text"] for line in block["lines"][-3:])),
            ),
            reverse=True,
        )
        chosen = []
        seen = set()
        for block in pinned_blocks + ranked:
            if block["id"] in seen:
                continue
            seen.add(block["id"])
            chosen.append(_trim_block(block, facts))
            if len(chosen) >= 3 and len(chosen) >= len(pinned_blocks):
                break
        return chosen

    def _notebook_index(self) -> list[dict]:
        index = []
        for block in self.store.blocks_for_prompt():
            preview_lines = block["lines"][-3:]
            preview = " ".join(line["text"] for line in preview_lines)
            attr = ""
            for line in reversed(block["lines"]):
                if line.get("attribute"):
                    attr = line["attribute"]
                    break
            index.append(
                {
                    "block_id": block["id"],
                    "topic": block["topic"],
                    "preview": preview[:180],
                    "preview_attr": attr,
                }
            )
        return index

    def _hash_lookup(self, query: str, index: dict) -> list[dict]:
        hits = []
        seen = set()
        for key, facts in index.items():
            name = entity_names(key)
            if name and name in query:
                for fact in facts:
                    if fact["id"] in seen:
                        continue
                    seen.add(fact["id"])
                    hits.append(fact)
        return hits

    def _call_json(self, prompt: str, fallback, required: tuple[str, ...]) -> tuple[dict, bool]:
        try:
            data = loads_json_object(self.llm.complete(prompt))
            if any(key not in data for key in required):
                raise ValueError("missing keys")
            return data, False
        except Exception:
            self.fallback_count += 1
            return fallback(), True


def _valid_origin(data: dict) -> bool:
    origin = data.get("new_origin")
    circle = data.get("poison_circle")
    return isinstance(origin, dict) and isinstance(circle, dict) and "radius_r" in circle


def _split_zones(collapsed: list[dict]) -> tuple[list[dict], list[dict]]:
    outer, inner = [], []
    seen = set()
    for bucket in collapsed:
        for fact in bucket["facts"]:
            if fact["id"] in seen:
                continue
            seen.add(fact["id"])
            if fact.get("zone") == "OUTER_ANTI_INTUITIVE":
                outer.append(fact)
            else:
                inner.append(fact)
    return outer, inner


def _rank_by_overlap(facts: list[dict], query: str) -> list[dict]:
    if len(facts) < 2:
        return facts
    query_grams = char_bigrams(query)
    return sorted(
        facts,
        key=lambda fact: jaccard(query_grams, char_bigrams(fact.get("text") or "")),
        reverse=True,
    )


def _context_blob(outer: list[dict], inner: list[dict]) -> str:
    chunks = []
    for fact in outer + inner:
        chunks.append(fact.get("text") or "")
    return "\n".join(chunks)


def _is_time_travel(query: str) -> bool:
    import re

    if re.search(r"第[0-9一二两三四五六七八九十]+次", query or "") and "之前" in (query or ""):
        return True
    return bool(re.search(r"最早|原本|以前|上次|之前", query or ""))


def _trim_block(block: dict, facts: list[dict]) -> dict:
    fact_attrs = {fact.get("attribute") for fact in facts if fact.get("attribute")}
    fact_keys = set()
    for fact in facts:
        fact_keys.update(fact.get("zkeys") or [])
    pinned = []
    rest = []
    for line in block["lines"]:
        if line.get("attribute") in fact_attrs and fact_keys.intersection(line.get("zkeys") or []):
            pinned.append(line)
        else:
            rest.append(line)
    shown = pinned + rest[-5:]
    # Preserve order and drop duplicates.
    seen = set()
    lines = []
    for line in shown:
        if line["id"] in seen:
            continue
        seen.add(line["id"])
        lines.append(line)
    lines.sort(key=lambda line: block["lines"].index(line) if line in block["lines"] else 0)
    return {"id": block["id"], "topic": block["topic"], "lines": lines}
