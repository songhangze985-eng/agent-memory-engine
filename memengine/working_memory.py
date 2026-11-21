"""Task lifecycle and explicit promotion through the fact writer."""

from math import isfinite
from uuid import uuid4

from memengine.storage.base import RecallFilter
from memengine.verification import supports_claim


_KINDS = {"tool_evidence", "temp_fact", "provisional_claim", "open_question"}
_VERIFIED = {"tool_verified", "user_confirmed"}
_MISSING = "task not found or expired"


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value.strip()


def _key(scope, task_id):
    key = (*scope, task_id)
    if any(not isinstance(value, str) or not value.strip() for value in key):
        raise ValueError("a task requires a user, agent, session and task ID")
    return key


def _reference(ref):
    if type(ref) is int:
        ref = {"kind": "turn", "id": ref}
    if isinstance(ref, str):
        kind, separator, ident = ref.partition(":")
        ref = {"kind": kind, "id": int(ident) if kind == "turn" and ident.isascii() and ident.isdecimal() else ident}
        if not separator:
            raise ValueError("source references require a kind and ID")
    if not isinstance(ref, dict) or set(ref) != {"kind", "id"}:
        raise ValueError("source references require kind and id")
    kind, ident = ref["kind"], ref["id"]
    if kind == "turn" and type(ident) is int and ident > 0:
        return {"kind": kind, "id": ident}
    if kind in ("fact", "tool_call", "item"):
        return {"kind": kind, "id": _text(ident, "reference id")}
    raise ValueError("invalid source reference")


class TaskWorkspace:
    def start_task(self, goal, task_id=None, ttl_sec=86400, *, user_id="default", agent_id="default", session_id="default"):
        now = self._now()
        expires = None
        if ttl_sec is not None:
            if type(ttl_sec) not in (int, float) or not isfinite(ttl_sec) or ttl_sec <= 0:
                raise ValueError("ttl_sec must be positive and finite")
            expires = now + ttl_sec
        key = _key((user_id, agent_id, session_id), uuid4().hex if task_id is None else task_id)
        with self.store.transaction():
            if not self.store.create_task(key, _text(goal, "goal"), now, expires):
                raise ValueError("task ID already exists in this session")
        return key[3]

    def _task(self, key, now, *, allowed=None):
        task = self.store.task(key)
        if task is None or (task["expires_at"] is not None and now >= task["expires_at"]):
            raise ValueError(_MISSING)
        if allowed is not None and task["status"] not in allowed:
            raise ValueError(f"task is {task['status']}")
        return task

    def get_task_workspace(self, task_id, *, user_id="default", agent_id="default", session_id="default", limit=None):
        key = _key((user_id, agent_id, session_id), task_id)
        with self.store.read_snapshot():
            now = self._now()
            task = self._task(key, now)
            return {**task, "items": self.store.task_items(key, limit=limit, as_of=now)}

    def record_task_evidence(self, task_id, content, item_type, source_refs=(), *,
                             user_id="default", agent_id="default", session_id="default"):
        if item_type not in _KINDS:
            raise ValueError("invalid work item type")
        key = _key((user_id, agent_id, session_id), task_id)
        item_id, now = uuid4().hex, self._now()
        refs = [_reference(ref) for ref in source_refs]
        with self.store.transaction():
            task = self._task(key, now, allowed={"active"})
            for ref in refs:
                self._source(key, ref)
            self.store.insert_task_item(key, item_id, item_type, _text(content, "content"), refs, now)
            self.store.set_task_state(key, "active", task["checkpoint"], now)
        return item_id

    def record_provisional_claim(self, task_id, claim, source_refs=(), status="unverified", **scope):
        if status != "unverified":
            raise ValueError("new claims must be unverified")
        return self.record_task_evidence(task_id, claim, "provisional_claim", source_refs, **scope)

    def _change_task(self, task_id, status, allowed, checkpoint=None, *,
                     user_id="default", agent_id="default", session_id="default"):
        key = _key((user_id, agent_id, session_id), task_id)
        now = self._now()
        with self.store.transaction():
            task = self._task(key, now, allowed=allowed)
            state = task["checkpoint"] if checkpoint is None else checkpoint
            self.store.set_task_state(key, status, state, now)
        return state

    def checkpoint_task(self, task_id, state_dict, **scope):
        if not isinstance(state_dict, dict):
            raise ValueError("checkpoint must be a dictionary")
        return self._change_task(task_id, "active", {"active"}, state_dict, **scope)

    def pause_task(self, task_id, **scope):
        return self._change_task(task_id, "paused", {"active"}, **scope)

    def resume_task(self, task_id, **scope):
        return self._change_task(task_id, "active", {"active", "paused"}, **scope)

    def _source(self, key, ref):
        scope, ident = key[:3], ref["id"]
        kind = ref["kind"]
        if kind == "turn":
            rows = self.store.get_turn_window([ident], 0, scope[0], scope[1], session_id=scope[2])
        elif kind == "fact":
            rows = self.store.facts_by_ids([ident], scope=scope)
        elif kind == "tool_call":
            rows = [row for row in self.store.tool_calls(scope, call_id=ident)
                    if row["task_id"] in (None, key[3])]
        else:
            row = self.store.task_item(key, ident)
            rows = [row] if row is not None else []
        if not rows:
            raise ValueError("source is outside the task scope or missing")
        return rows[0]

    def _corroboration(self, key, item, status, ref, now):
        source = self._source(key, ref)
        kind = ref["kind"]
        if status == "user_confirmed" and kind == "turn" and source["role"] == "user":
            if (not self.store.inactive_temporary_sources([source["turn_id"]], scope=key[:3], as_of=now, include_superseded=True)
                    and supports_claim(item["content"], source["text"])):
                return [source["turn_id"]], None, None
        if status == "tool_verified" and kind == "tool_call" and source["success"]:
            scope = key[:3]
            raw = self.store.get_turn_window(source["source_turn_ids"], 0, scope[0], scope[1], session_id=scope[2])
            blocked = self.store.inactive_temporary_sources(source["source_turn_ids"], scope=scope, as_of=now,
                                                            include_superseded=True)
            facts = self.store.facts_by_ids(source["derived_from_ids"], scope=scope)
            current = RecallFilter(as_of=now)
            for turn in raw:
                if turn["role"] == "tool" and turn["turn_id"] not in blocked and supports_claim(item["content"], turn["text"]):
                    return [turn["turn_id"]], source["tool_call_id"], None
            for fact in facts:
                if (fact["status"] == "active" and current.matches(fact)
                        and fact["epistemic_scope"] in {"world", "experience"}
                        and supports_claim(item["content"], fact["plain"])):
                    return [], source["tool_call_id"], fact["id"]
        raise ValueError("verification requires supporting user confirmation or successful tool evidence")

    def verify_task_item(self, task_id, item_id, status, verifier_ref=None, *,
                         user_id="default", agent_id="default", session_id="default"):
        if status not in (_VERIFIED | {"rejected"}):
            raise ValueError("invalid verification status")
        key = _key((user_id, agent_id, session_id), task_id)
        now = self._now()
        with self.store.transaction():
            task = self._task(key, now, allowed={"active"})
            item = self._source(key, {"kind": "item", "id": item_id})
            ref = _reference(verifier_ref) if verifier_ref is not None or status in _VERIFIED else None
            turns, support_fact = [], None
            if status in _VERIFIED:
                turns, _, support_fact = self._corroboration(key, item, status, ref, now)
            self.store.verify_task_item(key, item_id, status, ref, next(iter(turns), None), support_fact)
            self.store.set_task_state(key, "active", task["checkpoint"], now)

    def commit_task(self, task_id, promote_verified_only=True, *,
                    user_id="default", agent_id="default", session_id="default"):
        if promote_verified_only is not True:
            raise ValueError("only verified items can be promoted")
        scope = (user_id, agent_id, session_id)
        key, now = _key(scope, task_id), self._now()
        with self.store.transaction():
            task = self._task(key, now, allowed={"active", "committed"})
            items = self.store.task_items(key)
            promoted = {item["item_id"]: item["promoted_fact_id"] for item in items if item["promoted_fact_id"]}
            if task["status"] == "committed":
                return promoted
            for item in items:
                status = item["verification_status"]
                if status not in _VERIFIED or item["item_type"] not in {"provisional_claim", "tool_evidence"}:
                    continue
                turns, tool, support_fact = self._corroboration(key, item, status, item["verifier_ref"], now)
                if support_fact is not None:
                    deadline = self.store.evidence_fact_states([support_fact], scope=scope)[support_fact]["expires_at"]
                else:
                    deadline = self.store.temporary_source_deadline(turns, scope=scope, as_of=now)
                fact_id = self.remember(item["content"], user_id=user_id, agent_id=agent_id, session_id=session_id,
                                        source_turn_ids=turns, tool_call_id=tool, timestamp=now, expires_at=deadline,
                                        _promotion_task=task_id)
                self.store.commit_buffered_turns(turns, scope=scope)
                self.store.promote_task_item(key, item["item_id"], fact_id)
                promoted[item["item_id"]] = fact_id
            self.store.set_task_state(key, "committed", task["checkpoint"], now)
            return promoted
