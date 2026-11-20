"""Host-scoped memory tools for Chat Completions, MCP and local agent loops."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from math import isfinite
import re

from memengine.extract import extract_facts
from memengine.store import RecallFilter

MEMORY_TYPES = ("fact", "observation", "procedural")
_TEXT = {"type": "string", "minLength": 1}
_TURN = {"type": "integer", "minimum": 1, "maximum": 2**63 - 1}
_TIME = {"type": ["number", "string", "null"], "description": "Unix seconds or an ISO-8601 date or datetime. UTC if no offset."}
_TOOLS = {
    "memory_search": ("Find relevant scoped memories. An empty list indicates no evidence passed the gate.", ("query",), {
        "query": _TEXT,
        "memory_types": {"type": ["array", "null"], "items": {"type": "string", "enum": list(MEMORY_TYPES)},
                         "minItems": 1, "maxItems": 3, "uniqueItems": True},
        "time_start": {**_TIME, "description": "Inclusive event time. " + _TIME["description"]},
        "time_end": {**_TIME, "description": "Exclusive event time. " + _TIME["description"]},
        "include_cold": {"type": "boolean", "default": False},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 40, "default": 10},
    }),
    "memory_drilldown": ("Read raw turns and their same-session neighbors.", ("turn_ids",), {
        "turn_ids": {"type": "array", "items": _TURN, "minItems": 1, "maxItems": 40},
        "window": {"type": "integer", "minimum": 0, "maximum": 10, "default": 1},
    }),
    "memory_history": ("Inspect all revisions for an exact entity or memory ID.", ("entity_or_memory_id",), {
        "entity_or_memory_id": _TEXT,
    }),
    "memory_remember": ("Write grounded content. Procedural text uses 'When <trigger>: <technique>' unless the host supplies a tool trace.", ("content",), {
        "content": _TEXT,
        "memory_type": {"type": "string", "enum": ["fact", "procedural"], "default": "fact"},
        "is_constraint": {"type": "boolean", "default": False},
        "source_turn_id": {**_TURN, "type": ["integer", "null"]},
    }),
    "memory_forget": ("Revoke one scoped fact chain by ID. Hard deletion purges the derived chain and its indexes.", ("memory_id",), {
        "memory_id": _TEXT,
        "hard": {"type": "boolean", "default": False},
        "reason": {"type": "string", "default": ""},
    }),
}


def _schema(name):
    _, required, properties = _TOOLS[name]
    return {"type": "object", "properties": deepcopy(properties), "required": list(required), "additionalProperties": False}


def _validate(value, schema, field):
    kinds = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
    kind = {bool: "boolean", int: "integer", float: "number", str: "string", list: "array", type(None): "null"}.get(type(value))
    if kind not in kinds and not (kind == "integer" and "number" in kinds):
        raise ValueError(f"invalid type for {field}")
    if value is None:
        return
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"invalid value for {field}")
    if kind in {"integer", "number"}:
        if kind == "number" and not isfinite(value):
            raise ValueError(f"{field} must be finite")
        if not schema.get("minimum", value) <= value <= schema.get("maximum", value):
            raise ValueError(f"{field} out of range")
    if kind == "string" and len(value.strip()) < schema.get("minLength", 0):
        raise ValueError(f"{field} must not be blank")
    if kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", len(value)):
            raise ValueError(f"{field} has an invalid number of items")
        for entry in value:
            _validate(entry, schema["items"], field)
        if schema.get("uniqueItems") and len(set(value)) != len(value):
            raise ValueError(f"duplicate items in {field}")


def _timestamp(value):
    if isinstance(value, str):
        try:
            text = value.strip()
            moment = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
        except ValueError as exc:
            raise ValueError("expected an ISO-8601 date or datetime") from exc
        return (moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)).timestamp()
    if value is not None:
        try:
            number = float(value)
        except OverflowError as exc:
            raise ValueError("event time must be finite") from exc
        if not isfinite(number):
            raise ValueError("event time must be finite")
        return number
    return None


def _summary(fact):
    return {"id": fact["id"], "memory_type": fact["memory_type"], "status": fact["status"], "event_time": fact["event_time_start"],
            "score": fact.get("final_score"), "source_turn_ids": fact["source_turn_ids"], "content": fact["plain"]}


class MemoryTools(list):
    """JSON-serializable Chat Completions definitions with a Py-only scoped dispatcher."""

    def __init__(self, engine, user_id, agent_id, session_id):
        super().__init__({"type": "function", "function": {"name": name, "description": definition[0], "parameters": _schema(name)}}
                         for name, definition in _TOOLS.items())
        self._engine = engine
        self._scope = (user_id, agent_id, session_id)

    def mcp_definitions(self):
        return [{"name": name, "description": definition[0], "inputSchema": _schema(name)} for name, definition in _TOOLS.items()]

    def dispatch_tool(self, name, arguments, *, tool_call_id=None):
        if not isinstance(name, str) or name not in _TOOLS:
            raise ValueError("unknown memory tool")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        _, required, properties = _TOOLS[name]
        if not isinstance(arguments, dict) or not set(required) <= arguments.keys() or arguments.keys() - properties.keys():
            raise ValueError("missing or unexpected tool arguments")
        for key, value in arguments.items():
            _validate(value, properties[key], key)
        handler = getattr(self, "_" + name)
        if name == "memory_remember":
            return handler(**arguments, tool_call_id=tool_call_id)
        return handler(**arguments)

    def _memory_search(self, query, memory_types=None, time_start=None, time_end=None, include_cold=False, top_k=10):
        start, end = _timestamp(time_start), _timestamp(time_end)
        if start is not None and end is not None and start >= end:
            raise ValueError("time_start must precede time_end")
        user_id, agent_id, session_id = self._scope
        result = self._engine.retrieve(query.strip(), user_id=user_id, agent_id=agent_id, session_id=session_id,
                                       read_only=True, include_cold=include_cold, top_k=top_k,
                                       filters=RecallFilter(tuple(memory_types or MEMORY_TYPES), start, end, session_id is not None))
        return [_summary(fact) for fact in self._visible(result["ranked"])]

    def _visible(self, facts):
        if self._scope[2] is None:
            return facts
        ids = sorted({turn_id for fact in facts for turn_id in fact["source_turn_ids"]})
        visible = {row["turn_id"] for row in self._memory_drilldown(ids, window=0)}
        return [fact for fact in facts if set(fact["source_turn_ids"]) <= visible]

    def _memory_drilldown(self, turn_ids, window=1):
        user_id, agent_id, session_id = self._scope
        return self._engine.store.get_turn_window(turn_ids, window, user_id, agent_id, by_id=True, session_id=session_id)

    def _memory_history(self, entity_or_memory_id):
        versions = self._engine.store.memory_history(entity_or_memory_id.strip(), scope=self._scope)
        return [{**_summary(fact), **{key: fact[key] for key in ("root_fact_id", "revision", "supersedes_id", "valid_until", "ingest_time")}}
                for fact in self._visible(versions)]

    def _memory_remember(self, content, memory_type="fact", is_constraint=False, source_turn_id=None, *, tool_call_id=None):
        user_id, agent_id, session_id = self._scope
        if source_turn_id is not None:
            anchors = self._memory_drilldown([source_turn_id], window=0)
            if not anchors:
                raise ValueError("source turn not found in the tool scope")
            session_id = anchors[0]["session_id"]
        elif tool_call_id is not None:
            calls = self._engine.store.tool_calls(self._scope, call_id=tool_call_id)
            if not calls:
                raise ValueError("registered tool trace not found in the tool scope")
            session_id = calls[0]["session_id"]
        else:
            raise ValueError("source_turn_id or trusted tool context is required")
        trigger = ""
        if memory_type == "procedural":
            rule = re.fullmatch(r"\s*When\s+(.+?):\s*(\S.*)", content, re.I | re.S)
            if rule:
                trigger, content = rule.groups()
        zkeys = list(dict.fromkeys(key for fact in extract_facts(content) for key in fact["zkeys"]))
        memory_id = self._engine.remember(content, memory_type=memory_type, is_constraint=is_constraint, zkeys=zkeys,
                                          source_turn_ids=() if source_turn_id is None else (source_turn_id,),
                                          tool_call_id=tool_call_id, trigger=trigger, user_id=user_id,
                                          agent_id=agent_id, session_id=session_id)
        return {"id": memory_id, "memory_type": memory_type}

    def _memory_forget(self, memory_id, hard=False, reason=""):
        user_id, agent_id, _ = self._scope
        memory_id = memory_id.strip()
        with self._engine.store.transaction():
            rows = self._engine.store.facts_by_ids([memory_id], scope=self._scope)
            deleted = bool(self._visible(rows)) and self._engine.forget(memory_id, hard=hard, user_id=user_id, agent_id=agent_id)
        return {"id": memory_id, "deleted": deleted, "hard": hard, "reason": reason}
