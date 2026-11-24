"""Single unified tool registry generating OpenAI tools, MCP, CLI, and HTTP adapters."""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    required: tuple[str, ...]
    http_method: str = "POST"
    http_path: str = ""


_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="memory_retain",
        description="Conversational ingest track (ungated): store raw turn in L0 + FTS5 and extract bi-temporal facts.",
        parameters={
            "text": {"type": "string", "minLength": 1},
            "role": {"type": "string", "default": "user"},
            "speaker": {"type": ["string", "null"]},
            "timestamp": {"type": ["number", "null"]},
            "mention_time": {"type": ["number", "string", "null"]},
            "idempotency_key": {"type": ["string", "null"]},
        },
        required=("text",),
        http_path="/v2/memory/retain",
    ),
    ToolSpec(
        name="memory_search",
        description="Unified 4-channel recall (vector + BM25[facts+L0 FTS5] + entity_graph + exact/version).",
        parameters={
            "query": {"type": "string", "minLength": 1},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            "reference_time": {"type": ["number", "null"]},
            "include_cold": {"type": "boolean", "default": False},
            "mode": {"type": "string", "enum": ["single_query", "reconstructive"], "default": "reconstructive"},
        },
        required=("query",),
        http_path="/v2/memory/search",
    ),
    ToolSpec(
        name="memory_build_context",
        description="Assemble bounded context pack with full or progressive disclosure and conversational/coding profile.",
        parameters={
            "query": {"type": "string", "minLength": 1},
            "token_budget": {"type": "integer", "minimum": 64, "maximum": 32768, "default": 2048},
            "disclosure": {"type": "string", "enum": ["full", "progressive"], "default": "full"},
            "profile": {"type": "string", "enum": ["conversational", "coding_agent"], "default": "conversational"},
            "reference_time": {"type": ["number", "null"]},
        },
        required=("query",),
        http_path="/v2/memory/context",
    ),
    ToolSpec(
        name="memory_answer",
        description="Answer a question strictly from verified memory evidence with abstention and deterministic calculation.",
        parameters={
            "query": {"type": "string", "minLength": 1},
            "token_budget": {"type": "integer", "minimum": 64, "maximum": 32768, "default": 2048},
            "reference_time": {"type": ["number", "null"]},
        },
        required=("query",),
        http_path="/v2/memory/answer",
    ),
    ToolSpec(
        name="memory_reflect",
        description="Synthesize an evidence-backed reflection and store it as a unified kind='belief' fact.",
        parameters={
            "query": {"type": "string", "minLength": 1},
            "token_budget": {"type": "integer", "minimum": 64, "maximum": 32768, "default": 2048},
            "reference_time": {"type": ["number", "null"]},
        },
        required=("query",),
        http_path="/v2/memory/reflect",
    ),
    ToolSpec(
        name="memory_write_rule",
        description="Procedural rule track: 3-question salience gate + [触发条件]/[动作]/[验证结果] state machine + optimistic lock.",
        parameters={
            "path": {"type": "string", "minLength": 1},
            "title": {"type": "string", "minLength": 1},
            "trigger_condition": {"type": "string", "minLength": 1},
            "action": {"type": "string", "minLength": 1},
            "verification_result": {"type": "string", "default": ""},
            "verification_status": {"type": "string", "enum": ["unverified", "verified"], "default": "verified"},
            "salience_tag": {"type": "string", "enum": ["normal", "preference", "taboo", "core_rule", "warning"], "default": "core_rule"},
            "always_active": {"type": "boolean", "default": False},
            "visible_in_context": {"type": "boolean", "default": False},
            "reusable_pattern": {"type": "boolean", "default": True},
            "future_pitfall_risk": {"type": "boolean", "default": True},
            "expected_version": {"type": ["integer", "null"]},
        },
        required=("path", "title", "trigger_condition", "action"),
        http_path="/v2/memory/rules",
    ),
    ToolSpec(
        name="memory_tree_outline",
        description="Derived VFS view: compact ls -R outline of path-bound rules (<=50-char index summaries).",
        parameters={"dir_path": {"type": "string", "default": "/"}},
        required=(),
        http_path="/v2/memory/tree",
    ),
    ToolSpec(
        name="memory_read_note",
        description="Derived VFS view: read full Markdown content of one path-bound unified fact.",
        parameters={"path": {"type": "string", "minLength": 1}},
        required=("path",),
        http_path="/v2/memory/note",
    ),
    ToolSpec(
        name="memory_compact_directory",
        description="Derived VFS view: merge fragmented rules in a directory into one GUIDE.md with batch rollback snapshot.",
        parameters={"dir_path": {"type": "string", "minLength": 1}},
        required=("dir_path",),
        http_path="/v2/memory/compact",
    ),
    ToolSpec(
        name="memory_rollback",
        description="Restore unified facts from pre-mutation snapshots by path, snapshot_id, or batch_id.",
        parameters={
            "path": {"type": ["string", "null"]},
            "snapshot_id": {"type": ["string", "null"]},
            "batch_id": {"type": ["string", "null"]},
        },
        required=(),
        http_path="/v2/memory/rollback",
    ),
    ToolSpec(
        name="memory_drilldown",
        description="Read raw L0 turns and their same-session neighbors.",
        parameters={
            "turn_ids": {"type": "array", "items": {"type": "integer", "minimum": 1}, "minItems": 1},
            "window": {"type": "integer", "minimum": 0, "maximum": 10, "default": 1},
        },
        required=("turn_ids",),
        http_path="/v2/memory/drilldown",
    ),
    ToolSpec(
        name="memory_history",
        description="Inspect full bi-temporal revision history for an entity or memory ID.",
        parameters={"entity_or_memory_id": {"type": "string", "minLength": 1}},
        required=("entity_or_memory_id",),
        http_path="/v2/memory/history",
    ),
    ToolSpec(
        name="memory_forget",
        description="Soft-delete or hard-purge a scoped memory chain by ID or exact query.",
        parameters={
            "target": {"type": ["string", "null"]},
            "query": {"type": ["string", "null"]},
            "hard": {"type": "boolean", "default": False},
        },
        required=(),
        http_path="/v2/memory/forget",
    ),
)

TOOL_SPECS_BY_NAME: dict[str, ToolSpec] = {spec.name: spec for spec in _TOOL_SPECS}


def tool_json_schema(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": deepcopy(spec.parameters),
        "required": list(spec.required),
        "additionalProperties": False,
    }


class UnifiedToolRegistry:
    """Single source of truth dispatching unified engine tools for HTTP, CLI, MCP, and OpenAI."""

    def __init__(self, engine, *, user_id: str = "default", agent_id: str = "default", session_id: str | None = None):
        self.engine = engine
        self.user_id = user_id
        self.agent_id = agent_id
        self.session_id = session_id

    @staticmethod
    def specs() -> tuple[ToolSpec, ...]:
        return _TOOL_SPECS

    def openai_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": tool_json_schema(spec),
                },
            }
            for spec in _TOOL_SPECS
        ]

    def mcp_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "inputSchema": tool_json_schema(spec),
            }
            for spec in _TOOL_SPECS
        ]

    def dispatch(self, name: str, arguments: dict[str, Any] | str | None = None) -> Any:
        if name not in TOOL_SPECS_BY_NAME:
            raise ValueError(f"unknown unified memory tool: {name!r}")
        if isinstance(arguments, str):
            args = json.loads(arguments) if arguments.strip() else {}
        elif arguments is None:
            args = {}
        elif isinstance(arguments, dict):
            args = dict(arguments)
        else:
            raise ValueError("arguments must be a dict or JSON object string")

        spec = TOOL_SPECS_BY_NAME[name]
        missing = [req for req in spec.required if req not in args or args[req] is None]
        if missing:
            raise ValueError(f"missing required arguments for {name}: {', '.join(missing)}")
        unexpected = set(args.keys()) - set(spec.parameters.keys())
        if unexpected:
            raise ValueError(f"unexpected arguments for {name}: {', '.join(sorted(unexpected))}")

        handler: Callable[..., Any] = getattr(self, f"_handle_{name}")
        return handler(**args)

    def _scope_session(self) -> str:
        return self.session_id or "default"

    def _handle_memory_retain(self, **kwargs):
        return self.engine.ingest_turn(
            user_id=self.user_id,
            agent_id=self.agent_id,
            session_id=self._scope_session(),
            **kwargs,
        )

    def _handle_memory_search(self, **kwargs):
        res = self.engine.recall(
            user_id=self.user_id,
            agent_id=self.agent_id,
            session_id=self.session_id,
            **kwargs,
        )
        return {
            "empty_recall": res["empty_recall"],
            "count": len(res["ranked"]),
            "items": [
                {
                    "id": f["id"],
                    "kind": f.get("kind", "world"),
                    "status": f.get("status", "active"),
                    "vfs_path": f.get("vfs_path"),
                    "content": f.get("plain", f.get("text", "")),
                    "source_turn_ids": f.get("source_turn_ids", []),
                    "event_time": f.get("event_time_start"),
                }
                for f in res["ranked"]
            ],
        }

    def _handle_memory_build_context(self, **kwargs):
        ctx = self.engine.build_context(
            user_id=self.user_id,
            agent_id=self.agent_id,
            session_id=self.session_id,
            **kwargs,
        )
        return {
            "system_prefix": ctx["system_prefix"],
            "dynamic_memory": ctx["dynamic_memory"],
            "disclosure": ctx["disclosure"],
            "profile": ctx["profile"],
            "empty_recall": ctx["empty_recall"],
            "token_breakdown": ctx["token_breakdown"],
        }

    def _handle_memory_answer(self, **kwargs):
        ans = self.engine.answer(
            user_id=self.user_id,
            agent_id=self.agent_id,
            session_id=self.session_id,
            **kwargs,
        )
        return {
            "response": ans["response"],
            "abstained": ans["abstained"],
            "partial": ans.get("partial", False),
            "claims": ans.get("claims", []),
        }

    def _handle_memory_reflect(self, **kwargs):
        ref = self.engine.reflect(
            user_id=self.user_id,
            agent_id=self.agent_id,
            session_id=self._scope_session(),
            **kwargs,
        )
        return {
            "response": ref["response"],
            "abstained": ref["abstained"],
            "belief_written": ref["belief_written"],
            "belief_fact_id": ref["belief_fact_id"],
        }

    def _handle_memory_write_rule(self, **kwargs):
        return self.engine.ingest_rule(
            user_id=self.user_id,
            agent_id=self.agent_id,
            session_id=self._scope_session(),
            **kwargs,
        )

    def _handle_memory_tree_outline(self, dir_path: str = "/"):
        return {"dir_path": dir_path, "outline": self.engine.tree_outline(dir_path, user_id=self.user_id, agent_id=self.agent_id)}

    def _handle_memory_read_note(self, path: str):
        note = self.engine.read_note(path, user_id=self.user_id, agent_id=self.agent_id)
        return {"found": note is not None, "note": note}

    def _handle_memory_compact_directory(self, dir_path: str):
        return self.engine.compact_directory(dir_path, user_id=self.user_id, agent_id=self.agent_id)

    def _handle_memory_rollback(self, path=None, snapshot_id=None, batch_id=None):
        return self.engine.rollback(user_id=self.user_id, agent_id=self.agent_id, path=path, snapshot_id=snapshot_id, batch_id=batch_id)

    def _handle_memory_drilldown(self, turn_ids: list[int], window: int = 1):
        return self.engine.drilldown(turn_ids, window=window, user_id=self.user_id, agent_id=self.agent_id, session_id=self.session_id)

    def _handle_memory_history(self, entity_or_memory_id: str):
        return self.engine.history(entity_or_memory_id, user_id=self.user_id, agent_id=self.agent_id, session_id=self.session_id)

    def _handle_memory_forget(self, target=None, query=None, hard: bool = False):
        deleted = self.engine.forget(target, query=query, hard=hard, user_id=self.user_id, agent_id=self.agent_id)
        return {"deleted": bool(deleted), "hard": hard}
