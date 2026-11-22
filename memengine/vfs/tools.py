"""Claude-style Virtual Filesystem tools for Chat Completions & MCP (host-bound tenant scope)."""
from __future__ import annotations

import json
from typing import Any

from memengine.vfs.models import LAYERS, SALIENCE_TAGS, VERIFICATION_STATUSES

_SPECS = (
    {
        "name": "mem_tree",
        "description": "Inspect the Markdown memory directory tree under a path prefix.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "/"},
                "layer": {"type": "string", "enum": sorted(LAYERS)},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "mem_ls",
        "description": "List immediate subdirectories and Markdown memory files under a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "/"},
                "layer": {"type": "string", "enum": sorted(LAYERS)},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "mem_read",
        "description": "Read a Markdown memory node, its metadata, optimistic-lock version, and L0 provenance IDs.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "mem_write",
        "description": "Create or overwrite a Markdown memory node with optional optimistic lock version check.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "title": {"type": "string"},
                "content": {"type": "string"},
                "summary": {"type": "string"},
                "layer": {"type": "string", "enum": sorted(LAYERS), "default": "L2"},
                "confidence_score": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.7},
                "verification_status": {"type": "string", "enum": sorted(VERIFICATION_STATUSES), "default": "unverified"},
                "salience_tag": {"type": "string", "enum": sorted(SALIENCE_TAGS), "default": "normal"},
                "source_transcript_ids": {"type": "array", "items": {"type": "string"}},
                "expected_version": {"type": "integer", "minimum": 0},
            },
            "required": ["path", "title", "content"],
            "additionalProperties": False,
        },
    },
    {
        "name": "mem_edit",
        "description": "Apply a targeted string patch to an existing Markdown memory node with optimistic locking.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_str": {"type": "string"},
                "new_str": {"type": "string"},
                "expected_version": {"type": "integer", "minimum": 1},
                "verification_status": {"type": "string", "enum": sorted(VERIFICATION_STATUSES)},
                "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
                "salience_tag": {"type": "string", "enum": sorted(SALIENCE_TAGS)},
                "source_transcript_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["path", "old_str", "new_str"],
            "additionalProperties": False,
        },
    },
    {
        "name": "mem_search",
        "description": "Search curated L2 Markdown notes (vector + BM25). Never scans raw L0 transcripts.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "path_prefix": {"type": "string"},
                "verified_only": {"type": "boolean", "default": False},
                "salience_tags": {"type": "array", "items": {"type": "string", "enum": sorted(SALIENCE_TAGS)}},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 30, "default": 8},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "mem_grep",
        "description": "Regex/text grep over Markdown notes under a directory prefix.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path_prefix": {"type": "string", "default": "/"},
                "layer": {"type": "string", "enum": sorted(LAYERS)},
                "max_matches": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    },
    {
        "name": "mem_read_transcripts",
        "description": "Lazy provenance drilldown into L0 raw transcript chunks by transcript IDs.",
        "parameters": {
            "type": "object",
            "properties": {"transcript_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["transcript_ids"],
            "additionalProperties": False,
        },
    },
    {
        "name": "mem_rollback",
        "description": "Rollback a single Markdown node to a prior snapshot or revert an entire dream_batch_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "snapshot_id": {"type": "string"},
                "dream_batch_id": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
)

_BY_NAME = {spec["name"]: spec for spec in _SPECS}


class VFSTools(list):
    """Host-scoped OpenAI Chat Completions tool list + MCP exporter + dispatcher."""

    def __init__(self, engine, *, user_id: str = "default", agent_id: str = "default"):
        super().__init__({"type": "function", "function": spec} for spec in _SPECS)
        self.engine = engine
        self.user_id = user_id
        self.agent_id = agent_id

    def mcp_definitions(self) -> list[dict]:
        return [
            {"name": spec["name"], "description": spec["description"], "inputSchema": spec["parameters"]}
            for spec in _SPECS
        ]

    def dispatch_tool(self, name: str, arguments: str | dict[str, Any]) -> Any:
        if name not in _BY_NAME:
            raise ValueError(f"unknown VFS tool: {name}")
        raw = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
        allowed = set(_BY_NAME[name]["parameters"]["properties"])
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unexpected tool arguments for {name}: {sorted(unknown)}")
        scope = {"user_id": self.user_id, "agent_id": self.agent_id}
        if name == "mem_tree":
            return self.engine.tree(path=raw.get("path", "/"), layer=raw.get("layer"), **scope)
        if name == "mem_ls":
            return self.engine.ls(path=raw.get("path", "/"), layer=raw.get("layer"), **scope)
        if name == "mem_read":
            return self.engine.read(raw["path"], **scope)
        if name == "mem_write":
            return self.engine.write(
                raw["path"],
                title=raw["title"],
                content=raw["content"],
                summary=raw.get("summary", ""),
                layer=raw.get("layer", "L2"),
                confidence_score=raw.get("confidence_score", 0.7),
                verification_status=raw.get("verification_status", "unverified"),
                salience_tag=raw.get("salience_tag", "normal"),
                source_transcript_ids=raw.get("source_transcript_ids", ()),
                expected_version=raw.get("expected_version"),
                **scope,
            )
        if name == "mem_edit":
            return self.engine.edit(
                raw["path"],
                old_str=raw["old_str"],
                new_str=raw["new_str"],
                expected_version=raw.get("expected_version"),
                verification_status=raw.get("verification_status"),
                confidence_score=raw.get("confidence_score"),
                salience_tag=raw.get("salience_tag"),
                source_transcript_ids=raw.get("source_transcript_ids", ()),
                **scope,
            )
        if name == "mem_search":
            tags = tuple(raw["salience_tags"]) if raw.get("salience_tags") else None
            return self.engine.search(
                raw["query"],
                path_prefix=raw.get("path_prefix"),
                verified_only=bool(raw.get("verified_only", False)),
                salience_tags=tags,
                top_k=int(raw.get("top_k", 8)),
                **scope,
            )
        if name == "mem_grep":
            return self.engine.grep(
                raw["pattern"],
                path_prefix=raw.get("path_prefix", "/"),
                layer=raw.get("layer"),
                max_matches=int(raw.get("max_matches", 20)),
                **scope,
            )
        if name == "mem_read_transcripts":
            return self.engine.read_transcripts(list(raw["transcript_ids"]), **scope)
        if name == "mem_rollback":
            if raw.get("dream_batch_id"):
                return self.engine.rollback_batch(raw["dream_batch_id"], **scope)
            if raw.get("path"):
                return self.engine.rollback(raw["path"], snapshot_id=raw.get("snapshot_id"), **scope)
            raise ValueError("mem_rollback requires either dream_batch_id or path")
        raise ValueError(f"unhandled VFS tool: {name}")
