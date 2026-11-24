"""Zero-dependency JSON-RPC 2.0 MCP stdio server exposing UnifiedToolRegistry."""
from __future__ import annotations

import json
import sys
from contextlib import closing
from pathlib import Path
from typing import Any, TextIO

from agentmem_core.adapters.registry import UnifiedToolRegistry
from agentmem_core.core.engine import UnifiedMemoryEngine
from agentmem_core.vfs.models import OptimisticLockError

MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_VERSION = "0.3.0"


class MCPServer:
    """Standards-compliant MCP JSON-RPC 2.0 server backed by a single UnifiedMemoryEngine."""

    def __init__(
        self,
        engine: UnifiedMemoryEngine,
        *,
        user_id: str = "default",
        agent_id: str = "default",
        session_id: str | None = None,
        profile: str = "coding",
        repo_id: str | None = None,
        project_id: str | None = None,
        repo_root: str | Path | None = None,
    ):
        self.engine = engine
        self.user_id = user_id
        self.agent_id = agent_id
        self.session_id = session_id
        self.profile = profile
        self.repo_id = repo_id
        self.project_id = project_id
        self.repo_root = str(repo_root) if repo_root is not None else None
        self.registry = UnifiedToolRegistry(engine, user_id=user_id, agent_id=agent_id, session_id=session_id)

    def _inject_defaults(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        args = dict(arguments or {})
        if tool_name in {"memory_ambient_context", "memory_build_context", "memory_ingest_event"}:
            args.setdefault("profile", self.profile)
            if self.repo_id and "repo_id" not in args:
                args["repo_id"] = self.repo_id
            if self.project_id and "project_id" not in args:
                args["project_id"] = self.project_id
        if tool_name in {"memory_ingest_event", "memory_verify_anchors", "memory_write_rule"}:
            if self.repo_root and "repo_root" not in args:
                args["repo_root"] = self.repo_root
        if tool_name == "memory_write_rule" and self.repo_id and "scope_key" not in args:
            args["scope_key"] = f"repo:{self.repo_id}"
        return args

    def handle_message(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        """Handle one JSON-RPC 2.0 request/notification; return response dict or None for notifications."""
        method = msg.get("method")
        req_id = msg.get("id")
        params = msg.get("params") or {}

        if method == "notifications/initialized" or (req_id is None and str(method or "").startswith("notifications/")):
            return None

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "agentmem-core", "version": SERVER_VERSION},
                },
            }

        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": self.registry.mcp_definitions()},
            }

        if method == "tools/call":
            tool_name = str(params.get("name") or "")
            raw_args = params.get("arguments") or {}
            try:
                args = self._inject_defaults(tool_name, raw_args)
                output = self.registry.dispatch(tool_name, args)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(output, ensure_ascii=False, default=str)}],
                        "isError": False,
                    },
                }
            except (ValueError, OptimisticLockError) as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps({"error": str(exc)}, ensure_ascii=False)}],
                        "isError": True,
                    },
                }
            except Exception as exc:  # pragma: no cover - defensive RPC error envelope
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32603, "message": str(exc)},
                }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    def serve_stdio(self, stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
        """Run JSON-RPC 2.0 loop supporting both newline-delimited JSON and Content-Length framing."""
        in_stream = stdin or sys.stdin
        out_stream = stdout or sys.stdout

        while True:
            line = in_stream.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.lower().startswith("content-length:"):
                length = int(stripped.split(":", 1)[1].strip())
                # Consume remaining header lines up to blank separator
                while True:
                    hdr = in_stream.readline()
                    if not hdr or not hdr.strip():
                        break
                body = in_stream.read(length)
                use_framing = True
            else:
                body = stripped
                use_framing = False

            try:
                msg = json.loads(body)
            except json.JSONDecodeError as exc:
                resp: dict[str, Any] | None = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"},
                }
            else:
                resp = self.handle_message(msg)

            if resp is None:
                continue
            encoded = json.dumps(resp, ensure_ascii=False, default=str)
            if use_framing:
                raw_bytes = encoded.encode("utf-8")
                out_stream.write(f"Content-Length: {len(raw_bytes)}\r\n\r\n{encoded}")
            else:
                out_stream.write(encoded + "\n")
            out_stream.flush()
        return 0


def run_mcp_stdio(
    *,
    db_path: str | Path = ":memory:",
    db_url: str | None = None,
    user_id: str = "default",
    agent_id: str = "default",
    session_id: str | None = None,
    profile: str = "coding",
    repo_id: str | None = None,
    project_id: str | None = None,
    repo_root: str | Path | None = None,
) -> int:
    engine = UnifiedMemoryEngine(db_path=db_path, db_url=db_url, profile=profile)
    with closing(engine):
        server = MCPServer(
            engine,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            profile=profile,
            repo_id=repo_id,
            project_id=project_id,
            repo_root=repo_root,
        )
        return server.serve_stdio()
