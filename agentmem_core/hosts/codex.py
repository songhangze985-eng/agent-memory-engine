"""OpenAI Codex CLI host integration: .codex/config.toml MCP server, AGENTS.md snippet & notify hook."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentmem_core.core.code_anchors import normalize_rel_path
from agentmem_core.core.engine import UnifiedMemoryEngine
from agentmem_core.core.signals import AgentEvent

_CODEX_BLOCK_START = "# >>> agentmem-core codex mcp >>>"
_CODEX_BLOCK_END = "# <<< agentmem-core codex mcp <<<"


def _default_repo_id(repo_root: Path) -> str:
    return repo_root.resolve().name or "repo"


def _default_db_path(repo_root: Path) -> Path:
    return repo_root.resolve() / ".agentmem" / "memory.db"


def render_codex_config_toml(
    repo_root: str | Path,
    *,
    db_path: str | Path | None = None,
    user_id: str = "default",
    profile: str = "coding",
    repo_id: str | None = None,
    command: str = "agentmem-core",
) -> str:
    root = Path(repo_root).resolve()
    eff_db = str(Path(db_path).resolve()) if db_path else str(_default_db_path(root))
    eff_repo = repo_id or _default_repo_id(root)
    args = [
        "--db-url",
        eff_db,
        "mcp-serve",
        "--profile",
        profile,
        "--user-id",
        user_id,
        "--repo-id",
        eff_repo,
        "--repo-root",
        str(root),
    ]
    args_toml = ", ".join(json.dumps(a) for a in args)
    return "\n".join(
        [
            _CODEX_BLOCK_START,
            "[mcp_servers.agentmem]",
            f"command = {json.dumps(command)}",
            f"args = [{args_toml}]",
            "startup_timeout_sec = 10",
            "tool_timeout_sec = 30",
            _CODEX_BLOCK_END,
            "",
        ]
    )


def render_codex_agents_snippet(repo_id: str) -> str:
    return "\n".join(
        [
            "## Agent Memory (`agentmem-core` via MCP)",
            f"- Before editing unfamiliar files in `{repo_id}`, call `memory_ambient_context` with active files/errors (0 LLM calls, <10ms).",
            "- After resolving a test/build failure (`fail -> pass`) or receiving a durable user correction, call `memory_ingest_event` or `memory_write_rule` with `anchors`.",
            "- After major code or dependency changes, call `memory_verify_anchors` so outdated rules are flagged `stale`.",
            "",
        ]
    )


def handle_codex_hook(
    event_name: str,
    payload: dict[str, Any],
    engine: UnifiedMemoryEngine,
    *,
    user_id: str = "default",
    agent_id: str = "codex",
    repo_id: str | None = None,
    repo_root: str | Path | None = None,
    profile: str = "coding",
) -> dict[str, Any]:
    """Process Codex CLI notify/turn/tool JSON payloads through C1/C2/C3 unified pipelines."""
    root = Path(repo_root).resolve() if repo_root is not None else None
    eff_repo = repo_id or (_default_repo_id(root) if root is not None else "default")
    session_id = str(payload.get("thread_id") or payload.get("session_id") or "codex-session")
    ev = event_name.strip().casefold()

    if ev in {"ambient", "pre-turn", "prompt"}:
        query = str(payload.get("prompt") or payload.get("query") or "")
        files = [normalize_rel_path(p) for p in (payload.get("active_files") or ())]
        amb = engine.ambient_context(
            query,
            active_files=files,
            error_text=str(payload.get("error_text") or ""),
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            repo_id=eff_repo,
            profile=profile,
        )
        return {
            "host": "codex",
            "event": event_name,
            "context": "\n\n".join(p for p in (amb["system_prefix"], amb["dynamic_memory"]) if p),
            "llm_calls": amb["llm_calls"],
            "ambient_ms": amb["ambient_ms"],
        }

    if ev in {"tool", "exec", "notify", "agent-turn-complete"}:
        if "command" in payload or "exit_code" in payload:
            res = engine.ingest_event(
                AgentEvent(
                    event_type="tool_result",
                    tool_name=str(payload.get("tool_name") or "shell"),
                    command=str(payload.get("command") or ""),
                    exit_code=payload.get("exit_code"),
                    text=str(payload.get("output") or payload.get("text") or ""),
                    file_paths=tuple(normalize_rel_path(p) for p in (payload.get("file_paths") or ())),
                ),
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                repo_id=eff_repo,
                repo_root=root,
                profile=profile,
            )
            return {"host": "codex", "event": event_name, "result": res}
        changed = [normalize_rel_path(p) for p in (payload.get("changed_files") or ())]
        if changed or root is not None:
            drift = engine.verify_anchors(repo_root=root, changed_files=changed or None, user_id=user_id, agent_id=agent_id)
            return {"host": "codex", "event": event_name, "stale_check": drift}

    return {"host": "codex", "event": event_name, "ignored": True}


def install_codex(
    repo_root: str | Path,
    *,
    db_path: str | Path | None = None,
    user_id: str = "default",
    profile: str = "coding",
    repo_id: str | None = None,
    command: str = "agentmem-core",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Install/merge agentmem-core MCP server stanza into ``.codex/config.toml``."""
    root = Path(repo_root).resolve()
    eff_repo = repo_id or _default_repo_id(root)
    config_path = root / ".codex" / "config.toml"
    stanza = render_codex_config_toml(
        root, db_path=db_path, user_id=user_id, profile=profile, repo_id=eff_repo, command=command
    )

    existing = config_path.read_text(encoding="utf-8") if config_path.is_file() else ""
    if _CODEX_BLOCK_START in existing and _CODEX_BLOCK_END in existing:
        before = existing.split(_CODEX_BLOCK_START, 1)[0].rstrip()
        after = existing.split(_CODEX_BLOCK_END, 1)[1].lstrip()
        merged = "\n\n".join(p for p in (before, stanza.strip(), after) if p) + "\n"
    else:
        merged = (existing.rstrip() + "\n\n" + stanza) if existing.strip() else stanza

    if not dry_run:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(merged, encoding="utf-8")

    return {
        "host": "codex",
        "dry_run": dry_run,
        "files": [str(config_path)],
        "config_toml": merged,
        "agents_md_snippet": render_codex_agents_snippet(eff_repo),
    }
