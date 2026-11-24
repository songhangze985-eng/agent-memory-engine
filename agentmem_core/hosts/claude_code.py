"""Claude Code (cc) host integration: .mcp.json + UserPromptSubmit / PostToolUse / Stop hooks."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agentmem_core.core.code_anchors import normalize_rel_path
from agentmem_core.core.engine import UnifiedMemoryEngine
from agentmem_core.core.signals import AgentEvent

_FILE_PATH_HINT = re.compile(r"\b([A-Za-z0-9_./-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|toml|json|yaml|yml|sql|sh|md))\b")


def _default_repo_id(repo_root: Path) -> str:
    return repo_root.resolve().name or "repo"


def _default_db_path(repo_root: Path) -> Path:
    return repo_root.resolve() / ".agentmem" / "memory.db"


def render_claude_code_mcp_config(
    repo_root: str | Path,
    *,
    db_path: str | Path | None = None,
    user_id: str = "default",
    profile: str = "coding",
    repo_id: str | None = None,
    command: str = "agentmem-core",
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    eff_db = str(Path(db_path).resolve()) if db_path else str(_default_db_path(root))
    eff_repo = repo_id or _default_repo_id(root)
    return {
        "mcpServers": {
            "agentmem": {
                "command": command,
                "args": [
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
                ],
            }
        }
    }


def render_claude_code_hooks_config(
    repo_root: str | Path,
    *,
    db_path: str | Path | None = None,
    user_id: str = "default",
    profile: str = "coding",
    repo_id: str | None = None,
    command: str = "agentmem-core",
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    eff_db = str(Path(db_path).resolve()) if db_path else str(_default_db_path(root))
    eff_repo = repo_id or _default_repo_id(root)
    base = (
        f'{command} --db-url "{eff_db}" host hook claude-code {{event}} '
        f'--profile {profile} --user-id "{user_id}" --repo-id "{eff_repo}" --repo-root "{root}"'
    )
    return {
        "hooks": {
            "UserPromptSubmit": [
                {"hooks": [{"type": "command", "command": base.format(event="UserPromptSubmit")}]}
            ],
            "PostToolUse": [
                {"matcher": "Bash|Write|Edit|MultiEdit", "hooks": [{"type": "command", "command": base.format(event="PostToolUse")}]}
            ],
            "Stop": [
                {"hooks": [{"type": "command", "command": base.format(event="Stop")}]}
            ],
        }
    }


def handle_claude_code_hook(
    event_name: str,
    payload: dict[str, Any],
    engine: UnifiedMemoryEngine,
    *,
    user_id: str = "default",
    agent_id: str = "claude-code",
    repo_id: str | None = None,
    repo_root: str | Path | None = None,
    profile: str = "coding",
) -> dict[str, Any]:
    """Process one Claude Code hook JSON payload using C1 anchors, C2 signals, and C3 zero-LLM ambient recall."""
    root = Path(repo_root).resolve() if repo_root is not None else None
    eff_repo = repo_id or (_default_repo_id(root) if root is not None else "default")
    session_id = str(payload.get("session_id") or "cc-session")
    ev = event_name.strip()

    if ev == "UserPromptSubmit":
        prompt = str(payload.get("prompt") or payload.get("user_prompt") or "")
        # Also detect durable user corrections automatically (C2)
        corr = engine.ingest_event(
            AgentEvent(event_type="user_message", text=prompt),
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            scope_key=f"user:{user_id}",
            repo_id=eff_repo,
            repo_root=root,
            profile=profile,
        )
        # Extract referenced file paths from prompt for O(log N) anchor hit (C3 zero-LLM)
        active_files = list(dict.fromkeys(normalize_rel_path(m.group(1)) for m in _FILE_PATH_HINT.finditer(prompt)))
        amb = engine.ambient_context(
            prompt,
            active_files=active_files,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            repo_id=eff_repo,
            profile=profile,
        )
        blocks = [p for p in (amb.get("system_prefix", ""), amb.get("dynamic_memory", "")) if p and p.strip()]
        additional = "\n\n".join(blocks)
        return {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": additional,
            },
            "meta": {
                "llm_calls": amb["llm_calls"],
                "ambient_ms": amb["ambient_ms"],
                "injected_items": len(amb["items"]),
                "correction_captured": bool(corr.get("written")),
            },
        }

    if ev == "PostToolUse":
        tool_name = str(payload.get("tool_name") or "")
        tool_input = payload.get("tool_input") or {}
        tool_resp = payload.get("tool_response") or {}
        if isinstance(tool_resp, str):
            resp_text = tool_resp
            exit_code = payload.get("exit_code")
        else:
            resp_text = str(tool_resp.get("stdout") or "") + "\n" + str(tool_resp.get("stderr") or tool_resp.get("output") or "")
            exit_code = tool_resp.get("exit_code", tool_resp.get("exitCode", payload.get("exit_code")))

        if tool_name in {"Write", "Edit", "MultiEdit"}:
            fpath = tool_input.get("file_path") or tool_input.get("path") or ""
            rel = _relativize(fpath, root)
            res = engine.ingest_event(
                AgentEvent(
                    event_type="file_edit",
                    tool_name=tool_name,
                    text=str(tool_input.get("new_string") or tool_input.get("content") or "")[:240],
                    file_paths=(rel,) if rel else (),
                ),
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                repo_id=eff_repo,
                repo_root=root,
                profile=profile,
            )
            return {"continue": True, "event": "PostToolUse", "result": res}

        cmd = str(tool_input.get("command") or "")
        files = list(dict.fromkeys(normalize_rel_path(m.group(1)) for m in _FILE_PATH_HINT.finditer(cmd + "\n" + resp_text)))
        res = engine.ingest_event(
            AgentEvent(
                event_type="tool_result",
                tool_name=tool_name or "Bash",
                command=cmd,
                exit_code=int(exit_code) if exit_code is not None else None,
                text=resp_text.strip(),
                file_paths=tuple(files),
            ),
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            repo_id=eff_repo,
            repo_root=root,
            profile=profile,
        )
        return {"continue": True, "event": "PostToolUse", "result": res}

    if ev == "Stop":
        drift = engine.verify_anchors(repo_root=root, user_id=user_id, agent_id=agent_id)
        return {"continue": True, "event": "Stop", "stale_check": drift}

    return {"continue": True, "event": ev, "ignored": True}


def _relativize(path_str: str, repo_root: Path | None) -> str:
    if not path_str:
        return ""
    p = Path(path_str)
    if repo_root is not None and p.is_absolute():
        try:
            return normalize_rel_path(p.resolve().relative_to(repo_root.resolve()))
        except ValueError:
            pass
    return normalize_rel_path(path_str)


def install_claude_code(
    repo_root: str | Path,
    *,
    db_path: str | Path | None = None,
    user_id: str = "default",
    profile: str = "coding",
    repo_id: str | None = None,
    command: str = "agentmem-core",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Install/merge agentmem-core MCP server and hooks into a Claude Code (cc) workspace."""
    root = Path(repo_root).resolve()
    mcp_path = root / ".mcp.json"
    hooks_path = root / ".claude" / "settings.local.json"

    mcp_fragment = render_claude_code_mcp_config(
        root, db_path=db_path, user_id=user_id, profile=profile, repo_id=repo_id, command=command
    )
    hooks_fragment = render_claude_code_hooks_config(
        root, db_path=db_path, user_id=user_id, profile=profile, repo_id=repo_id, command=command
    )

    merged_mcp: dict[str, Any] = {}
    if mcp_path.is_file():
        merged_mcp = json.loads(mcp_path.read_text(encoding="utf-8"))
    merged_mcp.setdefault("mcpServers", {}).update(mcp_fragment["mcpServers"])

    merged_hooks: dict[str, Any] = {}
    if hooks_path.is_file():
        merged_hooks = json.loads(hooks_path.read_text(encoding="utf-8"))
    merged_hooks.setdefault("hooks", {}).update(hooks_fragment["hooks"])

    if not dry_run:
        mcp_path.write_text(json.dumps(merged_mcp, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        hooks_path.parent.mkdir(parents=True, exist_ok=True)
        hooks_path.write_text(json.dumps(merged_hooks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "host": "claude-code",
        "dry_run": dry_run,
        "files": [str(mcp_path), str(hooks_path)],
        "mcp_config": merged_mcp,
        "hooks_config": merged_hooks,
    }
