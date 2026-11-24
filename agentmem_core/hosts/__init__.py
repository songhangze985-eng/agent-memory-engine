"""General-purpose agent host adapters: Claude Code (cc), OpenAI Codex CLI (codex), OpenCode & MCP."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from agentmem_core.core.engine import UnifiedMemoryEngine
from agentmem_core.hosts.claude_code import (
    handle_claude_code_hook,
    install_claude_code,
    render_claude_code_hooks_config,
    render_claude_code_mcp_config,
)
from agentmem_core.hosts.codex import (
    handle_codex_hook,
    install_codex,
    render_codex_agents_snippet,
    render_codex_config_toml,
)
from agentmem_core.hosts.opencode import install_opencode, render_opencode_config

HOST_ALIASES: dict[str, str] = {
    "cc": "claude-code",
    "claude": "claude-code",
    "claude-code": "claude-code",
    "codex": "codex",
    "codex-cli": "codex",
    "opencode": "opencode",
}

SUPPORTED_HOSTS = ("claude-code", "codex", "opencode")


def resolve_host_name(name: str) -> str:
    canonical = HOST_ALIASES.get(str(name or "").strip().casefold())
    if canonical is None:
        raise ValueError(f"unsupported host {name!r}; expected one of {sorted(HOST_ALIASES.keys())}")
    return canonical


def install_host(
    host: str,
    repo_root: str | Path,
    *,
    db_path: str | Path | None = None,
    user_id: str = "default",
    profile: str = "coding",
    repo_id: str | None = None,
    command: str = "agentmem-core",
    dry_run: bool = False,
) -> dict[str, Any]:
    canonical = resolve_host_name(host)
    if canonical == "claude-code":
        return install_claude_code(
            repo_root, db_path=db_path, user_id=user_id, profile=profile, repo_id=repo_id, command=command, dry_run=dry_run
        )
    if canonical == "codex":
        return install_codex(
            repo_root, db_path=db_path, user_id=user_id, profile=profile, repo_id=repo_id, command=command, dry_run=dry_run
        )
    return install_opencode(
        repo_root, db_path=db_path, user_id=user_id, profile=profile, repo_id=repo_id, command=command, dry_run=dry_run
    )


def handle_host_hook(
    host: str,
    event_name: str,
    payload: dict[str, Any],
    engine: UnifiedMemoryEngine,
    *,
    user_id: str = "default",
    repo_id: str | None = None,
    repo_root: str | Path | None = None,
    profile: str = "coding",
) -> dict[str, Any]:
    canonical = resolve_host_name(host)
    if canonical == "claude-code":
        return handle_claude_code_hook(
            event_name, payload, engine, user_id=user_id, repo_id=repo_id, repo_root=repo_root, profile=profile
        )
    if canonical == "codex":
        return handle_codex_hook(
            event_name, payload, engine, user_id=user_id, repo_id=repo_id, repo_root=repo_root, profile=profile
        )
    raise ValueError(f"host {canonical!r} uses MCP directly and does not define shell hook events")


__all__ = [
    "SUPPORTED_HOSTS",
    "handle_claude_code_hook",
    "handle_codex_hook",
    "handle_host_hook",
    "install_claude_code",
    "install_codex",
    "install_host",
    "install_opencode",
    "render_claude_code_hooks_config",
    "render_claude_code_mcp_config",
    "render_codex_agents_snippet",
    "render_codex_config_toml",
    "render_opencode_config",
    "resolve_host_name",
]
