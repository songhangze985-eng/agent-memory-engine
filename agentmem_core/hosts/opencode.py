"""OpenCode & generic MCP host configuration generator."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _default_repo_id(repo_root: Path) -> str:
    return repo_root.resolve().name or "repo"


def _default_db_path(repo_root: Path) -> Path:
    return repo_root.resolve() / ".agentmem" / "memory.db"


def render_opencode_config(
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
        "mcp": {
            "agentmem": {
                "type": "local",
                "command": [
                    command,
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
                "enabled": True,
            }
        }
    }


def install_opencode(
    repo_root: str | Path,
    *,
    db_path: str | Path | None = None,
    user_id: str = "default",
    profile: str = "coding",
    repo_id: str | None = None,
    command: str = "agentmem-core",
    dry_run: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    cfg_path = root / "opencode.json"
    frag = render_opencode_config(
        root, db_path=db_path, user_id=user_id, profile=profile, repo_id=repo_id, command=command
    )
    merged: dict[str, Any] = {}
    if cfg_path.is_file():
        merged = json.loads(cfg_path.read_text(encoding="utf-8"))
    merged.setdefault("mcp", {}).update(frag["mcp"])
    if not dry_run:
        cfg_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "host": "opencode",
        "dry_run": dry_run,
        "files": [str(cfg_path)],
        "config": merged,
    }
