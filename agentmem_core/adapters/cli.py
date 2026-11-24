"""CLI adapter generated directly from UnifiedToolRegistry."""
from __future__ import annotations

import argparse
import json
from contextlib import closing

from agentmem_core.adapters.registry import TOOL_SPECS_BY_NAME, UnifiedToolRegistry
from agentmem_core.core.engine import UnifiedMemoryEngine


def dispatch_cli_tool(
    tool_name: str,
    arguments_json: str,
    *,
    db_url: str | None = None,
    db_path: str = ":memory:",
    user_id: str = "default",
    agent_id: str = "default",
    session_id: str | None = None,
) -> dict:
    engine = UnifiedMemoryEngine(db_path=db_path, db_url=db_url)
    with closing(engine):
        registry = UnifiedToolRegistry(engine, user_id=user_id, agent_id=agent_id, session_id=session_id)
        return registry.dispatch(tool_name, arguments_json)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unified Memory Engine CLI (generated from UnifiedToolRegistry)")
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--db-path", default=":memory:")
    parser.add_argument("--user-id", default="default")
    parser.add_argument("--agent-id", default="default")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("tool", choices=["list-tools", *sorted(TOOL_SPECS_BY_NAME.keys())])
    parser.add_argument("--args", default="{}", help="JSON object of tool arguments")
    args = parser.parse_args(argv)
    if args.tool == "list-tools":
        print(json.dumps([s.name for s in UnifiedToolRegistry.specs()], ensure_ascii=False, indent=2))
        return 0
    result = dispatch_cli_tool(
        args.tool,
        args.args,
        db_url=args.db_url,
        db_path=args.db_path,
        user_id=args.user_id,
        agent_id=args.agent_id,
        session_id=args.session_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
