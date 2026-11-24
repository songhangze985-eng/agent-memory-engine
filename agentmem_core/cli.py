"""Command line entry points for local memory and HTTP serving."""
import argparse
from contextlib import closing
import json
import sys

from agentmem_core.engine import CognitiveMemoryEngine
from agentmem_core.vfs import VFSMemoryEngine


def main(argv=None):
    parser = argparse.ArgumentParser(prog="agentmem-core")
    parser.add_argument("--db-url", help="SQLite, LanceDB, or PostgreSQL DSN (AGENTMEM_CORE_DB_URL / MEMENGINE_DB_URL)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-db")
    commands.add_parser("health")
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    mcp_cmd = commands.add_parser("mcp-serve", help="Run JSON-RPC 2.0 stdio MCP server for Claude Code, Codex, OpenCode & MCP clients")
    mcp_cmd.add_argument("--profile", default="coding")
    mcp_cmd.add_argument("--user-id", default="default")
    mcp_cmd.add_argument("--agent-id", default="default")
    mcp_cmd.add_argument("--session-id", default=None)
    mcp_cmd.add_argument("--repo-id", default=None)
    mcp_cmd.add_argument("--project-id", default=None)
    mcp_cmd.add_argument("--repo-root", default=None)

    amb_cmd = commands.add_parser("ambient", help="Run zero-LLM millisecond ambient context assembly")
    amb_cmd.add_argument("query", nargs="?", default="")
    amb_cmd.add_argument("--file", action="append", dest="active_files", default=[])
    amb_cmd.add_argument("--symbol", action="append", dest="active_symbols", default=[])
    amb_cmd.add_argument("--error", dest="error_text", default="")
    amb_cmd.add_argument("--profile", default="coding")
    amb_cmd.add_argument("--user-id", default="default")
    amb_cmd.add_argument("--agent-id", default="default")
    amb_cmd.add_argument("--repo-id", default=None)
    amb_cmd.add_argument("--project-id", default=None)
    amb_cmd.add_argument("--token-budget", type=int, default=None)

    host_cmd = commands.add_parser("host", help="Install or handle hooks for general-purpose coding hosts (claude-code/cc, codex, opencode)")
    host_sub = host_cmd.add_subparsers(dest="host_action", required=True)
    host_install = host_sub.add_parser("install")
    host_install.add_argument("host_name", choices=("claude-code", "cc", "codex", "opencode"))
    host_install.add_argument("--repo-root", default=".")
    host_install.add_argument("--user-id", default="default")
    host_install.add_argument("--profile", default="coding")
    host_install.add_argument("--repo-id", default=None)
    host_install.add_argument("--dry-run", action="store_true")

    host_hook = host_sub.add_parser("hook")
    host_hook.add_argument("host_name", choices=("claude-code", "cc", "codex"))
    host_hook.add_argument("event_name")
    host_hook.add_argument("--repo-root", default=None)
    host_hook.add_argument("--user-id", default="default")
    host_hook.add_argument("--profile", default="coding")
    host_hook.add_argument("--repo-id", default=None)
    host_hook.add_argument("--payload", default=None, help="Optional JSON payload string (reads stdin if omitted)")
    vfs_cmd = commands.add_parser("vfs", help="v2 Virtual Filesystem memory engine commands")
    vfs_cmd.add_argument("--transcript-dir", help="Optional L0 chunked transcript directory")
    vfs_cmd.add_argument("--user-id", default="default")
    vfs_cmd.add_argument("--agent-id", default="default")
    vfs_sub = vfs_cmd.add_subparsers(dest="vfs_action", required=True)
    vfs_sub.add_parser("health")
    vfs_tree = vfs_sub.add_parser("tree")
    vfs_tree.add_argument("--path", default="/")
    vfs_tree.add_argument("--layer", choices=("L1", "L2", "L3"), default="L2")
    vfs_read = vfs_sub.add_parser("read-note")
    vfs_read.add_argument("path")
    vfs_rule = vfs_sub.add_parser("write-rule")
    vfs_rule.add_argument("path")
    vfs_rule.add_argument("--title", required=True)
    vfs_rule.add_argument("--trigger", required=True)
    vfs_rule.add_argument("--action", required=True)
    vfs_rule.add_argument("--verification", default="")
    vfs_rule.add_argument("--always-active", action="store_true")
    vfs_search = vfs_sub.add_parser("search")
    vfs_search.add_argument("query")
    vfs_search.add_argument("--top-k", type=int, default=8)
    vfs_search.add_argument("--verified-only", action="store_true")
    vfs_ctx = vfs_sub.add_parser("context")
    vfs_ctx.add_argument("query")
    vfs_ctx.add_argument("--token-budget", type=int, default=2048)
    vfs_ctx.add_argument("--disclosure", choices=("progressive", "full"), default="progressive")
    vfs_compact = vfs_sub.add_parser("compact")
    vfs_compact.add_argument("dir_path")
    for name in ("retain", "recall", "reflect"):
        command = commands.add_parser(name)
        command.add_argument("--user-id", default="default")
        command.add_argument("--agent-id", default="default")
        if name != "reflect":
            command.add_argument("--session-id")
        if name == "retain":
            command.add_argument("text")
            command.add_argument("--ttl-seconds", type=float)
            command.add_argument("--role", default="user")
            command.add_argument("--mention-time")
            command.add_argument("--speaker")
        elif name == "recall":
            command.add_argument("query")
            command.add_argument("--mode", choices=("single_query", "reconstructive"), default="single_query")
            command.add_argument("--scope", action="append", dest="scopes")
            command.add_argument("--top-k", type=int)
            command.add_argument("--reference-time")
            command.add_argument("--packing-mode", choices=("token_budget", "top_k"))
            command.add_argument("--token-budget", type=int)
            command.add_argument("--include-cold", action="store_true")
        else:
            command.add_argument("--force", action="store_true")
            command.add_argument("--query")
            command.add_argument("--reference-time")
    args = vars(parser.parse_args(argv))
    command, db_url = args.pop("command"), args.pop("db_url")
    if command == "serve":
        try:
            import uvicorn
            from agentmem_core.server import create_app
        except ImportError:
            parser.error("serve requires pip install 'agentmem-core[server]'")
        uvicorn.run(create_app(db_url=db_url), **args)
        return 0
    if command == "mcp-serve":
        from agentmem_core.adapters.mcp_server import run_mcp_stdio

        return run_mcp_stdio(db_url=db_url, **args)
    if command == "ambient":
        from agentmem_core.core import UnifiedMemoryEngine

        engine = UnifiedMemoryEngine(db_url=db_url, profile=args["profile"])
        with closing(engine):
            result = engine.ambient_context(**args)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    if command == "host":
        from agentmem_core.core import UnifiedMemoryEngine
        from agentmem_core.hosts import handle_host_hook, install_host

        action = args.pop("host_action")
        host_name = args.pop("host_name")
        if action == "install":
            result = install_host(
                host_name,
                args["repo_root"],
                db_path=db_url,
                user_id=args["user_id"],
                profile=args["profile"],
                repo_id=args["repo_id"],
                dry_run=args["dry_run"],
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        raw_payload = args["payload"]
        if raw_payload is None:
            raw_payload = sys.stdin.read() if not sys.stdin.isatty() else "{}"
        payload = json.loads(raw_payload) if raw_payload.strip() else {}
        engine = UnifiedMemoryEngine(db_url=db_url, profile=args["profile"])
        with closing(engine):
            result = handle_host_hook(
                host_name,
                args["event_name"],
                payload,
                engine,
                user_id=args["user_id"],
                repo_id=args["repo_id"],
                repo_root=args["repo_root"],
                profile=args["profile"],
            )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    if command == "vfs":
        action = args.pop("vfs_action")
        transcript_dir = args.pop("transcript_dir")
        scope = {"user_id": args.pop("user_id"), "agent_id": args.pop("agent_id")}
        try:
            vfs = VFSMemoryEngine(db_url=db_url, transcript_dir=transcript_dir)
            with closing(vfs):
                if action == "health":
                    result = vfs.store.health_check()
                elif action == "tree":
                    result = {"outline": vfs.tree_outline(path=args["path"], layer=args["layer"], **scope)}
                elif action == "read-note":
                    result = vfs.read_note(args["path"], **scope)
                elif action == "write-rule":
                    result = vfs.write_rule(
                        args["path"], title=args["title"], trigger_condition=args["trigger"],
                        action=args["action"], verification_result=args["verification"],
                        always_active=args["always_active"], **scope,
                    )
                elif action == "search":
                    result = vfs.search(args["query"], top_k=args["top_k"], verified_only=args["verified_only"], **scope)
                elif action == "context":
                    result = vfs.build_context(args["query"], token_budget=args["token_budget"], disclosure=args["disclosure"], **scope)
                else:
                    result = vfs.compact_directory(args["dir_path"], **scope)
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    try:
        engine = CognitiveMemoryEngine(db_url=db_url)
        with closing(engine.store):
            if command in {"health", "init-db"}:
                result = engine.store.health_check()
            elif command == "retain":
                args["session_id"] = args["session_id"] or "default"
                result = engine.retain(**args)
            elif command == "recall":
                result = engine.recall(**args, read_only=True)
            else:
                query = args.pop("query")
                reference = args.pop("reference_time")
                force = args.pop("force")
                result = engine.reflect(query, reference_time=reference, **args) if query else engine.maintain(force=force, **args)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
