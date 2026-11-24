"""Command line entry points for local memory and HTTP serving."""
import argparse
from contextlib import closing
import json

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
