"""Command line entry points for local memory and HTTP serving."""
import argparse
from contextlib import closing
import json

from memengine.engine import CognitiveMemoryEngine


def main(argv=None):
    parser = argparse.ArgumentParser(prog="memengine")
    parser.add_argument("--db-url", help="SQLite path or PostgreSQL DSN (MEMENGINE_DB_URL)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-db")
    commands.add_parser("health")
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
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
            from memengine.server import create_app
        except ImportError:
            parser.error("serve requires pip install 'memengine[server]'")
        uvicorn.run(create_app(db_url=db_url), **args)
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
