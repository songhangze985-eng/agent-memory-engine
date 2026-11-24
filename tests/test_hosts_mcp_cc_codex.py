"""End-to-end tests for universal MCP stdio server & general-purpose hosts: Claude Code (cc), Codex, OpenCode."""
from __future__ import annotations

import io
import json

from agentmem_core import UnifiedMemoryEngine
from agentmem_core.adapters.mcp_server import MCPServer
from agentmem_core.cli import main as cli_main
from agentmem_core.hosts import handle_host_hook, install_host


def test_mcp_stdio_server_full_rpc_lifecycle(tmp_path):
    repo = tmp_path / "svc"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")

    engine = UnifiedMemoryEngine(tmp_path / "mcp.db", profile="coding")
    try:
        server = MCPServer(engine, user_id="dev", repo_id="svc", repo_root=repo, profile="coding")
        requests = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "memory_write_rule",
                    "arguments": {
                        "path": "/rules/app.md",
                        "title": "App runner flag",
                        "trigger_condition": "Running src/app.py locally",
                        "action": "Pass --reload-dir src when running uvicorn",
                        "verification_result": "Avoids watching .venv",
                        "anchors": ["src/app.py::run"],
                    },
                },
            },
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "memory_ambient_context",
                    "arguments": {
                        "query": "uvicorn reload",
                        "active_files": ["src/app.py"],
                        "active_symbols": ["run"],
                    },
                },
            },
        ]
        stdin = io.StringIO("\n".join(json.dumps(r) for r in requests) + "\n")
        stdout = io.StringIO()
        assert server.serve_stdio(stdin=stdin, stdout=stdout) == 0

        responses = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
        assert len(responses) == 4  # notification emits no response
        assert responses[0]["result"]["serverInfo"]["name"] == "agentmem-core"
        tool_names = {t["name"] for t in responses[1]["result"]["tools"]}
        assert {"memory_ambient_context", "memory_ingest_event", "memory_verify_anchors", "memory_write_rule"} <= tool_names

        write_out = json.loads(responses[2]["result"]["content"][0]["text"])
        assert write_out["accepted"] is True
        assert len(write_out["anchors"]) == 1

        amb_out = json.loads(responses[3]["result"]["content"][0]["text"])
        assert amb_out["llm_calls"] == 0
        assert "--reload-dir src" in amb_out["dynamic_memory"]
    finally:
        engine.close()


def test_claude_code_cc_install_and_hooks_e2e(tmp_path):
    repo = tmp_path / "payments"
    (repo / "src").mkdir(parents=True)
    target = repo / "src" / "stripe_client.py"
    target.write_text("TIMEOUT = 5\n", encoding="utf-8")

    # 1. Install Claude Code (.mcp.json + .claude/settings.local.json) via 'cc' alias
    installed = install_host("cc", repo, user_id="alice", profile="coding")
    assert installed["host"] == "claude-code"
    assert (repo / ".mcp.json").is_file()
    assert (repo / ".claude" / "settings.local.json").is_file()
    mcp_cfg = json.loads((repo / ".mcp.json").read_text(encoding="utf-8"))
    assert "agentmem" in mcp_cfg["mcpServers"]
    assert "mcp-serve" in mcp_cfg["mcpServers"]["agentmem"]["args"]

    # 2. E2E hook flow: PostToolUse failure -> Edit -> PostToolUse pass -> UserPromptSubmit zero-LLM ambient recall!
    engine = UnifiedMemoryEngine(tmp_path / "cc.db", profile="coding")
    try:
        r1 = handle_host_hook(
            "cc",
            "PostToolUse",
            {
                "session_id": "cc-1",
                "tool_name": "Bash",
                "tool_input": {"command": "pytest tests/test_stripe.py"},
                "tool_response": {"exit_code": 1, "stderr": "ReadTimeout: stripe_client.py timed out"},
            },
            engine,
            user_id="alice",
            repo_id="payments",
            repo_root=repo,
        )
        assert r1["result"]["signal"] == "failure_buffered"

        target.write_text("TIMEOUT = 30\n", encoding="utf-8")
        handle_host_hook(
            "cc",
            "PostToolUse",
            {
                "session_id": "cc-1",
                "tool_name": "Edit",
                "tool_input": {"file_path": str(target), "new_string": "Raise Stripe TIMEOUT to 30s in src/stripe_client.py"},
            },
            engine,
            user_id="alice",
            repo_id="payments",
            repo_root=repo,
        )

        r2 = handle_host_hook(
            "cc",
            "PostToolUse",
            {
                "session_id": "cc-1",
                "tool_name": "Bash",
                "tool_input": {"command": "uv run pytest tests/test_stripe.py"},
                "tool_response": {"exit_code": 0, "stdout": "5 passed"},
            },
            engine,
            user_id="alice",
            repo_id="payments",
            repo_root=repo,
        )
        assert r2["result"]["signal"] == "fail_to_pass"
        assert r2["result"]["written"] is True

        # Next user prompt referencing src/stripe_client.py receives zero-LLM ambient guidance
        r3 = handle_host_hook(
            "cc",
            "UserPromptSubmit",
            {"session_id": "cc-2", "prompt": "Fix retry logic in src/stripe_client.py"},
            engine,
            user_id="alice",
            repo_id="payments",
            repo_root=repo,
        )
        assert r3["meta"]["llm_calls"] == 0
        assert "TIMEOUT to 30s" in r3["hookSpecificOutput"]["additionalContext"]
    finally:
        engine.close()


def test_codex_and_opencode_install_and_cli_hooks(tmp_path, capsys):
    repo = tmp_path / "analytics"
    repo.mkdir()
    db_path = tmp_path / "codex.db"

    # Codex install (.codex/config.toml) + idempotent re-install
    res1 = install_host("codex", repo, db_path=db_path, user_id="bob")
    assert res1["host"] == "codex"
    toml_file = repo / ".codex" / "config.toml"
    assert toml_file.is_file()
    content1 = toml_file.read_text(encoding="utf-8")
    assert "[mcp_servers.agentmem]" in content1
    assert "mcp-serve" in content1

    res2 = install_host("codex", repo, db_path=db_path, user_id="bob")
    assert toml_file.read_text(encoding="utf-8").count("[mcp_servers.agentmem]") == 1
    assert "memory_ambient_context" in res2["agents_md_snippet"]

    # OpenCode install (opencode.json)
    oc = install_host("opencode", repo, db_path=db_path, user_id="bob")
    assert oc["host"] == "opencode"
    assert (repo / "opencode.json").is_file()

    # CLI host hook invocation for codex
    payload = json.dumps({"prompt": "Check build rules", "active_files": []})
    rc = cli_main(
        [
            "--db-url",
            str(db_path),
            "host",
            "hook",
            "codex",
            "ambient",
            "--repo-root",
            str(repo),
            "--payload",
            payload,
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["host"] == "codex"
    assert out["llm_calls"] == 0
