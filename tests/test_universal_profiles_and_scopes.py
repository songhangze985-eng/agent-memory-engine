"""Contract tests for G3 & M1 convergence: declarative profiles, zero profile branching in core, scope chains & shims."""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from agentmem_core import MemoryProfile, ScopeChain, UnifiedMemoryEngine, list_profiles, load_profile
from agentmem_core.compat import UnifiedV1Shim, UnifiedVFSShim

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = REPO_ROOT / "agentmem_core" / "core"


def test_zero_profile_branches_in_core_codebase():
    """G3 / M1 hard invariant: zero ``if profile`` or string-equality profile branches in core."""
    branch_re = re.compile(
        r"\bif\b[^\n]*\bprofile\b|"
        r"\bprofile\s*==\s*['\"](?:conversational|coding|coding_agent|hybrid)['\"]|"
        r"\bprofile_name\s*==",
    )
    violations: list[str] = []
    for py_file in sorted(CORE_DIR.rglob("*.py")):
        src = py_file.read_text(encoding="utf-8")
        # Ignore comments/docstrings via line check and AST comparison check
        for lineno, line in enumerate(src.splitlines(), start=1):
            code = line.split("#", 1)[0]
            if branch_re.search(code):
                # Allow pure None / isinstance type-dispatch in loader/resolver helpers
                if "isinstance(" in code or "profile is None" in code or "profile is not None" in code:
                    continue
                violations.append(f"{py_file.name}:{lineno}: {line.strip()}")
        tree = ast.parse(src, filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                left_src = ast.unparse(node.left)
                if "profile" in left_src and any(
                    isinstance(c, ast.Constant) and c.value in {"conversational", "coding", "coding_agent", "hybrid"}
                    for c in node.comparators
                ):
                    violations.append(f"{py_file.name}:{node.lineno}: AST profile string branch ({ast.unparse(node)})")
    assert violations == [], f"Profile branching found in core:\n" + "\n".join(violations)


def test_declarative_profile_loader_presets_and_custom_yaml(tmp_path):
    names = list_profiles()
    assert {"conversational", "coding", "hybrid"}.issubset(set(names))

    conv = load_profile("conversational")
    coding = load_profile("coding")
    alias = load_profile("coding_agent")
    assert conv.name == "conversational"
    assert conv.disclosure == "full"
    assert conv.include_directory_outline is False
    assert conv.enable_ambient_llm is False

    assert coding.name == "coding"
    assert coding.disclosure == "progressive"
    assert coding.include_directory_outline is True
    assert coding.preload_always_active is True
    assert coding.enable_code_anchors is True
    assert coding.stale_policy == "exclude"
    assert alias == coding

    # Custom YAML profile file loads cleanly with zero code changes
    custom_yaml = tmp_path / "custom_ops.yaml"
    custom_yaml.write_text(
        "\n".join(
            [
                "name: custom_ops",
                "disclosure: progressive",
                "default_token_budget: 1024",
                "preload_paths:",
                "  - /runbook.md",
                "preload_always_active: true",
                "include_directory_outline: false",
                "enable_code_anchors: true",
                "stale_policy: warn",
                "scope_hierarchy:",
                "  - session",
                "  - project",
                "  - user",
                "  - global",
                "ambient_token_budget: 384",
                "max_writes_per_window: 16",
            ]
        ),
        encoding="utf-8",
    )
    custom = load_profile(custom_yaml)
    assert custom.name == "custom_ops"
    assert custom.preload_paths == ("/runbook.md",)
    assert custom.include_directory_outline is False
    assert custom.stale_policy == "warn"
    assert custom.ambient_token_budget == 384

    # G4 invariant: ambient LLM must always be False
    with pytest.raises(ValueError, match="enable_ambient_llm"):
        MemoryProfile(name="illegal", enable_ambient_llm=True)


def test_same_core_passes_conversational_and_coding_contracts(tmp_path):
    """G3 contract: single UnifiedMemoryEngine instance serves both profiles cleanly."""
    engine = UnifiedMemoryEngine(tmp_path / "dual_profile.db")
    try:
        # 1. Conversational contract
        engine.ingest_turn(
            "I prefer concise Chinese explanations and pytest for Python testing.",
            user_id="alice",
            session_id="chat-1",
            scope_key="user:alice",
            profile="conversational",
            timestamp=1_700_000_000.0,
        )
        conv_ctx = engine.build_context(
            "What testing framework do I prefer?",
            user_id="alice",
            profile="conversational",
            reference_time=1_700_000_100.0,
        )
        assert conv_ctx["profile"] == "conversational"
        assert conv_ctx["disclosure"] == "full"
        assert "Directory Outline" not in conv_ctx["dynamic_memory"]
        assert "pytest" in conv_ctx["dynamic_memory"].lower()

        # 2. Coding contract on the exact same core and single fact table
        engine.ingest_rule(
            "/strategies/test.md",
            title="Use uv run pytest",
            trigger_condition="Running unit tests in payments repo",
            action="Execute uv run pytest -q instead of system pytest",
            verification_result="All 120 tests green",
            user_id="alice",
            repo_id="payments",
            profile="coding",
            always_active=True,
            timestamp=1_700_000_200.0,
        )
        code_ctx = engine.build_context(
            "How should I run tests in payments?",
            user_id="alice",
            repo_id="payments",
            profile="coding",
            reference_time=1_700_000_300.0,
        )
        assert code_ctx["profile"] == "coding"
        assert code_ctx["disclosure"] == "progressive"
        assert "/strategies/test.md" in code_ctx["system_prefix"]
        assert "Directory Outline" in code_ctx["dynamic_memory"]

        # Verify single fact ledger: ZERO legacy vfs_nodes table created
        tables = {
            r[0]
            for r in engine.store.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "facts" in tables
        assert "fact_unified_meta" in tables
        assert "vfs_nodes" not in tables
    finally:
        engine.close()


def test_scope_chain_sharing_and_project_override_precedence(tmp_path):
    """User preferences share across chat & coding; repo/project rules override user global prefs."""
    engine = UnifiedMemoryEngine(tmp_path / "scopes.db")
    try:
        # User global preference (shared across conversational and coding)
        engine.ingest_rule(
            "/prefs/test_runner.md",
            title="Default Test Runner",
            trigger_condition="Running Python unit tests",
            action="Use plain pytest -q",
            verification_result="User default preference",
            user_id="alice",
            scope_key="user:alice",
            override_key="test_runner",
            timestamp=1_700_000_000.0,
        )
        engine.ingest_turn(
            "I always prefer concise answers with bullet points.",
            user_id="alice",
            scope_key="user:alice",
            timestamp=1_700_000_010.0,
        )

        # Repo-specific rule overriding test_runner ONLY inside repo:payments
        engine.ingest_rule(
            "/prefs/test_runner.md",
            title="Payments Test Runner",
            trigger_condition="Running Python unit tests in payments",
            action="Use uv run pytest -m unit --tb=short",
            verification_result="Required by payments CI",
            user_id="alice",
            scope_key="repo:payments",
            override_key="test_runner",
            timestamp=1_700_000_020.0,
        )

        # Inside repo:payments -> repo rule overrides user preference; user general preference still inherited!
        payments_chain = ScopeChain(("session:s1", "repo:payments", "user:alice", "global"))
        note_pay = engine.read_note("/prefs/test_runner.md", user_id="alice", scope_chain=payments_chain)
        assert note_pay is not None
        assert note_pay["scope_key"] == "repo:payments"
        assert "uv run pytest -m unit" in note_pay["content"]

        rec_pay = engine.recall("Running Python unit tests", user_id="alice", scope_chain=payments_chain)
        top_scopes = [f["scope_key"] for f in rec_pay["ranked"] if f.get("override_key") == "test_runner"]
        assert top_scopes == ["repo:payments"]
        assert any(o["scope_key"] == "user:alice" for o in rec_pay["scope_overridden"])

        # Shared user preference is visible in repo:payments too
        rec_style = engine.recall("concise answers bullet points", user_id="alice", scope_chain=payments_chain)
        assert any("concise answers" in f["plain"].lower() for f in rec_style["ranked"])

        # Inside repo:analytics -> repo:payments rule is invisible, user:alice preference applies!
        analytics_chain = ScopeChain(("session:s2", "repo:analytics", "user:alice", "global"))
        note_ana = engine.read_note("/prefs/test_runner.md", user_id="alice", scope_chain=analytics_chain)
        assert note_ana is not None
        assert note_ana["scope_key"] == "user:alice"
        assert "plain pytest" in note_ana["content"]
    finally:
        engine.close()


def test_v1_and_v2_vfs_shims_forward_without_legacy_storage(tmp_path):
    """M1 verification: /v1/* and /v2/vfs/* shims forward cleanly to UnifiedMemoryEngine with zero vfs_nodes table."""
    engine = UnifiedMemoryEngine(tmp_path / "shim_only.db")
    try:
        v1 = UnifiedV1Shim(engine)
        vfs = UnifiedVFSShim(engine)

        r = v1.retain_turn("Deploy service Orion on port 8443.", user_id="u1", session_id="s1")
        assert r["turn_id"] >= 1
        found = v1.recall("Orion port 8443", user_id="u1")
        assert not found["empty_recall"]

        w = vfs.write_rule(
            path="/ops/orion.md",
            title="Orion Port",
            trigger_condition="Starting Orion service",
            action="Bind to port 8443",
            verification_result="Healthcheck 200",
            user_id="u1",
        )
        assert w["written"] is True
        assert vfs.read_note("/ops/orion.md", user_id="u1")["title"] == "Orion Port"

        # Verify HTTP server runs both /v1/* and /v2/vfs/* strictly on unified core without creating vfs_nodes
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient
        from agentmem_core.server import create_app

        app = create_app(unified_engine=engine)
        with TestClient(app) as client:
            r1 = client.post("/v1/retain", json={"user_id": "u2", "text": "Use Python 3.12 for builds."})
            assert r1.status_code == 200
            assert r1.headers.get("Deprecation") == "true"

            r2 = client.post(
                "/v2/vfs/write-rule",
                json={
                    "user_id": "u2",
                    "path": "/build/python.md",
                    "title": "Python Build",
                    "trigger_condition": "Building wheel",
                    "action": "Use Python 3.12",
                    "verification_result": "Wheel built",
                },
            )
            assert r2.status_code == 200 and r2.json()["written"] is True
            assert r2.headers.get("Deprecation") == "true"

            r3 = client.post("/v2/vfs/read-note", json={"user_id": "u2", "path": "/build/python.md"})
            assert r3.status_code == 200 and "Python 3.12" in r3.json()["detail_body"]

        tables = {
            r[0]
            for r in engine.store.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "vfs_nodes" not in tables
    finally:
        engine.close()
