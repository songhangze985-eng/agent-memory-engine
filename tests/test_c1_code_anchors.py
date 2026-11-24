"""C1 verification tests: code anchors (file, symbol, line_range, package_version) & staleness detection."""
from __future__ import annotations

from agentmem_core import CodeAnchor, UnifiedMemoryEngine, capture_anchor, load_profile


def test_file_and_symbol_anchors_mark_rule_stale_after_code_edit(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    auth_file = repo / "src" / "auth.py"
    auth_file.write_text(
        "def verify_token(tok: str) -> bool:\n"
        "    return tok.startswith('v1_')\n\n"
        "def helper() -> int:\n"
        "    return 42\n",
        encoding="utf-8",
    )

    engine = UnifiedMemoryEngine(tmp_path / "c1.db")
    try:
        sym_anchor = capture_anchor(repo, file_path="src/auth.py", symbol_name="verify_token")
        assert sym_anchor.anchor_type == "symbol"
        assert sym_anchor.content_hash != ""

        w = engine.ingest_rule(
            "/rules/auth_prefix.md",
            title="Legacy v1 token prefix",
            trigger_condition="Validating auth tokens in src/auth.py",
            action="Expect tokens prefixed with v1_",
            verification_result="Verified in auth.py",
            user_id="dev",
            repo_id="auth-svc",
            anchors=[sym_anchor],
            repo_root=repo,
            always_active=True,
        )
        assert w["accepted"] is True
        assert len(w["anchors"]) == 1

        # Fresh before edit -> visible in coding profile ambient & build_context
        amb_fresh = engine.ambient_context(
            "validate token",
            active_files=["src/auth.py"],
            active_symbols=["verify_token"],
            user_id="dev",
            repo_id="auth-svc",
            profile="coding",
        )
        assert not amb_fresh["empty_recall"]
        assert amb_fresh["excluded_stale_ids"] == []
        assert "v1_" in amb_fresh["dynamic_memory"]

        # Unrelated edit to helper() does NOT invalidate verify_token symbol anchor!
        auth_file.write_text(
            "def verify_token(tok: str) -> bool:\n"
            "    return tok.startswith('v1_')\n\n"
            "def helper() -> int:\n"
            "    return 99\n",
            encoding="utf-8",
        )
        chk1 = engine.verify_anchors(repo_root=repo, user_id="dev")
        assert chk1["stale_count"] == 0

        # Editing verify_token invalidates the symbol anchor -> rule becomes stale!
        auth_file.write_text(
            "def verify_token(tok: str) -> bool:\n"
            "    return tok.startswith('v2_jwt_')\n\n"
            "def helper() -> int:\n"
            "    return 99\n",
            encoding="utf-8",
        )
        chk2 = engine.verify_anchors(repo_root=repo, user_id="dev")
        assert chk2["stale_count"] == 1
        assert "symbol_modified:src/auth.py::verify_token" in chk2["stale_facts"][0]["stale_reason"]

        meta = engine.ledger.get_meta(w["fact_id"])
        assert meta["staleness_status"] == "stale"

        # Coding profile (stale_policy='exclude') MUST NOT inject the stale rule as active advice!
        amb_stale = engine.ambient_context(
            "validate token",
            active_files=["src/auth.py"],
            active_symbols=["verify_token"],
            user_id="dev",
            repo_id="auth-svc",
            profile="coding",
        )
        assert "v1_" not in amb_stale["dynamic_memory"]
        assert "v1_" not in amb_stale["system_prefix"]

        # Warn policy explicitly annotates [STALE: ...] so agent never blindly trusts it
        warn_prof = load_profile("coding").with_overrides(stale_policy="warn")
        ctx_warn = engine.build_context(
            "Validating auth tokens in src/auth.py",
            user_id="dev",
            repo_id="auth-svc",
            profile=warn_prof,
            disclosure="full",
        )
        assert "[STALE:" in ctx_warn["dynamic_memory"] or "[STALE:" in ctx_warn["system_prefix"]
    finally:
        engine.close()


def test_package_version_and_file_deletion_anchors_mark_stale(tmp_path):
    repo = tmp_path / "pkg_repo"
    repo.mkdir()
    pyproject = repo / "pyproject.toml"
    pyproject.write_text('[project]\ndependencies = ["pydantic==1.10.15"]\n', encoding="utf-8")

    engine = UnifiedMemoryEngine(tmp_path / "c1_pkg.db")
    try:
        pkg_anc = capture_anchor(repo, file_path="pyproject.toml", package_name="pydantic")
        assert pkg_anc.version_spec == "==1.10.15"

        w = engine.ingest_rule(
            "/rules/pydantic_v1.md",
            title="Pydantic v1 model serialization",
            trigger_condition="Serializing Pydantic BaseModel",
            action="Use model.dict() instead of model.model_dump()",
            verification_result="Works on pydantic==1.10.15",
            user_id="dev",
            repo_id="api",
            anchors=[pkg_anc],
            repo_root=repo,
        )
        assert w["accepted"] is True

        # Upgrade pydantic to v2 -> rule MUST flip to stale!
        pyproject.write_text('[project]\ndependencies = ["pydantic==2.7.1"]\n', encoding="utf-8")
        res = engine.verify_anchors(repo_root=repo, user_id="dev")
        assert res["stale_count"] == 1
        assert "package_version_changed:pydantic" in res["stale_facts"][0]["stale_reason"]

        # Deleting an anchored file also marks stale
        helper_file = repo / "scripts" / "legacy.sh"
        helper_file.parent.mkdir(parents=True)
        helper_file.write_text("echo old\n", encoding="utf-8")
        w2 = engine.ingest_rule(
            "/rules/legacy_script.md",
            title="Legacy deploy script",
            trigger_condition="Deploying service",
            action="Run scripts/legacy.sh",
            verification_result="Worked before migration",
            user_id="dev",
            repo_id="api",
            anchors=[CodeAnchor(file_path="scripts/legacy.sh")],
            repo_root=repo,
        )
        helper_file.unlink()
        res2 = engine.verify_anchors(repo_root=repo, user_id="dev", fact_id=w2["fact_id"])
        assert res2["stale_count"] == 1
        assert "file_missing:scripts/legacy.sh" in res2["stale_facts"][0]["stale_reason"]
    finally:
        engine.close()
