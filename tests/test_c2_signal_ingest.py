"""C2 event replay tests: fail->pass extraction, user corrections, drift detection, and G4 write cost cap."""
from __future__ import annotations

from agentmem_core import AgentEvent, UnifiedMemoryEngine, load_profile


def test_event_replay_fail_to_pass_and_user_correction_extraction(tmp_path):
    repo = tmp_path / "demo_repo"
    (repo / "src").mkdir(parents=True)
    target = repo / "src" / "db.py"
    target.write_text("POOL_SIZE = 5\n", encoding="utf-8")

    engine = UnifiedMemoryEngine(tmp_path / "c2.db")
    try:
        # 1. Routine noisy passing command -> zero writes
        r_noise = engine.ingest_event(
            AgentEvent(event_type="tool_result", tool_name="bash", command="git status", exit_code=0, text="clean"),
            user_id="dev",
            repo_id="billing",
            session_id="s1",
            repo_root=repo,
            profile="coding",
        )
        assert r_noise["written"] is False
        assert r_noise["signal"] is None

        # 2. Failure -> file edit -> passing command => automatic fail_to_pass rule with code anchor!
        r_fail = engine.ingest_event(
            AgentEvent(
                event_type="test_result",
                tool_name="bash",
                command="pytest tests/test_db.py",
                exit_code=1,
                text="OperationalError: SQLite database is locked on concurrent write",
                file_paths=("src/db.py",),
                timestamp=1000.0,
            ),
            user_id="dev",
            repo_id="billing",
            session_id="s1",
            repo_root=repo,
            profile="coding",
        )
        assert r_fail["signal"] == "failure_buffered"
        assert r_fail["written"] is False

        target.write_text("POOL_SIZE = 5\nBUSY_TIMEOUT_MS = 5000\n", encoding="utf-8")
        r_edit = engine.ingest_event(
            AgentEvent(
                event_type="file_edit",
                text="Enable WAL mode and busy_timeout=5000 in src/db.py",
                file_paths=("src/db.py",),
                timestamp=1010.0,
            ),
            user_id="dev",
            repo_id="billing",
            session_id="s1",
            repo_root=repo,
            profile="coding",
        )
        assert r_edit["written"] is False

        r_pass = engine.ingest_event(
            AgentEvent(
                event_type="test_result",
                tool_name="bash",
                command="uv run pytest tests/test_db.py",
                exit_code=0,
                text="12 passed in 0.4s",
                file_paths=("src/db.py",),
                timestamp=1020.0,
            ),
            user_id="dev",
            repo_id="billing",
            session_id="s1",
            repo_root=repo,
            profile="coding",
        )
        assert r_pass["signal"] == "fail_to_pass"
        assert r_pass["written"] is True
        assert r_pass["fact_id"] is not None
        assert r_pass["scope_key"] == "repo:billing"
        assert len(r_pass["anchors"]) == 1

        # Ambient recall immediately retrieves the learned rule on the same file/error
        amb = engine.ambient_context(
            "SQLite database is locked",
            active_files=["src/db.py"],
            user_id="dev",
            repo_id="billing",
            profile="coding",
        )
        assert not amb["empty_recall"]
        assert "busy_timeout=5000" in amb["dynamic_memory"]

        # 3. User correction signal -> automatically extracted
        r_corr = engine.ingest_event(
            AgentEvent(
                event_type="user_message",
                text="No, don't use unittest here, always use pytest fixtures with tmp_path.",
                override_key="test_style",
                timestamp=1030.0,
            ),
            user_id="dev",
            repo_id="billing",
            session_id="s1",
            scope_key="user:dev",
            profile="coding",
        )
        assert r_corr["signal"] == "user_correction"
        assert r_corr["written"] is True
        assert r_corr["scope_key"] == "user:dev"

        # 4. Subsequent code edit triggers automatic C1 staleness detection on the learned rule!
        target.write_text("ENGINE = 'postgres'\n", encoding="utf-8")
        r_drift = engine.ingest_event(
            AgentEvent(event_type="file_edit", file_paths=("src/db.py",), text="Switched to postgres", timestamp=1040.0),
            user_id="dev",
            repo_id="billing",
            session_id="s1",
            repo_root=repo,
            profile="coding",
        )
        assert r_drift["signal"] == "file_drift"
        assert r_drift["stale_check"]["stale_count"] >= 1
    finally:
        engine.close()


def test_signal_write_cost_upper_bound_and_deduplication(tmp_path):
    """G4: Verify write-cap and deduplication stop runaway writes in tight tool loops."""
    engine = UnifiedMemoryEngine(tmp_path / "c2_cap.db")
    capped_prof = load_profile("coding").with_overrides(max_writes_per_window=2, write_window_seconds=3600.0)
    try:
        # Duplicate user correction is suppressed
        ev1 = AgentEvent(event_type="user_message", text="Always use ruff check before commit.", timestamp=100.0)
        w1 = engine.ingest_event(ev1, user_id="dev", repo_id="r1", profile=capped_prof)
        w1_dup = engine.ingest_event(ev1, user_id="dev", repo_id="r1", profile=capped_prof)
        assert w1["written"] is True
        assert w1_dup["written"] is False
        assert w1_dup["reason"] == "duplicate_signal"

        # Second unique correction succeeds
        w2 = engine.ingest_event(
            AgentEvent(event_type="user_message", text="Never commit .env secrets.", timestamp=110.0),
            user_id="dev",
            repo_id="r1",
            profile=capped_prof,
        )
        assert w2["written"] is True
        assert w2["writes_in_window"] == 2

        # Third unique correction within window is strictly capped by G4 write budget!
        w3 = engine.ingest_event(
            AgentEvent(event_type="user_message", text="Always run mypy strict mode.", timestamp=120.0),
            user_id="dev",
            repo_id="r1",
            profile=capped_prof,
        )
        assert w3["written"] is False
        assert w3["reason"] == "write_budget_exceeded"
    finally:
        engine.close()
