"""Tests for Milestone M1: Unified core, dual ingest tracks, L0 FTS5, derived VFS/Profile views, adapters & /v1 shim."""
from __future__ import annotations

import pytest

from memengine import OptimisticLockError, RecallConfig, UnifiedMemoryEngine, UnifiedToolRegistry
from memengine.adapters import mcp_tool_definitions, openai_tool_definitions
from memengine.contrib import QdrantL2Index, S3TranscriptStore


def test_dual_ingest_tracks_and_l0_fts5_zero_vector_invariant(tmp_path):
    engine = UnifiedMemoryEngine(tmp_path / "m1.db")
    try:
        # 1. Conversational track: NO salience gate, idempotent, writes L0 + FTS5 + unified fact
        r1 = engine.ingest_turn(
            "I graduated with a degree in Mechanical Engineering at Stanford.",
            timestamp=1_700_000_000.0,
            session_id="s1",
            idempotency_key="turn-key-1",
        )
        assert r1["deduplicated"] is False
        assert r1["track"] == "conversational"
        assert r1["transcript_id"].startswith("tr_")

        r1_dup = engine.ingest_turn(
            "I graduated with a degree in Mechanical Engineering at Stanford.",
            timestamp=1_700_000_000.0,
            session_id="s1",
            idempotency_key="turn-key-1",
        )
        assert r1_dup["deduplicated"] is True
        assert r1_dup["turn_id"] == r1["turn_id"]

        # Verify L0 cold storage invariant: 0 L0 vectors, FTS5 indexed
        ep_stats = engine.episodes.stats()
        assert ep_stats["l0_vectors"] == 0
        assert ep_stats["fts_indexed_turns"] >= 1
        fts_hits = engine.episodes.search_fts("Mechanical Engineering")
        assert len(fts_hits) == 1

        # Verify stem-aware grounded answer works (graduate -> graduated)
        ans = engine.answer("What degree did I graduate with?", reference_time=1_700_010_000.0)
        assert ans["abstained"] is False
        assert "Mechanical Engineering" in ans["response"]

        # Verify strict abstention on unanswerable question
        ans_abs = engine.answer("What is the serial number of my quantum submarine?", reference_time=1_700_010_000.0)
        assert ans_abs["abstained"] is True
    finally:
        engine.close()


def test_procedural_rule_track_salience_lock_vfs_view_and_rollback(tmp_path):
    engine = UnifiedMemoryEngine(tmp_path / "m1_rules.db")
    try:
        # Rejected by 3-question salience gate (visible in current context)
        rejected = engine.ingest_rule(
            "/strategies/tmp.md",
            title="Transient noise",
            trigger_condition="When looking at screen",
            action="Read current line",
            visible_in_context=True,
        )
        assert rejected["accepted"] is False
        assert engine.read_note("/strategies/tmp.md") is None

        # Accepted verified rule stored directly in unified facts table (kind='rule')
        w1 = engine.ingest_rule(
            "/strategies/db/pool.md",
            title="PG Pool Sizing",
            trigger_condition="PG connections spike above 80%",
            action="Cap pool max_size to 20 and enable idle timeout",
            verification_result="Latency p95 dropped 40%",
            always_active=True,
            salience_tag="core_rule",
            expected_version=0,
        )
        assert w1["accepted"] is True
        assert w1["version"] == 1
        note = engine.read_note("/strategies/db/pool.md")
        assert note is not None
        assert note["kind"] == "rule"
        assert note["version"] == 1
        assert note["always_active"] is True

        # Optimistic lock rejects stale version
        with pytest.raises(OptimisticLockError):
            engine.ingest_rule(
                "/strategies/db/pool.md",
                title="Stale update",
                trigger_condition="When stale",
                action="Overwrite blindly",
                verification_result="Tested",
                expected_version=0,
            )

        # Update with correct expected_version=1 -> v2, then rollback to v1
        w2 = engine.ingest_rule(
            "/strategies/db/pool.md",
            title="PG Pool Sizing v2",
            trigger_condition="PG connections spike above 90%",
            action="Cap pool max_size to 16",
            verification_result="Zero connection errors",
            expected_version=1,
        )
        assert w2["version"] == 2
        assert engine.read_note("/strategies/db/pool.md")["title"] == "PG Pool Sizing v2"

        rb = engine.rollback(path="/strategies/db/pool.md")
        assert rb["count"] == 1
        assert engine.read_note("/strategies/db/pool.md")["title"] == "PG Pool Sizing"

        # Coding-agent profile prepends Stage-1 preload + Stage-2 ls -R directory outline + progressive disclosure
        ctx = engine.build_context(
            "PostgreSQL connection pool",
            profile="coding_agent",
            disclosure="progressive",
        )
        assert "/strategies/db/pool.md" in ctx["system_prefix"]
        assert "Directory Outline" in ctx["dynamic_memory"]
        assert ctx["disclosure"] == "progressive"

        # Directory compaction + lossless batch rollback
        engine.ingest_rule(
            "/strategies/db/index.md",
            title="GIN Index Rule",
            trigger_condition="Full text query slow",
            action="Add GIN index on tsvector",
            verification_result="10x faster",
        )
        comp = engine.compact_directory("/strategies/db/")
        assert comp["compacted"] is True
        assert comp["merged_count"] == 2
        assert engine.read_note("/strategies/db/GUIDE.md") is not None

        rb_batch = engine.rollback(batch_id=comp["batch_id"])
        assert rb_batch["count"] == 2
        assert engine.read_note("/strategies/db/pool.md") is not None
        assert engine.read_note("/strategies/db/index.md") is not None
    finally:
        engine.close()


def test_unified_tool_registry_adapters_and_v1_deprecation_shim(tmp_path):
    engine = UnifiedMemoryEngine(tmp_path / "m1_http.db")
    try:
        oai = openai_tool_definitions(engine)
        mcp = mcp_tool_definitions(engine)
        assert len(oai) == len(mcp) == len(UnifiedToolRegistry.specs())
        assert {t["function"]["name"] for t in oai} == {t["name"] for t in mcp}

        # Dispatch through UnifiedToolRegistry directly
        reg = engine.as_tools(user_id="u1", agent_id="a1", session_id="s1")
        ret = reg.dispatch("memory_retain", {"text": "Alice lives in Kyoto and works as an architect."})
        assert ret["turn_id"] >= 1
        ref = reg.dispatch("memory_reflect", {"query": "Where does Alice live?"})
        assert ref["abstained"] is False
        assert ref["belief_written"] is True

        # HTTP unified endpoints + deprecated /v1 shim headers on shared core (optional [server] extra)
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient
        from memengine.server import create_app

        app = create_app(unified_engine=engine)
        with TestClient(app) as client:
            r_tools = client.get("/v2/memory/tools")
            assert r_tools.status_code == 200
            assert len(r_tools.json()["openai_tools"]) == len(UnifiedToolRegistry.specs())

            r_retain = client.post("/v2/memory/retain", json={"user_id": "u2", "agent_id": "a2", "text": "Bob plays violin every Friday."})
            assert r_retain.status_code == 200
            r_ans = client.post("/v2/memory/answer", json={"user_id": "u2", "agent_id": "a2", "query": "What instrument does Bob play?"})
            assert r_ans.status_code == 200
            assert "violin" in r_ans.json()["response"].lower()

            # Legacy /v1/* route still works AND carries Deprecation + Sunset headers
            r_v1 = client.post("/v1/recall", json={"user_id": "u2", "agent_id": "a2", "query": "violin"})
            assert r_v1.status_code == 200
            assert r_v1.headers.get("Deprecation") == "true"
            assert "Sunset" in r_v1.headers
    finally:
        engine.close()


def test_feature_flag_bipolar_off_and_contrib_split():
    # Contrib adapters exist cleanly outside core
    assert QdrantL2Index is not None
    assert S3TranscriptStore is not None

    engine = UnifiedMemoryEngine(":memory:", config=RecallConfig(enable_bipolar=False, enable_recency_rerank=True))
    try:
        engine.ingest_turn("Project Orion deploys on Kubernetes 1.30.", timestamp=1_700_000_000.0)
        res = engine.recall("Project Orion Kubernetes", reference_time=1_700_000_100.0)
        assert res["empty_recall"] is False
        assert res["gate_stats"]["bipolar_enabled"] is False
    finally:
        engine.close()
