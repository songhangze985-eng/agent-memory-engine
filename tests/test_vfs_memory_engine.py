"""Comprehensive tests for the v2 VFS Memory Engine (L0 chunks, L1/L2/L3 tree, L2-only index, optimistic lock, dream rollback)."""
from contextlib import closing
import io
import os
import uuid

import pytest

from agentmem_core.vfs import (
    LocalChunkTranscriptStore,
    OptimisticLockError,
    PostgresVFSStore,
    QdrantL2Index,
    S3TranscriptStore,
    VFSMemoryEngine,
)
from agentmem_core.vector import HashedTfidfEmbedder


def _exercise_vfs_engine(engine: VFSMemoryEngine):
    alice = dict(user_id="alice", agent_id="coder")
    bob = dict(user_id="bob", agent_id="coder")

    # 1. L0 raw transcripts MUST NEVER create vector embeddings
    t1 = engine.append_transcript("Remember: I am severely allergic to sodium benzoate.", session_id="s1", timestamp=1000.0, **alice)
    t2 = engine.append_transcript("I prefer quiet history podcasts for my commute.", session_id="s1", timestamp=1001.0, **alice)
    for i in range(40):
        engine.append_transcript(f"Small talk noise turn {i} about weather and lunch.", session_id="s1", timestamp=1002.0 + i, **alice)
    engine.append_transcript("Bob secret token BOB-999.", session_id="s1", timestamp=1050.0, **bob)
    assert engine.store.vector_index_count(**alice) == 0

    # 2. L1 working scratchpad and L3 manifest MUST NOT enter L2 vector index
    l1 = engine.write(
        "/working/db-migration.md",
        title="DB Migration Draft",
        content="# Draft\nNeed to benchmark VectorChord RaBitQ before cutover.",
        layer="L1",
        verification_status="verified",
        source_transcript_ids=[t1.transcript_id],
        **alice,
    )
    assert l1["version"] == 1
    engine.refresh_l3_index(**alice)
    assert engine.store.vector_index_count(**alice) == 0

    # 3. L2 curated Markdown notes DO enter L2 vector + BM25 index
    pg_node = engine.write(
        "/strategies/database/postgres.md",
        title="PostgreSQL Vector Strategy",
        summary="Use VectorChord RaBitQ for disk-friendly vector search and BM25 for lexical ranking.",
        content="# PostgreSQL Vector Strategy\n- Prefer VectorChord RaBitQ over RAM-heavy HNSW.\n- Never vectorize L0 raw logs.",
        layer="L2",
        confidence_score=0.92,
        verification_status="verified",
        salience_tag="core_rule",
        source_transcript_ids=[t1.transcript_id],
        **alice,
    )
    assert pg_node["path"] == "/strategies/database/postgres.md"
    assert pg_node["version"] == 1
    assert engine.store.vector_index_count(**alice) == 1

    # Bob writes his own L2 note; multi-tenant isolation must be strict
    engine.write(
        "/strategies/database/postgres.md",
        title="Bob Secret DB Rule",
        content="Bob private credentials BOB-SECRET-777",
        layer="L2",
        verification_status="verified",
        **bob,
    )
    alice_hits = engine.search("PostgreSQL VectorChord RaBitQ", path_prefix="/strategies/", verified_only=True, **alice)
    assert len(alice_hits) == 1 and alice_hits[0]["path"] == "/strategies/database/postgres.md"
    assert "BOB-SECRET-777" not in alice_hits[0]["content"]
    assert not engine.search("BOB-SECRET-777", **alice)

    # 4. Optimistic locking prevents silent concurrent overwrite
    updated = engine.edit(
        "/strategies/database/postgres.md",
        old_str="Never vectorize L0 raw logs.",
        new_str="Never vectorize L0 raw logs; keep them in chunked object storage.",
        expected_version=1,
        **alice,
    )
    assert updated["version"] == 2
    with pytest.raises(OptimisticLockError) as exc_info:
        engine.edit(
            "/strategies/database/postgres.md",
            old_str="Prefer VectorChord RaBitQ",
            new_str="Conflicting edit",
            expected_version=1,
            **alice,
        )
    assert exc_info.value.expected_version == 1 and exc_info.value.actual_version == 2

    # 5. Dreaming consolidation (L0 + L1 -> L2) + L3 tree index + L0 provenance drilldown
    dream = engine.dream_consolidate(session_id="s1", **alice)
    assert "/taboos/constraints.md" in dream["updated_paths"]
    assert "/preferences/user.md" in dream["updated_paths"]
    assert "/strategies/db-migration.md" in dream["updated_paths"]
    assert dream["l0_scanned"] == 42
    assert engine.store.vector_index_count(**alice) == 4  # only the 4 L2 nodes!

    ctx = engine.build_context("sodium benzoate allergy and VectorChord database strategy", token_budget=2048, **alice)
    assert ctx["system_prefix"].startswith("# Memory Directory Index (L3)")
    assert "/taboos/constraints.md" in ctx["system_prefix"]
    assert ctx["l2_vector_count"] == 4
    assert t1.transcript_id in ctx["source_transcript_ids"]
    drilled = engine.read_transcripts([t1.transcript_id], **alice)
    assert len(drilled) == 1 and "sodium benzoate" in drilled[0]["text"]
    assert not engine.read_transcripts([t1.transcript_id], **bob)

    # 6. Bad dreaming run -> atomic batch rollback restores modified notes & purges hallucinated ones
    def corrupt_dream(transcripts, l1_nodes, existing_l2):
        return [
            {
                "path": "/strategies/database/postgres.md",
                "title": "Corrupted Note",
                "content": "BAD DREAM HALLUCINATION",
                "layer": "L2",
                "verification_status": "unverified",
            },
            {
                "path": "/strategies/hallucinated.md",
                "title": "Spurious Hallucination",
                "content": "Completely fabricated rule",
                "layer": "L2",
                "verification_status": "unverified",
            },
        ]

    bad = engine.dream_consolidate(synthesizer=corrupt_dream, **alice)
    assert "BAD DREAM HALLUCINATION" in engine.read("/strategies/database/postgres.md", **alice)["content"]
    assert engine.read("/strategies/hallucinated.md", **alice) is not None
    assert engine.store.vector_index_count(**alice) == 5

    engine.rollback_batch(bad["dream_batch_id"], **alice)
    restored = engine.read("/strategies/database/postgres.md", **alice)
    assert "Never vectorize L0 raw logs; keep them in chunked object storage." in restored["content"]
    assert engine.read("/strategies/hallucinated.md", **alice) is None
    assert engine.store.vector_index_count(**alice) == 4

    # 7. Demoting L2 -> L1 immediately removes vector embedding
    engine.write(
        "/strategies/db-migration.md",
        title="DB Migration Draft",
        content="Demoted back to L1 scratchpad",
        layer="L1",
        **alice,
    )
    assert engine.store.vector_index_count(**alice) == 3

    # 8. Path traversal protection
    with pytest.raises(ValueError):
        engine.write("../../etc/passwd", title="bad", content="nope", **alice)


def test_sqlite_vfs_end_to_end(tmp_path):
    engine = VFSMemoryEngine(tmp_path / "vfs.db", transcript_dir=tmp_path / "l0")
    with closing(engine):
        _exercise_vfs_engine(engine)
        stats = engine.transcripts.stats(user_id="alice", agent_id="coder")
        assert stats["chunk_files"] >= 1 and stats["indexed_transcripts"] == 42


def test_vfs_claude_style_tools_and_mcp_definitions(tmp_path):
    engine = VFSMemoryEngine(tmp_path / "tools.db", transcript_dir=tmp_path / "l0")
    with closing(engine):
        rec = engine.append_transcript("Always run WAL mode on SQLite.", user_id="alice", agent_id="dev", session_id="s1")
        tools = engine.as_tools(user_id="alice", agent_id="dev")
        assert len(tools) == 12
        mcp_names = {item["name"] for item in tools.mcp_definitions()}
        assert {
            "mem_tree", "mem_ls", "read_note", "mem_read", "mem_write_rule", "mem_write",
            "mem_edit", "mem_search", "mem_grep", "mem_compact_dir", "mem_read_transcripts", "mem_rollback",
        } == mcp_names

        # Salience-gated structured rule write via tool
        skipped = tools.dispatch_tool("mem_write_rule", {
            "path": "/strategies/sqlite/noise.md",
            "title": "Visible UI Noise",
            "trigger_condition": "Looking at screen",
            "action": "Notice red button",
            "visible_in_context": True,
        })
        assert skipped["written"] is False and skipped["skipped_reason"] == "visible_in_current_context"

        written = tools.dispatch_tool("mem_write", {
            "path": "/strategies/sqlite/wal.md",
            "title": "SQLite WAL Rule",
            "content": "# SQLite WAL\nAlways enable WAL mode.",
            "layer": "L2",
            "verification_status": "verified",
            "salience_tag": "core_rule",
            "source_transcript_ids": [rec.transcript_id],
        })
        assert written["version"] == 1
        assert "/strategies/" in tools.dispatch_tool("mem_ls", {"path": "/"})["directories"]
        assert "wal.md" in tools.dispatch_tool("mem_tree", {"path": "/"})
        edited = tools.dispatch_tool("mem_edit", {
            "path": "/strategies/sqlite/wal.md",
            "old_str": "Always enable WAL mode.",
            "new_str": "Always enable WAL mode with busy_timeout=5000.",
            "expected_version": 1,
        })
        assert edited["version"] == 2
        assert tools.dispatch_tool("mem_grep", {"pattern": "busy_timeout"})[0]["path"] == "/strategies/sqlite/wal.md"
        # Progressive disclosure in mem_search tool: default omits full body; read_note pulls full detail_body
        search_hits = tools.dispatch_tool("mem_search", {"query": "SQLite WAL busy_timeout"})
        assert search_hits[0]["path"] == "/strategies/sqlite/wal.md"
        assert "content" not in search_hits[0] and "detail_body" not in search_hits[0]
        assert len(search_hits[0]["index_summary"]) <= 50
        full_note = tools.dispatch_tool("read_note", {"path": "/strategies/sqlite/wal.md"})
        assert "busy_timeout=5000" in full_note["detail_body"]
        raw = tools.dispatch_tool("mem_read_transcripts", {"transcript_ids": [rec.transcript_id]})
        assert raw[0]["text"] == "Always run WAL mode on SQLite."
        rolled = tools.dispatch_tool("mem_rollback", {"path": "/strategies/sqlite/wal.md"})
        assert "busy_timeout=5000" not in rolled["content"]
        # Host scope cannot be overridden via tool arguments
        with pytest.raises(ValueError, match="unexpected tool arguments"):
            tools.dispatch_tool("mem_read", {"path": "/strategies/sqlite/wal.md", "user_id": "bob"})


def test_three_stage_retrieval_rerank_and_version_precision(tmp_path):
    from agentmem_core.chrono import DAY
    from agentmem_core.vfs.store import W_UNVERIFIED, W_VERIFIED

    engine = VFSMemoryEngine(tmp_path / "three_stage.db", transcript_dir=tmp_path / "l0")
    with closing(engine):
        u = dict(user_id="alice", agent_id="arch")
        base_ts = 1_700_000_000.0

        # Stage 1: Deterministic Pre-load (L1 /global.md + L2 always_active)
        engine.write("/global.md", title="Global Persona", content="Never deploy without migration dry-run.", layer="L1", verification_status="verified", **u)
        engine.write_rule(
            "/taboos/no_truncate.md",
            title="No Hard Truncation",
            trigger_condition="Assembling prompt memory context",
            action="Use progressive disclosure index_summary instead of cutting rules mid-sentence",
            verification_result="Verified zero causal breaks in evaluation",
            salience_tag="taboo",
            always_active=True,
            timestamp=base_ts - 180 * DAY,
            **u,
        )
        # Stage 2 & Stage 3 candidates: verified vs unverified + old vs new
        engine.write(
            "/strategies/db/unverified_guess.md",
            title="Guess Pool Size",
            content="PostgreSQL VectorChord RaBitQ pool size tuning checklist",
            layer="L2",
            verification_status="unverified",
            salience_tag="normal",
            timestamp=base_ts,
            **u,
        )
        engine.write(
            "/strategies/db/verified_rule.md",
            title="Verified Pool Size",
            content="PostgreSQL VectorChord RaBitQ pool size tuning checklist",
            layer="L2",
            verification_status="verified",
            salience_tag="normal",
            timestamp=base_ts,
            **u,
        )
        engine.write(
            "/notes/stale.md",
            title="Stale Cache Note",
            content="Redis cluster failover timeout parameter guidance",
            layer="L2",
            verification_status="verified",
            salience_tag="normal",
            timestamp=base_ts - 120 * DAY,
            **u,
        )
        engine.write(
            "/notes/fresh.md",
            title="Fresh Cache Note",
            content="Redis cluster failover timeout parameter guidance",
            layer="L2",
            verification_status="verified",
            salience_tag="normal",
            timestamp=base_ts - 1 * DAY,
            **u,
        )

        # Stage 1 unconditional pre-load even on completely unrelated query
        ctx = engine.build_context("What is the capital of France?", token_budget=2048, **u)
        preloaded_paths = {item["path"] for item in ctx["preloaded"]}
        assert "/global.md" in preloaded_paths
        assert "/taboos/no_truncate.md" in preloaded_paths
        assert "Never deploy without migration dry-run." in ctx["dynamic_memory"]

        # Stage 2 tree_outline exposes only paths + <=50-char index_summary (never full body)
        outline = engine.tree_outline(**u)
        assert "/strategies/db/" in outline and "verified_rule.md" in outline
        assert "PostgreSQL VectorChord RaBitQ pool size tuning checklist" not in outline

        # Stage 3 Rerank formula exact check: Final Score = Semantic_Sim * W_verified * W_recency
        hits = engine.search("PostgreSQL VectorChord RaBitQ pool size tuning checklist", reference_time=base_ts, **u)
        assert hits[0]["path"] == "/strategies/db/verified_rule.md"
        assert hits[1]["path"] == "/strategies/db/unverified_guess.md"
        b0, b1 = hits[0]["score_breakdown"], hits[1]["score_breakdown"]
        assert b0["w_verified"] == W_VERIFIED and b1["w_verified"] == W_UNVERIFIED
        assert abs(b0["final_score"] - b0["semantic_sim"] * b0["w_verified"] * b0["w_recency"]) < 1e-4

        # Recency check: fresh note beats 120-day-old identical note; always_active taboo never decays
        rhits = engine.search("Redis cluster failover timeout parameter guidance", reference_time=base_ts, **u)
        assert rhits[0]["path"] == "/notes/fresh.md"
        assert rhits[0]["score_breakdown"]["w_recency"] > rhits[1]["score_breakdown"]["w_recency"]
        thits = engine.search("progressive disclosure index_summary cutting rules", reference_time=base_ts, **u)
        assert thits[0]["path"] == "/taboos/no_truncate.md"
        assert thits[0]["score_breakdown"]["w_recency"] == 1.0

        # Grafted exact-identifier & version-precision guardrails
        engine.write("/releases/py3112.md", title="Python v3.11.2", content="Python v3.11.2 enables fast taskgroups.", layer="L2", verification_status="verified", **u)
        engine.write("/releases/py3114.md", title="Python v3.11.4", content="Python v3.11.4 patches SSL.", layer="L2", verification_status="verified", **u)
        engine.write("/releases/py3114rc.md", title="Python v3.11.4-rc.1", content="Python v3.11.4-rc.1 preview.", layer="L2", verification_status="verified", **u)
        vhits = engine.search("Python v3.11.2", **u)
        assert [h["path"] for h in vhits] == ["/releases/py3112.md"]
        assert engine.search("Python v3.12.0", **u) == []
        # Stable release outranks prerelease on containment query
        branch_hits = engine.search("Python v3.11", **u)
        assert branch_hits[0]["path"] in {"/releases/py3112.md", "/releases/py3114.md"}
        assert branch_hits[0]["path"] != "/releases/py3114rc.md"


def test_salience_gate_progressive_disclosure_and_directory_compaction(tmp_path):
    engine = VFSMemoryEngine(tmp_path / "compact.db", transcript_dir=tmp_path / "l0", max_notes_per_dir=10, max_dir_tokens=3000)
    with closing(engine):
        u = dict(user_id="alice", agent_id="dev")
        # Salience 3-question gate
        assert engine.write_rule("/x/a.md", title="A", trigger_condition="t", action="a", visible_in_context=True, **u)["written"] is False
        assert engine.write_rule("/x/b.md", title="B", trigger_condition="t", action="a", reusable_pattern=False, future_pitfall_risk=False, **u)["written"] is False

        # State machine forces unverified if verification_result is empty/placeholder
        hyp = engine.write_rule(
            "/strategies/db/hyp.md",
            title="Tentative Index Guess",
            trigger_condition="When CPU spikes on pgvector",
            action="Switch HNSW to VectorChord RaBitQ",
            verification_result="TBD",
            verification_status="verified",
            confidence_score=0.98,
            **u,
        )
        assert hyp["written"] is True
        assert hyp["verification_status"] == "unverified"
        assert hyp["confidence_score"] <= 0.65
        assert "[触发条件]" in hyp["detail_body"] and "[动作]" in hyp["detail_body"] and "[验证结果]" in hyp["detail_body"]
        assert len(hyp["index_summary"]) <= 50

        # Insert 11 notes into /strategies/db/ -> triggers compaction threshold (>10)
        tids = []
        for i in range(11):
            rec = engine.append_transcript(f"Proven DB rule evidence #{i}", session_id="s1", **u)
            tids.append(rec.transcript_id)
            engine.write_rule(
                f"/strategies/db/rule_{i:02d}.md",
                title=f"DB Rule {i}",
                trigger_condition=f"Database write burst #{i}",
                action=f"Batch commits in WAL mode #{i} to prevent disk thrash",
                verification_result=f"Benchmarked #{i}: P95 < 35ms",
                notes=("Long operational rationale detail sentence for progressive disclosure check. " * 20),
                source_transcript_ids=[rec.transcript_id],
                **u,
            )

        alerts = engine.check_compaction_needed(parent_path="/strategies/db/", **u)
        assert len(alerts) == 1 and alerts[0]["note_count"] == 12

        # Progressive disclosure saves tokens vs full dump and never truncates detail_body mid-sentence
        ctx_prog = engine.build_context("Database write burst WAL mode disk thrash", token_budget=2048, disclosure="progressive", **u)
        ctx_full = engine.build_context("Database write burst WAL mode disk thrash", token_budget=2048, disclosure="full", **u)
        assert ctx_prog["token_breakdown"]["dynamic_tokens"] < ctx_full["token_breakdown"]["dynamic_tokens"]
        assert ctx_prog["compaction_alerts"]
        for hit in ctx_prog["l2_hits"]:
            assert len(hit["index_summary"]) <= 50
            assert hit["detail_body"] == engine.read_note(hit["path"], **u)["detail_body"]

        # Compact directory into one unified GUIDE.md and verify atomic rollback
        compacted = engine.compact_directory("/strategies/db/", **u)
        assert compacted["compacted"] is True
        assert compacted["guide_path"] == "/strategies/db/GUIDE.md"
        assert compacted["merged_count"] == 12
        guide = engine.read_note("/strategies/db/GUIDE.md", **u)
        assert "## Verified Rules" in guide["detail_body"]
        assert "## Unverified Hypotheses" in guide["detail_body"]
        assert set(tids).issubset(set(guide["source_transcript_ids"]))
        assert engine.store.directory_stats(**u)[0]["note_count"] == 1

        # Rollback restores all 12 original fragments cleanly
        engine.rollback_batch(compacted["dream_batch_id"], **u)
        assert engine.read_note("/strategies/db/GUIDE.md", **u) is None
        assert engine.store.directory_stats(**u)[0]["note_count"] == 12


def test_s3_transcript_store_adapter():
    bucket_objects = {}

    class FakeS3:
        def put_object(self, *, Bucket, Key, Body, ContentType=None):
            bucket_objects[(Bucket, Key)] = Body

        def get_object(self, *, Bucket, Key):
            return {"Body": io.BytesIO(bucket_objects[(Bucket, Key)])}

        def list_objects_v2(self, *, Bucket, Prefix, ContinuationToken=None):
            contents = [{"Key": k} for (b, k) in sorted(bucket_objects) if b == Bucket and k.startswith(Prefix)]
            return {"Contents": contents, "IsTruncated": False}

    store = S3TranscriptStore("mem-bucket", client=FakeS3(), chunk_size=2)
    r1 = store.append(user_id="alice", agent_id="a1", session_id="s1", role="user", text="first", timestamp=100.0)
    r2 = store.append(user_id="alice", agent_id="a1", session_id="s1", role="assistant", text="second", timestamp=101.0)
    assert len(bucket_objects) == 1  # auto-flushed gzip chunk at chunk_size=2
    fetched = store.get_by_ids([r1.transcript_id, r2.transcript_id], user_id="alice", agent_id="a1")
    assert [r.text for r in fetched] == ["first", "second"]
    assert not store.get_by_ids([r1.transcript_id], user_id="bob", agent_id="a1")


def test_qdrant_l2_only_adapter(tmp_path):
    points = {}

    def fake_transport(method, url, body):
        if method == "GET" and "/collections/" in url:
            return {"status": "ok"}
        if method == "PUT" and "/points" in url:
            for pt in body["points"]:
                points[pt["id"]] = pt
            return {"status": "ok"}
        if method == "POST" and "/points/delete" in url:
            for pid in body["points"]:
                points.pop(pid, None)
            return {"status": "ok"}
        if method == "POST" and "/points/search" in url:
            qvec = body["vector"]
            must = body["filter"]["must"]
            scored = []
            for pt in points.values():
                p = pt["payload"]
                if not all(
                    (p.get(c["key"]) == c["match"]["value"])
                    if "value" in c["match"]
                    else (p.get(c["key"]) in c["match"]["any"])
                    for c in must
                ):
                    continue
                dot = sum(a * b for a, b in zip(qvec, pt["vector"]))
                scored.append({"id": pt["id"], "score": dot, "payload": p})
            scored.sort(key=lambda item: -item["score"])
            return {"result": scored[: body["limit"]]}
        raise AssertionError(f"unexpected call: {method} {url}")

    qdrant = QdrantL2Index("http://127.0.0.1:6333", vector_dim=512, transport=fake_transport)
    engine = VFSMemoryEngine(tmp_path / "qdrant_vfs.db", transcript_dir=tmp_path / "l0", external_vector_index=qdrant)
    with closing(engine):
        engine.append_transcript("L0 raw dialogue should never reach Qdrant.", user_id="alice", agent_id="a1")
        engine.write("/working/scratch.md", title="L1", content="L1 scratchpad", layer="L1", user_id="alice", agent_id="a1")
        assert len(points) == 0  # neither L0 nor L1 reached Qdrant!
        engine.write(
            "/strategies/qdrant.md",
            title="Qdrant L2 Rule",
            content="Use Qdrant strictly for L2 curated rules.",
            layer="L2",
            verification_status="verified",
            user_id="alice",
            agent_id="a1",
        )
        assert len(points) == 1
        hits = engine.search("Qdrant curated rules", user_id="alice", agent_id="a1")
        assert hits and hits[0]["path"] == "/strategies/qdrant.md"
        assert engine.store.health_check()["vector"] == "qdrant-l2"
        engine.delete("/strategies/qdrant.md", user_id="alice", agent_id="a1")
        assert len(points) == 0


def test_vfs_lancedb_l2_only_backend(tmp_path):
    pytest.importorskip("lancedb")
    vfs_dir = tmp_path / "lance_vfs"
    engine = VFSMemoryEngine(f"lancedb:///{vfs_dir}")
    with closing(engine):
        assert engine.store.health_check()["vector"] == "lancedb-l2"
        engine.append_transcript("L0 raw transcript must never enter LanceDB.", user_id="alice", agent_id="dev")
        engine.write("/working/draft.md", title="Draft", content="L1 scratchpad", layer="L1", user_id="alice", agent_id="dev")
        assert engine.store.vector_index_count(user_id="alice", agent_id="dev") == 0

        engine.write_rule(
            "/strategies/db/lance.md",
            title="LanceDB RaBitQ Rule",
            trigger_condition="Scaling local L2 rules beyond 1,000 notes",
            action="Use embedded LanceDB IVF_RQ disk-mapped vector index",
            verification_result="P95 stays below 35ms at 10K scale",
            user_id="alice",
            agent_id="dev",
        )
        assert engine.store.vector_index_count(user_id="alice", agent_id="dev") == 1
        opt = engine.optimize_index()
        assert opt["rows"] == 1
        hits = engine.search("LanceDB IVF_RQ disk-mapped vector index", user_id="alice", agent_id="dev")
        assert hits and hits[0]["path"] == "/strategies/db/lance.md"
        engine.delete("/strategies/db/lance.md", user_id="alice", agent_id="dev")
        assert engine.search("LanceDB IVF_RQ disk-mapped vector index", user_id="alice", agent_id="dev") == []


def test_vfs_thread_safe_concurrent_optimistic_locking(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    engine = VFSMemoryEngine(tmp_path / "concurrent.db", transcript_dir=tmp_path / "l0")
    with closing(engine):
        # 1. Concurrent first-create race on the same path -> exactly 1 winner, rest OptimisticLockError
        def create_worker(i: int):
            try:
                engine.write("/shared/race.md", title=f"W{i}", content=f"Body {i}", expected_version=0, user_id="alice", agent_id="team")
                return "ok"
            except OptimisticLockError:
                return "conflict"

        with ThreadPoolExecutor(max_workers=8) as pool:
            create_results = list(pool.map(create_worker, range(8)))
        assert create_results.count("ok") == 1
        assert create_results.count("conflict") == 7

        # 2. Concurrent version-1 edit race -> exactly 1 winner, 7 conflicts, zero SQLite lock errors
        current = engine.read_note("/shared/race.md", user_id="alice", agent_id="team")
        assert current["version"] == 1

        def edit_worker(i: int):
            try:
                engine.edit(
                    "/shared/race.md",
                    old_str=current["content"],
                    new_str=f"Updated by worker {i}",
                    expected_version=1,
                    user_id="alice",
                    agent_id="team",
                )
                return "ok"
            except OptimisticLockError:
                return "conflict"

        with ThreadPoolExecutor(max_workers=8) as pool:
            edit_results = list(pool.map(edit_worker, range(8)))
        assert edit_results.count("ok") == 1
        assert edit_results.count("conflict") == 7
        assert engine.read_note("/shared/race.md", user_id="alice", agent_id="team")["version"] == 2


def test_vfs_http_server_and_cli_connectivity(tmp_path, capsys, monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    _ = fastapi
    from fastapi.testclient import TestClient
    import agentmem_core.client as client_mod
    from agentmem_core.cli import main as cli_main
    from agentmem_core.client import MemoryClient
    from agentmem_core.server import create_app

    v1_db = tmp_path / "v1.db"
    vfs_db = tmp_path / "vfs.db"
    app = create_app(db_url=str(v1_db), vfs_db_url=str(vfs_db), api_key="top-secret")
    headers = {"Authorization": "Bearer top-secret"}

    with TestClient(app) as client:
        assert client.get("/v2/vfs/health").status_code == 401
        health = client.get("/v2/vfs/health", headers=headers)
        assert health.status_code == 200 and health.json()["backend"] == "sqlite-vfs"

        tr = client.post("/v2/vfs/transcripts", headers=headers, json={"text": "Always run WAL mode.", "user_id": "u1", "agent_id": "a1"})
        assert tr.status_code == 200
        tid = tr.json()["transcript_id"]

        r1 = client.post("/v2/vfs/write-rule", headers=headers, json={
            "path": "/strategies/db/wal.md",
            "title": "WAL Mode",
            "trigger_condition": "Opening SQLite database",
            "action": "Enable PRAGMA journal_mode=WAL",
            "verification_result": "Passed concurrent stress test",
            "always_active": True,
            "source_transcript_ids": [tid],
            "user_id": "u1",
            "agent_id": "a1",
        })
        assert r1.status_code == 200 and r1.json()["written"] is True and r1.json()["version"] == 1

        # Stale optimistic lock returns HTTP 409 Conflict
        conflict = client.post("/v2/vfs/write", headers=headers, json={
            "path": "/strategies/db/wal.md", "title": "Stale", "content": "Overwrite",
            "expected_version": 0, "user_id": "u1", "agent_id": "a1",
        })
        assert conflict.status_code == 409 and conflict.json()["actual_version"] == 1

        # Missing note returns HTTP 404 Not Found
        missing = client.post("/v2/vfs/read-note", headers=headers, json={"path": "/missing.md", "user_id": "u1", "agent_id": "a1"})
        assert missing.status_code == 404

        found = client.post("/v2/vfs/read-note", headers=headers, json={"path": "/strategies/db/wal.md", "user_id": "u1", "agent_id": "a1"})
        assert found.status_code == 200 and "PRAGMA journal_mode=WAL" in found.json()["detail_body"]

        tree = client.post("/v2/vfs/tree", headers=headers, json={"path": "/", "user_id": "u1", "agent_id": "a1"})
        assert tree.status_code == 200 and "wal.md" in tree.json()["outline"]

        ctx = client.post("/v2/vfs/context", headers=headers, json={"query": "unrelated question", "disclosure": "progressive", "user_id": "u1", "agent_id": "a1"})
        assert ctx.status_code == 200 and any(p["path"] == "/strategies/db/wal.md" for p in ctx.json()["preloaded"])

        schema = client.get("/v2/vfs/tools/schema", headers=headers)
        assert schema.status_code == 200 and len(schema.json()) == 12
        disp = client.post("/v2/vfs/tools/dispatch", headers=headers, json={"name": "read_note", "arguments": {"path": "/strategies/db/wal.md"}, "user_id": "u1", "agent_id": "a1"})
        assert disp.status_code == 200 and disp.json()["path"] == "/strategies/db/wal.md"

        # Standard-library MemoryClient v2 VFS methods end-to-end
        def _urlopen(request, timeout):
            resp = client.request(
                request.get_method(),
                request.full_url.removeprefix("http://testserver"),
                content=request.data,
                headers=request.headers,
            )
            resp.raise_for_status()
            return io.BytesIO(resp.content)

        monkeypatch.setattr(client_mod, "urlopen", _urlopen)
        sdk = MemoryClient("http://testserver", api_key="top-secret")
        assert sdk.vfs_health()["backend"] == "sqlite-vfs"
        sdk_rule = sdk.vfs_write_rule(
            "/strategies/db/busy.md",
            title="Busy Timeout",
            trigger_condition="Concurrent local readers",
            action="Set PRAGMA busy_timeout=5000",
            verification_result="Zero SQLITE_BUSY errors",
            user_id="u1",
            agent_id="a1",
        )
        assert sdk_rule["written"] is True
        assert "busy_timeout=5000" in sdk.vfs_read_note("/strategies/db/busy.md", user_id="u1", agent_id="a1")["detail_body"]
        assert sdk.vfs_search("PRAGMA busy_timeout", user_id="u1", agent_id="a1")[0]["path"] == "/strategies/db/busy.md"
        assert "busy.md" in sdk.vfs_tree("/", user_id="u1", agent_id="a1")["outline"]
        assert sdk.vfs_compact("/strategies/db/", user_id="u1", agent_id="a1")["compacted"] is True

    # CLI end-to-end check on the same VFS database
    cli_db = tmp_path / "cli_vfs.db"
    assert cli_main(["--db-url", str(cli_db), "vfs", "write-rule", "/strategies/cli.md", "--title", "CLI Rule", "--trigger", "Running CLI", "--action", "Use vfs subcommand", "--verification", "Tested"]) == 0
    assert "CLI Rule" in capsys.readouterr().out
    assert cli_main(["--db-url", str(cli_db), "vfs", "tree"]) == 0
    assert "cli.md" in capsys.readouterr().out
    assert cli_main(["--db-url", str(cli_db), "vfs", "context", "CLI subcommand", "--disclosure", "progressive"]) == 0
    assert "progressive" in capsys.readouterr().out


def test_postgres_vfs_extension_tiers_without_server():
    calls = []

    class Recorder:
        def execute(self, sql, args=()):
            calls.append(sql)
            return self

        def fetchone(self):
            return (1,)

        def fetchall(self):
            return []

    for flags, expected_vec, expected_lex, expected_ddl in (
        ({"pgvector": True, "vchord": True, "vchord_bm25": True}, "vchordrq", "vchord_bm25", "USING vchordrq"),
        ({"pgvector": True, "vectorscale": True, "pg_search": True}, "diskann", "pg_search", "USING diskann"),
    ):
        calls.clear()
        store = PostgresVFSStore.__new__(PostgresVFSStore)
        store.embedder = HashedTfidfEmbedder()
        store.external_vector_index = None
        store.pgvector = flags.get("pgvector", False)
        store.vchord = flags.get("vchord", False)
        store.vectorscale = flags.get("vectorscale", False)
        store.pg_search = flags.get("pg_search", False)
        store.vchord_bm25 = flags.get("vchord_bm25", False)
        store.conn = Recorder()
        store._bootstrap()
        ddl = "\n".join(calls)
        assert expected_ddl in ddl
        health = store.health_check()
        assert health["vector"] == expected_vec and health["lexical"] == expected_lex and health["l2_only_embedding"]


def test_postgres_vfs_live_contract(tmp_path):
    dsn = os.getenv("TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("TEST_POSTGRES_DSN is required for live PostgreSQL VFS contract test")
    import psycopg
    from psycopg import sql

    schema = "test_vfs_" + uuid.uuid4().hex[:12]
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        try:
            configured = psycopg.conninfo.make_conninfo(dsn, options=f"-csearch_path={schema},public")
            from urllib.parse import urlencode

            uri = "postgresql:///?" + urlencode(psycopg.conninfo.conninfo_to_dict(configured))
            engine = VFSMemoryEngine(uri, transcript_dir=tmp_path / "pg_l0")
            with closing(engine):
                _exercise_vfs_engine(engine)
                # Exercise new 3-stage retrieval, exact-identifier table, salience rule, and directory compaction on live PG
                pg_u = dict(user_id="pg_opt", agent_id="arch")
                engine.write("/global.md", title="Global", content="PG global identity.", layer="L1", verification_status="verified", **pg_u)
                r = engine.write_rule(
                    "/strategies/pg/r1.md",
                    title="PG Rule 1",
                    trigger_condition="Using PostgreSQL 16",
                    action="Prefer VectorChord RaBitQ over HNSW",
                    verification_result="Verified on live PG16",
                    always_active=True,
                    **pg_u,
                )
                assert r["written"] and r["verification_status"] == "verified"
                engine.write("/releases/py3112.md", title="Python v3.11.2", content="Python v3.11.2 active.", layer="L2", verification_status="verified", **pg_u)
                engine.write("/releases/py3114.md", title="Python v3.11.4", content="Python v3.11.4 active.", layer="L2", verification_status="verified", **pg_u)
                assert [h["path"] for h in engine.search("Python v3.11.2", **pg_u)] == ["/releases/py3112.md"]
                assert engine.search("Python v3.12.0", **pg_u) == []
                ctx = engine.build_context("unrelated topic", disclosure="progressive", **pg_u)
                assert {p["path"] for p in ctx["preloaded"]} == {"/global.md", "/strategies/pg/r1.md"}
                engine.write_rule("/strategies/pg/r2.md", title="PG Rule 2", trigger_condition="When writing L0", action="Keep L0 unvectorized", verification_result="Verified", **pg_u)
                comp = engine.compact_directory("/strategies/pg/", **pg_u)
                assert comp["compacted"] and comp["guide_path"] == "/strategies/pg/GUIDE.md"
                engine.rollback_batch(comp["dream_batch_id"], **pg_u)
                assert engine.read_note("/strategies/pg/GUIDE.md", **pg_u) is None
                health = engine.store.health_check()
                assert health["ok"] and health["backend"] == "postgresql-vfs" and health["l2_only_embedding"] and health["exact_identifiers_l2"]
        finally:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
