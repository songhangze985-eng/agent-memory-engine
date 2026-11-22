"""Comprehensive tests for the v2 VFS Memory Engine (L0 chunks, L1/L2/L3 tree, L2-only index, optimistic lock, dream rollback)."""
from contextlib import closing
import io
import os
import uuid

import pytest

from memengine.vfs import (
    LocalChunkTranscriptStore,
    OptimisticLockError,
    PostgresVFSStore,
    QdrantL2Index,
    S3TranscriptStore,
    VFSMemoryEngine,
)
from memengine.vector import HashedTfidfEmbedder


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
        assert len(tools) == 9
        mcp_names = {item["name"] for item in tools.mcp_definitions()}
        assert {"mem_tree", "mem_ls", "mem_read", "mem_write", "mem_edit", "mem_search", "mem_grep", "mem_read_transcripts", "mem_rollback"} == mcp_names

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
        assert tools.dispatch_tool("mem_search", {"query": "SQLite WAL busy_timeout"})[0]["path"] == "/strategies/sqlite/wal.md"
        raw = tools.dispatch_tool("mem_read_transcripts", {"transcript_ids": [rec.transcript_id]})
        assert raw[0]["text"] == "Always run WAL mode on SQLite."
        rolled = tools.dispatch_tool("mem_rollback", {"path": "/strategies/sqlite/wal.md"})
        assert "busy_timeout=5000" not in rolled["content"]
        # Host scope cannot be overridden via tool arguments
        with pytest.raises(ValueError, match="unexpected tool arguments"):
            tools.dispatch_tool("mem_read", {"path": "/strategies/sqlite/wal.md", "user_id": "bob"})


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
                health = engine.store.health_check()
                assert health["ok"] and health["backend"] == "postgresql-vfs" and health["l2_only_embedding"]
        finally:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
