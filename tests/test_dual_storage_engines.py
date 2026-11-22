"""Contract, security, rollback, ANN index, and extension-tier tests for dual storage engines."""
from contextlib import closing
from unittest.mock import patch

import numpy as np
import pytest

from memengine.engine import CognitiveMemoryEngine
from memengine.router import match_entities
from memengine.storage import open_store
from memengine.storage.base import RecallFilter
from memengine.storage.postgres import PostgresStore, _query
from memengine.storage.vector_cache import VectorView
from memengine.vector import HashedTfidfEmbedder
from tests.test_storage_parity import exercise, exercise_workspace


@pytest.mark.parametrize("scheme", ["lancedb:///", "lance:///", "sqlite+lance:///"])
def test_lance_hybrid_parity_and_uri_schemes(tmp_path, scheme):
    pytest.importorskip("lancedb")
    target = tmp_path / "ledger.db" if scheme == "sqlite+lance:///" else tmp_path / "hybrid_dir"
    uri = f"{scheme}{target}"
    with closing(open_store(uri)) as store:
        exercise(store)
        health = store.health_check()
        assert health["ok"] and health["backend"] == "sqlite+lancedb"
        assert health["vector"] in {"lancedb-arrow", "lancedb-ivf-rq"}
        assert "fact_vectors_512" in health["lance_tables"]


def test_lance_hybrid_workspace_survives_reopen(tmp_path):
    pytest.importorskip("lancedb")
    uri = f"lancedb:///{tmp_path / 'ws'}"
    exercise_workspace(lambda: CognitiveMemoryEngine(db_url=uri))


def test_lance_hybrid_rollback_external_writes_and_injection_safety(tmp_path):
    pytest.importorskip("lancedb")
    base = tmp_path / "tx_test"
    uri = f"lancedb:///{base}"

    def put(store, text, scope=("alice", "agent", "s1")):
        with store.transaction():
            block = store.create_block("Atlas", 1, scope=scope)
            return store.insert_fact(block_id=block, formatted=text, plain=text, ingest_time=1, scope=scope)

    with closing(open_store(uri)) as store:
        kept = put(store, "Atlas cobalt blueprint")
        assert store.search_vector_topk("cobalt", user_id="alice", agent_id="agent", ids_only=True)[0]["id"] == kept

        # Rollback must discard staged LanceDB writes and keep in-tx visibility consistent.
        with pytest.raises(RuntimeError):
            with store.transaction():
                put(store, "Atlas transient emerald")
                assert store.search_vector_topk("emerald", user_id="alice", agent_id="agent", ids_only=True)
                raise RuntimeError("abort")
        assert not store.search_vector_topk("emerald", user_id="alice", agent_id="agent", ids_only=True)

        # SQL-injection-shaped tenant identifiers must never bypass DataFusion prefilter isolation.
        evil_user = "alice' OR '1'='1"
        assert not store.search_vector_topk("cobalt", user_id=evil_user, agent_id="agent", ids_only=True)
        with pytest.raises(ValueError, match="null bytes"):
            store.search_vector_topk("cobalt", user_id="alice\x00", agent_id="agent", ids_only=True)

        # External raw SQLite write must be detected via PRAGMA data_version and reconciled.
        with closing(open_store(base / "ledger.db")) as raw_sqlite:
            external = put(raw_sqlite, "Atlas crimson external")
        hits = store.search_vector_topk("crimson", user_id="alice", agent_id="agent", ids_only=True)
        assert hits and hits[0]["id"] == external


def test_lance_hybrid_builds_rabitq_ivf_rq_ann_and_fts_index(tmp_path):
    pytest.importorskip("lancedb")
    uri = f"lancedb:///{tmp_path / 'ann_test'}"
    with closing(open_store(uri)) as store:
        with store.transaction():
            block = store.create_block("Scale", 1)
            target_id = None
            for idx in range(300):
                text = f"Archive record ITEM_{idx:04d} tracks subsystem node_{idx % 17} telemetry."
                fid = store.insert_fact(block_id=block, formatted=text, plain=text, ingest_time=float(idx))
                if idx == 142:
                    target_id = fid
        indexed = store.optimize_index(include_fts=True)
        assert "fact_vectors_512" in indexed
        health = store.health_check()
        assert health["vector"] == "lancedb-ivf-rq"
        assert "fact_vectors_512" in health["ann_indexed"]
        hits = store.search_vector_topk("ITEM_0142 telemetry", top_n=5, ids_only=True)
        assert any(row["id"] == target_id for row in hits)


def test_entity_key_index_matches_full_scan_parity(tmp_path):
    engine = CognitiveMemoryEngine(tmp_path / "entities.db")
    with closing(engine.store) as store:
        samples = [
            "Alice Chen prefers quiet podcasts during her morning commute.",
            "Bob likes classic movies and sci-fi books.",
            "老周对苯甲酸钠过敏，不能吃含防腐剂的零食。",
            "Project QUEUE_00421 deployed in A组 cluster.",
        ]
        for idx, text in enumerate(samples):
            engine.add(text, timestamp=idx + 1, speaker="Alice Chen" if idx == 0 else None)

        queries = [
            "What did Chen say about Alice?",          # non-adjacent multi-word Latin
            "Does Bob watch a movie?",                 # plural folding: movies <-> movie
            "老周对哪种防腐剂过敏？",                    # CJK substring
            "Check A组 status for QUEUE_00421",         # mixed CJK + exact token
            "Totally unrelated query xyz",             # zero hits
        ]
        for query in queries:
            full_keys = store.entity_keys()
            cached_keys = store.entity_keys(query=query)
            assert match_entities(query, cached_keys) == match_entities(query, full_keys)


def test_write_path_candidate_blocks_avoids_full_snapshot(tmp_path):
    engine = CognitiveMemoryEngine(tmp_path / "write_bounded.db")
    with closing(engine.store) as store:
        for idx in range(15):
            engine.add(f"Project ITEM_{idx:03d} runs on node {idx}.", timestamp=idx + 1)
        with patch.object(store, "snapshot", side_effect=AssertionError("full snapshot must not be called on write")):
            result = engine.add("Project ITEM_003 moved to node 99.", timestamp=100)
        assert result["operations"]


def test_vector_view_amortized_append_reuses_backing_buffer():
    view = VectorView()
    v1 = np.ones(8, dtype="<f4").tobytes()
    v2 = (np.ones(8, dtype="<f4") * 2).tobytes()
    groups1 = view.groups_for([("l1", "m", 8, 0, v1)])
    base_ptr = groups1[0][2].ctypes.data
    groups2 = view.groups_for([("l1", "m", 8, 0, None), ("l2", "m", 8, 0, v2)])
    assert groups2[0][2].ctypes.data == base_ptr
    assert groups2[0][2].shape == (2, 8)


class _SqlRecorder:
    def __init__(self):
        self.calls = []

    def execute(self, sql, args=()):
        self.calls.append((_query(sql)[0], args))
        return self

    def __iter__(self):
        return iter(())

    def fetchone(self):
        return (1,)

    def fetchall(self):
        return []


def _fake_pg(*, pgvector=True, vchord=False, vectorscale=False, pg_search=False, vchord_bm25=False, trigram=True):
    store = PostgresStore.__new__(PostgresStore)
    store.embedder = store.fallback_embedder = HashedTfidfEmbedder()
    store._write_vectors = None
    store.pgvector = pgvector
    store.vchord = vchord
    store.vectorscale = vectorscale
    store.pg_search = pg_search
    store.pg_tokenizer = vchord_bm25
    store.vchord_bm25 = vchord_bm25
    store.trigram = trigram
    store.conn = _SqlRecorder()
    return store


def test_postgres_vchordrq_and_vchord_bm25_ddl_and_queries():
    store = _fake_pg(pgvector=True, vchord=True, vchord_bm25=True, trigram=True)
    store._bootstrap()
    ddl = "\n".join(sql for sql, _ in store.conn.calls)
    assert "USING vchordrq ((vec::vector(512)) vector_cosine_ops)" in ddl
    assert "residual_quantization = true" in ddl
    assert "USING bm25 (bm25_tokens bm25_ops)" in ddl
    assert "gin_trgm_ops" not in ddl  # skipped when native BM25 is active
    health = store.health_check()
    assert health["vector"] == "vchordrq" and health["lexical"] == "vchord_bm25" and not health["degraded"]

    opts = dict(user_id="alice", agent_id="agent", session_id="s", filters=RecallFilter(as_of=10), ids_only=True, top_n=7)
    store.search_lexical("Atlas", **opts)
    sql, _ = store.conn.calls[-1]
    assert "<&> to_bm25query('idx_facts_fts_vchord_bm25'" in sql


def test_postgres_diskann_and_paradedb_pg_search_ddl_and_queries():
    store = _fake_pg(pgvector=True, vectorscale=True, pg_search=True, trigram=True)
    store._bootstrap()
    ddl = "\n".join(sql for sql, _ in store.conn.calls)
    assert "USING diskann ((vec::vector(512)) vector_cosine_ops)" in ddl
    assert "USING bm25 (rowid, fts_tokens) WITH (key_field='rowid')" in ddl
    assert "gin_trgm_ops" not in ddl
    health = store.health_check()
    assert health["vector"] == "diskann" and health["lexical"] == "pg_search" and not health["degraded"]

    opts = dict(user_id="alice", agent_id="agent", session_id="s", filters=RecallFilter(as_of=10), ids_only=True, top_n=7)
    store.search_lexical("Atlas blueprint", **opts)
    sql, _ = store.conn.calls[-1]
    assert "@@@" in sql and "-paradedb.score(x.rowid)" in sql


def test_postgres_vchord_lists_rejects_injection(monkeypatch):
    store = _fake_pg(pgvector=True, vchord=True)
    monkeypatch.setenv("MEMENGINE_VCHORD_LISTS", "64; DROP TABLE facts")
    with pytest.raises(ValueError, match="MEMENGINE_VCHORD_LISTS"):
        store._vchord_options()
