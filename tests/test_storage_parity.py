from contextlib import closing
import os
import uuid

import pytest

from memengine.identifiers import extract_identifiers
from memengine.engine import CognitiveMemoryEngine
from memengine.storage import open_store
from memengine.storage.base import RecallFilter


def exercise(store):
    scope = ("alice", "agent", "session")
    def put(text, scope, **options):
        turn = store.append_turn(text, 1, scope=scope)
        with store.transaction():
            block = store.create_block("Atlas", 1, scope=scope)
            return store.insert_fact(block_id=block, formatted=text, plain=text, ingest_time=1, scope=scope,
                                     source_turn_ids=[turn], zkeys=["实体:Atlas"], **options), turn

    a, turn = put("Atlas configuration v3.11", scope)
    b, _ = put("Atlas configuration v3.12", scope, valid_from=5, expires_at=10)
    c, alien = put("Atlas configuration v3.11", ("bob", "agent", "session"))
    opts = dict(user_id="alice", agent_id="agent", ids_only=True)
    exact = store.search_exact_identifiers(extract_identifiers("v3.11"), **opts)
    assert [row["id"] for row in exact] == [a]
    for as_of, expected in ((4, {a}), (5, {a, b}), (10, {a})):
        filters = RecallFilter(as_of=as_of)
        for search in (store.search_vector_topk, store.search_lexical, store.search_entity_inverted):
            argument = ["Atlas"] if search == store.search_entity_inverted else "Atlas"
            assert {row["id"] for row in search(argument, filters=filters, **opts)} == expected
        ids = store.search_temporal_range(filters=filters, **opts)
        assert {row["id"] for row in ids} == expected
    assert [row["turn_id"] for row in store.fetch_raw_window([turn, alien], 0, "alice", "agent")] == [turn]
    assert store.memory_history(c, scope=scope) == []
    assert not store.delete_fact(c, hard=True, scope=scope)
    newer = store.supersede_fact(a, formatted="Atlas configuration v3.13", plain="Atlas configuration v3.13",
                                ingest_time=7, scope=scope, current_turn_id=turn)
    assert [row["status"] for row in store.get_fact_chain(a, user_id="alice", agent_id="agent")] == ["superseded", "active"]
    assert store.get_line(newer, scope=scope, include_deleted=True)["id"] == newer
    cold = store.archive_cold(scope, now=20, half_life_days=30, inactive_days=365, superseded_days=365)
    assert b in cold
    assert b in {row["id"] for row in store.search_temporal_range(user_id="alice", agent_id="agent",
        start_ts=5, end_ts=10, include_cold=True, filters=RecallFilter(historical_window=(5, 10)), ids_only=True)}
    assert store.delete_fact(newer, hard=True, scope=scope)
    assert not store.search_exact_identifiers(extract_identifiers("v3.11"), **opts)
    chronicle = {"mention_time": 1717200000.0, "speaker": "Alice Chen"}
    d, _ = put("I used the Atlas environment yesterday.", scope, chronicle=chronicle)
    e, _ = put("I used the local Atlas environment yesterday.", scope, chronicle=chronicle)
    rich = store.facts_by_ids([d, e], scope=scope)
    assert all(row["occurrence_start"] == 1717113600.0 for row in rich)
    assert all("Alice Chen used" in row["normalized_text"] for row in rich)
    assert store.adjacent_links([d], scope=scope, filters=RecallFilter())[0]["target"] == e
    successor = store.supersede_fact(d, formatted="Atlas environment updated", plain="Atlas environment updated",
                                     ingest_time=8, scope=scope, chronicle=chronicle)
    observation, _ = put("Atlas environment history", scope, memory_type="observation", derived_from_ids=[d, e],
                         chronicle={"mention_time": 1717200000.0, "canonical_entity_id": rich[0]["canonical_entity_id"]})
    assert [row["id"] for row in store.list_observations(user_id="alice", agent_id="agent")] == [observation]
    assert store.delete_fact(successor, scope=scope)
    assert not store.list_observations(user_id="alice", agent_id="agent")
    stale, = store.list_observations(user_id="alice", agent_id="agent", include_stale=True)
    assert stale["id"] == observation and stale["is_stale"]
    for search in (store.search_vector_topk, store.search_lexical):
        assert not search("Atlas environment history", filters=RecallFilter(memory_types=("observation",)), **opts)
    assert store.health_check()["ok"]


def exercise_workspace(reopen):
    owner = {"user_id": "team", "agent_id": "agent", "session_id": "session"}
    text = "Atlas checkpoint is BLUE-482."
    engine = reopen()
    with closing(engine.store):
        turn = engine.buffer_turn(text, **owner)
        task = engine.start_task("Check Atlas", task_id="shared-name", **owner)
        item = engine.record_provisional_claim(task, text, [turn], **owner)
        engine.checkpoint_task(task, {"stage": 3, "turn": turn}, **owner)
        engine.start_task("Sibling task", task_id=task, **{**owner, "session_id": "other"})
        assert not engine.store.snapshot(scope=("team", "agent", None))
    engine = reopen()
    with closing(engine.store):
        recent, = engine.get_recent_turns(**owner)
        assert (recent["turn_id"], recent["extraction_state"], recent["text"]) == (turn, "buffered", text)
        workspace = engine.get_task_workspace(task, **owner)
        assert workspace["status"] == "active" and workspace["checkpoint"] == {"stage": 3, "turn": turn}
        assert [row["item_id"] for row in workspace["items"]] == [item]
        assert not engine.get_task_workspace(task, **{**owner, "session_id": "other"})["items"]
        for field in ("user_id", "agent_id"):
            other = {**owner, field: "foreign"}
            with pytest.raises(ValueError, match="task not found"):
                engine.get_task_workspace(task, **other)
            with pytest.raises(ValueError, match="task not found"):
                engine.verify_task_item(task, item, "user_confirmed", turn, **other)
        engine.pause_task(task, **owner)
        assert engine.resume_task(task, **owner) == {"stage": 3, "turn": turn}
        engine.verify_task_item(task, item, "user_confirmed", turn, **owner)
        promoted = engine.commit_task(task, **owner)
        assert set(promoted) == {item}
        assert engine.commit_task(task, **owner) == promoted
        assert engine.get_task_workspace(task, **owner)["items"][0]["promoted_fact_id"] == promoted[item]


def test_sqlite_workspace_survives_reopen(tmp_path):
    exercise_workspace(lambda: CognitiveMemoryEngine(tmp_path / "workspace.db"))


@pytest.mark.parametrize("file", [False, True])
def test_sqlite_parity(tmp_path, file):
    with closing(open_store(tmp_path / "ledger.db" if file else ":memory:")) as store:
        exercise(store)


def test_sqlite_cache_sees_updates_rollback_and_external_writes(tmp_path):
    path = tmp_path / "shared.sqlite"
    def put(store, text):
        with store.transaction():
            block = store.create_block("Atlas", 1)
            return store.insert_fact(block_id=block, formatted=text, plain=text, ingest_time=1)
    with closing(open_store(path)) as first, closing(open_store(path)) as second:
        fact = put(first, "Atlas blue")
        assert first.search_vector_topk("blue", ids_only=True)[0]["id"] == fact
        assert first.search_vector_topk("blue", ids_only=True)[0]["id"] == fact
        first.conn.execute("UPDATE fact_vectors SET vec=zeroblob(dim*4) WHERE fact_id=?", (fact,))
        assert not first.search_vector_topk("blue", ids_only=True)
        first.conn.rollback()
        assert first.search_vector_topk("blue", ids_only=True)[0]["id"] == fact
        with pytest.raises(RuntimeError):
            with first.transaction():
                put(first, "Atlas green")
                assert first.search_vector_topk("green", ids_only=True)
                raise RuntimeError("rollback")
        assert not first.search_vector_topk("green", ids_only=True)
        external = put(second, "Atlas red")
        assert first.search_vector_topk("red", ids_only=True)[0]["id"] == external
        first.delete_fact(external, hard=True)
        assert not first.search_vector_topk("red", ids_only=True)


@pytest.mark.parametrize("degraded", [False, True])
def test_postgres_contract(monkeypatch, degraded):
    dsn = os.getenv("TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("TEST_POSTGRES_DSN is required for the PostgreSQL integration test")
    import psycopg
    from psycopg import sql
    from memengine.storage.postgres import PostgresStore
    schema = "test_" + uuid.uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        if not degraded:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        if degraded:
            monkeypatch.setattr(PostgresStore, "_enable_extension", lambda *_: False)
        try:
            configured = psycopg.conninfo.make_conninfo(dsn, options=f"-csearch_path={schema},public")
            with closing(PostgresStore(configured)) as store:
                exercise(store)
                health = store.health_check()
                assert health["degraded"] is degraded
                if not degraded:
                    assert health["hnsw_dimensions"] == [512]
                    store.conn.execute("SELECT 'Atlasx' <% 'Atlas configuration'").fetchone()
                    indexes = store.conn.execute("SELECT indexname FROM pg_indexes WHERE schemaname=current_schema()")
                    assert "idx_fact_vectors_hnsw_512" in {row["indexname"] for row in indexes}
            from urllib.parse import urlencode

            uri = "postgresql:///?" + urlencode(psycopg.conninfo.conninfo_to_dict(configured))
            exercise_workspace(lambda: CognitiveMemoryEngine(db_url=uri))
        finally:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def test_postgres_query_is_parameterized_and_keeps_tenant_and_time_filters():
    from memengine.storage.postgres import _query
    where, args = RecallFilter(as_of=8, scopes=("world",)).sql("f.user_id=? AND f.agent_id=?", ("alice", "agent"))
    sql, _ = _query("SELECT f.id FROM facts f WHERE " + where)
    assert "f.user_id=%s AND f.agent_id=%s" in sql and "f.expires_at>%s" in sql
    assert "alice" not in sql and args[:2] == ("alice", "agent") and sql.count("%s") == len(args)


@pytest.mark.parametrize("native", [False, True])
def test_postgres_schema_and_bounded_index_sql_without_server(native):
    from memengine.storage.postgres import PostgresStore, _query
    from memengine.vector import HashedTfidfEmbedder

    class Recorder:
        def __init__(self):
            self.calls = []

        def execute(self, sql, args=()):
            self.calls.append((_query(sql)[0], args))
            return self

        def __iter__(self):
            return iter(())

        def fetchone(self):
            return None

        def fetchall(self):
            return []

    store = PostgresStore.__new__(PostgresStore)
    store.embedder = store.fallback_embedder = HashedTfidfEmbedder()
    store._write_vectors = None
    store.pgvector = store.trigram = native
    store.conn = Recorder()
    store._bootstrap()
    ddl = "\n".join(statement for statement, _ in store.conn.calls)
    assert "VIRTUAL" not in ddl and "AUTOINCREMENT" not in ddl
    assert "document TSVECTOR" in ddl and "ON DELETE CASCADE" in ddl and "vec REAL[]" in ddl
    assert ("USING hnsw" in ddl) is native and ("gin_trgm_ops" in ddl) is native
    options = dict(user_id="alice", agent_id="trial", session_id="s", filters=RecallFilter(as_of=10),
                   ids_only=True, top_n=7)
    store.search_vector_topk("Atlas", **options)
    sql, parameters = store.conn.calls[-1]
    assert ("<=>" in sql) is native and ("unnest" in sql) is not native
    assert "f.expires_at>%s" in sql and "LIMIT %s" in sql
    assert parameters[1:4] == ("alice", "trial", "s") and parameters[-1] == 7
    store.search_lexical("Atlas", **options)
    sql, parameters = store.conn.calls[-1]
    assert "document @@ q.ts" in sql and "f.expires_at>%s" in sql
    assert ("word_similarity" in sql) is native and parameters[2:5] == ("alice", "trial", "s")
