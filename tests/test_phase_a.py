import json
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pytest

from agentmem_core import CognitiveMemoryEngine
from agentmem_core.distance import bipolar_weight, compute_bipolar_radius
from agentmem_core.store import Store
from agentmem_core.temporal import parse_temporal_intent
from agentmem_core.textutil import normalize_entity, tokenize_for_fts
from agentmem_core.vector import HashedTfidfEmbedder


def stamp(date):
    return datetime.fromisoformat(date).replace(tzinfo=timezone.utc).timestamp()


def test_version_ledger_preserves_all_three_raw_sources_and_search_indexes():
    engine = CognitiveMemoryEngine()
    first = engine.add("老周对花粉过敏。", timestamp=1)
    second = engine.add("[step:8] 老周对苯甲酸钠过敏。", timestamp=2)
    third = engine.add("老周对牛奶过敏。", timestamp=3)
    ids = [entry["operations"][0]["line_id"] for entry in (first, second, third)]
    chain = engine.store.get_fact_chain(ids[0])
    assert [fact["id"] for fact in chain] == ids
    assert [fact["status"] for fact in chain] == ["superseded", "superseded", "active"]
    assert [fact["valid_until"] for fact in chain] == [2, 3, None]
    assert [fact["supersedes_id"] for fact in chain] == [None, *ids[:2]]
    assert chain[-1]["source_turn_ids"] == [1, 2, 3] and chain[-1]["source_step_ids"] == [8]
    assert {fact["seq"] for fact in chain} == {1}
    assert ids[1] in {fact["id"] for fact in engine.store.search_vector("苯甲酸钠")}
    assert ids[1] in {fact["id"] for fact in engine.store.search_bm25("苯甲酸钠")}
    assert len(engine.store.search_entity_inverted(["老周"])) == 3
    current = engine.retrieve("老周现在对什么过敏？", timestamp=4, read_only=True, include_raw_turns=True)
    assert current["ranked"][0]["id"] == ids[-1]
    assert any(fact["text"].startswith("[历史已变更]") for fact in current["ranked"] if fact["historical"])
    assert [turn["text"] for turn in current["drilldown_turns"]] == ["老周对花粉过敏。", "[step:8] 老周对苯甲酸钠过敏。", "老周对牛奶过敏。"]
    assert "牛奶" in engine.render() and "花粉" not in engine.render()
    assert not engine.delete(ids[0], user_id="wrong", hard=True)
    assert engine.delete(ids[0], hard=True)
    assert all(engine.store.conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
               for table in ("facts", "fact_entities", "facts_fts", "fact_vectors"))
    assert len(engine.store.get_turn_window([1], window=2)) == 3


def test_raw_logs_are_immediate_lossless_and_windows_cannot_escape_scope():
    engine = CognitiveMemoryEngine()
    ids = {"first": [], "second": []}
    for index in range(5):
        for session in ids:
            raw = f"  {session} message {index}\n"
            ids[session].append(engine.add(raw, timestamp=index, session_id=session, role="assistant", sync=False)["turn_id"])
    denied = engine.add("another tenant", timestamp=1, user_id="different", session_id="first", sync=False)["turn_id"]
    selected = [ids["first"][1], ids["first"][2], ids["second"][-1], denied]
    turns = engine.store.get_turn_window(selected, window=1)
    expected = sorted(ids["first"][:4] + ids["second"][-2:])
    assert [turn["turn_id"] for turn in turns] == expected
    assert turns[0]["text"] == "  first message 0\n" and turns[0]["role"] == "assistant"
    assert not engine.store.snapshot() and len(engine.pending) == 11
    assert engine.store.get_turn_window([denied], window=0, user_id="different")[0]["turn_id"] == denied
    engine.drain(user_id="default")
    source_ids = {source for fact in engine.store.snapshot(scope=("default", "default", None)) for source in fact["source_turn_ids"]}
    assert source_ids == set(ids["first"] + ids["second"])


def test_fts_tokenizer_and_native_bm25_handle_cjk_versions_and_metacharacters():
    engine = CognitiveMemoryEngine()
    allergy = engine.add("老周对苯甲酸钠过敏。", timestamp=1)["operations"][0]["line_id"]
    version = engine.add("Build v2.4.1-rc3 uses module_id under src/app.", timestamp=2)["operations"][0]["line_id"]
    engine.add("Build v2.4.1-rc4 uses another module.", timestamp=3)
    assert tokenize_for_fts("苯甲酸钠") == "苯 甲 酸 钠 苯甲 甲酸 酸钠"
    assert tokenize_for_fts("v2.4.1-rc3 module_id src/app") == "v2.4.1-rc3 module_id src/app"
    assert engine.store.search_bm25('"(苯甲酸钠):"')[0]["id"] == allergy
    results = engine.store.search_bm25(':("v2.4.1-rc3")')
    assert [fact["id"] for fact in results] == [version]
    assert results[0]["bm25_score"] < 0
    assert not engine.store.search_bm25('"(*)::!!!')


def test_one_hop_entity_normalization_is_exact_and_never_uses_tags():
    store = Store()
    with store.transaction():
        block = store.create_block("test", 0)
        ids = [store.insert_fact(block_id=block, formatted=key, plain=key, ingest_time=index,
                                 zkeys=[key, "类目:共享", "实体:第二跳"] if index == 0 else [key, "类目:共享"])
               for index, key in enumerate(["人物:老周", "实体:老周", "老周", "实体:第二跳", "实体:他人"])]
    assert normalize_entity(" 人物： 老 周 ") == normalize_entity("实体:老周") == "老周"
    assert normalize_entity("Entity: Interstellar") == "interstellar"
    assert {fact["id"] for fact in store.search_entity_inverted(["老周"])} == set(ids[:3])
    assert store.search_entity_inverted(["类目:共享", "标签：老周", "话题:时事杂讯"]) == []


def test_temporal_lookup_prefers_correct_superseded_year_and_preserves_event_time():
    engine = CognitiveMemoryEngine()
    block = engine.store.create_block("home", stamp("2024-06-01"))
    fact_id = engine.store.insert_fact(block_id=block, formatted="Alex lived in Rome", plain="Alex lived in Rome",
                                      ingest_time=stamp("2024-06-01"), zkeys=["实体:Alex"], attribute="住址")
    for year, place in ((2025, "Paris"), (2026, "Berlin")):
        fact_id = engine.store.supersede_fact(fact_id, formatted=f"Alex住在{place}", plain=f"Alex住在{place}",
                                              ingest_time=stamp(f"{year}-06-01"), zkeys=["实体:Alex"], attribute="住址")
    now = stamp("2026-11-19")
    intent = parse_temporal_intent("Alex去年住在哪里？", now)
    assert intent.is_historical and (intent.start_ts, intent.end_ts) == (stamp("2025-01-01"), stamp("2026-01-01"))
    assert parse_temporal_intent("before the second revision", now).rev_index == 1
    assert parse_temporal_intent("上周三发生了什么？", now).start_ts == stamp("2026-11-11")
    assert parse_temporal_intent("What happened on 2025-08-19?", now).end_ts == stamp("2025-08-20")
    facts = engine.store.search_temporal(start_ts=intent.start_ts, end_ts=intent.end_ts)
    assert len(facts) == 1 and facts[0]["status"] == "superseded"
    retrieved = engine.retrieve("Alex去年住在哪里？", timestamp=now, read_only=True)
    assert retrieved["ranked"][0]["id"] == facts[0]["id"]
    assert "Paris" in engine.answer("Alex去年住在哪里？", retrieved=retrieved)["response"]
    assert "Berlin" not in engine.answer("Alex去年住在哪里？", retrieved=retrieved)["response"]
    before = engine.retrieve("Alex上次住在哪里？", timestamp=now, read_only=True)
    assert before["ranked"][0]["id"] == facts[0]["id"] and before["ranked"][0]["temporal_match"]
    original = "Yesterday I reviewed that release."
    written = engine.add(original, timestamp=now)
    row = engine.store.get_line(written["operations"][0]["line_id"])
    assert "2026-11-18" in row["plain"] and row["event_time_start"] == stamp("2026-11-18")
    assert engine.store.get_turn_window([written["turn_id"]], window=0)[0]["text"] == original


def test_vector_cache_survives_reopen_and_retrieval_sql_is_bounded(tmp_path):
    class Counting(HashedTfidfEmbedder):
        def __init__(self):
            super().__init__()
            self.calls = []

        def encode(self, texts):
            self.calls.extend(texts)
            return super().encode(texts)

    path = tmp_path / "ledger.db"
    embedder = Counting()
    engine = CognitiveMemoryEngine(path, vector=embedder)
    for index in range(300):
        engine.add(f"Atlas release notes record {index}.", timestamp=index)
    assert len(embedder.calls) == 300
    assert engine.store.conn.execute("SELECT count(*) FROM fact_vectors").fetchone()[0] == 300
    vector = engine.store.conn.execute("SELECT dim,vec FROM fact_vectors LIMIT 1").fetchone()
    assert len(vector[1]) == vector[0] * 4 and np.linalg.norm(np.frombuffer(vector[1], dtype="<f4")) == pytest.approx(1)
    engine.store.close()
    reopened = CognitiveMemoryEngine(path, vector=embedder)
    sql = []
    reopened.store.conn.set_trace_callback(sql.append)
    result = reopened.retrieve("What do Atlas release notes say?", timestamp=301, read_only=True, include_raw_turns=True)
    assert result["ranked"] and result["drilldown_turns"]
    assert len(embedder.calls) == 301
    top_level = [statement for statement in sql if not statement.startswith("--")]
    assert len(top_level) < 25
    assert not any("timeline" in statement for statement in top_level)
    assert not any(statement.lstrip().startswith(("UPDATE", "INSERT", "DELETE")) for statement in top_level)
    assert reopened.store.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_poles_and_notebook_spacing_survive_new_storage():
    engine = CognitiveMemoryEngine()
    for _ in range(8):
        engine.add("团队每周三下午三点开周会。")
    engine.add("老周对花粉过敏。")
    habit, exception = engine.store.snapshot()
    assert len(habit["source_turn_ids"]) == 8
    assert bipolar_weight(compute_bipolar_radius(habit, 8)) == bipolar_weight(compute_bipolar_radius(exception, 8)) == 1.0
    rendered = engine.render()
    assert rendered.count("每周三") == 1 and "\n\n# 健康禁忌" in rendered


def test_legacy_lines_migrate_versions_and_only_verifiable_source_pointers(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as db:
        db.executescript("""
        CREATE TABLE blocks(id TEXT PRIMARY KEY, topic TEXT, created_at REAL);
        INSERT INTO blocks VALUES('b1','test',1);
        CREATE TABLE ingest(id INTEGER PRIMARY KEY,text TEXT,ts REAL);
        INSERT INTO ingest VALUES(1,'[step 1] old fact',1),(2,'[step 2] new fact',2);
        CREATE TABLE lines(id TEXT PRIMARY KEY,block_id TEXT,seq INTEGER,formatted TEXT,plain TEXT,revision INTEGER,
                           created_at REAL,updated_at REAL,active INTEGER,source_step_ids TEXT);
        INSERT INTO lines VALUES('l9','b1',1,'new fact','new fact',1,1,2,1,'[1,2]');
        CREATE TABLE line_history(line_id TEXT,revision INTEGER,formatted TEXT,plain TEXT,zkeys TEXT,t_start REAL,t_end REAL,source_step_ids TEXT);
        INSERT INTO line_history VALUES('l9',0,'old fact','old fact','["实体:Atlas"]',1,2,'[1]');
        CREATE TABLE line_zkeys(line_id TEXT,zkey TEXT);
        INSERT INTO line_zkeys VALUES('l9','实体:Atlas');
        """)
    first = Store(path)
    chain = first.get_fact_chain("l9")
    assert [row["source_turn_ids"] for row in chain] == [[1], [1, 2]]
    assert [row["status"] for row in chain] == ["superseded", "active"]
    assert first.get_line("l9")["plain"] == "new fact"
    assert first.search_bm25("old fact")[0]["id"] == "l9"
    assert first.conn.execute("PRAGMA foreign_key_check").fetchall() == []
    first.close()
    reopened = Store(path)
    assert reopened.get_fact_chain("l9") == chain
    assert reopened.conn.execute("SELECT count(*) FROM raw_turns").fetchone()[0] == 2
    obs_id = reopened.insert_observation("Current color", derived_from_fact_ids=[chain[-1]["id"]], source_turn_ids=[2])
    assert reopened.list_observations()[0]["id"] == obs_id
    assert reopened.list_observations(user_id="another") == []
