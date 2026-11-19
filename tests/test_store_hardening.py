import sqlite3
from concurrent.futures import ThreadPoolExecutor

from memengine.store import Store
from memengine.textutil import estimate_tokens
from memengine.timeline import TimelineBook


def test_legacy_database_migrates_idempotently(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as legacy:
        legacy.executescript("""
            CREATE TABLE ingest(id INTEGER PRIMARY KEY, text TEXT NOT NULL, ts REAL);
            INSERT INTO ingest VALUES(1, 'legacy dialogue', 10);
            CREATE TABLE line_zkeys(line_id TEXT NOT NULL, zkey TEXT NOT NULL);
            INSERT INTO line_zkeys VALUES('old', '类目:影视'), ('old', '类目:影视');
        """)
    for _ in range(2):
        store = Store(path)
        assert store.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert store.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert store.history_tokens() == estimate_tokens("legacy dialogue")
        assert store.get_meta("schema_version") == "2"
        assert tuple(store.conn.execute("SELECT user_id, role, event_ts FROM raw_turns").fetchone()) == ("default", "user", 10)
        assert not store.conn.execute("SELECT name FROM sqlite_master WHERE name IN ('ingest','line_zkeys')").fetchall()
        store.close()


def test_timeline_bounds_and_topic_shifts():
    book = TimelineBook(delta_t=60)
    for second in range(50):
        book.record(2, {"topic": "same"}, second)
    assert [node["event_count"] for node in book.layer2] == [20, 20, 10]
    book.record(2, {"topic": "different"}, 51)
    book.record(2, {"topic": "different", "origin_shifted": True}, 52)
    assert len(book.layer2) == 5
    duration = TimelineBook(delta_t=60, max_node_duration=5)
    for second in (0, 5, 6):
        duration.record(1, {}, second)
    assert [node["t_start"] for node in duration.layer1] == [0, 6]


def test_two_connections_archive_atomic_consecutive_revisions(tmp_path):
    db = tmp_path / "revisions.db"
    store = Store(db)
    with store.transaction():
        block = store.create_block("test", 0)
        line = store.insert_line(block_id=block, formatted="[step 0] initial", plain="[step 0] initial",
                                 attribute="revision", value="initial", surprise=0.25, raw_dist=None,
                                 layout="APPEND_ADJACENT_LINE", isolated=False, zkeys=["实体:test"], timestamp=0)

    def update(index):
        writer = Store(db)
        try:
            assert writer.update_line(line, formatted=f"[step {index}] revision {index}",
                                      plain=f"[step {index}] revision {index}", attribute="revision", value=str(index),
                                      surprise=0.95, raw_dist=None, layout="OVERWRITE_SAME_LINE",
                                      zkeys=["实体:test"], timestamp=index, is_conflict=True)
        finally:
            writer.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(update, [1, 2]))
    current = store.get_line(line)
    assert current["revision"] == 2
    assert sorted(current["source_step_ids"]) == [0, 1, 2]
    older = store.historical_versions([line])
    assert [revision["revision"] for revision in older] == [0, 1]
    assert {older[1]["value"], current["value"]} == {"1", "2"}
    store.close()
