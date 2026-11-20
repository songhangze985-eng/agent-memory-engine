"""One-time migration from the split notebook schema to the fact ledger."""

from collections import defaultdict
import json

from memengine.textutil import merge_step_ids, step_ids_in, strip_step_tags, tokenize_for_fts
from memengine.lifecycle import content_hash, is_constraint


def migrate_identifiers(store, tokenizer):
    conn = store.conn
    rebuild = []
    for table in ("facts_fts", "facts_cold_fts", "raw_turns_fts"):
        schema = conn.execute("SELECT sql FROM sqlite_master WHERE name=?", (table,)).fetchone()[0]
        if tokenizer not in schema:
            rebuild.append(table)
            conn.execute(f"DROP TABLE {table}")
            columns = "" if table == "raw_turns_fts" else "fact_id UNINDEXED,user_id UNINDEXED,agent_id UNINDEXED,"
            conn.execute(f'CREATE VIRTUAL TABLE {table} USING fts5({columns}fts_tokens, tokenize="{tokenizer}")')
    missing = store.get_meta("identifier_index_version") != "1"
    if not (missing or rebuild):
        return
    for table, tier in (("facts_fts", "hot"), ("facts_cold_fts", "cold")):
        if table in rebuild:
            conn.executemany(f"INSERT INTO {table}(rowid,fact_id,user_id,agent_id,fts_tokens) VALUES (?,?,?,?,?)",
                             ((*row[:4], tokenize_for_fts(row[4])) for row in conn.execute(
                                 "SELECT rowid,id,user_id,agent_id,plain FROM facts WHERE tier=?", (tier,))))
    if missing or "raw_turns_fts" in rebuild:
        conn.execute("DELETE FROM raw_turns_fts")
        conn.executemany("INSERT INTO raw_turns_fts(rowid,fts_tokens) VALUES (?,?)",
                         ((row[0], tokenize_for_fts(row[1])) for row in conn.execute("SELECT turn_id,text FROM raw_turns")))
    if missing:
        for row in conn.execute("SELECT id,plain FROM facts"):
            store.index_identifiers(row[0], row[1])
    store.set_meta("identifier_index_version", "1")


def migrate_lifecycle(store):
    if store.get_meta("lifecycle_schema_version") == "1":
        return
    conn = store.conn
    fields = {
        "memory_type": "TEXT NOT NULL DEFAULT 'fact' CHECK(memory_type IN ('fact','observation','procedural'))",
        "confidence": "REAL NOT NULL DEFAULT 1.0", "derived_from_ids": "TEXT NOT NULL DEFAULT '[]'",
        "is_constraint": "INTEGER NOT NULL DEFAULT 0", "tier": "TEXT NOT NULL DEFAULT 'hot' CHECK(tier IN ('hot','cold'))",
        "last_accessed_at": "REAL", "content_hash": "TEXT", "mention_count": "INTEGER NOT NULL DEFAULT 1",
        "tool_call_ids": "TEXT NOT NULL DEFAULT '[]'", "trigger_text": "TEXT NOT NULL DEFAULT ''",
        "success_count": "INTEGER NOT NULL DEFAULT 0",
    }
    for table, columns in (("facts", fields), ("raw_turns", {"extraction_state": "TEXT NOT NULL DEFAULT 'done'"}),
                           ("fact_entities", {"tier": "TEXT NOT NULL DEFAULT 'hot'"})):
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, definition in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_scope_hash ON facts(user_id,agent_id,content_hash,tier,status,session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_scope_tier ON facts(user_id,agent_id,tier,status,session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_event_time_hot ON facts(user_id,agent_id,event_time_start) WHERE tier='hot'")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fe_hot ON fact_entities(user_id,agent_id,key_type,entity_norm) WHERE tier='hot'")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_turns_pending ON raw_turns(user_id,agent_id,session_id,extraction_state,turn_id)")
    conn.executemany("UPDATE facts SET content_hash=?,is_constraint=? WHERE id=?",
                     [(content_hash(row["plain"]), int(is_constraint(row["plain"])), row["id"])
                      for row in conn.execute("SELECT id,plain FROM facts WHERE content_hash IS NULL")])
    store.set_meta("lifecycle_schema_version", "1")


def migrate_observation_rows(store):
    if store.get_meta("lifecycle_sources_migrated") == "1":
        return
    conn = store.conn
    for row in conn.execute("SELECT * FROM observations ORDER BY created_at,id").fetchall():
        scope = (row["user_id"], row["agent_id"], "default")
        block = store.create_block("Observations", row["created_at"], scope=scope)
        data = store._version_data(fact_id=row["id"], block_id=block, seq=1, scope=scope, formatted=row["content"],
                                   plain=row["content"], attribute="", value="", surprise=.25, layout=None, isolated=False,
                                   is_conflict=False, role="assistant", ingest_time=row["created_at"], event_time_start=None,
                                   status="active" if row["status"] == "active" else "deleted",
                                   valid_until=None if row["status"] == "active" else row["updated_at"],
                                   source_turn_ids=json.loads(row["source_turn_ids"]))
        data.update(memory_type="observation", confidence=row["confidence"], derived_from_ids=row["derived_from_fact_ids"])
        store._insert_version(data, (), offline=True)
    conn.execute("DELETE FROM observations")
    for row in conn.execute("SELECT key,value FROM meta WHERE key LIKE 'core_snapshot:%'").fetchall():
        user_id, agent_id = json.loads(row["key"].split(":", 1)[1])
        snapshot = json.loads(row["value"])
        store.set_core_snapshot(**snapshot, user_id=user_id, agent_id=agent_id)
    store.set_meta("lifecycle_sources_migrated", "1")


def migrate_legacy(store):
    conn = store.conn
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table in ("blocks", "timeline"):
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for column in ("user_id", "agent_id", "session_id"):
            if column not in columns:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT NOT NULL DEFAULT 'default'")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_timeline_scope ON timeline(user_id,agent_id,session_id,layer,position)")
    if not tables.intersection({"ingest", "lines", "line_history", "line_zkeys"}):
        return

    def legacy_rows(table):
        return (dict(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid")) if table in tables else ()

    def scope(row):
        return tuple(row.get(name, "default") for name in ("user_id", "agent_id", "session_id"))

    conn.execute("DELETE FROM meta WHERE key LIKE 'history_tokens:%'")
    exact_sources, tagged_sources = defaultdict(list), defaultdict(list)
    for turn in legacy_rows("ingest"):
        stamp = turn.get("created_at")
        if stamp is None:
            stamp = turn.get("ts")
        if stamp is None:
            stamp = 0.0
        owner = scope(turn)
        turn_id = store.append_turn(turn["text"], stamp, scope=owner, role=turn.get("role", "user"),
                                    event_ts=turn.get("event_time"))
        exact_sources[(owner, stamp, strip_step_tags(turn["text"]))].append(turn_id)
        for step in step_ids_in(turn["text"]):
            tagged_sources[(owner, step)].append(turn_id)

    keys, history = defaultdict(list), defaultdict(list)
    for row in legacy_rows("line_zkeys"):
        keys[row["line_id"]].append(row["zkey"])
    for row in legacy_rows("line_history"):
        history[row["line_id"]].append(row)

    lines = list(legacy_rows("lines"))
    orphan_history = set(history) - {line["id"] for line in lines}
    if orphan_history:
        raise ValueError("Legacy archive contains revisions without their parent line; migration stopped")
    for line in lines:
        root = line["id"]
        owner = scope(line)
        active = bool(line.get("active", 1))
        revision = line.get("revision", 0)
        older = {row["revision"]: row for row in history[root] if row["revision"] < revision}
        versions = [(older[number], "superseded", number) for number in sorted(older)]
        versions.append((line, "active" if active else "deleted", revision))
        predecessor = None
        turns, steps = [], []
        for row, status, number in versions:
            stamp = (row.get("updated_at") if row is line else row.get("t_start"))
            if stamp is None:
                stamp = line.get("created_at") or 0.0
            current_steps = merge_step_ids(json.loads(row.get("source_step_ids") or "[]"),
                                           step_ids_in(row["plain"]), step_ids_in(row["formatted"]))
            verified = [ids[0] for step in current_steps if len(ids := tagged_sources[(owner, step)]) == 1]
            exact = exact_sources[(owner, stamp, strip_step_tags(row["plain"]))]
            if len(exact) == 1:
                verified.extend(exact)
            turns, steps = merge_step_ids(turns, verified), merge_step_ids(steps, current_steps)
            fact_id = root if predecessor is None else f"{root}#{number}"
            valid_until = row.get("valid_until")
            if valid_until is None:
                valid_until = row.get("t_end")
            if status != "active" and valid_until is None:
                valid_until = stamp
            data = store._version_data(fact_id=fact_id, block_id=line["block_id"], seq=line["seq"], scope=owner,
                                       formatted=row["formatted"], plain=row["plain"], attribute=row.get("attribute"),
                                       value=row.get("value"), surprise=row.get("surprise"), layout=row.get("layout"),
                                       isolated=row.get("isolated", line.get("isolated")), is_conflict=row.get("is_conflict", False),
                                       role=row.get("role", line.get("role", "user")), event_time_start=row.get("event_time"),
                                       ingest_time=stamp, source_turn_ids=turns, source_step_ids=steps, status=status,
                                       revision=number, supersedes_id=predecessor, root_fact_id=root,
                                       valid_until=valid_until, hit_count=row.get("hit_count", 0))
            store._insert_version(data, keys[root] if row is line else json.loads(row.get("zkeys") or "[]"), offline=True)
            predecessor = fact_id
    for prefix, table in (("l", "facts"), ("b", "blocks")):
        allocated = [int(row[0][1:]) for row in conn.execute(f"SELECT id FROM {table}")
                     if row[0].startswith(prefix) and row[0][1:].isdigit()]
        store.set_meta(f"seq_{prefix}", str(max([int(store.get_meta(f"seq_{prefix}") or 0), *allocated])))
    for old in ("line_zkeys", "line_history", "lines", "ingest"):
        if old in tables:
            conn.execute(f"DROP TABLE {old}")
