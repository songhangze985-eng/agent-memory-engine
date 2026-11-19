"""Scoped SQLite persistence for the notebook, revisions and dual timeline."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from memengine.textutil import ensure_step_tags, estimate_tokens, merge_step_ids, step_ids_in

DEFAULT_SCOPE = ("default", "default", "default")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS blocks (
  id TEXT PRIMARY KEY, topic TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS lines (
  id TEXT PRIMARY KEY, block_id TEXT NOT NULL REFERENCES blocks(id), seq INTEGER NOT NULL,
  formatted TEXT NOT NULL, plain TEXT NOT NULL, attribute TEXT, value TEXT, surprise REAL,
  raw_dist REAL, layout TEXT, isolated INTEGER NOT NULL DEFAULT 0,
  revision INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL, updated_at REAL NOT NULL,
  active INTEGER NOT NULL DEFAULT 1, source_step_ids TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS line_zkeys (
  line_id TEXT NOT NULL REFERENCES lines(id) ON DELETE CASCADE, zkey TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS line_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT, line_id TEXT NOT NULL REFERENCES lines(id) ON DELETE CASCADE,
  revision INTEGER NOT NULL, formatted TEXT NOT NULL, plain TEXT NOT NULL, attribute TEXT,
  value TEXT, zkeys TEXT NOT NULL, surprise REAL, layout TEXT, t_start REAL, t_end REAL,
  source_step_ids TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS timeline (
  id INTEGER PRIMARY KEY AUTOINCREMENT, layer INTEGER NOT NULL, position INTEGER NOT NULL,
  t_start REAL NOT NULL, t_end REAL NOT NULL, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ingest (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, ts REAL);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

_SCOPE_COLUMNS = {name: "TEXT NOT NULL DEFAULT 'default'" for name in ("user_id", "agent_id", "session_id")}
_EVENT_COLUMNS = {"event_time": "REAL", "valid_until": "REAL", "is_conflict": "INTEGER NOT NULL DEFAULT 0"}
_ROLE = {"role": "TEXT NOT NULL DEFAULT 'user'"}
_STEPS = {"source_step_ids": "TEXT NOT NULL DEFAULT '[]'"}


def key_type(key: str) -> str:
    if key.startswith("类目:"):
        return "category"
    if key == "话题:时事杂讯":
        return "noise"
    return "entity"


def _scope_where(scope, alias=""):
    where = f"{alias}user_id = ? AND {alias}agent_id = ?"
    return (where, scope[:2]) if scope[2] is None else (where + f" AND {alias}session_id = ?", scope)


def _dump(value):
    return json.dumps(value, ensure_ascii=False)


def _history_key(scope):
    escaped = [part.replace("%", "%25").replace(":", "%3A") for part in scope[:2]]
    return "history_tokens:" + ":".join(escaped)


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, timeout=5)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("PRAGMA busy_timeout=5000; PRAGMA journal_mode=WAL; "
                                "PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON;" + _SCHEMA)
        self._migrate()

    def _migrate(self):
        with self.transaction():
            for table, additions in {
                "blocks": {**_SCOPE_COLUMNS, **_EVENT_COLUMNS},
                "lines": {**_SCOPE_COLUMNS, **_EVENT_COLUMNS, **_ROLE, **_STEPS},
                "ingest": {**_SCOPE_COLUMNS, **_EVENT_COLUMNS, **_ROLE, "created_at": "REAL"},
                "timeline": {**_SCOPE_COLUMNS, **_EVENT_COLUMNS, "created_at": "REAL"},
                "line_history": {**_EVENT_COLUMNS, **_STEPS, **_ROLE},
                "line_zkeys": {"key_type": "TEXT NOT NULL DEFAULT 'entity'"},
            }.items():
                existing = {row["name"] for row in self.conn.execute(f"PRAGMA table_info({table})")}
                for column, definition in additions.items():
                    if column not in existing:
                        self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
                if "event_time" in additions and "event_time" not in existing:
                    source = {"ingest": "ts", "timeline": "t_start", "line_history": "t_start"}.get(table, "created_at")
                    self.conn.execute(f"UPDATE {table} SET event_time = {source}")
                if "created_at" in additions and "created_at" not in existing:
                    self.conn.execute(f"UPDATE {table} SET created_at = event_time")
                if table == "line_history" and "valid_until" not in existing:
                    self.conn.execute("UPDATE line_history SET valid_until = t_end")
                if table == "line_zkeys" and "key_type" not in existing:
                    self.conn.execute("""UPDATE line_zkeys SET key_type = CASE WHEN zkey LIKE '类目:%' THEN 'category'
                                      WHEN zkey = '话题:时事杂讯' THEN 'noise' ELSE 'entity' END""")
            if self.get_meta("scoped_schema_v1") is None:
                self.conn.execute("DELETE FROM line_zkeys WHERE rowid NOT IN (SELECT MIN(rowid) FROM line_zkeys GROUP BY line_id, zkey)")
                totals = {}
                for row in self.conn.execute("SELECT user_id, agent_id, text FROM ingest"):
                    key = _history_key((row['user_id'], row['agent_id']))
                    totals[key] = totals.get(key, 0) + estimate_tokens(row["text"])
                for key, total in totals.items():
                    self.set_meta(key, str(total))
                self.set_meta("scoped_schema_v1", "1")
            for name, target in {
                "idx_lines_block": "lines(block_id, seq)",
                "idx_line_zkeys_line": "line_zkeys(line_id)",
                "idx_line_zkeys_type_key": "line_zkeys(key_type, zkey)",
                "idx_line_history_line": "line_history(line_id, revision)",
                "idx_lines_scope": "lines(user_id, agent_id, active, session_id)",
                "idx_timeline_scope": "timeline(user_id, agent_id, session_id, layer, position)",
            }.items():
                self.conn.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {target}")
            self.conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_line_zkey ON line_zkeys(line_id, zkey)")

    @contextmanager
    def transaction(self):
        if self.conn.in_transaction:
            self.conn.execute("SAVEPOINT memory_write")
            try:
                yield
            except BaseException:
                self.conn.execute("ROLLBACK TO memory_write")
                raise
            finally:
                self.conn.execute("RELEASE memory_write")
        else:
            with self.conn:
                self.conn.execute("BEGIN IMMEDIATE")
                yield

    def close(self) -> None:
        self.conn.close()

    def commit(self) -> None:
        self.conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else row["value"]

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT INTO meta VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))

    def next_id(self, prefix: str) -> str:
        row = self.conn.execute("""INSERT INTO meta VALUES(?, '1')
                                   ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1
                                   RETURNING value""", (f"seq_{prefix}",)).fetchone()
        return f"{prefix}{row['value']}"

    def create_block(self, topic: str, timestamp: float, *, scope=DEFAULT_SCOPE, event_time=None, is_conflict=False) -> str:
        block_id = self.next_id("b")
        self.conn.execute("""INSERT INTO blocks(id, topic, created_at, event_time, user_id, agent_id, session_id, is_conflict)
                             VALUES(?, ?, ?, ?, ?, ?, ?, ?)""", (block_id, topic, timestamp, event_time if event_time is not None else timestamp, *scope, is_conflict))
        return block_id

    def get_block(self, block_id: str, *, scope=DEFAULT_SCOPE) -> dict | None:
        where, args = _scope_where(scope)
        row = self.conn.execute(f"SELECT * FROM blocks WHERE id = ? AND {where}", (block_id, *args)).fetchone()
        return dict(row) if row is not None else None

    def block_by_topic(self, topic: str, *, scope=DEFAULT_SCOPE) -> dict | None:
        where, args = _scope_where(scope)
        row = self.conn.execute(f"SELECT * FROM blocks WHERE topic = ? AND {where} ORDER BY created_at LIMIT 1", (topic, *args)).fetchone()
        return dict(row) if row is not None else None

    def insert_line(self, *, block_id, formatted, plain, attribute, value, surprise, raw_dist, layout,
                    isolated, zkeys, timestamp, source_step_ids=None, scope=DEFAULT_SCOPE, role="user",
                    event_time=None, is_conflict=False) -> str:
        line_id = self.next_id("l")
        seq = self.conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM lines WHERE block_id = ?", (block_id,)).fetchone()[0]
        steps = merge_step_ids(source_step_ids, step_ids_in(plain), step_ids_in(formatted))
        self.conn.execute("""INSERT INTO lines(id, block_id, seq, formatted, plain, attribute, value, surprise, raw_dist,
                             layout, isolated, created_at, updated_at, source_step_ids, user_id, agent_id, session_id,
                             role, event_time, is_conflict) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                          (line_id, block_id, seq, formatted, plain, attribute, value, surprise, raw_dist, layout,
                           isolated, timestamp, timestamp, _dump(steps), *scope, role,
                           event_time if event_time is not None else timestamp, is_conflict))
        self._replace_zkeys(line_id, zkeys)
        return line_id

    def get_line(self, line_id: str, *, scope=DEFAULT_SCOPE, include_deleted=False) -> dict | None:
        where, args = _scope_where(scope)
        row = self.conn.execute(f"SELECT * FROM lines WHERE id = ? AND {where} AND (active = 1 OR ?)",
                                (line_id, *args, include_deleted)).fetchone()
        if row is None:
            return None
        keys = [key[0] for key in self.conn.execute("SELECT zkey FROM line_zkeys WHERE line_id = ? ORDER BY zkey", (line_id,))]
        return self._fact(row, keys)

    def archive_line(self, line: dict, t_end: float) -> None:
        self.conn.execute("""INSERT INTO line_history(line_id, revision, formatted, plain, attribute, value, zkeys,
                             surprise, layout, t_start, t_end, valid_until, source_step_ids, role, event_time, is_conflict)
                             VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                          (line["id"], line["revision"], line["formatted"], line["plain"], line["attribute"], line["value"],
                           _dump(line["zkeys"]), line["surprise"], line["layout"], line["updated_at"], t_end, t_end,
                           _dump(line["source_step_ids"]), line["role"], line["event_time"], line["is_conflict"]))

    def update_line(self, line_id, *, formatted, plain, attribute, value, surprise, raw_dist, layout,
                    isolated=None, zkeys, timestamp, source_step_ids=None, scope=DEFAULT_SCOPE, role="user",
                    event_time=None, is_conflict=False) -> bool:
        with self.transaction():
            current = self.get_line(line_id, scope=scope)
            if current is None:
                return False
            self.archive_line(current, timestamp)
            steps = merge_step_ids(current["source_step_ids"], source_step_ids, step_ids_in(plain), step_ids_in(formatted))
            self.conn.execute("""UPDATE lines SET formatted=?, plain=?, attribute=?, value=?, surprise=?, raw_dist=?,
                                 layout=?, isolated=?, revision=revision+1, updated_at=?, source_step_ids=?, role=?,
                                 event_time=?, is_conflict=? WHERE id=?""",
                              (formatted, plain, attribute, value, surprise, raw_dist, layout,
                               current["isolated"] if isolated is None else isolated, timestamp, _dump(steps), role,
                               event_time if event_time is not None else timestamp, is_conflict, line_id))
            self._replace_zkeys(line_id, zkeys)
            return True

    def delete_line(self, line_id, *, timestamp, hard=False, scope=DEFAULT_SCOPE) -> bool:
        with self.transaction():
            line = self.get_line(line_id, scope=scope, include_deleted=hard)
            if line is None:
                return False
            if hard:
                self.conn.execute("DELETE FROM line_history WHERE line_id = ?", (line_id,))
                self.conn.execute("DELETE FROM line_zkeys WHERE line_id = ?", (line_id,))
                self.conn.execute("DELETE FROM lines WHERE id = ?", (line_id,))
            else:
                self.archive_line(line, timestamp)
                self.conn.execute("UPDATE lines SET active=0, valid_until=? WHERE id=?", (timestamp, line_id))
            return True

    def _replace_zkeys(self, line_id, keys):
        self.conn.execute("DELETE FROM line_zkeys WHERE line_id = ?", (line_id,))
        self.conn.executemany("INSERT INTO line_zkeys(line_id, zkey, key_type) VALUES(?, ?, ?)",
                              [(line_id, key, key_type(key)) for key in dict.fromkeys(keys)])

    def append_ingest(self, text, timestamp, *, scope=DEFAULT_SCOPE, role="user", event_time=None):
        self.conn.execute("""INSERT INTO ingest(text, ts, created_at, event_time, user_id, agent_id, session_id, role)
                             VALUES(?, ?, ?, ?, ?, ?, ?, ?)""", (text, timestamp, timestamp, event_time if event_time is not None else timestamp, *scope, role))
        self.conn.execute("""INSERT INTO meta VALUES(?, ?) ON CONFLICT(key) DO UPDATE
                             SET value = CAST(value AS INTEGER) + CAST(excluded.value AS INTEGER)""",
                          (_history_key(scope), str(estimate_tokens(text))))

    def history_tokens(self, scope=DEFAULT_SCOPE):
        return int(self.get_meta(_history_key(scope)) or 0)

    @staticmethod
    def _fact(row, keys):
        fact = dict(row)
        steps = merge_step_ids(json.loads(fact.get("source_step_ids") or "[]"), step_ids_in(fact["plain"]), step_ids_in(fact["formatted"]))
        return {**fact, "text": ensure_step_tags(fact["plain"], steps), "zkeys": keys,
                "entity_z_keys": [key for key in keys if key_type(key) == "entity"],
                "category_z_keys": [key for key in keys if key_type(key) == "category"],
                "isolated": bool(fact.get("isolated")), "historical": False,
                "is_conflict_or_override": bool(fact.get("is_conflict")), "source_step_ids": steps}

    def snapshot(self, *, scope=DEFAULT_SCOPE) -> list[dict]:
        if not self.conn.in_transaction:
            with self.conn:
                self.conn.execute("BEGIN")
                return self._snapshot_rows(scope)
        return self._snapshot_rows(scope)

    def _snapshot_rows(self, scope):
        where, args = _scope_where(scope, "l.")
        rows = self.conn.execute(f"""SELECT l.*, b.topic FROM lines l JOIN blocks b ON b.id = l.block_id
                                     WHERE l.active = 1 AND {where} ORDER BY b.created_at, b.id, l.seq""", args).fetchall()
        ids = {row["id"] for row in rows}
        if not ids:
            return []
        keys = {}
        for row in self.conn.execute("SELECT line_id, zkey FROM line_zkeys WHERE line_id IN (SELECT value FROM json_each(?)) ORDER BY zkey", (_dump(list(ids)),)):
            keys.setdefault(row["line_id"], []).append(row["zkey"])
        return [self._fact(row, keys.get(row["id"], [])) for row in rows]

    def greeting_keys(self, query, *, scope=DEFAULT_SCOPE):
        where, args = _scope_where(scope, "l.")
        rows = self.conn.execute(f"""SELECT DISTINCT k.zkey FROM lines l JOIN line_zkeys k ON k.line_id = l.id
                                     WHERE l.active=1 AND {where} AND k.key_type='entity'
                                     AND instr(lower(?), lower(substr(k.zkey, instr(k.zkey, ':') + 1))) > 0""",
                                 (*args, query))
        return [row[0] for row in rows]

    def blocks_for_prompt(self, *, scope=DEFAULT_SCOPE, snapshot=None) -> list[dict]:
        blocks = {}
        for fact in self.snapshot(scope=scope) if snapshot is None else snapshot:
            block = blocks.setdefault(fact["block_id"], {"id": fact["block_id"], "topic": fact["topic"], "lines": []})
            block["lines"].append(fact)
        return list(blocks.values())

    def z_index(self, *, scope=DEFAULT_SCOPE, snapshot=None) -> dict[str, list[dict]]:
        index = {}
        for fact in self.snapshot(scope=scope) if snapshot is None else snapshot:
            for key in fact["zkeys"]:
                index.setdefault(key, []).append(fact)
        return index

    def historical_versions(self, line_ids, *, scope=DEFAULT_SCOPE) -> list[dict]:
        if not line_ids:
            return []
        where, args = _scope_where(scope, "l.")
        rows = self.conn.execute(f"""SELECT h.*, l.block_id, l.session_id, b.topic FROM line_history h
                                     JOIN lines l ON l.id = h.line_id JOIN blocks b ON b.id = l.block_id
                                     WHERE {where} AND h.line_id IN (SELECT value FROM json_each(?))
                                     ORDER BY h.line_id, h.revision""", (*args, _dump(list(line_ids))))
        versions = []
        for row in rows:
            fact = self._fact(row, json.loads(row["zkeys"]))
            versions.append({**fact, "id": f"{row['line_id']}#{row['revision']}", "historical": True,
                             "updated_at": row["t_start"], "is_conflict_or_override": True})
        return versions

    def save_timeline_node(self, layer_no, position, node, *, scope=DEFAULT_SCOPE, event_time=None):
        payload = _dump({key: value for key, value in node.items() if key not in {"t_start", "t_end"}})
        where, args = _scope_where(scope)
        conflict = bool(node.get("is_conflict"))
        updated = self.conn.execute(f"""UPDATE timeline SET t_end=?, payload=?, is_conflict=?
                                       WHERE {where} AND layer=? AND position=?""",
                                    (node["t_end"], payload, conflict, *args, layer_no, position))
        if not updated.rowcount:
            self.conn.execute("""INSERT INTO timeline(layer, position, t_start, t_end, payload, created_at, event_time,
                                 user_id, agent_id, session_id, is_conflict) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                              (layer_no, position, node["t_start"], node["t_end"], payload, node["t_start"],
                               event_time if event_time is not None else node["t_start"], *scope, conflict))

    def load_timeline(self, layer_no, *, scope=DEFAULT_SCOPE) -> list[dict]:
        where, args = _scope_where(scope)
        rows = self.conn.execute(f"SELECT t_start, t_end, payload FROM timeline WHERE {where} AND layer = ? ORDER BY position, id", (*args, layer_no))
        return [{"t_start": row["t_start"], "t_end": row["t_end"], **json.loads(row["payload"])} for row in rows]

    def last_timeline_node(self, layer_no, *, scope=DEFAULT_SCOPE):
        where, args = _scope_where(scope)
        row = self.conn.execute(f"""SELECT position, t_start, t_end, payload FROM timeline WHERE {where} AND layer = ?
                                    ORDER BY position DESC, id DESC LIMIT 1""", (*args, layer_no)).fetchone()
        if row is None:
            return -1, None
        return row["position"], {"t_start": row["t_start"], "t_end": row["t_end"], **json.loads(row["payload"])}

    def active_snapshot(self, *, scope=DEFAULT_SCOPE) -> list[dict]:
        return [{"block_id": block["id"], "topic": block["topic"], "lines": block["lines"]}
                for block in self.blocks_for_prompt(scope=scope)]
