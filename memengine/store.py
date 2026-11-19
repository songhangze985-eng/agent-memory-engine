"""SQLite raw-turn journal and versioned 2D fact ledger."""

from __future__ import annotations

import json
import sqlite3
import time
import unicodedata
from contextlib import contextmanager
from http.client import HTTPException
from pathlib import Path

import numpy as np

from memengine.textutil import (ensure_step_tags, estimate_tokens, merge_step_ids, normalize_entity,
                                step_ids_in, strip_step_tags, tokenize_for_fts)
from memengine.vector import HashedTfidfEmbedder, hashed_content_terms, keep_hashed_dense_hit, normalize_vectors

DEFAULT_SCOPE = ("default", "default", "default")
_IDS = "SELECT value FROM json_each(?)"
_FTS_STOP = frozenset("a an the is was were am are be to of in it at on with for and or i my me you your we our what which when where how who".split())

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS blocks (
  id TEXT PRIMARY KEY, topic TEXT NOT NULL, created_at REAL NOT NULL,
  user_id TEXT NOT NULL DEFAULT 'default', agent_id TEXT NOT NULL DEFAULT 'default', session_id TEXT NOT NULL DEFAULT 'default'
);
CREATE TABLE IF NOT EXISTS raw_turns (
  turn_id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL DEFAULT 'default',
  agent_id TEXT NOT NULL DEFAULT 'default', session_id TEXT NOT NULL DEFAULT 'default',
  role TEXT NOT NULL DEFAULT 'user', text TEXT NOT NULL, step_tag INTEGER,
  event_ts REAL NOT NULL, ingest_ts REAL NOT NULL, token_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_raw_turns_scope_session ON raw_turns(user_id,agent_id,session_id,turn_id);
CREATE INDEX IF NOT EXISTS idx_raw_turns_event_ts ON raw_turns(user_id,agent_id,event_ts);
CREATE TABLE IF NOT EXISTS facts (
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL DEFAULT 'default', agent_id TEXT NOT NULL DEFAULT 'default',
  session_id TEXT NOT NULL DEFAULT 'default', block_id TEXT NOT NULL REFERENCES blocks(id), seq INTEGER NOT NULL,
  role TEXT NOT NULL DEFAULT 'user', formatted TEXT NOT NULL, plain TEXT NOT NULL,
  attribute TEXT NOT NULL DEFAULT '', value TEXT NOT NULL DEFAULT '', layout TEXT NOT NULL DEFAULT 'APPEND_ADJACENT_LINE',
  isolated INTEGER NOT NULL DEFAULT 0, is_conflict INTEGER NOT NULL DEFAULT 0, surprise REAL NOT NULL DEFAULT 0.25,
  status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','superseded','deleted')),
  revision INTEGER NOT NULL DEFAULT 0, supersedes_id TEXT REFERENCES facts(id) ON DELETE SET NULL,
  root_fact_id TEXT NOT NULL, event_time_start REAL NOT NULL, event_time_end REAL,
  ingest_time REAL NOT NULL, valid_until REAL, source_turn_ids TEXT NOT NULL DEFAULT '[]',
  source_step_ids TEXT NOT NULL DEFAULT '[]', hit_count INTEGER NOT NULL DEFAULT 0,
  CHECK ((status = 'active') = (valid_until IS NULL)), UNIQUE(root_fact_id, revision)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_active_root ON facts(root_fact_id) WHERE status='active';
CREATE INDEX IF NOT EXISTS idx_facts_scope_status ON facts(user_id,agent_id,status,session_id);
CREATE INDEX IF NOT EXISTS idx_facts_block_seq ON facts(block_id,seq);
CREATE INDEX IF NOT EXISTS idx_facts_root_rev ON facts(root_fact_id,revision);
CREATE INDEX IF NOT EXISTS idx_facts_event_time ON facts(user_id,agent_id,event_time_start);
CREATE INDEX IF NOT EXISTS idx_facts_supersedes ON facts(supersedes_id);
CREATE TABLE IF NOT EXISTS fact_entities (
  fact_id TEXT NOT NULL REFERENCES facts(id) ON DELETE CASCADE, user_id TEXT NOT NULL DEFAULT 'default',
  agent_id TEXT NOT NULL DEFAULT 'default', entity_norm TEXT NOT NULL, raw_key TEXT NOT NULL,
  key_type TEXT NOT NULL DEFAULT 'entity' CHECK(key_type IN ('entity','tag','noise')),
  UNIQUE(fact_id, entity_norm, key_type)
);
CREATE INDEX IF NOT EXISTS idx_fe_lookup ON fact_entities(user_id,agent_id,key_type,entity_norm);
CREATE INDEX IF NOT EXISTS idx_fe_fact ON fact_entities(fact_id);
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(fact_id UNINDEXED,user_id UNINDEXED,agent_id UNINDEXED,
  fts_tokens, tokenize="unicode61 tokenchars '_.-/'");
CREATE TRIGGER IF NOT EXISTS remove_fact_fts AFTER DELETE ON facts BEGIN
  DELETE FROM facts_fts WHERE rowid=old.rowid;
END;
CREATE TABLE IF NOT EXISTS fact_vectors (
  fact_id TEXT PRIMARY KEY REFERENCES facts(id) ON DELETE CASCADE, user_id TEXT NOT NULL DEFAULT 'default',
  agent_id TEXT NOT NULL DEFAULT 'default', dim INTEGER NOT NULL, model_name TEXT NOT NULL, vec BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fv_scope ON fact_vectors(user_id,agent_id,model_name);
CREATE TABLE IF NOT EXISTS observations (
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL DEFAULT 'default', agent_id TEXT NOT NULL DEFAULT 'default',
  content TEXT NOT NULL, confidence REAL NOT NULL DEFAULT 0.7, derived_from_fact_ids TEXT NOT NULL DEFAULT '[]',
  source_turn_ids TEXT NOT NULL DEFAULT '[]', status TEXT NOT NULL DEFAULT 'active',
  created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obs_scope_status ON observations(user_id,agent_id,status);
CREATE TABLE IF NOT EXISTS timeline (
  id INTEGER PRIMARY KEY AUTOINCREMENT, layer INTEGER NOT NULL, position INTEGER NOT NULL,
  t_start REAL NOT NULL, t_end REAL NOT NULL, payload TEXT NOT NULL,
  user_id TEXT NOT NULL DEFAULT 'default', agent_id TEXT NOT NULL DEFAULT 'default', session_id TEXT NOT NULL DEFAULT 'default'
);
"""


def key_type(key: str) -> str:
    key = unicodedata.normalize("NFKC", key).strip()
    if key == "话题:时事杂讯":
        return "noise"
    prefix, separator, _ = key.partition(":")
    if separator and prefix.strip().casefold() in {"类目", "category", "tag", "标签", "话题", "topic"}:
        return "tag"
    return "entity"


def _scope_where(scope, alias=""):
    where = f"{alias}user_id = ? AND {alias}agent_id = ?"
    return (where, scope[:2]) if scope[2] is None else (where + f" AND {alias}session_id = ?", scope)


def _status_where(include_superseded):
    return "f.status IN ('active','superseded')" if include_superseded else "f.status='active'"


def _dump(value):
    return json.dumps(value, ensure_ascii=False)


def _history_key(scope):
    escaped = [part.replace("%", "%25").replace(":", "%3A") for part in scope[:2]]
    return "history_tokens:" + ":".join(escaped)


class Store:
    def __init__(self, path: str | Path = ":memory:", *, embedder=None):
        self.path = str(path)
        self.embedder = embedder if hasattr(embedder, "encode") and hasattr(embedder, "model_name") else HashedTfidfEmbedder()
        self.fallback_embedder = HashedTfidfEmbedder()
        self.conn = sqlite3.connect(self.path, timeout=5)
        self.conn.row_factory = sqlite3.Row
        try:
            self.conn.executescript("PRAGMA busy_timeout=5000; PRAGMA journal_mode=WAL; "
                                    "PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON;" + _SCHEMA)
            self._migrate()
        except BaseException:
            self.conn.close()
            raise

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

    @contextmanager
    def read_snapshot(self):
        if self.conn.in_transaction:
            yield
        else:
            with self.conn:
                self.conn.execute("BEGIN")
                yield

    def _migrate(self):
        from memengine.migration import migrate_legacy

        with self.transaction():
            if self.get_meta("schema_version") != "2":
                migrate_legacy(self)
                self.set_meta("schema_version", "2")

    def close(self):
        self.conn.close()

    def commit(self):
        self.conn.commit()

    def get_meta(self, key):
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row is not None else None

    def set_meta(self, key, value):
        self.conn.execute("INSERT INTO meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def next_id(self, prefix):
        row = self.conn.execute("""INSERT INTO meta VALUES(?, '1') ON CONFLICT(key) DO UPDATE
                                   SET value=CAST(value AS INTEGER)+1 RETURNING value""", (f"seq_{prefix}",)).fetchone()
        return f"{prefix}{row[0]}"

    def append_turn(self, text, timestamp=None, *, event_ts=None, ingest_ts=None, scope=DEFAULT_SCOPE,
                    role="user", user_id=None, agent_id=None, session_id=None) -> int:
        scope = tuple(default if override is None else override for default, override in
                      zip(scope, (user_id, agent_id, session_id)))
        stamp = time.time() if timestamp is None else timestamp
        steps = step_ids_in(text)
        tokens = estimate_tokens(text)
        with self.transaction():
            cursor = self.conn.execute("""INSERT INTO raw_turns(user_id,agent_id,session_id,role,text,step_tag,event_ts,ingest_ts,token_count)
                                          VALUES(?,?,?,?,?,?,?,?,?)""",
                                       (*scope, role, text, steps[0] if steps else None,
                                        stamp if event_ts is None else event_ts, stamp if ingest_ts is None else ingest_ts, tokens))
            self.conn.execute("""INSERT INTO meta VALUES(?,?) ON CONFLICT(key) DO UPDATE
                                 SET value=CAST(value AS INTEGER)+CAST(excluded.value AS INTEGER)""",
                              (_history_key(scope), str(tokens)))
            return cursor.lastrowid

    def history_tokens(self, scope=DEFAULT_SCOPE):
        return int(self.get_meta(_history_key(scope)) or 0)

    def get_turn_window(self, turn_ids: list[int], window: int = 1, user_id="default", agent_id="default") -> list[dict]:
        if window < 0:
            raise ValueError("window must be nonnegative")
        if not turn_ids:
            return []
        rows = self.conn.execute(f"""WITH anchors AS (
               SELECT * FROM raw_turns WHERE user_id=? AND agent_id=? AND turn_id IN ({_IDS})
             ), neighbors AS (
               SELECT turn_id FROM anchors
               UNION SELECT n.turn_id FROM anchors a JOIN raw_turns n USING(user_id,agent_id,session_id)
                 WHERE n.turn_id<a.turn_id AND n.turn_id >= coalesce((
                   SELECT turn_id FROM raw_turns WHERE user_id=a.user_id AND agent_id=a.agent_id AND session_id=a.session_id
                     AND turn_id<a.turn_id ORDER BY turn_id DESC LIMIT 1 OFFSET ?), 0) AND ?>0
               UNION SELECT n.turn_id FROM anchors a JOIN raw_turns n USING(user_id,agent_id,session_id)
                 WHERE n.turn_id>a.turn_id AND n.turn_id <= coalesce((
                   SELECT turn_id FROM raw_turns WHERE user_id=a.user_id AND agent_id=a.agent_id AND session_id=a.session_id
                     AND turn_id>a.turn_id ORDER BY turn_id LIMIT 1 OFFSET ?), 9223372036854775807) AND ?>0
             ) SELECT * FROM raw_turns WHERE turn_id IN (SELECT turn_id FROM neighbors) ORDER BY turn_id""",
             (user_id, agent_id, _dump(turn_ids), max(0, window - 1), window, max(0, window - 1), window))
        return [dict(row) for row in rows]

    def create_block(self, topic, timestamp, *, scope=DEFAULT_SCOPE, event_time=None, is_conflict=False):
        block_id = self.next_id("b")
        self.conn.execute("INSERT INTO blocks(id,topic,created_at,user_id,agent_id,session_id) VALUES(?,?,?,?,?,?)",
                          (block_id, topic, timestamp, *scope))
        return block_id

    def get_block(self, block_id, *, scope=DEFAULT_SCOPE):
        where, args = _scope_where(scope)
        row = self.conn.execute(f"SELECT * FROM blocks WHERE id=? AND {where}", (block_id, *args)).fetchone()
        return dict(row) if row else None

    def _encoded(self, text, *, offline=False):
        backend = self.fallback_embedder if offline else self.embedder
        try:
            vector = normalize_vectors(backend.encode([strip_step_tags(text)]))[0]
        except (OSError, ValueError, TypeError, KeyError, HTTPException):
            backend = self.fallback_embedder
            vector = normalize_vectors(backend.encode([strip_step_tags(text)]))[0]
        return backend.model_name, len(vector), vector.astype("<f4", copy=False).tobytes()

    def _insert_version(self, data, zkeys, *, offline=False):
        columns = tuple(data)
        cursor = self.conn.execute(f"INSERT INTO facts({','.join(columns)}) VALUES({','.join('?' for _ in columns)})",
                                   tuple(data.values()))
        fact_id, user_id, agent_id = (data[key] for key in ("id", "user_id", "agent_id"))
        entities = [(fact_id, user_id, agent_id, normalize_entity(key), key, key_type(key)) for key in dict.fromkeys(zkeys)]
        self.conn.executemany("INSERT OR IGNORE INTO fact_entities VALUES(?,?,?,?,?,?)",
                              (record for record in entities if record[3]))
        self.conn.execute("INSERT INTO facts_fts(rowid,fact_id,user_id,agent_id,fts_tokens) VALUES(?,?,?,?,?)",
                          (cursor.lastrowid, fact_id, user_id, agent_id, tokenize_for_fts(data["plain"])))
        model, dim, vector = self._encoded(data["plain"], offline=offline)
        self.conn.execute("INSERT INTO fact_vectors VALUES(?,?,?,?,?,?)", (fact_id, user_id, agent_id, dim, model, vector))
        return fact_id

    @staticmethod
    def _version_data(*, fact_id, block_id, seq, scope, formatted, plain, attribute, value, surprise, layout, isolated,
                      is_conflict, role, ingest_time, event_time_start, event_time_end=None, source_turn_ids=None,
                      source_step_ids=None, status="active", revision=0, supersedes_id=None, root_fact_id=None,
                      valid_until=None, hit_count=0):
        return {"id": fact_id, "user_id": scope[0], "agent_id": scope[1], "session_id": scope[2], "block_id": block_id,
                "seq": seq, "role": role, "formatted": formatted, "plain": plain, "attribute": attribute or "", "value": value or "",
                "surprise": surprise if surprise is not None else 0.25, "layout": layout or "APPEND_ADJACENT_LINE",
                "isolated": int(bool(isolated)), "is_conflict": int(bool(is_conflict)), "status": status,
                "revision": revision, "supersedes_id": supersedes_id, "root_fact_id": root_fact_id or fact_id,
                "event_time_start": event_time_start if event_time_start is not None else ingest_time,
                "event_time_end": event_time_end, "ingest_time": ingest_time, "valid_until": valid_until,
                "source_turn_ids": _dump(merge_step_ids(source_turn_ids)),
                "source_step_ids": _dump(merge_step_ids(source_step_ids, step_ids_in(plain), step_ids_in(formatted))),
                "hit_count": hit_count}

    def insert_fact(self, *, block_id, formatted, plain, attribute="", value="", surprise=0.25,
                    layout="APPEND_ADJACENT_LINE", isolated=False, zkeys=(), ingest_time,
                    event_time_start=None, event_time_end=None, source_turn_ids=None, source_step_ids=None,
                    scope=DEFAULT_SCOPE, role="user", is_conflict=False) -> str:
        with self.transaction():
            if not self.get_block(block_id, scope=scope):
                raise ValueError("block not in fact scope")
            self._validate_turn_ids(source_turn_ids, scope)
            fact_id = self.next_id("l")
            seq = self.conn.execute("SELECT coalesce(max(seq),0)+1 FROM facts WHERE block_id=?", (block_id,)).fetchone()[0]
            data = self._version_data(fact_id=fact_id, block_id=block_id, seq=seq, scope=scope, formatted=formatted,
                                      plain=plain, attribute=attribute, value=value, surprise=surprise, layout=layout,
                                      isolated=isolated, is_conflict=is_conflict, role=role, ingest_time=ingest_time,
                                      event_time_start=event_time_start, event_time_end=event_time_end,
                                      source_turn_ids=source_turn_ids, source_step_ids=source_step_ids)
            return self._insert_version(data, zkeys)

    def _validate_turn_ids(self, turn_ids, scope):
        if turn_ids:
            where, args = _scope_where(scope)
            count = self.conn.execute(f"SELECT count(*) FROM raw_turns WHERE {where} AND turn_id IN ({_IDS})",
                                      (*args, _dump(list(turn_ids)))).fetchone()[0]
            if count != len(set(turn_ids)):
                raise ValueError("source turns must exist in the fact scope")

    def _resolve_fact(self, fact_id, *, scope, status):
        where, args = _scope_where(scope)
        return self.conn.execute(f"""SELECT * FROM facts WHERE {where} AND (id=? OR root_fact_id=? OR root_fact_id=(
                 SELECT root_fact_id FROM facts WHERE id=?)) AND {status} ORDER BY revision DESC LIMIT 1""",
                                 (*args, fact_id, fact_id, fact_id)).fetchone()

    def _exact_fact(self, fact_id, scope):
        where, args = _scope_where(scope)
        return self.conn.execute(f"SELECT * FROM facts WHERE {where} AND id=?", (*args, fact_id)).fetchone()

    def supersede_fact(self, old_fact_id, *, formatted, plain, attribute="", value="", surprise=0.85,
                       layout="OVERWRITE_SAME_LINE", isolated=None, zkeys=(), ingest_time, event_time_start=None,
                       event_time_end=None, current_turn_id=None, current_step_ids=None,
                       scope=DEFAULT_SCOPE, role="user", is_conflict=False) -> str | None:
        with self.transaction():
            old = self._exact_fact(old_fact_id, scope)
            if old is None or old["status"] != "active":
                return None
            turns = [] if current_turn_id is None else [current_turn_id]
            self._validate_turn_ids(turns, scope)
            ingest_time = max(ingest_time, old["ingest_time"])
            self.conn.execute("UPDATE facts SET status='superseded',valid_until=? WHERE id=?", (ingest_time, old["id"]))
            owner = (old["user_id"], old["agent_id"], old["session_id"])
            data = self._version_data(fact_id=self.next_id("l"), block_id=old["block_id"], seq=old["seq"], scope=owner,
                                      formatted=formatted, plain=plain, attribute=attribute, value=value, surprise=surprise,
                                      layout=layout, isolated=old["isolated"] if isolated is None else isolated,
                                      is_conflict=is_conflict, role=role, ingest_time=ingest_time, event_time_start=event_time_start,
                                      event_time_end=event_time_end, root_fact_id=old["root_fact_id"], supersedes_id=old["id"],
                                      revision=old["revision"] + 1, hit_count=old["hit_count"],
                                      source_turn_ids=merge_step_ids(json.loads(old["source_turn_ids"]), turns),
                                      source_step_ids=merge_step_ids(json.loads(old["source_step_ids"]), current_step_ids))
            return self._insert_version(data, zkeys)

    def delete_fact(self, fact_id, *, hard=False, ingest_time=None, scope=DEFAULT_SCOPE):
        with self.transaction():
            row = self._resolve_fact(fact_id, scope=scope, status="1" if hard else "status='active'")
            if row is None:
                return False
            if hard:
                self.conn.execute("DELETE FROM facts WHERE root_fact_id=?", (row["root_fact_id"],))
            else:
                stamp = time.time() if ingest_time is None else ingest_time
                self.conn.execute("UPDATE facts SET status='deleted',valid_until=? WHERE id=?",
                                  (max(stamp, row["ingest_time"]), row["id"]))
            return True

    def increment_hit_counts(self, fact_ids):
        if fact_ids:
            self.conn.execute(f"UPDATE facts SET hit_count=hit_count+1 WHERE status!='deleted' AND id IN ({_IDS})",
                              (_dump(list(fact_ids)),))

    def merge_line_mentions(self, fact_id, step_ids, *, scope=DEFAULT_SCOPE, turn_id=None):
        with self.transaction():
            row = self._resolve_fact(fact_id, scope=scope, status="status='active'")
            if row is not None:
                turns = [turn_id] if turn_id is not None else []
                self._validate_turn_ids(turns, scope)
                self.conn.execute("UPDATE facts SET source_turn_ids=?,source_step_ids=? WHERE id=?",
                                  (_dump(merge_step_ids(json.loads(row["source_turn_ids"]), turns)),
                                   _dump(merge_step_ids(json.loads(row["source_step_ids"]), step_ids)), row["id"]))

    @staticmethod
    def _fact(row, keys):
        fact = dict(row)
        steps = json.loads(fact["source_step_ids"])
        return {**fact, "text": ensure_step_tags(fact["plain"], steps), "zkeys": keys,
                "entity_z_keys": [key for key in keys if key_type(key) == "entity"],
                "category_z_keys": [key for key in keys if key_type(key) == "tag"],
                "isolated": bool(fact["isolated"]), "historical": fact["status"] == "superseded",
                "active": int(fact["status"] == "active"), "event_time": fact["event_time_start"],
                "updated_at": fact["ingest_time"], "is_conflict_or_override": bool(fact["is_conflict"]),
                "source_step_ids": steps, "source_turn_ids": json.loads(fact["source_turn_ids"])}

    def _load_facts(self, predicate, args, order="b.created_at,b.id,f.seq,f.revision"):
        with self.read_snapshot():
            rows = self.conn.execute(f"SELECT f.*,b.topic FROM facts f JOIN blocks b ON b.id=f.block_id WHERE {predicate} ORDER BY {order}",
                                     args).fetchall()
            if not rows:
                return []
            keys = {}
            for row in self.conn.execute(f"SELECT fact_id,raw_key FROM fact_entities WHERE fact_id IN ({_IDS}) ORDER BY raw_key",
                                         (_dump([row["id"] for row in rows]),)):
                keys.setdefault(row["fact_id"], []).append(row["raw_key"])
        return [self._fact(row, keys.get(row["id"], [])) for row in rows]

    def facts_by_ids(self, fact_ids, *, scope=DEFAULT_SCOPE):
        if not fact_ids:
            return []
        where, args = _scope_where(scope, "f.")
        return self._load_facts(f"{where} AND f.id IN ({_IDS})", (*args, _dump(list(fact_ids))))

    def snapshot(self, *, scope=DEFAULT_SCOPE, include_superseded=False):
        where, args = _scope_where(scope, "f.")
        return self._load_facts(f"{where} AND {_status_where(include_superseded)}", args)

    def _searchable(self, include_superseded):
        return _status_where(include_superseded) + (" AND NOT EXISTS (SELECT 1 FROM facts removed "
                 "WHERE removed.root_fact_id=f.root_fact_id AND removed.status='deleted')" if include_superseded else "")

    def get_fact_chain(self, root_fact_id, *, user_id="default", agent_id="default", session_id=None):
        where, args = _scope_where((user_id, agent_id, session_id), "f.")
        return self._load_facts(f"{where} AND f.root_fact_id=?", (*args, root_fact_id), "f.revision")

    def blocks_for_prompt(self, *, scope=DEFAULT_SCOPE, snapshot=None):
        blocks = {}
        for fact in self.snapshot(scope=scope) if snapshot is None else snapshot:
            block = blocks.setdefault(fact["block_id"], {"id": fact["block_id"], "topic": fact["topic"], "lines": []})
            block["lines"].append(fact)
        return list(blocks.values())

    def z_index(self, *, scope=DEFAULT_SCOPE, snapshot=None):
        index = {}
        for fact in self.snapshot(scope=scope) if snapshot is None else snapshot:
            for key in fact["zkeys"]:
                index.setdefault(key, []).append(fact)
        return index

    def entity_keys(self, *, scope=DEFAULT_SCOPE):
        where, args = _scope_where(scope, "f.")
        rows = self.conn.execute(f"""SELECT DISTINCT e.raw_key FROM fact_entities e JOIN facts f ON f.id=e.fact_id
                 WHERE {where} AND {self._searchable(True)} AND e.key_type='entity' ORDER BY e.raw_key""", args)
        return [row[0] for row in rows]

    def search_entity_inverted(self, entities, *, user_id="default", agent_id="default", session_id=None,
                               include_superseded=True, top_n=40, ids_only=False):
        normalized = sorted({normalize_entity(key) for key in entities if key_type(key) == "entity"} - {""})
        if not normalized or top_n <= 0:
            return []
        scope = (user_id, agent_id, session_id)
        where, args = _scope_where(scope, "f.")
        rows = self.conn.execute(f"""SELECT f.id,count(*) AS entity_matches FROM fact_entities e JOIN facts f ON f.id=e.fact_id
             WHERE e.user_id=? AND e.agent_id=? AND e.key_type='entity' AND e.entity_norm IN ({_IDS})
               AND {where} AND {self._searchable(include_superseded)} GROUP BY f.id
             ORDER BY entity_matches DESC, f.ingest_time DESC,f.id LIMIT ?""",
                                 (user_id, agent_id, _dump(normalized), *args, top_n))
        return self._search_results(rows, scope, ids_only)

    def search_bm25(self, query, *, user_id="default", agent_id="default", session_id=None,
                    include_superseded=True, top_n=40, ids_only=False):
        terms = list(dict.fromkeys(tokenize_for_fts(query).split()))
        terms = [term for term in terms if term not in _FTS_STOP] or terms
        if not terms or top_n <= 0:
            return []
        scope = (user_id, agent_id, session_id)
        where, args = _scope_where(scope, "f.")
        expression = " OR ".join(f'"{term}"' for term in terms[:128])
        rows = self.conn.execute(f"""SELECT f.id,bm25(facts_fts) AS bm25_score FROM facts_fts JOIN facts f ON f.rowid=facts_fts.rowid
             WHERE facts_fts MATCH ? AND {where} AND {self._searchable(include_superseded)} ORDER BY bm25_score,f.id LIMIT ?""",
                                 (expression, *args, top_n))
        return self._search_results(rows, scope, ids_only)

    def search_temporal(self, user_id="default", agent_id="default", start_ts=None, end_ts=None,
                        include_superseded=True, top_n=40, *, session_id=None, order_hint=None,
                        rev_index=None, entities=(), previous_only=False, ids_only=False):
        scope = (user_id, agent_id, session_id)
        where, args = _scope_where(scope, "f.")
        conditions = [where, self._searchable(include_superseded)]
        params = list(args)
        if start_ts is not None:
            conditions.append("f.event_time_start>=?")
            params.append(start_ts)
        if end_ts is not None:
            conditions.append("f.event_time_start<?")
            params.append(end_ts)
        if rev_index is not None:
            conditions.append("f.revision=?")
            params.append(rev_index)
        if previous_only:
            conditions.append("f.status='superseded' AND NOT EXISTS (SELECT 1 FROM facts newer "
                              "WHERE newer.root_fact_id=f.root_fact_id AND newer.status='superseded' AND newer.revision>f.revision)")
        if entities:
            conditions.append(f"EXISTS(SELECT 1 FROM fact_entities e WHERE e.fact_id=f.id AND e.key_type='entity' AND e.entity_norm IN ({_IDS}))")
            params.append(_dump(list(entities)))
        order = "ASC" if order_hint == "earliest" else "DESC"
        rows = self.conn.execute(f"SELECT f.id FROM facts f INDEXED BY idx_facts_event_time WHERE {' AND '.join(conditions)} "
                                 f"ORDER BY f.event_time_start {order},f.revision {order},f.id LIMIT ?", (*params, max(0, top_n)))
        return self._search_results(rows, scope, ids_only)

    def search_vector(self, query, *, user_id="default", agent_id="default", session_id=None,
                      include_superseded=True, top_n=40, ids_only=False):
        if not query.strip() or top_n <= 0:
            return []
        scope = (user_id, agent_id, session_id)
        where, args = _scope_where(scope, "f.")
        rows = self.conn.execute(f"""SELECT v.*, f.plain FROM fact_vectors v JOIN facts f ON f.id=v.fact_id
             WHERE v.user_id=? AND v.agent_id=? AND {where} AND {self._searchable(include_superseded)} ORDER BY v.fact_id""",
                                 (user_id, agent_id, *args)).fetchall()
        groups = {}
        for row in rows:
            groups.setdefault((row["model_name"], row["dim"]), []).append(row)
        hits = []
        query_terms = hashed_content_terms(query)
        for backend in {self.fallback_embedder.model_name: self.fallback_embedder, self.embedder.model_name: self.embedder}.values():
            selected = [(size, values) for (model, size), values in groups.items() if model == backend.model_name]
            if not selected:
                continue
            try:
                encoded = normalize_vectors(backend.encode([query]))[0]
            except (OSError, ValueError, TypeError, KeyError, HTTPException):
                continue
            hashed = isinstance(backend, HashedTfidfEmbedder)
            for dim, values in selected:
                if dim != len(encoded):
                    continue
                matrix = np.frombuffer(b"".join(row["vec"] for row in values), dtype="<f4").reshape(-1, dim)
                for row, score in zip(values, matrix @ encoded):
                    score = float(score)
                    if hashed:
                        if not keep_hashed_dense_hit(score, query_terms, row["plain"]):
                            continue
                    elif score <= 0:
                        continue
                    hits.append({"id": row["fact_id"], "vector_score": score})
        hits.sort(key=lambda fact: (-fact["vector_score"], fact["id"]))
        return self._search_results(hits[:top_n], scope, ids_only)

    def _search_results(self, rows, scope, ids_only):
        hits = [dict(row) for row in rows]
        if ids_only:
            return hits
        facts = {fact["id"]: fact for fact in self.facts_by_ids([row["id"] for row in hits], scope=scope)}
        return [{**facts[row["id"]], **row} for row in hits if row["id"] in facts]

    def insert_observation(self, content, *, user_id="default", agent_id="default", confidence=0.7,
                           derived_from_fact_ids=(), source_turn_ids=(), timestamp=None):
        stamp = time.time() if timestamp is None else timestamp
        scope = (user_id, agent_id, None)
        with self.transaction():
            self._validate_turn_ids(source_turn_ids, scope)
            if derived_from_fact_ids and len(self.facts_by_ids(derived_from_fact_ids, scope=scope)) != len(set(derived_from_fact_ids)):
                raise ValueError("observation facts must exist in the observation scope")
            obs_id = self.next_id("o")
            self.conn.execute("INSERT INTO observations VALUES(?,?,?,?,?,?,?,'active',?,?)",
                              (obs_id, user_id, agent_id, content, confidence, _dump(list(derived_from_fact_ids)),
                               _dump(list(source_turn_ids)), stamp, stamp))
            return obs_id

    def list_observations(self, *, user_id="default", agent_id="default"):
        rows = self.conn.execute("SELECT * FROM observations WHERE user_id=? AND agent_id=? AND status='active' ORDER BY created_at,id",
                                 (user_id, agent_id))
        return [{**dict(row), "derived_from_fact_ids": json.loads(row["derived_from_fact_ids"]),
                 "source_turn_ids": json.loads(row["source_turn_ids"])} for row in rows]

    def insert_line(self, *, timestamp, event_time=None, raw_dist=None, **fields):
        return self.insert_fact(ingest_time=timestamp, event_time_start=event_time, **fields)

    def update_line(self, line_id, *, timestamp, event_time=None, raw_dist=None, source_step_ids=None, **fields):
        with self.transaction():
            row = self._resolve_fact(line_id, scope=fields.get("scope", DEFAULT_SCOPE), status="status='active'")
            return row is not None and self.supersede_fact(row["id"], ingest_time=timestamp, event_time_start=event_time,
                                                           current_step_ids=source_step_ids, **fields) is not None

    def get_line(self, line_id, *, scope=DEFAULT_SCOPE, include_deleted=False):
        with self.read_snapshot():
            row = self._resolve_fact(line_id, scope=scope, status="1" if include_deleted else "status='active'")
            return self.facts_by_ids([row["id"]], scope=scope)[0] if row is not None else None

    def historical_versions(self, line_ids, *, scope=DEFAULT_SCOPE):
        if not line_ids:
            return []
        where, args = _scope_where(scope, "f.")
        return self._load_facts(f"{where} AND f.root_fact_id IN (SELECT root_fact_id FROM facts WHERE id IN ({_IDS})) "
                                "AND f.status!='active'", (*args, _dump(list(line_ids))), "f.root_fact_id,f.revision")

    def active_snapshot(self, *, scope=DEFAULT_SCOPE):
        return [{"block_id": block["id"], "topic": block["topic"], "lines": block["lines"]}
                for block in self.blocks_for_prompt(scope=scope)]

    def save_timeline_node(self, layer, position, node, *, scope=DEFAULT_SCOPE, event_time=None):
        where, args = _scope_where(scope)
        payload = _dump({key: value for key, value in node.items() if key not in {"t_start", "t_end"}})
        updated = self.conn.execute(f"UPDATE timeline SET t_end=?,payload=? WHERE {where} AND layer=? AND position=?",
                                    (node["t_end"], payload, *args, layer, position))
        if not updated.rowcount:
            self.conn.execute("INSERT INTO timeline(layer,position,t_start,t_end,payload,user_id,agent_id,session_id) VALUES(?,?,?,?,?,?,?,?)",
                              (layer, position, node["t_start"], node["t_end"], payload, *scope))

    def load_timeline(self, layer, *, scope=DEFAULT_SCOPE):
        where, args = _scope_where(scope)
        rows = self.conn.execute(f"SELECT * FROM timeline WHERE {where} AND layer=? ORDER BY position,id", (*args, layer))
        return [self._timeline_node(row) for row in rows]

    def last_timeline_node(self, layer, *, scope=DEFAULT_SCOPE):
        where, args = _scope_where(scope)
        row = self.conn.execute(f"SELECT * FROM timeline WHERE {where} AND layer=? ORDER BY position DESC,id DESC LIMIT 1",
                                (*args, layer)).fetchone()
        return (row["position"], self._timeline_node(row)) if row else (-1, None)

    @staticmethod
    def _timeline_node(row):
        return {"t_start": row["t_start"], "t_end": row["t_end"], **json.loads(row["payload"])}
