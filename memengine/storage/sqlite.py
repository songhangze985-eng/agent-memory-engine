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

from memengine.anchors import turn_context

from memengine.identifiers import extract_identifiers
from memengine.lifecycle import DAY, PROTECTED, content_hash, explicit_preference, is_constraint, is_taboo
from memengine.textutil import (ensure_step_tags, estimate_tokens, merge_step_ids, normalize_entity,
                                step_ids_in, strip_step_tags, tokenize_for_fts)
from memengine.storage.base import DEFAULT_SCOPE, IDS as _IDS, SCOPES, scope_where as _scope_where
from memengine.storage.chronicle import Chronicle
from memengine.storage.vector_cache import EntityKeyIndex, IncrementalVectorCache
from memengine.storage.workspace import WorkspaceStore
from memengine.vector import HashedTfidfEmbedder, normalize_vectors

_FTS_STOP = frozenset("a an the is was were am are be to of in it at on with for and or i my me you your we our what which when where how who".split())
FTS_TOKENIZER = "unicode61 tokenchars '._-/@#+:'"

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
  fts_tokens, tokenize="{fts_tokenizer}");
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
CREATE VIRTUAL TABLE IF NOT EXISTS facts_cold_fts USING fts5(fact_id UNINDEXED,user_id UNINDEXED,agent_id UNINDEXED,
  fts_tokens, tokenize="{fts_tokenizer}");
CREATE VIRTUAL TABLE IF NOT EXISTS raw_turns_fts USING fts5(fts_tokens, tokenize="{fts_tokenizer}");
CREATE TABLE IF NOT EXISTS fact_anchors (
  fact_id TEXT PRIMARY KEY REFERENCES facts(id) ON DELETE CASCADE,
  turn_id INTEGER NOT NULL REFERENCES raw_turns(turn_id),
  content TEXT NOT NULL, model_name TEXT NOT NULL, dim INTEGER NOT NULL, vec BLOB NOT NULL,
  peer INTEGER NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS facts_anchor_fts USING fts5(fts_tokens, tokenize="{fts_tokenizer}");
CREATE TRIGGER IF NOT EXISTS remove_anchor_fts AFTER DELETE ON facts BEGIN
  DELETE FROM facts_anchor_fts WHERE rowid=old.rowid;
END;
CREATE TABLE IF NOT EXISTS exact_identifiers (
  memory_id TEXT NOT NULL REFERENCES facts(id) ON DELETE CASCADE, ident TEXT NOT NULL,
  ident_norm TEXT NOT NULL, ident_type TEXT NOT NULL, UNIQUE(memory_id,ident_norm)
);
CREATE INDEX IF NOT EXISTS idx_exact_ident_norm ON exact_identifiers(ident_norm);
CREATE INDEX IF NOT EXISTS idx_exact_ident_mem ON exact_identifiers(memory_id);
CREATE TRIGGER IF NOT EXISTS remove_cold_fact_fts AFTER DELETE ON facts BEGIN
  DELETE FROM facts_cold_fts WHERE rowid=old.rowid;
END;
CREATE TABLE IF NOT EXISTS core_profiles (
  user_id TEXT NOT NULL, agent_id TEXT NOT NULL, profile_text TEXT NOT NULL,
  procedural_rules TEXT NOT NULL, prefix_version TEXT NOT NULL, updated_at REAL NOT NULL,
  PRIMARY KEY(user_id,agent_id)
);
CREATE TABLE IF NOT EXISTS tool_calls (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, agent_id TEXT NOT NULL, session_id TEXT NOT NULL,
  tool_call_id TEXT NOT NULL, trigger_text TEXT NOT NULL, technique TEXT NOT NULL, success INTEGER NOT NULL,
  source_turn_ids TEXT NOT NULL, derived_from_ids TEXT NOT NULL, created_at REAL NOT NULL,
  UNIQUE(user_id,agent_id,tool_call_id)
);
CREATE INDEX IF NOT EXISTS idx_tool_call_cursor ON tool_calls(user_id,agent_id,seq);
CREATE TABLE IF NOT EXISTS timeline (
  id INTEGER PRIMARY KEY AUTOINCREMENT, layer INTEGER NOT NULL, position INTEGER NOT NULL,
  t_start REAL NOT NULL, t_end REAL NOT NULL, payload TEXT NOT NULL,
  user_id TEXT NOT NULL DEFAULT 'default', agent_id TEXT NOT NULL DEFAULT 'default', session_id TEXT NOT NULL DEFAULT 'default'
);
""".format(fts_tokenizer=FTS_TOKENIZER)


def key_type(key: str) -> str:
    key = unicodedata.normalize("NFKC", key).strip()
    if key == "话题:时事杂讯":
        return "noise"
    prefix, separator, _ = key.partition(":")
    if separator and prefix.strip().casefold() in {"类目", "category", "tag", "标签", "话题", "topic"}:
        return "tag"
    return "entity"


def _status_where(include_superseded):
    return "f.status IN ('active','superseded')" if include_superseded else "f.status='active'"


def _live_observation_sources(alias):
    return f"""NOT EXISTS (
        SELECT 1 FROM json_each({alias}.derived_from_ids) j
        WHERE NOT EXISTS (
            SELECT 1 FROM facts source WHERE source.id=j.value
              AND source.user_id={alias}.user_id AND source.agent_id={alias}.agent_id
              AND (source.status='active' OR ({alias}.canonical_entity_id IS NOT NULL
                  AND source.status='superseded' AND EXISTS (
                      SELECT 1 FROM facts active WHERE active.root_fact_id=source.root_fact_id
                        AND active.status='active'
                  )))
        )
    )"""


def _dump(value):
    return json.dumps(value, ensure_ascii=False)


def _history_key(scope):
    escaped = [part.replace("%", "%25").replace(":", "%3A") for part in scope[:2]]
    return "history_tokens:" + ":".join(escaped)


class SQLiteStore(Chronicle, WorkspaceStore):
    _json_aggregate = "json_group_array"
    _json_pair = "json_array"
    def __init__(self, path: str | Path = ":memory:", *, embedder=None):
        self.path = str(path)
        self.embedder = embedder if hasattr(embedder, "encode") and hasattr(embedder, "model_name") else HashedTfidfEmbedder()
        self.fallback_embedder = HashedTfidfEmbedder()
        self._write_vectors = None
        self.vector_cache = IncrementalVectorCache()
        self.entity_cache = EntityKeyIndex()
        self.enable_vector_cache = True
        self._vector_view = None
        self._write_depth = 0
        self._write_seq = 0
        self._snapshot_depth = 0
        self.conn = sqlite3.connect(self.path, timeout=5, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.create_function("ame_vector_missing", 3, self._vector_missing)
        try:
            self.conn.executescript("PRAGMA busy_timeout=5000; PRAGMA journal_mode=WAL; "
                                    "PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON;" + _SCHEMA)
            self._migrate()
        except BaseException:
            self.conn.close()
            raise

    def _vector_missing(self, key, generation, external_version):
        return self._vector_view.missing(key, generation, external_version)

    @contextmanager
    def transaction(self):
        self._write_depth += 1
        try:
            with self._transaction():
                yield
        finally:
            self._write_depth -= 1
            if self._write_depth == 0:
                self._write_seq += 1

    @contextmanager
    def _transaction(self):
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
                self._snapshot_depth += 1
                try:
                    yield
                finally:
                    self._snapshot_depth -= 1

    def _migrate(self):
        from memengine.migration import migrate_legacy, migrate_lifecycle, migrate_observation_rows, migrate_identifiers

        with self.transaction():
            migrate_lifecycle(self)
            self.migrate_chronicle()
            if self.get_meta("schema_version") != "2":
                migrate_legacy(self)
                self.set_meta("schema_version", "2")
            migrate_observation_rows(self)
            migrate_identifiers(self, FTS_TOKENIZER)
            columns = {row[1] for row in self.conn.execute("PRAGMA table_info(facts)")}
            if "epistemic_scope" not in columns:
                self.conn.execute("ALTER TABLE facts ADD COLUMN epistemic_scope TEXT NOT NULL DEFAULT 'world' "
                                  "CHECK(epistemic_scope IN ('world','experience','opinion','observation','procedural'))")
                self.conn.execute("UPDATE facts SET epistemic_scope=memory_type WHERE memory_type!='fact'")
            for column in ("valid_from", "expires_at"):
                if column not in columns:
                    self.conn.execute(f"ALTER TABLE facts ADD COLUMN {column} REAL")
            if "cache_generation" not in {row[1] for row in self.conn.execute("PRAGMA table_info(fact_vectors)")}:
                self.conn.execute("ALTER TABLE fact_vectors ADD COLUMN cache_generation INTEGER NOT NULL DEFAULT 0")
            self.conn.execute("""CREATE TRIGGER IF NOT EXISTS fact_vector_cache_version AFTER UPDATE OF vec,model_name,dim ON fact_vectors
                BEGIN UPDATE fact_vectors SET cache_generation=cache_generation+1 WHERE fact_id=NEW.fact_id; END""")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_validity ON facts(user_id,agent_id,event_time_start,expires_at)")
            self.migrate_workspace()
            self.migrate_allergy_slots()

    def health_check(self):
        return {"ok": self.conn.execute("SELECT 1").fetchone()[0] == 1, "backend": "sqlite",
                "vector": "numpy", "journal_mode": self.conn.execute("PRAGMA journal_mode").fetchone()[0]}

    def search_lexical(self, *args, **kwargs):
        return self.search_bm25(*args, **kwargs)

    def search_exact_identifiers(self, *args, **kwargs):
        return self.search_identifiers(*args, **kwargs)

    def search_temporal_range(self, **kwargs):
        return self.search_temporal(**kwargs)

    def search_vector_topk(self, *args, **kwargs):
        return self.search_vector(*args, **kwargs)

    def fetch_raw_window(self, *args, **kwargs):
        return self.get_turn_window(*args, **kwargs)

    def close(self):
        self.conn.close()

    def commit(self):
        self.conn.commit()

    def get_meta(self, key):
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row is not None else None

    def recall_metadata(self, scope, control_key, *, legacy):
        history = _history_key(scope)
        maintenance = self.maintenance_key(scope)
        keys = [control_key, history, maintenance, *(("origin", "radius") if legacy else ())]
        rows = self.conn.execute(f"SELECT key,value FROM meta WHERE key IN ({_IDS})", (_dump(keys),))
        values = {row["key"]: row["value"] for row in rows}
        return values, int(values.get(history) or 0), json.loads(values.get(maintenance) or "{}")

    def set_meta(self, key, value):
        self.conn.execute("INSERT INTO meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def next_id(self, prefix):
        row = self.conn.execute("""INSERT INTO meta VALUES(?, '1') ON CONFLICT(key) DO UPDATE
                                   SET value=CAST(meta.value AS INTEGER)+1 RETURNING value""", (f"seq_{prefix}",)).fetchone()
        return f"{prefix}{row[0]}"

    def append_turn(self, text, timestamp=None, *, event_ts=None, ingest_ts=None, scope=DEFAULT_SCOPE,
                    role="user", user_id=None, agent_id=None, session_id=None, extraction_state="done", speaker=None, mention_time=None) -> int:
        scope = tuple(default if override is None else override for default, override in
                      zip(scope, (user_id, agent_id, session_id)))
        stamp = time.time() if timestamp is None else timestamp
        steps = step_ids_in(text)
        tokens = estimate_tokens(text)
        with self.transaction():
            cursor = self.conn.execute("""INSERT INTO raw_turns(user_id,agent_id,session_id,role,text,step_tag,event_ts,ingest_ts,token_count,extraction_state,speaker,mention_time)
                                          VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                                       (*scope, role, text, steps[0] if steps else None,
                                        stamp if event_ts is None else event_ts, stamp if ingest_ts is None else ingest_ts, tokens, extraction_state, speaker,
                                        mention_time))
            self.conn.execute("INSERT INTO raw_turns_fts(rowid,fts_tokens) VALUES (?,?)", (cursor.lastrowid, tokenize_for_fts(text)))
            self.conn.execute("""INSERT INTO meta VALUES(?,?) ON CONFLICT(key) DO UPDATE
                                 SET value=CAST(meta.value AS INTEGER)+CAST(excluded.value AS INTEGER)""",
                              (_history_key(scope), str(tokens)))
            return cursor.lastrowid

    def history_tokens(self, scope=DEFAULT_SCOPE):
        return int(self.get_meta(_history_key(scope)) or 0)

    def session_turns(self, scope, *, pending=False, limit=None, before=None, as_of=None, for_context=False):
        where, args = _scope_where(scope)
        if pending:
            where += " AND extraction_state!='done'"
        if before is not None:
            where += " AND turn_id<?"
            args += (before,)
        if for_context:
            rows = self.conn.execute(f"""WITH recent AS (
                SELECT * FROM raw_turns WHERE {where} AND extraction_state!='pending' AND ingest_ts<=?
                ORDER BY turn_id DESC LIMIT ?
            ) SELECT r.* FROM recent r WHERE NOT EXISTS (
                SELECT 1 FROM facts f WHERE f.user_id=r.user_id AND f.agent_id=r.agent_id AND f.session_id=r.session_id
                AND (f.status!='active' OR f.expires_at IS NOT NULL OR f.valid_from IS NOT NULL)
                AND (f.status!='active' OR f.expires_at<=? OR f.valid_from>?)
                AND (f.origin_turn=r.turn_id OR (f.origin_turn IS NULL AND EXISTS (
                    SELECT 1 FROM json_each(f.source_turn_ids) j WHERE r.turn_id=j.value
                )))
            ) ORDER BY r.turn_id""", (*args, as_of, limit, as_of, as_of))
            return [dict(row) for row in rows]
        if limit is None:
            return [dict(row) for row in self.conn.execute(f"SELECT * FROM raw_turns WHERE {where} ORDER BY turn_id", args)]
        rows = self.conn.execute(f"SELECT * FROM raw_turns WHERE {where} ORDER BY turn_id DESC LIMIT ?", (*args, limit))
        return [dict(row) for row in reversed(rows.fetchall())]

    def pending_turn_ids(self, turn_ids, scope):
        where, args = _scope_where(scope)
        return {row[0] for row in self.conn.execute(f"SELECT turn_id FROM raw_turns WHERE {where} AND extraction_state!='done' "
                                                    f"AND turn_id IN ({_IDS})", (*args, _dump(list(turn_ids))))}

    def finish_turns(self, turn_ids):
        self.conn.execute(f"UPDATE raw_turns SET extraction_state='done' WHERE turn_id IN ({_IDS})", (_dump(list(turn_ids)),))

    def commit_buffered_turns(self, turn_ids, *, scope):
        if turn_ids:
            where, args = _scope_where(scope)
            self.conn.execute(f"""UPDATE raw_turns SET extraction_state='done'
                WHERE {where} AND extraction_state='buffered' AND turn_id IN ({_IDS})""",
                (*args, _dump(list(turn_ids))))

    def get_turn_window(self, turn_ids: list[int], window: int = 1, user_id="default", agent_id="default",
                        *, by_id=False, session_id=None) -> list[dict]:
        if window < 0:
            raise ValueError("window must be nonnegative")
        if not turn_ids:
            return []
        if by_id:
            where, args = _scope_where((user_id, agent_id, session_id), "a.")
            rows = self.conn.execute(f"""SELECT DISTINCT n.* FROM raw_turns a
                JOIN raw_turns n USING(user_id,agent_id,session_id)
                WHERE {where} AND a.turn_id IN ({_IDS})
                  AND n.extraction_state!='buffered'
                  AND n.turn_id BETWEEN a.turn_id-? AND a.turn_id+? ORDER BY n.turn_id""",
                                     (*args, _dump(turn_ids), window, window))
            return [dict(row) for row in rows]
        session_clause = "" if session_id is None else "AND session_id=?"
        scope_params = (user_id, agent_id) if session_id is None else (user_id, agent_id, session_id)
        rows = self.conn.execute(f"""WITH anchors AS (
               SELECT * FROM raw_turns WHERE user_id=? AND agent_id=? {session_clause} AND turn_id IN ({_IDS})
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
             (*scope_params, _dump(turn_ids), max(0, window - 1), window, max(0, window - 1), window))
        return [dict(row) for row in rows]

    def inactive_temporary_sources(self, turn_ids, *, scope, as_of, include_superseded=False):
        if not turn_ids or as_of is None:
            return set()
        where, args = _scope_where(scope, "f.")
        stale_source = ("f.status!='active'" if include_superseded else
                        "f.status='superseded' AND (f.attribute_slot LIKE 'allergy:%' OR f.attribute_slot LIKE 'preference:%')")
        inactive = f"""((f.expires_at IS NOT NULL AND (f.expires_at<=? OR f.valid_from>?))
            OR ({stale_source} AND (f.origin_turn IS NULL OR f.origin_turn=CAST(j.value AS BIGINT))))"""
        rows = self.conn.execute(f"""SELECT DISTINCT j.value AS turn_id
            FROM facts f, json_each(f.source_turn_ids) j
            WHERE {where} AND {inactive}
              AND j.value IN ({_IDS})""", (*args, as_of, as_of, _dump(sorted(turn_ids))))
        return {int(row["turn_id"]) for row in rows}

    def temporary_source_deadline(self, turn_ids, *, scope, as_of):
        if not turn_ids:
            return None
        where, args = _scope_where(scope, "f.")
        return self.conn.execute(f"""SELECT min(f.expires_at) FROM facts f, json_each(f.source_turn_ids) j
            WHERE {where} AND f.status='active' AND f.expires_at>? AND f.valid_from<=?
              AND j.value IN ({_IDS}) AND (f.origin_turn IS NULL OR f.origin_turn=CAST(j.value AS BIGINT))""",
                                 (*args, as_of, as_of, _dump(turn_ids))).fetchone()[0]

    def create_block(self, topic, timestamp, *, scope=DEFAULT_SCOPE, event_time=None, is_conflict=False):
        block_id = self.next_id("b")
        self.conn.execute("INSERT INTO blocks(id,topic,created_at,user_id,agent_id,session_id) VALUES(?,?,?,?,?,?)",
                          (block_id, topic, timestamp, *scope))
        return block_id

    def get_block(self, block_id, *, scope=DEFAULT_SCOPE):
        where, args = _scope_where(scope)
        row = self.conn.execute(f"SELECT * FROM blocks WHERE id=? AND {where}", (block_id, *args)).fetchone()
        return dict(row) if row else None

    @contextmanager
    def reuse_write_vectors(self):
        prior = self._write_vectors
        self._write_vectors = {}
        try:
            yield
        finally:
            self._write_vectors = prior

    def _encoded(self, text, *, offline=False):
        backend = self.fallback_embedder if offline else self.embedder
        clean = strip_step_tags(text)
        vector = (self._write_vectors or {}).get((backend.model_name, clean))
        try:
            if vector is None:
                vector = normalize_vectors(backend.encode([clean]))[0]
        except (OSError, ValueError, TypeError, KeyError, HTTPException):
            backend = self.fallback_embedder
            vector = (self._write_vectors or {}).get((backend.model_name, clean))
            if vector is None:
                vector = normalize_vectors(backend.encode([clean]))[0]
        return backend.model_name, len(vector), vector.astype("<f4", copy=False).tobytes()

    def _insert_version(self, data, zkeys, *, offline=False, chronicle=None):
        data.setdefault("content_hash", content_hash(data["plain"]))
        data["is_constraint"] = int(bool(data.get("is_constraint") or is_constraint(data["plain"])))
        zkeys = self.populate_chronicle(data, zkeys, chronicle)
        indexed = data.get("normalized_text") or data["plain"]
        columns = tuple(data)
        cursor = self.conn.execute(f"INSERT INTO facts({','.join(columns)}) VALUES({','.join('?' for _ in columns)})",
                                   tuple(data.values()))
        fact_id, user_id, agent_id = (data[key] for key in ("id", "user_id", "agent_id"))
        entities = [(fact_id, user_id, agent_id, normalize_entity(key), key, key_type(key)) for key in dict.fromkeys(zkeys)]
        self.conn.executemany("INSERT OR IGNORE INTO fact_entities(fact_id,user_id,agent_id,entity_norm,raw_key,key_type) VALUES(?,?,?,?,?,?)",
                              (record for record in entities if record[3]))
        self.conn.execute("INSERT INTO facts_fts(rowid,fact_id,user_id,agent_id,fts_tokens) VALUES(?,?,?,?,?)",
                          (cursor.lastrowid, fact_id, user_id, agent_id, tokenize_for_fts(indexed)))
        self.index_identifiers(fact_id, data["plain"])
        self.index_versions(fact_id, data["plain"])
        model, dim, vector = self._encoded(indexed, offline=offline)
        self.conn.execute("INSERT INTO fact_vectors(fact_id,user_id,agent_id,dim,model_name,vec) VALUES(?,?,?,?,?,?)", (fact_id, user_id, agent_id, dim, model, vector))
        if data.get("memory_type", "fact") == "fact":
            self._index_turn_anchor(data, cursor.lastrowid, offline=offline)
        if chronicle and data.get("memory_type", "fact") == "fact":
            self.link_chronicle(data, zkeys, model, vector)
        return fact_id

    def _index_turn_anchor(self, data, fact_rowid, *, offline):
        source_ids = json.loads(data["source_turn_ids"])
        if not source_ids:
            return
        turn_id = max(source_ids)
        scope = (data["user_id"], data["agent_id"], data["session_id"])
        rows = self.conn.execute(
            "SELECT turn_id,role,text,ingest_ts FROM raw_turns WHERE turn_id IN (?,?) "
            "AND user_id=? AND agent_id=? AND session_id=? "
            "AND (turn_id=? OR extraction_state!='buffered') ORDER BY turn_id",
            (turn_id - 1, turn_id, *scope, turn_id)).fetchall()
        if len(rows) != 2:
            return
        previous, current = rows
        context = turn_context(current, previous)
        if not context:
            return
        indexed = f"{strip_step_tags(data['plain'])}\n{context}"
        model, dim, vector = self._encoded(indexed, offline=offline)
        self.conn.execute("INSERT INTO fact_anchors VALUES(?,?,?,?,?,?,?)",
                          (data["id"], previous["turn_id"], context, model, dim, vector, int(current["role"] == previous["role"])))
        self.conn.execute("INSERT INTO facts_anchor_fts(rowid,fts_tokens) VALUES(?,?)",
                          (fact_rowid, tokenize_for_fts(indexed)))

    def search_turn_anchors(self, query, semantic_query, *, user_id="default", agent_id="default", session_id=None,
                            top_n=35, include_cold=False, filters=None):
        scope = (user_id, agent_id, session_id)
        where, args = _scope_where(scope, "f.")
        if filters is not None:
            where, args = filters.sql(where, args)
        eligible = f"{where} AND {self._searchable(True, include_cold, filters)}"
        vectors = self.conn.execute(f"""SELECT a.* FROM facts f JOIN fact_anchors a ON a.fact_id=f.id
                                    WHERE {eligible}""", args).fetchall()
        if not vectors:
            return [], []
        terms = list(dict.fromkeys(word for word in tokenize_for_fts(query).split() if word not in _FTS_STOP))
        lexical = []
        if terms:
            expression = " OR ".join(f'"{term}"' for term in terms[:128])
            lexical = [dict(row) for row in self.conn.execute(f"""SELECT f.id,bm25(facts_anchor_fts) AS anchor_bm25_score, a.peer, a.turn_id AS context_anchor_turn_id
                       FROM facts_anchor_fts JOIN facts f ON f.rowid=facts_anchor_fts.rowid
                       JOIN fact_anchors a ON a.fact_id=f.id
                       WHERE facts_anchor_fts MATCH ? AND {eligible} ORDER BY anchor_bm25_score,f.id LIMIT ?""",
                       (expression, *args, top_n))]
        dense = self._rank_vectors(vectors, semantic_query, top_n)
        return lexical, dense

    def index_identifiers(self, fact_id, text):
        self.conn.executemany("INSERT OR IGNORE INTO exact_identifiers VALUES(?,?,?,?)",
                              [(fact_id, item.ident, item.ident_norm, item.ident_type) for item in extract_identifiers(text)])

    def search_identifiers(self, identifiers, *, user_id="default", agent_id="default", session_id=None,
                           include_cold=False, include_superseded=True, top_n=40, ids_only=False, filters=None):
        if not identifiers:
            return []
        scope = (user_id, agent_id, session_id)
        where, args = _scope_where(scope, "f.")
        if filters is not None:
            where, args = filters.sql(where, args)
        rows = self.conn.execute(f"""SELECT f.id,count(*) AS exact_ident_hit FROM exact_identifiers e CROSS JOIN facts f ON f.id=e.memory_id
            WHERE e.ident_norm IN ({_IDS}) AND {where} AND {self._searchable(include_superseded, include_cold, filters)}
            GROUP BY f.id ORDER BY exact_ident_hit DESC,f.event_time_start DESC,f.id LIMIT ?""",
            (_dump([item.ident_norm for item in identifiers]), *args, top_n))
        return self._search_results(rows, scope, ids_only)

    def identifiers_for(self, ids):
        matches = {}
        for row in self.conn.execute(f"SELECT memory_id,ident_norm,ident_type FROM exact_identifiers WHERE memory_id IN ({_IDS})",
                                     (_dump(list(ids)),)):
            matches.setdefault(row["memory_id"], {})[row["ident_norm"]] = row["ident_type"]
        return matches

    def session_orders(self, scope, sessions):
        rows = self.conn.execute(f"SELECT session_id,min(turn_id) FROM raw_turns WHERE user_id=? AND agent_id=? "
                                 f"AND session_id IN ({_IDS}) GROUP BY session_id", (*scope[:2], _dump(sorted(sessions))))
        return {row[0]: row[1] for row in rows}

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
                    scope=DEFAULT_SCOPE, role="user", is_conflict=False, memory_type="fact", confidence=1.0,
                    derived_from_ids=(), tool_call_ids=(), trigger_text="", success_count=0,
                    is_constraint=False, offline=False, epistemic_scope=None, valid_from=None, expires_at=None, chronicle=None) -> str:
        with self.transaction():
            if not self.get_block(block_id, scope=scope):
                raise ValueError("block not in fact scope")
            reference_scope = scope if memory_type == "fact" and not tool_call_ids else (*scope[:2], None)
            self._validate_turn_ids(source_turn_ids, reference_scope)
            self.validate_fact_ids(derived_from_ids, (*scope[:2], None))
            fact_id = self.next_id("l")
            seq = self.conn.execute("SELECT coalesce(max(seq),0)+1 FROM facts WHERE block_id=?", (block_id,)).fetchone()[0]
            data = self._version_data(fact_id=fact_id, block_id=block_id, seq=seq, scope=scope, formatted=formatted,
                                      plain=plain, attribute=attribute, value=value, surprise=surprise, layout=layout,
                                      isolated=isolated, is_conflict=is_conflict, role=role, ingest_time=ingest_time,
                                      event_time_start=event_time_start, event_time_end=event_time_end,
                                      source_turn_ids=source_turn_ids, source_step_ids=source_step_ids)
            epistemic_scope = epistemic_scope or (memory_type if memory_type != "fact" else "world")
            if epistemic_scope not in SCOPES or not 0 <= confidence <= 1:
                raise ValueError("invalid epistemic scope or confidence")
            if expires_at is not None:
                valid_from = ingest_time if valid_from is None else valid_from
                if not expires_at > valid_from:
                    raise ValueError("expires_at must follow valid_from")
            data.update(epistemic_scope=epistemic_scope, valid_from=valid_from, expires_at=expires_at,
                        memory_type=memory_type, confidence=confidence, derived_from_ids=_dump(list(derived_from_ids)),
                        tool_call_ids=_dump(list(tool_call_ids)), trigger_text=trigger_text, success_count=success_count,
                        is_constraint=int(is_constraint))
            return self._insert_version(data, zkeys, offline=offline, chronicle=chronicle)

    def _validate_turn_ids(self, turn_ids, scope):
        if turn_ids:
            if any(type(turn_id) is not int or turn_id < 1 for turn_id in turn_ids):
                raise ValueError("source_turn_ids must be positive integers")
            where, args = _scope_where(scope)
            count = self.conn.execute(f"SELECT count(*) FROM raw_turns WHERE {where} AND turn_id IN ({_IDS})",
                                      (*args, _dump(list(turn_ids)))).fetchone()[0]
            if count != len(set(turn_ids)):
                raise ValueError("source turns must exist in the fact scope")

    def validate_fact_ids(self, fact_ids, scope):
        if fact_ids:
            where, args = _scope_where(scope)
            count = self.conn.execute(f"SELECT count(*) FROM facts WHERE {where} AND id IN ({_IDS})",
                                      (*args, _dump(list(fact_ids)))).fetchone()[0]
            if count != len(set(fact_ids)):
                raise ValueError("derived facts must exist in the memory scope")

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
                       scope=DEFAULT_SCOPE, role="user", is_conflict=False, source_turn_ids=(), chronicle=None) -> str | None:
        with self.transaction():
            old = self._exact_fact(old_fact_id, scope)
            if old is None or old["status"] != "active":
                return None
            turns = merge_step_ids(source_turn_ids, [] if current_turn_id is None else [current_turn_id])
            self._validate_turn_ids(turns, scope if old["memory_type"] == "fact" else (*scope[:2], None))
            ingest_time = max(ingest_time, old["ingest_time"])
            self.conn.execute("UPDATE facts SET status='superseded',valid_until=? WHERE id=?", (ingest_time, old["id"]))
            owner = (old["user_id"], old["agent_id"], old["session_id"])
            self.invalidate_observations(owner, [old["id"]], ingest_time)
            replace_sources = old["memory_type"] == "observation" and chronicle is not None and "derived_from_ids" in chronicle
            data = self._version_data(fact_id=self.next_id("l"), block_id=old["block_id"], seq=old["seq"], scope=owner,
                                      formatted=formatted, plain=plain, attribute=attribute, value=value, surprise=surprise,
                                      layout=layout, isolated=old["isolated"] if isolated is None else isolated,
                                      is_conflict=is_conflict, role=role, ingest_time=ingest_time, event_time_start=event_time_start,
                                      event_time_end=event_time_end, root_fact_id=old["root_fact_id"], supersedes_id=old["id"],
                                      revision=old["revision"] + 1, hit_count=old["hit_count"],
                                      source_turn_ids=turns if replace_sources else merge_step_ids(json.loads(old["source_turn_ids"]), turns),
                                      source_step_ids=current_step_ids if replace_sources else
                                      merge_step_ids(json.loads(old["source_step_ids"]), current_step_ids))
            for name in ("memory_type", "confidence", "derived_from_ids", "tool_call_ids", "trigger_text", "success_count", "is_constraint", "epistemic_scope", "valid_from", "expires_at"):
                data[name] = old[name]
            return self._insert_version(data, zkeys, chronicle=chronicle)

    def delete_fact(self, fact_id, *, hard=False, ingest_time=None, scope=DEFAULT_SCOPE):
        with self.transaction():
            row = self._resolve_fact(fact_id, scope=scope, status="TRUE" if hard else "status='active'")
            if row is None:
                return False
            stamp = max(time.time() if ingest_time is None else ingest_time, row["ingest_time"])
            if hard:
                removed = self.conn.execute("DELETE FROM facts WHERE root_fact_id=? RETURNING id", (row["root_fact_id"],)).fetchall()
                removed_ids = [item["id"] for item in removed]
            else:
                removed_ids = [item["id"] for item in self.conn.execute(
                    "SELECT id FROM facts WHERE root_fact_id=?", (row["root_fact_id"],))]
                self.conn.execute("UPDATE facts SET status='deleted',valid_until=? WHERE id=?", (stamp, row["id"]))
            self.invalidate_observations(scope, removed_ids, stamp)
            self._forget_vectors(scope, removed_ids if hard else [row["id"]])
            return True

    def invalidate_observations(self, scope, source_ids, stamp, *, compiled_line=""):
        rows = self.conn.execute(f"""UPDATE facts AS o SET is_stale=1
            WHERE o.user_id=? AND o.agent_id=? AND o.memory_type='observation'
              AND o.status='active' AND o.is_stale=0 AND EXISTS (
                  SELECT 1 FROM json_each(o.derived_from_ids) j WHERE j.value IN ({_IDS})
              ) RETURNING plain""", (*scope[:2], _dump(source_ids))).fetchall()
        if not rows and not compiled_line:
            return
        invalid = {" ".join(strip_step_tags(text).split()) for text in (compiled_line, *(row["plain"] for row in rows))}
        user, agent = scope[:2]
        snapshot = self.get_core_snapshot(user_id=user, agent_id=agent)
        lines = snapshot["core_profile"].splitlines()
        retained = [line for line in lines if line not in invalid]
        if retained != lines:
            self.set_core_snapshot("\n".join(retained), snapshot["procedural_rules"],
                                   user_id=user, agent_id=agent, timestamp=stamp)

    def _forget_vectors(self, scope, ids):
        if hasattr(self, "vector_cache"):
            self.vector_cache.discard(scope[:2], ids)

    def increment_hit_counts(self, fact_ids, accessed_at=None):
        if fact_ids:
            self.conn.execute(f"UPDATE facts SET hit_count=hit_count+1, "
                              f"last_accessed_at=coalesce(max(coalesce(last_accessed_at,ingest_time),?),last_accessed_at) "
                              f"WHERE status!='deleted' AND id IN ({_IDS})", (accessed_at, _dump(list(fact_ids))))

    def merge_line_mentions(self, fact_id, step_ids, *, scope=DEFAULT_SCOPE, turn_id=None, turn_ids=(), accessed_at=None,
                            exact_version=False):
        with self.transaction():
            row = (self._exact_fact(fact_id, scope) if exact_version else
                   self._resolve_fact(fact_id, scope=scope, status="status='active'"))
            if row is not None and row["status"] == "active" and (not exact_version or row["tier"] == "hot"):
                turns = merge_step_ids(turn_ids, [turn_id] if turn_id is not None else [])
                self._validate_turn_ids(turns, scope)
                previous = json.loads(row["source_turn_ids"])
                merged = merge_step_ids(previous, turns)
                self.conn.execute("""UPDATE facts SET source_turn_ids=?,source_step_ids=?,mention_count=mention_count+?,
                                  last_accessed_at=coalesce(max(coalesce(last_accessed_at,ingest_time),?),last_accessed_at) WHERE id=?""",
                                  (_dump(merged), _dump(merge_step_ids(json.loads(row["source_step_ids"]), step_ids)),
                                   len(merged)-len(previous), accessed_at, row["id"]))
                return True
            return False

    def content_matches(self, digest, scope, *, memory_type="fact", include_cold=False):
        where, args = _scope_where(scope, "f.")
        where += " AND f.status='active' AND f.content_hash=?"
        args += (digest,)
        if not include_cold:
            where += " AND f.tier='hot'"
        if memory_type is not None:
            where += " AND f.memory_type=?"
            args += (memory_type,)
        return self._load_facts(where, args)

    @staticmethod
    def _fact(row, keys):
        fact = dict(row)
        steps = json.loads(fact["source_step_ids"])
        turns = json.loads(fact["source_turn_ids"])
        user_fact = fact["memory_type"] == "fact" and fact["role"] == "user" and fact["epistemic_scope"] == "world"
        return {**fact, "text": ensure_step_tags(fact["plain"], steps), "zkeys": keys,
                "entity_z_keys": [key for key in keys if key_type(key) == "entity"],
                "category_z_keys": [key for key in keys if key_type(key) == "tag"],
                "is_taboo": user_fact and is_taboo(fact["plain"]),
                "explicit_preference": explicit_preference(fact["plain"], fact["speaker"]) if user_fact and turns else None,
                "isolated": bool(fact["isolated"]), "historical": fact["status"] == "superseded",
                "active": int(fact["status"] == "active"), "event_time": fact["event_time_start"],
                "updated_at": fact["ingest_time"], "is_conflict_or_override": bool(fact["is_conflict"]),
                "source_step_ids": steps, "source_turn_ids": turns,
                "derived_from_ids": json.loads(fact["derived_from_ids"]), "tool_call_ids": json.loads(fact["tool_call_ids"])}

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

    def recall_facts_by_ids(self, fact_ids, *, scope, identifiers=False):
        if not fact_ids:
            return []
        where, args = _scope_where(scope, "f.")
        aggregate, pair = self._json_aggregate, self._json_pair
        exact = (f", (SELECT {aggregate}({pair}(i.ident_norm,i.ident_type)) FROM exact_identifiers i "
                 "WHERE i.memory_id=f.id) AS recall_identifiers") if identifiers else ""
        rows = self.conn.execute(f"""SELECT f.*,b.topic,
                (SELECT {aggregate}(e.raw_key) FROM fact_entities e WHERE e.fact_id=f.id) AS recall_keys{exact}
            FROM facts f JOIN blocks b ON b.id=f.block_id
            WHERE {where} AND f.id IN ({_IDS}) ORDER BY b.created_at,b.id,f.seq,f.revision""",
            (*args, _dump(list(fact_ids))))
        facts = []
        for row in rows:
            data = dict(row)
            keys = data.pop("recall_keys")
            exact = data.pop("recall_identifiers", None)
            fact = self._fact(data, sorted((json.loads(keys) if isinstance(keys, str) else keys) or ()))
            if identifiers:
                fact["identifiers"] = dict((json.loads(exact) if isinstance(exact, str) else exact) or ())
            facts.append(fact)
        return facts

    def evidence_fact_states(self, fact_ids, *, scope):
        if not fact_ids:
            return {}
        where, args = _scope_where(scope)
        rows = self.conn.execute(f"""SELECT id,status,is_stale,valid_from,expires_at FROM facts
            WHERE {where} AND id IN ({_IDS})""", (*args, _dump(list(fact_ids))))
        return {row["id"]: dict(row) for row in rows}

    def snapshot(self, *, scope=DEFAULT_SCOPE, include_superseded=False, include_cold=True, include_derived=False):
        where, args = _scope_where(scope, "f.")
        kinds = "" if include_derived else " AND f.memory_type='fact'"
        tier = "" if include_cold else " AND f.tier='hot'"
        return self._load_facts(f"{where} AND {_status_where(include_superseded)}{kinds}{tier}", args)

    def _searchable(self, include_superseded, include_cold=False, filters=None):
        tier = "" if include_cold else " AND f.tier='hot'"
        if filters is None or filters.memory_types is None and filters.scopes is None:
            types = " AND f.memory_type!='observation'"
        else:
            types = f" AND (f.memory_type!='observation' OR (f.is_stale=0 AND {_live_observation_sources('f')}))"
        return _status_where(include_superseded) + tier + types + (" AND NOT EXISTS (SELECT 1 FROM facts removed "
                 "WHERE removed.root_fact_id=f.root_fact_id AND removed.status='deleted')" if include_superseded else "")

    def get_fact_chain(self, root_fact_id, *, user_id="default", agent_id="default", session_id=None):
        where, args = _scope_where((user_id, agent_id, session_id), "f.")
        return self._load_facts(f"{where} AND f.root_fact_id=?", (*args, root_fact_id), "f.revision")

    def memory_history(self, key, *, scope=DEFAULT_SCOPE):
        where, args = _scope_where(scope, "f.")
        with self.read_snapshot():
            match = self._exact_fact(key, scope)
            if match is not None:
                return self._load_facts(f"{where} AND f.root_fact_id=?", (*args, match["root_fact_id"]), "f.revision")
            roots = [row[0] for row in self.conn.execute(f"""SELECT DISTINCT f.root_fact_id FROM fact_entities e
                JOIN facts f ON f.id=e.fact_id WHERE e.user_id=? AND e.agent_id=? AND e.key_type='entity'
                AND e.entity_norm=? AND {where}""", (*scope[:2], normalize_entity(key), *args))]
            if not roots:
                return []
            return self._load_facts(f"{where} AND f.root_fact_id IN ({_IDS})", (*args, _dump(roots)), "f.root_fact_id,f.revision")

    def blocks_for_prompt(self, *, scope=DEFAULT_SCOPE, snapshot=None):
        blocks = {}
        for fact in self.snapshot(scope=scope) if snapshot is None else snapshot:
            block = blocks.setdefault(fact["block_id"], {"id": fact["block_id"], "topic": fact["topic"], "lines": []})
            block["lines"].append(fact)
        return list(blocks.values())

    def candidate_blocks(self, *, scope=DEFAULT_SCOPE, topics=(), explicit=(), max_recent_blocks=24, lines_per_block=5):
        """Bounded write-path block fetch avoiding O(N) full-session snapshots."""
        where, args = _scope_where(scope, "f.")
        base_where = f"{where} AND f.status='active' AND f.memory_type='fact' AND f.expires_at IS NULL AND f.epistemic_scope='world'"
        block_where, block_args = _scope_where(scope, "b.")
        block_ids = [row[0] for row in self.conn.execute(
            f"SELECT b.id FROM blocks b WHERE {block_where} ORDER BY b.created_at DESC, b.id DESC LIMIT ?",
            (*block_args, max_recent_blocks))]
        clean_topics = sorted({topic for topic in topics if topic})
        if clean_topics:
            block_ids.extend(row[0] for row in self.conn.execute(
                f"SELECT b.id FROM blocks b WHERE {block_where} AND b.topic IN ({_IDS}) ORDER BY b.created_at DESC LIMIT 16",
                (*block_args, _dump(clean_topics))))
        pinned_ids = set()
        clean_entities = sorted({entity for entity in explicit if entity})
        if clean_entities:
            for row in self.conn.execute(
                f"""SELECT DISTINCT f.id, f.block_id FROM fact_entities e JOIN facts f ON f.id=e.fact_id
                    WHERE e.user_id=? AND e.agent_id=? AND e.key_type='entity' AND e.entity_norm IN ({_IDS})
                      AND {base_where}""",
                (*scope[:2], _dump(clean_entities), *args)):
                pinned_ids.add(row["id"])
                block_ids.append(row["block_id"])
        unique_blocks = list(dict.fromkeys(block_ids))
        if not unique_blocks and not pinned_ids:
            return []
        fact_ids = set(pinned_ids)
        if unique_blocks:
            for row in self.conn.execute(
                f"""WITH ranked AS (
                      SELECT f.id, row_number() OVER (PARTITION BY f.block_id ORDER BY f.seq DESC, f.revision DESC) AS rn
                      FROM facts f WHERE {base_where} AND f.block_id IN ({_IDS})
                    ) SELECT id FROM ranked WHERE rn<=?""",
                (*args, _dump(unique_blocks), lines_per_block)):
                fact_ids.add(row["id"])
        if not fact_ids:
            return []
        facts = self._load_facts(f"{where} AND f.id IN ({_IDS})", (*args, _dump(sorted(fact_ids))))
        return self.blocks_for_prompt(scope=scope, snapshot=facts)

    def z_index(self, *, scope=DEFAULT_SCOPE, snapshot=None):
        index = {}
        for fact in self.snapshot(scope=scope) if snapshot is None else snapshot:
            for key in fact["zkeys"]:
                index.setdefault(key, []).append(fact)
        return index

    def _entity_keys_select(self, scope, include_cold):
        where, args = _scope_where(scope, "f.")
        tier = "" if include_cold else "AND e.tier='hot'"
        return (f"""SELECT DISTINCT e.raw_key FROM fact_entities e JOIN facts f ON f.id=e.fact_id
                 WHERE e.user_id=? AND e.agent_id=? AND e.key_type='entity' {tier}
                 AND {where} AND {self._searchable(True, include_cold)}""", (*scope[:2], *args))

    def _can_use_local_cache(self):
        return (self.enable_vector_cache and hasattr(self, "entity_cache") and not self._write_depth
                and (not self.conn.in_transaction or self._snapshot_depth > 0))

    def _cached_entity_candidates(self, query, scope, include_cold, external_version=None):
        if external_version is None:
            external_version = self.conn.execute("PRAGMA data_version").fetchone()[0]
        stamp = (external_version, self._write_seq)
        cache_key = (*scope, bool(include_cold))
        payload = self.entity_cache.get(cache_key, stamp)
        if payload is None:
            select, args = self._entity_keys_select(scope, include_cold)
            raw_keys = [row[0] for row in self.conn.execute(f"{select} ORDER BY e.raw_key", args)]
            payload = self.entity_cache.put(cache_key, stamp, raw_keys)
        return self.entity_cache.candidates(payload, query)

    def entity_keys(self, *, scope=DEFAULT_SCOPE, include_cold=False, query=None):
        if query is not None and self._can_use_local_cache():
            return self._cached_entity_candidates(query, scope, include_cold)
        select, args = self._entity_keys_select(scope, include_cold)
        rows = self.conn.execute(f"{select} ORDER BY e.raw_key", args)
        return [row[0] for row in rows]

    def recall_entity_keys(self, query, *, scope, include_cold=False):
        if self._can_use_local_cache():
            cache_key = (*scope, bool(include_cold))
            cached = self.entity_cache._entries.get(cache_key)
            if cached is not None and cached[0][1] == self._write_seq:
                rows = self.conn.execute(f"""WITH matched AS (
                      SELECT min(entity_id) AS id FROM entity_aliases WHERE user_id=? AND agent_id=? AND alias IN ({_IDS})
                      GROUP BY alias HAVING count(*)=1
                    ), dv AS (SELECT data_version FROM pragma_data_version)
                    SELECT 'entity:' || a.alias AS key, dv.data_version AS ver
                    FROM entity_aliases a JOIN matched m ON a.entity_id=m.id CROSS JOIN dv
                    UNION ALL SELECT NULL, data_version FROM dv
                    ORDER BY key""", self._alias_arguments(query, scope)).fetchall()
                external_version = rows[0]["ver"] if rows else 0
                if cached[0][0] == external_version:
                    aliases = sorted({row["key"] for row in rows if row["key"] is not None})
                    return self.entity_cache.candidates(cached[1], query), aliases
            select, args = self._entity_keys_select(scope, include_cold)
            hits = self.conn.execute(f"""WITH matched AS (
                  SELECT min(entity_id) AS id FROM entity_aliases WHERE user_id=? AND agent_id=? AND alias IN ({_IDS})
                  GROUP BY alias HAVING count(*)=1
                ), dv AS (SELECT data_version FROM pragma_data_version)
                SELECT raw_key AS key, 0 AS is_alias, dv.data_version AS ver FROM ({select}) direct_keys CROSS JOIN dv
                UNION ALL SELECT DISTINCT 'entity:' || a.alias, 1, dv.data_version FROM entity_aliases a JOIN matched m ON a.entity_id=m.id CROSS JOIN dv
                UNION ALL SELECT NULL, 2, data_version FROM dv
                ORDER BY is_alias, key""", (*self._alias_arguments(query, scope), *args)).fetchall()
            external_version = hits[0]["ver"] if hits else 0
            direct = [row["key"] for row in hits if row["is_alias"] == 0 and row["key"] is not None]
            aliases = [row["key"] for row in hits if row["is_alias"] == 1 and row["key"] is not None]
            payload = self.entity_cache.put(cache_key, (external_version, self._write_seq), direct)
            return self.entity_cache.candidates(payload, query), aliases
        select, args = self._entity_keys_select(scope, include_cold)
        hits = self.conn.execute(f"""WITH matched AS (
              SELECT min(entity_id) AS id FROM entity_aliases WHERE user_id=? AND agent_id=? AND alias IN ({_IDS})
              GROUP BY alias HAVING count(*)=1
            ) SELECT raw_key AS key,0 AS is_alias FROM ({select}) direct_keys
            UNION ALL SELECT DISTINCT 'entity:' || a.alias,1 FROM entity_aliases a JOIN matched m ON a.entity_id=m.id
            ORDER BY is_alias,key""", (*self._alias_arguments(query, scope), *args))
        direct, aliases = [], []
        for row in hits:
            (aliases if row["is_alias"] else direct).append(row["key"])
        return direct, aliases

    def search_entity_inverted(self, entities, *, user_id="default", agent_id="default", session_id=None,
                               include_superseded=True, top_n=40, ids_only=False, include_cold=False, filters=None):
        normalized = sorted({normalize_entity(key) for key in entities if key_type(key) == "entity"} - {""})
        if not normalized or top_n <= 0:
            return []
        scope = (user_id, agent_id, session_id)
        where, args = _scope_where(scope, "f.")
        if filters is not None:
            where, args = filters.sql(where, args)
        tier = "" if include_cold else "AND e.tier='hot'"
        rows = self.conn.execute(f"""SELECT f.id,count(*) AS entity_matches FROM fact_entities e CROSS JOIN facts f ON f.id=e.fact_id
             WHERE e.user_id=? AND e.agent_id=? AND e.key_type='entity' {tier} AND e.entity_norm IN ({_IDS})
               AND {where} AND {self._searchable(include_superseded, include_cold, filters)} GROUP BY f.id
             ORDER BY entity_matches DESC, f.ingest_time DESC,f.id LIMIT ?""",
                                 (user_id, agent_id, _dump(normalized), *args, top_n))
        return self._search_results(rows, scope, ids_only)

    def search_bm25(self, query, *, user_id="default", agent_id="default", session_id=None,
                    include_superseded=True, top_n=40, ids_only=False, include_cold=False, filters=None):
        terms = list(dict.fromkeys(tokenize_for_fts(query).split()))
        terms = [term for term in terms if term not in _FTS_STOP] or terms
        if not terms or top_n <= 0:
            return []
        scope = (user_id, agent_id, session_id)
        where, args = _scope_where(scope, "f.")
        if filters is not None:
            where, args = filters.sql(where, args)
        expression = " OR ".join(f'"{term}"' for term in terms[:128])
        hits = []
        for table in ("facts_fts", "facts_cold_fts") if include_cold else ("facts_fts",):
            hits.extend(self.conn.execute(f"""SELECT f.id,bm25({table}) AS bm25_score FROM {table} JOIN facts f ON f.rowid={table}.rowid
                 WHERE {table} MATCH ? AND {where} AND {self._searchable(include_superseded, include_cold, filters)} ORDER BY bm25_score,f.id LIMIT ?""",
                                          (expression, *args, top_n)))
        return self._search_results(sorted(hits, key=lambda hit: (hit["bm25_score"], hit["id"]))[:top_n], scope, ids_only)

    def search_raw_turns_fts(self, query, *, user_id="default", agent_id="default", session_id=None, top_n=40):
        """Zero-vector L0 episode fallback: BM25 search directly over ``raw_turns_fts``."""
        terms = list(dict.fromkeys(tokenize_for_fts(query).split()))
        terms = [term for term in terms if term not in _FTS_STOP] or terms
        if not terms or top_n <= 0:
            return []
        scope = (user_id, agent_id, session_id)
        where, args = _scope_where(scope, "r.")
        expression = " OR ".join(f'"{term}"' for term in terms[:128])
        rows = self.conn.execute(
            f"""SELECT r.turn_id, r.session_id, r.role, r.text, r.event_ts, r.ingest_ts,
                       r.speaker, r.mention_time, bm25(raw_turns_fts) AS bm25_score
                FROM raw_turns_fts
                JOIN raw_turns r ON r.turn_id = raw_turns_fts.rowid
                WHERE raw_turns_fts MATCH ? AND {where}
                ORDER BY bm25_score, r.turn_id LIMIT ?""",
            (expression, *args, top_n),
        ).fetchall()
        return [dict(row) for row in rows]

    def facts_for_turn_ids(self, turn_ids, *, user_id="default", agent_id="default", session_id=None,
                           include_superseded=False, include_cold=False):
        """Map L0 ``turn_id`` values back to derived fact IDs in the unified ledger."""
        ids = sorted({int(tid) for tid in turn_ids if tid is not None})
        if not ids:
            return {}
        scope = (user_id, agent_id, session_id)
        where, args = _scope_where(scope, "f.")
        status_sql = "f.status IN ('active','superseded')" if include_superseded else "f.status='active'"
        tier_sql = "" if include_cold else " AND f.tier='hot'"
        rows = self.conn.execute(
            f"""SELECT CAST(je.value AS INTEGER) AS tid, f.id
                FROM facts f, json_each(f.source_turn_ids) je
                WHERE {where} AND {status_sql}{tier_sql}
                  AND CAST(je.value AS INTEGER) IN ({_IDS})
                ORDER BY f.revision DESC, f.ingest_time DESC, f.id""",
            (*args, _dump(ids)),
        ).fetchall()
        mapping: dict[int, list[str]] = {}
        for row in rows:
            mapping.setdefault(int(row["tid"]), []).append(row["id"])
        return mapping

    def search_temporal(self, user_id="default", agent_id="default", start_ts=None, end_ts=None,
                        include_superseded=True, top_n=40, *, session_id=None, order_hint=None,
                        rev_index=None, entities=(), previous_only=False, ids_only=False, include_cold=False, filters=None):
        scope = (user_id, agent_id, session_id)
        where, args = _scope_where(scope, "f.")
        if filters is not None:
            where, args = filters.sql(where, args)
        conditions = [where, self._searchable(include_superseded, include_cold, filters)]
        params = list(args)
        if start_ts is not None:
            conditions.append("(f.event_time_start>=? OR f.expires_at>?)")
            params.extend((start_ts, start_ts))
        if end_ts is not None:
            conditions.append("(f.event_time_start<? OR (f.expires_at IS NOT NULL AND f.valid_from<?))")
            params.extend((end_ts, end_ts))
        if rev_index is not None:
            conditions.append("f.revision=?")
            params.append(rev_index)
        if previous_only:
            conditions.append("(f.expires_at IS NOT NULL OR (f.status='superseded' AND NOT EXISTS (SELECT 1 FROM facts newer "
                              "WHERE newer.root_fact_id=f.root_fact_id AND newer.status='superseded' AND newer.revision>f.revision)))")
        if entities:
            conditions.append(f"EXISTS(SELECT 1 FROM fact_entities e WHERE e.fact_id=f.id AND e.key_type='entity' AND e.entity_norm IN ({_IDS}))")
            params.append(_dump(list(entities)))
        order = "ASC" if order_hint == "earliest" else "DESC"
        index = "idx_facts_event_time" if include_cold else "idx_facts_event_time_hot"
        rows = self.conn.execute(f"SELECT f.id FROM facts f INDEXED BY {index} WHERE {' AND '.join(conditions)} "
                                 f"ORDER BY f.event_time_start {order},f.revision {order},f.id LIMIT ?", (*params, max(0, top_n)))
        return self._search_results(rows, scope, ids_only)

    def search_vector(self, query, *, user_id="default", agent_id="default", session_id=None,
                      include_superseded=True, top_n=40, ids_only=False, include_cold=False, memory_type=None, filters=None):
        if not query.strip() or top_n <= 0:
            return []
        scope = (user_id, agent_id, session_id)
        where, args = _scope_where(scope, "f.")
        if filters is not None:
            where, args = filters.sql(where, args)
        if memory_type is not None:
            where += " AND f.memory_type=?"
            args += (memory_type,)
        index = "" if include_cold else "INDEXED BY idx_facts_scope_tier"
        source = f"""FROM facts f {index} CROSS JOIN fact_vectors v ON f.id=v.fact_id
             WHERE {where} AND {self._searchable(include_superseded, include_cold, filters)} ORDER BY v.fact_id"""
        if not self.enable_vector_cache or self._write_depth or self.conn.in_transaction and not self._snapshot_depth:
            rows = self.conn.execute(f"SELECT v.* {source}", args).fetchall()
            return self._search_results(self._rank_vectors(rows, query, top_n), scope, ids_only)
        view = self.vector_cache.view(scope[:2])
        self._vector_view = view
        try:
            rows = self.conn.execute(f"""SELECT v.fact_id,v.model_name,v.dim,v.cache_generation,
                 CASE WHEN ame_vector_missing(v.fact_id,v.cache_generation,(SELECT data_version FROM pragma_data_version))
                 THEN v.vec END AS vec {source}""", args).fetchall()
        finally:
            self._vector_view = None
        hits = self._rank_matrix(view.groups_for(rows), query, top_n)
        return self._search_results(hits, scope, ids_only)

    def _rank_vectors(self, rows, query, top_n):
        groups = {}
        for row in rows:
            groups.setdefault((row["model_name"], row["dim"]), []).append(row)
        matrices = [(key, [r["fact_id"] for r in values],
                     np.frombuffer(b"".join(r["vec"] for r in values), dtype="<f4").reshape(-1, key[1]))
                    for key, values in groups.items()]
        return self._rank_matrix(matrices, query, top_n)

    def _rank_matrix(self, matrices, query, top_n):
        hits = []
        for backend in {self.fallback_embedder.model_name: self.fallback_embedder, self.embedder.model_name: self.embedder}.values():
            selected = [(size, ids, matrix) for (model, size), ids, matrix in matrices if model == backend.model_name]
            if not selected:
                continue
            try:
                encode = backend.encode if self._write_vectors is not None else getattr(backend, "encode_queries", backend.encode)
                encoded = normalize_vectors(encode([query]))[0]
            except (OSError, ValueError, TypeError, KeyError, HTTPException):
                continue
            if self._write_vectors is not None:
                self._write_vectors[backend.model_name, query] = encoded
            for dim, ids, matrix in selected:
                if dim != len(encoded):
                    continue
                hits.extend({"id": fact_id, "vector_score": float(score)}
                            for fact_id, score in zip(ids, matrix @ encoded) if score > 0)
        hits.sort(key=lambda fact: (-fact["vector_score"], fact["id"]))
        return hits[:top_n]

    def _search_results(self, rows, scope, ids_only):
        hits = [dict(row) for row in rows]
        if ids_only:
            return hits
        facts = {fact["id"]: fact for fact in self.facts_by_ids([row["id"] for row in hits], scope=scope)}
        return [{**facts[row["id"]], **row} for row in hits if row["id"] in facts]

    def insert_observation(self, content, *, user_id="default", agent_id="default", confidence=0.7,
                           derived_from_fact_ids=(), source_turn_ids=(), timestamp=None, offline=False):
        stamp = time.time() if timestamp is None else timestamp
        scope = (user_id, agent_id, "default")
        with self.transaction():
            block = self.create_block("Observations", stamp, scope=scope)
            return self.insert_fact(block_id=block, formatted=content, plain=content, ingest_time=stamp, scope=scope,
                                    memory_type="observation", confidence=confidence, derived_from_ids=derived_from_fact_ids,
                                    source_turn_ids=source_turn_ids, offline=offline)

    def list_observations(self, *, user_id="default", agent_id="default", session_id=None,
                          include_cold=False, include_stale=False):
        live = _live_observation_sources("o")
        selected = f"o.*, NOT ({live}) AS invalid_sources" if include_stale else "o.*"
        sql = f"SELECT {selected} FROM facts o WHERE o.user_id=? AND o.agent_id=? AND o.status='active' AND o.memory_type='observation'"
        args = (user_id, agent_id)
        if not include_cold:
            sql += " AND o.tier='hot'"
        if not include_stale:
            sql += f" AND o.is_stale=0 AND {live}"
        if session_id is not None:
            sql += """ AND NOT EXISTS (
                       SELECT 1 FROM json_each(o.derived_from_ids) j
                       LEFT JOIN facts f ON f.id=j.value AND f.user_id=o.user_id AND f.agent_id=o.agent_id AND f.session_id=?
                       WHERE f.id IS NULL)
                       AND NOT EXISTS (
                       SELECT 1 FROM json_each(o.source_turn_ids) j
                       LEFT JOIN raw_turns r ON r.turn_id=j.value AND r.user_id=o.user_id AND r.agent_id=o.agent_id AND r.session_id=?
                       WHERE r.turn_id IS NULL)"""
            args += (session_id, session_id)
        rows = self.conn.execute(sql + " ORDER BY ingest_time,id", args)
        observations = []
        for row in rows:
            fact = self._fact(row, [])
            if fact.pop("invalid_sources", False):
                fact["is_stale"] = 1
            observations.append({**fact, "content": fact["plain"], "created_at": fact["ingest_time"],
                                 "derived_from_fact_ids": fact["derived_from_ids"]})
        return observations

    def get_core_snapshot(self, *, user_id="default", agent_id="default"):
        row = self.conn.execute("SELECT * FROM core_profiles WHERE user_id=? AND agent_id=?", (user_id, agent_id)).fetchone()
        if row is None:
            return {"core_profile": "", "procedural_rules": []}
        return {"core_profile": row["profile_text"], "procedural_rules": json.loads(row["procedural_rules"]),
                "prefix_version": row["prefix_version"], "updated_at": row["updated_at"]}

    def set_core_snapshot(self, core_profile: str, procedural_rules: list[str], *, user_id="default", agent_id="default",
                          timestamp=None):
        from memengine.assembler import render_prefix
        from hashlib import sha256

        if not isinstance(core_profile, str) or not isinstance(procedural_rules, list) or not all(
                isinstance(rule, str) for rule in procedural_rules):
            raise ValueError("core_profile must be a string and procedural_rules a list of strings")
        prefix = render_prefix({"core_profile": core_profile, "procedural_rules": procedural_rules})
        version = sha256(prefix.encode()).hexdigest()[:12]
        with self.transaction():
            self.conn.execute("""INSERT INTO core_profiles VALUES(?,?,?,?,?,?) ON CONFLICT(user_id,agent_id) DO UPDATE SET
                profile_text=excluded.profile_text, procedural_rules=excluded.procedural_rules,
                prefix_version=excluded.prefix_version,updated_at=excluded.updated_at
                WHERE core_profiles.profile_text!=excluded.profile_text OR core_profiles.procedural_rules!=excluded.procedural_rules""",
                              (user_id, agent_id, core_profile, _dump(procedural_rules), version,
                               time.time() if timestamp is None else timestamp))
        return version

    def record_tool_call(self, call_id, trigger_text, technique, *, scope, success, source_turn_ids, derived_from_ids, timestamp,
                         task_id=None):
        with self.transaction():
            self._validate_turn_ids(source_turn_ids, scope)
            self.validate_fact_ids(derived_from_ids, (*scope[:2], None))
            self.conn.execute("""INSERT INTO tool_calls(user_id,agent_id,session_id,tool_call_id,trigger_text,technique,success,
                              source_turn_ids,derived_from_ids,created_at,task_id) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                              ON CONFLICT(user_id,agent_id,tool_call_id) DO NOTHING""",
                              (*scope, call_id, trigger_text, technique, int(success), _dump(list(source_turn_ids)),
                               _dump(list(derived_from_ids)), timestamp, task_id))

    def tool_calls(self, scope, *, call_id=None, after=0):
        where, args = _scope_where(scope)
        if call_id is not None:
            where += " AND tool_call_id=?"
            args += (call_id,)
        rows = self.conn.execute(f"SELECT * FROM tool_calls WHERE {where} AND seq>? ORDER BY seq", (*args, after))
        return [{**dict(row), "source_turn_ids": json.loads(row["source_turn_ids"]),
                 "derived_from_ids": json.loads(row["derived_from_ids"])} for row in rows]

    def update_opinion_confidence(self, fact_id, scope, confidence):
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        where, args = _scope_where(scope)
        self.conn.execute(f"UPDATE facts SET confidence=0.75*confidence+0.25*? "
                          f"WHERE id=? AND {where} AND epistemic_scope='opinion' AND status='active'",
                          (confidence, fact_id, *args))

    def merge_derived_sources(self, fact, refs, turns, calls=(), *, successful=()):
        turn_ids = merge_step_ids(fact["source_turn_ids"], turns)
        old_calls = fact["tool_call_ids"]
        self.conn.execute("""UPDATE facts SET derived_from_ids=?,source_turn_ids=?,tool_call_ids=?,success_count=success_count+?
                             WHERE id=? AND status='active'""",
                          (_dump(list(dict.fromkeys([*fact["derived_from_ids"], *refs]))), _dump(turn_ids),
                           _dump(list(dict.fromkeys([*old_calls, *calls]))), len(set(successful) - set(old_calls)), fact["id"]))

    @staticmethod
    def maintenance_key(scope):
        return "decay:" + _dump(scope[:2])

    def archive_cold(self, scope, *, now, half_life_days, inactive_days, superseded_days):
        with self.transaction():
            key = self.maintenance_key(scope)
            previous = self.get_meta(key)
            if previous:
                now = max(now, json.loads(previous)["as_of"])
            rows = self.conn.execute(f"""UPDATE facts SET tier='cold' WHERE user_id=? AND agent_id=? AND tier='hot'
                AND ((expires_at IS NOT NULL AND expires_at<=?) OR (NOT {PROTECTED} AND
                ((status='superseded' AND valid_until<?) OR
                (status='active' AND coalesce(last_accessed_at,ingest_time)<? AND hit_count=0)))) RETURNING id,rowid""",
                                     (*scope[:2], now, now - superseded_days * DAY, now - inactive_days * DAY)).fetchall()
            if rows:
                ids = _dump([row["rowid"] for row in rows])
                self.conn.execute(f"INSERT INTO facts_cold_fts(rowid,fact_id,user_id,agent_id,fts_tokens) "
                                  f"SELECT rowid,fact_id,user_id,agent_id,fts_tokens FROM facts_fts WHERE rowid IN ({_IDS})", (ids,))
                self.conn.execute(f"DELETE FROM facts_fts WHERE rowid IN ({_IDS})", (ids,))
                self.conn.execute(f"UPDATE fact_entities SET tier='cold' WHERE fact_id IN ({_IDS})",
                                  (_dump([row["id"] for row in rows]),))
            self.set_meta(key, _dump({"as_of": now, "half_life_days": half_life_days}))
        ids = [row["id"] for row in rows]
        self._forget_vectors(scope, ids)
        return ids

    def insert_line(self, *, timestamp, event_time=None, raw_dist=None, **fields):
        return self.insert_fact(ingest_time=timestamp, event_time_start=event_time, **fields)

    def update_line(self, line_id, *, timestamp, event_time=None, raw_dist=None, source_step_ids=None, **fields):
        with self.transaction():
            row = self._resolve_fact(line_id, scope=fields.get("scope", DEFAULT_SCOPE), status="status='active'")
            return row is not None and self.supersede_fact(row["id"], ingest_time=timestamp, event_time_start=event_time,
                                                           current_step_ids=source_step_ids, **fields) is not None

    def get_line(self, line_id, *, scope=DEFAULT_SCOPE, include_deleted=False):
        with self.read_snapshot():
            row = self._resolve_fact(line_id, scope=scope, status="TRUE" if include_deleted else "status='active'")
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
