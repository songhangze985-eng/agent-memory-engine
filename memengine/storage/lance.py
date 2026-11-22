"""Embedded SQLite ACID ledger paired with LanceDB (Rust + Apache Arrow) disk-native ANN & BM25."""
from __future__ import annotations

import json
import math
import os
import re
from contextlib import contextmanager
from http.client import HTTPException
from pathlib import Path

import numpy as np

from memengine.storage.base import DEFAULT_SCOPE, IDS as _IDS, scope_where as _scope_where
from memengine.storage.sqlite import SQLiteStore, _dump
from memengine.textutil import tokenize_for_fts
from memengine.vector import normalize_vectors

_DEFAULT_ANN_THRESHOLD = 2048
_DEFAULT_FLUSH_BATCH = 512
_TABLE_NAME_SAFE = re.compile(r"^[a-z0-9_]+$")


def _quote_sql(value: str) -> str:
    if "\x00" in value:
        raise ValueError("null bytes are not allowed in filter values")
    return "'" + value.replace("'", "''") + "'"


def _finite_float(value) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite numeric filter: {value!r}")
    return number


def _resolve_paths(target: str | Path) -> tuple[Path, Path]:
    path = Path(target).expanduser().resolve()
    if path.suffix in {".db", ".sqlite", ".sqlite3"}:
        return path, path.with_suffix(".lance")
    path.mkdir(parents=True, exist_ok=True)
    return path / "ledger.db", path / "lance"


class LanceHybridStore(SQLiteStore):
    """ACID relational ledger in SQLite WAL + zero-copy Arrow/LanceDB Rust vector & FTS engine.

    - All 15 relational tables, version chains, FK cascades, working memory, and 2-hop graph links
      stay strictly ACID in SQLite.
    - High-volume vector search (``fact_vectors`` & ``fact_anchors``) runs in embedded LanceDB
      using Rust SIMD mmap over Apache Arrow and automatic RaBitQ (``IVF_RQ``) quantized ANN
      indexing once tables pass ``ann_threshold`` — eliminating O(N*D) NumPy matrix scans and RAM churn.
    - All LanceDB mutations are staged per SQLite transaction / savepoint and flushed only after
      outermost commit (discarded on rollback), with ``PRAGMA data_version`` external-write recovery.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        embedder=None,
        ann_threshold: int | None = None,
        flush_batch_size: int = _DEFAULT_FLUSH_BATCH,
        enable_lance_fts: bool = False,
    ):
        try:
            import lancedb
            import pyarrow as pa
        except ImportError as exc:
            raise RuntimeError("LanceDB support requires: pip install 'memengine[lance]'") from exc
        self._lancedb = lancedb
        self._pa = pa
        ledger_path, lance_dir = _resolve_paths(path)
        self.lance_dir = lance_dir
        self.lance_dir.mkdir(parents=True, exist_ok=True)
        self.ann_threshold = int(
            os.getenv("MEMENGINE_LANCE_ANN_THRESHOLD", _DEFAULT_ANN_THRESHOLD)
            if ann_threshold is None
            else ann_threshold
        )
        self.flush_batch_size = max(1, int(flush_batch_size))
        self.enable_lance_fts = bool(enable_lance_fts)
        self._lance_tx_stack: list[list[tuple]] = []
        self._lance_dirty: list[tuple] = []
        self._lance_tables: dict[str, object] = {}
        self._ann_indexed: set[str] = set()
        self._fts_indexed: set[str] = set()
        self._lance_db = lancedb.connect(str(self.lance_dir))
        super().__init__(ledger_path, embedder=embedder)
        self._lance_data_version = self._pragma_data_version()
        self._has_deleted_facts = bool(self.conn.execute("SELECT 1 FROM facts WHERE status='deleted' LIMIT 1").fetchone())
        self._bootstrap_lance()

    def _pragma_data_version(self) -> int:
        return int(self.conn.execute("PRAGMA data_version").fetchone()[0])

    def _table_names(self) -> set[str]:
        result = self._lance_db.list_tables()
        names = getattr(result, "tables", result)
        return set(names)

    def _open_table(self, name: str):
        if name in self._lance_tables:
            return self._lance_tables[name]
        if name in self._table_names():
            table = self._lance_db.open_table(name)
            self._lance_tables[name] = table
            return table
        return None

    def _fact_table_name(self, dim: int) -> str:
        name = f"fact_vectors_{int(dim)}"
        assert _TABLE_NAME_SAFE.match(name)
        return name

    def _anchor_table_name(self, dim: int) -> str:
        name = f"fact_anchors_{int(dim)}"
        assert _TABLE_NAME_SAFE.match(name)
        return name

    def _fact_schema(self, dim: int):
        pa = self._pa
        return pa.schema([
            pa.field("fact_id", pa.string()),
            pa.field("user_id", pa.string()),
            pa.field("agent_id", pa.string()),
            pa.field("session_id", pa.string()),
            pa.field("status", pa.string()),
            pa.field("tier", pa.string()),
            pa.field("memory_type", pa.string()),
            pa.field("epistemic_scope", pa.string()),
            pa.field("model_name", pa.string()),
            pa.field("fts_tokens", pa.string()),
            pa.field("event_time_start", pa.float64()),
            pa.field("valid_from", pa.float64()),
            pa.field("expires_at", pa.float64()),
            pa.field("is_stale", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), int(dim))),
        ])

    def _anchor_schema(self, dim: int):
        pa = self._pa
        return pa.schema([
            pa.field("fact_id", pa.string()),
            pa.field("user_id", pa.string()),
            pa.field("agent_id", pa.string()),
            pa.field("session_id", pa.string()),
            pa.field("status", pa.string()),
            pa.field("tier", pa.string()),
            pa.field("memory_type", pa.string()),
            pa.field("epistemic_scope", pa.string()),
            pa.field("model_name", pa.string()),
            pa.field("event_time_start", pa.float64()),
            pa.field("valid_from", pa.float64()),
            pa.field("expires_at", pa.float64()),
            pa.field("is_stale", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), int(dim))),
        ])

    def _stage(self, op: tuple):
        if self._lance_tx_stack:
            self._lance_tx_stack[-1].append(op)
        else:
            self._lance_dirty.append(op)
            if len(self._lance_dirty) >= self.flush_batch_size:
                self.flush_index()

    @contextmanager
    def transaction(self):
        self._lance_tx_stack.append([])
        committed = False
        try:
            with super().transaction():
                yield
            committed = True
        finally:
            frame = self._lance_tx_stack.pop()
            if committed and frame:
                if self._lance_tx_stack:
                    self._lance_tx_stack[-1].extend(frame)
                else:
                    self._lance_dirty.extend(frame)
                    if len(self._lance_dirty) >= self.flush_batch_size:
                        self.flush_index()

    def _bootstrap_lance(self):
        sql_facts = self.conn.execute("SELECT count(*) FROM fact_vectors").fetchone()[0]
        existing_tables = [name for name in self._table_names() if name.startswith("fact_vectors_")]
        lance_facts = sum(self._open_table(name).count_rows() for name in existing_tables)
        if sql_facts != lance_facts:
            self.rebuild_index()
        else:
            for name in existing_tables:
                table = self._open_table(name)
                if table is not None:
                    for idx in table.list_indices():
                        if idx.columns == ["vector"]:
                            self._ann_indexed.add(name)
                        elif idx.columns == ["fts_tokens"]:
                            self._fts_indexed.add(name)

    def _insert_version(self, data, zkeys, *, offline=False, chronicle=None):
        fact_id = super()._insert_version(data, zkeys, offline=offline, chronicle=chronicle)
        row = self.conn.execute(
            """SELECT f.id, f.user_id, f.agent_id, f.session_id, f.status, f.tier,
                      f.memory_type, f.epistemic_scope, f.event_time_start, f.valid_from,
                      f.expires_at, f.is_stale, f.normalized_text, f.plain,
                      v.model_name, v.dim, v.vec
               FROM facts f JOIN fact_vectors v ON v.fact_id=f.id WHERE f.id=?""",
            (fact_id,),
        ).fetchone()
        if row is not None:
            indexed = row["normalized_text"] or row["plain"]
            self._stage((
                "upsert_fact",
                int(row["dim"]),
                {
                    "fact_id": row["id"],
                    "user_id": row["user_id"],
                    "agent_id": row["agent_id"],
                    "session_id": row["session_id"],
                    "status": row["status"],
                    "tier": row["tier"],
                    "memory_type": row["memory_type"],
                    "epistemic_scope": row["epistemic_scope"],
                    "model_name": row["model_name"],
                    "fts_tokens": tokenize_for_fts(indexed),
                    "event_time_start": float(row["event_time_start"]),
                    "valid_from": None if row["valid_from"] is None else float(row["valid_from"]),
                    "expires_at": None if row["expires_at"] is None else float(row["expires_at"]),
                    "is_stale": int(row["is_stale"]),
                    "vec_bytes": row["vec"],
                },
            ))
        return fact_id

    def _index_turn_anchor(self, data, fact_rowid, *, offline):
        super()._index_turn_anchor(data, fact_rowid, offline=offline)
        row = self.conn.execute(
            """SELECT f.id, f.user_id, f.agent_id, f.session_id, f.status, f.tier,
                      f.memory_type, f.epistemic_scope, f.event_time_start, f.valid_from,
                      f.expires_at, f.is_stale, a.model_name, a.dim, a.vec
               FROM facts f JOIN fact_anchors a ON a.fact_id=f.id WHERE f.id=?""",
            (data["id"],),
        ).fetchone()
        if row is not None:
            self._stage((
                "upsert_anchor",
                int(row["dim"]),
                {
                    "fact_id": row["id"],
                    "user_id": row["user_id"],
                    "agent_id": row["agent_id"],
                    "session_id": row["session_id"],
                    "status": row["status"],
                    "tier": row["tier"],
                    "memory_type": row["memory_type"],
                    "epistemic_scope": row["epistemic_scope"],
                    "model_name": row["model_name"],
                    "event_time_start": float(row["event_time_start"]),
                    "valid_from": None if row["valid_from"] is None else float(row["valid_from"]),
                    "expires_at": None if row["expires_at"] is None else float(row["expires_at"]),
                    "is_stale": int(row["is_stale"]),
                    "vec_bytes": row["vec"],
                },
            ))

    def revise_slot(self, data):
        before = data.get("supersedes_id")
        super().revise_slot(data)
        superseded = data.get("supersedes_id")
        if superseded and superseded != before:
            self._stage(("set_status", [superseded], "superseded"))

    def supersede_fact(self, old_fact_id, **kwargs):
        new_id = super().supersede_fact(old_fact_id, **kwargs)
        if new_id is not None:
            self._stage(("set_status", [old_fact_id], "superseded"))
        return new_id

    def delete_fact(self, fact_id, *, hard=False, ingest_time=None, scope=DEFAULT_SCOPE):
        with self.transaction():
            row = self._resolve_fact(fact_id, scope=scope, status="TRUE" if hard else "status='active'")
            if row is None:
                return False
            affected = (
                [r["id"] for r in self.conn.execute("SELECT id FROM facts WHERE root_fact_id=?", (row["root_fact_id"],))]
                if hard
                else [row["id"]]
            )
            deleted = super().delete_fact(fact_id, hard=hard, ingest_time=ingest_time, scope=scope)
            if deleted:
                if hard:
                    self._stage(("delete_facts", affected))
                else:
                    self._has_deleted_facts = True
                    self._stage(("set_status", affected, "deleted"))
            return deleted

    def invalidate_observations(self, scope, source_ids, stamp, *, compiled_line=""):
        stale_before = {
            row["id"]
            for row in self.conn.execute(
                f"""SELECT o.id FROM facts o
                    WHERE o.user_id=? AND o.agent_id=? AND o.memory_type='observation'
                      AND o.status='active' AND o.is_stale=0 AND EXISTS (
                          SELECT 1 FROM json_each(o.derived_from_ids) j WHERE j.value IN ({_IDS})
                      )""",
                (*scope[:2], _dump(source_ids)),
            )
        }
        super().invalidate_observations(scope, source_ids, stamp, compiled_line=compiled_line)
        if stale_before:
            self._stage(("mark_stale", sorted(stale_before)))

    def link_chronicle(self, data, zkeys, model, vector):
        super().link_chronicle(data, zkeys, model, vector)
        if data.get("canonical_entity_id") and data["epistemic_scope"] in ("world", "experience"):
            stale = [
                row["id"]
                for row in self.conn.execute(
                    """SELECT id FROM facts WHERE user_id=? AND agent_id=?
                       AND canonical_entity_id=? AND memory_type='observation' AND status='active' AND is_stale=1""",
                    (data["user_id"], data["agent_id"], data["canonical_entity_id"]),
                )
            ]
            if stale:
                self._stage(("mark_stale", stale))

    def archive_cold(self, scope, *, now, half_life_days, inactive_days, superseded_days):
        archived = super().archive_cold(
            scope, now=now, half_life_days=half_life_days,
            inactive_days=inactive_days, superseded_days=superseded_days,
        )
        if archived:
            self._stage(("set_tier", archived, "cold"))
        return archived

    def _rows_to_arrow(self, rows: list[dict], dim: int, *, include_fts: bool):
        pa = self._pa
        raw_vecs = np.frombuffer(b"".join(row["vec_bytes"] for row in rows), dtype="<f4")
        vec_array = pa.FixedSizeListArray.from_arrays(pa.array(raw_vecs, type=pa.float32()), int(dim))
        columns = {
            "fact_id": pa.array([row["fact_id"] for row in rows], type=pa.string()),
            "user_id": pa.array([row["user_id"] for row in rows], type=pa.string()),
            "agent_id": pa.array([row["agent_id"] for row in rows], type=pa.string()),
            "session_id": pa.array([row["session_id"] for row in rows], type=pa.string()),
            "status": pa.array([row["status"] for row in rows], type=pa.string()),
            "tier": pa.array([row["tier"] for row in rows], type=pa.string()),
            "memory_type": pa.array([row["memory_type"] for row in rows], type=pa.string()),
            "epistemic_scope": pa.array([row["epistemic_scope"] for row in rows], type=pa.string()),
            "model_name": pa.array([row["model_name"] for row in rows], type=pa.string()),
            "event_time_start": pa.array([row["event_time_start"] for row in rows], type=pa.float64()),
            "valid_from": pa.array([row["valid_from"] for row in rows], type=pa.float64()),
            "expires_at": pa.array([row["expires_at"] for row in rows], type=pa.float64()),
            "is_stale": pa.array([row["is_stale"] for row in rows], type=pa.int32()),
            "vector": vec_array,
        }
        if include_fts:
            columns["fts_tokens"] = pa.array([row["fts_tokens"] for row in rows], type=pa.string())
        schema = self._fact_schema(dim) if include_fts else self._anchor_schema(dim)
        return pa.table(columns, schema=schema)

    def _get_or_create_table(self, name: str, dim: int, *, include_fts: bool, initial_batch=None):
        table = self._open_table(name)
        if table is not None:
            return table, False
        schema = self._fact_schema(dim) if include_fts else self._anchor_schema(dim)
        if initial_batch is not None:
            table = self._lance_db.create_table(name, data=initial_batch, schema=schema)
            self._lance_tables[name] = table
            return table, True
        table = self._lance_db.create_table(name, schema=schema)
        self._lance_tables[name] = table
        return table, False

    def flush_index(self):
        """Coalesce and flush all committed SQLite mutations to LanceDB in Arrow batches."""
        if not self._lance_dirty:
            return
        ops = self._lance_dirty
        self._lance_dirty = []
        upserts_fact: dict[str, tuple[int, dict]] = {}
        upserts_anchor: dict[str, tuple[int, dict]] = {}
        status_updates: dict[str, str] = {}
        tier_updates: dict[str, str] = {}
        stale_ids: set[str] = set()
        deleted_ids: set[str] = set()

        for op in ops:
            kind = op[0]
            if kind == "upsert_fact":
                _, dim, row = op
                fid = row["fact_id"]
                deleted_ids.discard(fid)
                upserts_fact[fid] = (dim, dict(row))
            elif kind == "upsert_anchor":
                _, dim, row = op
                fid = row["fact_id"]
                deleted_ids.discard(fid)
                upserts_anchor[fid] = (dim, dict(row))
            elif kind == "set_status":
                _, fids, new_status = op
                for fid in fids:
                    if fid in upserts_fact:
                        upserts_fact[fid][1]["status"] = new_status
                    else:
                        status_updates[fid] = new_status
                    if fid in upserts_anchor:
                        upserts_anchor[fid][1]["status"] = new_status
            elif kind == "set_tier":
                _, fids, new_tier = op
                for fid in fids:
                    if fid in upserts_fact:
                        upserts_fact[fid][1]["tier"] = new_tier
                    else:
                        tier_updates[fid] = new_tier
                    if fid in upserts_anchor:
                        upserts_anchor[fid][1]["tier"] = new_tier
            elif kind == "mark_stale":
                _, fids = op
                for fid in fids:
                    if fid in upserts_fact:
                        upserts_fact[fid][1]["is_stale"] = 1
                    else:
                        stale_ids.add(fid)
                    if fid in upserts_anchor:
                        upserts_anchor[fid][1]["is_stale"] = 1
            elif kind == "delete_facts":
                _, fids = op
                for fid in fids:
                    upserts_fact.pop(fid, None)
                    upserts_anchor.pop(fid, None)
                    status_updates.pop(fid, None)
                    tier_updates.pop(fid, None)
                    stale_ids.discard(fid)
                    deleted_ids.add(fid)

        # 1. Batch-insert new facts per dimension (zero-copy Arrow FixedSizeList)
        for group_map, name_fn, has_fts in (
            (upserts_fact, self._fact_table_name, True),
            (upserts_anchor, self._anchor_table_name, False),
        ):
            by_dim: dict[int, list[dict]] = {}
            for dim, row in group_map.values():
                by_dim.setdefault(dim, []).append(row)
            for dim, rows in by_dim.items():
                name = name_fn(dim)
                arrow_table = self._rows_to_arrow(rows, dim, include_fts=has_fts)
                table, created = self._get_or_create_table(name, dim, include_fts=has_fts, initial_batch=arrow_table)
                if not created:
                    table.add(arrow_table)
                if has_fts and name not in self._ann_indexed and table.count_rows() >= self.ann_threshold:
                    self._build_table_indexes(name, table, include_fts=self.enable_lance_fts)

        # 2. Apply batched updates/deletes across existing Lance tables
        all_tables = [self._open_table(name) for name in self._table_names()
                      if name.startswith(("fact_vectors_", "fact_anchors_"))]
        all_tables = [t for t in all_tables if t is not None]
        if deleted_ids:
            predicate = "fact_id IN (" + ",".join(_quote_sql(fid) for fid in sorted(deleted_ids)) + ")"
            for table in all_tables:
                table.delete(predicate)
        if status_updates:
            by_status: dict[str, list[str]] = {}
            for fid, st in status_updates.items():
                by_status.setdefault(st, []).append(fid)
            for st, fids in by_status.items():
                predicate = "fact_id IN (" + ",".join(_quote_sql(fid) for fid in sorted(fids)) + ")"
                for table in all_tables:
                    table.update(where=predicate, values={"status": st})
        if tier_updates:
            by_tier: dict[str, list[str]] = {}
            for fid, tier in tier_updates.items():
                by_tier.setdefault(tier, []).append(fid)
            for tier, fids in by_tier.items():
                predicate = "fact_id IN (" + ",".join(_quote_sql(fid) for fid in sorted(fids)) + ")"
                for table in all_tables:
                    table.update(where=predicate, values={"tier": tier})
        if stale_ids:
            predicate = "fact_id IN (" + ",".join(_quote_sql(fid) for fid in sorted(stale_ids)) + ")"
            for table in all_tables:
                table.update(where=predicate, values={"is_stale": 1})

    def _build_table_indexes(self, name: str, table, *, include_fts: bool = False):
        count = table.count_rows()
        if count >= 256:
            from lancedb.index import IvfRq

            partitions = max(4, min(256, int(math.sqrt(count))))
            table.create_index("vector", config=IvfRq(distance_type="cosine", num_partitions=partitions), replace=True)
            self._ann_indexed.add(name)
        if include_fts and count > 0:
            from lancedb.index import FTS

            table.create_index(
                "fts_tokens",
                config=FTS(base_tokenizer="whitespace", stem=False, remove_stop_words=False, lower_case=False),
                replace=True,
            )
            self._fts_indexed.add(name)
        table.optimize()

    def optimize_index(self, *, include_fts: bool | None = None):
        """Explicitly compact Lance fragments and build disk-friendly RaBitQ (IVF_RQ) ANN indexes."""
        self.flush_index()
        use_fts = self.enable_lance_fts if include_fts is None else bool(include_fts)
        indexed = []
        for name in sorted(self._table_names()):
            if not name.startswith(("fact_vectors_", "fact_anchors_")):
                continue
            table = self._open_table(name)
            if table is None:
                continue
            self._build_table_indexes(name, table, include_fts=use_fts and name.startswith("fact_vectors_"))
            indexed.append(name)
        return indexed

    def rebuild_index(self):
        """Reconstruct LanceDB Arrow tables from the authoritative SQLite ledger."""
        self._lance_dirty.clear()
        for name in list(self._table_names()):
            if name.startswith(("fact_vectors_", "fact_anchors_")):
                self._lance_db.drop_table(name)
        self._lance_tables.clear()
        self._ann_indexed.clear()
        self._fts_indexed.clear()

        fact_rows_by_dim: dict[int, list[dict]] = {}
        for row in self.conn.execute(
            """SELECT f.id, f.user_id, f.agent_id, f.session_id, f.status, f.tier,
                      f.memory_type, f.epistemic_scope, f.event_time_start, f.valid_from,
                      f.expires_at, f.is_stale, f.normalized_text, f.plain,
                      v.model_name, v.dim, v.vec
               FROM facts f JOIN fact_vectors v ON v.fact_id=f.id"""
        ):
            indexed = row["normalized_text"] or row["plain"]
            fact_rows_by_dim.setdefault(int(row["dim"]), []).append({
                "fact_id": row["id"],
                "user_id": row["user_id"],
                "agent_id": row["agent_id"],
                "session_id": row["session_id"],
                "status": row["status"],
                "tier": row["tier"],
                "memory_type": row["memory_type"],
                "epistemic_scope": row["epistemic_scope"],
                "model_name": row["model_name"],
                "fts_tokens": tokenize_for_fts(indexed),
                "event_time_start": float(row["event_time_start"]),
                "valid_from": None if row["valid_from"] is None else float(row["valid_from"]),
                "expires_at": None if row["expires_at"] is None else float(row["expires_at"]),
                "is_stale": int(row["is_stale"]),
                "vec_bytes": row["vec"],
            })
        for dim, rows in fact_rows_by_dim.items():
            name = self._fact_table_name(dim)
            arrow_table = self._rows_to_arrow(rows, dim, include_fts=True)
            table = self._lance_db.create_table(name, data=arrow_table, schema=self._fact_schema(dim))
            self._lance_tables[name] = table
            if len(rows) >= self.ann_threshold:
                self._build_table_indexes(name, table, include_fts=self.enable_lance_fts)

        anchor_rows_by_dim: dict[int, list[dict]] = {}
        for row in self.conn.execute(
            """SELECT f.id, f.user_id, f.agent_id, f.session_id, f.status, f.tier,
                      f.memory_type, f.epistemic_scope, f.event_time_start, f.valid_from,
                      f.expires_at, f.is_stale, a.model_name, a.dim, a.vec
               FROM facts f JOIN fact_anchors a ON a.fact_id=f.id"""
        ):
            anchor_rows_by_dim.setdefault(int(row["dim"]), []).append({
                "fact_id": row["id"],
                "user_id": row["user_id"],
                "agent_id": row["agent_id"],
                "session_id": row["session_id"],
                "status": row["status"],
                "tier": row["tier"],
                "memory_type": row["memory_type"],
                "epistemic_scope": row["epistemic_scope"],
                "model_name": row["model_name"],
                "event_time_start": float(row["event_time_start"]),
                "valid_from": None if row["valid_from"] is None else float(row["valid_from"]),
                "expires_at": None if row["expires_at"] is None else float(row["expires_at"]),
                "is_stale": int(row["is_stale"]),
                "vec_bytes": row["vec"],
            })
        for dim, rows in anchor_rows_by_dim.items():
            name = self._anchor_table_name(dim)
            arrow_table = self._rows_to_arrow(rows, dim, include_fts=False)
            table = self._lance_db.create_table(name, data=arrow_table, schema=self._anchor_schema(dim))
            self._lance_tables[name] = table
            if len(rows) >= self.ann_threshold:
                self._build_table_indexes(name, table, include_fts=False)

        self._lance_data_version = self._pragma_data_version()
        self._has_deleted_facts = bool(self.conn.execute("SELECT 1 FROM facts WHERE status='deleted' LIMIT 1").fetchone())

    def _ensure_lance_synced(self):
        current = self._pragma_data_version()
        if current != self._lance_data_version:
            self.rebuild_index()
            return
        self.flush_index()

    @staticmethod
    def _lance_predicate(scope, *, model_name, include_superseded, include_cold, memory_type, filters) -> str:
        clauses = [
            f"user_id = {_quote_sql(scope[0])}",
            f"agent_id = {_quote_sql(scope[1])}",
            f"model_name = {_quote_sql(model_name)}",
        ]
        if scope[2] is not None:
            clauses.append(f"session_id = {_quote_sql(scope[2])}")
        if include_superseded:
            clauses.append("status IN ('active', 'superseded')")
        else:
            clauses.append("status = 'active'")
        if not include_cold:
            clauses.append("tier = 'hot'")
        if memory_type is not None:
            clauses.append(f"memory_type = {_quote_sql(memory_type)}")
        if filters is None or (filters.memory_types is None and filters.scopes is None):
            clauses.append("memory_type != 'observation'")
        else:
            clauses.append("(memory_type != 'observation' OR is_stale = 0)")
        if filters is not None:
            if filters.memory_types is not None:
                vals = ",".join(_quote_sql(v) for v in filters.memory_types)
                clauses.append(f"memory_type IN ({vals})" if vals else "1 = 0")
            if filters.scopes is not None:
                vals = ",".join(_quote_sql(v) for v in filters.scopes)
                clauses.append(f"epistemic_scope IN ({vals})" if vals else "1 = 0")
            if filters.time_start is not None:
                clauses.append(f"event_time_start >= {_finite_float(filters.time_start)}")
            if filters.time_end is not None:
                clauses.append(f"event_time_start < {_finite_float(filters.time_end)}")
            if filters.as_of is not None:
                as_of = _finite_float(filters.as_of)
                clauses.append(f"(valid_from IS NULL OR valid_from <= {as_of})")
                clauses.append(f"(expires_at IS NULL OR expires_at > {as_of})")
            if filters.historical_window is not None:
                lower, upper = filters.historical_window
                if lower is not None:
                    clauses.append(f"(expires_at IS NULL OR expires_at > {_finite_float(lower)})")
                if upper is not None:
                    clauses.append(f"(expires_at IS NULL OR valid_from < {_finite_float(upper)})")
        return " AND ".join(clauses)

    def _verify_relational_ids(self, candidate_ids, scope, include_superseded, include_cold, filters):
        if not candidate_ids:
            return set()
        where, args = _scope_where(scope, "f.")
        if filters is not None:
            where, args = filters.sql(where, args)
        rows = self.conn.execute(
            f"SELECT f.id FROM facts f WHERE {where} AND {self._searchable(include_superseded, include_cold, filters)} "
            f"AND f.id IN ({_IDS})",
            (*args, _dump(list(candidate_ids))),
        )
        return {row["id"] for row in rows}

    def _lance_vector_search(
        self,
        *,
        table_name_fn,
        query: str,
        scope,
        include_superseded: bool,
        include_cold: bool,
        top_n: int,
        memory_type=None,
        filters=None,
    ) -> list[dict]:
        if not query.strip() or top_n <= 0:
            return []
        self._ensure_lance_synced()
        needs_verify = bool(
            (include_superseded and self._has_deleted_facts)
            or (filters is not None and (filters.session_provenance or filters.memory_types is not None or filters.scopes is not None))
        )
        fetch_k = max(top_n * 2, top_n + 16) if needs_verify else top_n
        hits = []
        for backend in {self.fallback_embedder.model_name: self.fallback_embedder, self.embedder.model_name: self.embedder}.values():
            dim = getattr(backend, "dim", None)
            if dim is not None and self._open_table(table_name_fn(dim)) is None:
                continue
            try:
                encode = backend.encode if self._write_vectors is not None else getattr(backend, "encode_queries", backend.encode)
                encoded = normalize_vectors(encode([query]))[0].astype(np.float32, copy=False)
            except (OSError, ValueError, TypeError, KeyError, HTTPException):
                continue
            if self._write_vectors is not None:
                self._write_vectors[backend.model_name, query] = encoded
            table = self._open_table(table_name_fn(len(encoded)))
            if table is None:
                continue
            predicate = self._lance_predicate(
                scope,
                model_name=backend.model_name,
                include_superseded=include_superseded,
                include_cold=include_cold,
                memory_type=memory_type,
                filters=filters,
            )
            rows = (
                table.search(encoded, query_type="vector")
                .metric("cosine")
                .where(predicate, prefilter=True)
                .select(["fact_id", "_distance"])
                .limit(fetch_k)
                .to_list()
            )
            for row in rows:
                score = min(1.0, 1.0 - float(row["_distance"]))
                if score > 0:
                    hits.append({"id": row["fact_id"], "vector_score": score})
        if needs_verify and hits:
            allowed = self._verify_relational_ids(
                [hit["id"] for hit in hits], scope, include_superseded, include_cold, filters
            )
            hits = [hit for hit in hits if hit["id"] in allowed]
        hits.sort(key=lambda fact: (-fact["vector_score"], fact["id"]))
        return hits[:top_n]

    def search_vector(
        self,
        query,
        *,
        user_id="default",
        agent_id="default",
        session_id=None,
        include_superseded=True,
        top_n=40,
        ids_only=False,
        include_cold=False,
        memory_type=None,
        filters=None,
    ):
        # Inside an uncommitted write transaction, read from SQLite so uncommitted mutations/rollbacks stay exact.
        if self._write_depth or (self.conn.in_transaction and not self._snapshot_depth):
            return super().search_vector(
                query, user_id=user_id, agent_id=agent_id, session_id=session_id,
                include_superseded=include_superseded, top_n=top_n, ids_only=ids_only,
                include_cold=include_cold, memory_type=memory_type, filters=filters,
            )
        scope = (user_id, agent_id, session_id)
        hits = self._lance_vector_search(
            table_name_fn=self._fact_table_name,
            query=query,
            scope=scope,
            include_superseded=include_superseded,
            include_cold=include_cold,
            top_n=top_n,
            memory_type=memory_type,
            filters=filters,
        )
        return self._search_results(hits, scope, ids_only)

    def search_turn_anchors(
        self,
        query,
        semantic_query,
        *,
        user_id="default",
        agent_id="default",
        session_id=None,
        top_n=35,
        include_cold=False,
        filters=None,
    ):
        if self._write_depth or (self.conn.in_transaction and not self._snapshot_depth):
            return super().search_turn_anchors(
                query, semantic_query, user_id=user_id, agent_id=agent_id,
                session_id=session_id, top_n=top_n, include_cold=include_cold, filters=filters,
            )
        self._ensure_lance_synced()
        if not any(name.startswith("fact_anchors_") for name in self._table_names()):
            return [], []
        scope = (user_id, agent_id, session_id)
        where, args = _scope_where(scope, "f.")
        if filters is not None:
            where, args = filters.sql(where, args)
        eligible = f"{where} AND {self._searchable(True, include_cold, filters)}"
        from memengine.storage.sqlite import _FTS_STOP

        terms = list(dict.fromkeys(word for word in tokenize_for_fts(query).split() if word not in _FTS_STOP))
        lexical = []
        if terms:
            expression = " OR ".join(f'"{term}"' for term in terms[:128])
            lexical = [
                dict(row)
                for row in self.conn.execute(
                    f"""SELECT f.id,bm25(facts_anchor_fts) AS anchor_bm25_score, a.peer, a.turn_id AS context_anchor_turn_id
                        FROM facts_anchor_fts JOIN facts f ON f.rowid=facts_anchor_fts.rowid
                        JOIN fact_anchors a ON a.fact_id=f.id
                        WHERE facts_anchor_fts MATCH ? AND {eligible} ORDER BY anchor_bm25_score,f.id LIMIT ?""",
                    (expression, *args, top_n),
                )
            ]
        dense = self._lance_vector_search(
            table_name_fn=self._anchor_table_name,
            query=semantic_query,
            scope=scope,
            include_superseded=True,
            include_cold=include_cold,
            top_n=top_n,
            filters=filters,
        )
        return lexical, dense

    def health_check(self):
        base = super().health_check()
        self.flush_index()
        tables = sorted(name for name in self._table_names() if name.startswith(("fact_vectors_", "fact_anchors_")))
        return {
            **base,
            "backend": "sqlite+lancedb",
            "vector": "lancedb-ivf-rq" if self._ann_indexed else "lancedb-arrow",
            "lexical": "lancedb-fts+sqlite-fts5" if self._fts_indexed else "sqlite-fts5",
            "lance_dir": str(self.lance_dir),
            "lance_tables": tables,
            "ann_indexed": sorted(self._ann_indexed),
        }

    def close(self):
        try:
            self.flush_index()
        finally:
            super().close()
