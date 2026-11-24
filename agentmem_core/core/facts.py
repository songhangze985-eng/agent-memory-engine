"""Unified Fact Layer metadata, kind taxonomy, scope chains, staleness, optimistic locking & snapshots."""
from __future__ import annotations

import json
import time
import uuid

from agentmem_core.core.scopes import ScopeChain, normalize_override_key, normalize_scope_key
from agentmem_core.vfs.models import (
    OptimisticLockError,
    make_index_summary,
    normalize_dir_path,
    normalize_node_path,
)

FACT_KINDS = ("world", "experience", "preference", "belief", "rule")
SALIENCE_TAGS = ("normal", "preference", "taboo", "core_rule", "warning")
VERIFICATION_STATES = ("unverified", "verified")
STALENESS_STATES = ("fresh", "stale", "unverified")


class UnifiedFactLedger:
    """Single-source-of-truth metadata overlay on the core ``facts`` table.

    Unifies bi-temporal facts, preferences, distilled beliefs, and procedural rules
    (with hierarchical scope keys, override keys, code-anchor staleness state, VFS path
    projection, optimistic locking, and pre-mutation rollback snapshots) without
    duplicating facts into a second storage table.
    """

    def __init__(self, store):
        self.store = store
        self._ensure_schema()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        try:
            self.store.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        except Exception:
            # Column already exists (SQLite OperationalError / PG DuplicateColumn)
            try:
                self.store.conn.rollback()
            except Exception:
                pass

    def _ensure_schema(self) -> None:
        self.store.conn.execute(
            """CREATE TABLE IF NOT EXISTS fact_unified_meta (
                fact_id TEXT PRIMARY KEY REFERENCES facts(id) ON DELETE CASCADE,
                user_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'world' CHECK(kind IN ('world','experience','preference','belief','rule')),
                scope_key TEXT NOT NULL DEFAULT 'global',
                override_key TEXT NOT NULL DEFAULT '',
                vfs_path TEXT,
                parent_path TEXT,
                title TEXT NOT NULL DEFAULT '',
                index_summary TEXT NOT NULL DEFAULT '',
                verification_status TEXT NOT NULL DEFAULT 'verified' CHECK(verification_status IN ('unverified','verified')),
                salience_tag TEXT NOT NULL DEFAULT 'normal',
                always_active INTEGER NOT NULL DEFAULT 0,
                staleness_status TEXT NOT NULL DEFAULT 'fresh',
                stale_reason TEXT NOT NULL DEFAULT '',
                stale_detected_at DOUBLE PRECISION,
                version INTEGER NOT NULL DEFAULT 1,
                source_transcript_ids TEXT NOT NULL DEFAULT '[]',
                created_at DOUBLE PRECISION NOT NULL,
                updated_at DOUBLE PRECISION NOT NULL
            )"""
        )
        # Forward-compatible migration for existing databases built before universal scope/staleness columns
        self._ensure_column("fact_unified_meta", "scope_key", "TEXT NOT NULL DEFAULT 'global'")
        self._ensure_column("fact_unified_meta", "override_key", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("fact_unified_meta", "staleness_status", "TEXT NOT NULL DEFAULT 'fresh'")
        self._ensure_column("fact_unified_meta", "stale_reason", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("fact_unified_meta", "stale_detected_at", "DOUBLE PRECISION")

        self.store.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fum_scope_kind ON fact_unified_meta(user_id, agent_id, kind)"
        )
        self.store.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fum_scope_key ON fact_unified_meta(user_id, agent_id, scope_key)"
        )
        self.store.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fum_scope_path ON fact_unified_meta(user_id, agent_id, vfs_path)"
        )
        self.store.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fum_scope_parent ON fact_unified_meta(user_id, agent_id, parent_path)"
        )
        self.store.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fum_stale ON fact_unified_meta(user_id, agent_id, staleness_status)"
        )
        self.store.conn.execute(
            """CREATE TABLE IF NOT EXISTS fact_unified_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                batch_id TEXT,
                fact_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                scope_key TEXT NOT NULL DEFAULT 'global',
                vfs_path TEXT,
                kind TEXT NOT NULL,
                title TEXT NOT NULL,
                index_summary TEXT NOT NULL,
                content TEXT NOT NULL,
                verification_status TEXT NOT NULL,
                salience_tag TEXT NOT NULL,
                always_active INTEGER NOT NULL,
                version INTEGER NOT NULL,
                status TEXT NOT NULL,
                source_transcript_ids TEXT NOT NULL,
                created_at DOUBLE PRECISION NOT NULL
            )"""
        )
        self._ensure_column("fact_unified_snapshots", "scope_key", "TEXT NOT NULL DEFAULT 'global'")
        self.store.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fus_batch ON fact_unified_snapshots(user_id, agent_id, batch_id)"
        )
        self.store.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fus_path ON fact_unified_snapshots(user_id, agent_id, vfs_path)"
        )
        self.store.conn.commit()

    def attach(
        self,
        *,
        fact_id: str,
        user_id: str = "default",
        agent_id: str = "default",
        kind: str = "world",
        scope_key: str | None = None,
        override_key: str = "",
        vfs_path: str | None = None,
        title: str = "",
        content: str = "",
        index_summary: str = "",
        verification_status: str = "verified",
        salience_tag: str = "normal",
        always_active: bool = False,
        staleness_status: str = "fresh",
        stale_reason: str = "",
        source_transcript_ids: list[str] | tuple[str, ...] = (),
        expected_version: int | None = None,
        batch_id: str | None = None,
        timestamp: float | None = None,
    ) -> dict:
        if kind not in FACT_KINDS:
            raise ValueError(f"kind must be one of {FACT_KINDS}, got {kind!r}")
        if verification_status not in VERIFICATION_STATES:
            raise ValueError(f"verification_status must be one of {VERIFICATION_STATES}")
        if staleness_status not in STALENESS_STATES:
            raise ValueError(f"staleness_status must be one of {STALENESS_STATES}")
        now = time.time() if timestamp is None else float(timestamp)
        norm_scope = normalize_scope_key(scope_key or (f"user:{user_id}" if user_id else "global"))
        norm_override = normalize_override_key(override_key)
        norm_path: str | None = None
        parent_path: str | None = None
        if vfs_path:
            parent_path, norm_path = normalize_node_path(vfs_path)

        existing = (
            self.get_active_by_path(norm_path, user_id=user_id, agent_id=agent_id, scope_key=norm_scope)
            if norm_path
            else None
        )
        if expected_version is not None:
            actual_version = existing["version"] if existing else 0
            if actual_version != expected_version:
                raise OptimisticLockError(norm_path or fact_id, expected_version, actual_version)

        next_version = 1
        if existing is not None:
            self._save_snapshot(existing, batch_id=batch_id, timestamp=now)
            next_version = int(existing["version"]) + 1
            if existing["fact_id"] != fact_id:
                self.store.conn.execute(
                    "UPDATE fact_unified_meta SET vfs_path=NULL, parent_path=NULL, updated_at=? WHERE fact_id=?",
                    (now, existing["fact_id"]),
                )

        summary = make_index_summary(title or norm_path or fact_id, content, index_summary)
        tids = list(dict.fromkeys(str(t) for t in source_transcript_ids if t))
        stale_ts = now if staleness_status == "stale" else None
        self.store.conn.execute(
            """INSERT INTO fact_unified_meta(
                   fact_id, user_id, agent_id, kind, scope_key, override_key, vfs_path, parent_path,
                   title, index_summary, verification_status, salience_tag, always_active,
                   staleness_status, stale_reason, stale_detected_at,
                   version, source_transcript_ids, created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (fact_id) DO UPDATE SET
                   kind=excluded.kind,
                   scope_key=excluded.scope_key,
                   override_key=excluded.override_key,
                   vfs_path=excluded.vfs_path,
                   parent_path=excluded.parent_path,
                   title=excluded.title,
                   index_summary=excluded.index_summary,
                   verification_status=excluded.verification_status,
                   salience_tag=excluded.salience_tag,
                   always_active=excluded.always_active,
                   staleness_status=excluded.staleness_status,
                   stale_reason=excluded.stale_reason,
                   stale_detected_at=excluded.stale_detected_at,
                   version=excluded.version,
                   source_transcript_ids=excluded.source_transcript_ids,
                   updated_at=excluded.updated_at""",
            (
                fact_id,
                user_id,
                agent_id,
                kind,
                norm_scope,
                norm_override,
                norm_path,
                parent_path,
                title,
                summary,
                verification_status,
                salience_tag,
                int(bool(always_active)),
                staleness_status,
                stale_reason,
                stale_ts,
                next_version,
                json.dumps(tids, ensure_ascii=False),
                now,
                now,
            ),
        )
        return self.get_meta(fact_id)

    def set_staleness(
        self,
        fact_id: str,
        *,
        staleness_status: str,
        stale_reason: str = "",
        timestamp: float | None = None,
    ) -> dict | None:
        if staleness_status not in STALENESS_STATES:
            raise ValueError(f"staleness_status must be one of {STALENESS_STATES}")
        now = time.time() if timestamp is None else float(timestamp)
        detected = now if staleness_status == "stale" else None
        self.store.conn.execute(
            """UPDATE fact_unified_meta
               SET staleness_status=?, stale_reason=?, stale_detected_at=?, updated_at=?
               WHERE fact_id=?""",
            (staleness_status, stale_reason, detected, now, fact_id),
        )
        self.store.conn.commit()
        return self.get_meta(fact_id)

    def _save_snapshot(self, row: dict, *, batch_id: str | None = None, timestamp: float | None = None) -> str:
        snap_id = f"snap_{uuid.uuid4().hex[:14]}"
        now = time.time() if timestamp is None else float(timestamp)
        tids = row.get("source_transcript_ids") or []
        tids_json = tids if isinstance(tids, str) else json.dumps(list(tids), ensure_ascii=False)
        self.store.conn.execute(
            """INSERT INTO fact_unified_snapshots(
                   snapshot_id, batch_id, fact_id, user_id, agent_id, scope_key, vfs_path, kind, title,
                   index_summary, content, verification_status, salience_tag, always_active,
                   version, status, source_transcript_ids, created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                snap_id,
                batch_id,
                row["fact_id"],
                row["user_id"],
                row["agent_id"],
                row.get("scope_key", "global"),
                row.get("vfs_path"),
                row["kind"],
                row.get("title", ""),
                row.get("index_summary", ""),
                row.get("plain", row.get("content", "")),
                row["verification_status"],
                row["salience_tag"],
                int(bool(row.get("always_active"))),
                int(row["version"]),
                row.get("status", "active"),
                tids_json,
                now,
            ),
        )
        return snap_id

    @staticmethod
    def _decode_row(row) -> dict:
        data = dict(row)
        raw_tids = data.get("source_transcript_ids")
        if isinstance(raw_tids, str):
            try:
                data["source_transcript_ids"] = json.loads(raw_tids)
            except Exception:
                data["source_transcript_ids"] = []
        data["always_active"] = bool(data.get("always_active"))
        data.setdefault("scope_key", "global")
        data.setdefault("override_key", "")
        data.setdefault("staleness_status", "fresh")
        data.setdefault("stale_reason", "")
        return data

    def get_meta(self, fact_id: str) -> dict | None:
        row = self.store.conn.execute(
            """SELECT m.*, f.plain, f.status, f.confidence, f.source_turn_ids, f.event_time_start, f.ingest_time
               FROM fact_unified_meta m
               JOIN facts f ON f.id = m.fact_id
               WHERE m.fact_id = ?""",
            (fact_id,),
        ).fetchone()
        return self._decode_row(row) if row is not None else None

    def get_active_by_path(
        self,
        path: str,
        *,
        user_id: str = "default",
        agent_id: str = "default",
        scope_key: str | None = None,
        scope_chain: ScopeChain | None = None,
    ) -> dict | None:
        _, norm = normalize_node_path(path)
        if scope_key is not None:
            norm_scope = normalize_scope_key(scope_key)
            row = self.store.conn.execute(
                """SELECT m.*, f.plain, f.status, f.confidence, f.source_turn_ids, f.event_time_start, f.ingest_time
                   FROM fact_unified_meta m
                   JOIN facts f ON f.id = m.fact_id
                   WHERE m.user_id=? AND m.agent_id=? AND m.scope_key=? AND m.vfs_path=? AND f.status='active'
                   ORDER BY m.version DESC, m.updated_at DESC LIMIT 1""",
                (user_id, agent_id, norm_scope, norm),
            ).fetchone()
            return self._decode_row(row) if row is not None else None

        rows = self.store.conn.execute(
            """SELECT m.*, f.plain, f.status, f.confidence, f.source_turn_ids, f.event_time_start, f.ingest_time
               FROM fact_unified_meta m
               JOIN facts f ON f.id = m.fact_id
               WHERE m.user_id=? AND m.agent_id=? AND m.vfs_path=? AND f.status='active'
               ORDER BY m.version DESC, m.updated_at DESC""",
            (user_id, agent_id, norm),
        ).fetchall()
        decoded = [self._decode_row(r) for r in rows]
        if not decoded:
            return None
        if scope_chain is None:
            return decoded[0]
        scoped = [r for r in decoded if scope_chain.contains(r.get("scope_key"))]
        if not scoped:
            return None
        scoped.sort(key=lambda r: (scope_chain.rank(r.get("scope_key")), -int(r["version"]), -float(r["updated_at"])))
        return scoped[0]

    def list_path_facts(
        self,
        *,
        user_id: str = "default",
        agent_id: str = "default",
        path_prefix: str | None = None,
        kinds: tuple[str, ...] | None = None,
        always_active_only: bool = False,
        scope_chain: ScopeChain | None = None,
        include_stale: bool = True,
    ) -> list[dict]:
        clauses = ["m.user_id=?", "m.agent_id=?", "m.vfs_path IS NOT NULL", "f.status='active'"]
        params: list = [user_id, agent_id]
        if path_prefix:
            pref = normalize_dir_path(path_prefix)
            clauses.append("m.vfs_path LIKE ?")
            params.append(f"{pref}%")
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            clauses.append(f"m.kind IN ({placeholders})")
            params.extend(kinds)
        if always_active_only:
            clauses.append("m.always_active=1")
        if not include_stale:
            clauses.append("m.staleness_status != 'stale'")
        rows = self.store.conn.execute(
            f"""SELECT m.*, f.plain, f.status, f.confidence, f.source_turn_ids, f.event_time_start, f.ingest_time
                FROM fact_unified_meta m
                JOIN facts f ON f.id = m.fact_id
                WHERE {' AND '.join(clauses)}
                ORDER BY m.vfs_path""",
            tuple(params),
        ).fetchall()
        decoded = [self._decode_row(r) for r in rows]
        if scope_chain is None:
            return decoded
        # Filter to scope_chain and let highest-precedence scope win per vfs_path
        winners: dict[str, tuple[int, dict]] = {}
        for row in decoded:
            rank = scope_chain.rank(row.get("scope_key"))
            if rank is None:
                continue
            vpath = row["vfs_path"]
            prev = winners.get(vpath)
            if prev is None or rank < prev[0] or (rank == prev[0] and row["updated_at"] > prev[1]["updated_at"]):
                winners[vpath] = (rank, {**row, "scope_rank": rank})
        return [item for _, item in sorted(winners.values(), key=lambda pair: (pair[1]["vfs_path"], pair[0]))]

    def list_scope_facts(
        self,
        *,
        user_id: str = "default",
        agent_id: str = "default",
        scope_chain: ScopeChain | None = None,
        kinds: tuple[str, ...] | None = None,
        include_stale: bool = True,
        limit: int = 200,
    ) -> list[dict]:
        clauses = ["m.user_id=?", "m.agent_id=?", "f.status='active'"]
        params: list = [user_id, agent_id]
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            clauses.append(f"m.kind IN ({placeholders})")
            params.extend(kinds)
        if not include_stale:
            clauses.append("m.staleness_status != 'stale'")
        if scope_chain is not None:
            placeholders = ",".join("?" for _ in scope_chain.scopes)
            clauses.append(f"m.scope_key IN ({placeholders})")
            params.extend(scope_chain.scopes)
        params.append(int(limit))
        rows = self.store.conn.execute(
            f"""SELECT m.*, f.plain, f.status, f.confidence, f.source_turn_ids, f.event_time_start, f.ingest_time
                FROM fact_unified_meta m
                JOIN facts f ON f.id = m.fact_id
                WHERE {' AND '.join(clauses)}
                ORDER BY m.updated_at DESC LIMIT ?""",
            tuple(params),
        ).fetchall()
        return [self._decode_row(r) for r in rows]

    def meta_map(self, fact_ids: list[str]) -> dict[str, dict]:
        ids = list(dict.fromkeys(fact_ids))
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self.store.conn.execute(
            f"""SELECT m.*, f.plain, f.status, f.confidence, f.source_turn_ids, f.event_time_start, f.ingest_time
                FROM fact_unified_meta m
                JOIN facts f ON f.id = m.fact_id
                WHERE m.fact_id IN ({placeholders})""",
            tuple(ids),
        ).fetchall()
        return {row["fact_id"]: self._decode_row(row) for row in rows}

    def list_snapshots(
        self,
        *,
        user_id: str = "default",
        agent_id: str = "default",
        vfs_path: str | None = None,
        batch_id: str | None = None,
    ) -> list[dict]:
        clauses = ["user_id=?", "agent_id=?"]
        params: list = [user_id, agent_id]
        if vfs_path:
            _, norm = normalize_node_path(vfs_path)
            clauses.append("vfs_path=?")
            params.append(norm)
        if batch_id:
            clauses.append("batch_id=?")
            params.append(batch_id)
        rows = self.store.conn.execute(
            f"SELECT * FROM fact_unified_snapshots WHERE {' AND '.join(clauses)} ORDER BY created_at DESC",
            tuple(params),
        ).fetchall()
        return [self._decode_row(r) for r in rows]

    def get_snapshot(self, snapshot_id: str, *, user_id: str = "default", agent_id: str = "default") -> dict | None:
        row = self.store.conn.execute(
            "SELECT * FROM fact_unified_snapshots WHERE snapshot_id=? AND user_id=? AND agent_id=?",
            (snapshot_id, user_id, agent_id),
        ).fetchone()
        return self._decode_row(row) if row is not None else None
