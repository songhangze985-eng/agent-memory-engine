"""L0 Episode Log: gzipped JSONL cold store (zero vectorization) + turn-level FTS5 fallback."""
from __future__ import annotations

import hashlib
import tempfile
import time
from pathlib import Path
from typing import Iterator

from memengine.vfs.models import TranscriptRecord
from memengine.vfs.transcripts import LocalChunkTranscriptStore, TranscriptStore


def default_episode_dir(db_url: str | Path | None) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    raw = str(db_url or "")
    if raw and not raw.startswith(("postgresql://", "postgres://", ":memory:")):
        cleaned = raw
        for prefix in (
            "lancedb:///",
            "lance:///",
            "sqlite+lance:///",
            "lancedb://",
            "lance://",
            "sqlite+lance://",
            "sqlite:///",
        ):
            if cleaned.startswith(prefix):
                from urllib.parse import unquote

                cleaned = unquote(cleaned.removeprefix(prefix))
                break
        base = Path(cleaned).expanduser().resolve()
        target = (base / "l0_episodes") if base.suffix not in {".db", ".sqlite", ".sqlite3"} else (base.parent / f"{base.stem}_l0_episodes")
        return target, None
    tmp = tempfile.TemporaryDirectory(prefix="memengine_l0_ep_")
    return Path(tmp.name), tmp


class EpisodeLog:
    """Source-of-truth L0 episode log (never vectorized) with FTS5 lexical turn fallback."""

    def __init__(
        self,
        store,
        *,
        db_url: str | Path | None = None,
        transcript_dir: str | Path | None = None,
        transcript_store: TranscriptStore | None = None,
    ):
        self.store = store
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        if transcript_store is not None:
            self.cold_store = transcript_store
        else:
            if transcript_dir is None:
                target, self._temp_dir = default_episode_dir(db_url)
            else:
                target = Path(transcript_dir)
            self.cold_store = LocalChunkTranscriptStore(target)
        self._is_pg = type(store).__name__.startswith("Postgres")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.store.conn.execute(
            """CREATE TABLE IF NOT EXISTS ingest_idempotency (
                user_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                turn_id BIGINT NOT NULL,
                transcript_id TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                created_at DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (user_id, agent_id, idempotency_key)
            )"""
        )
        self.store.conn.execute(
            """CREATE TABLE IF NOT EXISTS episode_turn_map (
                turn_id BIGINT PRIMARY KEY,
                transcript_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                session_id TEXT NOT NULL
            )"""
        )
        self.store.conn.commit()

    def close(self) -> None:
        try:
            self.cold_store.flush()
        finally:
            if self._temp_dir is not None:
                self._temp_dir.cleanup()
                self._temp_dir = None

    def lookup_idempotency(self, *, user_id: str, agent_id: str, idempotency_key: str) -> dict | None:
        row = self.store.conn.execute(
            """SELECT turn_id, transcript_id, payload_hash, created_at
               FROM ingest_idempotency
               WHERE user_id=? AND agent_id=? AND idempotency_key=?""",
            (user_id, agent_id, idempotency_key),
        ).fetchone()
        return dict(row) if row is not None else None

    def record_idempotency(
        self,
        *,
        user_id: str,
        agent_id: str,
        idempotency_key: str,
        turn_id: int,
        transcript_id: str,
        text: str,
        timestamp: float,
    ) -> None:
        digest = hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:32]
        self.store.conn.execute(
            """INSERT INTO ingest_idempotency(user_id, agent_id, idempotency_key, turn_id, transcript_id, payload_hash, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (user_id, agent_id, idempotency_key) DO NOTHING""",
            (user_id, agent_id, idempotency_key, int(turn_id), transcript_id, digest, float(timestamp)),
        )

    def append_cold(
        self,
        *,
        turn_id: int,
        user_id: str,
        agent_id: str,
        session_id: str,
        role: str,
        text: str,
        timestamp: float | None = None,
        speaker: str | None = None,
        metadata: dict | None = None,
    ) -> TranscriptRecord:
        stamp = time.time() if timestamp is None else float(timestamp)
        meta = dict(metadata or {})
        meta["turn_id"] = int(turn_id)
        rec = self.cold_store.append(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            role=role,
            text=text,
            timestamp=stamp,
            speaker=speaker,
            metadata=meta,
        )
        self.store.conn.execute(
            """INSERT INTO episode_turn_map(turn_id, transcript_id, user_id, agent_id, session_id)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT (turn_id) DO UPDATE SET transcript_id=excluded.transcript_id""",
            (int(turn_id), rec.transcript_id, user_id, agent_id, session_id),
        )
        return rec

    def search_fts(
        self,
        query: str,
        *,
        user_id: str = "default",
        agent_id: str = "default",
        session_id: str | None = None,
        top_n: int = 30,
    ) -> list[dict]:
        """Zero-vector turn-level BM25 fallback over L0 raw turns."""
        return self.store.search_raw_turns_fts(
            query,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            top_n=top_n,
        )

    def indexed_turn_ids(self, *, user_id: str = "default", agent_id: str = "default", session_id: str | None = None) -> list[int]:
        if session_id is not None:
            rows = self.store.conn.execute(
                "SELECT turn_id FROM raw_turns WHERE user_id=? AND agent_id=? AND session_id=?",
                (user_id, agent_id, session_id),
            ).fetchall()
        else:
            rows = self.store.conn.execute(
                "SELECT turn_id FROM raw_turns WHERE user_id=? AND agent_id=?",
                (user_id, agent_id),
            ).fetchall()
        return [int(r["turn_id"]) for r in rows]

    def iter_episodes(
        self,
        *,
        user_id: str = "default",
        agent_id: str = "default",
        session_id: str | None = None,
        since_ts: float | None = None,
        until_ts: float | None = None,
    ) -> Iterator[TranscriptRecord]:
        return self.cold_store.iter_session(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            since_ts=since_ts,
            until_ts=until_ts,
        )

    def stats(self, *, user_id: str = "default", agent_id: str = "default") -> dict:
        cold = self.cold_store.stats(user_id=user_id, agent_id=agent_id) if hasattr(self.cold_store, "stats") else {}
        row = self.store.conn.execute(
            "SELECT count(*) AS cnt FROM raw_turns WHERE user_id=? AND agent_id=?",
            (user_id, agent_id),
        ).fetchone()
        return {
            **cold,
            "fts_indexed_turns": int(row["cnt"]) if row else 0,
            "l0_vectors": 0,  # Hard invariant: L0 episodes are NEVER vectorized.
        }
