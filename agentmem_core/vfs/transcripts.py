"""Decoupled L0 raw transcript object storage (chunked JSONL.GZ, zero vectorization)."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Protocol

from agentmem_core.vfs.models import TranscriptRecord

_SAFE_KEY = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _safe_segment(value: str | None) -> str:
    text = str(value) if value is not None and str(value).strip() else "default"
    if _SAFE_KEY.match(text) and text not in {".", ".."}:
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    return f"h_{digest}"


def _day_bucket(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), timezone.utc).strftime("%Y-%m-%d")


class TranscriptStore(Protocol):
    """L0 raw log storage protocol — strictly append/chunk-oriented, never vectorized."""

    def append(
        self,
        *,
        user_id: str,
        agent_id: str,
        session_id: str,
        role: str,
        text: str,
        timestamp: float | None = None,
        speaker: str | None = None,
        metadata: dict | None = None,
    ) -> TranscriptRecord: ...

    def flush(self) -> None: ...

    def get_by_ids(self, transcript_ids: list[str], *, user_id: str, agent_id: str) -> list[TranscriptRecord]: ...

    def iter_session(
        self,
        *,
        user_id: str,
        agent_id: str,
        session_id: str | None = None,
        since_ts: float | None = None,
        until_ts: float | None = None,
    ) -> Iterator[TranscriptRecord]: ...


class LocalChunkTranscriptStore:
    """S3/OSS-compatible chunked JSONL.GZ local object store for L0 raw transcripts.

    Object key layout:
        ``<root>/<user_seg>/<agent_seg>/<session_seg>/<YYYY-MM-DD>/chunk_<seq:06d>.jsonl.gz``

    A lightweight per-tenant index file maps ``transcript_id -> object_key`` so lazy provenance
    drilldowns only decompress the exact gzip chunk(s) referenced by an L2 Markdown note.
    """

    def __init__(self, root: str | Path, *, chunk_size: int = 64):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.chunk_size = max(1, int(chunk_size))
        self._lock = threading.RLock()
        self._buffers: dict[tuple[str, str, str, str], list[dict]] = {}
        self._index_cache: dict[tuple[str, str], dict[str, str]] = {}

    def _tenant_dir(self, user_id: str, agent_id: str) -> Path:
        path = (self.root / _safe_segment(user_id) / _safe_segment(agent_id)).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("tenant directory escapes transcript store root")
        return path

    def _index_path(self, user_id: str, agent_id: str) -> Path:
        return self._tenant_dir(user_id, agent_id) / "_manifest.jsonl"

    def _load_index(self, user_id: str, agent_id: str) -> dict[str, str]:
        key = (user_id, agent_id)
        if key in self._index_cache:
            return self._index_cache[key]
        mapping: dict[str, str] = {}
        manifest = self._index_path(user_id, agent_id)
        if manifest.exists():
            with manifest.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        entry = json.loads(line)
                        mapping[entry["id"]] = entry["obj"]
        self._index_cache[key] = mapping
        return mapping

    def _chunk_dir(self, user_id: str, agent_id: str, session_id: str, day: str) -> Path:
        path = (self._tenant_dir(user_id, agent_id) / _safe_segment(session_id) / day).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("chunk directory escapes transcript store root")
        return path

    def _flush_bucket(self, bucket: tuple[str, str, str, str]) -> str | None:
        rows = self._buffers.pop(bucket, None)
        if not rows:
            return None
        user_id, agent_id, session_id, day = bucket
        folder = self._chunk_dir(user_id, agent_id, session_id, day)
        folder.mkdir(parents=True, exist_ok=True)
        existing = sorted(folder.glob("chunk_*.jsonl.gz"))
        seq = len(existing) + 1
        while True:
            target = folder / f"chunk_{seq:06d}.jsonl.gz"
            if not target.exists():
                break
            seq += 1
        payload = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode("utf-8")
        with gzip.open(target, "wb", compresslevel=5) as gz:
            gz.write(payload)
        rel_key = target.relative_to(self.root).as_posix()
        manifest = self._index_path(user_id, agent_id)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        index = self._load_index(user_id, agent_id)
        with manifest.open("a", encoding="utf-8") as fh:
            for row in rows:
                index[row["transcript_id"]] = rel_key
                fh.write(json.dumps({"id": row["transcript_id"], "obj": rel_key}, ensure_ascii=False) + "\n")
        return rel_key

    def flush(self) -> None:
        with self._lock:
            for bucket in list(self._buffers):
                self._flush_bucket(bucket)

    def append(
        self,
        *,
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
        tid = f"tr_{int(stamp * 1000)}_{uuid.uuid4().hex[:10]}"
        record = TranscriptRecord(
            transcript_id=tid,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            role=role,
            text=text,
            timestamp=stamp,
            speaker=speaker,
            metadata=dict(metadata or {}),
        )
        bucket = (user_id, agent_id, session_id, _day_bucket(stamp))
        with self._lock:
            buf = self._buffers.setdefault(bucket, [])
            buf.append({
                "transcript_id": record.transcript_id,
                "user_id": record.user_id,
                "agent_id": record.agent_id,
                "session_id": record.session_id,
                "role": record.role,
                "text": record.text,
                "timestamp": record.timestamp,
                "speaker": record.speaker,
                "metadata": record.metadata,
            })
            if len(buf) >= self.chunk_size:
                self._flush_bucket(bucket)
        return record

    def _read_chunk(self, rel_key: str) -> list[TranscriptRecord]:
        target = (self.root / rel_key).resolve()
        if not target.is_relative_to(self.root) or not target.exists():
            return []
        records = []
        with gzip.open(target, "rb") as gz:
            for raw in gz.read().decode("utf-8").split("\n"):
                if not raw.strip():
                    continue
                item = json.loads(raw)
                records.append(TranscriptRecord(
                    transcript_id=item["transcript_id"],
                    user_id=item["user_id"],
                    agent_id=item["agent_id"],
                    session_id=item["session_id"],
                    role=item["role"],
                    text=item["text"],
                    timestamp=float(item["timestamp"]),
                    speaker=item.get("speaker"),
                    metadata=item.get("metadata") or {},
                ))
        return records

    def get_by_ids(self, transcript_ids: list[str], *, user_id: str, agent_id: str) -> list[TranscriptRecord]:
        if not transcript_ids:
            return []
        wanted = set(transcript_ids)
        found: dict[str, TranscriptRecord] = {}
        with self._lock:
            # Check unflushed in-memory buffers first
            for (u, a, _, _), rows in self._buffers.items():
                if u != user_id or a != agent_id:
                    continue
                for item in rows:
                    if item["transcript_id"] in wanted:
                        found[item["transcript_id"]] = TranscriptRecord(
                            transcript_id=item["transcript_id"],
                            user_id=item["user_id"],
                            agent_id=item["agent_id"],
                            session_id=item["session_id"],
                            role=item["role"],
                            text=item["text"],
                            timestamp=float(item["timestamp"]),
                            speaker=item.get("speaker"),
                            metadata=item.get("metadata") or {},
                        )
            remaining = wanted - found.keys()
            if remaining:
                index = self._load_index(user_id, agent_id)
                chunks = {index[tid] for tid in remaining if tid in index}
                for rel_key in sorted(chunks):
                    for rec in self._read_chunk(rel_key):
                        if rec.transcript_id in remaining and rec.user_id == user_id and rec.agent_id == agent_id:
                            found[rec.transcript_id] = rec
        return [found[tid] for tid in transcript_ids if tid in found]

    def iter_session(
        self,
        *,
        user_id: str,
        agent_id: str,
        session_id: str | None = None,
        since_ts: float | None = None,
        until_ts: float | None = None,
    ) -> Iterator[TranscriptRecord]:
        self.flush()
        base = self._tenant_dir(user_id, agent_id)
        if not base.exists():
            return
        if session_id is not None:
            search_root = base / _safe_segment(session_id)
            chunk_files = sorted(search_root.rglob("chunk_*.jsonl.gz")) if search_root.exists() else []
        else:
            chunk_files = sorted(base.rglob("chunk_*.jsonl.gz"))
        for file_path in chunk_files:
            rel_key = file_path.relative_to(self.root).as_posix()
            for rec in self._read_chunk(rel_key):
                if rec.user_id != user_id or rec.agent_id != agent_id:
                    continue
                if session_id is not None and rec.session_id != session_id:
                    continue
                if since_ts is not None and rec.timestamp < since_ts:
                    continue
                if until_ts is not None and rec.timestamp >= until_ts:
                    continue
                yield rec

    def stats(self, *, user_id: str, agent_id: str) -> dict:
        self.flush()
        base = self._tenant_dir(user_id, agent_id)
        chunks = list(base.rglob("chunk_*.jsonl.gz")) if base.exists() else []
        total_bytes = sum(p.stat().st_size for p in chunks)
        return {
            "backend": "local_chunk_object_store",
            "chunk_files": len(chunks),
            "compressed_bytes": total_bytes,
            "indexed_transcripts": len(self._load_index(user_id, agent_id)),
        }


from agentmem_core.contrib.s3_transcripts import S3TranscriptStore  # noqa: F401 (re-exported for backwards compatibility)
