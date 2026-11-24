"""Optional external S3/OSS L0 transcript adapter (non-core contrib)."""
from __future__ import annotations

import gzip
import json
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Iterator

from memengine.vfs.models import TranscriptRecord

_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_segment(value: str) -> str:
    return _SAFE_SEGMENT.sub("_", (value or "default").strip()) or "default"


def _day_bucket(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


class S3TranscriptStore:
    """S3 / Aliyun OSS / MinIO adapter sharing the chunked JSONL.GZ key layout.

    Moved out of core into ``memengine.contrib``.
    """

    def __init__(self, bucket: str, *, prefix: str = "transcripts", client=None, chunk_size: int = 64):
        if client is None:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError("S3TranscriptStore requires boto3 or an injected S3-compatible client") from exc
            client = boto3.client("s3")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = client
        self.chunk_size = max(1, int(chunk_size))
        self._buffers: dict[tuple[str, str, str, str], list[dict]] = {}
        self._id_to_key: dict[tuple[str, str, str], str] = {}

    def _object_key(self, user_id: str, agent_id: str, session_id: str, day: str, chunk_id: str) -> str:
        parts = [self.prefix, _safe_segment(user_id), _safe_segment(agent_id), _safe_segment(session_id), day, f"chunk_{chunk_id}.jsonl.gz"]
        return "/".join(part for part in parts if part)

    def _flush_bucket(self, bucket_key: tuple[str, str, str, str]) -> str | None:
        rows = self._buffers.pop(bucket_key, None)
        if not rows:
            return None
        user_id, agent_id, session_id, day = bucket_key
        key = self._object_key(user_id, agent_id, session_id, day, uuid.uuid4().hex[:12])
        payload = gzip.compress(("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode("utf-8"))
        self.client.put_object(Bucket=self.bucket, Key=key, Body=payload, ContentType="application/gzip")
        for row in rows:
            self._id_to_key[(user_id, agent_id, row["transcript_id"])] = key
        return key

    def flush(self) -> None:
        for bucket_key in list(self._buffers):
            self._flush_bucket(bucket_key)

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
        record = TranscriptRecord(tid, user_id, agent_id, session_id, role, text, stamp, speaker, dict(metadata or {}))
        bucket_key = (user_id, agent_id, session_id, _day_bucket(stamp))
        buf = self._buffers.setdefault(bucket_key, [])
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
            self._flush_bucket(bucket_key)
        return record

    def _fetch_chunk(self, key: str) -> list[TranscriptRecord]:
        body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        records = []
        for raw in gzip.decompress(body).decode("utf-8").split("\n"):
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
        self.flush()
        wanted = set(transcript_ids)
        found: dict[str, TranscriptRecord] = {}
        keys = {self._id_to_key[(user_id, agent_id, tid)] for tid in wanted if (user_id, agent_id, tid) in self._id_to_key}
        for key in sorted(keys):
            for rec in self._fetch_chunk(key):
                if rec.transcript_id in wanted and rec.user_id == user_id and rec.agent_id == agent_id:
                    found[rec.transcript_id] = rec
        missing = wanted - found.keys()
        if missing:
            for rec in self.iter_session(user_id=user_id, agent_id=agent_id):
                if rec.transcript_id in missing:
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
        parts = [self.prefix, _safe_segment(user_id), _safe_segment(agent_id)]
        if session_id is not None:
            parts.append(_safe_segment(session_id))
        prefix = "/".join(part for part in parts if part) + "/"
        token = None
        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            resp = self.client.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".jsonl.gz"):
                    continue
                for rec in self._fetch_chunk(key):
                    if rec.user_id != user_id or rec.agent_id != agent_id:
                        continue
                    if session_id is not None and rec.session_id != session_id:
                        continue
                    if since_ts is not None and rec.timestamp < since_ts:
                        continue
                    if until_ts is not None and rec.timestamp >= until_ts:
                        continue
                    yield rec
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
