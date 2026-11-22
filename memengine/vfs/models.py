"""Core data models, path normalization, and invariants for the VFS memory architecture."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

LAYERS = frozenset({"L1", "L2", "L3"})
VERIFICATION_STATUSES = frozenset({"unverified", "verified"})
SALIENCE_TAGS = frozenset({"normal", "preference", "taboo", "core_rule", "warning"})

_INVALID_SEGMENT = re.compile(r"[\x00-\x1f\\]")


class OptimisticLockError(RuntimeError):
    """Raised when a concurrent writer mutated a memory node before this write committed."""

    def __init__(self, path: str, expected_version: int, actual_version: int):
        self.path = path
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            f"optimistic lock conflict on {path}: expected version {expected_version}, found {actual_version}"
        )


def normalize_dir_path(raw: str | None) -> str:
    """Normalize a folder path to canonical form like '/' or '/strategies/database/'."""
    text = (raw or "/").strip()
    if not text:
        return "/"
    if len(text) > 512 or _INVALID_SEGMENT.search(text):
        raise ValueError(f"invalid VFS directory path: {raw!r}")
    parts = [segment for segment in text.split("/") if segment]
    if any(segment in {".", ".."} for segment in parts):
        raise ValueError(f"path traversal is not allowed in VFS path: {raw!r}")
    return "/" if not parts else "/" + "/".join(parts) + "/"


def normalize_node_path(raw: str) -> tuple[str, str]:
    """Normalize a file node path like '/strategies/database/pg.md' -> (parent_path, full_path)."""
    text = (raw or "").strip()
    if not text or len(text) > 512 or _INVALID_SEGMENT.search(text):
        raise ValueError(f"invalid VFS node path: {raw!r}")
    parts = [segment for segment in text.split("/") if segment]
    if not parts or any(segment in {".", ".."} for segment in parts):
        raise ValueError(f"invalid VFS node path: {raw!r}")
    filename = parts[-1]
    if not filename.endswith(".md"):
        filename = f"{filename}.md"
    parent_parts = parts[:-1]
    parent_path = "/" if not parent_parts else "/" + "/".join(parent_parts) + "/"
    return parent_path, parent_path + filename


def validate_node_fields(
    *,
    layer: str,
    confidence_score: float,
    verification_status: str,
    salience_tag: str,
) -> None:
    if layer not in LAYERS:
        raise ValueError(f"layer must be one of {sorted(LAYERS)}, got {layer!r}")
    if not (0.0 <= float(confidence_score) <= 1.0):
        raise ValueError(f"confidence_score must be in [0, 1], got {confidence_score!r}")
    if verification_status not in VERIFICATION_STATUSES:
        raise ValueError(
            f"verification_status must be one of {sorted(VERIFICATION_STATUSES)}, got {verification_status!r}"
        )
    if salience_tag not in SALIENCE_TAGS:
        raise ValueError(f"salience_tag must be one of {sorted(SALIENCE_TAGS)}, got {salience_tag!r}")


@dataclass(frozen=True)
class TranscriptRecord:
    transcript_id: str
    user_id: str
    agent_id: str
    session_id: str
    role: str
    text: str
    timestamp: float
    speaker: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryNode:
    node_id: str
    user_id: str
    agent_id: str
    parent_path: str
    path: str
    title: str
    summary: str
    content: str
    layer: str
    confidence_score: float
    verification_status: str
    salience_tag: str
    source_transcript_ids: tuple[str, ...]
    version: int
    is_deleted: bool
    created_at: float
    updated_at: float

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "parent_path": self.parent_path,
            "path": self.path,
            "title": self.title,
            "summary": self.summary,
            "content": self.content,
            "layer": self.layer,
            "confidence_score": self.confidence_score,
            "verification_status": self.verification_status,
            "salience_tag": self.salience_tag,
            "source_transcript_ids": list(self.source_transcript_ids),
            "version": self.version,
            "is_deleted": self.is_deleted,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
