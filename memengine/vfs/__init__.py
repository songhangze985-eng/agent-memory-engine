"""Virtual Filesystem (VFS) Memory Engine v2 — 3-stage hybrid retrieval, salience gating & progressive disclosure."""
from memengine.vfs.engine import VFSMemoryEngine
from memengine.vfs.index import L2VectorIndex, LanceL2Index, QdrantL2Index
from memengine.vfs.models import (
    LAYERS,
    MAX_INDEX_SUMMARY_CHARS,
    SALIENCE_TAGS,
    SALIENCE_WRITE_GUIDE,
    VERIFICATION_STATUSES,
    MemoryNode,
    OptimisticLockError,
    TranscriptRecord,
    enforce_rule_state_machine,
    evaluate_salience_gate,
    format_structured_rule,
    make_index_summary,
    parse_structured_rule,
)
from memengine.vfs.store import PostgresVFSStore, SQLiteVFSStore, W_UNVERIFIED, W_VERIFIED, open_vfs_store
from memengine.vfs.tools import VFSTools
from memengine.vfs.transcripts import LocalChunkTranscriptStore, S3TranscriptStore, TranscriptStore

__all__ = [
    "LAYERS",
    "MAX_INDEX_SUMMARY_CHARS",
    "SALIENCE_TAGS",
    "SALIENCE_WRITE_GUIDE",
    "VERIFICATION_STATUSES",
    "W_VERIFIED",
    "W_UNVERIFIED",
    "MemoryNode",
    "OptimisticLockError",
    "TranscriptRecord",
    "evaluate_salience_gate",
    "format_structured_rule",
    "parse_structured_rule",
    "enforce_rule_state_machine",
    "make_index_summary",
    "TranscriptStore",
    "LocalChunkTranscriptStore",
    "S3TranscriptStore",
    "L2VectorIndex",
    "LanceL2Index",
    "QdrantL2Index",
    "SQLiteVFSStore",
    "PostgresVFSStore",
    "open_vfs_store",
    "VFSMemoryEngine",
    "VFSTools",
]
