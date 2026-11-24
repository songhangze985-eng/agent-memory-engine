"""Unified cognitive memory engine: L0+FTS5, single bi-temporal fact layer, derived VFS/Profile views & tool registry."""

from memengine.adapters import UnifiedToolRegistry
from memengine.core import FACT_KINDS, QueryPlan, RecallConfig, UnifiedMemoryEngine
from memengine.engine import CognitiveMemoryEngine
from memengine.vfs import OptimisticLockError, VFSMemoryEngine, open_vfs_store

__all__ = [
    "FACT_KINDS",
    "OptimisticLockError",
    "QueryPlan",
    "RecallConfig",
    "UnifiedMemoryEngine",
    "UnifiedToolRegistry",
    "CognitiveMemoryEngine",
    "VFSMemoryEngine",
    "open_vfs_store",
]
__version__ = "0.3.0"
