"""Unified cognitive memory engine: L0+FTS5, single bi-temporal fact layer, derived VFS/Profile views & tool registry."""

from agentmem_core.adapters import UnifiedToolRegistry
from agentmem_core.core import FACT_KINDS, QueryPlan, RecallConfig, UnifiedMemoryEngine
from agentmem_core.engine import CognitiveMemoryEngine
from agentmem_core.vfs import OptimisticLockError, VFSMemoryEngine, open_vfs_store

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
