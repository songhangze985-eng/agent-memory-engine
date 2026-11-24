"""Unified cognitive memory engine: declarative profiles, scope chains, L0+FTS5, single fact layer & adapters."""

from agentmem_core.adapters import UnifiedToolRegistry
from agentmem_core.core import (
    FACT_KINDS,
    MemoryProfile,
    ProfileBinding,
    QueryPlan,
    RecallConfig,
    ScopeChain,
    UnifiedMemoryEngine,
    list_profiles,
    load_profile,
)
from agentmem_core.engine import CognitiveMemoryEngine
from agentmem_core.vfs import OptimisticLockError, VFSMemoryEngine, open_vfs_store

__all__ = [
    "FACT_KINDS",
    "MemoryProfile",
    "OptimisticLockError",
    "ProfileBinding",
    "QueryPlan",
    "RecallConfig",
    "ScopeChain",
    "UnifiedMemoryEngine",
    "UnifiedToolRegistry",
    "CognitiveMemoryEngine",
    "VFSMemoryEngine",
    "list_profiles",
    "load_profile",
    "open_vfs_store",
]
__version__ = "0.3.0"
