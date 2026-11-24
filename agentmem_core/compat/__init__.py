"""Backwards-compatibility shims (deprecated)."""
from agentmem_core.compat.v1_shim import (
    V1_DEPRECATION_MSG,
    V1_SUNSET_DATE,
    V2_VFS_DEPRECATION_MSG,
    UnifiedV1Shim,
    UnifiedVFSShim,
    attach_v1_deprecation_middleware,
)

__all__ = [
    "V1_DEPRECATION_MSG",
    "V1_SUNSET_DATE",
    "V2_VFS_DEPRECATION_MSG",
    "UnifiedV1Shim",
    "UnifiedVFSShim",
    "attach_v1_deprecation_middleware",
]
