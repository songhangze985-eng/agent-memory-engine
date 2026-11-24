"""Environment variable helper supporting ``AGENTMEM_CORE_*`` with legacy ``MEMENGINE_*`` fallback."""
from __future__ import annotations

import os


def getenv_compat(suffix: str, default: str | None = None) -> str | None:
    """Read ``AGENTMEM_CORE_<suffix>`` first, falling back to deprecated ``MEMENGINE_<suffix>``."""
    val = os.getenv(f"AGENTMEM_CORE_{suffix}")
    if val is not None and val != "":
        return val
    legacy = os.getenv(f"MEMENGINE_{suffix}")
    if legacy is not None and legacy != "":
        return legacy
    return default
