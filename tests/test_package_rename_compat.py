"""Tests for package rename to ``agentmem-core`` (``agentmem_core``) & zero-downtime ``memengine`` shim."""
from __future__ import annotations

import importlib
import subprocess
import sys
import warnings

import agentmem_core
from agentmem_core.env import getenv_compat


def test_agentmem_core_primary_exports_and_env_compat(monkeypatch):
    assert hasattr(agentmem_core, "UnifiedMemoryEngine")
    assert hasattr(agentmem_core, "CognitiveMemoryEngine")
    assert hasattr(agentmem_core, "VFSMemoryEngine")
    assert hasattr(agentmem_core, "UnifiedToolRegistry")

    monkeypatch.delenv("AGENTMEM_CORE_DB_URL", raising=False)
    monkeypatch.setenv("MEMENGINE_DB_URL", "sqlite:///legacy.db")
    assert getenv_compat("DB_URL") == "sqlite:///legacy.db"

    monkeypatch.setenv("AGENTMEM_CORE_DB_URL", "sqlite:///primary.db")
    assert getenv_compat("DB_URL") == "sqlite:///primary.db"


def test_legacy_memengine_shim_emits_deprecation_and_preserves_identity():
    # Remove cached shim entries to verify DeprecationWarning on first import
    for key in [k for k in sys.modules if k == "memengine" or k.startswith("memengine.")]:
        sys.modules.pop(key, None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        legacy = importlib.import_module("memengine")
        legacy_core = importlib.import_module("memengine.core")
        legacy_plan = importlib.import_module("memengine.core.plan")
        legacy_contrib = importlib.import_module("memengine.contrib")

    assert any(
        issubclass(w.category, DeprecationWarning) and "agentmem_core" in str(w.message)
        for w in caught
    )
    assert legacy.UnifiedMemoryEngine is agentmem_core.UnifiedMemoryEngine
    assert legacy_core.UnifiedMemoryEngine is agentmem_core.UnifiedMemoryEngine
    assert legacy_plan.RecallConfig is agentmem_core.RecallConfig
    assert legacy_contrib.QdrantL2Index is importlib.import_module("agentmem_core.contrib").QdrantL2Index


def test_cli_entrypoints_work_for_both_agentmem_core_and_legacy_memengine():
    for module in ("agentmem_core.cli", "memengine.cli"):
        proc = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert "agentmem-core" in proc.stdout
