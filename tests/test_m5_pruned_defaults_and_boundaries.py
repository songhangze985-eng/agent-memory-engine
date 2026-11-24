"""Tests for Milestone M5: ablation-pruned defaults, core/contrib boundary & package rename plan."""
from __future__ import annotations

import ast
from pathlib import Path

from agentmem_core.core import RecallConfig, UnifiedMemoryEngine


def test_m5_pruned_defaults_disable_bipolar_and_recency_by_default():
    cfg = RecallConfig()
    assert cfg.enable_bipolar is False
    assert cfg.enable_recency_rerank is False
    assert cfg.enable_episode_fts is True
    assert cfg.enable_entity_graph is True
    assert cfg.enable_slot_supersede is True
    assert cfg.enable_graph_enum is True
    assert cfg.enable_deterministic_calc is True

    engine = UnifiedMemoryEngine(":memory:")
    try:
        engine.ingest_turn("Alice uses PostgreSQL 16 for production workloads.", timestamp=1_700_000_000.0)
        res = engine.recall("What database does Alice use in production?", reference_time=1_700_000_100.0)
        assert res["empty_recall"] is False
        assert res["gate_stats"]["bipolar_enabled"] is False
        assert res["query_plan"].allow_recency_tiebreak is False
    finally:
        engine.close()


def test_core_packages_have_zero_contrib_dependencies():
    repo_root = Path(__file__).resolve().parents[1]
    core_dirs = [
        repo_root / "agentmem_core" / "core",
        repo_root / "agentmem_core" / "views",
        repo_root / "agentmem_core" / "adapters",
        repo_root / "agentmem_core" / "storage",
    ]
    forbidden = {"agentmem_core.contrib", "qdrant_client", "boto3"}
    offenders: list[str] = []
    for directory in core_dirs:
        for path in sorted(directory.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if any(alias.name == f or alias.name.startswith(f + ".") for f in forbidden):
                            offenders.append(f"{path.name}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    if any(node.module == f or node.module.startswith(f + ".") for f in forbidden):
                        offenders.append(f"{path.name}: from {node.module}")
    assert offenders == [], f"Forbidden contrib/non-core imports in core: {offenders}"


def test_rename_plan_document_exists():
    repo_root = Path(__file__).resolve().parents[1]
    plan = (repo_root / "docs" / "RENAME_PLAN.md").read_text(encoding="utf-8")
    assert "chronomem" in plan
    assert "Zero Breaking Changes" in plan
