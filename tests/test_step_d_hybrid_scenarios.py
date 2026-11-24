"""Step D & SWE external benchmark tests: dual-profile mounting, RealMem hybrid eval, SWE-Bench-CL adapter."""
from __future__ import annotations

import json

from agentmem_core import UnifiedMemoryEngine
from eval.coding.external_benchmarks import evaluate_swe_context_stream, load_swebench_cl_jsonl
from eval.coding.run_coding_eval import run_coding_eval
from eval.hybrid.realmem_eval import run_realmem_hybrid_eval


def test_realmem_style_hybrid_project_eval_passes_all_checks():
    rep = run_realmem_hybrid_eval(seed=42)
    assert rep["gate_pass"] is True, f"Failed checks: {[c for c in rep['checks'] if not c['passed']]}"
    assert rep["passed"] == rep["total"] == 8


def test_step_c_coding_reuse_eval_passes_g2_gate():
    rep = run_coding_eval(seed=42)
    assert rep["gate_pass"] is True
    s = rep["summary"]
    assert s["with_memory_success_rate"] > s["no_memory_success_rate"]
    assert s["with_memory_avg_steps"] < s["no_memory_avg_steps"]
    assert s["with_memory_avg_tokens"] < s["no_memory_avg_tokens"]
    assert s["stale_rule_misleading_rate"] == 0.0
    assert s["false_injection_rate"] == 0.0
    assert s["ambient_llm_calls"] == 0


def test_single_agent_mounts_conversational_and_coding_profiles_without_interference(tmp_path):
    engine = UnifiedMemoryEngine(tmp_path / "dual_mount.db")
    try:
        conv = engine.bind_profile("conversational", user_id="u1", session_id="chat-1")
        code = engine.bind_profile("coding", user_id="u1", repo_id="billing", project_id="fin", session_id="code-1")

        # Write via conversational binding -> routes to user:u1
        r_conv = conv.ingest_turn("I live in Kyoto and prefer concise replies.", timestamp=1_700_000_000.0)
        assert r_conv["turn_id"] >= 1

        # Write via coding binding -> routes to repo:billing
        r_code = code.ingest_rule(
            "/billing/rounding.md",
            title="Banker's Rounding",
            trigger_condition="Computing invoice tax in billing",
            action="Use Decimal ROUND_HALF_EVEN in src/tax.py",
            verification_result="Passes accounting suite",
            anchors=["src/tax.py"],
            timestamp=1_700_000_100.0,
        )
        assert r_code["accepted"] is True
        assert r_code["meta"]["scope_key"] == "repo:billing"

        # Conversational retrieval sees Kyoto, does NOT leak repo:billing rule or directory outline
        ctx_conv = conv.build_context("Where do I live?", reference_time=1_700_000_200.0)
        assert "Kyoto" in ctx_conv["dynamic_memory"]
        assert "ROUND_HALF_EVEN" not in ctx_conv["dynamic_memory"]
        assert "Directory Outline" not in ctx_conv["dynamic_memory"]

        # Coding retrieval sees repo:billing rule AND inherits user:u1 preference!
        amb_code = code.ambient_context("invoice tax rounding", active_files=["src/tax.py"])
        assert amb_code["llm_calls"] == 0
        assert "ROUND_HALF_EVEN" in amb_code["dynamic_memory"]
        rec_pref = code.recall("concise replies", reference_time=1_700_000_200.0)
        assert any("concise replies" in f["plain"].lower() for f in rec_pref["ranked"])
    finally:
        engine.close()


def test_swebench_cl_and_swe_contextbench_stream_adapter(tmp_path):
    stream_path = tmp_path / "swe_cl.jsonl"
    rows = [
        {
            "instance_id": "repo__issue-101",
            "repo": "acme/core",
            "base_commit": "abc1234",
            "order_index": 1,
            "problem_statement": "Fix timeout bug in src/client.py retry loop",
            "gold_files": ["src/client.py"],
            "gold_fix_summary": "Set exponential backoff cap to 15s in src/client.py",
        },
        {
            "instance_id": "repo__issue-102",
            "repo": "acme/core",
            "base_commit": "def5678",
            "order_index": 2,
            "problem_statement": "Add jitter to retry loop in src/client.py",
            "gold_files": ["src/client.py"],
            "gold_fix_summary": "Add random uniform jitter inside retry loop",
        },
    ]
    stream_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    tasks = load_swebench_cl_jsonl(stream_path)
    assert len(tasks) == 2

    engine = UnifiedMemoryEngine(tmp_path / "swe.db", profile="coding")
    try:
        res = evaluate_swe_context_stream(tasks, engine=engine)
        assert res["tasks_processed"] == 2
        assert res["forward_transfer_hit_rate"] == 1.0
        assert res["avg_context_recall"] == 1.0
        assert res["avg_context_precision"] == 1.0
    finally:
        engine.close()
