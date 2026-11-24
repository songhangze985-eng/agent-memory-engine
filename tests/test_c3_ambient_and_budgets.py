"""C3 & G4 tests: zero-LLM ambient planner, millisecond latency, anti-false-injection, and question <=2 LLM cap."""
from __future__ import annotations

import pytest

from agentmem_core import BudgetExceededError, CodeAnchor, UnifiedMemoryEngine
from agentmem_core.core.ambient import enforce_llm_budget


class _PoisonLLM:
    """Raises immediately if anything attempts an LLM call during the ambient path."""

    def complete(self, prompt: str, **kwargs):
        raise AssertionError("Ambient path MUST NEVER invoke LLM!")


class _CountingLLM:
    def __init__(self):
        self.calls = 0

    def complete(self, prompt: str, **kwargs):
        self.calls += 1
        return f"mock reply #{self.calls}"


def test_ambient_planner_is_strictly_zero_llm_and_millisecond_fast(tmp_path):
    engine = UnifiedMemoryEngine(tmp_path / "c3.db", llm=_PoisonLLM())
    try:
        engine.ingest_rule(
            "/ops/openblas.md",
            title="OpenBLAS Thread Cap",
            trigger_condition="Running numpy or pytest on macOS",
            action="Set OPENBLAS_NUM_THREADS=1 to avoid thread contention hang",
            verification_result="Tests finish in 3s",
            user_id="dev",
            repo_id="ml-core",
            anchors=[CodeAnchor(file_path="tests/conftest.py")],
        )
        engine.ingest_rule(
            "/ops/pg_port.md",
            title="Local Postgres Port",
            trigger_condition="Connecting to docker postgres",
            action="Use port 5433 instead of 5432",
            verification_result="Connected cleanly",
            user_id="dev",
            repo_id="ml-core",
            anchors=[CodeAnchor(file_path="docker-compose.yml")],
        )

        # Anchor + error hit with PoisonLLM attached -> 0 LLM calls and fast!
        res = engine.ambient_context(
            "pytest hang on macOS",
            active_files=["tests/conftest.py"],
            error_text="OPENBLAS thread contention",
            user_id="dev",
            repo_id="ml-core",
            profile="coding",
        )
        assert res["path"] == "ambient"
        assert res["llm_calls"] == 0
        assert res["ambient_ms"] < 250.0
        assert not res["empty_recall"]
        assert "OPENBLAS_NUM_THREADS=1" in res["dynamic_memory"]
        assert len(res["anchor_hits"]) == 1

        # Unrelated query -> anti-false-injection floor prevents noisy memory pollution
        unrelated = engine.ambient_context(
            "How do I center a CSS flexbox div in Safari?",
            active_files=["frontend/styles.css"],
            user_id="dev",
            repo_id="ml-core",
            profile="coding",
        )
        assert unrelated["llm_calls"] == 0
        assert unrelated["empty_recall"] is True
        assert unrelated["items"] == []
    finally:
        engine.close()


def test_g4_llm_budget_guard_enforces_zero_ambient_and_max_two_question_calls(tmp_path):
    llm = _CountingLLM()
    engine = UnifiedMemoryEngine(tmp_path / "g4.db", llm=llm)
    try:
        # 0-call ambient guard rejects even the 1st LLM call
        with pytest.raises(BudgetExceededError):
            with enforce_llm_budget(engine, max_calls=0, path_name="ambient"):
                engine._engine.llm.complete("forbidden ambient call")

        # 2-call question guard allows planner (1) + reader (2), rejects 3rd call
        with enforce_llm_budget(engine, max_calls=2, path_name="question") as guard:
            engine._engine.llm.complete("planner")
            engine._engine.llm.complete("reader")
            assert guard.calls == 2
            with pytest.raises(BudgetExceededError):
                engine._engine.llm.complete("illegal 3rd call")

        # Real engine.answer() respects <=2 LLM budget and reports call count
        engine.ingest_turn("Project Zephyr uses SQLite WAL mode.", user_id="u1", timestamp=1_700_000_000.0)
        ans = engine.answer("What storage mode does Project Zephyr use?", user_id="u1", reference_time=1_700_000_100.0)
        assert ans["llm_calls"] <= ans["llm_budget"] <= 2
    finally:
        engine.close()
