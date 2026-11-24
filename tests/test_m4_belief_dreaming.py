"""Tests for Milestone M4: evidence-backed belief dreaming (deterministic + live LLM contract + L0 rebuild)."""
from __future__ import annotations

import json

from memengine.chrono import parse_session_clock
from memengine.core import UnifiedMemoryEngine
from memengine.core.dreaming import synthesize_beliefs_with_llm


class _FakeLLM:
    def complete(self, prompt: str) -> str:
        return json.dumps({
            "beliefs": [
                {
                    "text": "User prefers cool and refreshing cold dishes and drinks.",
                    "domain": "preference",
                    "confidence": 0.9,
                    "derived_from_ids": ["l1", "l2"],
                },
                {
                    "text": "Hallucinated uncited belief that must be rejected.",
                    "domain": "theme",
                    "confidence": 0.9,
                    "derived_from_ids": ["nonexistent_id"],
                },
            ]
        })


def test_m4_deterministic_belief_dreaming_reflective_and_open_domain():
    engine = UnifiedMemoryEngine(":memory:")
    try:
        t1 = parse_session_clock("2023-10-05")
        t2 = parse_session_clock("2023-10-12")
        engine.ingest_turn(
            "I'm really into iced coffee; there's something about its refreshing kick that keeps me coming back.",
            timestamp=t1,
            session_id="s1",
            speaker="user",
        )
        engine.ingest_turn(
            "I'm a big fan of Cold Soba along with Iced Coffee; both are perfect for a refreshing meal!",
            timestamp=t2,
            session_id="s2",
            speaker="user",
        )
        engine.ingest_turn(
            "Sam: I was travelling through San Francisco and Yosemite in October 2023.",
            timestamp=t2 + 100,
            session_id="s3",
            speaker="Sam",
        )

        cons = engine.consolidate(min_new_facts=1, force=True, timestamp=t2 + 500)
        assert cons["belief_count"] >= 2
        assert cons["belief_mode"] == "deterministic"

        # Every distilled belief has kind='belief' and non-empty source_turn_ids
        metas = engine.ledger.meta_map(cons["belief_ids"])
        assert all(m["kind"] == "belief" for m in metas.values())
        assert all(m["source_turn_ids"] for m in metas.values())

        # Reflective preference query retrieves the belief with inferred "Cool and Refreshing" cue
        ctx_food = engine.build_context("According to the dishes I mentioned, which flavor might I prefer?", reference_time=t2 + 1000)
        assert "cool" in ctx_food["dynamic_memory"].lower()

        # Open-domain spatial inference answers US state ("California") grounded in San Francisco/Yosemite
        ans_geo = engine.answer("Which US state was Sam travelling in during October 2023?", reference_time=t2 + 1000)
        assert ans_geo["abstained"] is False
        assert "california" in ans_geo["response"].lower()
    finally:
        engine.close()


def test_m4_llm_belief_synthesizer_enforces_strict_citations():
    facts = [
        {"id": "l1", "plain": "I love iced coffee.", "source_turn_ids": [1], "mention_time": 1_700_000_000.0},
        {"id": "l2", "plain": "Cold Soba is refreshing.", "source_turn_ids": [2], "mention_time": 1_700_010_000.0},
    ]
    proposals = synthesize_beliefs_with_llm(_FakeLLM(), facts)
    # Only the validly cited belief passes; the hallucinated uncited belief is rejected!
    assert len(proposals) == 1
    assert proposals[0]["derived_from_ids"] == ["l1", "l2"]
    assert proposals[0]["source_turn_ids"] == [1, 2]
