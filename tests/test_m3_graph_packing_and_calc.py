"""Tests for Milestone M3: entity-graph enumeration, coverage-first packing & deterministic arithmetic."""
from __future__ import annotations

from memengine.chrono import parse_session_clock
from memengine.core import UnifiedMemoryEngine


def test_m3_deterministic_sum_and_diff_across_sessions():
    engine = UnifiedMemoryEngine(":memory:")
    try:
        t1 = parse_session_clock("2023-03-01")
        t2 = parse_session_clock("2023-04-01")
        t3 = parse_session_clock("2023-05-01")
        engine.ingest_turn("My most popular YouTube video got 1,248 views this week!", timestamp=t1, session_id="s1")
        engine.ingest_turn("My most popular TikTok video reached 750 views yesterday!", timestamp=t2, session_id="s2")
        ans_sum = engine.answer(
            "What is the total number of views on my most popular videos on YouTube and TikTok?",
            reference_time=t3,
        )
        assert ans_sum["abstained"] is False
        assert "1,998" in ans_sum["response"]

        engine.ingest_turn("In Hawaii I spent $420 per night on hotel accommodations.", timestamp=t1 + 100, session_id="s3")
        engine.ingest_turn("In Tokyo I spent $150 per night on hotel accommodations.", timestamp=t2 + 100, session_id="s4")
        ans_diff = engine.answer(
            "How much more did I spend on accommodations per night in Hawaii compared to Tokyo?",
            reference_time=t3,
        )
        assert ans_diff["abstained"] is False
        assert "$270" in ans_diff["response"]
    finally:
        engine.close()


def test_m3_distinct_session_counting_and_multi_hop_synthesis():
    engine = UnifiedMemoryEngine(":memory:")
    try:
        base = parse_session_clock("2023-02-01")
        engine.ingest_turn("I visited my primary care doctor today for an annual checkup.", timestamp=base, session_id="s1")
        engine.ingest_turn("I visited an ENT specialist doctor about my sinus allergies.", timestamp=base + 10 * 86400, session_id="s2")
        engine.ingest_turn("I visited a dermatologist doctor for a quick skin check.", timestamp=base + 20 * 86400, session_id="s3")
        ans_cnt = engine.answer("How many different doctors did I visit?", reference_time=base + 30 * 86400)
        assert ans_cnt["abstained"] is False
        assert ans_cnt["response"].startswith("3")

        # Multi-hop distinct-session synthesis
        engine.ingest_turn("Melanie: My dog Oliver loves fetching sticks at the park.", timestamp=base + 100, session_id="p1", speaker="Melanie")
        engine.ingest_turn("Melanie: My cat Luna slept on my laptop all morning.", timestamp=base + 5 * 86400, session_id="p2", speaker="Melanie")
        ans_pets = engine.answer("What are Melanie's pets' names?", reference_time=base + 30 * 86400)
        assert ans_pets["abstained"] is False
        assert "Oliver" in ans_pets["response"] and "Luna" in ans_pets["response"]
    finally:
        engine.close()
