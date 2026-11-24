"""Tests for Milestone M2: slot-level supersede, as_of/time-window filtering, recency tie-breaker & temporal reasoning."""
from __future__ import annotations

from agentmem_core.chrono import parse_session_clock
from agentmem_core.core import RecallConfig, UnifiedMemoryEngine, build_query_plan
from agentmem_core.core.supersede import extract_state_slot, resolve_relative_occurrence


def test_slot_extraction_and_supersede_chain():
    s1 = extract_state_slot("I've actually watched 4 MCU films in the last 3 months.", speaker="user")
    s2 = extract_state_slot("I've watched 5 MCU films recently.", speaker="user")
    assert s1 is not None and s2 is not None
    assert s1.slot_key == s2.slot_key
    assert s1.slot_value == "4" and s2.slot_value == "5"

    engine = UnifiedMemoryEngine(":memory:")
    try:
        t1 = parse_session_clock("2023-03-01")
        t2 = parse_session_clock("2023-04-15")
        r1 = engine.ingest_turn("I've actually watched 4 MCU films so far.", timestamp=t1, session_id="s1")
        r2 = engine.ingest_turn("I've actually watched 5 MCU films so far.", timestamp=t2, session_id="s2")
        assert len(r2["superseded_fact_ids"]) >= 1

        # Current-state query excludes superseded old value and returns latest value
        ans = engine.answer("How many MCU films have I watched?", reference_time=parse_session_clock("2023-05-01"))
        assert ans["abstained"] is False
        assert "5" in ans["response"]

        # Change-direction query sees both old and new via supersede chain
        ans_dir = engine.answer("Did I increase or decrease the number of MCU films watched?", reference_time=parse_session_clock("2023-05-01"))
        assert "increased" in ans_dir["response"].lower()
    finally:
        engine.close()


def test_recency_disabled_on_historical_and_temporal_queries():
    cfg = RecallConfig(enable_recency_rerank=True)
    p_hist = build_query_plan("What gadget did I invest in before getting the Air Fryer?", reference_time=1_700_000_000.0, config=cfg)
    assert p_hist.is_historical is True
    assert p_hist.allow_recency_tiebreak is False

    p_temp = build_query_plan("Which event happened first, the festival or Spanish classes?", reference_time=1_700_000_000.0, config=cfg)
    assert p_temp.is_temporal is True
    assert p_temp.temporal_order_ask == "first"
    assert p_temp.allow_recency_tiebreak is False

    p_curr = build_query_plan("What is my favorite coffee shop?", reference_time=1_700_000_000.0, config=cfg)
    assert p_curr.is_historical is False
    assert p_curr.is_temporal is False
    assert p_curr.allow_recency_tiebreak is True


def test_relative_occurrence_temporal_order_and_days_ago():
    base = parse_session_clock("2023-05-27")
    assert abs(resolve_relative_occurrence("I attended a cultural festival yesterday.", base) - (base - 86400.0)) < 1.0
    assert abs(resolve_relative_occurrence("I have been taking Spanish classes for the past three months.", base) - (base - 90 * 86400.0)) < 1.0

    engine = UnifiedMemoryEngine(":memory:")
    try:
        engine.ingest_turn("I attended a cultural festival in my hometown yesterday.", timestamp=base, mention_time=base, session_id="s1")
        engine.ingest_turn("I have been taking Spanish classes for the past three months.", timestamp=base + 100, mention_time=base + 100, session_id="s2")
        ans = engine.answer(
            "Which event happened first, my attendance at a cultural festival or the start of my Spanish classes?",
            reference_time=base + 200,
        )
        assert ans["abstained"] is False
        assert "spanish classes" in ans["response"].lower()

        t_read = parse_session_clock("2023-03-20")
        t_q = parse_session_clock("2023-04-01")
        engine.ingest_turn("I finally got around to reading the March 15th issue of The New Yorker today.", timestamp=t_read, mention_time=t_read, session_id="s3")
        ans_days = engine.answer("How many days ago did I read the March 15th issue of The New Yorker?", reference_time=t_q)
        assert ans_days["abstained"] is False
        assert "12 days" in ans_days["response"]
    finally:
        engine.close()
