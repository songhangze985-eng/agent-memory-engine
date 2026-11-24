from datetime import datetime, timezone

from agentmem_core import CognitiveMemoryEngine
from agentmem_core.textutil import estimate_tokens


def keep(engine, text, stamp=1, **kwargs):
    turn = engine.store.append_turn(text, stamp)
    return engine.remember(text, source_turn_ids=[turn], timestamp=stamp, **kwargs)


def test_scopes_and_smoothed_opinion():
    engine = CognitiveMemoryEngine()
    scopes = ["world", "experience", "opinion", "observation", "procedural"]
    memories = {scope: keep(engine, f"Atlas calibration {scope}", epistemic_scope=scope, confidence=.4)
                for scope in scopes}
    assert keep(engine, "Atlas calibration opinion", epistemic_scope="opinion", confidence=.8) == memories["opinion"]
    for scope in scopes:
        result = engine.retrieve("Atlas calibration", scopes=[scope], read_only=True)
        assert [hit["id"] for hit in result["ranked"]] == [memories[scope]]
    assert engine.store.get_line(memories["opinion"])["confidence"] == .5
    out = engine.build_context("Atlas calibration", scopes=["opinion", "experience"])
    assert "[Agent历史动作]" in out["dynamic_memory"] and "[观点 置信度 0.50]" in out["dynamic_memory"]
    engine.store.close()


def test_temporary_override_expires_without_superseding_and_can_be_recalled():
    engine = CognitiveMemoryEngine()
    day = 86400
    start = datetime(2025, 5, 1, tzinfo=timezone.utc).timestamp()
    fact = keep(engine, "我常喝美式咖啡。", start-day, zkeys=["实体:咖啡"])
    added = engine.add("拔牙后一周内禁饮咖啡。", timestamp=start)
    temporary = added["operations"][0]["line_id"]
    assert added["operations"][0]["action"] == "TEMPORARY_OVERRIDE"
    assert engine.store.get_line(fact)["status"] == "active"
    during = engine.retrieve("喝咖啡有什么限制？", timestamp=start+3*day, read_only=True)
    winner = during["ranked"][0]
    assert (winner["id"], winner["r_m"], winner["bipolar_weight"]) == (temporary, 1, 1)
    assert "临时生效至" in winner["state_tag"]
    after = engine.retrieve("喝咖啡有什么限制？", timestamp=start+10*day, read_only=True)
    assert after["ranked"][0]["id"] == fact and temporary not in {row["id"] for row in after["ranked"]}
    then = engine.retrieve("2025-05-04 当时喝咖啡有什么限制？", timestamp=start+10*day, read_only=True)
    assert temporary in {row["id"] for row in then["ranked"]}
    assert "禁饮" not in engine.build_context("喝咖啡有什么限制？", timestamp=start+10*day)["dynamic_memory"]
    assert engine.store.get_line(fact)["revision"] == 0
    engine.store.close()


def test_reconstruction_recovers_the_bridge_fact_with_shared_budget():
    engine = CognitiveMemoryEngine(per_channel_k=1)
    keep(engine, "Alice recommended AtlasToolkit to Bob.", zkeys=["实体:Alice", "实体:AtlasToolkit"])
    target = keep(engine, "AtlasToolkit – v2.4.1.", zkeys=["实体:AtlasToolkit"])
    question = "Which tool did Alice recommend to Bob and which version did it use?"
    single = engine.retrieve(question, read_only=True, token_budget=80)
    rebuilt = engine.retrieve(question, read_only=True, token_budget=80, mode="reconstructive")
    assert target not in {row["id"] for row in single["ranked"]}
    assert rebuilt["reconstructed"] and "实体:AtlasToolkit" in rebuilt["bridge_entities"]
    assert target in {row["id"] for row in rebuilt["ranked"]}
    assert estimate_tokens(rebuilt["context"]) <= 80
    assert engine.build_context(question, mode="reconstructive", token_budget=80)["token_breakdown"]["total"] <= 80
    engine.store.close()


def test_timed_advice_does_not_replace_a_user_preference():
    engine = CognitiveMemoryEngine()
    advice = "Plan a menu for the next 7 days. Avoid wasting food."
    engine.add(advice, role="assistant", timestamp=1)
    assert engine.store.snapshot()
    assert all(fact["expires_at"] is None for fact in engine.store.snapshot())
    engine.store.close()
