from datetime import datetime, timezone
import json

import pytest

from memengine import CognitiveMemoryEngine
from memengine.lifecycle import DAY
from memengine.reranker import CalibratedLocalReranker
from memengine.store import Store
from memengine.textutil import estimate_tokens
from memengine.vector import HashedTfidfEmbedder


def engine(**kwargs):
    return CognitiveMemoryEngine(vector=kwargs.pop("vector", HashedTfidfEmbedder()),
                                 reranker=CalibratedLocalReranker(), **kwargs)


class LayoutSpy:
    def __init__(self):
        self.prompts = []
        self.operations = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return json.dumps({"write_operations": self.operations})


def test_zero_llm_hash_and_semantic_dedup_preserve_numeric_and_state_changes():
    class SimilarVectors(HashedTfidfEmbedder):
        def encode(self, texts):
            return super().encode(["same test embedding"] * len(texts))

    writer = LayoutSpy()
    memory = engine(llm=writer, vector=SimilarVectors())

    def operation(text, target=""):
        return {"action": "OVERWRITE_SAME_LINE" if target else "APPEND_ADJACENT_LINE",
                "formatted_line": text, "target_line_id": target, "entity_z_keys": ["实体:Atlas"], "attribute": "budget"}

    original = "Atlas release budget is 1000 yuan"
    writer.operations = [operation(original)]
    first = memory.add(original, timestamp=1)
    root = first["operations"][0]["line_id"]
    exact = memory.add(" ATLAS RELEASE BUDGET IS 1000 YUAN. ", timestamp=2)
    near = memory.add("The Atlas release budget is 1000 yuan.", timestamp=3)
    assert [x["operations"][0]["dedup_level"] for x in (exact, near)] == ["hash", "semantic"]
    assert len(writer.prompts) == 1
    row = memory.store.get_line(root)
    assert row["source_turn_ids"] == [first["turn_id"], exact["turn_id"], near["turn_id"]]
    assert row["mention_count"] == 3 and row["hit_count"] == 0

    for instant, updated in ((4, "Atlas release budget is 2400 yuan"), (5, "Atlas release budget is not 2400 yuan")):
        writer.operations = [operation(updated, memory.store.get_line(root)["id"])]
        assert memory.add(updated, timestamp=instant)["operations"][0]["action"] == "OVERWRITE_SAME_LINE"
    assert len(writer.prompts) == 3
    chain = memory.store.get_fact_chain(root)
    assert [fact["status"] for fact in chain] == ["superseded", "superseded", "active"]
    assert chain[-1]["source_turn_ids"] == [1, 2, 3, 4, 5]


def test_session_flush_batches_after_reopen_resolves_time_and_keeps_raw_sources(tmp_path):
    path = tmp_path / "session.db"
    writer = LayoutSpy()
    memory = engine(db_path=path, llm=writer, working_window=2, clock=lambda: 1736121600)
    scope = {"user_id": "alice", "agent_id": "assistant", "session_id": "one"}
    first_id = memory.buffer_turn("Atlas discussed launch", **scope)
    first = memory.working_memory(**scope)[0]
    assert first["turn_id"] == first_id and first["extraction_state"] == "buffered"
    second = memory.add("It shipped yesterday.", trigger="session_end", **scope)
    sibling = memory.add("Different session payload.", trigger="session_end", **{**scope, "session_id": "two"})
    assert not writer.prompts and not memory.store.snapshot(scope=("alice", "assistant", None))
    memory.store.close()

    memory = engine(db_path=path, llm=writer, clock=lambda: 1736121600)
    writer.operations = [{"action": "APPEND_ADJACENT_LINE", "formatted_line": "Atlas shipped yesterday.",
                          "attribute": "release", "source_turn_ids": [first_id, second["turn_id"]]}]
    result = memory.flush_session(**scope)
    assert result["turn_ids"] == [first_id, second["turn_id"]] and len(writer.prompts) == 1
    assert "It shipped yesterday." in writer.prompts[0] and "Different session payload." not in writer.prompts[0]
    fact = memory.store.get_line(result["operations"][0]["line_id"], scope=("alice", "assistant", "one"))
    assert "2025-01-05" in fact["plain"] and fact["event_time_start"] == 1736035200
    assert fact["source_turn_ids"] == [first_id, second["turn_id"]]
    assert memory.working_memory(**scope)[0]["extraction_state"] == "done"
    assert memory.flush_session(**scope) == {"turn_ids": [], "operations": []}
    assert memory.working_memory(**{**scope, "session_id": "two"})[0]["turn_id"] == sibling["turn_id"]
    assert memory.working_memory(**scope)[-1]["text"] == "It shipped yesterday."


def test_offline_session_flush_resolves_one_unambiguous_antecedent():
    memory = engine()
    memory.add("张敏将前往京都参加设计评审。", timestamp=1, trigger="session_end")
    turn = memory.add("她对花粉过敏。", timestamp=2, trigger="session_end")
    assert not memory.has_facts()
    assert len(memory.flush_session("default")["operations"]) == 2
    allergy = next(row for row in memory.store.snapshot() if row["attribute"] == "过敏")
    assert "张敏对花粉过敏" in allergy["plain"] and allergy["source_turn_ids"] == [turn["turn_id"]]
    assert memory.working_memory()[-1]["text"] == "她对花粉过敏。"


def test_distillation_provenance_success_count_and_stable_profile_fingerprint():
    memory = engine(clock=lambda: 100)
    assert memory.consolidate()["skipped"]
    turns = [memory.add(f"Atlas recorded build item {index}.", timestamp=index) for index in range(8)]
    fact_id = turns[0]["operations"][0]["line_id"]
    for call_id in ("release:1", "release:2"):
        memory.record_tool_call(call_id, trigger="Atlas release", technique="Verify the build manifest", success=True,
                                source_turn_ids=[turns[0]["turn_id"]], derived_from_ids=[fact_id])
    condensed = memory.consolidate()
    assert condensed["new_facts"] == 8 and condensed["observation_ids"] and len(condensed["procedural_ids"]) == 1
    obs = memory.store.list_observations()[0]
    assert obs["memory_type"] == "observation" and len(obs["derived_from_ids"]) >= 2 and len(obs["source_turn_ids"]) >= 2
    rule = memory.store.get_line(condensed["procedural_ids"][0])
    assert rule["memory_type"] == "procedural" and rule["derived_from_ids"] == [fact_id]
    assert rule["source_turn_ids"] == [turns[0]["turn_id"]] and rule["success_count"] == 2
    assert rule["tool_call_ids"] == ["release:1", "release:2"]
    core = memory.store.get_core_snapshot()
    ctx = memory.build_context("How do I perform the Atlas release?")
    assert "Verify the build manifest" in ctx["system_prefix"]
    assert ctx["prefix_version"] == core["prefix_version"] and estimate_tokens(ctx["system_prefix"]) <= 1000
    assert memory.consolidate()["skipped"]
    assert memory.consolidate(force=True, timestamp=101)["prefix_version"] == core["prefix_version"]
    assert memory.store.get_core_snapshot()["updated_at"] == 100
    memory.add("老周对花粉过敏。", timestamp=102)
    revised = memory.consolidate(force=True, timestamp=103)
    assert revised["prefix_version"] != core["prefix_version"]
    assert "老周对花粉过敏" in memory.store.get_core_snapshot()["core_profile"]
    assert memory.forget(rule["id"], timestamp=104)
    assert "Verify the build manifest" not in memory.build_context("How do I perform the Atlas release?")["system_prefix"]


def test_archive_protects_constraints_procedures_and_high_confidence_observations():
    now = datetime(2025, 1, 15, tzinfo=timezone.utc).timestamp()
    memory = engine(clock=lambda: now)
    start = datetime(2024, 8, 15, tzinfo=timezone.utc).timestamp()
    allergy = memory.add("老周对花粉过敏。", timestamp=start)["operations"][0]["line_id"]
    old = memory.add("设计评审的预算为5000元。", timestamp=start)["operations"][0]["line_id"]
    memory.add("设计评审的预算改为8000元。", timestamp=now)
    fragment = memory.add("园艺种植简报介绍东北寒冷地区防霜养护技术。", timestamp=now)
    frag = fragment["operations"][0]["line_id"]
    memory.add("园艺种植简报介绍东北寒冷地区防霜养护技术。", timestamp=now)
    recent = memory.add("Atlas assembly uses ceramic rollers.", timestamp=now)
    recent_id = recent["operations"][0]["line_id"]
    memory.recall("What rollers does the Atlas assembly use?", timestamp=now, mutate_stats=True)
    rule = memory.remember("Use a certified calibration fixture", memory_type="procedural", trigger="calibration",
                           source_turn_ids=[recent["turn_id"]], timestamp=start)
    observation = memory.store.insert_observation("Recurring process observation", confidence=0.9,
                                                  source_turn_ids=[recent["turn_id"]], timestamp=start)
    aged = now + 120 * DAY
    archived = memory.decay_and_archive(timestamp=aged)["archived_ids"]
    assert old in archived and frag in archived and not {allergy, rule, observation, recent_id}.intersection(archived)
    protected = memory.store.facts_by_ids([allergy, rule, observation, recent_id])
    assert {fact["tier"] for fact in protected} == {"hot"}
    assert memory.store.get_line(allergy)["is_constraint"] == 1
    assert memory.retrieve("老周对什么过敏？", timestamp=aged, read_only=True)["ranked"][0]["decay_factor"] == 1
    assert memory.retrieve("What fixture should I use for calibration?", timestamp=aged, read_only=True)["ranked"][0]["decay_factor"] == 1
    warmed = memory.retrieve("What rollers does the Atlas assembly use?", timestamp=aged, read_only=True)
    assert next(fact for fact in warmed["ranked"] if fact["id"] == recent_id)["decay_factor"] == pytest.approx(1 / 16)

    assert not memory.store.search_bm25("防霜养护技术")
    assert memory.store.search_bm25("防霜养护技术", include_cold=True)[0]["id"] == frag
    assert frag not in [fact["id"] for fact in memory.store.search_vector("防霜养护技术")]
    assert frag not in [fact["id"] for fact in memory.retrieve("防霜养护技术有哪些？", timestamp=aged, read_only=True)["ranked"]]
    cold = memory.retrieve("防霜养护技术有哪些？", timestamp=aged, read_only=True, include_cold=True)
    assert any(fact["id"] == frag and fact["tier"] == "cold" for fact in cold["ranked"])
    history = memory.retrieve("设计评审2024年的预算是多少？", timestamp=aged, read_only=True)
    assert history["ranked"][0]["id"] == old and "5000" in history["ranked"][0]["plain"]
    assert history["ranked"][0]["decay_factor"] == 1.0


def test_remember_requires_registered_provenance_and_forget_is_scoped():
    memory = engine()
    with pytest.raises(ValueError, match="requires real"):
        memory.remember("invented rule")
    with pytest.raises(ValueError, match="registered tool"):
        memory.remember("invented rule", tool_call_id="unknown")
    memory.record_tool_call("tool-real", trigger="atlas", technique="Verify the build manifest", success=True, user_id="alice")
    row = memory.remember("Verify the build manifest", memory_type="procedural", tool_call_id="tool-real", user_id="alice")
    assert memory.store.get_line(row, scope=("alice", "default", None))["tool_call_ids"] == ["tool-real"]
    assert not memory.forget(row, hard=True)
    assert memory.forget(query="When atlas: Verify the build manifest", hard=True, user_id="alice")
    assert not memory.store.get_line(row, scope=("alice", "default", None), include_deleted=True)


def test_legacy_core_and_observations_migrate_idempotently(tmp_path):
    path = tmp_path / "old.db"
    store = Store(path)
    with store.transaction():
        store.set_meta('core_snapshot:["default", "default"]', json.dumps({"core_profile": "Legacy profile", "procedural_rules": ["Keep rule"]}))
        store.conn.execute("INSERT INTO observations VALUES('o23','default','default','Legacy note',.85,'[]','[]','active',1,1)")
        store.conn.execute("DELETE FROM meta WHERE key='lifecycle_sources_migrated'")
    store.close()
    for _ in range(2):
        store = Store(path)
        assert store.get_core_snapshot()["core_profile"] == "Legacy profile"
        assert [fact["id"] for fact in store.list_observations()] == ["o23"]
        store.close()
