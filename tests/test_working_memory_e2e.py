from contextlib import closing
import json

import pytest

from memengine.engine import CognitiveMemoryEngine
from memengine.llm import MockLLMClient


OWNER = {"user_id": "alice", "agent_id": "atlas", "session_id": "session"}
KEY = (OWNER["user_id"], OWNER["agent_id"], OWNER["session_id"])


class ProposedClaim(MockLLMClient):
    def __init__(self, claim, evidence_ids):
        self.claim = claim
        self.evidence_ids = evidence_ids

    def complete(self, prompt):
        if "慢系统综合推理与作答器" not in prompt.split("# Input", 1)[0]:
            return super().complete(prompt)
        return json.dumps({"final_answer": self.claim, "evidence_synthesis": "", "anti_intuitive_override": False,
                           "rejected_noise_ids": [],
                           "claims": [{"text": self.claim, "evidence_ids": self.evidence_ids}]})


def test_buffer_answer_and_restart_without_fact_write(tmp_path):
    path = tmp_path / "workspace.db"
    query = "What is the Atlas task checkpoint?"
    engine = CognitiveMemoryEngine(path)
    with closing(engine.store):
        engine.store.set_core_snapshot("Atlas prefers short reports.", [], user_id=OWNER["user_id"],
                                       agent_id=OWNER["agent_id"])
        before = engine.build_context(query, **OWNER)
        turn = engine.buffer_turn("Atlas task checkpoint is BLUE-482.", **OWNER)
        task = engine.start_task("Find Atlas checkpoint", **OWNER)
        engine.checkpoint_task(task, {"phase": 2, "turn": turn}, **OWNER)
        bundle = engine.build_context(query, task_id=task, **OWNER)
        assert bundle["system_prefix"] == before["system_prefix"] and "Atlas prefers short reports" in bundle["system_prefix"]
        assert bundle["prefix_version"] == before["prefix_version"]
        assert "BLUE-482" in bundle["dynamic_memory"]
        assert bundle["dynamic_memory"].index("Active Task Working Memory") < bundle["dynamic_memory"].index("Recent Session Buffer")
        assert bundle["evidence_map"][f"turn:{turn}"]["session_id"] == OWNER["session_id"]
        answer = engine.answer(query, task_id=task, **OWNER)
        assert "BLUE-482" in answer["response"] and not answer["abstained"]
        assert answer["claims"][0]["evidence_ids"] == [f"turn:{turn}"]
        assert not engine.store.snapshot(scope=KEY)
    engine = CognitiveMemoryEngine(path)
    with closing(engine.store):
        assert engine.get_task_workspace(task, **OWNER)["status"] == "active"
        assert engine.resume_task(task, **OWNER) == {"phase": 2, "turn": turn}
        assert [row["turn_id"] for row in engine.get_recent_turns(**OWNER)] == [turn]
        assert "BLUE-482" in engine.answer(query, task_id=task, **OWNER)["response"]
        assert engine.answer(query, **{**OWNER, "session_id": "other"})["abstained"]
        assert not engine.store.snapshot(scope=KEY)


def test_answer_rejects_other_versions_even_when_recent_turns_are_available():
    engine = CognitiveMemoryEngine(clock=lambda: 1735689600)
    query = "Which cache does Python v3.12.0 support?"
    with closing(engine.store):
        engine.retain("Python v3.11.4 supports the Elm cache.", **OWNER)
        engine.retain("Python v3.12.0 supports the Oak cache.", **{**OWNER, "user_id": "other"})
        response = engine.answer(query, token_budget=2048, **OWNER)
        assert response["gate_stats"]["empty_reason"] == "no_exact_version"
        assert response["abstained"] and not response["claims"]


def test_buffered_version_can_answer_with_only_its_supporting_clause():
    engine = CognitiveMemoryEngine()
    query = "Which cache does Python v3.12.0 support?"
    with closing(engine.store):
        turn = engine.buffer_turn("Python v3.11.4 supports the Elm cache; "
                                  "Python v3.12.0 supports the Oak cache.", **OWNER)
        response = engine.answer(query, **OWNER)
        assert not response["abstained"]
        assert "Oak cache" in response["response"] and "Elm cache" not in response["response"]
        assert response["claims"][0]["evidence_ids"] == [f"turn:{turn}"]


def test_workspace_evidence_must_match_the_requested_version():
    engine = CognitiveMemoryEngine()
    with closing(engine.store):
        content = "Python v3.11.4 supports the Elm cache."
        turn = engine.buffer_turn(content, **OWNER)
        task = engine.start_task("Verify the runtime cache", **OWNER)
        item = engine.record_provisional_claim(task, content, [turn], **OWNER)
        engine.verify_task_item(task, item, "user_confirmed", turn, **OWNER)
        options = {"task_id": task, "include_short_term": False, **OWNER}
        missing = engine.answer("Which cache does Python v3.12.0 support?", **options)
        assert missing["abstained"] and not missing["claims"]
        present = engine.answer("Which cache does Python v3.11.4 support?", **options)
        assert not present["abstained"]
        assert present["claims"][0]["evidence_ids"] == [f"item:{item}"]


def test_reflection_stages_unsupported_claim_and_gate_promotes_verified_source():
    claim = "Atlas uses Python v3.11.4."
    engine = CognitiveMemoryEngine(llm=ProposedClaim(claim, ["turn:1"]))
    with closing(engine.store):
        wrong = engine.buffer_turn("Atlas uses Python v3.11.2.", **OWNER)
        task = engine.start_task("Verify Atlas runtime", **OWNER)
        reflection = engine.reflect("Which Python version does Atlas use?", task_id=task, **OWNER)
        assert reflection["abstained"] and not reflection["claims"]
        item, = reflection["provisional_item_ids"]
        workspace = engine.get_task_workspace(task, **OWNER)
        assert workspace["items"][0]["verification_status"] == "unverified"
        assert not engine.recall("Python v3.11.4", user_id="alice", agent_id="atlas")["ranked"]
        with pytest.raises(ValueError, match="supporting"):
            engine.verify_task_item(task, item, "user_confirmed", wrong, **OWNER)
        with pytest.raises(ValueError, match="only verified"):
            engine.commit_task(task, promote_verified_only=False, **OWNER)
        verified = engine.buffer_turn(claim, **OWNER)
        with pytest.raises(ValueError, match="source is outside"):
            sibling = engine.start_task("Separate scope", **{**OWNER, "session_id": "sibling"})
            engine.record_provisional_claim(sibling, claim, [verified], **{**OWNER, "session_id": "sibling"})
        engine.verify_task_item(task, item, "user_confirmed", verified, **OWNER)
        engine.record_task_evidence(task, "Atlas uses Python v11.0.0.", "temp_fact", **OWNER)
        fact = engine.commit_task(task, **OWNER)[item]
        assert engine.commit_task(task, **OWNER) == {item: fact}
        assert [row["id"] for row in engine.recall("Atlas Python v3.11.4", user_id="alice", agent_id="atlas")["ranked"]] == [fact]
        assert engine.store.get_line(fact, scope=KEY)["source_turn_ids"] == [verified]
        assert [row["extraction_state"] for row in engine.get_recent_turns(**OWNER)] == ["buffered", "done"]


def test_only_successful_tool_evidence_can_verify_a_claim():
    engine = CognitiveMemoryEngine()
    with closing(engine.store):
        task = engine.start_task("Verify deployment", **OWNER)
        content = "Atlas deployment completed on 2024-05-05."
        turn = engine.buffer_turn(content, role="tool", **OWNER)
        item = engine.record_tool_call("deploy", trigger="deployment check", technique="read deployment report",
                                       success=True, source_turn_ids=[turn], task_id=task, evidence=content, **OWNER)
        engine.verify_task_item(task, item, "tool_verified", "tool_call:deploy", **OWNER)
        fact = engine.commit_task(task, **OWNER)[item]
        assert engine.store.get_line(fact, scope=KEY)["tool_call_ids"] == ["deploy"]


def test_repeated_answers_and_default_recall_leave_weights_unchanged():
    engine = CognitiveMemoryEngine()
    with closing(engine.store):
        row = engine.retain("Atlas checkpoint is BLUE-482.", **OWNER)
        ident = row["operations"][0]["line_id"]
        before = (engine.store.get_line(ident, scope=KEY)["hit_count"], engine._control_state(KEY))
        for _ in range(20):
            assert "BLUE-482" in engine.answer("What is the Atlas checkpoint?", **OWNER)["response"]
            assert engine.recall("What is the Atlas checkpoint?", **OWNER)["ranked"]
        assert (engine.store.get_line(ident, scope=KEY)["hit_count"], engine._control_state(KEY)) == before


def test_expired_and_foreign_evidence_cannot_validate_a_citation():
    claim = "Atlas budget is 100 euros."
    engine = CognitiveMemoryEngine(llm=ProposedClaim(claim, ["fact:l1", "turn:1"]), clock=lambda: 100)
    with closing(engine.store):
        engine.retain(claim, timestamp=10, ttl_seconds=5, **OWNER)
        engine.buffer_turn("Atlas uses a separate accounting system.", timestamp=30, **OWNER)
        engine.retain(claim, **{**OWNER, "user_id": "other"})
        query = "What is the Atlas budget?"
        answer = engine.answer(query, **OWNER)
        assert answer["abstained"] and not answer["claims"]
        assert "100 euros" not in answer["dynamic_memory"]
        assert engine.reflect("What is the Zebra warranty?", **OWNER)["abstained"]


def test_context_sql_has_one_session_read_and_at_most_two_task_reads():
    engine = CognitiveMemoryEngine()
    with closing(engine.store):
        engine.buffer_turn("Atlas checkpoint is BLUE-482.", **OWNER)
        task = engine.start_task("Check Atlas", **OWNER)
        for index in range(40):
            engine.record_task_evidence(task, f"Check pending telemetry {index}.", "open_question", **OWNER)
        for task_id in (None, task):
            queries = []
            engine.store.conn.set_trace_callback(queries.append)
            try:
                engine.build_context("What is the Atlas checkpoint?", task_id=task_id, **OWNER)
            finally:
                engine.store.conn.set_trace_callback(None)
            assert sum("WITH recent AS (" in query for query in queries) == 1
            assert sum("FROM working_" in query for query in queries) == (0 if task_id is None else 2)


def test_explicit_retain_moves_a_buffered_turn_without_copying_it():
    engine = CognitiveMemoryEngine()
    with closing(engine.store):
        turn = engine.buffer_turn("Atlas checkpoint is BLUE-482.", **OWNER)
        result = engine.retain(buffered_turn_id=turn, **OWNER)
        assert result["turn_id"] == turn and result["operations"]
        assert not engine.retain(buffered_turn_id=turn, **OWNER)["operations"]
        row, = engine.get_recent_turns(**OWNER)
        assert row["turn_id"] == turn and row["extraction_state"] == "done"
        assert engine.store.get_line(result["operations"][0]["line_id"], scope=KEY)["source_turn_ids"] == [turn]


def test_expired_verifier_is_downgraded_and_commit_rolls_back():
    text = "Atlas budget is 100 euros."
    clock = [11]
    engine = CognitiveMemoryEngine(llm=ProposedClaim(text, []), clock=lambda: clock[0])
    with closing(engine.store):
        task = engine.start_task("Verify the Atlas budget", **OWNER)
        safe = "Atlas accounting report exists."
        turn = engine.buffer_turn(safe, **OWNER)
        safe_item = engine.record_provisional_claim(task, safe, [turn], **OWNER)
        engine.verify_task_item(task, safe_item, "user_confirmed", turn, **OWNER)
        clock[0] = 12
        retained = engine.retain(text, ttl_seconds=5, timestamp=10, **OWNER)["operations"][0]["line_id"]
        item = engine.record_tool_call("budget", trigger="check budget", technique="read accounts", success=True,
                                       derived_from_ids=[retained], task_id=task, evidence=text, **OWNER)
        engine.verify_task_item(task, item, "tool_verified", "tool_call:budget", **OWNER)
        assert engine.get_task_workspace(task, **OWNER)["items"][-1]["support_current"]
        clock[0] = 16
        bundle = engine.build_context("What is the Atlas budget?", task_id=task, **OWNER)
        assert bundle["evidence_map"][f"item:{item}"]["verification_status"] == "unverified"
        with pytest.raises(ValueError, match="supporting"):
            engine.commit_task(task, **OWNER)
        facts = engine.store.snapshot(scope=KEY)
        assert [row["id"] for row in facts] == [retained]
        assert engine.get_task_workspace(task, **OWNER)["status"] == "active"


def test_promotion_keeps_its_verifying_facts_expiration():
    text = "Atlas budget is 100 euros."
    clock = [12]
    engine = CognitiveMemoryEngine(clock=lambda: clock[0])
    with closing(engine.store):
        retained = engine.retain(text, ttl_seconds=5, timestamp=10, **OWNER)["operations"][0]["line_id"]
        task = engine.start_task("Verify the Atlas budget", **OWNER)
        item = engine.record_tool_call("budget", trigger="check budget", technique="read accounts", success=True,
                                       derived_from_ids=[retained], task_id=task, evidence=text, **OWNER)
        engine.verify_task_item(task, item, "tool_verified", "tool_call:budget", **OWNER)
        promoted = engine.commit_task(task, **OWNER)[item]
        assert engine.store.get_line(promoted, scope=KEY)["expires_at"] == 15
        clock[0] = 16
        assert engine.answer("What is the Atlas budget?", **OWNER)["abstained"]
