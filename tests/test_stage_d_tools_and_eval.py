import json
from pathlib import Path
import sys

import pytest

from eval.evidence_ablation import Question, Turn, measure, measure_pair, run_cli
from eval.locomo.run import groups as locomo_groups
from eval.longmemeval.run import groups as longmem_groups
from eval.metrics import evidence_metrics, retrieval_report
from eval.paired import evaluate_pair, make_engine
from memengine.lifecycle import DAY


def test_all_five_tool_schemas_are_json_serializable_and_enforced():
    engine = make_engine("mock")
    tools = engine.as_tools("alice", "planner", "work")
    names = {"memory_search", "memory_drilldown", "memory_history", "memory_remember", "memory_forget"}
    assert {item["function"]["name"] for item in json.loads(json.dumps(tools))} == names
    schemas = {item["name"]: item["inputSchema"] for item in tools.mcp_definitions()}
    for name in names:
        function = next(item["function"] for item in tools if item["function"]["name"] == name)
        assert function["parameters"] == schemas[name]
        assert schemas[name]["additionalProperties"] is False
        with pytest.raises(ValueError, match="missing or unexpected"):
            tools.dispatch_tool(name, {})
    for name, args in (("memory_search", {"query": "Atlas", "top_k": True}),
                       ("memory_search", {"query": "Atlas", "time_start": float("nan")}),
                       ("memory_search", {"query": "Atlas", "time_start": 2, "time_end": 1}),
                       ("memory_search", {"query": "Atlas", "memory_types": ["fact", "fact"]}),
                       ("memory_drilldown", {"turn_ids": [0]}), ("memory_history", {"entity_or_memory_id": " "}),
                       ("memory_remember", {"content": "Atlas", "tool_call_id": "untrusted"}),
                       ("memory_forget", {"memory_id": "x", "hard": "true"})):
        with pytest.raises(ValueError):
            tools.dispatch_tool(name, args)


def test_tools_follow_evidence_versions_and_never_cross_scope():
    engine = make_engine("mock")
    owner = {"user_id": "alice", "agent_id": "planner", "session_id": "work"}
    other = {**owner, "session_id": "private"}
    first = engine.add("设计评审的预算为5000元。", timestamp=10, **owner)
    hidden = engine.add("设计评审的预算为9000元。", timestamp=11, **other)
    last = engine.add("设计评审的预算改为8000元。", timestamp=15, **owner)
    tools = engine.as_tools(**owner)
    dispatch = tools.dispatch_tool
    root = first["operations"][0]["line_id"]
    latest = last["operations"][0]["line_id"]
    before = engine.store.conn.total_changes
    history = dispatch("memory_history", json.dumps({"entity_or_memory_id": latest}))
    assert history == dispatch("memory_history", {"entity_or_memory_id": "事件:设计评审"})
    assert [(v["id"], v["status"], v["event_time"]) for v in history] == [(root, "superseded", 10), (latest, "active", 15)]
    assert history[1]["supersedes_id"] == root and history[0]["valid_until"] == 15
    selected = dispatch("memory_search", {"query": "设计评审预算", "memory_types": ["fact"],
                                          "time_start": "1970-01-01T00:00:10Z", "time_end": 15})
    assert [row["id"] for row in selected] == [root]
    assert set(selected[0]) >= {"id", "memory_type", "status", "event_time", "score", "source_turn_ids"}
    raw = dispatch("memory_drilldown", {"turn_ids": [first["turn_id"], last["turn_id"], hidden["turn_id"]], "window": 2})
    assert [item["turn_id"] for item in raw] == [first["turn_id"], last["turn_id"]]
    assert engine.store.conn.total_changes == before

    with pytest.raises(ValueError, match="source_turn_id"):
        dispatch("memory_remember", {"content": "An invented rule"})
    with pytest.raises(ValueError, match="scope"):
        dispatch("memory_remember", {"content": "An invented rule", "source_turn_id": hidden["turn_id"]})
    written = dispatch("memory_remember", {"content": "When Atlas deploys: verify the release manifest",
                                             "memory_type": "procedural", "source_turn_id": last["turn_id"]})
    fact = engine.store.facts_by_ids([written["id"]], scope=("alice", "planner", "work"))[0]
    assert fact["trigger_text"] == "Atlas deploys" and fact["source_turn_ids"] == [last["turn_id"]]
    assert dispatch("memory_search", {"query": "Atlas release manifest", "memory_types": ["procedural"]})[0]["id"] == written["id"]
    wrong_id = hidden["operations"][0]["line_id"]
    assert not dispatch("memory_forget", {"memory_id": wrong_id, "hard": True})["deleted"]
    assert dispatch("memory_forget", {"memory_id": written["id"], "reason": "obsolete"})["deleted"]
    assert dispatch("memory_history", {"entity_or_memory_id": written["id"]})[0]["status"] == "deleted"
    assert dispatch("memory_forget", {"memory_id": written["id"], "hard": True})["deleted"]
    assert dispatch("memory_history", {"entity_or_memory_id": written["id"]}) == []


def test_dispatch_tool_accepts_trusted_host_context_and_searches_cold_and_observations():
    engine = make_engine("mock")
    engine.set_time(0)
    engine.record_tool_call("42:tool", trigger="Atlas deploy", technique="Validate the manifest", success=True)
    rule = engine.dispatch_tool("memory_remember", {"content": "Validate the manifest", "memory_type": "procedural"},
                                tool_call_id="42:tool")
    assert engine.store.get_line(rule["id"])["tool_call_ids"] == ["42:tool"]
    with pytest.raises(ValueError, match="scope"):
        engine.dispatch_tool("memory_remember", {"content": "Validate the manifest"}, tool_call_id="42:tool", user_id="bob")
    obs = engine.store.insert_observation("Frost experiments confirm leaf damage below freezing", timestamp=0, confidence=0.65)
    engine.decay_and_archive(timestamp=120 * DAY)
    args = {"query": "Frost experiments confirm leaf damage below freezing", "memory_types": ["observation"]}
    assert engine.dispatch_tool("memory_search", args) == []
    cold = engine.dispatch_tool("memory_search", {**args, "include_cold": True})
    assert cold[0]["id"] == obs


def test_report_denominators_token_components_and_prefix_comparisons():
    assert evidence_metrics([{"none"}, {"a"}, {"a", "x"}, {"b"}], {"a", "b", "c"}) == {
        "recall_at_5": 2 / 3, "recall_at_10": 2 / 3, "mrr_at_10": 0.5}
    rows = [{"unanswerable": neg, "full": {"empty": empty, "latency_ms": latency, "prefix_hit": hit,
                                          "injected_tokens": 10, "prefix_tokens": 2, "dynamic_tokens": 5, "drilled_raw_tokens": 3}}
            for neg, empty, latency, hit in ((True, True, 10, None), (True, False, 20, True),
                                             (False, True, 30, False), (False, False, 40, None))]
    report = retrieval_report(rows, "full")
    assert report["abstention_rate"] == report["empty_recall_precision"] == report["false_empty_rate"] == 0.5
    assert report["prefix_hit_rate"] == 0.5 and report["prefix_comparisons"] == 2
    assert report["p50_retrieval_ms"] == 25 and report["p95_retrieval_ms"] == pytest.approx(38.5)
    assert report["avg_injected_tokens"] == sum(report[f"avg_{kind}_tokens"] for kind in ("prefix", "dynamic", "drilled_raw"))
    assert retrieval_report(rows[2:], "full")["abstention_rate"] is None


def test_pair_maps_raw_ids_and_never_counts_neighbor_as_retrieved_fact():
    engine = make_engine("mock")
    primary = engine.add("[step 99] Atlas assembled ceramic axles.", timestamp=1)
    neighbor = engine.store.append_turn("Atlas design approval still pending.", 2.0)
    raw_ids = {primary["turn_id"]: Turn("D1:1", "session-1", "user", "axle", 1),
               neighbor: Turn("D1:2", "session-1", "user", "approval", 2)}
    engine.store.set_core_snapshot("Atlas profile", [])
    pair = evaluate_pair(engine, "What axles did Atlas assemble?", 3)
    question = Question("q", "temporal", "ignored", 3, frozenset({"D1:1", "D1:2"}))
    scores = measure_pair(engine, question, pair, raw_ids, pair.context["prefix_version"])
    assert scores["bipolar"]["recall_at_10"] == scores["rrf"]["recall_at_10"] == 0.5
    assert scores["bipolar"]["drilled_evidence_recall"] == 0.5
    assert scores["bipolar"]["drilled_turn_refs"] == ["D1:2"]
    assert scores["bipolar"]["prefix_hit"] and scores["rrf"]["prefix_hit"] is None
    assert scores["bipolar"]["injected_tokens"] <= 2048


def test_benchmark_adapters_keep_categories_and_fixture_cannot_claim_official_score(tmp_path, monkeypatch, capsys):
    locomo = [{"sample_id": "example", "conversation": {"session_1": [{"speaker": "Alex", "dia_id": "D1:1", "text": "Atlas"}]},
               "qa": [{"question": f"Question {category}?", "category": category, "evidence": ["D1:1"]} for category in range(1, 6)]}]
    turns, qs = next(locomo_groups(locomo, 5, 42))
    assert turns[0].ref == "D1:1" and {q.category[2:] for q in qs} == {"multi-hop", "temporal", "open-domain", "single-hop", "adversarial"}
    assert sum(q.unanswerable for q in qs) == 1
    fixture = Path(__file__).with_name("fixtures") / "longmemeval_synthetic.json"
    output = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", ["fixture", "--sample", "2", "--mode", "mock", "--data-dir", str(fixture), "--out", str(output)])
    run_cli("LongMemEval-S", (fixture.name,), "https://example.invalid", tmp_path, longmem_groups)
    report = json.loads(output.read_text())
    assert report["mode"] == "mock" and report["dataset_status"] == "unverified input, not an official score"
    assert "not an official score" in capsys.readouterr().out
    assert set(report["by_category"]) == {"knowledge-update", "abstention"}
    src_turns, q = next(group for group in longmem_groups(json.loads(fixture.read_text()), 2, 42)
                        if group[1][0].category == "knowledge-update")
    prior = {"id": "old", "text": "old", "status": "superseded", "status_weight": 0.35, "source_step_ids": [0]}
    new = {"id": "new", "text": "new", "status": "active", "source_step_ids": [1]}
    assert measure([prior, new], src_turns, q[0], {})["update_dominates"]
