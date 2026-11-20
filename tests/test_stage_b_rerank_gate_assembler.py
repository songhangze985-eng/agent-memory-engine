import io
import json
from hashlib import sha256

import pytest

from memengine import CognitiveMemoryEngine
from memengine.distance import BELT, INNER, OUTER, bipolar_weight, compute_bipolar_radius
from memengine.gate import precision_gate
from memengine.reranker import CalibratedLocalReranker, OpenAICompatibleReranker
from memengine.textutil import estimate_tokens
from memengine.vector import HashedTfidfEmbedder


def engine(**kwargs):
    kwargs.setdefault("reranker", CalibratedLocalReranker())
    return CognitiveMemoryEngine(vector=HashedTfidfEmbedder(), **kwargs)


class FixedScores:
    is_calibrated = True

    def __init__(self, by_text=None):
        self.by_text = by_text or {}
        self.seen = []

    def score(self, query, candidates):
        self.seen.append([fact["id"] for fact in candidates])
        return [self.by_text.get(fact["plain"], 0.8) for fact in candidates]


def test_exact_poles_and_center():
    peak = compute_bipolar_radius({"source_turn_ids": list(range(8))}, 8)
    edge = compute_bipolar_radius({"source_turn_ids": [1], "isolated": True}, 8)
    assert (peak, edge) == (0.0, 1.0)
    assert bipolar_weight(peak) == bipolar_weight(edge) == 1.0
    assert bipolar_weight(compute_bipolar_radius({"source_turn_ids": [1, 2]}, 8)) == 0.10


def test_history_has_status_weight_and_explicit_chronology():
    texts = ["Alex lived in Rome", "Alex lived in Paris", "Alex lived in Berlin"]
    memory = engine(reranker=FixedScores(dict(zip(texts, (0.5, 0.7, 0.9)))), cliff_ratio=0)
    with memory.store.transaction():
        block = memory.store.create_block("Home", 100)
        args = {"attribute": "住址", "zkeys": ["实体:Alex"]}
        first = memory.store.insert_fact(block_id=block, formatted=texts[0], plain=texts[0],
                                         ingest_time=100, event_time_start=20, **args)
        second = memory.store.supersede_fact(first, formatted=texts[1], plain=texts[1],
                                             ingest_time=200, event_time_start=10, **args)
        third = memory.store.supersede_fact(second, formatted=texts[2], plain=texts[2],
                                            ingest_time=300, event_time_start=30, **args)
    regular = memory.retrieve("Where does Alex live now?", timestamp=500, read_only=True)
    assert regular["ranked"][0]["id"] == third
    old = next(fact for fact in regular["ranked"] if fact["id"] == first)
    assert old["state_tag"] == "[历史已变更]"
    assert old["final_score"] == pytest.approx(old["ce_score"] * 0.35)
    assert "[历史已变更]" not in memory.build_context("Where does Alex live now?", drilldown_top_n=0)["dynamic_memory"]
    historical = memory.retrieve("Where did Alex live earliest?", timestamp=500, read_only=True)
    assert historical["parsed_query"]["query_type"] == "TIME_TRAVEL"
    assert [fact["id"] for fact in historical["ranked"]] == [second, first, third]
    for old in historical["ranked"][:2]:
        assert old["state_tag"] == "[历史快照]" and old["status_weight"] == 1
        assert old["final_score"] == old["ce_score"]
    response = memory.answer("Where did Alex live earliest?", retrieved=historical)["response"]
    assert "Paris" in response and "Rome" not in response and "Berlin" not in response


def test_history_gate_keeps_the_entity_mention_needed_to_resolve_related_facts():
    memory = engine()
    for person, city, project, amounts in (("张敏", "京都", "设计评审", (5000, 8000, 3000)),
                                          ("李楠", "青岛", "供应商会谈", (2000, 4500, 1200))):
        memory.add(f"{person}将前往{city}参加{project}。")
        for amount in amounts:
            for index in range(12):
                memory.add(f"公共服务信息简报第{index}期介绍城市公共活动。")
            memory.add(f"{project}的预算改为{amount}元。")
    query = "在第二次修改预算之前，张敏这次出行的预算是多少？"
    result = memory.answer(query)
    assert any(fact["entity_anchor"] and "设计评审" in fact["plain"] for fact in result["ranked"])
    assert "8000" in result["response"] and "4500" not in result["response"]


def test_uncalibrated_gate_preserves_weak_relevance_and_read_only_answer():
    memory = engine(enable_casual_bypass=False)
    memory.add("老周对苯甲酸钠过敏，不能食用含该防腐剂的食品。")
    memory.add("The bicycle uses hydraulic disc brakes, with mineral oil in its system.")
    negative = "How should I reset a video codec?"
    before = memory.store.conn.total_changes
    result = memory.retrieve(negative, read_only=True)
    assert result["gate_stats"]["candidate_count"] > 0
    assert 0 <= result["gate_stats"]["max_ce_score"] < result["gate_stats"]["tau_abs"] == 0.12
    assert result["ranked"] and not result["empty_recall"]
    assert result["gate_stats"]["empty_reason"] is None
    assert not result["gate_stats"]["cliff_dropped_count"]
    absent = engine(enable_casual_bypass=False)
    assert absent.answer(negative)["response"] == "No relevant memory found."
    memory.answer(negative)
    assert memory.store.conn.total_changes == before
    assert memory.retrieve("老周对哪种防腐剂过敏？", read_only=True)["ranked"][0]["ce_score"] > 0.12
    memory.allow_empty_recall = False
    assert not memory.retrieve(negative, read_only=True)["empty_recall"]


def test_cliff_requires_confident_evidence_and_belt_does_not_refill():
    def ranked(rows):
        return [{"id": name, "zone": zone, "final_score": final, "ce_score": ce, "text": name * 80}
                for name, zone, final, ce in rows]

    config = dict(tau_abs=0.12, allow_empty_recall=True, cliff_ratio=0.45, belt_prune_ratio=0.35, min_keep=2)
    facts = ranked([("inner", INNER, 1.0, 1.0), ("outer", OUTER, 0.70, 0.70), ("tail", OUTER, 0.30, 0.30)])
    selected, stats = precision_gate(facts, **config)
    assert [fact["id"] for fact in selected] == ["inner", "outer"]
    assert stats["cliff_dropped_count"] == 1
    assert precision_gate(facts, **{**config, "min_keep": 3})[0] == facts

    facts = ranked([("inner", INNER, 1.0, 1.0), ("keep-belt", BELT, 0.35, 0.60),
                    ("prune-belt", BELT, 0.34, 0.58), ("outer", OUTER, 0.27, 0.27)])
    selected, stats = precision_gate(facts, **config)
    assert [fact["id"] for fact in selected] == ["inner", "keep-belt", "outer"]
    assert stats["belt_pruned_count"] == 1
    assert estimate_tokens("\n".join(f["text"] for f in selected)) < estimate_tokens("\n".join(f["text"] for f in facts))
    only_belt = ranked([("one", BELT, 0.95, 0.95), ("two", BELT, 0.30, 0.30)])
    assert precision_gate(only_belt, **config)[0] == only_belt[:1]


def test_four_channels_are_bounded_and_ce_gets_only_the_fused_top_40(monkeypatch):
    scorer = FixedScores()
    memory = engine(reranker=scorer)
    with memory.store.transaction():
        block = memory.store.create_block("Atlas", 1)
        ids = [memory.store.insert_fact(block_id=block, formatted=f"Atlas evidence {index}",
                                        plain=f"Atlas evidence {index}", ingest_time=index, zkeys=["实体:Atlas"])
               for index in range(50)]
    seen = {}

    def channel(name, start, **fields):
        def search(*args, top_n, **kwargs):
            seen[name] = top_n
            return [{"id": fact_id, **fields} for fact_id in ids[start:start + top_n]]
        return search

    monkeypatch.setattr(memory, "search_vector", channel("vector", 0, vector_score=0.6))
    monkeypatch.setattr(memory.store, "search_bm25", channel("lexical", 15, bm25_score=-2))
    monkeypatch.setattr(memory.store, "search_entity_inverted", channel("entity", 0))
    monkeypatch.setattr(memory.store, "search_temporal", channel("time", 15))
    result = memory.retrieve("Where did Atlas appear earliest?", timestamp=100, read_only=True)
    assert seen == dict(vector=35, lexical=35, entity=35, time=35)
    assert len(scorer.seen) == 1 and len(set(scorer.seen[0])) == result["gate_stats"]["candidate_count"] == 40


def test_remote_contract_calibrated_threshold_and_local_fallback(monkeypatch):
    monkeypatch.setenv("RERANK_BASE_URL", "https://reranker.example/v1")
    monkeypatch.setenv("RERANK_API_KEY", "test-key")
    monkeypatch.setenv("RERANK_MODEL", "test-model")
    calls = []

    def respond(request, timeout):
        calls.append((request.full_url, dict(request.header_items()), json.loads(request.data), timeout))
        docs = calls[-1][2]["documents"]
        rows = [{"index": index, "relevance_score": (index + 1) * 0.15} for index in reversed(range(len(docs)))]
        return io.StringIO(json.dumps({"results": rows}))

    monkeypatch.setattr("urllib.request.urlopen", respond)
    remote = OpenAICompatibleReranker.from_env()
    assert remote.is_calibrated and remote.score("query", [{"plain": "A"}, {"plain": "B"}]) == [0.15, 0.30]
    assert calls[0][:3] == ("https://reranker.example/v1/rerank",
                            {"Content-type": "application/json", "Authorization": "Bearer test-key"},
                            {"query": "query", "model": "test-model", "documents": ["A", "B"]})
    memory = engine(reranker=remote)
    memory.add("The Atlas prototype requires ceramic bearings for its motor.")
    query = "Which bearings does the Atlas prototype require?"
    result = memory.retrieve(query, read_only=True)
    assert result["gate_stats"]["tau_abs"] == 0.18 and result["empty_recall"]

    def unavailable(*args, **kwargs):
        raise OSError("offline fixture")

    monkeypatch.setattr("urllib.request.urlopen", unavailable)
    fallback = memory.retrieve(query, read_only=True)
    assert not fallback["empty_recall"]
    assert fallback["gate_stats"]["reranker_fallback"] and fallback["gate_stats"]["tau_abs"] == 0.12


def test_assembly_keeps_prefix_bytes_and_deduplicates_scoped_raw_windows():
    memory = engine(reranker=FixedScores(), enable_casual_bypass=False)
    scope = dict(user_id="alice", agent_id="planner", session_id="atlas")
    owner = {key: scope[key] for key in ("user_id", "agent_id")}
    memory.store.set_core_snapshot("Atlas researcher", ["Preserve release decisions"], **owner)
    memory.store.set_core_snapshot("Other tenant", ["Private rule"], user_id="bob", agent_id="planner")
    a = memory.add("Atlas motors use ceramic bearings.", timestamp=1, **scope)
    b = memory.add("Atlas couplings use titanium sleeves.", timestamp=2, **scope)
    outside = memory.add("Atlas external tenant secret.", timestamp=3, **{**scope, "user_id": "bob"})
    c = memory.add("Atlas pumps use silicone seals.", timestamp=4, **scope)
    other = memory.add("Atlas unrelated session secret.", timestamp=5, **{**scope, "session_id": "other"})
    tail = memory.store.append_turn("Atlas pumps need a bench test.\n", 6, **scope)
    first_fact = a["operations"][0]["line_id"]
    memory.store.insert_observation("Atlas part choices reflect durability.", derived_from_fact_ids=[first_fact], **owner)
    memory.store.insert_observation("Atlas part choices reflect durability.", derived_from_fact_ids=[first_fact], **owner)
    memory.store.insert_observation("Mixed-session private note", source_turn_ids=[a["turn_id"], other["turn_id"]], **owner)
    queries = ["Atlas release details?", "What are the Atlas bearings?", "Atlas pumps?", "Atlas sleeves?", "Atlas coupling?"]
    before = memory.store.conn.total_changes
    calls = []
    memory.store.conn.set_trace_callback(calls.append)
    contexts = [memory.build_context(query, **scope, drilldown_window=2, token_budget=500) for query in queries]
    memory.store.conn.set_trace_callback(None)
    assert memory.store.conn.total_changes == before
    assert sum("SELECT DISTINCT n.* FROM raw_turns" in call for call in calls) == 2 * len(contexts)
    prefixes = {ctx["system_prefix"].encode() for ctx in contexts}
    assert len(prefixes) == 1
    prefix, = prefixes
    for result in contexts:
        assert result["prefix_version"] == sha256(prefix).hexdigest()[:12]
        assert result["evidence_turn_ids"] in ([], [tail])
        assert outside["turn_id"] not in result["evidence_turn_ids"] and other["turn_id"] not in result["evidence_turn_ids"]
        assert "Atlas motors use ceramic bearings." in result["dynamic_memory"]
        assert "private note" not in result["dynamic_memory"]
        assert result["dynamic_memory"].count("Atlas part choices reflect durability.") == 1
        assert result["items"][0]["kind"] == "fact"
        assert all(item["turn_id"] == tail for item in result["items"] if item["kind"] == "raw_turn")
        tokens = result["token_breakdown"]
        assert tokens["total"] == tokens["system_prefix"] + estimate_tokens(result["dynamic_memory"]) <= 500
    assert any(result["evidence_turn_ids"] == [tail] for result in contexts)
    assert memory.build_context(queries[0], **scope, drilldown_top_n=0)["evidence_turn_ids"] == []
    # The numeric ID range differs from same-session adjacency across interleaved writes.
    narrow = memory.store.get_turn_window([b["turn_id"]], window=1, by_id=True, **scope)
    assert [row["turn_id"] for row in narrow] == [a["turn_id"], b["turn_id"]]
    size = estimate_tokens(prefix.decode())
    tiny = memory.build_context(queries[0], **scope, token_budget=size)
    assert tiny["system_prefix"].encode() == prefix and tiny["empty_recall"]
    assert tiny["dynamic_memory"] == "" and tiny["gate_stats"]["empty_reason"] == "budget"
    with pytest.raises(ValueError, match="stable system prefix"):
        memory.build_context(queries[0], **scope, token_budget=size - 1)
