from contextlib import closing
import json

import numpy as np
import pytest

from eval import regression_guard
from memengine import CognitiveMemoryEngine
from memengine import neural_onnx
from memengine.neural import NeuralBackendUnavailableError
from memengine.reranker import CalibratedLocalReranker
from memengine.store import Store
from memengine.textutil import estimate_tokens


def put(store, plain, time, *, scope=("default", "default", "conversation"), role="user"):
    with store.transaction():
        raw = store.append_turn(plain, time, scope=scope, role=role)
        block = store.create_block("Conversation", time, scope=scope)
        fact = store.insert_fact(block_id=block, scope=scope, plain=plain, formatted=plain,
                                 attribute="gateway", ingest_time=time, role=role, source_turn_ids=[raw])
    return fact, raw


def test_adjacent_reply_indexes_both_channels_without_changing_fact_or_scope(tmp_path):
    path = tmp_path / "chat.db"
    store = Store(path)
    first, first_turn = put(store, "Alice: Where was Project Egret calibrated? " + " laboratory instruments" * 28, 1)
    reply, reply_turn = put(store, "Mina: It was calibrated at Hudson.", 2)
    other, _ = put(store, "Yes, it was Hudson.", 3, scope=("default", "default", "other"))
    raw = store.get_line(reply, scope=("default", "default", "conversation"))
    assert raw["plain"] == "Mina: It was calibrated at Hudson."
    assert raw["source_turn_ids"] == [reply_turn]
    turn, context, dim, vector = store.conn.execute("SELECT turn_id,content,dim,vec FROM fact_anchors WHERE fact_id=?", (reply,)).fetchone()
    assert turn == first_turn and "Project Egret" in context and estimate_tokens(context) <= 40
    assert len(vector) == 4 * dim
    assert not store.conn.execute("SELECT 1 FROM fact_anchors WHERE fact_id IN (?,?)", (first, other)).fetchall()
    store.close()

    reopened = Store(path)
    lexical, dense = reopened.search_turn_anchors("Project Egret?", "Project Egret", session_id="conversation")
    assert lexical[0]["id"] == dense[0]["id"] == reply
    assert lexical[0]["context_anchor_turn_id"] == first_turn
    assert reopened.search_turn_anchors("Project Egret", "Project Egret", session_id="other") == ([], [])
    assert reopened.delete_fact(reply, hard=True, scope=("default", "default", "conversation"))
    assert not reopened.conn.execute("SELECT 1 FROM fact_anchors WHERE fact_id=?", (reply,)).fetchall()
    reopened.close()


def test_temporal_weight_cannot_promote_a_weakly_relevant_new_fact():
    class Fixed:
        is_calibrated = True

        def score(self, query, candidates):
            return [0.74 if "East" in fact["plain"] else 0.58 for fact in candidates]

    memory = CognitiveMemoryEngine(reranker=Fixed(), enable_casual_bypass=False)
    old, _ = put(memory.store, "Atlas routing currently uses the East gateway.", 5)
    new, _ = put(memory.store, "Atlas routing currently uses the West gateway.", 6)
    result = memory.retrieve("What is the Atlas routing gateway?", read_only=True)
    assert [fact["id"] for fact in result["ranked"][:2]] == [old, new]
    assert all(fact["temporal_weight"] == 1 for fact in result["ranked"])


def test_local_rrf_near_tie_stabilization_preserves_a_clear_relevance_gap():
    backend = CalibratedLocalReranker()
    first, second = {"id": "first", "ce_score": 0.69}, {"id": "second", "ce_score": 0.74}
    assert backend.stabilize_shortlist([second, first], [first, second]) == [first, second]
    weak = {**first, "ce_score": 0.10}
    assert backend.stabilize_shortlist([second, weak], [weak, second]) == [second, weak]
    exception = {**second, "is_constraint": True}
    assert backend.stabilize_shortlist([exception, first], [first, exception]) == [exception, first]


def test_onnx_is_normalized_and_calibrated_without_torch(monkeypatch):
    calls = []

    class Fake:
        embedding_size = 2

        def embed(self, inputs, batch_size):
            calls.append(inputs)
            yield from [[3.0, 4.0]] * len(inputs)

        def rerank(self, query, documents, batch_size):
            calls.append((query, documents))
            yield from [3, -3]

    monkeypatch.setattr(neural_onnx, "_create_model", lambda *args: Fake())
    embed = neural_onnx.OnnxNeuralEmbedder("intfloat/multilingual-e5-small")
    assert str(embed.encode(["doc"]).dtype) == "float32"
    assert np.allclose(embed.encode_queries(["ask"]), [[0.6, 0.8]])
    assert calls == [["passage: doc"], ["query: ask"]]
    assert embed.encode([]).shape == (0, 2)
    rerank = neural_onnx.OnnxCrossEncoderReranker()
    assert rerank.is_calibrated
    assert rerank.score("search", [{"plain": "one"}, {"plain": "two"}]) == pytest.approx([0.9525741268, 0.0474258732])


def test_onnx_preferred_then_torch_fallback(monkeypatch):
    from memengine import neural

    monkeypatch.setattr(neural_onnx, "OnnxNeuralEmbedder", lambda: "embed")
    monkeypatch.setattr(neural_onnx, "OnnxCrossEncoderReranker", lambda: "rank")
    assert neural.load_local_backends() == ("embed", "rank")

    def unavailable():
        raise NeuralBackendUnavailableError("weights missing")

    monkeypatch.setattr(neural_onnx, "OnnxCrossEncoderReranker", unavailable)
    monkeypatch.setattr(neural, "LocalNeuralEmbedder", lambda: "fallback-embed")
    monkeypatch.setattr(neural, "LocalCrossEncoderReranker", lambda: "fallback-rank")
    assert neural.load_local_backends() == ("fallback-embed", "fallback-rank")


def test_onnx_query_encoding_reused_only_in_one_recall_scope(monkeypatch):
    calls = []

    class Fake:
        def embed(self, inputs, batch_size):
            calls.extend(inputs)
            yield from [[3.0, 4.0]] * len(inputs)

    monkeypatch.setattr(neural_onnx, "_create_model", lambda *args: Fake())
    embedder = neural_onnx.OnnxNeuralEmbedder("intfloat/multilingual-e5-small")
    with embedder.cache_query_encodings():
        first = embedder.encode_queries(["ask"])
        first[:] = 0
        assert np.allclose(embedder.encode_queries(["ask"]), [[.6, .8]])
        embedder.encode_queries(["another"])
    with embedder.cache_query_encodings():
        embedder.encode_queries(["ask"])
    assert calls == ["query: ask", "query: another", "query: ask"]

    memory = CognitiveMemoryEngine(vector=embedder, reranker=CalibratedLocalReranker(), enable_casual_bypass=False)
    with closing(memory.store):
        put(memory.store, "Alice: Where was Project Egret calibrated? " + " laboratory instruments" * 28, 1)
        put(memory.store, "Mina: It was calibrated at Hudson.", 2)
        calls.clear()
        for _ in range(2):
            memory.build_context("Where was Project Egret calibrated?", session_id="conversation",
                                 timestamp=3, token_budget=512, include_short_term=False)
        assert len(calls) == 2 and calls[0] == calls[1]


def test_guard_checks_cohort_numeric_floor_and_pytest_exit(tmp_path, monkeypatch):
    name = "03-membench"
    items = [{"id": "fixture", "evidence_steps": [2]}]
    digest = regression_guard.cohort_digest(items, ("id", "evidence_steps"))
    monkeypatch.setattr(regression_guard, "COHORTS", {name: (1, ("id", "evidence_steps"), digest)})
    report = {"backend": "hashed", "mode": "mock", "seeds": [42, 43, 44, 45, 46], "items": items,
              "top_k": 10, "token_budget": 2048,
              "paired_retrieval": {"bipolar": {"recall_at_5": .60555555, "recall_at_10": .800757575,
                                                 "mrr_at_10": .70, "answerable_n": 99, "false_empty": 0,
                                                 "avg_injected_tokens": 530}}}
    out = tmp_path / f"{name}.json"
    out.write_text(json.dumps(report))
    log = tmp_path / "01-pytest.log"
    log.write_text("==== 127 passed in 2s ====\nExit code: 0\n")
    assert all(row[-1] for row in regression_guard.validate(tmp_path)[0])
    report["items"][0]["evidence_steps"] = [10]
    report["paired_retrieval"]["bipolar"]["recall_at_10"] = .8
    out.write_text(json.dumps(report))
    log.write_text("==== 127 passed in 2s ====\nExit code: 1\n")
    failed = {row[1] for row in regression_guard.validate(tmp_path)[0] if not row[-1]}
    assert failed == {"exit", "frozen cohort", "recall_at_10"}


def test_consolidated_ablation_preserves_facts_and_measures_observation_sources():
    from eval.evidence_ablation import Question, Turn, measure_pair
    from eval.paired import attach_consolidated_context, evaluate_pair

    engine = CognitiveMemoryEngine(enable_casual_bypass=False)
    fact, turn_id = put(engine.store, "Atlas uses a calibrated laboratory sensor.", 1)
    untouched = evaluate_pair(engine, "Which sensor does Atlas use?", 2)
    engine.store.set_core_snapshot("Atlas laboratory profile", [])
    engine.store.insert_observation("Atlas tracks sensor calibration.", derived_from_fact_ids=[fact], source_turn_ids=[turn_id])
    attached = attach_consolidated_context(engine, untouched, "Which sensor does Atlas use?")
    raw_ids = {turn_id: Turn("official-turn-1", "conversation", "user", "sensor", 1)}
    q = Question("q", "reflective", "Which sensor does Atlas use?", 2, frozenset({"official-turn-1"}))
    before = measure_pair(engine, q, untouched, raw_ids)["bipolar"]
    after = measure_pair(engine, q, attached, raw_ids)["bipolar"]
    assert before["ranked_turns"] == after["ranked_turns"] == [["official-turn-1"]]
    assert after["recall_at_10"] == before["recall_at_10"] == 1
    assert after["prefix_tokens"] > 0 and after["observation_count"] == after["observation_gold_supports"] == 1
    assert after["observation_only_gold_supports"] == 0
