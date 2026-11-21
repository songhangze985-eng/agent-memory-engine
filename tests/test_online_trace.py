from eval.online_trace import diagnose
from eval.paired import make_engine


def test_retrieval_trace_observes_actual_call_without_changing_it():
    engine = make_engine("mock")
    stamp = "2025-01-01T00:00:00Z"
    engine.retain("I prefer quiet podcasts for my commute.", timestamp=1735689500, session_id="sample")
    options = {"reference_time": stamp, "mode": "reconstructive", "token_budget": 2048,
               "include_short_term": False}
    untraced = engine.build_context("Which podcasts do I prefer for my commute?", **options)
    writes = engine.store.conn.total_changes
    passes = []
    traced = engine.build_context("Which podcasts do I prefer for my commute?", _trace=passes, **options)
    assert engine.store.conn.total_changes == writes
    assert untraced.keys() == traced.keys()
    assert {key: value for key, value in untraced.items() if key != "retrieval_ms"} == {
        key: value for key, value in traced.items() if key != "retrieval_ms"
    }
    evidence = [row["id"] for row in traced["items"] if row["kind"] == "fact"]
    assert evidence
    assert len(passes) == 1
    observed = passes[0]
    assert observed["rerank_calls"] == 1
    assert list(observed["ce_scores"]) == observed["fused"]
    assert observed["packed"] == evidence
    assert set(evidence) <= set(observed["gate_kept"])


def test_diagnosis_distinguishes_candidate_gate_and_packing_losses():
    trace = {"fused": ["a", "b", "c"], "ce_scores": {"a": .3, "b": .8, "c": .7},
             "reranked": ["a", "b", "c"], "gate_input": ["a", "b", "c"], "gate_kept": ["a", "c"],
             "packing_input": ["a", "c"], "packed": ["a"],
             "sources": {"a": ["a:0"], "b": ["a:1"], "c": ["b:0"]},
             "sessions": {"a": "a", "b": "a", "c": "b"}}
    rows = {row["ref"]: row for row in diagnose(trace, [], {}, {"a:0", "a:1", "b:0", "c:0"})}
    assert rows["a:0"]["outcome"] == "missing_from_context"
    assert rows["a:1"]["outcome"] == "missing_from_gate_kept"
    assert rows["a:1"]["ranks"]["ce"] == 1
    assert rows["a:1"]["same_session_above_reranked"] == 1
    assert rows["b:0"]["outcome"] == "missing_from_packed"
    assert rows["c:0"]["outcome"] == "missing_from_fused"
