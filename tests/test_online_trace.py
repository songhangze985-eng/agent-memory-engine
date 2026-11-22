from eval.evidence_ablation import Question, Turn
from eval.online_trace import FUSION_ABLATIONS, diagnose, run_group
from eval.paired import make_engine
from eval.packing_replay import oracle_order
from memengine.retrieval.rerank_budget import RerankBudget, shortlist


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


def test_packing_replay_reproduces_context_and_keeps_budget():
    stamp = 1735689500
    turns = [
        Turn("a:0", "a", "user", "I prefer quiet history podcasts on my commute.", stamp),
        Turn("b:0", "b", "user", "My commute lasts forty minutes.", stamp + 10),
    ]
    question = Question("test", "preference", "Which podcasts do I prefer for my commute?",
                        stamp + 20, frozenset({"a:0"}), frozenset({"a"}))
    row, = run_group((turns, [question]), backend="hashed", mode="reconstructive", packing_replay=True)
    versions = row["packing_replay"]
    assert versions["admitted_gold_coverage"] == 1
    assert versions["baseline"]["recall_at_10"] == row["recall_at_10"]
    assert versions["baseline"]["tokens"] == row["tokens"]
    assert versions["ce_order"]["tokens"] <= 2048
    assert versions["gold_first"]["recall_at_10"] == 1


def test_gold_first_reordering_covers_distinct_sources_and_keeps_pool():
    distractor = {"id": "x", "text": "Related distractor"}
    long_dup = {"id": "a", "text": "A longer duplicate with the same source"}
    short = {"id": "b", "text": "Short"}
    independent = {"id": "c", "text": "Other evidence"}
    pool = [distractor, long_dup, short, independent]
    sources = {"x": ["noise"], "a": ["one"], "b": ["one"], "c": ["two"]}
    assert oracle_order(pool, sources, {"one", "two"}) == [short, independent, distractor, long_dup]


def test_fusion_ablation_keeps_control_and_provenance():
    stamp = 1735689500
    turns = [Turn("a:0", "a", "user", "I prefer quiet podcasts for my commute.", stamp)]
    question = Question("test", "preference", "Which podcasts do I prefer for my commute?",
                        stamp + 20, frozenset({"a:0"}), frozenset({"a"}))
    group = turns, [question]
    control, = run_group(group, backend="hashed", mode="reconstructive")
    results = run_group(group, backend="hashed", mode="reconstructive", fusion_ablation=True)
    assert [row["fusion_variant"] for row in results] == list(FUSION_ABLATIONS)
    assert {row["id"] for row in results} == {control["id"]}
    assert results[0]["context_fact_ids"] == control["context_fact_ids"]
    assert results[0]["gold"] == control["gold"]
    assert all(row["tokens"] <= 2048 for row in results)


def test_ce_shortlist_protects_pinned_and_same_slot_evidence():
    candidates = [{"canonical_entity_id": "alice", "attribute_slot": "workplace"},
                  {}, {}, {}, {}, {}, {},
                  {"canonical_entity_id": "alice", "attribute_slot": "workplace"},
                  {"exact_ident_hit": 1}]
    assert shortlist(candidates, 7) == [0, 1, 2, 3, 4, 7, 8]
    assert shortlist(candidates, 1) == [8]
    assert shortlist(candidates, 0) == []


def test_ce_budget_preserves_measured_scores_across_passes():
    calls = []

    def score(query, candidates):
        calls.extend(row["id"] for row in candidates)
        return [.8] * len(candidates)

    candidates = [{"id": "first", "plain": "other version"},
                  {"id": "exact", "plain": "requested version", "exact_ident_hit": 1}]
    budget = RerankBudget(1)
    assert budget.score("version", candidates, score) == ([0.0, .8], ["exact"])
    assert budget.score("version", list(reversed(candidates)), score) == ([.8, 0.0], [])
    assert calls == ["exact"]


def test_runtime_trace_shares_ce_quota_across_reconstruction_passes(monkeypatch):
    class Scorer:
        wide_pair_limit = 1
        is_calibrated = True

        def score(self, query, candidates):
            return [.8] * len(candidates)

    memory = make_engine("mock")
    memory.reranker = Scorer()
    memory.retain("I prefer quiet history podcasts on my commute.", timestamp=10)
    memory.retain("I also like travel documentaries.", timestamp=12)
    monkeypatch.setattr("memengine.engine.bridge_keys", lambda *args: ["entity:commute"])
    observed = []
    result = memory.retrieve("What podcasts do I prefer on my commute?", mode="reconstructive",
                             timestamp=15, token_budget=2048, read_only=True, _trace=observed)
    first, second = observed
    ident, = first["ce_scored_ids"]
    assert second["ce_scored_ids"] == []
    assert second["ce_scores"][ident] == .8
    assert result["gate_stats"]["ce_pair_limit"] == result["gate_stats"]["ce_pairs_scored"] == 1
    memory.store.close()
