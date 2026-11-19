from unittest.mock import patch

import pytest

from memengine import engine as engine_module
from memengine.distance import BELT, INNER, NEWS_KEY, OUTER, bipolar_weight, compute_bipolar_radius, score_bipolar
from memengine.engine import CognitiveMemoryEngine, _fit_context, _prune_belt
from memengine.store import Store
from memengine.vector import HashedTfidfEmbedder


def _score(fact, raw_rrf=1.0, c_max=8, radius=0.25):
    return score_bipolar({**fact, "rrf_score": raw_rrf}, c_max=c_max, rrf_max=1.0,
                         radius=radius, bipolar_lambda=0.50)


def test_exact_poles_and_mirrored_weights():
    assert bipolar_weight(0.0) == bipolar_weight(1.0) == 1.0
    assert bipolar_weight(0.5) == 0.10
    assert bipolar_weight(0.25) == bipolar_weight(0.75) == 0.20
    assert bipolar_weight(0.03) == bipolar_weight(0.97) == 1.0


def test_frequency_geometry_and_outer_overrides_are_not_semantic_distance():
    habit = {"source_step_ids": list(range(8))}
    single = {"source_step_ids": [9], "raw_dist": 0.99}
    noise = {"zkeys": [NEWS_KEY], "hit_count": 100}
    assert _score(habit)["zone"] == INNER
    assert compute_bipolar_radius(habit, 8) == 0.0
    assert _score(habit)["bipolar_weight"] == _score(single)["bipolar_weight"] == 1.0
    assert _score(single)["zone"] == OUTER
    assert _score(single)["bipolar_weight"] == 1.0
    consolidated = {"source_step_ids": list(range(10))}
    assert _score(consolidated, c_max=10)["score"] == _score(single, c_max=10)["score"]
    assert _score(habit)["score"] > _score(noise)["score"]
    mid = _score({"source_step_ids": [2, 3]}, radius=1)
    assert mid["r_m"] == 0.50 and mid["zone"] == BELT and mid["bipolar_weight"] == 0.10
    assert _score(noise)["zone"] == BELT and compute_bipolar_radius(noise, 8) == 0.5
    assert compute_bipolar_radius({**habit, "isolated": True}, 8) == 1.0
    assert _score({**habit, "layout": "OVERWRITE_SAME_LINE"})["zone"] == OUTER
    for field in ("is_conflict", "historical"):
        revised = _score({**habit, field: True})
        assert revised["zone"] == OUTER and revised["override_bonus"] == 0.05


def test_low_rrf_singleton_cannot_displace_relevant_top_k():
    facts = [_score({"id": "irrelevant", "text": "weak"}, raw_rrf=1e-6),
             _score({"id": "ordinary", "text": "relevant", "hit_count": 1}, raw_rrf=0.85),
             _score({"id": "habit", "text": "habit", "hit_count": 7}, raw_rrf=1.0)]
    ranked = sorted(facts, key=lambda fact: -fact["score"])
    assert [fact["id"] for fact in _fit_context(ranked, 2, 100)] == ["habit", "ordinary"]


def test_all_singletons_keep_exact_rrf_order_across_z_buckets():
    engine = CognitiveMemoryEngine(vector=HashedTfidfEmbedder())
    for index in range(12):
        engine.add(f"[step {index}] Alice and Casey watched the movie Film{index}.", timestamp=index)
    original = engine_module.weighted_rrf
    captured = []

    def observe(*args, **kwargs):
        ranked = original(*args, **kwargs)
        captured.extend(ranked)
        return ranked

    with patch.object(engine_module, "weighted_rrf", observe):
        result = engine.retrieve("Which movies did Alice and Casey watch?", read_only=True)
    assert len(captured) >= 10
    assert [fact["id"] for fact in result["ranked"]] == [fact["id"] for fact in captured[:10]]
    assert not result["empty_retrieval"] and result["outer_triggered_count"] == 10
    assert result["inner_count"] == result["belt_count"] == result["belt_compressed_count"] == 0


def test_belt_only_is_retained_and_tiny_budget_truncates_one_candidate():
    engine = CognitiveMemoryEngine(vector=HashedTfidfEmbedder(), enable_casual_bypass=False)
    engine.add("今天市面公开讨论园艺种植技术以及家庭绿植养护方法。", timestamp=2)
    result = engine.retrieve("园艺种植技术？", read_only=True, token_budget=3)
    assert not result["empty_retrieval"] and result["belt_count"] == 1
    assert result["ranked"][0]["context_truncated"]
    assert result["context_tokens"] == 3 and result["belt_compressed_count"] == 0
    assert engine.retrieve("园艺种植技术？", read_only=True, top_k=0)["empty_retrieval"]


def test_belt_prune_keeps_poles_and_refills_in_global_order():
    ranked = [{"id": name, "zone": zone, "bipolar_score": score} for name, zone, score in
              (("high", INNER, 1.0), ("at_threshold", BELT, 0.35), ("low", BELT, 0.34), ("outer", OUTER, 0.02))]
    assert [fact["id"] for fact in _prune_belt(ranked, 0.35, 2)] == ["high", "at_threshold", "outer"]
    assert _prune_belt(ranked, 0, 2) == ranked
    assert _prune_belt([], 0.35, 2) == []
    belt = [{**fact, "zone": BELT, "text": fact["id"]} for fact in ranked]
    assert _fit_context(_prune_belt(belt, 0.90, 2), 10, 2048) == belt[:2]


def test_belt_pruning_runs_before_top_k_and_preserves_empty_floor(monkeypatch):
    engine = CognitiveMemoryEngine(vector=HashedTfidfEmbedder(), enable_casual_bypass=False)
    for number in range(4):
        engine.add(f"城市园艺种植讨论会第{number}期介绍多种家庭绿植养护方法。", timestamp=number)
    original = engine_module.weighted_rrf

    def ranked(*args, **kwargs):
        return [{**fact, "rrf_score": score} for fact, score in
                zip(original(*args, **kwargs), (0.01, 0.003, 0.002, 0.001))]

    monkeypatch.setattr(engine_module, "weighted_rrf", ranked)
    result = engine.retrieve("城市园艺种植讨论会？", read_only=True)
    assert result["belt_count"] == 2 and not result["empty_retrieval"]
    assert result["belt_pruned_count"] == result["belt_compressed_count"] == 2
    assert engine.retrieve("城市园艺种植讨论会？", read_only=True, belt_prune_ratio=0)["belt_count"] == 4


def test_hit_counts_batch_scope_and_read_only_sql():
    engine = CognitiveMemoryEngine(vector=HashedTfidfEmbedder())
    query = "What films did Alice watch?"
    for tenant in ("alice", "bob"):
        for tag in range(3):
            engine.add(f"[step {tag}] Alice watched film Film{tag}", timestamp=tag, user_id=tenant)
    sql = []
    engine.store.conn.set_trace_callback(sql.append)
    result = engine.retrieve(query, user_id="alice", read_only=True)
    assert len(result["ranked"]) == 3
    assert not any(statement.lstrip().split()[0] in {"UPDATE", "INSERT", "DELETE", "REPLACE"} for statement in sql)
    sql.clear()
    engine.retrieve(query, user_id="alice")
    assert sum("UPDATE facts SET hit_count" in statement for statement in sql) == 1
    assert {fact["hit_count"] for fact in engine.store.snapshot(scope=("alice", "default", None))} == {1}
    assert {fact["hit_count"] for fact in engine.store.snapshot(scope=("bob", "default", None))} == {0}
    sql.clear()
    engine.answer(query, user_id="alice")
    assert not any(statement.lstrip().split()[0] in {"UPDATE", "INSERT", "DELETE", "REPLACE"} for statement in sql)


def test_noop_reinforces_source_steps_without_overwriting():
    engine = CognitiveMemoryEngine(vector=HashedTfidfEmbedder())
    for step in range(8):
        engine.add(f"[step {step}] 团队每周三下午三点开周会。", timestamp=step)
    rows = engine.store.snapshot()
    assert len(rows) == 1 and rows[0]["revision"] == 0
    assert rows[0]["source_step_ids"] == list(range(8))
    assert not engine.store.historical_versions([rows[0]["id"]])


def test_hit_count_persists_for_existing_active_fact(tmp_path):
    path = tmp_path / "legacy.db"
    engine = CognitiveMemoryEngine(db_path=path, vector=HashedTfidfEmbedder())
    line_id = engine.add("团队每周三下午三点开周会。", timestamp=1)["operations"][0]["line_id"]
    engine.store.close()
    store = Store(path)
    assert store.get_line(line_id)["hit_count"] == 0
    with store.transaction():
        store.increment_hit_counts([line_id, line_id, "does-not-exist"])
    store.close()
    reopened = Store(path)
    assert reopened.get_line(line_id)["hit_count"] == 1
