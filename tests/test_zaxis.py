from memengine.policy import decide_origin
from memengine.zaxis import collapse_and_scan_z_axis, expand_entities, order_far_to_near


def test_far_to_near_scan_isolates_buckets_and_dedups():
    index = {
        "人物:近": [
            {"id": "a", "raw_dist": 0.1, "text": "near only", "entity_anchor": True},
            {"id": "s", "raw_dist": 0.2, "text": "shared", "entity_anchor": True},
        ],
        "人物:远": [
            {"id": "b", "raw_dist": 0.9, "text": "far only", "entity_anchor": True, "isolated": True},
            {"id": "s", "raw_dist": 0.2, "text": "shared", "entity_anchor": True},
        ],
    }
    collapsed = collapse_and_scan_z_axis(index, ["人物:远", "人物:近"], radius_r=0.45, epsilon=0.05)
    assert [bucket["z_bucket"] for bucket in collapsed] == ["人物:远", "人物:近"]
    far_ids = [fact["id"] for fact in collapsed[0]["facts"]]
    near_ids = [fact["id"] for fact in collapsed[1]["facts"]]
    assert far_ids == ["b", "s"]
    assert near_ids == ["a"]
    zones = {fact["id"]: fact["zone"] for bucket in collapsed for fact in bucket["facts"]}
    assert zones["b"] == "OUTER_ANTI_INTUITIVE"
    assert zones["a"] == "INNER_INTUITIVE"
    assert zones["s"] == "INNER_INTUITIVE"
    assert collapsed[0]["facts"][0]["d_asymp"] == 0.9


def test_raw_zero_becomes_epsilon_inside_the_circle():
    index = {"人物:甲": [{"id": "z", "raw_dist": 0, "text": "touch", "vector_score": 1.0}]}
    collapsed = collapse_and_scan_z_axis(index, ["人物:甲"], radius_r=0.45, epsilon=0.05)
    fact = collapsed[0]["facts"][0]
    assert fact["d_asymp"] == 0.05
    assert fact["zone"] == "INNER_INTUITIVE"


def test_expand_does_not_cross_unrelated_axes():
    index = {
        "人物:张敏": [{"id": "t", "zkeys": ["人物:张敏", "地点:京都", "事件:设计评审"]}],
        "地点:京都": [{"id": "t", "zkeys": ["人物:张敏", "地点:京都", "事件:设计评审"]}],
        "事件:设计评审": [
            {"id": "t", "zkeys": ["人物:张敏", "地点:京都", "事件:设计评审"]},
            {"id": "b", "zkeys": ["事件:设计评审"]},
        ],
        "人物:李楠": [{"id": "r", "zkeys": ["人物:李楠", "事件:供应商会谈"]}],
        "事件:供应商会谈": [{"id": "r", "zkeys": ["人物:李楠", "事件:供应商会谈"]}],
    }
    reached = set(expand_entities(index, ["人物:张敏"], hops=2))
    assert "事件:设计评审" in reached
    assert "人物:李楠" not in reached
    assert "事件:供应商会谈" not in reached


def test_far_to_near_order_and_unranked_keys_stay_far():
    decision = decide_origin({}, [], ["人物:张敏", "人物:老周"], "老周对苯甲酸钠过敏")
    order = decision["poison_circle"]["z_axis_scan_order_far_to_near"]
    assert order[0] == "人物:张敏"
    assert order[-1] == "人物:老周"
    assert decision["poison_circle"]["radius_r"] == 0.45
    assert decision["poison_circle"]["anti_intuitive_weight"] >= 0.85
    repaired = order_far_to_near(["人物:老周"], ["人物:老周", "事件:设计评审"])
    assert repaired[0] == "事件:设计评审"
    assert repaired[-1] == "人物:老周"
