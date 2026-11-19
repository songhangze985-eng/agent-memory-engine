from memengine.policy import decide_origin
from memengine.zaxis import collapse_and_scan_z_axis, order_far_to_near


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
    collapsed = collapse_and_scan_z_axis(index, ["人物:远", "人物:近"])
    assert [bucket["z_bucket"] for bucket in collapsed] == ["人物:远", "人物:近"]
    far_ids = [fact["id"] for fact in collapsed[0]["facts"]]
    near_ids = [fact["id"] for fact in collapsed[1]["facts"]]
    assert far_ids == ["b", "s"]
    assert near_ids == ["a"]
    assert collapsed[0]["facts"] == index["人物:远"]


def test_collapse_preserves_precomputed_zone_and_score():
    index = {"人物:甲": [{"id": "z", "zone": "POISON_BELT", "score": 0.10}]}
    collapsed = collapse_and_scan_z_axis(index, ["人物:甲"])
    fact = collapsed[0]["facts"][0]
    assert fact == index["人物:甲"][0]


def test_far_to_near_order_and_unranked_keys_stay_far():
    decision = decide_origin({}, [], ["人物:张敏", "人物:老周"], "老周对苯甲酸钠过敏")
    order = decision["poison_circle"]["z_axis_scan_order_far_to_near"]
    assert order[0] == "人物:张敏"
    assert order[-1] == "人物:老周"
    assert decision["poison_circle"]["radius_r"] == 0.25
    repaired = order_far_to_near(["人物:老周"], ["人物:老周", "事件:设计评审"])
    assert repaired[0] == "事件:设计评审"
    assert repaired[-1] == "人物:老周"
