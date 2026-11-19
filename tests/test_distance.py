from memengine.distance import asymptotic_distance, compute_raw_dist


def test_epsilon_is_a_lower_bound():
    assert asymptotic_distance(0, 0.05) == 0.05
    assert asymptotic_distance(0.04, 0.05) == 0.05
    assert asymptotic_distance(-1, 0.05) == 0.05
    assert asymptotic_distance(0.05, 0.05) == 0.05
    assert asymptotic_distance(0.5, 0.05) == 0.5
    assert asymptotic_distance("nope", 0.05) == 0.05


def test_identical_anchor_can_be_raw_zero_but_not_asymptotic_zero():
    fact = {
        "text": "工作安排周会时间",
        "surprise": 0.0,
        "layout": "APPEND_ADJACENT_LINE",
        "isolated": False,
    }
    origin = {
        "x_topic": "工作安排周会时间",
        "y_subtopic": "",
        "decision_anchor": "",
    }
    raw = compute_raw_dist(fact, origin)
    assert raw == 0.0
    assert asymptotic_distance(raw) == 0.05


def test_layout_and_surprise_do_not_change_semantic_distance():
    fact = {
        "text": "工作安排周会时间",
        "surprise": 0.1,
        "layout": "INSERT_ISOLATED_LINE",
        "isolated": True,
    }
    origin = {"x_topic": "工作安排", "y_subtopic": "周会时间", "decision_anchor": "工作安排周会时间"}
    assert compute_raw_dist(fact, origin) == compute_raw_dist({**fact, "isolated": False, "layout": "APPEND_ADJACENT_LINE", "surprise": 0.9}, origin)
