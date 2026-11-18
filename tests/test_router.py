from memengine.router import CASUAL, TASK, route_query


KEYS = [
    "人物:老周",
    "禁忌:苯甲酸钠",
    "人物:苏晚",
    "事件:门禁密码",
    "事件:仓库备用钥匙",
    "地点:绿色铁盒",
    "事件:北岛项目",
    "事件:周会",
    "人物:张敏",
    "事件:设计评审",
    "话题:时事杂讯",
]


def test_casual_when_no_entity_and_no_syntax():
    assert route_query("你好", KEYS) == CASUAL
    assert route_query("今天天气真好", KEYS) == CASUAL
    assert route_query("最近还好吗", KEYS) == CASUAL


def test_entity_name_hit_routes_to_task():
    assert route_query("老王在吗", ["人物:老王"]) == TASK
    assert route_query("老周对哪种防腐剂过敏？", KEYS) == TASK


def test_syntax_triggers_follow_the_spec():
    assert route_query("我们改一下", []) == TASK
    assert route_query("之前的记录", []) == TASK
    assert route_query("帮我", []) == CASUAL
    assert route_query("请你帮我做一个完整的方案", []) == CASUAL
    assert len("请你帮我做一个完整的方案") == 12
    assert route_query("请你帮我做一个完整的方案哦", []) == TASK
    assert route_query("今天的特别节目", []) == CASUAL
    assert route_query("你别去", []) == TASK


def test_english_cues_use_word_boundaries_and_mcq_is_task():
    assert route_query("hello", []) == CASUAL
    assert route_query("Please change the meeting", []) == TASK
    assert route_query("What is the education level of the subordinate?", []) == TASK
    question = "What is the education level?\nA. Bachelor\nB. Associate Degree\nC. High School\nD. Master"
    assert route_query(question, ["entity:subordinate"]) == TASK
    assert route_query("Anniversary party", ["entity:ann"]) == CASUAL
    assert route_query("Ann is here", ["person:Ann"]) == TASK


def test_decision_pattern_needs_length():
    assert route_query("请帮我", []) == CASUAL
    assert route_query("门锁坏了怎么办", []) == CASUAL
    assert route_query("请帮我确认老周是不是对花生过敏，并给个方案。", KEYS) == TASK
