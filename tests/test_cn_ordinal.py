"""Chinese ordinals must agree between time parsing and revision travel."""

from memengine.engine import CognitiveMemoryEngine
from memengine.policy import travel_revision
from memengine.temporal import parse_temporal_intent
from memengine.textutil import parse_cn_number

NOW = 1000.0


def test_parse_cn_number_reads_compound_numerals():
    assert parse_cn_number("十二") == 12
    assert parse_cn_number("二十") == 20
    assert parse_cn_number("二十三") == 23
    assert parse_cn_number("十") == 10
    assert parse_cn_number("二") == 2
    assert parse_cn_number("两") == 2
    assert parse_cn_number("十一") == 11
    assert parse_cn_number("九十九") == 99
    assert parse_cn_number("12") == 12
    assert parse_cn_number("十二三") == 1


def test_temporal_and_revision_policy_share_ordinals():
    expected = {"十二": 11, "二十": 19, "二十三": 22, "十": 9, "二": 1}
    for numeral, index in expected.items():
        query = f"在第{numeral}次修改预算之前"
        assert parse_temporal_intent(query, NOW).rev_index == index
        assert travel_revision(query) == index
    spaced = "在第 十二 次修改预算之前"
    assert parse_temporal_intent(spaced, NOW).rev_index == 11
    assert travel_revision(spaced) == 11
    digits = "在第12次修改预算之前"
    assert parse_temporal_intent(digits, NOW).rev_index == 11
    assert travel_revision(digits) == 11
    assert parse_temporal_intent("before the second revision", NOW).rev_index == 1
    assert travel_revision("before the second revision") == "previous"


def test_before_nth_budget_edit_uses_that_revision():
    engine = CognitiveMemoryEngine(db_path=":memory:")
    engine.add("设计评审的预算为1000元。", timestamp=100)
    for step, value in enumerate((2000, 3000, 4000, 5000), start=1):
        engine.add(f"设计评审的预算改为{value}元。", timestamp=100 + step * 100)

    second = engine.answer("在第二次修改预算之前", timestamp=900)
    assert "2000" in second["response"], second["response"]
    assert "1000" not in second["response"]

    fourth = engine.answer("在第四次修改预算之前", timestamp=900)
    assert "4000" in fourth["response"], fourth["response"]
    assert "5000" not in fourth["response"]

    fifth = engine.answer("在第五次修改预算之前", timestamp=900)
    assert "5000" in fifth["response"], fifth["response"]
    assert "4000" not in fifth["response"]

    twelfth = engine.answer("在第十二次修改预算之前", timestamp=900)
    assert twelfth["temporal_intent"]["rev_index"] == 11
    assert "5000" in twelfth["response"], twelfth["response"]
    assert "4000" not in twelfth["response"]
    assert "1000" not in twelfth["response"]

    earlier = engine.answer("在第十二次修改预算以前", timestamp=900)
    assert earlier["temporal_intent"]["rev_index"] == 11
    assert "5000" in earlier["response"], earlier["response"]
    assert "4000" not in earlier["response"]
    assert "1000" not in earlier["response"]
