from eval.corpus import FACTS, FILLER, NOISE_SENTENCES
from agentmem_core.extract import extract_facts


def test_fact_bank_extracts_the_rare_value():
    assert len(FACTS) == 20
    for fact in FACTS:
        found = extract_facts(fact["text"])
        assert len(found) == 1, fact["text"]
        assert found[0]["value"] == fact["keywords"][0]
        assert fact["keywords"][0] in found[0]["text"]
        assert found[0]["zkeys"]


def test_noise_and_filler_are_not_facts():
    for sentence in NOISE_SENTENCES + [FILLER]:
        assert extract_facts(sentence) == [], sentence


def test_meeting_and_budget_patterns():
    habit = extract_facts("团队每周三下午三点开周会。")
    change = extract_facts("本周周会改到周五上午十点。")
    assert habit[0]["attribute"] == "周会时间"
    assert change[0]["attribute"] == "周会时间"
    assert habit[0]["zkeys"] == change[0]["zkeys"]
    assert "事件:周会" in habit[0]["zkeys"]
    assert habit[0]["value"] != change[0]["value"]
    initial = extract_facts("设计评审的预算为5000元。")
    revised = extract_facts("设计评审的预算改为8000元。")
    assert initial[0]["value"] == "5000"
    assert revised[0]["value"] == "8000"
    assert initial[0]["attribute"] == revised[0]["attribute"] == "预算"
