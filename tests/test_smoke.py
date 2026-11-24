from agentmem_core.engine import CognitiveMemoryEngine
from agentmem_core.router import CASUAL, TASK


def test_isolated_fact_survives_noise_and_reload(tmp_path):
    path = tmp_path / "memory.db"
    engine = CognitiveMemoryEngine(db_path=path)
    engine.add("食品工业再次讨论防腐剂的标签写法。过敏季门诊量有所上升。", timestamp=1)
    engine.add("老周对苯甲酸钠过敏，不能食用含该防腐剂的食品。", timestamp=2)
    engine.add("社区开了一场密码学讲座。出差补贴标准维持不变。", timestamp=3)
    first = engine.answer("老周对哪种防腐剂过敏？", timestamp=50)
    assert first["route"] == TASK
    assert "苯甲酸钠" in first["response"]
    assert engine.fallback_count == 0
    keys = set(engine.store.z_index())
    assert "人物:老周" in keys
    assert any(line["isolated"] for block in engine.store.active_snapshot() for line in block["lines"] if "苯甲酸钠" in line["plain"])

    reloaded = CognitiveMemoryEngine(db_path=path)
    second = reloaded.answer("老周对哪种防腐剂过敏？", timestamp=80)
    assert "苯甲酸钠" in second["response"]


def test_overwrite_suppresses_the_repeated_habit():
    engine = CognitiveMemoryEngine(db_path=":memory:")
    for offset in range(4):
        engine.add("团队每周三下午三点开周会。", timestamp=100 + offset)
    engine.add("本周周会改到周五上午十点。", timestamp=200)
    result = engine.answer("这周的周会安排在什么时候？", timestamp=300)
    assert "周五" in result["response"], result["response"]
    assert "周三" not in result["response"], result["response"]
    active = [line for block in engine.store.active_snapshot() for line in block["lines"] if line["attribute"] == "周会时间"]
    assert len(active) == 1
    assert active[0]["revision"] >= 1


def test_multihop_and_time_travel_do_not_cross_people():
    engine = CognitiveMemoryEngine(db_path=":memory:")
    engine.add("张敏将前往京都参加设计评审。", timestamp=1)
    engine.add("设计评审的预算为5000元。", timestamp=100)
    engine.add("设计评审的预算改为8000元。", timestamp=200)
    engine.add("设计评审的预算改为3000元。", timestamp=300)
    engine.add("李楠将前往青岛参加供应商会谈。", timestamp=400)
    engine.add("供应商会谈的预算为2000元。", timestamp=500)
    current = engine.answer("张敏这次出行的预算是多少？", timestamp=600)
    assert "3000" in current["response"], current["response"]
    assert "8000" not in current["response"], current["response"]
    assert "5000" not in current["response"], current["response"]
    assert "2000" not in current["response"], current["response"]
    previous = engine.answer("在第二次修改预算之前，张敏这次出行的预算是多少？", timestamp=700)
    assert "8000" in previous["response"], previous["response"]
    assert "3000" not in previous["response"], previous["response"]
    assert "5000" not in previous["response"], previous["response"]
    assert engine.fallback_count == 0


def test_casual_greeting_does_not_dump_memory():
    engine = CognitiveMemoryEngine(db_path=":memory:")
    engine.add("苏晚的门禁密码是7341。", timestamp=1)
    result = engine.answer("你好", timestamp=20)
    assert result["route"] == CASUAL
    assert "7341" not in result["response"]
    casual_off = CognitiveMemoryEngine(db_path=":memory:", enable_casual_bypass=False, enable_timeline_buffer=False)
    casual_off.add("苏晚的门禁密码是7341。", timestamp=1)
    forced = casual_off.answer("你好", timestamp=20)
    assert forced["route"] == TASK
