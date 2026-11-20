"""Synthetic long-dialogue fixtures shared by both stress runners."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StressCase:
    person: str
    turns: tuple[tuple[str, str], ...]
    positive: str
    negative: str
    negative_kind: str


def stress_dialogue(index):
    person = ["老周", "张敏", "李楠", "王宁", "苏晚"][index % 5]
    food = ["花生", "牛奶", "海鲜", "芝麻", "杏仁"][index // 5]
    turns = [("habit", f"{person}每周三下午三点开周会，周会餐饮安排照常。")] * (6 + index % 5)
    for note in range(14 + index % 5):
        marker = "甲乙丙丁戊己庚辛壬癸"[note // 10] + "甲乙丙丁戊己庚辛壬癸"[note % 10]
        text = f"{person}周会餐饮方案{marker}的预算为{1000 + note}元，周会餐饮安排计划讨论与通知下达。"
        turns.extend([("background", text)] * 2)
    for note in range(12):
        turns.append(("noise", f"通用资讯栏目报道第{note:02d}期城市公共服务进展与社区节庆活动。"))
    turns.append(("taboo", f"{person}对{food}过敏。"))
    queries = (f"{person}周会餐饮安排有什么需要注意的例外？", f"请确认{person}周会餐饮的特别情况。",
               f"{person}周会餐饮方案有哪些限制？", f"安排周会餐饮时，{person}有什么饮食限制？")
    negatives = (
        ("unrelated", f"What is the calibration frequency of spectrometer ZX-{900 + index}?"),
        ("unseen-entity", "贺琦对什么食物过敏？"),
        ("unseen-attribute", f"{person}的车辆牌照号码是多少？"),
        ("unseen-event", f"{person}的甲壳虫观察协会秋季研讨会在哪里举办？"),
    )
    kind, negative = negatives[index // 5]
    return StressCase(person, tuple(turns), queries[index // 5], negative, kind)
