"""Rule extractor for objective facts.

The spec's three prompts start from an already extracted fact list. v1 does
that step with patterns instead of a fourth model call. Sentences that match
nothing are not facts (the engine may archive them as low-weight news).
"""

from __future__ import annotations

import re

from agentmem_core.categories import attach_dual_zkeys
from agentmem_core.textutil import split_sentences

_GREETING = re.compile(
    r"^(你好|您好|嗨|早上好|晚上好|哈哈|呵呵|谢谢|再见|周末愉快|辛苦啦|最近还好吗|吃过饭了吗|今天天气真好|好的|在吗|"
    r"hi|hello|hey|thanks|thank you|ok|okay|goodbye|bye)"
    r"[啊呀呢吧!！。.~～\s]*$",
    re.IGNORECASE,
)
_STOP = frozenset(
    "the and for are was were you your that this with from have has had not can but "
    "his her its our their what which who when where why how about into over than then".split()
)
_LATIN_FACT = re.compile(r"\b[A-Za-z]{3,}\b")


def is_greeting(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    if _GREETING.match(stripped):
        return True
    return all(_GREETING.match(part) for part in split_sentences(stripped))


def _fact(text, zkeys, attribute, value, topic, isolated) -> dict:
    body = text.strip()
    return {
        "text": body,
        "zkeys": attach_dual_zkeys(zkeys, body),
        "attribute": attribute,
        "value": str(value),
        "topic": topic,
        "isolated": bool(isolated),
        "is_greeting": False,
    }


def _from_sentence(sentence: str) -> dict | None:
    allergy_revoked = re.search(
        r"(?P<person>[\u4e00-\u9fa5]{2,3})(?:不再|已经不)对(?P<value>[^，。]{1,16})过敏", sentence
    )
    if allergy_revoked:
        person, value = allergy_revoked.group("person", "value")
        return {**_fact(sentence, [f"人物:{person}"], "过敏", value, "健康禁忌", False), "delete": True}
    forget = re.search(r"(?:请)?忘掉(?:(?P<person>[\u4e00-\u9fa5]{2,3})的)?(?P<attr>门禁密码|保险柜密码|工卡密码|备用银行卡密码)$", sentence)
    if forget:
        person, attribute = forget.group("person", "attr")
        keys = [f"人物:{person}"] if person else [f"事件:{attribute}"]
        return {**_fact(sentence, keys, attribute, "", "私密凭据", False), "delete": True}
    if re.search(r"取消(?:这次|本次|本周|下周|的)?周会|周会(?:已|被)?取消", sentence):
        return {**_fact(sentence, ["事件:周会"], "周会时间", "", "工作安排", False), "delete": True}
    allergy = re.search(
        r"(?P<person>[\u4e00-\u9fa5]{2,3})对(?P<value>[\u4e00-\u9fa5A-Za-z0-9]{1,16})过敏",
        sentence,
    )
    if allergy:
        person, value = allergy.group("person"), allergy.group("value")
        fact = _fact(
            sentence,
            [f"人物:{person}", f"禁忌:{value}"],
            "过敏",
            value,
            "健康禁忌",
            True,
        )
        return {**fact, "is_conflict_or_override": "不能" in sentence or "禁止" in sentence}

    secret = re.search(
        r"(?P<person>[\u4e00-\u9fa5]{2,3})的(?P<attr>门禁密码|保险柜密码|工卡密码|备用银行卡密码)是\s*(?P<value>[0-9]{3,8})",
        sentence,
    )
    if secret:
        person, attr, value = secret.group("person"), secret.group("attr"), secret.group("value")
        return {**_fact(sentence, [f"人物:{person}", f"事件:{attr}"], attr, value, "私密凭据", True),
                "is_conflict_or_override": True}

    desk = re.search(
        r"(?P<person>[\u4e00-\u9fa5]{2,3})的备用工位在(?P<value>[\u4e00-\u9fa5A-Za-z0-9]{1,16})",
        sentence,
    )
    if desk:
        person, value = desk.group("person"), desk.group("value")
        return _fact(sentence, [f"人物:{person}", f"地点:{value}"], "备用工位", value, "工位", True)

    located = re.search(
        r"(?P<obj>仓库备用钥匙|南栈钥匙|备用印章|旧档案箱)放在(?P<value>[\u4e00-\u9fa5A-Za-z0-9]{1,16})",
        sentence,
    )
    if located:
        obj, value = located.group("obj"), located.group("value")
        return {**_fact(sentence, [f"事件:{obj}", f"地点:{value}"], "存放位置", value, "物品位置", True),
                "is_conflict_or_override": True}

    internal = re.search(
        r"(?P<event>[\u4e00-\u9fa5]{2,12})的内部代号是(?P<value>[\u4e00-\u9fa5A-Za-z0-9]{1,8})",
        sentence,
    )
    if internal:
        event, value = internal.group("event"), internal.group("value")
        return {**_fact(sentence, [f"事件:{event}"], "代号", value, "项目代号", True),
                "is_conflict_or_override": True}

    code = re.search(
        r"(?P<event>[\u4e00-\u9fa5]{2,12})代号是(?P<value>[\u4e00-\u9fa5A-Za-z0-9]{1,8})",
        sentence,
    )
    if code:
        event, value = code.group("event"), code.group("value")
        return _fact(sentence, [f"事件:{event}"], "代号", value, "项目代号", True)

    passphrase = re.search(
        r"(?P<event>[\u4e00-\u9fa5]{2,12})口令是(?P<value>[\u4e00-\u9fa5A-Za-z0-9]{1,8})",
        sentence,
    )
    if passphrase:
        event, value = passphrase.group("event"), passphrase.group("value")
        return _fact(sentence, [f"事件:{event}"], "口令", value, "私密凭据", True)

    trip = re.search(
        r"(?P<person>[\u4e00-\u9fa5]{2,3})将前往(?P<place>[\u4e00-\u9fa5]{2,8})参加(?P<event>[\u4e00-\u9fa5]{2,12})",
        sentence,
    )
    if trip:
        person, place, event = trip.group("person"), trip.group("place"), trip.group("event")
        return _fact(
            sentence,
            [f"人物:{person}", f"地点:{place}", f"事件:{event}"],
            "行程",
            place,
            "出行",
            False,
        )

    budget_change = re.search(
        r"(?P<event>[\u4e00-\u9fa5]{2,12})的预算改为(?P<value>[0-9]+)",
        sentence,
    )
    if budget_change:
        event, value = budget_change.group("event"), budget_change.group("value")
        return _fact(sentence, [f"事件:{event}"], "预算", value, "出行", False)

    budget = re.search(
        r"(?P<event>[\u4e00-\u9fa5]{2,12})的预算为(?P<value>[0-9]+)",
        sentence,
    )
    if budget:
        event, value = budget.group("event"), budget.group("value")
        return _fact(sentence, [f"事件:{event}"], "预算", value, "出行", False)

    meeting = re.search(r"每周(?P<value>[周一二三四五六日天][^，。]{0,12})开周会", sentence)
    if meeting:
        value = meeting.group("value")
        return _fact(sentence, ["事件:周会"], "周会时间", value, "工作安排", False)

    moved = re.search(r"周会改到(?P<value>[^，。]{2,20})", sentence)
    if moved:
        return _fact(sentence, ["事件:周会"], "周会时间", moved.group("value"), "工作安排", False)

    movie = re.search(r"(?:电影|影片|看过|看了|喜欢)\s*《([^》]+)》", sentence)
    if movie:
        title = movie[1]
        return _fact(sentence, [f"实体:{title}", "类目:影视"], f"影片:{title}", sentence, "影视", False)

    return None


def english_entity_keys(text: str) -> list[str]:
    """Only proper names are graph keys; generic terms stay in the vector channel."""
    keys = []
    seen = set()
    for name in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", text or ""):
        if name.lower() in _STOP:
            continue
        key = f"person:{name}"
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def english_turn_fact(text: str) -> dict | None:
    raw = (text or "").strip()
    if not raw or re.search(r"[\u4e00-\u9fff]", raw):
        return None
    keys = english_entity_keys(raw)
    if not _LATIN_FACT.search(raw):
        return None
    step = re.search(r"\[step (\d+)\]", raw)
    attribute = f"turn:{step.group(1)}" if step else ""
    return _fact(raw, keys, attribute, raw[:80], "dialogue", False)


def extract_facts(text: str) -> list[dict]:
    if is_greeting(text):
        return []
    found = []
    seen = set()
    for sentence in split_sentences(text):
        parts = re.split(r"[，,；;]", sentence)
        separate = [fact for part in parts if (fact := _from_sentence(part.strip()))] if len(parts) > 1 else []
        batch = separate if len(separate) > 1 else [_from_sentence(sentence)]
        for fact in batch:
            if fact is None:
                continue
            marker = (fact["attribute"], fact["value"], tuple(fact["zkeys"]), bool(fact.get("delete")))
            if marker in seen:
                continue
            seen.add(marker)
            found.append(fact)
    if found or re.search(r"[\u4e00-\u9fff]", text or ""):
        return found
    fallback = english_turn_fact(text or "")
    return [fallback] if fallback else []
