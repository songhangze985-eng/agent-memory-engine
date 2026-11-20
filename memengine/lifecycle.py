"""Conservative pre-write identity and maintenance-time memory decay."""

import re
import unicodedata
from hashlib import sha256

from memengine.textutil import strip_step_tags

DAY = 86400
PROTECTED = "(status='active' AND (memory_type='procedural' OR (memory_type='observation' AND confidence>=0.8) OR is_constraint=1 OR isolated=1))"
_CONSTRAINT = re.compile(r"过敏|禁忌|禁止|不得|不能|切勿|\b(?:allerg\w*|must not|never|prohibited)\b", re.I)
_DECISIVE = re.compile(r"(?<!\w)[+-]?\d[\d.,:/-]*\d|\d|[一二两三四五六七八九十]+(?=[年月日点分周次元])|"
                       r"周[一二三四五六日天]|上午|下午|昨天|今天|明天|去年|上周|下周|不再|没有|不是|取消|撤销|禁止|不|"
                       r"\b(?:not|never|no|without|cancel\w*|yesterday|today|tomorrow|monday|tuesday|"
                       r"wednesday|thursday|friday|saturday|sunday|am|pm|enabled|disabled|active|inactive)\b", re.I)
_ACTION = re.compile(r"取消|撤销|忘掉|忘记|删除|不再|改为|改到|更改|\b(?:forget|remove|delete|cancel|no longer|changed to)\b", re.I)


def normalized_content(text):
    return " ".join(unicodedata.normalize("NFKC", strip_step_tags(text)).casefold().split()).rstrip("。.!！ ")


def content_hash(text):
    return sha256(normalized_content(text).encode()).hexdigest()


def is_constraint(text):
    return bool(_CONSTRAINT.search(text))


def changes_state(text):
    return bool(_ACTION.search(text))


def matching_attributes(left, right):
    return _DECISIVE.findall(normalized_content(left)) == _DECISIVE.findall(normalized_content(right))


def resolve_subject(text, previous, extract):
    if not re.match(r"^(?:他|她|\b(?:he|she)\b)", text, re.I):
        return text
    people = {key.split(":", 1)[1] for fact in extract(previous) for key in fact.get("zkeys", ())
              if key.startswith(("人物:", "person:"))}
    if len(people) != 1:
        return text
    return re.sub(r"^(?:他|她|he\b|she\b)", lambda _: next(iter(people)), text, count=1, flags=re.I)


def decay_factor(fact, as_of, half_life_days):
    if as_of is None or (fact["status"] == "active" and (
            fact["memory_type"] == "procedural" or (fact["memory_type"] == "observation" and fact["confidence"] >= 0.8)
            or fact["is_constraint"] or fact["isolated"])):
        return 1.0
    accessed = fact["last_accessed_at"]
    elapsed = max(0, as_of - (fact["ingest_time"] if accessed is None else accessed)) / DAY
    return 2 ** (-elapsed / half_life_days)
