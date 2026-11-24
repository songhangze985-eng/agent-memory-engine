"""Conservative pre-write identity and maintenance-time memory decay."""

import re
import unicodedata
from hashlib import sha256

from agentmem_core.linguistics import strip_speaker_prefix
from agentmem_core.textutil import strip_step_tags

DAY = 86400
PROTECTED = "(status='active' AND (memory_type='procedural' OR (memory_type='observation' AND confidence>=0.8) OR is_constraint=1 OR isolated=1))"
_CONSTRAINT = re.compile(r"过敏|禁忌|禁止|不得|不能|切勿|\b(?:allerg\w*|must not|never|prohibited)\b", re.I)
_TABOO = re.compile(
    r"^(?:(?i:I|we|he|she|they)|[A-Z][\w'-]*(?: [A-Z][\w'-]*){0,2})"
    r"(?i:(?:\s+(?:am|is|are)|['’](?:m|s|re))\s+(?:currently\s+)?allergic\s+to\b|"
    r"\s+(?:must avoid|cannot eat|can't eat|must not eat)\b)|"
    r"^(?:我|他|她|[\u4e00-\u9fff]{2,4})(?:现在|一直)?"
    r"(?:对[^，。；!?！？\n]{1,24}过敏|的?(?:饮食|健康)?禁忌(?:是|为)|必须避免|不能(?:吃|食用|饮用))")
_TABOO_HEADING = re.compile(r"^[^。；!?！？\n:：]{0,24}(?:禁忌|allergies|dietary restrictions)\s*[:：]\s*", re.I)
_TABOO_CONTEXT = re.compile(
    r"[?？]|^(?:如果|假如|假设|据说|听说|曾经|以前)|不再对|(?:没|不)\s*过敏|不是对|并非对")
_ALLERGY_ASSERTIONS = (
    re.compile(r"(?P<subject>I|[A-Z][\w'-]*(?: [A-Z][\w'-]*){0,2})"
               r"(?i:(?:\s+(?:am|is)|['’](?:m|s))\s+(?:(?:currently|no longer|not)\s+)?"
               r"allergic\s+to\s+(?P<allergen>[\w-]+(?: [\w-]+){0,4}))[.!！。]?"),
    re.compile(r"(?P<subject>我|[\u4e00-\u9fff]{2,4}?)(?:不再|已经不|现在)?对"
               r"(?P<allergen>[\w-]{1,24}?)(?:不)?过敏[.!！。]?"),
)
CURRENT_STATE_SLOT_PREFIXES = ("allergy:", "preference:")
LEGACY_PREFERENCE_SUMMARY = "多轮明确表达偏好"
_EXPLICIT_PREFERENCE = (
    re.compile(r"(?P<subject>[a-z][\w'-]*(?: [a-z][\w'-]*){0,2}?)\s+(?:(?:now|really|currently)\s+)?"
               r"(?:(?P<avoid>dislikes?|(?:don't|don’t|doesn't|doesn’t|do not|does not)\s+(?:like|enjoy)|prefers? not to)|"
               r"(?:likes?|prefers?|loves?|enjoys?))\s+(?P<value>[^.?!;\r\n]{1,180})[.!]?", re.I),
    re.compile(r"(?P<subject>我|[\u4e00-\u9fff]{2,4}?)(?:现在|一直)?"
               r"(?:(?P<avoid>不太喜欢|不喜欢|不爱|讨厌)|(?:更|很|最|比较)?(?:偏好|喜欢|爱看|爱听))"
               r"(?P<value>[^。！？；?\r\n]{1,100})[。！]?"),
)
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


def is_taboo(text):
    body = _TABOO_HEADING.sub("", strip_step_tags(text).strip())
    return bool(_TABOO.search(body) and not _TABOO_CONTEXT.search(body))


def allergy_slot(text, speaker):
    """Resolve a whole explicit allergy statement to its speaker's state slot."""
    if not speaker:
        return None
    body = strip_step_tags(text).strip()
    for pattern in _ALLERGY_ASSERTIONS:
        match = pattern.fullmatch(body)
        if match and normalized_content(match["subject"]) in {"i", "我", normalized_content(speaker)}:
            return "allergy:" + normalized_content(match["allergen"])
    return None


def explicit_preference(text, speaker):
    body = strip_speaker_prefix(strip_step_tags(text).strip(), speaker)
    if "?" in body or "？" in body:
        return None
    subjects = {"i", "我"}
    if speaker:
        subjects.add(normalized_content(speaker))
    for pattern in _EXPLICIT_PREFERENCE:
        match = pattern.fullmatch(body)
        if match and normalized_content(match["subject"]) in subjects:
            return {"value": match["value"].strip(), "polarity": "negative" if match["avoid"] else "positive"}
    return None


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
