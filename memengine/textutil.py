"""Small text helpers shared by routing, distance, and eval."""

from __future__ import annotations

import re
import unicodedata

_CN_NUM = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def tokenize(text: str) -> set[str]:
    """Spec router tokenization: word-ish spans, including CJK."""
    return set(re.findall(r"[\w\u4e00-\u9fa5]+", text or ""))


def char_bigrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text or "")
    if len(compact) < 2:
        return set(compact)
    return {compact[i : i + 2] for i in range(len(compact) - 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def estimate_tokens(text: str) -> int:
    """CJK character ≈ 1 token; other characters ≈ 4 per token."""
    if not text:
        return 0
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def parse_cn_number(raw: str) -> int:
    if raw.isdigit():
        return int(raw)
    return _CN_NUM.get(raw, 1)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"[。！？!\n]+", text or "")
    return [part.strip() for part in parts if part.strip()]


_STEP_TAG = re.compile(r"\[step(?:\s+|:\s*)(\d+)\]", re.I)


def strip_step_tags(text: str) -> str:
    return _STEP_TAG.sub("", text).strip()


def normalize_entity(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw).strip()
    prefix, separator, value = text.partition(":")
    if separator and prefix.strip().casefold() in {"实体", "人物", "类目", "话题", "禁忌", "事件", "地点",
                                            "entity", "person", "category", "topic", "tag", "place", "event"}:
        text = value
    return re.sub(r"\s+", "", text).casefold()


def tokenize_for_fts(text: str) -> str:
    terms = []
    for token in re.findall(r"[\u3400-\u9fff]+|[A-Za-z0-9_.\-/]+", strip_step_tags(text)):
        if "\u3400" <= token[0] <= "\u9fff":
            terms.extend(token)
            terms.extend(token[index:index + 2] for index in range(len(token) - 1))
        else:
            token = token.strip(".-/").lower()
            if any(char.isalnum() for char in token):
                terms.append(token)
    return " ".join(terms)


def step_ids_in(text: str) -> list[int]:
    """Preserve first-seen order of ``[step N]`` markers."""
    found: list[int] = []
    for match in _STEP_TAG.finditer(text or ""):
        step = int(match.group(1))
        if step not in found:
            found.append(step)
    return found


def merge_step_ids(*groups) -> list[int]:
    found: list[int] = []
    for group in groups:
        for step in group or []:
            step = int(step)
            if step not in found:
                found.append(step)
    return found


def ensure_step_tags(text: str, ids) -> str:
    """Make sure every source step id appears as a ``[step N]`` tag in text."""
    raw = text or ""
    missing = [int(step) for step in (ids or []) if f"[step {int(step)}]" not in raw]
    if not missing:
        return raw
    prefix = " ".join(f"[step {step}]" for step in missing)
    return f"{prefix} {raw}".strip()
