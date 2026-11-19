"""Small text helpers shared by routing, distance, and eval."""

from __future__ import annotations

import re

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
