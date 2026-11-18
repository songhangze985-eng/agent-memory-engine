"""Scoring helpers. Definitions are the ones used in eval/results."""

from __future__ import annotations

import math
import re


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * p
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def answer_correct(response: str, question: dict) -> bool:
    text = response or ""
    for keyword in question.get("keywords") or []:
        if keyword not in text:
            return False
    for banned in question.get("forbidden") or []:
        if banned and banned in text:
            return False
    return True


def affirms(text: str, assertion: str) -> bool:
    """True when ``assertion`` appears without a nearby negation."""
    if not assertion or assertion not in text:
        return False
    start = 0
    while True:
        index = text.find(assertion, start)
        if index < 0:
            return False
        prefix = text[max(0, index - 8) : index]
        if not re.search(r"[不没无非]", prefix):
            return True
        start = index + len(assertion)


def trap_hallucinated(response: str, question: dict) -> bool:
    text = response or ""
    if affirms(text, question.get("false_assertion") or ""):
        return True
    for token in question.get("protected") or []:
        if token and token in text:
            return True
    return False
