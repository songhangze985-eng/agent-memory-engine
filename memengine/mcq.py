"""Attach one choice letter after a System-2 reply.

The spec reasoner prompt stays a natural-language reply. When the user
question is multiple choice, this wrapper appends a single letter so a
harness can parse it. The letter is the option whose text overlaps the
reply the most; an existing trailing letter is kept.
"""

from __future__ import annotations

import re

from memengine.textutil import char_bigrams, jaccard

_CHOICE_LINE = re.compile(r"(?m)^\s*([ABCD])[\.\)]\s+(.+?)\s*$")


def parse_mcq_choices(query: str) -> dict[str, str]:
    found = {}
    for match in _CHOICE_LINE.finditer(query or ""):
        found[match.group(1)] = match.group(2).strip()
    if len(found) < 2:
        return {}
    return found


def _best_letter(response: str, choices: dict[str, str]) -> str:
    letters = [letter for letter in ("A", "B", "C", "D") if letter in choices]
    if not letters:
        return "A"
    if not (response or "").strip():
        return letters[0]
    base = char_bigrams(response)
    best = letters[0]
    best_score = -1.0
    for letter in letters:
        score = jaccard(base, char_bigrams(choices[letter]))
        if score > best_score:
            best_score = score
            best = letter
    return best


def end_with_choice(query: str, response: str) -> str:
    choices = parse_mcq_choices(query)
    if not choices:
        return response or ""
    text = (response or "").rstrip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and re.fullmatch(r"[ABCD]", lines[-1]) and lines[-1] in choices:
        return text
    letter = _best_letter(text, choices)
    if not text:
        return letter
    return text + "\n" + letter
