"""Lightweight deterministic stemming and content-term tokenization shared across core modules."""
from __future__ import annotations

import re

from memengine.textutil import strip_step_tags

_WORD = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")
_STOP = frozenset(
    "a an the and or but in on at to for of with by from as is are was were be been being "
    "i you he she it we they my your his her its our their me him us them this that these those "
    "do does did have has had would could should will can may might must about into over after before "
    "what which who whom whose when where why how tell please give show know remember recall "
    "user assistant said mentioned told one some any many much very just really also still "
    "的 了 是 在 我 你 他 她 它 们 这 那 什么 哪个 哪些 怎么 为什么 多少 几 吗 呢 啊 吧".split()
)


def light_stem(word: str) -> str:
    """Lightweight deterministic English suffix normalizer (fixes graduate/graduated mismatch)."""
    w = word.casefold()
    if len(w) <= 3 or not w.isascii() or w.isdigit():
        return w
    for suffix, repl in (
        ("ies", "y"),
        ("ied", "y"),
        ("ing", ""),
        ("ations", "at"),
        ("ation", "at"),
        ("ments", "ment"),
        ("ness", ""),
        ("ed", ""),
        ("es", ""),
        ("ly", ""),
        ("s", ""),
    ):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            stem = w[: -len(suffix)] + repl
            if len(stem) >= 3:
                return stem
    return w


def stem_terms(text: str) -> set[str]:
    cleaned = strip_step_tags(text or "")
    out: set[str] = set()
    for match in _WORD.finditer(cleaned):
        tok = match.group(0).casefold()
        if tok in _STOP or (len(tok) == 1 and tok.isascii() and not tok.isdigit()):
            continue
        out.add(light_stem(tok))
    return out
