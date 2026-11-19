"""Step 1: millisecond rule router.

Chinese cues follow the spec list. A one-character cue is ignored when it
only occurs inside a longer word (别 inside 特别). English cues use word
boundaries. Entity names match normalized tokens, not raw substrings, so
``ann`` does not hit ``anniversary``. CJK names still match as a substring
because those queries are not space-separated.
"""

from __future__ import annotations

import re

from memengine.textutil import tokenize

_MUTATION_ZH = re.compile(r"(改|换|推迟|提前|取消|其实|不过|但是|别|不要|冲突)")
_TEMPORAL_ZH = re.compile(r"(之前|上次|原本|以前|最早|后来|目前|现在|对比)")
_DECISION_ZH = re.compile(r"(安排|规划|预算|推荐|选择|怎么办|帮我|方案)")
_MUTATION_EN = re.compile(
    r"\b(change|changed|reschedule|cancel|canceled|cancelled|instead|but|don't|do not|conflict|postpone|postponed|update|updated)\b",
    re.IGNORECASE,
)
_TEMPORAL_EN = re.compile(
    r"\b(before|after|previously|originally|earlier|later|currently|last time|used to)\b",
    re.IGNORECASE,
)
_DECISION_EN = re.compile(
    r"\b(plan|schedule|budget|recommend|choose|choice|help me|option|what should)\b",
    re.IGNORECASE,
)
_QUESTION_EN = re.compile(r"\b(what|which|who|whom|whose|when|where|why|how)\b", re.IGNORECASE)
_MCQ_LINE = re.compile(r"(?m)^\s*[ABCD][\.\)]\s+\S")
_LATIN = re.compile(r"[A-Za-z0-9]{3,}")
_STOP = frozenset(
    "the and for are was were you your that this with from have has had not can but".split()
)
# 别 inside these words is not the negation cue.
_BI_WORDS = ("特别", "区别", "别人", "性别", "类别", "级别", "别的", "别墅", "鉴别", "差别")

TASK = "TASK_POISON_CIRCLE"
CASUAL = "CASUAL_2D_BYPASS"


def entity_names(key: str) -> str:
    return key.split(":", 1)[1] if ":" in key else key


def _cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def latin_tokens(text: str) -> set[str]:
    return {tok.lower() for tok in _LATIN.findall(text or "") if tok.lower() not in _STOP}


def _token_forms(token: str) -> set[str]:
    """Light plural folding so ``movies`` matches ``movie``."""
    folded = (token or "").lower()
    forms = {folded}
    if len(folded) > 4 and folded.endswith("ies"):
        forms.add(folded[:-3] + "y")
    if len(folded) > 3 and folded.endswith("s") and not folded.endswith("ss"):
        forms.add(folded[:-1])
    return forms


def _name_matches(name: str, query: str, query_tokens: set[str]) -> bool:
    if not name:
        return False
    if _cjk(name):
        return len(name) >= 2 and name in query
    parts = latin_tokens(name)
    if not parts:
        folded = re.sub(r"[^a-z0-9]+", "", name.lower())
        query_forms = set().union(*(_token_forms(tok) for tok in query_tokens)) if query_tokens else set()
        return len(folded) >= 3 and (folded in query_tokens or folded in query_forms)
    query_forms = set().union(*(_token_forms(tok) for tok in query_tokens)) if query_tokens else set()
    return all(_token_forms(part) & query_forms for part in parts)


def entity_hit(query: str, entity_keys) -> bool:
    tokens = latin_tokens(query)
    for key in entity_keys:
        if _name_matches(entity_names(key), query, tokens):
            return True
    return False


def match_entities(query: str, entity_keys) -> list[str]:
    tokens = latin_tokens(query)
    hits = []
    for key in sorted(entity_keys):
        if _name_matches(entity_names(key), query, tokens):
            hits.append(key)
    return hits


def _zh_trigger(pattern: re.Pattern, text: str) -> bool:
    for match in pattern.finditer(text):
        if match.group(0) != "别":
            return True
        start = match.start()
        hidden = False
        for word in _BI_WORDS:
            at = word.find("别")
            begin = start - at
            if begin >= 0 and text[begin : begin + len(word)] == word:
                hidden = True
                break
        if not hidden:
            return True
    return False


def route_query(query: str, entity_keys) -> str:
    """Return TASK_POISON_CIRCLE or CASUAL_2D_BYPASS."""
    query = query or ""
    tokenize(query)  # spec tokenization step; matching itself is boundary-aware
    hit = entity_hit(query, entity_keys)
    mcq = len(_MCQ_LINE.findall(query)) >= 2
    question = len(query) > 12 and bool(_QUESTION_EN.search(query))
    syntax_hit = bool(
        _zh_trigger(_MUTATION_ZH, query)
        or _zh_trigger(_TEMPORAL_ZH, query)
        or (len(query) > 12 and _zh_trigger(_DECISION_ZH, query))
        or _MUTATION_EN.search(query)
        or _TEMPORAL_EN.search(query)
        or (len(query) > 12 and _DECISION_EN.search(query))
        or mcq
        or question
    )
    if hit or syntax_hit:
        return TASK
    return CASUAL
