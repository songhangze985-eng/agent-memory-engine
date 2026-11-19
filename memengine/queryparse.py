"""Parse a question stem into entities, controlled categories, and a vector query."""

from __future__ import annotations

import re

from memengine.categories import (
    CATEGORY_ENUM,
    attach_dual_zkeys,
    category_synonyms,
    extract_explicit_entities,
    map_categories,
    validate_category_keys,
)

_REFLECTIVE = re.compile(
    r"\b(prefer|preference|kind of|type of|what kind|probably|might|overall|in common|personality|habit|style)\b"
    r"|共同点|偏好|性格|习惯|风格|可能喜欢|更喜欢",
    re.I,
)
_CJK = re.compile(r"[\u4e00-\u9fff]")


def history_language(text: str) -> str:
    return "zh" if _CJK.search(text or "") else "en"


def expand_semantic_query(stem: str, categories: list[str]) -> str:
    language = history_language(stem)
    parts = [stem.strip()]
    for key in categories:
        extra = category_synonyms(key, language=language)
        if extra:
            parts.append(extra)
    return " ".join(part for part in parts if part)


def parse_query_rules(stem: str) -> dict:
    text = (stem or "").strip()
    categories = map_categories(text, limit=3)
    query_type = "REFLECTIVE" if _REFLECTIVE.search(text) else "FACTUAL"
    return {
        "query_type": query_type,
        "entities": extract_explicit_entities(text),
        "categories": categories,
        "semantic_vector_query": expand_semantic_query(text, categories),
    }


def normalize_parsed_query(data: dict | None, stem: str) -> dict:
    fallback = parse_query_rules(stem)
    raw = data if isinstance(data, dict) else {}
    query_type = str(raw.get("query_type") or fallback["query_type"]).upper()
    if query_type not in {"REFLECTIVE", "FACTUAL"}:
        query_type = fallback["query_type"]
    categories = validate_category_keys(raw.get("categories") or [], fallback_text=stem, limit=3)
    if not categories:
        categories = list(fallback["categories"])
    entities = []
    seen = set()
    for item in raw.get("entities") or []:
        key = str(item).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        entities.append(key)
    # The parser must not invent titles: if the model invented an entity
    # that is not a visible proper noun and not in the stem, drop it.
    stem_l = (stem or "").lower()
    kept = []
    for key in entities:
        name = key.split(":", 1)[-1]
        if name and name.lower() in stem_l:
            kept.append(key)
    if not kept:
        kept = list(fallback["entities"])
    semantic = str(raw.get("semantic_vector_query") or "").strip() or fallback["semantic_vector_query"]
    if not semantic:
        semantic = expand_semantic_query(stem, categories)
    return {
        "query_type": query_type,
        "entities": kept,
        "categories": categories,
        "semantic_vector_query": semantic,
    }



_CHOICE_LINE = re.compile(r"(?m)^\s*[ABCD][\.\)]\s+.*$")
_REPLY_LETTER = re.compile(r"(?im)^\s*Reply with only one letter:.*$")
_TIME_PREFIX = re.compile(r"^\(current time is [^)]*\)\s*")


def strip_query_options(text: str) -> str:
    """Remove MCQ choices and letter instructions from a possibly-formatted question."""
    raw = _REPLY_LETTER.sub("", text or "")
    raw = _CHOICE_LINE.sub("", raw)
    raw = _TIME_PREFIX.sub("", raw.strip())
    lines = [line.rstrip() for line in raw.splitlines() if line.strip()]
    return "\n".join(lines).strip()


# Re-exported so write-side callers have one import path.
__all__ = [
    "CATEGORY_ENUM",
    "attach_dual_zkeys",
    "expand_semantic_query",
    "history_language",
    "normalize_parsed_query",
    "parse_query_rules",
]
