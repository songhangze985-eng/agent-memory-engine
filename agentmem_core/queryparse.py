"""Parse a question stem into entities, controlled categories, and a vector query."""

from __future__ import annotations

import re

from agentmem_core.categories import (
    category_synonyms,
    extract_explicit_entities,
    map_categories,
)
from agentmem_core.retrieval.coverage import explicit_facets

_REFLECTIVE = re.compile(
    r"\b(prefer|preference|kind of|type of|what kind|probably|might|overall|in common|personality|habit|style)\b"
    r"|共同点|偏好|性格|习惯|风格|可能喜欢|更喜欢|爱看.{0,5}(?:啥|什么)|喜欢.{0,5}(?:啥|什么)",
    re.I,
)
_PREFERENCE_QUERY = re.compile(
    r"\b(?:prefers?|preferences?|dislikes?|recommend(?:ation|ations)?|suggest(?:ion|ions)?)\b|"
    r"\b(?:what|which)\b[^?.!]{0,64}\b(?:like|likes|enjoy|enjoys)\b|偏好|喜好|推荐|更喜欢|"
    r"(?:喜欢|爱看|爱听).{0,5}(?:啥|什么|哪)", re.I)
_CJK = re.compile(r"[\u4e00-\u9fff]")
_TIME_TRAVEL = re.compile(r"最早|原本|以前|上次|之前|\b(?:previously|originally|earlier|last time|used to|before)\b", re.I)


def history_language(text: str) -> str:
    return "zh" if _CJK.search(text or "") else "en"


def expand_semantic_query(stem: str, categories: list[str]) -> str:
    language = history_language(stem)
    parts = [stem.strip()]
    for key in categories:
        extra = category_synonyms(key, language=language)
        if extra:
            parts.append(extra)
    return "\n".join(part for part in parts if part)


def parse_query_rules(stem: str, *, categories=()) -> dict:
    text = (stem or "").strip()
    keys = map_categories(text, limit=3, categories=categories)
    if _TIME_TRAVEL.search(text):
        query_type = "TIME_TRAVEL"
    else:
        query_type = "REFLECTIVE" if _REFLECTIVE.search(text) else "FACTUAL"
    return {
        "query_type": query_type,
        "preference_query": bool(_PREFERENCE_QUERY.search(text)),
        "facets": explicit_facets(text),
        "entities": extract_explicit_entities(text),
        "categories": keys,
        "semantic_vector_query": expand_semantic_query(text, keys),
    }
