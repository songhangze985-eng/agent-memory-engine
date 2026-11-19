"""Step 1: millisecond rule router.

The trigger regexes follow the spec pseudocode. Entity hit extends the
pseudocode's full-key substring test with the entity name after the colon,
because stored keys look like ``人物:老王`` while users say ``老王``.
"""

from __future__ import annotations

import re

from memengine.textutil import tokenize

MUTATION_PATTERN = re.compile(r"(改|换|推迟|提前|取消|其实|不过|但是|别|不要|冲突)")
TEMPORAL_PATTERN = re.compile(r"(之前|上次|原本|以前|最早|后来|目前|现在|对比)")
DECISION_PATTERN = re.compile(r"(安排|规划|预算|推荐|选择|怎么办|帮我|方案)")

TASK = "TASK_POISON_CIRCLE"
CASUAL = "CASUAL_2D_BYPASS"


def entity_names(key: str) -> str:
    return key.split(":", 1)[1] if ":" in key else key


def entity_hit(query: str, entity_keys) -> bool:
    tokens = tokenize(query)
    for key in entity_keys:
        name = entity_names(key)
        if not name:
            continue
        if key in query or name in query or name in tokens:
            return True
    return False


def match_entities(query: str, entity_keys) -> list[str]:
    hits = []
    for key in sorted(entity_keys):
        name = entity_names(key)
        if not name:
            continue
        if key in query or name in query:
            hits.append(key)
    return hits


def route_query(query: str, entity_keys) -> str:
    """Return TASK_POISON_CIRCLE or CASUAL_2D_BYPASS."""
    query = query or ""
    # Retained from the spec: tokens are the lexical view of the utterance.
    tokens = tokenize(query)
    hit = entity_hit(query, entity_keys) or any(
        entity_names(key) in tokens for key in entity_keys if entity_names(key)
    )
    syntax_hit = bool(
        MUTATION_PATTERN.search(query)
        or TEMPORAL_PATTERN.search(query)
        or (len(query) > 12 and DECISION_PATTERN.search(query))
    )
    if hit or syntax_hit:
        return TASK
    return CASUAL
