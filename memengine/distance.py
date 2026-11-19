"""Independent semantic distance and structural surprise in the poison circle."""

from __future__ import annotations

import math

EPSILON = 0.05
MIN_REL = 0.15
INNER = "INNER_INTUITIVE"
OUTER = "OUTER_ANTI_INTUITIVE"
DROPPED = "POISONED_DROPPED"


def unit_interval(value, default=0.0):
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return min(1.0, max(0.0, parsed)) if math.isfinite(parsed) else default


def asymptotic_distance(raw_dist: float, epsilon: float = EPSILON) -> float:
    return max(epsilon, unit_interval(raw_dist))


def semantic_origin_text(origin):
    return " ".join(str(origin.get(key) or "") for key in ("x_topic", "y_subtopic", "decision_anchor")).strip()


def compute_raw_dist(fact: dict, origin: dict | None) -> float:
    from memengine.vector import HashedTfidfEmbedder

    hit = HashedTfidfEmbedder().search(semantic_origin_text(origin or {}), [fact], 1)[0]
    return 1.0 - unit_interval(hit["vector_score"])


def evaluate_circle(fact, radius, epsilon=EPSILON):
    distance = asymptotic_distance(fact.get("raw_dist", 1.0), epsilon)
    exceptional = fact.get("layout") == "INSERT_ISOLATED_LINE" or fact.get("isolated")
    conflict = fact.get("is_conflict_or_override") or fact.get("historical")
    if conflict:
        surprise = 0.95
    elif fact.get("topic") == "时事杂讯":
        surprise = 0.08
    elif exceptional:
        surprise = 0.90
    elif fact.get("layout") == "OVERWRITE_SAME_LINE":
        surprise = 0.85
    else:
        surprise = 0.25
    related = conflict and fact.get("historical") or fact.get("entity_anchor") or fact.get("vector_score", 0) > MIN_REL
    zone = DROPPED
    if related and (conflict or distance > radius and surprise >= 0.75):
        zone = OUTER
    elif related and distance <= radius and not exceptional:
        zone = INNER
    return {**fact, "d_asymp": distance, "p_radial": epsilon / distance, "surprise": surprise, "zone": zone}


def score_fact(fact, now, intuitive_weight, anti_intuitive_weight):
    age = max(0.0, now - float(fact.get("event_time") if fact.get("event_time") is not None else fact.get("updated_at", now)))
    recency = 2 ** (-age / (30 * 86400))
    rrf = fact.get("rrf_norm", 0.0)
    if fact["zone"] == INNER:
        return intuitive_weight * (0.6 * rrf + 0.4 * fact["p_radial"]) + 0.15 * recency
    return anti_intuitive_weight * (0.45 * rrf + 0.55 * fact["surprise"]) + 0.15 * recency
