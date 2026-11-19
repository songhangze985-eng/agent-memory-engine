"""Frequency coordinates and symmetric asymptotic attraction."""

from __future__ import annotations

import math

EPSILON = 0.05
MIN_REL = 0.15
NEWS_TOPIC = "时事杂讯"
NEWS_KEY = "话题:时事杂讯"
INNER = "INNER_INTUITIVE"
OUTER = "OUTER_ANTI_INTUITIVE"
BELT = "POISON_BELT"


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


def is_noise(fact: dict) -> bool:
    return fact.get("topic") == NEWS_TOPIC or set(fact.get("zkeys") or ()) == {NEWS_KEY}


def consolidation_count(fact: dict) -> int:
    return max(1, len(fact.get("source_step_ids") or ()), len(fact.get("source_turn_ids") or ())) + int(fact.get("hit_count") or 0)


def is_override(fact: dict) -> bool:
    return bool(fact.get("is_conflict") or fact.get("is_conflict_or_override") or fact.get("historical"))


def compute_bipolar_radius(fact: dict, c_max: int) -> float:
    if is_noise(fact):
        return 0.5
    count = consolidation_count(fact)
    if count == 1 or is_override(fact) or fact.get("isolated") or fact.get("layout") in {
        "INSERT_ISOLATED_LINE", "OVERWRITE_SAME_LINE",
    }:
        return 1.0
    if c_max <= 1:
        return 0.0
    return min(0.50, max(0.0, 1.0 - math.log(count) / math.log(c_max)))


def bipolar_weight(r: float, eps_in: float = EPSILON, eps_out: float = EPSILON) -> float:
    if not 0 < eps_in <= 1 or not 0 < eps_out <= 1:
        raise ValueError("epsilon must be in (0, 1]")
    radial = unit_interval(r)
    return max(eps_in / max(eps_in, radial), eps_out / max(eps_out, 1.0 - radial))


def inner_boundary(radius: float) -> float:
    return min(0.35, max(0.15, radius))


def score_bipolar(fact: dict, *, c_max: int, rrf_max: float, radius: float,
                  bipolar_lambda: float, epsilon: float = EPSILON) -> dict:
    radial = compute_bipolar_radius(fact, c_max)
    weight = bipolar_weight(radial, epsilon, epsilon)
    relevance = fact["rrf_score"] / (rrf_max + 1e-12)
    inner = inner_boundary(radius)
    zone = INNER if radial <= inner else OUTER if radial >= 1.0 - inner else BELT
    bonus = 0.05 if is_override(fact) and not is_noise(fact) else 0.0
    multiplier = bipolar_lambda + (1.0 - bipolar_lambda) * weight
    score = relevance * multiplier + bonus
    return {**fact, "r_m": radial, "bipolar_weight": weight, "rel_norm": relevance,
            "override_bonus": bonus, "score": score, "bipolar_score": score, "zone": zone}
