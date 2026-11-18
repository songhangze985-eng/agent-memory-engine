"""Asymptotic distance to the 2D origin.

``d = max(eps, raw_dist)`` with ``eps = 0.05``. Raw distance is not Euclidean:
the origin is a symbolic ``(topic, subtopic, anchor)`` triple. See DESIGN.md.
"""

from __future__ import annotations

from memengine.textutil import char_bigrams, jaccard


def asymptotic_distance(raw_dist: float, epsilon: float = 0.05) -> float:
    try:
        raw = float(raw_dist)
    except (TypeError, ValueError):
        return float(epsilon)
    if raw < epsilon:
        return float(epsilon)
    return raw


def compute_raw_dist(fact: dict, origin: dict | None) -> float:
    """Blend surprise with lexical distance to the current origin.

    Isolated lines (single-shot exceptions) are pushed to at least 0.72 so
    they sit outside the default radius 0.45 and the backtrack radius 0.62.
    A raw value of 0 is allowed here; the epsilon floor is applied later.
    """
    surprise = fact.get("surprise")
    try:
        surprise_f = float(0.5 if surprise is None else surprise)
    except (TypeError, ValueError):
        surprise_f = 0.5
    surprise_f = min(1.0, max(0.0, surprise_f))
    if fact.get("layout") == "INSERT_ISOLATED_LINE" or fact.get("isolated"):
        return max(0.72, surprise_f)
    origin_text = ""
    if origin:
        origin_text = " ".join(
            str(origin.get(key, "")) for key in ("x_topic", "y_subtopic", "decision_anchor")
        )
    lexical = 1.0 - jaccard(char_bigrams(fact.get("text") or ""), char_bigrams(origin_text))
    return min(1.0, max(0.0, 0.5 * lexical + 0.5 * surprise_f))
