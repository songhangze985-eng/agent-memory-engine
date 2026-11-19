"""Parallel entity-axis expansion and a single far-to-near collapse."""

from __future__ import annotations

from memengine.distance import evaluate_circle
from memengine.store import key_type


def expand_entities(z_index: dict, seeds, hops: int = 2) -> list[str]:
    """Walk shared Z keys up to ``hops`` so a weak cross-axis fact is visible."""
    keys = {key for key in seeds if key_type(key) == "entity"}
    frontier = keys.copy()
    for _ in range(max(0, hops)):
        extra: set[str] = set()
        for key in frontier:
            for fact in z_index.get(key, []):
                for zkey in fact.get("entity_z_keys", fact.get("zkeys", [])):
                    if key_type(zkey) == "entity":
                        extra.add(zkey)
        frontier = extra - keys
        if not frontier:
            break
        keys |= extra
    return sorted(keys)


def order_far_to_near(controller_order, scan_keys) -> list[str]:
    """Keep the controller's far-to-near order; unranked keys stay on the far side."""
    scan = list(scan_keys)
    allowed = set(scan)
    ordered = list(dict.fromkeys(key for key in controller_order if isinstance(key, str) and key in allowed))
    covered = set(ordered)
    missing = [key for key in scan if key not in covered]
    return missing + ordered


def collapse_and_scan_z_axis(z_hash_index: dict, far_to_near: list, radius_r: float, epsilon: float = 0.05) -> list[dict]:
    collapsed_line = []
    seen_fact_ids: set[str] = set()
    for entity in far_to_near:
        bucket = []
        for fact in z_hash_index.get(entity, []):
            fact_id = fact["id"]
            if fact_id in seen_fact_ids:
                continue
            bucket.append(evaluate_circle(fact, radius_r, epsilon))
            seen_fact_ids.add(fact_id)
        if bucket:
            collapsed_line.append({"z_bucket": entity, "facts": bucket})
    return collapsed_line
