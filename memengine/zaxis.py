"""Step 3: parallel Z-axis collapse and far-to-near reverse scan.

``collapse_and_scan_z_axis`` follows the spec pseudocode: it reverses
``matched_entities``, buckets facts by entity, and never repeats a fact id.
The engine passes entities in near-to-far order so this reverse yields
far-to-near. Facts from different entities stay in different buckets.
"""

from __future__ import annotations

from memengine.distance import asymptotic_distance


def expand_entities(z_index: dict, seeds, hops: int = 2) -> list[str]:
    """Walk shared Z keys up to ``hops`` so a weak cross-axis fact is visible."""
    keys = set(seeds)
    for _ in range(max(0, hops)):
        extra: set[str] = set()
        for key in keys:
            for fact in z_index.get(key, []):
                extra.update(fact.get("zkeys") or [])
        keys |= extra
    return sorted(keys)


def order_far_to_near(controller_order, scan_keys) -> list[str]:
    """Keep the controller's far-to-near order; unranked keys stay on the far side."""
    scan = list(scan_keys)
    allowed = set(scan)
    ordered = [key for key in controller_order if key in allowed]
    missing = [key for key in scan if key not in ordered]
    return missing + ordered


def collapse_and_scan_z_axis(z_hash_index: dict, matched_entities: list, radius_r: float, epsilon: float = 0.05) -> list[dict]:
    collapsed_line = []
    seen_fact_ids: set[str] = set()
    for entity in reversed(list(matched_entities)):
        bucket = []
        for fact in z_hash_index.get(entity, []):
            fact_id = fact["id"]
            if fact_id in seen_fact_ids:
                continue
            d_asymp = asymptotic_distance(fact.get("raw_dist", epsilon), epsilon)
            zone = "INNER_INTUITIVE" if d_asymp <= radius_r else "OUTER_ANTI_INTUITIVE"
            bucket.append({**fact, "d_asymp": d_asymp, "zone": zone})
            seen_fact_ids.add(fact_id)
        if bucket:
            collapsed_line.append({"z_bucket": entity, "facts": bucket})
    return collapsed_line
