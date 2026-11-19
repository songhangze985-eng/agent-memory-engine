"""Parallel entity buckets and a single far-to-near collapse."""

from __future__ import annotations

def order_far_to_near(controller_order, scan_keys) -> list[str]:
    """Keep the controller's far-to-near order; unranked keys stay on the far side."""
    scan = list(scan_keys)
    allowed = set(scan)
    ordered = list(dict.fromkeys(key for key in controller_order if isinstance(key, str) and key in allowed))
    covered = set(ordered)
    missing = [key for key in scan if key not in covered]
    return missing + ordered


def collapse_and_scan_z_axis(z_hash_index: dict, far_to_near: list) -> list[dict]:
    collapsed_line = []
    seen_fact_ids: set[str] = set()
    for entity in far_to_near:
        bucket = []
        for fact in z_hash_index.get(entity, []):
            fact_id = fact["id"]
            if fact_id in seen_fact_ids:
                continue
            bucket.append(fact)
            seen_fact_ids.add(fact_id)
        if bucket:
            collapsed_line.append({"z_bucket": entity, "facts": bucket})
    return collapsed_line
