"""Request-scoped cross-encoder candidate accounting."""
from memengine.reranker import checked_scores


PAIR_TOKENS = 250
BATCH_SIZE = 10


def shortlist(candidates, limit):
    pinned = [index for index, row in enumerate(candidates) if row.get("exact_ident_hit")]
    head = candidates[:min(5, limit)]
    slots = {(row["canonical_entity_id"], row["attribute_slot"]) for row in head
             if row.get("canonical_entity_id") and row.get("attribute_slot")}
    peers = [index for index, row in enumerate(candidates)
             if (row.get("canonical_entity_id"), row.get("attribute_slot")) in slots]
    indexes = list(dict.fromkeys([*pinned, *range(len(head)), *peers, *range(len(candidates))]))
    return sorted(indexes[:limit])


class RerankBudget:
    def __init__(self, limit):
        self.limit = limit
        self.remaining = limit
        self._scores = {}

    def score(self, query, candidates, score_pairs):
        keys = [(query, row["id"], row["plain"]) for row in candidates]
        missing = [index for index, key in enumerate(keys) if key not in self._scores]
        available = [candidates[index] for index in missing]
        positions = [missing[index] for index in shortlist(available, self.remaining)]
        selected = [candidates[index] for index in positions]
        if selected:
            scores = checked_scores(score_pairs(query, selected), len(selected))
            self._scores.update((keys[index], value) for index, value in zip(positions, scores))
            self.remaining -= len(positions)
        # The wide plan adds deterministic evidence for every candidate after this step.
        return [self._scores.get(key, 0.0) for key in keys], [row["id"] for row in selected]
