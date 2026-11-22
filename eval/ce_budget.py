"""Bound CE work in paired offline retrieval runs."""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from unittest.mock import patch


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


@contextmanager
def limit_cross_encoder(engine, pair_limit, scored_ids):
    reranker = engine.reranker
    backend = reranker._encoder.model
    tokenizer = deepcopy(backend.tokenizer)
    truncation = tokenizer.truncation or {}
    max_length = min(PAIR_TOKENS, truncation.get("max_length", PAIR_TOKENS))
    tokenizer.enable_truncation(max_length=max_length, direction=truncation.get("direction", "right"))
    score_all = reranker.score

    def score(query, candidates):
        remaining = max(0, pair_limit - sum(map(len, scored_ids)))
        indices = shortlist(candidates, remaining)
        selected = [candidates[index] for index in indices]
        scored_ids.append([row["id"] for row in selected])
        cross_scores = score_all(query, selected)
        if len(indices) == len(candidates):
            return cross_scores
        # The wide plan blends its deterministic score into every candidate after this call.
        scores = [0.0] * len(candidates)
        for index, cross_score in zip(indices, cross_scores, strict=True):
            scores[index] = cross_score
        return scores

    with ExitStack() as stack:
        stack.enter_context(patch.object(backend, "tokenizer", tokenizer))
        stack.enter_context(patch.object(reranker, "batch_size", min(pair_limit, BATCH_SIZE)))
        stack.enter_context(patch.object(reranker, "score", score))
        yield
