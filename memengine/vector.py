"""Pluggable local vector index: hashed TF-IDF over word tokens and char n-grams.

No external embedding service. Vectors are seed-independent (MD5 buckets).
The retrieval path can swap this class for another object that implements
``search(query, facts, top_n)``.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np

_WORD = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", re.UNICODE)
_LATIN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    raw = text or ""
    tokens: list[str] = []
    for match in _WORD.findall(raw):
        if _LATIN.fullmatch(match):
            tokens.append(match.lower())
        else:
            tokens.append(match)
    compact = re.sub(r"\s+", "", raw.lower())
    for width in (2, 3):
        if len(compact) >= width:
            tokens.extend(compact[i : i + width] for i in range(len(compact) - width + 1))
    return tokens


def _bucket(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.md5(token.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "little") % dim
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    return index, sign


class HashedTfidfEmbedder:
    """Sparse hashed TF-IDF, L2-normalized, compared with cosine (dot product)."""

    def __init__(self, dim: int = 512):
        self.dim = int(dim)

    def _tf_counts(self, text: str) -> Counter:
        return Counter(tokenize(text))

    def _idf(self, docs: list[Counter]) -> dict[str, float]:
        n_docs = max(1, len(docs))
        df: Counter = Counter()
        for counts in docs:
            df.update(counts.keys())
        return {term: math.log((1 + n_docs) / (1 + freq)) + 1.0 for term, freq in df.items()}

    def embed(self, text: str, idf: dict[str, float] | None = None) -> np.ndarray:
        weights = idf or {}
        vector = np.zeros(self.dim, dtype=np.float64)
        counts = self._tf_counts(text)
        for term, count in counts.items():
            index, sign = _bucket(term, self.dim)
            vector[index] += sign * (1.0 + math.log(count)) * weights.get(term, 1.0)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def search(self, query: str, facts: list[dict], top_n: int = 30) -> list[dict]:
        if not facts:
            return []
        docs = [self._tf_counts(str(fact.get("text") or "")) for fact in facts]
        idf = self._idf(docs)
        query_vec = self.embed(query, idf)
        scored = []
        for fact, counts in zip(facts, docs):
            vector = np.zeros(self.dim, dtype=np.float64)
            for term, count in counts.items():
                index, sign = _bucket(term, self.dim)
                vector[index] += sign * (1.0 + math.log(count)) * idf.get(term, 1.0)
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector /= norm
            scored.append((float(query_vec @ vector), fact))
        scored.sort(key=lambda item: item[0], reverse=True)
        hits = []
        for score, fact in scored[: max(1, top_n)]:
            item = dict(fact)
            item["vector_score"] = score
            hits.append(item)
        return hits


def unique_facts(z_index: dict) -> list[dict]:
    seen: set[str] = set()
    facts: list[dict] = []
    for bucket in (z_index or {}).values():
        for fact in bucket:
            fact_id = fact.get("id")
            if not fact_id or fact_id in seen:
                continue
            seen.add(fact_id)
            facts.append(fact)
    return facts


def weighted_rrf(vec_hits: list[dict], hash_hits: list[dict], w_vec: float = 0.65, w_hash: float = 0.35, k: int = 60) -> list[dict]:
    """Weighted Reciprocal Rank Fusion. Rank is 1-based. Identity is fact id."""
    scores: dict[str, float] = {}
    keep: dict[str, dict] = {}
    for weight, hits in ((w_vec, vec_hits or []), (w_hash, hash_hits or [])):
        for rank, fact in enumerate(hits, start=1):
            fact_id = str(fact.get("id") or "")
            if not fact_id:
                continue
            scores[fact_id] = scores.get(fact_id, 0.0) + weight / (k + rank)
            if fact_id not in keep:
                keep[fact_id] = dict(fact)
            else:
                keep[fact_id] = {**fact, **keep[fact_id]}
            keep[fact_id]["rrf_score"] = scores[fact_id]
    ordered = sorted(keep.values(), key=lambda fact: (-float(fact.get("rrf_score") or 0.0), str(fact.get("id"))))
    return ordered
