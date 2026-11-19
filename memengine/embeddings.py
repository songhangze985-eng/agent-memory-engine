"""Deterministic hashing embedder for the naive RAG baseline.

Uses MD5 so vectors do not depend on PYTHONHASHSEED. This is not a neural
embedding model; the full memory path does not retrieve by vector.
"""

from __future__ import annotations

import hashlib

import numpy as np


class HashingEmbedder:
    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float64)
        compact = re_space(text)
        if len(compact) < 2:
            compact = compact + "_"
        for index in range(len(compact) - 1):
            digest = hashlib.md5(compact[index : index + 2].encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector


def re_space(text: str) -> str:
    return "".join((text or "").lower().split())
