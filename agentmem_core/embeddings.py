"""Remote semantic embeddings and deterministic hashing for the RAG baseline."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from collections import OrderedDict
from http.client import HTTPException

import numpy as np

from agentmem_core.vector import HashedTfidfEmbedder, rank_vector_hits


class OpenAICompatibleEmbedder:
    def __init__(self, base_url: str, api_key: str, model: str, *, timeout: float = 10.0):
        base = base_url.rstrip("/")
        self.url = base if base.endswith("/embeddings") else base + "/embeddings"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.fallback = HashedTfidfEmbedder()
        self._vectors = OrderedDict()

    @classmethod
    def from_env(cls):
        url = os.getenv("EMBEDDING_BASE_URL")
        key = os.getenv("EMBEDDING_API_KEY")
        model = os.getenv("EMBEDDING_MODEL")
        return cls(url, key or "", model) if url and model else None

    def _embed(self, texts):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = json.dumps({"model": self.model, "input": texts}).encode()
        request = urllib.request.Request(self.url, data=body, headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            rows = sorted(json.load(response)["data"], key=lambda item: item["index"])
        if [row["index"] for row in rows] != list(range(len(texts))):
            raise ValueError("embedding size or indices do not match inputs")
        vectors = np.asarray([row["embedding"] for row in rows], dtype=np.float64)
        if vectors.ndim != 2 or not vectors.shape[1] or not np.isfinite(vectors).all():
            raise ValueError("invalid embedding vectors")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms > 0, norms, 1)

    @property
    def model_name(self):
        return f"openai:{self.url}:{self.model}"

    def encode(self, texts):
        return np.asarray(self._embed(texts), dtype="<f4")

    def search_many(self, queries, facts, top_n):
        if not facts or not queries or top_n <= 0:
            return [[] for _ in queries]
        active = [fact for fact in facts if not fact.get("historical")]
        identities = [(fact.get("id"), fact.get("revision"), hashlib.sha256(fact["text"].encode()).digest())
                      for fact in active]
        keys = {id(fact): key for fact, key in zip(active, identities)}
        working = {key: self._vectors[key] for key in identities if key in self._vectors}
        missing = [fact for fact in facts if keys.get(id(fact)) not in working]
        try:
            query_vectors = self._embed(queries)
            fresh = {}
            for start in range(0, len(missing), 128):
                batch = missing[start:start + 128]
                fresh.update(zip(map(id, batch), self._embed([fact["text"] for fact in batch])))
            vectors = np.stack([fresh[id(fact)] if id(fact) in fresh else working[keys[id(fact)]] for fact in facts])
            scores = np.clip(query_vectors @ vectors.T, 0, 1)
            for fact, key in zip(active, identities):
                self._vectors[key] = fresh[id(fact)] if id(fact) in fresh else working[key]
                self._vectors.move_to_end(key)
            while len(self._vectors) > 4096:
                self._vectors.popitem(last=False)
            return [rank_vector_hits(facts, row, top_n) for row in scores]
        except (OSError, HTTPException, ValueError, KeyError, TypeError):
            return self.fallback.search_many(queries, facts, top_n)

    def search(self, query, facts, top_n=30):
        return self.search_many([query], facts, top_n)[0]

    def forget(self, line_id):
        for key in list(self._vectors):
            if key[0] == line_id:
                del self._vectors[key]


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
