"""Ablation baselines written for this repo.

Baseline B is an original CRUD-plus-window stand-in. It does not use Mem0,
Letta, Zep, or any other memory-library code.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np

from memengine.embeddings import HashingEmbedder
from memengine.extract import extract_facts
from memengine.textutil import estimate_tokens

REFUSAL = "根据现有记忆，没有足够的直接关联可以确认这一点。"


def _chunks(text: str, size: int) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    return [raw[i : i + size] for i in range(0, len(raw), size)]


class NaiveRAG:
    """Baseline A: fixed character chunks and top-5 hashing-embedding search."""

    def __init__(self, chunk_size: int = 200, top_k: int = 5):
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.chunks: list[str] = []
        self.vectors: list[np.ndarray] = []
        self.full: list[str] = []
        self.embedder = HashingEmbedder()

    def add(self, text: str, timestamp: float | None = None, sync: bool = True) -> dict:
        del timestamp, sync
        self.full.append(text or "")
        for piece in _chunks(text or "", self.chunk_size):
            self.chunks.append(piece)
            self.vectors.append(self.embedder.embed(piece))
        return {"operations": [], "queued": False}

    def answer(self, query: str, timestamp: float | None = None) -> dict:
        del timestamp
        started = time.perf_counter()
        chosen = self._retrieve(query)
        elapsed = (time.perf_counter() - started) * 1000
        response = "\n".join(chosen) if chosen else "没有检索到内容。"
        history = "\n".join(self.full)
        return {
            "route": "NAIVE_RAG",
            "response": response,
            "retrieval_ms": elapsed,
            "context_tokens": estimate_tokens(response),
            "history_tokens": estimate_tokens(history),
            "reasoning": None,
            "llm_fallback": False,
        }

    def timeline_stats(self):
        return None

    def _retrieve(self, query: str) -> list[str]:
        if not self.vectors:
            return []
        query_vec = self.embedder.embed(query)
        matrix = np.vstack(self.vectors)
        scores = matrix @ query_vec
        count = min(self.top_k, len(self.chunks))
        order = np.argsort(-scores)[:count]
        return [self.chunks[int(index)] for index in order]


class CrudMemory:
    """Baseline B: from-scratch entity CRUD plus a fixed raw-turn window.

    Records upsert when the attribute matches and any Z key overlaps.
    Retrieval returns direct entity hits and records that share a key with
    them (one key join). Older values are discarded. This is not Mem0/Letta.
    """

    def __init__(self, window: int = 8):
        self.window: deque[str] = deque(maxlen=window)
        self.full: list[str] = []
        self.records: dict[tuple, dict] = {}

    def add(self, text: str, timestamp: float | None = None, sync: bool = True) -> dict:
        del sync
        self.full.append(text or "")
        self.window.append(text or "")
        for fact in extract_facts(text or ""):
            self._upsert(fact, timestamp)
        return {"operations": [], "queued": False}

    def _upsert(self, fact: dict, timestamp: float | None) -> None:
        stored = dict(fact)
        stored["updated_at"] = 0 if timestamp is None else float(timestamp)
        incoming = set(fact.get("zkeys") or [])
        for key in list(self.records):
            attribute, zkeys = key
            if attribute == fact.get("attribute") and incoming.intersection(zkeys):
                merged = tuple(sorted(set(zkeys) | incoming))
                self.records.pop(key)
                self.records[(attribute, merged)] = stored
                return
        self.records[(fact.get("attribute") or "", tuple(sorted(incoming)))] = stored

    def answer(self, query: str, timestamp: float | None = None) -> dict:
        del timestamp
        started = time.perf_counter()
        direct = []
        for fact in self.records.values():
            names = [key.split(":", 1)[-1] for key in fact.get("zkeys") or []]
            if any(name and name in query for name in names):
                direct.append(fact)
        keys = set()
        for fact in direct:
            keys.update(fact.get("zkeys") or [])
        linked = []
        if keys:
            for fact in self.records.values():
                if keys.intersection(fact.get("zkeys") or []):
                    linked.append(fact)
        hits = []
        seen = set()
        for fact in direct + linked:
            marker = (fact.get("attribute"), fact.get("text"))
            if marker in seen:
                continue
            seen.add(marker)
            hits.append(fact)
        elapsed = (time.perf_counter() - started) * 1000
        if hits:
            response = "。".join(fact["text"].rstrip("。") for fact in hits) + "。"
        else:
            response = REFUSAL
        history = "\n".join(self.full)
        context = response + "\n" + "\n".join(self.window)
        return {
            "route": "CRUD",
            "response": response,
            "retrieval_ms": elapsed,
            "context_tokens": estimate_tokens(context),
            "history_tokens": estimate_tokens(history),
            "reasoning": None,
            "llm_fallback": False,
        }

    def timeline_stats(self):
        return None
