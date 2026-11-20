"""Offline hybrid semantic ranking and weighted reciprocal rank fusion."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np

from memengine.categories import map_categories
from memengine.router import match_entities
from memengine.textutil import strip_step_tags, tokenize_for_fts


def normalize_vectors(vectors):
    result = np.asarray(vectors, dtype="<f4")
    if result.ndim != 2 or not result.shape[1] or not np.isfinite(result).all():
        raise ValueError("invalid embedding vectors")
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    return result / np.where(norms > 0, norms, 1.0)

_WORD = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", re.UNICODE)
_LATIN = re.compile(r"[A-Za-z0-9]+")
_HAN_TRIGRAM = re.compile(r"[\u4e00-\u9fff]{3}")
_HAN = re.compile(r"[\u4e00-\u9fff]")
_HAN_FUNCTION = frozenset("的了这那该是有在和与就吗呢呀啊么啥")
_STOP = frozenset("the can this and for was with what when where why which how i of in a to is it you your my at".split())
_QUERY_STOP = _STOP | frozenset("do does did might would could kind type prefer should about mentioned please tell".split())
_BROAD_CATEGORIES = {"类目:人物", "类目:兴趣习惯", "类目:情绪观点", "类目:其他/通用"}


def _latin_terms(text, stop):
    terms = set(_LATIN.findall(text.lower())) - stop
    return {word[:-3] + "y" if len(word) > 4 and word.endswith("ies") else
            word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word for word in terms}


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

    @property
    def model_name(self):
        return f"hash-tfidf-static-v2:{self.dim}"

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            clean = strip_step_tags(text)
            terms = [term for term in tokenize_for_fts(clean, legacy=True).split() if term not in _QUERY_STOP]
            weights = {}
            counts = Counter(terms)
            for term in terms:
                if term.isascii() and term.isalpha() and len(term) >= 4:
                    for index in range(len(term) - 2):
                        feature = "~" + term[index:index + 3]
                        counts[feature] += 1
                        weights[feature] = 0.35
            vectors.append(self._embed_counts(counts, weights))
        return np.asarray(vectors, dtype="<f4").reshape(len(texts), self.dim)

    def _tf_counts(self, text: str) -> Counter:
        return Counter(tokenize(text))

    def _idf(self, docs: list[Counter]) -> dict[str, float]:
        n_docs = max(1, len(docs))
        df: Counter = Counter()
        for counts in docs:
            df.update(counts.keys())
        return {term: math.log((1 + n_docs) / (1 + freq)) + 1.0 for term, freq in df.items()}

    def embed(self, text: str, idf: dict[str, float] | None = None) -> np.ndarray:
        return self._embed_counts(self._tf_counts(text), idf or {})

    def _embed_counts(self, counts, weights) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float64)
        for term, count in counts.items():
            index, sign = _bucket(term, self.dim)
            vector[index] += sign * (1.0 + math.log(count)) * weights.get(term, 1.0)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def search(self, query: str, facts: list[dict], top_n: int = 30) -> list[dict]:
        return self.search_many([query], facts, top_n)[0]

    def search_many(self, queries: list[str], facts: list[dict], top_n: int) -> list[list[dict]]:
        if not facts or top_n <= 0:
            return [[] for _ in queries]
        texts = [str(fact.get("text") or "") for fact in facts]
        docs = [self._tf_counts(text) for text in texts]
        idf = self._idf(docs)
        vectors = np.stack([self._embed_counts(counts, idf) for counts in docs])
        lengths = [sum(counts.values()) for counts in docs]
        mean_length = max(1, sum(lengths) / len(lengths))
        word_docs = [_latin_terms(text, _STOP) for text in texts]
        entity_keys = {key for fact in facts for key in fact.get("entity_z_keys", [])}
        known_categories = sorted({key.split(":", 1)[1] for fact in facts for key in fact.get("category_z_keys", [])})
        rankings = []
        for query in queries:
            stem = query.split("\n", 1)[0]
            keywords = _latin_terms(stem, _QUERY_STOP)
            han_query = bool(_HAN.search(stem))
            categories = set(map_categories(stem, categories=known_categories)) - _BROAD_CATEGORIES if not match_entities(stem, entity_keys) else set()
            qterms = set(tokenize(query)) - _STOP
            weights = {term: idf.get(term, math.log(1 + len(docs)) + 1) for term in qterms}
            qweight = sum(weights.values()) or 1
            cosine = vectors @ self.embed(query, idf)
            scores = []
            for index, (text, counts, length) in enumerate(zip(texts, docs, lengths)):
                shared = qterms & counts.keys()
                saturation = 1.2 * (0.25 + 0.75 * length / mean_length)
                bm25 = sum(weights[term] * counts[term] / (counts[term] + saturation) for term in shared) / qweight
                containment = sum(weights[term] for term in shared) / max(1, min(qweight, sum(idf[term] for term in counts if term not in _STOP)))
                score = (0.50 * max(0.0, float(cosine[index])) + 0.20 * bm25 + 0.30 * containment) if shared else 0.0
                if han_query and not any(_HAN_TRIGRAM.fullmatch(term) and _HAN_FUNCTION.isdisjoint(term) for term in shared):
                    score *= 0.70
                word_overlap = keywords.intersection(word_docs[index])
                if len(word_overlap) >= 2 or word_overlap and len(keywords) == 1:
                    score = max(score, 0.90 * len(word_overlap) / len(keywords))
                if categories.intersection(facts[index].get("category_z_keys", [])):
                    score = max(score, 0.56 + 0.40 * score)
                if text.strip().casefold() == query.strip().casefold():
                    score = 1.0
                scores.append(min(1.0, score))
            rankings.append(rank_vector_hits(facts, scores, top_n))
        return rankings


def rank_vector_hits(facts, scores, top_n):
    pairs = sorted(zip(scores, facts), key=lambda pair: -pair[0])
    return [{**fact, "vector_score": float(score)} for score, fact in pairs[:max(0, top_n)]]


def weighted_rrf(vec_hits: list[dict], hash_hits: list[dict], w_vec: float = 0.65, w_hash: float = 0.35, k: int = 60,
                 *, extra_channels=()) -> list[dict]:
    """Weighted Reciprocal Rank Fusion. Rank is 1-based. Identity is fact id."""
    scores: dict[str, float] = {}
    keep: dict[str, dict] = {}
    channels = ((w_vec, vec_hits or []), (w_hash, hash_hits or []), *extra_channels)
    total_weight = sum(weight for weight, hits in channels if hits) or 1
    for weight, hits in channels:
        seen = set()
        for rank, fact in enumerate(hits, start=1):
            fact_id = str(fact.get("id") or "")
            if not fact_id or fact_id in seen:
                continue
            seen.add(fact_id)
            scores[fact_id] = scores.get(fact_id, 0.0) + weight * fact.get("rrf_factor", 1.0) / (k + rank)
            if fact_id not in keep:
                keep[fact_id] = dict(fact)
            else:
                keep[fact_id] = {**fact, **keep[fact_id]}
            keep[fact_id]["rrf_score"] = scores[fact_id]
            keep[fact_id]["rrf_norm"] = scores[fact_id] * (k + 1) / total_weight
    ordered = sorted(keep.values(), key=lambda fact: (-float(fact.get("rrf_score") or 0.0), str(fact.get("id"))))
    return ordered
