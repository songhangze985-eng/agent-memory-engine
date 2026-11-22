"""Batch relevance scoring with remote and deterministic local implementations."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from collections import Counter
from math import isfinite, log
from typing import Protocol

from memengine.distance import unit_interval
from memengine.linguistics import stem
from memengine.temporal import competition_terms
from memengine.textutil import tokenize_for_fts

_STOP = frozenset("a an the is was were are be to of in it at on with for and or i my me you your we our "
                  "what which when where why how who do does did can could would should tell please about "
                  "kind type some something remember recall mentioned know".split())
_CJK_FILLER = re.compile(r"什么时候|什么|哪些|哪种|多少|一下|怎么样|怎么|哪里|在哪|如何|是否|请问|请确认|"
                         r"上次|原来|原本|最早|以前|之前|现在|当前|去年|上个月|上周|昨天|最新")
_CONSTRAINT_QUERY = re.compile(r"禁忌|限制|例外|特别情况|需要注意|注意事项|过敏|"
                               r"\b(?:allerg\w*|exceptions?|restrictions?|contraindications?|precautions?|avoid)\b", re.I)
_CONSTRAINT_FACT = re.compile(r"过敏|禁忌|禁止|不得|不能|切勿|无障碍|例外|"
                              r"\b(?:allerg\w*|exceptions?|restrictions?|contraindications?|avoid|prohibited|must not)\b", re.I)


def taboo_query(query):
    return bool(_CONSTRAINT_QUERY.search(query))


def anchored_relevance(query, candidates):
    terms = competition_terms(query)
    matching = [terms & competition_terms(fact["plain"]) for fact in candidates]
    frequencies = Counter(term for matches in matching for term in matches)
    weights = {term: 1 + log((len(candidates) + 1) / (frequency + 1)) for term, frequency in frequencies.items()}
    overlaps = [sum(weights[term] for term in sorted(matches)) for matches in matching]
    maximum = max(overlaps, default=0) or 1
    return [0.30 * overlap / maximum + 0.10 * max(fact["vector_score"], fact.get("anchor_vector_score", 0.0))
            for fact, overlap in zip(candidates, overlaps)]


def evidence_pair_scores(query, candidates):
    """Discount generic terms and prefer the clause that contains the evidence."""
    def canonical(terms):
        return {stem(word) if word.isascii() and word.isalpha() else word for word in terms if word not in _STOP}

    wanted = canonical(competition_terms(query))
    documents = [fact.get("normalized_text") or fact["plain"] for fact in candidates]
    overlaps = [wanted & canonical(competition_terms(text)) for text in documents]
    frequencies = Counter(word for shared in overlaps for word in shared)
    weights = {word: 1 + log((len(candidates) + 1) / (frequencies[word] + 1)) for word in sorted(wanted)}
    denominator = sum(weights.values()) or 1
    output = []
    for text, shared in zip(documents, overlaps):
        coverage = sum(weights[word] for word in sorted(shared)) / denominator
        clause = 0.0
        if len(shared) > 1:
            for value in re.split(r"[!?;。！？；\n]|\.(?=\s)", text):
                local = canonical(competition_terms(value)) & shared
                if len(local) > 1:
                    clause = max(clause, sum(weights[word] for word in sorted(local)) / denominator)
        output.append((coverage, clause))
    return output


class Reranker(Protocol):
    is_calibrated: bool

    def score(self, query: str, candidates: list[dict]) -> list[float]:
        """Return one relevance score per candidate in input order."""


def checked_scores(scores, count):
    if len(scores) != count or any(type(score) is bool or not isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("reranker must return one finite [0, 1] score per candidate")
    return scores


class OpenAICompatibleReranker:
    is_calibrated = True

    def __init__(self, base_url: str, api_key: str, model: str, *, timeout: float = 10.0):
        base = base_url.rstrip("/")
        self.url = base if base.endswith("/rerank") else base + "/rerank"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_env(cls):
        url, model = os.getenv("RERANK_BASE_URL"), os.getenv("RERANK_MODEL")
        return cls(url, os.getenv("RERANK_API_KEY", ""), model) if url and model else None

    def score(self, query, candidates):
        if not candidates:
            return []
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model, "query": query, "documents": [item["plain"] for item in candidates]}
        request = urllib.request.Request(self.url, data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.load(response)
        rows = body.get("results", body.get("data"))
        indexed = {row["index"]: row["relevance_score"] for row in rows}
        if (len(rows) != len(candidates) or any(type(row["index"]) is not int for row in rows)
                or set(indexed) != set(range(len(candidates)))):
            raise ValueError("rerank response indices do not match inputs")
        return checked_scores([indexed[index] for index in range(len(candidates))], len(candidates))


class CalibratedLocalReranker:
    """Piecewise confidence from lexical coverage, cosine and a weak RRF prior."""

    is_calibrated = False  # Local confidence uses the offline threshold.

    def rank_long_evidence(self, ranked, baseline):
        """Use an ordinal RRF prior when long facts are competing for the byte budget."""
        order = {fact["id"]: index for index, fact in enumerate(baseline)}
        scored = []
        for fact in ranked:
            prior = 1 / (1 + 0.25 * order.get(fact["id"], 35))
            score = (0.75 * fact["ce_score"] + 0.25 * prior) * fact["status_weight"] * fact["temporal_weight"]
            score *= fact["bipolar_factor"] * fact["decay_factor"] * fact.get("attention_weight", 1.0)
            scored.append({**fact, "score": score, "final_score": score, "bipolar_score": score})
        return sorted(scored, key=lambda fact: (-fact["final_score"], fact["id"]))

    def stabilize_shortlist(self, ranked, rrf):
        """Give near-tied ordinary evidence a stable RRF head."""
        if len(ranked) < 2 or not rrf:
            return ranked
        leading = ranked[0]
        pinned = next((fact for fact in ranked if fact["id"] == rrf[0]["id"]), None)
        if not pinned or pinned.get("temporal_role") == "older":
            return ranked
        if any(fact.get("is_constraint") or fact.get("attention_weight", 1.0) > 1.0 or fact.get("is_conflict_or_override")
               or fact.get("exact_ident_hit") or fact.get("version_mismatch")
               or fact.get("bipolar_weight", 1.0) != 1.0 for fact in (leading, pinned)):
            return ranked
        # A clearly stronger relevance signal can displace the first RRF result.
        if pinned["ce_score"] < 0.85 * leading["ce_score"]:
            return ranked
        # Compare the lead candidates on rank-normalized RRF and bounded relevance.
        if pinned["id"] != leading["id"]:
            rrf_order = {fact["id"]: index for index, fact in enumerate(rrf)}
            prior = 1 / (1 + 0.25 * rrf_order[leading["id"]])
            if 0.85 + 0.15 * pinned["ce_score"] > 0.85 * prior + 0.15 * leading["ce_score"]:
                return [pinned, *(fact for fact in ranked if fact["id"] != pinned["id"])]
        return ranked

    def score(self, query, candidates):
        terms = set(tokenize_for_fts(_CJK_FILLER.sub(" ", query)).split()) - _STOP
        # CJK bigrams carry specificity. Keep separate single-character queries usable.
        weights = {term: (2 if not term.isascii() and len(term) > 1 else
                          0.25 if not term.isascii() and len(term) == 1 else 1) for term in terms}
        total = sum(weights.values()) or 1
        constraint_query = taboo_query(query)
        ranked = []
        for fact in candidates:
            text = fact["plain"]
            shared = terms.intersection(tokenize_for_fts(text).split())
            coverage = sum(weights[term] for term in sorted(shared)) / total
            vector = unit_interval(fact.get("vector_score"))
            prior = unit_interval(fact.get("rrf_norm"))
            anchor = fact.get("entity_anchor", False)
            score = 0.60 * coverage + 0.28 * vector + 0.12 * prior
            if anchor and coverage > 0:
                # A direct entity mention can connect separately indexed facts to the question.
                score = max(score, 0.22 + 0.30 * coverage)
            if fact.get("category_match"):
                score = max(score, 0.19 + 0.24 * vector)
            if fact.get("temporal_match") and (anchor or coverage >= 0.15):
                score = max(score, 0.32 + 0.46 * coverage)
            if constraint_query:
                constraint = bool(_CONSTRAINT_FACT.search(text))
                if constraint and (anchor or coverage >= 0.25 or vector >= 0.45):
                    score = max(score, 0.56 + 0.30 * coverage)
                elif not constraint:
                    score *= 0.65
            ranked.append(unit_interval((score - 0.08) / 0.58))
        return ranked
