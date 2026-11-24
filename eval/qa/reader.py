"""Pluggable QA readers: OpenAI-compatible LLM reader + deterministic offline reader."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from eval.qa.judge import content_tokens, looks_like_abstention
from agentmem_core.llm import OpenAICompatibleClient
from agentmem_core.textutil import strip_step_tags

_SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|\n+")

_LLM_READER_PROMPT = """Answer the user's question concisely using ONLY the retrieved memory context below.
If the memory context does not contain enough evidence to answer the question, reply exactly:
"I don't have enough information to answer this question."
If a fact was updated over time, answer with the latest updated value unless the question explicitly asks about an earlier time.
For date or counting questions, show the exact date or exact count clearly in your first sentence.

Reference Date: {reference_date}
Memory Context:
{context}

Question: {question}
Answer:"""

_LLM_MCQ_PROMPT = """Choose the single best option (A, B, C, or D) based ONLY on the retrieved memory context below.
Reply with ONLY the single letter A, B, C, or D.

Reference Date: {reference_date}
Memory Context:
{context}

Question: {question}
{choices_block}
Answer letter:"""


@dataclass(frozen=True)
class ReaderOutput:
    answer: str
    abstained: bool
    predicted_choice: str | None = None


def _best_extractive_answer(query: str, context: str, *, max_words: int = 60) -> str:
    """Select the most query-relevant concise sentence(s) from assembled memory context."""
    q_toks = set(content_tokens(query))
    lines = [strip_step_tags(s).strip() for s in _SENT_SPLIT.split(context or "") if s.strip()]
    # Strip markdown heading lines when sentence candidates exist
    candidates = [ln for ln in lines if not ln.startswith("#") and len(ln) >= 3] or lines
    if not candidates:
        return ""
    scored = []
    for idx, sent in enumerate(candidates):
        s_toks = set(content_tokens(sent))
        overlap = len(q_toks & s_toks)
        # Prefer higher query overlap, earlier rank in context (already ranked by engine)
        score = overlap / max(1, len(q_toks)) + 0.08 / (idx + 1)
        scored.append((score, idx, sent))
    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen: list[str] = []
    words = 0
    for _, _, sent in scored[:3]:
        w = len(sent.split())
        if chosen and words + w > max_words:
            break
        chosen.append(sent)
        words += w
    text = " ".join(chosen)
    split_words = text.split()
    if len(split_words) > max_words:
        text = " ".join(split_words[:max_words])
    return text


def select_mcq_choice_deterministic(query: str, context: str, choices: dict[str, str]) -> str:
    """Deterministic lexical overlap choice selector for offline MemBench evaluation."""
    if not choices:
        return ""
    ctx_toks = set(content_tokens(context))
    q_toks = set(content_tokens(query))
    best_letter = next(iter(choices))
    best_score = -1.0
    for letter, text in choices.items():
        c_toks = set(content_tokens(text))
        if not c_toks:
            score = 0.0
        else:
            # Prefer tokens appearing in context that distinguish this option from question stem
            novel = c_toks - q_toks or c_toks
            ctx_hit = len(novel & ctx_toks) / len(novel)
            all_hit = len(c_toks & ctx_toks) / len(c_toks)
            score = 0.70 * ctx_hit + 0.30 * all_hit
        if score > best_score:
            best_score = score
            best_letter = letter
    return best_letter


class DeterministicEngineReader:
    """Offline deterministic reader: uses engine.answer() when available, else focused extraction."""

    name = "MockLLMClient/DeterministicReader"

    def answer_from_engine(self, engine_answer_result: dict, query: str) -> ReaderOutput:
        resp = str(engine_answer_result.get("response") or "").strip()
        abstained = bool(engine_answer_result.get("abstained")) or looks_like_abstention(resp)
        if abstained:
            return ReaderOutput(answer=resp or "Insufficient evidence to answer.", abstained=True)
        # Keep response concise (<= 65 words) for strict judge evaluation
        words = resp.split()
        if len(words) > 65:
            resp = _best_extractive_answer(query, resp, max_words=60) or " ".join(words[:60])
        return ReaderOutput(answer=resp, abstained=False)

    def answer_from_context(
        self,
        query: str,
        context: str,
        *,
        reference_date: str = "",
        empty_recall: bool = False,
        top_score: float = 1.0,
        min_score: float = 0.05,
        choices: dict[str, str] | None = None,
        **_ignored,
    ) -> ReaderOutput:
        if choices:
            letter = select_mcq_choice_deterministic(query, context, choices)
            return ReaderOutput(answer=f"{letter}. {choices.get(letter, '')}", abstained=False, predicted_choice=letter)
        if empty_recall or not (context or "").strip() or top_score < min_score:
            return ReaderOutput(answer="Insufficient evidence to answer.", abstained=True)
        extracted = _best_extractive_answer(query, context, max_words=60)
        if not extracted:
            return ReaderOutput(answer="Insufficient evidence to answer.", abstained=True)
        # Abstain if extracted text shares ZERO content tokens with the question
        q_toks = set(content_tokens(query))
        e_toks = set(content_tokens(extracted))
        if q_toks and not (q_toks & e_toks):
            return ReaderOutput(answer="Insufficient evidence to answer.", abstained=True)
        return ReaderOutput(answer=extracted, abstained=False)


class OpenAICompatibleReader:
    """Live LLM QA reader using OpenAI-compatible Chat Completions (Hindsight-aligned prompt)."""

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        resolved_model = model or os.environ.get("LLM_MODEL") or "gpt-4o-mini"
        resolved_base = base_url or os.environ.get("LLM_BASE_URL")
        resolved_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.client = OpenAICompatibleClient(base_url=resolved_base, api_key=resolved_key, model=resolved_model)
        self.name = f"llm:{self.client.model}"

    def answer_from_context(
        self,
        query: str,
        context: str,
        *,
        reference_date: str = "",
        empty_recall: bool = False,
        choices: dict[str, str] | None = None,
        **_ignored,
    ) -> ReaderOutput:
        if choices:
            choices_block = "\n".join(f"{k}. {v}" for k, v in choices.items())
            prompt = _LLM_MCQ_PROMPT.format(
                reference_date=reference_date or "unknown",
                context=context or "(empty)",
                question=query,
                choices_block=choices_block,
            )
            raw = self.client.complete(prompt).strip()
            m = re.search(r"\b([ABCD])\b", raw.upper())
            letter = m.group(1) if m else select_mcq_choice_deterministic(query, context, choices)
            return ReaderOutput(answer=raw or letter, abstained=False, predicted_choice=letter)

        if empty_recall or not (context or "").strip():
            return ReaderOutput(answer="I don't have enough information to answer this question.", abstained=True)
        prompt = _LLM_READER_PROMPT.format(
            reference_date=reference_date or "unknown",
            context=context,
            question=query,
        )
        raw = self.client.complete(prompt).strip()
        return ReaderOutput(answer=raw, abstained=looks_like_abstention(raw))
