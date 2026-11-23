"""Pluggable QA judges: Hindsight-aligned LLM judge + reproducible deterministic offline judge."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from memengine.llm import OpenAICompatibleClient, loads_json_object

_STOPWORDS = frozenset(
    "a an the and or but in on at to for of with by from as is are was were be been being "
    "i you he she it we they my your his her its our their me him us them this that these those "
    "do does did have has had would could should will can may might must about into over after before "
    "user assistant said mentioned told".split()
)

_REFUSAL_RE = re.compile(
    r"(?:no relevant memory|insufficient evidence|cannot verify|could not verify|"
    r"don't have enough information|do not have enough information|not mentioned|"
    r"no information|don't know|do not know|cannot answer|unable to determine|"
    r"not provided|none mentioned|no record|没有找到|无法核实|证据不足|未提及|不清楚)",
    re.I,
)

_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
    "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
    "once": "1", "twice": "2", "three times": "3",
}

_DATE_DMY = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*,?\s*(\d{4})\b",
    re.I,
)
_DATE_MDY = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+"
    r"(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})\b",
    re.I,
)
_DATE_YMD = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")
_WORD_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*|[\u4e00-\u9fff]")


@dataclass(frozen=True)
class JudgeVerdict:
    correct: bool
    score: float
    label: str
    reason: str
    borderline: bool = False


def canonicalize_dates(text: str) -> str:
    """Normalize common English/ISO dates to YYYY-MM-DD for fair comparison."""
    def _dmy(m: re.Match) -> str:
        d, mon, y = int(m.group(1)), _MONTHS[m.group(2).lower()], int(m.group(3))
        return f"{y:04d}-{mon:02d}-{d:02d}"

    def _mdy(m: re.Match) -> str:
        mon, d, y = _MONTHS[m.group(1).lower()], int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{mon:02d}-{d:02d}"

    def _ymd(m: re.Match) -> str:
        y, mon, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{mon:02d}-{d:02d}"

    out = _DATE_DMY.sub(_dmy, text or "")
    out = _DATE_MDY.sub(_mdy, out)
    out = _DATE_YMD.sub(_ymd, out)
    return out


def normalize_answer_text(text: str) -> str:
    cleaned = canonicalize_dates(str(text or "")).casefold()
    for word, digit in _NUM_WORDS.items():
        cleaned = re.sub(rf"\b{re.escape(word)}\b", digit, cleaned)
    tokens = _WORD_TOKEN.findall(cleaned)
    return " ".join(tokens)


def content_tokens(text: str) -> list[str]:
    norm = normalize_answer_text(text)
    return [tok for tok in norm.split() if tok not in _STOPWORDS]


def looks_like_abstention(text: str, abstained_flag: bool = False) -> bool:
    if abstained_flag:
        return True
    stripped = (text or "").strip()
    if not stripped:
        return True
    return bool(_REFUSAL_RE.search(stripped))


class DeterministicQAJudge:
    """Transparent, zero-API deterministic QA judge for reproducible offline baselines.

    Every report stamps ``judge="deterministic-v1"`` so offline deterministic scores
    are never confused with live LLM-as-a-Judge scores.
    """

    name = "deterministic-v1"

    def evaluate(
        self,
        *,
        question: str,
        gold_answer: str,
        predicted_answer: str,
        category: str,
        unanswerable: bool = False,
        abstained: bool = False,
        mcq_gold: str | None = None,
        mcq_predicted: str | None = None,
    ) -> JudgeVerdict:
        # 1. Multiple-choice (MemBench)
        if mcq_gold:
            gold_letter = mcq_gold.strip().upper()[:1]
            pred_letter = (mcq_predicted or "").strip().upper()[:1]
            if not pred_letter and predicted_answer:
                m = re.search(r"\b([ABCD])\b", predicted_answer.strip().upper())
                pred_letter = m.group(1) if m else ""
            ok = bool(gold_letter and pred_letter == gold_letter)
            return JudgeVerdict(
                correct=ok,
                score=1.0 if ok else 0.0,
                label="correct" if ok else "wrong",
                reason="mcq_exact_match" if ok else f"mcq_mismatch({pred_letter or 'NONE'}!={gold_letter})",
                borderline=False,
            )

        # 2. Unanswerable / abstention / adversarial
        is_abstained = looks_like_abstention(predicted_answer, abstained)
        if unanswerable:
            return JudgeVerdict(
                correct=is_abstained,
                score=1.0 if is_abstained else 0.0,
                label="correct_abstention" if is_abstained else "hallucinated_on_unanswerable",
                reason="abstained_correctly" if is_abstained else "failed_to_abstain",
                borderline=False,
            )

        if is_abstained:
            return JudgeVerdict(
                correct=False,
                score=0.0,
                label="false_abstention",
                reason="abstained_on_answerable_question",
                borderline=False,
            )

        # 3. Answerable open-ended QA
        gold_norm = normalize_answer_text(gold_answer)
        pred_norm = normalize_answer_text(predicted_answer)
        if not gold_norm:
            return JudgeVerdict(False, 0.0, "empty_gold", "missing_gold_answer", False)

        pred_words = pred_norm.split()
        focused_pred = " ".join(pred_words[:80])
        gold_toks = content_tokens(gold_answer)
        pred_toks = content_tokens(focused_pred)
        gold_set = set(gold_toks) or set(gold_norm.split())
        pred_set = set(pred_toks) or set(focused_pred.split())

        overlap = len(gold_set & pred_set)
        recall = overlap / len(gold_set) if gold_set else 0.0
        precision = overlap / len(pred_set) if pred_set else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Exact normalized phrase containment within focused answer window (<=80 words)
        contained = bool(re.search(rf"(?<!\w){re.escape(gold_norm)}(?!\w)", focused_pred))
        if contained:
            # Guard: for short single-number answers (counts, days, ages), avoid false positives
            # from incidental numbers inside a long un-focused excerpt.
            if re.fullmatch(r"\d+", gold_norm) and len(pred_words) > 35:
                # Require the number to appear in the leading answer clause (first 20 words)
                lead = " ".join(pred_words[:20])
                if not re.search(rf"(?<!\w){re.escape(gold_norm)}(?!\w)", lead):
                    return JudgeVerdict(False, round(f1, 4), "numeric_not_focused", f"number_buried_in_excerpt(f1={f1:.2f})", borderline=True)
            return JudgeVerdict(True, 1.0, "exact_span_match", "normalized_gold_in_focused_answer", False)

        # Short factual / date / entity answers: require high content-token recall in concise answer
        if len(gold_set) <= 4:
            if recall >= 0.85 and len(pred_words) <= 65:
                return JudgeVerdict(True, round(max(f1, recall), 4), "token_recall_match", f"short_gold_recall={recall:.2f}", False)
            if f1 >= 0.50:
                return JudgeVerdict(True, round(f1, 4), "token_f1_match", f"f1={f1:.2f}", False)
            borderline = (0.30 <= f1 < 0.50) or (recall >= 0.50)
            return JudgeVerdict(False, round(f1, 4), "mismatch", f"short_gold_miss(r={recall:.2f},f1={f1:.2f})", borderline)

        # Longer preference / open-domain / multi-item answers
        if f1 >= 0.40 or (recall >= 0.65 and len(pred_words) <= 85):
            return JudgeVerdict(True, round(max(f1, recall), 4), "semantic_overlap_match", f"r={recall:.2f},f1={f1:.2f}", False)

        borderline = (0.28 <= f1 < 0.40) or (recall >= 0.45)
        return JudgeVerdict(False, round(f1, 4), "mismatch", f"long_gold_miss(r={recall:.2f},f1={f1:.2f})", borderline)


_LLM_JUDGE_PROMPT = """You are an impartial, strict evaluator grading a memory-augmented AI assistant's answer against a reference gold answer (aligned with LongMemEval / LoCoMo / Hindsight QA protocol).

Category: {category}
Question: {question}
Gold Answer: {gold_answer}
Predicted Answer: {predicted_answer}

Grading rules:
- If Category is "abstention" or "5 adversarial": mark correct=true ONLY if the predicted answer clearly states that the information is unknown, not mentioned, or cannot be verified from memory.
- If Category is "knowledge-update": mark correct=true ONLY if the predicted answer provides the LATEST updated value (mentioning the older value as prior history is fine, but answering with only the outdated value is FALSE).
- If Category is "temporal-reasoning" or "2 temporal": equivalent date/duration formats (e.g., "7 May 2023" vs "May 7, 2023" vs "2023-05-07") are correct; wrong dates or wrong counts are FALSE.
- Otherwise: mark correct=true iff the predicted answer contains the exact factual answer or semantically equivalent response to the gold answer without contradicting it.

Reply ONLY with a JSON object:
{{"correct": true/false, "reason": "brief explanation"}}
"""


class OpenAICompatibleJudge:
    """Hindsight / LongMemEval aligned LLM-as-a-Judge using OpenAI-compatible Chat Completions."""

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        resolved_model = model or os.environ.get("JUDGE_MODEL") or os.environ.get("LLM_MODEL") or "gpt-4o-mini"
        resolved_base = base_url or os.environ.get("JUDGE_BASE_URL") or os.environ.get("LLM_BASE_URL")
        resolved_key = api_key or os.environ.get("JUDGE_API_KEY") or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.client = OpenAICompatibleClient(base_url=resolved_base, api_key=resolved_key, model=resolved_model)
        self.name = f"llm-judge:{self.client.model}"
        self._fallback = DeterministicQAJudge()

    def evaluate(
        self,
        *,
        question: str,
        gold_answer: str,
        predicted_answer: str,
        category: str,
        unanswerable: bool = False,
        abstained: bool = False,
        mcq_gold: str | None = None,
        mcq_predicted: str | None = None,
    ) -> JudgeVerdict:
        if mcq_gold:
            return self._fallback.evaluate(
                question=question,
                gold_answer=gold_answer,
                predicted_answer=predicted_answer,
                category=category,
                unanswerable=unanswerable,
                abstained=abstained,
                mcq_gold=mcq_gold,
                mcq_predicted=mcq_predicted,
            )
        prompt = _LLM_JUDGE_PROMPT.format(
            category=category,
            question=question,
            gold_answer=gold_answer if not unanswerable else "(UNANSWERABLE — must abstain)",
            predicted_answer=predicted_answer or "(EMPTY / ABSTAINED)",
        )
        raw = self.client.complete(prompt)
        data = loads_json_object(raw)
        ok = bool(data.get("correct"))
        reason = str(data.get("reason") or "")
        return JudgeVerdict(
            correct=ok,
            score=1.0 if ok else 0.0,
            label="llm_judge_pass" if ok else "llm_judge_fail",
            reason=reason,
            borderline=False,
        )
