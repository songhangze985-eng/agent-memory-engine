"""MemBench metrics.

Accuracy compares the chosen option letter with ``ground_truth`` (paper §3.5,
official env equality after the agent emits one of A/B/C/D).
Recall@10 is the fraction of evidence step ids present in the first 10
retrieved step ids (paper §3.5; their retrieval memories return indices).
Efficiency is mean write time per ``add`` and mean read time per ``search``.
Capacity is accuracy grouped by how many tokens were written, a cross-item
sketch of the paper's accuracy-versus-token curve.
"""

from __future__ import annotations

import json
import re

from memengine.textutil import char_bigrams, estimate_tokens, jaccard

_LETTER = re.compile(r"\b([ABCD])\b")


def parse_choice(text: str | None) -> str | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict):
        choice = str(obj.get("choice") or "").strip()
        if choice[:1] in "ABCD":
            return choice[:1]
    if raw[0] in "ABCD" and (len(raw) == 1 or raw[1] in " .):]\n"):
        return raw[0]
    found = list(dict.fromkeys(_LETTER.findall(raw)))
    if len(found) == 1:
        return found[0]
    return None


def choice_correct(response: str | None, ground_truth: str) -> bool:
    gold = (ground_truth or "").strip()
    if not gold:
        return False
    if (response or "").strip() == gold:
        return True
    parsed = parse_choice(response)
    return parsed is not None and parsed == gold


def recall_at_10(retrieved_steps: list[int], evidence_steps: list[int]) -> float | None:
    gold = set(evidence_steps)
    if not gold:
        return None
    top = []
    for step in retrieved_steps:
        if step not in top:
            top.append(step)
        if len(top) == 10:
            break
    hit = sum(1 for step in top if step in gold)
    return hit / len(gold)


def extract_steps(texts: list[str]) -> list[int]:
    steps = []
    for text in texts:
        for match in re.finditer(r"\[step (\d+)\]", text or ""):
            steps.append(int(match.group(1)))
    return steps


def lexical_choice(context: str, choices: dict[str, str]) -> str:
    """Deterministic offline chooser. Empty context always returns A."""
    letters = [letter for letter in ("A", "B", "C", "D") if letter in choices]
    if not letters:
        return "A"
    if not (context or "").strip():
        return letters[0]
    ctx = char_bigrams(context)
    best_letter = letters[0]
    best_score = -1.0
    for letter in letters:
        score = jaccard(ctx, char_bigrams(choices[letter]))
        if score > best_score:
            best_score = score
            best_letter = letter
    return best_letter


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def accuracy(flags: list[bool]) -> float | None:
    if not flags:
        return None
    return sum(1 for flag in flags if flag) / len(flags)


def capacity_bins(rows: list[dict], key: str = "same_head_correct") -> list[dict]:
    usable = [row for row in rows if row.get("ingest_tokens") is not None]
    if not usable:
        return []
    ordered = sorted(usable, key=lambda row: row["ingest_tokens"])
    width = max(1, len(ordered) // 3)
    bins = []
    for start, name in ((0, "low"), (width, "mid"), (width * 2, "high")):
        chunk = ordered[start : start + width if name != "high" else len(ordered)]
        if name == "high":
            chunk = ordered[start:]
        if not chunk:
            continue
        bins.append(
            {
                "bin": name,
                "n": len(chunk),
                "min_tokens": chunk[0]["ingest_tokens"],
                "max_tokens": chunk[-1]["ingest_tokens"],
                "accuracy": accuracy([bool(row.get(key)) for row in chunk]),
            }
        )
    return bins


def estimate_context_tokens(texts: list[str]) -> int:
    return estimate_tokens("\n".join(texts))
