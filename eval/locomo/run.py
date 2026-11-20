"""Evaluate LoCoMo turn evidence against a pure RRF control."""

import re
from pathlib import Path

from memengine.chrono import parse_session_clock
from eval.evidence_ablation import Question, Turn, run_cli, sample_rows

URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
CATEGORIES = {1: "1 multi-hop", 2: "2 temporal", 3: "3 open-domain", 4: "4 single-hop", 5: "5 adversarial"}


def parse_turns(conversation):
    sessions = sorted((int(match[1]), key) for key in conversation
                      if (match := re.fullmatch(r"session_(\d+)", key)))
    turns = []
    for number, key in sessions:
        date = conversation.get(f"{key}_date_time")
        stamp = (parse_session_clock(date)
                 if date else (turns[-1].timestamp + 1 if turns else 0.0))
        for index, item in enumerate(conversation[key]):
            caption = item.get("blip_caption") or ""
            text = f"{item['speaker']}: {item['text']}" + (f" [Image: {caption}]" if caption else "")
            turns.append(Turn(str(item["dia_id"]), str(number), "user", text, stamp + index / 1000, mention_time=stamp if date else None, speaker=item["speaker"]))
    if not turns or len({turn.ref for turn in turns}) != len(turns):
        raise ValueError("LoCoMo conversation has no turns or duplicate dia_id values")
    return turns


def groups(data, count, seed):
    pool = [(ordinal, index, item) for ordinal, dialog in enumerate(data) for index, item in enumerate(dialog["qa"])]
    if not pool:
        raise ValueError("No LoCoMo questions found")
    selected = sample_rows(pool, count, seed, lambda row: CATEGORIES[int(row[2]["category"])])
    by_dialog = {}
    for ordinal, index, item in selected:
        by_dialog.setdefault(ordinal, []).append((index, item))
    for ordinal, chosen in by_dialog.items():
        dialog = data[ordinal]
        turns = parse_turns(dialog["conversation"])
        stamp = max(turn.timestamp for turn in turns) + 1
        questions = []
        for index, item in chosen:
            date = parse_session_clock(item.get("question_date"))
            questions.append(Question(f"{dialog['sample_id']}:{index}", CATEGORIES[int(item["category"])], item["question"],
                                      stamp if date is None else date, frozenset(map(str, item.get("evidence") or ())),
                                      unanswerable=int(item["category"]) == 5))
        yield turns, questions


if __name__ == "__main__":
    run_cli("LoCoMo", ("locomo10.json",), URL, Path(__file__).with_name(".cache"), groups)
