"""Evaluate LoCoMo turn evidence against a pure RRF control."""

import re
from datetime import datetime, timezone
from pathlib import Path

from eval.evidence_ablation import Question, Turn, run_cli, sample_rows

URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
CATEGORIES = {1: "1 multi-hop", 2: "2 temporal", 3: "3 open-domain", 4: "4 single-hop", 5: "5 adversarial"}


def parse_turns(conversation):
    sessions = sorted((int(match[1]), key) for key in conversation
                      if (match := re.fullmatch(r"session_(\d+)", key)))
    turns = []
    for number, key in sessions:
        date = conversation.get(f"{key}_date_time")
        stamp = (datetime.strptime(date, "%I:%M %p on %d %B, %Y").replace(tzinfo=timezone.utc).timestamp()
                 if date else (turns[-1].timestamp + 1 if turns else 0.0))
        for index, item in enumerate(conversation[key]):
            caption = item.get("blip_caption") or ""
            text = f"{item['speaker']}: {item['text']}" + (f" [Image: {caption}]" if caption else "")
            turns.append(Turn(str(item["dia_id"]), str(number), "user", text, stamp + index / 1000))
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
        questions = [Question(f"{dialog['sample_id']}:{index}", CATEGORIES[int(item["category"])], item["question"],
                              stamp, frozenset(map(str, item.get("evidence") or ())),
                              unanswerable=int(item["category"]) == 5) for index, item in chosen]
        yield turns, questions


if __name__ == "__main__":
    run_cli("LoCoMo", ("locomo10.json",), URL, Path(__file__).with_name(".cache"), groups)
