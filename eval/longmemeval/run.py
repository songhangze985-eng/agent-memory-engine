"""Evaluate LongMemEval-S evidence without an external model or judge."""

import re
from datetime import datetime, timezone
from pathlib import Path

from eval.evidence_ablation import Question, Turn, run_cli, sample_rows

URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"


def _stamp(value):
    date = re.sub(r"\s*\([A-Za-z]+\)", "", value)
    pattern = "%Y/%m/%d %H:%M" if ":" in date else "%Y/%m/%d"
    return datetime.strptime(date, pattern).replace(tzinfo=timezone.utc).timestamp()


def category(row):
    return "abstention" if row["question_id"].endswith("_abs") else row["question_type"]


def parse_item(row):
    ordered = sorted(enumerate(zip(row["haystack_dates"], row["haystack_session_ids"], row["haystack_sessions"], strict=True)),
                     key=lambda item: (_stamp(item[1][0]), item[0]))
    turns = []
    for ordinal, (date, session_id, entries) in ordered:
        stamp = _stamp(date)
        for index, item in enumerate(entries):
            role = item["role"]
            turns.append(Turn(f"{session_id}:{index}", session_id, role, f"{role}: {item['content']}",
                              stamp + index / 1000, has_answer=item.get("has_answer") is True,
                              ingest_session=f"session-{ordinal}"))
    bucket = category(row)
    sessions = frozenset(row.get("answer_session_ids") or ())
    evidence = frozenset(turn.ref for turn in turns if turn.has_answer)
    latest, older = None, frozenset()
    if bucket == "knowledge-update":
        known_sessions = set(row["haystack_session_ids"])
        pairs = {(name[:-2] + "_1", name) for name in sessions & known_sessions
                 if name.endswith("_2") and name[:-2] + "_1" in known_sessions}
        if len(pairs) == 1:
            previous, latest = pairs.pop()
            if {previous, latest} <= {turn.session for turn in turns if turn.has_answer}:
                older = frozenset({previous})
            else:
                latest = None
    question = Question(row["question_id"], bucket, row["question"], _stamp(row["question_date"]),
                        evidence, sessions, bucket == "abstention", older, latest)
    return turns, question


def groups(data, count, seed):
    for row in sample_rows(data, count, seed, category):
        turns, question = parse_item(row)
        yield turns, [question]


if __name__ == "__main__":
    run_cli("LongMemEval-S", ("longmemeval_s.json", "longmemeval_s_cleaned.json"), URL,
            Path(__file__).with_name(".cache"), groups)
