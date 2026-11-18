"""Normalize one official trajectory into turns, a question, and evidence ids."""

from __future__ import annotations

import re
from datetime import datetime, timezone


_TIME = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2})")


def parse_timestamp(raw: str | None, fallback: float) -> float:
    match = _TIME.search(raw or "")
    if not match:
        return fallback
    moment = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    return moment.timestamp()


def _sessions(message_list) -> list[list[dict]]:
    if not message_list:
        return []
    if isinstance(message_list[0], dict):
        return [message_list]
    return [session for session in message_list if isinstance(session, list)]


def flatten_turns(message_list) -> list[dict]:
    turns = []
    for session_index, session in enumerate(_sessions(message_list)):
        for local_index, message in enumerate(session):
            if not isinstance(message, dict):
                continue
            sid = message.get("sid", message.get("mid", local_index))
            turns.append(
                {
                    "global": len(turns),
                    "session": session_index,
                    "local": local_index,
                    "sid": sid,
                    "message": message,
                }
            )
    return turns


def evidence_steps(target_step_id, turns: list[dict]) -> list[int]:
    """Map official target_step_id values onto flattened 0-based step ids."""
    found: list[int] = []
    for target in target_step_id or []:
        if isinstance(target, int):
            found.append(target)
            continue
        if isinstance(target, list) and len(target) == 1 and isinstance(target[0], int):
            found.append(target[0])
            continue
        if isinstance(target, list) and len(target) >= 2:
            local, session = target[0], target[1]
            matches = [
                turn["global"]
                for turn in turns
                if turn["session"] == session and (turn["local"] == local or turn["sid"] == local)
            ]
            found.append(matches[0] if matches else int(local))
    return found


def render_turn(turn: dict) -> str:
    message = turn["message"]
    place = message.get("place") or ""
    when = message.get("time") or ""
    if "message" in message:
        body = str(message.get("message") or "")
    else:
        user = message.get("user_message") or message.get("user") or ""
        agent = message.get("assistant_message") or message.get("assistant") or ""
        body = f"user: {user}\nagent: {agent}"
    return f"[step {turn['global']}] {body} (place: {place}; time: {when})".strip()


def normalize_trajectory(raw: dict, *, category: str, subtype: str, scenario: str) -> dict:
    turns = flatten_turns(raw.get("message_list") or [])
    qa = raw.get("QA") or {}
    choices = qa.get("choices") or {}
    if not isinstance(choices, dict):
        choices = {}
    ordered = {letter: str(choices[letter]) for letter in ("A", "B", "C", "D") if letter in choices}
    rendered = []
    for index, turn in enumerate(turns):
        text = render_turn(turn)
        stamp = parse_timestamp(turn["message"].get("time"), float(index))
        rendered.append({"text": text, "ts": stamp, "step": turn["global"]})
    return {
        "tid": raw.get("tid"),
        "category": category,
        "subtype": subtype,
        "scenario": scenario,
        "turns": rendered,
        "question": str(qa.get("question") or ""),
        "q_time": str(qa.get("time") or ""),
        "choices": ordered,
        "ground_truth": str(qa.get("ground_truth") or "").strip(),
        "answer_text": str(qa.get("answer") or ""),
        "evidence_steps": evidence_steps(qa.get("target_step_id"), turns),
    }


def format_question(item: dict) -> str:
    lines = [f"(current time is {item['q_time']}) {item['question']}"]
    for letter, text in item["choices"].items():
        lines.append(f"{letter}. {text}")
    lines.append("Reply with only one letter: A, B, C, or D.")
    return "\n".join(lines)
