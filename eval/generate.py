"""Seeded dataset builders. Sizes follow the spec; ``small`` is about 10–20%."""

from __future__ import annotations

import random

from eval.corpus import (
    CHANGE,
    FACTS,
    FILLER,
    GREETINGS,
    HABIT,
    MEETING_QUESTION,
    NOISE_SENTENCES,
    SCENARIOS,
    TEST4_BACKGROUND,
    TRAPS,
)
from memengine.textutil import estimate_tokens

SCALES = {
    "small": {
        "noise_tokens": 12_000,
        "n_facts": 4,
        "gap_rounds": 10,
        "reinforce": 6,
        "post_change_gap": 1,
        "n_scenarios": 2,
        "hop_gap": 16,
        "n_casual": 10,
        "n_traps": 10,
    },
    "full": {
        "noise_tokens": 100_000,
        "n_facts": 20,
        "gap_rounds": 50,
        "reinforce": 30,
        "post_change_gap": 3,
        "n_scenarios": 2,
        "hop_gap": 80,
        "n_casual": 50,
        "n_traps": 50,
    },
}


def _stamp(utterances: list[dict], start: float = 1_700_000_000.0, step: float = 1.0) -> list[dict]:
    stamped = []
    clock = start
    for item in utterances:
        stamped.append({**item, "ts": clock})
        clock += step
    return stamped


def _pack_noise(rng: random.Random, target_tokens: int) -> list[str]:
    packed = []
    total = 0
    while total < target_tokens:
        parts = []
        while estimate_tokens("".join(parts)) < 90:
            sentence = rng.choice(NOISE_SENTENCES)
            if not sentence.endswith("。"):
                sentence += "。"
            parts.append(sentence)
        text = "".join(parts)
        packed.append(text)
        total += estimate_tokens(text)
    return packed


def build_test1(cfg: dict, rng: random.Random) -> dict:
    facts = FACTS[: cfg["n_facts"]]
    noise = _pack_noise(rng, cfg["noise_tokens"])
    inserts = sorted(rng.sample(range(len(noise) + 1), len(facts)))
    sequence = []
    fact_iter = iter(facts)
    cursor = 0
    for position in inserts:
        while cursor < position:
            sequence.append(noise[cursor])
            cursor += 1
        sequence.append(next(fact_iter)["text"])
    sequence.extend(noise[cursor:])
    sequence.extend(rng.choice(NOISE_SENTENCES) + "。" for _ in range(cfg["gap_rounds"]))
    utterances = _stamp([{"text": text, "kind": "ingest"} for text in sequence])
    questions = []
    ask_at = utterances[-1]["ts"] + 30
    for offset, fact in enumerate(facts):
        questions.append(
            {
                "query": fact["question"],
                "keywords": list(fact["keywords"]),
                "forbidden": [],
                "expected_route": "TASK_POISON_CIRCLE",
                "measure_compression": True,
                "ts": ask_at + offset,
            }
        )
    return {"name": "test1", "utterances": utterances, "questions": questions}


def build_test2(cfg: dict, rng: random.Random) -> dict:
    del rng
    rows = [{"text": HABIT, "kind": "ingest"} for _ in range(cfg["reinforce"])]
    rows.append({"text": CHANGE, "kind": "ingest"})
    rows.extend({"text": FILLER, "kind": "ingest"} for _ in range(cfg["post_change_gap"]))
    utterances = _stamp(rows, start=1_700_100_000.0)
    questions = [
        {
            "query": MEETING_QUESTION,
            "keywords": ["周五"],
            "forbidden": ["周三"],
            "expected_route": "TASK_POISON_CIRCLE",
            "measure_compression": True,
            "ts": utterances[-1]["ts"] + 30,
        }
    ]
    return {"name": "test2", "utterances": utterances, "questions": questions}


def _other_numbers(scenario_index: int, keep: str) -> list[str]:
    banned = []
    for index, scenario in enumerate(SCENARIOS):
        for number in scenario["budgets"]:
            if index == scenario_index and number == keep:
                continue
            banned.append(number)
    return banned


def build_test3(cfg: dict, rng: random.Random) -> dict:
    del rng
    rows = []
    scenarios = SCENARIOS[: cfg["n_scenarios"]]
    for scenario in scenarios:
        person, place, event = scenario["person"], scenario["place"], scenario["event"]
        initial, first, second = scenario["budgets"]
        rows.append({"text": f"{person}将前往{place}参加{event}。"})
        rows.extend({"text": FILLER} for _ in range(cfg["hop_gap"]))
        rows.append({"text": f"{event}的预算为{initial}元。"})
        rows.extend({"text": FILLER} for _ in range(cfg["hop_gap"]))
        rows.append({"text": f"{event}的预算改为{first}元。"})
        rows.extend({"text": FILLER} for _ in range(cfg["hop_gap"]))
        rows.append({"text": f"{event}的预算改为{second}元。"})
    utterances = _stamp([{"text": row["text"], "kind": "ingest"} for row in rows], start=1_700_200_000.0)
    questions = []
    ask_at = utterances[-1]["ts"] + 50
    for index, scenario in enumerate(scenarios):
        initial, first, second = scenario["budgets"]
        person = scenario["person"]
        questions.append(
            {
                "query": f"{person}这次出行的预算是多少？",
                "keywords": [second],
                "forbidden": _other_numbers(index, second),
                "expected_route": "TASK_POISON_CIRCLE",
                "measure_compression": True,
                "ts": ask_at,
            }
        )
        ask_at += 5
        questions.append(
            {
                "query": f"在第二次修改预算之前，{person}这次出行的预算是多少？",
                "keywords": [first],
                "forbidden": _other_numbers(index, first),
                "expected_route": "TASK_POISON_CIRCLE",
                "measure_compression": True,
                "ts": ask_at,
            }
        )
        ask_at += 5
    return {"name": "test3", "utterances": utterances, "questions": questions}


def _expand(items: list, target: int, label: str) -> list:
    if target <= len(items):
        return items[:target]
    expanded = []
    copy_id = 0
    while len(expanded) < target:
        for item in items:
            if len(expanded) >= target:
                break
            if copy_id == 0:
                expanded.append(item)
            else:
                suffix = f"（{copy_id}）"
                if label == "greeting":
                    expanded.append(item + suffix)
                else:
                    clone = dict(item)
                    clone["query"] = item["query"] + suffix
                    expanded.append(clone)
        copy_id += 1
    return expanded


def build_test4(cfg: dict, rng: random.Random) -> dict:
    del rng
    greetings = _expand(GREETINGS, cfg["n_casual"], "greeting")
    traps = _expand(TRAPS, cfg["n_traps"], "trap")
    background = _stamp(
        [{"text": text, "kind": "ingest"} for text in TEST4_BACKGROUND],
        start=1_700_300_000.0,
    )
    questions = []
    ask_at = background[-1]["ts"] + 40
    for text in greetings:
        questions.append(
            {
                "query": text,
                "keywords": [],
                "forbidden": [],
                "expected_route": "CASUAL_2D_BYPASS",
                "trap": False,
                "measure_compression": False,
                "ts": ask_at,
            }
        )
        ask_at += 1
    for trap in traps:
        questions.append(
            {
                "query": trap["query"],
                "keywords": [],
                "forbidden": [],
                "expected_route": trap["expected_route"],
                "trap": True,
                "false_assertion": trap.get("false_assertion") or "",
                "protected": list(trap.get("protected") or []),
                "measure_compression": False,
                "ts": ask_at,
            }
        )
        ask_at += 1
    return {"name": "test4", "utterances": background, "questions": questions}


def build_all(scale: str, seed: int) -> dict:
    if scale not in SCALES:
        raise KeyError(scale)
    cfg = SCALES[scale]
    return {
        "scale": scale,
        "seed": seed,
        "config": cfg,
        "tests": {
            "test1": build_test1(cfg, random.Random(seed)),
            "test2": build_test2(cfg, random.Random(seed + 1)),
            "test3": build_test3(cfg, random.Random(seed + 2)),
            "test4": build_test4(cfg, random.Random(seed + 3)),
        },
    }
