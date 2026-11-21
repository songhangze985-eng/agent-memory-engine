"""Generate grouped, fully synthetic evidence and abstention cases for offline calibration."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import random


@dataclass(frozen=True)
class Evidence:
    slot: str
    text: str
    values: tuple[str, ...]
    role: str = "user"


@dataclass(frozen=True)
class Scenario:
    intent: str
    query: str
    distractor: str
    evidence: tuple[Evidence, ...]


SCENARIOS = (
    Scenario("single-session-user",
             "Where did I leave the spare key after the {event} rehearsal?",
             "Before the {event} rehearsal I made sure there was a spare key.",
             (Evidence("place", "After the {event} rehearsal I left the spare key {value}.",
                       ("in the prop cabinet", "under the sound desk", "in the costume drawer",
                        "in the equipment case", "beside the ticket tray")),)),
    Scenario("single-session-assistant",
             "Which documentary did you recommend for the {event} discussion?",
             "For the {event} discussion I asked for a documentary recommendation.",
             (Evidence("documentary", "For the {event} discussion I recommended the documentary {value}.",
                       ("A Map of Rain", "Listening to Winter", "A Room of Tides", "The Quiet Orbit",
                        "Where Lanterns Drift"), "assistant"),)),
    Scenario("single-session-preference",
             "What snack do I prefer during {event} workshops?",
             "At the {event} workshop we put snacks out before the break.",
             (Evidence("snack", "During {event} workshops I prefer {value} as a snack.",
                       ("salted pistachios", "dried apricots", "rice crackers", "apple slices", "roasted chickpeas")),)),
    Scenario("multi-session",
             "Which book and which podcast did I choose for the {event} commute?",
             "On the {event} commute I compared a few options for books and podcasts.",
             (Evidence("book", "For the {event} commute I chose the book {value}.",
                       ("Ink Harbor", "The Orchard Wall", "A Balcony of Stars", "Under Paper Skies", "The Ledger Tree")),
              Evidence("podcast", "For the {event} commute I chose the podcast {value}.",
                       ("Quiet Archives", "Footnotes at Dawn", "City Echoes", "The Field Record", "Passing Beacons")))),
    Scenario("temporal-reasoning",
             "On what date did I send my first application for {event}?",
             "I planned to send an application for {event} in early March.",
             (Evidence("sent_date", "I sent my first application for {event} on {value}.",
                       ("2024-03-12", "2024-03-15", "2024-03-18", "2024-03-22", "2024-03-26")),)),
    Scenario("knowledge-update",
             "Who is the current coordinator for {event}?",
             "Tarin Vale used to be the coordinator for {event}.",
             (Evidence("current_coordinator", "For {event}, {value} has replaced Tarin Vale as coordinator.",
                       ("Cella Wren", "Neris Brook", "Elian March", "Vara Bell", "Dalen Rose")),)),
)

EVENT_STARTS = ("Amber Birch Cedar Clover Coral Elm Fern Hazel Indigo Iris Juniper Lark Laurel Maple "
                "Moss Olive Onyx Opal Orchid Pine Quartz Reed Rowan Sage Saffron Silver Slate Sorrel Spruce Willow").split()
EVENT_ENDS = ("Alcove Arch Bank Base Bay Beacon Bell Bend Bluff Bridge Brook Canal Cape Circle Cliff Court "
              "Cove Crest Dale Dock Dune Edge Fair Fen Field Fold Fork Gate Glen Grove Harbor Haven Heath Hill "
              "Hollow Isle Lake Lane Ledge Light Loft Marsh Meadow Mill Moor Nook Park Path Pier Plain Point "
              "Pond Port Quarry Reach Ridge Rock Shore Terrace Vale Yard").split()
GROUPS_PER_INTENT = 140
SEED = 42
START = 1711929600
DAY = 86400


def build_cases(*, seed=SEED, groups_per_intent=GROUPS_PER_INTENT):
    rng = random.Random(seed)
    events = []
    for start in EVENT_STARTS:
        endings = rng.sample(EVENT_ENDS, len(EVENT_ENDS))
        events.extend((f"{start} {left}", f"{start} {right}") for left, right in zip(endings[::2], endings[1::2]))
    rng.shuffle(events)
    if not 1 <= groups_per_intent <= len(events) // len(SCENARIOS):
        raise ValueError("groups per intent exceeds the unique event combinations")
    for scenario in SCENARIOS:
        for index in range(groups_per_intent):
            event, wrong_event = events.pop()
            group = f"{scenario.intent}-{index:03d}"
            common = {"group": group, "intent": scenario.intent, "user_id": "owner",
                      "split": "holdout" if index % 5 == 0 else "calibration", "timestamp": START + 10 * DAY}
            query = scenario.query.format(event=event)
            distractors = [
                {"ref": "d1", "user_id": "owner", "session_id": "background", "role": "user",
                 "text": scenario.distractor.format(event=event), "timestamp": START},
                {"ref": "d2", "user_id": "owner", "session_id": "background", "role": "user",
                 "text": f"I added the {wrong_event} meeting to my calendar.", "timestamp": START + 1},
            ]
            supports, expected = [], {}
            for offset, evidence in enumerate(scenario.evidence, 1):
                value = rng.choice(evidence.values)
                ref = f"e{offset}"
                supports.append({"ref": ref, "user_id": "owner", "session_id": f"evidence-{offset}",
                                 "timestamp": START + offset * DAY, "role": evidence.role,
                                 "text": evidence.text.format(event=event, value=value)})
                expected[evidence.slot] = {"source_ref": ref, "value": value}
            unknown = dict.fromkeys(expected)
            variants = [
                ("answerable", "complete", query, distractors + supports, expected),
                ("evidence_removed", "unsupported", query, distractors, unknown),
                ("false_premise", "unsupported", scenario.query.format(event=wrong_event),
                 distractors + supports, unknown),
                ("cross_user", "unsupported", query,
                 distractors + [{**turn, "user_id": "neighbor"} for turn in supports], unknown),
            ]
            if len(supports) > 1:
                omitted = rng.choice(list(expected))
                partial = {**expected, omitted: None}
                kept = [turn for turn in supports if turn["ref"] != expected[omitted]["source_ref"]]
                variants.append(("partial_evidence", "partial", query, distractors + kept, partial))
            for variant, outcome, question, turns, slots in variants:
                yield {**common, "id": f"{group}/{variant}", "variant": variant, "outcome": outcome,
                       "query": question, "turns": turns, "expected_slots": slots}


def write_cohort(path, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256()
    counts = Counter()
    with path.open("wb") as stream:
        for case in build_cases(seed=seed):
            line = (json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            stream.write(line)
            digest.update(line)
            counts[case["split"], case["variant"]] += 1
    return {"seed": seed, "groups_per_intent": GROUPS_PER_INTENT, "sha256": digest.hexdigest(),
            "counts": {split: {variant: count for (part, variant), count in sorted(counts.items()) if part == split}
                       for split in ("calibration", "holdout")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Destination JSONL file")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    metadata = write_cohort(args.out, args.seed)
    args.out.with_suffix(".manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                                       encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
