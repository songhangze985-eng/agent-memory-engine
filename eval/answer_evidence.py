"""Pair pre-workspace and current answers with fixed, zero-API reasoner outputs."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import subprocess
import sys

from eval.static_report import table, write_result
from agentmem_core.engine import CognitiveMemoryEngine
from agentmem_core.llm import MockLLMClient
from agentmem_core.reranker import CalibratedLocalReranker
from agentmem_core.vector import HashedTfidfEmbedder


def answer_cases():
    settings = (
        ("Atlas", "runtime", "uses Python v3.11.4", "uses Python v3.11.2"),
        ("Beacon", "deployment date", "deployed on 2024-06-15", "deployed on 2024-05-15"),
        ("Cedar", "budget", "has a budget of 250 euros", "has a budget of 2500 euros"),
        ("Draco", "region", "uses the Tokyo region", "uses the Sydney region"),
        ("Echo", "inspection date", "passed inspection on 2024-02-29", "passed inspection on 2024-02-28"),
        ("Fjord", "build number", "uses build ALPHA-74", "uses build ALPHA-75"),
        ("Grove", "storage version", "uses PostgreSQL v16.2", "uses PostgreSQL v16.3"),
        ("Harbor", "checkpoint", "has checkpoint KEY-982", "has checkpoint KEY-983"),
        ("Iris", "temperature", "requires 18 degrees", "requires 28 degrees"),
        ("Juniper", "retry limit", "allows 3 retries", "allows 13 retries"),
    )
    for name, slot, true_value, false_value in settings:
        query = f"What is the current {name} {slot}?"
        support, false = f"{name} {true_value}.", f"{name} {false_value}."
        for case, candidate in (("retained", support), ("buffered", support), ("conflicting", false),
                                ("stale", false), ("missing", false)):
            yield {"id": f"{name.casefold()}-{case}", "case": case, "query": query,
                   "support": support, "proposal": candidate, "decoy": f"{name} tracks a separate {slot} audit."}


class _AnswerProposal(MockLLMClient):
    """Give both revisions the same reasoner output for each question."""

    def __init__(self, proposal):
        self.proposal, self.citations = proposal, []

    def complete(self, prompt):
        if "慢系统综合推理与作答器" not in prompt.split("# Input", 1)[0]:
            return super().complete(prompt)
        return json.dumps({"final_answer": self.proposal, "evidence_synthesis": "", "anti_intuitive_override": False,
                           "rejected_noise_ids": [], "claims": [{"text": self.proposal, "evidence_ids": self.citations}]})


def capture_answers():
    items = []
    for case in answer_cases():
        model = _AnswerProposal(case["proposal"])
        engine = CognitiveMemoryEngine(llm=model, vector=HashedTfidfEmbedder(), reranker=CalibratedLocalReranker(),
                                       enable_casual_bypass=False)
        with closing(engine.store):
            kind = case["case"]
            if kind == "buffered":
                turn = engine.buffer_turn(case["support"], timestamp=100)
                model.citations = [f"turn:{turn if turn is not None else 1}"]
            elif kind in {"retained", "conflicting", "stale"}:
                text = case["proposal"] if kind == "stale" else case["support"]
                expiry = {"ttl_seconds": 1} if kind == "stale" else {}
                turn = engine.retain(text, timestamp=100, **expiry)
                model.citations = [f"fact:{turn['operations'][0]['line_id']}"]
            if kind in {"stale", "missing"}:
                engine.retain(case["decoy"], timestamp=101)
                if kind == "missing":
                    model.citations = ["fact:l1"]
            result = engine.answer(case["query"], timestamp=110)
            response = result["response"]
            items.append({**case, "response": response, "cited_proposal": model.citations,
                          "abstained": result.get("abstained", response in {
                              "", "No relevant memory found.", "没有找到相关记忆。", "Insufficient evidence to answer."})})
    return items


def answer_metrics(items):
    surfaced = [row for row in items if not row["abstained"]]
    answerable = [row for row in items if row["case"] in {"retained", "buffered"}]
    stale = [row for row in items if row["case"] == "stale"]
    negatives = [row for row in items if row["case"] in {"conflicting", "stale", "missing"}]

    def correct(row):
        return not row["abstained"] and row["case"] in {"retained", "buffered"} and row["response"] == row["support"]

    return {"claim_support_rate": sum(map(correct, surfaced)) / max(1, len(surfaced)),
            "stale_fact_misuse_rate": sum(not row["abstained"] and row["proposal"] in row["response"] for row in stale) / len(stale),
            "abstention_rate": sum(row["abstained"] for row in negatives) / len(negatives),
            "answerable_accuracy": sum(map(correct, answerable)) / len(answerable)}


ANSWER_FLOORS = {"claim_support_rate": 1.0, "stale_fact_misuse_rate": 0.0,
                 "abstention_rate": 1.0, "answerable_accuracy": .95}
METRIC_NAMES = {"claim_support_rate": "断言证据支持率", "stale_fact_misuse_rate": "过期事实误用率",
                "abstention_rate": "无证据拒答率", "answerable_accuracy": "可回答题正确率"}
PROTOCOL = "fixed-reasoner-zero-api-v1"


def matches_cohort(items):
    expected = list(answer_cases())
    return len(items) == len(expected) and all(
        all(row[key] == value for key, value in case.items()) for row, case in zip(items, expected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    baseline = parser.add_mutually_exclusive_group(required=True)
    baseline.add_argument("--baseline-repo", type=Path, help="pre-workspace source checkout")
    baseline.add_argument("--baseline", type=Path, help="recorded baseline answer JSON")
    parser.add_argument("--out", type=Path, default=Path("eval/results/final-sprint/answer-evidence.json"))
    args = parser.parse_args()
    if args.baseline_repo:
        code = ("import json,runpy,sys;"
                "capture=runpy.run_path(sys.argv[1])['capture_answers'];"
                "print(json.dumps(capture(),ensure_ascii=False))")
        before_items = json.loads(subprocess.check_output(
            [sys.executable, "-c", code, str(Path(__file__).resolve())], cwd=args.baseline_repo, text=True))
    else:
        before_items = json.loads(args.baseline.read_text())["baseline"]
    if not matches_cohort(before_items):
        parser.error("baseline uses a different answer cohort")
    items = capture_answers()
    before, after = answer_metrics(before_items), answer_metrics(items)
    report = {"benchmark": "Paired Answer Evidence", "protocol": PROTOCOL,
              "baseline": before_items, "items": items, "before": before, "after": after}
    write_result(report, args.out, table(("指标", "改造前", "改造后", "门槛"), [
        (name, f"{before[key]:.2%}", f"{after[key]:.2%}",
         ("≤ " if key == "stale_fact_misuse_rate" else "≥ ") + f"{ANSWER_FLOORS[key]:.0%}")
        for key, name in METRIC_NAMES.items()
    ]))
    return int(any((after[key] > limit if key == "stale_fact_misuse_rate" else after[key] < limit)
                   for key, limit in ANSWER_FLOORS.items()))
