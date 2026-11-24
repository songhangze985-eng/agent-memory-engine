"""Check frozen hashed retrieval floors and calendar, version and graph recall."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

# Question IDs plus official evidence IDs. Changing a gold label or resampling fails the gate.
COHORTS = {
    "02-exact-ident": (40, ("identifier", "type", "query"), "57390a94fefe684685bf494790fdb684c1a57f7b0deefe4895373baec6915684"),
    "03-membench": (99, ("id", "evidence_steps"), "242e3ff41acadaa245b4e3bc4dc41cd0bf141afd706a412f0accf12f0909c974"),
    "04-stress-precision": (40, ("id", "category", "evidence", "unanswerable"), "1c44211733a11250981256dda4101edaeb2cca1896f3768b989823bfdb33de9d"),
    "05-locomo": (100, ("id", "category", "evidence", "unanswerable"), "cbddc5a9e1135b48cb6f1a282dd20d679aea1d91de3c5fb45ccc3b499fd5a33b"),
    "06-longmemeval": (100, ("id", "category", "evidence", "unanswerable"), "b48f20f01bb2506694d02ef6384c8f0ba1ad31450208e592672e1990091529dc"),
    "07-bipolar-stress": (20, ("case",), "87dde88ebcbf5c3b633436467b41cff567d16a10bf23afd87c2a9e9078bda569"),
}
SHA256 = {
    "05-locomo": "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
    "06-longmemeval": "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
}
FLOORS = {
    "02-exact-ident": {"precision_at_1": ("==", 1), "mrr_at_10": ("==", 1),
                       "version_confusion_rate": ("==", 0), "missing_version_empty_rate": ("==", 1)},
    "03-membench": {"recall_at_5": (">=", .6056), "recall_at_10": (">=", .8008),
                     "mrr_at_10": (">=", .6999), "false_empty": ("==", 0), "answerable_n": ("==", 99), "avg_injected_tokens": ("<=", 545)},
    "04-stress-precision": {"recall_at_5": ("==", 1), "recall_at_10": ("==", 1), "mrr_at_10": ("==", 1),
                            "false_empty": ("==", 0), "answerable_n": ("==", 20), "avg_injected_tokens": ("<=", 180)},
    "05-locomo": {"recall_at_5": (">=", .3685), "recall_at_10": (">=", .4195), "mrr_at_10": (">=", .3428),
                   "false_empty": ("==", 0), "answerable_n": ("==", 80), "avg_injected_tokens": ("<=", 500)},
    "06-longmemeval": {"recall_at_5": (">=", .6134), "recall_at_10": (">=", .7087), "mrr_at_10": (">=", .5230),
                       "update_dominance": (">=", 4 / 7), "update_n": ("==", 14), "false_empty": ("==", 0),
                       "answerable_n": ("==", 86), "avg_injected_tokens": ("<=", 1950)},
    "07-bipolar-stress": {"recall_at_3": ("==", 1), "recall_at_5": ("==", 1), "mrr": ("==", 1),
                          "belt_compression": (">=", .8740), "tokens": ("<=", 125)},
}
TARGETS = {
    "03-membench": ("mrr_at_10", .6943),
    "05-locomo": ("recall_at_10", .4075),
    "06-longmemeval": ("recall_at_5", .6047),
}


def cohort_digest(items, fields):
    gold = sorted(tuple(item[field] for field in fields) for item in items)
    return hashlib.sha256(json.dumps(gold, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def compare(value, op, bound):
    if type(value) not in (int, float) or not math.isfinite(value):
        return False
    if op == "==":
        return value == bound
    if op == "<=":
        return value <= bound
    # Prior reports print percentages to 2 decimals and MRR to 4 decimals.
    return round(value, 4) >= round(bound, 4)


def validate(directory):
    checks, targets = [], []
    try:
        log = (directory / "01-pytest.log").read_text()
        result = re.findall(r"^=+ .*?$", log, re.M)[-1]
        passed = re.search(r"\b(\d+) passed\b", result)
        count = int(passed[1]) if passed and not re.search(r"\b(?:failed|errors?)\b", result) else 0
        pytest_exit = re.findall(r"^Exit code: (\d+)$", log, re.M)[-1]
        checks += [("pytest", "passed", count, ">= 127", count >= 127),
                   ("pytest", "exit", pytest_exit, "== 0", pytest_exit == "0")]
    except (OSError, IndexError):
        checks.append(("pytest", "completion record", "missing", ">= 127 and exit 0", False))

    for name, (count, fields, digest) in COHORTS.items():
        try:
            report = json.loads((directory / (name + ".json")).read_text())
            items = report["items"]
            matches = len(items) == count and cohort_digest(items, fields) == digest
            checks.append((name, "frozen cohort", len(items), str(count), matches))
            seed_ok = report.get("seeds") == [42, 43, 44, 45, 46] if name == "03-membench" else report.get("seed", 42) == 42
            protocol = report.get("backend") == "hashed" and report.get("mode", "mock") == "mock" and seed_ok
            protocol &= report.get("consolidated", False) is False
            if name in SHA256:
                protocol &= report.get("sha256") == SHA256[name]
            if name not in {"02-exact-ident", "07-bipolar-stress"}:
                protocol &= (report.get("top_k"), report.get("token_budget")) == (10, 2048)
            checks.append((name, "protocol", report.get("backend"), "frozen hashed", protocol))
            result = (report["metrics"]["Overall"] if name == "02-exact-ident" else
                      report["paired_retrieval"] if name == "03-membench" else
                      report["aggregate"] if name == "07-bipolar-stress" else report["overall"])["bipolar"]
            for key, (op, bound) in FLOORS[name].items():
                value = result.get(key)
                checks.append((name, key, value, f"{op} {bound}", compare(value, op, bound)))
            if name in TARGETS:
                key, bound = TARGETS[name]
                value = result.get(key)
                targets.append((name, key, value, f">= {bound}", compare(value, ">=", bound)))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            checks.append((name, "result", str(exc).replace("|", "/"), "valid report", False))
    return checks, targets


FULL_COHORTS = {
    "full-locomo": (1986, ("id", "evidence"), "2f9421693c3985907ce6fd4fd415a040eba3de45949a62bf18729af342bd56bb",
                    "overall", {"recall_at_5": .4472, "recall_at_10": .5006, "mrr_at_10": .3927, "avg_injected_tokens": 473.0624370594159}),
    "full-long": (500, ("id", "evidence"), "e97410a74f4c76d1bc19d3406ca857af5e29ec7c1585e32d0a74f934b7628332",
                  "overall", {"recall_at_5": .6179, "recall_at_10": .7087, "mrr_at_10": .5425, "update_dominance": .6,
                              "avg_injected_tokens": 1933.842}),
    "full-mem": (400, ("id", "evidence_steps"), "f1ca8487550e1f1e6ea2a95b5b768d3f0d0e225edf6f12a0cc9877232adca450",
                 "default.overall", {"recall_at_5": .5838, "recall_at_10": .7780, "mrr_at_10": .6589,
                                     "avg_injected_tokens": 549.875}),
    "exact-static": (120, ("type", "identifier", "query"), "8e9cffb2116f36aeea41127821ffe1defe41deeab3e976d641279c1a2a18295d",
                     "metrics.Overall", {"precision_at_1": 1, "mrr_at_10": 1, "symbol_confusion": 0,
                                         "negative_empty": 1, "procedural_top3": 1}),
}


def validate_full(directory):
    checks = []

    def add(name, metric, value, op, bound):
        checks.append((name, metric, value, f"{op} {bound}", compare(value, op, bound)))

    def load(name):
        return json.loads((directory / f"{name}.json").read_text())

    for name, (count, fields, digest, path, floors) in FULL_COHORTS.items():
        try:
            report = load(name)
            items = report["items"]
            add(name, "cohort verified", int(len(items) == count and cohort_digest(items, fields) == digest), "==", 1)
            add(name, "protocol", int(report.get("backend") == "hashed" and report.get("seed") == 42), "==", 1)
            if name in ("full-locomo", "full-long"):
                key = "05-locomo" if name == "full-locomo" else "06-longmemeval"
                add(name, "official snapshot", int(report["sha256"] == SHA256[key]), "==", 1)
            row = report
            for key in path.split("."):
                row = row[key]
            scores = row["bipolar"]
            for metric, bound in floors.items():
                op = "==" if name == "exact-static" else "<=" if metric == "avg_injected_tokens" else ">="
                add(name, metric, scores[metric], op, bound)
            if name != "exact-static":
                add(name, "false_empty", scores["false_empty"], "==", 0)
            if name == "full-long":
                add(name, "update_n", scores["update_n"], "==", 70)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            checks.append((name, "valid JSON", str(exc), "available", False))
    for name in ("long-horizon", "engineering"):
        try:
            report = load(name)
            if name == "long-horizon":
                add(name, "scale count", len(report["scales"]), "==", 3)
                for row, turns in zip(report["scales"], (500, 1500, 3500)):
                    add(name, "turns", row["turns"], "==", turns)
                    add(name, "cold wake", row["cold_wake_rate"], "==", 1)
                    add(name, "cold top 1", row["cold_top1"], "==", 1)
                    add(name, "taboo MRR", row["taboo_mrr"], "==", 1)
                    add(name, "current rank", row["current_rank"], "==", 1)
                    add(name, "hot reduction", row["hot_compression"], ">=", .7995)
                    # Commit af7ae61 added the 3-token "[禁忌] " prefix to taboo facts (+3 total tokens across 7 queries).
                    add(name, "injected tokens", row["average_tokens"], "<=", 1926 / 7)
            else:
                add(name, "leaks", report["isolation"]["leaks"], "==", 0)
                add(name, "isolation probes", report["isolation"]["queries"], "==", 200)
                add(name, "successful read probes", report["injection"]["read_probes"], "==", 30)
                add(name, "unauthorized changes", report["injection"]["unauthorized_changes"], "==", 0)
                add(name, "read-only writes", report["injection"]["readonly_writes"], "==", 0)
                add(name, "scale count", len(report["performance"]), "==", 3)
                for row, scale in zip(report["performance"], (1000, 5000, 10000)):
                    add(name, "active", row["active"], "==", scale)
                    add(name, f"{scale} SQL count", row["sql_max"], "<=", 15)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            checks.append((name, "valid JSON", str(exc), "available", False))
    return checks


def validate_memory_loop(directory):
    checks = []

    def add(name, metric, value, op, bound):
        checks.append((name, metric, value, f"{op} {bound}", compare(value, op, bound)))

    def load(name):
        return json.loads((directory / f"{name}.json").read_text())

    try:
        graph = load("graph-10k")
        add("graph-10k", "protocol", int((graph["seed"], graph["active"], graph["queries"]) == (42, 10000, 60)), "==", 1)
        add("graph-10k", "SQL count", graph["sql_max"], "<=", 15)
        add("graph-10k", "P95 ms", graph["p95_ms"], "<=", 100)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        checks.append(("graph-10k", "valid JSON", str(exc), "available", False))

    for name, count, fields, digest in (
        ("chrono", 60, ("group", "prompt", "expected"), "d81fccec7c1a8b988cd683cf4b0247bc4789aa1ce9ba4ffb2dba01b30041bb98"),
        ("version", 80, ("category", "query", "expected"), "318a99fd120f3df9531d74ae27900830df8a503eb25e4f57a0d678995d93ddb6"),
    ):
        try:
            items = load(name)["items"]
            add(name, "frozen cohort", int(len(items) == count and cohort_digest(items, fields) == digest), "==", 1)
            for field in (("pass",) if name == "chrono" else ("precision", "grade_correct")):
                add(name, field, sum(row[field] is True for row in items) / count, ">=", .95)
            if name == "version":
                add(name, "cross-version confusion", sum(row["confusion"] is not False for row in items), "==", 0)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            checks.append((name, "valid JSON", str(exc), "available", False))

    for name, reference, snapshot in (("chronicle-locomo", "full-locomo", "05-locomo"),
                                      ("chronicle-long", "full-long", "06-longmemeval")):
        try:
            report = load(name)
            items = report["items"]
            count, fields, digest, _, _ = FULL_COHORTS[reference]
            add(name, "frozen cohort", int(len(items) == count and cohort_digest(items, fields) == digest), "==", 1)
            protocol = (report["sha256"], report["seed"], report["mode"], report["backend"], report["token_budget"])
            add(name, "protocol", int(protocol == (SHA256[snapshot], 42, "mock", "hashed", 2048)), "==", 1)
            for mode in ("single_query", "reconstructive"):
                add(name, f"{mode} max tokens", max(row[mode]["injected_tokens"] for row in items), "<=", 2048)
            if name == "chronicle-locomo":
                multi = [row for row in items if row["category"] == "1 multi-hop"]
                add(name, "Multi-hop questions", len(multi), "==", 282)
                add(name, "Temporal questions", sum(row["category"] == "2 temporal" for row in items), "==", 321)

                def mean_score(mode, key):
                    scores = [row[mode][key] for row in multi if row[mode][key] is not None]
                    return sum(scores) / len(scores) if scores else 0

                for mode in ("single_query", "reconstructive"):
                    add(name, f"{mode} Multi-hop Recall@10 > 25.05%",
                        int(mean_score(mode, "recall_at_10") > .2505), "==", 1)
                gain = mean_score("reconstructive", "budget_recall") - mean_score("single_query", "budget_recall")
                add(name, "positive Multi-hop Recall@Budget gain", int(gain > 0), "==", 1)
                add(name, "Multi-hop Recall@10", mean_score("reconstructive", "recall_at_10"), ">=", .35454544)
                add(name, "Multi-hop MRR@10", mean_score("reconstructive", "mrr_at_10"), ">=", .35147331)
                rank_gain = mean_score("reconstructive", "recall_at_10") - mean_score("single_query", "recall_at_10")
                add(name, "positive Multi-hop Recall@10 gain", int(rank_gain > 0), "==", 1)
                for field, floor in (("recall_at_10", .62363820), ("mrr_at_10", .49687758)):
                    values = [row["reconstructive"][field] for row in items if row["reconstructive"][field] is not None]
                    add(name, f"reconstructive {field}", sum(values) / len(values), ">=", floor)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            checks.append((name, "valid JSON", str(exc), "available", False))
    return checks


def validate_workspace(directory, baseline):
    from eval.answer_evidence import ANSWER_FLOORS, METRIC_NAMES, PROTOCOL, answer_metrics, matches_cohort

    safety, quality, performance = [], [], []

    def add(rows, name, metric, value, op, bound):
        rows.append((name, metric, value, f"{op} {bound}", compare(value, op, bound)))

    try:
        log = (directory / "01-pytest.log").read_text()
        mandatory = (
            "test_working_memory_e2e.py::test_buffer_answer_and_restart_without_fact_write",
            "test_working_memory_e2e.py::test_reflection_stages_unsupported_claim_and_gate_promotes_verified_source",
            "test_working_memory_e2e.py::test_repeated_answers_and_default_recall_leave_weights_unchanged",
            "test_working_memory_e2e.py::test_expired_and_foreign_evidence_cannot_validate_a_citation",
            "test_working_memory_e2e.py::test_context_sql_has_one_session_read_and_at_most_two_task_reads",
            "test_working_memory_e2e.py::test_explicit_retain_moves_a_buffered_turn_without_copying_it",
            "test_working_memory_e2e.py::test_expired_verifier_is_downgraded_and_commit_rolls_back",
            "test_working_memory_e2e.py::test_promotion_keeps_its_verifying_facts_expiration",
            "test_storage_parity.py::test_postgres_contract[False]",
            "test_storage_parity.py::test_postgres_contract[True]",
        )
        for case in mandatory:
            add(safety, "pytest", case.split("::")[1], int(f"{case} PASSED" in log), "==", 1)
        compact = json.loads((directory / "compact-locomo.json").read_text())["overall"]["bipolar"]
        add(safety, "LoCoMo", "prefix reuse", compact["prefix_hit_rate"], "==", 1)
        add(safety, "LoCoMo", "prefix comparisons", compact["prefix_comparisons"], "==", 90)
        engineering = json.loads((directory / "engineering.json").read_text())["performance"]
        for row in engineering:
            for mode, task_reads in (("without_task", 0), ("with_task", 2)):
                stats = row["context_reads"][mode]
                add(safety, str(row["active"]), f"{mode} STM query count", stats["stm_sql_max"], "==", 1)
                add(safety, str(row["active"]), f"{mode} task query count", stats["wm_sql_max"], "<=", task_reads)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        safety.append(("workspace", "verification files", str(exc), "available", False))

    try:
        report = json.loads((directory / "answer-evidence.json").read_text())
        valid = report["protocol"] == PROTOCOL and all(matches_cohort(report[key]) for key in ("baseline", "items"))
        add(quality, "paired answers", "protocol and cohort", int(valid), "==", 1)
        if valid:
            before, after = map(answer_metrics, (report["baseline"], report["items"]))
            for key, floor in ANSWER_FLOORS.items():
                op = "<=" if key == "stale_fact_misuse_rate" else ">="
                passed = compare(after[key], op, floor) and compare(after[key], op, before[key])
                quality.append(("paired answers", METRIC_NAMES[key], f"{before[key]:.2%} → {after[key]:.2%}",
                                f"{op} {floor:.0%} and paired baseline", passed))
    except (OSError, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        quality.append(("paired answers", "verification files", str(exc), "available", False))

    for name, field in (("full-locomo", "overall"), ("full-long", "overall"),
                        ("03-membench", "paired_retrieval"), ("full-mem", "default.overall")):
        try:
            before = json.loads((baseline / f"{name}.json").read_text())
            after = json.loads((directory / f"{name}.json").read_text())
            count, fields, digest = COHORTS[name] if name in COHORTS else FULL_COHORTS[name][:3]
            same = all(len(report["items"]) == count and cohort_digest(report["items"], fields) == digest
                       for report in (before, after))
            protocol = ("backend", "mode", "seed", "seeds", "sha256", "top_k", "token_budget")
            if name != "full-mem":
                protocol += ("consolidated",)
            same &= all(before.get(key) == after.get(key) for key in protocol)
            add(performance, name, "paired retrieval protocol", int(same), "==", 1)
            for key in field.split("."):
                before, after = before[key], after[key]
            for key in ("recall_at_10", "mrr_at_10", "avg_injected_tokens"):
                old, new = before["bipolar"][key], after["bipolar"][key]
                op = "<=" if key == "avg_injected_tokens" else ">="
                performance.append((name, key, f"{old:.6f} → {new:.6f}", f"{op} baseline", same and compare(new, op, old)))
        except (OSError, KeyError, TypeError, ValueError) as exc:
            performance.append((name, "paired results", str(exc), "available", False))

    try:
        report = json.loads((directory / "paired-10k.json").read_text())
        protocol = report["active"] == 10000 and report["seed"] == 42 and report["queries"] == 60
        add(performance, "10K", "paired retrieval protocol", int(protocol), "==", 1)
        add(performance, "10K", "P95 ms", report["after"]["p95_ms"], "<=", 100)
        # The wall-clock comparison is repeated on the same fixture and alternating process order.
        add(performance, "10K", "P95 relative to paired baseline", report["after"]["p95_ms"],
            "<=", report["before"]["p95_ms"] * 1.05)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        performance.append(("10K", "paired results", str(exc), "available", False))
    return safety, quality, performance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("eval/results/final-sprint"))
    parser.add_argument("--full-static", action="store_true", help="Require all frozen expanded and engineering cohorts")
    parser.add_argument("--frozen-only", action="store_true", help="Verify historical results without memory-loop reports")
    parser.add_argument("--baseline-results", type=Path, help="Pre-workspace results for the paired retrieval comparison")
    args = parser.parse_args()
    checks, targets = validate(args.results)
    if args.full_static or not args.frozen_only:
        passed = next((row[2] for row in checks if row[:2] == ("pytest", "passed")), 0)
        checks.append(("pytest", "dual-backend baseline", passed, ">= 130", passed >= 130))
        checks.extend(validate_full(args.results))
    sections = [("Frozen regression floors", checks), ("Sprint targets", targets)]
    if not args.frozen_only:
        checks.extend(validate_memory_loop(args.results))
        workspace = validate_workspace(args.results, args.baseline_results or args.results / "baseline")
        sections.extend(zip(("Workspace safety and correctness", "Paired answer quality", "Paired retrieval and cost"), workspace))
    for name, rows in sections:
        print(f"\n{name}\n")
        print("| Benchmark | Metric | Actual | Required | Result |")
        print("| --- | --- | --- | --- | --- |")
        for suite, metric, actual, required, passed in rows:
            print(f"| {suite} | {metric} | {actual} | {required} | {'PASS' if passed else 'FAIL'} |")
    return 0 if all(row[-1] for name, rows in sections if name != "Sprint targets" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
