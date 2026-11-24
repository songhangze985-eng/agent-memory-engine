"""Shared offline evaluation; gold labels are only read after retrieval."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import shutil
import tempfile
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from dataclasses import dataclass, field
from functools import partial
from http.client import HTTPException
from pathlib import Path

from eval.metrics import evidence_metrics, mean, print_retrieval_reports, retrieval_report
from eval.paired import LABELS, TOKEN_BUDGET, TOP_K, add_backend_argument, cli_backend, evaluate_pair, make_engine, attach_consolidated_context
from agentmem_core.llm import build_llm_from_env
from agentmem_core.textutil import estimate_tokens

MODES = ("bipolar", "rrf")
OFFICIAL_SHA256 = {"LoCoMo": "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
                  "LongMemEval-S": "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"}


@dataclass(frozen=True)
class Turn:
    ref: str
    session: str
    role: str
    text: str
    timestamp: float
    has_answer: bool = False
    ingest_session: str | None = None
    mention_time: float | None = None
    speaker: str | None = None


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    text: str
    timestamp: float
    evidence: frozenset[str] = field(default_factory=frozenset)
    evidence_sessions: frozenset[str] = field(default_factory=frozenset)
    unanswerable: bool = False
    older_sessions: frozenset[str] = field(default_factory=frozenset)
    latest_session: str | None = None


def resolve_dataset(path: Path, names: tuple[str, ...], url: str) -> Path:
    if path.is_file():
        return path
    if path.suffix.lower() == ".json" and not path.is_dir():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    for name in names:
        if (path / name).is_file():
            return path / name
    path.mkdir(parents=True, exist_ok=True)
    dest = path / names[0]
    request = urllib.request.Request(url, headers={"User-Agent": "memengine-benchmark", "Accept-Encoding": "gzip"})
    temp_path = None
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with tempfile.NamedTemporaryFile(dir=path, delete=False) as tmp:
                temp_path = Path(tmp.name)
                if response.headers.get("Content-Encoding") == "gzip":
                    with gzip.GzipFile(fileobj=response) as decoded:
                        shutil.copyfileobj(decoded, tmp)
                else:
                    shutil.copyfileobj(response, tmp)
        temp_path.replace(dest)
    except (OSError, EOFError, HTTPException) as exc:
        raise RuntimeError(f"Could not download {url}: {exc}. Pass --data-dir /path/to/{names[0]} to run offline.") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return dest


def read_dataset(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Expected a nonempty JSON array in {path}")
    return payload


def sample_rows(rows, count, seed, category):
    if count is None:
        return list(rows)
    if count <= 0:
        raise ValueError("--sample must be positive")
    rng = random.Random(seed)
    groups = {}
    for row in rows:
        groups.setdefault(category(row), []).append(row)
    for group in groups.values():
        rng.shuffle(group)
    order = sorted(groups)
    rng.shuffle(order)
    chosen = []
    while order and len(chosen) < count:
        for name in order:
            chosen.append(groups[name].pop())
            if len(chosen) == count:
                break
        order = [name for name in order if groups[name]]
    return chosen


def fact_turns(fact, turns, raw_ids=None):
    if fact.get("context_truncated"):
        return []
    if raw_ids is not None:
        return [raw_ids[turn_id] for turn_id in fact.get("source_turn_ids", ()) if turn_id in raw_ids]
    return [turns[index] for index in fact.get("source_step_ids", []) if 0 <= index < len(turns)]


def measure(facts, turns, question, archive, *, raw_ids=None):
    mapped = [fact_turns(fact, turns, raw_ids) for fact in facts]
    ranked_turns = [{turn.ref for turn in entries} for entries in mapped]
    ranked_sessions = [{turn.session for turn in entries} for entries in mapped]
    row = {**evidence_metrics(ranked_turns, question.evidence if not question.unanswerable else set()),
           "session": evidence_metrics(ranked_sessions, question.evidence_sessions if not question.unanswerable else set()),
           "tokens": estimate_tokens("\n".join(fact["text"] for fact in facts)), "empty": not facts,
           "ranked_turns": [sorted(refs) for refs in ranked_turns],
           "ranked_sessions": [sorted(refs) for refs in ranked_sessions]}
    if question.unanswerable:
        noise = [fact["text"] for fact, refs in zip(facts, ranked_turns) if not refs.intersection(question.evidence)]
        row.update(noise_leakage=bool(noise), redundant_tokens=estimate_tokens("\n".join(noise)))
    if question.latest_session and question.older_sessions:
        def version_refs(entries):
            return {turn.session for turn in entries if turn.has_answer}

        versions = [version_refs(entries) for entries in mapped]
        latest_rank = next((rank for rank, (fact, refs) in enumerate(zip(facts, versions), 1)
                            if fact.get("status", "active") == "active" and question.latest_session in refs), None)
        archived_by_rank = [set().union(*(version_refs(fact_turns(revision, turns, raw_ids)) & question.older_sessions
                                         for revision in archive.get(fact["id"], ())))
                            if not fact.get("historical") and question.latest_session in refs else set()
                            for fact, refs in zip(facts, versions)]
        archived = any(archived_by_rank)
        first_old = next((rank for rank, (refs, archived_refs) in enumerate(zip(versions, archived_by_rank), 1)
                          if (refs & question.older_sessions) - archived_refs), None)
        old_downweighted = first_old is not None and facts[first_old - 1].get("status") == "superseded" \
            and facts[first_old - 1].get("status_weight", 1.0) < 1.0
        row.update(update_dominates=bool(latest_rank is not None and
                                        (archived if first_old is None else latest_rank < first_old or old_downweighted)),
                   update_latest_rank=latest_rank, update_old_rank=first_old, update_archive_verified=archived,
                   update_old_downweighted=old_downweighted)
    return row


def measure_pair(engine, question, pair, raw_ids, previous_prefix=None):
    full = [item for item in pair.context["items"] if item["kind"] == "fact"]
    facts = full + pair.baseline
    archive = {}
    if question.latest_session:
        ids = {fact["id"] for fact in facts if fact.get("status") == "active" and fact["revision"]}
        for revision in engine.store.historical_versions(ids, scope=("default", "default", None)):
            for fact in facts:
                if fact["id"] in ids and fact["root_fact_id"] == revision["root_fact_id"]:
                    archive.setdefault(fact["id"], []).append(revision)
    row = {}
    for label, ranked in (("bipolar", full), ("rrf", pair.baseline)):
        metrics = measure(ranked, (), question, archive, raw_ids=raw_ids)
        metrics.update(pair.efficiency(label, previous_prefix))
        if label == "bipolar":
            metrics["empty"] = pair.context["empty_recall"]
            metrics["gate_empty"] = pair.retrieved["empty_recall"]
            observations = [item for item in pair.context["items"]
                            if item["kind"] == "observation" or item.get("memory_type") == "observation"]
            supports = {turn.ref for item in observations for turn in fact_turns(item, (), raw_ids=raw_ids)}
            new_supports = supports.difference(ref for refs in metrics["ranked_turns"] for ref in refs)
            metrics.update(observation_count=len(observations),
                           observation_gold_supports=len(supports & question.evidence),
                           observation_only_gold_supports=len(new_supports & question.evidence))
            metrics["drilled_turn_refs"] = [raw_ids[turn_id].ref for turn_id in metrics["evidence_turn_ids"] if turn_id in raw_ids]
            if question.evidence and not question.unanswerable:
                metrics["drilled_evidence_recall"] = len(question.evidence.intersection(metrics["drilled_turn_refs"])) / len(question.evidence)
            if question.unanswerable:
                metrics["noise_leakage"] |= bool(set(metrics["drilled_turn_refs"]) - question.evidence)
        row[label] = metrics
    return row


def retain_source_turn(engine, index, turn, *, chronicle=True):
    provenance = dict(mention_time=turn.mention_time, speaker=turn.speaker) if chronicle else {}
    return engine.retain(f"[step {index}] {turn.text}", timestamp=turn.timestamp,
                         session_id=turn.ingest_session or turn.session, role=turn.role, **provenance)


def _evaluate_group(group, mode="mock", backend="hashed", consolidate=False):
    turns, questions = group
    rows = []
    engine = make_engine(mode, backend=backend)
    with closing(engine.store):
        raw_ids = {}
        for index, turn in enumerate(turns):
            stored = retain_source_turn(engine, index, turn, chronicle=False)
            raw_ids[stored["turn_id"]] = turn
        sources = {turn.ref for turn in turns}
        pairs = [evaluate_pair(engine, question.text, question.timestamp) for question in questions]
        if consolidate and questions:
            engine.consolidate(timestamp=max((turn.timestamp for turn in turns), default=questions[0].timestamp))
        previous_prefix = None
        for question, pair in zip(questions, pairs):
            if consolidate:
                pair = attach_consolidated_context(engine, pair, question.text)
            scores = measure_pair(engine, question, pair, raw_ids, previous_prefix)
            previous_prefix = scores["bipolar"]["prefix_version"]
            rows.append({"id": question.id, "category": question.category, "unanswerable": question.unanswerable,
                         "evidence": sorted(question.evidence), "evidence_sessions": sorted(question.evidence_sessions),
                         "unmapped_evidence": sorted(question.evidence - sources), "latest_session": question.latest_session,
                         "older_sessions": sorted(question.older_sessions), "n_turns": len(turns),
                         "candidate_count": len(pair.candidates), "outer": pair.retrieved["outer_triggered_count"],
                         "belt_pruned": pair.retrieved["belt_pruned_count"], "belt_compressed": pair.retrieved["belt_compressed_count"],
                         **scores})
    return rows


def paired_rows(groups, workers=1, mode="mock", backend="hashed", consolidate=False):
    def collect(batches):
        rows = []
        for batch in batches:
            before = len(rows)
            rows.extend(batch)
            if before // 10 != len(rows) // 10:
                print(f"Evaluated {len(rows)} questions", flush=True)
        return rows

    evaluate = partial(_evaluate_group, mode=mode, backend=backend, consolidate=consolidate)
    if workers == 1:
        return collect(map(evaluate, groups))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return collect(pool.map(evaluate, groups))


def summarize(rows):
    report = {"n": len(rows), "answerable": sum(not row["unanswerable"] for row in rows),
              "unmapped_evidence_questions": sum(bool(row["unmapped_evidence"]) for row in rows),
              "belt_pruned": sum(row["belt_pruned"] for row in rows)}
    for mode in MODES:
        def avg(key, *, session=False):
            scores = [(row[mode]["session"] if session else row[mode]).get(key) for row in rows]
            return mean([value for value in scores if value is not None])

        report[mode] = {**retrieval_report(rows, mode),
                        "turn_scored": sum(row[mode]["recall_at_10"] is not None for row in rows),
                        "session": {key: avg(key, session=True) for key in ("recall_at_5", "recall_at_10", "mrr_at_10")},
                        "session_scored": sum(row[mode]["session"]["recall_at_10"] is not None for row in rows),
                        "tokens": avg("tokens"), "empty": sum(row[mode]["empty"] for row in rows),
                        "answerable_empty": sum(row[mode]["empty"] and not row["unanswerable"] for row in rows),
                        "noise_leakage": avg("noise_leakage"), "redundant_tokens": avg("redundant_tokens"),
                        "update_dominance": avg("update_dominates"),
                        "update_scored": sum("update_dominates" in row[mode] for row in rows)}
    baseline = report["rrf"]["tokens"]
    report["token_savings"] = 1.0 - report["bipolar"]["tokens"] / baseline if baseline else None
    return report


def print_tables(report):
    def percent(value):
        return "—" if value is None else f"{value:.2%}"

    def number(value):
        return "—" if value is None else f"{value:.2f}"

    groups = {"Overall": report["overall"], **report["by_category"]}
    print_retrieval_reports((category, LABELS[mode], group[mode]) for category, group in groups.items() for mode in MODES)
    for sessions in (False, True):
        if sessions and not report["overall"]["rrf"]["session_scored"]:
            continue
        print("\nSession evidence" if sessions else "\nTurn evidence")
        print("| Category | N | Gold N | Mode | Recall@5 | Recall@10 | MRR@10 | Avg tokens | Empty (answerable) |")
        print("| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
        for category, group in groups.items():
            for mode in MODES:
                row = group[mode]
                metrics = row["session"] if sessions else row
                gold_n = row["session_scored" if sessions else "turn_scored"]
                print(f"| {category} | {group['n']} | {gold_n} | {mode} | {percent(metrics['recall_at_5'])} | "
                      f"{percent(metrics['recall_at_10'])} | {number(metrics['mrr_at_10'])} | "
                      f"{number(row['tokens'])} | {row['empty']} ({row['answerable_empty']}) |")
    print("\n| Category | Tokens saved vs RRF | Belt pruned | Noise leakage (bipolar / RRF) | "
          "Avg redundant tokens (bipolar / RRF) | Update dominance (bipolar / RRF; N) |")
    print("| --- | ---: | ---: | --- | --- | --- |")
    for category, group in groups.items():
        bipolar, rrf = (group[mode] for mode in MODES)
        print(f"| {category} | {percent(group['token_savings'])} | {group['belt_pruned']} | "
              f"{percent(bipolar['noise_leakage'])} / {percent(rrf['noise_leakage'])} | "
              f"{number(bipolar['redundant_tokens'])} / {number(rrf['redundant_tokens'])} | "
              f"{percent(bipolar['update_dominance'])} / {percent(rrf['update_dominance'])}; {bipolar['update_scored']} |")


def run_cli(benchmark, names, url, default_cache, group_loader):
    parser = argparse.ArgumentParser(description=f"{benchmark} paired evidence retrieval")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sample", type=int, default=100)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--chronicle", action="store_true", help="Use real speaker/calendar metadata and equal-budget graph ablation")
    parser.add_argument("--offline", action="store_true", help="Require an existing dataset file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=default_cache, help="Local JSON file or download/cache directory")
    parser.add_argument("--out", type=Path, default=Path(f"{benchmark.lower()}-ablation.json"))
    parser.add_argument("--workers", type=int, default=1, help="Parallel independent conversations or haystacks")
    parser.add_argument("--mode", choices=("mock", "live"), default="mock")
    parser.add_argument("--consolidate", action="store_true", help="Build a core profile after ingesting each conversation")
    add_backend_argument(parser)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if (args.offline or args.all or args.chronicle) and (args.backend != "hashed" or args.mode != "mock"):
        parser.error("static evaluation requires --mode mock --backend hashed")
    backend = cli_backend(parser, args.backend)
    if backend == "local-neural" and args.workers != 1:
        parser.error("local-neural requires --workers 1 so the model stays in one process")
    model = None
    if args.mode == "live":
        _, resolved, model = build_llm_from_env()
        if resolved != "live":
            parser.error("live mode requires LLM_API_KEY or OPENAI_API_KEY")
    try:
        if args.offline or args.all or args.chronicle:
            path = args.data_dir if args.data_dir.is_file() else next(
                (args.data_dir / name for name in names if (args.data_dir / name).is_file()), None)
            if path is None:
                raise FileNotFoundError(f"missing local data: {args.data_dir}")
        else:
            path = resolve_dataset(args.data_dir, names, url)
        count = None if args.all else args.sample
        if args.chronicle:
            if args.consolidate:
                parser.error("--chronicle uses a frozen ingest-only comparison")
            from eval.chronicle_bench import run
            return run(benchmark, group_loader, path, args.out, count, args.seed, args.workers)
        data = read_dataset(path)
        rows = paired_rows(group_loader(data, count, args.seed), args.workers, args.mode, backend, args.consolidate)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"{benchmark}: {exc}\n")
    with path.open("rb") as stream:
        sha = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    categories = sorted({row["category"] for row in rows})
    verified = sha.hexdigest() == OFFICIAL_SHA256.get(benchmark)
    report = {"benchmark": benchmark, "mode": args.mode, "backend": backend, "model": model, "file": str(path.resolve()), "dataset_source": url,
              "dataset_status": "verified official snapshot" if verified else "unverified input, not an official score",
              "sha256": sha.hexdigest(), "seed": args.seed, "requested": "all" if args.all else args.sample, "top_k": TOP_K,
              "token_budget": TOKEN_BUDGET, "workers": args.workers, "consolidated": args.consolidate,
              "prefix_budget": "separate" if args.consolidate else "included",
              "overall": summarize(rows), "items": rows,
              "by_category": {category: summarize([row for row in rows if row["category"] == category])
                              for category in categories}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{benchmark}: {len(rows)} questions, mode={args.mode}, backend={backend}, seed={args.seed}, Top-{TOP_K}, {TOKEN_BUDGET} tokens")
    print(f"Dataset source: {url}; {report['dataset_status']}; evidence scoring requires no judge API")
    print(f"Dataset: {path.resolve()} (SHA-256: {report['sha256']})")
    print_tables(report)
    from eval.static_report import official_table
    args.out.with_suffix(".md").write_text(official_table(report), encoding="utf-8")
    print(f"\nRaw JSON: {args.out.resolve()}")
    if any(report["overall"][mode]["answerable_empty"] for mode in MODES):
        parser.exit(1, "Acceptance failed: at least one answerable question had empty retrieval.\n")


if __name__ == "__main__":
    from eval.answer_evidence import main

    raise SystemExit(main())
