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
from pathlib import Path

from eval.bipolar_ablation import TOKEN_BUDGET, TOP_K, paired_retrieve
from eval.membench.score import mean
from memengine.engine import CognitiveMemoryEngine
from memengine.llm import MockLLMClient
from memengine.textutil import estimate_tokens
from memengine.vector import HashedTfidfEmbedder

MODES = ("bipolar", "rrf")


@dataclass(frozen=True)
class Turn:
    ref: str
    session: str
    role: str
    text: str
    timestamp: float
    has_answer: bool = False
    ingest_session: str | None = None


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
    except (OSError, EOFError) as exc:
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


def evidence_metrics(ranked_refs, gold):
    if not gold:
        return {"recall_at_5": None, "recall_at_10": None, "mrr_at_10": None}
    hits = [set(refs) & gold for refs in ranked_refs[:10]]
    first = next((rank for rank, refs in enumerate(hits, 1) if refs), None)
    return {"recall_at_5": len(set().union(*hits[:5])) / len(gold),
            "recall_at_10": len(set().union(*hits)) / len(gold),
            "mrr_at_10": 1 / first if first else 0.0}


def fact_turns(fact, turns):
    if fact.get("context_truncated"):
        return []
    return [turns[index] for index in fact.get("source_step_ids", []) if 0 <= index < len(turns)]


def measure(facts, turns, question, archive):
    mapped = [fact_turns(fact, turns) for fact in facts]
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
        latest_rank = next((rank for rank, refs in enumerate(versions, 1) if question.latest_session in refs), None)
        archived_by_rank = [set().union(*(version_refs(fact_turns(revision, turns)) & question.older_sessions
                                         for revision in archive.get(fact["id"], ())))
                            if not fact.get("historical") and question.latest_session in refs else set()
                            for fact, refs in zip(facts, versions)]
        archived = any(archived_by_rank)
        first_old = next((rank for rank, (refs, archived_refs) in enumerate(zip(versions, archived_by_rank), 1)
                          if (refs & question.older_sessions) - archived_refs), None)
        row.update(update_dominates=bool(latest_rank is not None and
                                        (archived if first_old is None else latest_rank < first_old)),
                   update_latest_rank=latest_rank, update_old_rank=first_old, update_archive_verified=archived)
    return row


def _evaluate_group(group):
    turns, questions = group
    rows = []
    engine = CognitiveMemoryEngine(llm=MockLLMClient(), vector=HashedTfidfEmbedder(), enable_casual_bypass=False)
    with closing(engine.store):
        for index, turn in enumerate(turns):
            engine.add(f"[step {index}] {turn.text}", timestamp=turn.timestamp,
                       session_id=turn.ingest_session or turn.session, role=turn.role)
        sources = {turn.ref for turn in turns}
        for question in questions:
            result, baseline, candidates = paired_retrieve(engine, question.text, question.timestamp)
            active = result["ranked"]
            archive = {}
            if question.latest_session:
                ids = {fact["id"] for fact in active + baseline if not fact.get("historical") and fact["revision"]}
                for revision in engine.store.historical_versions(ids, scope=("default", "default", None)):
                    for fact in active + baseline:
                        if fact["id"] in ids and fact["root_fact_id"] == revision["root_fact_id"]:
                            archive.setdefault(fact["id"], []).append(revision)
            rows.append({"id": question.id, "category": question.category, "unanswerable": question.unanswerable,
                         "evidence": sorted(question.evidence), "evidence_sessions": sorted(question.evidence_sessions),
                         "unmapped_evidence": sorted(question.evidence - sources), "latest_session": question.latest_session,
                         "older_sessions": sorted(question.older_sessions), "n_turns": len(turns),
                         "candidate_count": len(candidates), "outer": result["outer_triggered_count"],
                         "belt_pruned": result["belt_pruned_count"], "belt_compressed": result["belt_compressed_count"],
                         "bipolar": measure(active, turns, question, archive),
                         "rrf": measure(baseline, turns, question, archive)})
    return rows


def paired_rows(groups, workers=1):
    def collect(batches):
        rows = []
        for batch in batches:
            before = len(rows)
            rows.extend(batch)
            if before // 10 != len(rows) // 10:
                print(f"Evaluated {len(rows)} questions", flush=True)
        return rows

    if workers == 1:
        return collect(map(_evaluate_group, groups))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return collect(pool.map(_evaluate_group, groups))


def summarize(rows):
    report = {"n": len(rows), "answerable": sum(not row["unanswerable"] for row in rows),
              "unmapped_evidence_questions": sum(bool(row["unmapped_evidence"]) for row in rows),
              "belt_pruned": sum(row["belt_pruned"] for row in rows)}
    for mode in MODES:
        def avg(key, *, session=False):
            scores = [(row[mode]["session"] if session else row[mode]).get(key) for row in rows]
            return mean([value for value in scores if value is not None])

        report[mode] = {**{key: avg(key) for key in ("recall_at_5", "recall_at_10", "mrr_at_10")},
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
    parser = argparse.ArgumentParser(description=f"{benchmark} offline paired evidence retrieval")
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=default_cache, help="Local JSON file or download/cache directory")
    parser.add_argument("--out", type=Path, default=Path(f"{benchmark.lower()}-ablation.json"))
    parser.add_argument("--workers", type=int, default=1, help="Parallel independent conversations or haystacks")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        path = resolve_dataset(args.data_dir, names, url)
        data = read_dataset(path)
        rows = paired_rows(group_loader(data, args.sample, args.seed), args.workers)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"{benchmark}: {exc}\n")
    with path.open("rb") as stream:
        sha = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    categories = sorted({row["category"] for row in rows})
    report = {"benchmark": benchmark, "mode": "offline-evidence-only", "file": str(path.resolve()),
              "sha256": sha.hexdigest(), "seed": args.seed, "requested": args.sample, "top_k": TOP_K,
              "token_budget": TOKEN_BUDGET, "workers": args.workers, "overall": summarize(rows), "items": rows,
              "by_category": {category: summarize([row for row in rows if row["category"] == category])
                              for category in categories}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{benchmark}: {len(rows)} questions, seed={args.seed}, Top-{TOP_K}, {TOKEN_BUDGET} tokens; no LLM API")
    print(f"Dataset: {path.resolve()} (SHA-256: {report['sha256']})")
    print_tables(report)
    print(f"\nRaw JSON: {args.out.resolve()}")
    if any(report["overall"][mode]["answerable_empty"] for mode in MODES):
        parser.exit(1, "Acceptance failed: at least one answerable question had empty retrieval.\n")
