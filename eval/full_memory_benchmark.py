"""Full-course memory evaluation runner (v1 Ledger + v2 VFS) with MemBench-weighted scoring."""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from pathlib import Path

from eval.locomo.run import groups as locomo_groups
from eval.longmemeval.run import groups as longmemeval_groups
from eval.membench.catalog import CATEGORIES
from eval.membench.items import parse_timestamp, question_stem
from eval.membench_static import run as run_v1_membench, sample_items as sample_membench_items
from eval.metrics import percentile
from eval.vfs_ledger_ablation import (
    ablate_exact_identifiers,
    ablate_version_precision,
    bench_vfs_retrieval_and_output_optimizations,
)
from memengine.textutil import estimate_tokens
from memengine.vfs import VFSMemoryEngine

_STEP_PATH = re.compile(r"/steps/step_(\d+)\.md$")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _eval_vfs_membench_item(item: dict) -> dict:
    """Evaluate one official MemBench trajectory on v2 VFSMemoryEngine (apples-to-apples per-step)."""
    with tempfile.TemporaryDirectory() as tmp:
        vfs = VFSMemoryEngine(Path(tmp) / "mb.db", transcript_dir=Path(tmp) / "l0")
        with closing(vfs):
            t_ing = time.perf_counter()
            for idx, turn in enumerate(item["turns"]):
                step_id = int(turn.get("step", idx))
                ts = float(turn.get("ts", idx))
                tr = vfs.append_transcript(turn["text"], session_id="s1", timestamp=ts)
                vfs.write(
                    f"/steps/step_{step_id:04d}.md",
                    title=f"Step {step_id}",
                    summary=turn["text"][:160],
                    content=turn["text"],
                    layer="L2",
                    verification_status="verified",
                    source_transcript_ids=[tr.transcript_id],
                    timestamp=ts,
                )
            ingest_ms = (time.perf_counter() - t_ing) * 1000
            stamp = parse_timestamp(item.get("q_time"), max((turn["ts"] for turn in item["turns"]), default=0.0) + 1.0)
            stem = question_stem(item)

            c0 = time.process_time()
            t0 = time.perf_counter()
            hits = vfs.search(stem, reference_time=stamp, top_k=10)
            query_ms = (time.perf_counter() - t0) * 1000
            cpu_ms = (time.process_time() - c0) * 1000

            ctx_prog = vfs.build_context(
                stem, reference_time=stamp, token_budget=2048, disclosure="progressive", include_tree_outline=False, top_k=10
            )
            full_tokens = sum(estimate_tokens(h["detail_body"]) for h in hits)
            prog_tokens = ctx_prog["token_breakdown"]["dynamic_tokens"]

            ranked_steps: list[int] = []
            for h in hits:
                m = _STEP_PATH.search(h["path"])
                if m:
                    ranked_steps.append(int(m.group(1)))

            gold = set(int(s) for s in (item.get("evidence_steps") or []))
            if gold:
                r5 = len(gold & set(ranked_steps[:5])) / len(gold)
                r10 = len(gold & set(ranked_steps[:10])) / len(gold)
                first = next((rank for rank, s in enumerate(ranked_steps[:10], 1) if s in gold), None)
                mrr = 1.0 / first if first else 0.0
            else:
                r5 = r10 = mrr = None

            return {
                "category": item["category"],
                "subtype": item["subtype"],
                "has_gold": bool(gold),
                "recall_at_5": r5,
                "recall_at_10": r10,
                "mrr_at_10": mrr,
                "query_ms": query_ms,
                "cpu_ms": cpu_ms,
                "ingest_ms": ingest_ms,
                "prog_tokens": prog_tokens,
                "full_tokens": full_tokens,
            }


def eval_vfs_membench(data_dir: Path, *, seed: int = 42, per_category: int = 200, workers: int = 4) -> dict:
    """Evaluate v2 VFS on MemBench (4 official categories x per_category)."""
    items, missing = sample_membench_items(data_dir, seed, per_category)
    if workers <= 1:
        rows = [_eval_vfs_membench_item(it) for it in items]
    else:
        with ProcessPoolExecutor(workers) as pool:
            rows = list(pool.map(_eval_vfs_membench_item, items))

    def _agg(subset: list[dict]) -> dict:
        labeled = [r for r in subset if r["has_gold"]]
        durs = [r["query_ms"] for r in subset]
        prog = _mean([r["prog_tokens"] for r in subset])
        full = _mean([r["full_tokens"] for r in subset])
        return {
            "n": len(subset),
            "labeled": len(labeled),
            "recall_at_5": round(_mean([r["recall_at_5"] for r in labeled]), 6),
            "recall_at_10": round(_mean([r["recall_at_10"] for r in labeled]), 6),
            "mrr_at_10": round(_mean([r["mrr_at_10"] for r in labeled]), 6),
            "query_p50_ms": round(percentile(durs, 0.5), 2),
            "query_p95_ms": round(percentile(durs, 0.95), 2),
            "avg_prog_tokens": round(prog, 1),
            "avg_full_tokens": round(full, 1),
            "token_reduction_rate": round((full - prog) / full, 4) if full > 0 else 0.0,
        }

    return {
        "benchmark": "MemBench (v2 VFS)",
        "seed": seed,
        "per_category": per_category,
        "total_items": len(rows),
        "uncached_subtypes": missing,
        "overall": _agg(rows),
        "by_category": {cat: _agg([r for r in rows if r["category"] == cat]) for cat in CATEGORIES},
    }


def _eval_vfs_conversation_group(group_tuple) -> dict:
    turns, questions = group_tuple
    with tempfile.TemporaryDirectory() as tmp:
        vfs = VFSMemoryEngine(Path(tmp) / "vfs.db", transcript_dir=Path(tmp) / "l0")
        with closing(vfs):
            t_ing = time.perf_counter()
            by_session: dict[str, list[tuple[str, str, str, float | None]]] = {}
            turn_to_session: dict[str, str] = {}
            for turn in turns:
                rec = vfs.append_transcript(
                    turn.text,
                    session_id=turn.ingest_session or turn.session or "default",
                    role=turn.role,
                    timestamp=turn.timestamp,
                    speaker=turn.speaker,
                )
                by_session.setdefault(turn.session, []).append((rec.transcript_id, turn.ref, turn.text, turn.timestamp))
                turn_to_session[rec.transcript_id] = turn.session

            for ordinal, (orig_session, entries) in enumerate(by_session.items()):
                tids = [tid for tid, _, _, _ in entries]
                body = "\n".join(text for _, _, text, _ in entries)
                summary = entries[0][2][:160] if entries else orig_session
                ts_vals = [ts for _, _, _, ts in entries if ts is not None]
                session_ts = max(ts_vals) if ts_vals else float(ordinal + 1)
                vfs.write(
                    f"/sessions/s_{ordinal:04d}_{orig_session}.md",
                    title=f"Session {orig_session}",
                    summary=summary,
                    content=body,
                    layer="L2",
                    verification_status="verified",
                    source_transcript_ids=tids,
                    timestamp=session_ts,
                )
            ingest_ms = (time.perf_counter() - t_ing) * 1000
            l2_vecs = vfs.store.vector_index_count(user_id="default", agent_id="default")

            q_rows = []
            for q in questions:
                c0 = time.process_time()
                t0 = time.perf_counter()
                hits = vfs.search(q.text, reference_time=q.timestamp, top_k=10)
                q_ms = (time.perf_counter() - t0) * 1000
                c_ms = (time.process_time() - c0) * 1000

                gold_sessions = set(q.evidence_sessions) if q.evidence_sessions else {t.session for t in turns if t.ref in q.evidence}
                if not gold_sessions or q.unanswerable:
                    q_rows.append({"answerable": False, "q_ms": q_ms, "c_ms": c_ms})
                    continue

                ranked_sessions = []
                for hit in hits:
                    hit_sessions = {turn_to_session[tid] for tid in hit.get("source_transcript_ids", ()) if tid in turn_to_session}
                    ranked_sessions.append(hit_sessions)

                hit5 = set().union(*ranked_sessions[:5]) if ranked_sessions[:5] else set()
                hit10 = set().union(*ranked_sessions[:10]) if ranked_sessions[:10] else set()
                r5 = len(gold_sessions & hit5) / len(gold_sessions)
                r10 = len(gold_sessions & hit10) / len(gold_sessions)
                first = next((idx for idx, sset in enumerate(ranked_sessions[:10], 1) if sset & gold_sessions), None)
                mrr = 1.0 / first if first else 0.0
                upd = None
                if q.latest_session and q.older_sessions:
                    latest_rank = next((idx for idx, sset in enumerate(ranked_sessions, 1) if q.latest_session in sset), None)
                    older_rank = next((idx for idx, sset in enumerate(ranked_sessions, 1) if sset & set(q.older_sessions)), None)
                    upd = 1.0 if (latest_rank is not None and (older_rank is None or latest_rank < older_rank)) else 0.0
                q_rows.append({
                    "answerable": True,
                    "r5": r5,
                    "r10": r10,
                    "mrr": mrr,
                    "upd": upd,
                    "q_ms": q_ms,
                    "c_ms": c_ms,
                })
            return {"ingest_ms": ingest_ms, "l2_vecs": l2_vecs, "q_rows": q_rows}


def eval_vfs_full_dataset(name: str, dataset_path: Path, group_fn, *, sample: int | None, seed: int = 42, workers: int = 4) -> dict:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    grouped = list(group_fn(data, sample, seed))
    if workers <= 1:
        results = [_eval_vfs_conversation_group(g) for g in grouped]
    else:
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(_eval_vfs_conversation_group, grouped))

    all_q = [r for res in results for r in res["q_rows"]]
    ans = [r for r in all_q if r["answerable"]]
    upds = [r["upd"] for r in ans if r["upd"] is not None]
    durs = [r["q_ms"] for r in all_q]
    return {
        "dataset": name,
        "questions": len(all_q),
        "answerable": len(ans),
        "session_recall_at_5": round(_mean([r["r5"] for r in ans]), 6),
        "session_recall_at_10": round(_mean([r["r10"] for r in ans]), 6),
        "session_mrr_at_10": round(_mean([r["mrr"] for r in ans]), 6),
        "update_dominance": round(_mean(upds), 6) if upds else None,
        "update_n": len(upds),
        "avg_l2_vectors": round(_mean([res["l2_vecs"] for res in results]), 2),
        "query_p50_ms": round(percentile(durs, 0.5), 2),
        "query_p95_ms": round(percentile(durs, 0.95), 2),
        "query_cpu_total_ms": round(sum(r["c_ms"] for r in all_q), 1),
    }


def compute_weighted_summary(
    *,
    membench_r10: float,
    longmemeval_r10: float,
    locomo_r10: float,
    technical_p1: float,
) -> dict:
    """Transparent MemBench-weighted composite retrieval index (MemBench 40% / LongMemEval 25% / LoCoMo 20% / Tech 15%)."""
    weights = {"membench": 0.40, "longmemeval": 0.25, "locomo": 0.20, "technical": 0.15}
    score = (
        weights["membench"] * membench_r10
        + weights["longmemeval"] * longmemeval_r10
        + weights["locomo"] * locomo_r10
        + weights["technical"] * technical_p1
    )
    return {
        "weights": weights,
        "components": {
            "membench_r10": round(membench_r10, 6),
            "longmemeval_r10": round(longmemeval_r10, 6),
            "locomo_r10": round(locomo_r10, 6),
            "technical_p1": round(technical_p1, 6),
        },
        "weighted_composite": round(score, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locomo", type=Path, default=Path("/private/tmp/ame-benchmark-data/locomo10.json"))
    parser.add_argument("--longmemeval", type=Path, default=Path("/private/tmp/ame-benchmark-data/longmemeval_s_cleaned.json"))
    parser.add_argument("--memdata", type=Path, default=Path("/private/tmp/agent-memory-engine.Fl1ikV/eval/membench/.cache/MemData"))
    parser.add_argument("--membench-per-cat", type=int, default=200)
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 4) - 1)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-v1-membench", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    t0 = time.perf_counter()
    vfs_membench = eval_vfs_membench(args.memdata, seed=args.seed, per_category=args.membench_per_cat, workers=args.workers)
    vfs_locomo = eval_vfs_full_dataset("LoCoMo (Full 1,986Q)", args.locomo, locomo_groups, sample=None, seed=args.seed, workers=args.workers)
    vfs_longmem = eval_vfs_full_dataset("LongMemEval-S (Full 500Q)", args.longmemeval, longmemeval_groups, sample=None, seed=args.seed, workers=args.workers)
    vfs_exact = ablate_exact_identifiers(seed=args.seed)
    vfs_version = ablate_version_precision()
    vfs_opt = bench_vfs_retrieval_and_output_optimizations()

    tech_p1 = 0.5 * (vfs_exact["vfs_l2"]["precision_at_1"] + vfs_version["vfs_l2"]["precision_at_1"])
    vfs_weighted = compute_weighted_summary(
        membench_r10=vfs_membench["overall"]["recall_at_10"],
        longmemeval_r10=vfs_longmem["session_recall_at_10"],
        locomo_r10=vfs_locomo["session_recall_at_10"],
        technical_p1=tech_p1,
    )

    payload = {
        "seed": args.seed,
        "workers": args.workers,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "vfs_membench": vfs_membench,
        "vfs_locomo_full": vfs_locomo,
        "vfs_longmemeval_full": vfs_longmem,
        "vfs_exact_120": vfs_exact,
        "vfs_version_80": vfs_version,
        "vfs_retrieval_output_opt": vfs_opt,
        "vfs_membench_weighted_summary": vfs_weighted,
    }

    if not args.skip_v1_membench:
        t1 = time.perf_counter()
        v1_mb = run_v1_membench(args.memdata, args.seed, args.workers, args.membench_per_cat)
        payload["v1_membench_expanded"] = {
            "per_category": args.membench_per_cat,
            "total_items": len(v1_mb["items"]),
            "elapsed_s": round(time.perf_counter() - t1, 1),
            "default": v1_mb["default"],
            "consolidated": v1_mb["consolidated"],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
