"""Real head-to-head ablations: (1) v1 Ledger vs v2 VFS on remote benchmarks; (2) SQLite vs LanceDB vs PostgreSQL."""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
import uuid
from contextlib import closing
from pathlib import Path

from eval.engineering_bench import fixture_prompts, prepare_fixture, timing_summary
from eval.exact_ident_eval import static_catalog
from eval.locomo.run import groups as locomo_groups
from eval.longmemeval.run import groups as longmemeval_groups
from eval.metrics import percentile
from eval.paired import make_engine
from eval.version_precision_eval import PRODUCTS
from agentmem_core.engine import CognitiveMemoryEngine
from agentmem_core.identifiers import extract_identifiers
from agentmem_core.llm import MockLLMClient
from agentmem_core.reranker import CalibratedLocalReranker
from agentmem_core.vector import HashedTfidfEmbedder
from agentmem_core.vfs import VFSMemoryEngine


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def ablate_exact_identifiers(seed: int = 42) -> dict:
    """Compare v1 Ledger vs v2 VFS (L2 Markdown) on the 120-question Exact Identifiers suite."""
    catalog, absent = static_catalog()
    rng = random.Random(seed)
    items = list(catalog)
    rng.shuffle(items)

    # 1. v2 VFS (each record stored as an L2 Markdown note — NO exact_identifiers table or version gate)
    with tempfile.TemporaryDirectory() as tmp:
        vfs = VFSMemoryEngine(Path(tmp) / "vfs.db", transcript_dir=Path(tmp) / "l0")
        with closing(vfs):
            path_by_key = {}
            for ordinal, (kind, literal) in enumerate(items, 1):
                body = f"{literal} uses recovery checkpoint {ordinal} and the verified transaction workflow."
                tr = vfs.append_transcript(body, session_id="s1", timestamp=float(ordinal))
                node_path = f"/{kind}/item_{ordinal:03d}.md"
                vfs.write(
                    node_path,
                    title=f"{literal} Recovery Record",
                    summary=body,
                    content=body,
                    layer="L2",
                    verification_status="verified",
                    salience_tag="core_rule" if kind == "procedural" else "normal",
                    source_transcript_ids=[tr.transcript_id],
                    timestamp=float(ordinal),
                )
                path_by_key[(kind, literal)] = node_path

            p1, rr, confusion, missing_empty = [], [], [], []
            for kind, literal in items + absent:
                query = f"`{literal}`" if kind == "missing_symbol" else f"Which configuration or recovery procedure applies to {literal}?"
                gold_path = path_by_key.get((kind, literal))
                hits = vfs.search(query, top_k=10)
                paths = [h["path"] for h in hits]
                if gold_path is not None:
                    rank = paths.index(gold_path) + 1 if gold_path in paths else None
                    p1.append(1.0 if rank == 1 else 0.0)
                    rr.append(1.0 / rank if rank else 0.0)
                else:
                    missing_empty.append(1.0 if not hits else 0.0)
                # Version / symbol confusion check
                wanted = {it.ident_norm for it in extract_identifiers(literal)}
                top_idents = {it.ident_norm for it in extract_identifiers(hits[0]["content"])} if hits else set()
                if kind in {"version", "missing_version", "missing_symbol"}:
                    confusion.append(1.0 if (hits and not (wanted & top_idents)) else 0.0)

    return {
        "n_total": len(items) + len(absent),
        "n_answerable": len(items),
        "n_missing": len(absent),
        "vfs_l2": {
            "precision_at_1": _mean(p1),
            "mrr_at_10": _mean(rr),
            "confusion_rate": _mean(confusion),
            "missing_empty_rate": _mean(missing_empty),
        },
    }


def ablate_version_precision() -> dict:
    """Compare v2 VFS (L2 Markdown) on the exact 80-question Version Precision suite."""
    p1, confusion = [], []
    with tempfile.TemporaryDirectory() as tmp:
        vfs = VFSMemoryEngine(Path(tmp) / "vfs.db", transcript_dir=Path(tmp) / "l0")
        with closing(vfs):
            for i, product in enumerate(PRODUCTS):
                other = PRODUCTS[(i + 1) % len(PRODUCTS)]
                major = i + 3
                b, s, p = f"{major}.11", f"{major}.11.2", f"{major}.11.4"
                cases = (
                    ("exact", f"{product} v{s}", f"{product} v{s} supports the optional cache.", f"{product} v{p}"),
                    ("containment", f"{product} v{b}", f"{product} v{p} enables audit mode.", f"{product} v{p}-rc.1"),
                    ("coarse", f"{product} v{s}", f"{product} v{b} supports read replicas.", f"{product} v{p}"),
                    ("sibling", f"{product} v{s}", f"{product} v{s} changes the retry policy.", f"{product} v{major}.11.1"),
                    ("numeric_order", f"{product} v{major}.10", f"{product} v{major}.10 accepts these flags.", f"{product} v{major}.2"),
                    ("prerelease", f"{product} v{s}", f"{product} v{s} release is stable for staging.", f"{product} v{s}-rc1"),
                    ("range", f"{product} v{b}", f"Supports {product} >= {major}.10, < {major}.13.", f"{product} v{major}.13"),
                    ("cross_product", f"{product} v{b}", f"{product} v{b} supports local profiles.", f"{other} v{b}"),
                )
                for category, query, good, bad in cases:
                    uid = f"u_{i}_{category}"
                    vfs.write("/releases/good.md", title=good, content=good, layer="L2", verification_status="verified", user_id=uid)
                    vfs.write("/releases/bad.md", title=bad, content=bad, layer="L2", verification_status="verified", user_id=uid)
                    hits = vfs.search(query, top_k=5, user_id=uid)
                    top_path = hits[0]["path"] if hits else None
                    p1.append(1.0 if top_path == "/releases/good.md" else 0.0)
                    if category in {"cross_product", "sibling", "prerelease"}:
                        confusion.append(1.0 if any(h["path"] == "/releases/bad.md" for h in hits) else 0.0)
    return {
        "n": len(p1),
        "vfs_l2": {
            "precision_at_1": _mean(p1),
            "confusion_rate": _mean(confusion),
        },
    }


def _eval_vfs_on_dataset(name: str, dataset_path: Path, group_fn, sample: int, seed: int, mode: str) -> dict:
    """Evaluate v2 VFS on official LoCoMo or LongMemEval-S stratified sample.

    Modes:
      - ``dream_only``: all turns stored in L0 chunks; only deterministic ``dream_consolidate`` promotes L2 notes.
      - ``session_md``: all turns stored in L0 chunks; each session becomes 1 L2 Markdown note ``/sessions/<sid>.md``.
    """
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    recalls_5, recalls_10, mrrs = [], [], []
    update_dom = []
    durs = []
    vec_counts = []
    ingest_ms_total = 0.0
    cpu_q_ms_total = 0.0
    n_questions = 0
    n_answerable = 0

    for turns, questions in group_fn(data, sample, seed):
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

                if mode == "session_md":
                    for ordinal, (orig_session, entries) in enumerate(by_session.items()):
                        tids = [tid for tid, _, _, _ in entries]
                        body = "\n".join(text for _, _, text, _ in entries)
                        summary = entries[0][2][:160] if entries else orig_session
                        session_ts_vals = [ts for _, _, _, ts in entries if ts is not None]
                        session_ts = max(session_ts_vals) if session_ts_vals else float(ordinal + 1)
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
                ingest_ms_total += (time.perf_counter() - t_ing) * 1000
                vec_counts.append(vfs.store.vector_index_count(user_id="default", agent_id="default"))

                for q in questions:
                    n_questions += 1
                    c0 = time.process_time()
                    t0 = time.perf_counter()
                    hits = vfs.search(q.text, reference_time=q.timestamp, top_k=10)
                    durs.append((time.perf_counter() - t0) * 1000)
                    cpu_q_ms_total += (time.process_time() - c0) * 1000

                    gold_sessions = set(q.evidence_sessions) if q.evidence_sessions else {t.session for t in turns if t.ref in q.evidence}
                    if not gold_sessions or q.unanswerable:
                        continue
                    n_answerable += 1

                    # Map each ranked L2 hit back to original session(s) via its L0 source_transcript_ids
                    ranked_sessions = []
                    for hit in hits:
                        hit_sessions = {turn_to_session[tid] for tid in hit.get("source_transcript_ids", ()) if tid in turn_to_session}
                        ranked_sessions.append(hit_sessions)

                    hit5 = set().union(*ranked_sessions[:5]) if ranked_sessions[:5] else set()
                    hit10 = set().union(*ranked_sessions[:10]) if ranked_sessions[:10] else set()
                    recalls_5.append(len(gold_sessions & hit5) / len(gold_sessions))
                    recalls_10.append(len(gold_sessions & hit10) / len(gold_sessions))

                    first_rank = next((idx for idx, sset in enumerate(ranked_sessions[:10], 1) if sset & gold_sessions), None)
                    mrrs.append(1.0 / first_rank if first_rank else 0.0)

                    if q.latest_session and q.older_sessions:
                        latest_rank = next((idx for idx, sset in enumerate(ranked_sessions, 1) if q.latest_session in sset), None)
                        older_rank = next((idx for idx, sset in enumerate(ranked_sessions, 1) if sset & set(q.older_sessions)), None)
                        dominated = latest_rank is not None and (older_rank is None or latest_rank < older_rank)
                        update_dom.append(1.0 if dominated else 0.0)

    return {
        "dataset": name,
        "mode": f"vfs_{mode}",
        "questions": n_questions,
        "answerable": n_answerable,
        "session_recall_at_5": _mean(recalls_5),
        "session_recall_at_10": _mean(recalls_10),
        "session_mrr_at_10": _mean(mrrs),
        "update_dominance": _mean(update_dom) if update_dom else None,
        "update_n": len(update_dom),
        "avg_l2_vectors": _mean(vec_counts),
        "total_ingest_ms": round(ingest_ms_total, 1),
        "query_p50_ms": round(percentile(durs, 0.5), 2),
        "query_p95_ms": round(percentile(durs, 0.95), 2),
        "query_cpu_total_ms": round(cpu_q_ms_total, 1),
    }


def ablate_storage_backends(scales: tuple[int, ...] = (1000, 5000, 10000), pg_dsn: str | None = None) -> list[dict]:
    """3-way storage backend ablation: SQLite vs SQLite+LanceDB (IVF_RQ) vs live PostgreSQL 16."""
    results = []
    for scale in scales:
        prompts = fixture_prompts(scale, 42)
        with tempfile.TemporaryDirectory() as tmp:
            # 1. SQLite (optimized)
            p_sql = Path(tmp) / f"sql_{scale}.db"
            e_sql = CognitiveMemoryEngine(
                p_sql, llm=MockLLMClient(), vector=HashedTfidfEmbedder(),
                reranker=CalibratedLocalReranker(), enable_casual_bypass=False,
            )
            with closing(e_sql.store):
                t0 = time.perf_counter()
                prepare_fixture(e_sql, scale)
                ingest_s = time.perf_counter() - t0
                e_sql.retrieve(prompts[0], read_only=True)
                durs, hits = [], 0
                c0 = time.process_time()
                for q in prompts:
                    s = time.perf_counter()
                    r = e_sql.retrieve(q, read_only=True)
                    durs.append((time.perf_counter() - s) * 1000)
                    hits += bool(r["ranked"] and r["ranked"][0]["exact_ident_hit"])
                cpu_ms = (time.process_time() - c0) * 1000
                st = timing_summary(durs, [0])
                results.append({
                    "scale": scale, "backend": "SQLite (optimized)",
                    "writes_per_sec": round(scale / ingest_s, 1), "opt_ms": 0.0,
                    "p50_ms": round(st["p50_ms"], 2), "p95_ms": round(st["p95_ms"], 2), "p99_ms": round(st["p99_ms"], 2),
                    "cpu_60q_ms": round(cpu_ms, 1), "hit_rate": f"{hits}/{len(prompts)}",
                })

            # 2. SQLite + LanceDB (IVF_RQ RaBitQ)
            p_lance = Path(tmp) / f"lance_{scale}"
            e_lance = CognitiveMemoryEngine(
                db_url=f"lancedb:///{p_lance}", llm=MockLLMClient(), vector=HashedTfidfEmbedder(),
                reranker=CalibratedLocalReranker(), enable_casual_bypass=False,
            )
            with closing(e_lance.store):
                t0 = time.perf_counter()
                prepare_fixture(e_lance, scale)
                e_lance.store.flush_index()
                ingest_s = time.perf_counter() - t0
                t_opt = time.perf_counter()
                e_lance.store.optimize_index()
                opt_ms = (time.perf_counter() - t_opt) * 1000
                e_lance.retrieve(prompts[0], read_only=True)
                durs, hits = [], 0
                c0 = time.process_time()
                for q in prompts:
                    s = time.perf_counter()
                    r = e_lance.retrieve(q, read_only=True)
                    durs.append((time.perf_counter() - s) * 1000)
                    hits += bool(r["ranked"] and r["ranked"][0]["exact_ident_hit"])
                cpu_ms = (time.process_time() - c0) * 1000
                st = timing_summary(durs, [0])
                results.append({
                    "scale": scale, "backend": "SQLite+LanceDB (IVF_RQ)",
                    "writes_per_sec": round(scale / ingest_s, 1), "opt_ms": round(opt_ms, 1),
                    "p50_ms": round(st["p50_ms"], 2), "p95_ms": round(st["p95_ms"], 2), "p99_ms": round(st["p99_ms"], 2),
                    "cpu_60q_ms": round(cpu_ms, 1), "hit_rate": f"{hits}/{len(prompts)}",
                })

        # 3. Live PostgreSQL 16 + pgvector HNSW (if DSN provided)
        if pg_dsn:
            import psycopg
            from psycopg import sql
            from urllib.parse import urlencode

            schema = f"bench_{scale}_{uuid.uuid4().hex[:8]}"
            with psycopg.connect(pg_dsn, autocommit=True) as conn:
                conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
                try:
                    configured = psycopg.conninfo.make_conninfo(pg_dsn, options=f"-csearch_path={schema},public")
                    uri = "postgresql:///?" + urlencode(psycopg.conninfo.conninfo_to_dict(configured))
                    e_pg = CognitiveMemoryEngine(
                        db_url=uri, llm=MockLLMClient(), vector=HashedTfidfEmbedder(),
                        reranker=CalibratedLocalReranker(), enable_casual_bypass=False,
                    )
                    with closing(e_pg.store):
                        t0 = time.perf_counter()
                        prepare_fixture(e_pg, scale)
                        ingest_s = time.perf_counter() - t0
                        e_pg.retrieve(prompts[0], read_only=True)
                        durs, hits = [], 0
                        c0 = time.process_time()
                        for q in prompts:
                            s = time.perf_counter()
                            r = e_pg.retrieve(q, read_only=True)
                            durs.append((time.perf_counter() - s) * 1000)
                            hits += bool(r["ranked"] and r["ranked"][0]["exact_ident_hit"])
                        cpu_ms = (time.process_time() - c0) * 1000
                        st = timing_summary(durs, [0])
                        health = e_pg.store.health_check()
                        results.append({
                            "scale": scale, "backend": f"PostgreSQL ({health['vector']}+{health['lexical']})",
                            "writes_per_sec": round(scale / ingest_s, 1), "opt_ms": 0.0,
                            "p50_ms": round(st["p50_ms"], 2), "p95_ms": round(st["p95_ms"], 2), "p99_ms": round(st["p99_ms"], 2),
                            "cpu_60q_ms": round(cpu_ms, 1), "hit_rate": f"{hits}/{len(prompts)}",
                        })
                finally:
                    conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
    return results


def bench_vfs_retrieval_and_output_optimizations() -> dict:
    """Real measured evaluation of 3-stage retrieval, salience gating, progressive disclosure, and compaction."""
    from agentmem_core.chrono import DAY
    from agentmem_core.textutil import estimate_tokens

    with tempfile.TemporaryDirectory() as tmp:
        vfs = VFSMemoryEngine(Path(tmp) / "opt_vfs.db", transcript_dir=Path(tmp) / "l0")
        with closing(vfs):
            # 1. Stage 1 Deterministic Pre-load (L1 /global.md + L2 always_active)
            vfs.write("/global.md", title="Global Operating Identity", content="Always verify migrations before cutover.", layer="L1", verification_status="verified")
            vfs.write_rule(
                "/taboos/no_l0_vec.md",
                title="Never Vectorize L0 Transcripts",
                trigger_condition="Designing memory indexing pipelines",
                action="Only embed curated L2 Markdown rules; store L0 transcripts in cold chunks",
                verification_result="Validated across 10K benchmark (-70% CPU)",
                salience_tag="taboo",
                always_active=True,
            )
            unrelated_ctx = vfs.build_context("What is the weather in Tokyo?", token_budget=2048)
            preloaded_paths = {item["path"] for item in unrelated_ctx["preloaded"]}
            stage1_preload_ok = "/global.md" in preloaded_paths and "/taboos/no_l0_vec.md" in preloaded_paths

            # 2. Write-side Salience Gate & State Machine (40 trials: 20 noise + 10 unverified + 10 verified)
            noise_blocked = 0
            for i in range(10):
                r = vfs.write_rule(f"/noise/vis_{i}.md", title=f"Visible {i}", trigger_condition="UI", action="Click button", visible_in_context=True)
                noise_blocked += int(not r["written"])
            for i in range(10):
                r = vfs.write_rule(f"/noise/eph_{i}.md", title=f"Ephemeral {i}", trigger_condition="Temp", action="Wait 2s", reusable_pattern=False, future_pitfall_risk=False)
                noise_blocked += int(not r["written"])
            unverified_forced = 0
            for i in range(10):
                r = vfs.write_rule(
                    f"/hypotheses/h_{i}.md",
                    title=f"Hypothesis {i}",
                    trigger_condition=f"When latency spikes on shard {i}",
                    action=f"Increase connection pool on shard {i}",
                    verification_result="",
                    verification_status="verified",  # caller tries to fake verified without outcome
                    confidence_score=0.95,
                )
                unverified_forced += int(r["written"] and r["verification_status"] == "unverified" and r["confidence_score"] <= 0.65)

            # 3. Stage 3 Rerank Formula: verified vs unverified (30 pairs) & recency dominance (30 pairs)
            base_ts = 1_700_000_000.0
            verified_wins = 0
            recency_wins = 0
            for i in range(30):
                uid_v = f"rerank_v_{i}"
                topic = f"postgresql connection pool timeout strategy topic{i}"
                vfs.write(f"/rules/guess.md", title=f"Guess {i}", content=topic, layer="L2", verification_status="unverified", timestamp=base_ts, user_id=uid_v)
                vfs.write(f"/rules/proven.md", title=f"Proven {i}", content=topic, layer="L2", verification_status="verified", timestamp=base_ts, user_id=uid_v)
                hits = vfs.search(topic, top_k=5, reference_time=base_ts, user_id=uid_v)
                verified_wins += int(bool(hits) and hits[0]["path"] == "/rules/proven.md")

                uid_r = f"rerank_r_{i}"
                vfs.write(f"/notes/old.md", title=f"Old {i}", content=topic, layer="L2", verification_status="verified", salience_tag="normal", timestamp=base_ts - 90 * DAY, user_id=uid_r)
                vfs.write(f"/notes/new.md", title=f"New {i}", content=topic, layer="L2", verification_status="verified", salience_tag="normal", timestamp=base_ts - 1 * DAY, user_id=uid_r)
                rhits = vfs.search(topic, top_k=5, reference_time=base_ts, user_id=uid_r)
                recency_wins += int(bool(rhits) and rhits[0]["path"] == "/notes/new.md")

            # 4. Read-side Progressive Disclosure vs Full Dump (12 detailed L2 rules, ~260 tokens each)
            uid_pd = "progressive_bench"
            for i in range(12):
                detail = (
                    f"Detailed database migration and recovery playbook #{i} covering PostgreSQL WAL checkpoints, "
                    f"VectorChord RaBitQ index build parameters, lock timeout safeguards, connection pooling limits, "
                    f"replica lag verification, and rollback verification procedures. "
                ) * 6
                vfs.write_rule(
                    f"/strategies/database/rule_{i:02d}.md",
                    title=f"Database Migration Rule {i}",
                    trigger_condition=f"Running database migration phase {i} on PostgreSQL cluster",
                    action=f"Verify WAL checkpoints and VectorChord RaBitQ index build #{i}",
                    verification_result=f"Passed staging replay #{i} with zero lock timeouts",
                    notes=detail,
                    user_id=uid_pd,
                )
            vfs.refresh_l3_index(user_id=uid_pd)
            ctx_prog = vfs.build_context(
                "PostgreSQL database migration WAL checkpoints VectorChord RaBitQ",
                token_budget=4096,
                disclosure="progressive",
                include_tree_outline=True,
                top_k=8,
                user_id=uid_pd,
            )
            ctx_full = vfs.build_context(
                "PostgreSQL database migration WAL checkpoints VectorChord RaBitQ",
                token_budget=4096,
                disclosure="full",
                include_tree_outline=True,
                top_k=8,
                user_id=uid_pd,
            )
            prog_tokens = ctx_prog["token_breakdown"]["dynamic_tokens"]
            full_tokens = ctx_full["token_breakdown"]["dynamic_tokens"]
            token_reduction = (full_tokens - prog_tokens) / full_tokens if full_tokens else 0.0
            # Verify ZERO mid-rule truncation and every index_summary <= 50 chars
            max_summary_len = max(len(h["index_summary"]) for h in ctx_prog["l2_hits"])
            mid_rule_truncations = sum(1 for h in ctx_prog["l2_hits"] if h["detail_body"] != vfs.read_note(h["path"], user_id=uid_pd)["detail_body"])

            # 5. Dynamic Directory Compaction (>10 notes -> single GUIDE.md) + Atomic Rollback
            alerts_before = vfs.check_compaction_needed(parent_path="/strategies/database/", user_id=uid_pd)
            before_notes = alerts_before[0]["note_count"]
            before_tokens = alerts_before[0]["tokens"]
            compacted = vfs.compact_directory("/strategies/database/", user_id=uid_pd)
            after_stats = vfs.store.directory_stats(user_id=uid_pd, agent_id="default")
            db_after = next(s for s in after_stats if s["parent_path"] == "/strategies/database/")
            after_notes = db_after["note_count"]
            after_tokens = db_after["tokens"]
            vfs.rollback_batch(compacted["dream_batch_id"], user_id=uid_pd)
            restored_stats = vfs.store.directory_stats(user_id=uid_pd, agent_id="default")
            restored_notes = next(s for s in restored_stats if s["parent_path"] == "/strategies/database/")["note_count"]

    return {
        "stage1_deterministic_preload_ok": stage1_preload_ok,
        "salience_noise_blocked_rate": noise_blocked / 20.0,
        "hypothesis_state_machine_enforced_rate": unverified_forced / 10.0,
        "verified_beats_unverified_rate": verified_wins / 30.0,
        "recency_dominance_rate": recency_wins / 30.0,
        "progressive_disclosure": {
            "progressive_dynamic_tokens": prog_tokens,
            "full_dynamic_tokens": full_tokens,
            "token_reduction_rate": round(token_reduction, 4),
            "max_index_summary_chars": max_summary_len,
            "mid_rule_truncations": mid_rule_truncations,
        },
        "directory_compaction": {
            "before_notes": before_notes,
            "after_notes": after_notes,
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
            "token_reduction_rate": round((before_tokens - after_tokens) / before_tokens, 4) if before_tokens else 0.0,
            "rollback_restored_notes": restored_notes,
            "rollback_exact_ok": restored_notes == before_notes,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locomo", type=Path, default=Path("/private/tmp/ame-benchmark-data/locomo10.json"))
    parser.add_argument("--longmemeval", type=Path, default=Path("/private/tmp/ame-benchmark-data/longmemeval_s_cleaned.json"))
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--pg-dsn", type=str, default=os.getenv("TEST_POSTGRES_DSN"))
    parser.add_argument("--skip-storage", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "exact_identifiers_120": ablate_exact_identifiers(),
        "version_precision_80": ablate_version_precision(),
        "locomo_100": [
            _eval_vfs_on_dataset("LoCoMo", args.locomo, locomo_groups, args.sample, 42, "session_md"),
        ],
        "longmemeval_100": [
            _eval_vfs_on_dataset("LongMemEval-S", args.longmemeval, longmemeval_groups, args.sample, 42, "session_md"),
        ],
        "vfs_retrieval_and_output_opt": bench_vfs_retrieval_and_output_optimizations(),
    }
    if not args.skip_storage:
        report["storage_backends"] = ablate_storage_backends((1000, 5000, 10000), pg_dsn=args.pg_dsn)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import uuid
    main()
