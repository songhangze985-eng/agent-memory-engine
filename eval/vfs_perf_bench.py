"""Real performance & scalability benchmark for v2 VFSMemoryEngine across SQLite, LanceDB (IVF_RQ), and PostgreSQL."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

from eval.metrics import percentile
from memengine.vfs import OptimisticLockError, VFSMemoryEngine


def _run_l0_stress(n_turns: int = 10_000) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        vfs = VFSMemoryEngine(Path(tmp) / "l0_stress.db", transcript_dir=Path(tmp) / "l0")
        with closing(vfs):
            t0 = time.perf_counter()
            c0 = time.process_time()
            for i in range(n_turns):
                vfs.append_transcript(
                    f"Turn {i}: investigating worker pool shard_{i % 64} and checking staging deployment logs.",
                    session_id=f"sess_{i // 200}",
                    timestamp=1_700_000_000.0 + i,
                )
            vfs.transcripts.flush()
            wall_s = time.perf_counter() - t0
            cpu_ms = (time.process_time() - c0) * 1000
            stats = vfs.transcripts.stats(user_id="default", agent_id="default")
            vec_count = vfs.store.vector_index_count(user_id="default", agent_id="default")
    return {
        "l0_turns": n_turns,
        "turns_per_sec": round(n_turns / wall_s, 1),
        "wall_ms": round(wall_s * 1000, 1),
        "cpu_ms": round(cpu_ms, 1),
        "chunk_files": stats["chunk_files"],
        "compressed_kb": round(stats["compressed_bytes"] / 1024.0, 1),
        "l2_vectors_created": vec_count,
    }


def _populate_and_bench_vfs(
    vfs: VFSMemoryEngine,
    *,
    label: str,
    n_notes: int,
    n_queries: int = 60,
    optimize: bool = False,
) -> dict:
    base_ts = 1_700_000_000.0
    vfs.write(
        "/global.md",
        title="Global Ops Policy",
        content="Always run canary checks and verify backups before schema migration.",
        layer="L1",
        verification_status="verified",
        timestamp=base_ts,
    )
    vfs.write_rule(
        "/taboos/no_blind_cutover.md",
        title="No Blind Cutover",
        trigger_condition="Executing production database cutover",
        action="Validate checksums and keep rollback snapshot ready",
        verification_result="Zero data loss across 12 releases",
        salience_tag="taboo",
        always_active=True,
        timestamp=base_ts,
    )

    domains = ("database", "networking", "caching", "storage", "auth", "queue", "search", "deploy")
    t0 = time.perf_counter()
    for i in range(n_notes):
        dom = domains[i % len(domains)]
        ver = f"v{3 + (i % 5)}.{i % 20}.{i % 10}"
        code = f"ERR_{dom.upper()}_{i:04d}"
        vfs.write_rule(
            f"/strategies/{dom}/rule_{i:05d}.md",
            title=f"{dom.title()} Rule {i} ({code})",
            trigger_condition=f"When service_{dom}_{i % 32} emits {code} on engine {ver}",
            action=f"Apply mitigation profile_{i:05d} for {dom} subsystem and drain shard_{i % 64}",
            verification_result=f"Verified on staging cluster #{i % 16} with P95 < 25ms",
            notes=f"Operational context for {code} on {ver}: check connection limits, WAL checkpoints, and replica lag.",
            salience_tag="core_rule" if i % 10 == 0 else "normal",
            timestamp=base_ts + i * 60.0,
        )
    ingest_s = time.perf_counter() - t0
    opt_ms = 0.0
    opt_info = {"indexed": False, "index_type": "native"}
    if optimize:
        t_opt = time.perf_counter()
        opt_info = vfs.optimize_index()
        opt_ms = (time.perf_counter() - t_opt) * 1000
    vfs.refresh_l3_index(timestamp=base_ts + n_notes * 60.0)

    # Build 60 deterministic mixed queries targeting exact rules across the full range
    step = max(1, n_notes // n_queries)
    targets = [min(n_notes - 1, idx * step) for idx in range(n_queries)]
    queries = [
        (
            f"/strategies/{domains[t % len(domains)]}/rule_{t:05d}.md",
            f"How should we mitigate ERR_{domains[t % len(domains)].upper()}_{t:04d} on {domains[t % len(domains)]} subsystem?",
        )
        for t in targets
    ]

    # Warmup
    vfs.search(queries[0][1], top_k=8)

    search_durs = []
    top1_hits = 0
    c0 = time.process_time()
    for gold_path, q in queries:
        s = time.perf_counter()
        hits = vfs.search(q, top_k=8)
        search_durs.append((time.perf_counter() - s) * 1000)
        if hits and hits[0]["path"] == gold_path:
            top1_hits += 1
    search_cpu_ms = (time.process_time() - c0) * 1000

    # Progressive vs Full build_context benchmark (20 queries)
    ctx_prog_durs, ctx_prog_toks, ctx_full_toks = [], [], []
    for _, q in queries[:20]:
        s = time.perf_counter()
        cp = vfs.build_context(q, token_budget=2048, disclosure="progressive", include_tree_outline=False, top_k=8)
        ctx_prog_durs.append((time.perf_counter() - s) * 1000)
        ctx_prog_toks.append(cp["token_breakdown"]["dynamic_tokens"])
        cf = vfs.build_context(q, token_budget=2048, disclosure="full", include_tree_outline=False, top_k=8)
        ctx_full_toks.append(cf["token_breakdown"]["dynamic_tokens"])

    # 8-thread concurrent optimistic lock stress check
    vfs.write("/shared/bench_lock.md", title="Lock Target", content="initial", expected_version=0)

    def _contend(worker: int) -> str:
        try:
            vfs.edit("/shared/bench_lock.md", old_str="initial", new_str=f"win_{worker}", expected_version=1)
            return "ok"
        except OptimisticLockError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(_contend, range(8)))

    health = vfs.store.health_check()
    return {
        "scale_l2_notes": n_notes,
        "backend": label,
        "vector_engine": health["vector"],
        "lexical_engine": health["lexical"],
        "index_info": opt_info,
        "l2_writes_per_sec": round(n_notes / ingest_s, 1),
        "opt_ms": round(opt_ms, 1),
        "search_p50_ms": round(percentile(search_durs, 0.5), 2),
        "search_p95_ms": round(percentile(search_durs, 0.95), 2),
        "search_p99_ms": round(percentile(search_durs, 0.99), 2),
        "search_cpu_60q_ms": round(search_cpu_ms, 1),
        "top1_accuracy": f"{top1_hits}/{len(queries)}",
        "context_prog_p95_ms": round(percentile(ctx_prog_durs, 0.95), 2),
        "avg_prog_tokens": round(sum(ctx_prog_toks) / len(ctx_prog_toks), 1),
        "avg_full_tokens": round(sum(ctx_full_toks) / len(ctx_full_toks), 1),
        "concurrency_8threads_ok": outcomes.count("ok") == 1 and outcomes.count("conflict") == 7,
    }


def run_vfs_perf_suite(scales: tuple[int, ...] = (500, 2000, 5000), pg_dsn: str | None = None) -> dict:
    rows = []
    for scale in scales:
        with tempfile.TemporaryDirectory() as tmp:
            # 1. SQLite VFS
            vfs_sql = VFSMemoryEngine(Path(tmp) / f"vfs_{scale}.db", transcript_dir=Path(tmp) / f"l0_sql_{scale}")
            with closing(vfs_sql):
                rows.append(_populate_and_bench_vfs(vfs_sql, label="SQLite VFS", n_notes=scale, optimize=False))

            # 2. SQLite + LanceDB VFS (IVF_RQ RaBitQ)
            lance_dir = Path(tmp) / f"vfs_lance_{scale}"
            vfs_lance = VFSMemoryEngine(f"lancedb:///{lance_dir}")
            with closing(vfs_lance):
                rows.append(_populate_and_bench_vfs(vfs_lance, label="SQLite+LanceDB VFS (IVF_RQ)", n_notes=scale, optimize=True))

        # 3. Live PostgreSQL 16 VFS (if DSN provided)
        if pg_dsn:
            import psycopg
            from psycopg import sql
            from urllib.parse import urlencode

            schema = f"vfs_perf_{scale}_{uuid.uuid4().hex[:8]}"
            with psycopg.connect(pg_dsn, autocommit=True) as conn:
                conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                try:
                    configured = psycopg.conninfo.make_conninfo(pg_dsn, options=f"-csearch_path={schema},public")
                    uri = "postgresql:///?" + urlencode(psycopg.conninfo.conninfo_to_dict(configured))
                    with tempfile.TemporaryDirectory() as tmp_pg:
                        vfs_pg = VFSMemoryEngine(uri, transcript_dir=Path(tmp_pg) / "pg_l0")
                        with closing(vfs_pg):
                            rows.append(_populate_and_bench_vfs(vfs_pg, label="PostgreSQL 16 VFS", n_notes=scale, optimize=False))
                finally:
                    conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))

    return {
        "l0_cold_ingest_10k": _run_l0_stress(10_000),
        "vfs_scale_matrix": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", type=int, nargs="+", default=[500, 2000, 5000])
    parser.add_argument("--pg-dsn", type=str, default=os.getenv("TEST_POSTGRES_DSN"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run_vfs_perf_suite(tuple(args.scales), pg_dsn=args.pg_dsn)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
