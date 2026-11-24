"""C3 Ambient Zero-LLM Planner latency & precision stress benchmark."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from contextlib import closing
from pathlib import Path

from agentmem_core import CodeAnchor, UnifiedMemoryEngine
from eval.metrics import percentile
from eval.qa.provenance import build_provenance, render_provenance_markdown


class _ForbidLLM:
    def complete(self, *args, **kwargs):
        raise AssertionError("Ambient benchmark must execute 0 LLM calls")


def run_ambient_benchmark(*, num_facts: int = 1000, num_queries: int = 240, seed: int = 42) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "ambient_bench.db"
        engine = UnifiedMemoryEngine(db, llm=_ForbidLLM())
        with closing(engine):
            t_ingest0 = time.perf_counter()
            # 1. Populate background corpus (conversational facts + repo rules + code anchors)
            for i in range(num_facts):
                if i % 5 == 0:
                    mod = f"src/module_{i // 5:03d}.py"
                    engine.ingest_rule(
                        f"/rules/mod_{i // 5:03d}.md",
                        title=f"Rule for module_{i // 5:03d}",
                        trigger_condition=f"Editing {mod} or hitting ERR_MOD_{i // 5:03d}",
                        action=f"Use safe_handler_{i // 5:03d}(timeout=30) with v2.{i % 10}.0 API",
                        verification_result=f"Verified test_mod_{i // 5:03d} pass",
                        user_id="bench",
                        repo_id="core-repo",
                        scope_key="repo:core-repo",
                        anchors=[CodeAnchor(file_path=mod, anchor_type="symbol", symbol_name=f"safe_handler_{i // 5:03d}")],
                        timestamp=1_700_000_000.0 + i,
                    )
                else:
                    engine.ingest_turn(
                        f"Session note {i}: service alpha_{i} runs worker pool size {4 + (i % 8)} on node_{i % 20}.",
                        user_id="bench",
                        session_id=f"s_{i // 20}",
                        scope_key="user:bench",
                        timestamp=1_700_000_000.0 + i,
                    )
            ingest_sec = round(time.perf_counter() - t_ingest0, 3)

            # Mark 10 rules stale to measure stale exclusion rate
            stale_files = [f"src/module_{k:03d}.py" for k in range(10)]
            drift = engine.verify_anchors(changed_files=stale_files, user_id="bench")

            latencies_ms: list[float] = []
            llm_calls_total = 0
            anchor_hit_ok = 0
            anchor_total = 0
            stale_leaked = 0
            stale_total = 0
            false_injections = 0
            distractor_total = 0
            tokens_list: list[int] = []

            for q_idx in range(num_queries):
                bucket = q_idx % 4
                if bucket == 0:
                    # Fresh code-anchor + error hit
                    m_id = 10 + (q_idx % 80)
                    res = engine.ambient_context(
                        f"ERR_MOD_{m_id:03d} timeout",
                        active_files=[f"src/module_{m_id:03d}.py"],
                        active_symbols=[f"safe_handler_{m_id:03d}"],
                        user_id="bench",
                        repo_id="core-repo",
                        profile="coding",
                    )
                    anchor_total += 1
                    if f"safe_handler_{m_id:03d}" in res["dynamic_memory"]:
                        anchor_hit_ok += 1
                elif bucket == 1:
                    # Stale rule trap — MUST NOT inject stale rule!
                    m_id = q_idx % 10
                    res = engine.ambient_context(
                        f"ERR_MOD_{m_id:03d}",
                        active_files=[f"src/module_{m_id:03d}.py"],
                        active_symbols=[f"safe_handler_{m_id:03d}"],
                        user_id="bench",
                        repo_id="core-repo",
                        profile="coding",
                    )
                    stale_total += 1
                    if f"safe_handler_{m_id:03d}" in res["dynamic_memory"]:
                        stale_leaked += 1
                elif bucket == 2:
                    # Lexical conversational + identifier lookup
                    n_id = (q_idx * 7) % max(1, num_facts - 1)
                    if n_id % 5 == 0:
                        n_id += 1
                    res = engine.ambient_context(
                        f"alpha_{n_id} worker pool size",
                        user_id="bench",
                        profile="conversational",
                    )
                else:
                    # Completely unrelated distractor -> measures false-injection rate
                    res = engine.ambient_context(
                        f"quantum banana recipe kaleidoscope_unseen_{q_idx}",
                        active_files=[f"unrelated/file_{q_idx}.css"],
                        user_id="bench",
                        repo_id="core-repo",
                        profile="coding",
                    )
                    distractor_total += 1
                    if not res["empty_recall"]:
                        false_injections += 1

                latencies_ms.append(float(res["ambient_ms"]))
                llm_calls_total += int(res["llm_calls"])
                tokens_list.append(int(res["token_breakdown"]["total"]))

            prov = build_provenance(
                engine="UnifiedMemoryEngine (AmbientPlanner)",
                embedder="none (zero-vector hot path)",
                reranker="none (deterministic lexical + code-anchor)",
                llm="none (enforced max_calls=0)",
                judge="deterministic-assertion",
                protocol="C3 Ambient Stress Benchmark (G4 latency & zero-LLM gate)",
                granularity="fact + code-anchor",
                seed=seed,
                dataset=f"synthetic_ambient_corpus(facts={num_facts}, queries={num_queries})",
            )
            p50 = round(percentile(latencies_ms, 0.50), 3)
            p95 = round(percentile(latencies_ms, 0.95), 3)
            p99 = round(percentile(latencies_ms, 0.99), 3)
            avg_ms = round(sum(latencies_ms) / max(1, len(latencies_ms)), 3)

            return {
                "provenance": prov,
                "corpus": {"num_facts": num_facts, "num_queries": num_queries, "ingest_sec": ingest_sec, "stale_marked": drift["stale_count"]},
                "metrics": {
                    "llm_calls_total": llm_calls_total,
                    "avg_ms": avg_ms,
                    "p50_ms": p50,
                    "p95_ms": p95,
                    "p99_ms": p99,
                    "avg_tokens": round(sum(tokens_list) / max(1, len(tokens_list)), 1),
                    "fresh_anchor_recall": round(anchor_hit_ok / max(1, anchor_total), 4),
                    "stale_leak_rate": round(stale_leaked / max(1, stale_total), 4),
                    "false_injection_rate": round(false_injections / max(1, distractor_total), 4),
                },
                "gate_pass": (
                    llm_calls_total == 0
                    and p95 < 50.0
                    and stale_leaked == 0
                    and false_injections == 0
                    and anchor_hit_ok == anchor_total
                ),
            }


def render_markdown(report: dict) -> str:
    m = report["metrics"]
    c = report["corpus"]
    lines = [
        "# C3 Zero-LLM Ambient Planner Stress Benchmark Report",
        "",
        render_provenance_markdown(report["provenance"]),
        "",
        f"- **Gate Status**: `{'PASS' if report['gate_pass'] else 'FAIL'}`",
        f"- **Corpus**: `{c['num_facts']}` facts/rules/anchors (ingested in `{c['ingest_sec']}s`), `{c['stale_marked']}` marked stale, `{c['num_queries']}` ambient queries",
        "",
        "| Metric | Measured Value | Target / Invariant |",
        "| :--- | ---: | :--- |",
        f"| Total LLM Calls | **{m['llm_calls_total']}** | `== 0` (G4 hard invariant) |",
        f"| Latency Avg | {m['avg_ms']:.3f} ms | millisecond-class |",
        f"| Latency P50 | {m['p50_ms']:.3f} ms | millisecond-class |",
        f"| Latency **P95** | **{m['p95_ms']:.3f} ms** | `< 50 ms` (G4 ambient SLA) |",
        f"| Latency P99 | {m['p99_ms']:.3f} ms | millisecond-class |",
        f"| Avg Injected Tokens | {m['avg_tokens']:.1f} | `<= ambient_token_budget` |",
        f"| Fresh Anchor Recall | {m['fresh_anchor_recall'] * 100:.2f}% | `100%` |",
        f"| Stale Rule Leak Rate | {m['stale_leak_rate'] * 100:.2f}% | `0.00%` |",
        f"| False Injection Rate | {m['false_injection_rate'] * 100:.2f}% | `0.00%` |",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=int, default=1000)
    parser.add_argument("--queries", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("eval/results/c3_ambient_perf_report.json"))
    parser.add_argument("--md-out", type=Path, default=Path("eval/results/c3_ambient_perf_report.md"))
    args = parser.parse_args()

    rep = run_ambient_benchmark(num_facts=args.facts, num_queries=args.queries, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = render_markdown(rep)
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(md, encoding="utf-8")
    print(md)
    return 0 if rep["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
