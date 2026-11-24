# C3 Zero-LLM Ambient Planner Stress Benchmark Report

| Provenance Field | Value |
| :--- | :--- |
| `engine` | `UnifiedMemoryEngine (AmbientPlanner)` |
| `embedder` | `none (zero-vector hot path)` |
| `reranker` | `none (deterministic lexical + code-anchor)` |
| `llm` | `none (enforced max_calls=0)` |
| `judge` | `deterministic-assertion` |
| `protocol` | `C3 Ambient Stress Benchmark (G4 latency & zero-LLM gate)` |
| `granularity` | `fact + code-anchor` |
| `commit_hash` | `396a87e-dirty` |
| `seed` | `42` |
| `dataset` | `synthetic_ambient_corpus(facts=1000, queries=240)` |
| `dataset_sha256` | `synthetic/in-memory` |
| `timestamp_utc` | `2026-11-24T06:06:44Z` |

- **Gate Status**: `PASS`
- **Corpus**: `1000` facts/rules/anchors (ingested in `6.39s`), `10` marked stale, `240` ambient queries

| Metric | Measured Value | Target / Invariant |
| :--- | ---: | :--- |
| Total LLM Calls | **0** | `== 0` (G4 hard invariant) |
| Latency Avg | 3.203 ms | millisecond-class |
| Latency P50 | 2.870 ms | millisecond-class |
| Latency **P95** | **4.892 ms** | `< 50 ms` (G4 ambient SLA) |
| Latency P99 | 5.417 ms | millisecond-class |
| Avg Injected Tokens | 182.0 | `<= ambient_token_budget` |
| Fresh Anchor Recall | 100.00% | `100%` |
| Stale Rule Leak Rate | 0.00% | `0.00%` |
| False Injection Rate | 0.00% | `0.00%` |
