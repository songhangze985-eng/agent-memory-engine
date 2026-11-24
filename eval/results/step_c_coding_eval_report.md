# Step C Coding Profile Experience-Reuse & Hygiene Evaluation (G2)

| Provenance Field | Value |
| :--- | :--- |
| `engine` | `UnifiedMemoryEngine (coding profile)` |
| `extractor_model` | `C2 SignalExtractor (fail_to_pass)` |
| `reader_model` | `DeterministicPairedCodingSimulator` |
| `judge` | `deterministic-fix-token-match` |
| `judge_prompt_version` | `hindsight-aligned-v1` |
| `embedder` | `none (C3 zero-vector ambient path)` |
| `reranker` | `none (code-anchor + exact + FTS5)` |
| `top_k` | `10` |
| `token_budget` | `2048` |
| `split` | `full` |
| `protocol` | `Step C Coding Experience-Reuse + Staleness + False-Injection Eval (G2)` |
| `protocol_version` | `1.0.0` |
| `protocol_sha256` | `e232828db8cf2bc690f5f32ed4bd8a6b87b2a96ae56b5e4ae479f686f6ba5db4` |
| `granularity` | `procedural-rule + code-anchor` |
| `commit_hash` | `0aa3ec8-dirty` |
| `seed` | `42` |
| `dataset` | `coding_reuse_pairs(n=30, stale_traps=10, distractors=10)` |
| `dataset_sha256` | `synthetic/in-memory` |
| `timestamp_utc` | `2026-11-24T06:22:26Z` |

- **G2 Gate Status**: `PASS`

## Headline Metrics (30 Experience-Reuse Pairs + 10 Stale Traps + 10 Distractors)

| Metric | No Memory | With Memory (`coding` profile) | Delta / Target |
| :--- | ---: | ---: | :--- |
| Task B Success Rate | 60.00% | **100.00%** | `+40.00 pp` |
| Avg Steps / Task | 4.40 | **1.00** | `-77.3%` |
| Avg Tokens / Task | 2391.7 | **648.7** | `-72.9%` |
| Pitfall Fix Recall (via C2->C3) | — | **100.00%** | 30/30 auto-captured |
| False-Injection Rate (10 unrelated) | — | **0.00%** | target `<= 5%` |
| Stale Drift Detection Rate (10 traps) | — | **100.00%** | target `100%` |
| Stale-Rule Misleading Rate (10 traps) | — | **0.00%** | target `0.00%` |
| Ambient P95 Latency / LLM Calls | — | **3.879 ms / 0 LLM** | G4 millisecond + 0 LLM |

## Breakdown by Pitfall Class

| Pitfall Class | N | No-Mem Success | With-Mem Success | No-Mem Steps | With-Mem Steps | No-Mem Tokens | With-Mem Tokens | Fix Recall |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `version_diff` | 10 | 90.00% | **100.00%** | 4.10 | **1.00** | 2250.0 | **581.5** | 100.00% |
| `project_convention` | 10 | 50.00% | **100.00%** | 4.20 | **1.00** | 2185.0 | **643.9** | 100.00% |
| `env_error` | 10 | 40.00% | **100.00%** | 4.90 | **1.00** | 2740.0 | **720.6** | 100.00% |
