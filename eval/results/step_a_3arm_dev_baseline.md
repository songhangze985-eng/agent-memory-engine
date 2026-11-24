# Frozen-Protocol End-to-End QA Baseline Report

- **Protocol**: `eval/protocol.yaml` (`version=1.0.0`, `sha256=e232828db8cf2bc6...`)
- **Split**: `dev` (`dev_ratio=0.25`) | **Seeds**: `[42, 137, 2026]` | **Arms**: `['full_context', 'turn_rrf', 'unified']`

## Cross-Arm Summary

| Benchmark | Arm | Split | Seeds | QA Acc (Mean) | Spread (Min–Max) | Turn R@10 | Avg Tokens | P95 ms |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `longmemeval` | `full_context` | `dev` | `[42, 137, 2026]` | **30.83%** | 25.00%–40.00% | 1.01% | 126895.0 | 271.79 |
| `longmemeval` | `turn_rrf` | `dev` | `[42, 137, 2026]` | **35.83%** | 32.50%–40.00% | 79.17% | 1651.4 | 1.67 |
| `longmemeval` | `unified` | `dev` | `[42, 137, 2026]` | **30.83%** | 27.50%–35.00% | 72.66% | 2039.2 | 203.62 |
| `locomo` | `full_context` | `dev` | `[42, 137, 2026]` | **5.00%** | 2.50%–7.50% | 2.86% | 25048.0 | 88.20 |
| `locomo` | `turn_rrf` | `dev` | `[42, 137, 2026]` | **9.17%** | 6.25%–12.50% | 38.26% | 384.6 | 0.99 |
| `locomo` | `unified` | `dev` | `[42, 137, 2026]` | **12.92%** | 11.25%–13.75% | 49.87% | 2000.7 | 82.91 |

## LongMemEval-S — `full_context` (Multi-Seed: `[42, 137, 2026]`)

| Provenance Field | Value |
| :--- | :--- |
| `engine` | `full_context` |
| `extractor_model` | `DeterministicRuleExtractor` |
| `reader_model` | `MockLLMClient/DeterministicReader` |
| `judge` | `deterministic-v1` |
| `judge_prompt_version` | `hindsight-aligned-v1` |
| `embedder` | `none` |
| `reranker` | `none` |
| `top_k` | `10` |
| `token_budget` | `2048` |
| `split` | `dev` |
| `protocol` | `LongMemEval-S end-to-end QA + turn-level retrieval + 6-bucket attribution` |
| `protocol_version` | `1.0.0` |
| `protocol_sha256` | `e232828db8cf2bc690f5f32ed4bd8a6b87b2a96ae56b5e4ae479f686f6ba5db4` |
| `granularity` | `full-context (no memory engine)` |
| `commit_hash` | `0aa3ec8-dirty` |
| `seed` | `42` |
| `seeds` | `[42, 137, 2026]` |
| `dataset` | `LongMemEval-S (dev, n=40)` |
| `dataset_sha256` | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| `timestamp_utc` | `2026-11-24T06:18:32Z` |

| Metric | Mean | Min | Max | Std | ±95% CI |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **QA Accuracy** | **30.83%** | 25.00% | 40.00% | 8.04% | ±19.97% |
| Turn R@10 | 1.01% | 0.00% | 2.30% | 1.18% | ±2.92% |
| Turn MRR@10 | 0.0072 | 0.0000 | 0.0173 | 0.0090 | ±0.0224 |
| Avg Injected Tokens | 126895.0 | 126712.0 | 127093.1 | 191.0 | ±474.5 |
| P95 Latency (ms) | 271.79 | 269.96 | 274.66 | 2.51 | ±6.25 |

### Per-Category QA Accuracy (Across Seeds)

| Category | N | Mean | Min | Max |
| :--- | ---: | ---: | ---: | ---: |
| `single-session-user` | 6 | **74.44%** | 40.00% | 100.00% |
| `single-session-assistant` | 5 | **64.44%** | 60.00% | 66.67% |
| `single-session-preference` | 5 | **0.00%** | 0.00% | 0.00% |
| `knowledge-update` | 6 | **44.44%** | 33.33% | 50.00% |
| `temporal-reasoning` | 6 | **31.11%** | 0.00% | 60.00% |
| `multi-session` | 6 | **0.00%** | 0.00% | 0.00% |
| `abstention` | 6 | **0.00%** | 0.00% | 0.00% |

### Representative Error Attribution (Seed 42: 30 wrong / 40 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 0 | 0.00% |
| `recall_miss` | 0 | 0.00% |
| `ranking_drop` | 0 | 0.00% |
| `assembly_truncation` | 0 | 0.00% |
| `reasoning_error` | 26 | 86.67% |
| `judge_noise` | 4 | 13.33% |

## LongMemEval-S — `turn_rrf` (Multi-Seed: `[42, 137, 2026]`)

| Provenance Field | Value |
| :--- | :--- |
| `engine` | `turn_rrf` |
| `extractor_model` | `DeterministicRuleExtractor` |
| `reader_model` | `MockLLMClient/DeterministicReader` |
| `judge` | `deterministic-v1` |
| `judge_prompt_version` | `hindsight-aligned-v1` |
| `embedder` | `HashedTfidfEmbedder` |
| `reranker` | `none` |
| `top_k` | `10` |
| `token_budget` | `2048` |
| `split` | `dev` |
| `protocol` | `LongMemEval-S end-to-end QA + turn-level retrieval + 6-bucket attribution` |
| `protocol_version` | `1.0.0` |
| `protocol_sha256` | `e232828db8cf2bc690f5f32ed4bd8a6b87b2a96ae56b5e4ae479f686f6ba5db4` |
| `granularity` | `turn-level RRF (BM25 + dense, zero extraction)` |
| `commit_hash` | `0aa3ec8-dirty` |
| `seed` | `42` |
| `seeds` | `[42, 137, 2026]` |
| `dataset` | `LongMemEval-S (dev, n=40)` |
| `dataset_sha256` | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| `timestamp_utc` | `2026-11-24T06:18:47Z` |

| Metric | Mean | Min | Max | Std | ±95% CI |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **QA Accuracy** | **35.83%** | 32.50% | 40.00% | 3.82% | ±9.49% |
| Turn R@10 | 79.17% | 77.94% | 80.15% | 1.12% | ±2.79% |
| Turn MRR@10 | 0.7070 | 0.6848 | 0.7472 | 0.0348 | ±0.0866 |
| Avg Injected Tokens | 1651.4 | 1647.7 | 1658.9 | 6.5 | ±16.1 |
| P95 Latency (ms) | 1.67 | 1.54 | 1.78 | 0.12 | ±0.30 |

### Per-Category QA Accuracy (Across Seeds)

| Category | N | Mean | Min | Max |
| :--- | ---: | ---: | ---: | ---: |
| `single-session-user` | 6 | **86.67%** | 60.00% | 100.00% |
| `single-session-assistant` | 5 | **76.67%** | 66.67% | 83.33% |
| `single-session-preference` | 5 | **0.00%** | 0.00% | 0.00% |
| `knowledge-update` | 6 | **44.44%** | 33.33% | 50.00% |
| `temporal-reasoning` | 6 | **42.22%** | 16.67% | 60.00% |
| `multi-session` | 6 | **0.00%** | 0.00% | 0.00% |
| `abstention` | 6 | **0.00%** | 0.00% | 0.00% |

### Representative Error Attribution (Seed 42: 27 wrong / 40 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 0 | 0.00% |
| `recall_miss` | 1 | 3.70% |
| `ranking_drop` | 1 | 3.70% |
| `assembly_truncation` | 0 | 0.00% |
| `reasoning_error` | 21 | 77.78% |
| `judge_noise` | 4 | 14.81% |

## LongMemEval-S — `unified` (Multi-Seed: `[42, 137, 2026]`)

| Provenance Field | Value |
| :--- | :--- |
| `engine` | `unified` |
| `extractor_model` | `DeterministicRuleExtractor` |
| `reader_model` | `MockLLMClient/DeterministicReader` |
| `judge` | `deterministic-v1` |
| `judge_prompt_version` | `hindsight-aligned-v1` |
| `embedder` | `HashedTfidfEmbedder` |
| `reranker` | `CalibratedLocalReranker` |
| `top_k` | `10` |
| `token_budget` | `2048` |
| `split` | `dev` |
| `protocol` | `LongMemEval-S end-to-end QA + turn-level retrieval + 6-bucket attribution` |
| `protocol_version` | `1.0.0` |
| `protocol_sha256` | `e232828db8cf2bc690f5f32ed4bd8a6b87b2a96ae56b5e4ae479f686f6ba5db4` |
| `granularity` | `turn-level (primary); session-level supplementary` |
| `commit_hash` | `0aa3ec8-dirty` |
| `seed` | `42` |
| `seeds` | `[42, 137, 2026]` |
| `dataset` | `LongMemEval-S (dev, n=40)` |
| `dataset_sha256` | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| `timestamp_utc` | `2026-11-24T06:19:48Z` |

| Metric | Mean | Min | Max | Std | ±95% CI |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **QA Accuracy** | **30.83%** | 27.50% | 35.00% | 3.82% | ±9.49% |
| Turn R@10 | 72.66% | 68.24% | 77.21% | 4.49% | ±11.15% |
| Turn MRR@10 | 0.5601 | 0.4782 | 0.6103 | 0.0716 | ±0.1778 |
| Avg Injected Tokens | 2039.2 | 2038.0 | 2040.7 | 1.4 | ±3.4 |
| P95 Latency (ms) | 203.62 | 187.34 | 213.91 | 14.26 | ±35.43 |

### Per-Category QA Accuracy (Across Seeds)

| Category | N | Mean | Min | Max |
| :--- | ---: | ---: | ---: | ---: |
| `single-session-user` | 6 | **86.67%** | 60.00% | 100.00% |
| `single-session-assistant` | 5 | **58.89%** | 50.00% | 66.67% |
| `single-session-preference` | 5 | **0.00%** | 0.00% | 0.00% |
| `knowledge-update` | 6 | **38.89%** | 33.33% | 50.00% |
| `temporal-reasoning` | 6 | **28.89%** | 20.00% | 33.33% |
| `multi-session` | 6 | **0.00%** | 0.00% | 0.00% |
| `abstention` | 6 | **0.00%** | 0.00% | 0.00% |

### Representative Error Attribution (Seed 42: 26 wrong / 40 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 0 | 0.00% |
| `recall_miss` | 1 | 3.85% |
| `ranking_drop` | 1 | 3.85% |
| `assembly_truncation` | 0 | 0.00% |
| `reasoning_error` | 19 | 73.08% |
| `judge_noise` | 5 | 19.23% |

## LoCoMo — `full_context` (Multi-Seed: `[42, 137, 2026]`)

| Provenance Field | Value |
| :--- | :--- |
| `engine` | `full_context` |
| `extractor_model` | `DeterministicRuleExtractor` |
| `reader_model` | `MockLLMClient/DeterministicReader` |
| `judge` | `deterministic-v1` |
| `judge_prompt_version` | `hindsight-aligned-v1` |
| `embedder` | `none` |
| `reranker` | `none` |
| `top_k` | `10` |
| `token_budget` | `2048` |
| `split` | `dev` |
| `protocol` | `LoCoMo end-to-end QA + turn-level retrieval + 6-bucket attribution` |
| `protocol_version` | `1.0.0` |
| `protocol_sha256` | `e232828db8cf2bc690f5f32ed4bd8a6b87b2a96ae56b5e4ae479f686f6ba5db4` |
| `granularity` | `full-context (no memory engine)` |
| `commit_hash` | `0aa3ec8-dirty` |
| `seed` | `42` |
| `seeds` | `[42, 137, 2026]` |
| `dataset` | `LoCoMo (dev, n=80)` |
| `dataset_sha256` | `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4` |
| `timestamp_utc` | `2026-11-24T06:21:36Z` |

| Metric | Mean | Min | Max | Std | ±95% CI |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **QA Accuracy** | **5.00%** | 2.50% | 7.50% | 2.50% | ±6.21% |
| Turn R@10 | 2.86% | 0.62% | 4.43% | 1.99% | ±4.93% |
| Turn MRR@10 | 0.0167 | 0.0132 | 0.0192 | 0.0031 | ±0.0077 |
| Avg Injected Tokens | 25048.0 | 24784.4 | 25269.4 | 245.2 | ±609.3 |
| P95 Latency (ms) | 88.20 | 85.08 | 93.17 | 4.35 | ±10.81 |

### Per-Category QA Accuracy (Across Seeds)

| Category | N | Mean | Min | Max |
| :--- | ---: | ---: | ---: | ---: |
| `1 multi-hop` | 16 | **2.08%** | 0.00% | 6.25% |
| `2 temporal` | 16 | **4.17%** | 0.00% | 12.50% |
| `3 open-domain` | 16 | **0.00%** | 0.00% | 0.00% |
| `4 single-hop` | 16 | **18.75%** | 12.50% | 25.00% |
| `5 adversarial` | 16 | **0.00%** | 0.00% | 0.00% |

### Representative Error Attribution (Seed 42: 78 wrong / 80 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 2 | 2.56% |
| `recall_miss` | 0 | 0.00% |
| `ranking_drop` | 0 | 0.00% |
| `assembly_truncation` | 0 | 0.00% |
| `reasoning_error` | 70 | 89.74% |
| `judge_noise` | 6 | 7.69% |

## LoCoMo — `turn_rrf` (Multi-Seed: `[42, 137, 2026]`)

| Provenance Field | Value |
| :--- | :--- |
| `engine` | `turn_rrf` |
| `extractor_model` | `DeterministicRuleExtractor` |
| `reader_model` | `MockLLMClient/DeterministicReader` |
| `judge` | `deterministic-v1` |
| `judge_prompt_version` | `hindsight-aligned-v1` |
| `embedder` | `HashedTfidfEmbedder` |
| `reranker` | `none` |
| `top_k` | `10` |
| `token_budget` | `2048` |
| `split` | `dev` |
| `protocol` | `LoCoMo end-to-end QA + turn-level retrieval + 6-bucket attribution` |
| `protocol_version` | `1.0.0` |
| `protocol_sha256` | `e232828db8cf2bc690f5f32ed4bd8a6b87b2a96ae56b5e4ae479f686f6ba5db4` |
| `granularity` | `turn-level RRF (BM25 + dense, zero extraction)` |
| `commit_hash` | `0aa3ec8-dirty` |
| `seed` | `42` |
| `seeds` | `[42, 137, 2026]` |
| `dataset` | `LoCoMo (dev, n=80)` |
| `dataset_sha256` | `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4` |
| `timestamp_utc` | `2026-11-24T06:21:40Z` |

| Metric | Mean | Min | Max | Std | ±95% CI |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **QA Accuracy** | **9.17%** | 6.25% | 12.50% | 3.15% | ±7.82% |
| Turn R@10 | 38.26% | 37.81% | 39.06% | 0.70% | ±1.74% |
| Turn MRR@10 | 0.3139 | 0.2884 | 0.3478 | 0.0306 | ±0.0759 |
| Avg Injected Tokens | 384.6 | 378.3 | 389.6 | 5.8 | ±14.3 |
| P95 Latency (ms) | 0.99 | 0.96 | 1.02 | 0.03 | ±0.08 |

### Per-Category QA Accuracy (Across Seeds)

| Category | N | Mean | Min | Max |
| :--- | ---: | ---: | ---: | ---: |
| `1 multi-hop` | 16 | **0.00%** | 0.00% | 0.00% |
| `2 temporal` | 16 | **14.58%** | 6.25% | 25.00% |
| `3 open-domain` | 16 | **6.25%** | 6.25% | 6.25% |
| `4 single-hop` | 16 | **25.00%** | 18.75% | 31.25% |
| `5 adversarial` | 16 | **0.00%** | 0.00% | 0.00% |

### Representative Error Attribution (Seed 42: 75 wrong / 80 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 2 | 2.67% |
| `recall_miss` | 21 | 28.00% |
| `ranking_drop` | 10 | 13.33% |
| `assembly_truncation` | 0 | 0.00% |
| `reasoning_error` | 37 | 49.33% |
| `judge_noise` | 5 | 6.67% |

## LoCoMo — `unified` (Multi-Seed: `[42, 137, 2026]`)

| Provenance Field | Value |
| :--- | :--- |
| `engine` | `unified` |
| `extractor_model` | `DeterministicRuleExtractor` |
| `reader_model` | `MockLLMClient/DeterministicReader` |
| `judge` | `deterministic-v1` |
| `judge_prompt_version` | `hindsight-aligned-v1` |
| `embedder` | `HashedTfidfEmbedder` |
| `reranker` | `CalibratedLocalReranker` |
| `top_k` | `10` |
| `token_budget` | `2048` |
| `split` | `dev` |
| `protocol` | `LoCoMo end-to-end QA + turn-level retrieval + 6-bucket attribution` |
| `protocol_version` | `1.0.0` |
| `protocol_sha256` | `e232828db8cf2bc690f5f32ed4bd8a6b87b2a96ae56b5e4ae479f686f6ba5db4` |
| `granularity` | `turn-level (primary); session-level supplementary` |
| `commit_hash` | `0aa3ec8-dirty` |
| `seed` | `42` |
| `seeds` | `[42, 137, 2026]` |
| `dataset` | `LoCoMo (dev, n=80)` |
| `dataset_sha256` | `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4` |
| `timestamp_utc` | `2026-11-24T06:21:56Z` |

| Metric | Mean | Min | Max | Std | ±95% CI |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **QA Accuracy** | **12.92%** | 11.25% | 13.75% | 1.44% | ±3.59% |
| Turn R@10 | 49.87% | 46.19% | 54.23% | 4.07% | ±10.10% |
| Turn MRR@10 | 0.4159 | 0.3746 | 0.4631 | 0.0445 | ±0.1106 |
| Avg Injected Tokens | 2000.7 | 1994.5 | 2010.2 | 8.4 | ±20.8 |
| P95 Latency (ms) | 82.91 | 65.06 | 95.66 | 15.92 | ±39.56 |

### Per-Category QA Accuracy (Across Seeds)

| Category | N | Mean | Min | Max |
| :--- | ---: | ---: | ---: | ---: |
| `1 multi-hop` | 16 | **4.17%** | 0.00% | 6.25% |
| `2 temporal` | 16 | **25.00%** | 18.75% | 31.25% |
| `3 open-domain` | 16 | **6.25%** | 6.25% | 6.25% |
| `4 single-hop` | 16 | **29.17%** | 18.75% | 37.50% |
| `5 adversarial` | 16 | **0.00%** | 0.00% | 0.00% |

### Representative Error Attribution (Seed 42: 69 wrong / 80 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 2 | 2.90% |
| `recall_miss` | 12 | 17.39% |
| `ranking_drop` | 1 | 1.45% |
| `assembly_truncation` | 0 | 0.00% |
| `reasoning_error` | 50 | 72.46% |
| `judge_noise` | 4 | 5.80% |
