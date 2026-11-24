# End-to-End QA & Turn-Level Retrieval Evaluation Report

## MemBench — `unified`

| Provenance Field | Value |
| :--- | :--- |
| `engine` | `unified` |
| `embedder` | `HashedTfidfEmbedder` |
| `reranker` | `CalibratedLocalReranker` |
| `llm` | `MockLLMClient/DeterministicReader` |
| `judge` | `exact-mcq-match` |
| `protocol` | `MemBench end-to-end MCQ QA + step-level retrieval + 6-bucket attribution` |
| `granularity` | `step-level (identical across all engines)` |
| `commit_hash` | `2b328ac-dirty` |
| `seed` | `42` |
| `dataset` | `MemBench (n=200, 50/cat)` |
| `dataset_sha256` | `10737d41e6db88a2bc5179dab9598d0cb21abbdffa23376d6336c3e862dc877c` |
| `timestamp_utc` | `2026-11-24T04:55:00Z` |

| Category | N | QA Acc | Turn R@5 | Turn R@10 | Turn MRR@10 | Supp. Session R@10 | Update Dom | Avg Tokens | P95 ms |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Overall** | 200 | **53.00%** | 62.28% | 82.03% | 0.7649 | 100.00% | — | 1365.7 | 85.57 |
| `participation_factual` | 50 | **42.00%** | 63.58% | 75.65% | 0.7057 | 100.00% | — | 1925.1 | 131.44 |
| `participation_reflective` | 50 | **58.00%** | 53.17% | 81.90% | 0.7308 | 100.00% | — | 1733.4 | 66.43 |
| `observation_factual` | 50 | **66.00%** | 73.27% | 88.61% | 0.7747 | 100.00% | — | 797.8 | 34.74 |
| `observation_reflective` | 50 | **46.00%** | 59.10% | 81.97% | 0.8485 | 100.00% | — | 1006.4 | 33.45 |

### Error Attribution (94 wrong / 200 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 0 | 0.00% |
| `recall_miss` | 1 | 1.06% |
| `ranking_drop` | 0 | 0.00% |
| `assembly_truncation` | 0 | 0.00% |
| `reasoning_error` | 93 | 98.94% |
| `judge_noise` | 0 | 0.00% |

## LoCoMo — `unified`

| Provenance Field | Value |
| :--- | :--- |
| `engine` | `unified` |
| `embedder` | `HashedTfidfEmbedder` |
| `reranker` | `CalibratedLocalReranker` |
| `llm` | `MockLLMClient/DeterministicReader` |
| `judge` | `deterministic-v1` |
| `protocol` | `LoCoMo end-to-end QA + turn-level retrieval + 6-bucket attribution` |
| `granularity` | `turn-level (primary); session-level supplementary` |
| `commit_hash` | `2b328ac-dirty` |
| `seed` | `42` |
| `dataset` | `LoCoMo (n=200)` |
| `dataset_sha256` | `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4` |
| `timestamp_utc` | `2026-11-24T04:55:22Z` |

| Category | N | QA Acc | Turn R@5 | Turn R@10 | Turn MRR@10 | Supp. Session R@10 | Update Dom | Avg Tokens | P95 ms |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Overall** | 200 | **12.00%** | 46.11% | 54.64% | 0.4412 | 82.74% | — | 2038.6 | 106.95 |
| *Answerable Only* | 160 | **15.00%** | 46.11% | 54.64% | 0.4412 | 82.74% | — | 2040.0 | 110.76 |
| `1 multi-hop` | 40 | **5.00%** | 20.27% | 29.24% | 0.3016 | 71.32% | — | 2041.5 | 111.14 |
| `2 temporal` | 40 | **22.50%** | 77.92% | 79.17% | 0.7133 | 89.17% | — | 2042.0 | 94.33 |
| `3 open-domain` | 40 | **10.00%** | 30.05% | 38.99% | 0.2564 | 77.28% | — | 2041.2 | 139.15 |
| `4 single-hop` | 40 | **22.50%** | 55.00% | 70.00% | 0.4798 | 92.50% | — | 2035.2 | 95.81 |
| `5 adversarial` | 40 | **0.00%** | — | — | — | — | — | 2033.2 | 101.54 |

### Error Attribution (176 wrong / 200 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 1 | 0.57% |
| `recall_miss` | 31 | 17.61% |
| `ranking_drop` | 2 | 1.14% |
| `assembly_truncation` | 3 | 1.70% |
| `reasoning_error` | 128 | 72.73% |
| `judge_noise` | 11 | 6.25% |

## LongMemEval-S — `unified`

| Provenance Field | Value |
| :--- | :--- |
| `engine` | `unified` |
| `embedder` | `HashedTfidfEmbedder` |
| `reranker` | `CalibratedLocalReranker` |
| `llm` | `MockLLMClient/DeterministicReader` |
| `judge` | `deterministic-v1` |
| `protocol` | `LongMemEval-S end-to-end QA + turn-level retrieval + 6-bucket attribution` |
| `granularity` | `turn-level (primary); session-level supplementary` |
| `commit_hash` | `2b328ac-dirty` |
| `seed` | `42` |
| `dataset` | `LongMemEval-S (n=100)` |
| `dataset_sha256` | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| `timestamp_utc` | `2026-11-24T04:57:23Z` |

| Category | N | QA Acc | Turn R@5 | Turn R@10 | Turn MRR@10 | Supp. Session R@10 | Update Dom | Avg Tokens | P95 ms |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Overall** | 100 | **31.00%** | 63.28% | 72.62% | 0.4890 | 89.17% | 64.29% | 2038.2 | 284.06 |
| *Answerable Only* | 86 | **36.05%** | 63.28% | 72.62% | 0.4890 | 89.17% | 64.29% | 2038.3 | 290.71 |
| `single-session-user` | 15 | **26.67%** | 86.67% | 93.33% | 0.5328 | 100.00% | — | 2037.7 | 239.68 |
| `single-session-assistant` | 14 | **71.43%** | 100.00% | 100.00% | 0.7405 | 100.00% | — | 2039.4 | 285.87 |
| `single-session-preference` | 14 | **0.00%** | 10.71% | 23.81% | 0.0748 | 57.14% | — | 2037.3 | 277.90 |
| `knowledge-update` | 14 | **50.00%** | 82.14% | 85.71% | 0.6369 | 96.43% | 64.29% | 2039.0 | 289.08 |
| `temporal-reasoning` | 15 | **46.67%** | 63.33% | 73.33% | 0.5900 | 92.89% | — | 2038.4 | 294.08 |
| `multi-session` | 14 | **21.43%** | 35.12% | 57.98% | 0.3490 | 87.50% | — | 2038.2 | 269.85 |
| `abstention` | 14 | **0.00%** | — | — | — | — | — | 2037.4 | 261.26 |

### Error Attribution (69 wrong / 100 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 0 | 0.00% |
| `recall_miss` | 4 | 5.80% |
| `ranking_drop` | 5 | 7.25% |
| `assembly_truncation` | 1 | 1.45% |
| `reasoning_error` | 50 | 72.46% |
| `judge_noise` | 9 | 13.04% |
