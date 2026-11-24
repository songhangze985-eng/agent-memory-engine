# End-to-End QA & Turn-Level Retrieval Evaluation Report

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
| `commit_hash` | `9479cc0-dirty` |
| `seed` | `42` |
| `dataset` | `LongMemEval-S (n=100)` |
| `dataset_sha256` | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| `timestamp_utc` | `2026-11-24T04:17:31Z` |

| Category | N | QA Acc | Turn R@5 | Turn R@10 | Turn MRR@10 | Supp. Session R@10 | Update Dom | Avg Tokens | P95 ms |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Overall** | 100 | **24.00%** | 62.69% | 71.61% | 0.4919 | 87.71% | 57.14% | 2037.6 | 215.87 |
| *Answerable Only* | 86 | **27.91%** | 62.69% | 71.61% | 0.4919 | 87.71% | 57.14% | 2036.8 | 212.89 |
| `single-session-user` | 15 | **33.33%** | 86.67% | 93.33% | 0.5340 | 100.00% | — | 2032.3 | 185.10 |
| `single-session-assistant` | 14 | **78.57%** | 100.00% | 100.00% | 0.7405 | 100.00% | — | 2029.6 | 232.87 |
| `single-session-preference` | 14 | **0.00%** | 10.71% | 23.81% | 0.0740 | 57.14% | — | 2043.0 | 198.04 |
| `knowledge-update` | 14 | **35.71%** | 78.57% | 85.71% | 0.6726 | 96.43% | 57.14% | 2041.3 | 192.80 |
| `temporal-reasoning` | 15 | **13.33%** | 60.00% | 73.33% | 0.5778 | 92.89% | — | 2036.1 | 222.41 |
| `multi-session` | 14 | **7.14%** | 38.69% | 51.79% | 0.3435 | 78.57% | — | 2039.1 | 206.24 |
| `abstention` | 14 | **0.00%** | — | — | — | — | — | 2042.6 | 217.00 |

### Error Attribution (76 wrong / 100 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 0 | 0.00% |
| `recall_miss` | 4 | 5.26% |
| `ranking_drop` | 6 | 7.89% |
| `assembly_truncation` | 0 | 0.00% |
| `reasoning_error` | 57 | 75.00% |
| `judge_noise` | 9 | 11.84% |

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
| `commit_hash` | `9479cc0-dirty` |
| `seed` | `42` |
| `dataset` | `LoCoMo (n=200)` |
| `dataset_sha256` | `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4` |
| `timestamp_utc` | `2026-11-24T04:17:51Z` |

| Category | N | QA Acc | Turn R@5 | Turn R@10 | Turn MRR@10 | Supp. Session R@10 | Update Dom | Avg Tokens | P95 ms |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Overall** | 200 | **8.50%** | 46.96% | 54.61% | 0.4452 | 82.02% | — | 2007.9 | 62.43 |
| *Answerable Only* | 160 | **10.62%** | 46.96% | 54.61% | 0.4452 | 82.02% | — | 2010.1 | 61.52 |
| `1 multi-hop` | 40 | **0.00%** | 20.63% | 29.11% | 0.3125 | 71.32% | — | 2011.5 | 61.50 |
| `2 temporal` | 40 | **22.50%** | 79.17% | 79.17% | 0.7246 | 89.17% | — | 2010.1 | 60.72 |
| `3 open-domain` | 40 | **2.50%** | 26.51% | 36.29% | 0.2158 | 71.54% | — | 2008.5 | 63.79 |
| `4 single-hop` | 40 | **17.50%** | 60.00% | 72.50% | 0.5105 | 95.00% | — | 2010.1 | 58.42 |
| `5 adversarial` | 40 | **0.00%** | — | — | — | — | — | 1999.5 | 63.19 |

### Error Attribution (183 wrong / 200 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 1 | 0.55% |
| `recall_miss` | 34 | 18.58% |
| `ranking_drop` | 1 | 0.55% |
| `assembly_truncation` | 0 | 0.00% |
| `reasoning_error` | 140 | 76.50% |
| `judge_noise` | 7 | 3.83% |

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
| `commit_hash` | `9479cc0-dirty` |
| `seed` | `42` |
| `dataset` | `MemBench (n=200, 50/cat)` |
| `dataset_sha256` | `10737d41e6db88a2bc5179dab9598d0cb21abbdffa23376d6336c3e862dc877c` |
| `timestamp_utc` | `2026-11-24T04:18:05Z` |

| Category | N | QA Acc | Turn R@5 | Turn R@10 | Turn MRR@10 | Supp. Session R@10 | Update Dom | Avg Tokens | P95 ms |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Overall** | 200 | **51.50%** | 64.55% | 83.44% | 0.7068 | 100.00% | — | 1189.1 | 46.86 |
| `participation_factual` | 50 | **44.00%** | 68.00% | 77.32% | 0.7154 | 100.00% | — | 1857.4 | 51.92 |
| `participation_reflective` | 50 | **56.00%** | 55.97% | 83.57% | 0.6006 | 100.00% | — | 1534.2 | 40.17 |
| `observation_factual` | 50 | **66.00%** | 79.13% | 90.81% | 0.8005 | 100.00% | — | 572.7 | 19.92 |
| `observation_reflective` | 50 | **40.00%** | 55.10% | 82.07% | 0.7107 | 100.00% | — | 792.2 | 18.06 |

### Error Attribution (97 wrong / 200 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 0 | 0.00% |
| `recall_miss` | 1 | 1.03% |
| `ranking_drop` | 0 | 0.00% |
| `assembly_truncation` | 0 | 0.00% |
| `reasoning_error` | 96 | 98.97% |
| `judge_noise` | 0 | 0.00% |
