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
| `commit_hash` | `b2a43b6-dirty` |
| `seed` | `42` |
| `dataset` | `LongMemEval-S (n=100)` |
| `dataset_sha256` | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| `timestamp_utc` | `2026-11-24T04:36:00Z` |

| Category | N | QA Acc | Turn R@5 | Turn R@10 | Turn MRR@10 | Supp. Session R@10 | Update Dom | Avg Tokens | P95 ms |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Overall** | 100 | **30.00%** | 63.28% | 71.61% | 0.4867 | 87.71% | 64.29% | 2037.8 | 217.91 |
| *Answerable Only* | 86 | **34.88%** | 63.28% | 71.61% | 0.4867 | 87.71% | 64.29% | 2037.1 | 217.30 |
| `single-session-user` | 15 | **26.67%** | 86.67% | 93.33% | 0.5340 | 100.00% | — | 2032.3 | 185.79 |
| `single-session-assistant` | 14 | **78.57%** | 100.00% | 100.00% | 0.7405 | 100.00% | — | 2029.6 | 233.61 |
| `single-session-preference` | 14 | **0.00%** | 10.71% | 23.81% | 0.0740 | 57.14% | — | 2043.0 | 189.76 |
| `knowledge-update` | 14 | **50.00%** | 82.14% | 85.71% | 0.6369 | 96.43% | 64.29% | 2041.3 | 211.25 |
| `temporal-reasoning` | 15 | **46.67%** | 63.33% | 73.33% | 0.5833 | 92.89% | — | 2037.5 | 216.42 |
| `multi-session` | 14 | **7.14%** | 35.12% | 51.79% | 0.3411 | 78.57% | — | 2039.1 | 220.41 |
| `abstention` | 14 | **0.00%** | — | — | — | — | — | 2042.6 | 206.49 |

### Error Attribution (70 wrong / 100 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 0 | 0.00% |
| `recall_miss` | 4 | 5.71% |
| `ranking_drop` | 6 | 8.57% |
| `assembly_truncation` | 0 | 0.00% |
| `reasoning_error` | 50 | 71.43% |
| `judge_noise` | 10 | 14.29% |

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
| `commit_hash` | `b2a43b6-dirty` |
| `seed` | `42` |
| `dataset` | `LoCoMo (n=200)` |
| `dataset_sha256` | `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4` |
| `timestamp_utc` | `2026-11-24T04:36:20Z` |

| Category | N | QA Acc | Turn R@5 | Turn R@10 | Turn MRR@10 | Supp. Session R@10 | Update Dom | Avg Tokens | P95 ms |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Overall** | 200 | **9.00%** | 46.39% | 54.29% | 0.4362 | 81.06% | — | 2007.2 | 61.45 |
| *Answerable Only* | 160 | **11.25%** | 46.39% | 54.29% | 0.4362 | 81.06% | — | 2009.3 | 60.63 |
| `1 multi-hop` | 40 | **2.50%** | 20.27% | 30.36% | 0.3057 | 70.04% | — | 2011.1 | 62.00 |
| `2 temporal` | 40 | **22.50%** | 79.17% | 79.17% | 0.7321 | 89.17% | — | 2010.4 | 57.12 |
| `3 open-domain` | 40 | **2.50%** | 27.19% | 36.29% | 0.2127 | 71.54% | — | 2009.9 | 63.05 |
| `4 single-hop` | 40 | **17.50%** | 57.50% | 70.00% | 0.4778 | 92.50% | — | 2006.0 | 58.09 |
| `5 adversarial` | 40 | **0.00%** | — | — | — | — | — | 1998.5 | 61.76 |

### Error Attribution (182 wrong / 200 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 1 | 0.55% |
| `recall_miss` | 34 | 18.68% |
| `ranking_drop` | 1 | 0.55% |
| `assembly_truncation` | 0 | 0.00% |
| `reasoning_error` | 137 | 75.27% |
| `judge_noise` | 9 | 4.95% |

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
| `commit_hash` | `b2a43b6-dirty` |
| `seed` | `42` |
| `dataset` | `MemBench (n=200, 50/cat)` |
| `dataset_sha256` | `10737d41e6db88a2bc5179dab9598d0cb21abbdffa23376d6336c3e862dc877c` |
| `timestamp_utc` | `2026-11-24T04:36:33Z` |

| Category | N | QA Acc | Turn R@5 | Turn R@10 | Turn MRR@10 | Supp. Session R@10 | Update Dom | Avg Tokens | P95 ms |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Overall** | 200 | **51.50%** | 64.55% | 83.44% | 0.7065 | 100.00% | — | 1189.0 | 48.25 |
| `participation_factual` | 50 | **44.00%** | 68.00% | 77.32% | 0.7265 | 100.00% | — | 1856.7 | 59.00 |
| `participation_reflective` | 50 | **56.00%** | 55.97% | 83.57% | 0.6006 | 100.00% | — | 1534.2 | 44.58 |
| `observation_factual` | 50 | **66.00%** | 79.13% | 90.81% | 0.7882 | 100.00% | — | 572.7 | 21.48 |
| `observation_reflective` | 50 | **40.00%** | 55.10% | 82.07% | 0.7107 | 100.00% | — | 792.2 | 20.32 |

### Error Attribution (97 wrong / 200 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 0 | 0.00% |
| `recall_miss` | 1 | 1.03% |
| `ranking_drop` | 0 | 0.00% |
| `assembly_truncation` | 0 | 0.00% |
| `reasoning_error` | 96 | 98.97% |
| `judge_noise` | 0 | 0.00% |
