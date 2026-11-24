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
| `commit_hash` | `4ba561d-dirty` |
| `seed` | `42` |
| `dataset` | `LongMemEval-S (n=100)` |
| `dataset_sha256` | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| `timestamp_utc` | `2026-11-24T04:50:16Z` |

| Category | N | QA Acc | Turn R@5 | Turn R@10 | Turn MRR@10 | Supp. Session R@10 | Update Dom | Avg Tokens | P95 ms |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Overall** | 100 | **31.00%** | 63.28% | 72.03% | 0.4880 | 89.17% | 64.29% | 2038.3 | 302.74 |
| *Answerable Only* | 86 | **36.05%** | 63.28% | 72.03% | 0.4880 | 89.17% | 64.29% | 2037.6 | 298.79 |
| `single-session-user` | 15 | **26.67%** | 86.67% | 93.33% | 0.5340 | 100.00% | — | 2032.3 | 250.74 |
| `single-session-assistant` | 14 | **71.43%** | 100.00% | 100.00% | 0.7405 | 100.00% | — | 2032.9 | 278.60 |
| `single-session-preference` | 14 | **0.00%** | 10.71% | 23.81% | 0.0740 | 57.14% | — | 2043.0 | 295.50 |
| `knowledge-update` | 14 | **50.00%** | 82.14% | 85.71% | 0.6369 | 96.43% | 64.29% | 2041.3 | 298.18 |
| `temporal-reasoning` | 15 | **46.67%** | 63.33% | 70.00% | 0.5833 | 92.89% | — | 2037.5 | 278.67 |
| `multi-session` | 14 | **21.43%** | 35.12% | 57.98% | 0.3490 | 87.50% | — | 2039.0 | 306.58 |
| `abstention` | 14 | **0.00%** | — | — | — | — | — | 2042.6 | 284.28 |

### Error Attribution (69 wrong / 100 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 0 | 0.00% |
| `recall_miss` | 4 | 5.80% |
| `ranking_drop` | 6 | 8.70% |
| `assembly_truncation` | 0 | 0.00% |
| `reasoning_error` | 50 | 72.46% |
| `judge_noise` | 9 | 13.04% |

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
| `commit_hash` | `4ba561d-dirty` |
| `seed` | `42` |
| `dataset` | `LoCoMo (n=200)` |
| `dataset_sha256` | `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4` |
| `timestamp_utc` | `2026-11-24T04:50:43Z` |

| Category | N | QA Acc | Turn R@5 | Turn R@10 | Turn MRR@10 | Supp. Session R@10 | Update Dom | Avg Tokens | P95 ms |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Overall** | 200 | **10.00%** | 46.39% | 53.85% | 0.4356 | 81.06% | — | 1991.8 | 117.86 |
| *Answerable Only* | 160 | **12.50%** | 46.39% | 53.85% | 0.4356 | 81.06% | — | 1996.2 | 117.86 |
| `1 multi-hop` | 40 | **5.00%** | 20.27% | 28.61% | 0.3032 | 70.04% | — | 1986.1 | 123.24 |
| `2 temporal` | 40 | **22.50%** | 79.17% | 79.17% | 0.7321 | 89.17% | — | 2010.4 | 114.73 |
| `3 open-domain` | 40 | **2.50%** | 27.19% | 36.29% | 0.2127 | 71.54% | — | 2001.7 | 119.17 |
| `4 single-hop` | 40 | **20.00%** | 57.50% | 70.00% | 0.4778 | 92.50% | — | 1986.5 | 101.56 |
| `5 adversarial` | 40 | **0.00%** | — | — | — | — | — | 1974.5 | 99.87 |

### Error Attribution (180 wrong / 200 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 1 | 0.56% |
| `recall_miss` | 34 | 18.89% |
| `ranking_drop` | 1 | 0.56% |
| `assembly_truncation` | 0 | 0.00% |
| `reasoning_error` | 133 | 73.89% |
| `judge_noise` | 11 | 6.11% |

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
| `commit_hash` | `4ba561d-dirty` |
| `seed` | `42` |
| `dataset` | `MemBench (n=200, 50/cat)` |
| `dataset_sha256` | `10737d41e6db88a2bc5179dab9598d0cb21abbdffa23376d6336c3e862dc877c` |
| `timestamp_utc` | `2026-11-24T04:51:01Z` |

| Category | N | QA Acc | Turn R@5 | Turn R@10 | Turn MRR@10 | Supp. Session R@10 | Update Dom | Avg Tokens | P95 ms |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Overall** | 200 | **51.50%** | 64.55% | 83.44% | 0.7065 | 100.00% | — | 1189.0 | 74.42 |
| `participation_factual` | 50 | **44.00%** | 68.00% | 77.32% | 0.7265 | 100.00% | — | 1856.7 | 82.97 |
| `participation_reflective` | 50 | **56.00%** | 55.97% | 83.57% | 0.6006 | 100.00% | — | 1534.2 | 73.65 |
| `observation_factual` | 50 | **66.00%** | 79.13% | 90.81% | 0.7882 | 100.00% | — | 572.7 | 27.50 |
| `observation_reflective` | 50 | **40.00%** | 55.10% | 82.07% | 0.7107 | 100.00% | — | 792.2 | 23.43 |

### Error Attribution (97 wrong / 200 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 0 | 0.00% |
| `recall_miss` | 1 | 1.03% |
| `ranking_drop` | 0 | 0.00% |
| `assembly_truncation` | 0 | 0.00% |
| `reasoning_error` | 96 | 98.97% |
| `judge_noise` | 0 | 0.00% |
