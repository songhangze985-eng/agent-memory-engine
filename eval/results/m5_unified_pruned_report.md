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
| `commit_hash` | `7a20d51-dirty` |
| `seed` | `42` |
| `dataset` | `LongMemEval-S (n=100)` |
| `dataset_sha256` | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| `timestamp_utc` | `2026-11-24T05:24:15Z` |

| Category | N | QA Acc | Turn R@5 | Turn R@10 | Turn MRR@10 | Supp. Session R@10 | Update Dom | Avg Tokens | P95 ms |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Overall** | 100 | **31.00%** | 63.86% | 72.62% | 0.4893 | 89.17% | 64.29% | 2038.6 | 245.09 |
| *Answerable Only* | 86 | **36.05%** | 63.86% | 72.62% | 0.4893 | 89.17% | 64.29% | 2038.8 | 243.07 |
| `single-session-user` | 15 | **26.67%** | 86.67% | 93.33% | 0.5328 | 100.00% | — | 2037.7 | 200.28 |
| `single-session-assistant` | 14 | **71.43%** | 100.00% | 100.00% | 0.7405 | 100.00% | — | 2039.8 | 252.38 |
| `single-session-preference` | 14 | **0.00%** | 10.71% | 23.81% | 0.0748 | 57.14% | — | 2038.5 | 196.05 |
| `knowledge-update` | 14 | **50.00%** | 82.14% | 85.71% | 0.6369 | 96.43% | 64.29% | 2038.9 | 246.55 |
| `temporal-reasoning` | 15 | **46.67%** | 63.33% | 73.33% | 0.5900 | 92.89% | — | 2039.7 | 233.91 |
| `multi-session` | 14 | **21.43%** | 38.69% | 57.98% | 0.3506 | 87.50% | — | 2038.2 | 254.62 |
| `abstention` | 14 | **0.00%** | — | — | — | — | — | 2037.5 | 255.44 |

### Error Attribution (69 wrong / 100 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 0 | 0.00% |
| `recall_miss` | 4 | 5.80% |
| `ranking_drop` | 5 | 7.25% |
| `assembly_truncation` | 1 | 1.45% |
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
| `commit_hash` | `7a20d51-dirty` |
| `seed` | `42` |
| `dataset` | `LoCoMo (n=200)` |
| `dataset_sha256` | `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4` |
| `timestamp_utc` | `2026-11-24T05:24:33Z` |

| Category | N | QA Acc | Turn R@5 | Turn R@10 | Turn MRR@10 | Supp. Session R@10 | Update Dom | Avg Tokens | P95 ms |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Overall** | 200 | **12.00%** | 46.11% | 54.48% | 0.4432 | 82.61% | — | 2038.2 | 71.90 |
| *Answerable Only* | 160 | **15.00%** | 46.11% | 54.48% | 0.4432 | 82.61% | — | 2039.5 | 72.17 |
| `1 multi-hop` | 40 | **5.00%** | 20.89% | 28.61% | 0.3122 | 70.47% | — | 2041.1 | 77.04 |
| `2 temporal` | 40 | **22.50%** | 77.92% | 79.17% | 0.7133 | 89.17% | — | 2042.0 | 71.53 |
| `3 open-domain` | 40 | **10.00%** | 29.37% | 38.99% | 0.2473 | 77.62% | — | 2040.3 | 71.22 |
| `4 single-hop` | 40 | **22.50%** | 55.00% | 70.00% | 0.4851 | 92.50% | — | 2034.5 | 71.16 |
| `5 adversarial` | 40 | **0.00%** | — | — | — | — | — | 2033.3 | 68.92 |

### Error Attribution (176 wrong / 200 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 1 | 0.57% |
| `recall_miss` | 31 | 17.61% |
| `ranking_drop` | 2 | 1.14% |
| `assembly_truncation` | 3 | 1.70% |
| `reasoning_error` | 129 | 73.30% |
| `judge_noise` | 10 | 5.68% |

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
| `commit_hash` | `7a20d51-dirty` |
| `seed` | `42` |
| `dataset` | `MemBench (n=200, 50/cat)` |
| `dataset_sha256` | `10737d41e6db88a2bc5179dab9598d0cb21abbdffa23376d6336c3e862dc877c` |
| `timestamp_utc` | `2026-11-24T05:24:49Z` |

| Category | N | QA Acc | Turn R@5 | Turn R@10 | Turn MRR@10 | Supp. Session R@10 | Update Dom | Avg Tokens | P95 ms |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Overall** | 200 | **53.00%** | 64.03% | 82.74% | 0.7763 | 100.00% | — | 1377.2 | 56.44 |
| `participation_factual` | 50 | **42.00%** | 63.58% | 75.65% | 0.7057 | 100.00% | — | 1925.0 | 60.49 |
| `participation_reflective` | 50 | **56.00%** | 51.43% | 81.00% | 0.7484 | 100.00% | — | 1747.4 | 51.99 |
| `observation_factual` | 50 | **68.00%** | 75.27% | 88.61% | 0.7847 | 100.00% | — | 811.6 | 29.71 |
| `observation_reflective` | 50 | **46.00%** | 65.83% | 85.70% | 0.8663 | 100.00% | — | 1024.7 | 26.93 |

### Error Attribution (94 wrong / 200 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 0 | 0.00% |
| `recall_miss` | 1 | 1.06% |
| `ranking_drop` | 0 | 0.00% |
| `assembly_truncation` | 0 | 0.00% |
| `reasoning_error` | 93 | 98.94% |
| `judge_noise` | 0 | 0.00% |
