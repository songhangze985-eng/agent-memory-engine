# End-to-End QA & Turn-Level Retrieval Evaluation Report

## LongMemEval-S — `v1_chronicle`

| Provenance Field | Value |
| :--- | :--- |
| `engine` | `v1_chronicle` |
| `embedder` | `HashedTfidfEmbedder` |
| `reranker` | `CalibratedLocalReranker` |
| `llm` | `MockLLMClient/DeterministicReader` |
| `judge` | `deterministic-v1` |
| `protocol` | `LongMemEval-S end-to-end QA + turn-level retrieval + 6-bucket attribution` |
| `granularity` | `turn-level (primary); session-level supplementary` |
| `commit_hash` | `dc65002-dirty` |
| `seed` | `42` |
| `dataset` | `LongMemEval-S (n=100)` |
| `dataset_sha256` | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| `timestamp_utc` | `2026-11-23T17:14:20Z` |

| Category | N | QA Acc | Turn R@5 | Turn R@10 | Turn MRR@10 | Supp. Session R@10 | Update Dom | Avg Tokens | P95 ms |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Overall** | 100 | **10.00%** | 62.69% | 71.61% | 0.4919 | 87.71% | 57.14% | 2037.6 | 233.06 |
| *Answerable Only* | 86 | **10.47%** | 62.69% | 71.61% | 0.4919 | 87.71% | 57.14% | 2036.8 | 231.46 |
| `single-session-user` | 15 | **13.33%** | 86.67% | 93.33% | 0.5340 | 100.00% | — | 2032.1 | 186.82 |
| `single-session-assistant` | 14 | **28.57%** | 100.00% | 100.00% | 0.7405 | 100.00% | — | 2029.6 | 260.17 |
| `single-session-preference` | 14 | **0.00%** | 10.71% | 23.81% | 0.0740 | 57.14% | — | 2043.0 | 214.79 |
| `knowledge-update` | 14 | **7.14%** | 78.57% | 85.71% | 0.6726 | 96.43% | 57.14% | 2041.3 | 216.63 |
| `temporal-reasoning` | 15 | **13.33%** | 60.00% | 73.33% | 0.5778 | 92.89% | — | 2036.1 | 229.45 |
| `multi-session` | 14 | **0.00%** | 38.69% | 51.79% | 0.3435 | 78.57% | — | 2039.1 | 220.95 |
| `abstention` | 14 | **7.14%** | — | — | — | — | — | 2042.6 | 202.18 |

### Error Attribution (90 wrong / 100 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 0 | 0.00% |
| `recall_miss` | 4 | 4.44% |
| `ranking_drop` | 6 | 6.67% |
| `assembly_truncation` | 0 | 0.00% |
| `reasoning_error` | 69 | 76.67% |
| `judge_noise` | 11 | 12.22% |

## LongMemEval-S — `v2_session_md`

| Provenance Field | Value |
| :--- | :--- |
| `engine` | `v2_session_md` |
| `embedder` | `HashedTfidfEmbedder` |
| `reranker` | `VFS_W_verified_x_W_recency` |
| `llm` | `MockLLMClient/DeterministicReader` |
| `judge` | `deterministic-v1` |
| `protocol` | `LongMemEval-S end-to-end QA + turn-level retrieval + 6-bucket attribution` |
| `granularity` | `turn-level (session_md projected 1-turn/hit; session-level supplementary)` |
| `commit_hash` | `dc65002-dirty` |
| `seed` | `42` |
| `dataset` | `LongMemEval-S (n=100)` |
| `dataset_sha256` | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| `timestamp_utc` | `2026-11-23T17:14:33Z` |

| Category | N | QA Acc | Turn R@5 | Turn R@10 | Turn MRR@10 | Supp. Session R@10 | Update Dom | Avg Tokens | P95 ms |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Overall** | 100 | **9.00%** | 36.05% | 36.43% | 0.3866 | 96.28% | 85.71% | 1317.9 | 8.01 |
| *Answerable Only* | 86 | **9.30%** | 36.05% | 36.43% | 0.3866 | 96.28% | 85.71% | 1309.5 | 8.09 |
| `single-session-user` | 15 | **0.00%** | 40.00% | 40.00% | 0.4000 | 100.00% | — | 1367.8 | 7.57 |
| `single-session-assistant` | 14 | **57.14%** | 78.57% | 78.57% | 0.7857 | 100.00% | — | 1564.9 | 8.25 |
| `single-session-preference` | 14 | **0.00%** | 7.14% | 7.14% | 0.0536 | 100.00% | — | 895.9 | 7.96 |
| `knowledge-update` | 14 | **0.00%** | 53.57% | 53.57% | 0.6071 | 96.43% | 85.71% | 1631.0 | 7.00 |
| `temporal-reasoning` | 15 | **0.00%** | 30.00% | 30.00% | 0.3667 | 94.22% | — | 1105.1 | 6.67 |
| `multi-session` | 14 | **0.00%** | 7.14% | 9.52% | 0.1071 | 86.90% | — | 1302.9 | 7.61 |
| `abstention` | 14 | **7.14%** | — | — | — | — | — | 1369.6 | 5.89 |

### Error Attribution (91 wrong / 100 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 0 | 0.00% |
| `recall_miss` | 0 | 0.00% |
| `ranking_drop` | 1 | 1.10% |
| `assembly_truncation` | 0 | 0.00% |
| `reasoning_error` | 89 | 97.80% |
| `judge_noise` | 1 | 1.10% |

## LongMemEval-S — `v2_turn_md`

| Provenance Field | Value |
| :--- | :--- |
| `engine` | `v2_turn_md` |
| `embedder` | `HashedTfidfEmbedder` |
| `reranker` | `VFS_W_verified_x_W_recency` |
| `llm` | `MockLLMClient/DeterministicReader` |
| `judge` | `deterministic-v1` |
| `protocol` | `LongMemEval-S end-to-end QA + turn-level retrieval + 6-bucket attribution` |
| `granularity` | `turn-level (primary); session-level supplementary` |
| `commit_hash` | `dc65002-dirty` |
| `seed` | `42` |
| `dataset` | `LongMemEval-S (n=100)` |
| `dataset_sha256` | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| `timestamp_utc` | `2026-11-23T17:14:55Z` |

| Category | N | QA Acc | Turn R@5 | Turn R@10 | Turn MRR@10 | Supp. Session R@10 | Update Dom | Avg Tokens | P95 ms |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Overall** | 100 | **24.00%** | 64.83% | 80.08% | 0.5801 | 92.36% | 57.14% | 1578.9 | 19.23 |
| *Answerable Only* | 86 | **24.42%** | 64.83% | 80.08% | 0.5801 | 92.36% | 57.14% | 1598.1 | 19.24 |
| `single-session-user` | 15 | **53.33%** | 86.67% | 100.00% | 0.7296 | 100.00% | — | 1716.4 | 15.23 |
| `single-session-assistant` | 14 | **57.14%** | 85.71% | 92.86% | 0.7245 | 100.00% | — | 1614.8 | 22.88 |
| `single-session-preference` | 14 | **0.00%** | 34.52% | 52.38% | 0.2114 | 85.71% | — | 1616.6 | 19.15 |
| `knowledge-update` | 14 | **21.43%** | 89.29% | 92.86% | 0.7321 | 96.43% | 57.14% | 1628.8 | 15.49 |
| `temporal-reasoning` | 15 | **6.67%** | 56.67% | 76.67% | 0.6130 | 89.56% | — | 1685.1 | 17.91 |
| `multi-session` | 14 | **7.14%** | 35.12% | 64.52% | 0.4567 | 82.14% | — | 1312.1 | 19.24 |
| `abstention` | 14 | **21.43%** | — | — | — | — | — | 1461.0 | 15.39 |

### Error Attribution (76 wrong / 100 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 0 | 0.00% |
| `recall_miss` | 6 | 7.89% |
| `ranking_drop` | 4 | 5.26% |
| `assembly_truncation` | 3 | 3.95% |
| `reasoning_error` | 54 | 71.05% |
| `judge_noise` | 9 | 11.84% |

## LoCoMo — `v1_chronicle`

| Provenance Field | Value |
| :--- | :--- |
| `engine` | `v1_chronicle` |
| `embedder` | `HashedTfidfEmbedder` |
| `reranker` | `CalibratedLocalReranker` |
| `llm` | `MockLLMClient/DeterministicReader` |
| `judge` | `deterministic-v1` |
| `protocol` | `LoCoMo end-to-end QA + turn-level retrieval + 6-bucket attribution` |
| `granularity` | `turn-level (primary); session-level supplementary` |
| `commit_hash` | `dc65002-dirty` |
| `seed` | `42` |
| `dataset` | `LoCoMo (n=200)` |
| `dataset_sha256` | `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4` |
| `timestamp_utc` | `2026-11-23T17:17:04Z` |

| Category | N | QA Acc | Turn R@5 | Turn R@10 | Turn MRR@10 | Supp. Session R@10 | Update Dom | Avg Tokens | P95 ms |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Overall** | 200 | **6.00%** | 46.96% | 54.61% | 0.4452 | 82.02% | — | 2008.7 | 121.10 |
| *Answerable Only* | 160 | **5.62%** | 46.96% | 54.61% | 0.4452 | 82.02% | — | 2011.0 | 121.39 |
| `1 multi-hop` | 40 | **2.50%** | 20.63% | 29.11% | 0.3125 | 71.32% | — | 2013.9 | 122.58 |
| `2 temporal` | 40 | **0.00%** | 79.17% | 79.17% | 0.7246 | 89.17% | — | 2011.4 | 99.75 |
| `3 open-domain` | 40 | **5.00%** | 26.51% | 36.29% | 0.2158 | 71.54% | — | 2008.5 | 125.20 |
| `4 single-hop` | 40 | **15.00%** | 60.00% | 72.50% | 0.5105 | 95.00% | — | 2010.1 | 155.33 |
| `5 adversarial` | 40 | **7.50%** | — | — | — | — | — | 1999.5 | 109.96 |

### Error Attribution (188 wrong / 200 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 1 | 0.53% |
| `recall_miss` | 34 | 18.09% |
| `ranking_drop` | 1 | 0.53% |
| `assembly_truncation` | 0 | 0.00% |
| `reasoning_error` | 144 | 76.60% |
| `judge_noise` | 8 | 4.26% |

## LoCoMo — `v2_session_md`

| Provenance Field | Value |
| :--- | :--- |
| `engine` | `v2_session_md` |
| `embedder` | `HashedTfidfEmbedder` |
| `reranker` | `VFS_W_verified_x_W_recency` |
| `llm` | `MockLLMClient/DeterministicReader` |
| `judge` | `deterministic-v1` |
| `protocol` | `LoCoMo end-to-end QA + turn-level retrieval + 6-bucket attribution` |
| `granularity` | `turn-level (session_md projected 1-turn/hit; session-level supplementary)` |
| `commit_hash` | `dc65002-dirty` |
| `seed` | `42` |
| `dataset` | `LoCoMo (n=200)` |
| `dataset_sha256` | `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4` |
| `timestamp_utc` | `2026-11-23T17:17:05Z` |

| Category | N | QA Acc | Turn R@5 | Turn R@10 | Turn MRR@10 | Supp. Session R@10 | Update Dom | Avg Tokens | P95 ms |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Overall** | 200 | **7.50%** | 28.72% | 31.76% | 0.2498 | 82.07% | — | 1689.0 | 5.39 |
| *Answerable Only* | 160 | **4.38%** | 28.72% | 31.76% | 0.2498 | 82.07% | — | 1710.6 | 5.39 |
| `1 multi-hop` | 40 | **0.00%** | 12.30% | 14.10% | 0.2238 | 76.17% | — | 1812.8 | 5.57 |
| `2 temporal` | 40 | **0.00%** | 51.67% | 52.92% | 0.3808 | 89.17% | — | 1698.4 | 5.00 |
| `3 open-domain` | 40 | **2.50%** | 14.86% | 21.78% | 0.1474 | 72.04% | — | 1474.9 | 4.88 |
| `4 single-hop` | 40 | **15.00%** | 35.00% | 37.50% | 0.2394 | 90.00% | — | 1856.4 | 4.87 |
| `5 adversarial` | 40 | **20.00%** | — | — | — | — | — | 1602.4 | 4.98 |

### Error Attribution (185 wrong / 200 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 1 | 0.54% |
| `recall_miss` | 1 | 0.54% |
| `ranking_drop` | 13 | 7.03% |
| `assembly_truncation` | 10 | 5.41% |
| `reasoning_error` | 153 | 82.70% |
| `judge_noise` | 7 | 3.78% |

## LoCoMo — `v2_turn_md`

| Provenance Field | Value |
| :--- | :--- |
| `engine` | `v2_turn_md` |
| `embedder` | `HashedTfidfEmbedder` |
| `reranker` | `VFS_W_verified_x_W_recency` |
| `llm` | `MockLLMClient/DeterministicReader` |
| `judge` | `deterministic-v1` |
| `protocol` | `LoCoMo end-to-end QA + turn-level retrieval + 6-bucket attribution` |
| `granularity` | `turn-level (primary); session-level supplementary` |
| `commit_hash` | `dc65002-dirty` |
| `seed` | `42` |
| `dataset` | `LoCoMo (n=200)` |
| `dataset_sha256` | `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4` |
| `timestamp_utc` | `2026-11-23T17:17:08Z` |

| Category | N | QA Acc | Turn R@5 | Turn R@10 | Turn MRR@10 | Supp. Session R@10 | Update Dom | Avg Tokens | P95 ms |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Overall** | 200 | **6.50%** | 37.30% | 42.41% | 0.3309 | 71.93% | — | 489.4 | 17.92 |
| *Answerable Only* | 160 | **7.50%** | 37.30% | 42.41% | 0.3309 | 71.93% | — | 489.2 | 17.49 |
| `1 multi-hop` | 40 | **0.00%** | 14.60% | 19.50% | 0.2505 | 63.82% | — | 494.5 | 16.75 |
| `2 temporal` | 40 | **0.00%** | 67.92% | 69.17% | 0.5169 | 80.00% | — | 481.7 | 14.59 |
| `3 open-domain` | 40 | **5.00%** | 12.32% | 21.94% | 0.1402 | 52.24% | — | 479.0 | 19.40 |
| `4 single-hop` | 40 | **25.00%** | 52.50% | 57.50% | 0.4015 | 90.00% | — | 501.4 | 17.49 |
| `5 adversarial` | 40 | **2.50%** | — | — | — | — | — | 490.1 | 19.15 |

### Error Attribution (187 wrong / 200 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 1 | 0.53% |
| `recall_miss` | 49 | 26.20% |
| `ranking_drop` | 28 | 14.97% |
| `assembly_truncation` | 0 | 0.00% |
| `reasoning_error` | 103 | 55.08% |
| `judge_noise` | 6 | 3.21% |

## MemBench — `v1_chronicle`

| Provenance Field | Value |
| :--- | :--- |
| `engine` | `v1_chronicle` |
| `embedder` | `HashedTfidfEmbedder` |
| `reranker` | `CalibratedLocalReranker` |
| `llm` | `MockLLMClient/DeterministicReader` |
| `judge` | `exact-mcq-match` |
| `protocol` | `MemBench end-to-end MCQ QA + step-level retrieval + 6-bucket attribution` |
| `granularity` | `step-level (identical across all engines)` |
| `commit_hash` | `dc65002-dirty` |
| `seed` | `42` |
| `dataset` | `MemBench (n=200, 50/cat)` |
| `dataset_sha256` | `10737d41e6db88a2bc5179dab9598d0cb21abbdffa23376d6336c3e862dc877c` |
| `timestamp_utc` | `2026-11-23T17:17:24Z` |

| Category | N | QA Acc | Turn R@5 | Turn R@10 | Turn MRR@10 | Supp. Session R@10 | Update Dom | Avg Tokens | P95 ms |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Overall** | 200 | **51.50%** | 64.55% | 83.44% | 0.7068 | 100.00% | — | 1189.1 | 57.30 |
| `participation_factual` | 50 | **44.00%** | 68.00% | 77.32% | 0.7154 | 100.00% | — | 1857.4 | 65.14 |
| `participation_reflective` | 50 | **56.00%** | 55.97% | 83.57% | 0.6006 | 100.00% | — | 1534.2 | 54.01 |
| `observation_factual` | 50 | **66.00%** | 79.13% | 90.81% | 0.8005 | 100.00% | — | 572.7 | 28.71 |
| `observation_reflective` | 50 | **40.00%** | 55.10% | 82.07% | 0.7107 | 100.00% | — | 792.2 | 21.53 |

### Error Attribution (97 wrong / 200 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 0 | 0.00% |
| `recall_miss` | 1 | 1.03% |
| `ranking_drop` | 0 | 0.00% |
| `assembly_truncation` | 0 | 0.00% |
| `reasoning_error` | 96 | 98.97% |
| `judge_noise` | 0 | 0.00% |

## MemBench — `v2_session_md`

| Provenance Field | Value |
| :--- | :--- |
| `engine` | `v2_session_md` |
| `embedder` | `HashedTfidfEmbedder` |
| `reranker` | `VFS_W_verified_x_W_recency` |
| `llm` | `MockLLMClient/DeterministicReader` |
| `judge` | `exact-mcq-match` |
| `protocol` | `MemBench end-to-end MCQ QA + step-level retrieval + 6-bucket attribution` |
| `granularity` | `step-level (identical across all engines)` |
| `commit_hash` | `dc65002-dirty` |
| `seed` | `42` |
| `dataset` | `MemBench (n=200, 50/cat)` |
| `dataset_sha256` | `10737d41e6db88a2bc5179dab9598d0cb21abbdffa23376d6336c3e862dc877c` |
| `timestamp_utc` | `2026-11-23T17:17:31Z` |

| Category | N | QA Acc | Turn R@5 | Turn R@10 | Turn MRR@10 | Supp. Session R@10 | Update Dom | Avg Tokens | P95 ms |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Overall** | 200 | **50.50%** | 53.04% | 68.41% | 0.6153 | 98.00% | — | 640.3 | 6.60 |
| `participation_factual` | 50 | **48.00%** | 61.13% | 69.20% | 0.6666 | 100.00% | — | 766.1 | 7.66 |
| `participation_reflective` | 50 | **52.00%** | 49.53% | 76.17% | 0.5842 | 100.00% | — | 826.3 | 3.19 |
| `observation_factual` | 50 | **66.00%** | 66.28% | 82.59% | 0.6848 | 92.00% | — | 449.0 | 4.31 |
| `observation_reflective` | 50 | **36.00%** | 35.20% | 45.70% | 0.5257 | 100.00% | — | 519.9 | 2.73 |

### Error Attribution (99 wrong / 200 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 0 | 0.00% |
| `recall_miss` | 13 | 13.13% |
| `ranking_drop` | 4 | 4.04% |
| `assembly_truncation` | 1 | 1.01% |
| `reasoning_error` | 81 | 81.82% |
| `judge_noise` | 0 | 0.00% |

## MemBench — `v2_turn_md`

| Provenance Field | Value |
| :--- | :--- |
| `engine` | `v2_turn_md` |
| `embedder` | `HashedTfidfEmbedder` |
| `reranker` | `VFS_W_verified_x_W_recency` |
| `llm` | `MockLLMClient/DeterministicReader` |
| `judge` | `exact-mcq-match` |
| `protocol` | `MemBench end-to-end MCQ QA + step-level retrieval + 6-bucket attribution` |
| `granularity` | `step-level (identical across all engines)` |
| `commit_hash` | `dc65002-dirty` |
| `seed` | `42` |
| `dataset` | `MemBench (n=200, 50/cat)` |
| `dataset_sha256` | `10737d41e6db88a2bc5179dab9598d0cb21abbdffa23376d6336c3e862dc877c` |
| `timestamp_utc` | `2026-11-23T17:17:37Z` |

| Category | N | QA Acc | Turn R@5 | Turn R@10 | Turn MRR@10 | Supp. Session R@10 | Update Dom | Avg Tokens | P95 ms |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Overall** | 200 | **49.50%** | 52.79% | 68.14% | 0.6146 | 98.00% | — | 640.4 | 6.42 |
| `participation_factual` | 50 | **46.00%** | 61.13% | 69.20% | 0.6661 | 100.00% | — | 766.5 | 7.57 |
| `participation_reflective` | 50 | **52.00%** | 49.53% | 76.17% | 0.5842 | 100.00% | — | 826.3 | 3.69 |
| `observation_factual` | 50 | **64.00%** | 65.28% | 81.48% | 0.6824 | 92.00% | — | 449.0 | 3.98 |
| `observation_reflective` | 50 | **36.00%** | 35.20% | 45.70% | 0.5257 | 100.00% | — | 519.9 | 2.91 |

### Error Attribution (101 wrong / 200 total)

| Failure Bucket | Count | Share of Errors |
| :--- | ---: | ---: |
| `extraction_miss` | 0 | 0.00% |
| `recall_miss` | 13 | 12.87% |
| `ranking_drop` | 5 | 4.95% |
| `assembly_truncation` | 1 | 0.99% |
| `reasoning_error` | 82 | 81.19% |
| `judge_noise` | 0 | 0.00% |
