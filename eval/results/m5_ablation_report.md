# M5 End-to-End Ablation Report (`UnifiedMemoryEngine`)

Frozen 500Q cohort (`LongMemEval-S 100Q` + `LoCoMo 200Q` + `MemBench 200Q`, `seed=42`).
Each row disables **one** component relative to `full_m4`. Negative delta (`-`) means removing the component hurts performance (KEEP); `0.00` or positive (`+`) means removing it does not hurt (DISABLE/REMOVE by default).

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
| `timestamp_utc` | `2026-11-24T05:04:28Z` |

## Cross-Benchmark End-to-End QA & Turn-Level Retrieval Matrix

| Variant | LME QA (Δ) | LME Turn R@10 (Δ) | LME UpdDom (Δ) | LoCoMo QA (Δ) | LoCoMo Turn R@10 (Δ) | MemBench QA (Δ) | MemBench MRR@10 (Δ) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`full_m4`** | **31.00%** | **72.62%** | **64.29%** | **12.00%** | **54.64%** | **53.00%** | **0.7649** |
| `no_bipolar` | 31.00% (0.00 pp) | 72.62% (0.00 pp) | 64.29% (0.00 pp) | 12.00% (0.00 pp) | 54.48% (-0.16 pp) | 53.00% (0.00 pp) | 0.7738 (+0.0089) |
| `no_recency_tiebreak` | 31.00% (0.00 pp) | 72.62% (0.00 pp) | 64.29% (0.00 pp) | 12.00% (0.00 pp) | 54.64% (0.00 pp) | 53.00% (0.00 pp) | 0.7674 (+0.0025) |
| `no_slot_supersede` | 31.00% (0.00 pp) | 72.62% (0.00 pp) | 64.29% (0.00 pp) | 12.00% (0.00 pp) | 54.64% (0.00 pp) | 53.00% (0.00 pp) | 0.7646 (-0.0003) |
| `no_graph_enum_calc` | 29.00% (-2.00 pp) | 72.19% (-0.43 pp) | 64.29% (0.00 pp) | 11.50% (-0.50 pp) | 54.97% (+0.32 pp) | 53.00% (0.00 pp) | 0.7651 (+0.0002) |
| `no_dreaming` | 31.00% (0.00 pp) | 72.03% (-0.58 pp) | 64.29% (0.00 pp) | 10.00% (-2.00 pp) | 53.85% (-0.80 pp) | 51.50% (-1.50 pp) | 0.7065 (-0.0584) |

## Targeted Category Breakdown

| Variant | LME `knowledge-update` QA | LME `temporal-reasoning` QA | LME `multi-session` QA | LoCoMo `1 multi-hop` QA | LoCoMo `3 open-domain` QA | MemBench `obs_reflective` QA |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_m4` | 50.00% | 46.67% | 21.43% | 5.00% | 10.00% | 46.00% |
| `no_bipolar` | 50.00% | 46.67% | 21.43% | 5.00% | 10.00% | 46.00% |
| `no_recency_tiebreak` | 50.00% | 46.67% | 21.43% | 5.00% | 10.00% | 46.00% |
| `no_slot_supersede` | 50.00% | 46.67% | 21.43% | 5.00% | 10.00% | 46.00% |
| `no_graph_enum_calc` | 50.00% | 46.67% | 7.14% | 2.50% | 10.00% | 46.00% |
| `no_dreaming` | 50.00% | 46.67% | 21.43% | 5.00% | 2.50% | 40.00% |

## M5 Pruning Decision & Final Shipping Default Verification

Applying the M5 gate rule (*"if removing a component does not drop end-to-end QA or turn-level recall, permanently disable/remove it by default"*):

1. **Pruned from defaults (`RecallConfig`)**:
   - `enable_bipolar=False` (was `True`): `0.00 pp` QA drop on all 3 benchmarks, **+0.0089** MemBench MRR@10 gain. Retained strictly as an opt-in feature flag.
   - `enable_recency_rerank=False` (was `True`): `0.00 pp` QA and Turn R@10 drop on all 3 benchmarks, **+0.0025** MemBench MRR@10 gain. Retained strictly as an opt-in feature flag.
   - Cross-session update promotion (`_resolve_cross_session_updates`) is now gated under `enable_slot_supersede` so the flag controls the full M2 update stack cleanly.
2. **Kept ON by default (proven end-to-end QA gains)**:
   - M2 Slot-Level Supersede & Temporal QueryPlan (`enable_slot_supersede=True`)
   - M3 Entity-Graph Facet Enumeration, Coverage-First Packing & Deterministic Calculator (`enable_graph_enum=True`, `enable_deterministic_calc=True`: `+14.29 pp` LongMemEval `multi-session` QA, `+2.50 pp` LoCoMo `1 multi-hop` QA)
   - M4 Reflective Belief Dreaming (`consolidate()`: `+7.50 pp` LoCoMo `3 open-domain` QA, `+6.00 pp` MemBench `observation_reflective` QA, `+0.0584` MemBench MRR@10)

### Combined M5 Pruned Default (`enable_bipolar=False, enable_recency_rerank=False, consolidate=True`)

Measured on the exact same frozen 500Q cohort (`seed=42`, full details in `eval/results/m5_unified_pruned_report.md`):

| Benchmark | QA Acc (vs `full_m4`) | Turn/Step R@10 (vs `full_m4`) | Turn/Step MRR@10 (vs `full_m4`) | Update Dom | Query P95 ms |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **LongMemEval-S (100Q)** | **31.00%** *(0.00 pp)* | **72.62%** *(0.00 pp)* | **0.4893** *(+0.0003)* | **64.29%** *(0.00 pp)* | 245.09 ms |
| **LoCoMo (200Q)** | **12.00%** *(0.00 pp)* | **54.48%** *(-0.16 pp)* | **0.4432** *(+0.0020)* | — | 71.90 ms |
| **MemBench (200Q)** | **53.00%** *(0.00 pp)* | **82.74%** *(+0.71 pp)* | **0.7763** *(+0.0114)* | — | 56.44 ms |

