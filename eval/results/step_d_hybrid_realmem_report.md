# Step D Mixed-Profile & Scope-Chain Verification Report (RealMem-Style)

| Provenance Field | Value |
| :--- | :--- |
| `engine` | `UnifiedMemoryEngine (conversational + coding + hybrid bindings)` |
| `extractor_model` | `DeterministicRuleExtractor + C2 SignalExtractor` |
| `reader_model` | `MockLLMClient/DeterministicReader` |
| `judge` | `deterministic-assertion` |
| `judge_prompt_version` | `hindsight-aligned-v1` |
| `embedder` | `HashedTfidfEmbedder` |
| `reranker` | `CalibratedLocalReranker` |
| `top_k` | `10` |
| `token_budget` | `2048` |
| `split` | `full` |
| `protocol` | `Step D RealMem-Style Mixed-Profile Project Benchmark` |
| `protocol_version` | `1.0.0` |
| `protocol_sha256` | `e232828db8cf2bc690f5f32ed4bd8a6b87b2a96ae56b5e4ae479f686f6ba5db4` |
| `granularity` | `fact + rule + code-anchor + scope-chain` |
| `commit_hash` | `0aa3ec8-dirty` |
| `seed` | `42` |
| `dataset` | `realmem_hybrid_project_suite(checks=8)` |
| `dataset_sha256` | `synthetic/in-memory` |
| `timestamp_utc` | `2026-11-24T06:22:27Z` |

- **Gate Status**: `PASS` (`8/8` checks passed)

| Check ID | Status | Detail |
| :--- | :---: | :--- |
| `shared_user_pref_in_coding` | **PASS** | user:maya pref inherited in coding(atlas) |
| `repo_overrides_user_pref` | **PASS** | repo:atlas-api test_cmd overrides user:maya |
| `user_pref_preserved_in_other_repo` | **PASS** | user:maya test_cmd intact in zephyr-ui |
| `zero_cross_repo_contamination` | **PASS** | atlas-api and zephyr-ui strictly isolated |
| `coding_signal_ambient_hit` | **PASS** | fail_to_pass rule recalled with 0 LLM calls |
| `mid_project_stale_excluded` | **PASS** | pydantic upgrade flagged & excluded stale rule |
| `hybrid_knowledge_update` | **PASS** | response=Project Atlas launch budget is now 28000 USD after Q2 approval. |
| `dual_profile_non_interference` | **PASS** | conversational routing unaffected by coding rules |
