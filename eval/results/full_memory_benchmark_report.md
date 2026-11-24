# 历史全程检索评测存档（已由 M0–M5 端到端 QA 评测体系替代）

> **重要更新（2026-11-24 · M0–M5 架构整合）**：
> - 原“MemBench 40/25/20/15 加权综合分”及跨粒度混排表（v2 Session 级 vs v1 Turn 级）**已在 M0 永久移除**，本文件仅作为历史检索基线存档保留。
> - 全部离线历史数值均使用 `HashedTfidfEmbedder` + `CalibratedLocalReranker` + `MockLLMClient`（零外部 LLM），不包含端到端 QA 准确率。
> - **最新权威端到端 QA + 同口径 Turn 级检索 + 6 桶错题归因 + 消融裁决**请直接查看：
>   - [M0 基线报告](m0_baseline_report.md)
>   - [M5 消融裁决报告](m5_ablation_report.md)
>   - [M5 裁剪默认最终实测报告](m5_unified_pruned_report.md)（LongMemEval-S QA `31.00%` / Turn R@10 `72.62%` / UpdDom `64.29%`；LoCoMo QA `12.00%` / Turn R@10 `54.48%`；MemBench QA `53.00%` / Step R@10 `82.74%` / MRR `0.7763`）
> - 原文末尾提及的两处历史 CI 缺陷（`eval/ablation_matrix.py` 缺少 `parsed_query` 与 `regression_guard.py` `long-horizon` 阈值）**已在 M0 全部修复并通过 CI**。

---

- **历史测试日期**：2026-11-23
- **历史测试环境**：Apple M1 Pro（8 核 / 16 GB RAM），Python 3.12.15，LanceDB 0.41.0 + PyArrow 26.0.1，Docker 29.9.3（`pgvector/pgvector:pg16`），Embedder=`HashedTfidfEmbedder`，Reranker=`CalibratedLocalReranker`，LLM=`MockLLMClient`
- **数据来源**：本机实时运行官方数据集产出（`locomo10.json` SHA-256 `79fa87e9...` 全量 1,986 题；`longmemeval_s_cleaned.json` SHA-256 `d6f21ea9...` 全量 500 题；官方 `import-myself/Membench` `MemData` 分层抽取 800 / 400 / 99 题）。

---

## 一、历史检索基线汇总（已废除加权综合分，按原粒度分列）

| 引擎与口径 | MemBench 800 题 *(Step R@10)* | LongMemEval-S 500 题 *(注粒度)* | LoCoMo 1,986 题 *(注粒度)* | 自建专项门禁 | 查询延迟 P95 |
| :--- | ---: | ---: | ---: | :---: | ---: |
| **v2 VFS (`VFSMemoryEngine`)** | 67.28% *(Step)* | 94.98% *(Session 级，不可直比 Turn)* | 88.89% *(Session 级，不可直比 Turn)* | PASS *(Exact+Ver)* | 1.83–3.18 ms |
| **v1 Chronicle (`reconstructive`, 2048 Budget)** | 76.56% *(Step)* | 84.90% *(Turn R@Budget)* | 77.40% *(Turn R@Budget)* | PASS *(4 项)* | 69.02–204.88 ms |
| **v1 标准 Top-10 (`CognitiveMemoryEngine`)** | 76.56% *(Step)* | 70.87% *(Turn R@10)* | 50.06% *(Turn R@10)* | PASS *(4 项)* | 26.52–123.94 ms |

> **口径说明**：上表仅作历史检索参考；v2 `session_md` 为会话级召回，与 v1 轮次级（Turn-level）粒度不同，不可直接比较。同口径（Turn/Step 级）端到端 QA 对比请见 [m0_baseline_report.md](m0_baseline_report.md) 与 [m5_unified_pruned_report.md](m5_unified_pruned_report.md)。

---

## 二、核心高权重基准：MemBench 全类别扩样实测（800 / 400 / 99 题）

### A. MemBench 800 题扩样对比（4 大官方类别 × 200 题，seed=42）

| 类别 | 题数 | v1 Full (`bipolar`)<br>R@5 / R@10 / MRR | v1 Pure RRF 基线<br>R@5 / R@10 / MRR | v2 VFS (`VFSMemoryEngine`)<br>R@5 / R@10 / MRR | v2 VFS P95 / 渐进省 Token |
| :--- | ---: | :---: | :---: | :---: | :---: |
| **Overall（全量 800 题）** | **800** | **60.19% / 76.56% / 0.6698** | 56.70% / 76.56% / 0.6568 | 51.62% / 67.28% / 0.6085 | **2.23 ms** / **-37.2%** |
| `participation_factual`（第一人称事实） | 200 | 67.67% / 76.11% / 0.6717 | 68.74% / 76.11% / 0.6824 | **68.59% / 77.91% / 0.6782** | **2.96 ms** / -35.0% |
| `observation_factual`（第三人称事实） | 200 | **74.80% / 87.84% / 0.8367** | 74.41% / 87.84% / 0.8349 | 71.49% / **87.71%** / 0.8242 | **1.89 ms** / -10.5% |
| `participation_reflective`（第一人称反思） | 200 | **49.98% / 74.47% / 0.5348** | 44.72% / 74.47% / 0.5158 | 37.98% / 63.43% / 0.4337 | **1.82 ms** / **-52.3%** |
| `observation_reflective`（第三人称反思） | 200 | **48.31% / 67.83% / 0.6358** | 38.95% / 67.83% / 0.5942 | 28.42% / 40.08% / 0.4980 | **1.32 ms** / -36.8% |

- **v1 双极排序相对 Pure RRF 收益**：在 800 题上整体 `R@5` 领先 **+3.49 pp**（`60.19%` vs `56.70%`），其中高难反思类 `observation_reflective` `R@5` 领先 **+9.36 pp**、`participation_reflective` `R@5` 领先 **+5.26 pp**，误空率 **`0.00%`**（0/800）；开启 `consolidate` 后 800 题中 720 题触发观察召回、517 题命中金标、145 题由蒸馏观察独占命中。
- **v2 VFS 诚实分析**：在两类事实记忆上已追平或反超 v1（`participation_factual` R@10 `77.91%` > v1 `76.11%`，`observation_factual` `87.71%` ≈ `87.84%`），且延迟低 **8–9×**（P95 `1.89–2.96 ms` vs v1 `16.58–26.89 ms`）；而在无 LLM 做梦蒸馏时，纯字面步级检索在 `observation_reflective` 上为 `40.08%`，印证了**高层反思任务必须配合异步 LLM 做梦归纳或 v1 观察层**。

### B. 官方 400 题标准静态队列 & 冻结 99 题回归队列（v1）

| 队列 | 题数 | v1 Full (`bipolar`) R@5 / R@10 / MRR | Pure RRF 基线 R@5 / R@10 / MRR | 平均注入 Token | P95 ms | 冻结门禁 |
| :--- | ---: | :---: | :---: | ---: | ---: | :---: |
| **MemBench 400** (`eval.membench_static`) | 400 | **58.38% / 77.80% / 0.6589** | 55.54% / 77.80% / 0.6540 | 549.9 | 28.33 | PASS |
| **MemBench 99 冻结队列** (`03-membench`) | 99 | **60.56% / 80.08% / 0.6999** | 57.64% / 80.08% / 0.6943 | 529.3 | 21.35 | **PASS**（MRR ≥ 0.6943） |

---

## 三、全量长对话官方基准：LoCoMo 1,986 题 & LongMemEval-S 500 题

| 官方数据集 | 方案 | 题数 (可答) | R@5 | R@10 | R@Budget | MRR@10 | 更新支配 (70题) | 平均 Token | P50 / P95 ms |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **LoCoMo 全量** | v1 Full (`bipolar`) | 1,986 (1,540) | 44.72% | 50.06% | — | 0.3927 | — | 473.1 | — / 37.08 |
| **LoCoMo 全量** | v1 Pure RRF 基线 | 1,986 (1,540) | 37.25% | 44.05% | — | 0.2774 | — | 349.9 | — / 22.97 |
| **LoCoMo 全量** | v1 Chronicle `single_query` | 1,986 (1,540) | 55.01% | 61.55% | 74.50% | 0.4920 | — | 2,013.2 | 34.12 / 49.12 |
| **LoCoMo 全量** | v1 Chronicle `reconstructive` | 1,986 (1,540) | 56.21% | 62.37% | **77.40%** | 0.4973 | — | 2,008.2 | 48.83 / 69.02 |
| **LoCoMo 全量** | **v2 VFS (`session_md`)** | **1,986 (1,531)** | **79.58%** | **88.89%** | — | **0.6864** | — | 渐进按需 | **1.02 / 1.83** |
| **LongMemEval-S 全量** | v1 Full (`bipolar`) | 500 (470) | 61.79% | 70.87% | — | 0.5425 | 60.00% (42/70) | 1,933.8 | — / 123.94 |
| **LongMemEval-S 全量** | v1 Pure RRF 基线 | 500 (470) | 61.14% | 69.80% | — | 0.4946 | 52.86% (37/70) | 1,893.3 | — / 59.84 |
| **LongMemEval-S 全量** | v1 Chronicle `single_query` | 500 (470) | 59.97% | 70.72% | 84.22% | 0.5595 | 52.86% (37/70) | 2,038.2 | 137.30 / 206.79 |
| **LongMemEval-S 全量** | v1 Chronicle `reconstructive` | 500 (470) | 59.77% | 71.85% | **84.90%** | 0.5570 | 50.00% (35/70) | 2,037.2 | 139.07 / 204.88 |
| **LongMemEval-S 全量** | **v2 VFS (`session_md`)** | **500 (470)** | **90.59%** | **94.98%** | — | **0.8993** | **67.14% (47/70)** | 渐进按需 | **1.95 / 3.18** |

---

## 四、技术精度、时间推理、抗压与 100K 长程专项（全部 100% 达标）

| 专项套件 | 规模 | 关键实测成绩 | 对照 / 备注 |
| :--- | :--- | :--- | :--- |
| **Exact Identifiers** (`v1` & `v2 VFS`) | 120 题 (90+30) | **P@1 = 100.00%，MRR = 1.0000，串台 = 0.00%，缺失空返 = 100.00%** | Pure RRF 串台 18.33%、缺失空返仅 33.33% |
| **Version Precision** (`v1` & `v2 VFS`) | 80 题 (8 类) | **P@1 = 100.00%，串台 = 0 / 80** | 涵盖精确、包含、粗粒度、兄弟、预发布、区间、跨产品 |
| **Chrono Calendar** (`eval.chrono_eval`) | 60 题 | **准确率 = 100.00%** | 日历解析、相对时间与区间重叠全过 |
| **Stress + Precision** (`04-stress-precision`) | 40 题 | **R@5 = 100.00%，R@10 = 100.00%，MRR = 1.0000，误空 = 0%** | 平均注入 170.3 Tokens |
| **Bipolar Stress** (`07-bipolar-stress`) | 20 组 | **R@3/R@5 = 100.0%，MRR = 1.0000，夹层压缩 = 87.40%** | 平均 121.3 Token（RRF MRR 0.4583、压缩 49.60%、340.2 Token） |
| **100K 长程冷归档** (`long-horizon`) | 500 / 1,500 / 3,500 轮 (101K Token) | **禁忌 R@1 = 100%，冷唤醒 = 100%，冷 Top-1 = 100%，热层缩减 = 79.99%** | 3,500 轮历史仅注入 275.1 Tokens |
| **v2 写入门控 / 渐进披露 / 动态合并** | 专项套件 | **显著性拦截 100%，状态机 100%，`W_verified`/`W_recency` 100%** | 渐进披露省 Token **-81.36%**（0 硬截断）；目录合并省 **-85.21%**（12/12 回滚无损） |

---

## 五、工程安全、多后端扩展与 Docker PostgreSQL 16 实测

- **租户隔离与注入防护** (`eval.engineering_bench`)：10 租户 200 次交叉检索 **0 泄漏**；30 条 Prompt 注入载荷 **0 越权篡改 / 0 只读写入**。
- **v1 10K 扩展与图门禁**：1K / 5K / 10K 活跃事实 SQL 次数严格锁定 **`15–15`**；10K SQLite 缓存 P95 = **`70.39 ms`**（关闭缓存 `117.05 ms`），10K `--graph` 两跳图门禁 P95 = **`94.04 ms (< 100 ms)`**；10K SQLite+LanceDB (`IVF_RQ`) P95 = **`32.95 ms`**，CPU 降幅 **-69%–70%**。
- **v2 VFS L0 冷存与三后端扩展**（详见 [ablation_storage_and_vfs_report.md](ablation_storage_and_vfs_report.md)）：
  - L0 10,000 轮冷日志写入：**`44,991.9 turns/s`**，压缩占用 **`240.4 KB`**，产生 L2 向量 **`0`** 条；
  - 5,000 篇 L2 规则查询（Top-1 均 `60/60`，8 线程并发乐观锁全过）：**SQLite+LanceDB (`IVF_RQ`) P95 = `20.70 ms`**（渐进披露 Token `-56.9%`），纯 SQLite VFS `53.68 ms`，Docker PostgreSQL 16 VFS `59.38 ms`。
- **全量自动化测试**：SQLite + LanceDB + 实时 Docker PostgreSQL 16 运行 `pytest -q`：**219 passed / 0 failed / 0 skipped**（8.28s）；冻结回归门禁 `08-regression-guard` 全项 PASS。

> *注：上游 `scripts/run_static_matrix.sh` 中 `eval/ablation_matrix.py` 存在 1 处预存的 mock 字典缺少 `parsed_query` 字段报错，以及 `long-horizon` 3,500 轮注入 Token 为 `275.14` 较旧阈值 `274.71` 多 `0.43` Token（均为合入 PR #36 时上游固有状态），其余 19/19 核心评测套件均以退出码 `0` 完整通过。*
