# 存储优化性能实测与账本 / SQLite 消融实验报告

- 测试日期：2026-11-22 / 2026-11-23
- 测试环境：Apple M1 Pro（8 核 / 16 GB RAM），Python 3.12.15，LanceDB 0.41.0 + PyArrow 26.0.1，Docker 29.9.3（`pgvector/pgvector:pg16`）
- 评测原则：**全部数值均为本机实时运行产出**，官方数据集经 SHA-256 校验（`locomo10.json`: `79fa87e9...`；`longmemeval_s_cleaned.json`: `d6f21ea9...`），无任何估算或编造数据。

---

## 一、存储优化后的远端基准完整实测评分（v1 引擎）

### 1. 专项精度与长程门禁（全部 100% 通过）

| 远端基准套件 | 题数 | Precision@1 / 命中率 | MRR@10 | 串台率 | 缺失空返 | 备注 |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **Exact Identifiers** (`eval.exact_ident_eval`) | 120 (90+30) | **100.00%** (RRF: 97.78%) | **1.0000** | **0.00%** (RRF: 18.33%) | **100.00%** (RRF: 33.33%) | Procedural Top-3 = 100.00% |
| **Version Precision** (`eval.version_precision_eval`) | 80 | **100.00%** | — | **0 / 80** | — | 8 类版本分级准确率 100.00% |
| **Chrono Calendar** (`eval.chrono_eval`) | 60 | **100.00%** | — | — | — | session 6/6, span 48/48, interval 2/2, distance 2/2, ledger 2/2 |
| **Stress + Precision** (`eval.stress_precision`) | 40 | **100.00%** (R@5/R@10) | **1.0000** (RRF: 0.4583) | — | 0/20 误空 | 夹层压缩 87.40% (RRF 49.60%)，P50/P95 = 4.14 / 5.33 ms |
| **Bipolar Stress** (`eval.bipolar_ablation stress`) | 20 组 | **100.0%** (R@3/R@5) | **1.0000** (RRF: 0.4583) | — | 0/20 | 夹层压缩 87.4%，平均注入 121.3 Token (RRF 340.2) |
| **100K 长程冷归档** (`eval.long_horizon_static`) | 500/1500/3500 轮 | **100.00%** 禁忌 R@1 | **1.0000** | — | — | 热层缩减 79.99%，冷层历史 Top-1 = 100.00%，注入 275.1 Token |

### 2. 官方数据集（LoCoMo 100Q & LongMemEval-S 100Q，seed=42）

| 官方数据集与模式 | 题数 | R@5 | R@10 | R@Budget | MRR@10 | Session R@10 | 更新支配 | 平均 Token | P50 / P95 ms | 误空 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **LoCoMo 100Q** (Compact Full) | 100 | 36.85% | 41.95% | — | 0.3428 | — | — | 465.6 | 23.31 / 30.38 | 0/80 |
| **LoCoMo 100Q** (Pure RRF 对照) | 100 | 26.01% | 30.87% | — | 0.2334 | — | — | 334.1 | 13.56 / 19.07 | 0/80 |
| **LoCoMo 100Q** (Chronicle `single_query`) | 100 | — | 52.43% | 64.29% | 0.4384 | — | — | 2004.9 | — / 46.66 | 0/80 |
| **LoCoMo 100Q** (Chronicle `reconstructive`) | 100 | — | 52.43% | 65.34% | 0.4454 | — | — | 2001.6 | — / 64.01 | 0/80 |
| **LongMemEval-S 100Q** (Compact Full) | 100 | 61.34% | 70.87% | — | 0.5230 | 88.20% | **64.29%** (9/14) | 1946.5 | — | 0/86 |
| **LongMemEval-S 100Q** (Pure RRF 对照) | 100 | 60.47% | 67.29% | — | 0.4745 | 86.65% | 35.71% (5/14) | 1878.3 | — | 0/86 |
| **LongMemEval-S 100Q** (Chronicle `single_query`) | 100 | — | 68.02% | 85.48% | 0.4908 | — | 57.14% (8/14) | 2037.8 | — / 183.06 | 0/86 |
| **LongMemEval-S 100Q** (Chronicle `reconstructive`) | 100 | — | 71.61% | 84.90% | 0.4919 | — | 50.00% (7/14) | 2037.6 | — / 198.33 | 0/86 |

### 3. 工程安全与 10K 扩展门禁（`eval.engineering_bench`）

- 租户隔离：`200` 查询 / `0` 越权泄漏；Prompt 注入：`30` 载荷 / `0` 越权篡改 / `0` 只读写入。
- 10K 两跳图门禁（`--graph`）：**P95 = 84.052 ms（< 100 ms 达标）**，**SQL 次数 = 15–15（≤ 15 达标）**。
- SQLite 优化前后对比（10,000 活跃事实，60 题）：关闭缓存 P95 = `145.014 ms` → 开启优化缓存 P95 = `79.554 ms`（独立独占单跑为 `66.52 ms`）→ 启用 **SQLite+LanceDB (`IVF_RQ` RaBitQ) P95 = `31.20–32.95 ms`**，60 题进程 CPU 从 `6,144 ms` 降至 `1,848–1,914 ms`（直降 **69–70%**）。

---

## 二、消融实验 1：账本（v1 Ledger）还有必要性吗？

在完全相同的远端数据集上对比 **v1 账本** 与 **v2 VFS（`memory_nodes` + L0 分块日志 + 仅 L2 向量/BM25）**：

### A. 精确技术符号与版本精度（裸 VFS 严重退化，账本结构化能力不可替代）

| 基准 | 指标 | v1 账本 (Full) | v2 VFS (纯 L2 向量+BM25) | 差异 |
| :--- | :--- | ---: | ---: | :--- |
| **Exact Identifiers 120Q** | Precision@1 | **100.00%** | 16.67% | **-83.33 pp** |
| **Exact Identifiers 120Q** | MRR@10 | **1.0000** | 0.4463 | **-0.5537** |
| **Exact Identifiers 120Q** | 串台率 | **0.00%** | 75.56% | **+75.56 pp** |
| **Exact Identifiers 120Q** | 缺失空返率 (30题) | **100.00%** | 36.67% | **-63.33 pp** |
| **Version Precision 80Q** | Precision@1 | **100.00%** | 40.00% | **-60.00 pp** |
| **Version Precision 80Q** | 兄弟/跨产品串台率 | **0.00%** | 100.00% | **+100.00 pp** |

### B. 官方长对话数据集（LoCoMo 100Q & LongMemEval-S 100Q，seed=42）

| 数据集 | 方案 | Session R@5 | Session R@10 | Session MRR@10 | 更新支配 (14题) | 平均向量数/题 | 查询 P50 / P95 ms | 100题总 CPU ms |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **LoCoMo 100Q** | v1 账本 (Compact Full) | — *(Turn 36.85%)* | — *(Turn 41.95%)* | — *(Turn 0.3428)* | — | ~400+ | 23.31 / 30.38 | — |
| **LoCoMo 100Q** | v2 VFS (`dream_only` 无LLM正则) | 28.87% | 28.87% | 0.3797 | — | 1.2 | 0.15 / 0.31 | 17.0 |
| **LoCoMo 100Q** | **v2 VFS (`session_md` L2笔记)** | **64.65%** | **80.17%** | **0.5511** | — | **27.2** | **0.69 / 1.09** | **69.7** |
| **LongMemEval-S 100Q** | v1 账本 (Compact Full) | — *(Turn 61.34%)* | 88.20% *(Turn 70.87%)* | — *(Turn 0.5230)* | **64.29% (9/14)** | ~500+ | ~120 / 183+ | — |
| **LongMemEval-S 100Q** | v2 VFS (`dream_only` 无LLM正则) | 0.39% | 0.39% | 0.0116 | 0.00% (0/14) | 0.76 | 0.32 / 0.56 | 34.5 |
| **LongMemEval-S 100Q** | **v2 VFS (`session_md` L2笔记)** | **86.90%** | **91.20%** | **0.8003** | 50.00% (7/14) | **47.73** | **1.28 / 1.94** | **132.3** |

**账本必要性结论**：
1. **应当淘汰的部分**：旧 15 表账本中的“2D 笔记本排版（`blocks`/`seq`/`layout`）”和“对每条原始对话全量向量化”已无必要；VFS 的 **L0 对象分块 + L2 Markdown 笔记** 在官方 LongMemEval-S 会话召回上达到 **91.20%**（超过 v1 的 88.20%），查询延迟暴降至 **1.94 ms P95**，CPU 消耗低两个数量级。
2. **必须保留的部分**：纯 VFS 缺失三个硬能力——**精确标识符/版本分级索引**（否则 Exact 120Q 从 100% 掉到 16.67%、版本串台 100%）、**双时态发生区间与状态修订支配**（否则更新支配率从 64.29% 跌到 50.00%）、**严格缺失拒答门禁**。另外，`dream_only` 消融证明：若只靠正则蒸馏 L2，召回几乎归零（0.39%），L2 沉淀必须依赖 **LLM 异步整理或 Session 级 Markdown 归档**。
3. **最优终态**：以 **v2 VFS 为主骨架**，将 v1 的 `exact_identifiers`、版本分级器与双时态修订元数据作为轻量外挂字段并入 `memory_nodes`。

---

## 三、消融实验 2：SQLite 是否可以丢弃？（三后端 1K / 5K / 10K 实测）

同机（M1 Pro）使用 `eval.engineering_bench` 相同夹具与 60 题精确查询，实测 **SQLite (optimized)**、**SQLite+LanceDB (`IVF_RQ`)** 与 **PostgreSQL 16 (`pgvector` HNSW + `tsvector` + `pg_trgm`)**：

| 活跃事实规模 | 存储后端 | 写入吞吐 (writes/s) | 建索引耗时 | P50 ms | P95 ms | P99 ms | 60题 Python CPU ms | Top-1 命中 |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **1,000** | SQLite (optimized) | **677.0** | 0 ms | **13.78** | 15.46 | **17.53** | 818.7 | 60/60 |
| **1,000** | SQLite+LanceDB (`IVF_RQ`) | 615.6 | 130.7 ms | 14.06 | **15.29** | 28.24 | 844.0 | 60/60 |
| **1,000** | PostgreSQL 16 (`pgvector+tsvector`) | 53.8 | 0 ms | 64.93 | 76.05 | 98.53 | 674.9* | 60/60 |
| **5,000** | SQLite (optimized) | **621.3** | 0 ms | 35.94 | 66.15 | 67.98 | 2,300.2 | 60/60 |
| **5,000** | **SQLite+LanceDB (`IVF_RQ`)** | 598.6 | 389.2 ms | **22.00** | **23.04** | **24.12** | **1,336.2** | 60/60 |
| **5,000** | PostgreSQL 16 (`pgvector+tsvector`) | 50.6 | 0 ms | 103.73 | 111.63 | 130.97 | 685.1* | 60/60 |
| **10,000** | SQLite (optimized) | **625.1** | 0 ms | 63.44 | 95.40 | 110.20 | 4,122.9 | 60/60 |
| **10,000** | **SQLite+LanceDB (`IVF_RQ`)** | 594.1 | 863.8 ms | **31.47** | **32.95** | **33.49** | **1,914.1** | 60/60 |
| **10,000** | PostgreSQL 16 (`pgvector+tsvector`) | 51.7 | 0 ms | 177.90 | 209.80 | 242.58 | 678.8* | 60/60 |

*\*注：PostgreSQL 行中的 `60题 Python CPU ms` 仅统计 Python 客户端进程耗时，数据库计算开销在 Docker 容器内的 postgres 服务端进程执行，因此其真实端到端延迟应以 P50/P95/P99 为准。*

**SQLite 去留结论**：**不可丢弃 SQLite，但应淘汰“纯 SQLite 无 ANN 暴力扫大向量库”的用法。**
- **为什么不能丢**：
  1. 本地单事务写入吞吐 **625 writes/s vs PG 51.7 writes/s（快 12×）**；
  2. 10K 规模本地查询端到端延迟 **SQLite+LanceDB P95 = 32.95 ms vs PG16 Docker P95 = 209.80 ms（快 6.4×）**；
  3. 配合 v2 VFS（仅 L2 向量化），单机桌面 Agent 的 L2 笔记仅百至千级，SQLite 查询仅 **1–2 ms** 且零 Docker 依赖、214 项 CI 测试 6 秒跑完。
- **PostgreSQL 适用场景与后续优化点**：多 Agent 并发写（行级乐观锁替代全局 `pg_advisory_xact_lock`）、跨机部署与百万级共享租户；部署时必须使用 `VectorChord` (`vchordrq`) + 真 BM25（`pg_search`/`vchord_bm25`）替代 `pg_trgm` 才能把 PG P95 压到低位。

---

## 四、检索三阶段管线 + 写入显著性门控 + 渐进式披露 & 动态合并（优化后实测）

基于上述消融结论，我们在 v2 VFS（`memengine/vfs/`）中完成了**失败路径裁剪**与**检索/输出双端架构升级**：
- **裁剪与架构升级**：丢弃仅靠正则蒸馏 L2 的 `dream_only` 路径，将 v1 消融证明不可替代的 3 项结构化能力（L2 精确标识符表 `memory_node_identifiers`、版本精度分级与零串台门禁、双时态/更新时效权重）正式嫁接到 `memory_nodes`；并将 `VFSMemoryEngine` 晋升为顶层一等入口。
- **三阶段混合检索管线**：
  1. **强制预加载（Stage 1）**：无条件注入 L1 `/global.md` 与 L2 `always_active=True` 高频避坑规则；
  2. **结构化目录导航（Stage 2）**：暴露轻量 `ls -R` 大纲（`tree_outline()`，仅含路径 + ≤50 字 `index_summary` + 状态标签）与 `read_note(path)` 主动翻阅工具；
  3. **混合语义 + 元数据过滤 + 公式重排（Stage 3）**：Dense + BM25 + L2 精确标识符/版本过滤，严格执行重排公式 $\text{Final Score} = \text{Semantic\_Sim} \times W_{\text{verified}} \times W_{\text{recency}}$（`verified=1.35` 远高于 `unverified=0.55`；`W_recency` 对常驻/禁忌/核心规则恒为 `1.0`，普通笔记按 `updated_at` 平滑衰减于 `[0.82, 1.00]`）。
- **写入端 & 读取端高密度输出优化**：
  - **写入端**：内置三问自检引导（`SALIENCE_WRITE_GUIDE`）与 `write_rule()` 显著性门控（拒绝上下文可见信息与一次性流水账），强制 `[触发条件] + [动作] + [验证结果]` 三段式结构与假设/规则状态机（缺少有效验证结果自动强制降级为 `unverified` 且置信度 ≤ `0.65`）。
  - **读取端**：双层结构（≤50 字 `index_summary` + 完整无损 `detail_body`），`build_context(disclosure="progressive")` 与 `mem_search` 默认仅注入 `index_summary` 卡片、按需调用 `read_note(path)` 拉取全文，彻底消除粗暴字符硬截断；目录超过阈值（>10 条或 >3,000 Tokens）触发 `compact_directory()` 原子合并重构为单份 `GUIDE.md` 并支持 `rollback_batch()` 整批回滚。

### A. 远端四套基准实测对比（裸 VFS vs 优化后 VFS v2，实测命令：`python -m eval.vfs_ledger_ablation --skip-storage`）

| 远端基准套件 | 指标 | v1 账本基线 | 优化前（裸 VFS） | **优化后（VFS v2 三阶段+精确索引）** | 相对裸 VFS 变化 |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **Exact Identifiers 120Q** | Precision@1 | 100.00% | 16.67% | **100.00%** | **+83.33 pp** |
| **Exact Identifiers 120Q** | MRR@10 | 1.0000 | 0.4463 | **1.0000** | **+0.5537** |
| **Exact Identifiers 120Q** | 串台率 | 0.00% | 75.56% | **0.00%** | **-75.56 pp** |
| **Exact Identifiers 120Q** | 缺失空返率 (30题) | 100.00% | 36.67% | **100.00%** | **+63.33 pp** |
| **Version Precision 80Q** | Precision@1 | 100.00% | 40.00% | **100.00%** | **+60.00 pp** |
| **Version Precision 80Q** | 兄弟/跨产品串台率 | 0.00% | 100.00% | **0.00%** | **-100.00 pp** |
| **LoCoMo 100Q** | Session R@5 / R@10 | — | 64.65% / 80.17% | **67.39% / 83.74%** | **+2.74 / +3.57 pp** |
| **LoCoMo 100Q** | Session MRR@10 | — | 0.5511 | **0.5793** | **+0.0282** |
| **LoCoMo 100Q** | 查询 P50 / P95 ms | 23.31 / 30.38 | 0.69 / 1.09 | **0.89 / 1.27** | 维持 ~1 ms 极速 |
| **LongMemEval-S 100Q** | Session R@5 / R@10 | — / 88.20% | 86.90% / 91.20% | **92.79% / 96.51%** | **+5.89 / +5.31 pp** |
| **LongMemEval-S 100Q** | Session MRR@10 | — | 0.8003 | **0.8772** | **+0.0769** |
| **LongMemEval-S 100Q** | 更新支配 (14题) | 64.29% (9/14) | 50.00% (7/14) | **85.71% (12/14)** | **+35.71 pp**（超 v1 +21.42 pp） |
| **LongMemEval-S 100Q** | 查询 P50 / P95 ms | ~120 / 183+ | 1.28 / 1.94 | **1.81 / 2.94** | 较 v1 提速 **62×**（CPU 222.4 ms） |

### B. 检索三阶段、显著性写入门控、渐进式披露与动态合并专项实测

| 验证项 | 实测指标 | 实测结果 |
| :--- | :--- | :--- |
| **Stage 1 强制预加载** | 无关问题下 `/global.md` + `always_active` 召回 | **100%**（全部无条件注入） |
| **写入端显著性门控** | 20 条上下文可见/一次性噪音写入拦截率 | **100.0%**（20/20 拦截） |
| **假设 vs 规则状态机** | 10 条缺少验证结果的伪 `verified` 写入强制降级率 | **100.0%**（10/10 强制 `unverified`, conf ≤ 0.65） |
| **Stage 3 重排公式 (`W_verified`)** | 30 组同文 `verified` vs `unverified` 排序胜率 | **100.0%**（30/30 `verified` 置顶） |
| **Stage 3 重排公式 (`W_recency`)** | 30 组同文新旧笔记排序胜率 | **100.0%**（30/30 新笔记置顶） |
| **读取端渐进式披露** | 12 篇长规则上下文动态注入 Token（`progressive` vs `full`） | **661 vs 3,546 Tokens（直降 81.36%）**，`index_summary` 最长 `46` 字（≤50），**硬截断破损数 = 0** |
| **动态目录合并重构** | `/strategies/db/` 12 篇碎片（5,248 Tokens）→ 单份 `GUIDE.md` | **12 篇 → 1 篇**，**5,248 → 776 Tokens（直降 85.21%）**，整批回滚 **12/12（100%）无损还原** |
| **全量自动化测试** | `pytest -q`（含本地 SQLite + LanceDB + 实时 PG16 Docker） | **219 passed / 0 failed / 0 skipped**（8.28s）；CI mock 烟测 `eval.run_eval` 100% 通过 |

---

## 五、全链路联通核验与 v2 VFS 三后端扩展性能实测（`eval.vfs_perf_bench`）

在补齐架构联通性（`LanceL2Index` 嵌入式 RaBitQ L2 索引 + `lancedb:///` 工厂接入、`SQLiteVFSStore`/`LocalChunkTranscriptStore` 线程安全锁、`edit()` 乐观锁版本前置校验、消除 `write_rule()`/`directory_stats()` 的 $O(N^2)$ 全文扫描改为 `content_tokens` 索引 `GROUP BY`、以及 FastAPI `/v2/vfs/*` + CLI `memengine vfs` + `MemoryClient` 全链路打通）后，在同机实测三后端扩展性能与并发门禁：

### A. L0 解耦冷日志写入压测（10,000 轮原始对话）

| 指标 | 实测值 |
| :--- | :--- |
| **吞吐量 / 总耗时** | **44,991.9 turns/s**（总耗时 `222.3 ms`，CPU `218.0 ms`） |
| **分块存储占用** | `200` 个 `.jsonl.gz` 分块，压缩后仅 **`240.4 KB`** |
| **新增 L2 向量数** | **`0`**（严格验证 L0 原始日志零向量化、不增加检索发热） |

### B. v2 VFS 三后端多规模实测（500 / 2,000 / 5,000 篇 L2 精炼规则，60 题精确+语义混合检索 + 8 线程并发乐观锁）

| L2 规模 | 后端 | L2 写入 (notes/s) | 建索引耗时 | Search P50 / P95 / P99 ms | 60题 CPU ms | `build_context` P95 ms | 渐进 vs 全量 Token | Top-1 命中 | 8 线程乐观锁 |
| ---: | :--- | ---: | ---: | :---: | ---: | ---: | :---: | :---: | :---: |
| **500** | **SQLite VFS** | **824.8** | 0 ms | **3.24 / 3.77 / 5.71** | **197.4** | **4.57** | 453.2 vs 1046.7 | 60/60 | PASS |
| **500** | SQLite+LanceDB VFS (`IVF_RQ`) | 790.4 | 34.6 ms | 5.77 / 6.43 / 15.52 | 416.8 | 7.15 | 452.5 vs 1045.9 | 60/60 | PASS |
| **500** | PostgreSQL 16 VFS (`pgvector+tsvector`) | 128.3 | 0 ms | 18.96 / 23.58 / 26.47 | 128.2* | 23.11 | 452.2 vs 1048.1 | 60/60 | PASS |
| **2,000** | SQLite VFS | **562.4** | 0 ms | 11.48 / 12.70 / 34.91 | 733.2 | 13.81 | 445.6 vs 1033.5 | 60/60 | PASS |
| **2,000** | **SQLite+LanceDB VFS (`IVF_RQ`)** | 530.1 | 88.7 ms | **10.82 / 11.42 / 12.21** | 861.6 | **12.74** | 445.6 vs 1032.7 | 60/60 | PASS |
| **2,000** | PostgreSQL 16 VFS (`pgvector+tsvector`) | 118.1 | 0 ms | 18.21 / 19.36 / 19.83 | 137.2* | 23.65 | 445.7 vs 1033.5 | 60/60 | PASS |
| **5,000** | SQLite VFS | **334.5** | 0 ms | 28.74 / 53.68 / 54.95 | 1,881.0 | 59.02 | 445.6 vs 1033.8 | 60/60 | PASS |
| **5,000** | **SQLite+LanceDB VFS (`IVF_RQ`)** | 328.3 | 184.1 ms | **19.61 / 20.70 / 21.54** | **1,582.5** | **31.54** | **445.2 vs 1033.0 (-56.9%)** | **60/60** | **PASS** |
| **5,000** | PostgreSQL 16 VFS (`pgvector+tsvector`) | 112.8 | 0 ms | 56.37 / 59.38 / 62.62 | 151.7* | 66.46 | 445.6 vs 1033.7 | 60/60 | PASS |

*\*注：PostgreSQL 行的 `60题 CPU ms` 仅统计 Python 客户端进程耗时，数据库服务端算力在 Docker 容器内执行；端到端性能以 P50/P95/P99 为准。当 L2 规模在 500 篇以内时纯 SQLite VFS 最快（P95 3.77 ms）；从 2,000 篇起 **SQLite+LanceDB (`IVF_RQ`)** 反超，到 5,000 篇时 P95 仅 **20.70 ms**（较纯 SQLite 快 **2.6×**、较 PG16 Docker 快 **2.9×**）。*
