# Agent Memory Engine（认知与虚拟文件系统记忆引擎）

面向强模型 Agent 的高性能双代混合记忆引擎，融合 **v2 虚拟文件系统记忆引擎（`VFSMemoryEngine`）** 与 **v1 双极双时态认知账本（`CognitiveMemoryEngine`）**：

- **混合分层存储**：以关系数据库（SQLite WAL / PostgreSQL 16+）承载虚拟文件系统与状态机，L0 原始对话解耦至分块冷存储（本地 `.jsonl.gz` / S3 / OSS）且**绝不向量化**，仅对精炼 L2 Markdown 规则建立向量、BM25 与精确标识符索引（支持 LanceDB `IVF_RQ`、VectorChord `vchordrq`、DiskANN、ParadeDB `pg_search` 与 Qdrant）。
- **三阶段混合检索**：强制预加载（L1 `/global.md` + L2 `always_active`）→ 结构化目录导航（`ls -R` 大纲 + `read_note` 主动翻阅）→ 混合语义 + 元数据过滤 + 公式重排，结合精确符号/版本分级零串台门禁。
- **高密度读写输出**：写入端三问显著性门控与 `[触发条件] + [动作] + [验证结果]` 假设/规则状态机；读取端以 `<=50` 字 `index_summary` 渐进式披露替代粗暴字符硬截断，目录超阈值自动合并重构为 `GUIDE.md` 且支持整批原子回滚。

---

## 系统架构图

为保证全局清晰、逻辑闭环且排布舒展无重叠，系统架构拆分为两张互补视图：
- **图 1（全局分层存储与读写演化闭环）**：展示具体输入如何经治理层分流进入**短期记忆（L0/STM）**、**工作记忆（L1/WM）**与**长期记忆（L2+L3/LTM）**，三层如何晋升、同步索引与安全回滚，以及最终输出如何闭环回写。
- **图 2（查询调用、三阶段混合检索与渐进披露闭环）**：展示一条查询请求如何触发 Stage 1/2/3 多路召回、精确符号门禁、公式重排、证据链核验、渐进式披露输出，以及 Agent 按需工具钻取的完整回路。

### 图 1：全局分层架构、三层记忆存储与演化闭环

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 52, 'rankSpacing': 58, 'curve': 'basis', 'padding': 18}}}%%
flowchart TB
  subgraph Inputs["1. 外部输入（明确载荷）"]
    InDlg["对话流输入<br/>text · role · speaker · timestamp · session_id"]
    InRule["任务证据 / 规则输入<br/>path · trigger · action · verification · expected_version"]
    InQry["查询与上下文请求<br/>query · user_id / agent_id · token_budget · reference_time"]
  end

  Gateway["2. 统一接入与租户隔离层<br/>Python SDK · FastAPI /v2/vfs/* & /v1/* · CLI · OpenAI / MCP 12 Tools<br/>宿主强制绑定 user_id / agent_id / session_id 作用域"]

  subgraph WriteGov["3. 写入治理与并发事务层"]
    Gate["显著性三问门控 + 三段式状态机<br/>① 拦截界面可见与一次性流水账<br/>② 强制结构：[触发条件] + [动作] + [验证结果]<br/>③ 缺有效验证结果 → 自动锁为 unverified（conf ≤ 0.65）"]
    Tx["ACID 事务与乐观锁防线<br/>行级 expected_version 校验（冲突即返 HTTP 409）<br/>写前自动生成版本快照 · 成功返回新 version"]
    Gate -->|"通过显著性门控"| Tx
  end

  subgraph Storage["4. 分层混合存储层（SQLite WAL / LanceDB IVF_RQ / PostgreSQL 16+）"]
    L0[("短期 / 原始记忆 · L0 / STM<br/>Local JSONL.GZ / S3 分块冷存（v1 turns 缓冲）<br/>按 user/agent/session/date 分片 · 严禁向量化")]
    L1[("工作记忆 · L1 / WM<br/>/global.md 全局身份 + /working/*.md 草稿<br/>v1 任务检查点与临时证据 · 不进长期向量库")]
    L2[("长期记忆 · L2 + L3 / LTM（memory_nodes 表）<br/>/strategies/ · /taboos/ · /preferences/ + L3 /INDEX.md<br/>含 verification_status · always_active · content_tokens · source_ids")]
    L2Idx[("仅 L2 同步三路检索索引（L0 / L1 物理隔离）<br/>① 向量 ANN：LanceDB IVF_RQ / VectorChord vchordrq / Qdrant<br/>② 词法检索：SQLite FTS5 / ParadeDB pg_search / vchord_bm25<br/>③ 精确符号与 8 类版本表：memory_node_identifiers")]
    Evo["离线演化与安全回滚引擎<br/>• 异步做梦：L0 + L1 提炼晋升 L2<br/>• 动态合并：单目录超 10 条或 3,000T → GUIDE.md<br/>• 原子回滚：rollback_batch 一键撤销坏梦与错误合并"]

    L0 -.->|"source_transcript_ids 溯源 / 做梦提炼"| Evo
    L1 -->|"核验通过 / 任务显式 commit 晋升"| L2
    L2 -->|"仅活跃 L2 节点实时同步"| L2Idx
    L2 <-->|"目录膨胀重构 / 整批快照回滚"| Evo
  end

  Engine["5. 检索调度、渐进装配与证据核验中枢（调用细节见图 2）<br/>v2：Stage 1 预加载 + Stage 2 目录导航 + Stage 3 混合重排 + 渐进披露<br/>v1：双极渐近环 + Chronicle 双时态两跳图 + 单从句证据链核验"]

  subgraph Outputs["6. 具体输出（明确字段与状态）"]
    OutCtx["输出 A · 高密度上下文包（送入 LLM）<br/>system_prefix（≤360T L3 前缀缓存）<br/>+ Preload + ls -R 大纲 + ≤50字摘要卡"]
    OutAns["输出 B · 核验作答 / 工具回读<br/>已证结论+引文 / 部分作答+缺失子问 / 严格拒答<br/>或 read_note 完整正文 / L0 原始对话"]
    OutMeta["输出 C · 治理与状态元数据<br/>score_breakdown · source_transcript_ids<br/>compaction_alerts 目录告警 · 404/409 状态"]
  end

  InDlg --> Gateway
  InRule --> Gateway
  InQry --> Gateway

  Gateway -->|"① 原始对话直落冷存"| L0
  Gateway -->|"② 全局身份 / 任务草稿"| L1
  Gateway -->|"③ 规则与经验写入"| Gate
  Gateway -->|"④ 查询调用（scope / budget / time）"| Engine

  Tx -->|"原子落盘 + 记录快照"| L2

  L0 -.->|"按溯源 ID 懒加载原话"| Engine
  L1 -->|"全局身份 + 活跃任务态"| Engine
  L2 -->|"预加载规则 + 目录树 + 全文"| Engine
  L2Idx -->|"三路 Top-K 候选与精确匹配"| Engine

  Engine --> OutCtx
  Engine --> OutAns
  Engine --> OutMeta

  OutAns -.->|"执行验证产生新结果 → 回写升级 verified（完整闭环）"| Gateway

  classDef io fill:#eff6ff,stroke:#3b82f6,stroke-width:1.5px,color:#111827
  classDef ctrl fill:#f5f3ff,stroke:#7c3aed,stroke-width:1.5px,color:#111827
  classDef store fill:#ecfdf5,stroke:#10b981,stroke-width:1.5px,color:#111827
  classDef idx fill:#fff7ed,stroke:#f97316,stroke-width:1.5px,color:#111827
  classDef out fill:#f0fdf4,stroke:#16a34a,stroke-width:1.5px,color:#111827
  class InDlg,InRule,InQry io
  class Gateway,Gate,Tx,Engine ctrl
  class L0,L1,L2,Evo store
  class L2Idx idx
  class OutCtx,OutAns,OutMeta out
```

### 图 2：查询调用链路、三阶段混合检索与渐进式披露闭环

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 46, 'rankSpacing': 54, 'curve': 'basis', 'padding': 16}}}%%
flowchart TB
  QIn["查询输入载荷<br/>query · user_id / agent_id · path_prefix<br/>verified_only · reference_time · token_budget · disclosure"]

  Dispatch["检索调度入口<br/>VFSMemoryEngine.build_context() / search()<br/>兼容 v1 CognitiveMemoryEngine.recall() / answer()"]

  QIn --> Dispatch

  subgraph Stage1["Stage 1 · 强制预加载（无条件命中）"]
    S1["get_preloaded_nodes()<br/>无条件拉取 L1 /global.md<br/>+ L2 always_active=True 高频避坑规则<br/>不受语义相似度波动影响"]
  end

  subgraph Stage2["Stage 2 · 结构化目录导航"]
    S2["tree_outline()（轻量 ls -R 大纲）<br/>仅暴露 path + ≤50字 index_summary + 状态徽章<br/>不注入冗长正文 · 供强模型主动浏览选读"]
  end

  subgraph Stage3["Stage 3 · Hybrid RAG 多路并行检索（仅查活跃 L2）"]
    C1["向量 ANN 通道<br/>LanceDB IVF_RQ / VectorChord / Qdrant"]
    C2["词法 BM25 通道<br/>FTS5 / ParadeDB pg_search / vchord_bm25"]
    C3["精确符号与版本通道<br/>memory_node_identifiers（O(1) 查表）<br/>8 类版本分级 · 稳定版优先于 rc"]
    Fuse["RRF 融合 + 精确缺失门禁<br/>查无对应版本号 / 错误码 / 代码符号 → 严格空返 []"]
    Rerank["公式重排 + 时效与日历加权<br/>Final = Semantic_Sim × W_verified × W_recency<br/>verified=1.35 vs unverified=0.55 · 禁忌常驻 W_recency=1.0"]

    C1 --> Fuse
    C2 --> Fuse
    C3 --> Fuse
    Fuse --> Rerank
  end

  Dispatch -->|"① 必载规则"| S1
  Dispatch -->|"② 目录导航"| S2
  Dispatch -->|"③ 并行多路召回（每路 LIMIT fetch_k）"| C1
  Dispatch --> C2
  Dispatch --> C3

  Assemble["渐进式披露与预算装配器（杜绝中途硬截断）<br/>• progressive（默认）：仅注入 ≤50字 index_summary 卡片<br/>• full：整篇装入，超预算整篇退化为摘要卡，绝不截断因果前提<br/>• v1 核验路径：admissible_evidence + assess_claims 单从句核验"]

  S1 --> Assemble
  S2 --> Assemble
  Rerank --> Assemble

  OutBundle["上下文输出包（注入 LLM Prompt）<br/>• system_prefix：L3 /INDEX.md（≤360T 字节稳定缓存）<br/>• dynamic_memory：Stage 1 + Stage 2 + Stage 3 渐进卡片<br/>• score_breakdown · source_transcript_ids · compaction_alerts"]

  OutVerified["核验作答输出（v1 answer / reflect）<br/>• 证据充分：已证结论 + evidence_ids 引文<br/>• 部分有据：partial=True 仅答已证部分 + 列出缺失子问<br/>• 证据不足：abstained=True 严格拒答不编造"]

  Assemble --> OutBundle
  Assemble --> OutVerified

  AgentLoop["Agent 决策与按需工具调用闭环<br/>① 命中高相关摘要卡 → read_note(path) 拉取完整 detail_body<br/>② 需复核原始语境 → mem_read_transcripts(ids) 懒加载 L0 原话<br/>③ 任务验证规则有效 → mem_write_rule() 回写升级 verified"]

  OutBundle --> AgentLoop
  AgentLoop -.->|"按需补查 / 完整正文与原文钻取 / 新验证结果回流"| Dispatch

  classDef inNode fill:#eff6ff,stroke:#3b82f6,stroke-width:1.5px,color:#111827
  classDef s1 fill:#fef2f2,stroke:#ef4444,stroke-width:1.5px,color:#111827
  classDef s2 fill:#f0f9ff,stroke:#0284c7,stroke-width:1.5px,color:#111827
  classDef s3 fill:#fff7ed,stroke:#f97316,stroke-width:1.5px,color:#111827
  classDef core fill:#f5f3ff,stroke:#7c3aed,stroke-width:1.5px,color:#111827
  classDef out fill:#f0fdf4,stroke:#16a34a,stroke-width:1.5px,color:#111827
  class QIn inNode
  class Dispatch,Assemble,AgentLoop core
  class S1 s1
  class S2 s2
  class C1,C2,C3,Fuse,Rerank s3
  class OutBundle,OutVerified out
```

---

## 核心原理概述

### 1. 分层混合存储与防发热设计（L0 / L1 / L2 / L3）
- **为什么用关系库模拟虚拟文件系统**：强 LLM 对“目录树 + Markdown 文本”有天然读写亲和力，但纯文件难以处理多 Agent 并发写冲突与错误回滚。核心表 `memory_nodes`（`node_id`、`parent_path`、`path`、`title`、`summary`、`content`、`layer`、`confidence_score`、`verification_status`、`salience_tag`、`always_active`、`content_tokens`、`source_transcript_ids`、`version`）在保留目录树体验的同时，提供 **ACID 事务、行级乐观锁（冲突返回 `OptimisticLockError` / HTTP 409）、版本快照、整批回滚（`rollback_batch`）与严格租户隔离**。
- **L0 解耦冷存与仅 L2 向量化**：原始对话按 `user/agent/session/date` 分块压缩落盘（`.jsonl.gz` 或 S3/OSS），**永不进入向量索引**，彻底杜绝海量过时对话噪声淹没规则或引发大库检索 CPU 发烫。
- **多后端能力分级自动适配**：
  - **SQLite WAL（默认，≤500 条 L2 最轻量）**：零依赖、毫秒级检索与 CI 极速运行。
  - **SQLite + LanceDB（`lancedb:///path`，本地 2K–50K 推荐）**：SQLite 负责 ACID 元数据与精确表，Rust + Apache Arrow/Lance 接管磁盘零拷贝向量与 **RaBitQ (`IVF_RQ`)** 量化 ANN，免外部服务即可压降 70% CPU。
  - **PostgreSQL 16+（`postgresql://`，多 Agent 分布式并发）**：启动自动探测升级——向量层 `vchordrq` (VectorChord) → `diskann` (pgvectorscale) → `hnsw` (pgvector) → `sql-array`；词法层 `pg_search` (ParadeDB BM25) → `vchord_bm25` → `tsvector`；亦支持外挂 `QdrantL2Index`。

### 2. 三阶段混合检索管线与精确符号门禁
传统“Query → Top-K 向量相似度”容易漏掉字面不相似但逻辑强相关的规则，也常在版本号/错误码上串台。本引擎采用三阶段管线：
1. **Stage 1 强制预加载**：无论查询主题是什么，无条件加载 L1 `/global.md` 与 L2 中 `always_active=True` 的高频避坑规则。
2. **Stage 2 结构化目录导航（Agentic Tree Search）**：通过 `tree_outline()` 向 Agent 暴露类似 `ls -R` 的轻量目录大纲（仅路径 + ≤50 字 `index_summary` + 状态标签），支持模型主动调用 `read_note(path)` 按需翻阅完整笔记。
3. **Stage 3 混合语义 + 元数据过滤 + 公式重排**：融合 Dense 向量 + BM25/FTS5 + L2 精确标识符表（`memory_node_identifiers`，写入建表、查询 $O(1)$ 命中）与 8 类版本精度分级（精确/包含/粗粒度/兄弟/预发布/区间/跨产品），按以下公式重排：

$$\text{Final Score} = \text{Semantic\_Sim} \times W_{\text{verified}} \times W_{\text{recency}}$$

- **已验证规则压倒未验证假设**：`verified` 权重 `1.35`，`unverified` 权重 `0.55`，避免将单次猜测当作真理。
- **时效与常驻分离**：`always_active` / `taboo` / `core_rule` 的 $W_{\text{recency}} \equiv 1.0$ 永不衰减；普通笔记按 `updated_at` 平滑衰减于 `[0.82, 1.00]`，既确保新更新稳定压过旧事实，又不让近期弱噪声掩盖数月前的高相关记忆。
- **严格缺失空返**：针对版本号、错误码、代码符号、文件路径执行精确核验，不存在时返回空集，杜绝版本串台。

### 3. 写入显著性门控、渐进式披露与动态合并
- **写入端三问引导与状态机**：提供轻量自检提示 `SALIENCE_WRITE_GUIDE`（不死板限制“只记失败”）：① 当前上下文能否直接看到？② 是一次性状态还是可复用规律？③ 未来新会话没有它会不会踩坑？`write_rule()` 强制采用 `[触发条件] + [动作] + [验证结果]` 结构，缺少有效验证结果时自动降级为 `unverified` 且 `confidence_score <= 0.65`。
- **读取端渐进式披露替代硬截断**：每条 L2 节点维护 `index_summary`（≤50 字一句话结论）与无损 `detail_body`。`build_context(disclosure="progressive")` 与 `mem_search` 默认仅返回摘要卡片，由模型按需 `read_note(path)` 拉取全文；即使在 `disclosure="full"` 触及预算，也只整篇退化为 `index_summary`，**绝不中途截断规则的因果前提**。
- **动态目录合并重构（Compaction）**：基于持久化 `content_tokens` 以单条索引 SQL `GROUP BY` 监测目录膨胀；单目录超 `10` 条或 `3,000` Tokens 时触发 `compact_directory()`，将碎片重构为结构化单份 `GUIDE.md`（保留 `Verified Rules` / `Unverified Hypotheses` 分组与全量 L0 溯源 ID），并挂接 `dream_batch_id` 支持整批原子回滚。

### 4. 单次事实召回与证据链核验（v1 认知账本兼容）
- **双极渐近环**：只提过一次的孤立事实、禁忌与显式覆写被推向高权外极（`r = 1.0`），与反复巩固的高频核心事实（`r -> 0`）同享高权重，仅压低中间低价值杂讯带（实测夹层压缩 `87.4%`）。
- **严格证据链核验**：`answer()` / `reflect()` 经 `admissible_evidence()` 与 `assess_claims()` 校验单从句直接支撑、极性、版本与时间一致性；完全无证据直接拒答（`abstained=True`），复合子问仅部分有据时只输出有引文的已证结论并明确列出缺失子问（`partial=True`）。

完整设计公式与历史规格见 [设计说明](docs/DESIGN.md) 与 [规格文档](docs/SPEC.md)。

---

## 全程记忆评测实测结果（增加 MemBench 权重）

全部成绩均于 **2026-11-23** 在 **Apple M1 Pro（8 核 / 16 GB，Python 3.12，LanceDB 0.41.0，Docker `pgvector/pgvector:pg16`）** 实时运行官方数据集产出（`locomo10.json` 全量 1,986 题、`longmemeval_s_cleaned.json` 全量 500 题、官方 `import-myself/Membench` `MemData` 4 大类扩样 800 / 400 / 99 题），**无估算与编造**。完整细项见 [全程记忆评测完整实测报告](eval/results/full_memory_benchmark_report.md) 与 [存储优化与 VFS 消融实测报告](eval/results/ablation_storage_and_vfs_report.md)。

### 1. MemBench 加权综合总览（MemBench 40% + LongMemEval-S 25% + LoCoMo 20% + 技术专项 15%）

| 引擎与评测模式 | MemBench 800 题<br>*(权重 40%)* | LongMemEval-S 全量 500 题<br>*(权重 25%)* | LoCoMo 全量 1,986 题<br>*(权重 20%)* | 技术精确/版本/日历/压力<br>*(权重 15%)* | **加权综合分** | 查询 P95 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **v2 VFS (`VFSMemoryEngine`)** | 67.28% *(Step R@10)* | **94.98%** *(Session R@10)* | **88.89%** *(Session R@10)* | **100.00%** *(Exact+Ver)* | **83.43%** | **1.83–3.18 ms** |
| **v1 Chronicle (`reconstructive`, 2048 Budget)** | **76.56%** *(Step R@10)* | 84.90% *(R@Budget)* | 77.40% *(R@Budget)* | **100.00%** *(4项全满)* | **82.33%** | 69.02–204.88 ms |
| **v1 标准 Top-10 (`CognitiveMemoryEngine`)** | **76.56%** *(Step R@10)* | 70.87% *(Turn R@10)* | 50.06% *(Turn R@10)* | **100.00%** *(4项全满)* | **73.35%** | 26.52–123.94 ms |

### 2. 核心高权重基准：MemBench 全类别实测（800 / 400 / 99 题）

| MemBench 队列 / 类别 | 题数 | v1 Full (`bipolar`)<br>R@5 / R@10 / MRR | v1 Pure RRF 基线<br>R@5 / R@10 / MRR | v2 VFS (`VFSMemoryEngine`)<br>R@5 / R@10 / MRR | 关键亮点 |
| :--- | ---: | :---: | :---: | :---: | :--- |
| **MemBench 800 总计（4×200）** | **800** | **60.19% / 76.56% / 0.6698** | 56.70% / 76.56% / 0.6568 | 51.62% / 67.28% / 0.6085 | v1 R@5 +3.49 pp、0 误空；v2 P95 **2.23 ms** / 省 Token **-37.2%** |
| └ `participation_factual` | 200 | 67.67% / 76.11% / 0.6717 | 68.74% / 76.11% / 0.6824 | **68.59% / 77.91% / 0.6782** | v2 R@10 反超 v1（+1.80 pp），延迟低 9×（2.96 ms） |
| └ `observation_factual` | 200 | **74.80% / 87.84% / 0.8367** | 74.41% / 87.84% / 0.8349 | 71.49% / **87.71%** / 0.8242 | v1/v2 事实召回双双逼近 88%（v2 P95 1.89 ms） |
| └ `participation_reflective` | 200 | **49.98% / 74.47% / 0.5348** | 44.72% / 74.47% / 0.5158 | 37.98% / 63.43% / 0.4337 | v1 R@5 领先 RRF +5.26 pp；v2 渐进省 Token -52.3% |
| └ `observation_reflective` | 200 | **48.31% / 67.83% / 0.6358** | 38.95% / 67.83% / 0.5942 | 28.42% / 40.08% / 0.4980 | v1 R@5 领先 RRF **+9.36 pp**（145 题由观察蒸馏独占命中） |
| **MemBench 400 标准静态** | 400 | **58.38% / 77.80% / 0.6589** | 55.54% / 77.80% / 0.6540 | — | 平均注入 549.9 Tokens |
| **MemBench 99 冻结门禁** | 99 | **60.56% / 80.08% / 0.6999** | 57.64% / 80.08% / 0.6943 | — | 冻结回归门禁 **PASS**（MRR ≥ 0.6943） |

### 3. 全量长对话、技术精度与压缩专项实测

| 评测套件 | 规模 | 实测成绩 | 说明 |
| :--- | :--- | :--- | :--- |
| **LongMemEval-S 全量** | 500 题（470 可答） | v2 VFS **Session R@10 = 94.98%，MRR = 0.8993，更新支配 = 67.14% (47/70)，P95 = 3.18 ms** | v1 Turn R@10 = 70.87%（Chronicle R@Budget = 84.90%，更新支配 60.00%） |
| **LoCoMo 全量** | 1,986 题（1,540 可答） | v2 VFS **Session R@10 = 88.89%，MRR = 0.6864，P95 = 1.83 ms** | v1 Turn R@10 = 50.06%（Chronicle `reconstructive` R@10 = 62.37%，R@Budget = 77.40%） |
| **Exact Identifiers** | 120 题（90+30） | v1 & v2 双双 **P@1 = 100.00%，MRR = 1.0000，串台 = 0.00%，缺失空返 = 100.00%** | Pure RRF 串台 18.33%、缺失空返仅 33.33% |
| **Version Precision** | 80 题（8 类） | v1 & v2 双双 **P@1 = 100.00%，版本串台 = 0 / 80** | 覆盖精确/包含/粗粒度/兄弟/预发布/区间/跨产品 |
| **Chrono Calendar** | 60 题 | **准确率 = 100.00%** | 日历解析、跨度与区间重叠全过 |
| **Stress + Bipolar 抗压** | 40 题 / 20 组 | **R@5/R@10 = 100.00%，MRR = 1.0000，夹层压缩 = 87.40%** | 平均仅注入 121.3 Tokens（RRF 为 340.2 Tokens） |
| **100K 长程冷归档** | 3,500 轮（101K Token） | **单次禁忌 R@1 = 100%，冷唤醒 = 100%，冷 Top-1 = 100%，热层缩减 = 79.99%** | 仅注入 275.1 Tokens |
| **v2 读写密度优化专项** | 渐进披露 / 动态合并 | 显著性拦截 **100%**；渐进披露 **3,546 → 661 Tokens (-81.36%)，0 硬截断** | 动态合并 **12 篇 → 1 篇 `GUIDE.md`（-85.21% Tokens）**，回滚 12/12 无损 |

### 4. 存储扩展、低发热压测与安全门禁（SQLite / LanceDB `IVF_RQ` / Docker PostgreSQL 16）

| 压测维度 | 后端 / 规模 | 吞吐 / 延迟 | 关键门禁 |
| :--- | :--- | :--- | :--- |
| **v2 L0 冷日志写入** | 10,000 轮 `.jsonl.gz` | **44,991.9 turns/s**（222.3 ms），仅占 **240.4 KB** | 新增 L2 向量 = **0**（零检索发热） |
| **v2 VFS 500 L2 规则** | SQLite VFS | 写入 **824.8 notes/s**，Search P95 = **3.77 ms** | Top-1 **60/60**，8 线程乐观锁 **PASS** |
| **v2 VFS 5,000 L2 规则** | **SQLite+LanceDB (`IVF_RQ`)** | Search P95 = **20.70 ms**，CPU 最省（渐进 Token **-56.9%**） | 较纯 SQLite（53.68 ms）快 **2.6×**、较 PG16 Docker（59.38 ms）快 **2.9×** |
| **v1 10K 扩展与图门禁** | SQLite / LanceDB `IVF_RQ` | 10K LanceDB P95 = **32.95 ms（CPU -70%）**；`--graph` P95 = **94.04 ms** | SQL 严格 **15–15**；租户隔离 **0/200 泄漏**；注入防护 **0/30 篡改** |
| **全量自动化测试** | SQLite + LanceDB + 实时 PG16 Docker | `pytest -q`：**219 passed / 0 failed / 0 skipped**（8.28s） | 冻结回归门禁 `08-regression-guard` 与 CI 全绿 |

---

## 快速开始

### 安装

```bash
pip install -e ".[dev]"        # 基础版（Python 3.10+、NumPy、SQLite WAL/FTS5）
pip install -e ".[lance]"      # 嵌入式 LanceDB + PyArrow（RaBitQ IVF_RQ 零拷贝 ANN）
pip install -e ".[postgres]"   # PostgreSQL 16+ 驱动与连接池（VectorChord / DiskANN / BM25）
pip install -e ".[server]"     # FastAPI + Uvicorn HTTP 服务
pip install -e ".[all]"        # 以上全部后端与服务组件
```

### v2 虚拟文件系统记忆引擎（推荐）

```python
from memengine import VFSMemoryEngine

# 支持 SQLite（默认）、lancedb:///path（嵌入式 RaBitQ ANN）、postgresql://...
vfs = VFSMemoryEngine("lancedb:///tmp/agent_vfs")
scope = {"user_id": "alice", "agent_id": "coder"}

# 1. L0 原始对话冷存（零向量化）
tr = vfs.append_transcript("压测发现全量向量化 L0 会导致 CPU 飙升。", session_id="s1", **scope)

# 2. Stage 1 全局身份 + 显著性门控结构化规则写入
vfs.write("/global.md", title="Global Policy", content="变更存储架构前必须跑真实压测。", layer="L1", verification_status="verified", **scope)
vfs.write_rule(
    "/strategies/db/l2_only.md",
    title="仅对 L2 精炼规则建向量索引",
    trigger_condition="设计大规模 Agent 记忆存储管线",
    action="L0 原始对话按块落冷存，仅对 L2 Markdown 规则建立向量与精确符号索引",
    verification_result="10K 压测 CPU 下降 70%，L0 写入 44,991 turns/s",
    always_active=True,
    source_transcript_ids=[tr.transcript_id],
    **scope,
)

# 3. 三阶段检索 + 渐进式披露（默认仅注入 <=50 字 index_summary，按需 read_note）
ctx = vfs.build_context("如何防止大库向量检索发烫？", disclosure="progressive", **scope)
full_note = vfs.read_note("/strategies/db/l2_only.md", **scope)

# 4. 导出 12 个 OpenAI / MCP 风格文件工具（mem_tree / read_note / mem_write_rule / mem_search / mem_compact_dir / mem_rollback ...）
tools = vfs.as_tools(**scope)
mcp_schema = tools.mcp_definitions()
```

### v1 认知账本引擎（完全兼容）

```python
from memengine import CognitiveMemoryEngine

engine = CognitiveMemoryEngine(db_path="memory.db", categories=["基础设施"])
scope = {"user_id": "alice", "agent_id": "assistant"}
written = engine.add("老周对苯甲酸钠过敏。", **scope, session_id="chat-1", role="user")
evidence = engine.retrieve("老周对哪种防腐剂过敏？", **scope, top_k=10, token_budget=2048)
print(engine.answer("老周对哪种防腐剂过敏？", **scope, retrieved=evidence)["response"])
```

---

## CLI 与 HTTP 服务（`/v2/vfs/*` & `/v1/*`）

通过 `--db-url` 或环境变量 `MEMENGINE_DB_URL` / `MEMENGINE_VFS_DB_URL` 切换 SQLite、LanceDB（`lancedb:///dir`）或 PostgreSQL（`postgresql://...`）：

```bash
# v2 VFS CLI
python3 -m memengine.cli --db-url lancedb:///tmp/vfs vfs write-rule /strategies/wal.md \
  --title "SQLite WAL" --trigger "本地并发读写" --action "开启 WAL 与 busy_timeout=5000" --verification "8 线程压测通过"
python3 -m memengine.cli --db-url lancedb:///tmp/vfs vfs tree
python3 -m memengine.cli --db-url lancedb:///tmp/vfs vfs context "本地并发读写策略" --disclosure progressive

# v1 CLI 与 HTTP 服务
python3 -m memengine.cli --db-url ./memory.db retain "拔牙后一周内禁饮咖啡。" --ttl-seconds 604800
python3 -m memengine.cli --db-url ./memory.db recall "最近喝咖啡有什么限制？" --mode reconstructive
MEMENGINE_DB_URL=./memory.db MEMENGINE_API_KEY=your-key memengine serve
```

| 路由组 | 端点 | 功能说明 |
| :--- | :--- | :--- |
| **v2 VFS** | `GET /v2/vfs/health`、`POST /v2/vfs/transcripts` | 健康检查与 L0 分块日志写入 |
| **v2 VFS** | `POST /v2/vfs/write`、`/write-rule` | Markdown 节点与显著性门控规则写入（乐观锁冲突返回 **HTTP 409**） |
| **v2 VFS** | `POST /v2/vfs/tree`、`/read-note`、`/search`、`/context` | Stage 2 目录大纲、`read_note` 渐进拉取（缺失返回 **HTTP 404**）、Stage 3 混合检索与上下文装配 |
| **v2 VFS** | `POST /v2/vfs/compact`、`/rollback`、`GET/POST /v2/vfs/tools/*` | 动态目录合并重构、整批回滚与 12 个 Claude 风格 VFS 工具调度 |
| **v1 Ledger** | `GET /health`、`POST /v1/retain`、`/recall`、`/answer`、`/reflect`、`/consolidate` | 双时态写入、重构召回、证据核验作答与离线蒸馏 |
| **v1 Ledger** | `/v1/tasks/*`、`/v1/drilldown`、`/v1/history/*`、`/v1/tools/*` | 任务工作区检查点、原话钻取、版本链与 v1 工具协议 |

标准库客户端 `MemoryClient(url, api_key=...)` 同时提供 v1 方法与 `vfs_health` / `vfs_write_rule` / `vfs_read_note` / `vfs_tree` / `vfs_search` / `vfs_context` / `vfs_compact`。Docker 部署默认使用 `tensorchord/vchord-suite:pg16-latest`（可通过 `MEMENGINE_PG_IMAGE` 切换为 `paradedb/paradedb:latest` 或 `pgvector/pgvector:pg16`）。

---

## 模型与环境变量配置

默认离线运行确定性 `MockLLMClient` 与本地 `HashedTfidfEmbedder`，亦可通过环境变量启用在线 LLM / 向量 / 精排服务或本地 ONNX / Sentence-Transformers 神经后端：

| 环境变量 | 含义 |
| :--- | :--- |
| `MEMENGINE_DB_URL` / `MEMENGINE_VFS_DB_URL` | 存储连接地址（SQLite、`lancedb:///...`、`postgresql://...`）。 |
| `MEMENGINE_API_KEY` | HTTP 服务 Bearer 鉴权密钥。 |
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | OpenAI 兼容 Chat 接口配置。 |
| `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL` / `EMBEDDING_API_KEY` | 远程 `/embeddings` 向量编码服务。 |
| `RERANK_BASE_URL` / `RERANK_MODEL` / `RERANK_API_KEY` | 远程 `/rerank` 精排服务。 |
| `LOCAL_EMBED_MODEL` / `LOCAL_RERANK_MODEL` | 本机 ONNX（`pip install -e ".[onnx]"`）或 Sentence-Transformers（`".[neural]"`）模型配置。 |

---

## 评测与复现命令

```bash
# 1. 全量单元测试（设置 TEST_POSTGRES_DSN 自动纳入实时 PostgreSQL 16 合约测试）
python3 -m pytest -q

# 2. MemBench 加权全程评测（MemBench 800 + 全量 LoCoMo 1,986 + 全量 LongMemEval-S 500 + Exact/Version）
python3 -m eval.full_memory_benchmark --membench-per-cat 200 --workers 6 --out /tmp/full-memory-bench.json

# 3. v2 VFS 三后端扩展性能与 8 线程并发乐观锁压测
python3 -m eval.vfs_perf_bench --scales 500 2000 5000 --out /tmp/vfs-perf.json

# 4. v1 vs v2 VFS 消融与三后端 1K/5K/10K 对比
python3 -m eval.vfs_ledger_ablation --out /tmp/vfs-ablation.json

# 5. 官方 22 项零 API 静态矩阵（含冻结回归门禁）
bash scripts/run_static_matrix.sh /tmp/static-results /path/to/locomo10.json \
  /path/to/longmemeval_s_cleaned.json /path/to/MemData /path/to/03-membench-ref.json
```

历史基准报告存档：[全程记忆评测报告](eval/results/full_memory_benchmark_report.md) · [存储与 VFS 消融报告](eval/results/ablation_storage_and_vfs_report.md) · [Chronicle 实跑报告](eval/results/chronicle_report.md) · [静态矩阵报告](eval/results/static_benchmark_report.md)
