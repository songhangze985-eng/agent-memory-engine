# 检索失分诊断

冻结基线使用 Python 3.12、SQLite、确定性 Hashed TF-IDF 及本地词法精排。LoCoMo 与 LongMemEval-S 校验官方数据快照 SHA-256。MemBench 使用固定的缓存样本、种子和去重规则。`frozen` 重放锁定的评测入口，`chronicle` 另行测量已经存在的说话人、日历和图检索路径。

## 统计口径

- Gold-in-Store 按题计算金标原话落盘覆盖和有事实溯源的覆盖。事实覆盖包括历史版本，因而与查询时的可召回性分开。R@10 和 MRR@10 采用实际装填后的事实顺序，预算覆盖另计注入的原话证据。
- 候选漏斗另跑一次加深观测，每路深度上限为 100，在最终一次 RRF 处取切片计算 R@100、R@50、R@20。A 是候选 @50 和最终前十都没有的金标份额。B 是候选 @50 有但最终前十没有的份额，包含精排、裁剪和装填的影响。加深观测会改变候选组成，表中的装填损失因此只提供定位线索。
- 单路消融统一以 @50 为口径。边际贡献用同一候选集的影子 RRF 比较，分别排除每一路。精确标识符、原话锚点、独立精排及门禁保留在主漏斗，因此影子融合与主漏斗的值可能不同。正负边际只说明这次融合中该路的贡献。
- Adversarial 和独列的 abstention 无正向证据分母，单报空返。单词原形共现和零词法共现是语义漏召的代理量。未逐题标注同义/蕴含，不能把所有零共现都归为同义改写。
- 漏斗 P95 在同一台主机的多进程任务中采集。10K、SQL 和前缀红线由隔离的工程门禁验证。

## 六项源码核查

| 核查项 | 冻结入口及已有能力 | 对齐缺口 |
| --- | --- | --- |
| 1. Retain、说话人与时钟 | `eval/locomo/run.py:13` 解析会话日期和实际说话人。`eval/longmemeval/run.py:15` 解析 haystack 与问题日期，源数据只提供对话角色。`eval/evidence_ablation.py:193` 冻结入库调用丢弃 `speaker` 和 `mention_time`。从 `memengine/storage/chronicle.py:113` 可写入发生区间及规范实体。 | `memengine/linguistics.py:47` 的确定性映射覆盖 I/my/me 等单数第一人称，未处理 we/our/us 的群体语义。LoCoMo 问题没有官方 `question_date` 字段，当前以最后一轮为回退锚点。LongMemEval-S 传入问题日期后，冻结路径只用它作为检索时钟，不启用绝对时间窗口计划。 |
| 2. 四路召回 | `memengine/retrieval/operators.py:11` 依次执行稠密或哈希向量、全文、实体倒排和条件时序查询。`memengine/storage/sqlite.py:793` 用 FTS5 BM25 取词法候选。`memengine/storage/postgres.py:279` 用 tsvector/pg_trgm。 | Agent Memory Engine 的四路查询在同一存储连接上串行；冻结路径只做实体倒排，发生区间索引和关系图通过显式计划启用。Hindsight 的 TEMPR 并行生成候选，再做跨编码器精排与预算装填。 |
| 3. 2 跳扩散与径向门禁 | `memengine/storage/chronicle.py:208` 持久化实体、语义、日历、因果边，`memengine/retrieval/activation.py:26` 执行两个批量前沿查询。`memengine/retrieval/operators.py:44` 重新搜索新实体和反馈词。 | 只按源点度数衰减，活跃枢纽的目标度数没有参与。反馈词查和源图扩散一起抬升大量邻居，`memengine/engine.py:690` 的激活加分缺少路径证据约束。`memengine/gate.py:40` 的夹层裁剪只保护精确标识符，未识别证据桥。 |
| 4. 精排、截断和装填 | `eval/paired.py:91` 固定 Top-10，`memengine/engine.py:544` 对显式 Token 模式提供更宽候选，`memengine/assembler.py:122` 合并稳定前缀、动态事实、任务上下文和原话。 | 冻结入口的常规路径还会使用 `memengine/engine.py:726` 的 RRF 证据下限，剔除部分高分但不在 RRF 装填范围内的候选。预算模式以分数顺序取事实，多跳桥和原话尚未形成独立装填约束。 |
| 5. 同义及语义模型 | `memengine/reranker.py:118` 的确定性精排使用字面覆盖、向量分数与 RRF，`memengine/retrieval/operators.py:18` 只在宽计划中使用词干查询扩展。`memengine/neural_onnx.py:29` 已提供本地 FastEmbed 接口。 | 加深哈希候选改善深度，无法保证召回同义语句。词干和零词法共现计数见漏斗，离线神经模型需用实际权重后另跑同口径比较。 |
| 6. 核验与拒答 | `memengine/reflection.py:25` 最多三轮补证并经 `admissible_evidence` 过滤，`memengine/verification.py:24` 做逐断言验证。 | 入口依赖一个子句里的连续词串，日期与版本检查发生在字面命中之后。同义的 bought/purchased、单位换算 1 km/1000 m、等价的自然语言/ISO 日期均被拒绝。时态、主客体变化及矛盾值仍应分别拒绝。 |

Hindsight 官方工程说明详见 [How Hindsight works](https://docs.hindsight.vectorize.io/enterprise/how-hindsight-works/) 与 [Retrieval strategies](https://github.com/vectorize-io/hindsight/blob/main/skills/hindsight-docs/references/developer/retrieval.md)。这里采用证据检索指标，与 Hindsight 发布的端到端回答准确率是两个口径。

## 冻结 LoCoMo 与 MemBench 的失血定位

- LoCoMo 原话及事实溯源均覆盖 99.61% 的金标。A 类占 37.65%，B 类占 12.30%。Multi-hop 的 A 类达到 64.16%，B 类为 10.78%。优先改善多跳候选的组成，再优化前十取舍。
- LoCoMo 的词法路独立 R@50 为 67.78%，高于哈希语义路的 56.90%。移除哈希语义路后，影子融合改善 2.24 个百分点。MemBench 400 的词法路独立 R@50 为 58.41%，哈希语义路为 92.62%，移除它损失 35.08 个百分点。查询结构需要参与通道权重决策。
- MemBench 99 和 400 的 Gold-in-Store 均为 100%。MemBench 400 的 A 类占 6.39%，B 类占 15.80%，优先精排和证据装填。400 题共识别 109 个漏召原话实例，其中 39 个仅有词干共现，65 个无词法共现。LoCoMo 对应为 1102、5、12；大量漏召仍有共享字词，关键是匹配具体实体及动作。
- 上述三个队列的 R@10、MRR@10、Token 与原锁定结果吻合。基线启用 PostgreSQL 后 160 项测试通过，证据和冻结回归门禁退出码均为 0。

## 已部署的 LoCoMo 日历和图路径

官方文件中的 5,882 条 LoCoMo 对话均解析到说话人和会话时间。432 条原话可解析出具备起止边界的发生区间。1,986 个问题均使用最后一轮会话日历作为问题日期的回退值。

现有宽候选配合说话人、日历和重构使 LoCoMo 候选 R@50 从 61.51% 升至 75.35%，Multi-hop 从 34.98% 升至 52.32%。但同样使用日历写入和 2,048 Token 预算时，已有重构比已有单次检索的 Multi-hop R@10 低 1.69 个百分点，预算内事实覆盖高 2.37 个百分点。需要让有价值的桥节点进入前十。

加深路径中识别到 249 个带图激活的金标候选，含 95 个 Multi-hop 金标候选。其中落入夹层的数量为 0。日历写入会跳过旧去重路径（初始代码的 `memengine/engine.py:333`），许多候选因此仍处在单次提及的外圈。这批数据支持优先排查种子、边质量及图重排。夹层保护针对有明确图证据的链路。

本机已实测离线 FastEmbed 384 维 MiniLM 和 MS MARCO MiniLM 交叉编码器。量化嵌入权重通过发布方 SHA-256 校验。对 “What city did Alice make her home in?”，嵌入对 “Alice settled in London.” 和无关内容分别得到 0.76873、0.12672 余弦，交叉编码器得分为 0.99809、0.00001。该例验证模型通路，全量增益另行评估。权重来源见 [MiniLM INT8](https://huggingface.co/Xenova/all-MiniLM-L6-v2/tree/main/onnx) 与 [MS MARCO MiniLM](https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2/tree/main/onnx)。

## LongMemEval-S 全量结论与改动顺序

LongMemEval-S 的 500 题重放锁定了 70.87% R@10、0.5425 MRR@10 和 1,933.8 Token，原话和事实溯源覆盖均为 100%。总体 A 类 14.24%，B 类 14.89%。Multi-session 同时在候选与精排失分（23.95%、23.86%），preference 为 34.44%、28.89%，temporal-reasoning 为 16.61%、14.67%，knowledge-update 为 3.47%、8.10%。164 个漏召原话实例中仅词干共现 30 个，无词法共现 54 个。

官方文件的 246,750 条 LongMemEval-S 原话均能解析会话时间和角色身份，500 个问题均提供问题日期。24,644 条原话包含可解析的显式发生区间，其余原话仍保留提及时间。

同预算的现有宽路径中，LongMemEval-S 单次的 R@10 为 70.84%，重构为 66.77%。knowledge-update 的最新态支配为 52.86% 和 51.43%，冻结入口为 60.00%。全局开启现有重构会拖累多类问题。优先级为保留来源元数据，改善 LoCoMo 多跳与 LongMemEval multi-session/temporal 的候选，收紧图桥接，随后处理 MemBench 与 preference 的精排失血，再单独评估预算装填和核验。

15 题同种子、同入口的离线可行性测试中，Hashed 与 ONNX 的 R@10 分别为 44.44% 和 51.39%，MRR 为 0.2940 和 0.3958。神经轨 P95 为 1,964.98 ms。12 题有正向金标，这组结果用于可行性和耗时定位，最终以全量矩阵为准。

## locomo · frozen · hashed

| 题型 | N | 金标题 | Raw | Fact | 候选@100 | 候选@50 | 候选@20 | R@10 | MRR@10 | 预算覆盖 | A 缺候选 | B 候选未装配 | Top-10 装填损失 | Token | P95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | 1986 | 1536 | 99.61% | 99.61% | 73.05% | 61.51% | 50.04% | 50.06% | 0.3927 | 53.58% | 37.65% | 12.30% | 3.55% | 473.1 | 327.05 |
| 1 multi-hop | 282 | 282 | 99.43% | 99.43% | 47.85% | 34.98% | 21.62% | 25.05% | 0.2715 | 26.64% | 64.16% | 10.78% | 2.10% | 490.6 | 323.78 |
| 2 temporal | 321 | 321 | 99.69% | 99.69% | 79.41% | 69.03% | 55.63% | 61.66% | 0.4791 | 64.41% | 29.41% | 8.93% | 3.37% | 442.1 | 326.16 |
| 3 open-domain | 96 | 92 | 96.74% | 96.74% | 41.04% | 31.57% | 23.68% | 25.67% | 0.1662 | 26.76% | 66.74% | 7.59% | 0.06% | 467.4 | 359.05 |
| 4 single-hop | 841 | 841 | 99.96% | 99.96% | 82.58% | 70.81% | 60.33% | 56.68% | 0.4251 | 61.41% | 28.72% | 14.61% | 4.48% | 478.6 | 327.34 |
| 5 adversarial | 446 | 0 | — | — | — | — | — | — | — | — | — | — | — | 475.0 | 320.03 |

| 题型 | 检索路 | 独立 R@50 | 移除此路 R@50 | 四路+桥接 R@50 | 边际百分点 |
| --- | --- | --- | --- | --- | --- |
| Overall | semantic | 56.90% | 66.41% | 64.17% | -2.24 |
| Overall | lexical | 67.78% | 56.66% | 64.17% | +7.51 |
| Overall | entity_graph | 20.70% | 64.58% | 64.17% | -0.42 |
| Overall | temporal | 1.78% | 64.16% | 64.17% | +0.01 |
| Overall | reconstructive | 0.00% | 64.17% | 64.17% | +0.00 |
| 1 multi-hop | semantic | 30.41% | 39.61% | 37.44% | -2.18 |
| 1 multi-hop | lexical | 41.19% | 29.18% | 37.44% | +8.26 |
| 1 multi-hop | entity_graph | 11.33% | 38.09% | 37.44% | -0.65 |
| 1 multi-hop | temporal | 0.71% | 37.08% | 37.44% | +0.35 |
| 1 multi-hop | reconstructive | 0.00% | 37.44% | 37.44% | +0.00 |
| 2 temporal | semantic | 62.56% | 74.95% | 71.26% | -3.69 |
| 2 temporal | lexical | 75.21% | 62.85% | 71.26% | +8.41 |
| 2 temporal | entity_graph | 25.83% | 71.26% | 71.26% | +0.00 |
| 2 temporal | temporal | 0.73% | 70.90% | 71.26% | +0.36 |
| 2 temporal | reconstructive | 0.00% | 71.26% | 71.26% | +0.00 |
| 3 open-domain | semantic | 26.88% | 34.00% | 32.05% | -1.95 |
| 3 open-domain | lexical | 35.28% | 27.81% | 32.05% | +4.24 |
| 3 open-domain | entity_graph | 16.03% | 33.72% | 32.05% | -1.67 |
| 3 open-domain | temporal | 0.00% | 33.14% | 32.05% | -1.09 |
| 3 open-domain | reconstructive | 0.00% | 32.05% | 32.05% | +0.00 |
| 4 single-hop | semantic | 66.90% | 75.68% | 73.94% | -1.74 |
| 4 single-hop | lexical | 77.41% | 66.67% | 73.94% | +7.27 |
| 4 single-hop | entity_graph | 22.39% | 74.30% | 73.94% | -0.36 |
| 4 single-hop | temporal | 2.73% | 74.06% | 73.94% | -0.12 |
| 4 single-hop | reconstructive | 0.00% | 73.94% | 73.94% | +0.00 |
| 5 adversarial | semantic | — | — | — | — |
| 5 adversarial | lexical | — | — | — | — |
| 5 adversarial | entity_graph | — | — | — | — |
| 5 adversarial | temporal | — | — | — | — |
| 5 adversarial | reconstructive | — | — | — | — |

| 题型 | 图金标候选 | 图金标夹层 | 夹层裁掉 | 可核查漏召证据 | 仅词干共现 | 无字面或词干共现 | 负样本空返 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | 0 | 0 | 0 | 1102 | 5 | 12 | 0/446 |
| 1 multi-hop | 0 | 0 | 0 | 571 | 5 | 6 | 0/0 |
| 2 temporal | 0 | 0 | 0 | 121 | 0 | 1 | 0/0 |
| 3 open-domain | 0 | 0 | 0 | 146 | 0 | 1 | 0/0 |
| 4 single-hop | 0 | 0 | 0 | 264 | 0 | 4 | 0/0 |
| 5 adversarial | 0 | 0 | 0 | 0 | 0 | 0 | 0/446 |

## membench99 · frozen · hashed

| 题型 | N | 金标题 | Raw | Fact | 候选@100 | 候选@50 | 候选@20 | R@10 | MRR@10 | 预算覆盖 | A 缺候选 | B 候选未装配 | Top-10 装填损失 | Token | P95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | 99 | 99 | 100.00% | 100.00% | 93.98% | 93.48% | 91.60% | 80.08% | 0.6999 | 80.08% | 6.52% | 13.40% | 1.88% | 529.3 | 182.22 |
| observation_factual | 25 | 25 | 100.00% | 100.00% | 89.50% | 89.50% | 89.50% | 80.00% | 0.7482 | 80.00% | 10.50% | 9.50% | 2.00% | 245.3 | 95.17 |
| observation_reflective | 24 | 24 | 100.00% | 100.00% | 90.42% | 90.42% | 90.42% | 79.51% | 0.7743 | 79.51% | 9.58% | 10.90% | 3.06% | 533.3 | 96.98 |
| participation_factual | 25 | 25 | 100.00% | 100.00% | 98.00% | 96.00% | 94.50% | 88.50% | 0.7527 | 88.50% | 4.00% | 7.50% | 1.50% | 561.7 | 189.65 |
| participation_reflective | 25 | 25 | 100.00% | 100.00% | 97.87% | 97.87% | 91.93% | 72.27% | 0.5276 | 72.27% | 2.13% | 25.60% | 1.00% | 777.0 | 175.45 |

| 题型 | 检索路 | 独立 R@50 | 移除此路 R@50 | 四路+桥接 R@50 | 边际百分点 |
| --- | --- | --- | --- | --- | --- |
| Overall | semantic | 92.90% | 62.05% | 93.48% | +31.42 |
| Overall | lexical | 62.05% | 92.90% | 93.48% | +0.58 |
| Overall | entity_graph | 14.56% | 93.48% | 93.48% | +0.00 |
| Overall | temporal | 0.00% | 93.48% | 93.48% | +0.00 |
| Overall | reconstructive | 0.00% | 93.48% | 93.48% | +0.00 |
| observation_factual | semantic | 89.50% | 80.50% | 89.50% | +9.00 |
| observation_factual | lexical | 80.50% | 89.50% | 89.50% | +0.00 |
| observation_factual | entity_graph | 24.00% | 89.50% | 89.50% | +0.00 |
| observation_factual | temporal | 0.00% | 89.50% | 89.50% | +0.00 |
| observation_factual | reconstructive | 0.00% | 89.50% | 89.50% | +0.00 |
| observation_reflective | semantic | 90.42% | 16.11% | 90.42% | +74.31 |
| observation_reflective | lexical | 16.11% | 90.42% | 90.42% | +0.00 |
| observation_reflective | entity_graph | 0.00% | 90.42% | 90.42% | +0.00 |
| observation_reflective | temporal | 0.00% | 90.42% | 90.42% | +0.00 |
| observation_reflective | reconstructive | 0.00% | 90.42% | 90.42% | +0.00 |
| participation_factual | semantic | 95.50% | 95.50% | 96.00% | +0.50 |
| participation_factual | lexical | 95.50% | 95.50% | 96.00% | +0.50 |
| participation_factual | entity_graph | 31.00% | 96.00% | 96.00% | +0.00 |
| participation_factual | temporal | 0.00% | 96.00% | 96.00% | +0.00 |
| participation_factual | reconstructive | 0.00% | 96.00% | 96.00% | +0.00 |
| participation_reflective | semantic | 96.07% | 54.27% | 97.87% | +43.60 |
| participation_reflective | lexical | 54.27% | 96.07% | 97.87% | +1.80 |
| participation_reflective | entity_graph | 2.67% | 97.87% | 97.87% | +0.00 |
| participation_reflective | temporal | 0.00% | 97.87% | 97.87% | +0.00 |
| participation_reflective | reconstructive | 0.00% | 97.87% | 97.87% | +0.00 |

| 题型 | 图金标候选 | 图金标夹层 | 夹层裁掉 | 可核查漏召证据 | 仅词干共现 | 无字面或词干共现 | 负样本空返 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | 0 | 0 | 0 | 18 | 2 | 16 | 0/0 |
| observation_factual | 0 | 0 | 0 | 5 | 0 | 5 | 0/0 |
| observation_reflective | 0 | 0 | 0 | 9 | 1 | 8 | 0/0 |
| participation_factual | 0 | 0 | 0 | 2 | 0 | 2 | 0/0 |
| participation_reflective | 0 | 0 | 0 | 2 | 1 | 1 | 0/0 |

## membench · frozen · hashed

| 题型 | N | 金标题 | Raw | Fact | 候选@100 | 候选@50 | 候选@20 | R@10 | MRR@10 | 预算覆盖 | A 缺候选 | B 候选未装配 | Top-10 装填损失 | Token | P95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | 400 | 400 | 100.00% | 100.00% | 95.00% | 93.61% | 90.85% | 77.80% | 0.6589 | 77.80% | 6.39% | 15.80% | 2.42% | 549.9 | 181.10 |
| observation_factual | 100 | 100 | 100.00% | 100.00% | 97.86% | 97.86% | 96.24% | 89.74% | 0.8363 | 89.74% | 2.14% | 8.12% | 0.81% | 333.6 | 130.76 |
| observation_reflective | 100 | 100 | 100.00% | 100.00% | 87.45% | 87.45% | 87.45% | 69.42% | 0.6555 | 69.42% | 12.55% | 18.03% | 4.28% | 522.0 | 134.89 |
| participation_factual | 100 | 100 | 100.00% | 100.00% | 96.40% | 90.85% | 85.22% | 76.20% | 0.6390 | 76.20% | 9.15% | 14.65% | 1.25% | 572.1 | 186.29 |
| participation_reflective | 100 | 100 | 100.00% | 100.00% | 98.28% | 98.28% | 94.47% | 75.87% | 0.5046 | 75.87% | 1.72% | 22.42% | 3.33% | 771.9 | 195.86 |

| 题型 | 检索路 | 独立 R@50 | 移除此路 R@50 | 四路+桥接 R@50 | 边际百分点 |
| --- | --- | --- | --- | --- | --- |
| Overall | semantic | 92.62% | 58.53% | 93.61% | +35.08 |
| Overall | lexical | 58.41% | 92.55% | 93.61% | +1.06 |
| Overall | entity_graph | 18.18% | 93.61% | 93.61% | +0.00 |
| Overall | temporal | 0.38% | 93.61% | 93.61% | +0.00 |
| Overall | reconstructive | 0.00% | 93.61% | 93.61% | +0.00 |
| observation_factual | semantic | 97.66% | 80.97% | 97.86% | +16.88 |
| observation_factual | lexical | 80.97% | 97.66% | 97.86% | +0.20 |
| observation_factual | entity_graph | 33.08% | 97.86% | 97.86% | +0.00 |
| observation_factual | temporal | 0.50% | 97.86% | 97.86% | +0.00 |
| observation_factual | reconstructive | 0.00% | 97.86% | 97.86% | +0.00 |
| observation_reflective | semantic | 87.20% | 9.77% | 87.45% | +77.68 |
| observation_reflective | lexical | 8.57% | 87.45% | 87.45% | +0.00 |
| observation_reflective | entity_graph | 1.20% | 87.20% | 87.45% | +0.25 |
| observation_reflective | temporal | 0.00% | 87.45% | 87.45% | +0.00 |
| observation_reflective | reconstructive | 0.00% | 87.45% | 87.45% | +0.00 |
| participation_factual | semantic | 89.50% | 86.58% | 90.85% | +4.27 |
| participation_factual | lexical | 87.30% | 88.97% | 90.85% | +1.88 |
| participation_factual | entity_graph | 38.17% | 91.10% | 90.85% | -0.25 |
| participation_factual | temporal | 1.00% | 90.85% | 90.85% | +0.00 |
| participation_factual | reconstructive | 0.00% | 90.85% | 90.85% | +0.00 |
| participation_reflective | semantic | 96.13% | 56.80% | 98.28% | +41.48 |
| participation_reflective | lexical | 56.80% | 96.13% | 98.28% | +2.15 |
| participation_reflective | entity_graph | 0.25% | 98.28% | 98.28% | +0.00 |
| participation_reflective | temporal | 0.00% | 98.28% | 98.28% | +0.00 |
| participation_reflective | reconstructive | 0.00% | 98.28% | 98.28% | +0.00 |

| 题型 | 图金标候选 | 图金标夹层 | 夹层裁掉 | 可核查漏召证据 | 仅词干共现 | 无字面或词干共现 | 负样本空返 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | 0 | 0 | 0 | 109 | 39 | 65 | 0/0 |
| observation_factual | 0 | 0 | 0 | 12 | 11 | 1 | 0/0 |
| observation_reflective | 0 | 0 | 0 | 52 | 9 | 43 | 0/0 |
| participation_factual | 0 | 0 | 0 | 38 | 17 | 16 | 0/0 |
| participation_reflective | 0 | 0 | 0 | 7 | 2 | 5 | 0/0 |

## locomo · chronicle · hashed

| 题型 | N | 金标题 | Raw | Fact | 候选@100 | 候选@50 | 候选@20 | R@10 | MRR@10 | 预算覆盖 | A 缺候选 | B 候选未装配 | Top-10 装填损失 | Token | P95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | 1986 | 1536 | 99.61% | 99.61% | 83.83% | 75.35% | 59.36% | 57.79% | 0.4451 | 77.00% | 24.16% | 18.05% | 2.35% | 2013.4 | 484.77 |
| 1 multi-hop | 282 | 282 | 99.43% | 99.43% | 65.49% | 52.32% | 34.96% | 29.08% | 0.2985 | 53.52% | 47.37% | 23.55% | 2.24% | 2023.3 | 448.33 |
| 2 temporal | 321 | 321 | 99.69% | 99.69% | 88.24% | 80.92% | 64.75% | 71.70% | 0.5700 | 81.54% | 17.94% | 10.36% | 2.34% | 1988.8 | 451.36 |
| 3 open-domain | 96 | 92 | 96.74% | 96.74% | 52.75% | 39.44% | 29.63% | 24.28% | 0.1859 | 41.07% | 60.56% | 15.17% | 1.69% | 1987.6 | 423.82 |
| 4 single-hop | 841 | 841 | 99.96% | 99.96% | 91.70% | 84.88% | 68.75% | 65.77% | 0.4750 | 87.08% | 14.76% | 19.46% | 2.46% | 2017.7 | 523.67 |
| 5 adversarial | 446 | 0 | — | — | — | — | — | — | — | — | — | — | — | 2022.2 | 483.79 |

| 题型 | 检索路 | 独立 R@50 | 移除此路 R@50 | 四路+桥接 R@50 | 边际百分点 |
| --- | --- | --- | --- | --- | --- |
| Overall | semantic | 66.07% | 72.73% | 73.69% | +0.95 |
| Overall | lexical | 72.59% | 67.66% | 73.69% | +6.03 |
| Overall | entity_graph | 20.70% | 73.14% | 73.69% | +0.54 |
| Overall | temporal | 2.63% | 73.64% | 73.69% | +0.04 |
| Overall | reconstructive | 39.16% | 73.53% | 73.69% | +0.16 |
| 1 multi-hop | semantic | 44.71% | 52.12% | 52.75% | +0.63 |
| 1 multi-hop | lexical | 52.02% | 46.02% | 52.75% | +6.73 |
| 1 multi-hop | entity_graph | 11.33% | 53.13% | 52.75% | -0.38 |
| 1 multi-hop | temporal | 0.71% | 52.57% | 52.75% | +0.18 |
| 1 multi-hop | reconstructive | 27.89% | 52.27% | 52.75% | +0.48 |
| 2 temporal | semantic | 71.96% | 79.39% | 80.63% | +1.25 |
| 2 temporal | lexical | 79.23% | 73.96% | 80.63% | +6.67 |
| 2 temporal | entity_graph | 25.83% | 79.80% | 80.63% | +0.83 |
| 2 temporal | temporal | 1.04% | 80.74% | 80.63% | -0.10 |
| 2 temporal | reconstructive | 41.87% | 81.98% | 80.63% | -1.35 |
| 3 open-domain | semantic | 39.17% | 41.89% | 41.91% | +0.03 |
| 3 open-domain | lexical | 41.78% | 38.10% | 41.91% | +3.82 |
| 3 open-domain | entity_graph | 16.03% | 43.79% | 41.91% | -1.88 |
| 3 open-domain | temporal | 1.09% | 41.37% | 41.91% | +0.54 |
| 3 open-domain | reconstructive | 16.21% | 44.02% | 41.91% | -2.11 |
| 4 single-hop | semantic | 73.92% | 80.48% | 81.53% | +1.05 |
| 4 single-hop | lexical | 80.32% | 75.74% | 81.53% | +5.79 |
| 4 single-hop | entity_graph | 22.39% | 80.52% | 81.53% | +1.01 |
| 4 single-hop | temporal | 4.04% | 81.53% | 81.53% | +0.00 |
| 4 single-hop | reconstructive | 44.41% | 80.66% | 81.53% | +0.87 |
| 5 adversarial | semantic | — | — | — | — |
| 5 adversarial | lexical | — | — | — | — |
| 5 adversarial | entity_graph | — | — | — | — |
| 5 adversarial | temporal | — | — | — | — |
| 5 adversarial | reconstructive | — | — | — | — |

| 题型 | 图金标候选 | 图金标夹层 | 夹层裁掉 | 可核查漏召证据 | 仅词干共现 | 无字面或词干共现 | 负样本空返 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | 249 | 0 | 0 | 775 | 0 | 12 | 0/446 |
| 1 multi-hop | 95 | 0 | 0 | 430 | 0 | 6 | 0/0 |
| 2 temporal | 36 | 0 | 0 | 75 | 0 | 1 | 0/0 |
| 3 open-domain | 9 | 0 | 0 | 134 | 0 | 1 | 0/0 |
| 4 single-hop | 109 | 0 | 0 | 136 | 0 | 4 | 0/0 |
| 5 adversarial | 0 | 0 | 0 | 0 | 0 | 0 | 0/446 |

## longmemeval · frozen · hashed

| 题型 | N | 金标题 | Raw | Fact | 候选@100 | 候选@50 | 候选@20 | R@10 | MRR@10 | 预算覆盖 | A 缺候选 | B 候选未装配 | Top-10 装填损失 | Token | P95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | 500 | 470 | 100.00% | 100.00% | 91.60% | 85.66% | 78.71% | 70.87% | 0.5425 | 70.87% | 14.24% | 14.89% | 2.84% | 1933.8 | 522.10 |
| abstention | 30 | 0 | — | — | — | — | — | — | — | — | — | — | — | 1919.6 | 339.45 |
| knowledge-update | 72 | 72 | 100.00% | 100.00% | 98.61% | 96.53% | 93.29% | 88.43% | 0.6704 | 88.43% | 3.47% | 8.10% | 0.00% | 1929.7 | 260.85 |
| multi-session | 121 | 121 | 100.00% | 100.00% | 86.34% | 76.05% | 64.27% | 52.19% | 0.4090 | 52.19% | 23.95% | 23.86% | 5.12% | 1967.8 | 682.12 |
| single-session-assistant | 56 | 56 | 100.00% | 100.00% | 98.21% | 98.21% | 91.07% | 89.29% | 0.7355 | 89.29% | 1.79% | 8.93% | 0.00% | 1883.8 | 302.39 |
| single-session-preference | 30 | 30 | 100.00% | 100.00% | 81.11% | 65.56% | 52.22% | 36.67% | 0.1643 | 36.67% | 34.44% | 28.89% | 0.00% | 1692.8 | 231.11 |
| single-session-user | 64 | 64 | 100.00% | 100.00% | 98.44% | 95.31% | 93.75% | 90.62% | 0.7021 | 90.62% | 4.69% | 4.69% | 1.56% | 1974.6 | 566.22 |
| temporal-reasoning | 127 | 127 | 100.00% | 100.00% | 88.77% | 82.99% | 77.43% | 68.71% | 0.5209 | 68.71% | 16.61% | 14.67% | 4.86% | 1965.6 | 214.46 |

| 题型 | 检索路 | 独立 R@50 | 移除此路 R@50 | 四路+桥接 R@50 | 边际百分点 |
| --- | --- | --- | --- | --- | --- |
| Overall | semantic | 79.05% | 88.09% | 85.94% | -2.15 |
| Overall | lexical | 88.74% | 79.83% | 85.94% | +6.11 |
| Overall | entity_graph | 34.66% | 85.55% | 85.94% | +0.39 |
| Overall | temporal | 0.28% | 85.94% | 85.94% | +0.00 |
| Overall | reconstructive | 0.00% | 85.94% | 85.94% | +0.00 |
| abstention | semantic | — | — | — | — |
| abstention | lexical | — | — | — | — |
| abstention | entity_graph | — | — | — | — |
| abstention | temporal | — | — | — | — |
| abstention | reconstructive | — | — | — | — |
| knowledge-update | semantic | 93.06% | 98.15% | 96.53% | -1.62 |
| knowledge-update | lexical | 98.15% | 93.75% | 96.53% | +2.78 |
| knowledge-update | entity_graph | 47.45% | 96.53% | 96.53% | +0.00 |
| knowledge-update | temporal | 0.00% | 96.53% | 96.53% | +0.00 |
| knowledge-update | reconstructive | 0.00% | 96.53% | 96.53% | +0.00 |
| multi-session | semantic | 65.72% | 79.89% | 76.32% | -3.57 |
| multi-session | lexical | 81.16% | 67.09% | 76.32% | +9.23 |
| multi-session | entity_graph | 18.25% | 76.18% | 76.32% | +0.14 |
| multi-session | temporal | 0.00% | 76.32% | 76.32% | +0.00 |
| multi-session | reconstructive | 0.00% | 76.32% | 76.32% | +0.00 |
| single-session-assistant | semantic | 91.07% | 96.43% | 98.21% | +1.79 |
| single-session-assistant | lexical | 98.21% | 91.07% | 98.21% | +7.14 |
| single-session-assistant | entity_graph | 76.79% | 96.43% | 98.21% | +1.79 |
| single-session-assistant | temporal | 0.00% | 98.21% | 98.21% | +0.00 |
| single-session-assistant | reconstructive | 0.00% | 98.21% | 98.21% | +0.00 |
| single-session-preference | semantic | 58.89% | 70.00% | 65.56% | -4.44 |
| single-session-preference | lexical | 71.67% | 58.89% | 65.56% | +6.67 |
| single-session-preference | entity_graph | 3.33% | 67.22% | 65.56% | -1.67 |
| single-session-preference | temporal | 0.00% | 65.56% | 65.56% | +0.00 |
| single-session-preference | reconstructive | 0.00% | 65.56% | 65.56% | +0.00 |
| single-session-user | semantic | 92.97% | 96.88% | 95.31% | -1.56 |
| single-session-user | lexical | 96.88% | 92.97% | 95.31% | +2.34 |
| single-session-user | entity_graph | 19.53% | 95.31% | 95.31% | +0.00 |
| single-session-user | temporal | 1.56% | 95.31% | 95.31% | +0.00 |
| single-session-user | reconstructive | 0.00% | 95.31% | 95.31% | +0.00 |
| temporal-reasoning | semantic | 76.27% | 86.35% | 83.78% | -2.57 |
| temporal-reasoning | lexical | 86.38% | 77.45% | 83.78% | +6.33 |
| temporal-reasoning | entity_graph | 39.50% | 82.86% | 83.78% | +0.92 |
| temporal-reasoning | temporal | 0.26% | 83.78% | 83.78% | +0.00 |
| temporal-reasoning | reconstructive | 0.00% | 83.78% | 83.78% | +0.00 |

| 题型 | 图金标候选 | 图金标夹层 | 夹层裁掉 | 可核查漏召证据 | 仅词干共现 | 无字面或词干共现 | 负样本空返 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | 0 | 0 | 0 | 164 | 30 | 54 | 0/30 |
| abstention | 0 | 0 | 0 | 0 | 0 | 0 | 0/30 |
| knowledge-update | 0 | 0 | 0 | 5 | 1 | 1 | 0/0 |
| multi-session | 0 | 0 | 0 | 96 | 22 | 37 | 0/0 |
| single-session-assistant | 0 | 0 | 0 | 1 | 0 | 1 | 0/0 |
| single-session-preference | 0 | 0 | 0 | 15 | 1 | 1 | 0/0 |
| single-session-user | 0 | 0 | 0 | 3 | 0 | 1 | 0/0 |
| temporal-reasoning | 0 | 0 | 0 | 44 | 6 | 13 | 0/0 |

## longmemeval · chronicle · hashed

| 题型 | N | 金标题 | Raw | Fact | 候选@100 | 候选@50 | 候选@20 | R@10 | MRR@10 | 预算覆盖 | A 缺候选 | B 候选未装配 | Top-10 装填损失 | Token | P95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | 500 | 470 | 100.00% | 100.00% | 92.84% | 88.30% | 74.82% | 66.77% | 0.5238 | 82.09% | 11.60% | 21.63% | 1.51% | 2044.5 | 847.12 |
| abstention | 30 | 0 | — | — | — | — | — | — | — | — | — | — | — | 2044.2 | 791.09 |
| knowledge-update | 72 | 72 | 100.00% | 100.00% | 99.31% | 98.61% | 93.06% | 81.25% | 0.6568 | 95.14% | 1.39% | 17.36% | 2.55% | 2044.4 | 365.54 |
| multi-session | 121 | 121 | 100.00% | 100.00% | 88.37% | 81.71% | 58.68% | 46.61% | 0.3898 | 70.11% | 18.29% | 35.10% | 1.31% | 2044.4 | 500.90 |
| single-session-assistant | 56 | 56 | 100.00% | 100.00% | 98.21% | 96.43% | 91.07% | 91.07% | 0.7455 | 94.64% | 3.57% | 5.36% | 0.00% | 2044.1 | 198.72 |
| single-session-preference | 30 | 30 | 100.00% | 100.00% | 68.33% | 60.56% | 31.11% | 18.33% | 0.0712 | 43.33% | 37.78% | 43.89% | 1.11% | 2045.2 | 294.34 |
| single-session-user | 64 | 64 | 100.00% | 100.00% | 100.00% | 96.88% | 94.53% | 90.62% | 0.6183 | 98.44% | 3.12% | 6.25% | 0.00% | 2044.6 | 1313.37 |
| temporal-reasoning | 127 | 127 | 100.00% | 100.00% | 93.25% | 87.38% | 73.10% | 66.48% | 0.5374 | 81.50% | 12.62% | 20.89% | 2.62% | 2044.6 | 466.46 |

| 题型 | 检索路 | 独立 R@50 | 移除此路 R@50 | 四路+桥接 R@50 | 边际百分点 |
| --- | --- | --- | --- | --- | --- |
| Overall | semantic | 81.86% | 90.89% | 87.84% | -3.05 |
| Overall | lexical | 90.36% | 82.81% | 87.84% | +5.03 |
| Overall | entity_graph | 39.47% | 87.31% | 87.84% | +0.53 |
| Overall | temporal | 3.18% | 87.76% | 87.84% | +0.08 |
| Overall | reconstructive | 62.81% | 88.34% | 87.84% | -0.50 |
| abstention | semantic | — | — | — | — |
| abstention | lexical | — | — | — | — |
| abstention | entity_graph | — | — | — | — |
| abstention | temporal | — | — | — | — |
| abstention | reconstructive | — | — | — | — |
| knowledge-update | semantic | 97.22% | 97.45% | 98.61% | +1.16 |
| knowledge-update | lexical | 98.15% | 95.14% | 98.61% | +3.47 |
| knowledge-update | entity_graph | 50.69% | 97.45% | 98.61% | +1.16 |
| knowledge-update | temporal | 0.00% | 98.61% | 98.61% | +0.00 |
| knowledge-update | reconstructive | 76.85% | 98.61% | 98.61% | +0.00 |
| multi-session | semantic | 72.52% | 85.10% | 82.99% | -2.11 |
| multi-session | lexical | 86.89% | 73.58% | 82.99% | +9.41 |
| multi-session | entity_graph | 25.30% | 81.45% | 82.99% | +1.54 |
| multi-session | temporal | 1.24% | 82.99% | 82.99% | +0.00 |
| multi-session | reconstructive | 52.47% | 83.46% | 82.99% | -0.47 |
| single-session-assistant | semantic | 91.07% | 98.21% | 94.64% | -3.57 |
| single-session-assistant | lexical | 94.64% | 91.07% | 94.64% | +3.57 |
| single-session-assistant | entity_graph | 80.36% | 92.86% | 94.64% | +1.79 |
| single-session-assistant | temporal | 0.00% | 94.64% | 94.64% | +0.00 |
| single-session-assistant | reconstructive | 82.14% | 94.64% | 94.64% | +0.00 |
| single-session-preference | semantic | 52.22% | 67.22% | 62.22% | -5.00 |
| single-session-preference | lexical | 71.67% | 53.89% | 62.22% | +8.33 |
| single-session-preference | entity_graph | 13.33% | 58.89% | 62.22% | +3.33 |
| single-session-preference | temporal | 0.00% | 62.22% | 62.22% | +0.00 |
| single-session-preference | reconstructive | 23.33% | 62.22% | 62.22% | +0.00 |
| single-session-user | semantic | 96.09% | 98.44% | 96.88% | -1.56 |
| single-session-user | lexical | 96.88% | 96.09% | 96.88% | +0.78 |
| single-session-user | entity_graph | 21.09% | 98.44% | 96.88% | -1.56 |
| single-session-user | temporal | 3.12% | 96.88% | 96.88% | +0.00 |
| single-session-user | reconstructive | 65.62% | 98.44% | 96.88% | -1.56 |
| temporal-reasoning | semantic | 77.82% | 91.26% | 84.86% | -6.40 |
| temporal-reasoning | lexical | 88.50% | 81.10% | 84.86% | +3.75 |
| temporal-reasoning | entity_graph | 44.02% | 85.80% | 84.86% | -0.94 |
| temporal-reasoning | temporal | 9.00% | 84.57% | 84.86% | +0.29 |
| temporal-reasoning | reconstructive | 64.09% | 85.49% | 84.86% | -0.63 |

| 题型 | 图金标候选 | 图金标夹层 | 夹层裁掉 | 可核查漏召证据 | 仅词干共现 | 无字面或词干共现 | 负样本空返 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | 80 | 0 | 0 | 130 | 14 | 49 | 0/30 |
| abstention | 0 | 0 | 0 | 0 | 0 | 0 | 0/30 |
| knowledge-update | 7 | 0 | 0 | 2 | 1 | 1 | 0/0 |
| multi-session | 39 | 0 | 0 | 75 | 10 | 34 | 0/0 |
| single-session-assistant | 3 | 0 | 0 | 2 | 0 | 1 | 0/0 |
| single-session-preference | 5 | 0 | 0 | 16 | 0 | 2 | 0/0 |
| single-session-user | 2 | 0 | 0 | 2 | 0 | 1 | 0/0 |
| temporal-reasoning | 24 | 0 | 0 | 33 | 3 | 10 | 0/0 |
