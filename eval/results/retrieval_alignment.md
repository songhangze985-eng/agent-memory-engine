# 检索对齐实测

冻结入库和宽检索使用同一官方队列。R@10 和 MRR@10 都以实际装填的前十条事实计算。预算覆盖还计入已注入原话。Token 按实际上下文估算。候选 R@50 来自每路加深到 100 的诊断通路。未经逐题语义标注的词法缺口见失分报告。

冻结管线继续独立运行严格门禁。宽检索以 2,048 Token 装填，原冻结平均 Token 是独立成本约束。

## Hashed 冻结入口与预算检索对比

本节的预算检索快照用于完整漏斗和逐路消融。后续的日历路由调整与数据库批量读取以末尾的当前代码配对复跑为准。

| 基准 | 题型 | N | R@50 | R@10 | MRR@10 | Token | 预算覆盖 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| locomo | Overall | 1986 | 61.51% → 75.63% | 50.06% → 61.22% | 0.3927 → 0.4847 | 473.1 → 2006.5 | 77.34% |
| locomo | 1 multi-hop | 282 | 34.98% → 53.39% | 25.05% → 35.63% | 0.2715 → 0.3521 | 490.6 → 2022.4 | 53.85% |
| locomo | 2 temporal | 321 | 69.03% → 81.83% | 61.66% → 74.74% | 0.4791 → 0.6239 | 442.1 → 1982.4 | 83.18% |
| locomo | 3 open-domain | 96 | 31.57% → 44.28% | 25.67% → 30.26% | 0.1662 → 0.2239 | 467.4 → 1965.8 | 43.62% |
| locomo | 4 single-hop | 841 | 70.81% → 84.15% | 56.68% → 68.03% | 0.4251 → 0.5046 | 478.6 → 2011.6 | 86.68% |
| locomo | 5 adversarial | 446 | — → — | — → — | — → — | 475.0 → 2013.2 | — |
| longmemeval | Overall | 500 | 85.66% → 89.63% | 70.87% → 71.14% | 0.5425 → 0.5568 | 1933.8 → 2037.3 | 84.55% |
| longmemeval | abstention | 30 | — → — | — → — | — → — | 1919.6 → 2039.9 | — |
| longmemeval | knowledge-update | 72 | 96.53% → 98.61% | 88.43% → 80.79% | 0.6704 → 0.6633 | 1929.7 → 2041.5 | 97.92% |
| longmemeval | multi-session | 121 | 76.05% → 82.85% | 52.19% → 50.87% | 0.4090 → 0.4379 | 1967.8 → 2035.9 | 72.62% |
| longmemeval | single-session-assistant | 56 | 98.21% → 96.43% | 89.29% → 91.07% | 0.7355 → 0.7318 | 1883.8 → 2033.5 | 94.64% |
| longmemeval | single-session-preference | 30 | 65.56% → 62.22% | 36.67% → 30.56% | 0.1643 → 0.1084 | 1692.8 → 2041.9 | 53.89% |
| longmemeval | single-session-user | 64 | 95.31% → 98.44% | 90.62% → 92.19% | 0.7021 → 0.6796 | 1974.6 → 2033.2 | 98.44% |
| longmemeval | temporal-reasoning | 127 | 82.99% → 90.03% | 68.71% → 75.17% | 0.5209 → 0.5767 | 1965.6 → 2038.4 | 84.15% |
| membench99 | Overall | 99 | 93.48% → 97.98% | 80.08% → 82.17% | 0.6999 → 0.6812 | 529.3 → 1116.2 | 94.60% |
| membench99 | observation_factual | 25 | 89.50% → 96.00% | 80.00% → 79.50% | 0.7482 → 0.7167 | 245.3 → 371.4 | 90.00% |
| membench99 | observation_reflective | 24 | 90.42% → 100.00% | 79.51% → 86.88% | 0.7743 → 0.8090 | 533.3 → 665.2 | 94.51% |
| membench99 | participation_factual | 25 | 96.00% → 96.00% | 88.50% → 89.50% | 0.7527 → 0.7560 | 561.7 → 2032.1 | 96.00% |
| membench99 | participation_reflective | 25 | 97.87% → 100.00% | 72.27% → 73.00% | 0.5276 → 0.4482 | 777.0 → 1377.9 | 97.87% |
| membench | Overall | 400 | 93.61% → 98.23% | 77.80% → 79.08% | 0.6589 → 0.6788 | 549.9 → 1162.6 | 94.33% |
| membench | observation_factual | 100 | 97.86% → 100.00% | 89.74% → 90.25% | 0.8363 → 0.8532 | 333.6 → 465.7 | 99.24% |
| membench | observation_reflective | 100 | 87.45% → 100.00% | 69.42% → 76.92% | 0.6555 → 0.7351 | 522.0 → 697.6 | 90.47% |
| membench | participation_factual | 100 | 90.85% → 92.92% | 76.20% → 74.97% | 0.6390 → 0.6523 | 572.1 → 1962.2 | 88.80% |
| membench | participation_reflective | 100 | 98.28% → 100.00% | 75.87% → 74.17% | 0.5046 → 0.4744 | 771.9 → 1524.7 | 98.80% |

## 实际装配逐路消融

每一路分别在同一题上关闭。边际值为完整检索减去关闭该路之后的结果。single_query 是关闭整条重构路径的同预算对照。

| 基准 | 题型 | 关闭路径 | R@10 | Δ R@10 百分点 | MRR@10 | Δ MRR@10 |
| --- | --- | --- | --- | --- | --- | --- |
| locomo | Overall | semantic | 62.24% | -1.01 | 0.4873 | -0.0026 |
| locomo | Overall | lexical | 58.35% | +2.87 | 0.4689 | +0.0158 |
| locomo | Overall | entity_graph | 60.55% | +0.68 | 0.4806 | +0.0041 |
| locomo | Overall | temporal | 62.28% | -1.05 | 0.4960 | -0.0113 |
| locomo | Overall | reconstructive | 61.12% | +0.10 | 0.4842 | +0.0005 |
| locomo | Overall | single_query | 60.45% | +0.77 | 0.4793 | +0.0055 |
| locomo | 1 multi-hop | semantic | 35.57% | +0.07 | 0.3530 | -0.0009 |
| locomo | 1 multi-hop | lexical | 32.25% | +3.39 | 0.3384 | +0.0137 |
| locomo | 1 multi-hop | entity_graph | 34.58% | +1.06 | 0.3471 | +0.0050 |
| locomo | 1 multi-hop | temporal | 35.34% | +0.29 | 0.3496 | +0.0025 |
| locomo | 1 multi-hop | reconstructive | 34.78% | +0.85 | 0.3492 | +0.0029 |
| locomo | 1 multi-hop | single_query | 34.14% | +1.50 | 0.3459 | +0.0062 |
| locomo | 2 temporal | semantic | 75.67% | -0.93 | 0.6254 | -0.0015 |
| locomo | 2 temporal | lexical | 71.39% | +3.35 | 0.6042 | +0.0197 |
| locomo | 2 temporal | entity_graph | 74.77% | -0.03 | 0.6187 | +0.0053 |
| locomo | 2 temporal | temporal | 75.05% | -0.31 | 0.6203 | +0.0037 |
| locomo | 2 temporal | reconstructive | 74.53% | +0.21 | 0.6232 | +0.0007 |
| locomo | 2 temporal | single_query | 74.22% | +0.52 | 0.6168 | +0.0071 |
| locomo | 3 open-domain | semantic | 31.08% | -0.82 | 0.2460 | -0.0221 |
| locomo | 3 open-domain | lexical | 31.02% | -0.75 | 0.2403 | -0.0164 |
| locomo | 3 open-domain | entity_graph | 31.35% | -1.09 | 0.2250 | -0.0011 |
| locomo | 3 open-domain | temporal | 30.81% | -0.54 | 0.2257 | -0.0018 |
| locomo | 3 open-domain | reconstructive | 29.18% | +1.09 | 0.2262 | -0.0023 |
| locomo | 3 open-domain | single_query | 31.35% | -1.09 | 0.2207 | +0.0031 |
| locomo | 4 single-hop | semantic | 69.46% | -1.43 | 0.5061 | -0.0015 |
| locomo | 4 single-hop | lexical | 65.12% | +2.91 | 0.4860 | +0.0186 |
| locomo | 4 single-hop | entity_graph | 67.02% | +1.01 | 0.5006 | +0.0040 |
| locomo | 4 single-hop | temporal | 69.88% | -1.84 | 0.5273 | -0.0227 |
| locomo | 4 single-hop | reconstructive | 68.33% | -0.30 | 0.5047 | -0.0001 |
| locomo | 4 single-hop | single_query | 67.20% | +0.83 | 0.4998 | +0.0048 |
| longmemeval | Overall | semantic | 68.37% | +2.77 | 0.5151 | +0.0417 |
| longmemeval | Overall | lexical | 68.91% | +2.23 | 0.5416 | +0.0152 |
| longmemeval | Overall | entity_graph | 70.78% | +0.36 | 0.5665 | -0.0097 |
| longmemeval | Overall | temporal | 71.17% | -0.04 | 0.5602 | -0.0034 |
| longmemeval | Overall | reconstructive | 71.14% | +0.00 | 0.5474 | +0.0095 |
| longmemeval | Overall | single_query | 70.72% | +0.42 | 0.5595 | -0.0026 |
| longmemeval | knowledge-update | semantic | 79.86% | +0.93 | 0.6263 | +0.0371 |
| longmemeval | knowledge-update | lexical | 81.94% | -1.16 | 0.6393 | +0.0241 |
| longmemeval | knowledge-update | entity_graph | 81.02% | -0.23 | 0.6832 | -0.0199 |
| longmemeval | knowledge-update | temporal | 82.18% | -1.39 | 0.6656 | -0.0023 |
| longmemeval | knowledge-update | reconstructive | 80.79% | +0.00 | 0.6564 | +0.0069 |
| longmemeval | knowledge-update | single_query | 82.64% | -1.85 | 0.6729 | -0.0096 |
| longmemeval | multi-session | semantic | 47.04% | +3.83 | 0.3811 | +0.0568 |
| longmemeval | multi-session | lexical | 48.69% | +2.18 | 0.4283 | +0.0096 |
| longmemeval | multi-session | entity_graph | 52.37% | -1.50 | 0.4565 | -0.0186 |
| longmemeval | multi-session | temporal | 51.69% | -0.83 | 0.4462 | -0.0083 |
| longmemeval | multi-session | reconstructive | 50.18% | +0.69 | 0.4260 | +0.0119 |
| longmemeval | multi-session | single_query | 51.45% | -0.58 | 0.4382 | -0.0003 |
| longmemeval | single-session-assistant | semantic | 91.07% | +0.00 | 0.7843 | -0.0525 |
| longmemeval | single-session-assistant | lexical | 89.29% | +1.79 | 0.7323 | -0.0005 |
| longmemeval | single-session-assistant | entity_graph | 89.29% | +1.79 | 0.7342 | -0.0024 |
| longmemeval | single-session-assistant | temporal | 91.07% | +0.00 | 0.7318 | +0.0000 |
| longmemeval | single-session-assistant | reconstructive | 91.07% | +0.00 | 0.7313 | +0.0006 |
| longmemeval | single-session-assistant | single_query | 89.29% | +1.79 | 0.7357 | -0.0039 |
| longmemeval | single-session-preference | semantic | 25.00% | +5.56 | 0.0809 | +0.0275 |
| longmemeval | single-session-preference | lexical | 23.33% | +7.22 | 0.0812 | +0.0272 |
| longmemeval | single-session-preference | entity_graph | 29.44% | +1.11 | 0.1120 | -0.0036 |
| longmemeval | single-session-preference | temporal | 30.56% | +0.00 | 0.1084 | +0.0000 |
| longmemeval | single-session-preference | reconstructive | 32.78% | -2.22 | 0.1053 | +0.0031 |
| longmemeval | single-session-preference | single_query | 30.56% | +0.00 | 0.1079 | +0.0006 |
| longmemeval | single-session-user | semantic | 89.06% | +3.12 | 0.6115 | +0.0681 |
| longmemeval | single-session-user | lexical | 90.62% | +1.56 | 0.6526 | +0.0270 |
| longmemeval | single-session-user | entity_graph | 90.62% | +1.56 | 0.6788 | +0.0008 |
| longmemeval | single-session-user | temporal | 92.19% | +0.00 | 0.6913 | -0.0117 |
| longmemeval | single-session-user | reconstructive | 92.19% | +0.00 | 0.6623 | +0.0173 |
| longmemeval | single-session-user | single_query | 90.62% | +1.56 | 0.6793 | +0.0003 |
| longmemeval | temporal-reasoning | semantic | 71.97% | +3.20 | 0.5150 | +0.0617 |
| longmemeval | temporal-reasoning | lexical | 71.63% | +3.54 | 0.5629 | +0.0137 |
| longmemeval | temporal-reasoning | entity_graph | 74.12% | +1.05 | 0.5820 | -0.0053 |
| longmemeval | temporal-reasoning | temporal | 73.73% | +1.44 | 0.5741 | +0.0025 |
| longmemeval | temporal-reasoning | reconstructive | 75.30% | -0.13 | 0.5666 | +0.0100 |
| longmemeval | temporal-reasoning | single_query | 73.60% | +1.57 | 0.5792 | -0.0025 |
| membench99 | Overall | semantic | 68.68% | +13.49 | 0.6386 | +0.0426 |
| membench99 | Overall | lexical | 80.41% | +1.76 | 0.6766 | +0.0045 |
| membench99 | Overall | entity_graph | 82.17% | +0.00 | 0.6812 | +0.0000 |
| membench99 | Overall | temporal | 82.17% | +0.00 | 0.6812 | +0.0000 |
| membench99 | Overall | reconstructive | 81.68% | +0.50 | 0.6805 | +0.0007 |
| membench99 | Overall | single_query | 81.68% | +0.50 | 0.6805 | +0.0007 |
| membench99 | observation_factual | semantic | 72.50% | +7.00 | 0.7364 | -0.0197 |
| membench99 | observation_factual | lexical | 79.50% | +0.00 | 0.7208 | -0.0041 |
| membench99 | observation_factual | entity_graph | 79.50% | +0.00 | 0.7167 | +0.0000 |
| membench99 | observation_factual | temporal | 79.50% | +0.00 | 0.7167 | +0.0000 |
| membench99 | observation_factual | reconstructive | 82.00% | -2.50 | 0.7171 | -0.0004 |
| membench99 | observation_factual | single_query | 82.00% | -2.50 | 0.7171 | -0.0004 |
| membench99 | observation_reflective | semantic | 48.40% | +38.47 | 0.6181 | +0.1910 |
| membench99 | observation_reflective | lexical | 83.61% | +3.26 | 0.8090 | +0.0000 |
| membench99 | observation_reflective | entity_graph | 86.88% | +0.00 | 0.8090 | +0.0000 |
| membench99 | observation_reflective | temporal | 86.88% | +0.00 | 0.8090 | +0.0000 |
| membench99 | observation_reflective | reconstructive | 83.61% | +3.26 | 0.8090 | +0.0000 |
| membench99 | observation_reflective | single_query | 83.61% | +3.26 | 0.8090 | +0.0000 |
| membench99 | participation_factual | semantic | 88.00% | +1.50 | 0.8000 | -0.0440 |
| membench99 | participation_factual | lexical | 89.00% | +0.50 | 0.7373 | +0.0187 |
| membench99 | participation_factual | entity_graph | 89.50% | +0.00 | 0.7560 | +0.0000 |
| membench99 | participation_factual | temporal | 89.50% | +0.00 | 0.7560 | +0.0000 |
| membench99 | participation_factual | reconstructive | 89.50% | +0.00 | 0.7547 | +0.0013 |
| membench99 | participation_factual | single_query | 89.50% | +0.00 | 0.7547 | +0.0013 |
| membench99 | participation_reflective | semantic | 65.00% | +8.00 | 0.3992 | +0.0490 |
| membench99 | participation_reflective | lexical | 69.67% | +3.33 | 0.4447 | +0.0034 |
| membench99 | participation_reflective | entity_graph | 73.00% | +0.00 | 0.4482 | +0.0000 |
| membench99 | participation_reflective | temporal | 73.00% | +0.00 | 0.4482 | +0.0000 |
| membench99 | participation_reflective | reconstructive | 71.67% | +1.33 | 0.4462 | +0.0020 |
| membench99 | participation_reflective | single_query | 71.67% | +1.33 | 0.4462 | +0.0020 |
| membench | Overall | semantic | 66.85% | +12.23 | 0.6194 | +0.0594 |
| membench | Overall | lexical | 78.08% | +0.99 | 0.6720 | +0.0067 |
| membench | Overall | entity_graph | 79.33% | -0.25 | 0.6793 | -0.0005 |
| membench | Overall | temporal | 79.08% | +0.00 | 0.6790 | -0.0002 |
| membench | Overall | reconstructive | 78.34% | +0.74 | 0.6797 | -0.0010 |
| membench | Overall | single_query | 78.34% | +0.74 | 0.6797 | -0.0010 |
| membench | observation_factual | semantic | 80.67% | +9.58 | 0.8282 | +0.0251 |
| membench | observation_factual | lexical | 89.79% | +0.46 | 0.8469 | +0.0063 |
| membench | observation_factual | entity_graph | 90.75% | -0.50 | 0.8482 | +0.0050 |
| membench | observation_factual | temporal | 90.25% | +0.00 | 0.8532 | +0.0000 |
| membench | observation_factual | reconstructive | 89.59% | +0.66 | 0.8487 | +0.0045 |
| membench | observation_factual | single_query | 89.59% | +0.66 | 0.8487 | +0.0045 |
| membench | observation_reflective | semantic | 47.88% | +29.03 | 0.5482 | +0.1869 |
| membench | observation_reflective | lexical | 74.12% | +2.80 | 0.7192 | +0.0159 |
| membench | observation_reflective | entity_graph | 76.92% | +0.00 | 0.7351 | +0.0000 |
| membench | observation_reflective | temporal | 76.92% | +0.00 | 0.7351 | +0.0000 |
| membench | observation_reflective | reconstructive | 74.90% | +2.02 | 0.7338 | +0.0013 |
| membench | observation_reflective | single_query | 74.90% | +2.02 | 0.7338 | +0.0013 |
| membench | participation_factual | semantic | 74.38% | +0.60 | 0.6740 | -0.0217 |
| membench | participation_factual | lexical | 75.47% | -0.50 | 0.6528 | -0.0005 |
| membench | participation_factual | entity_graph | 75.47% | -0.50 | 0.6595 | -0.0072 |
| membench | participation_factual | temporal | 74.97% | +0.00 | 0.6532 | -0.0008 |
| membench | participation_factual | reconstructive | 74.95% | +0.02 | 0.6645 | -0.0122 |
| membench | participation_factual | single_query | 74.95% | +0.02 | 0.6645 | -0.0122 |
| membench | participation_reflective | semantic | 64.48% | +9.68 | 0.4272 | +0.0472 |
| membench | participation_reflective | lexical | 72.95% | +1.22 | 0.4693 | +0.0051 |
| membench | participation_reflective | entity_graph | 74.17% | +0.00 | 0.4744 | +0.0000 |
| membench | participation_reflective | temporal | 74.17% | +0.00 | 0.4744 | +0.0000 |
| membench | participation_reflective | reconstructive | 73.92% | +0.25 | 0.4719 | +0.0025 |
| membench | participation_reflective | single_query | 73.92% | +0.25 | 0.4719 | +0.0025 |

## 本地模型实测

Track B 使用校验过的量化 MiniLM 向量和 MS MARCO MiniLM 交叉编码器，单模型线程为 1。全量任务与工程测量并行运行时曾暂停，因此本表中的 P95 是原始墙钟读数。独立 10K 工程压测提供可复现的引擎延迟。

| 基准 | 轨道 | N | R@50 | R@10 | MRR@10 | Token | P95 ms | 有金标误空 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| locomo | track-a | 1986 | 75.63% | 61.22% | 0.4847 | 2006.5 | 379.74 | 0 |
| longmemeval | track-a | 500 | 89.63% | 71.14% | 0.5568 | 2037.3 | 1099.94 | 0 |
| membench99 | track-a | 99 | 97.98% | 82.17% | 0.6812 | 1116.2 | 334.29 | 0 |
| membench99 | track-b | 99 | 98.99% | 40.29% | 0.3868 | 403.1 | 8121.71 | 54 |
| membench99 | track-b-blended | 99 | 98.99% | 91.67% | 0.8301 | 1157.6 | 11977.04 | 0 |
| membench | track-a | 400 | 98.23% | 79.08% | 0.6788 | 1162.6 | 196.51 | 0 |
| membench | track-b | 400 | 98.31% | 36.08% | 0.3529 | 393.1 | 8830.67 | 217 |
| membench | track-b-blended | 400 | 98.31% | 87.25% | 0.7771 | 1188.0 | 7822.95 | 0 |

## 增量抽样

各行使用相同问题 ID。不同时间的主机负载不同，延迟列为原始观测。

| 基准 | 阶段 | N | Δ R@50 百分点 | Δ R@10 百分点 | Δ MRR | Δ Token | Δ P95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| locomo | Retain | 60 | +0.00 | +2.21 | -0.0126 | -3.5 | -512.16 |
| locomo | 扩散与通道 | 60 | +3.78 | -1.95 | +0.0077 | -14.8 | +31.81 |
| locomo | 确定性精排 | 60 | +0.00 | +0.30 | +0.0122 | +0.2 | +4.04 |
| longmemeval | Retain | 35 | +0.00 | +0.00 | +0.0000 | -0.3 | -603.15 |
| longmemeval | 扩散与通道 | 35 | +2.78 | +7.78 | +0.0380 | +0.8 | +22.78 |
| longmemeval | 确定性精排 | 35 | +0.00 | +0.00 | +0.0035 | -0.2 | +39.62 |
| membench | 确定性精排 | 48 | +5.07 | +0.64 | +0.0461 | +603.5 | -108.81 |
| locomo | 装填与日期补查 | 60 | +0.00 | +0.00 | +0.0000 | -3.6 | +207.80 |
| longmemeval | 装填与日期补查 | 35 | +0.00 | +0.00 | +0.0000 | -5.9 | +1429.93 |
| membench | 装填与日期补查 | 48 | +0.00 | +0.00 | +0.0000 | +0.0 | +126.99 |

## 当前代码同预算复跑

| 基准 | 题型 | 模式 | N | R@10 | MRR@10 | 预算覆盖 | Token | 更新支配 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| locomo | Overall | single_query | 1986 | 61.55% | 0.4918 | 74.49% | 2013.1 | — |
| locomo | Overall | reconstructive | 1986 | 62.36% | 0.4969 | 77.27% | 2008.2 | — |
| locomo | 1 multi-hop | single_query | 282 | 34.08% | 0.3453 | 51.98% | 2023.1 | — |
| locomo | 1 multi-hop | reconstructive | 282 | 35.45% | 0.3515 | 53.94% | 2023.7 | — |
| locomo | 2 temporal | single_query | 321 | 75.67% | 0.6251 | 83.07% | 1993.3 | — |
| locomo | 2 temporal | reconstructive | 321 | 75.99% | 0.6340 | 83.59% | 1982.4 | — |
| locomo | 3 open-domain | single_query | 96 | 31.89% | 0.2224 | 44.16% | 1983.3 | — |
| locomo | 3 open-domain | reconstructive | 96 | 31.89% | 0.2276 | 44.16% | 1968.7 | — |
| locomo | 4 single-hop | single_query | 841 | 68.61% | 0.5196 | 82.08% | 2016.9 | — |
| locomo | 4 single-hop | reconstructive | 841 | 69.52% | 0.5228 | 86.31% | 2012.7 | — |
| locomo | 5 adversarial | single_query | 446 | — | — | — | 2020.2 | — |
| locomo | 5 adversarial | reconstructive | 446 | — | — | — | 2017.0 | — |
| longmemeval | Overall | single_query | 500 | 70.72% | 0.5595 | 84.22% | 2038.2 | 52.86% |
| longmemeval | Overall | reconstructive | 500 | 71.14% | 0.5568 | 84.55% | 2037.3 | 50.00% |
| longmemeval | abstention | single_query | 30 | — | — | — | 2040.8 | — |
| longmemeval | abstention | reconstructive | 30 | — | — | — | 2039.9 | — |
| longmemeval | knowledge-update | single_query | 72 | 82.64% | 0.6729 | 95.83% | 2041.1 | 52.86% |
| longmemeval | knowledge-update | reconstructive | 72 | 80.79% | 0.6633 | 97.92% | 2041.5 | 50.00% |
| longmemeval | multi-session | single_query | 121 | 51.45% | 0.4382 | 73.10% | 2038.8 | — |
| longmemeval | multi-session | reconstructive | 121 | 50.87% | 0.4379 | 72.62% | 2035.9 | — |
| longmemeval | single-session-assistant | single_query | 56 | 89.29% | 0.7357 | 94.64% | 2034.5 | — |
| longmemeval | single-session-assistant | reconstructive | 56 | 91.07% | 0.7318 | 94.64% | 2033.5 | — |
| longmemeval | single-session-preference | single_query | 30 | 30.56% | 0.1079 | 55.56% | 2038.9 | — |
| longmemeval | single-session-preference | reconstructive | 30 | 30.56% | 0.1084 | 53.89% | 2041.9 | — |
| longmemeval | single-session-user | single_query | 64 | 90.62% | 0.6793 | 96.88% | 2036.9 | — |
| longmemeval | single-session-user | reconstructive | 64 | 92.19% | 0.6796 | 98.44% | 2033.2 | — |
| longmemeval | temporal-reasoning | single_query | 127 | 73.60% | 0.5792 | 84.02% | 2037.5 | — |
| longmemeval | temporal-reasoning | reconstructive | 127 | 75.17% | 0.5767 | 84.15% | 2038.4 | — |

## 工程门禁

| 版本 | P50 ms | P95 ms | P99 ms | SQL |
| --- | --- | --- | --- | --- |
| before | 73.022 | 78.021 | 78.731 | 15 |
| after | 73.670 | 78.648 | 79.364 | 15 |

| 规模 | 查询 | P95 ms | SQL min | SQL max |
| --- | --- | --- | --- | --- |
| 10000 | 60 | 84.132 | 15 | 15 |

- PostgreSQL 16.16 / pgvector 0.8.7 容器通过鉴权、会话恢复、任务显式提交和有日历元数据的 Recall。
- 60 道日历题全对。80 道版本题首位和分级均全对，串台 0。
- 证据核验的断言支持率与无证据拒答率均为 100%。过期事实误用为 0%。
- LoCoMo 前缀复用 90/90。10K 原路径和图路径分别为 78.65 ms 和 84.13 ms，均为 15 条 SQL。
