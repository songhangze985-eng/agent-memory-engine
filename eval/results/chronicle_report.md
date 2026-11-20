# 真实日历、版本精度与记忆闭环实跑报告

2026-11-20 · Python 3.12 · 零 API `mock` / `hashed`。SQLite 跑评测，pytest 另覆盖 PostgreSQL 16.2 实连、SQL 合同和 HTTP 接口。两套官方全量用 seed 42、4 个 worker；全部使用真实会话日历、说话人与提问参考时间。报告的 Token 和延迟按实际装配、检索记录计算。

## 实现与验证

| 能力 | 实现位置 | 验证 |
| --- | --- | --- |
| 现实发生区间与提及时间 | `memengine/chrono.py`、`storage/chronicle.py` 的 `populate_chronicle` 与 `occurrence_candidates` | 日历 60/60。含中英会话格式、相对时间、跨月、事件间隔 |
| 版本粒度与槽位修订 | `memengine/versioning.py` 的 `score_version_match`、`Chronicle.revise_slot` | 80/80 首位正确，80/80 分级正确，跨版本串台 0 |
| 规范实体、原话和关系边 | `Chronicle.canonical_entity`、`link_chronicle`、`retrieval/activation.py` | 真实 `raw_turns` 强指针、四类边、两跳限制、跨租户隔离 |
| 预算召回、实体保鲜与只读核验 | `CognitiveMemoryEngine.recall` / `reflect`、`consolidator.py`、`reflection.py` | 150 项 pytest 通过。只读核验最多三轮，引用可追溯至检出记忆与原话 |

## LoCoMo 全量 1,986 题

官方 `snap-research/locomo` 的 `locomo10.json`，SHA-256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`。前十条事实用于 Recall@10 和 MRR@10，全部已装配事实用于 Recall@Budget。两个新模式都受 2,048 Token 约束，证据按官方 `dia_id` 对照。无金证据的题不纳入召回或 MRR 分母。

| 题型 | 模式 | 题数 | Recall@10 | Recall@Budget | MRR@10 | 平均 Token | P95 ms | 可回答误空 | 更新支配 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | 历史 Top-10 | 1986 | 50.06% | — | 0.3927 | 473.1 | 78.91 | 0 | — |
| Overall | single_query | 1986 | 57.95% | 73.41% | 0.4510 | 1977.5 | 64.81 | 0 | — |
| Overall | reconstructive | 1986 | 57.79% | 76.94% | 0.4451 | 2013.4 | 95.33 | 0 | — |
| 1 multi-hop | 历史 Top-10 | 282 | 25.05% | — | 0.2715 | 490.6 | 80.05 | 0 | — |
| 1 multi-hop | single_query | 282 | 30.77% | 51.15% | 0.2916 | 1999.1 | 65.12 | 0 | — |
| 1 multi-hop | reconstructive | 282 | 29.08% | 53.52% | 0.2985 | 2023.3 | 93.22 | 0 | — |
| 2 temporal | 历史 Top-10 | 321 | 61.66% | — | 0.4791 | 442.1 | 76.26 | 0 | — |
| 2 temporal | single_query | 321 | 71.29% | 80.43% | 0.5790 | 1930.8 | 59.04 | 0 | — |
| 2 temporal | reconstructive | 321 | 71.70% | 81.54% | 0.5700 | 1988.8 | 82.82 | 0 | — |
| 3 open-domain | 历史 Top-10 | 96 | 25.67% | — | 0.1662 | 467.4 | 81.25 | 0 | — |
| 3 open-domain | single_query | 96 | 30.26% | 42.63% | 0.2128 | 1958.1 | 65.20 | 0 | — |
| 3 open-domain | reconstructive | 96 | 24.28% | 41.07% | 0.1859 | 1987.6 | 90.12 | 0 | — |
| 4 single-hop | 历史 Top-10 | 841 | 56.68% | — | 0.4251 | 478.6 | 78.95 | 0 | — |
| 4 single-hop | single_query | 841 | 65.00% | 81.57% | 0.4816 | 1981.8 | 64.64 | 0 | — |
| 4 single-hop | reconstructive | 841 | 65.77% | 86.96% | 0.4750 | 2017.7 | 91.24 | 0 | — |
| 5 adversarial | 历史 Top-10 | 446 | — | — | — | 475.0 | 78.74 | 0 | — |
| 5 adversarial | single_query | 446 | — | — | — | 1993.5 | 65.75 | 0 | — |
| 5 adversarial | reconstructive | 446 | — | — | — | 2022.2 | 108.60 | 0 | — |

- 全量固定前十相对历史的变化为 +7.89 个百分点（单路）和 +7.74 个百分点（图模式）。图模式总体在同预算下变化 +3.53 个百分点。
- 282 道多跳题，前一轮两种紧凑模式的 Recall@10 均为 25.05%。本轮图模式的 Recall@10 为 29.08%，相对历史 +4.03 个百分点，相对本轮单路 -1.69 个百分点。图模式 Recall@Budget 达到 53.52%，达到 52% 目标；同预算净增 +2.37 个百分点，低于要求的 +10%。
- 多路径聚合实验把多跳 Recall@Budget 推到 54.17%，净增 3.02 个百分点，同时将该组 Recall@10 降至 28.96%，总体 MRR@10 为 0.4410。本轮保留最强路径扩散实现。完整实验输出在日志包内。
- 预算由实际选中全量平均 53.0 / 50.9 条事实消耗至 1,977.5 / 2,013.4 Token。全量延迟采用各自记录的 P95，跑批期间有同机并行评测。

原始结果文件 `locomo-verified.json`、`legacy/full-locomo.json`、`locomo-multipath-trial.json`。

## LongMemEval-S 全量 500 题

官方 `xiaowu0162/longmemeval-cleaned` 的 `longmemeval_s_cleaned.json`，SHA-256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`。会话日期为 `haystack_dates`，提问以 `question_date` 为相对时间锚点。500 题中 76 题的数据包含晚于提问日期的会话，44 题包含晚于提问日期的标注答案会话。两组使用同一官方 haystack。提问日期用于解析与临时规则的时效，评测保持官方提供的证据全集。

| 题型 | 模式 | 题数 | Recall@10 | Recall@Budget | MRR@10 | 平均 Token | P95 ms | 可回答误空 | 更新支配 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | 历史 Top-10 | 500 | 70.87% | — | 0.5425 | 1933.8 | 125.29 | 0 | 42/70 (60.00%) |
| Overall | single_query | 500 | 70.84% | 85.00% | 0.5607 | 2044.5 | 154.12 | 0 | 37/70 (52.86%) |
| Overall | reconstructive | 500 | 66.77% | 82.09% | 0.5238 | 2044.5 | 203.86 | 0 | 36/70 (51.43%) |
| temporal-reasoning | 历史 Top-10 | 127 | 68.71% | — | 0.5209 | 1965.6 | 116.52 | 0 | — |
| temporal-reasoning | single_query | 127 | 73.33% | 85.72% | 0.5745 | 2044.3 | 148.31 | 0 | — |
| temporal-reasoning | reconstructive | 127 | 66.48% | 81.50% | 0.5374 | 2044.6 | 176.09 | 0 | — |
| multi-session | 历史 Top-10 | 121 | 52.19% | — | 0.4090 | 1967.8 | 125.29 | 0 | — |
| multi-session | single_query | 121 | 50.66% | 73.54% | 0.4415 | 2044.6 | 150.91 | 0 | — |
| multi-session | reconstructive | 121 | 46.61% | 70.11% | 0.3898 | 2044.4 | 191.53 | 0 | — |
| knowledge-update | 历史 Top-10 | 72 | 88.43% | — | 0.6704 | 1929.7 | 105.43 | 0 | 42/70 (60.00%) |
| knowledge-update | single_query | 72 | 85.88% | 97.22% | 0.6933 | 2044.8 | 141.48 | 0 | 37/70 (52.86%) |
| knowledge-update | reconstructive | 72 | 81.25% | 95.14% | 0.6568 | 2044.4 | 168.69 | 0 | 36/70 (51.43%) |
| abstention | 历史 Top-10 | 30 | — | — | — | 1919.6 | 103.43 | 0 | — |
| abstention | single_query | 30 | — | — | — | 2044.8 | 134.66 | 0 | — |
| abstention | reconstructive | 30 | — | — | — | 2044.2 | 173.84 | 0 | — |
| single-session-assistant | 历史 Top-10 | 56 | 89.29% | — | 0.7355 | 1883.8 | 122.42 | 0 | — |
| single-session-assistant | single_query | 56 | 89.29% | 94.64% | 0.7289 | 2044.7 | 162.64 | 0 | — |
| single-session-assistant | reconstructive | 56 | 91.07% | 94.64% | 0.7455 | 2044.1 | 189.14 | 0 | — |
| single-session-preference | 历史 Top-10 | 30 | 36.67% | — | 0.1643 | 1692.8 | 85.46 | 0 | — |
| single-session-preference | single_query | 30 | 28.89% | 55.56% | 0.1197 | 2044.4 | 128.51 | 0 | — |
| single-session-preference | reconstructive | 30 | 18.33% | 43.33% | 0.0712 | 2045.2 | 157.97 | 0 | — |
| single-session-user | 历史 Top-10 | 64 | 90.62% | — | 0.7021 | 1974.6 | 146.42 | 0 | — |
| single-session-user | single_query | 64 | 90.62% | 96.88% | 0.6692 | 2044.5 | 187.78 | 0 | — |
| single-session-user | reconstructive | 64 | 90.62% | 98.44% | 0.6183 | 2044.6 | 240.38 | 0 | — |

- 单路 Recall@10 由 70.87% 到 70.84%，变化 -0.03 个百分点。单路时间推理 73.33%，多会话 50.66%。单路 MRR@10 由 0.5425 到 0.5607。
- 图模式 Recall@10 为 66.77%，相对单路 -4.06 个百分点。其预算内 Recall 为 82.09%，单路为 85.00%。知识更新支配由 42/70 (60.00%) 到 37/70 (52.86%)（单路）和 36/70 (51.43%)（图模式）。全量显著提升和更新类改善均未达成。
- 前十固定为单路，再用图补尾的离线重放会保住原单路头部，LongMemEval-S 全量预算内 Recall 仍低于单路。这项重放按固定事实条数模拟，未计真 Token，未纳入交付排序。

原始结果文件 `longmemeval-verified.json`、`legacy/full-long.json`。

## MemBench、冻结回归与 10K 工程

MemBench 99 题使用 seed 42–46 的冻结队列并验证题目与证据 ID。MemBench 400 使用 seed 42 的已有官方对话缓存矩阵。表中为零 API 的检索归因分数。历史冻结跑法以 Top-10 与 2,048 Token 上限校验同口径回归。

| 题库 | 系统 | Recall@5 | Recall@10 | MRR@10 | 可回答误空 | 平均 Token |
| --- | --- | --- | --- | --- | --- | --- |
| 原 99 题 | Full System | 60.56% | 80.08% | 0.6999 | 0 | 529.3 |
| 原 99 题 | Pure RRF | 57.64% | 80.08% | 0.6943 | 0 | 545.2 |
| 缓存 400 题 | Full System | 58.38% | 77.80% | 0.6589 | 0 | 549.9 |
| 缓存 400 题 | Pure RRF | 55.54% | 77.80% | 0.6540 | 0 | 561.3 |

工程基准是 SQLite 与本机哈希索引的暖查询，每个规模执行 60 次确定性标识符检索。以下为独占跑批的测量。

| 活跃事实 | 向量缓存 | P50 ms | P95 ms | P99 ms | SQL 范围 |
| --- | --- | --- | --- | --- | --- |
| 1000 | 关闭 | 18.233 | 19.008 | 30.850 | 15–15 |
| 1000 | 开启 | 14.058 | 14.597 | 15.944 | 15–15 |
| 5000 | 关闭 | 61.929 | 71.144 | 73.137 | 15–15 |
| 5000 | 开启 | 42.105 | 47.535 | 60.320 | 15–15 |
| 10000 | 关闭 | 115.924 | 129.276 | 135.040 | 15–15 |
| 10000 | 开启 | 75.828 | 80.417 | 83.271 | 15–15 |

10K 缓存查询 P95 历史为 77.218 ms，本轮为 80.417 ms。每次检索执行 15 条 SQL。该基准测量普通检索；全量官方的图与日历查询延迟在上表单列。


原始结果文件 `legacy/03-membench.json`、`legacy/full-mem.json`、`engineering-isolated.json`。压力、精确标识和历史冻结矩阵见 `legacy/`。最新门禁见 `verified-regression-guard.log`，退出码为 0。专项结果见 `chrono-release.json` 与 `version_precision-release.json`。

## 复现

```bash
python3 -m pytest -v
python3 -m eval.chrono_eval --out /tmp/chrono.json
python3 -m eval.version_precision_eval --out /tmp/version.json
python3 -m eval.chronicle_bench locomo --seed 42 --workers 4 \
  --data-dir /path/to/locomo10.json --out /tmp/chronicle-locomo.json
python3 -m eval.chronicle_bench longmemeval --seed 42 --workers 4 \
  --data-dir /path/to/longmemeval_s_cleaned.json --out /tmp/chronicle-long.json
OPENBLAS_NUM_THREADS=1 python3 -m eval.engineering_bench --out /tmp/engineering.json
bash scripts/run_static_matrix.sh /path/to/results /path/to/locomo10.json \
  /path/to/longmemeval_s_cleaned.json /path/to/MemData /path/to/frozen-99.json
python3 -m eval.regression_guard --results /path/to/results
```

设置 `TEST_POSTGRES_DSN` 可执行 PostgreSQL 实连契约。日志包收录完整标准输出、标准错误及退出码，保留调试时发生的失败和重跑；JSON 包含逐题评分与真实证据映射。历史前十结果来自上一轮正式归档的 `matrix/full-locomo.json`、`matrix/full-long.json` 与 `reconstructive-release.json`。
