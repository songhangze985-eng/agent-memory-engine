# 双数据库与双模回忆验证报告

2026-11-20 本机实跑。基线为 130 项单测。改造后 `python3 -m pytest -v` 共 144 项通过，含 PostgreSQL 实连，退出码 0。`eval.regression_guard --full-static` 退出码 0。

## 冻结证据检索

SQLite、mock、hashed。LoCoMo、LongMemEval-S 和 MemBench 400 使用固定种子 42。MemBench 99 使用冻结种子 42–46。LoCoMo 与 LongMemEval-S 的官方快照 SHA-256 及全部样本证据 ID 均由门禁核对。MemBench 400 从已有缓存按四类各抽 100 题。

| 基准 | 题数 | R@5 | R@10 | MRR@10 | 平均 Token | 误空 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoCoMo | 1986 | 44.72% | 50.06% | 0.3927 | 473.1 | 0/1540 |
| LongMemEval-S | 500 | 61.79% | 70.87% | 0.5425 | 1933.8 | 0/470 |
| MemBench 99 | 99 | 60.56% | 80.08% | 0.6999 | 529.3 | 0/99 |
| MemBench 400 | 400 | 58.38% | 77.80% | 0.6589 | 549.9 | 0/400 |

LongMemEval-S 全量更新支配为 42/70。Exact 120 的 90 道有答案题均首位命中，30 道缺失题全部空返。3,500 轮长程夹具中，单次禁忌和 5 个冷层旧版本均首位命中，平均注入 274.7 Token。200 次隔离查询零泄漏，30 条载荷零越权篡改与只读写入。

首轮全量门禁发现两道 LongMemEval-S 的助手建议被误识别为临时状态。将自动推断限制到用户陈述中的同一子句后，冻结指标恢复。显式有效期仍可用于其他角色的写入。

## SQLite 增量向量缓存

同一个 SQLite 文件、同一组 60 个精确标识符查询，每种配置先热身一次。单位为毫秒。SQL 统计为单次 `retrieve()` 的顶层语句数。

| 活跃记录 | 关闭 P50 / P95 / P99 | 开启 P50 / P95 / P99 | SQL |
| ---: | ---: | ---: | ---: |
| 1,000 | 17.959 / 18.893 / 21.650 | 13.823 / 14.125 / 17.048 | 15 → 15 |
| 5,000 | 58.634 / 63.233 / 75.695 | 40.375 / 43.464 / 44.341 | 15 → 15 |
| 10,000 | 110.961 / 114.411 / 121.643 | 73.651 / 77.218 / 77.315 | 15 → 15 |

10K 记录的配对 P95 下降 32.5%，上一版独立运行的 P95 为 111.757 ms。缓存仍从 SQL 获取当次符合范围、状态及有效期的候选键，暖查询复用连续 float32 矩阵，外部提交使本地视图失效。

## LoCoMo Multi-hop 回忆模式

同一组官方 282 道多跳题。同一对话、同一事实库、交错运行单次与重构模式。两组共用 Top-10 和 2048 Token 预算，耗时统计为检索阶段，包含二跳的首轮与补召回。

| 模式 | 题数 | R@10 | MRR@10 | 平均 Token | P50 ms | P95 ms | 二跳触发 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `single_query` | 282 | 25.05% | 0.2715 | 490.6 | 37.366 | 60.124 | 0 |
| `reconstructive` | 282 | 25.05% | 0.2714 | 491.2 | 61.326 | 100.352 | 182 |

有明确版本目标的桥接夹具能补回二跳证据。官方多跳集未见 Recall 净增，平均 Token 和延迟上升。默认保持 `single_query`。

## 实连与部署

- PostgreSQL 16.2、pgvector 0.6.2、pg_trgm 1.6。本机实连实测了 HNSW 索引、全文、时效过滤、版本链、冷热迁移、多租户隔离及 SQL 数组降级。后者使用同一实库关闭扩展分支验证。
- FastAPI、标准库客户端及 CLI 通过本机回环连接跑通六个操作。Bearer 鉴权验证了未授权的 401 响应。
- Docker Compose 配置解析通过。镜像拉取耗时后改用本机 PostgreSQL 二进制完成实连验证。容器整栈启动未验收。

## 复现

```bash
TEST_POSTGRES_DSN=postgresql://user:password@localhost/test python3 -m pytest -v
bash scripts/run_static_matrix.sh /path/to/results /path/to/locomo10.json \
  /path/to/longmemeval_s_cleaned.json /path/to/MemData /path/to/previous-99.json
python3 -m eval.regression_guard --results /path/to/results --full-static
python3 -m eval.engineering_bench --out /path/to/engineering.json
python3 -m eval.reconstructive_bench --data-dir /path/to/locomo10.json --out /path/to/reconstructive.json
```
