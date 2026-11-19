# MemBench 分层抽样实测

本次为 mock 模式：作答头是确定性的词面重叠选择器，引擎 answer() 走离线规则。这些数字不是记忆质量分数。

- 数据：https://github.com/import-myself/Membench/tree/main/MemData ，许可证说明：README 徽章写着 MIT（https://opensource.org/licenses/MIT），仓库里没有 LICENSE 文件，GitHub 也未登记许可证。
- 划分：官方 MemData 四类（participation/observation × factual/reflective），不是论文 Sub-dataset 1/2。
- 抽样：每类最多 20 条，种子 42，实际 80 条。
- 模式：`mock`，模型 `None`。

| 系统 | 参与·事实 | 参与·反思 | 观察·事实 | 观察·反思 | 总体准确率 | Recall@10 | 写入 ms/次 | 读取 ms/次 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 引擎 answer() | 30.0% | 35.0% | 50.0% | 25.0% | 35.0% | — | 7.319 | 22.131 |
| 引擎检索 + 同一作答头 | 30.0% | 35.0% | 50.0% | 25.0% | 35.0% | 56.7% | 7.319 | 22.131 |
| 无记忆基线 | 15.0% | 35.0% | 25.0% | 20.0% | 23.8% | — | — | — |
| 全上下文基线 | 30.0% | 35.0% | 50.0% | 25.0% | 35.0% | — | — | — |

准确率：抽出的选项字母与 `ground_truth` 相同。Recall@10 只对检索结果计算：前 10 个命中步骤里，证据步骤被覆盖的比例。
写入时间是 `add` 的平均墙钟时间，读取时间是 `search` 的平均墙钟时间，对应论文里记忆写入 / 检索的效率，不含作答模型时间。
容量分箱（同一作答头，按写入 token 从少到多分成三段）见 JSON 的 `capacity_bins`。
token 数为字符估计（中文 1 字约 1 token，其余约 4 字符 1 token）。本次客户端没有返回计费，`cost_usd` 为空。

论文 Table 5（Qwen2.5-7B-Instruct，Sub-dataset 1）的报告值写在 JSON 的 `paper_reported_not_this_run`，不是上表。
