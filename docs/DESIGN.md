# 记忆引擎设计

这是实现契约。`SPEC.md` 是原始提案；其中的伪代码和目标数字不是当前实现或实测结果。

## 写入与 2D 排版

`CognitiveMemoryEngine.add` 先写入 `raw_turns`，保存包括空白和外部 step 标签的原文，返回自增 `turn_id`。随后按 `(user_id, agent_id, session_id)` 查候选段落。相似度最高的 8 个段落各保留最近 5 行，另保留所有命中实体的活跃行。Live 通过 `notebook_layout.txt` 一次抽取和排版，提供最近同会话上下文供指代消解，并以当前轮次的 UTC 日期解析相对时间。原始消息与角色做 HTML 转义后放入 `untrusted_utterance`。引擎验证操作类型、允许目标、目标属性和删除意图。硬删除只能通过显式接口触发。

| 操作 | 存储与布局 |
| --- | --- |
| `OVERWRITE_SAME_LINE` | 同一块、同一行序生成新事实 ID，继承 `root_fact_id`，以 `supersedes_id` 链接旧版本、增加修订号并对来源 turn ID 和 step ID 分别求并集。旧版本封口为 `superseded`，保留索引。 |
| `APPEND_ADJACENT_LINE` | 在同一主题追加相邻行。 |
| `INSERT_ISOLATED_LINE` | 为一次性约束或例外留空行；渲染时与邻行隔开。 |
| `DELETE_LINE` | 将活跃版本置为 `deleted` 并设置 `valid_until`，整条链从普通检索隐藏。 |
| `NOOP` | 确认重复事实时，仅给命中版本补充来源 turn ID 和 step ID。 |

每个操作自带文本、属性和值。`plain` 来自清洗后的 `formatted`，新事实都绑定原文 `turn_id`。Mock 及解析失败的回退用 `extract_facts` 与 `decide_layout`。类目是可选软标签，`categories=[...]` 扩展标签词表，自由实体和未知标签均可保存。

## 读入、分区与排序

1. **路由。** 纯寒暄且无已知实体、疑问或记忆线索时直接返回 `CASUAL_2D_BYPASS`。任务查询在本地解析意图、扩展语义查询并运行 `parse_temporal_intent`，从候选预览和实体线索计算原点与边界。检索控制全程无需 LLM。
2. **召回。** 在一个 SQLite 读取快照中，向量、FTS5 BM25、实体倒排和时间四路各最多取 40 条，权重依次为 `0.40 / 0.35 / 0.15 / 0.25`，以 `k=60` 做加权 RRF。向量以 NumPy 批量计算余弦；全文以英文代码词元和 CJK 单字、相邻双字索引；实体键去前缀、去空白、大小写归一后直接查一个倒排桶。时间路仅在识别历史意图或 UTC 时间窗口时启动，用 `event_time_start` 范围和可选修订号查询。候选详情和键集中两条 SQL 加载。普通查询的 `superseded` 版本乘 `0.35` 并加 `[历史已变更]`；历史查询恢复旧版本权重，明确窗口内有匹配时降低窗外候选权重。
3. **巩固频次坐标。** `c_m = max(1, len(source_step_ids), len(source_turn_ids)) + hit_count`，`c_max` 取当前 RRF 池非杂讯最大频次。新闻主题或仅有新闻 Z 键时，`r_m=0.5`；`c_m=1`、孤立、覆写、冲突或历史修订时，`r_m=1`；其余事实用 `clamp(1 - ln(c_m)/ln(c_max), 0, 0.50)`，`c_max<=1` 时取 `0`。`c_m=c_max=8` 时精确落在 `r_m=0`；`c_m=2, c_max=8` 时被截在 `r_m=0.50`。
4. **对称引力与三区。** `W(r) = max(eps/max(eps,r), eps/max(eps,1-r))`，两端默认 `eps=0.05`：`W(0)=W(1)=1`，`W(0.5)=0.10`。`R_in=clamp(R,0.15,0.35)`，`R_out=1-R_in`。`r<=R_in` 为 `INNER_INTUITIVE`，`r>=R_out` 为 `OUTER_ANTI_INTUITIVE`，中间为 `POISON_BELT`。由远及近分 Z 桶，同一事实 ID 只分给最先扫到的桶，无分区配额。
5. **全局排序与夹层软裁剪。** `Rel_norm=RRF/(max_pool_RRF+1e-12)`，`bipolar_score = Rel_norm * (lambda + (1-lambda) * W) + beta`。`bipolar_lambda` 默认 `0.50`；显式冲突或时间旅行历史修订加 `0.05`。按分数降序，同分保持 RRF 顺序。仅裁掉 `bipolar_score < belt_prune_ratio * max_score` 的夹层事实，默认比例 `0.35`，`0` 禁用；按原排序恢复到 `min(min_keep, 候选数)`，`min_keep` 默认 `2`。参数可在构造或单次检索指定。两极不参与软裁剪；全员普通单次时与纯 RRF 同序。
6. **预算与诊断。** 软裁剪后按 `top_k` 与估算 `token_budget` 截断；正预算下至少保留首条可容纳的事实，若无完整条目可容纳则截短首条。空候选、寒暄或显式零预算仍可能返回空。`ranked` 保留全局顺序，`outer` / `inner` / `belt` 是分区视图；`belt_pruned_count` 单计软裁剪，`belt_compressed_count` 还包括预算或槽位移除的夹层事实。外围计数包含普通单次事实，不等于显式例外数。默认 `lambda=0.50` 时，同等 RRF 相关性的夹层分数是两极的 `55%`，高于 `35%` 裁剪线；高相关夹层因此会保留，这不是强制压缩配额。

配置 `EMBEDDING_BASE_URL`（基础路径或完整 `/embeddings` 路径）和 `EMBEDDING_MODEL` 时使用 `OpenAICompatibleEmbedder`，`EMBEDDING_API_KEY` 由服务端要求决定。`fact_vectors` 按版本保存模型名、维度和 L2 归一的 float32 向量。默认使用本地 `HashedTfidfEmbedder`，远程写入编码失败时保存本地向量；查询只对成功编码的同模型同维度向量计算相似度。模型变更后，已存的其他远程模型向量需要重新索引才能进入向量通道。FTS5、实体和时间通道独立可用。

## 隔离、时钟与持久化

- `add(text, *, user_id='default', agent_id='default', session_id='default', role='user', timestamp=None, event_time=None, sync=True)`。原文同步持久化。`sync=False` 仅将抽取与事实索引加入当前进程队列，非只读任务查询默认先刷对应用户/Agent 的作业。
- `retrieve(query, *, user_id='default', agent_id='default', session_id=None, top_k=10, token_budget=2048, timestamp=None, read_only=False, flush_pending=True, include_raw_turns=False)`。`search` 是兼容封装。指定 `session_id` 时限定会话。打开 `include_raw_turns` 时，`drilldown_turns` 返回前三条事实所指向原文及同一会话的相邻轮，去重并按 `turn_id` 排序。`timestamp` 是本次调用的时钟参考，历史检索基于事件时间及修订。`set_time` 固定时钟，`reset_clock` 恢复原时钟。
- `delete(line_id, *, hard=False, user_id='default', agent_id='default', timestamp=None)`。硬删除覆盖整条版本链及其实体、FTS5 和向量索引。独立 `raw_turns` 日志仍可经原文接口读取，数据擦除若涵盖原文、备份或远程留存，需要分别处理。
- `answer(query, *, user_id='default', agent_id='default', timestamp=None, retrieved=None)`。复用同租户的检索结果，或发起只读检索；不刷盘、不新建时间节点、不移动持久化原点或增加命中次数。System 2 按 `ranked` 接收证据；仍可拒绝新闻及无实体锚点、弱语义关联的冲突或孤立证据，不反写检索结果。选择题适配留在 `eval/`。寒暄返回空 `response`，由调用方接管。
- 非只读检索批量递增最终选中事实版本的 `hit_count`。`facts` 的两个时间视角分别是 `event_time_start/end` 与 `ingest_time/valid_until`，后者用于记录版本入库和失效。`get_fact_chain` 可按修订顺序审计；`render` 仅看活跃行。
- 文件型 SQLite 使用 WAL、`synchronous=NORMAL`、`busy_timeout=5000`、外键，写事务使用 `BEGIN IMMEDIATE`。打开旧库时迁移 `ingest`、`lines`、`line_history`、`line_zkeys`，事务内校验并重建索引，使用 `meta.schema_version=2` 记录版本。原始历史 Token 在入库时按用户/Agent 增量累计。
- `timeline` 仅保留写入审计兼容，默认节点合并间隔 10 秒、时长 300 秒、最多 20 个事件。`observations` 提供按租户插入和读取接口，`add()` 不生成反思。
