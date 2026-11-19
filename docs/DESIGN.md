# 记忆引擎设计

这是实现契约。`SPEC.md` 是原始提案；其中的伪代码和目标数字不是当前实现或实测结果。

## 写入与 2D 排版

`CognitiveMemoryEngine.add` 按 `(user_id, agent_id, session_id)` 查候选段落。相似度最高的 8 个段落各保留最近 5 行，另保留所有命中实体的活跃行（不受这两个上限约束）。Live 只调用一次 `notebook_layout.txt`；原始消息与角色做 HTML 转义后放入 `untrusted_utterance`，候选和原文都标为数据。引擎验证操作类型、允许目标、目标属性和删除意图；不接受模型决定硬删除。提示词隔离是防护的一层，不把模型输出当作权限。

| 操作 | 存储与布局 |
| --- | --- |
| `OVERWRITE_SAME_LINE` | 保留行 ID 与行序，事务内归档旧版本、增加修订号、合并来源 step ID。 |
| `APPEND_ADJACENT_LINE` | 在同一主题追加相邻行。 |
| `INSERT_ISOLATED_LINE` | 为一次性约束或例外留空行；渲染时与邻行隔开。 |
| `DELETE_LINE` | 归档并将原行置为 `active=0`，设置 `valid_until`。 |
| `NOOP` | 不修改笔记本。 |

每个操作自带文本、属性和值，不与另一份事实列表按位置配对。`plain` 来自清洗后的 `formatted`；检索文本可以另加来源 step 标记。Mock 及解析失败的回退用 `extract_facts` 与 `decide_layout`；规则识别不了的长中文杂讯可进入单独新闻桶。默认 12 类加 `其他/通用`；`categories=[...]` 扩展允许列表，默认类目的中英文同义词映射到同一个键。

## 读入、分区与排序

1. **路由。** 纯寒暄且无已知实体、疑问或记忆线索时直接返回 `CASUAL_2D_BYPASS`，不调用任何模型，也不触发待写队列。其他输入进入 `query_origin_controller.txt`：一次返回查询类型（`FACTUAL`、`REFLECTIVE`、`TIME_TRAVEL`）、实体、类目、语义扩写、原点和毒圈配置。旧 `query_parser.txt` / `origin_boundary.txt` 保留为兼容模板，不参与主链路。无实体命中时，控制器只看向量预检 Top-10 导出的最多 15 个实体键和有限主题摘要。
2. **召回。** 单次读取作用域内的活跃行快照。实体键做至多 2-Hop；类目只在没有实体候选时做直接召回，不能当图边，新闻键也不参与扩展。语义扩写 Top-30 与 Z Hash 用 `0.65/0.35` 加权 RRF（`k=60`，用理论最大分归一化）。`TIME_TRAVEL` 可从相关行加入 `line_history` 修订。删除行不进检索，历史仍可显式审计。
3. **距离与惊奇解耦。** `d_origin = 1 - max(sim(扩写, 事实), sim(原点三元组, 事实))`。远程为截断到 `[0,1]` 的余弦相似度；离线融合 BM25、哈希 TF-IDF 余弦、词元与字粒度匹配；泛类目可用同义归一化辅助离线召回。`d_asymp = max(0.05, d_origin)`；`P_radial = 0.05 / d_asymp`。`S` 仅取结构标签：冲突/历史 `0.95`，孤立行 `0.90`，普通覆写 `0.85`，相邻行 `0.25`，新闻 `0.08`。
4. **Z 轴坍缩与三态分区。** 直接消费控制器的由远及近顺序；未排到的轴放远侧。同一事实 ID 只分给最先扫到的一个桶。实体图命中、历史回溯、或扩写查询通道的相似度 `> 0.15` 才算有锚点；候选决定的原点只参与距离，不能反过来为自己制造关联。有锚点且显式冲突/历史，或有锚点、`d_asymp > R` 且 `S >= 0.75`，为 `OUTER_ANTI_INTUITIVE`。非孤立、非冲突的行只有在有锚点且 `d_asymp <= R` 时才进 `INNER_INTUITIVE`。其他情况是 `POISONED_DROPPED`；例如圈内、未标冲突的孤立行也不满足两个入选条件。无关文本不会用“最近记忆”兜底。
5. **排序与配额。** 圈内用 `w_in * (0.6 * RRF_norm + 0.4 * P_radial) + 0.15 * Recency`，圈外用 `w_anti * (0.45 * RRF_norm + 0.55 * S) + 0.15 * Recency`。权重用控制器值，默认 `0.40` 和 `0.90`；半径限制到 `[epsilon, 1]`。`Recency = 2^(-max(0, now-event_time)/30天)`，缺失事件时间时用更新时间。圈外最多 `ceil(top_k*0.4)`，有历史回溯或已标冲突时放宽到 `ceil(top_k*0.6)`。两个区各按分数排序；当两个区都可容纳时，在 Token 预算里为至少一条圈内背景留空间。`token_budget` 约束返回的证据正文估算，不含系统指令或元数据的协议开销。

配置 `EMBEDDING_BASE_URL`（基础路径或完整 `/embeddings` 路径）和 `EMBEDDING_MODEL` 时使用 `OpenAICompatibleEmbedder`；`EMBEDDING_API_KEY` 可选，取决于服务是否要求鉴权。当前行向量在进程内按行 ID、修订和文本摘要缓存（最多 4096），本地硬删除清除对应缓存；无服务或请求失败时整批降级 `HashedTfidfEmbedder`，不混用不同向量空间。远程服务会接收查询和该作用域的记忆文本，接入前需评估数据边界。

## 隔离、时钟与持久化

- `add(text, *, user_id='default', agent_id='default', session_id='default', role='user', timestamp=None, event_time=None, sync=True)`。默认同步落盘；`sync=False` 暂存在当前实例内，任务查询默认先刷对应用户/Agent 的待写作业。纯寒暄和 `read_only=True` 不刷盘；待写队列不跨进程持久化。
- `retrieve(query, *, user_id='default', agent_id='default', session_id=None, top_k=10, token_budget=2048, timestamp=None, read_only=False, flush_pending=True)`。`search` 为兼容封装。`session_id=None` 跨当前用户/Agent 的会话查记忆；指定会话则只查该会话。控制状态按用户、Agent、会话独立保存；跨会话查询用该租户的默认控制流。`timestamp` 只影响当次调用，不是 SQLite 快照的 as-of 过滤器；时间旅行使用行修订。显式 `set_time` 固定时钟，`reset_clock` 恢复构造时注入的时钟或系统时钟。
- `delete(line_id, *, hard=False, user_id='default', agent_id='default', timestamp=None)`。硬删除只覆盖本库中的 `lines`、`line_zkeys`、`line_history` 和当前实例向量缓存；独立的原始 ingest 记录、历史备份、其他进程缓存及远程供应商留存不在此契约范围内，不能把它当作端到端的数据擦除。
- `answer(query, *, user_id='default', agent_id='default', timestamp=None, retrieved=None)`。复用同租户的检索结果，或发起只读检索；不刷盘、不新建时间节点、不移动持久化原点。只使用 `evidence_synthesis`、`anti_intuitive_override`、`rejected_noise_ids`、`final_answer`。选择题适配在 `eval/` 而不在理由生成器。寒暄返回空 `response`，由调用方接管。
- 文件型 SQLite 使用 WAL、`synchronous=NORMAL`、`busy_timeout=5000`、外键和幂等迁移；`:memory:` 使用 SQLite 内存日志。行、Z 键批量读取，写事务用 `BEGIN IMMEDIATE`，被更新的修订不会丢失旧版本。原始历史 Token 在入库时按用户/Agent 增量累计。
- 双时间轴分别记录原点移动、写入执行。默认仅在间隔不超过 10 秒、节点时长不超过 300 秒且未满 20 个事件时合并；原点迁移或主题切换新开节点。SQLite 仅追加或更新最后一个节点。
