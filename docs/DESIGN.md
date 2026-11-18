# 设计说明

本文把规格里的五步链路对应到代码，并记下规格没有写死、v1 自己选定的做法。规格原文的目标分数只在 README 里作为作者目标列出，不在这里当作实测。

## 步骤对照

| 规格步骤 | 代码 | 行为 |
| --- | --- | --- |
| 1 毫秒级分流 | `memengine/router.py` `route_query` | 实体命中，或状态变更 / 时间回溯正则命中，或问句长于 12 且命中决策词，返回 `TASK_POISON_CIRCLE`；否则 `CASUAL_2D_BYPASS`。正则与规格伪代码一致。 |
| 2 渐近原点与双层时间轴 | `memengine/distance.py`、`memengine/timeline.py` | `d = max(eps, raw_dist)`，默认 `eps = 0.05`。层 1 记原点，层 2 记圈内 / 圈外条数。间隔 `<= Δt`（默认 10 秒）时合并，并按伪代码 `update` 载荷、把 `t_end` 推到当前时刻。 |
| 3 Z 轴坍缩与由远及近 | `memengine/zaxis.py` | `collapse_and_scan_z_axis` 按伪代码把输入列表反转后分桶，同一个事实 id 只出现一次，不同实体不进同一个桶。`d_asymp <= R` 为 `INNER_INTUITIVE`，否则为 `OUTER_ANTI_INTUITIVE`。 |
| 4 慢系统只读推理 | `prompts/system2_reasoner.txt`、`memengine/policy.py` `plan_answer` | 引擎只采用 `final_response` 和 `reasoning_trace`。返回体里若带有写入操作、新原点或新半径，一律丢弃。`answer` 不会把推理结果写回笔记本，也不会改毒圈半径。 |
| 5 3D 笔记本写入 | `prompts/notebook_layout.txt`、`memengine/notebook.py` | `OVERWRITE_SAME_LINE` 覆盖原行并归档旧值；`APPEND_ADJACENT_LINE` 在同一块内追加，渲染为单个 `\n`；`INSERT_ISOLATED_LINE` 与邻行之间空一行，渲染为 `\n\n`；`NOOP` 不改存储。 |

对外接口在 `memengine/engine.py`：`add`、`search`、`answer`。持久化是 SQLite（`memengine/store.py`）。三份提示词在 `prompts/`，占位符与规格相同，由 `memengine/prompts.py` 替换。

模型调用只有 `memengine/llm.py` 一个入口。`OpenAICompatibleClient` 读取 `LLM_BASE_URL`、`LLM_API_KEY`（或 `OPENAI_API_KEY`）、`LLM_MODEL`（默认 `Qwen2.5-7B-Instruct`），向 `{base}/chat/completions` 发温度 0 的请求。没有密钥时用 `MockLLMClient`：它读已经填好的提示词，再走同一套确定性策略，方便单测和 CI。解析失败时引擎退回这套策略，并累计 `fallback_count`。

## 规格留白时的选择

**原点不是数值平面。** 规格要求移动原点 `(x0, y0)`，提示词给出的却是主题、子维度和决策句。v1 把原点存成 `{x_topic, y_subtopic, decision_anchor}`，不建立欧氏坐标。

**`raw_dist` 的算法。** 规格只要求 `d = max(eps, raw_dist)`，没有定义 `raw_dist`。v1 在查询时计算：孤立行（`INSERT_ISOLATED_LINE`）取 `max(0.72, surprise)`，从而落在默认半径 0.45 和回溯半径 0.62 之外；其余行取 `0.5 * (1 - 字符二元组 Jaccard) + 0.5 * surprise`。`surprise` 来自排版策略：孤立 0.90，同行覆写 0.86，紧凑追加 0.28，新闻归档 0.12。字面距离为 0 时原始距离可以为 0，渐近距离仍被抬到 `eps`。

**实体命中多认冒号后面的名字。** 伪代码用完整键做子串匹配。库里的键是 `人物:老王` 这种形式，用户只会说「老王」。v1 在完整键之外，也用冒号后的实体名做子串匹配。

**由远及近的两次反转。** 提示词要求控制器输出 far-to-near。伪代码函数会把输入再反转一次。引擎把 far-to-near 先反转成 near-to-far 再交给该函数，扫描顺序仍是由远及近。控制器没排到的键放在远侧。直接命中的实体再沿共享 Z 键扩展两跳，用来接住「人 → 事件 → 预算」这种跨轴事实；新闻桶的键是 `话题:时事杂讯`，个人事实不挂这个键，扩展不会把新闻流卷进来。

**事实抽取不是第四个提示词。** 规格的三份提示词从「已经抽好的客观事实」开始。v1 用 `memengine/extract.py` 的模式完成这一步，不另加模型。寒暄和短句 `NOOP`。够长、又抽不出事实的句子，归档进主题「时事杂讯」，只挂低权重新闻键，这样 2D 平面上确实有噪声流，而 Z 轴检索仍按实体分桶。

**时间旅行靠行修订，不靠时间轴载荷。** 层 1 / 层 2 的载荷按规格只保存原点和圈内外计数，伪代码在窗口内还会用最新载荷覆盖旧载荷。预算这类「第二次修改之前」的原文因此另存在 `line_history`。回溯问句会把这些修订当作圈外事实交给慢系统，按修订号选取：第 N 次修改之前取修订 `N-1`（初次写入是修订 0）。这是为了让覆写之后仍能回答规格里的历史问题。

**异步写入是进程内队列。** `add(..., sync=False)` 先决定排版、把作业放进 `pending`，`drain()` 才落库。默认 `sync=True`，在本轮调用内写完。推理路径不调用 `drain`。v1 没有后台线程。

**闲聊旁路不扫毒圈。** 闲聊路径只做实体名哈希探测。路由已经保证这条路径没有实体命中，因此回复是一句固定短句，不附带记忆正文，也不调用模型。

**权重。** 控制器仍输出 `anti_intuitive_weight = 0.90`、`intuitive_weight = 0.40`。v1 的选择规则是字典序：同一属性上，圈外事实覆盖圈内事实，再比修订号。没有把两个权重乘成一个分数。

**嵌入只给基线 A。** 主路径不走向量。基线 A 使用 MD5 字符二元组哈希向量（`memengine/embeddings.py`），避免 `PYTHONHASHSEED` 导致结果漂移。这不是神经网络嵌入。

**基线 B。** `eval/baselines.py` 的 `CrudMemory` 是为本仓库写的实体键 CRUD：同属性且 Z 键相交就覆盖为最新值，回答时取问句里的实体，并带上一层共享键上的其他记录，另保留 8 轮原始窗口供上下文 token 统计。它不包含 Mem0、Letta 或其他记忆库的代码。

## 规格原文

仓库内 [SPEC.md](SPEC.md) 保留了与所有者 Google Doc 对应的规格正文，便于离线对照。提示词固定在仓库根目录 `prompts/`；`memengine/prompts.py` 会按该布局解析，并兼容将来把同名文件放进包内目录的安装方式。

## 已知缺口

- 抽取器只认识过敏、密码、代号、行程、预算、周会等模式。模式之外的个人事实会进新闻桶，之后按实体问也检索不到。
- 路由器是字符子串。规格里的 `别` 会命中「特别」，短实体名也会误命中。
- 时间轴在会话持续有写入时几乎收成一个节点，层 1 只留下会话里最后一次原点。细粒度历史不在时间轴上。
- 慢系统的离线实现会返回有实体路径的已存事实。问句里的错误谓词（例如把苯甲酸钠问成花生）不会被编造成真，但回复也不一定显式否定这个诱饵。
- 小规模语料里，罕见词本身就很显眼，朴素 RAG 仍可能把孤立事实排进 Top-5。淹没效果要到更大规模、更多近义噪声才会显现。
- 没有真正的后台写入线程，没有多用户，也没有主路径上的神经嵌入。
