# Hangze 提供的认知记忆引擎规格（原文要点，完整保留）

来源：仓库所有者提供的规格说明（对应 Google Doc
`https://docs.google.com/document/d/1LEKYBqyd1SHkCChLorU3MOyotBaRUte0hVPLhzL-uZ0/`）。
本文件保留原始提案。旧公式、四操作排版、逆序伪代码与 Prompt 示例不是运行时契约；当前实现见 [DESIGN.md](DESIGN.md)。目标成绩表也不是本仓库实测。

## 一、整体运行思路与五步执行链路
混合架构：确定性代码（毫秒级规则路由、哈希坍缩、时间区间合并）+ 大模型语义（3D 排版写入、原点决策、慢系统反直觉推理）。

1. 步骤1 毫秒级分流网关（实体命中 + 句法特征触发器）：输入命中 Z轴实体哈希表任一键，或匹配状态变更/时间回溯/决策规划句法特征 -> TASK_POISON_CIRCLE（3D 渐近原点与毒圈引擎）；否则 CASUAL_2D_BYPASS（按哈希索引直接检索快速回复）。
2. 步骤2 2D 渐近原点定位与双层时间轴区间缓冲：控制核心识别当前任务决策，移动 2D 平面原点 (x0,y0)。事实点到原点距离强制渐近下界 d = max(eps, d_raw)（eps=0.05），永不触达原点。若距上次记录时间跨度 < Δt，合并进上一个时间节点、更新 t_end（层1记录原点移动；层2记录边界内外事实数量变化）。
3. 步骤3 平行 Z 轴坍缩隔离与“由远及近”逆向筛选：多条平行 Z 轴（人物、地点、事件、禁忌等）坍缩为带哈希索引的 1D 隔离线：同源 Z 轴事实垂直对齐，异源严格分桶隔离不重叠。从 Z 轴联想最远端向近端逆向扫描，优先打捞圈外高权重反直觉事实（例外、冲突、单次孤立规则），与圈内高压缩直觉锚点组装候选上下文。
4. 步骤4 慢系统（System 2）反直觉规划与幻觉校验：多步推理与自反思校验（实体路径可达性、时间有效性），生成回复。严格读写隔离：推理结论绝不反向修改毒圈边界。
5. 步骤5 异步 3D 笔记本落库：新客观事实后台异步写入：2D 平面“同行覆写 / 紧凑换行 / 孤立隔行”，更新 Z 轴哈希索引，触发时间轴第二层区间数量更新。

### 核心调度伪代码
```python
import re
import time

class CognitiveMemoryEngine:
    def __init__(self, delta_t_buffer=10.0, epsilon=0.05):
        self.delta_t = delta_t_buffer
        self.epsilon = epsilon
        self.notebook_2d = {}              # {block_id: lines}
        self.z_hash_index = {}             # {entity_key: [fact_records]}
        self.timeline_layer1 = []          # [{"t_start","t_end","origin"}]
        self.timeline_layer2 = []          # [{"t_start","t_end","in_cnt","out_cnt"}]
        self.current_origin = None

    def route_trigger(self, query: str) -> str:
        tokens = set(re.findall(r'[\w\u4e00-\u9fa5]+', query))
        entity_hit = any(k in query for k in self.z_hash_index.keys())
        mutation_pattern = r'(改|换|推迟|提前|取消|其实|不过|但是|别|不要|冲突)'
        temporal_pattern = r'(之前|上次|原本|以前|最早|后来|目前|现在|对比)'
        decision_pattern = r'(安排|规划|预算|推荐|选择|怎么办|帮我|方案)'
        syntax_hit = bool(
            re.search(mutation_pattern, query) or
            re.search(temporal_pattern, query) or
            (len(query) > 12 and re.search(decision_pattern, query))
        )
        return "TASK_POISON_CIRCLE" if (entity_hit or syntax_hit) else "CASUAL_2D_BYPASS"

    def record_timeline_buffered(self, layer: list, payload: dict):
        now = time.time()
        if layer and (now - layer[-1]["t_end"] <= self.delta_t):
            layer[-1]["t_end"] = now
            layer[-1].update(payload)
        else:
            layer.append({"t_start": now, "t_end": now, **payload})

    def collapse_and_scan_z_axis(self, matched_entities: list, radius_r: float):
        collapsed_line = []
        seen_fact_ids = set()
        for z_rank, entity in enumerate(reversed(matched_entities)):
            bucket = []
            for fact in self.z_hash_index.get(entity, []):
                if fact["id"] not in seen_fact_ids:
                    d_asymp = max(self.epsilon, fact["raw_dist"])
                    zone = "INNER_INTUITIVE" if d_asymp <= radius_r else "OUTER_ANTI_INTUITIVE"
                    bucket.append({**fact, "d_asymp": d_asymp, "zone": zone})
                    seen_fact_ids.add(fact["id"])
            if bucket:
                collapsed_line.append({"z_bucket": entity, "facts": bucket})
        return collapsed_line
```

## 二、三套提示词（JSON 契约输出）

### Prompt 1：2D 原点锚定与毒圈边界反直觉权重调节（任务态入口）
```text
# Role
你是认知记忆引擎的【核心控制决策器（Origin & Boundary Controller）】。你的职责是根据用户当前问题，确定 2D 记忆平面上的任务原点坐标，并动态调节毒圈边界的渗透权重（默认设定：毒圈外反直觉事实的审查权重高于圈内直觉事实）。

# Input Data
- 当前历史原点：{current_origin_json}
- 2D 笔记本主题索引目录：{notebook_2d_index}
- Z 轴命中实体列表：{matched_z_entities}
- 用户当前输入：{user_query}

# Core Rules
1. **原点移动（Origin Shift）**：定位与当前核心决策最匹配的 2D 主题块作为新原点。注意：所有相关度均为相对值，强相关锚点必须保持渐近距离（d >= 0.05），严禁判定为绝对匹配（d = 0）。
2. **毒圈边界半径与反直觉渗透调节**：
   - 常规任务：收紧毒圈半径（如 R = 0.45），大幅压缩圈内直觉背景，同时将圈外“反直觉例外/冲突/单次禁忌”的渗透权重调至最高（anti_intuitive_weight >= 0.85）。
   - 多跳回溯任务：适度扩大跨 Z 轴扫描深度，确保弱关联事实（分布于边界 R 两侧）能被完整捕获。
3. **Z 轴扫描排序**：将命中的 Z 轴实体按“语义联想距离从最远端到最近端”排序，以便执行逆向漏斗筛选。

# Output Schema (Strict JSON)
{
  "origin_shifted": true,
  "new_origin": {
    "x_topic": "主题块名称",
    "y_subtopic": "子维度名称",
    "decision_anchor": "一句话概括当前核心决策目标"
  },
  "poison_circle": {
    "radius_r": 0.45,
    "anti_intuitive_weight": 0.90,
    "intuitive_weight": 0.40,
    "z_axis_scan_order_far_to_near": ["最远端Z轴实体", "次远端Z轴实体", "最近端Z轴实体"]
  }
}
```

### Prompt 2：慢系统（System 2）反直觉规划与防幻觉回复（核心推理层）
```text
# Role
你是记忆引擎的【慢系统推理规划器（System 2 Anti-Intuitive Reasoner）】。你接收经过“由远及近逆向扫描”筛选出的高压缩记忆切片。你的核心任务是利用圈外反直觉事实修正圈内直觉惯性，并在严防幻觉的前提下生成精准回复。

# Memory Context
- 当前决策原点：{new_origin_json}
- 毒圈外·高权反直觉事实（例外/冲突/单次约束，优先审查）：
{outer_anti_intuitive_facts}
- 毒圈内·高压缩直觉事实（渐近常规背景，仅供参考）：
{inner_intuitive_facts}
- 用户当前问题：{user_query}

# Anti-Hallucination & Reasoning Protocol (必须严格执行)
1. **逆向冲突消解（从远及近）**：优先检查【毒圈外反直觉事实】是否与【毒圈内直觉事实】或用户的当前假设存在时间冲突、条件限制或隐性禁忌（如过敏、改期、预算上限）。若存在冲突，无条件以最新有效的反直觉事实推翻直觉结论。
2. **Z 轴路径可达性自检（防过度脑补）**：对于每一条拟采用的圈外反直觉事实，必须验证其与当前问题之间存在显式的实体逻辑链。若无显式逻辑链支撑，判定为 `[NOISE]` 并丢弃，严禁强行关联编造。
3. **渐近警戒原则**：圈内事实距离原点始终存在微隙，不可将其视为 100% 绝对真理；若关键信息缺失，如实说明已知边界并向用户确认，绝不臆测。
4. **只读隔离声明**：你的推理过程仅用于生成回复，禁止在输出中包含任何修改记忆库或调整毒圈边界的指令。

# Output Schema (Strict JSON)
{
  "reasoning_trace": {
    "anti_intuitive_overrides": ["列举被圈外反直觉事实修正或拦截的直觉假设，无则留空"],
    "rejected_noise_ids": ["被判定为无显式实体路径的噪声事实ID"],
    "verification_status": "VERIFIED | PARTIAL_INFO | CONFLICT_RESOLVED"
  },
  "final_response": "直接面向用户的最终高质量回复（自然、准确、简洁，不暴露内部记忆标签）"
}
```

### Prompt 3：3D 笔记本异步排版与 Z 轴切片写入（存储维护层）
```text
# Role
你是记忆存储层的【3D 笔记本排版路由引擎（3D Notebook Layout Router）】。在对话结束后，负责将本轮提取的新事实写入 2D 平面笔记本，并挂载正交 Z 轴哈希切片。

# Input Data
- 局部相似 2D 段落块（Top-3）：{candidate_2d_blocks}
- 本轮对话抽取的客观事实列表：{extracted_facts}
- 当前时间戳：{timestamp}

# 3D Layout Rules
1. **X-Y 平面排版操作（四选一）**：
   - `OVERWRITE_SAME_LINE`（同一数据同行覆写）：新事实是对已有同一实体同一属性的更新（如时间修改、状态变更），直接在原行覆盖替换旧值，保持单一事实源。
   - `APPEND_ADJACENT_LINE`（类似数据不隔行但换行）：新事实与某已有段落属于同一主题/场景，紧挨着该段落下一行追加（单换行 `\n`），形成紧凑语义块。
   - `INSERT_ISOLATED_LINE`（单次独立事实隔行放置）：全新的孤立事件或冷门细节，上下保留空行（双换行 `\n\n`）独立存放，防止特征被前后噪声稀释。
   - `NOOP`（忽略）：纯寒暄或重复废话，不入库。
2. **Z 轴正交切片提取**：为每条入库事实提取标准化的 Z 轴哈希键（格式：`维度:实体名`，如 `人物:老王`、`地点:东京`、`禁忌:花生`、`事件:项目上线`），确保同一事实在不同 Z 轴上可被纵向对齐检索且互不污染物理排版。

# Output Schema (Strict JSON)
{
  "write_operations": [
    {
      "action": "OVERWRITE_SAME_LINE | APPEND_ADJACENT_LINE | INSERT_ISOLATED_LINE | NOOP",
      "target_block_id": "目标段落ID（新建时为 NEW）",
      "target_line_id": "被覆写的原行ID（仅 OVERWRITE 时填写）",
      "formatted_line": "- [2026-11-18] 标准化后的单行精简事实文本",
      "z_axis_keys": ["人物:xxx", "地点:xxx", "事件:xxx"],
      "surprise_score": 0.85
    }
  ]
}
```

## 三、测试方案

### 四大测试集
| 模块 | 构造方式 | 检验目标 |
|---|---|---|
| Test 1 单次孤立事实抗淹没（对应 MemBench 参与式/观察式事实） | 100K token 无关新闻噪声流中随机插入 20 条仅出现 1 次的冷门细节（如“对某种防腐剂过敏”），间隔 50 轮后提问 | 2D 隔行留白 + 圈外反直觉默认优先能否捞出单次低频事实 |
| Test 2 高频旧记忆覆写与冲突（对应知识动态更新） | 前 30 轮反复强化同一设定（“每周三下午开周会”），第 31 轮改口一次（“本周周会改到周五上午”），第 35 轮提问 | 2D 同行覆写 + Z 轴逆向扫描能否压制高频旧习惯 |
| Test 3 跨 Z 轴多跳与时间旅行（对应反思推理 + 历史回溯） | “人物、地点、预算变更”分散在相隔 80 轮的孤立行中；提问跨维度组合推理，以及“在第二次修改预算之前的状态是什么” | 平行 Z 轴坍缩扫描的多跳精度，双层时间轴的历史快照回溯 |
| Test 4 闲聊旁路与零关联防幻觉 | 50 条纯寒暄 + 50 条“词面相似但实体路径无关”的陷阱问题 | 触发器分流准确率，慢系统路径自检对幻觉的拦截率 |

### 消融对照组（同底座模型，如 Qwen2.5-7B-Instruct，同测试集）
1. Baseline A Naive RAG：固定 chunk + Top-5 向量检索
2. Baseline B Mem0/Letta 复现版：常规 CRUD 记忆库 + 标准上下文窗口管理
3. Variant C：完整架构去掉时间轴区间缓冲与闲聊旁路
4. Full System：3D 笔记本 + 渐近原点毒圈 + 平行 Z 轴坍缩 + 双层时间区间缓冲 + 闲聊旁路

### 目标成绩（作者预估的验收目标，尚无实测）
| 指标 | Naive RAG | Mem0/Letta | 目标 |
|---|---|---|---|
| 单次孤立事实准确率 | 42.5% | 68.0% | > 91.0% |
| 知识动态更新准确率 | 31.0% | 74.5% | > 94.0% |
| 跨维度多跳推理准确率 | 38.0% | 65.0% | > 85.0% |
| 上下文 Token 压缩率（vs 全量历史） | 65.0% | 78.0% | > 92.0% |
| 无关联陷阱幻觉率（越低越好） | 24.0% | 16.5% | < 4.5% |
| 时间轴存储压缩比 | — | — | 减少 60%~75% 节点 |
| P95 检索延迟 | 180 ms | 850 ms | 闲聊 < 15 ms，任务态 < 120 ms |
