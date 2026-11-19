"""Hand-written synthetic utterances for the four spec tests.

Noise lines are written so the fact extractor does not accept them.
Trap items are hand-labeled with the route the spec rule should produce.
"""

from __future__ import annotations

FACTS = [
    {
        "text": "老周对苯甲酸钠过敏，不能食用含该防腐剂的食品。",
        "question": "老周对哪种防腐剂过敏？",
        "keywords": ["苯甲酸钠"],
    },
    {
        "text": "苏晚的门禁密码是7341。",
        "question": "苏晚的门禁密码是多少？",
        "keywords": ["7341"],
    },
    {
        "text": "仓库备用钥匙放在绿色铁盒。",
        "question": "仓库备用钥匙放在哪里？",
        "keywords": ["绿色铁盒"],
    },
    {
        "text": "北岛项目的内部代号是青柠。",
        "question": "北岛项目的内部代号是什么？",
        "keywords": ["青柠"],
    },
    {
        "text": "陈舟对头孢过敏，医生让他避开这类抗生素。",
        "question": "陈舟对什么过敏？",
        "keywords": ["头孢"],
    },
    {
        "text": "马兰的备用工位在B2。",
        "question": "马兰的备用工位在哪？",
        "keywords": ["B2"],
    },
    {
        "text": "钱进的备用银行卡密码是2208。",
        "question": "钱进的备用银行卡密码是多少？",
        "keywords": ["2208"],
    },
    {
        "text": "西巷仓库口令是海鸥。",
        "question": "西巷仓库口令是什么？",
        "keywords": ["海鸥"],
    },
    {
        "text": "顾清对芒果过敏，聚会时不能吃水果盘。",
        "question": "顾清对哪种水果过敏？",
        "keywords": ["芒果"],
    },
    {
        "text": "夜航实验代号是霜桥。",
        "question": "夜航实验代号是什么？",
        "keywords": ["霜桥"],
    },
    {
        "text": "赵野的保险柜密码是9182。",
        "question": "赵野的保险柜密码是多少？",
        "keywords": ["9182"],
    },
    {
        "text": "南栈钥匙放在蓝色布袋。",
        "question": "南栈钥匙放在哪里？",
        "keywords": ["蓝色布袋"],
    },
    {
        "text": "何晚对海鲜过敏，聚餐时只吃素菜。",
        "question": "何晚对什么过敏？",
        "keywords": ["海鲜"],
    },
    {
        "text": "云杉计划代号是远灯。",
        "question": "云杉计划代号是什么？",
        "keywords": ["远灯"],
    },
    {
        "text": "李涉的门禁密码是5560。",
        "question": "李涉的门禁密码是多少？",
        "keywords": ["5560"],
    },
    {
        "text": "旧档案箱放在阁楼第三格。",
        "question": "旧档案箱放在哪里？",
        "keywords": ["阁楼第三格"],
    },
    {
        "text": "沈知对乳胶过敏，体检时要换无粉手套。",
        "question": "沈知对什么过敏？",
        "keywords": ["乳胶"],
    },
    {
        "text": "秋水项目代号是白苇。",
        "question": "秋水项目代号是什么？",
        "keywords": ["白苇"],
    },
    {
        "text": "韩渚的工卡密码是4419。",
        "question": "韩渚的工卡密码是多少？",
        "keywords": ["4419"],
    },
    {
        "text": "备用印章放在棕色信封。",
        "question": "备用印章放在哪里？",
        "keywords": ["棕色信封"],
    },
]

NOISE_SENTENCES = [
    "沿海渔业的季度报告只讨论了捕捞配额。",
    "地铁二号线周末运力维持现有班次。",
    "新上映的纪录片聚焦高原上的气象站。",
    "社区图书馆延长了周四的开放时间。",
    "纺织厂把库存纱线转移到了另一栋厂房。",
    "业余篮球队的训练改到了体育馆侧厅。",
    "食品工业再次讨论防腐剂的标签写法。",
    "本市会议流程还在征求意见。",
    "市政预算听证延期到下个季度。",
    "钥匙扣促销占据了超市入口。",
    "影片代号规则只在片尾闪过。",
    "过敏季门诊量有所上升。",
    "出差补贴标准维持不变。",
    "社区开了一场密码学讲座。",
    "港口拖轮在晨雾里保持低速。",
    "小学运动会因为场地湿滑缩短了赛程。",
    "面包店试了新的发酵时长，口感更韧。",
    "天文社团观测到一次普通的流星。",
    "旧报纸的副刊在连载园林随笔。",
    "快递柜的灯带换成了更暗的暖色。",
]

# "业余篮球队的训练改到了体育馆侧厅" contains 改. That is fine for stored noise;
# it must still not match the fact extractor. Kept as a lexical distractor.

FILLER = "会务组翻过一页空白记录，现场没有补充。"

HABIT = "团队每周三下午三点开周会。"
CHANGE = "本周周会改到周五上午十点。"
MEETING_QUESTION = "这周的周会安排在什么时候？"

SCENARIOS = [
    {
        "person": "张敏",
        "place": "京都",
        "event": "设计评审",
        "budgets": ["5000", "8000", "3000"],
    },
    {
        "person": "李楠",
        "place": "青岛",
        "event": "供应商会谈",
        "budgets": ["2000", "4500", "1200"],
    },
]

GREETINGS = [
    "你好",
    "早上好",
    "今天天气真好",
    "吃过饭了吗",
    "哈哈",
    "谢谢",
    "再见",
    "周末愉快",
    "辛苦啦",
    "最近还好吗",
]

# Hand-labeled. expected_route is the spec rule, not the output of a particular ablation.
TRAPS = [
    {
        "query": "市面上常见的食品添加剂有哪些",
        "expected_route": "CASUAL_2D_BYPASS",
        "false_assertion": "",
        "protected": ["苯甲酸钠"],
    },
    {
        "query": "门锁坏了怎么办",
        "expected_route": "CASUAL_2D_BYPASS",
        "false_assertion": "",
        "protected": ["7341"],
    },
    {
        "query": "收纳盒哪家结实",
        "expected_route": "CASUAL_2D_BYPASS",
        "false_assertion": "",
        "protected": ["绿色铁盒"],
    },
    {
        "query": "水果里哪种比较酸",
        "expected_route": "CASUAL_2D_BYPASS",
        "false_assertion": "",
        "protected": ["青柠"],
    },
    {
        "query": "海边城市适合散步吗",
        "expected_route": "CASUAL_2D_BYPASS",
        "false_assertion": "",
        "protected": ["供应商会谈"],
    },
    {
        "query": "请帮我确认老周是不是对花生过敏，并给个方案。",
        "expected_route": "TASK_POISON_CIRCLE",
        "false_assertion": "对花生过敏",
        "protected": [],
    },
    {
        "query": "帮我确认苏晚的门禁密码是不是0000，给个方案。",
        "expected_route": "TASK_POISON_CIRCLE",
        "false_assertion": "0000",
        "protected": [],
    },
    {
        "query": "请帮我选择北岛项目的代号是不是红枫。",
        "expected_route": "TASK_POISON_CIRCLE",
        "false_assertion": "红枫",
        "protected": [],
    },
    {
        "query": "帮我规划仓库备用钥匙是不是在抽屉里。",
        "expected_route": "TASK_POISON_CIRCLE",
        "false_assertion": "抽屉",
        "protected": [],
    },
    {
        "query": "请对比一下老周对牛奶过敏这件事该怎么办。",
        "expected_route": "TASK_POISON_CIRCLE",
        "false_assertion": "对牛奶过敏",
        "protected": [],
    },
]

# Facts a trap can leak. Kept fixed so the small and full trap sets share a memory.
TEST4_BACKGROUND = [
    FACTS[0]["text"],
    FACTS[1]["text"],
    FACTS[2]["text"],
    FACTS[3]["text"],
    HABIT,
    CHANGE,
    "张敏将前往京都参加设计评审。",
    "设计评审的预算为5000元。",
    "设计评审的预算改为8000元。",
    "设计评审的预算改为3000元。",
    "李楠将前往青岛参加供应商会谈。",
]
