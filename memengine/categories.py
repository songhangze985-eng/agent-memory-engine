"""Write/query-shared controlled category enum and keyword mapping.

Unknown category labels are dropped. If none remain, a deterministic
English/Chinese keyword map fills in at most three keys from the enum.
"""

from __future__ import annotations

import re

CATEGORY_ENUM = [
    "影视",
    "书籍",
    "音乐",
    "饮食",
    "地点",
    "人物",
    "消费",
    "工作学习",
    "运动健康",
    "日程计划",
    "兴趣习惯",
    "情绪观点",
]

CATEGORY_PREFIX = "类目:"
ENTITY_PREFIX = "实体:"

# Synonyms that are not in the enum but should collapse onto it.
_ALIASES = {
    "电影": "影视",
    "影片": "影视",
    "电视剧": "影视",
    "动漫": "影视",
    "movie": "影视",
    "film": "影视",
    "书": "书籍",
    "小说": "书籍",
    "book": "书籍",
    "novel": "书籍",
    "歌": "音乐",
    "歌曲": "音乐",
    "music": "音乐",
    "菜": "饮食",
    "食物": "饮食",
    "food": "饮食",
    "city": "地点",
    "person": "人物",
    "people": "人物",
}

# Keyword -> category, longest-first matching on the raw text.
_KEYWORDS: list[tuple[str, str]] = [
    ("documentary", "影视"),
    ("animation", "影视"),
    ("watched", "影视"),
    ("movies", "影视"),
    ("movie", "影视"),
    ("films", "影视"),
    ("film", "影视"),
    ("cinema", "影视"),
    ("anime", "影视"),
    ("电视剧", "影视"),
    ("纪录片", "影视"),
    ("电影", "影视"),
    ("影片", "影视"),
    ("动漫", "影视"),
    ("综艺", "影视"),
    ("片子", "影视"),
    ("novels", "书籍"),
    ("novel", "书籍"),
    ("books", "书籍"),
    ("book", "书籍"),
    ("reading", "书籍"),
    ("manga", "书籍"),
    ("comic", "书籍"),
    ("paper", "书籍"),
    ("essay", "书籍"),
    ("read", "书籍"),
    ("论文", "书籍"),
    ("漫画", "书籍"),
    ("小说", "书籍"),
    ("文章", "书籍"),
    ("阅读", "书籍"),
    ("书", "书籍"),
    ("concert", "音乐"),
    ("album", "音乐"),
    ("songs", "音乐"),
    ("song", "音乐"),
    ("music", "音乐"),
    ("band", "音乐"),
    ("演唱会", "音乐"),
    ("专辑", "音乐"),
    ("乐队", "音乐"),
    ("歌曲", "音乐"),
    ("音乐", "音乐"),
    ("歌", "音乐"),
    ("restaurant", "饮食"),
    ("flavor", "饮食"),
    ("dishes", "饮食"),
    ("dish", "饮食"),
    ("snack", "饮食"),
    ("drink", "饮食"),
    ("taste", "饮食"),
    ("food", "饮食"),
    ("meal", "饮食"),
    ("allergy", "饮食"),
    ("过敏", "饮食"),
    ("忌口", "饮食"),
    ("餐厅", "饮食"),
    ("饮品", "饮食"),
    ("零食", "饮食"),
    ("菜品", "饮食"),
    ("口味", "饮食"),
    ("味道", "饮食"),
    ("爱吃", "饮食"),
    ("菜", "饮食"),
    ("country", "地点"),
    ("cities", "地点"),
    ("city", "地点"),
    ("place", "地点"),
    ("lived", "地点"),
    ("景点", "地点"),
    ("居住", "地点"),
    ("国家", "地点"),
    ("城市", "地点"),
    ("地点", "地点"),
    ("去过", "地点"),
    ("colleague", "人物"),
    ("friends", "人物"),
    ("friend", "人物"),
    ("family", "人物"),
    ("person", "人物"),
    ("people", "人物"),
    ("同事", "人物"),
    ("朋友", "人物"),
    ("家人", "人物"),
    ("名人", "人物"),
    ("人物", "人物"),
    ("budget", "消费"),
    ("brand", "消费"),
    ("shopping", "消费"),
    ("bought", "消费"),
    ("buy", "消费"),
    ("price", "消费"),
    ("理财", "消费"),
    ("预算", "消费"),
    ("品牌", "消费"),
    ("购物", "消费"),
    ("消费", "消费"),
    ("exam", "工作学习"),
    ("school", "工作学习"),
    ("project", "工作学习"),
    ("skill", "工作学习"),
    ("study", "工作学习"),
    ("work", "工作学习"),
    ("job", "工作学习"),
    ("考试", "工作学习"),
    ("学校", "工作学习"),
    ("技能", "工作学习"),
    ("项目", "工作学习"),
    ("职业", "工作学习"),
    ("学习", "工作学习"),
    ("工作", "工作学习"),
    ("workout", "运动健康"),
    ("fitness", "运动健康"),
    ("sleep", "运动健康"),
    ("health", "运动健康"),
    ("sport", "运动健康"),
    ("作息", "运动健康"),
    ("睡眠", "运动健康"),
    ("健身", "运动健康"),
    ("运动", "运动健康"),
    ("健康", "运动健康"),
    ("reschedule", "日程计划"),
    ("deadline", "日程计划"),
    ("meeting", "日程计划"),
    ("schedule", "日程计划"),
    ("appointment", "日程计划"),
    ("trip", "日程计划"),
    ("改期", "日程计划"),
    ("截止", "日程计划"),
    ("出行", "日程计划"),
    ("约定", "日程计划"),
    ("会议", "日程计划"),
    ("周会", "日程计划"),
    ("日程", "日程计划"),
    ("lifestyle", "兴趣习惯"),
    ("hobby", "兴趣习惯"),
    ("habit", "兴趣习惯"),
    ("game", "兴趣习惯"),
    ("pet", "兴趣习惯"),
    ("生活方式", "兴趣习惯"),
    ("习惯", "兴趣习惯"),
    ("宠物", "兴趣习惯"),
    ("游戏", "兴趣习惯"),
    ("兴趣", "兴趣习惯"),
    ("preference", "情绪观点"),
    ("prefer", "情绪观点"),
    ("personality", "情绪观点"),
    ("opinion", "情绪观点"),
    ("attitude", "情绪观点"),
    ("values", "情绪观点"),
    ("like", "情绪观点"),
    ("love", "情绪观点"),
    ("审美", "情绪观点"),
    ("性格", "情绪观点"),
    ("价值观", "情绪观点"),
    ("态度", "情绪观点"),
    ("观点", "情绪观点"),
    ("吐槽", "情绪观点"),
    ("喜欢", "情绪观点"),
]

_KEYWORD_RE = [
    (re.compile(r"(?<![a-z])" + re.escape(word) + r"(?![a-z])", re.I) if re.search(r"[A-Za-z]", word) else re.compile(re.escape(word)), cat)
    for word, cat in _KEYWORDS
]

_STOP_PROPER = frozenset(
    "according accordingto what which who whom whose when where why how the "
    "this that these those i my me we our you your reply letter current time "
    "monday tuesday wednesday thursday friday saturday sunday drama war "
    "mystery animation romance children sci fi musical action thriller "
    "western comedy horror documentary".split()
)


def category_key(name: str) -> str:
    return f"{CATEGORY_PREFIX}{name}"


def entity_key(name: str) -> str:
    raw = (name or "").strip()
    if not raw:
        return ""
    if raw.startswith(ENTITY_PREFIX) or raw.startswith(CATEGORY_PREFIX):
        return raw
    if ":" in raw:
        return raw
    return f"{ENTITY_PREFIX}{raw}"


def _bare_category(raw: str) -> str:
    text = (raw or "").strip()
    if ":" in text:
        text = text.split(":", 1)[1].strip()
    return text


def normalize_category_key(raw: str) -> str | None:
    """Return a valid ``类目:枚举名`` key, or None if it cannot be mapped."""
    bare = _bare_category(raw)
    if not bare:
        return None
    if bare in CATEGORY_ENUM:
        return category_key(bare)
    alias = _ALIASES.get(bare) or _ALIASES.get(bare.lower())
    if alias in CATEGORY_ENUM:
        return category_key(alias)
    return None


def map_categories(text: str, limit: int = 3) -> list[str]:
    """Deterministic keyword map used by the mock/rule path and as fallback."""
    raw = text or ""
    found: list[str] = []
    for pattern, cat in _KEYWORD_RE:
        key = category_key(cat)
        if key in found:
            continue
        if pattern.search(raw):
            found.append(key)
            if len(found) >= limit:
                break
    return found


def validate_category_keys(keys, fallback_text: str = "", limit: int = 3) -> list[str]:
    """Drop labels outside the enum; fill from the keyword map if nothing valid remains."""
    valid: list[str] = []
    for raw in keys or []:
        key = normalize_category_key(str(raw))
        if key and key not in valid:
            valid.append(key)
        if len(valid) >= limit:
            return valid
    if not valid:
        for key in map_categories(fallback_text, limit=limit):
            if key not in valid:
                valid.append(key)
    return valid[:limit]


def extract_explicit_entities(text: str) -> list[str]:
    """Proper nouns actually present in the stem; never invent titles."""
    keys: list[str] = []
    seen: set[str] = set()
    for name in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", text or ""):
        if name.lower() in _STOP_PROPER:
            continue
        key = entity_key(name)
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def attach_dual_zkeys(
    existing,
    text: str,
    extra_entity_keys=None,
    extra_category_keys=None,
    default_category: str | None = "兴趣习惯",
) -> list[str]:
    """Keep prior keys, add ``实体:`` / ``类目:`` keys, guarantee one enum category on write."""
    keys: list[str] = []
    for raw in list(existing or []) + list(extra_entity_keys or []):
        item = str(raw).strip()
        if not item or item in keys:
            continue
        keys.append(item)
    cats = validate_category_keys(extra_category_keys, fallback_text=text)
    if not cats and default_category:
        cats = [category_key(default_category)]
    for key in cats:
        if key not in keys:
            keys.append(key)
    return keys


def category_synonyms(category: str, language: str = "en") -> str:
    """Expansion terms for the vector channel, in the language of the history."""
    en = {
        "影视": "movie film cinema watched show director plot like review title",
        "书籍": "book novel read reading author chapter like review title",
        "音乐": "song album music band concert listen like",
        "饮食": "dish food restaurant flavor taste meal drink like prefer",
        "地点": "place city country visit lived location",
        "人物": "person people family friend colleague name",
        "消费": "buy bought brand budget price shopping",
        "工作学习": "work job school project skill exam study",
        "运动健康": "sport fitness health sleep workout",
        "日程计划": "meeting schedule appointment trip deadline",
        "兴趣习惯": "hobby habit game pet lifestyle",
        "情绪观点": "prefer preference like love opinion attitude personality",
    }
    zh = {
        "影视": "电影 影片 看过 剧情 导演 喜欢 评价 影视作品",
        "书籍": "书 小说 读过 作者 章节 喜欢 评价",
        "音乐": "歌 专辑 乐队 听过 喜欢",
        "饮食": "菜 食物 餐厅 口味 味道 爱吃 喜欢",
        "地点": "地方 城市 国家 去过 住过",
        "人物": "人 家人 朋友 同事 名字",
        "消费": "买 品牌 预算 价格 购物",
        "工作学习": "工作 学校 项目 技能 考试 学习",
        "运动健康": "运动 健身 健康 睡眠",
        "日程计划": "会议 日程 约定 出行 截止",
        "兴趣习惯": "兴趣 习惯 游戏 宠物",
        "情绪观点": "喜欢 偏好 态度 性格 观点",
    }
    table = zh if language == "zh" else en
    bare = _bare_category(category)
    return table.get(bare, bare)
