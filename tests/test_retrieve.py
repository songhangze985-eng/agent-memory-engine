from eval.membench.items import format_question, question_stem, strip_mcq_options
from agentmem_core.categories import (
    CATEGORY_ENUM,
    attach_dual_zkeys,
    normalize_category_key,
    validate_category_keys,
)
from agentmem_core.engine import CognitiveMemoryEngine
from agentmem_core.llm import MockLLMClient
from agentmem_core.queryparse import parse_query_rules
from agentmem_core.vector import score_quantiles, weighted_rrf


def test_category_enum_is_the_fixed_twelve():
    assert CATEGORY_ENUM == [
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


def test_unknown_categories_are_dropped_and_aliases_collapse():
    assert normalize_category_key("影片") == "类目:影视"
    assert normalize_category_key("类目:电影") == "类目:影视"
    assert normalize_category_key("made_up") is None
    keys = validate_category_keys(["影片", "自造类目", "书籍"], fallback_text="")
    assert keys == ["类目:影视", "类目:书籍"]
    filled = validate_category_keys(["自造类目"], fallback_text="the movies I mentioned")
    assert filled == ["类目:影视"]
    attached = attach_dual_zkeys(["实体:Ran"], "I watched the movie Ran", extra_category_keys=["影片"])
    assert "实体:Ran" in attached
    assert "类目:影视" in attached


def test_query_parser_maps_hypernym_stems():
    movies = parse_query_rules("What do the movies I mentioned have in common?")
    assert "类目:影视" in movies["categories"]
    assert movies["entities"] == []
    books = parse_query_rules("What kind of books might I prefer?")
    assert "类目:书籍" in books["categories"]


def test_weighted_rrf_prefers_vector_ranks_by_default():
    vec = [{"id": "v1", "text": "vector-first"}, {"id": "both", "text": "shared"}, {"id": "v3", "text": "vector-third"}]
    hashed = [{"id": "h1", "text": "hash-first"}, {"id": "both", "text": "shared"}]
    merged = weighted_rrf(vec, hashed, w_vec=0.65, w_hash=0.35, k=60)
    ids = [fact["id"] for fact in merged]
    assert ids[0] == "both"
    assert "v1" in ids
    assert ids.index("v1") < ids.index("h1")
    assert set(ids) == {"v1", "both", "v3", "h1"}


def test_score_quantiles_scale_bm25_in_the_correct_direction():
    hits = [{"id": "strong", "bm25_score": -9}, {"id": "middle", "bm25_score": -5},
            {"id": "weak", "bm25_score": -1}]
    scores = score_quantiles(hits, "bm25_score", lower_is_better=True)
    assert scores["strong"] == 1 and scores["weak"] == 0
    assert abs(scores["middle"] - .5) < 1e-12
    assert not score_quantiles([hits[0]], "bm25_score")
    ranked = weighted_rrf([], hits, normalized_channels=({}, scores))
    assert [row["id"] for row in ranked] == ["strong", "middle", "weak"]
    assert ranked[-1]["rrf_score"] == 0


def test_unrelated_query_does_not_fall_back_to_recent_facts():
    engine = CognitiveMemoryEngine(db_path=":memory:", llm=MockLLMClient())
    try:
        engine.add("[step 0] user: I really dig the movie Ran\nagent: Nice.", timestamp=1.0)
        assert engine.has_facts()
        engine.vector = type("NoVectorMatch", (), {"search": lambda self, query, facts, top_n: []})()
        retrieved = engine.search("What should I pack for a ski trip tomorrow?")
        texts = [fact.get("text") or "" for fact in list(retrieved.get("hits") or []) + list(retrieved.get("outer") or []) + list(retrieved.get("inner") or [])]
        assert not texts
        assert retrieved["empty_retrieval"] is True
    finally:
        engine.store.close()


def test_search_uses_question_stem_and_ignores_options():
    item = {
        "question": "What type of movies might I prefer?",
        "q_time": "'2024-10-02 08:00' Wednesday",
        "choices": {
            "A": "Zucchini cookbook deluxe",
            "B": "Classic samurai cinema such as Ran",
            "C": "Alpine skiing manuals",
            "D": "Tax audit worksheets",
        },
    }
    stem = question_stem(item)
    formatted = format_question(item)
    assert "Zucchini" in formatted and "Ran" in formatted
    assert "Zucchini" not in stem and "Alpine" not in stem
    assert strip_mcq_options(formatted) == stem

    engine = CognitiveMemoryEngine(db_path=":memory:", llm=MockLLMClient())
    try:
        engine.add("[step 0] user: I really dig the movie Ran\nagent: Nice.", timestamp=1.0)
        retrieved = engine.search(stem)
        parsed = retrieved["parsed_query"]
        blob = " ".join(
            [
                str(parsed),
                str(parsed.get("semantic_vector_query") or ""),
                " ".join(parsed.get("entities") or []),
            ]
        )
        assert "Zucchini" not in blob
        assert "Alpine" not in blob
        assert "Tax audit" not in blob
        assert "类目:影视" in (parsed.get("categories") or [])
    finally:
        engine.store.close()



def test_shared_category_does_not_overwrite_different_entities():
    from agentmem_core.policy import decide_layout

    first = {
        "text": "设计评审的预算改为3000元",
        "zkeys": ["事件:设计评审", "类目:消费"],
        "attribute": "预算",
        "value": "3000",
        "topic": "出行",
        "isolated": False,
    }
    second = {
        "text": "供应商会谈的预算为2000元",
        "zkeys": ["事件:供应商会谈", "类目:消费"],
        "attribute": "预算",
        "value": "2000",
        "topic": "出行",
        "isolated": False,
    }
    # Simulate the first fact already sitting in a block.
    blocks = [
        {
            "id": "b1",
            "topic": "出行",
            "lines": [
                {
                    "id": "l1",
                    "text": first["text"],
                    "attribute": "预算",
                    "value": "3000",
                    "zkeys": first["zkeys"],
                }
            ],
        }
    ]
    ops = decide_layout(blocks, [second], 1)["write_operations"]
    assert ops[0]["action"] != "OVERWRITE_SAME_LINE"
    assert ops[0]["action"] == "APPEND_ADJACENT_LINE"
