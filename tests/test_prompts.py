from importlib.resources import files

from memengine.extract import extract_facts
from memengine.llm import MockLLMClient, loads_json_object
from memengine.prompts import load_prompt, prompt_dir, render_prompt


_PROMPT_FILES = (
    "origin_boundary.txt",
    "system2_reasoner.txt",
    "notebook_layout.txt",
    "query_parser.txt",
    "query_origin_controller.txt",
)


def test_prompt_dir_resolves_spec_files():
    directory = prompt_dir()
    for name in _PROMPT_FILES:
        assert (directory / name).is_file()


def test_prompt_templates_resolve_via_importlib_resources():
    root = files("prompts")
    for name in _PROMPT_FILES:
        assert root.joinpath(name).read_text(encoding="utf-8").strip()


def test_prompt_files_keep_the_spec_contract():
    origin = load_prompt("origin")
    reason = load_prompt("reason")
    layout = load_prompt("layout")
    query = load_prompt("query")
    assert "Origin & Boundary Controller" in origin
    assert "{current_origin_json}" in origin
    assert "慢系统综合推理与作答器" in reason
    assert "只读隔离" in reason
    assert "{retrieved_top_k_memories}" in reason
    assert "{user_question}" in reason
    assert "无损事实抽取与分类写入器" in layout
    assert "{raw_turn_text}" in layout
    assert "{step_id}" in layout
    assert "{query_stem}" in query
    assert "查询路由与上位类目解析器" in query
    for action in ("OVERWRITE_SAME_LINE", "APPEND_ADJACENT_LINE", "INSERT_ISOLATED_LINE", "DELETE_LINE", "NOOP"):
        assert action in layout
    for name in ("影视", "书籍", "音乐", "饮食", "地点", "人物", "消费", "工作学习", "运动健康", "日程计划", "兴趣习惯", "情绪观点"):
        assert name in layout
        assert name in query


def test_mock_client_fills_all_contracts():
    client = MockLLMClient()
    origin = render_prompt(
        "origin",
        current_origin_json={},
        notebook_2d_index=[{"block_id": "b1", "topic": "健康禁忌", "preview": "老周", "preview_attr": "过敏"}],
        matched_z_entities=["人物:老周", "人物:张敏"],
        user_query="老周对哪种防腐剂过敏？",
    )
    assert "{current_origin_json}" not in origin
    origin_data = loads_json_object(client.complete(origin))
    assert origin_data["poison_circle"]["z_axis_scan_order_far_to_near"][-1] == "人物:老周"

    outer = [
        {
            "id": "l1",
            "text": "老周对苯甲酸钠过敏",
            "zkeys": ["人物:老周", "禁忌:苯甲酸钠"],
            "attribute": "过敏",
            "value": "苯甲酸钠",
            "revision": 0,
            "historical": False,
            "zone": "OUTER_ANTI_INTUITIVE",
        }
    ]
    reason = render_prompt(
        "reason",
        query_type="FACTUAL",
        retrieved_top_k_memories=outer,
        user_question='"老周对哪种防腐剂过敏？"',
    )
    reason_data = loads_json_object(client.complete(reason))
    assert "苯甲酸钠" in reason_data["final_answer"]
    assert "write_operations" not in reason_data
    assert set(reason_data) == {"evidence_synthesis", "anti_intuitive_override", "rejected_noise_ids", "final_answer"}

    facts = extract_facts("老周对苯甲酸钠过敏，不能食用含该防腐剂的食品。")
    layout = render_prompt(
        "layout",
        step_id="0",
        candidate_2d_blocks=[],
        raw_turn_text="[step 0] 老周对苯甲酸钠过敏，不能食用含该防腐剂的食品。",
    )
    layout_data = loads_json_object(client.complete(layout))
    assert layout_data["write_operations"][0]["action"] == "INSERT_ISOLATED_LINE"
    assert any(str(key).startswith("类目:") for key in layout_data["write_operations"][0].get("category_z_keys") or layout_data["write_operations"][0].get("z_axis_keys") or [])

    query = render_prompt("query", query_stem="What type of movies might I prefer?")
    query_data = loads_json_object(client.complete(query))
    assert "类目:影视" in query_data["categories"]
    assert query_data["query_type"] in {"REFLECTIVE", "FACTUAL"}
