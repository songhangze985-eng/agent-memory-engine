from memengine.extract import extract_facts
from memengine.llm import MockLLMClient, loads_json_object
from memengine.prompts import load_prompt, prompt_dir, render_prompt


def test_prompt_dir_resolves_spec_files():
    directory = prompt_dir()
    assert (directory / "origin_boundary.txt").is_file()
    assert (directory / "system2_reasoner.txt").is_file()
    assert (directory / "notebook_layout.txt").is_file()


def test_prompt_files_keep_the_spec_contract():
    origin = load_prompt("origin")
    reason = load_prompt("reason")
    layout = load_prompt("layout")
    assert "Origin & Boundary Controller" in origin
    assert "{current_origin_json}" in origin
    assert "System 2 Anti-Intuitive Reasoner" in reason
    assert "只读隔离声明" in reason
    assert "3D Notebook Layout Router" in layout
    for action in ("OVERWRITE_SAME_LINE", "APPEND_ADJACENT_LINE", "INSERT_ISOLATED_LINE", "NOOP"):
        assert action in layout


def test_mock_client_fills_all_three_contracts():
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
        new_origin_json={"x_topic": "健康禁忌", "y_subtopic": "过敏", "decision_anchor": "老周"},
        outer_anti_intuitive_facts=outer,
        inner_intuitive_facts=[],
        user_query="老周对哪种防腐剂过敏？",
    )
    reason_data = loads_json_object(client.complete(reason))
    assert "苯甲酸钠" in reason_data["final_response"]
    assert "write_operations" not in reason_data

    facts = extract_facts("老周对苯甲酸钠过敏，不能食用含该防腐剂的食品。")
    layout = render_prompt(
        "layout",
        candidate_2d_blocks=[],
        extracted_facts=facts,
        timestamp="1000",
    )
    layout_data = loads_json_object(client.complete(layout))
    assert layout_data["write_operations"][0]["action"] == "INSERT_ISOLATED_LINE"
