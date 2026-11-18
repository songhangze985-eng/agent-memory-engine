from memengine.notebook import apply_write_op, render_block
from memengine.store import Store


def _op(action, block, line="", formatted="x", zkeys=None, surprise=0.5):
    return {
        "action": action,
        "target_block_id": block,
        "target_line_id": line,
        "formatted_line": formatted,
        "z_axis_keys": zkeys or [],
        "surprise_score": surprise,
    }


def test_append_is_single_newline_and_isolated_is_blank_line():
    store = Store(":memory:")
    alias = {}
    apply_write_op(
        store,
        _op("APPEND_ADJACENT_LINE", "NEW:工作", formatted="甲"),
        topic="工作",
        attribute="周会时间",
        value="周三",
        plain="甲",
        timestamp=10,
        alias=alias,
    )
    apply_write_op(
        store,
        _op("APPEND_ADJACENT_LINE", "NEW:工作", formatted="乙"),
        topic="工作",
        attribute="其他",
        value="乙",
        plain="乙",
        timestamp=11,
        alias=alias,
    )
    block_id = alias["NEW:工作"]
    lines = store.blocks_for_prompt()[0]["lines"]
    rendered = render_block(lines)
    assert rendered == "甲\n乙"
    assert "\n\n" not in rendered
    apply_write_op(
        store,
        _op("INSERT_ISOLATED_LINE", block_id, formatted="丙"),
        topic="工作",
        attribute="代号",
        value="丙",
        plain="丙",
        timestamp=12,
        alias=alias,
    )
    lines = store.blocks_for_prompt()[0]["lines"]
    rendered = render_block(lines)
    assert rendered == "甲\n乙\n\n丙"


def test_overwrite_archives_the_previous_value_and_updates_zkeys():
    store = Store(":memory:")
    alias = {}
    line_id = apply_write_op(
        store,
        _op("INSERT_ISOLATED_LINE", "NEW:ISO", formatted="旧", zkeys=["人物:老周", "禁忌:花粉"], surprise=0.9),
        topic="健康禁忌",
        attribute="过敏",
        value="花粉",
        plain="老周对花粉过敏",
        timestamp=10,
        alias=alias,
    )
    apply_write_op(
        store,
        _op(
            "OVERWRITE_SAME_LINE",
            alias["NEW:ISO"],
            line=line_id,
            formatted="新",
            zkeys=["人物:老周", "禁忌:苯甲酸钠"],
            surprise=0.86,
        ),
        topic="健康禁忌",
        attribute="过敏",
        value="苯甲酸钠",
        plain="老周对苯甲酸钠过敏",
        timestamp=20,
        alias=alias,
    )
    active = [line for block in store.active_snapshot() for line in block["lines"]]
    assert len(active) == 1
    assert active[0]["id"] == line_id
    assert active[0]["plain"] == "老周对苯甲酸钠过敏"
    assert active[0]["revision"] == 1
    history = store.historical_versions([line_id])
    assert len(history) == 1
    assert history[0]["text"] == "老周对花粉过敏"
    assert history[0]["revision"] == 0
    indexed = store.z_index()
    assert indexed["禁忌:苯甲酸钠"][0]["text"] == "老周对苯甲酸钠过敏"
    assert "禁忌:花粉" not in indexed


def test_noop_does_not_change_lines():
    store = Store(":memory:")
    apply_write_op(
        store,
        _op("APPEND_ADJACENT_LINE", "NEW:t", formatted="只此一行"),
        topic="杂项",
        attribute="代号",
        value="青柠",
        plain="只此一行",
        timestamp=1,
        alias={},
    )
    before = store.active_snapshot()
    apply_write_op(
        store,
        _op("NOOP", ""),
        topic="杂项",
        attribute="代号",
        value="青柠",
        plain="只此一行",
        timestamp=2,
        alias={},
    )
    assert store.active_snapshot() == before
