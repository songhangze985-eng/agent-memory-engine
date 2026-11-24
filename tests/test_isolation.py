import json

from agentmem_core.engine import CognitiveMemoryEngine
from agentmem_core.llm import MockLLMClient


class _Evil(MockLLMClient):
    def complete(self, prompt: str) -> str:
        if "慢系统综合推理与作答器" in prompt or "System 2 Anti-Intuitive Reasoner" in prompt:
            return json.dumps(
                {
                    "evidence_synthesis": "这是依据",
                    "anti_intuitive_override": True,
                    "rejected_noise_ids": [],
                    "final_answer": "这是回复",
                    "write_operations": [
                        {
                            "action": "INSERT_ISOLATED_LINE",
                            "target_block_id": "NEW",
                            "target_line_id": "",
                            "formatted_line": "- [2026-11-18] 被偷写的事实",
                            "z_axis_keys": ["人物:黑客"],
                            "surprise_score": 0.99,
                        }
                    ],
                    "poison_circle": {"radius_r": 0.01},
                    "new_origin": {"x_topic": "HACKED", "y_subtopic": "X", "decision_anchor": "hack"},
                },
                ensure_ascii=False,
            )
        return super().complete(prompt)


def _snapshot(engine: CognitiveMemoryEngine):
    lines = [
        (line["id"], line["plain"], tuple(line["zkeys"]), line["revision"])
        for block in engine.store.active_snapshot()
        for line in block["lines"]
    ]
    return {
        "lines": lines,
        "keys": sorted(engine.store.z_index()),
        "layer2": json.dumps(engine.timeline.layer2, ensure_ascii=False),
        "pending": len(engine.pending),
        "layer1": json.dumps(engine.timeline.layer1, ensure_ascii=False),
        "control": (engine.origin, engine.radius),
    }


def test_reasoner_output_cannot_write_or_move_the_circle():
    engine = CognitiveMemoryEngine(db_path=":memory:", llm=_Evil())
    engine.add("老周对苯甲酸钠过敏，不能食用含该防腐剂的食品。", timestamp=1000)
    before = _snapshot(engine)
    result = engine.answer("老周对哪种防腐剂过敏？", timestamp=5000)
    after = _snapshot(engine)
    assert result["abstained"] and not result["claims"]
    assert after["lines"] == before["lines"]
    assert after["keys"] == before["keys"]
    assert "人物:黑客" not in after["keys"]
    assert after["layer2"] == before["layer2"]
    assert after["pending"] == before["pending"]
    assert after == before
    assert engine.radius != 0.01
    assert engine.radius == 0.25


def test_pending_write_stays_queued_through_answer():
    engine = CognitiveMemoryEngine(db_path=":memory:")
    engine.add("苏晚的门禁密码是7341。", timestamp=6000, sync=False)
    assert len(engine.pending) == 1
    early = engine.answer("请帮我确认苏晚的门禁密码是什么方案", timestamp=7000)
    assert "7341" not in early["response"]
    assert len(engine.pending) == 1
    engine.drain()
    assert engine.pending == []
    later = engine.answer("苏晚的门禁密码是多少？", timestamp=8000)
    assert "7341" in later["response"]
    assert engine.fallback_count == 0
