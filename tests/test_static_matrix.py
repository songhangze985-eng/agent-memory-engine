import json

import pytest

from eval.paired import make_engine
from eval.static_report import FROZEN, summary_table
from memengine.assembler import assemble_context, render_prefix
from memengine.temporal import temporal_competition


def test_summary_uses_the_matching_dataset_and_named_metrics(tmp_path):
    for i, (filename, dataset, field) in enumerate(FROZEN):
        metrics = {"mrr_at_10": (i + 1) / 10, "recall_at_10": .8, "recall_at_5": .5, "avg_injected_tokens": 123}
        (tmp_path / filename).write_text(json.dumps({"benchmark": dataset, "items": [i],
                                                    field: {"bipolar": metrics, "rrf": metrics}}))
    report = summary_table(tmp_path)
    assert "| MemBench | 1 | 50.00% / 80.00% / 0.1000" in report
    assert "| LoCoMo | 1 | 50.00% / 80.00% / 0.2000" in report
    source = tmp_path / FROZEN[1][0]
    source.write_text((tmp_path / FROZEN[0][0]).read_text())
    with pytest.raises(ValueError, match="expected LoCoMo"):
        summary_table(tmp_path)


def test_assembly_deduplicates_profile_and_observation():
    engine = make_engine("mock")
    try:
        row = engine.add("Atlas uses a calibrated laboratory sensor.", timestamp=1)
        fact = row["operations"][0]["line_id"]
        engine.store.insert_observation("Atlas tracks sensor calibration.", derived_from_fact_ids=[fact],
                                        source_turn_ids=[row["turn_id"]])
        prefix = render_prefix({"core_profile": "ATLAS  tracks sensor calibration", "procedural_rules": []})
        query = "Which sensor does Atlas use?"
        result = assemble_context(engine.store, engine.retrieve(query, read_only=True), prefix=prefix,
                                  token_budget=2048, drilldown_top_n=0, drilldown_window=0, query=query)
        assert not any(item["kind"] == "observation" for item in result["items"])
        assert result["token_breakdown"]["facts"] > 0
    finally:
        engine.store.close()


def test_temporal_quantity_can_change_without_changing_number_in_product_name():
    attributes = {"attribute": "", "identifiers": {}, "role": "user", "entity_z_keys": ["实体:Acme Meter 3"]}
    older = {**attributes, "id": "old", "session_id": "first", "source_turn_ids": [1], "event_time_start": 1,
             "ingest_time": 1, "plain": "I've used the Acme Meter 3 for six months."}
    newer = {**attributes, "id": "new", "session_id": "second", "source_turn_ids": [2], "event_time_start": 2,
             "ingest_time": 2, "plain": "I've used the Acme Meter 3 for nine months."}
    question = "How long have I used the Acme Meter 3?"
    assert temporal_competition([older, newer], {}, question, [1, .95]) == {"new": "latest", "old": "older"}
    name_only = {**newer, "plain": "I have used the Acme Meter 3."}
    assert temporal_competition([older, name_only], {}, question, [1, .95]) == {}
