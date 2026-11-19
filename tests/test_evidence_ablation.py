import json
from pathlib import Path

import pytest

from eval.evidence_ablation import Question, Turn, evidence_metrics, measure, paired_rows, resolve_dataset, sample_rows, summarize
from eval.locomo.run import groups as locomo_groups
from eval.longmemeval.run import groups as longmem_groups, parse_item


@pytest.fixture
def longmem_fixture():
    path = Path(__file__).with_name("fixtures") / "longmemeval_synthetic.json"
    return json.loads(path.read_text())


def test_locomo_ids_categories_and_official_session_order():
    payload = [{"sample_id": "demo", "conversation": {
        "session_10": [{"speaker": "Alex", "dia_id": "D10:1", "text": "A lake.", "blip_caption": "A boat"}],
        "session_2": [{"speaker": "Jamie", "dia_id": "D2:1", "text": "A river."}],
        "session_2_date_time": "1:56 pm on 8 May, 2023",
    }, "qa": [{"category": 1, "question": "What waters did they visit?", "evidence": ["D2:1", "D10:1"]},
                {"category": 4, "question": "Who saw a lake?", "evidence": ["D10:1"]},
                {"category": 5, "question": "Who went diving?", "evidence": ["D10:1"]}]}]
    (turns, questions), = locomo_groups(payload, 3, 42)
    assert [turn.ref for turn in turns] == ["D2:1", "D10:1"]
    assert turns[0].timestamp < turns[1].timestamp
    assert "A boat" in turns[1].text
    by_category = {question.category: question for question in questions}
    assert by_category["1 multi-hop"].evidence == {"D2:1", "D10:1"}
    assert by_category["4 single-hop"].id == "demo:1"
    assert by_category["5 adversarial"].unanswerable


def test_longmem_turn_session_ids_and_unanswerable_buckets(longmem_fixture):
    turns, question = parse_item(longmem_fixture[0])
    assert [turn.ref for turn in turns] == ["answer_color_1:0", "answer_color_2:0", "answer_color_2:1"]
    assert all("answer" not in turn.ingest_session for turn in turns)
    assert [turn.role for turn in turns] == ["user", "user", "assistant"]
    assert question.evidence == {"answer_color_1:0", "answer_color_2:0"}
    assert question.evidence_sessions == {"answer_color_1", "answer_color_2"}
    assert question.latest_session == "answer_color_2" and question.older_sessions == {"answer_color_1"}
    _, abstention = parse_item(longmem_fixture[1])
    assert abstention.category == "abstention" and abstention.unanswerable


def test_ranked_evidence_recall_is_deduped_and_mrr_uses_fact_rank():
    ranked = [{"noise"}, {"A"}, {"A"}, set(), {"other"}, {"B", "C"}] + [set()] * 4 + [{"D"}]
    assert evidence_metrics(ranked, {"A", "B", "C", "D"}) == {
        "recall_at_5": 0.25, "recall_at_10": 0.75, "mrr_at_10": 0.5,
    }
    assert evidence_metrics([], {"A"})["mrr_at_10"] == 0
    assert evidence_metrics(ranked, set())["recall_at_10"] is None


def test_update_dominance_uses_turn_labels_and_verifies_archive(longmem_fixture):
    turns, question = parse_item(longmem_fixture[0])
    old = {"id": "old", "text": "old", "source_step_ids": [0]}
    new = {"id": "new", "text": "new", "source_step_ids": [1]}
    reply = {"id": "reply", "text": "okay", "source_step_ids": [2]}
    assert measure([reply, new, old], turns, question, {})["update_dominates"]
    assert not measure([old, new], turns, question, {})["update_dominates"]
    assert not measure([new], turns, question, {})["update_dominates"]
    assert measure([{"id": "new", "text": "new", "source_step_ids": [0, 1]}],
                   turns, question, {})["update_dominates"] is False
    assert measure([new], turns, question, {"new": [old]})["update_archive_verified"]
    assert measure([new], turns, question, {"new": [old]})["update_dominates"]
    assert measure([{**new, "source_step_ids": [0, 1]}], turns, question, {"new": [old]})["update_dominates"]
    assert not measure([old, new], turns, question, {"new": [old]})["update_dominates"]


def test_adversarial_noise_uses_evidence_not_the_trap_answer():
    turns = [Turn("D1:1", "1", "user", "fact", 0), Turn("D1:2", "1", "user", "noise", 1)]
    question = Question("trap", "5 adversarial", "Unsupported?", 2, frozenset({"D1:1"}), unanswerable=True)
    gold = {"text": "fact", "source_step_ids": [0]}
    extra = {"text": "noise", "source_step_ids": [1]}
    result = measure([gold, extra], turns, question, {})
    assert result["recall_at_10"] is None and result["noise_leakage"] and result["redundant_tokens"] == 2
    assert not measure([gold], turns, question, {})["noise_leakage"]
    assert measure([{**gold, "context_truncated": True}], turns, question, {})["noise_leakage"]


def test_paired_synthetic_run_is_offline_even_with_embedding_env(longmem_fixture, monkeypatch):
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "must-not-be-used")

    def reject_network(*args, **kwargs):
        pytest.fail("offline evaluation attempted network access")

    monkeypatch.setattr("urllib.request.urlopen", reject_network)
    rows = paired_rows(longmem_groups(longmem_fixture, 2, 42))
    assert {row["category"] for row in rows} == {"abstention", "knowledge-update"}
    summary = summarize(rows)
    assert summary["n"] == 2 and summary["answerable"] == 1
    assert summary["bipolar"]["turn_scored"] == 1 and summary["bipolar"]["session_scored"] == 1
    assert summary["bipolar"]["answerable_empty"] == 0
    assert summary["token_savings"] == 0


def test_local_fixture_and_failed_download_do_not_trigger_synthetic_fallback(tmp_path, monkeypatch):
    path = tmp_path / "sample.json"
    path.write_text("[]")
    assert resolve_dataset(tmp_path, ("sample.json",), "https://example.invalid/test") == path
    assert resolve_dataset(path, ("different.json",), "https://example.invalid/test") == path
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        resolve_dataset(tmp_path / "missing.json", ("sample.json",), "https://example.invalid/test")

    def unavailable(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr("urllib.request.urlopen", unavailable)
    with pytest.raises(RuntimeError, match="Pass --data-dir"):
        resolve_dataset(tmp_path, ("missing.json",), "https://example.invalid/test")
    assert not (tmp_path / "missing.json").exists()


def test_stratified_sampling_is_seeded_and_balanced_without_duplicates():
    pool = [("first", index) for index in range(20)] + [("second", index) for index in range(6)]
    first = sample_rows(pool, 10, 42, lambda row: row[0])
    assert first == sample_rows(pool, 10, 42, lambda row: row[0])
    assert len(set(first)) == 10 and sum(row[0] == "first" for row in first) == 5
