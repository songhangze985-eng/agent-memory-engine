from eval.membench.items import evidence_steps, flatten_turns, format_question, normalize_trajectory
from eval.membench.score import choice_correct, lexical_choice, parse_choice, recall_at_10
from eval.membench.run import _eval_item


def test_parse_choice_matches_the_official_letter():
    assert parse_choice("B") == "B"
    assert parse_choice('{"choice": "D"}') == "D"
    assert parse_choice("The answer is C.") == "C"
    assert parse_choice("I think both A and B work") is None
    assert choice_correct("B", "B")
    assert not choice_correct("根据现有记忆，没有足够的直接关联可以确认这一点。", "B")


def test_nested_session_evidence_ids():
    turns = flatten_turns(
        [
            [{"mid": 0, "message": "a"}, {"mid": 2, "message": "b"}],
            [{"mid": 0, "message": "c"}],
        ]
    )
    assert [turn["global"] for turn in turns] == [0, 1, 2]
    assert evidence_steps([[2, 0], [0, 1]], turns) == [1, 2]


def test_lexical_chooser_uses_context_and_not_the_label():
    choices = {"A": "Bachelor", "B": "Associate Degree", "C": "High School", "D": "Master"}
    assert lexical_choice("", choices) == "A"
    context = "[step 0] My subordinate's education is an Associate Degree."
    assert lexical_choice(context, choices) == "B"


def test_mock_item_scores_full_context_and_leaves_native_unscored(tmp_path):
    item = normalize_trajectory(
        {
            "tid": 7,
            "message_list": [
                {
                    "message": "My subordinate's education is an Associate Degree.",
                    "time": "'2024-10-01 08:00' Tuesday",
                    "place": "Boston, MA",
                }
            ],
            "QA": {
                "question": "What is the education level of the subordinate?",
                "answer": "Associate Degree",
                "target_step_id": [0],
                "choices": {
                    "A": "Bachelor",
                    "B": "Associate Degree",
                    "C": "High School",
                    "D": "Master",
                },
                "ground_truth": "B",
                "time": "'2024-10-02 08:00' Wednesday",
            },
        },
        category="observation_factual",
        subtype="simple",
        scenario="roles",
    )
    assert "Reply with only one letter" in format_question(item)
    row = _eval_item(item, client=None, mock=True)
    assert row["route"] == "TASK_POISON_CIRCLE"
    assert row["full_context_correct"] is True
    assert row["no_memory_choice"] == "A"
    assert row["no_memory_correct"] is False
    assert row["native_response"].rstrip().endswith("B")
    assert row["native_correct"] is True
    assert row["same_head_correct"] is True
    assert row["same_head_choice"] != row["no_memory_choice"]
    assert row["recall_at_10"] == 1.0
    assert row["n_turns"] == 1
    assert row["ingest_tokens"] > 0


def _sample_item(tid=7, category="observation_factual"):
    return normalize_trajectory(
        {
            "tid": tid,
            "message_list": [
                {
                    "message": "My subordinate's education is an Associate Degree.",
                    "time": "'2024-10-01 08:00' Tuesday",
                    "place": "Boston, MA",
                }
            ],
            "QA": {
                "question": "What is the education level of the subordinate?",
                "answer": "Associate Degree",
                "target_step_id": [0],
                "choices": {
                    "A": "Bachelor",
                    "B": "Associate Degree",
                    "C": "High School",
                    "D": "Master",
                },
                "ground_truth": "B",
                "time": "'2024-10-02 08:00' Wednesday",
            },
        },
        category=category,
        subtype="simple",
        scenario="roles",
    )


def test_checkpoint_resume_skips_finished(tmp_path):
    from eval.membench.run import _evaluate_all, _item_key

    first = _sample_item(7)
    second = _sample_item(8)
    path = tmp_path / "out.jsonl"
    first_row = _eval_item(first, client=None, mock=True)
    path.write_text(__import__("json").dumps(first_row, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = _evaluate_all([first, second], client=None, mock=True, concurrency=1, checkpoint_path=path)
    assert rows[0]["tid"] == 7
    assert rows[1]["tid"] == 8
    assert rows[1]["full_context_correct"] is True
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(_item_key(__import__("json").loads(line)) == _item_key(first) for line in lines if not __import__("json").loads(line).get("_checkpoint_meta"))
    # A second resume should not append another copy of already-finished items.
    before = path.read_text(encoding="utf-8")
    again = _evaluate_all([first, second], client=None, mock=True, concurrency=1, checkpoint_path=path)
    assert [row["tid"] for row in again] == [7, 8]
    assert path.read_text(encoding="utf-8") == before


def test_fatal_llm_error_keeps_checkpoint(tmp_path):
    from eval.membench.run import _evaluate_all
    from memengine.llm import FatalLLMError

    first = _sample_item(7)
    second = _sample_item(8)
    path = tmp_path / "out.jsonl"
    path.write_text(__import__("json").dumps(_eval_item(first, client=None, mock=True), ensure_ascii=False) + "\n", encoding="utf-8")

    class Boom:
        def complete(self, prompt: str) -> str:
            raise FatalLLMError(402, "LLM HTTP 402")

    try:
        _evaluate_all([first, second], Boom(), mock=False, concurrency=1, checkpoint_path=path)
    except SystemExit as exc:
        assert "402" in str(exc)
    else:
        raise AssertionError("expected SystemExit")
    saved = [
        __import__("json").loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not __import__("json").loads(line).get("_checkpoint_meta")
    ]
    assert [row["tid"] for row in saved] == [7]


def test_eval_item_records_notebook_debug_fields():
    item = _sample_item(7)
    row = _eval_item(item, client=None, mock=True)
    assert row["notebook_n_lines"] >= 1
    assert 0 in row["evidence_in_notebook"]
    assert row["evidence_missing_from_notebook"] == []
    assert row["write_actions"]
    assert "retrieved_preview" in row
    assert row["retrieved_steps"]


def test_search_falls_back_when_question_names_no_entities():
    from memengine.engine import CognitiveMemoryEngine
    from memengine.llm import MockLLMClient

    engine = CognitiveMemoryEngine(db_path=":memory:", llm=MockLLMClient())
    engine.add("[step 0] user: I really dig the movie Ran\nagent: Nice.", timestamp=1.0)
    retrieved = engine.search("What type might I prefer?")
    try:
        assert retrieved["entity_fallback"] is True
        texts = " ".join(fact.get("text") or "" for fact in (retrieved["outer"] + retrieved["inner"]))
        assert "[step 0]" in texts
        assert "Ran" in texts
    finally:
        engine.store.close()


def test_eval_writes_diagnostic_artifact(tmp_path):
    item = _sample_item(7, category="observation_factual")
    row = _eval_item(item, client=None, mock=True, artifacts_dir=tmp_path)
    path = tmp_path / "observation_factual_7.json"
    assert path.is_file()
    data = __import__("json").loads(path.read_text(encoding="utf-8"))
    assert data["tid"] == 7
    assert data["question_stem"] == item["question"]
    assert "Zucchini" not in data["question_stem"]
    assert "notebook_2d" in data
    assert "z_index" in data
    assert "empty_retrieval" in data
    assert row["empty_retrieval"] is False
    assert row["parsed_query"]["categories"] or row["notebook_n_lines"] >= 1


def test_question_stem_never_includes_choices():
    from eval.membench.items import question_stem

    item = _sample_item(7)
    stem = question_stem(item)
    formatted = format_question(item)
    assert "Associate Degree" in formatted
    assert stem == "What is the education level of the subordinate?"
    assert "A." not in stem and "Bachelor" not in stem
