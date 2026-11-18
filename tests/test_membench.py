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
