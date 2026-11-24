"""Unit tests for M0 end-to-end QA harness, turn-level retrieval, 6-bucket attribution, and regression gates."""
from __future__ import annotations

from eval.ablation_matrix import _stress
from eval.attribution.classify import ATTRIBUTION_BUCKETS, classify_failure, summarize_attribution
from eval.evidence_ablation import Turn
from eval.gates.chrono import check_chrono_gate
from eval.gates.exact import check_exact_gate
from eval.gates.version import check_version_gate
from eval.qa.harness import QAItem, evaluate_group_qa, summarize_qa_rows
from eval.qa.judge import DeterministicQAJudge, canonicalize_dates
from eval.qa.provenance import build_provenance, render_provenance_markdown
from eval.retrieval.turn_level import evaluate_turn_retrieval_item, project_hit_to_turns


def test_provenance_stamps_all_required_fields():
    prov = build_provenance(
        engine="v1_chronicle",
        embedder="HashedTfidfEmbedder",
        reranker="CalibratedLocalReranker",
        llm="MockLLMClient/DeterministicReader",
        judge="deterministic-v1",
        protocol="test-protocol",
        granularity="turn-level",
        seed=42,
        dataset="synthetic",
    )
    for field in ("engine", "embedder", "reranker", "llm", "judge", "commit_hash", "seed", "dataset_sha256", "granularity"):
        assert field in prov and prov[field] not in (None, "")
    md = render_provenance_markdown(prov)
    assert "HashedTfidfEmbedder" in md
    assert "deterministic-v1" in md


def test_deterministic_judge_dates_abstention_and_mcq():
    judge = DeterministicQAJudge()
    assert canonicalize_dates("7 May 2023") == "2023-05-07"
    assert canonicalize_dates("May 7, 2023") == "2023-05-07"

    v_date = judge.evaluate(
        question="When did Caroline go to the group?",
        gold_answer="7 May 2023",
        predicted_answer="Caroline went to the support group on 2023-05-07.",
        category="2 temporal",
    )
    assert v_date.correct is True

    v_abs_ok = judge.evaluate(
        question="What is Bob's favorite spaceship?",
        gold_answer="",
        predicted_answer="Insufficient evidence to answer.",
        category="abstention",
        unanswerable=True,
        abstained=True,
    )
    assert v_abs_ok.correct is True

    v_abs_fail = judge.evaluate(
        question="What is Bob's favorite spaceship?",
        gold_answer="",
        predicted_answer="Bob likes the Millennium Falcon.",
        category="abstention",
        unanswerable=True,
        abstained=False,
    )
    assert v_abs_fail.correct is False

    v_mcq = judge.evaluate(
        question="Which color?",
        gold_answer="Blue",
        predicted_answer="B. Blue",
        category="observation_factual",
        mcq_gold="B",
        mcq_predicted="B",
    )
    assert v_mcq.correct is True


def test_six_bucket_attribution_covers_every_stage():
    assert classify_failure(
        correct=True, unanswerable=False, gold_turns={"t1"}, ingested_turns={"t1"},
        recalled_turns={"t1"}, ranked_turns={"t1"}, packed_turns={"t1"},
    ) is None
    assert classify_failure(
        correct=False, unanswerable=False, gold_turns={"t1"}, ingested_turns=set(),
        recalled_turns=set(), ranked_turns=set(), packed_turns=set(),
    ) == "extraction_miss"
    assert classify_failure(
        correct=False, unanswerable=False, gold_turns={"t1"}, ingested_turns={"t1"},
        recalled_turns=set(), ranked_turns=set(), packed_turns=set(),
    ) == "recall_miss"
    assert classify_failure(
        correct=False, unanswerable=False, gold_turns={"t1"}, ingested_turns={"t1"},
        recalled_turns={"t1"}, ranked_turns=set(), packed_turns=set(),
    ) == "ranking_drop"
    assert classify_failure(
        correct=False, unanswerable=False, gold_turns={"t1"}, ingested_turns={"t1"},
        recalled_turns={"t1"}, ranked_turns={"t1"}, packed_turns=set(),
    ) == "assembly_truncation"
    assert classify_failure(
        correct=False, unanswerable=False, gold_turns={"t1"}, ingested_turns={"t1"},
        recalled_turns={"t1"}, ranked_turns={"t1"}, packed_turns={"t1"}, borderline_judge=True,
    ) == "judge_noise"
    assert classify_failure(
        correct=False, unanswerable=False, gold_turns={"t1"}, ingested_turns={"t1"},
        recalled_turns={"t1"}, ranked_turns={"t1"}, packed_turns={"t1"}, borderline_judge=False,
    ) == "reasoning_error"

    summary = summarize_attribution([
        {"correct": False, "category": "multi-session", "attribution": "recall_miss"},
        {"correct": True, "category": "multi-session", "attribution": None},
    ])
    assert summary["total_wrong"] == 1
    assert summary["by_bucket"]["recall_miss"]["count"] == 1
    assert set(summary["by_bucket"]) == set(ATTRIBUTION_BUCKETS)


def test_turn_level_projection_prevents_session_blob_inflation():
    turn_texts = {f"s1:{i}": f"Unrelated filler sentence number {i}" for i in range(10)}
    turn_texts["s1:7"] = "Alice graduated with a degree in Computer Science."
    projected = project_hit_to_turns("What degree did Alice graduate with?", list(turn_texts.keys()), turn_texts)
    assert projected == ["s1:7"]

    item = evaluate_turn_retrieval_item(
        ranked_turn_sets=[{"s1:7"}],
        ranked_session_sets=[{"s1"}],
        gold_turns={"s1:7"},
        gold_sessions={"s1"},
        latest_session="s2",
        older_sessions={"s1"},
    )
    assert item["turn_r10"] == 1.0
    assert item["update_dominance"] == 0.0


def test_end_to_end_qa_harness_v1_and_v2_and_ablation_fix():
    turns = [
        Turn("s1:0", "s1", "user", "user: I graduated with a degree in Mechanical Engineering.", 1_700_000_000.0, has_answer=True, ingest_session="s1", mention_time=1_700_000_000.0, speaker="user"),
        Turn("s2:0", "s2", "user", "user: My favorite coffee shop is Blue Bottle in Seattle.", 1_700_100_000.0, has_answer=False, ingest_session="s2", mention_time=1_700_100_000.0, speaker="user"),
    ]
    qas = [
        QAItem(
            id="q1",
            category="single-session-user",
            question="What degree did I graduate with?",
            gold_answer="Mechanical Engineering",
            timestamp=1_700_200_000.0,
            evidence_turns=frozenset({"s1:0"}),
            evidence_sessions=frozenset({"s1"}),
        ),
        QAItem(
            id="q2",
            category="abstention",
            question="What is the serial number of my quantum submarine?",
            gold_answer="",
            timestamp=1_700_200_000.0,
            unanswerable=True,
        ),
    ]
    for kind in ("v1_chronicle", "v2_session_md"):
        rows = evaluate_group_qa(turns, qas, engine_kind=kind)
        assert len(rows) == 2
        summary = summarize_qa_rows(rows)
        assert summary["overall"]["n"] == 2
        assert summary["overall"]["turn_r10"] == 1.0

    # Regression: eval/ablation_matrix._rank_only must no longer fail on missing parsed_query
    stress_rows = _stress(0)
    assert len(stress_rows) == 1
    assert all(str(stage) in stress_rows[0] for stage in range(6))


def test_pass_fail_regression_gates_fast():
    assert check_chrono_gate()["passed"] is True
    assert check_version_gate()["passed"] is True
    assert check_exact_gate()["passed"] is True
