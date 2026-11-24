"""Step A & G5 tests: frozen protocol.yaml, dev/test split disjointness, provenance guard & 3-arm harness."""
from __future__ import annotations

import pytest

from eval.evidence_ablation import Turn
from eval.qa.harness import QAItem, evaluate_group_qa, summarize_qa_rows
from eval.qa.protocol import (
    DEFAULT_PROTOCOL_PATH,
    aggregate_multi_seed_reports,
    load_protocol,
    partition_by_split,
    validate_provenance,
)
from eval.qa.provenance import build_provenance


def test_frozen_protocol_yaml_loads_and_declares_required_sections():
    proto, sha = load_protocol(DEFAULT_PROTOCOL_PATH)
    assert len(sha) == 64
    assert proto["protocol_version"] == "1.0.0"
    assert set(proto["arms"]) == {"full_context", "turn_rrf", "unified"}
    assert "open_20b" in proto["backbones"]
    assert "frontier" in proto["backbones"]
    assert len(proto["judge"]["seeds"]) >= 3
    assert proto["splits"]["tuning_split"] == "dev"
    assert proto["splits"]["final_report_split"] == "test"
    assert "hindsight_open_20b" in proto["reference_lines"]


def test_dev_and_test_splits_are_strictly_disjoint_and_deterministic():
    ids = [f"q_{i:04d}" for i in range(400)]
    dev = partition_by_split(ids, lambda x: x, split="dev", dev_ratio=0.25)
    test = partition_by_split(ids, lambda x: x, split="test", dev_ratio=0.25)
    full = partition_by_split(ids, lambda x: x, split="full", dev_ratio=0.25)

    assert set(dev).isdisjoint(set(test))
    assert set(dev) | set(test) == set(full)
    assert 70 <= len(dev) <= 130  # ~25% of 400
    assert partition_by_split(ids, lambda x: x, split="dev", dev_ratio=0.25) == dev


def test_g5_provenance_validator_rejects_missing_metadata():
    valid = build_provenance(
        engine="unified",
        embedder="HashedTfidfEmbedder",
        reranker="CalibratedLocalReranker",
        llm="MockLLMClient/DeterministicReader",
        extractor_model="DeterministicRuleExtractor",
        judge="deterministic-v1",
        protocol="Step A test",
        granularity="turn-level",
        seed=42,
        split="dev",
        dataset="synthetic",
    )
    validate_provenance(valid)
    assert valid["extractor_model"] == "DeterministicRuleExtractor"
    assert valid["reader_model"] == "MockLLMClient/DeterministicReader"
    assert valid["split"] == "dev"
    assert len(valid["protocol_sha256"]) == 64

    broken = dict(valid)
    broken.pop("dataset_sha256")
    with pytest.raises(ValueError, match="dataset_sha256"):
        validate_provenance(broken)


def test_three_arm_baseline_harness_and_multiseed_aggregation():
    turns = [
        Turn("s1:0", "s1", "user", "User: I moved to Seattle in January 2024.", 1_705_000_000.0, has_answer=False, speaker="User"),
        Turn("s2:0", "s2", "user", "User: Actually I relocated from Seattle to Austin in June 2024.", 1_718_000_000.0, has_answer=True, speaker="User"),
        Turn("s3:0", "s3", "user", "User: My favorite coffee shop is Blue Bottle.", 1_720_000_000.0, has_answer=False, speaker="User"),
    ]
    qas = [
        QAItem(
            id="q1",
            category="knowledge-update",
            question="Where do I live now?",
            gold_answer="Austin",
            timestamp=1_725_000_000.0,
            evidence_turns=frozenset({"s2:0"}),
            evidence_sessions=frozenset({"s2"}),
            older_sessions=frozenset({"s1"}),
            latest_session="s2",
        ),
        QAItem(
            id="q2",
            category="abstention",
            question="What is my passport number?",
            gold_answer="",
            timestamp=1_725_000_000.0,
            unanswerable=True,
        ),
    ]

    for arm in ("full_context", "turn_rrf", "unified"):
        rows = evaluate_group_qa(turns, qas, engine_kind=arm, token_budget=1024)
        assert len(rows) == 2
        summary = summarize_qa_rows(rows)
        assert summary["overall"]["n"] == 2
        assert "attribution" in summary

    # Multi-seed aggregation produces intervals
    seed_reps = []
    for seed in (42, 137, 2026):
        rows = evaluate_group_qa(turns, qas, engine_kind="unified", token_budget=1024)
        prov = build_provenance(
            engine="unified",
            embedder="HashedTfidfEmbedder",
            reranker="CalibratedLocalReranker",
            llm="MockLLMClient/DeterministicReader",
            judge="deterministic-v1",
            protocol="test",
            granularity="turn-level",
            seed=seed,
            split="dev",
            dataset="synthetic",
        )
        seed_reps.append({"benchmark": "Synthetic", "provenance": prov, "summary": summarize_qa_rows(rows)})

    agg = aggregate_multi_seed_reports(seed_reps)
    assert agg["seeds"] == [42, 137, 2026]
    assert agg["overall"]["qa_accuracy"]["min"] <= agg["overall"]["qa_accuracy"]["mean"] <= agg["overall"]["qa_accuracy"]["max"]
