from collections import Counter, defaultdict
from contextlib import closing
import hashlib

from eval.abstention_cohort import build_cases, write_cohort
from eval.paired import make_engine


def test_abstention_cohort_keeps_anchor_groups_together(tmp_path):
    groups = defaultdict(dict)
    cases = list(build_cases())
    assert len({case["id"] for case in cases}) == len(cases) == 3500
    assert Counter(case["outcome"] for case in cases) == {"complete": 840, "unsupported": 2520, "partial": 140}
    assert len({case["query"] for case in cases if case["variant"] in {"answerable", "false_premise"}}) == 1680
    for case in cases:
        groups[case["group"]][case["variant"]] = case
    assert len(groups) == 840
    for variants in groups.values():
        positive = variants["answerable"]
        removed = variants["evidence_removed"]
        wrong = variants["false_premise"]
        neighbor = variants["cross_user"]
        assert len({case["split"] for case in variants.values()}) == 1
        assert positive["query"] == removed["query"] == neighbor["query"] != wrong["query"]
        assert positive["turns"] == wrong["turns"]
        support_refs = {slot["source_ref"] for slot in positive["expected_slots"].values()}
        assert all(turn["ref"] not in support_refs for turn in removed["turns"])
        assert [turn for turn in neighbor["turns"] if turn["user_id"] == "owner"] == removed["turns"]
        assert {turn["ref"] for turn in neighbor["turns"] if turn["user_id"] == "neighbor"} == support_refs
        for case in variants.values():
            owner_turns = {turn["ref"]: turn for turn in case["turns"] if turn["user_id"] == case["user_id"]}
            known = [slot for slot in case["expected_slots"].values() if slot is not None]
            assert case["outcome"] == ("unsupported" if not known else
                                       "complete" if len(known) == len(case["expected_slots"]) else "partial")
            assert all(slot["value"] in owner_turns[slot["source_ref"]]["text"] for slot in known)

    path = tmp_path / "cohort.jsonl"
    manifest = write_cohort(path, 42)
    assert manifest["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest["counts"]["holdout"]["cross_user"] == 168
    assert manifest == write_cohort(path, 42)


def test_near_neighbor_support_stays_in_the_other_tenant():
    case = next(case for case in build_cases(groups_per_intent=1) if case["variant"] == "cross_user")
    engine = make_engine("mock")
    with closing(engine.store):
        for turn in case["turns"]:
            engine.retain(turn["text"], user_id=turn["user_id"], session_id=turn["session_id"],
                          timestamp=turn["timestamp"], role=turn["role"])
        options = {"timestamp": case["timestamp"], "include_short_term": False, "token_budget": 512}
        owner = engine.build_context(case["query"], user_id=case["user_id"], **options)
        neighbor = engine.build_context(case["query"], user_id="neighbor", **options)
        support = next(turn["text"] for turn in case["turns"] if turn["user_id"] == "neighbor")
        assert support not in owner["dynamic_memory"]
        assert support in neighbor["dynamic_memory"]
        assert all(row["user_id"] == case["user_id"] for row in owner["evidence_map"].values())


def test_calibration_counts_source_groups_once_and_keeps_threshold_ties():
    from eval.abstention_calibration import measure, score_floors

    rows = [
        {"group": str(index), "runtime_group": "multi-facet", "outcome": "complete",
         "features": {"support": 0.75}, "empty": False, "variant": "answerable"}
        for index in range(49)
    ]
    rows.append({**rows[0], "outcome": "partial", "variant": "partial_evidence", "features": {"support": 0.5}})
    assert score_floors(rows[1:49], "support")["multi-facet"]["threshold"] is None
    thresholds = score_floors(rows, "support")
    assert thresholds["multi-facet"] == {"groups": 49, "threshold": 0.5}

    negatives = [
        {**rows[0], "outcome": "unsupported", "variant": "evidence_removed", "features": {"support": 0.25}},
        {**rows[0], "outcome": "unsupported", "variant": "evidence_removed", "features": {"support": 0.0}, "empty": True},
    ]
    measured, _ = measure([rows[-1], *negatives], "support", thresholds)
    assert measured["partial"]["total_rejection_rate"] == 0
    assert measured["unsupported"]["baseline_empty"] == 1
    assert measured["unsupported"]["additional_rejected"] == 1
    assert measured["unsupported"]["total_rejection_rate"] == 1


def test_coordinated_queries_reuse_scope_without_matching_a_different_choice():
    from memengine.retrieval.abstention import abstention_features

    engine = make_engine("mock")
    with closing(engine.store):
        engine.retain("On the Harbor Line trip I chose the book The Lantern Atlas.", timestamp=100, session_id="book")
        engine.retain("On the Quarry Line trip I chose the podcast Morning Signals.", timestamp=101, session_id="other")
        engine.retain("On the Harbor Line trip I discussed several podcasts.", timestamp=102, session_id="discussion")
        query = "Which book and which podcast did I choose on the Harbor Line trip?"
        context = engine.answer(query, timestamp=110, token_budget=256, include_short_term=False)
        book = "Which book did I choose on the Harbor Line trip"
        podcast = "which podcast did I choose on the Harbor Line trip"
        assert context["parsed_query"]["facets"] == [book, podcast]
        assert set(context["answered_facets"]) == {book}
        assert context["unanswered_facets"] == [podcast]
        assert abstention_features(query, context["ranked"], [book, podcast])["facet_support"] == 1.0


def test_facet_support_requires_one_visible_statement_with_a_value():
    from memengine.retrieval.coverage import supported_facet

    template = {"role": "user", "status": "active", "source_turn_ids": [1], "speaker": "Alice", "entity_anchor": True}
    target = ["Which book did I choose on the Harbor Line trip"]
    for text in (
        "[2024-05-01 | Alice] The book I chose on the Harbor Line trip.",
        "I discussed a book on the Harbor Line trip. I chose a podcast called Morning Signals.",
        "I did not choose the book The Lantern Atlas on the Harbor Line trip.",
        "I chose the book The Lantern Atlas on the Quarry Line trip.",
    ):
        assert not supported_facet(target, [{**template, "text": text, "plain": text}], require_entity=True)
