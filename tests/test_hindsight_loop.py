from contextlib import closing

from memengine.chrono import parse_session_clock, resolve_temporal_spans
from memengine.engine import CognitiveMemoryEngine
from memengine.retrieval.activation import spreading_activation
from memengine.storage.base import RecallFilter
from memengine.verification import supports_claim


def retain(engine, text, *, speaker="Alice Chen", mention_time="2024-06-15", session_id="s1", user_id="default"):
    result = engine.retain(text, speaker=speaker, mention_time=mention_time,
                           session_id=session_id, user_id=user_id)
    return result["operations"][0]["line_id"]


def test_speaker_occurrence_and_scoped_version_chain():
    engine = CognitiveMemoryEngine()
    with closing(engine.store) as store:
        first = retain(engine, "I use Python v3.11.2 since June 1, 2024.")
        second = retain(engine, "I upgraded to Python v3.11.4 yesterday.",
                        mention_time="2024-06-17", speaker="Alice", session_id="s2")
        old, latest = store.facts_by_ids([first, second], scope=("default", "default", None))
        assert old["canonical_entity_id"] == latest["canonical_entity_id"]
        assert "Alice Chen use Python" in old["normalized_text"]
        assert "occurrence: 2024-06-01" in old["normalized_text"]
        assert "[2024-06-15 | Alice Chen]" in old["formatted"]
        assert latest["mention_time"] == parse_session_clock("2024-06-17")
        assert latest["occurrence_start"] == parse_session_clock("2024-06-16")
        assert old["status"] == "superseded" and latest["supersedes_id"] == first
        assert latest["attribute_slot"] == "runtime_version:python" and latest["revision"] == 1
        assert latest["source_turn_ids"] == [old["origin_turn"], latest["origin_turn"]]
        result = engine.recall("Alice Chen Python v3.11", reference_time="2024-06-18",
                               read_only=True, token_budget=384)
        assert result["ranked"][0]["id"] == second
        assert result["ranked"][0]["version_grade"] == "CONTAINED_SPECIFIC"
        engine.consolidate(force=True, timestamp=parse_session_clock("2024-06-18"))
        profile = next(row for row in store.list_observations() if row["canonical_entity_id"])
        assert profile["derived_from_ids"] == [first, second]
        assert "entity:alice chen" in store.aliases_for_query("What did Chen upgrade?", ("default", "default", None))


def test_named_month_relative_to_the_mention_year():
    for anchor, first, last in (("2024-03-11", "2023-05-01", "2023-05-31"),
                                ("2024-06-01", "2024-05-01", "2024-05-31")):
        span, = resolve_temporal_spans("I visited there last May.", anchor)
        assert (span.iso_start, span.iso_end, span.precision) == (first, last, "month")


def test_compatible_and_wildcard_ranges_recall_only_satisfying_versions():
    engine = CognitiveMemoryEngine()
    with closing(engine.store):
        old = retain(engine, "Python v3.11.1 has a cache.")
        patch = retain(engine, "Python v3.11.4 has a cache.")
        minor = retain(engine, "Python v3.12.0 has a cache.")
        retain(engine, "Rust v3.11.4 has a cache.")
        for query, expected in (("Python ~= 3.11.2", {patch}), ("Python ~= 3.11", {old, patch, minor}),
                                ("Python v3.11.x", {old, patch}), ("Python >= 3.12.0", {minor})):
            ranked = engine.recall(query, reference_time="2024-06-16", token_budget=384, read_only=True)["ranked"]
            assert {row["id"] for row in ranked} == expected
            assert {row["version_grade"] for row in ranked} == {"RANGE_SATISFIED"}


def test_four_graph_edges_and_two_hop_owner_isolation():
    engine = CognitiveMemoryEngine()
    with closing(engine.store) as store:
        first = retain(engine, "Did you calibrate the Atlas laboratory sensors yesterday?")
        second = retain(engine, "I calibrated the Atlas laboratory sensors yesterday because the gateway was offline.")
        third = retain(engine, "I calibrated Atlas laboratory sensors this week.", session_id="s2")
        alien = retain(engine, "I calibrated Atlas laboratory sensors this week.", user_id="bob")
        kinds = {row[0] for row in store.conn.execute("SELECT DISTINCT link_type FROM memory_links")}
        assert kinds == {"entity", "temporal", "semantic", "causal"}
        graph = spreading_activation(store, [{"id": first}], scope=("default", "default", None),
                                     filters=RecallFilter(), top_n=12)
        assert second in {row["id"] for row in graph}
        assert third in {row["id"] for row in graph}
        assert alien not in {row["id"] for row in graph}
        assert max(row["graph_depth"] for row in graph) <= 2
        assert all(0 < row["activation"] <= .65 for row in graph)


def test_calendar_index_and_plural_speaker_keep_original_turn():
    engine = CognitiveMemoryEngine()
    with closing(engine.store) as store:
        ident = retain(engine, "We visited Kyoto yesterday.", mention_time="2024-06-16")
        row = store.get_line(ident, scope=("default", "default", "s1"))
        assert "Alice Chen and others visited Kyoto" in row["normalized_text"]
        assert "[occurrence: 2024-06-15] 2024-06 2024" in row["normalized_text"]
        original = store.get_turn_window([row["origin_turn"]], window=0, by_id=True)
        assert original[0]["text"] == "We visited Kyoto yesterday."


def test_explicit_preference_slot_archives_only_the_replaced_preference():
    engine = CognitiveMemoryEngine()
    with closing(engine.store) as store:
        breakfast = retain(engine, "I prefer green tea for breakfast.", mention_time="2024-05-01")
        work = retain(engine, "I prefer navy blue for work.", mention_time="2024-05-02")
        latest = retain(engine, "I now prefer black coffee for breakfast.", mention_time="2024-05-03")
        scope = ("default", "default", "s1")
        rows = {row["id"]: row for row in store.facts_by_ids([breakfast, work, latest], scope=scope)}
        assert rows[latest]["supersedes_id"] == breakfast
        assert rows[breakfast]["status"] == "superseded"
        assert rows[work]["status"] == "active"


def test_claim_equivalences_preserve_subject_polarity_and_slot_values():
    options = {"equivalences": True}
    assert supports_claim("Alice purchased a piano.", "Alice bought a piano.", **options)
    assert supports_claim("The trip was 1 kilometer.", "The trip was 1000 meters.", **options)
    assert supports_claim("Alice arrived on May 3, 2024.", "Alice arrived on 2024-05-03.", **options)
    assert supports_claim("Alice purchased a piano.", "I bought a piano.", aliases=("Alice", "Alice Chen"),
                          speaker="Alice Chen", **options)
    assert not supports_claim("Alice purchased a piano.", "Bob bought a piano.", aliases=("Alice",), **options)
    assert not supports_claim("Alice purchased a piano.", "Alice didn't buy a piano.", **options)
    assert not supports_claim("Alice painted the shed.", "Alice was painting the shed.", **options)
    assert not supports_claim("The trip was 1 kilometer.", "The trip was 900 meters.", **options)
    assert not supports_claim("The temperature was 1.", "The temperature was -1.", **options)
    assert not supports_claim("Alice arrived on May 3, 2024.", "Alice arrived on 2024-05-04.", **options)


def test_observation_is_marked_stale_and_fresh_facts_are_checked():
    engine = CognitiveMemoryEngine()
    with closing(engine.store) as store:
        retain(engine, "I live in River City.", mention_time="2024-01-01", session_id="s1")
        retain(engine, "I work at Atlas Building.", mention_time="2024-02-01", session_id="s2")
        engine.consolidate(force=True, timestamp=parse_session_clock("2024-02-02"))
        obs = next(row for row in store.list_observations() if row["canonical_entity_id"])
        assert obs["is_stale"] == 0
        fresh = retain(engine, "I moved to Pine City yesterday.", mention_time="2024-03-02", session_id="s3")
        assert store.get_line(obs["id"])["is_stale"] == 1
        context = engine.build_context("Where does Alice Chen live?", reference_time="2024-03-03", token_budget=256)
        assert fresh in {row["id"] for row in context["items"] if row["kind"] == "fact"}
        assert obs["plain"] not in context["dynamic_memory"]
        engine.consolidate(force=True, timestamp=parse_session_clock("2024-03-03"))
        new = next(row for row in store.list_observations() if row["canonical_entity_id"])
        assert new["id"] != obs["id"] and new["supersedes_id"] == obs["id"]
        assert not new["is_stale"] and fresh in new["derived_from_ids"]


def test_deleted_source_invalidates_context_and_rebuilds_observation_provenance():
    engine = CognitiveMemoryEngine()
    with closing(engine.store) as store:
        address = retain(engine, "I live in River City.", mention_time="2024-01-01")
        for text, session in (("I work at Atlas Building.", "s2"), ("I study at Elm College.", "s3"),
                              ("I purchased a blue canoe.", "s4")):
            retain(engine, text, mention_time="2024-02-01", session_id=session)
        engine.consolidate(force=True, timestamp=parse_session_clock("2024-03-01"))
        before = store.list_observations()
        group = next(row for row in before if not row["canonical_entity_id"])
        assert group["plain"] in store.get_core_snapshot()["core_profile"]
        address_turn = store.get_line(address, scope=("default", "default", None))["source_turn_ids"][0]

        assert engine.delete(address, timestamp=parse_session_clock("2024-03-02"))
        stale = store.list_observations(include_stale=True)
        assert not store.list_observations()
        assert {row["id"] for row in stale if row["is_stale"]} == {row["id"] for row in before}
        before_read = store.conn.total_changes
        for query in ("Where does Alice Chen work?", "What did Alice study?"):
            context = engine.build_context(query, reference_time="2024-03-03", token_budget=256)
            assert group["plain"] not in context["system_prefix"]
            assert all(row["kind"] != "observation" for row in context["items"])
        assert store.conn.total_changes == before_read

        engine.consolidate(force=True, timestamp=parse_session_clock("2024-03-03"))
        rebuilt = store.list_observations()
        assert {row["supersedes_id"] for row in rebuilt} == {row["id"] for row in before}
        for row in rebuilt:
            assert address not in row["derived_from_ids"] and address_turn not in row["source_turn_ids"]
            assert len(row["derived_from_ids"]) == 3 and not row["is_stale"]
        assert group["plain"] in store.get_core_snapshot()["core_profile"]


def test_taboo_attention_requires_current_relevant_user_evidence():
    engine = CognitiveMemoryEngine()
    with closing(engine.store) as store:
        allergy = retain(engine, "I am allergic to peanuts.", speaker="Alice", session_id="s1")
        ordinary = retain(engine, "I cannot open the peanut container.", speaker="Alice", session_id="s1")
        negated = retain(engine, "I am not allergic to strawberries.", speaker="Alice", session_id="s1")
        question = retain(engine, "Alice is allergic to dust?", speaker="Alice", session_id="s1")
        other_person = retain(engine, "Bob is allergic to peanuts.", speaker="Bob", session_id="s2")
        foreign = retain(engine, "I am allergic to walnuts.", speaker="Alice", user_id="foreign")
        assistant = engine.retain("I am allergic to milk.", role="assistant", speaker="assistant",
                                  session_id="s1")["operations"][0]["line_id"]

        before = store.conn.total_changes
        for budget in (None, 256):
            context = engine.build_context("What is Alice allergic to?", token_budget=budget, include_short_term=False)
            facts = {row["id"]: row for row in context["ranked"]}
            assert context["ranked"][0]["id"] == allergy
            assert facts[allergy]["is_taboo"] and len(facts[allergy]["source_turn_ids"]) == 1
            assert facts[allergy]["attention_weight"] > 1
            assert "[禁忌]" in context["dynamic_memory"]
            assert facts[other_person]["is_taboo"] and facts[other_person]["attention_weight"] == 1
            assert foreign not in facts
            for ident in (ordinary, negated, question, assistant):
                assert not facts[ident]["is_taboo"]
        ordinary_query = engine.recall("What did Alice say about the peanut container?")
        assert ordinary_query["ranked"][0]["id"] == ordinary
        assert all(row["attention_weight"] == 1 for row in ordinary_query["ranked"])
        history = engine.recall("What was Alice allergic to before?")
        assert any(row["id"] == allergy for row in history["ranked"])
        assert all(row["attention_weight"] == 1 for row in history["ranked"])
        assert store.conn.total_changes == before


def test_retelling_a_taboo_does_not_move_it_to_the_frequency_belt():
    engine = CognitiveMemoryEngine()
    with closing(engine.store):
        for stamp in range(1, 8):
            engine.retain("The allergen review covers the usual snack inventory.", timestamp=stamp)
        text = "I am allergic to peanuts."
        first = engine.retain(text, timestamp=8)["operations"][0]["line_id"]
        for stamp in (9, 10):
            engine.retain(text, timestamp=stamp)
            result = engine.recall("What am I allergic to?", timestamp=stamp)
            lead = result["ranked"][0]
            assert lead["id"] == first and not lead["isolated"]
            assert lead["mention_count"] == stamp - 7
            assert lead["attention_weight"] > 1 and lead["bipolar_weight"] == 1


def test_reflect_fills_two_event_gap_and_cites_only_retrieved_memory():
    engine = CognitiveMemoryEngine()
    with closing(engine.store) as store:
        first = retain(engine, "I opened the Atlas lab at the launch on 2024-01-01.", mention_time="2024-01-20")
        second = retain(engine, "I ran the Atlas lab inspection on 2024-01-12.", mention_time="2024-01-21", session_id="s2")
        before = store.conn.total_changes
        answer = engine.reflect("How many days between the launch and the inspection?",
                                reference_time="2024-01-22", token_budget=256)
        assert 1 < answer["rounds_used"] <= 3
        assert answer["answer"] == "11 days"
        assert set(answer["cited_memory_ids"]) == {first, second}
        assert len(answer["cited_turn_ids"]) == 2
        assert [row["occurrence_start"] for row in answer["temporal_chain"]] == [
            parse_session_clock("2024-01-01"), parse_session_clock("2024-01-12")]
        assert store.conn.total_changes == before
