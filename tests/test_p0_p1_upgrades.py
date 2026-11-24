import sqlite3
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from agentmem_core import CognitiveMemoryEngine
from agentmem_core.identifiers import extract_identifiers
from agentmem_core.neural import LocalCrossEncoderReranker, LocalNeuralEmbedder, NeuralBackendUnavailableError
from agentmem_core.reranker import CalibratedLocalReranker
from agentmem_core.store import Store
from agentmem_core.textutil import estimate_tokens
from agentmem_core.vector import HashedTfidfEmbedder


class WeakReranker:
    is_calibrated = True

    def score(self, query, facts):
        return [0.01] * len(facts)


def engine(**kwargs):
    config = dict(vector=HashedTfidfEmbedder(), reranker=CalibratedLocalReranker(), enable_casual_bypass=False)
    return CognitiveMemoryEngine(**{**config, **kwargs})


def put(memory, plain, stamp, *, entity=None, attribute="", raw=None, session="default"):
    scope = ("default", "default", session)
    with memory.store.transaction():
        turn = memory.store.append_turn(raw or plain, stamp, scope=scope)
        block = memory.store.create_block("Records", stamp, scope=scope)
        fact = memory.store.insert_fact(block_id=block, plain=plain, formatted=plain, ingest_time=stamp, scope=scope,
                                        source_turn_ids=[turn], attribute=attribute, zkeys=[entity] if entity else ())
    return fact, turn


def test_precise_symbols_and_complete_versions():
    memory = engine()
    values = ("v3.10", "v3.11", "v3.12", "v3.11.1", "ERR_403", "agent-memory-engine", "CognitiveMemoryEngine",
              "memengine/store.py", "Qwen2.5-7B", "《青梧发布手册》")
    ids = {value: put(memory, f"Recovery configuration and release bug information for {value}.", stamp)[0]
           for stamp, value in enumerate(values, 1)}
    full = memory.retrieve("Where is the v3.11 configuration for the release bug?", read_only=True)
    assert full["ranked"][0]["id"] == ids["v3.11"]
    assert full["ranked"][0]["exact_ident_hit"] == 1
    assert full["gate_stats"]["version_mismatch_count"] == 3
    assert {ids[key] for key in ("v3.10", "v3.12", "v3.11.1")}.isdisjoint(item["id"] for item in full["ranked"])
    for value in values:
        hit = memory.retrieve(f"Which record covers {value}?", read_only=True)
        assert hit["ranked"][0]["id"] == ids[value]
    for value in ("v3.11", "ERR_403", "agent-memory-engine", "memengine/store.py", "Qwen2.5-7B"):
        assert [fact["id"] for fact in memory.store.search_bm25(value)] == [ids[value]]
    assert extract_identifiers("3.11.2")[0].major_minor_family == "v3.11"
    assert extract_identifiers("v2.4.1+build.2")[0].ident_norm == "v2.4.1+build.2"
    comparison = memory.retrieve("从 v3.10 升级到 v3.11 改了什么？", read_only=True)
    assert {item["id"] for item in comparison["ranked"][:2]} == {ids["v3.10"], ids["v3.11"]}
    assert comparison["gate_stats"]["version_mismatch_count"] == 0
    missing = memory.retrieve("Which configuration is for v3.13?", read_only=True)
    assert missing["empty_recall"] and missing["gate_stats"]["empty_reason"] == "no_exact_version"


def test_fts_symbol_migration_rebuilds_raw_hot_and_cold(tmp_path):
    path = tmp_path / "legacy.db"
    original = engine(db_path=path)
    cold, raw = put(original, "Old runtime was v3.10.", 1)
    hot, _ = put(original, "Service uses v3.11.", 95 * 86400)
    original.decay_and_archive(timestamp=100 * 86400, cold_inactive_days=90)
    original.store.close()
    with sqlite3.connect(path) as conn:
        for name in ("facts_fts", "facts_cold_fts", "raw_turns_fts"):
            conn.execute(f"DROP TABLE {name}")
            columns = "fact_id UNINDEXED,user_id UNINDEXED,agent_id UNINDEXED," if name != "raw_turns_fts" else ""
            conn.execute(f"CREATE VIRTUAL TABLE {name} USING fts5({columns}fts_tokens, tokenize='unicode61')")
        conn.execute("DROP TABLE exact_identifiers")
        conn.execute("DELETE FROM meta WHERE key='identifier_index_version'")
    for _ in range(2):
        migrated = Store(path)
        assert [fact["id"] for fact in migrated.search_bm25("v3.11")] == [hot]
        assert not migrated.search_bm25("v3.10")
        assert [fact["id"] for fact in migrated.search_bm25("v3.10", include_cold=True)] == [cold]
        assert migrated.conn.execute("SELECT rowid FROM raw_turns_fts WHERE raw_turns_fts MATCH ?", ('"v3.10"',)).fetchone()[0] == raw
        assert migrated.identifiers_for([hot])[hot] == {"v3.11": "version"}
        assert migrated.conn.execute("PRAGMA foreign_key_check").fetchall() == []
        migrated.close()


def test_temporal_competition_is_read_only_and_historical_queries_keep_old_state():
    memory = engine()
    first = datetime(2025, 2, 1, tzinfo=timezone.utc).timestamp()
    last = datetime(2025, 8, 1, tzinfo=timezone.utc).timestamp()
    old, _ = put(memory, "Atlas routing currently uses the East gateway.", first,
                 entity="实体:Atlas", attribute="gateway", session="s1")
    new, _ = put(memory, "Atlas routing currently uses the West gateway.", last,
                 entity="实体:Atlas", attribute="gateway", session="s2")
    changes = memory.store.conn.total_changes
    current = memory.retrieve("What gateway does Atlas routing use currently?", timestamp=last + 1, read_only=True)
    assert [hit["id"] for hit in current["ranked"][:2]] == [new, old]
    assert [hit["temporal_weight"] for hit in current["ranked"][:2]] == [1.12, 0.65]
    assert current["ranked"][1]["state_tag"] == "[可能已过时/较早记录]"
    assert all(hit["status"] == "active" for hit in current["ranked"])
    context = memory.build_context("What gateway does Atlas routing use?")
    assert "[2025-08 | 最新]" in context["dynamic_memory"] and "[2025-02 | 较早]" in context["dynamic_memory"]
    historical = memory.retrieve("What gateway did Atlas routing use on 2025-02-01?", timestamp=last + 1, read_only=True)
    assert historical["ranked"][0]["id"] == old and historical["ranked"][0]["temporal_role"] is None
    assert memory.store.conn.total_changes == changes


def test_residual_budget_skips_redundancy_and_never_displaces_candidate_atoms():
    class Confident(WeakReranker):
        def score(self, query, facts):
            return [0.8] * len(facts)

    memory = engine(reranker=Confident())
    same = "Atlas calibration is verified at the North laboratory."
    first, redundant_turn = put(memory, same, 1)
    long = "unindexed detail " * 30 + "factory inspection code Atlas calibration wobble " + " additional references" * 50
    second, long_turn = put(memory, "Atlas calibration needs an inspection certificate.", 2, raw=long)
    query = "What does Atlas calibration require?"
    context = memory.build_context(query, token_budget=1024)
    assert context["system_prefix"] == "" and context["token_breakdown"]["prefix_tokens"] == 0
    assert context["evidence_turn_ids"] == [long_turn] and redundant_turn not in context["evidence_turn_ids"]
    excerpt = next(item["text"] for item in context["items"] if item["kind"] == "raw_turn")
    assert "factory inspection code" in excerpt and estimate_tokens(excerpt) <= 120
    tight = memory.build_context(query, token_budget=35)
    assert {item["id"] for item in tight["items"] if item["kind"] == "fact"} == {first, second}
    tokens = tight["token_breakdown"]
    assert tokens["drilldown_tokens"] <= int(35 * 0.25) and tokens["total_injected_tokens"] <= 35


def test_calibrated_gate_abstains_but_protects_exact_hits():
    calibrated = engine(reranker=WeakReranker())
    fact, _ = put(calibrated, "Version v3.11 stores the kayak rental information.", 1)
    weak = calibrated.retrieve("Where should I rent a kayak?", read_only=True)
    assert weak["empty_recall"] and weak["gate_stats"]["empty_reason"] == "below_absolute_threshold"
    pinned = calibrated.retrieve("Which release documents v3.11?", read_only=True)
    assert pinned["ranked"][0]["id"] == fact and pinned["gate_stats"]["max_ce_score"] == 0.01
    calibrated.reranker = CalibratedLocalReranker()
    assert not calibrated.retrieve("Where should I rent a kayak?", read_only=True)["empty_recall"]


def test_optional_neural_protocol_and_persistent_vectors(tmp_path, monkeypatch):
    import agentmem_core.neural as neural

    calls = []

    class Encoder:
        def encode(self, texts, **kwargs):
            calls.append(texts)
            return np.tile([3.0, 4.0], (len(texts), 1))

        def predict(self, pairs, **kwargs):
            calls.append(pairs)
            return [-2, 2]

    encoder = Encoder()
    with monkeypatch.context() as mock:
        mock.setattr(neural, "_load", lambda *args, **kwargs: (encoder, SimpleNamespace(nn=SimpleNamespace(Identity=object))))
        embedder = LocalNeuralEmbedder("unit-test/e5")
        assert np.allclose(embedder.encode(["one", "two"]), [[0.6, 0.8], [0.6, 0.8]])
        cross = LocalCrossEncoderReranker("unit-test/cross")
        scores = cross.score("q", [{"plain": "a"}, {"plain": "b"}])
        assert cross.is_calibrated and scores == pytest.approx([0.11920292, 0.88079708])
        path = tmp_path / "vectors.db"
        memory = engine(db_path=path, vector=embedder)
        put(memory, "A stored sentence.", 1)
        memory.store.close()
        store = Store(path, embedder=embedder)
        model, dim, vector = store.conn.execute("SELECT model_name,dim,vec FROM fact_vectors").fetchone()
        assert model == embedder.model_name and dim == 2 and np.frombuffer(vector, dtype="<f4") == pytest.approx([0.6, 0.8])
        assert store.search_vector("related question")[0]["vector_score"] == pytest.approx(1)
        assert calls[-2:] == [["passage: A stored sentence."], ["query: related question"]]
        store.close()
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(NeuralBackendUnavailableError, match=r"Install .\[neural\]"):
        LocalNeuralEmbedder("unit-test/missing")
