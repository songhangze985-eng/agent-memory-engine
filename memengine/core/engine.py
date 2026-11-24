"""UnifiedMemoryEngine: single core combining L0+FTS5, unified bi-temporal fact layer, VFS/Profile views & tool registry."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

from memengine.core.answer import unified_answer, unified_reflect
from memengine.core.assemble import unified_assemble
from memengine.core.calc import deterministic_calculate
from memengine.core.consolidate import rebuild_from_episodes, run_consolidation
from memengine.core.episodes import EpisodeLog
from memengine.core.facts import UnifiedFactLedger
from memengine.core.ingest import ingest_conversational_turn, ingest_procedural_rule
from memengine.core.plan import QueryPlan, RecallConfig, build_query_plan
from memengine.core.recall import unified_recall
from memengine.core.supersede import SlotSupersedeTracker
from memengine.engine import CognitiveMemoryEngine
from memengine.storage.base import RecallFilter
from memengine.views.profile import ProfileView
from memengine.views.vfs import VFSView


class UnifiedMemoryEngine:
    """Single consolidated memory engine replacing parallel v1/v2 engines.

    Architecture:
      - **L0 Episode Log** (``EpisodeLog``): ``.jsonl.gz`` cold chunks (0 L0 vectors) + SQLite/PG ``raw_turns_fts`` fallback.
      - **Unified Fact Layer** (``UnifiedFactLedger`` over SQLite+LanceDB or PostgreSQL):
        single ``facts`` ledger with bi-temporal fields, mandatory evidence,
        ``kind ∈ {world, experience, preference, belief, rule}``, status & slot-level supersede,
        optimistic lock versions, and rollback snapshots.
      - **Dual Ingest Tracks**: ungated conversational track (``ingest_turn``) + salience-gated
        procedural rule track (``ingest_rule``).
      - **Derived Views**: ``self.vfs`` (``VFSView``) and ``self.profile`` (``ProfileView``) — zero duplicate fact tables.
      - **Unified Retrieval & Assemble**: 4 parallel channels (vector + BM25[facts+L0 FTS5] + entity_graph + exact)
        + single ``assemble()`` with ``disclosure`` & ``profile`` parameters + feature-flagged reranking.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        *,
        db_url: str | None = None,
        transcript_dir: str | Path | None = None,
        transcript_store=None,
        llm=None,
        vector=None,
        reranker=None,
        clock: Callable[[], float] | None = None,
        config: RecallConfig | None = None,
        calculator=None,
        **engine_kwargs,
    ):
        self.config = config or RecallConfig()
        self.calculator = calculator if calculator is not None else (
            deterministic_calculate if self.config.enable_deterministic_calc else None
        )
        self._engine = CognitiveMemoryEngine(
            db_path=str(db_path),
            db_url=db_url,
            llm=llm,
            vector=vector,
            reranker=reranker,
            clock=clock,
            **engine_kwargs,
        )
        self.store = self._engine.store
        self.episodes = EpisodeLog(
            self.store,
            db_url=db_url or db_path,
            transcript_dir=transcript_dir,
            transcript_store=transcript_store,
        )
        self.ledger = UnifiedFactLedger(self.store)
        self.supersede = SlotSupersedeTracker(self.store)
        self.vfs = VFSView(self._engine, self.ledger)
        self.profile = ProfileView(self.store, self.ledger)

    def close(self) -> None:
        try:
            self.episodes.close()
        finally:
            self.store.close()

    def episode_indexed_turn_ids(self, *, user_id: str = "default", agent_id: str = "default", session_id: str | None = None) -> list[int]:
        return self.episodes.indexed_turn_ids(user_id=user_id, agent_id=agent_id, session_id=session_id)

    # ---- Dual Ingest Tracks ----

    def ingest_turn(
        self,
        text: str,
        *,
        user_id: str = "default",
        agent_id: str = "default",
        session_id: str = "default",
        role: str = "user",
        timestamp: float | None = None,
        event_time: float | None = None,
        mention_time: float | str | None = None,
        speaker: str | None = None,
        idempotency_key: str | None = None,
        ttl_seconds: float | None = None,
        expires_at: float | str | None = None,
        valid_from: float | str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Conversational ingest track (NO salience gate — maximizes conversational recall)."""
        return ingest_conversational_turn(
            self._engine,
            self.episodes,
            self.ledger,
            text,
            supersede=self.supersede if self.config.enable_slot_supersede else None,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            role=role,
            timestamp=timestamp,
            event_time=event_time,
            mention_time=mention_time,
            speaker=speaker,
            idempotency_key=idempotency_key,
            ttl_seconds=ttl_seconds,
            expires_at=expires_at,
            valid_from=valid_from,
            metadata=metadata,
        )

    def retain(self, text: str, **kwargs) -> dict:
        """Alias for conversational ``ingest_turn``."""
        return self.ingest_turn(text, **kwargs)

    def ingest_rule(
        self,
        path: str,
        *,
        title: str,
        trigger_condition: str,
        action: str,
        verification_result: str = "",
        notes: str = "",
        index_summary: str = "",
        user_id: str = "default",
        agent_id: str = "default",
        session_id: str = "default",
        confidence_score: float = 0.85,
        verification_status: str = "verified",
        salience_tag: str = "core_rule",
        always_active: bool = False,
        visible_in_context: bool = False,
        reusable_pattern: bool = True,
        future_pitfall_risk: bool = True,
        source_turn_ids: list[int] | tuple[int, ...] = (),
        source_transcript_ids: list[str] | tuple[str, ...] = (),
        expected_version: int | None = None,
        batch_id: str | None = None,
        timestamp: float | None = None,
    ) -> dict:
        """Procedural rule track (3-question salience gate + state machine + optimistic lock -> unified kind='rule')."""
        return ingest_procedural_rule(
            self._engine,
            self.episodes,
            self.ledger,
            path,
            title=title,
            trigger_condition=trigger_condition,
            action=action,
            verification_result=verification_result,
            notes=notes,
            index_summary=index_summary,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            confidence_score=confidence_score,
            verification_status=verification_status,
            salience_tag=salience_tag,
            always_active=always_active,
            visible_in_context=visible_in_context,
            reusable_pattern=reusable_pattern,
            future_pitfall_risk=future_pitfall_risk,
            source_turn_ids=source_turn_ids,
            source_transcript_ids=source_transcript_ids,
            expected_version=expected_version,
            batch_id=batch_id,
            timestamp=timestamp,
        )

    write_rule = ingest_rule

    # ---- Derived VFS View Shortcuts ----

    def read_note(self, path: str, *, user_id: str = "default", agent_id: str = "default") -> dict | None:
        return self.vfs.read_note(path, user_id=user_id, agent_id=agent_id)

    def tree_outline(self, dir_path: str = "/", *, user_id: str = "default", agent_id: str = "default") -> str:
        return self.vfs.tree_outline(dir_path, user_id=user_id, agent_id=agent_id)

    def compact_directory(self, dir_path: str, *, user_id: str = "default", agent_id: str = "default", timestamp: float | None = None) -> dict:
        return self.vfs.compact_directory(dir_path, user_id=user_id, agent_id=agent_id, timestamp=timestamp)

    def rollback(self, *, user_id: str = "default", agent_id: str = "default", path: str | None = None, snapshot_id: str | None = None, batch_id: str | None = None) -> dict:
        return self.vfs.rollback(user_id=user_id, agent_id=agent_id, path=path, snapshot_id=snapshot_id, batch_id=batch_id)

    # ---- Unified Recall, Assemble, Answer, Reflect ----

    def plan(self, query: str, *, reference_time: float | None = None, config: RecallConfig | None = None) -> QueryPlan:
        return build_query_plan(query, reference_time=reference_time, config=config or self.config)

    def recall(
        self,
        query: str,
        *,
        user_id: str = "default",
        agent_id: str = "default",
        session_id: str | None = None,
        reference_time: float | None = None,
        mode: str = "reconstructive",
        token_budget: int = 2048,
        top_k: int | None = None,
        packing_mode: str | None = None,
        include_cold: bool = False,
        filters: RecallFilter | None = None,
        config: RecallConfig | None = None,
        _trace: list[dict] | None = None,
    ) -> dict:
        return unified_recall(
            self._engine,
            self.episodes,
            self.ledger,
            query,
            supersede=self.supersede,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            reference_time=reference_time,
            mode=mode,
            token_budget=token_budget,
            top_k=top_k,
            packing_mode=packing_mode,
            include_cold=include_cold,
            filters=filters,
            config=config or self.config,
            _trace=_trace,
        )

    def build_context(
        self,
        query: str,
        *,
        user_id: str = "default",
        agent_id: str = "default",
        session_id: str | None = None,
        reference_time: float | None = None,
        mode: str = "reconstructive",
        token_budget: int = 2048,
        top_k: int | None = None,
        packing_mode: str | None = None,
        include_cold: bool = False,
        filters: RecallFilter | None = None,
        disclosure: Literal["full", "progressive"] | None = None,
        profile: Literal["conversational", "coding_agent"] | None = None,
        drilldown_top_n: int = 2,
        drilldown_window: int = 0,
        config: RecallConfig | None = None,
        _trace: list[dict] | None = None,
    ) -> dict:
        cfg = config or self.config
        eff_profile = profile or cfg.profile
        eff_disclosure = disclosure or cfg.disclosure
        recall_result = self.recall(
            query,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            reference_time=reference_time,
            mode=mode,
            token_budget=token_budget,
            top_k=top_k,
            packing_mode=packing_mode,
            include_cold=include_cold,
            filters=filters,
            config=cfg,
            _trace=_trace,
        )
        return unified_assemble(
            self.store,
            self.ledger,
            self.vfs,
            self.profile,
            recall_result,
            supersede=self.supersede,
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            token_budget=token_budget,
            drilldown_top_n=drilldown_top_n,
            drilldown_window=drilldown_window,
            disclosure=eff_disclosure,
            profile=eff_profile,
        )

    def _answer_from_context(self, query: str, bundle: dict) -> dict:
        return unified_answer(self._engine, query, bundle, calculator=self.calculator)

    def answer(
        self,
        query: str,
        *,
        user_id: str = "default",
        agent_id: str = "default",
        session_id: str | None = None,
        reference_time: float | None = None,
        token_budget: int = 2048,
        mode: str = "reconstructive",
        disclosure: Literal["full", "progressive"] = "full",
        profile: Literal["conversational", "coding_agent"] | None = None,
    ) -> dict:
        bundle = self.build_context(
            query,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            reference_time=reference_time,
            token_budget=token_budget,
            mode=mode,
            disclosure=disclosure,
            profile=profile,
        )
        return self._answer_from_context(query, bundle)

    def reflect(
        self,
        query: str,
        *,
        user_id: str = "default",
        agent_id: str = "default",
        session_id: str = "default",
        reference_time: float | None = None,
        token_budget: int = 2048,
    ) -> dict:
        bundle = self.build_context(
            query,
            user_id=user_id,
            agent_id=agent_id,
            session_id=None,
            reference_time=reference_time,
            token_budget=token_budget,
        )
        return unified_reflect(
            self._engine,
            self.episodes,
            self.ledger,
            query,
            bundle,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            timestamp=reference_time,
        )

    # ---- Maintenance & Lifecycle ----

    def consolidate(self, *, user_id: str = "default", agent_id: str = "default", min_new_facts: int = 8, force: bool = False, timestamp: float | None = None) -> dict:
        return run_consolidation(
            self._engine,
            self.episodes,
            self.ledger,
            user_id=user_id,
            agent_id=agent_id,
            min_new_facts=min_new_facts,
            force=force,
            timestamp=timestamp,
        )

    def rebuild_from_episodes(self, *, user_id: str = "default", agent_id: str = "default", session_id: str | None = None) -> dict:
        return rebuild_from_episodes(self, user_id=user_id, agent_id=agent_id, session_id=session_id)

    def forget(self, target=None, *, query: str | None = None, hard: bool = False, user_id: str = "default", agent_id: str = "default") -> bool:
        return self._engine.forget(target, query=query, hard=hard, user_id=user_id, agent_id=agent_id)

    def drilldown(self, turn_ids: list[int], *, window: int = 1, user_id: str = "default", agent_id: str = "default", session_id: str | None = None) -> list[dict]:
        return self.store.get_turn_window(turn_ids, window, user_id, agent_id, by_id=True, session_id=session_id)

    def history(self, entity_or_memory_id: str, *, user_id: str = "default", agent_id: str = "default", session_id: str | None = None) -> list[dict]:
        return self.store.memory_history(entity_or_memory_id, scope=(user_id, agent_id, session_id))

    def as_tools(self, *, user_id: str = "default", agent_id: str = "default", session_id: str | None = None):
        from memengine.adapters.registry import UnifiedToolRegistry

        return UnifiedToolRegistry(self, user_id=user_id, agent_id=agent_id, session_id=session_id)

    def dispatch_tool(self, name: str, arguments: dict | str, *, user_id: str = "default", agent_id: str = "default", session_id: str | None = None):
        return self.as_tools(user_id=user_id, agent_id=agent_id, session_id=session_id).dispatch(name, arguments)
