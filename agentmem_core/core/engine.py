"""UnifiedMemoryEngine: universal single core driven by declarative MemoryProfile & ScopeChain."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal

from agentmem_core.core.answer import unified_answer, unified_reflect
from agentmem_core.core.assemble import unified_assemble
from agentmem_core.core.calc import deterministic_calculate
from agentmem_core.core.consolidate import rebuild_from_episodes, run_consolidation
from agentmem_core.core.episodes import EpisodeLog
from agentmem_core.core.facts import UnifiedFactLedger
from agentmem_core.core.ingest import ingest_conversational_turn, ingest_procedural_rule
from agentmem_core.core.plan import QueryPlan, RecallConfig, build_query_plan
from agentmem_core.core.profiles import MemoryProfile, load_profile
from agentmem_core.core.recall import unified_recall
from agentmem_core.core.scopes import ScopeChain, resolve_write_scope
from agentmem_core.core.supersede import SlotSupersedeTracker
from agentmem_core.engine import CognitiveMemoryEngine
from agentmem_core.storage.base import RecallFilter
from agentmem_core.views.profile import ProfileView
from agentmem_core.views.vfs import VFSView


class ProfileBinding:
    """Lightweight handle binding a declarative MemoryProfile + ScopeChain to a shared UnifiedMemoryEngine.

    Enables a single agent to mount multiple profiles (e.g. ``conversational`` + ``coding``)
    simultaneously on one core without write-routing or retrieval-routing interference.
    """

    def __init__(
        self,
        engine: "UnifiedMemoryEngine",
        profile: str | Path | MemoryProfile,
        *,
        user_id: str = "default",
        agent_id: str = "default",
        session_id: str | None = None,
        repo_id: str | None = None,
        project_id: str | None = None,
        scope_chain: ScopeChain | list[str] | tuple[str, ...] | None = None,
    ):
        self.engine = engine
        self.profile = load_profile(profile)
        self.user_id = user_id
        self.agent_id = agent_id
        self.session_id = session_id
        self.repo_id = repo_id
        self.project_id = project_id
        self.scope_chain = ScopeChain.coerce(scope_chain) or ScopeChain.build(
            hierarchy=self.profile.scope_hierarchy,
            user_id=user_id,
            session_id=session_id,
            repo_id=repo_id,
            project_id=project_id,
        )

    def default_write_scope(self, explicit: str | None = None) -> str:
        return resolve_write_scope(
            explicit_scope_key=explicit,
            default_level=self.profile.default_write_scope_level,
            user_id=self.user_id,
            session_id=self.session_id,
            repo_id=self.repo_id,
            project_id=self.project_id,
        )

    def ingest_turn(self, text: str, *, scope_key: str | None = None, **kwargs: Any) -> dict:
        eff_scope = self.default_write_scope(scope_key)
        return self.engine.ingest_turn(
            text,
            user_id=self.user_id,
            agent_id=self.agent_id,
            session_id=kwargs.pop("session_id", None) or self.session_id or "default",
            scope_key=eff_scope,
            profile=self.profile,
            **kwargs,
        )

    def ingest_rule(self, path: str, *, scope_key: str | None = None, **kwargs: Any) -> dict:
        eff_scope = self.default_write_scope(scope_key)
        return self.engine.ingest_rule(
            path,
            user_id=self.user_id,
            agent_id=self.agent_id,
            session_id=kwargs.pop("session_id", None) or self.session_id or "default",
            scope_key=eff_scope,
            profile=self.profile,
            **kwargs,
        )

    def recall(self, query: str, **kwargs: Any) -> dict:
        return self.engine.recall(
            query,
            user_id=self.user_id,
            agent_id=self.agent_id,
            session_id=kwargs.pop("session_id", None),
            scope_chain=kwargs.pop("scope_chain", None) or self.scope_chain,
            profile=self.profile,
            **kwargs,
        )

    def build_context(self, query: str, **kwargs: Any) -> dict:
        return self.engine.build_context(
            query,
            user_id=self.user_id,
            agent_id=self.agent_id,
            session_id=kwargs.pop("session_id", None),
            scope_chain=kwargs.pop("scope_chain", None) or self.scope_chain,
            profile=self.profile,
            **kwargs,
        )

    def answer(self, query: str, **kwargs: Any) -> dict:
        return self.engine.answer(
            query,
            user_id=self.user_id,
            agent_id=self.agent_id,
            session_id=kwargs.pop("session_id", None),
            scope_chain=kwargs.pop("scope_chain", None) or self.scope_chain,
            profile=self.profile,
            **kwargs,
        )

    def read_note(self, path: str) -> dict | None:
        return self.engine.read_note(
            path,
            user_id=self.user_id,
            agent_id=self.agent_id,
            scope_chain=self.scope_chain,
        )

    def tree_outline(self, dir_path: str = "/") -> str:
        return self.engine.tree_outline(
            dir_path,
            user_id=self.user_id,
            agent_id=self.agent_id,
            scope_chain=self.scope_chain,
            include_stale=self.profile.stale_policy != "exclude",
        )


class UnifiedMemoryEngine:
    """Universal single-core memory engine parameterized by declarative MemoryProfile & ScopeChain.

    Architecture:
      - **L0 Episode Log** (``EpisodeLog``): ``.jsonl.gz`` cold chunks (0 L0 vectors) + SQLite/PG ``raw_turns_fts`` fallback.
      - **Unified Fact Layer** (``UnifiedFactLedger`` over SQLite+LanceDB or PostgreSQL):
        single ``facts`` ledger with bi-temporal fields, mandatory evidence, hierarchical ``scope_key``,
        code-anchor staleness state, ``kind ∈ {world, experience, preference, belief, rule}``,
        slot-level supersede, optimistic lock versions, and rollback snapshots.
      - **Declarative Profiles** (``MemoryProfile``): loaded from YAML/JSON with zero profile branching in core.
      - **Derived Views**: ``self.vfs`` (``VFSView``) and ``self.profile_view`` / ``self.profile`` (``ProfileView``).
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
        profile: str | Path | MemoryProfile | None = None,
        calculator=None,
        **engine_kwargs,
    ):
        self.default_profile = load_profile(profile)
        self.config = config or self.default_profile.recall
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
        self.profile_view = self.profile

    def close(self) -> None:
        try:
            self.episodes.close()
        finally:
            self.store.close()

    def bind_profile(
        self,
        profile: str | Path | MemoryProfile,
        *,
        user_id: str = "default",
        agent_id: str = "default",
        session_id: str | None = None,
        repo_id: str | None = None,
        project_id: str | None = None,
        scope_chain: ScopeChain | list[str] | tuple[str, ...] | None = None,
    ) -> ProfileBinding:
        """Bind a declarative profile and scope chain to this shared core instance."""
        return ProfileBinding(
            self,
            profile,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            repo_id=repo_id,
            project_id=project_id,
            scope_chain=scope_chain,
        )

    def _resolve_profile(self, profile: str | Path | MemoryProfile | None) -> MemoryProfile:
        return self.default_profile if profile is None else load_profile(profile)

    def _resolve_scope_chain(
        self,
        spec: MemoryProfile,
        *,
        scope_chain: ScopeChain | list[str] | tuple[str, ...] | str | None = None,
        user_id: str = "default",
        session_id: str | None = None,
        repo_id: str | None = None,
        project_id: str | None = None,
    ) -> ScopeChain | None:
        explicit = ScopeChain.coerce(scope_chain)
        if explicit is not None:
            return explicit
        if repo_id is not None or project_id is not None:
            return ScopeChain.build(
                hierarchy=spec.scope_hierarchy,
                user_id=user_id,
                session_id=session_id,
                repo_id=repo_id,
                project_id=project_id,
            )
        return None

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
        scope_key: str | None = None,
        repo_id: str | None = None,
        project_id: str | None = None,
        override_key: str = "",
        profile: str | Path | MemoryProfile | None = None,
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
        spec = self._resolve_profile(profile)
        eff_scope = resolve_write_scope(
            explicit_scope_key=scope_key,
            default_level=spec.default_write_scope_level,
            user_id=user_id,
            session_id=session_id,
            repo_id=repo_id,
            project_id=project_id,
        )
        return ingest_conversational_turn(
            self._engine,
            self.episodes,
            self.ledger,
            text,
            supersede=self.supersede if self.config.enable_slot_supersede else None,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            scope_key=eff_scope,
            override_key=override_key,
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
        scope_key: str | None = None,
        repo_id: str | None = None,
        project_id: str | None = None,
        override_key: str = "",
        profile: str | Path | MemoryProfile | None = None,
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
        spec = self._resolve_profile(profile)
        eff_scope = resolve_write_scope(
            explicit_scope_key=scope_key,
            default_level=spec.default_write_scope_level,
            user_id=user_id,
            session_id=session_id,
            repo_id=repo_id,
            project_id=project_id,
        )
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
            scope_key=eff_scope,
            override_key=override_key,
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

    def read_note(
        self,
        path: str,
        *,
        user_id: str = "default",
        agent_id: str = "default",
        scope_key: str | None = None,
        scope_chain: ScopeChain | list[str] | tuple[str, ...] | None = None,
    ) -> dict | None:
        return self.vfs.read_note(
            path,
            user_id=user_id,
            agent_id=agent_id,
            scope_key=scope_key,
            scope_chain=ScopeChain.coerce(scope_chain),
        )

    def tree_outline(
        self,
        dir_path: str = "/",
        *,
        user_id: str = "default",
        agent_id: str = "default",
        scope_chain: ScopeChain | list[str] | tuple[str, ...] | None = None,
        include_stale: bool = True,
    ) -> str:
        return self.vfs.tree_outline(
            dir_path,
            user_id=user_id,
            agent_id=agent_id,
            scope_chain=ScopeChain.coerce(scope_chain),
            include_stale=include_stale,
        )

    def compact_directory(self, dir_path: str, *, user_id: str = "default", agent_id: str = "default", timestamp: float | None = None) -> dict:
        return self.vfs.compact_directory(dir_path, user_id=user_id, agent_id=agent_id, timestamp=timestamp)

    def rollback(self, *, user_id: str = "default", agent_id: str = "default", path: str | None = None, snapshot_id: str | None = None, batch_id: str | None = None) -> dict:
        return self.vfs.rollback(user_id=user_id, agent_id=agent_id, path=path, snapshot_id=snapshot_id, batch_id=batch_id)

    # ---- Unified Recall, Assemble, Answer, Reflect ----

    def plan(
        self,
        query: str,
        *,
        reference_time: float | None = None,
        config: RecallConfig | None = None,
        profile: str | Path | MemoryProfile | None = None,
    ) -> QueryPlan:
        spec = self._resolve_profile(profile)
        cfg = spec.to_recall_config(config or self.config)
        return build_query_plan(query, reference_time=reference_time, config=cfg)

    def recall(
        self,
        query: str,
        *,
        user_id: str = "default",
        agent_id: str = "default",
        session_id: str | None = None,
        scope_chain: ScopeChain | list[str] | tuple[str, ...] | str | None = None,
        repo_id: str | None = None,
        project_id: str | None = None,
        profile: str | Path | MemoryProfile | None = None,
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
        spec = self._resolve_profile(profile)
        cfg = spec.to_recall_config(config or self.config)
        eff_chain = self._resolve_scope_chain(
            spec,
            scope_chain=scope_chain,
            user_id=user_id,
            session_id=session_id,
            repo_id=repo_id,
            project_id=project_id,
        )
        return unified_recall(
            self._engine,
            self.episodes,
            self.ledger,
            query,
            supersede=self.supersede,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            scope_chain=eff_chain,
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

    def build_context(
        self,
        query: str,
        *,
        user_id: str = "default",
        agent_id: str = "default",
        session_id: str | None = None,
        scope_chain: ScopeChain | list[str] | tuple[str, ...] | str | None = None,
        repo_id: str | None = None,
        project_id: str | None = None,
        reference_time: float | None = None,
        mode: str = "reconstructive",
        token_budget: int | None = None,
        top_k: int | None = None,
        packing_mode: str | None = None,
        include_cold: bool = False,
        filters: RecallFilter | None = None,
        disclosure: Literal["full", "progressive"] | None = None,
        profile: str | Path | MemoryProfile | None = None,
        drilldown_top_n: int = 2,
        drilldown_window: int = 0,
        config: RecallConfig | None = None,
        _trace: list[dict] | None = None,
    ) -> dict:
        spec = self._resolve_profile(profile)
        cfg = spec.to_recall_config(config or self.config)
        eff_budget = spec.default_token_budget if token_budget is None else int(token_budget)
        eff_disclosure = disclosure or (config.disclosure if config is not None else spec.disclosure)
        eff_chain = self._resolve_scope_chain(
            spec,
            scope_chain=scope_chain,
            user_id=user_id,
            session_id=session_id,
            repo_id=repo_id,
            project_id=project_id,
        )
        recall_result = self.recall(
            query,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            scope_chain=eff_chain,
            profile=spec,
            reference_time=reference_time,
            mode=mode,
            token_budget=eff_budget,
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
            scope_chain=eff_chain,
            token_budget=eff_budget,
            drilldown_top_n=drilldown_top_n,
            drilldown_window=drilldown_window,
            disclosure=eff_disclosure,
            profile=spec,
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
        scope_chain: ScopeChain | list[str] | tuple[str, ...] | str | None = None,
        repo_id: str | None = None,
        project_id: str | None = None,
        reference_time: float | None = None,
        token_budget: int = 2048,
        mode: str = "reconstructive",
        disclosure: Literal["full", "progressive"] = "full",
        profile: str | Path | MemoryProfile | None = None,
    ) -> dict:
        bundle = self.build_context(
            query,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            scope_chain=scope_chain,
            repo_id=repo_id,
            project_id=project_id,
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
        scope_chain: ScopeChain | list[str] | tuple[str, ...] | str | None = None,
        repo_id: str | None = None,
        project_id: str | None = None,
        profile: str | Path | MemoryProfile | None = None,
        reference_time: float | None = None,
        token_budget: int = 2048,
    ) -> dict:
        bundle = self.build_context(
            query,
            user_id=user_id,
            agent_id=agent_id,
            session_id=None,
            scope_chain=scope_chain,
            repo_id=repo_id,
            project_id=project_id,
            profile=profile,
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
        from agentmem_core.adapters.registry import UnifiedToolRegistry

        return UnifiedToolRegistry(self, user_id=user_id, agent_id=agent_id, session_id=session_id)

    def dispatch_tool(self, name: str, arguments: dict | str, *, user_id: str = "default", agent_id: str = "default", session_id: str | None = None):
        return self.as_tools(user_id=user_id, agent_id=agent_id, session_id=session_id).dispatch(name, arguments)
