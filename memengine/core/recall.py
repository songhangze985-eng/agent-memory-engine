"""Unified four-channel recall (vector + bm25[facts + L0 FTS5] + entity_graph + exact) with feature-flagged rerank."""
from __future__ import annotations

import math
import time
from contextlib import nullcontext

from memengine.core.answer import stem_terms
from memengine.core.episodes import EpisodeLog
from memengine.core.facts import UnifiedFactLedger
from memengine.core.plan import QueryPlan, RecallConfig, build_query_plan
from memengine.engine import _compact_evidence, _fit_context, baseline_rrf
from memengine.identifiers import eligible_identifiers
from memengine.storage.base import RecallFilter
from memengine.textutil import strip_step_tags


def _recency_weight(ts: float | None, reference_time: float | None, half_life_days: float = 90.0) -> float:
    if ts is None or reference_time is None:
        return 0.5
    age_days = max(0.0, (float(reference_time) - float(ts)) / 86400.0)
    return math.exp(-math.log(2.0) * age_days / max(1.0, half_life_days))


def _is_pinned(fact: dict) -> bool:
    return bool(
        fact.get("exact_ident_hit")
        or fact.get("version_confidence", 0) > 0
        or fact.get("temporary_priority")
        or fact.get("attention_weight", 1.0) > 1.0
    )


def _run_without_bipolar(
    engine,
    query: str,
    plan: QueryPlan,
    *,
    scope: tuple[str, str, str | None],
    top_k: int,
    token_budget: int,
    include_cold: bool,
    filters: RecallFilter | None,
    _trace: list[dict] | None,
) -> dict:
    """Feature-flagged recall path with bipolar ring & belt pruning disabled (for M5 ablation)."""
    t0 = time.perf_counter()
    direct = tuple(engine.store.entity_keys(scope=scope, query=query))
    parsed = dict(plan.parsed)
    if plan.is_historical:
        parsed["query_type"] = "TIME_TRAVEL"
    candidates = engine._recall(
        query,
        parsed,
        plan.temporal,
        scope,
        direct,
        include_cold=include_cold or plan.is_historical,
        filters=filters,
    )
    ranked = baseline_rrf(candidates)
    ce_scores, backend = engine._rerank(query, ranked)
    for fact, ce in zip(ranked, ce_scores):
        fact["ce_score"] = ce
        fact["final_score"] = 0.55 * min(1.0, fact.get("rrf_norm", 0.0)) + 0.45 * ce + 0.35 * fact.get("exact_hit_ratio", 0.0)
        if fact.get("version_mismatch"):
            fact["final_score"] *= 0.30
    ranked.sort(key=lambda f: (-f.get("version_confidence", 0), -f.get("exact_hit_ratio", 0), -f["final_score"], f["id"]))
    ranked, ident_mode = eligible_identifiers(ranked, query, plan.is_historical, engine.allow_empty_recall)
    if plan.exclude_superseded:
        ranked = [f for f in ranked if f.get("status") == "active"]
    selected = _fit_context(ranked, top_k, token_budget)
    gate_stats = {"tau_abs": 0.12, "bipolar_enabled": False, "reranker": type(backend).__name__}
    if ident_mode == "version" and not ranked:
        gate_stats["empty_reason"] = "no_exact_version"
    elif ident_mode == "symbol" and not ranked:
        gate_stats["empty_reason"] = "no_exact_identifier"
    if _trace is not None:
        _trace.append({
            "fused": [f["id"] for f in candidates],
            "reranked": [f["id"] for f in ranked],
            "packed": [f["id"] for f in selected],
        })
    return {
        "route": "TASK",
        "parsed_query": parsed,
        "temporal_intent": plan.temporal.__dict__,
        "matched_entities": list(direct),
        "ranked": selected,
        "empty_recall": not selected,
        "gate_stats": gate_stats,
        "scope": {"user_id": scope[0], "agent_id": scope[1], "session_id": scope[2]},
        "validity_as_of": plan.as_of,
        "packing_mode": "token_budget",
        "retrieval_ms": (time.perf_counter() - t0) * 1000.0,
    }


def unified_recall(
    engine,
    episodes: EpisodeLog,
    ledger: UnifiedFactLedger,
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
    plan: QueryPlan | None = None,
    _trace: list[dict] | None = None,
) -> dict:
    """Execute unified 4-channel recall + L0 FTS5 fallback + feature-flagged reranking."""
    cfg = config or RecallConfig()
    qplan = plan or build_query_plan(query, reference_time=reference_time, config=cfg)
    scope = (user_id, agent_id, session_id)
    effective_top_k = top_k if top_k is not None else (35 if packing_mode == "token_budget" or mode == "reconstructive" else 10)
    effective_packing = packing_mode or ("token_budget" if top_k is None else "top_k")

    # 1. Core 4-channel recall (vector + fact BM25 + entity graph + exact/version)
    if cfg.enable_bipolar:
        trace_arg = {"_trace": _trace} if _trace is not None else {}
        result = engine.retrieve(
            query,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            token_budget=token_budget,
            read_only=True,
            mutate_stats=False,
            mode=mode if cfg.enable_entity_graph else "single_query",
            filters=filters,
            include_cold=include_cold or qplan.is_historical,
            top_k=top_k,
            timestamp=reference_time,
            reference_time=reference_time,
            packing_mode=effective_packing,
            **trace_arg,
        )
    else:
        result = _run_without_bipolar(
            engine,
            query,
            qplan,
            scope=scope,
            top_k=effective_top_k,
            token_budget=token_budget,
            include_cold=include_cold,
            filters=filters,
            _trace=_trace,
        )

    # Preserve strict precision gate for exact identifiers & versions (zero confusion / 100% empty on missing)
    empty_reason = (result.get("gate_stats") or {}).get("empty_reason")
    strict_exact_empty = empty_reason in {"no_exact_version", "no_exact_identifier"}
    if strict_exact_empty:
        return {**result, "query_plan": qplan, "episode_fallback_turns": []}

    ranked = list(result.get("ranked") or ())

    # 2. Zero-vector L0 Turn-Level FTS5 Fallback Channel (recovers extraction misses & diluted long turns)
    episode_fallback_turns: list[dict] = []
    if cfg.enable_episode_fts and not qplan.identifiers and not qplan.versions:
        ep_hits = episodes.search_fts(
            query,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            top_n=30,
        )
        if ep_hits:
            turn_ids = [int(h["turn_id"]) for h in ep_hits]
            if _trace is not None:
                _trace.append({"episode_fts_turn_ids": turn_ids})
            turn_to_facts = engine.store.facts_for_turn_ids(
                turn_ids,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                include_superseded=qplan.is_historical,
                include_cold=include_cold or qplan.is_historical,
            )

            # Build L0-FTS5 fact rank order + capture un-extracted fallback turns
            ep_fact_order: list[str] = []
            seen_ep_fids: set[str] = set()
            for hit in ep_hits:
                tid = int(hit["turn_id"])
                fids = turn_to_facts.get(tid) or []
                if not fids:
                    episode_fallback_turns.append({
                        "kind": "episode_turn",
                        "turn_id": tid,
                        "session_id": hit["session_id"],
                        "role": hit["role"],
                        "speaker": hit.get("speaker"),
                        "text": strip_step_tags(hit["text"]).strip(),
                        "event_ts": hit["event_ts"],
                        "mention_time": hit.get("mention_time"),
                        "bm25_score": hit["bm25_score"],
                    })
                else:
                    for fid in fids:
                        if fid not in seen_ep_fids:
                            seen_ep_fids.add(fid)
                            ep_fact_order.append(fid)

            # Non-destructive, rank-preserving L0 FTS5 rescue:
            # Keep primary Top-10 & graph-activated ordering intact; only promote unambiguous
            # multi-term lexical direct hits from L0 FTS5 that primary missed from Top-10.
            q_stems = stem_terms(query)
            known_speakers = {str(h.get("speaker") or "").casefold() for h in ep_hits if h.get("speaker")}
            q_content_stems = {s for s in q_stems if s not in known_speakers} or q_stems
            strong_threshold = max(2, min(3, len(q_content_stems)))

            top10_ids = {f["id"] for f in ranked[:10]}
            by_id = {f["id"]: dict(f) for f in ranked}
            candidate_ids = [fid for fid in ep_fact_order[:12] if fid not in top10_ids]
            missing_ids = [fid for fid in candidate_ids if fid not in by_id]
            if missing_ids:
                for hydrated in engine.store.facts_by_ids(missing_ids, scope=scope):
                    if qplan.exclude_superseded and hydrated.get("status") != "active":
                        continue
                    hydrated = dict(hydrated)
                    if effective_packing == "token_budget":
                        hydrated = _compact_evidence(query, hydrated)
                    hydrated.setdefault("ce_score", 0.30)
                    hydrated.setdefault("final_score", 0.30)
                    hydrated.setdefault("rrf_norm", 0.30)
                    hydrated.setdefault("exact_hit_ratio", 0.0)
                    hydrated.setdefault("exact_ident_hit", False)
                    hydrated.setdefault("version_mismatch", False)
                    by_id[hydrated["id"]] = hydrated

            # Mark all facts corroborated by L0 FTS5 without scrambling primary order
            ep_set = set(ep_fact_order)
            for fact in ranked:
                if fact["id"] in ep_set:
                    fact["episode_fts_hit"] = True

            # Score L0 FTS5 rescues not in Top-10 by non-speaker stem coverage
            rescues: list[tuple[float, int, dict]] = []
            for ep_idx, fid in enumerate(candidate_ids):
                fact = by_id.get(fid)
                if fact is None:
                    continue
                if qplan.exclude_superseded and fact.get("status") != "active":
                    continue
                f_stems = stem_terms(fact.get("plain") or fact.get("text") or "")
                overlap_n = len(q_content_stems & f_stems)
                overlap_ratio = overlap_n / max(1, len(q_content_stems))
                if overlap_n >= strong_threshold and overlap_ratio >= 0.50:
                    fact["episode_fts_hit"] = True
                    rescues.append((overlap_ratio, -ep_idx, fact))

            if rescues:
                rescues.sort(key=lambda item: (-item[0], -item[1]))
                top_rescues = [fact for _, _, fact in rescues[:2]]
                rescue_ids = {f["id"] for f in top_rescues}
                # Remove promoted items from deeper tail before inserting cleanly after top-5 / protected slots
                base = [f for f in ranked if f["id"] not in rescue_ids]
                insert_at = min(len(base), 7)
                # Keep top-3 and any graph-activated/pinned slots ahead of rescues
                while insert_at < min(len(base), 9) and (
                    _is_pinned(base[insert_at]) or base[insert_at].get("activation") or base[insert_at].get("speaker_bridge")
                ):
                    insert_at += 1
                # Only insert if rescue strictly improves lexical coverage over displaced tail slot (or len < 10)
                if len(base) < 10 or (
                    insert_at < len(base)
                    and len(q_content_stems & stem_terms(top_rescues[0].get("plain") or top_rescues[0].get("text") or ""))
                    > len(q_content_stems & stem_terms(base[insert_at].get("plain") or base[insert_at].get("text") or ""))
                ):
                    ranked = [*base[:insert_at], *top_rescues, *base[insert_at:]]
                    if effective_packing == "top_k":
                        ranked = ranked[:effective_top_k]
                    else:
                        ranked = _fit_context(ranked, max(effective_top_k, 50), token_budget)

    # 3. Enrich with unified fact metadata (kind, vfs_path, verification_status, always_active)
    meta_by_id = ledger.meta_map([f["id"] for f in ranked])
    for fact in ranked:
        meta = meta_by_id.get(fact["id"])
        if meta:
            fact["kind"] = meta["kind"]
            fact["vfs_path"] = meta["vfs_path"]
            fact["index_summary"] = meta["index_summary"]
            fact["verification_status"] = meta["verification_status"]
            fact["salience_tag"] = meta["salience_tag"]
            fact["always_active"] = meta["always_active"]
        else:
            fact.setdefault("kind", "rule" if fact.get("memory_type") == "procedural" else "world")
            fact.setdefault("verification_status", "verified")
            fact.setdefault("always_active", False)

    # 4. Feature-flagged W_verified * W_recency tie-breaker (NEVER on historical or temporal queries!)
    if qplan.allow_recency_tiebreak and ranked:
        pinned = [f for f in ranked if _is_pinned(f)]
        pinned_ids = {f["id"] for f in pinned}
        rest = [f for f in ranked if f["id"] not in pinned_ids]
        for idx, fact in enumerate(rest):
            base = fact.get("unified_rrf_score", fact.get("final_score", 1.0 / (60.0 + idx)))
            w_ver = 1.0 if fact.get("verification_status") == "verified" else 0.85
            w_rec = _recency_weight(fact.get("event_time_start") or fact.get("ingest_time"), qplan.as_of)
            # Strict tie-breaker scale (<= 4% relative adjustment) so relevance always dominates
            fact["tiebreak_score"] = base * w_ver + 0.0006 * w_rec
        rest.sort(key=lambda f: (-f["tiebreak_score"], f["id"]))
        ranked = [*pinned, *rest]

    # 5. State filter: exclude superseded facts on non-historical current-state queries
    if qplan.exclude_superseded:
        ranked = [f for f in ranked if f.get("status") != "superseded" or f.get("temporal_role") == "older"]

    return {
        **result,
        "ranked": ranked,
        "empty_recall": not ranked and not episode_fallback_turns,
        "query_plan": qplan,
        "episode_fallback_turns": episode_fallback_turns[:6],
    }
