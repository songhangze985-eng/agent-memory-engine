"""Unified four-channel recall + L0 FTS5 rescue + M2 slot supersede, as_of & time-window filtering."""
from __future__ import annotations

import math
import re

from agentmem_core.core.text import stem_terms
from agentmem_core.core.episodes import EpisodeLog
from agentmem_core.core.facts import UnifiedFactLedger
from agentmem_core.core.graph_enum import coverage_first_session_pack, enumerate_entity_graph_candidates
from agentmem_core.core.plan import QueryPlan, RecallConfig, build_query_plan
from agentmem_core.core.scopes import ScopeChain, apply_scope_precedence
from agentmem_core.core.supersede import SlotSupersedeTracker, resolve_relative_occurrence
from agentmem_core.engine import _compact_evidence, _fit_context
from agentmem_core.storage.base import RecallFilter
from agentmem_core.textutil import strip_step_tags


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


def _fact_effective_ts(fact: dict) -> float:
    occ = fact.get("occurrence_start")
    if occ is not None:
        return float(occ)
    mention = fact.get("mention_time")
    rel = resolve_relative_occurrence(fact.get("plain") or fact.get("text") or "", mention)
    if rel is not None:
        return float(rel)
    if mention is not None:
        return float(mention)
    return float(fact.get("event_time_start") or fact.get("ingest_time") or 0.0)


def _fact_mention_ts(fact: dict) -> float:
    for key in ("mention_time", "event_time_start", "ingest_time"):
        val = fact.get(key)
        if val is not None:
            return float(val)
    return 0.0


_UPDATE_TURN_CUE = re.compile(
    r"\b(?:now|currently|switched|upgraded|changed|updated|moved\s+to|so\s+far|just\s+realized|"
    r"just\s+finished|actually\s+watched|limit|increased|decreased|instead|replaced|lost\s+\d+)\b|"
    r"现在|目前|改成|换成|升级|最新",
    re.I,
)
_MULTI_AGG_GUARD = re.compile(r"\b(?:all|both|every|list|altogether|combined|in\s+total)\b|全部|所有|列出|一共", re.I)


_SENT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")
_NUM_TOK_RE = re.compile(
    r"\b(\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\b",
    re.I,
)
_COUNT_OR_STATE_Q = re.compile(
    r"\b(?:how\s+many|how\s+much|how\s+long|currently|now|latest|most\s+recent|new\s+|did\s+i\s+finish|what\s+is\s+my)\b|目前|现在|最新|多少",
    re.I,
)


def _best_query_sentence(text: str, q_content: set[str]) -> tuple[str, set[str], set[str]]:
    """Extract the highest query-matching sentence window (plus following declarative sentence if rhetorical ?)."""
    cleaned = strip_step_tags(text or "")
    sents = [s.strip() for s in _SENT_RE.split(cleaned) if s.strip()] or [cleaned]
    best_idx = 0
    best_hits = -1
    for idx, sent in enumerate(sents):
        hits = len(q_content & stem_terms(sent))
        if hits > best_hits:
            best_hits = hits
            best_idx = idx
    snippet = sents[best_idx]
    if snippet.rstrip().endswith(("?", "？")) and best_idx + 1 < len(sents):
        snippet = snippet + " " + sents[best_idx + 1]
    stems = stem_terms(snippet)
    nums = {m.group(1).casefold() for m in _NUM_TOK_RE.finditer(snippet)}
    return snippet, stems, nums


def _resolve_cross_session_updates(ranked: list[dict], query: str, qplan: QueryPlan) -> list[dict]:
    """Promote chronologically newer state updates ONLY when strong query-focused update evidence exists.

    Strictly disabled on historical, temporal, and multi-item aggregation queries so
    LoCoMo temporal/multi-hop and LongMemEval multi-session queries are never degraded.
    """
    if qplan.is_historical or qplan.is_temporal or qplan.change_direction_ask or _MULTI_AGG_GUARD.search(query or ""):
        return ranked
    if len(ranked) < 2:
        return ranked

    q_stems = stem_terms(query)
    if not q_stems:
        return ranked

    top_slice = [dict(f) for f in ranked[:12]]
    tail = ranked[12:]
    known_speakers = {str(f.get("speaker") or "").casefold() for f in top_slice if f.get("speaker")}
    q_content = {s for s in q_stems if s not in known_speakers} or q_stems
    focused = [_best_query_sentence(f.get("plain") or f.get("text") or "", q_content) for f in top_slice]
    state_like_q = bool(_COUNT_OR_STATE_Q.search(query or ""))

    older_indices: set[int] = set()
    latest_indices: set[int] = set()
    for i in range(len(top_slice)):
        fi = top_slice[i]
        if _is_pinned(fi) or fi.get("role", "user") != "user" or not fi.get("session_id"):
            continue
        snip_i, stems_i, nums_i = focused[i]
        if len(q_content & stems_i) < 2:
            continue
        for j in range(i + 1, len(top_slice)):
            fj = top_slice[j]
            if _is_pinned(fj) or fj.get("role", "user") != "user" or not fj.get("session_id"):
                continue
            if fj["session_id"] == fi["session_id"]:
                continue
            snip_j, stems_j, nums_j = focused[j]
            if len(q_content & stems_j) < 2:
                continue
            non_num_i = {t for t in stems_i if not t.isdigit() and t not in known_speakers}
            non_num_j = {t for t in stems_j if not t.isdigit() and t not in known_speakers}
            shared_q = (q_content & non_num_i) & non_num_j
            if len(shared_q) < 2:
                continue
            sent_jac = len(non_num_i & non_num_j) / max(1, len(non_num_i | non_num_j))
            same_attr = bool(fi.get("attribute") and fi.get("attribute") == fj.get("attribute"))
            ti, tj = _fact_mention_ts(fi), _fact_mention_ts(fj)
            if abs(ti - tj) < 3600.0:
                continue
            newer_idx, older_idx = (j, i) if tj > ti else (i, j)
            newer_snip = focused[newer_idx][0]
            diff_nums = bool(nums_i and nums_j and nums_i != nums_j)
            has_cue = bool(_UPDATE_TURN_CUE.search(newer_snip))
            # Fire when BOTH query-focused sentences express competing values for the same user predicate:
            # 1) Same attribute, OR
            # 2) State/count question + differing numeric values + shared >=2 query content stems, OR
            # 3) High sentence-level template overlap (>=0.28) + update cue or differing numbers
            if not (
                same_attr
                or (state_like_q and diff_nums and sent_jac >= 0.16)
                or (sent_jac >= 0.28 and (has_cue or diff_nums))
            ):
                continue
            older_indices.add(older_idx)
            latest_indices.add(newer_idx)

    if not older_indices:
        return ranked

    strict_latest = latest_indices - older_indices
    for idx in strict_latest:
        top_slice[idx]["temporal_role"] = "latest"
        top_slice[idx]["is_update_latest"] = True
    for idx in older_indices:
        top_slice[idx]["temporal_role"] = "older"
        top_slice[idx]["state_tag"] = "[历史已变更]"

    # Swap competing latest directly into the older fact's higher rank slot
    reordered = list(top_slice)
    for old_idx in sorted(older_indices):
        for new_idx in sorted(strict_latest):
            pos_old = next((k for k, f in enumerate(reordered) if f["id"] == top_slice[old_idx]["id"]), None)
            pos_new = next((k for k, f in enumerate(reordered) if f["id"] == top_slice[new_idx]["id"]), None)
            if pos_old is not None and pos_new is not None and pos_old < pos_new:
                item_new = reordered.pop(pos_new)
                reordered.insert(pos_old, item_new)
    return [*reordered, *tail]


def unified_recall(
    engine,
    episodes: EpisodeLog,
    ledger: UnifiedFactLedger,
    query: str,
    *,
    supersede: SlotSupersedeTracker | None = None,
    user_id: str = "default",
    agent_id: str = "default",
    session_id: str | None = None,
    scope_chain: ScopeChain | None = None,
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
    """Execute unified 4-channel recall + L0 FTS5 rescue + M2 slot supersede & temporal filtering."""
    cfg = config or RecallConfig()
    qplan = plan or build_query_plan(query, reference_time=reference_time, config=cfg)
    scope = (user_id, agent_id, session_id)
    effective_top_k = top_k if top_k is not None else (35 if packing_mode == "token_budget" or mode == "reconstructive" else 10)
    effective_packing = packing_mode or ("token_budget" if top_k is None else "top_k")

    # 1. Core 4-channel recall (vector + fact BM25 + entity graph + exact/version)
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
        enable_bipolar=cfg.enable_bipolar,
        **trace_arg,
    )

    # Preserve strict precision gate for exact identifiers & versions (zero confusion / 100% empty on missing)
    empty_reason = (result.get("gate_stats") or {}).get("empty_reason")
    strict_exact_empty = empty_reason in {"no_exact_version", "no_exact_identifier"}
    if strict_exact_empty:
        return {**result, "query_plan": qplan, "episode_fallback_turns": []}

    ranked = list(result.get("ranked") or ())

    # 2. Zero-vector L0 Turn-Level FTS5 Rescue Channel
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
                include_superseded=not qplan.exclude_superseded,
                include_cold=include_cold or qplan.is_historical,
            )

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

            ep_set = set(ep_fact_order)
            for fact in ranked:
                if fact["id"] in ep_set:
                    fact["episode_fts_hit"] = True

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
                base = [f for f in ranked if f["id"] not in rescue_ids]
                insert_at = min(len(base), 7)
                while insert_at < min(len(base), 9) and (
                    _is_pinned(base[insert_at]) or base[insert_at].get("activation") or base[insert_at].get("speaker_bridge")
                ):
                    insert_at += 1
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

    # 3. M2 Query-Time `explicit_as_of` Filter (point-in-time historical view)
    if qplan.explicit_as_of is not None:
        cutoff = float(qplan.explicit_as_of)
        ranked = [
            f for f in ranked
            if _fact_mention_ts(f) <= cutoff and (f.get("valid_until") is None or float(f["valid_until"]) > cutoff)
        ]

    # 4. M2 Query-Time Time-Window Constraint Boost
    win_start, win_end = qplan.time_window
    if (win_start is not None or win_end is not None) and not qplan.identifiers and not qplan.versions and len(ranked) > 1:
        def _in_window(fact: dict) -> bool:
            ts = _fact_effective_ts(fact)
            if win_start is not None and ts < win_start - 86400.0:
                return False
            if win_end is not None and ts > win_end + 86400.0:
                return False
            return True

        in_win = [f for f in ranked if _is_pinned(f) or _in_window(f)]
        out_win = [f for f in ranked if not _is_pinned(f) and not _in_window(f)]
        if in_win:
            ranked = [*in_win, *out_win]

    # 5. M2 Slot-Level Supersede State Filter & Cross-Session Update Dominance
    superseded_ids: set[str] = set()
    superseding_ids: set[str] = set()
    if cfg.enable_slot_supersede and supersede is not None:
        superseded_ids = supersede.superseded_fact_ids(user_id=user_id, agent_id=agent_id)
        superseding_ids = supersede.superseding_fact_ids(user_id=user_id, agent_id=agent_id)

    if qplan.exclude_superseded:
        ranked = [
            f for f in ranked
            if f.get("status") != "superseded" and f["id"] not in superseded_ids
        ]
        # Lift active superseding latest facts cleanly ahead of older same-slot/same-topic peers
        if superseding_ids:
            for idx, fact in enumerate(ranked):
                if fact["id"] in superseding_ids:
                    fact["is_superseding_latest"] = True
                    fact["temporal_role"] = "latest"
            # Move any superseding latest fact in top-12 up to slot 0/1 if it matches >= 2 query stems
            q_stems_up = stem_terms(query)
            if q_stems_up and not qplan.is_temporal:
                promoted = [
                    f for f in ranked[:12]
                    if f.get("is_superseding_latest") and len(q_stems_up & stem_terms(f.get("plain") or f.get("text") or "")) >= 2
                ]
                if promoted:
                    p_ids = {f["id"] for f in promoted}
                    pinned_head = [f for f in ranked if _is_pinned(f) and f["id"] not in p_ids]
                    pin_ids = {f["id"] for f in pinned_head}
                    rest_tail = [f for f in ranked if f["id"] not in p_ids and f["id"] not in pin_ids]
                    ranked = [*pinned_head, *promoted, *rest_tail]
        if cfg.enable_slot_supersede:
            ranked = _resolve_cross_session_updates(ranked, query, qplan)

    # 5b. M3 Entity-Graph Enumeration & Coverage-First Distinct-Session Packing (aggregate / multi-hop)
    if (
        cfg.enable_graph_enum
        and (qplan.is_aggregate_query or qplan.is_multi_hop_query)
        and not qplan.identifiers
        and not qplan.versions
        and not any(f.get("is_superseding_latest") or f.get("is_update_latest") for f in ranked[:5])
    ):
        existing_ids = {f["id"] for f in ranked}
        enumerated = enumerate_entity_graph_candidates(
            engine,
            episodes,
            query,
            scope=scope,
            existing_ids=existing_ids,
            exclude_superseded=qplan.exclude_superseded,
            include_cold=include_cold or qplan.is_historical,
            packing_mode=effective_packing,
        )
        if enumerated:
            ranked = [*ranked, *enumerated]
        ranked = coverage_first_session_pack(
            ranked,
            query,
            is_aggregate=qplan.is_aggregate_query,
            is_multi_hop=qplan.is_multi_hop_query,
            pinned_fn=_is_pinned,
        )
        if effective_packing == "top_k":
            ranked = ranked[:effective_top_k]
        else:
            ranked = _fit_context(ranked, max(effective_top_k, 50), token_budget)

    # 6. Enrich with unified fact metadata (scope_key, override_key, staleness)
    meta_by_id = ledger.meta_map([f["id"] for f in ranked])
    for fact in ranked:
        meta = meta_by_id.get(fact["id"])
        if meta:
            fact["kind"] = meta["kind"]
            fact["scope_key"] = meta.get("scope_key", "global")
            fact["override_key"] = meta.get("override_key", "")
            fact["vfs_path"] = meta["vfs_path"]
            fact["index_summary"] = meta["index_summary"]
            fact["verification_status"] = meta["verification_status"]
            fact["salience_tag"] = meta["salience_tag"]
            fact["always_active"] = meta["always_active"]
            fact["staleness_status"] = meta.get("staleness_status", "fresh")
            fact["stale_reason"] = meta.get("stale_reason", "")
        else:
            fact.setdefault("kind", "rule" if fact.get("memory_type") == "procedural" else "world")
            fact.setdefault("scope_key", f"user:{user_id}" if user_id else "global")
            fact.setdefault("override_key", "")
            fact.setdefault("verification_status", "verified")
            fact.setdefault("always_active", False)
            fact.setdefault("staleness_status", "fresh")
            fact.setdefault("stale_reason", "")

    # 6b. Hierarchical Scope Chain Filtering & Precedence Resolution (project/repo overrides user/global)
    eff_scope_chain = ScopeChain.coerce(scope_chain)
    overridden_by_scope: list[dict] = []
    if eff_scope_chain is not None:
        ranked, overridden_by_scope = apply_scope_precedence(ranked, eff_scope_chain, suppress_overridden=True)

    # 7. M2 Demoted Recency Tie-Breaker: ONLY breaks near-ties (<=5% gap) on non-historical, non-temporal queries!
    if qplan.allow_recency_tiebreak and len(ranked) >= 2:
        for i in range(min(len(ranked) - 1, 9)):
            a, b = ranked[i], ranked[i + 1]
            if _is_pinned(a) or _is_pinned(b):
                continue
            sa = float(a.get("final_score") or 0.0)
            sb = float(b.get("final_score") or 0.0)
            if sa > 0 and sb > 0 and abs(sa - sb) / max(sa, sb) <= 0.05:
                wa = (1.0 if a.get("verification_status") == "verified" else 0.85) * _recency_weight(_fact_mention_ts(a), qplan.as_of)
                wb = (1.0 if b.get("verification_status") == "verified" else 0.85) * _recency_weight(_fact_mention_ts(b), qplan.as_of)
                if wb > wa + 0.05:
                    ranked[i], ranked[i + 1] = b, a

    return {
        **result,
        "ranked": ranked,
        "empty_recall": not ranked and not episode_fallback_turns,
        "query_plan": qplan,
        "scope_chain": eff_scope_chain.as_list() if eff_scope_chain is not None else None,
        "scope_overridden": [
            {"id": item["id"], "scope_key": item["scope_key"], "overridden_by_scope": item.get("overridden_by_scope")}
            for item in overridden_by_scope
        ],
        "episode_fallback_turns": episode_fallback_turns[:6],
    }
