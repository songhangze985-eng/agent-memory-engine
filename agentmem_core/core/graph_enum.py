"""Entity-graph enumeration and session-diverse coverage-first packing (Milestone M3)."""
from __future__ import annotations

import re

from agentmem_core.core.text import expand_category_stems, light_stem, stem_terms
from agentmem_core.engine import _compact_evidence
from agentmem_core.textutil import strip_step_tags

_MULTI_ENTITY_SPLIT = re.compile(
    r"\b(?:compared\s+to|versus|vs\.?|and\s+(?:my\s+|the\s+)?|or\s+(?:my\s+|the\s+)?)\b",
    re.I,
)
_PROPER_PHRASE = re.compile(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,2})\b")
_QUESTION_STARTERS = frozenset({
    "what", "which", "who", "when", "where", "why", "how", "did", "do", "does",
    "have", "has", "had", "is", "are", "was", "were", "can", "could", "before", "after",
})


def extract_query_facets(query: str) -> list[str]:
    """Extract sub-entity / comparison facets from multi-session or multi-hop questions.

    For example, 'views on YouTube and TikTok' yields ['YouTube', 'TikTok'];
    'accommodations in Hawaii compared to Tokyo' yields ['Hawaii', 'Tokyo'].
    """
    facets: list[str] = []
    seen: set[str] = set()
    for m in _PROPER_PHRASE.finditer(query or ""):
        phrase = m.group(1).strip()
        first = phrase.split()[0].casefold()
        if first in _QUESTION_STARTERS or len(phrase) < 2:
            continue
        key = phrase.casefold()
        if key not in seen:
            seen.add(key)
            facets.append(phrase)
    # Also split explicit "A compared to B" / "A and B" clauses when proper nouns weren't enough
    if len(facets) < 2 and _MULTI_ENTITY_SPLIT.search(query or ""):
        parts = [p.strip(" ?.,") for p in _MULTI_ENTITY_SPLIT.split(query or "") if p.strip()]
        if len(parts) >= 2:
            for part in parts[-2:]:
                toks = [t for t in re.findall(r"[A-Za-z0-9]+", part) if len(t) >= 3 and t.casefold() not in _QUESTION_STARTERS]
                if toks:
                    label = " ".join(toks[:3])
                    if label.casefold() not in seen:
                        seen.add(label.casefold())
                        facets.append(label)
    return facets[:4]


def enumerate_entity_graph_candidates(
    engine,
    episodes,
    query: str,
    *,
    scope: tuple[str, str, str | None],
    existing_ids: set[str],
    exclude_superseded: bool = True,
    include_cold: bool = False,
    packing_mode: str = "token_budget",
    max_added: int = 16,
) -> list[dict]:
    """Enumerate active facts across entity graph & per-facet FTS for aggregate / multi-hop queries."""
    user_id, agent_id, session_id = scope
    facets = extract_query_facets(query)
    q_stems = expand_category_stems(stem_terms(query))
    candidate_fids: list[str] = []
    seen_fids: set[str] = set(existing_ids)

    # 1. Entity-inverted enumeration for every named entity found in the query
    if facets:
        entity_keys = [f"entity:{f.casefold()}" for f in facets] + [f"实体:{f}" for f in facets] + [f"人物:{f}" for f in facets]
        inv_hits = engine.store.search_entity_inverted(
            entity_keys,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            include_superseded=not exclude_superseded,
            include_cold=include_cold,
            top_n=30,
            ids_only=True,
        )
        for h in inv_hits:
            fid = h["id"]
            if fid not in seen_fids:
                seen_fids.add(fid)
                candidate_fids.append(fid)

        # 2. Per-facet lexical & L0 FTS5 enumeration (including category hyponym expansion for multi-hop!)
        q_pred_words = [
            w for w in re.findall(r"[A-Za-z0-9]+", query)
            if len(w) >= 4 and w.casefold() not in _QUESTION_STARTERS and all(w.casefold() not in f.casefold() for f in facets)
        ]
        raw_pred_stems = stem_terms(" ".join(q_pred_words))
        expanded_only = sorted(expand_category_stems(raw_pred_stems) - raw_pred_stems)[:6]
        pred_tail = " ".join([*q_pred_words[:4], *expanded_only])
        for facet in facets:
            sub_q = f"{facet} {pred_tail}".strip()
            bm_hits = engine.store.search_lexical(
                sub_q,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                include_superseded=not exclude_superseded,
                include_cold=include_cold,
                top_n=10,
                ids_only=True,
            )
            for h in bm_hits:
                fid = h["id"]
                if fid not in seen_fids:
                    seen_fids.add(fid)
                    candidate_fids.append(fid)
            ep_hits = episodes.search_fts(sub_q, user_id=user_id, agent_id=agent_id, session_id=session_id, top_n=8)
            if ep_hits:
                t2f = engine.store.facts_for_turn_ids(
                    [int(e["turn_id"]) for e in ep_hits],
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    include_superseded=not exclude_superseded,
                    include_cold=include_cold,
                )
                for e in ep_hits:
                    for fid in t2f.get(int(e["turn_id"]), ()):
                        if fid not in seen_fids:
                            seen_fids.add(fid)
                            candidate_fids.append(fid)

    if not candidate_fids:
        return []

    hydrated_rows = engine.store.facts_by_ids(candidate_fids[:40], scope=scope)
    scored: list[tuple[int, float, dict]] = []
    facet_stems = [stem_terms(f) for f in facets]
    for row in hydrated_rows:
        if exclude_superseded and row.get("status") != "active":
            continue
        text = strip_step_tags(row.get("plain") or row.get("text") or "")
        f_stems = stem_terms(text)
        q_hit = len(q_stems & f_stems)
        facet_hit = sum(1 for fs in facet_stems if fs and (fs & f_stems))
        if q_hit < 2 and facet_hit == 0:
            continue
        item = dict(row)
        if packing_mode == "token_budget":
            item = _compact_evidence(query, item)
        item.setdefault("ce_score", 0.28)
        item.setdefault("final_score", 0.28)
        item.setdefault("rrf_norm", 0.28)
        item.setdefault("exact_hit_ratio", 0.0)
        item.setdefault("exact_ident_hit", False)
        item.setdefault("version_mismatch", False)
        item["graph_enumerated"] = True
        scored.append((facet_hit, q_hit / max(1, len(q_stems)), item))

    scored.sort(key=lambda x: (-x[0], -x[1], x[2]["id"]))
    return [item for _, _, item in scored[:max_added]]


def coverage_first_session_pack(
    ranked: list[dict],
    query: str,
    *,
    is_aggregate: bool,
    is_multi_hop: bool,
    pinned_fn,
) -> list[dict]:
    """Coverage-first diversity packing for multi-session & multi-hop queries.

    Prevents redundant same-session assistant echoes or repeated turns from monopolizing
    Top-10 when distinct matching sessions or uncovered query facets exist in the pool.
    """
    if not (is_aggregate or is_multi_hop) or len(ranked) <= 3:
        return ranked

    facets = extract_query_facets(query)
    facet_stems = [stem_terms(f) for f in facets if stem_terms(f)]
    q_stems = stem_terms(query)
    known_speakers = {str(f.get("speaker") or "").casefold() for f in ranked[:25] if f.get("speaker")}
    q_content = {s for s in q_stems if s not in known_speakers} or q_stems

    pool = list(ranked[:30])
    tail = list(ranked[30:])

    chosen: list[dict] = []
    chosen_ids: set[str] = set()
    session_counts: dict[str, int] = {}
    covered_facets: set[int] = set()

    def _add(fact: dict) -> None:
        chosen.append(fact)
        chosen_ids.add(fact["id"])
        sid = str(fact.get("session_id") or "")
        if sid:
            session_counts[sid] = session_counts.get(sid, 0) + 1
        f_stems = stem_terms(fact.get("plain") or fact.get("text") or "")
        for idx, fs in enumerate(facet_stems):
            if fs & f_stems:
                covered_facets.add(idx)

    # 1. Preserve pinned facts AND top-5 primary hits so single-session multi-turn evidence & MRR@10 stay intact!
    for fact in pool:
        if pinned_fn(fact) and fact["id"] not in chosen_ids:
            _add(fact)
    for fact in pool[:5]:
        if fact["id"] not in chosen_ids:
            _add(fact)

    # 2. Ensure uncovered explicit query facets (e.g. YouTube & TikTok, Hawaii & Tokyo) get represented in Top-10
    if len(facet_stems) >= 2:
        for f_idx, fs in enumerate(facet_stems):
            if f_idx in covered_facets or len(chosen) >= 10:
                continue
            best = None
            best_key = (-1, -1, 0)
            for p_idx, fact in enumerate(pool):
                if fact["id"] in chosen_ids:
                    continue
                t_stems = stem_terms(fact.get("plain") or fact.get("text") or "")
                f_hits = len(fs & t_stems)
                if f_hits == 0:
                    continue
                q_hits = len(expand_category_stems(q_content) & t_stems)
                if q_hits < 1:
                    continue
                is_user = 1 if fact.get("role", "user") == "user" else 0
                key = (f_hits, q_hits + is_user, -p_idx)
                if key > best_key:
                    best_key = key
                    best = fact
            if best is not None:
                best = {**best, "coverage_priority": len(chosen)}
                _add(best)

    # 3. Fill remaining Top-10 slots (#6..#10) preferring distinct sessions (max 2 per session first)
    expanded_q = expand_category_stems(q_content)
    for max_per_session in (2, 99):
        if len(chosen) >= 10:
            break
        for fact in pool:
            if len(chosen) >= 10:
                break
            if fact["id"] in chosen_ids:
                continue
            sid = str(fact.get("session_id") or "")
            if sid and session_counts.get(sid, 0) >= max_per_session:
                continue
            # Prefer facts that actually overlap expanded query terms when promoting from deeper tail
            if max_per_session == 2 and fact.get("graph_enumerated"):
                t_stems = stem_terms(fact.get("plain") or fact.get("text") or "")
                if len(expanded_q & t_stems) < 2:
                    continue
            _add(fact)

    rest = [f for f in pool if f["id"] not in chosen_ids]
    return [*chosen, *rest, *tail]
