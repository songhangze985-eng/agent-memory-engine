"""Single unified context assembly driven purely by declarative MemoryProfile configuration."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Literal

from agentmem_core.assembler import assemble_context
from agentmem_core.chrono import date_iso
from agentmem_core.core.facts import UnifiedFactLedger
from agentmem_core.core.profiles import MemoryProfile, load_profile
from agentmem_core.core.scopes import ScopeChain
from agentmem_core.core.supersede import SlotSupersedeTracker
from agentmem_core.textutil import estimate_tokens, strip_step_tags
from agentmem_core.views.profile import ProfileView
from agentmem_core.views.vfs import VFSView


def _apply_stale_policy(ranked: list[dict], policy: str) -> tuple[list[dict], list[str]]:
    """Enforce profile's declarative stale_policy ('exclude' | 'demote' | 'warn')."""
    excluded: list[str] = []
    fresh: list[dict] = []
    stale_items: list[dict] = []
    for fact in ranked:
        if fact.get("staleness_status") == "stale":
            if policy == "exclude":
                excluded.append(str(fact["id"]))
                continue
            annotated = dict(fact)
            reason = annotated.get("stale_reason") or "anchor_drifted"
            prefix = f"[STALE: {reason}] "
            if not str(annotated.get("plain", "")).startswith("[STALE:"):
                annotated["plain"] = prefix + str(annotated.get("plain", ""))
            if not str(annotated.get("text", "")).startswith("[STALE:"):
                annotated["text"] = prefix + str(annotated.get("text", annotated.get("plain", "")))
            stale_items.append(annotated)
        else:
            fresh.append(fact)
    if policy == "demote":
        return [*fresh, *stale_items], excluded
    if policy == "warn":
        # Preserve rank order while carrying [STALE: ...] warning annotations
        by_id = {f["id"]: f for f in stale_items}
        return [by_id.get(f["id"], f) for f in ranked], excluded
    return fresh, excluded


def unified_assemble(
    store,
    ledger: UnifiedFactLedger,
    vfs_view: VFSView,
    profile_view: ProfileView,
    recall_result: dict,
    *,
    supersede: SlotSupersedeTracker | None = None,
    query: str,
    user_id: str = "default",
    agent_id: str = "default",
    session_id: str | None = None,
    scope_chain: ScopeChain | None = None,
    token_budget: int = 2048,
    drilldown_top_n: int = 2,
    drilldown_window: int = 0,
    disclosure: Literal["full", "progressive"] | None = None,
    profile: str | Path | MemoryProfile = "conversational",
    workspace=None,
    recent=(),
) -> dict:
    """Unify reconstructive coverage-first assembly and progressive disclosure with zero profile branches."""
    spec = load_profile(profile)
    eff_disclosure = disclosure or spec.disclosure
    exclude_stale = spec.stale_policy == "exclude"

    prefix = profile_view.render_system_prefix(
        user_id=user_id,
        agent_id=agent_id,
        preload_paths=spec.preload_paths,
        preload_always_active=spec.preload_always_active,
        scope_chain=scope_chain,
        exclude_stale=exclude_stale,
    )
    prefix_tokens = estimate_tokens(prefix)
    if prefix_tokens > token_budget:
        prefix = ""
        prefix_tokens = 0

    filtered_ranked, excluded_stale_ids = _apply_stale_policy(
        list(recall_result.get("ranked") or ()),
        spec.stale_policy,
    )
    effective_recall = {
        **recall_result,
        "ranked": filtered_ranked,
        "excluded_stale_ids": excluded_stale_ids,
    }

    # Progressive disclosure mode: emit <=50-char index_summary list + optional outline
    if eff_disclosure == "progressive":
        remaining = token_budget - prefix_tokens
        sections: list[str] = []
        items: list[dict] = []
        used = 0

        if spec.include_directory_outline:
            outline = vfs_view.tree_outline(
                "/",
                user_id=user_id,
                agent_id=agent_id,
                scope_chain=scope_chain,
                include_stale=not exclude_stale,
            )
            outline_block = f"## Directory Outline (ls -R)\n{outline}"
            cost = estimate_tokens(outline_block) + 2
            if cost <= remaining:
                sections.append(outline_block)
                used += cost

        summary_lines = ["## Relevant Memories (Progressive Disclosure — call read_note(path) for full detail)"]
        for fact in filtered_ranked:
            vpath = fact.get("vfs_path") or f"fact:{fact['id']}"
            summary = fact.get("index_summary") or fact.get("plain", "")[:50]
            kind = fact.get("kind", "world")
            vstat = fact.get("verification_status", "verified")
            skey = fact.get("scope_key")
            scope_tag = f"|{skey}" if skey else ""
            stale_tag = "|STALE" if fact.get("staleness_status") == "stale" else ""
            line = f"- [{vpath}] ({kind}|{vstat}{scope_tag}{stale_tag}) {summary}"
            cost = estimate_tokens(line) + 1
            if used + cost > remaining:
                break
            summary_lines.append(line)
            used += cost
            items.append({**fact, "kind": "fact", "text": summary, "evidence_id": f"fact:{fact['id']}"})
        if len(summary_lines) > 1:
            sections.append("\n".join(summary_lines))

        dynamic = "\n\n".join(sections)
        dyn_tokens = estimate_tokens(dynamic)
        evidence_map = {item["evidence_id"]: item for item in items}
        return {
            **effective_recall,
            "system_prefix": prefix,
            "prefix_version": sha256(prefix.encode("utf-8")).hexdigest()[:12],
            "dynamic_memory": dynamic,
            "empty_recall": not items and not spec.include_directory_outline,
            "disclosure": "progressive",
            "profile": spec.name,
            "scope_chain": scope_chain.as_list() if scope_chain else None,
            "items": items,
            "evidence_map": evidence_map,
            "evidence_turn_ids": [],
            "token_breakdown": {
                "system_prefix": prefix_tokens,
                "dynamic_memory": dyn_tokens,
                "total": prefix_tokens + dyn_tokens,
                "prefix_tokens": prefix_tokens,
                "facts_tokens": dyn_tokens,
                "drilldown_tokens": 0,
                "total_injected_tokens": prefix_tokens + dyn_tokens,
            },
            "task_id": None,
            "task_status": workspace["status"] if workspace else None,
            "workspace_session_id": session_id or "default",
            "lexical_facet_coverage": {"facets": [], "covered": 0, "total": 0},
        }

    # Full mode: coverage-first reconstructive assembly + optional outline + L0 fallback turns
    outline_header = ""
    outline_header_tokens = 0
    if spec.include_directory_outline:
        outline = vfs_view.tree_outline(
            "/",
            user_id=user_id,
            agent_id=agent_id,
            scope_chain=scope_chain,
            include_stale=not exclude_stale,
        )
        if outline and "(empty)" not in outline:
            candidate_hdr = f"## Directory Outline (ls -R)\n{outline}"
            cost = estimate_tokens(candidate_hdr) + 2
            if cost <= max(128, (token_budget - prefix_tokens) // 4):
                outline_header = candidate_hdr
                outline_header_tokens = cost

    bundle = assemble_context(
        store,
        effective_recall,
        prefix=prefix,
        token_budget=max(64, token_budget - outline_header_tokens),
        drilldown_top_n=drilldown_top_n,
        drilldown_window=drilldown_window,
        query=query,
        workspace=workspace,
        recent=recent,
    )

    items = list(bundle["items"])
    evidence_map = dict(bundle["evidence_map"])
    evidence_turn_ids = list(bundle["evidence_turn_ids"])
    dynamic = bundle["dynamic_memory"]
    total_tokens = int(bundle["token_breakdown"]["total"])

    fallback_turns = list(effective_recall.get("episode_fallback_turns") or ())
    extra_snippets: list[str] = []
    seen_turns = set(evidence_turn_ids)
    seen_fact_ids = {item["id"] for item in items if item.get("kind") in {"fact", "observation"} and item.get("id")}
    for item in items:
        for tid in item.get("source_turn_ids") or ():
            seen_turns.add(int(tid))

    qplan = effective_recall.get("query_plan")
    if supersede is not None and qplan is not None and (qplan.is_historical or qplan.change_direction_ask):
        chains = supersede.chain_for_facts(list(seen_fact_ids), user_id=user_id, agent_id=agent_id)
        for skey, members in chains.items():
            for member in members:
                fid = member["fact_id"]
                if fid in seen_fact_ids:
                    continue
                d_str = date_iso(member.get("event_ts") or member.get("event_time_start")) or ""
                tag = "[历史版本" + (f" {d_str}" if d_str else "") + "]"
                snippet = f"{tag} {strip_step_tags(member['plain']).strip()}"
                cost = estimate_tokens(snippet) + 1
                if total_tokens + outline_header_tokens + cost > token_budget:
                    continue
                extra_snippets.append(snippet)
                seen_fact_ids.add(fid)
                ev_id = f"fact:{fid}"
                hist_item = {
                    "id": fid,
                    "kind": "fact",
                    "plain": member["plain"],
                    "text": snippet,
                    "status": member["status"],
                    "event_time_start": member["event_time_start"],
                    "mention_time": member["event_ts"],
                    "evidence_id": ev_id,
                    "historical_slot": skey,
                }
                items.append(hist_item)
                evidence_map[ev_id] = hist_item
                total_tokens += cost
    for ep in fallback_turns:
        tid = int(ep["turn_id"])
        if tid in seen_turns:
            continue
        snippet = f"[L0 Turn {tid}] {ep['text']}"
        cost = estimate_tokens(snippet) + 1
        if total_tokens + outline_header_tokens + cost > token_budget:
            continue
        extra_snippets.append(snippet)
        seen_turns.add(tid)
        evidence_turn_ids.append(tid)
        ev_id = f"turn:{tid}"
        ep_item = {**ep, "evidence_id": ev_id}
        items.append(ep_item)
        evidence_map[ev_id] = ep_item
        total_tokens += cost

    parts = [p for p in (outline_header, dynamic, "\n".join(extra_snippets)) if p]
    final_dynamic = "\n\n".join(parts)
    final_total = prefix_tokens + estimate_tokens(final_dynamic)
    breakdown = {
        **bundle["token_breakdown"],
        "dynamic_memory": estimate_tokens(final_dynamic),
        "total": final_total,
        "total_injected_tokens": final_total,
    }
    return {
        **effective_recall,
        **bundle,
        "dynamic_memory": final_dynamic,
        "empty_recall": not items,
        "disclosure": "full",
        "profile": spec.name,
        "scope_chain": scope_chain.as_list() if scope_chain else None,
        "items": items,
        "evidence_map": evidence_map,
        "evidence_turn_ids": evidence_turn_ids,
        "task_id": None,
        "task_status": workspace["status"] if workspace else None,
        "workspace_session_id": session_id or "default",
        "token_breakdown": breakdown,
    }
