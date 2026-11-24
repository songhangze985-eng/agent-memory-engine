"""Single unified context assembly: status filter, coverage-first packing, disclosure level, and coding profile."""
from __future__ import annotations

from hashlib import sha256
from typing import Literal

from memengine.assembler import assemble_context
from memengine.chrono import date_iso
from memengine.core.facts import UnifiedFactLedger
from memengine.core.plan import QueryPlan, RecallConfig
from memengine.core.supersede import SlotSupersedeTracker
from memengine.textutil import estimate_tokens, strip_step_tags
from memengine.views.profile import ProfileView
from memengine.views.vfs import VFSView


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
    token_budget: int = 2048,
    drilldown_top_n: int = 2,
    drilldown_window: int = 0,
    disclosure: Literal["full", "progressive"] = "full",
    profile: Literal["conversational", "coding_agent"] = "conversational",
    workspace=None,
    recent=(),
) -> dict:
    """Unify reconstructive coverage-first assembly and progressive disclosure into one function."""
    include_coding = profile == "coding_agent"
    prefix = profile_view.render_system_prefix(
        user_id=user_id,
        agent_id=agent_id,
        include_coding_preload=include_coding,
    )
    prefix_tokens = estimate_tokens(prefix)
    if prefix_tokens > token_budget:
        prefix = ""
        prefix_tokens = 0

    # Progressive disclosure mode: emit <=50-char index_summary list + directory outline (zero mid-rule truncation)
    if disclosure == "progressive":
        remaining = token_budget - prefix_tokens
        sections: list[str] = []
        items: list[dict] = []
        used = 0

        if include_coding:
            outline = vfs_view.tree_outline("/", user_id=user_id, agent_id=agent_id)
            outline_block = f"## Directory Outline (ls -R)\n{outline}"
            cost = estimate_tokens(outline_block) + 2
            if cost <= remaining:
                sections.append(outline_block)
                used += cost

        summary_lines = ["## Relevant Memories (Progressive Disclosure — call read_note(path) for full detail)"]
        ranked = list(recall_result.get("ranked") or ())
        for fact in ranked:
            vpath = fact.get("vfs_path") or f"fact:{fact['id']}"
            summary = fact.get("index_summary") or fact.get("plain", "")[:50]
            kind = fact.get("kind", "world")
            vstat = fact.get("verification_status", "verified")
            line = f"- [{vpath}] ({kind}|{vstat}) {summary}"
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
            **recall_result,
            "system_prefix": prefix,
            "prefix_version": sha256(prefix.encode("utf-8")).hexdigest()[:12],
            "dynamic_memory": dynamic,
            "empty_recall": not items and not include_coding,
            "disclosure": "progressive",
            "profile": profile,
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

    # Full mode: coverage-first reconstructive assembly + optional coding-agent outline + L0 fallback turns
    coding_header = ""
    coding_header_tokens = 0
    if include_coding:
        outline = vfs_view.tree_outline("/", user_id=user_id, agent_id=agent_id)
        if outline and "(empty)" not in outline:
            candidate_hdr = f"## Directory Outline (ls -R)\n{outline}"
            cost = estimate_tokens(candidate_hdr) + 2
            if cost <= max(128, (token_budget - prefix_tokens) // 4):
                coding_header = candidate_hdr
                coding_header_tokens = cost

    bundle = assemble_context(
        store,
        recall_result,
        prefix=prefix,
        token_budget=max(64, token_budget - coding_header_tokens),
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

    # Append un-extracted L0 FTS5 fallback turns if budget allows (closes extraction_miss gap!)
    fallback_turns = list(recall_result.get("episode_fallback_turns") or ())
    extra_snippets: list[str] = []
    seen_turns = set(evidence_turn_ids)
    seen_fact_ids = {item["id"] for item in items if item.get("kind") in {"fact", "observation"} and item.get("id")}
    for item in items:
        for tid in item.get("source_turn_ids") or ():
            seen_turns.add(int(tid))

    # M2 Historical / change-direction supersede-chain expansion
    qplan = recall_result.get("query_plan")
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
                if total_tokens + coding_header_tokens + cost > token_budget:
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
        if total_tokens + coding_header_tokens + cost > token_budget:
            continue
        extra_snippets.append(snippet)
        seen_turns.add(tid)
        evidence_turn_ids.append(tid)
        ev_id = f"turn:{tid}"
        ep_item = {**ep, "evidence_id": ev_id}
        items.append(ep_item)
        evidence_map[ev_id] = ep_item
        total_tokens += cost

    parts = [p for p in (coding_header, dynamic, "\n".join(extra_snippets)) if p]
    final_dynamic = "\n\n".join(parts)
    final_total = prefix_tokens + estimate_tokens(final_dynamic)
    breakdown = {
        **bundle["token_breakdown"],
        "dynamic_memory": estimate_tokens(final_dynamic),
        "total": final_total,
        "total_injected_tokens": final_total,
    }
    return {
        **recall_result,
        **bundle,
        "dynamic_memory": final_dynamic,
        "empty_recall": not items,
        "disclosure": "full",
        "profile": profile,
        "items": items,
        "evidence_map": evidence_map,
        "evidence_turn_ids": evidence_turn_ids,
        "task_id": None,
        "task_status": workspace["status"] if workspace else None,
        "workspace_session_id": session_id or "default",
        "token_breakdown": breakdown,
    }
