"""Derived Profile view rendering the stable <=360-token system prefix from unified facts."""
from __future__ import annotations

from agentmem_core.assembler import render_prefix
from agentmem_core.core.facts import UnifiedFactLedger
from agentmem_core.core.scopes import ScopeChain
from agentmem_core.textutil import estimate_tokens


class ProfileView:
    """Projects core profile + declaratively configured preload paths / always-active rules."""

    def __init__(self, store, ledger: UnifiedFactLedger, *, max_prefix_tokens: int = 360):
        self.store = store
        self.ledger = ledger
        self.max_prefix_tokens = int(max_prefix_tokens)

    def render_system_prefix(
        self,
        *,
        user_id: str = "default",
        agent_id: str = "default",
        include_coding_preload: bool = False,
        preload_paths: tuple[str, ...] = (),
        preload_always_active: bool = False,
        scope_chain: ScopeChain | None = None,
        exclude_stale: bool = True,
    ) -> str:
        snapshot = self.store.get_core_snapshot(user_id=user_id, agent_id=agent_id)
        base = render_prefix(snapshot)

        # Backward-compatible mapping of legacy boolean flag onto declarative fields
        eff_paths = tuple(preload_paths) if preload_paths else (("/global.md",) if include_coding_preload else ())
        eff_always = bool(preload_always_active or include_coding_preload)
        if not eff_paths and not eff_always:
            return base

        path_set = set(eff_paths)
        all_rows = self.ledger.list_path_facts(
            user_id=user_id,
            agent_id=agent_id,
            scope_chain=scope_chain,
            include_stale=not exclude_stale,
        )
        preload_rows = [
            r for r in all_rows
            if r["vfs_path"] in path_set or (eff_always and r["always_active"])
        ]
        preload_rows.sort(
            key=lambda r: (
                0 if r["vfs_path"] in path_set else 1,
                int(r.get("scope_rank", 99)),
                r["vfs_path"],
            )
        )
        sections = [base] if base else []
        used = estimate_tokens(base)
        for row in preload_rows:
            stale_tag = f" [STALE: {row['stale_reason']}]" if row.get("staleness_status") == "stale" else ""
            block = f"## [{row['vfs_path']}] {row['title'] or row['vfs_path']}{stale_tag}\n{row['plain'].strip()}"
            cost = estimate_tokens(block) + 2
            if used + cost <= self.max_prefix_tokens:
                sections.append(block)
                used += cost
            else:
                compact = f"- [{row['vfs_path']}]{stale_tag} {row['index_summary']}"
                c_cost = estimate_tokens(compact) + 1
                if used + c_cost <= self.max_prefix_tokens:
                    sections.append(compact)
                    used += c_cost
        return "\n\n".join(s for s in sections if s)
