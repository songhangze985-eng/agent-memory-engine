"""Derived Profile view rendering the stable <=360-token system prefix from unified facts."""
from __future__ import annotations

from memengine.assembler import render_prefix
from memengine.core.facts import UnifiedFactLedger
from memengine.textutil import estimate_tokens


class ProfileView:
    """Projects core profile + always-active rules into a bounded system prefix."""

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
    ) -> str:
        snapshot = self.store.get_core_snapshot(user_id=user_id, agent_id=agent_id)
        base = render_prefix(snapshot)
        if not include_coding_preload:
            return base

        # Coding-agent profile Stage-1 deterministic preload: /global.md + always_active rules
        preload_rows = [
            r for r in self.ledger.list_path_facts(user_id=user_id, agent_id=agent_id)
            if r["vfs_path"] == "/global.md" or r["always_active"]
        ]
        preload_rows.sort(key=lambda r: (0 if r["vfs_path"] == "/global.md" else 1, r["vfs_path"]))
        sections = [base] if base else []
        used = estimate_tokens(base)
        for row in preload_rows:
            block = f"## [{row['vfs_path']}] {row['title'] or row['vfs_path']}\n{row['plain'].strip()}"
            cost = estimate_tokens(block) + 2
            if used + cost <= self.max_prefix_tokens:
                sections.append(block)
                used += cost
            else:
                compact = f"- [{row['vfs_path']}] {row['index_summary']}"
                c_cost = estimate_tokens(compact) + 1
                if used + c_cost <= self.max_prefix_tokens:
                    sections.append(compact)
                    used += c_cost
        return "\n\n".join(s for s in sections if s)
