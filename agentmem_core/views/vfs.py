"""Derived VFS path-tree view over unified facts (kind='rule' and path-addressed facts)."""
from __future__ import annotations

import time
import uuid

from agentmem_core.core.facts import UnifiedFactLedger
from agentmem_core.core.scopes import ScopeChain
from agentmem_core.textutil import estimate_tokens
from agentmem_core.vfs.models import (
    make_index_summary,
    normalize_dir_path,
    normalize_node_path,
    parse_structured_rule,
)

DEFAULT_MAX_NOTES_PER_DIR = 10
DEFAULT_MAX_DIR_TOKENS = 3000


class VFSView:
    """Projects unified facts with ``vfs_path`` as a Claude-style Markdown directory tree.

    Zero separate fact table: all rules live in the unified ``facts`` ledger with
    ``kind='rule'`` (or path-bound ``belief``/``preference``), and VFS operations read,
    outline, compact, and rollback directly against that single source of truth.
    """

    def __init__(
        self,
        engine,
        ledger: UnifiedFactLedger,
        *,
        max_notes_per_dir: int = DEFAULT_MAX_NOTES_PER_DIR,
        max_dir_tokens: int = DEFAULT_MAX_DIR_TOKENS,
    ):
        self._engine = engine
        self._ledger = ledger
        self.max_notes_per_dir = max(2, int(max_notes_per_dir))
        self.max_dir_tokens = max(256, int(max_dir_tokens))

    def read_note(
        self,
        path: str,
        *,
        user_id: str = "default",
        agent_id: str = "default",
        scope_key: str | None = None,
        scope_chain: ScopeChain | None = None,
    ) -> dict | None:
        row = self._ledger.get_active_by_path(
            path,
            user_id=user_id,
            agent_id=agent_id,
            scope_key=scope_key,
            scope_chain=scope_chain,
        )
        if row is None:
            return None
        return {
            "fact_id": row["fact_id"],
            "path": row["vfs_path"],
            "parent_path": row["parent_path"],
            "kind": row["kind"],
            "scope_key": row.get("scope_key", "global"),
            "override_key": row.get("override_key", ""),
            "staleness_status": row.get("staleness_status", "fresh"),
            "stale_reason": row.get("stale_reason", ""),
            "title": row["title"],
            "index_summary": row["index_summary"],
            "content": row["plain"],
            "detail_body": row["plain"],
            "verification_status": row["verification_status"],
            "salience_tag": row["salience_tag"],
            "always_active": row["always_active"],
            "version": row["version"],
            "source_transcript_ids": row["source_transcript_ids"],
            "updated_at": row["updated_at"],
        }

    def preload_always_active(
        self,
        *,
        user_id: str = "default",
        agent_id: str = "default",
        preload_paths: tuple[str, ...] = ("/global.md",),
        scope_chain: ScopeChain | None = None,
        include_stale: bool = False,
    ) -> list[dict]:
        """Stage-1 deterministic preload for configured paths + ``always_active`` rules."""
        path_set = set(preload_paths)
        rows = self._ledger.list_path_facts(
            user_id=user_id,
            agent_id=agent_id,
            scope_chain=scope_chain,
            include_stale=include_stale,
        )
        selected = [
            r for r in rows
            if r["vfs_path"] in path_set or r["always_active"]
        ]
        selected.sort(key=lambda r: (0 if r["vfs_path"] in path_set else 1, int(r.get("scope_rank", 99)), r["vfs_path"]))
        return selected

    def tree_outline(
        self,
        dir_path: str = "/",
        *,
        user_id: str = "default",
        agent_id: str = "default",
        kinds: tuple[str, ...] | None = None,
        scope_chain: ScopeChain | None = None,
        include_stale: bool = True,
    ) -> str:
        """Compact ``ls -R`` style directory outline with <=50-char ``index_summary``."""
        norm_dir = normalize_dir_path(dir_path)
        rows = self._ledger.list_path_facts(
            user_id=user_id,
            agent_id=agent_id,
            path_prefix=norm_dir,
            kinds=kinds,
            scope_chain=scope_chain,
            include_stale=include_stale,
        )
        if not rows:
            return f"{norm_dir}\n(empty)"
        by_dir: dict[str, list[dict]] = {}
        for row in rows:
            by_dir.setdefault(row["parent_path"] or "/", []).append(row)
        lines: list[str] = []
        for parent in sorted(by_dir):
            lines.append(parent)
            for row in sorted(by_dir[parent], key=lambda r: r["vfs_path"]):
                fname = row["vfs_path"].rsplit("/", 1)[-1]
                pin = "|always_active" if row["always_active"] else ""
                stale = "|STALE" if row.get("staleness_status") == "stale" else ""
                lines.append(
                    f"- {fname} [{row['kind']}|{row['verification_status']}|v{row['version']}{pin}{stale}] — {row['index_summary']}"
                )
        return "\n".join(lines)

    def check_compaction_needed(
        self,
        dir_path: str,
        *,
        user_id: str = "default",
        agent_id: str = "default",
    ) -> dict:
        norm_dir = normalize_dir_path(dir_path)
        rows = [
            r for r in self._ledger.list_path_facts(user_id=user_id, agent_id=agent_id, path_prefix=norm_dir)
            if r["parent_path"] == norm_dir and not r["vfs_path"].endswith("/GUIDE.md")
        ]
        total_tokens = sum(estimate_tokens(r["plain"]) for r in rows)
        return {
            "dir_path": norm_dir,
            "note_count": len(rows),
            "total_tokens": total_tokens,
            "needs_compaction": len(rows) > self.max_notes_per_dir or total_tokens > self.max_dir_tokens,
        }

    def compact_directory(
        self,
        dir_path: str,
        *,
        user_id: str = "default",
        agent_id: str = "default",
        session_id: str = "default",
        timestamp: float | None = None,
    ) -> dict:
        """Merge fragmented directory rules into one structured ``GUIDE.md`` with full batch rollback."""
        norm_dir = normalize_dir_path(dir_path)
        now = time.time() if timestamp is None else float(timestamp)
        rows = [
            r for r in self._ledger.list_path_facts(user_id=user_id, agent_id=agent_id, path_prefix=norm_dir)
            if r["parent_path"] == norm_dir and not r["vfs_path"].endswith("/GUIDE.md")
        ]
        if len(rows) < 2:
            return {"compacted": False, "dir_path": norm_dir, "merged_count": 0, "guide_path": None, "batch_id": None}

        batch_id = f"compact_{uuid.uuid4().hex[:12]}"
        verified = [r for r in rows if r["verification_status"] == "verified"]
        unverified = [r for r in rows if r["verification_status"] != "verified"]
        sections = [f"# Standard Guide: {norm_dir}"]
        seen_sigs: set[str] = set()

        for heading, group in (("Verified Rules", verified), ("Unverified Hypotheses", unverified)):
            bullets: list[str] = []
            for row in group:
                parsed = parse_structured_rule(row["plain"])
                if parsed["trigger_condition"] and parsed["action"]:
                    sig = f"{parsed['trigger_condition'].casefold()}|{parsed['action'].casefold()}"
                    if sig in seen_sigs:
                        continue
                    seen_sigs.add(sig)
                    bullets.append(
                        f"### {row['title'] or row['vfs_path']}\n"
                        f"- [触发条件] {parsed['trigger_condition']}\n"
                        f"- [动作] {parsed['action']}\n"
                        f"- [验证结果] {parsed['verification_result'] or '已整合'}"
                    )
                else:
                    body = " ".join(ln.strip() for ln in row["plain"].splitlines() if ln.strip() and not ln.strip().startswith("#"))
                    if not body or body.casefold() in seen_sigs:
                        continue
                    seen_sigs.add(body.casefold())
                    bullets.append(f"### {row['title'] or row['vfs_path']}\n{body}")
            if bullets:
                sections.append(f"## {heading}\n\n" + "\n\n".join(bullets))

        guide_path = f"{norm_dir}GUIDE.md"
        guide_title = f"Standard Guide ({norm_dir.strip('/') or 'root'})"
        guide_content = "\n\n".join(sections)
        all_tids = list(dict.fromkeys(tid for r in rows for tid in r["source_transcript_ids"]))
        all_turns: list[int] = []
        for r in rows:
            raw_turns = r.get("source_turn_ids")
            if isinstance(raw_turns, str):
                import json

                try:
                    raw_turns = json.loads(raw_turns)
                except Exception:
                    raw_turns = []
            for tid in raw_turns or ():
                if int(tid) not in all_turns:
                    all_turns.append(int(tid))

        # Snapshot all source notes before superseding them so batch rollback restores 100% losslessly
        for row in rows:
            self._ledger._save_snapshot(row, batch_id=batch_id, timestamp=now)
            self._engine.store.conn.execute(
                "UPDATE facts SET status='superseded', valid_until=? WHERE id=? AND status='active'",
                (now, row["fact_id"]),
            )
            self._engine.store.conn.execute(
                "UPDATE fact_unified_meta SET updated_at=? WHERE fact_id=?",
                (now, row["fact_id"]),
            )

        guide_fact_id = self._engine.remember(
            guide_content,
            memory_type="procedural",
            source_turn_ids=all_turns,
            trigger=f"Guide for {norm_dir}",
            is_constraint=any(r["always_active"] for r in rows),
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            timestamp=now,
            confidence=max((float(r.get("confidence", 0.8)) for r in verified), default=0.75),
        )
        self._ledger.attach(
            fact_id=guide_fact_id,
            user_id=user_id,
            agent_id=agent_id,
            kind="rule",
            vfs_path=guide_path,
            title=guide_title,
            content=guide_content,
            index_summary=make_index_summary(guide_title, guide_content, f"{norm_dir.strip('/') or 'root'} 指南（合并 {len(rows)} 条）"),
            verification_status="verified" if verified else "unverified",
            salience_tag="core_rule",
            always_active=any(r["always_active"] for r in rows),
            source_transcript_ids=all_tids,
            batch_id=batch_id,
            timestamp=now,
        )
        self._engine.store.commit()
        before_tokens = sum(estimate_tokens(r["plain"]) for r in rows)
        after_tokens = estimate_tokens(guide_content)
        return {
            "compacted": True,
            "dir_path": norm_dir,
            "merged_count": len(rows),
            "guide_path": guide_path,
            "guide_fact_id": guide_fact_id,
            "batch_id": batch_id,
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
            "token_reduction_rate": round((before_tokens - after_tokens) / before_tokens, 4) if before_tokens else 0.0,
        }

    def rollback(
        self,
        *,
        user_id: str = "default",
        agent_id: str = "default",
        path: str | None = None,
        snapshot_id: str | None = None,
        batch_id: str | None = None,
        timestamp: float | None = None,
    ) -> dict:
        """Restore unified facts from pre-mutation snapshots (by path, snapshot_id, or batch_id)."""
        now = time.time() if timestamp is None else float(timestamp)
        snaps: list[dict] = []
        if snapshot_id:
            snap = self._ledger.get_snapshot(snapshot_id, user_id=user_id, agent_id=agent_id)
            if snap:
                snaps = [snap]
        elif batch_id:
            snaps = self._ledger.list_snapshots(user_id=user_id, agent_id=agent_id, batch_id=batch_id)
        elif path:
            all_snaps = self._ledger.list_snapshots(user_id=user_id, agent_id=agent_id, vfs_path=path)
            if all_snaps:
                snaps = [all_snaps[0]]
        else:
            raise ValueError("rollback requires path, snapshot_id, or batch_id")

        restored: list[str] = []
        for snap in snaps:
            vpath = snap.get("vfs_path")
            if vpath:
                # Deactivate any newer active fact currently holding this vfs_path
                current = self._ledger.get_active_by_path(vpath, user_id=user_id, agent_id=agent_id)
                if current and current["fact_id"] != snap["fact_id"]:
                    self._engine.store.conn.execute(
                        "UPDATE facts SET status='superseded', valid_until=? WHERE id=? AND status='active'",
                        (now, current["fact_id"]),
                    )
                    self._engine.store.conn.execute(
                        "UPDATE fact_unified_meta SET vfs_path=NULL, parent_path=NULL, updated_at=? WHERE fact_id=?",
                        (now, current["fact_id"]),
                    )
            # Re-activate original fact & restore its unified metadata
            self._engine.store.conn.execute(
                "UPDATE facts SET status='active', valid_until=NULL WHERE id=?",
                (snap["fact_id"],),
            )
            parent, norm = normalize_node_path(vpath) if vpath else (None, None)
            self._engine.store.conn.execute(
                """UPDATE fact_unified_meta
                   SET vfs_path=?, parent_path=?, title=?, index_summary=?, verification_status=?,
                       salience_tag=?, always_active=?, version=?, updated_at=?
                   WHERE fact_id=?""",
                (
                    norm,
                    parent,
                    snap["title"],
                    snap["index_summary"],
                    snap["verification_status"],
                    snap["salience_tag"],
                    int(bool(snap["always_active"])),
                    int(snap["version"]),
                    now,
                    snap["fact_id"],
                ),
            )
            restored.append(norm or snap["fact_id"])

        # If rolling back a compaction batch, deactivate any GUIDE.md created in the directory
        if batch_id and snaps:
            parents = {normalize_node_path(s["vfs_path"])[0] for s in snaps if s.get("vfs_path")}
            for pdir in parents:
                guide = self._ledger.get_active_by_path(f"{pdir}GUIDE.md", user_id=user_id, agent_id=agent_id)
                if guide:
                    self._engine.store.conn.execute(
                        "UPDATE facts SET status='superseded', valid_until=? WHERE id=? AND status='active'",
                        (now, guide["fact_id"]),
                    )
                    self._engine.store.conn.execute(
                        "UPDATE fact_unified_meta SET vfs_path=NULL, parent_path=NULL, updated_at=? WHERE fact_id=?",
                        (now, guide["fact_id"]),
                    )

        self._engine.store.commit()
        return {"restored": restored, "count": len(restored)}
