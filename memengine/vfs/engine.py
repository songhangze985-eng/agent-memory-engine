"""High-level VFS Memory Engine: L0 chunked logs + L1/L2/L3 Markdown tree + L2-only hybrid retrieval."""
from __future__ import annotations

import hashlib
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable

from memengine.lifecycle import explicit_preference, is_constraint, is_taboo
from memengine.textutil import estimate_tokens
from memengine.vfs.models import (
    MemoryNode,
    OptimisticLockError,
    TranscriptRecord,
    normalize_dir_path,
    normalize_node_path,
)
from memengine.vfs.store import SQLiteVFSStore, open_vfs_store
from memengine.vfs.transcripts import LocalChunkTranscriptStore, TranscriptStore

_INDEX_PATH = "/INDEX.md"


def _slugify(text: str, fallback: str = "note") -> str:
    slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", (text or "").strip()).strip("-").lower()
    return (slug[:48] or fallback) + ".md"


def _default_dream_synthesizer(
    transcripts: list[TranscriptRecord],
    l1_nodes: list[MemoryNode],
    existing_l2: list[MemoryNode],
) -> list[dict]:
    """Deterministic zero-LLM dreaming synthesizer promoting L0/L1 taboos, preferences, and rules to L2."""
    proposals: list[dict] = []
    taboo_lines: list[tuple[str, str]] = []
    pref_lines: list[tuple[str, str]] = []

    for rec in transcripts:
        if rec.role != "user":
            continue
        text = rec.text.strip()
        if not text:
            continue
        if (is_taboo(text) or is_constraint(text)) and not text.rstrip().endswith(("?", "？")):
            taboo_lines.append((text, rec.transcript_id))
        elif explicit_preference(text, rec.speaker) is not None:
            pref_lines.append((text, rec.transcript_id))

    if taboo_lines:
        existing_taboo = next((n for n in existing_l2 if n.path == "/taboos/constraints.md"), None)
        body_lines = list(existing_taboo.content.splitlines()) if existing_taboo else ["# Taboos & Hard Constraints"]
        seen = {line.strip() for line in body_lines}
        sources = []
        for text, tid in taboo_lines:
            bullet = f"- {text}"
            if bullet not in seen:
                body_lines.append(bullet)
                seen.add(bullet)
            sources.append(tid)
        proposals.append({
            "path": "/taboos/constraints.md",
            "title": "Taboos & Hard Constraints",
            "summary": f"{len(body_lines) - 1} hard user constraints and taboos",
            "content": "\n".join(body_lines),
            "layer": "L2",
            "confidence_score": 0.95,
            "verification_status": "verified",
            "salience_tag": "taboo",
            "source_transcript_ids": sources,
        })

    if pref_lines:
        existing_pref = next((n for n in existing_l2 if n.path == "/preferences/user.md"), None)
        body_lines = list(existing_pref.content.splitlines()) if existing_pref else ["# User Preferences"]
        seen = {line.strip() for line in body_lines}
        sources = []
        for text, tid in pref_lines:
            bullet = f"- {text}"
            if bullet not in seen:
                body_lines.append(bullet)
                seen.add(bullet)
            sources.append(tid)
        proposals.append({
            "path": "/preferences/user.md",
            "title": "User Preferences",
            "summary": f"{len(body_lines) - 1} explicit user preferences",
            "content": "\n".join(body_lines),
            "layer": "L2",
            "confidence_score": 0.88,
            "verification_status": "verified",
            "salience_tag": "preference",
            "source_transcript_ids": sources,
        })

    # Promote verified L1 working notes into L2 strategies automatically
    for note in l1_nodes:
        if note.verification_status == "verified":
            rel = note.path.removeprefix("/working/") if note.path.startswith("/working/") else note.path.lstrip("/")
            target = "/strategies/" + rel
            proposals.append({
                "path": target,
                "title": note.title,
                "summary": note.summary or note.title,
                "content": note.content,
                "layer": "L2",
                "confidence_score": max(0.85, note.confidence_score),
                "verification_status": "verified",
                "salience_tag": note.salience_tag if note.salience_tag != "normal" else "core_rule",
                "source_transcript_ids": list(note.source_transcript_ids),
            })
    return proposals


class VFSMemoryEngine:
    """Claude-style Virtual Filesystem Memory Engine.

    Architecture:
      - L0 (`transcript_store`): Chunked gzip object storage for raw dialogue transcripts (ZERO embeddings).
      - L1 (`memory_nodes`, layer='L1'): Active working/episodic Markdown scratchpads (NOT globally vectorized).
      - L2 (`memory_nodes`, layer='L2'): Curated semantic & procedural Markdown notes (ONLY layer vectorized/BM25-indexed).
      - L3 (`memory_nodes`, layer='L3'): Deterministic `/INDEX.md` directory tree & core rule manifest (KV-cache static prefix).
    """

    def __init__(
        self,
        db_url: str | Path | None = None,
        *,
        transcript_dir: str | Path | None = None,
        transcript_store: TranscriptStore | None = None,
        embedder=None,
        external_vector_index=None,
        prefix_token_budget: int = 360,
    ):
        self.store = open_vfs_store(db_url, embedder=embedder, external_vector_index=external_vector_index)
        self._temp_dir = None
        if transcript_store is not None:
            self.transcripts = transcript_store
        else:
            if transcript_dir is None:
                if db_url and not str(db_url).startswith(("postgresql://", "postgres://", ":memory:")):
                    base = Path(str(db_url).removeprefix("sqlite:///")).expanduser().resolve()
                    transcript_dir = base.parent / (base.stem + "_l0_transcripts")
                else:
                    self._temp_dir = tempfile.TemporaryDirectory(prefix="memengine_l0_")
                    transcript_dir = Path(self._temp_dir.name)
            self.transcripts = LocalChunkTranscriptStore(transcript_dir)
        self.prefix_token_budget = int(prefix_token_budget)

    def close(self) -> None:
        try:
            self.transcripts.flush()
        finally:
            self.store.close()
            if self._temp_dir is not None:
                self._temp_dir.cleanup()
                self._temp_dir = None

    # ---- L0: Decoupled Raw Transcripts (NEVER vectorized) ----

    def append_transcript(
        self,
        text: str,
        *,
        user_id: str = "default",
        agent_id: str = "default",
        session_id: str = "default",
        role: str = "user",
        timestamp: float | None = None,
        speaker: str | None = None,
        metadata: dict | None = None,
    ) -> TranscriptRecord:
        return self.transcripts.append(
            user_id=user_id, agent_id=agent_id, session_id=session_id,
            role=role, text=text, timestamp=timestamp, speaker=speaker, metadata=metadata,
        )

    def read_transcripts(
        self,
        transcript_ids: list[str],
        *,
        user_id: str = "default",
        agent_id: str = "default",
    ) -> list[dict]:
        records = self.transcripts.get_by_ids(transcript_ids, user_id=user_id, agent_id=agent_id)
        return [
            {
                "transcript_id": r.transcript_id,
                "session_id": r.session_id,
                "role": r.role,
                "speaker": r.speaker,
                "text": r.text,
                "timestamp": r.timestamp,
                "metadata": r.metadata,
            }
            for r in records
        ]

    # ---- L1 / L2 / L3: Virtual Filesystem Operations ----

    def write(
        self,
        path: str,
        *,
        title: str,
        content: str,
        summary: str = "",
        layer: str = "L2",
        confidence_score: float = 0.7,
        verification_status: str = "unverified",
        salience_tag: str = "normal",
        source_transcript_ids: list[str] | tuple[str, ...] = (),
        expected_version: int | None = None,
        dream_batch_id: str | None = None,
        reason: str = "write",
        user_id: str = "default",
        agent_id: str = "default",
        timestamp: float | None = None,
    ) -> dict:
        node = self.store.upsert_node(
            user_id=user_id,
            agent_id=agent_id,
            path=path,
            title=title,
            summary=summary,
            content=content,
            layer=layer,
            confidence_score=confidence_score,
            verification_status=verification_status,
            salience_tag=salience_tag,
            source_transcript_ids=source_transcript_ids,
            expected_version=expected_version,
            dream_batch_id=dream_batch_id,
            reason=reason,
            timestamp=timestamp,
        )
        return node.to_dict()

    def read(self, path: str, *, user_id: str = "default", agent_id: str = "default") -> dict | None:
        node = self.store.get_node(path, user_id=user_id, agent_id=agent_id)
        return node.to_dict() if node is not None else None

    def edit(
        self,
        path: str,
        *,
        old_str: str,
        new_str: str,
        expected_version: int | None = None,
        verification_status: str | None = None,
        confidence_score: float | None = None,
        salience_tag: str | None = None,
        source_transcript_ids: list[str] | tuple[str, ...] = (),
        dream_batch_id: str | None = None,
        reason: str = "edit",
        user_id: str = "default",
        agent_id: str = "default",
        timestamp: float | None = None,
    ) -> dict:
        existing = self.store.get_node(path, user_id=user_id, agent_id=agent_id)
        if existing is None:
            raise FileNotFoundError(f"VFS node not found: {path}")
        if not old_str:
            raise ValueError("old_str must be nonempty")
        count = existing.content.count(old_str)
        if count == 0:
            raise ValueError(f"old_str not found in {existing.path}")
        if count > 1:
            raise ValueError(f"old_str matched {count} locations in {existing.path}; provide a unique snippet")
        new_content = existing.content.replace(old_str, new_str, 1)
        lock_ver = existing.version if expected_version is None else expected_version
        node = self.store.upsert_node(
            user_id=user_id,
            agent_id=agent_id,
            path=existing.path,
            title=existing.title,
            summary=existing.summary,
            content=new_content,
            layer=existing.layer,
            confidence_score=existing.confidence_score if confidence_score is None else confidence_score,
            verification_status=existing.verification_status if verification_status is None else verification_status,
            salience_tag=existing.salience_tag if salience_tag is None else salience_tag,
            source_transcript_ids=source_transcript_ids,
            expected_version=lock_ver,
            dream_batch_id=dream_batch_id,
            reason=reason,
            timestamp=timestamp,
        )
        return node.to_dict()

    def delete(
        self,
        path: str,
        *,
        expected_version: int | None = None,
        dream_batch_id: str | None = None,
        reason: str = "delete",
        user_id: str = "default",
        agent_id: str = "default",
        timestamp: float | None = None,
    ) -> bool:
        return self.store.delete_node(
            user_id=user_id, agent_id=agent_id, path=path,
            expected_version=expected_version, dream_batch_id=dream_batch_id,
            reason=reason, timestamp=timestamp,
        )

    def ls(self, path: str = "/", *, user_id: str = "default", agent_id: str = "default", layer: str | None = None) -> dict:
        norm_dir = normalize_dir_path(path)
        all_nodes = self.store.list_nodes(user_id=user_id, agent_id=agent_id, parent_path=norm_dir, recursive=True, layer=layer)
        subdirs: set[str] = set()
        files: list[dict] = []
        for node in all_nodes:
            if node.parent_path == norm_dir:
                files.append({
                    "path": node.path,
                    "title": node.title,
                    "layer": node.layer,
                    "verification_status": node.verification_status,
                    "salience_tag": node.salience_tag,
                    "confidence_score": node.confidence_score,
                    "version": node.version,
                })
            elif node.parent_path.startswith(norm_dir):
                rest = node.parent_path[len(norm_dir):].strip("/")
                if rest:
                    first = rest.split("/", 1)[0]
                    subdirs.add(norm_dir + first + "/")
        return {"path": norm_dir, "directories": sorted(subdirs), "files": files}

    def tree(self, path: str = "/", *, user_id: str = "default", agent_id: str = "default", layer: str | None = None) -> str:
        norm_dir = normalize_dir_path(path)
        nodes = self.store.list_nodes(user_id=user_id, agent_id=agent_id, parent_path=norm_dir, recursive=True, layer=layer)
        if not nodes:
            return f"{norm_dir} (empty)"
        lines = [norm_dir]
        by_dir: dict[str, list[MemoryNode]] = {}
        for node in nodes:
            by_dir.setdefault(node.parent_path, []).append(node)
        for folder in sorted(by_dir):
            if folder != norm_dir:
                rel_folder = folder[len(norm_dir):]
                depth = rel_folder.strip("/").count("/") + 1
                lines.append(f"{'  ' * (depth - 1)}- {folder}")
            else:
                depth = 0
            indent = "  " * depth
            for node in by_dir[folder]:
                name = node.path.rsplit("/", 1)[-1]
                badge = f"[{node.layer}|{node.verification_status}|{node.salience_tag}|v{node.version}]"
                title_part = f" — {node.title}" if node.title else ""
                lines.append(f"{indent}- {name} {badge}{title_part}")
        return "\n".join(lines)

    def grep(
        self,
        pattern: str,
        *,
        path_prefix: str = "/",
        layer: str | None = None,
        user_id: str = "default",
        agent_id: str = "default",
        max_matches: int = 20,
    ) -> list[dict]:
        if not pattern or len(pattern) > 256:
            raise ValueError("grep pattern must be 1..256 characters")
        regex = re.compile(pattern, re.IGNORECASE)
        nodes = self.store.list_nodes(user_id=user_id, agent_id=agent_id, parent_path=path_prefix, recursive=True, layer=layer)
        matches = []
        for node in nodes:
            hit_lines = [line.strip() for line in node.content.splitlines() if regex.search(line)]
            if hit_lines or regex.search(node.title):
                matches.append({
                    "path": node.path,
                    "title": node.title,
                    "layer": node.layer,
                    "verification_status": node.verification_status,
                    "version": node.version,
                    "matched_lines": hit_lines[:3],
                })
                if len(matches) >= max_matches:
                    break
        return matches

    def search(
        self,
        query: str,
        *,
        path_prefix: str | None = None,
        verified_only: bool = False,
        salience_tags: tuple[str, ...] | None = None,
        top_k: int = 10,
        user_id: str = "default",
        agent_id: str = "default",
    ) -> list[dict]:
        """L2-only hybrid vector + BM25 search."""
        return self.store.search_l2(
            query,
            user_id=user_id,
            agent_id=agent_id,
            path_prefix=path_prefix,
            verified_only=verified_only,
            salience_tags=salience_tags,
            top_k=top_k,
        )

    # ---- Snapshots, L3 Index & Dreaming Consolidation ----

    def history(
        self,
        path: str | None = None,
        *,
        dream_batch_id: str | None = None,
        limit: int = 20,
        user_id: str = "default",
        agent_id: str = "default",
    ) -> list[dict]:
        return self.store.list_snapshots(
            user_id=user_id, agent_id=agent_id, path=path, dream_batch_id=dream_batch_id, limit=limit
        )

    def rollback(
        self,
        path: str,
        *,
        snapshot_id: str | None = None,
        user_id: str = "default",
        agent_id: str = "default",
        timestamp: float | None = None,
    ) -> dict:
        node = self.store.rollback_node(
            user_id=user_id, agent_id=agent_id, path=path, snapshot_id=snapshot_id, timestamp=timestamp
        )
        return node.to_dict()

    def rollback_batch(
        self,
        dream_batch_id: str,
        *,
        user_id: str = "default",
        agent_id: str = "default",
        timestamp: float | None = None,
    ) -> list[dict]:
        nodes = self.store.rollback_batch(
            user_id=user_id, agent_id=agent_id, dream_batch_id=dream_batch_id, timestamp=timestamp
        )
        self.refresh_l3_index(user_id=user_id, agent_id=agent_id, timestamp=timestamp)
        return [node.to_dict() for node in nodes]

    def refresh_l3_index(
        self,
        *,
        user_id: str = "default",
        agent_id: str = "default",
        dream_batch_id: str | None = None,
        timestamp: float | None = None,
    ) -> dict:
        """Render deterministic byte-stable L3 /INDEX.md directory map + verified core rules."""
        l2_nodes = self.store.list_nodes(user_id=user_id, agent_id=agent_id, layer="L2")
        lines = ["# Memory Directory Index (L3)"]
        if l2_nodes:
            lines.append("## L2 Knowledge Tree")
            for node in l2_nodes:
                tag = f"[{node.verification_status}" + (f"|{node.salience_tag}]" if node.salience_tag != "normal" else "]")
                entry = f"- `{node.path}` {tag}: {node.title or node.summary}"
                if estimate_tokens("\n".join([*lines, entry])) > self.prefix_token_budget:
                    break
                lines.append(entry)
        else:
            lines.append("(no L2 memory notes yet)")
        content = "\n".join(lines)
        existing = self.store.get_node(_INDEX_PATH, user_id=user_id, agent_id=agent_id)
        if existing is not None and existing.content == content and existing.layer == "L3":
            return existing.to_dict()
        return self.write(
            _INDEX_PATH,
            title="Memory Directory Index",
            summary="Deterministic L3 tree manifest for prompt prefix caching",
            content=content,
            layer="L3",
            confidence_score=1.0,
            verification_status="verified",
            salience_tag="core_rule",
            dream_batch_id=dream_batch_id,
            reason="refresh_l3_index",
            user_id=user_id,
            agent_id=agent_id,
            timestamp=timestamp,
        )

    def dream_consolidate(
        self,
        *,
        user_id: str = "default",
        agent_id: str = "default",
        session_id: str | None = None,
        since_ts: float | None = None,
        synthesizer: Callable[[list[TranscriptRecord], list[MemoryNode], list[MemoryNode]], list[dict]] | None = None,
        timestamp: float | None = None,
    ) -> dict:
        """Offline/async 'dreaming' consolidation: L0 chunks + L1 working notes -> L2 Markdown notes.

        All writes share a single ``dream_batch_id`` with pre-state snapshots so any bad dreaming
        run can be reverted atomically via ``rollback_batch(dream_batch_id)``.
        """
        batch_id = f"dream_{uuid.uuid4().hex[:12]}"
        now = time.time() if timestamp is None else float(timestamp)
        l0_records = list(
            self.transcripts.iter_session(user_id=user_id, agent_id=agent_id, session_id=session_id, since_ts=since_ts)
        )
        l1_nodes = self.store.list_nodes(user_id=user_id, agent_id=agent_id, layer="L1")
        l2_nodes = self.store.list_nodes(user_id=user_id, agent_id=agent_id, layer="L2")
        synth = _default_dream_synthesizer if synthesizer is None else synthesizer
        proposals = synth(l0_records, l1_nodes, l2_nodes)
        updated_paths = []
        for item in proposals:
            written = self.write(
                item["path"],
                title=item.get("title") or item["path"].rsplit("/", 1)[-1],
                summary=item.get("summary") or "",
                content=item["content"],
                layer=item.get("layer", "L2"),
                confidence_score=float(item.get("confidence_score", 0.8)),
                verification_status=item.get("verification_status", "unverified"),
                salience_tag=item.get("salience_tag", "normal"),
                source_transcript_ids=item.get("source_transcript_ids") or (),
                dream_batch_id=batch_id,
                reason=f"dream:{batch_id}",
                user_id=user_id,
                agent_id=agent_id,
                timestamp=now,
            )
            updated_paths.append(written["path"])
        if updated_paths:
            self.refresh_l3_index(user_id=user_id, agent_id=agent_id, dream_batch_id=batch_id, timestamp=now)
        return {
            "dream_batch_id": batch_id,
            "l0_scanned": len(l0_records),
            "l1_scanned": len(l1_nodes),
            "updated_paths": updated_paths,
        }

    def build_context(
        self,
        query: str,
        *,
        user_id: str = "default",
        agent_id: str = "default",
        token_budget: int = 2048,
        path_prefix: str | None = None,
        verified_only: bool = False,
        include_l1_working: bool = True,
        top_k: int = 8,
    ) -> dict:
        """Assemble KV-cache-friendly L3 static prefix + token-bounded L2 hits + active L1 scratchpad."""
        index_node = self.store.get_node(_INDEX_PATH, user_id=user_id, agent_id=agent_id)
        prefix = index_node.content if index_node is not None and index_node.layer == "L3" else ""
        prefix_tokens = estimate_tokens(prefix)
        if prefix_tokens > token_budget:
            raise ValueError("token_budget cannot fit L3 system_prefix")
        prefix_version = hashlib.sha256(prefix.encode("utf-8")).hexdigest()[:12]
        remaining = token_budget - prefix_tokens

        hits = self.search(
            query,
            user_id=user_id,
            agent_id=agent_id,
            path_prefix=path_prefix,
            verified_only=verified_only,
            top_k=top_k,
        )
        sections: list[str] = []
        selected: list[dict] = []
        used = 0
        for item in hits:
            header = f"### {item['path']} [{item['verification_status']}|{item['salience_tag']}|v{item['version']}]"
            block = f"{header}\n{item['content'].strip()}"
            cost = estimate_tokens(block) + 1
            if used + cost > remaining:
                continue
            sections.append(block)
            selected.append(item)
            used += cost

        l1_selected: list[dict] = []
        if include_l1_working and remaining - used > 64:
            l1_budget = min(384, (remaining - used) // 3)
            l1_used = 0
            for note in self.store.list_nodes(user_id=user_id, agent_id=agent_id, layer="L1"):
                block = f"### [L1 Working] {note.path}\n{note.content.strip()}"
                cost = estimate_tokens(block) + 1
                if l1_used + cost > l1_budget:
                    break
                sections.append(block)
                l1_selected.append(note.to_dict())
                l1_used += cost
                used += cost

        dynamic_memory = "\n\n".join(sections)
        transcript_ids = list(dict.fromkeys(
            tid for item in (*selected, *l1_selected) for tid in item.get("source_transcript_ids", ())
        ))
        return {
            "system_prefix": prefix,
            "prefix_version": prefix_version,
            "dynamic_memory": dynamic_memory,
            "empty_recall": not selected and not l1_selected,
            "l2_hits": selected,
            "l1_working": l1_selected,
            "source_transcript_ids": transcript_ids,
            "token_breakdown": {
                "prefix_tokens": prefix_tokens,
                "dynamic_tokens": estimate_tokens(dynamic_memory),
                "total_tokens": prefix_tokens + estimate_tokens(dynamic_memory),
            },
            "l2_vector_count": self.store.vector_index_count(user_id=user_id, agent_id=agent_id),
        }

    def as_tools(self, *, user_id: str = "default", agent_id: str = "default"):
        from memengine.vfs.tools import VFSTools

        return VFSTools(self, user_id=user_id, agent_id=agent_id)
