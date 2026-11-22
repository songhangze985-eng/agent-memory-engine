"""High-level VFS Memory Engine v2: 3-stage retrieval, salience-gated structured writes, progressive disclosure & directory compaction."""
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
    MAX_INDEX_SUMMARY_CHARS,
    SALIENCE_WRITE_GUIDE,
    MemoryNode,
    OptimisticLockError,
    TranscriptRecord,
    enforce_rule_state_machine,
    evaluate_salience_gate,
    format_structured_rule,
    make_index_summary,
    normalize_dir_path,
    normalize_node_path,
    parse_structured_rule,
)
from memengine.vfs.store import SQLiteVFSStore, open_vfs_store
from memengine.vfs.transcripts import LocalChunkTranscriptStore, TranscriptStore

_INDEX_PATH = "/INDEX.md"
_GLOBAL_PATH = "/global.md"
DEFAULT_MAX_NOTES_PER_DIR = 10
DEFAULT_MAX_DIR_TOKENS = 3000


def _default_dream_synthesizer(
    transcripts: list[TranscriptRecord],
    l1_nodes: list[MemoryNode],
    existing_l2: list[MemoryNode],
) -> list[dict]:
    """Deterministic fallback synthesizer promoting verified L1 notes and explicit constraints/preferences."""
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
            "summary": make_index_summary("Taboos & Hard Constraints", "\n".join(body_lines)),
            "content": "\n".join(body_lines),
            "layer": "L2",
            "confidence_score": 0.95,
            "verification_status": "verified",
            "salience_tag": "taboo",
            "always_active": True,
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
            "summary": make_index_summary("User Preferences", "\n".join(body_lines)),
            "content": "\n".join(body_lines),
            "layer": "L2",
            "confidence_score": 0.88,
            "verification_status": "verified",
            "salience_tag": "preference",
            "source_transcript_ids": sources,
        })

    for note in l1_nodes:
        if note.path == _GLOBAL_PATH:
            continue
        if note.verification_status == "verified":
            rel = note.path.removeprefix("/working/") if note.path.startswith("/working/") else note.path.lstrip("/")
            target = "/strategies/" + rel
            proposals.append({
                "path": target,
                "title": note.title,
                "summary": make_index_summary(note.title, note.content, note.summary),
                "content": note.content,
                "layer": "L2",
                "confidence_score": max(0.85, note.confidence_score),
                "verification_status": "verified",
                "salience_tag": note.salience_tag if note.salience_tag != "normal" else "core_rule",
                "always_active": note.always_active,
                "source_transcript_ids": list(note.source_transcript_ids),
            })
    return proposals


def _default_directory_refactorer(dir_path: str, notes: list[MemoryNode]) -> dict:
    """Deterministic high-density directory merger: merges fragmented notes into one standard guide."""
    verified = [n for n in notes if n.verification_status == "verified"]
    unverified = [n for n in notes if n.verification_status != "verified"]
    verified.sort(key=lambda n: (-int(n.always_active), -n.confidence_score, -n.updated_at, n.path))
    unverified.sort(key=lambda n: (-n.confidence_score, -n.updated_at, n.path))

    seen_rules: set[str] = set()
    sections = [f"# Standard Guide: {dir_path}"]

    def _emit_group(heading: str, group: list[MemoryNode]) -> None:
        bullets = []
        for note in group:
            parsed = parse_structured_rule(note.content)
            if parsed["trigger_condition"] and parsed["action"]:
                sig = f"{parsed['trigger_condition'].casefold()}|{parsed['action'].casefold()}"
                if sig in seen_rules:
                    continue
                seen_rules.add(sig)
                bullets.append(
                    f"### {note.title or note.path}\n"
                    f"- [触发条件] {parsed['trigger_condition']}\n"
                    f"- [动作] {parsed['action']}\n"
                    f"- [验证结果] {parsed['verification_result'] or '已整合'}"
                )
            else:
                body = " ".join(line.strip() for line in note.content.splitlines() if line.strip() and not line.strip().startswith("#"))
                sig = body.casefold()
                if not body or sig in seen_rules:
                    continue
                seen_rules.add(sig)
                bullets.append(f"### {note.title or note.path}\n{body}")
        if bullets:
            sections.append(f"## {heading}\n\n" + "\n\n".join(bullets))

    _emit_group("Verified Rules", verified)
    _emit_group("Unverified Hypotheses", unverified)

    all_sources = list(dict.fromkeys(tid for n in notes for tid in n.source_transcript_ids))
    any_always = any(n.always_active for n in notes)
    top_salience = next(
        (tag for tag in ("taboo", "core_rule", "warning", "preference") if any(n.salience_tag == tag for n in notes)),
        "core_rule" if verified else "normal",
    )
    content = "\n\n".join(sections)
    title = f"Standard Guide ({dir_path.strip('/') or 'root'})"
    summary = make_index_summary(
        title,
        content,
        f"{dir_path.strip('/') or 'root'} 标准指南（整合 {len(notes)} 条笔记）",
    )
    return {
        "path": f"{dir_path}GUIDE.md",
        "title": title,
        "summary": summary,
        "content": content,
        "layer": "L2",
        "confidence_score": max((n.confidence_score for n in verified), default=0.75),
        "verification_status": "verified" if verified else "unverified",
        "salience_tag": top_salience,
        "always_active": any_always,
        "source_transcript_ids": all_sources,
    }


class VFSMemoryEngine:
    """Claude-style Virtual Filesystem Memory Engine v2.

    Three-Stage Hybrid Retrieval Pipeline:
      1. **Deterministic Pre-load**: Unconditionally loads L1 ``/global.md`` + L2 ``always_active`` rules.
      2. **Agentic Tree Search**: Exposes compact ``ls -R`` outline (path + <=50-char ``index_summary``) + ``read_note(path)``.
      3. **Hybrid Semantic + Metadata RAG**: Dense + BM25 + L2 exact-identifier/version precision with
         ``Final Score = Semantic_Sim * W_verified * W_recency``.

    Write & Read Density Optimization:
      - Write-side salience self-check (3 questions) + ``[触发条件] + [动作] + [验证结果]`` hypothesis/rule state machine.
      - Read-side **Progressive Disclosure** (``index_summary`` <= 50 chars first; ``read_note(path)`` for full ``detail_body``; zero mid-rule hard truncation).
      - Dynamic directory compaction/refactor when a directory exceeds ``max_notes_per_dir`` (10) or ``max_dir_tokens`` (3,000).
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
        max_notes_per_dir: int = DEFAULT_MAX_NOTES_PER_DIR,
        max_dir_tokens: int = DEFAULT_MAX_DIR_TOKENS,
    ):
        self.store = open_vfs_store(db_url, embedder=embedder, external_vector_index=external_vector_index)
        self._temp_dir = None
        if transcript_store is not None:
            self.transcripts = transcript_store
        else:
            if transcript_dir is None:
                from urllib.parse import unquote

                raw_url = str(db_url or "")
                if raw_url and not raw_url.startswith(("postgresql://", "postgres://", ":memory:")):
                    cleaned = raw_url
                    for prefix in ("lancedb:///", "lance:///", "sqlite+lance:///", "lancedb://", "lance://", "sqlite+lance://", "sqlite:///"):
                        if cleaned.startswith(prefix):
                            cleaned = unquote(cleaned.removeprefix(prefix))
                            break
                    base = Path(cleaned).expanduser().resolve()
                    transcript_dir = (base / "l0_transcripts") if base.suffix not in {".db", ".sqlite", ".sqlite3"} else (base.parent / (base.stem + "_l0_transcripts"))
                else:
                    self._temp_dir = tempfile.TemporaryDirectory(prefix="memengine_l0_")
                    transcript_dir = Path(self._temp_dir.name)
            self.transcripts = LocalChunkTranscriptStore(transcript_dir)
        self.prefix_token_budget = int(prefix_token_budget)
        self.max_notes_per_dir = max(2, int(max_notes_per_dir))
        self.max_dir_tokens = max(256, int(max_dir_tokens))

    def optimize_index(self) -> dict:
        return self.store.optimize_index()

    def close(self) -> None:
        try:
            self.transcripts.flush()
        finally:
            self.store.close()
            if self._temp_dir is not None:
                self._temp_dir.cleanup()
                self._temp_dir = None

    @staticmethod
    def salience_guide() -> str:
        return SALIENCE_WRITE_GUIDE

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

    # ---- Write-Side: Salience Gate + Structured Rule State Machine + Raw VFS Write ----

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
        always_active: bool = False,
        occurrence_start: float | None = None,
        occurrence_end: float | None = None,
        attribute_slot: str = "",
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
            always_active=always_active,
            occurrence_start=occurrence_start,
            occurrence_end=occurrence_end,
            attribute_slot=attribute_slot,
            source_transcript_ids=source_transcript_ids,
            expected_version=expected_version,
            dream_batch_id=dream_batch_id,
            reason=reason,
            timestamp=timestamp,
        )
        return node.to_dict()

    def write_rule(
        self,
        path: str,
        *,
        title: str,
        trigger_condition: str,
        action: str,
        verification_result: str = "",
        notes: str = "",
        index_summary: str = "",
        layer: str = "L2",
        confidence_score: float = 0.85,
        verification_status: str = "verified",
        salience_tag: str = "core_rule",
        always_active: bool = False,
        visible_in_context: bool = False,
        reusable_pattern: bool = True,
        future_pitfall_risk: bool = True,
        source_transcript_ids: list[str] | tuple[str, ...] = (),
        expected_version: int | None = None,
        user_id: str = "default",
        agent_id: str = "default",
        timestamp: float | None = None,
    ) -> dict:
        """Salience-gated structured write enforcing [触发条件] + [动作] + [验证结果] and hypothesis/rule state machine."""
        allowed, gate_reason = evaluate_salience_gate(
            visible_in_context=visible_in_context,
            reusable_pattern=reusable_pattern,
            future_pitfall_risk=future_pitfall_risk,
        )
        if not allowed:
            return {"written": False, "skipped_reason": gate_reason, "path": path}

        content = format_structured_rule(
            title=title,
            trigger_condition=trigger_condition,
            action=action,
            verification_result=verification_result,
            notes=notes,
        )
        enforced_status, enforced_conf = enforce_rule_state_machine(
            verification_status=verification_status,
            confidence_score=confidence_score,
            verification_result=verification_result,
        )
        concise_summary = make_index_summary(
            title, content, index_summary or action, max_chars=MAX_INDEX_SUMMARY_CHARS
        )
        written = self.write(
            path,
            title=title,
            summary=concise_summary,
            content=content,
            layer=layer,
            confidence_score=enforced_conf,
            verification_status=enforced_status,
            salience_tag=salience_tag,
            always_active=always_active,
            source_transcript_ids=source_transcript_ids,
            expected_version=expected_version,
            reason="write_rule",
            user_id=user_id,
            agent_id=agent_id,
            timestamp=timestamp,
        )
        compaction = self.check_compaction_needed(
            parent_path=written["parent_path"], user_id=user_id, agent_id=agent_id
        )
        return {**written, "written": True, "skipped_reason": None, "compaction_needed": bool(compaction)}

    # ---- Read-Side: Progressive Disclosure & Agentic Navigation ----

    def read_note(self, path: str, *, user_id: str = "default", agent_id: str = "default") -> dict | None:
        """Read complete, untruncated detail_body and metadata of a Markdown memory node."""
        node = self.store.get_node(path, user_id=user_id, agent_id=agent_id)
        return node.to_dict() if node is not None else None

    def read(self, path: str, *, user_id: str = "default", agent_id: str = "default") -> dict | None:
        return self.read_note(path, user_id=user_id, agent_id=agent_id)

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
        always_active: bool | None = None,
        summary: str | None = None,
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
        if expected_version is not None and existing.version != expected_version:
            raise OptimisticLockError(existing.path, expected_version, existing.version)
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
            summary=existing.summary if summary is None else summary,
            content=new_content,
            layer=existing.layer,
            confidence_score=existing.confidence_score if confidence_score is None else confidence_score,
            verification_status=existing.verification_status if verification_status is None else verification_status,
            salience_tag=existing.salience_tag if salience_tag is None else salience_tag,
            always_active=existing.always_active if always_active is None else always_active,
            occurrence_start=existing.occurrence_start,
            occurrence_end=existing.occurrence_end,
            attribute_slot=existing.attribute_slot,
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
                    "index_summary": node.summary,
                    "layer": node.layer,
                    "verification_status": node.verification_status,
                    "salience_tag": node.salience_tag,
                    "always_active": node.always_active,
                    "confidence_score": node.confidence_score,
                    "version": node.version,
                })
            elif node.parent_path.startswith(norm_dir):
                rest = node.parent_path[len(norm_dir):].strip("/")
                if rest:
                    first = rest.split("/", 1)[0]
                    subdirs.add(norm_dir + first + "/")
        return {"path": norm_dir, "directories": sorted(subdirs), "files": files}

    def tree_outline(
        self,
        path: str = "/",
        *,
        user_id: str = "default",
        agent_id: str = "default",
        layer: str | None = "L2",
    ) -> str:
        """Stage 2 lightweight `ls -R` outline: ONLY path, concise <=50-char index_summary, and state badges."""
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
                flags = [node.layer, node.verification_status]
                if node.always_active:
                    flags.append("always_active")
                elif node.salience_tag != "normal":
                    flags.append(node.salience_tag)
                flags.append(f"v{node.version}")
                brief = node.summary or node.title
                lines.append(f"{indent}- {name} [{'|'.join(flags)}] — {brief}")
        return "\n".join(lines)

    def tree(self, path: str = "/", *, user_id: str = "default", agent_id: str = "default", layer: str | None = None) -> str:
        return self.tree_outline(path=path, user_id=user_id, agent_id=agent_id, layer=layer)

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
                    "index_summary": node.summary,
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
        reference_time: float | str | None = None,
        top_k: int = 10,
        user_id: str = "default",
        agent_id: str = "default",
    ) -> list[dict]:
        """Stage 3 L2-only Hybrid Dense + BM25 + Exact-Identifier/Version search."""
        return self.store.search_l2(
            query,
            user_id=user_id,
            agent_id=agent_id,
            path_prefix=path_prefix,
            verified_only=verified_only,
            salience_tags=salience_tags,
            reference_time=reference_time,
            top_k=top_k,
        )

    # ---- Dynamic Directory Compaction / Refactor ----

    def check_compaction_needed(
        self,
        *,
        parent_path: str | None = None,
        user_id: str = "default",
        agent_id: str = "default",
    ) -> list[dict]:
        """Return directories exceeding max_notes_per_dir (default 10) or max_dir_tokens (default 3000)."""
        stats = self.store.directory_stats(
            user_id=user_id, agent_id=agent_id, layer="L2", parent_path=parent_path
        )
        flagged = []
        for row in stats:
            reasons = []
            if row["note_count"] > self.max_notes_per_dir:
                reasons.append(f"notes={row['note_count']}>{self.max_notes_per_dir}")
            if row["tokens"] > self.max_dir_tokens:
                reasons.append(f"tokens={row['tokens']}>{self.max_dir_tokens}")
            if reasons:
                flagged.append({**row, "reasons": reasons})
        return flagged

    def compact_directory(
        self,
        dir_path: str,
        *,
        refactorer: Callable[[str, list[MemoryNode]], dict] | None = None,
        user_id: str = "default",
        agent_id: str = "default",
        timestamp: float | None = None,
    ) -> dict:
        """Merge scattered notes in one directory into a unified standard guide (atomic & rollback-safe)."""
        norm_dir = normalize_dir_path(dir_path)
        notes = self.store.list_nodes(
            user_id=user_id, agent_id=agent_id, parent_path=norm_dir, recursive=False, layer="L2"
        )
        if len(notes) <= 1:
            return {"compacted": False, "dir_path": norm_dir, "dream_batch_id": None, "merged_count": 0}
        batch_id = f"compact_{uuid.uuid4().hex[:12]}"
        now = time.time() if timestamp is None else float(timestamp)
        merged = (_default_directory_refactorer if refactorer is None else refactorer)(norm_dir, notes)
        guide_path = normalize_node_path(merged["path"])[1]
        guide = self.write(
            guide_path,
            title=merged["title"],
            summary=merged.get("summary", ""),
            content=merged["content"],
            layer="L2",
            confidence_score=float(merged.get("confidence_score", 0.85)),
            verification_status=merged.get("verification_status", "verified"),
            salience_tag=merged.get("salience_tag", "core_rule"),
            always_active=bool(merged.get("always_active", False)),
            source_transcript_ids=merged.get("source_transcript_ids") or (),
            dream_batch_id=batch_id,
            reason=f"compact:{norm_dir}",
            user_id=user_id,
            agent_id=agent_id,
            timestamp=now,
        )
        archived_paths = []
        for note in notes:
            if note.path == guide["path"]:
                continue
            if self.delete(
                note.path,
                dream_batch_id=batch_id,
                reason=f"compact_archive:{norm_dir}",
                user_id=user_id,
                agent_id=agent_id,
                timestamp=now,
            ):
                archived_paths.append(note.path)
        self.refresh_l3_index(user_id=user_id, agent_id=agent_id, dream_batch_id=batch_id, timestamp=now)
        return {
            "compacted": True,
            "dir_path": norm_dir,
            "dream_batch_id": batch_id,
            "guide_path": guide["path"],
            "merged_count": len(notes),
            "archived_paths": archived_paths,
        }

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
        """Render deterministic byte-stable L3 /INDEX.md directory outline using <=50-char index_summary."""
        l2_nodes = self.store.list_nodes(user_id=user_id, agent_id=agent_id, layer="L2")
        lines = ["# Memory Directory Index (L3)"]
        if l2_nodes:
            lines.append("## L2 Directory Outline (`ls -R` — call `read_note(path)` for full details)")
            for node in l2_nodes:
                flags = [node.verification_status]
                if node.always_active:
                    flags.append("always_active")
                elif node.salience_tag != "normal":
                    flags.append(node.salience_tag)
                entry = f"- `{node.path}` [{'|'.join(flags)}]: {node.summary or node.title}"
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
            summary="Deterministic L3 outline for prompt prefix cache",
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
        auto_compact: bool = False,
        timestamp: float | None = None,
    ) -> dict:
        """Offline/async dreaming consolidation with atomic batch rollback + optional directory compaction."""
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
                always_active=bool(item.get("always_active", False)),
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
        compacted_dirs = []
        if auto_compact:
            for alert in self.check_compaction_needed(user_id=user_id, agent_id=agent_id):
                res = self.compact_directory(alert["parent_path"], user_id=user_id, agent_id=agent_id, timestamp=now)
                if res["compacted"]:
                    compacted_dirs.append(res)
        return {
            "dream_batch_id": batch_id,
            "l0_scanned": len(l0_records),
            "l1_scanned": len(l1_nodes),
            "updated_paths": updated_paths,
            "compacted_dirs": compacted_dirs,
        }

    # ---- Three-Stage Context Assembly with Progressive Disclosure (ZERO mid-rule truncation!) ----

    def build_context(
        self,
        query: str,
        *,
        user_id: str = "default",
        agent_id: str = "default",
        token_budget: int = 2048,
        path_prefix: str | None = None,
        verified_only: bool = False,
        reference_time: float | str | None = None,
        disclosure: str = "progressive",
        include_tree_outline: bool = True,
        include_l1_working: bool = True,
        top_k: int = 8,
    ) -> dict:
        """Assemble 3-stage context using Progressive Disclosure instead of crude mid-rule hard truncation.

        - ``disclosure="progressive"`` (default): Stage 3 hits inject ONLY concise ``<=50``-char
          ``index_summary`` cards so the Agent pulls intact ``detail_body`` on demand via ``read_note(path)``.
        - ``disclosure="full"``: Injects whole intact ``detail_body`` blocks when they fit; if a full note
          exceeds remaining budget, gracefully degrades that whole note to its ``index_summary`` rather than
          slicing a rule mid-sentence.
        """
        if disclosure not in {"progressive", "full"}:
            raise ValueError("disclosure must be 'progressive' or 'full'")

        index_node = self.store.get_node(_INDEX_PATH, user_id=user_id, agent_id=agent_id)
        prefix = index_node.content if index_node is not None and index_node.layer == "L3" else ""
        prefix_tokens = estimate_tokens(prefix)
        if prefix_tokens > token_budget:
            raise ValueError("token_budget cannot fit L3 system_prefix")
        prefix_version = hashlib.sha256(prefix.encode("utf-8")).hexdigest()[:12]
        remaining = token_budget - prefix_tokens
        sections: list[str] = []
        used = 0
        preloaded_paths: set[str] = set()

        def _try_add(block: str) -> bool:
            nonlocal used
            cost = estimate_tokens(block) + 2
            if used + cost > remaining:
                return False
            sections.append(block)
            used += cost
            return True

        # Stage 1: Deterministic Pre-load (L1 /global.md + L2 always_active rules)
        global_node, always_active_nodes = self.store.get_preloaded_nodes(user_id=user_id, agent_id=agent_id)
        preloaded_items: list[dict] = []
        stage1_lines = []
        if global_node is not None:
            preloaded_items.append(global_node.to_dict())
            preloaded_paths.add(global_node.path)
            stage1_lines.append(f"### [Global L1] {global_node.path}\n{global_node.content.strip()}")
        for node in always_active_nodes:
            preloaded_items.append(node.to_dict())
            preloaded_paths.add(node.path)
            full_entry = f"### [Always Active L2|{node.verification_status}] {node.path}\n{node.content.strip()}"
            if estimate_tokens("\n\n".join([*stage1_lines, full_entry])) <= max(128, remaining // 2):
                stage1_lines.append(full_entry)
            else:
                stage1_lines.append(
                    f"- `{node.path}` [always_active|{node.verification_status}]: {node.summary} (call `read_note(\"{node.path}\")`)"
                )
        if stage1_lines:
            _try_add("## Stage 1: Deterministic Pre-load\n" + "\n\n".join(stage1_lines))

        # Stage 2: Agentic Tree Search Outline (`ls -R` lightweight directory map)
        outline_text = ""
        if include_tree_outline:
            outline_text = self.tree_outline(path=path_prefix or "/", user_id=user_id, agent_id=agent_id, layer="L2")
            if outline_text and not outline_text.endswith("(empty)"):
                block = (
                    "## Stage 2: L2 Directory Outline (`ls -R` — call `read_note(path)` for full `detail_body`)\n"
                    + outline_text
                )
                _try_add(block)

        # Stage 3: Hybrid Dense + BM25 + Exact-Identifier RAG with Progressive Disclosure
        hits = self.search(
            query,
            user_id=user_id,
            agent_id=agent_id,
            path_prefix=path_prefix,
            verified_only=verified_only,
            reference_time=reference_time,
            top_k=top_k,
        )
        selected: list[dict] = []
        stage3_blocks: list[str] = []
        for item in hits:
            if item["path"] in preloaded_paths:
                selected.append(item)
                continue
            badge = f"{item['verification_status']}|{item['salience_tag']}|score={item['score']:.3f}|v{item['version']}"
            summary_card = f"- `{item['path']}` [{badge}]: {item['index_summary']} → `read_note(\"{item['path']}\")`"
            if disclosure == "full":
                full_block = f"### {item['path']} [{badge}]\n{item['detail_body'].strip()}"
                if used + estimate_tokens("\n".join([*stage3_blocks, full_block])) + 8 <= remaining:
                    stage3_blocks.append(full_block)
                    selected.append({**item, "disclosed": "detail_body"})
                    continue
                # Graceful whole-note degradation to index_summary — NEVER cut mid-rule!
            if used + estimate_tokens("\n".join([*stage3_blocks, summary_card])) + 8 <= remaining:
                stage3_blocks.append(summary_card)
                selected.append({**item, "disclosed": "index_summary"})
        if stage3_blocks:
            _try_add("## Stage 3: Hybrid L2 Recall (Progressive Disclosure)\n" + "\n".join(stage3_blocks))

        # Optional L1 working notes (excluding /global.md already in Stage 1)
        l1_selected: list[dict] = []
        if include_l1_working and remaining - used > 64:
            l1_budget = min(384, (remaining - used) // 3)
            l1_used = 0
            l1_blocks = []
            for note in self.store.list_nodes(user_id=user_id, agent_id=agent_id, layer="L1"):
                if note.path == _GLOBAL_PATH:
                    continue
                block = f"### [L1 Working] {note.path}\n{note.content.strip()}"
                cost = estimate_tokens(block) + 1
                if l1_used + cost > l1_budget:
                    break
                l1_blocks.append(block)
                l1_selected.append(note.to_dict())
                l1_used += cost
            if l1_blocks:
                _try_add("## Active L1 Working Notes\n" + "\n\n".join(l1_blocks))

        dynamic_memory = "\n\n".join(sections)
        transcript_ids = list(dict.fromkeys(
            tid for item in (*preloaded_items, *selected, *l1_selected) for tid in item.get("source_transcript_ids", ())
        ))
        compaction_alerts = self.check_compaction_needed(user_id=user_id, agent_id=agent_id)
        return {
            "system_prefix": prefix,
            "prefix_version": prefix_version,
            "dynamic_memory": dynamic_memory,
            "empty_recall": not preloaded_items and not selected and not l1_selected,
            "disclosure": disclosure,
            "preloaded": preloaded_items,
            "directory_outline": outline_text,
            "l2_hits": selected,
            "l1_working": l1_selected,
            "source_transcript_ids": transcript_ids,
            "compaction_alerts": compaction_alerts,
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
