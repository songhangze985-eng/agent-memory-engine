"""C2: Signal-triggered memory extraction (fail-to-pass, user corrections, drift) + G4 write cost cap."""
from __future__ import annotations

import hashlib
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from agentmem_core.core.code_anchors import CodeAnchor, normalize_rel_path
from agentmem_core.core.profiles import MemoryProfile, load_profile
from agentmem_core.core.scopes import normalize_override_key, resolve_write_scope

EventType = Literal["tool_result", "user_message", "assistant_message", "file_edit", "git_commit", "test_result"]

_ERROR_LINE_RE = re.compile(
    r"(?:Error|Exception|Traceback|FAILED|ModuleNotFoundError|ImportError|AttributeError|"
    r"TypeError|SyntaxError|PermissionError|ConnectionRefusedError|OperationalError|"
    r"AssertionError|ENOENT|EACCES|ERR_|exit\s+code\s+[1-9]|报错|失败|异常)",
    re.I,
)
_USER_CORRECTION_RE = re.compile(
    r"(?:^\s*(?:no|nope|actually|wait|stop|wrong)\b[,:\s]+|"
    r"\b(?:don't\s+(?:use|run|commit|edit|do|call)|do\s+not\s+(?:use|run|commit|edit|do|call)|"
    r"never\s+(?:use|run|commit|push|modify|edit|do|call|hardcode)|stop\s+using|"
    r"use\s+.+\s+instead\s+of|always\s+(?:use|run|check|keep|set|prefer|enable|disable)|"
    r"remember\s+to\s+(?:always\s+)?|please\s+always|we\s+switched\s+to)\b|"
    r"不对|错了|不要用|别用|不要提交|改用|以后都|记住要|必须用|统一用)",
    re.I,
)
_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class AgentEvent:
    """Normalized agent runtime event consumed by the C2 signal extraction pipeline."""

    event_type: EventType
    text: str = ""
    tool_name: str = ""
    command: str = ""
    exit_code: int | None = None
    error_signature: str = ""
    file_paths: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    override_key: str = ""
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def coerce(cls, raw: "AgentEvent | dict[str, Any]") -> "AgentEvent":
        if isinstance(raw, AgentEvent):
            return raw
        files = raw.get("file_paths") or raw.get("files") or ()
        if isinstance(files, str):
            files = (files,)
        syms = raw.get("symbols") or ()
        if isinstance(syms, str):
            syms = (syms,)
        return cls(
            event_type=raw.get("event_type", "tool_result"),
            text=str(raw.get("text") or ""),
            tool_name=str(raw.get("tool_name") or ""),
            command=str(raw.get("command") or ""),
            exit_code=raw.get("exit_code"),
            error_signature=str(raw.get("error_signature") or ""),
            file_paths=tuple(normalize_rel_path(p) for p in files if str(p).strip()),
            symbols=tuple(str(s).strip() for s in syms if str(s).strip()),
            override_key=str(raw.get("override_key") or ""),
            timestamp=raw.get("timestamp"),
            metadata=dict(raw.get("metadata") or {}),
        )


def _extract_error_signature(event: AgentEvent) -> str:
    if event.error_signature.strip():
        return event.error_signature.strip()[:160]
    for line in (event.text or "").splitlines():
        clean = line.strip()
        if clean and _ERROR_LINE_RE.search(clean):
            return clean[:160]
    if event.exit_code not in (None, 0):
        cmd = event.command or event.tool_name or "command"
        return f"{cmd} exited with code {event.exit_code}"[:160]
    return ""


def _command_family(command: str, tool_name: str) -> str:
    tokens = (command or "").strip().split()
    if not tokens:
        return (tool_name or "tool").casefold()
    # Skip env assignments like FOO=1 or runners like uv run / npx / poetry run
    idx = 0
    while idx < len(tokens) and "=" in tokens[idx] and not tokens[idx].startswith("-"):
        idx += 1
    if idx < len(tokens) and tokens[idx] in {"uv", "poetry", "npx", "python", "python3"}:
        if idx + 1 < len(tokens) and tokens[idx + 1] in {"run", "-m"} and idx + 2 < len(tokens):
            return tokens[idx + 2].casefold()
    return tokens[min(idx, len(tokens) - 1)].casefold()


def _slug(text: str, *, max_len: int = 42) -> str:
    s = _SLUG_RE.sub("_", (text or "").casefold()).strip("_")
    return (s[:max_len].strip("_")) or "signal"


class WriteCostLimiter:
    """G4 write-cost upper bound enforcing bounded writes/window and near-duplicate suppression."""

    def __init__(self, *, max_writes_per_window: int = 32, window_seconds: float = 3600.0):
        self.max_writes_per_window = max(1, int(max_writes_per_window))
        self.window_seconds = max(1.0, float(window_seconds))
        self._buckets: dict[tuple[str, str, str], deque[float]] = {}
        self._seen_hashes: dict[tuple[str, str, str], set[str]] = {}

    def allow(
        self,
        *,
        user_id: str,
        agent_id: str,
        scope_key: str,
        fingerprint: str,
        now: float,
        max_writes: int | None = None,
        window_seconds: float | None = None,
    ) -> tuple[bool, str, int]:
        limit = max(1, int(max_writes if max_writes is not None else self.max_writes_per_window))
        win = max(1.0, float(window_seconds if window_seconds is not None else self.window_seconds))
        key = (user_id, agent_id, scope_key)
        bucket = self._buckets.setdefault(key, deque())
        seen = self._seen_hashes.setdefault(key, set())
        cutoff = now - win
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if not bucket:
            seen.clear()
        if fingerprint in seen:
            return False, "duplicate_signal", len(bucket)
        if len(bucket) >= limit:
            return False, "write_budget_exceeded", len(bucket)
        return True, "allowed", len(bucket)

    def record(self, *, user_id: str, agent_id: str, scope_key: str, fingerprint: str, now: float) -> int:
        key = (user_id, agent_id, scope_key)
        bucket = self._buckets.setdefault(key, deque())
        seen = self._seen_hashes.setdefault(key, set())
        bucket.append(now)
        seen.add(fingerprint)
        return len(bucket)


class SignalExtractor:
    """Detects high-signal events (fail->pass, user corrections, code drift) and persists bounded memories."""

    def __init__(self, engine, *, max_pending_per_session: int = 32):
        self.engine = engine
        self.max_pending = max(4, int(max_pending_per_session))
        self._pending_failures: dict[tuple[str, str, str], deque[dict[str, Any]]] = {}
        self._recent_edits: dict[tuple[str, str, str], deque[dict[str, Any]]] = {}
        self.limiter = WriteCostLimiter()

    def process_event(
        self,
        event: AgentEvent | dict[str, Any],
        *,
        user_id: str = "default",
        agent_id: str = "default",
        session_id: str = "default",
        scope_key: str | None = None,
        repo_id: str | None = None,
        project_id: str | None = None,
        repo_root: str | Path | None = None,
        profile: str | Path | MemoryProfile | None = None,
    ) -> dict[str, Any]:
        ev = AgentEvent.coerce(event)
        spec = load_profile(profile) if profile is not None else self.engine.default_profile
        now = time.time() if ev.timestamp is None else float(ev.timestamp)
        s_key = (user_id, agent_id, session_id)
        eff_scope = resolve_write_scope(
            explicit_scope_key=scope_key,
            default_level=spec.default_write_scope_level,
            user_id=user_id,
            session_id=session_id,
            repo_id=repo_id,
            project_id=project_id,
        )

        # 1. File edit / git commit -> record edit context & trigger C1 anchor staleness check
        if ev.event_type in ("file_edit", "git_commit"):
            edits = self._recent_edits.setdefault(s_key, deque(maxlen=self.max_pending))
            edits.append(
                {
                    "file_paths": ev.file_paths,
                    "symbols": ev.symbols,
                    "text": ev.text,
                    "timestamp": now,
                }
            )
            drift_res = None
            if ev.file_paths or repo_root is not None:
                drift_res = self.engine.verify_anchors(
                    repo_root=repo_root,
                    changed_files=ev.file_paths,
                    user_id=user_id,
                    agent_id=agent_id,
                    timestamp=now,
                )
            return {
                "signal": "file_drift" if (drift_res and drift_res["stale_count"] > 0) else None,
                "written": False,
                "fact_id": None,
                "stale_check": drift_res,
                "writes_in_window": 0,
            }

        # 2. User explicit correction / constraint / durable preference
        if ev.event_type == "user_message":
            if not _USER_CORRECTION_RE.search(ev.text or ""):
                return {"signal": None, "written": False, "reason": "no_correction_signal", "fact_id": None}

            fp = hashlib.sha256(f"corr:{eff_scope}:{ev.text.strip().casefold()}".encode("utf-8")).hexdigest()[:20]
            ok, reason, count = self.limiter.allow(
                user_id=user_id,
                agent_id=agent_id,
                scope_key=eff_scope,
                fingerprint=fp,
                now=now,
                max_writes=spec.max_writes_per_window,
                window_seconds=spec.write_window_seconds,
            )
            if not ok:
                return {"signal": "user_correction", "written": False, "reason": reason, "writes_in_window": count, "fact_id": None}

            slug = _slug(ev.override_key or ev.text[:48])
            path = f"/corrections/{slug}.md"
            anchors = self._build_anchors(ev.file_paths, ev.symbols)
            rule_res = self.engine.ingest_rule(
                path,
                title=f"User Correction: {ev.text.strip()[:48]}",
                trigger_condition=f"When handling task related to {ev.override_key or slug.replace('_', ' ')}",
                action=ev.text.strip(),
                verification_result="Explicitly instructed/corrected by user",
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                scope_key=eff_scope,
                override_key=normalize_override_key(ev.override_key or slug),
                salience_tag="preference",
                always_active=False,
                anchors=anchors,
                repo_root=repo_root,
                timestamp=now,
            )
            new_count = self.limiter.record(user_id=user_id, agent_id=agent_id, scope_key=eff_scope, fingerprint=fp, now=now)
            return {
                "signal": "user_correction",
                "written": bool(rule_res.get("accepted")),
                "fact_id": rule_res.get("fact_id"),
                "path": rule_res.get("path"),
                "scope_key": eff_scope,
                "writes_in_window": new_count,
            }

        # 3. Tool / test result: fail -> pass detection
        if ev.event_type in ("tool_result", "test_result"):
            err_sig = _extract_error_signature(ev)
            is_failure = (ev.exit_code is not None and ev.exit_code != 0) or (ev.exit_code is None and bool(err_sig))
            fam = _command_family(ev.command, ev.tool_name)
            pending = self._pending_failures.setdefault(s_key, deque(maxlen=self.max_pending))

            if is_failure:
                pending.append(
                    {
                        "family": fam,
                        "command": ev.command,
                        "tool_name": ev.tool_name,
                        "error_signature": err_sig or f"{fam} failed",
                        "text": ev.text[:500],
                        "file_paths": ev.file_paths,
                        "symbols": ev.symbols,
                        "timestamp": now,
                    }
                )
                return {
                    "signal": "failure_buffered",
                    "written": False,
                    "pending_failures": len(pending),
                    "error_signature": err_sig,
                    "fact_id": None,
                }

            # Passing command: check if it resolves a buffered failure!
            is_success = ev.exit_code == 0 or (ev.exit_code is None and not err_sig)
            if not is_success or not pending:
                return {"signal": None, "written": False, "reason": "no_matching_failure", "fact_id": None}

            matched_idx: int | None = None
            ev_files = set(ev.file_paths)
            for idx in range(len(pending) - 1, -1, -1):
                cand = pending[idx]
                if now - cand["timestamp"] > 1800.0:
                    continue
                same_family = cand["family"] == fam or (cand["tool_name"] and cand["tool_name"] == ev.tool_name)
                shared_files = bool(ev_files and set(cand["file_paths"]) & ev_files)
                if same_family or shared_files:
                    matched_idx = idx
                    break

            if matched_idx is None:
                return {"signal": None, "written": False, "reason": "unrelated_pass", "fact_id": None}

            failure = pending[matched_idx]
            del pending[matched_idx]

            # Gather files/edits touched between failure and passing verification
            edits = [
                e for e in self._recent_edits.get(s_key, ())
                if failure["timestamp"] <= e["timestamp"] <= now
            ]
            all_files = list(dict.fromkeys([*failure["file_paths"], *(p for e in edits for p in e["file_paths"]), *ev.file_paths]))
            all_syms = list(dict.fromkeys([*failure["symbols"], *(s for e in edits for s in e["symbols"]), *ev.symbols]))
            edit_notes = "; ".join(e["text"].strip() for e in edits if e["text"].strip())
            fix_parts = []
            if ev.command and ev.command != failure["command"]:
                fix_parts.append(f"Run `{ev.command}` instead of `{failure['command']}`")
            if edit_notes:
                fix_parts.append(edit_notes)
            if ev.text.strip() and not fix_parts:
                fix_parts.append(ev.text.strip()[:240])
            action_text = "; ".join(fix_parts) or f"Apply verified fix and run `{ev.command or fam}`"

            fp_src = f"f2p:{eff_scope}:{failure['error_signature'].casefold()}:{action_text.casefold()}"
            fp = hashlib.sha256(fp_src.encode("utf-8")).hexdigest()[:20]
            ok, reason, count = self.limiter.allow(
                user_id=user_id,
                agent_id=agent_id,
                scope_key=eff_scope,
                fingerprint=fp,
                now=now,
                max_writes=spec.max_writes_per_window,
                window_seconds=spec.write_window_seconds,
            )
            if not ok:
                return {"signal": "fail_to_pass", "written": False, "reason": reason, "writes_in_window": count, "fact_id": None}

            rule_slug = _slug(f"{fam}_{failure['error_signature'][:36]}")
            path = f"/learned/{rule_slug}.md"
            anchors = self._build_anchors(all_files, all_syms)
            rule_res = self.engine.ingest_rule(
                path,
                title=f"Fix {fam}: {failure['error_signature'][:42]}",
                trigger_condition=f"When `{failure['command'] or fam}` fails with: {failure['error_signature']}",
                action=action_text,
                verification_result=f"Verified passing (`exit_code=0`) via `{ev.command or ev.tool_name}`",
                notes=f"Files: {', '.join(all_files)}" if all_files else "",
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                scope_key=eff_scope,
                override_key=normalize_override_key(ev.override_key or rule_slug),
                salience_tag="core_rule",
                always_active=False,
                anchors=anchors,
                repo_root=repo_root,
                timestamp=now,
            )
            new_count = self.limiter.record(user_id=user_id, agent_id=agent_id, scope_key=eff_scope, fingerprint=fp, now=now)
            return {
                "signal": "fail_to_pass",
                "written": bool(rule_res.get("accepted")),
                "fact_id": rule_res.get("fact_id"),
                "path": rule_res.get("path"),
                "scope_key": eff_scope,
                "anchors": rule_res.get("anchors", []),
                "writes_in_window": new_count,
            }

        return {"signal": None, "written": False, "reason": "ignored_event_type", "fact_id": None}

    @staticmethod
    def _build_anchors(files: list[str] | tuple[str, ...], symbols: list[str] | tuple[str, ...]) -> list[CodeAnchor]:
        anchors: list[CodeAnchor] = []
        norm_files = [normalize_rel_path(f) for f in files if str(f).strip()]
        if symbols and norm_files:
            primary = norm_files[0]
            for sym in symbols:
                anchors.append(CodeAnchor(file_path=primary, anchor_type="symbol", symbol_name=sym))
            for f in norm_files[1:]:
                anchors.append(CodeAnchor(file_path=f, anchor_type="file"))
        else:
            for f in norm_files:
                anchors.append(CodeAnchor(file_path=f, anchor_type="file"))
        return anchors
