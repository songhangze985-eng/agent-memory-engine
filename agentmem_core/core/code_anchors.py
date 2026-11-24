"""C1: Fact-level code anchors (file / symbol / line_range / git_commit / package_version) & staleness tracking."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentmem_core.core.facts import UnifiedFactLedger
from agentmem_core.core.scopes import ScopeChain, normalize_scope_key

ANCHOR_TYPES = ("file", "symbol", "line_range", "git_commit", "package_version")
_GENERIC_SYMBOL_RE_TEMPLATE = r"(?:def|class|function|func|const|let|var|type|interface|struct)\s+{name}\b"
_DEP_LINE_RE = re.compile(
    r"""['"]?([A-Za-z0-9_.-]+)['"]?\s*(?:((?:==|>=|<=|~=|!=|>|<)[0-9A-Za-z._,*+-]+)|[:=]\s*['"]?([~^><=!*0-9A-Za-z._,+-]+)['"]?)"""
)


def normalize_rel_path(path: str | Path) -> str:
    """Normalize a repository-relative path into POSIX form without leading ./ or /."""
    raw = str(path or "").replace("\\", "/").strip()
    while raw.startswith("./"):
        raw = raw[2:]
    return raw.lstrip("/")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:24]


def extract_symbol_source(text: str, symbol_name: str, *, suffix: str = "") -> str | None:
    """Extract canonical source slice for a Python AST symbol or generic code symbol."""
    if not symbol_name:
        return None
    target = symbol_name.strip()
    if suffix.casefold() == ".py" or not suffix:
        try:
            tree = ast.parse(text)
            lines = text.splitlines()
            parts = target.split(".")
            if len(parts) == 2:
                cls_name, method_name = parts
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name == cls_name:
                        for child in node.body:
                            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                                end = getattr(child, "end_lineno", child.lineno)
                                return "\n".join(lines[child.lineno - 1 : end])
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == target:
                    end = getattr(node, "end_lineno", node.lineno)
                    return "\n".join(lines[node.lineno - 1 : end])
            if suffix.casefold() == ".py":
                return None
        except SyntaxError:
            if suffix.casefold() == ".py":
                return None

    # Generic fallback for TS/JS/Go/Rust/etc.
    pat = re.compile(_GENERIC_SYMBOL_RE_TEMPLATE.format(name=re.escape(target)))
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if pat.search(line):
            return "\n".join(lines[idx : min(len(lines), idx + 25)])
    return None


def extract_package_version(text: str, package_name: str, *, file_path: str = "") -> str | None:
    """Extract declared version spec for ``package_name`` from package.json, pyproject.toml, or requirements.txt."""
    if not package_name:
        return None
    target = package_name.strip().casefold()
    if file_path.endswith(".json"):
        try:
            data = json.loads(text)
            for section in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
                deps = data.get(section)
                if isinstance(deps, dict):
                    for k, v in deps.items():
                        if str(k).casefold() == target:
                            return str(v).strip()
        except Exception:
            pass
    for line in text.splitlines():
        clean = line.split("#", 1)[0].strip()
        if not clean:
            continue
        for match in _DEP_LINE_RE.finditer(clean):
            pkg = match.group(1).casefold()
            ver = (match.group(2) or match.group(3) or "").strip()
            if pkg == target and ver:
                return ver
    return None


@dataclass(frozen=True)
class CodeAnchor:
    """Typed code/repository anchor attached to a unified memory fact or rule."""

    file_path: str
    anchor_type: str = "file"
    symbol_name: str = ""
    line_start: int | None = None
    line_end: int | None = None
    content_hash: str = ""
    version_spec: str = ""
    commit_sha: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        atype = str(self.anchor_type or "file").strip().casefold()
        if atype not in ANCHOR_TYPES:
            raise ValueError(f"anchor_type must be one of {ANCHOR_TYPES}, got {atype!r}")
        object.__setattr__(self, "anchor_type", atype)
        object.__setattr__(self, "file_path", normalize_rel_path(self.file_path))

    @classmethod
    def coerce(cls, raw: "CodeAnchor | dict[str, Any] | str") -> "CodeAnchor":
        if isinstance(raw, CodeAnchor):
            return raw
        if isinstance(raw, str):
            if "::" in raw:
                fpath, sym = raw.split("::", 1)
                return cls(file_path=fpath, anchor_type="symbol", symbol_name=sym.strip())
            return cls(file_path=raw, anchor_type="file")
        if isinstance(raw, dict):
            atype = raw.get("anchor_type")
            if not atype:
                if raw.get("version_spec") or raw.get("package_name"):
                    atype = "package_version"
                elif raw.get("symbol_name"):
                    atype = "symbol"
                elif raw.get("line_start") is not None:
                    atype = "line_range"
                elif raw.get("commit_sha"):
                    atype = "git_commit"
                else:
                    atype = "file"
            sym = str(raw.get("symbol_name") or raw.get("package_name") or "")
            return cls(
                file_path=str(raw.get("file_path") or raw.get("path") or ""),
                anchor_type=str(atype),
                symbol_name=sym,
                line_start=raw.get("line_start"),
                line_end=raw.get("line_end"),
                content_hash=str(raw.get("content_hash") or ""),
                version_spec=str(raw.get("version_spec") or ""),
                commit_sha=str(raw.get("commit_sha") or ""),
                metadata=dict(raw.get("metadata") or {}),
            )
        raise ValueError(f"cannot coerce anchor from {type(raw).__name__}")


def capture_anchor(
    repo_root: str | Path,
    *,
    file_path: str | Path,
    symbol_name: str = "",
    line_start: int | None = None,
    line_end: int | None = None,
    package_name: str = "",
    commit_sha: str = "",
) -> CodeAnchor:
    """Capture a live content/AST/version hash anchor from ``repo_root``."""
    rel = normalize_rel_path(file_path)
    full = Path(repo_root) / rel
    text = full.read_text(encoding="utf-8", errors="replace") if full.is_file() else ""

    if package_name:
        ver = extract_package_version(text, package_name, file_path=rel) or ""
        return CodeAnchor(
            file_path=rel,
            anchor_type="package_version",
            symbol_name=package_name,
            version_spec=ver,
            content_hash=_sha256_text(f"{package_name}:{ver}") if ver else "",
            commit_sha=commit_sha,
        )
    if symbol_name:
        sym_src = extract_symbol_source(text, symbol_name, suffix=full.suffix) if text else None
        return CodeAnchor(
            file_path=rel,
            anchor_type="symbol",
            symbol_name=symbol_name,
            content_hash=_sha256_text(sym_src) if sym_src is not None else "",
            commit_sha=commit_sha,
        )
    if line_start is not None:
        lines = text.splitlines()
        s = max(1, int(line_start))
        e = int(line_end) if line_end is not None else s
        snippet = "\n".join(lines[s - 1 : e])
        return CodeAnchor(
            file_path=rel,
            anchor_type="line_range",
            line_start=s,
            line_end=e,
            content_hash=_sha256_text(snippet) if snippet else "",
            commit_sha=commit_sha,
        )
    return CodeAnchor(
        file_path=rel,
        anchor_type="git_commit" if commit_sha and not text else "file",
        content_hash=_sha256_text(text) if text else "",
        commit_sha=commit_sha,
    )


class CodeAnchorManager:
    """Stores fact-to-code anchors in SQLite/PG and detects drift -> marks facts ``stale``."""

    def __init__(self, store, ledger: UnifiedFactLedger):
        self.store = store
        self.ledger = ledger
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.store.conn.execute(
            """CREATE TABLE IF NOT EXISTS fact_code_anchors (
                anchor_id TEXT PRIMARY KEY,
                fact_id TEXT NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
                user_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                scope_key TEXT NOT NULL DEFAULT 'global',
                anchor_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                symbol_name TEXT NOT NULL DEFAULT '',
                line_start INTEGER,
                line_end INTEGER,
                content_hash TEXT NOT NULL DEFAULT '',
                version_spec TEXT NOT NULL DEFAULT '',
                commit_sha TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'fresh',
                created_at DOUBLE PRECISION NOT NULL,
                verified_at DOUBLE PRECISION NOT NULL
            )"""
        )
        self.store.conn.execute("CREATE INDEX IF NOT EXISTS idx_fca_fact ON fact_code_anchors(fact_id)")
        self.store.conn.execute("CREATE INDEX IF NOT EXISTS idx_fca_file ON fact_code_anchors(user_id, agent_id, file_path)")
        self.store.conn.execute("CREATE INDEX IF NOT EXISTS idx_fca_symbol ON fact_code_anchors(user_id, agent_id, symbol_name)")
        self.store.conn.commit()

    def attach(
        self,
        fact_id: str,
        anchors: list[CodeAnchor | dict[str, Any] | str] | tuple[CodeAnchor | dict[str, Any] | str, ...],
        *,
        user_id: str = "default",
        agent_id: str = "default",
        scope_key: str | None = None,
        repo_root: str | Path | None = None,
        timestamp: float | None = None,
    ) -> list[dict]:
        now = time.time() if timestamp is None else float(timestamp)
        norm_scope = normalize_scope_key(scope_key or (f"user:{user_id}" if user_id else "global"))
        attached: list[dict] = []
        for raw in anchors:
            anc = CodeAnchor.coerce(raw)
            if repo_root is not None and not anc.content_hash and anc.file_path:
                captured = capture_anchor(
                    repo_root,
                    file_path=anc.file_path,
                    symbol_name=anc.symbol_name if anc.anchor_type == "symbol" else "",
                    line_start=anc.line_start,
                    line_end=anc.line_end,
                    package_name=anc.symbol_name if anc.anchor_type == "package_version" else "",
                    commit_sha=anc.commit_sha,
                )
                anc = CodeAnchor(
                    file_path=anc.file_path,
                    anchor_type=anc.anchor_type,
                    symbol_name=anc.symbol_name,
                    line_start=anc.line_start,
                    line_end=anc.line_end,
                    content_hash=captured.content_hash,
                    version_spec=anc.version_spec or captured.version_spec,
                    commit_sha=anc.commit_sha,
                )
            aid = f"anc_{uuid.uuid4().hex[:14]}"
            self.store.conn.execute(
                """INSERT INTO fact_code_anchors(
                       anchor_id, fact_id, user_id, agent_id, scope_key, anchor_type,
                       file_path, symbol_name, line_start, line_end, content_hash,
                       version_spec, commit_sha, status, created_at, verified_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'fresh', ?, ?)""",
                (
                    aid,
                    fact_id,
                    user_id,
                    agent_id,
                    norm_scope,
                    anc.anchor_type,
                    anc.file_path,
                    anc.symbol_name,
                    anc.line_start,
                    anc.line_end,
                    anc.content_hash,
                    anc.version_spec,
                    anc.commit_sha,
                    now,
                    now,
                ),
            )
            attached.append(
                {
                    "anchor_id": aid,
                    "fact_id": fact_id,
                    "scope_key": norm_scope,
                    "anchor_type": anc.anchor_type,
                    "file_path": anc.file_path,
                    "symbol_name": anc.symbol_name,
                    "line_start": anc.line_start,
                    "line_end": anc.line_end,
                    "content_hash": anc.content_hash,
                    "version_spec": anc.version_spec,
                    "commit_sha": anc.commit_sha,
                    "status": "fresh",
                }
            )
        self.store.conn.commit()
        return attached

    def list_anchors(
        self,
        *,
        fact_id: str | None = None,
        user_id: str = "default",
        agent_id: str = "default",
        scope_chain: ScopeChain | None = None,
    ) -> list[dict]:
        clauses = ["a.user_id=?", "a.agent_id=?", "f.status='active'"]
        params: list[Any] = [user_id, agent_id]
        if fact_id is not None:
            clauses.append("a.fact_id=?")
            params.append(fact_id)
        if scope_chain is not None:
            ph = ",".join("?" for _ in scope_chain.scopes)
            clauses.append(f"a.scope_key IN ({ph})")
            params.extend(scope_chain.scopes)
        rows = self.store.conn.execute(
            f"""SELECT a.* FROM fact_code_anchors a
                JOIN facts f ON f.id = a.fact_id
                WHERE {' AND '.join(clauses)}
                ORDER BY a.created_at ASC""",
            tuple(params),
        ).fetchall()
        return [dict(r) for r in rows]

    def verify_anchors(
        self,
        *,
        repo_root: str | Path | None = None,
        changed_files: list[str] | tuple[str, ...] | set[str] | None = None,
        file_contents: dict[str, str] | None = None,
        user_id: str = "default",
        agent_id: str = "default",
        scope_chain: ScopeChain | None = None,
        fact_id: str | None = None,
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        """Verify anchors against workspace files / changed files and mark drifted facts ``stale``."""
        now = time.time() if timestamp is None else float(timestamp)
        anchors = self.list_anchors(
            fact_id=fact_id,
            user_id=user_id,
            agent_id=agent_id,
            scope_chain=scope_chain,
        )
        changed_norm = {normalize_rel_path(p) for p in (changed_files or ())}
        content_norm = {normalize_rel_path(k): v for k, v in (file_contents or {}).items()}
        root_path = Path(repo_root) if repo_root is not None else None

        by_fact: dict[str, list[dict]] = {}
        for row in anchors:
            by_fact.setdefault(row["fact_id"], []).append(row)

        stale_facts: list[dict[str, Any]] = []
        fresh_facts: list[str] = []

        for fid, fact_anchors in by_fact.items():
            reasons: list[str] = []
            for row in fact_anchors:
                rel = normalize_rel_path(row["file_path"])
                atype = row["anchor_type"]
                drift: str | None = None

                # Load current file text if available from memory map or repo_root
                text: str | None = None
                file_exists = False
                if rel in content_norm:
                    text = content_norm[rel]
                    file_exists = True
                elif root_path is not None and rel:
                    full = root_path / rel
                    if full.is_file():
                        text = full.read_text(encoding="utf-8", errors="replace")
                        file_exists = True

                if root_path is not None or rel in content_norm:
                    if not file_exists:
                        drift = f"file_missing:{rel}"
                    elif atype == "package_version":
                        cur_ver = extract_package_version(text or "", row["symbol_name"], file_path=rel)
                        if cur_ver is None:
                            drift = f"package_removed:{row['symbol_name']}@{rel}"
                        elif row["version_spec"] and cur_ver != row["version_spec"]:
                            drift = f"package_version_changed:{row['symbol_name']}({row['version_spec']}->{cur_ver})"
                    elif atype == "symbol":
                        sym_src = extract_symbol_source(text or "", row["symbol_name"], suffix=Path(rel).suffix)
                        if sym_src is None:
                            drift = f"symbol_removed:{rel}::{row['symbol_name']}"
                        elif row["content_hash"] and _sha256_text(sym_src) != row["content_hash"]:
                            drift = f"symbol_modified:{rel}::{row['symbol_name']}"
                    elif atype == "line_range" and row["line_start"] is not None:
                        lines = (text or "").splitlines()
                        s = max(1, int(row["line_start"]))
                        e = int(row["line_end"]) if row["line_end"] is not None else s
                        snippet = "\n".join(lines[s - 1 : e])
                        if not snippet:
                            drift = f"lines_missing:{rel}:{s}-{e}"
                        elif row["content_hash"] and _sha256_text(snippet) != row["content_hash"]:
                            drift = f"lines_modified:{rel}:{s}-{e}"
                    else:
                        if row["content_hash"] and _sha256_text(text or "") != row["content_hash"]:
                            drift = f"file_modified:{rel}"
                elif rel in changed_norm:
                    # Event-driven change notification without disk inspection
                    drift = f"file_changed:{rel}"

                new_status = "stale" if drift else "fresh"
                self.store.conn.execute(
                    "UPDATE fact_code_anchors SET status=?, verified_at=? WHERE anchor_id=?",
                    (new_status, now, row["anchor_id"]),
                )
                if drift:
                    reasons.append(drift)

            if reasons:
                reason_str = "; ".join(reasons)
                self.ledger.set_staleness(fid, staleness_status="stale", stale_reason=reason_str, timestamp=now)
                stale_facts.append({"fact_id": fid, "stale_reason": reason_str, "reasons": reasons})
            else:
                # Only promote back to fresh if explicitly verified against disk/content
                if root_path is not None or content_norm:
                    self.ledger.set_staleness(fid, staleness_status="fresh", stale_reason="", timestamp=now)
                fresh_facts.append(fid)

        self.store.conn.commit()
        return {
            "checked_anchors": len(anchors),
            "checked_facts": len(by_fact),
            "stale_count": len(stale_facts),
            "stale_facts": stale_facts,
            "fresh_facts": fresh_facts,
        }

    def find_by_context(
        self,
        *,
        active_files: list[str] | tuple[str, ...] = (),
        active_symbols: list[str] | tuple[str, ...] = (),
        user_id: str = "default",
        agent_id: str = "default",
        scope_chain: ScopeChain | None = None,
        include_stale: bool = False,
        limit: int = 20,
    ) -> list[dict]:
        """O(log N) direct anchor lookup for C3 zero-LLM ambient context injection."""
        norm_files = [normalize_rel_path(p) for p in active_files if str(p).strip()]
        norm_syms = [str(s).strip() for s in active_symbols if str(s).strip()]
        if not norm_files and not norm_syms:
            return []

        match_clauses: list[str] = []
        params: list[Any] = [user_id, agent_id]
        if norm_files:
            ph = ",".join("?" for _ in norm_files)
            match_clauses.append(f"a.file_path IN ({ph})")
            params.extend(norm_files)
        if norm_syms:
            ph = ",".join("?" for _ in norm_syms)
            match_clauses.append(f"a.symbol_name IN ({ph})")
            params.extend(norm_syms)

        clauses = [
            "a.user_id=?",
            "a.agent_id=?",
            "f.status='active'",
            f"({' OR '.join(match_clauses)})",
        ]
        if scope_chain is not None:
            ph = ",".join("?" for _ in scope_chain.scopes)
            clauses.append(f"m.scope_key IN ({ph})")
            params.extend(scope_chain.scopes)
        if not include_stale:
            clauses.append("m.staleness_status != 'stale'")
        params.append(int(limit))

        rows = self.store.conn.execute(
            f"""SELECT DISTINCT m.*, f.plain, f.status, f.confidence, f.source_turn_ids,
                       f.event_time_start, f.ingest_time, a.file_path AS matched_file, a.symbol_name AS matched_symbol
                FROM fact_code_anchors a
                JOIN facts f ON f.id = a.fact_id
                JOIN fact_unified_meta m ON m.fact_id = a.fact_id
                WHERE {' AND '.join(clauses)}
                ORDER BY m.always_active DESC, m.updated_at DESC
                LIMIT ?""",
            tuple(params),
        ).fetchall()
        results: list[dict] = []
        seen: set[str] = set()
        for r in rows:
            decoded = UnifiedFactLedger._decode_row(r)
            fid = decoded["fact_id"]
            if fid in seen:
                continue
            seen.add(fid)
            decoded["id"] = fid
            decoded["anchor_hit"] = True
            results.append(decoded)
        return results
