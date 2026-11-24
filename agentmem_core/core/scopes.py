"""Hierarchical scope chain model, normalization, and cross-scope precedence resolution."""
from __future__ import annotations

import re
from dataclasses import dataclass

_SCOPE_TOKEN = re.compile(r"[^a-zA-Z0-9._/@:-]+")
_OVERRIDE_SPACE = re.compile(r"\s+")

CANONICAL_LEVELS = ("session", "repo", "project", "user", "global")


def normalize_scope_key(scope_key: str | None) -> str:
    """Normalize a scope identifier such as ``repo:payments`` or ``user:alice``."""
    raw = str(scope_key or "").strip()
    if not raw or raw.casefold() == "global":
        return "global"
    cleaned = _SCOPE_TOKEN.sub("_", raw).strip(":")
    if ":" not in cleaned:
        return f"custom:{cleaned}"
    level, ident = cleaned.split(":", 1)
    level = level.strip().casefold() or "custom"
    ident = ident.strip()
    if level == "global" or not ident:
        return "global"
    return f"{level}:{ident}"


def scope_level(scope_key: str | None) -> str:
    norm = normalize_scope_key(scope_key)
    return "global" if norm == "global" else norm.split(":", 1)[0]


def normalize_override_key(key: str | None) -> str:
    """Normalize a semantic override key (e.g. ``test_runner``, ``code_style``)."""
    raw = str(key or "").strip().casefold()
    if not raw:
        return ""
    return _OVERRIDE_SPACE.sub("_", raw)


@dataclass(frozen=True)
class ScopeChain:
    """Ordered scopes from highest precedence (index 0) to lowest precedence (index N-1).

    Example coding chain:
      ``("session:s1", "repo:payments", "project:billing", "user:alice", "global")``

    Example conversational chain:
      ``("session:c1", "user:alice", "global")``
    """

    scopes: tuple[str, ...]

    def __post_init__(self) -> None:
        deduped: list[str] = []
        seen: set[str] = set()
        for raw in self.scopes:
            norm = normalize_scope_key(raw)
            if norm not in seen:
                seen.add(norm)
                deduped.append(norm)
        if not deduped:
            deduped = ["global"]
        object.__setattr__(self, "scopes", tuple(deduped))

    @classmethod
    def build(
        cls,
        *,
        hierarchy: tuple[str, ...] = ("session", "user", "global"),
        user_id: str | None = None,
        session_id: str | None = None,
        repo_id: str | None = None,
        project_id: str | None = None,
        extra_scopes: tuple[str, ...] | list[str] = (),
    ) -> "ScopeChain":
        """Build an ordered ScopeChain from a profile's declarative ``scope_hierarchy``."""
        mapping = {
            "session": f"session:{session_id}" if session_id else None,
            "repo": f"repo:{repo_id}" if repo_id else None,
            "project": f"project:{project_id}" if project_id else None,
            "user": f"user:{user_id}" if user_id else None,
            "global": "global",
        }
        ordered: list[str] = []
        for level in hierarchy:
            norm_level = str(level).strip().casefold()
            if ":" in norm_level:
                ordered.append(norm_level)
            else:
                val = mapping.get(norm_level)
                if val:
                    ordered.append(val)
        for ext in extra_scopes:
            if ext:
                ordered.append(str(ext))
        if "global" not in ordered:
            ordered.append("global")
        return cls(tuple(ordered))

    @classmethod
    def coerce(cls, value: "ScopeChain | list[str] | tuple[str, ...] | str | None") -> "ScopeChain | None":
        if value is None:
            return None
        if isinstance(value, ScopeChain):
            return value
        if isinstance(value, str):
            parts = [p.strip() for p in value.split(",") if p.strip()]
            return cls(tuple(parts)) if parts else None
        items = [str(x).strip() for x in value if str(x).strip()]
        return cls(tuple(items)) if items else None

    def rank(self, scope_key: str | None) -> int | None:
        norm = normalize_scope_key(scope_key)
        try:
            return self.scopes.index(norm)
        except ValueError:
            return None

    def contains(self, scope_key: str | None) -> bool:
        return self.rank(scope_key) is not None

    def as_list(self) -> list[str]:
        return list(self.scopes)


def resolve_write_scope(
    *,
    explicit_scope_key: str | None,
    default_level: str,
    user_id: str = "default",
    session_id: str | None = None,
    repo_id: str | None = None,
    project_id: str | None = None,
) -> str:
    """Resolve the canonical write scope key from explicit override or profile default level."""
    if explicit_scope_key:
        return normalize_scope_key(explicit_scope_key)
    level = str(default_level or "user").strip().casefold()
    if ":" in level:
        return normalize_scope_key(level)
    if level == "repo":
        if repo_id:
            return normalize_scope_key(f"repo:{repo_id}")
        if project_id:
            return normalize_scope_key(f"project:{project_id}")
    if level == "project" and project_id:
        return normalize_scope_key(f"project:{project_id}")
    if level == "session" and session_id:
        return normalize_scope_key(f"session:{session_id}")
    if level == "user" and user_id:
        return normalize_scope_key(f"user:{user_id}")
    if level == "global":
        return "global"
    if repo_id:
        return normalize_scope_key(f"repo:{repo_id}")
    if project_id:
        return normalize_scope_key(f"project:{project_id}")
    return normalize_scope_key(f"user:{user_id}") if user_id else "global"


def _conflict_key(fact: dict) -> str | None:
    okey = normalize_override_key(fact.get("override_key"))
    if okey:
        return f"override:{okey}"
    vpath = fact.get("vfs_path")
    if vpath:
        return f"path:{vpath}"
    slot = fact.get("slot_key")
    if slot:
        return f"slot:{slot}"
    return None


def apply_scope_precedence(
    facts: list[dict],
    scope_chain: ScopeChain | None,
    *,
    suppress_overridden: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Filter facts to ``scope_chain`` and resolve higher-precedence overrides.

    Returns ``(active_facts, overridden_facts)``.
    When ``scope_chain`` is ``None``, all facts pass through unchanged for backward compatibility.
    """
    if scope_chain is None:
        return list(facts), []

    admitted: list[dict] = []
    for fact in facts:
        skey = normalize_scope_key(fact.get("scope_key"))
        rank = scope_chain.rank(skey)
        if rank is None:
            continue
        item = dict(fact)
        item["scope_key"] = skey
        item["scope_rank"] = rank
        admitted.append(item)

    # Determine winning (lowest-rank = highest-precedence) scope per conflict key
    best_by_key: dict[str, tuple[int, str]] = {}
    for item in admitted:
        ckey = _conflict_key(item)
        if not ckey:
            continue
        rank = int(item["scope_rank"])
        prev = best_by_key.get(ckey)
        if prev is None or rank < prev[0]:
            best_by_key[ckey] = (rank, item["scope_key"])

    kept: list[dict] = []
    overridden: list[dict] = []
    for item in admitted:
        ckey = _conflict_key(item)
        if ckey and ckey in best_by_key:
            win_rank, win_scope = best_by_key[ckey]
            if int(item["scope_rank"]) > win_rank:
                item["scope_overridden"] = True
                item["overridden_by_scope"] = win_scope
                overridden.append(item)
                if suppress_overridden:
                    continue
        item["scope_overridden"] = False
        kept.append(item)

    # Stable precedence tie-break: keep relevance order, lift strictly higher-precedence
    # items ahead of lower-precedence items only when both have equal or near-equal rank/score
    # OR when an overriding fact matches a topic present in overridden facts.
    if overridden:
        overridden_keys = {_conflict_key(o) for o in overridden if _conflict_key(o)}
        winners = [f for f in kept if _conflict_key(f) in overridden_keys]
        others = [f for f in kept if _conflict_key(f) not in overridden_keys]
        kept = [*winners, *others]

    return kept, overridden
