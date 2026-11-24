"""Declarative MemoryProfile configuration loader (zero profile branching in core).

Profiles are loaded from YAML or JSON files (``profiles/<name>.yaml`` or an explicit path).
All behavioral differences between conversational, coding, and hybrid memory modes are
expressed as typed configuration fields rather than conditional branches on profile name.
"""
from __future__ import annotations

import json
import sysconfig
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from agentmem_core.core.plan import RecallConfig

# Backward-compatible preset aliases (resolved purely at config load time, never branched in core)
_PROFILE_ALIASES: dict[str, str] = {
    "coding_agent": "coding",
    "code": "coding",
    "chat": "conversational",
    "default": "conversational",
}

_BUILTIN_DEFAULTS: dict[str, dict[str, Any]] = {
    "conversational": {
        "name": "conversational",
        "description": "Long-horizon multi-session conversational memory profile.",
        "disclosure": "full",
        "default_token_budget": 2048,
        "preload_paths": [],
        "preload_always_active": False,
        "include_directory_outline": False,
        "enable_code_anchors": False,
        "stale_policy": "warn",
        "ingest_track": "conversational",
        "enable_salience_gate": False,
        "scope_hierarchy": ["session", "user", "global"],
        "default_write_scope_level": "user",
        "ambient_token_budget": 512,
        "ambient_top_k": 6,
        "enable_ambient_llm": False,
        "max_writes_per_window": 128,
        "write_window_seconds": 3600.0,
        "max_question_llm_calls": 2,
        "recall": {
            "enable_bipolar": False,
            "enable_recency_rerank": False,
            "enable_episode_fts": True,
            "enable_entity_graph": True,
            "enable_slot_supersede": True,
            "enable_graph_enum": True,
            "enable_deterministic_calc": True,
            "enable_code_anchors": False,
        },
    },
    "coding": {
        "name": "coding",
        "description": "Coding-agent procedural, anchor-verified, and repo-scoped memory profile.",
        "disclosure": "progressive",
        "default_token_budget": 2048,
        "preload_paths": ["/global.md"],
        "preload_always_active": True,
        "include_directory_outline": True,
        "enable_code_anchors": True,
        "stale_policy": "exclude",
        "ingest_track": "signal_driven",
        "enable_salience_gate": True,
        "scope_hierarchy": ["session", "repo", "project", "user", "global"],
        "default_write_scope_level": "repo",
        "ambient_token_budget": 768,
        "ambient_top_k": 8,
        "enable_ambient_llm": False,
        "max_writes_per_window": 32,
        "write_window_seconds": 3600.0,
        "max_question_llm_calls": 2,
        "recall": {
            "enable_bipolar": False,
            "enable_recency_rerank": False,
            "enable_episode_fts": True,
            "enable_entity_graph": True,
            "enable_slot_supersede": True,
            "enable_graph_enum": True,
            "enable_deterministic_calc": True,
            "enable_code_anchors": True,
        },
    },
    "hybrid": {
        "name": "hybrid",
        "description": "Unified project-oriented profile combining conversational recall and coding rules/anchors.",
        "disclosure": "full",
        "default_token_budget": 2048,
        "preload_paths": ["/global.md"],
        "preload_always_active": True,
        "include_directory_outline": True,
        "enable_code_anchors": True,
        "stale_policy": "exclude",
        "ingest_track": "conversational",
        "enable_salience_gate": False,
        "scope_hierarchy": ["session", "repo", "project", "user", "global"],
        "default_write_scope_level": "project",
        "ambient_token_budget": 768,
        "ambient_top_k": 8,
        "enable_ambient_llm": False,
        "max_writes_per_window": 64,
        "write_window_seconds": 3600.0,
        "max_question_llm_calls": 2,
        "recall": {
            "enable_bipolar": False,
            "enable_recency_rerank": False,
            "enable_episode_fts": True,
            "enable_entity_graph": True,
            "enable_slot_supersede": True,
            "enable_graph_enum": True,
            "enable_deterministic_calc": True,
            "enable_code_anchors": True,
        },
    },
}


@dataclass(frozen=True)
class MemoryProfile:
    """Declarative runtime profile controlling ingest, scope chain, ambient path, and assembly."""

    name: str = "conversational"
    description: str = ""
    disclosure: Literal["full", "progressive"] = "full"
    default_token_budget: int = 2048
    preload_paths: tuple[str, ...] = ()
    preload_always_active: bool = False
    include_directory_outline: bool = False
    enable_code_anchors: bool = False
    stale_policy: Literal["exclude", "demote", "warn"] = "warn"
    ingest_track: Literal["conversational", "procedural", "signal_driven"] = "conversational"
    enable_salience_gate: bool = False
    scope_hierarchy: tuple[str, ...] = ("session", "user", "global")
    default_write_scope_level: str = "user"
    ambient_token_budget: int = 512
    ambient_top_k: int = 6
    enable_ambient_llm: bool = False
    max_writes_per_window: int = 128
    write_window_seconds: float = 3600.0
    max_question_llm_calls: int = 2
    recall: RecallConfig = field(default_factory=RecallConfig)

    def __post_init__(self) -> None:
        if self.enable_ambient_llm:
            raise ValueError("Invariant violation (G4): enable_ambient_llm MUST be False")
        if self.disclosure not in ("full", "progressive"):
            raise ValueError(f"invalid disclosure mode: {self.disclosure!r}")
        if self.stale_policy not in ("exclude", "demote", "warn"):
            raise ValueError(f"invalid stale_policy: {self.stale_policy!r}")
        if self.max_question_llm_calls < 0 or self.max_question_llm_calls > 2:
            raise ValueError("Invariant violation (G4): max_question_llm_calls must be in [0, 2]")

    def with_overrides(self, **kwargs: Any) -> "MemoryProfile":
        return replace(self, **kwargs)

    def to_recall_config(self, base: RecallConfig | None = None) -> RecallConfig:
        src = base or self.recall
        return replace(
            src,
            enable_code_anchors=self.enable_code_anchors or src.enable_code_anchors,
            disclosure=self.disclosure,
            profile_name=self.name,
        )


def profiles_dir() -> Path | None:
    """Locate packaged ``profiles/`` directory for editable and wheel installs."""
    here = Path(__file__).resolve().parents[2]
    candidates = [
        here / "profiles",
        Path(sysconfig.get_path("data")) / "share" / "agentmem_core" / "profiles",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return None


def _parse_scalar(raw: str) -> Any:
    val = raw.strip()
    if not val:
        return ""
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    low = val.casefold()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none", "~"):
        return None
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in inner.split(",")]
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        return val


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Dependency-free YAML subset parser for profile and protocol configs (uses PyYAML if available)."""
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return loaded if isinstance(loaded, dict) else {}
    except ImportError:
        pass

    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    lines = text.splitlines()

    def _strip_comment(line: str) -> str:
        in_single = False
        in_double = False
        for idx, ch in enumerate(line):
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
            elif ch == "#" and not in_single and not in_double:
                if idx == 0 or line[idx - 1].isspace():
                    return line[:idx]
        return line

    idx = 0
    while idx < len(lines):
        raw_line = _strip_comment(lines[idx]).rstrip()
        idx += 1
        if not raw_line.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.lstrip(" ")

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        container = stack[-1][1]

        if stripped.startswith("- "):
            val_str = stripped[2:].strip()
            if isinstance(container, list):
                if ":" in val_str and not val_str.startswith(("'", '"', "http://", "https://")):
                    k, v = val_str.split(":", 1)
                    item: dict[str, Any] = {k.strip(): _parse_scalar(v) if v.strip() else {}}
                    container.append(item)
                    stack.append((indent + 1, item))
                else:
                    container.append(_parse_scalar(val_str))
            continue

        if ":" not in stripped:
            continue
        key, rest = stripped.split(":", 1)
        key = key.strip()
        rest = rest.strip()
        if rest:
            if isinstance(container, dict):
                container[key] = _parse_scalar(rest)
        else:
            # Look ahead to decide whether child is a list or nested dict
            next_is_list = False
            for peek in lines[idx:]:
                clean_peek = _strip_comment(peek).rstrip()
                if not clean_peek.strip():
                    continue
                next_indent = len(clean_peek) - len(clean_peek.lstrip(" "))
                if next_indent > indent and clean_peek.lstrip(" ").startswith("- "):
                    next_is_list = True
                break
            child: Any = [] if next_is_list else {}
            if isinstance(container, dict):
                container[key] = child
                stack.append((indent, child))
    return root


def _profile_from_dict(data: dict[str, Any]) -> MemoryProfile:
    raw = dict(data)
    recall_raw = raw.pop("recall", None) or {}
    recall_fields = {
        "enable_bipolar": bool(recall_raw.get("enable_bipolar", False)),
        "enable_recency_rerank": bool(recall_raw.get("enable_recency_rerank", False)),
        "enable_episode_fts": bool(recall_raw.get("enable_episode_fts", True)),
        "enable_entity_graph": bool(recall_raw.get("enable_entity_graph", True)),
        "enable_slot_supersede": bool(recall_raw.get("enable_slot_supersede", True)),
        "enable_graph_enum": bool(recall_raw.get("enable_graph_enum", True)),
        "enable_deterministic_calc": bool(recall_raw.get("enable_deterministic_calc", True)),
        "enable_code_anchors": bool(
            recall_raw.get("enable_code_anchors", raw.get("enable_code_anchors", False))
        ),
        "disclosure": str(raw.get("disclosure", "full")),
        "profile_name": str(raw.get("name", "conversational")),
    }
    recall_cfg = RecallConfig(**recall_fields)
    return MemoryProfile(
        name=str(raw.get("name", "conversational")),
        description=str(raw.get("description", "")),
        disclosure=str(raw.get("disclosure", "full")),  # type: ignore[arg-type]
        default_token_budget=int(raw.get("default_token_budget", 2048)),
        preload_paths=tuple(str(p) for p in (raw.get("preload_paths") or ())),
        preload_always_active=bool(raw.get("preload_always_active", False)),
        include_directory_outline=bool(raw.get("include_directory_outline", False)),
        enable_code_anchors=bool(raw.get("enable_code_anchors", False)),
        stale_policy=str(raw.get("stale_policy", "warn")),  # type: ignore[arg-type]
        ingest_track=str(raw.get("ingest_track", "conversational")),  # type: ignore[arg-type]
        enable_salience_gate=bool(raw.get("enable_salience_gate", False)),
        scope_hierarchy=tuple(str(s) for s in (raw.get("scope_hierarchy") or ("session", "user", "global"))),
        default_write_scope_level=str(raw.get("default_write_scope_level", "user")),
        ambient_token_budget=int(raw.get("ambient_token_budget", 512)),
        ambient_top_k=int(raw.get("ambient_top_k", 6)),
        enable_ambient_llm=bool(raw.get("enable_ambient_llm", False)),
        max_writes_per_window=int(raw.get("max_writes_per_window", 128)),
        write_window_seconds=float(raw.get("write_window_seconds", 3600.0)),
        max_question_llm_calls=int(raw.get("max_question_llm_calls", 2)),
        recall=recall_cfg,
    )


def load_profile(spec: str | Path | MemoryProfile | None = None) -> MemoryProfile:
    """Load a declarative MemoryProfile by preset name, YAML/JSON path, or existing instance."""
    if isinstance(spec, MemoryProfile):
        return spec
    if spec is None:
        return load_profile("conversational")

    path_candidate = Path(spec)
    if path_candidate.is_file():
        text = path_candidate.read_text(encoding="utf-8")
        if path_candidate.suffix.casefold() == ".json":
            return _profile_from_dict(json.loads(text))
        return _profile_from_dict(parse_simple_yaml(text))

    raw_name = str(spec).strip()
    canonical = _PROFILE_ALIASES.get(raw_name.casefold(), raw_name.casefold())
    pdir = profiles_dir()
    if pdir is not None:
        for suffix in (".yaml", ".yml", ".json"):
            file_path = pdir / f"{canonical}{suffix}"
            if file_path.is_file():
                text = file_path.read_text(encoding="utf-8")
                parsed = json.loads(text) if suffix == ".json" else parse_simple_yaml(text)
                return _profile_from_dict(parsed)

    if canonical in _BUILTIN_DEFAULTS:
        return _profile_from_dict(_BUILTIN_DEFAULTS[canonical])
    raise ValueError(f"unknown memory profile or missing profile config file: {spec!r}")


def list_profiles() -> list[str]:
    """List available built-in and packaged declarative profile names."""
    names = set(_BUILTIN_DEFAULTS.keys())
    pdir = profiles_dir()
    if pdir is not None:
        for child in pdir.iterdir():
            if child.is_file() and child.suffix.casefold() in (".yaml", ".yml", ".json"):
                names.add(child.stem)
    return sorted(names)
