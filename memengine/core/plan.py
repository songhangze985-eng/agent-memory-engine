"""QueryPlan and RecallConfig (feature-flagged bipolar ring, recency tie-breaker, coding profile)."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Literal

from memengine.chrono import TemporalSpan, resolve_temporal_spans
from memengine.identifiers import extract_identifiers
from memengine.queryparse import parse_query_rules
from memengine.temporal import TemporalIntent, parse_temporal_intent
from memengine.versioning import extract_versions

_UPDATE_CUE = re.compile(
    r"\b(?:current|currently|now|latest|most recent|new|newest|updated|still)\b|"
    r"目前|现在|最新|当前|改成了|换成",
    re.I,
)
_AGGREGATE_CUE = re.compile(
    r"\b(?:how many|how much|total|all|both|every|list|count|number of|altogether|combined)\b|"
    r"一共|总共|多少个|多少次|列出|全部|所有",
    re.I,
)
_TEMPORAL_MATH_CUE = re.compile(
    r"\b(?:how many days|how many weeks|how many months|how long|days between|weeks between|"
    r"months between|before|after|earlier|later|first|last|when did|what date|which came first)\b|"
    r"几天|几周|几个月|多久|相差|之前|之后|什么时候|哪天|最早|最晚",
    re.I,
)
_CODING_CUE = re.compile(
    r"\b(?:rule|rules|strategy|strategies|guide|playbook|workflow|runbook|pitfall|avoid|constraint|config|deployment|migration|sqlite|postgres|api|error|exception|stack\s*trace)\b|"
    r"规则|避坑|指南|策略|配置|报错|异常|规范",
    re.I,
)


@dataclass(frozen=True)
class RecallConfig:
    """Explicit feature flags so ablation can keep or drop every ranking component."""

    enable_bipolar: bool = True
    enable_recency_rerank: bool = False  # Tie-breaker ONLY on non-historical, non-temporal queries
    enable_episode_fts: bool = True  # Zero-vector L0 turn FTS5 fallback channel
    enable_entity_graph: bool = True
    profile: Literal["conversational", "coding_agent"] = "conversational"
    disclosure: Literal["full", "progressive"] = "full"


@dataclass(frozen=True)
class QueryPlan:
    query: str
    parsed: dict
    temporal: TemporalIntent
    as_of: float | None
    windows: tuple[TemporalSpan, ...] = ()
    identifiers: tuple = ()
    versions: tuple = ()
    is_historical: bool = False
    is_temporal: bool = False
    is_update_query: bool = False
    is_aggregate_query: bool = False
    is_coding_query: bool = False
    exclude_superseded: bool = True
    allow_recency_tiebreak: bool = False
    profile: Literal["conversational", "coding_agent"] = "conversational"
    extra: dict = field(default_factory=dict)


def build_query_plan(
    query: str,
    *,
    reference_time: float | None = None,
    config: RecallConfig | None = None,
) -> QueryPlan:
    cfg = config or RecallConfig()
    effective_time = time.time() if reference_time is None else float(reference_time)
    parsed = parse_query_rules(query)
    temporal = parse_temporal_intent(query, effective_time)
    windows = tuple(resolve_temporal_spans(query, effective_time))
    identifiers = tuple(extract_identifiers(query))
    versions = tuple(extract_versions(query))

    is_historical = bool(temporal.is_historical or parsed.get("query_type") == "TIME_TRAVEL")
    is_temporal = bool(
        is_historical
        or windows
        or temporal.start_ts is not None
        or temporal.end_ts is not None
        or _TEMPORAL_MATH_CUE.search(query or "")
    )
    is_update = bool(_UPDATE_CUE.search(query or "")) and not is_historical
    is_aggregate = bool(_AGGREGATE_CUE.search(query or ""))
    is_coding = cfg.profile == "coding_agent" or bool(identifiers or versions or _CODING_CUE.search(query or ""))

    # Hard rule from §2: recency is NEVER applied on historical or temporal queries!
    allow_recency = bool(cfg.enable_recency_rerank and not is_historical and not is_temporal)
    exclude_superseded = not is_historical

    return QueryPlan(
        query=query,
        parsed=parsed,
        temporal=temporal,
        as_of=reference_time,
        windows=windows,
        identifiers=identifiers,
        versions=versions,
        is_historical=is_historical,
        is_temporal=is_temporal,
        is_update_query=is_update,
        is_aggregate_query=is_aggregate,
        is_coding_query=is_coding,
        exclude_superseded=exclude_superseded,
        allow_recency_tiebreak=allow_recency,
        profile=cfg.profile,
    )
