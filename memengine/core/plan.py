"""QueryPlan and RecallConfig with as_of, time-window, historical/update detection & recency tie-breaker (M2)."""
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
    r"\b(?:current|currently|now|latest|most recent|new|newest|updated|still|so far)\b|"
    r"目前|现在|最新|当前|改成了|换成",
    re.I,
)
_HISTORICAL_BEFORE_CUE = re.compile(
    r"\b(?:before\s+(?:getting|buying|purchasing|switching|moving|changing|starting|the)|prior\s+to|"
    r"originally|used\s+to|previously|formerly|earlier\s+version|old\s+(?:limit|job|address|budget)|"
    r"when\s+i\s+(?:just\s+|first\s+)?started)\b|"
    r"之前|原本|以前|原先|最初|修改前|刚开始",
    re.I,
)
_AS_OF_CUE = re.compile(r"\b(?:as\s+of|by\s+the\s+end\s+of|back\s+in|at\s+the\s+time\s+of)\b|截至", re.I)
_CHANGE_DIR_CUE = re.compile(r"\b(?:increase\s+or\s+decrease|increased\s+or\s+decreased|more\s+or\s+less|higher\s+or\s+lower|go\s+up\s+or\s+down)\b|增加还是减少|涨了还是跌了", re.I)
_AGGREGATE_CUE = re.compile(
    r"\b(?:how many|how much|total|all|both|every|list|count|number of|altogether|combined|compared to|average)\b|"
    r"一共|总共|多少个|多少次|列出|全部|所有",
    re.I,
)
_MULTI_HOP_CUE = re.compile(
    r"\b(?:what\s+are\b|what\s+(?:types?|kinds?|styles?|activities|hobbies|places|books|movies|names|things|events|sports)\b|"
    r"which\s+(?:countries|cities|places|events|books|movies|people)\b|where\s+(?:has|have)\b|"
    r"names?\s+of\b|both\b|compared\s+to\b|\band\s+(?:her|his|their|my)\b)",
    re.I,
)
_ORDER_FIRST_CUE = re.compile(r"\b(?:which\s+(?:event\s+)?(?:happened|came|occurred|started|was)\s+first|happened\s+earlier|came\s+earlier)\b|哪件事先发生|哪个更早", re.I)
_ORDER_LAST_CUE = re.compile(r"\b(?:which\s+(?:event\s+)?(?:happened|came|occurred)\s+last|happened\s+most\s+recently)\b|哪个最晚", re.I)
_DURATION_CUE = re.compile(
    r"\b(?:how\s+many\s+(?:days|weeks|months|years)\s+(?:ago|passed|between|since|before|after|had\s+passed|have\s+passed)|"
    r"how\s+long\s+(?:ago|between|had\s+i|have\s+i|did\s+it\s+take))\b|几天前|相差几天|隔了几天|多久前|多久了",
    re.I,
)
_DATE_ASK_CUE = re.compile(r"\b(?:when\s+did|when\s+was|when\s+will|what\s+date|what\s+day|what\s+month|what\s+year)\b|什么时候|哪天|哪一天|几月|哪一年", re.I)
_CODING_CUE = re.compile(
    r"\b(?:rule|rules|strategy|strategies|guide|playbook|workflow|runbook|pitfall|avoid|constraint|config|deployment|migration|sqlite|postgres|api|error|exception|stack\s*trace)\b|"
    r"规则|避坑|指南|策略|配置|报错|异常|规范",
    re.I,
)


@dataclass(frozen=True)
class RecallConfig:
    """Explicit feature flags; M5 end-to-end ablation pruned bipolar & recency defaults."""

    enable_bipolar: bool = False  # M5 ablation: disabling saves complexity, 0.00pp QA drop, +0.0089 MemBench MRR@10
    enable_recency_rerank: bool = False  # M5 ablation: disabling yields 0.00pp QA/R@10 drop, +0.0025 MemBench MRR@10
    enable_episode_fts: bool = True
    enable_entity_graph: bool = True
    enable_slot_supersede: bool = True
    enable_graph_enum: bool = True
    enable_deterministic_calc: bool = True
    profile: Literal["conversational", "coding_agent"] = "conversational"
    disclosure: Literal["full", "progressive"] = "full"


@dataclass(frozen=True)
class QueryPlan:
    query: str
    parsed: dict
    temporal: TemporalIntent
    as_of: float | None
    explicit_as_of: float | None = None
    time_window: tuple[float | None, float | None] = (None, None)
    windows: tuple[TemporalSpan, ...] = ()
    identifiers: tuple = ()
    versions: tuple = ()
    is_historical: bool = False
    is_temporal: bool = False
    is_update_query: bool = False
    is_aggregate_query: bool = False
    is_multi_hop_query: bool = False
    is_coding_query: bool = False
    change_direction_ask: bool = False
    temporal_order_ask: Literal["first", "last", "duration", "date", None] = None
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

    before_cue = bool(_HISTORICAL_BEFORE_CUE.search(query or ""))
    change_dir = bool(_CHANGE_DIR_CUE.search(query or ""))
    is_historical = bool(
        temporal.is_historical
        or parsed.get("query_type") == "TIME_TRAVEL"
        or before_cue
    )

    # Resolve explicit temporal_order_ask
    temporal_order: Literal["first", "last", "duration", "date", None] = None
    if _ORDER_FIRST_CUE.search(query or ""):
        temporal_order = "first"
    elif _ORDER_LAST_CUE.search(query or ""):
        temporal_order = "last"
    elif _DURATION_CUE.search(query or ""):
        temporal_order = "duration"
    elif _DATE_ASK_CUE.search(query or ""):
        temporal_order = "date"

    # Compute explicit time window from chrono spans or temporal intent
    win_start = temporal.start_ts
    win_end = temporal.end_ts
    if windows and win_start is None and win_end is None:
        starts = [w.start for w in windows if w.start is not None]
        ends = [w.end for w in windows if w.end is not None]
        if starts:
            win_start = min(starts)
        if ends:
            win_end = max(ends)

    # Explicit past as_of point ("as of March 2024", "back in 2022")
    explicit_as_of: float | None = None
    if _AS_OF_CUE.search(query or "") and win_end is not None and win_end < effective_time:
        explicit_as_of = win_end
        is_historical = True

    is_temporal = bool(
        is_historical
        or windows
        or win_start is not None
        or win_end is not None
        or temporal_order is not None
    )
    is_update = bool(_UPDATE_CUE.search(query or "") or change_dir) and not is_historical
    is_aggregate = bool(_AGGREGATE_CUE.search(query or "")) and temporal_order != "duration"
    is_multi_hop = bool(_MULTI_HOP_CUE.search(query or ""))
    is_coding = cfg.profile == "coding_agent" or bool(identifiers or versions or _CODING_CUE.search(query or ""))

    # Strict M2 invariant: recency is ONLY a tie-breaker on non-historical, non-temporal queries!
    allow_recency = bool(cfg.enable_recency_rerank and not is_historical and not is_temporal)
    exclude_superseded = not is_historical and not change_dir

    return QueryPlan(
        query=query,
        parsed=parsed,
        temporal=temporal,
        as_of=effective_time,
        explicit_as_of=explicit_as_of,
        time_window=(win_start, win_end),
        windows=windows,
        identifiers=identifiers,
        versions=versions,
        is_historical=is_historical,
        is_temporal=is_temporal,
        is_update_query=is_update,
        is_aggregate_query=is_aggregate,
        is_multi_hop_query=is_multi_hop,
        is_coding_query=is_coding,
        change_direction_ask=change_dir,
        temporal_order_ask=temporal_order,
        exclude_superseded=exclude_superseded,
        allow_recency_tiebreak=allow_recency,
        profile=cfg.profile,
    )
