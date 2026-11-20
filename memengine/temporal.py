"""Deterministic query-time and event-time parsing in UTC."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import combinations
from math import sqrt

from memengine.textutil import normalize_entity, parse_before_ordinal, parse_cn_number, strip_step_tags, tokenize_for_fts
from memengine.vector import _QUERY_STOP

_DATE = re.compile(r"(?<!\d)((?:19|20)\d\d)[-/](\d{1,2})[-/](\d{1,2})(?!\d)")
_YEAR = re.compile(r"(?<![\w.-])((?:19|20)\d\d)(?![\d./-])|((?:19|20)\d\d)年")
_BEFORE_EN = re.compile(r"\bbefore\s+(?:the\s+)?(\d+|first|second|third|fourth|fifth)(?:st|nd|rd|th)?\s+"
                       r"(?:change|update|revision|edit)", re.I)
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
_EARLIEST = re.compile(r"最早|第一次|原本|\b(?:earliest|originally|initially|first time|initial)\b", re.I)
_LATEST = re.compile(r"最近|最新|当前|现在|\b(?:latest|current|most recent|now)\b", re.I)
_PREVIOUS = re.compile(r"之前|以前|上次|\b(?:previously|earlier|last time|used to|before|formerly)\b", re.I)
_WEEKDAY = {char: index for index, char in enumerate("一二三四五六日")}
_CONVERSATIONAL = frozenset("just really also still want think have has had about now recently currently new old "
                            "that they there been get got very going will time update changed".split())
_STATE_CUE = re.compile(r"现在|改为|搬到|搬去|更新|不再|更换|住在|工作于|现任|"
                        r"\b(?:current(?:ly)?|now|update[ds]?|switch(?:ed)?|chang(?:ed|e)|no longer|instead|"
                        r"moved|live[sd]? in|living in|works? (?:at|for)|used to|replac\w*|anymore|new job)\b", re.I)
_QUANTITY_QUERY = re.compile(r"多少|几(?:个|次|小时|年|月|部)|多久|\bhow (?:many|much|long|often)\b", re.I)
_CLAUSES = re.compile(r"[!?;。！？；\n]|\.(?=\s)|\s[-–—]\s")
_QUANTITY = re.compile(r"\b(?:\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b", re.I)
_NUMBER_WORDS = {name: str(index) for index, name in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}


def competition_terms(text):
    return {word for word in tokenize_for_fts(text).split()
            if word not in _QUERY_STOP and word not in _CONVERSATIONAL and (word.isascii() or len(word) > 1)}


def _measurement(text, target):
    if not target:
        return set(), set()
    clauses = _CLAUSES.split(text)
    choices = []
    for clause in clauses:
        # A number already named in the question can identify the subject rather than measure it.
        values = {_NUMBER_WORDS.get(word.lower(), word) for word in _QUANTITY.findall(clause)} - target
        if values:
            topics = competition_terms(clause) & target
            choices.append((topics, values))
    return max(choices, key=lambda item: len(item[0]), default=(set(), set()))


def temporal_competition(candidates, session_orders, query="", scores=None):
    """Resolve competing versions without changing either stored fact."""
    pool = candidates[:30]
    relevance = {fact["id"]: score for fact, score in zip(candidates, scores)} if scores is not None else {}
    clean = {fact["id"]: re.sub(r"^(?:user|assistant):\s*", "", strip_step_tags(fact["plain"]), flags=re.I) for fact in pool}
    terms = {key: competition_terms(body) for key, body in clean.items()}
    focus = competition_terms(query) if _QUANTITY_QUERY.search(query) else set()
    measurements = {key: _measurement(body, focus) for key, body in clean.items()}
    entities = {fact["id"]: {normalize_entity(key) for key in fact["entity_z_keys"]} |
                {key for key, kind in fact["identifiers"].items() if kind == "project_token"} for fact in pool}
    older, newer = set(), set()

    def position(fact):
        return (fact["event_time_start"], session_orders.get(fact["session_id"], 0),
                max(fact["source_turn_ids"], default=0), fact["ingest_time"])

    for a, b in combinations(pool, 2):
        aid, bid = a["id"], b["id"]
        linked = a.get("supersedes_id") == bid or b.get("supersedes_id") == aid
        av = {key for key, kind in a["identifiers"].items() if kind == "version"}
        bv = {key for key, kind in b["identifiers"].items() if kind == "version"}
        if not linked and av and bv and av.isdisjoint(bv):
            continue
        if not linked and (a["role"] != b["role"] or a.get("epistemic_scope") != b.get("epistemic_scope")):
            continue
        overlap = terms[aid] & terms[bid]
        similarity = len(overlap) / sqrt(max(1, len(terms[aid]) * len(terms[bid])))
        attribute = a["attribute"]
        shared_attribute = bool(attribute and not attribute.startswith(("turn:", "event:")) and attribute == b["attribute"])
        shared_entity = bool(entities[aid] & entities[bid])
        different_source = (a["session_id"] != b["session_id"] or
                            bool(a["source_turn_ids"] and b["source_turn_ids"]
                                 and set(a["source_turn_ids"]).isdisjoint(b["source_turn_ids"])))
        a_topics, a_values = measurements[aid]
        b_topics, b_values = measurements[bid]
        # Long conversational turns often bury a revised count in a single clause.
        # Compare the query-aligned portions only when both report conflicting quantities.
        common_topics = a_topics & b_topics
        quantity_slot = (len(common_topics) >= 2 and min(len(a_topics), len(b_topics)) >= max(2, len(focus) * 0.5)
                         and a_values.isdisjoint(b_values))
        if quantity_slot:
            similarity = max(similarity, len(common_topics) / sqrt(len(a_topics) * len(b_topics)))
        stateful = quantity_slot or bool(_STATE_CUE.search(clean[aid]) or _STATE_CUE.search(clean[bid]))
        if not (linked or shared_entity and (shared_attribute or stateful and similarity >= 0.52)
                or not shared_entity and stateful and different_source and len(overlap) >= 2 and similarity >= 0.68):
            continue
        first, second = position(a), position(b)
        if first != second:
            old, new = (aid, bid) if first < second else (bid, aid)
            if new in relevance and relevance[new] < 0.80 * relevance[old]:
                continue
            older.add(old)
            newer.add(new)
    return {**dict.fromkeys(newer, "latest"), **dict.fromkeys(older, "older")}


@dataclass(frozen=True)
class TemporalIntent:
    is_historical: bool = False
    start_ts: float | None = None
    end_ts: float | None = None
    order_hint: str | None = None
    rev_index: int | None = None


def _month(date, offset):
    year, month = divmod(date.year * 12 + date.month - 1 + offset, 12)
    return date.replace(year=year, month=month + 1, day=1)


def _interval(text, now):
    day = datetime.fromtimestamp(now, timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    match = _DATE.search(text)
    if match:
        try:
            start = day.replace(year=int(match[1]), month=int(match[2]), day=int(match[3]))
        except ValueError:
            return None, None
        return start, start + timedelta(days=1)
    match = _YEAR.search(text)
    if match or re.search(r"去年|\blast year\b", text, re.I):
        year = int(match[1] or match[2]) if match else day.year - 1
        start = day.replace(year=year, month=1, day=1)
        return start, start.replace(year=year + 1)
    if re.search(r"昨天|\byesterday\b", text, re.I):
        return day - timedelta(days=1), day
    if re.search(r"明天|\btomorrow\b", text, re.I):
        return day + timedelta(days=1), day + timedelta(days=2)
    for offset, pattern in ((-1, r"上个?月|\blast month\b"), (1, r"下个?月|\bnext month\b")):
        if re.search(pattern, text, re.I):
            return _month(day, offset), _month(day, offset + 1)
    week = re.search(r"([上下])周([一二三四五六日天])?|\b(last|next) week\b", text, re.I)
    if week:
        offset = -7 if week[1] == "上" or (week[3] or "").lower() == "last" else 7
        start = day + timedelta(days=offset - day.weekday())
        if week[2]:
            start += timedelta(days=_WEEKDAY.get(week[2], 6))
            return start, start + timedelta(days=1)
        return start, start + timedelta(days=7)
    return None, None


def parse_temporal_intent(query: str, now: float) -> TemporalIntent:
    index = parse_before_ordinal(query)
    if index is not None:
        return TemporalIntent(True, order_hint="before_n", rev_index=index)
    english = _BEFORE_EN.search(query)
    if english:
        number = english[1].lower()
        return TemporalIntent(True, order_hint="before_n", rev_index=max(0, _ORDINALS.get(number, parse_cn_number(number)) - 1))
    start, end = _interval(query, now)
    earliest = bool(_EARLIEST.search(query))
    previous = bool(_PREVIOUS.search(query))
    if start is not None:
        stamp, end_stamp = start.timestamp(), end.timestamp()
        if re.search(r"\bbefore\b|之前|以前", query, re.I):
            return TemporalIntent(True, end_ts=stamp, order_hint="latest")
        return TemporalIntent(stamp <= now, stamp, end_stamp, "earliest" if earliest else None)
    if earliest:
        return TemporalIntent(True, order_hint="earliest")
    if previous:
        return TemporalIntent(True, order_hint="latest")
    return TemporalIntent(order_hint="latest" if _LATEST.search(query) else None)


def absolute_event_text(text: str, now: float) -> tuple[str, float | None]:
    relative = re.compile(r"昨天|明天|[上下]周[一二三四五六日天]?|[上下]个?月|去年|"
                          r"\b(?:yesterday|tomorrow|last year|(?:last|next) (?:week|month))\b", re.I)
    intervals = []

    def replace(match):
        start, end = _interval(match[0], now)
        if start is None:
            return match[0]
        intervals.append(start.timestamp())
        if end - start == timedelta(days=1):
            return start.strftime("%Y-%m-%d")
        return f"{start:%Y-%m-%d}–{end - timedelta(days=1):%Y-%m-%d}"

    rewritten = relative.sub(replace, text)
    if not intervals:
        start, _ = _interval(text, now)
        return rewritten, start.timestamp() if start else None
    return rewritten, intervals[0]
