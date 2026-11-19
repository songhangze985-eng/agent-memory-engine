"""Deterministic query-time and event-time parsing in UTC."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from memengine.textutil import parse_cn_number

_DATE = re.compile(r"(?<!\d)((?:19|20)\d\d)[-/](\d{1,2})[-/](\d{1,2})(?!\d)")
_YEAR = re.compile(r"(?<![\w.-])((?:19|20)\d\d)(?![\d./-])|((?:19|20)\d\d)年")
_BEFORE_N = re.compile(r"第\s*([\d一二两三四五六七八九十]+)\s*次.{0,12}(?:之前|以前)|"
                       r"\bbefore\s+(?:the\s+)?(\d+|first|second|third|fourth|fifth)(?:st|nd|rd|th)?\s+"
                       r"(?:change|update|revision|edit)", re.I)
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
_EARLIEST = re.compile(r"最早|第一次|原本|\b(?:earliest|originally|initially|first time|initial)\b", re.I)
_LATEST = re.compile(r"最近|最新|当前|现在|\b(?:latest|current|most recent|now)\b", re.I)
_PREVIOUS = re.compile(r"之前|以前|上次|\b(?:previously|earlier|last time|used to|before|formerly)\b", re.I)
_WEEKDAY = {char: index for index, char in enumerate("一二三四五六日")}


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
    ordinal = _BEFORE_N.search(query)
    if ordinal:
        number = ordinal[1] or ordinal[2].lower()
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
