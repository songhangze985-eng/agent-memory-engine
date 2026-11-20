"""UTC calendar parsing and closed, second-resolution occurrence intervals."""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from memengine.textutil import parse_cn_number

UTC = timezone.utc
DAY = 86400
_MONTH = {name.casefold(): i for i, name in enumerate(calendar.month_name) if i}
_MONTH.update({name.casefold(): i for i, name in enumerate(calendar.month_abbr) if i})
_MONTH_RE = "|".join(sorted(_MONTH, key=len, reverse=True))
_WEEKDAYS = "|".join(calendar.day_name)
_WORD = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10, "a": 1, "an": 1}


@dataclass(frozen=True)
class TemporalSpan:
    start: float | None
    end: float | None
    precision: str
    text: str

    @property
    def iso_start(self):
        return date_iso(self.start)

    @property
    def iso_end(self):
        return date_iso(self.end)


def date_iso(stamp):
    return None if stamp is None else datetime.fromtimestamp(stamp, UTC).date().isoformat()


def parse_session_clock(value):
    if value is None or value == "":
        return None
    if type(value) in (int, float):
        return datetime.fromtimestamp(value, UTC).timestamp()
    if isinstance(value, datetime):
        stamp = value
    else:
        text = re.sub(r"\s*\([A-Za-z]{3,9}\)", "", value.strip())
        text = re.sub(r"^(\d{4})年(\d{1,2})月(\d{1,2})日$", r"\1/\2/\3", text)
        try:
            stamp = datetime.fromisoformat(text.removesuffix("Z") + ("+00:00" if text.endswith("Z") else ""))
        except ValueError:
            for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y", "%Y/%m/%d %H:%M:%S",
                        "%Y/%m/%d %H:%M", "%Y/%m/%d", "%Y-%m-%d", "%B %d, %Y"):
                try:
                    stamp = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"unsupported session calendar: {value!r}")
    return (stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC)).timestamp()


def _month(day, months):
    year, index = divmod(day.year * 12 + day.month - 1 + months, 12)
    return day.replace(year=year, month=index + 1, day=min(day.day, calendar.monthrange(year, index + 1)[1]))


def _number(text):
    if text in _WORD:
        return _WORD[text]
    if text == "零":
        return 0
    return parse_cn_number(text)


def _bounds(day, unit):
    if unit == "day":
        return day, day + timedelta(days=1)
    if unit == "week":
        start = day - timedelta(days=day.weekday())
        return start, start + timedelta(days=7)
    if unit == "month":
        start = day.replace(day=1)
        return start, _month(start, 1)
    start = day.replace(month=1, day=1)
    return start, start.replace(year=start.year + 1)


_ISO = r"(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?)?"
_CALENDAR = re.compile(
    rf"(?<![\d.])(?P<iso>{_ISO})(?![\d]|\.\d)|"
    rf"(?P<ymd>(?:(?:19|20)\d{{2}}年)?\d{{1,2}}月(?:\d{{1,2}}[日号])?)|"
    rf"(?P<month>\b(?:{_MONTH_RE})\.?\s+(?:\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?|\d{{4}})\b)|"
    rf"(?P<daymonth>\b\d{{1,2}}\s+(?:{_MONTH_RE})(?:,?\s+(?:19|20)\d{{2}})?\b)|"
    r"(?P<year>(?<![\w.-])(?:19|20)\d{2}(?![\d./-])(?:年)?)|"
    rf"(?P<engrel>\b(?:last night|yesterday|today|tomorrow|(?:last|this|next)\s+(?:week|month|year|spring|summer|fall|autumn|winter|{_WEEKDAYS}|{_MONTH_RE})|"
    rf"(?:\d+|{'|'.join(_WORD)})\s+(?:days?|weeks?|months?|years?)\s+ago|in\s+(?:\d+|{'|'.join(_WORD)})\s+(?:days?|weeks?))\b)|"
    r"(?P<zhrel>昨天|昨晚|前天|今天|明天|后天|去年|今年|明年|(?:上|这|本|下)周[一二三四五六日天]?|(?:上|这|本|下)个?月|"
    r"[\d零一二两三四五六七八九十]+\s*(?:天|周|个?月|年)前)", re.I)
_QUALIFIER = re.compile(r"(?:\b(before|after|since)\s+|(自|截至)\s*)$", re.I)
_UNRESOLVED = re.compile(r"\b(?:recently|the other day|someday|a while ago|last (?:christmas|holiday))\b|前些天|最近|前阵子", re.I)
_TEMPORAL = re.compile(r"\b(?:when|how long ago|how much time|how many (?:days|weeks|months|years)|between|earliest|latest)\b|多久|何时|哪天|什么时候|相隔|最早", re.I)


def _relative(term, day):
    simple = {"yesterday": -1, "today": 0, "tomorrow": 1, "last night": -1, "昨天": -1, "昨晚": -1,
              "前天": -2, "今天": 0, "明天": 1, "后天": 2}
    if term in ("last night", "昨晚"):
        return (day - timedelta(hours=6), day + timedelta(hours=6)), "day"
    if term in simple:
        return _bounds(day + timedelta(days=simple[term]), "day"), "day"
    if term in ("去年", "今年", "明年"):
        return _bounds(_month(day, 12 * ("去年", "今年", "明年").index(term) - 12), "year"), "year"
    if re.match(r"^[上这本下]", term):
        offset = {"上": -1, "这": 0, "本": 0, "下": 1}[term[0]]
        if term[1] == "周":
            week = _bounds(day + timedelta(weeks=offset), "week")
            if len(term) == 2:
                return week, "week"
            return _bounds(week[0] + timedelta(days=min(6, "一二三四五六日天".index(term[-1]))), "day"), "day"
        return _bounds(_month(day, offset), "month"), "month"
    if (match := re.fullmatch(r"([\d零一二两三四五六七八九十]+)\s*(天|周|个?月|年)前", term)):
        parts = (_number(match[1]), {"天": "day", "周": "week", "月": "month", "个月": "month", "年": "year"}[match[2]], -1)
    elif (match := re.fullmatch(r"(in )?(\d+|[a-z]+)\s+(day|week|month|year)s?(?: ago)?", term)) and (match[2].isdigit() or match[2] in _WORD):
        parts = (_number(match[2]), match[3], 1 if match[1] else -1)
    else:
        direction, unit = term.split()
        offset = {"last": -1, "this": 0, "next": 1}[direction]
        if unit in _MONTH:
            month = _MONTH[unit]
            year = day.year - (offset == -1 and month >= day.month) + (offset == 1 and month <= day.month)
            return _bounds(day.replace(year=year, month=month, day=1), "month"), "month"
        if unit.title() in calendar.day_name:
            weekday = list(calendar.day_name).index(unit.title())
            delta = (day.weekday() - weekday) % 7 or 7
            target = day - timedelta(days=delta) if offset == -1 else (
                day + timedelta(days=(weekday - day.weekday()) % 7 or 7) if offset == 1 else
                day + timedelta(days=weekday - day.weekday()))
            return _bounds(target, "day"), "day"
        if unit in ("spring", "summer", "fall", "autumn", "winter"):
            month = {"spring": 3, "summer": 6, "autumn": 9, "fall": 9, "winter": 12}[unit]
            pivot = day.year - int(unit == "winter" and day.month < 3)
            first = day.replace(year=pivot, month=month, day=1)
            if offset and ((first <= day and offset == 1) or (first > day and offset == -1)):
                first = first.replace(year=first.year + offset)
            elif offset == -1 and first <= day < _month(first, 3):
                first = first.replace(year=first.year - 1)
            if offset == 1 and first <= day:
                first = first.replace(year=first.year + 1)
            return (first, _month(first, 3)), "month"
        parts = (1, unit, offset)
        if unit == "week":
            return _bounds(day + timedelta(weeks=offset), "week"), "week"
        if unit in ("month", "year"):
            return _bounds(_month(day, offset * (12 if unit == "year" else 1)), unit), unit
    count, unit, sign = parts
    step = count * sign
    value = (_month(day, step * (12 if unit == "year" else 1)) if unit in ("month", "year")
             else day + timedelta(days=step * (7 if unit == "week" else 1)))
    return _bounds(value, "day"), "day"


def _absolute(match, day):
    token = match[0]
    base = day or datetime(2000, 1, 1, tzinfo=UTC)
    if match.lastgroup == "iso":
        value = datetime.fromtimestamp(parse_session_clock(token), UTC)
        return ((value, value), "exact") if ":" in token else (_bounds(value, "day"), "day")
    if match.lastgroup == "year":
        return _bounds(base.replace(year=int(token[:4]), month=1, day=1), "year"), "year"
    values = [int(i) for i in re.findall(r"\d+", token)]
    year = next((i for i in values if i >= 1900), day.year if day is not None else None)
    if year is None:
        return (None, None), "relative_unresolved"
    nums = [i for i in values if i < 1900]
    if match.lastgroup == "ymd":
        month, *nums = nums
    else:
        month = _MONTH[re.search(_MONTH_RE, token, re.I)[0].casefold()]
    unit = "day" if nums else "month"
    return _bounds(base.replace(year=year, month=month, day=nums[0] if nums else 1), unit), unit


def resolve_temporal_spans(text, anchor_time):
    stamp = parse_session_clock(anchor_time)
    day = (datetime.fromtimestamp(stamp, UTC).replace(hour=0, minute=0, second=0, microsecond=0)
           if stamp is not None else None)
    found = []
    for match in _CALENDAR.finditer(text):
        try:
            if match.lastgroup in ("engrel", "zhrel"):
                (start, end), unit = _relative(match[0].casefold(), day) if day is not None else ((None, None), "relative_unresolved")
            else:
                (start, end), unit = _absolute(match, day)
        except (ValueError, OverflowError):
            continue
        first, last = (start.timestamp(), end.timestamp() - (unit != "exact")) if start is not None else (None, None)
        qualifier = _QUALIFIER.search(text[:match.start()])
        if qualifier and first is not None:
            op = (qualifier[1] or qualifier[2]).lower()
            first, last = ((None, first - 1) if op == "before" else (last + 1, None) if op == "after" else
                           (first, stamp) if op in ("since", "自") else (None, last))
        found.append((match.start(), match.end(), TemporalSpan(first, last, unit, match[0])))
    output = []
    i = 0
    while i < len(found):
        start, end, span = found[i]
        if i + 1 < len(found):
            other_start, other_end, other = found[i+1]
            if (re.fullmatch(r"\s*(?:to|through|until|and|至|到|—|–|-)\s*", text[end:other_start], re.I)
                    and re.search(r"\b(?:from|between)\s*$|从\s*$", text[:start], re.I)
                    and span.start is not None and other.end is not None and span.start <= other.end):
                output.append(TemporalSpan(span.start, other.end, other.precision, text[start:other_end]))
                i += 2
                continue
        output.append(span)
        i += 1
    if not output and (match := _UNRESOLVED.search(text)):
        output.append(TemporalSpan(None, None, "relative_unresolved", match[0]))
    return output


def temporal_query(text):
    return bool(_TEMPORAL.search(text))


def interval_score(start, end, query):
    if start is None or end is None:
        return 0.0
    lower = start if query.start is None else query.start
    upper = end if query.end is None else query.end
    common = max(0.0, min(end, upper) - max(start, lower) + 1)
    return common / max(1.0, min(end - start + 1, upper - lower + 1))


def days_between(first, second):
    if first.start is None or second.start is None:
        return None
    return abs((datetime.fromtimestamp(first.start, UTC).date() - datetime.fromtimestamp(second.start, UTC).date()).days)


def occurrence(text, spans):
    reporting = re.search(r"\b(?:mentioned|reported|recalled|noted|said)\b|提到|说起|回忆", text, re.I)
    if reporting:
        suffix = text[reporting.end():]
        event = next((span for span in spans if span.start is not None and span.text in suffix), None)
        if event:
            return event
    return next((span for span in spans if span.start is not None), spans[0] if spans else None)
