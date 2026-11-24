"""Deterministic validity intervals for explicitly temporary memories."""
import math
import re
from datetime import datetime, timedelta, timezone

from agentmem_core.textutil import parse_cn_number

_TEMPORARY = re.compile(r"临时|暂时|禁饮|禁食|忌口|避免|吃药|\b(?:temporar\w+|avoid|refrain)\b", re.I)
_DURATION = re.compile(r"(?:未来|接下来)\s*([\d一二两三四五六七八九十]+)\s*(天|周)|"
                       r"([\d一二两三四五六七八九十]+)\s*(天|周)内|"
                       r"\b(?:for the next|for)\s+(\d+)\s+(days?|weeks?)\b", re.I)


def interval(text, now, *, ttl_seconds=None, expires_at=None, valid_from=None):
    if ttl_seconds is not None and expires_at is not None:
        raise ValueError("provide ttl_seconds or expires_at")
    start = now if valid_from is None else float(valid_from)
    if ttl_seconds is not None:
        expires_at = start + float(ttl_seconds)
    # A duration and a temporary directive must describe the same clause.
    if expires_at is None:
        text = next((clause for clause in re.split(r"[。！？!?；;\n]+|\.(?=\s)", text)
                     if _TEMPORARY.search(clause) and (
                         _DURATION.search(clause) or any(cue in clause for cue in ("这周", "本周", "临时改到", "暂时改到")))), "")
    if expires_at is None and text:
        duration = _DURATION.search(text)
        if duration:
            number = duration[1] or duration[3] or duration[5]
            amount = parse_cn_number(number)
            unit = duration[2] or duration[4] or duration[6]
            expires_at = start + amount * (7 if unit == "周" or unit.startswith("week") else 1) * 86400
        elif "这周" in text or "本周" in text:
            day = datetime.fromtimestamp(start, timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            expires_at = (day + timedelta(days=7-day.weekday())).timestamp()
        elif "临时改到" in text or "暂时改到" in text:
            day = datetime.fromtimestamp(start, timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            weekday = re.search(r"(下)?周([一二三四五六日天])", text)
            if weekday:
                target = min(6, "一二三四五六日天".index(weekday[2]))
                delta = target - day.weekday()
                delta = delta + 7 if weekday[1] else delta % 7
                expires_at = (day + timedelta(days=delta+1)).timestamp()
            elif "明天" in text:
                expires_at = (day + timedelta(days=2)).timestamp()
            else:
                expires_at = start + 86400
    if expires_at is None:
        return None, None
    end = float(expires_at)
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("expires_at must be finite and follow valid_from")
    return start, end
