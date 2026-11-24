"""Deterministic calculation module for answer(): counts, sums, diffs, percentages & date arithmetic (M3)."""
from __future__ import annotations

import re

from agentmem_core.chrono import date_iso
from agentmem_core.core.graph_enum import extract_query_facets
from agentmem_core.core.supersede import _parse_num
from agentmem_core.core.text import light_stem, stem_terms
from agentmem_core.textutil import strip_step_tags

_SUM_CUE = re.compile(
    r"\b(?:total\s+(?:number\s+of|amount\s+of|hours|minutes|days|weeks|cost|views|people|distance)?|"
    r"in\s+total|combined|altogether|sum\s+of)\b|一共|总共|合计|总计",
    re.I,
)
_DIFF_CUE = re.compile(
    r"\b(?:how\s+much\s+(?:more|less|faster|slower|longer|shorter|older|younger|cheaper|higher|lower)|"
    r"how\s+many\s+(?:more|fewer|years\s+older|years\s+younger|minutes\s+faster|hours\s+longer)|"
    r"difference\s+between|compared\s+to)\b|多花|少花|快了多少|大几岁|相差多少",
    re.I,
)
_COUNT_CUE = re.compile(
    r"^\s*how\s+many\s+(?:different\s+)?([a-zA-Z0-9\s-]{2,32}?)\s+(?:have|did|do|has|are|were)\b|多少个|多少次|几次",
    re.I,
)
_MONEY_RE = re.compile(r"\$\s*(\d[\d,]*(?:\.\d+)?)")
_DURATION_UNIT_RE = re.compile(
    r"\b(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|half)\s*"
    r"(hours?|hrs?|minutes?|mins?|weeks?|days?|months?|years?)\b",
    re.I,
)
_LARGE_NUM_RE = re.compile(r"\b(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(k|thousand|million|views|people|reached|followers|subscribers|courses|classes|sessions|hours|minutes|dollars|bikes|tournaments|wins)?\b", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|\n+")


def _extract_number_with_unit(text: str, query: str) -> tuple[float | None, str]:
    """Extract the most query-aligned numeric quantity (with unit) from an evidence snippet."""
    q_low = (query or "").casefold()
    # 1. Currency questions ("spend", "cost", "$", "per night", "price")
    if re.search(r"\b(?:spend|spent|cost|price|paid|dollar|accommodations?)\b|\$", q_low):
        m = _MONEY_RE.search(text)
        if m:
            val = _parse_num(m.group(1))
            if val is not None:
                return val, "$"

    # 2. Explicit time/duration unit questions ("hours", "minutes", "weeks", "years")
    want_unit = None
    for u_pat, u_norm in (
        (r"\bhours?\b", "hours"),
        (r"\bminutes?\b|\bfaster\b|\bslower\b", "minutes"),
        (r"\bweeks?\b", "weeks"),
        (r"\byears?\b|\bolder\b|\byounger\b", "years"),
        (r"\bdays?\b", "days"),
    ):
        if re.search(u_pat, q_low):
            want_unit = u_norm
            break

    if want_unit:
        for m in _DURATION_UNIT_RE.finditer(text):
            raw_n, raw_u = m.group(1).casefold(), m.group(2).casefold()
            val = 0.5 if raw_n == "half" else _parse_num(raw_n)
            if val is None:
                continue
            u_norm = (
                "hours" if raw_u.startswith(("hour", "hr"))
                else "minutes" if raw_u.startswith("min")
                else "weeks" if raw_u.startswith("week")
                else "years" if raw_u.startswith("year")
                else "days" if raw_u.startswith("day")
                else raw_u
            )
            if u_norm == want_unit:
                return val, want_unit
            # Convert days <-> weeks when question asks weeks
            if want_unit == "weeks" and u_norm == "days":
                return round(val / 7.0, 2), "weeks"
            if want_unit == "minutes" and u_norm == "hours":
                return round(val * 60.0, 2), "minutes"
        # Also handle "MM:SS" race times like "25:00" or "25 minutes"
        if want_unit == "minutes":
            mmss = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
            if mmss:
                return float(int(mmss.group(1)) + int(mmss.group(2)) / 60.0), "minutes"
        if want_unit == "years":
            age = re.search(r"\b(?:i(?:'m| am)|turned|was)\s+(\d{2})\b", text, re.I)
            if age:
                return float(age.group(1)), "years"

    # 3. General counts / views / reach (ignore 4-digit years 1990..2030 unless asked)
    q_stems = stem_terms(query)
    best_val: float | None = None
    best_score = -1
    for m in _LARGE_NUM_RE.finditer(text):
        raw = m.group(1).replace(",", "")
        unit = (m.group(2) or "").casefold()
        val = _parse_num(raw)
        if val is None:
            continue
        if unit in {"k", "thousand"}:
            val *= 1000.0
        elif unit == "million":
            val *= 1_000_000.0
        # Skip calendar years (2020-2026) when question isn't asking for a year
        if 1990 <= val <= 2030 and not unit and "year" not in q_low and "," not in m.group(1):
            continue
        # Score proximity to query stems
        start = max(0, m.start() - 45)
        end = min(len(text), m.end() + 45)
        ctx_stems = stem_terms(text[start:end])
        score = len(q_stems & ctx_stems) * 10 + (5 if unit and light_stem(unit) in q_stems else 0) + (3 if "," in m.group(1) else 0)
        if score > best_score:
            best_score = score
            best_val = val
    return best_val, ""


_META_QUANT = frozenset({"differ", "distinct", "variou", "total", "number", "count", "typic", "combin", "altogeth", "sever"})


def _fmt_num(val: float, unit: str = "") -> str:
    if abs(val - round(val)) < 1e-6:
        iv = int(round(val))
        s = f"{iv:,}" if abs(iv) >= 1000 else str(iv)
    else:
        s = f"{val:,.2f}".rstrip("0").rstrip(".")
    if unit == "$":
        return f"${s}"
    return f"{s} {unit}".strip() if unit else s


def deterministic_calculate(query: str, rows: list[dict], bundle: dict) -> dict | None:
    """Deterministic arithmetic for multi-session sum, difference, and distinct-session counting.

    Returns ``None`` whenever the evidence does not cleanly support deterministic math so
    control falls through safely to the grounded extractive/LLM reader.
    """
    qplan = bundle.get("query_plan")
    if qplan is None or not rows:
        return None
    # Do not intercept temporal duration/order/date queries (handled by M2 temporal resolvers)
    if qplan.temporal_order_ask is not None or qplan.change_direction_ask:
        return None

    q = query.strip()
    q_stems = stem_terms(q)
    facets = extract_query_facets(q)

    # ---- 1. Deterministic Difference / Comparison ("how much more/faster/older ... compared to ...") ----
    if _DIFF_CUE.search(q):
        vals: list[tuple[str, float, str, dict]] = []
        used_sessions: set[str] = set()
        if len(facets) >= 2:
            for facet in facets[:2]:
                f_stems = stem_terms(facet)
                best_pair = None
                for row in rows[:20]:
                    if row.get("role") == "assistant":
                        continue
                    text = strip_step_tags(row.get("plain") or row.get("text") or "")
                    if not (f_stems & stem_terms(text)):
                        continue
                    v, u = _extract_number_with_unit(text, q)
                    if v is not None:
                        best_pair = (facet, v, u, row)
                        break
                if best_pair:
                    vals.append(best_pair)
        else:
            # Compare two distinct sessions matching query (e.g., this year's 5K vs previous year's 5K)
            for row in rows[:20]:
                if row.get("role") == "assistant":
                    continue
                sid = str(row.get("session_id") or row.get("evidence_id"))
                if sid in used_sessions:
                    continue
                text = strip_step_tags(row.get("plain") or row.get("text") or "")
                if len(q_stems & stem_terms(text)) < 2:
                    continue
                v, u = _extract_number_with_unit(text, q)
                if v is not None:
                    used_sessions.add(sid)
                    vals.append((sid, v, u, row))
                if len(vals) >= 2:
                    break
        if len(vals) == 2 and vals[0][1] != vals[1][1]:
            diff = abs(vals[0][1] - vals[1][1])
            unit = vals[0][2] or vals[1][2]
            ans_str = _fmt_num(diff, unit)
            resp = f"{ans_str} (comparing {_fmt_num(vals[0][1], unit)} and {_fmt_num(vals[1][1], unit)})."
            return {
                "response": resp,
                "claims": [{"text": resp, "evidence_ids": [vals[0][3]["evidence_id"], vals[1][3]["evidence_id"]]}],
                "detail": {"op": "diff", "values": [vals[0][1], vals[1][1]], "result": diff, "unit": unit},
            }

    # ---- 2. Deterministic Summation ("total number of ... on A and B", "hours in total combined") ----
    if _SUM_CUE.search(q):
        items_found: list[tuple[float, str, dict]] = []
        used_sessions: set[str] = set()
        if len(facets) >= 2:
            for facet in facets:
                f_stems = stem_terms(facet)
                for row in rows[:20]:
                    if row.get("role") == "assistant":
                        continue
                    text = strip_step_tags(row.get("plain") or row.get("text") or "")
                    if not (f_stems & stem_terms(text)):
                        continue
                    v, u = _extract_number_with_unit(text, q)
                    if v is not None:
                        sid = str(row.get("session_id") or row.get("evidence_id"))
                        used_sessions.add(sid)
                        items_found.append((v, u, row))
                        break
        else:
            for row in rows[:20]:
                if row.get("role") == "assistant":
                    continue
                sid = str(row.get("session_id") or row.get("evidence_id"))
                if sid in used_sessions:
                    continue
                text = strip_step_tags(row.get("plain") or row.get("text") or "")
                if len(q_stems & stem_terms(text)) < 2:
                    continue
                v, u = _extract_number_with_unit(text, q)
                if v is not None:
                    used_sessions.add(sid)
                    items_found.append((v, u, row))
                if len(items_found) >= 5:
                    break
        if len(items_found) >= 2:
            total = sum(v for v, _, _ in items_found)
            unit = next((u for _, u, _ in items_found if u), "")
            ans_str = _fmt_num(total, unit)
            parts = " + ".join(_fmt_num(v, unit) for v, _, _ in items_found)
            resp = f"{ans_str} in total ({parts})."
            return {
                "response": resp,
                "claims": [{"text": resp, "evidence_ids": [r["evidence_id"] for _, _, r in items_found]}],
                "detail": {"op": "sum", "values": [v for v, _, _ in items_found], "result": total, "unit": unit},
            }

    # ---- 3. Deterministic Distinct-Session Event Counting ("How many different doctors/dinner parties/tournaments...?") ----
    m_cnt = _COUNT_CUE.search(q)
    if m_cnt and not qplan.is_update_query:
        # Guard: if a single latest turn already states an explicit total count (knowledge-update), don't sum sessions!
        if any(r.get("is_superseding_latest") or r.get("temporal_role") == "latest" for r in rows[:5]):
            return None
        target_noun_stems = stem_terms(m_cnt.group(1) or "")
        # Require >= 2 distinct matching sessions to trigger cross-session counting
        if not target_noun_stems:
            return None
        known_speakers = {str(r.get("speaker") or "").casefold() for r in rows[:20] if r.get("speaker")}
        q_non_speaker = {s for s in q_stems if s not in known_speakers and s not in _META_QUANT} or q_stems
        min_hits = 2 if len(q_non_speaker) >= 2 else 1
        matching_sessions: dict[str, dict] = {}
        for row in rows[:25]:
            if row.get("role") == "assistant":
                continue
            text = strip_step_tags(row.get("plain") or row.get("text") or "")
            t_stems = stem_terms(text)
            if not (target_noun_stems & t_stems):
                continue
            if len(q_non_speaker & t_stems) < min_hits:
                continue
            sid = str(row.get("session_id") or row.get("evidence_id"))
            if sid not in matching_sessions:
                matching_sessions[sid] = row
        # Only emit deterministic count when >= 2 distinct sessions match AND none already states an explicit higher count
        if len(matching_sessions) >= 2:
            # Check whether individual turns already carry an explicit count (e.g. "I've watched 5 MCU films")
            explicit_higher = False
            for row in matching_sessions.values():
                text = strip_step_tags(row.get("plain") or row.get("text") or "")
                v, _ = _extract_number_with_unit(text, q)
                if v is not None and v > len(matching_sessions) and v <= 50:
                    explicit_higher = True
                    break
            if not explicit_higher:
                n = len(matching_sessions)
                resp = f"{n} (across {n} distinct recorded sessions)."
                return {
                    "response": resp,
                    "claims": [{"text": resp, "evidence_ids": [r["evidence_id"] for r in matching_sessions.values()]}],
                    "detail": {"op": "count_distinct_sessions", "count": n},
                }

    return None
