"""Grounded answer() & reflect() with stem-aware verification, M2 temporal/update reasoning & M3 calc hook."""
from __future__ import annotations

import re
import time

from memengine.chrono import date_iso, resolve_temporal_spans
from memengine.core.episodes import EpisodeLog
from memengine.core.facts import UnifiedFactLedger
from memengine.core.supersede import _parse_num, resolve_relative_occurrence
from memengine.core.text import expand_category_stems, light_stem, stem_terms
from memengine.llm import MockLLMClient
from memengine.textutil import strip_step_tags
from memengine.verification import admissible_evidence

_WORD = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")
_SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|\n+")
_STOP = frozenset(
    "a an the and or but in on at to for of with by from as is are was were be been being "
    "i you he she it we they my your his her its our their me him us them this that these those "
    "do does did have has had would could should will can may might must about into over after before "
    "what which who whom whose when where why how tell please give show know remember recall "
    "user assistant said mentioned told one some any many much very just really also still "
    "的 了 是 在 我 你 他 她 它 们 这 那 什么 哪个 哪些 怎么 为什么 多少 几 吗 呢 啊 吧".split()
)
_TEMPORAL_Q = re.compile(
    r"\b(?:when|what date|what day|what month|what year|how long|how many days|how many weeks|how many months)\b|什么时候|哪天|几月|几年|几天",
    re.I,
)
_DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}|20\d{2})\b",
    re.I,
)
_NUM_IN_TEXT = re.compile(
    r"\b(\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\b",
    re.I,
)
_DURATION_PHRASE = re.compile(
    r"\b((?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:days?|weeks?|months?|years?))\b",
    re.I,
)
_OR_SPLIT_FIRST = re.compile(r"(?:first|earlier|last|recently)\s*[,:]?\s*(.+?)\s+or\s+(.+?)[\?？]?$", re.I)
_BETWEEN_SPLIT = re.compile(r"\bbetween\s+(?:the\s+day\s+)?(.+?)\s+and\s+(?:the\s+day\s+)?(.+?)[\?？]?$", re.I)
_SINCE_WHEN_SPLIT = re.compile(r"\bsince\s+(.+?)\s+when\s+(.+?)[\?？]?$", re.I)


def light_stem(word: str) -> str:
    """Lightweight deterministic English suffix normalizer (fixes graduate/graduated mismatch)."""
    w = word.casefold()
    if len(w) <= 3 or not w.isascii() or w.isdigit():
        return w
    for suffix, repl in (
        ("ies", "y"),
        ("ied", "y"),
        ("ing", ""),
        ("ations", "at"),
        ("ation", "at"),
        ("ments", "ment"),
        ("ness", ""),
        ("ed", ""),
        ("es", ""),
        ("ly", ""),
        ("s", ""),
    ):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            stem = w[: -len(suffix)] + repl
            if len(stem) >= 3:
                return stem
    return w


def stem_terms(text: str) -> set[str]:
    cleaned = strip_step_tags(text or "")
    out: set[str] = set()
    for match in _WORD.finditer(cleaned):
        tok = match.group(0).casefold()
        if tok in _STOP or (len(tok) == 1 and tok.isascii() and not tok.isdigit()):
            continue
        out.add(light_stem(tok))
    return out


def stem_evidence_matches_query(
    query: str,
    text: str,
    *,
    entity_anchor: bool = False,
    exact_hit: bool = False,
    speaker: str | None = None,
) -> tuple[bool, float, int]:
    """Return ``(matched, overlap_ratio, overlap_count)`` using stem-normalized content terms."""
    if exact_hit:
        return True, 1.0, 2
    raw_q_stems = stem_terms(query)
    if not raw_q_stems:
        return False, 0.0, 0
    q_stems = expand_category_stems(raw_q_stems)
    e_stems = stem_terms(text)
    overlap = q_stems & e_stems
    count = len(overlap)
    ratio = min(1.0, count / len(raw_q_stems))
    if count == 0:
        return False, 0.0, 0
    sp_stems = stem_terms(speaker) if speaker else set()
    non_speaker_q = raw_q_stems - sp_stems
    non_speaker_overlap = overlap - sp_stems
    if non_speaker_q and not non_speaker_overlap:
        return False, 0.0, 0
    if len(non_speaker_q) >= 3 and len(non_speaker_overlap) < 2 and ratio < 0.50:
        return False, ratio, count
    if len(raw_q_stems) <= 2:
        return (count >= 1), ratio, count
    if len(raw_q_stems) <= 4:
        return (count >= 2 or (entity_anchor and len(non_speaker_overlap) >= 1) or ratio >= 0.45), ratio, count
    return (count >= 2 and ratio >= 0.25) or (entity_anchor and len(non_speaker_overlap) >= 2), ratio, count


def resolve_row_event_ts(row: dict) -> float | None:
    """Resolve the true occurrence timestamp of an evidence row (handling relative offsets)."""
    text = strip_step_tags(row.get("plain") or row.get("text") or "")
    mention = row.get("mention_time") or row.get("event_time_start") or row.get("ingest_time")
    # Explicit relative expression in text overrides bare mention_time
    rel = resolve_relative_occurrence(text, mention)
    if rel is not None:
        return float(rel)
    if row.get("occurrence_start") is not None:
        return float(row["occurrence_start"])
    if mention is not None:
        # Check chrono explicit spans anchored at mention_time
        spans = resolve_temporal_spans(text, float(mention))
        if spans and spans[0].start is not None:
            return float(spans[0].start)
        return float(mention)
    return None


def _best_row_for_clause(clause: str, rows: list[dict], *, exclude_id: str | None = None) -> tuple[dict | None, int]:
    c_stems = stem_terms(clause)
    if not c_stems:
        return None, 0
    best_row = None
    best_score = (0, 0.0, 0)
    for idx, row in enumerate(rows[:20]):
        if exclude_id and row.get("evidence_id") == exclude_id:
            continue
        text = strip_step_tags(row.get("plain") or row.get("text") or "")
        hits = len(c_stems & stem_terms(text))
        if hits == 0:
            continue
        ratio = hits / len(c_stems)
        score = (hits, ratio, -idx)
        if score > best_score:
            best_score = score
            best_row = row
    return best_row, best_score[0]


def _answer_temporal_order(query: str, rows: list[dict], order: str) -> dict | None:
    """Deterministically resolve 'Which event happened first/last, A or B?' using resolved event timestamps."""
    m = _OR_SPLIT_FIRST.search(query.strip())
    if not m:
        return None
    clause_a, clause_b = m.group(1).strip(), m.group(2).strip()
    row_a, hits_a = _best_row_for_clause(clause_a, rows)
    row_b, hits_b = _best_row_for_clause(clause_b, rows, exclude_id=row_a["evidence_id"] if row_a else None)
    if not row_a or not row_b or hits_a == 0 or hits_b == 0:
        return None
    ts_a = resolve_row_event_ts(row_a)
    ts_b = resolve_row_event_ts(row_b)
    if ts_a is None or ts_b is None:
        return None
    a_wins = (ts_a <= ts_b) if order == "first" else (ts_a >= ts_b)
    winner = clause_a if a_wins else clause_b
    da, db = date_iso(ts_a) or "unknown", date_iso(ts_b) or "unknown"
    verb = "happened first" if order == "first" else "happened last"
    resp = f"{winner} {verb} ({clause_a}: {da}; {clause_b}: {db})."
    return {
        "response": resp,
        "claims": [{"text": resp, "evidence_ids": [row_a["evidence_id"], row_b["evidence_id"]]}],
        "detail": {"order": order, "clause_a": clause_a, "date_a": da, "clause_b": clause_b, "date_b": db},
    }


def _answer_temporal_duration(query: str, rows: list[dict], reference_time: float | None) -> dict | None:
    """Deterministically compute day/week/month differences for temporal-reasoning questions."""
    q = query.strip()
    # 1. Two-event difference: "between A and B" or "since A when B"
    m_pair = _BETWEEN_SPLIT.search(q) or _SINCE_WHEN_SPLIT.search(q)
    if m_pair:
        clause_a, clause_b = m_pair.group(1).strip(), m_pair.group(2).strip()
        row_a, hits_a = _best_row_for_clause(clause_a, rows)
        row_b, hits_b = _best_row_for_clause(clause_b, rows, exclude_id=row_a["evidence_id"] if row_a else None)
        if row_a and row_b and hits_a >= 1 and hits_b >= 1:
            ts_a = resolve_row_event_ts(row_a)
            ts_b = resolve_row_event_ts(row_b)
            if ts_a is not None and ts_b is not None:
                days = abs(round((ts_b - ts_a) / 86400.0))
                da, db = date_iso(ts_a) or "", date_iso(ts_b) or ""
                if re.search(r"\bweeks?\b", q, re.I):
                    weeks = round(days / 7.0)
                    resp = f"{weeks} weeks ({days} days, between {da} and {db})."
                elif re.search(r"\bmonths?\b", q, re.I):
                    months = max(1, round(days / 30.0))
                    resp = f"{months} months ({days} days, between {da} and {db})."
                else:
                    resp = f"{days} days (between {da} and {db})."
                return {
                    "response": resp,
                    "claims": [{"text": resp, "evidence_ids": [row_a["evidence_id"], row_b["evidence_id"]]}],
                    "detail": {"days": days, "date_a": da, "date_b": db},
                }

    # 2. Single-event "how many days/weeks/months ago ...?"
    if re.search(r"\b(?:ago|since)\b|天前|周前|个月前", q, re.I) and reference_time is not None:
        row, hits = _best_row_for_clause(q, rows)
        if row and hits >= 2:
            ts = resolve_row_event_ts(row)
            if ts is not None and ts <= float(reference_time) + 86400.0:
                days = max(0, round((float(reference_time) - ts) / 86400.0))
                d_ev = date_iso(ts) or ""
                d_ref = date_iso(reference_time) or ""
                if re.search(r"\bweeks?\b", q, re.I):
                    weeks = round(days / 7.0)
                    resp = f"{weeks} weeks ago ({days} days, on {d_ev} relative to {d_ref})."
                elif re.search(r"\bmonths?\b", q, re.I):
                    months = max(1, round(days / 30.0))
                    resp = f"{months} months ago ({days} days, on {d_ev} relative to {d_ref})."
                else:
                    resp = f"{days} days ago (on {d_ev} relative to {d_ref})."
                return {
                    "response": resp,
                    "claims": [{"text": resp, "evidence_ids": [row["evidence_id"]]}],
                    "detail": {"days": days, "event_date": d_ev, "reference_date": d_ref},
                }

    # 3. "How long had I been <X>?" where text explicitly states a duration ("two months", "three years")
    if re.search(r"\bhow\s+long\b", q, re.I):
        row, hits = _best_row_for_clause(q, rows)
        if row and hits >= 2:
            text = strip_step_tags(row.get("plain") or row.get("text") or "")
            m_dur = _DURATION_PHRASE.search(text)
            if m_dur:
                resp = f"{m_dur.group(1)} — {text[:180]}"
                return {
                    "response": resp,
                    "claims": [{"text": resp, "evidence_ids": [row["evidence_id"]]}],
                    "detail": {"extracted_duration": m_dur.group(1)},
                }
    return None


def _answer_change_direction(query: str, rows: list[dict]) -> dict | None:
    """Resolve 'increase or decrease' questions by comparing earliest vs latest numeric values."""
    q_stems = stem_terms(query) - {"increas", "decreas", "limit", "most", "recent", "chang"}
    matched_rows: list[tuple[float, float, dict]] = []
    for row in rows[:20]:
        text = strip_step_tags(row.get("plain") or row.get("text") or "")
        if len(q_stems & stem_terms(text)) < max(1, min(2, len(q_stems))):
            continue
        nums = [_parse_num(m.group(1)) for m in _NUM_IN_TEXT.finditer(text)]
        nums = [n for n in nums if n is not None]
        ts = row.get("mention_time") or row.get("event_time_start") or 0.0
        if nums:
            matched_rows.append((float(ts), nums[0], row))
    if len(matched_rows) < 2:
        return None
    matched_rows.sort(key=lambda item: item[0])
    old_ts, old_val, old_row = matched_rows[0]
    new_ts, new_val, new_row = matched_rows[-1]
    if old_val == new_val:
        return None
    direction = "increased" if new_val > old_val else "decreased"
    old_s = str(int(old_val) if old_val.is_integer() else old_val)
    new_s = str(int(new_val) if new_val.is_integer() else new_val)
    resp = f"You {direction} it (from {old_s} to {new_s})."
    return {
        "response": resp,
        "claims": [{"text": resp, "evidence_ids": [old_row["evidence_id"], new_row["evidence_id"]]}],
        "detail": {"direction": direction, "old": old_val, "new": new_val},
    }


def _collect_evidence_rows(bundle: dict, query: str) -> list[dict]:
    """Gather all admissible evidence items (facts + observations + L0 episode fallback + historical revisions)."""
    eligible = admissible_evidence(bundle, query=query)
    qplan = bundle.get("query_plan")
    include_hist = bool(qplan and (qplan.is_historical or qplan.change_direction_ask))
    rows: list[dict] = []
    seen_ids: set[str] = set()
    for ev_id, row in eligible.items():
        seen_ids.add(ev_id)
        rows.append({**row, "evidence_id": ev_id})
    for item in bundle.get("items") or ():
        ev_id = item.get("evidence_id") or (f"fact:{item['id']}" if item.get("id") else f"turn:{item.get('turn_id')}")
        if ev_id in seen_ids:
            continue
        if item.get("kind") == "episode_turn" or item.get("historical_slot") or (include_hist and item.get("kind") == "fact"):
            seen_ids.add(ev_id)
            rows.append({**item, "evidence_id": ev_id})
    return rows


_POLAR_Q = re.compile(r"^\s*(?:did|do|does|have|has|had|am|is|are|was|were|can|could)\s+(?:i|you|he|she|they|we)\b|^(?:before\s+[^,]+,\s*)(?:did|do|does|have|has)\s+", re.I)
_NEG_POLARITY = re.compile(r"\b(?:never|didn't|did not|haven't|have not|don't|do not|no longer|not yet|stopped|quit)\b|没有|从未|不再", re.I)
_PROPER_OR_ACRONYM = re.compile(r"\b(?:[A-Z]{2,}[A-Za-z0-9]*|[A-Z][a-z]+[A-Z][A-Za-z0-9]*)\b")


def _extract_focused_claims(query: str, rows: list[dict], *, qplan=None, max_words: int = 58) -> list[dict]:
    """Select concise, citation-verified sentence claims from ranked admissible evidence."""
    q_stems = stem_terms(query)
    wants_date = bool(_TEMPORAL_Q.search(query)) or (qplan is not None and qplan.temporal_order_ask == "date")
    wants_num = bool(re.search(r"\b(?:how\s+many|how\s+much)\b|多少", query, re.I))
    is_polar = bool(_POLAR_Q.search(query)) and not re.search(r"\bor\b", query, re.I)
    is_historical = bool(qplan and qplan.is_historical)
    # Rare acronyms / proper entities in query (e.g., MCU, EOS, Fitbit) deserve extra focus weight
    rare_stems = {light_stem(m.group(0)) for m in _PROPER_OR_ACRONYM.finditer(query or "")}

    # On non-historical current-state queries, exclude facts marked temporal_role='older' when a 'latest' exists
    has_latest = any(r.get("temporal_role") == "latest" or r.get("is_superseding_latest") for r in rows[:15])
    active_rows = (
        [r for r in rows if r.get("temporal_role") != "older" and r.get("status") != "superseded"]
        if (has_latest and not is_historical)
        else rows
    ) or rows

    candidates: list[tuple[float, int, str, dict]] = []
    for rank_idx, row in enumerate(active_rows[:15]):
        raw_text = row.get("plain") or row.get("text") or ""
        cleaned = strip_step_tags(raw_text).strip()
        if not cleaned:
            continue
        matched, row_ratio, row_hits = stem_evidence_matches_query(
            query,
            cleaned,
            entity_anchor=bool(row.get("entity_anchor")),
            exact_hit=bool(row.get("exact_ident_hit")),
            speaker=row.get("speaker"),
        )
        if not matched:
            continue

        resolved_ts = resolve_row_event_ts(row)
        occ = date_iso(resolved_ts) if resolved_ts is not None else (date_iso(row.get("occurrence_start")) or date_iso(row.get("mention_time")))
        # Strip leading "Speaker: " prefix BEFORE sentence splitting so greetings like "Melanie: Wow, Caroline!"
        # never steal speaker-name lexical score from the actual factual sentence in the turn!
        row_speaker = str(row.get("speaker") or "").strip()
        sp_stems = stem_terms(row_speaker) if row_speaker else set()
        body_no_speaker = re.sub(r"^[A-Za-z][A-Za-z0-9 _.-]{0,30}:\s+", "", cleaned, count=1) if ":" in cleaned[:36] else cleaned
        non_sp_q = expand_category_stems(q_stems - sp_stems) or expand_category_stems(q_stems)
        sentences = [s.strip() for s in _SENT_SPLIT.split(body_no_speaker) if len(s.strip()) >= 4] or [body_no_speaker]
        for s_idx, sent in enumerate(sentences):
            effective_sent = sent
            if sent.rstrip().endswith(("?", "？")) and s_idx + 1 < len(sentences) and not sentences[s_idx + 1].rstrip().endswith(("?", "？")):
                effective_sent = sentences[s_idx + 1]
                s_stems = stem_terms(sent + " " + effective_sent)
                is_question_sent = False
            else:
                s_stems = stem_terms(sent)
                is_question_sent = sent.rstrip().endswith(("?", "？"))
            # Score sentences strictly by non-speaker content stems so greetings ("Wow, Caroline!") score 0!
            s_hits = len(non_sp_q & s_stems)
            s_ratio = min(1.0, s_hits / max(1, len(q_stems - sp_stems) or len(q_stems)))
            if s_hits == 0 and len(sentences) > 1:
                continue
            rare_bonus = 0.22 * (len(rare_stems & s_stems) / max(1, len(rare_stems))) if rare_stems else 0.0
            num_bonus = 0.18 if (wants_num and _NUM_IN_TEXT.search(effective_sent)) else 0.0
            date_bonus = 0.22 if (wants_date and (_DATE_RE.search(effective_sent) or occ)) else 0.0
            pref_bonus = 0.12 if row.get("preference_priority") else 0.0
            taboo_bonus = 0.15 if row.get("is_taboo") else 0.0
            latest_bonus = 0.30 if (not is_historical and (row.get("is_superseding_latest") or row.get("temporal_role") == "latest")) else 0.0
            rank_bonus = 0.22 / (1.0 + 0.35 * rank_idx)
            score = (
                0.55 * s_ratio
                + 0.25 * row_ratio
                + rank_bonus
                + rare_bonus
                + num_bonus
                + date_bonus
                + pref_bonus
                + taboo_bonus
                + latest_bonus
                - (0.35 if is_question_sent else 0.0)
            )
            display = f"{occ}: {effective_sent}" if (wants_date and occ and occ not in effective_sent) else effective_sent
            if is_polar and not is_question_sent:
                prefix = "No. " if (_NEG_POLARITY.search(effective_sent) and not _NEG_POLARITY.search(query)) else "Yes. "
                if not display.lower().startswith(("yes", "no")):
                    display = prefix + display
            candidates.append((score, rank_idx * 100 + s_idx, display, row))

    if not candidates:
        return []

    candidates.sort(key=lambda item: (-item[0], item[1]))
    # Only apply multi-session 3-claim packing on genuine multi-hop listing queries (not single-fact knowledge-update counts!)
    is_multi_item = bool(qplan and qplan.is_multi_hop_query and not has_latest)
    max_claims = 3 if is_multi_item else 2
    effective_max_words = 62 if is_multi_item else max_words

    claims: list[dict] = []
    seen_norm: set[str] = set()
    seen_sessions: set[str] = set()
    used_words = 0

    ordered_candidates = candidates
    if is_multi_item and len(candidates) > 1:
        distinct_pass: list[tuple[float, int, str, dict]] = []
        remainder: list[tuple[float, int, str, dict]] = []
        seen_s: set[str] = set()
        for cand in candidates:
            sid = str(cand[3].get("session_id") or cand[3].get("evidence_id"))
            if sid not in seen_s:
                seen_s.add(sid)
                distinct_pass.append(cand)
            else:
                remainder.append(cand)
        ordered_candidates = [*distinct_pass, *remainder]

    for _, _, text, row in ordered_candidates:
        norm = text.casefold().strip()
        if norm in seen_norm:
            continue
        words = text.split()
        per_claim_cap = 26 if (is_multi_item and len(words) > 26) else effective_max_words
        if len(words) > per_claim_cap:
            text = " ".join(words[:per_claim_cap])
            words = text.split()
        if claims and used_words + len(words) > effective_max_words:
            break
        seen_norm.add(norm)
        sid = str(row.get("session_id") or "")
        if sid:
            seen_sessions.add(sid)
        claims.append({
            "text": text,
            "evidence_ids": [row["evidence_id"]],
        })
        used_words += len(words)
        if len(claims) >= max_claims:
            break
    return claims


def unified_answer(engine, query: str, bundle: dict, *, calculator=None) -> dict:
    """Answer strictly from verified evidence with M2 update/temporal reasoning & M3 deterministic calc."""
    chinese = bool(re.search(r"[\u3400-\u9fff]", query))
    refusal = "没有找到足够的证据。" if chinese else "Insufficient evidence to answer."

    if bundle.get("route") == "CASUAL":
        return {**bundle, "response": "", "reasoning": None, "llm_fallback": False, "claims": [], "abstained": False, "partial": False}

    if bundle.get("empty_recall"):
        return {
            **bundle,
            "response": "没有找到相关记忆。" if chinese else "No relevant memory found.",
            "reasoning": None,
            "llm_fallback": False,
            "claims": [],
            "abstained": True,
            "partial": False,
            "answered_facets": {},
            "unanswered_facets": [row["facet"] for row in (bundle.get("lexical_facet_coverage") or {}).get("facets", ())],
        }

    if not isinstance(engine.llm, MockLLMClient):
        return engine._answer_from_context(query, bundle)

    rows = _collect_evidence_rows(bundle, query)
    qplan = bundle.get("query_plan")

    # 1. Optional M3 deterministic calculator hook (counts & calendar math)
    if calculator is not None:
        calc_out = calculator(query, rows, bundle)
        if calc_out is not None:
            return {
                **bundle,
                "response": calc_out["response"],
                "reasoning": {"evidence_synthesis": calc_out["response"], "deterministic_calc": calc_out.get("detail")},
                "llm_fallback": False,
                "claims": calc_out.get("claims", []),
                "abstained": False,
                "partial": False,
            }

    # 2. M2 deterministic temporal & knowledge-update resolvers
    if qplan is not None:
        special = None
        if qplan.change_direction_ask:
            special = _answer_change_direction(query, rows)
        elif qplan.temporal_order_ask in {"first", "last"}:
            special = _answer_temporal_order(query, rows, qplan.temporal_order_ask)
        elif qplan.temporal_order_ask == "duration":
            special = _answer_temporal_duration(query, rows, qplan.as_of)
        if special is not None:
            return {
                **bundle,
                "response": special["response"],
                "reasoning": {"evidence_synthesis": special["response"], "temporal_update_resolution": special.get("detail")},
                "llm_fallback": False,
                "claims": special["claims"],
                "abstained": False,
                "partial": False,
            }

    # 3. Preserve v1 TIME_TRAVEL / facet planner when it answers cleanly
    query_type = (bundle.get("parsed_query") or {}).get("query_type")
    if query_type == "TIME_TRAVEL" or (bundle.get("lexical_facet_coverage") or {}).get("facets"):
        v1_ans = engine._answer_from_context(query, bundle)
        if not v1_ans.get("abstained"):
            return v1_ans

    # 4. Stem-aware grounded extractive answering with update dominance & strict abstention
    claims = _extract_focused_claims(query, rows, qplan=qplan)
    if not claims:
        return {
            **bundle,
            "response": refusal,
            "reasoning": {"evidence_synthesis": "", "anti_intuitive_override": False, "rejected_noise_ids": []},
            "llm_fallback": False,
            "claims": [],
            "abstained": True,
            "partial": False,
        }

    response = " ".join(c["text"] for c in claims)
    return {
        **bundle,
        "response": response,
        "reasoning": {"evidence_synthesis": response, "anti_intuitive_override": False, "rejected_noise_ids": []},
        "llm_fallback": False,
        "claims": claims,
        "abstained": False,
        "partial": False,
    }


def unified_reflect(
    engine,
    episodes: EpisodeLog,
    ledger: UnifiedFactLedger,
    query: str,
    bundle: dict,
    *,
    user_id: str = "default",
    agent_id: str = "default",
    session_id: str = "default",
    timestamp: float | None = None,
) -> dict:
    """Reflect on retrieved evidence and write back an evidence-backed ``kind='belief'`` fact."""
    ans = unified_answer(engine, query, bundle)
    if ans.get("abstained") or not ans.get("claims"):
        return {**ans, "belief_written": False, "belief_fact_id": None}

    now = time.time() if timestamp is None else float(timestamp)
    cited_ids = {eid for c in ans["claims"] for eid in c.get("evidence_ids", ())}
    ev_map = bundle.get("evidence_map") or {}
    source_turns: list[int] = []
    derived_facts: list[str] = []
    for eid in sorted(cited_ids):
        item = ev_map.get(eid)
        if not item:
            continue
        if item.get("kind") in {"fact", "observation"} and item.get("id"):
            derived_facts.append(item["id"])
            for tid in item.get("source_turn_ids") or ():
                if int(tid) not in source_turns:
                    source_turns.append(int(tid))
        elif item.get("turn_id"):
            if int(item["turn_id"]) not in source_turns:
                source_turns.append(int(item["turn_id"]))

    if not source_turns:
        return {**ans, "belief_written": False, "belief_fact_id": None}

    belief_text = ans["response"].strip()
    fact_id = engine.remember(
        belief_text,
        memory_type="fact",
        source_turn_ids=source_turns,
        epistemic_scope="opinion",
        confidence=0.80,
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        timestamp=now,
    )
    ledger.attach(
        fact_id=fact_id,
        user_id=user_id,
        agent_id=agent_id,
        kind="belief",
        title=f"Reflection: {query[:48]}",
        content=belief_text,
        verification_status="verified",
        salience_tag="normal",
        always_active=False,
        timestamp=now,
    )
    engine.store.commit()
    return {
        **ans,
        "belief_written": True,
        "belief_fact_id": fact_id,
        "derived_from_fact_ids": derived_facts,
        "source_turn_ids": source_turns,
    }
