"""Grounded answer() and reflect() output contract with stem-aware verification and strict abstention."""
from __future__ import annotations

import re
import time

from memengine.chrono import date_iso
from memengine.core.episodes import EpisodeLog
from memengine.core.facts import UnifiedFactLedger
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
_TEMPORAL_Q = re.compile(r"\b(?:when|what date|what day|what month|what year|how long|how many days|how many weeks|how many months)\b|什么时候|哪天|几月|几年|几天", re.I)
_DATE_RE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}|20\d{2})\b", re.I)


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
        if tok in _STOP or len(tok) == 1 and tok.isascii() and not tok.isdigit():
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
    q_stems = stem_terms(query)
    if not q_stems:
        return False, 0.0, 0
    e_stems = stem_terms(text)
    overlap = q_stems & e_stems
    count = len(overlap)
    ratio = count / len(q_stems)
    if count == 0:
        return False, 0.0, 0
    # Guard against speaker-only false matches on adversarial/unanswerable questions
    sp_stems = stem_terms(speaker) if speaker else set()
    non_speaker_q = q_stems - sp_stems
    non_speaker_overlap = overlap - sp_stems
    if non_speaker_q and not non_speaker_overlap:
        return False, 0.0, 0
    if len(non_speaker_q) >= 3 and len(non_speaker_overlap) < 2 and ratio < 0.50:
        return False, ratio, count
    if len(q_stems) <= 2:
        return (count >= 1), ratio, count
    if len(q_stems) <= 4:
        return (count >= 2 or (entity_anchor and len(non_speaker_overlap) >= 1) or ratio >= 0.45), ratio, count
    return (count >= 2 and ratio >= 0.25) or (entity_anchor and len(non_speaker_overlap) >= 2), ratio, count


def _collect_evidence_rows(bundle: dict, query: str) -> list[dict]:
    """Gather all admissible evidence items (facts + observations + L0 episode fallback turns)."""
    eligible = admissible_evidence(bundle, query=query)
    rows: list[dict] = []
    seen_ids: set[str] = set()
    for ev_id, row in eligible.items():
        seen_ids.add(ev_id)
        rows.append({**row, "evidence_id": ev_id})
    for item in bundle.get("items") or ():
        if item.get("kind") == "episode_turn":
            ev_id = item.get("evidence_id") or f"turn:{item['turn_id']}"
            if ev_id not in seen_ids:
                seen_ids.add(ev_id)
                rows.append({**item, "evidence_id": ev_id})
    return rows


def _extract_focused_claims(query: str, rows: list[dict], *, max_words: int = 58) -> list[dict]:
    """Select concise, citation-verified sentence claims from ranked admissible evidence."""
    q_stems = stem_terms(query)
    wants_date = bool(_TEMPORAL_Q.search(query))
    candidates: list[tuple[float, int, str, dict]] = []

    for rank_idx, row in enumerate(rows[:15]):
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

        # Attach date prefix if temporal question and row has occurrence/mention date
        occ = date_iso(row.get("occurrence_start")) or date_iso(row.get("mention_time"))
        sentences = [s.strip() for s in _SENT_SPLIT.split(cleaned) if len(s.strip()) >= 4] or [cleaned]
        for s_idx, sent in enumerate(sentences):
            s_stems = stem_terms(sent)
            s_hits = len(q_stems & s_stems)
            s_ratio = s_hits / max(1, len(q_stems))
            # Skip pure questions echoed by the user unless no declarative sentence exists
            is_question_sent = sent.rstrip().endswith(("?", "？"))
            date_bonus = 0.20 if (wants_date and (_DATE_RE.search(sent) or occ)) else 0.0
            pref_bonus = 0.12 if row.get("preference_priority") else 0.0
            taboo_bonus = 0.15 if row.get("is_taboo") else 0.0
            rank_bonus = 0.22 / (1.0 + 0.35 * rank_idx)
            # Combine sentence overlap, row overlap, and engine rank
            score = (
                0.55 * s_ratio
                + 0.25 * row_ratio
                + rank_bonus
                + date_bonus
                + pref_bonus
                + taboo_bonus
                - (0.28 if is_question_sent else 0.0)
            )
            display = f"[{occ}] {sent}" if (wants_date and occ and occ not in sent) else sent
            candidates.append((score, rank_idx * 100 + s_idx, display, row))

    if not candidates:
        return []

    candidates.sort(key=lambda item: (-item[0], item[1]))
    claims: list[dict] = []
    seen_norm: set[str] = set()
    used_words = 0
    for _, _, text, row in candidates:
        norm = text.casefold().strip()
        if norm in seen_norm:
            continue
        words = text.split()
        if len(words) > max_words:
            if claims:
                continue
            text = " ".join(words[:max_words])
            words = text.split()
        if claims and used_words + len(words) > max_words:
            break
        seen_norm.add(norm)
        claims.append({
            "text": text,
            "evidence_ids": [row["evidence_id"]],
        })
        used_words += len(words)
        if len(claims) >= 2:
            break
    return claims


def unified_answer(engine, query: str, bundle: dict, *, calculator=None) -> dict:
    """Answer strictly from verified evidence; supports live LLM, deterministic calc hook, and strict abstention."""
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

    # If a real LLM is wired into the engine, preserve the full v1 LLM + claim-verification contract
    if not isinstance(engine.llm, MockLLMClient):
        return engine._answer_from_context(query, bundle)

    rows = _collect_evidence_rows(bundle, query)

    # Optional M3 deterministic calculator hook (counts & calendar date differences)
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

    # For explicit TIME_TRAVEL / revision / facet queries where v1's rule planner already succeeds, keep it first
    query_type = (bundle.get("parsed_query") or {}).get("query_type")
    if query_type == "TIME_TRAVEL" or (bundle.get("lexical_facet_coverage") or {}).get("facets"):
        v1_ans = engine._answer_from_context(query, bundle)
        if not v1_ans.get("abstained"):
            return v1_ans

    # Stem-aware grounded extractive answering with strict abstention guard
    claims = _extract_focused_claims(query, rows)
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
