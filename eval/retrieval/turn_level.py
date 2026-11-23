"""Apples-to-apples turn-level retrieval scoring across engines.

Cross-version comparisons MUST use turn-level R@5 / R@10 / MRR@10.
Coarse session notes (e.g. legacy v2 ``session_md``) are projected to their
single highest-overlap turn per ranked hit so returning a 30-turn session
blob cannot inflate turn-level recall. Session-level recall is reported
strictly as a supplementary column (``supp_session_r10``).
"""
from __future__ import annotations

import re
from typing import Iterable

_WORD = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def _tokens(text: str) -> set[str]:
    return {m.group(0).casefold() for m in _WORD.finditer(text or "")}


def project_hit_to_turns(
    query: str,
    turn_refs: Iterable[str],
    turn_texts: dict[str, str],
    *,
    max_atomic_turns: int = 3,
) -> list[str]:
    """Return turn refs represented by one ranked retrieval unit.

    Atomic fact units typically link to 1-3 source turns and are kept as-is.
    Coarse multi-turn blobs (>3 turns, such as legacy ``session_md`` notes) are
    projected to their single best lexically matching turn so turn-level R@k
    remains comparable across granularities.
    """
    refs = list(dict.fromkeys(turn_refs))
    if len(refs) <= max_atomic_turns:
        return refs
    q_toks = _tokens(query)
    best_ref = max(
        refs,
        key=lambda ref: (len(q_toks & _tokens(turn_texts.get(ref, ""))), ref),
    )
    return [best_ref]


def evaluate_turn_retrieval_item(
    *,
    ranked_turn_sets: list[set[str]],
    ranked_session_sets: list[set[str]],
    gold_turns: set[str],
    gold_sessions: set[str],
    unanswerable: bool = False,
    latest_session: str | None = None,
    older_sessions: set[str] | frozenset[str] = frozenset(),
    superseded_older_archived: bool = False,
) -> dict:
    """Compute strictly turn-level retrieval metrics + supplementary session metrics."""
    if unanswerable or (not gold_turns and not gold_sessions):
        return {
            "answerable": False,
            "turn_r5": None,
            "turn_r10": None,
            "turn_mrr10": None,
            "supp_session_r10": None,
            "update_dominance": None,
            "update_top1": None,
            "update_strict_archived": None,
        }

    # Turn-level metrics (primary cross-version comparison metric)
    if gold_turns:
        t_top5 = set().union(*ranked_turn_sets[:5]) if ranked_turn_sets[:5] else set()
        t_top10 = set().union(*ranked_turn_sets[:10]) if ranked_turn_sets[:10] else set()
        turn_r5 = len(gold_turns & t_top5) / len(gold_turns)
        turn_r10 = len(gold_turns & t_top10) / len(gold_turns)
        first_turn = next(
            (rank for rank, tset in enumerate(ranked_turn_sets[:10], 1) if tset & gold_turns),
            None,
        )
        turn_mrr10 = (1.0 / first_turn) if first_turn is not None else 0.0
    else:
        turn_r5 = turn_r10 = turn_mrr10 = None

    # Supplementary session-level metric (explicitly marked supplementary)
    if gold_sessions:
        s_top10 = set().union(*ranked_session_sets[:10]) if ranked_session_sets[:10] else set()
        supp_session_r10 = len(gold_sessions & s_top10) / len(gold_sessions)
    else:
        supp_session_r10 = None

    # Knowledge-update dominance (unified definition + strict revision-archive check)
    update_dom = None
    update_top1 = None
    update_strict = None
    if latest_session and older_sessions:
        latest_rank = next(
            (rank for rank, sset in enumerate(ranked_session_sets[:10], 1) if latest_session in sset),
            None,
        )
        older_rank = next(
            (rank for rank, sset in enumerate(ranked_session_sets[:10], 1) if sset & set(older_sessions)),
            None,
        )
        # Unified definition: latest value is retrieved in top-10 AND either old value is
        # excluded/suppressed or latest ranks strictly ahead of old value.
        update_dom = 1.0 if (latest_rank is not None and (older_rank is None or latest_rank < older_rank)) else 0.0
        update_top1 = 1.0 if latest_rank == 1 else 0.0
        # Strict v1 revision-chain definition: if old value is absent, require explicit archive link.
        if latest_rank is None:
            update_strict = 0.0
        elif older_rank is None:
            update_strict = 1.0 if superseded_older_archived else 0.0
        else:
            update_strict = 1.0 if latest_rank < older_rank else 0.0

    return {
        "answerable": True,
        "turn_r5": turn_r5,
        "turn_r10": turn_r10,
        "turn_mrr10": turn_mrr10,
        "supp_session_r10": supp_session_r10,
        "update_dominance": update_dom,
        "update_top1": update_top1,
        "update_strict_archived": update_strict,
    }


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def summarize_turn_retrieval(rows: list[dict]) -> dict:
    ans = [r for r in rows if r.get("answerable")]
    t5 = [r["turn_r5"] for r in ans if r.get("turn_r5") is not None]
    t10 = [r["turn_r10"] for r in ans if r.get("turn_r10") is not None]
    tmrr = [r["turn_mrr10"] for r in ans if r.get("turn_mrr10") is not None]
    s10 = [r["supp_session_r10"] for r in ans if r.get("supp_session_r10") is not None]
    udom = [r["update_dominance"] for r in ans if r.get("update_dominance") is not None]
    utop1 = [r["update_top1"] for r in ans if r.get("update_top1") is not None]
    ustrict = [r["update_strict_archived"] for r in ans if r.get("update_strict_archived") is not None]
    return {
        "answerable": len(ans),
        "turn_r5": _avg(t5),
        "turn_r10": _avg(t10),
        "turn_mrr10": _avg(tmrr),
        "supp_session_r10": _avg(s10),
        "update_dominance": _avg(udom),
        "update_top1": _avg(utop1),
        "update_strict_archived": _avg(ustrict),
        "update_n": len(udom),
    }
