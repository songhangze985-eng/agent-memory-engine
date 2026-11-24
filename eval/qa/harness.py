"""Shared end-to-end QA + turn-level retrieval + 6-bucket attribution harness."""
from __future__ import annotations

import tempfile
import time
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.attribution.classify import ATTRIBUTION_BUCKETS, classify_failure, summarize_attribution
from eval.evidence_ablation import Turn, retain_source_turn
from eval.metrics import percentile
from eval.paired import make_engine
from eval.qa.judge import DeterministicQAJudge, OpenAICompatibleJudge
from eval.qa.reader import DeterministicEngineReader, OpenAICompatibleReader
from eval.retrieval.turn_level import (
    evaluate_turn_retrieval_item,
    project_hit_to_turns,
    summarize_turn_retrieval,
)
from memengine.textutil import estimate_tokens
from memengine.vfs import VFSMemoryEngine

TOKEN_BUDGET = 2048


@dataclass(frozen=True)
class QAItem:
    id: str
    category: str
    question: str
    gold_answer: str
    timestamp: float
    question_date: str = ""
    evidence_turns: frozenset[str] = frozenset()
    evidence_sessions: frozenset[str] = frozenset()
    unanswerable: bool = False
    older_sessions: frozenset[str] = frozenset()
    latest_session: str | None = None
    choices: dict[str, str] = field(default_factory=dict)
    mcq_gold: str | None = None


def _iso_date(ts: float, raw: str = "") -> str:
    if raw:
        return raw
    try:
        return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")
    except (OverflowError, OSError, ValueError):
        return ""


def make_reader(llm_model: str | None = None, *, base_url: str | None = None, api_key: str | None = None):
    if llm_model:
        return OpenAICompatibleReader(model=llm_model, base_url=base_url, api_key=api_key)
    return DeterministicEngineReader()


def make_judge(judge_model: str | None = None, *, base_url: str | None = None, api_key: str | None = None):
    if judge_model:
        return OpenAICompatibleJudge(model=judge_model, base_url=base_url, api_key=api_key)
    return DeterministicQAJudge()


def _fact_turn_refs(fact: dict, turn_id_to_ref: dict[int, str]) -> set[str]:
    refs: set[str] = set()
    for tid in fact.get("source_turn_ids") or ():
        if tid in turn_id_to_ref:
            refs.add(turn_id_to_ref[tid])
    return refs


def _fact_session_refs(fact: dict, turn_id_to_session: dict[int, str]) -> set[str]:
    refs: set[str] = set()
    for tid in fact.get("source_turn_ids") or ():
        if tid in turn_id_to_session:
            refs.add(turn_id_to_session[tid])
    return refs


def _eval_v1_group(
    turns: list[Turn],
    questions: list[QAItem],
    *,
    engine_kind: str,
    reader,
    judge,
    consolidate: bool = False,
    token_budget: int = TOKEN_BUDGET,
) -> list[dict[str, Any]]:
    """Run v1_standard, v1_chronicle, or unified engine over one dialogue group."""
    if engine_kind == "unified":
        from memengine.core import UnifiedMemoryEngine

        engine = UnifiedMemoryEngine(":memory:")
    else:
        engine = make_engine("mock")

    is_chronicle = engine_kind in {"v1_chronicle", "unified"}
    mode = "reconstructive" if is_chronicle else "single_query"
    packing_mode = "token_budget" if is_chronicle else "top_k"

    with closing(engine.store):
        turn_id_to_ref: dict[int, str] = {}
        turn_id_to_session: dict[int, str] = {}
        turn_texts: dict[str, str] = {t.ref: t.text for t in turns}
        for ordinal, turn in enumerate(turns):
            if engine_kind == "unified":
                rec = engine.ingest_turn(
                    f"[step {ordinal}] {turn.text}",
                    timestamp=turn.timestamp,
                    session_id=turn.ingest_session or turn.session,
                    role=turn.role,
                    mention_time=turn.timestamp if turn.mention_time is None else turn.mention_time,
                    speaker=turn.speaker,
                )
            elif is_chronicle:
                rec = retain_source_turn(engine, ordinal, turn)
            else:
                rec = engine.add(
                    f"[step {ordinal}] {turn.text}",
                    timestamp=turn.timestamp,
                    session_id=turn.ingest_session or turn.session,
                    role=turn.role,
                )
            tid = rec["turn_id"]
            turn_id_to_ref[tid] = turn.ref
            turn_id_to_session[tid] = turn.session

        if consolidate:
            max_ts = max((t.timestamp for t in turns), default=0.0) + 1.0
            engine.consolidate(min_new_facts=1, force=True, timestamp=max_ts)

        # Build complete fact_id -> turn_refs lookup across all ingested sessions for attribution
        all_facts = engine.store.snapshot(
            scope=("default", "default", None),
            include_superseded=True,
            include_cold=True,
            include_derived=True,
        )
        fact_to_turns: dict[str, set[str]] = {}
        ingested_turns: set[str] = set()
        for fact in all_facts:
            trefs = _fact_turn_refs(fact, turn_id_to_ref)
            fact_to_turns[fact["id"]] = trefs
            ingested_turns.update(trefs)
        # Unified core also indexes L0 turns via FTS5 fallback; include L0 fallback turns as ingested
        if engine_kind == "unified" and hasattr(engine, "episode_indexed_turn_ids"):
            for tid in engine.episode_indexed_turn_ids():
                if tid in turn_id_to_ref:
                    ingested_turns.add(turn_id_to_ref[tid])

        rows: list[dict[str, Any]] = []
        for q in questions:
            trace: list[dict] = []
            t0 = time.perf_counter()
            if engine_kind == "unified":
                bundle = engine.build_context(
                    q.question,
                    reference_time=q.timestamp,
                    mode=mode,
                    token_budget=token_budget,
                    packing_mode=packing_mode,
                    _trace=trace,
                )
            else:
                bundle = engine.build_context(
                    q.question,
                    reference_time=q.timestamp,
                    mode=mode,
                    token_budget=token_budget,
                    packing_mode=packing_mode,
                    top_k=None if is_chronicle else 10,
                    _trace=trace,
                )
            retrieval_ms = (time.perf_counter() - t0) * 1000.0

            # Extract trace sets for 6-bucket attribution
            recalled_turns: set[str] = set()
            for step in trace:
                for fid in step.get("fused") or ():
                    recalled_turns.update(fact_to_turns.get(fid, ()))
                for tid in step.get("episode_fts_turn_ids") or ():
                    if tid in turn_id_to_ref:
                        recalled_turns.add(turn_id_to_ref[tid])

            ranked_facts = list(bundle.get("ranked") or ())
            ranked_turn_sets: list[set[str]] = []
            ranked_session_sets: list[set[str]] = []
            ranked_turns_flat: set[str] = set()
            for fact in ranked_facts:
                trefs = _fact_turn_refs(fact, turn_id_to_ref)
                srefs = _fact_session_refs(fact, turn_id_to_session)
                projected = set(project_hit_to_turns(q.question, trefs, turn_texts))
                ranked_turn_sets.append(projected)
                ranked_session_sets.append(srefs)
                ranked_turns_flat.update(trefs)

            # Packed turns in final assembled context
            packed_turns: set[str] = set()
            packed_facts: list[dict] = []
            for item in bundle.get("items") or ():
                if item.get("kind") in {"fact", "observation"}:
                    packed_facts.append(item)
                    packed_turns.update(_fact_turn_refs(item, turn_id_to_ref))
                elif item.get("kind") in {"raw_turn", "session_turn", "episode_turn"}:
                    tid = item.get("turn_id")
                    if tid in turn_id_to_ref:
                        packed_turns.add(turn_id_to_ref[tid])
            for tid in bundle.get("evidence_turn_ids") or ():
                if tid in turn_id_to_ref:
                    packed_turns.add(turn_id_to_ref[tid])

            # Check strict archived supersede link for knowledge-update
            superseded_archived = False
            if q.latest_session and q.older_sessions:
                latest_ids = {
                    f["id"]
                    for f in ranked_facts[:10]
                    if f.get("status") == "active"
                    and f.get("revision")
                    and q.latest_session in _fact_session_refs(f, turn_id_to_session)
                }
                if latest_ids:
                    hist = engine.store.historical_versions(latest_ids, scope=("default", "default", None))
                    older_set = set(q.older_sessions)
                    for rev in hist:
                        if _fact_session_refs(rev, turn_id_to_session) & older_set:
                            superseded_archived = True
                            break

            gold_sessions = set(q.evidence_sessions) if q.evidence_sessions else {
                t.session for t in turns if t.ref in q.evidence_turns
            }
            ret_metrics = evaluate_turn_retrieval_item(
                ranked_turn_sets=ranked_turn_sets,
                ranked_session_sets=ranked_session_sets,
                gold_turns=set(q.evidence_turns),
                gold_sessions=gold_sessions,
                unanswerable=q.unanswerable,
                latest_session=q.latest_session,
                older_sessions=q.older_sessions,
                superseded_older_archived=superseded_archived,
            )

            # End-to-end QA Reader
            context_text = "\n\n".join(
                part for part in (bundle.get("system_prefix", ""), bundle.get("dynamic_memory", "")) if part
            )
            if isinstance(reader, DeterministicEngineReader):
                if q.choices:
                    out = reader.answer_from_context(q.question, context_text, empty_recall=bundle["empty_recall"], choices=q.choices)
                else:
                    ans_res = engine._answer_from_context(q.question, bundle)
                    out = reader.answer_from_engine(ans_res, q.question)
            else:
                out = reader.answer_from_context(
                    q.question,
                    context_text,
                    reference_date=_iso_date(q.timestamp, q.question_date),
                    empty_recall=bundle["empty_recall"],
                    choices=q.choices or None,
                )

            verdict = judge.evaluate(
                question=q.question,
                gold_answer=q.gold_answer,
                predicted_answer=out.answer,
                category=q.category,
                unanswerable=q.unanswerable,
                abstained=out.abstained,
                mcq_gold=q.mcq_gold,
                mcq_predicted=out.predicted_choice,
            )

            bucket = classify_failure(
                correct=verdict.correct,
                unanswerable=q.unanswerable,
                gold_turns=q.evidence_turns,
                ingested_turns=ingested_turns,
                recalled_turns=recalled_turns | ranked_turns_flat,
                ranked_turns=ranked_turns_flat,
                packed_turns=packed_turns,
                borderline_judge=verdict.borderline,
            )

            rows.append({
                "id": q.id,
                "category": q.category,
                "question": q.question,
                "gold_answer": q.gold_answer,
                "predicted_answer": out.answer,
                "abstained": out.abstained,
                "unanswerable": q.unanswerable,
                "correct": verdict.correct,
                "judge_score": verdict.score,
                "judge_label": verdict.label,
                "judge_reason": verdict.reason,
                "attribution": bucket,
                "injected_tokens": int((bundle.get("token_breakdown") or {}).get("total", 0)),
                "retrieval_ms": round(retrieval_ms, 3),
                **ret_metrics,
            })
        return rows


def _eval_vfs_group(
    turns: list[Turn],
    questions: list[QAItem],
    *,
    engine_kind: str,
    reader,
    judge,
    token_budget: int = TOKEN_BUDGET,
) -> list[dict[str, Any]]:
    """Run legacy v2 VFSMemoryEngine in either ``v2_session_md`` or ``v2_turn_md`` mode."""
    per_turn = engine_kind == "v2_turn_md"
    with tempfile.TemporaryDirectory() as tmp:
        vfs = VFSMemoryEngine(Path(tmp) / "vfs.db", transcript_dir=Path(tmp) / "l0")
        with closing(vfs):
            tid_to_ref: dict[str, str] = {}
            tid_to_session: dict[str, str] = {}
            turn_texts: dict[str, str] = {t.ref: t.text for t in turns}
            by_session: dict[str, list[tuple[str, Turn]]] = {}

            for ordinal, turn in enumerate(turns):
                rec = vfs.append_transcript(
                    turn.text,
                    session_id=turn.ingest_session or turn.session or "default",
                    role=turn.role,
                    timestamp=turn.timestamp,
                    speaker=turn.speaker,
                )
                tid_to_ref[rec.transcript_id] = turn.ref
                tid_to_session[rec.transcript_id] = turn.session
                if per_turn:
                    vfs.write(
                        f"/turns/t_{ordinal:05d}.md",
                        title=f"Turn {ordinal}",
                        summary=turn.text[:160],
                        content=turn.text,
                        layer="L2",
                        verification_status="verified",
                        source_transcript_ids=[rec.transcript_id],
                        timestamp=turn.timestamp,
                    )
                else:
                    by_session.setdefault(turn.session, []).append((rec.transcript_id, turn))

            if not per_turn:
                for ordinal, (orig_session, entries) in enumerate(by_session.items()):
                    tids = [tid for tid, _ in entries]
                    body = "\n".join(t.text for _, t in entries)
                    summary = entries[0][1].text[:160] if entries else orig_session
                    session_ts = max((t.timestamp for _, t in entries), default=float(ordinal + 1))
                    vfs.write(
                        f"/sessions/s_{ordinal:04d}_{orig_session}.md",
                        title=f"Session {orig_session}",
                        summary=summary,
                        content=body,
                        layer="L2",
                        verification_status="verified",
                        source_transcript_ids=tids,
                        timestamp=session_ts,
                    )

            ingested_turns = set(tid_to_ref.values())
            rows: list[dict[str, Any]] = []
            for q in questions:
                t0 = time.perf_counter()
                # Recall top-50 to separate recall_miss vs ranking_drop honestly
                wide_hits = vfs.search(q.question, reference_time=q.timestamp, top_k=50)
                top10_hits = wide_hits[:10]
                ctx = vfs.build_context(
                    q.question,
                    reference_time=q.timestamp,
                    token_budget=token_budget,
                    disclosure="full",
                    include_tree_outline=False,
                    include_l1_working=False,
                    top_k=10,
                )
                retrieval_ms = (time.perf_counter() - t0) * 1000.0

                recalled_turns: set[str] = set()
                for hit in wide_hits:
                    for tid in hit.get("source_transcript_ids") or ():
                        if tid in tid_to_ref:
                            recalled_turns.add(tid_to_ref[tid])

                ranked_turn_sets: list[set[str]] = []
                ranked_session_sets: list[set[str]] = []
                ranked_turns_flat: set[str] = set()
                for hit in top10_hits:
                    trefs = [tid_to_ref[tid] for tid in (hit.get("source_transcript_ids") or ()) if tid in tid_to_ref]
                    srefs = {tid_to_session[tid] for tid in (hit.get("source_transcript_ids") or ()) if tid in tid_to_session}
                    # Strict apples-to-apples turn projection for coarse session blobs
                    projected = set(project_hit_to_turns(q.question, trefs, turn_texts))
                    ranked_turn_sets.append(projected)
                    ranked_session_sets.append(srefs)
                    ranked_turns_flat.update(trefs)

                packed_paths = {item.get("path") for item in (ctx.get("l2_hits") or ())}
                packed_turns: set[str] = set()
                for tid in ctx.get("source_transcript_ids") or ():
                    if tid in tid_to_ref:
                        packed_turns.add(tid_to_ref[tid])
                for hit in top10_hits:
                    if hit.get("path") in packed_paths:
                        for tid in hit.get("source_transcript_ids") or ():
                            if tid in tid_to_ref:
                                packed_turns.add(tid_to_ref[tid])

                gold_sessions = set(q.evidence_sessions) if q.evidence_sessions else {
                    t.session for t in turns if t.ref in q.evidence_turns
                }
                ret_metrics = evaluate_turn_retrieval_item(
                    ranked_turn_sets=ranked_turn_sets,
                    ranked_session_sets=ranked_session_sets,
                    gold_turns=set(q.evidence_turns),
                    gold_sessions=gold_sessions,
                    unanswerable=q.unanswerable,
                    latest_session=q.latest_session,
                    older_sessions=q.older_sessions,
                    superseded_older_archived=False,
                )

                context_text = ctx.get("dynamic_memory") or ""
                top_score = float(top10_hits[0].get("final_score", top10_hits[0].get("score", 0.0))) if top10_hits else 0.0
                out = reader.answer_from_context(
                    q.question,
                    context_text,
                    reference_date=_iso_date(q.timestamp, q.question_date),
                    empty_recall=not top10_hits,
                    top_score=top_score,
                    min_score=0.05,
                    choices=q.choices or None,
                )

                verdict = judge.evaluate(
                    question=q.question,
                    gold_answer=q.gold_answer,
                    predicted_answer=out.answer,
                    category=q.category,
                    unanswerable=q.unanswerable,
                    abstained=out.abstained,
                    mcq_gold=q.mcq_gold,
                    mcq_predicted=out.predicted_choice,
                )

                bucket = classify_failure(
                    correct=verdict.correct,
                    unanswerable=q.unanswerable,
                    gold_turns=q.evidence_turns,
                    ingested_turns=ingested_turns,
                    recalled_turns=recalled_turns,
                    ranked_turns=ranked_turns_flat,
                    packed_turns=packed_turns,
                    borderline_judge=verdict.borderline,
                )

                rows.append({
                    "id": q.id,
                    "category": q.category,
                    "question": q.question,
                    "gold_answer": q.gold_answer,
                    "predicted_answer": out.answer,
                    "abstained": out.abstained,
                    "unanswerable": q.unanswerable,
                    "correct": verdict.correct,
                    "judge_score": verdict.score,
                    "judge_label": verdict.label,
                    "judge_reason": verdict.reason,
                    "attribution": bucket,
                    "injected_tokens": int((ctx.get("token_breakdown") or {}).get("total_tokens", estimate_tokens(context_text))),
                    "retrieval_ms": round(retrieval_ms, 3),
                    **ret_metrics,
                })
            return rows


def evaluate_group_qa(
    turns: list[Turn],
    questions: list[QAItem],
    *,
    engine_kind: str,
    llm_model: str | None = None,
    judge_model: str | None = None,
    consolidate: bool = False,
    token_budget: int = TOKEN_BUDGET,
) -> list[dict[str, Any]]:
    reader = make_reader(llm_model)
    judge = make_judge(judge_model)
    if engine_kind in {"v2_session_md", "v2_turn_md"}:
        return _eval_vfs_group(turns, questions, engine_kind=engine_kind, reader=reader, judge=judge, token_budget=token_budget)
    if engine_kind in {"v1_standard", "v1_chronicle", "unified"}:
        return _eval_v1_group(
            turns,
            questions,
            engine_kind=engine_kind,
            reader=reader,
            judge=judge,
            consolidate=consolidate,
            token_budget=token_budget,
        )
    raise ValueError(f"unsupported engine_kind: {engine_kind!r}")


def summarize_qa_rows(rows: list[dict[str, Any]], *, categories: list[str] | None = None) -> dict[str, Any]:
    def _agg(subset: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(subset)
        acc = sum(1 for r in subset if r["correct"]) / n if n else 0.0
        ret = summarize_turn_retrieval(subset)
        durs = [float(r["retrieval_ms"]) for r in subset]
        toks = [int(r["injected_tokens"]) for r in subset]
        return {
            "n": n,
            "qa_accuracy": round(acc, 6),
            "correct": sum(1 for r in subset if r["correct"]),
            "wrong": sum(1 for r in subset if not r["correct"]),
            **ret,
            "avg_injected_tokens": round(sum(toks) / n, 1) if n else 0.0,
            "p50_ms": round(percentile(durs, 0.50), 2) if durs else 0.0,
            "p95_ms": round(percentile(durs, 0.95), 2) if durs else 0.0,
        }

    ordered_cats = categories or sorted({str(r["category"]) for r in rows})
    by_cat = {cat: _agg([r for r in rows if r["category"] == cat]) for cat in ordered_cats if any(r["category"] == cat for r in rows)}
    answerable_only = [r for r in rows if not r.get("unanswerable")]
    return {
        "overall": _agg(rows),
        "answerable_only": _agg(answerable_only) if len(answerable_only) != len(rows) else None,
        "by_category": by_cat,
        "attribution": summarize_attribution(rows),
    }


def render_qa_markdown(report: dict[str, Any], title: str) -> str:
    from eval.qa.provenance import render_provenance_markdown

    lines = [f"## {title}", "", render_provenance_markdown(report["provenance"]), ""]
    lines.append(
        "| Category | N | QA Acc | Turn R@5 | Turn R@10 | Turn MRR@10 | Supp. Session R@10 | Update Dom | Avg Tokens | P95 ms |"
    )
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    def _fmt_pct(v) -> str:
        return "—" if v is None else f"{v:.2%}"

    def _fmt_mrr(v) -> str:
        return "—" if v is None else f"{v:.4f}"

    def _emit_row(label: str, m: dict[str, Any]) -> None:
        lines.append(
            f"| {label} | {m['n']} | **{_fmt_pct(m['qa_accuracy'])}** | {_fmt_pct(m['turn_r5'])} | "
            f"{_fmt_pct(m['turn_r10'])} | {_fmt_mrr(m['turn_mrr10'])} | {_fmt_pct(m['supp_session_r10'])} | "
            f"{_fmt_pct(m['update_dominance'])} | {m['avg_injected_tokens']:.1f} | {m['p95_ms']:.2f} |"
        )

    _emit_row("**Overall**", report["summary"]["overall"])
    if report["summary"].get("answerable_only"):
        _emit_row("*Answerable Only*", report["summary"]["answerable_only"])
    for cat, metrics in report["summary"]["by_category"].items():
        _emit_row(f"`{cat}`", metrics)

    attr = report["summary"]["attribution"]
    lines.extend([
        "",
        f"### Error Attribution ({attr['total_wrong']} wrong / {attr['total_questions']} total)",
        "",
        "| Failure Bucket | Count | Share of Errors |",
        "| :--- | ---: | ---: |",
    ])
    for bucket in ATTRIBUTION_BUCKETS:
        b = attr["by_bucket"][bucket]
        lines.append(f"| `{bucket}` | {b['count']} | {b['share_of_errors']:.2%} |")
    return "\n".join(lines)
