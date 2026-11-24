"""Dual-track ingestion: ungated Conversational Track + salience-gated Procedural Rule Track."""
from __future__ import annotations

import time

from memengine.chrono import parse_session_clock
from memengine.core.episodes import EpisodeLog
from memengine.core.facts import UnifiedFactLedger
from memengine.lifecycle import explicit_preference
from memengine.vfs.models import (
    enforce_rule_state_machine,
    evaluate_salience_gate,
    format_structured_rule,
    make_index_summary,
    normalize_node_path,
)


def _infer_conversational_kind(*, role: str, speaker: str | None, text: str, epistemic_scope: str | None) -> str:
    if role == "tool" or (speaker or "").casefold() in {"assistant", "agent"} or epistemic_scope == "experience":
        return "experience"
    if explicit_preference(text, speaker) is not None:
        return "preference"
    if epistemic_scope == "opinion":
        return "belief"
    return "world"


def ingest_conversational_turn(
    engine,
    episodes: EpisodeLog,
    ledger: UnifiedFactLedger,
    text: str,
    *,
    user_id: str = "default",
    agent_id: str = "default",
    session_id: str = "default",
    role: str = "user",
    timestamp: float | None = None,
    event_time: float | None = None,
    mention_time: float | str | None = None,
    speaker: str | None = None,
    idempotency_key: str | None = None,
    ttl_seconds: float | None = None,
    expires_at: float | str | None = None,
    valid_from: float | str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Conversational ingest track: NO salience gate — always logs L0 + FTS5 + extracts facts."""
    if idempotency_key:
        existing = episodes.lookup_idempotency(user_id=user_id, agent_id=agent_id, idempotency_key=idempotency_key)
        if existing is not None:
            return {
                "deduplicated": True,
                "turn_id": int(existing["turn_id"]),
                "transcript_id": existing["transcript_id"],
                "facts": [],
                "operations": [],
                "track": "conversational",
            }

    mention = parse_session_clock(mention_time)
    now = float(timestamp) if timestamp is not None else (mention if mention is not None else engine._now())

    # Delegate atomic turn + fact extraction to core ledger (populates raw_turns + raw_turns_fts + facts)
    res = engine.add(
        text,
        timestamp=now,
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        role=role,
        event_time=event_time,
        mention_time=now if mention is None else mention,
        speaker=speaker or role,
        ttl_seconds=ttl_seconds,
        expires_at=expires_at,
        valid_from=valid_from,
    )
    turn_id = int(res["turn_id"])

    # Always write cold L0 .jsonl.gz archive (zero L0 vectorization)
    rec = episodes.append_cold(
        turn_id=turn_id,
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        role=role,
        text=text,
        timestamp=now,
        speaker=speaker or role,
        metadata=metadata,
    )
    if idempotency_key:
        episodes.record_idempotency(
            user_id=user_id,
            agent_id=agent_id,
            idempotency_key=idempotency_key,
            turn_id=turn_id,
            transcript_id=rec.transcript_id,
            text=text,
            timestamp=now,
        )

    # Attach unified kind + L0 provenance pointer to every created/updated fact
    created_ids = [op["line_id"] for op in res.get("operations", ()) if op.get("line_id")]
    if created_ids:
        rows = engine.store.facts_by_ids(created_ids, scope=(user_id, agent_id, None))
        for fact in rows:
            kind = _infer_conversational_kind(
                role=role,
                speaker=speaker,
                text=fact["plain"],
                epistemic_scope=fact.get("epistemic_scope"),
            )
            salience = "taboo" if fact.get("is_taboo") else ("preference" if kind == "preference" else "normal")
            ledger.attach(
                fact_id=fact["id"],
                user_id=user_id,
                agent_id=agent_id,
                kind=kind,
                title="",
                content=fact["plain"],
                verification_status="verified",
                salience_tag=salience,
                always_active=bool(fact.get("is_taboo") and fact.get("is_constraint")),
                source_transcript_ids=[rec.transcript_id],
                timestamp=now,
            )
    engine.store.commit()
    return {
        **res,
        "deduplicated": False,
        "transcript_id": rec.transcript_id,
        "track": "conversational",
    }


def ingest_procedural_rule(
    engine,
    episodes: EpisodeLog,
    ledger: UnifiedFactLedger,
    path: str,
    *,
    title: str,
    trigger_condition: str,
    action: str,
    verification_result: str = "",
    notes: str = "",
    index_summary: str = "",
    user_id: str = "default",
    agent_id: str = "default",
    session_id: str = "default",
    confidence_score: float = 0.85,
    verification_status: str = "verified",
    salience_tag: str = "core_rule",
    always_active: bool = False,
    visible_in_context: bool = False,
    reusable_pattern: bool = True,
    future_pitfall_risk: bool = True,
    source_turn_ids: list[int] | tuple[int, ...] = (),
    source_transcript_ids: list[str] | tuple[str, ...] = (),
    expected_version: int | None = None,
    batch_id: str | None = None,
    timestamp: float | None = None,
) -> dict:
    """Procedural rule track: 3-question salience gate + state machine + optimistic lock -> unified kind='rule'."""
    accepted, reason = evaluate_salience_gate(
        visible_in_context=visible_in_context,
        reusable_pattern=reusable_pattern,
        future_pitfall_risk=future_pitfall_risk,
    )
    gate = {"accepted": accepted, "reason": reason}
    if not accepted:
        return {
            "accepted": False,
            "gate": gate,
            "fact_id": None,
            "path": normalize_node_path(path)[1],
            "track": "procedural_rule",
        }

    _, norm_path = normalize_node_path(path)
    now = time.time() if timestamp is None else float(timestamp)
    structured = format_structured_rule(
        title=title,
        trigger_condition=trigger_condition,
        action=action,
        verification_result=verification_result,
        notes=notes,
    )
    verification_status, confidence_score = enforce_rule_state_machine(
        verification_status=verification_status,
        confidence_score=confidence_score,
        verification_result=verification_result,
    )

    # Guarantee source turn exists in L0 so mandatory evidence invariant is never violated
    turns = list(source_turn_ids)
    tids = list(source_transcript_ids)
    if not turns:
        raw_text = f"[rule:{norm_path}] {title}: {structured}"
        turn_id = engine.store.append_turn(
            raw_text,
            now,
            scope=(user_id, agent_id, session_id),
            role="system",
            extraction_state="done",
            speaker="system",
            mention_time=now,
        )
        rec = episodes.append_cold(
            turn_id=turn_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            role="system",
            text=raw_text,
            timestamp=now,
            speaker="system",
            metadata={"vfs_path": norm_path, "kind": "rule"},
        )
        turns = [turn_id]
        if rec.transcript_id not in tids:
            tids.append(rec.transcript_id)

    # Check optimistic lock BEFORE writing fact so a lock conflict leaves the ledger untouched
    existing = ledger.get_active_by_path(norm_path, user_id=user_id, agent_id=agent_id)
    if expected_version is not None:
        from memengine.vfs.models import OptimisticLockError

        actual_version = int(existing["version"]) if existing else 0
        if actual_version != expected_version:
            raise OptimisticLockError(norm_path, expected_version, actual_version)

    # Write directly into the single unified facts table (kind='rule', memory_type='procedural')
    fact_id = engine.remember(
        structured,
        memory_type="procedural",
        source_turn_ids=turns,
        trigger=trigger_condition.strip(),
        is_constraint=bool(always_active or salience_tag == "taboo"),
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        timestamp=now,
        confidence=float(confidence_score),
    )

    summary = make_index_summary(title, structured, index_summary)
    meta = ledger.attach(
        fact_id=fact_id,
        user_id=user_id,
        agent_id=agent_id,
        kind="rule",
        vfs_path=norm_path,
        title=title.strip(),
        content=structured,
        index_summary=summary,
        verification_status=verification_status,
        salience_tag=salience_tag,
        always_active=always_active,
        source_transcript_ids=tids,
        expected_version=expected_version,
        batch_id=batch_id,
        timestamp=now,
    )
    # Soft-supersede the previous active fact now that its snapshot has been recorded
    if existing is not None and existing["fact_id"] != fact_id:
        engine.store.conn.execute(
            "UPDATE facts SET status='superseded', valid_until=? WHERE id=? AND status='active'",
            (now, existing["fact_id"]),
        )
    engine.store.commit()
    return {
        "accepted": True,
        "gate": gate,
        "fact_id": fact_id,
        "path": norm_path,
        "version": meta["version"],
        "index_summary": meta["index_summary"],
        "meta": meta,
        "track": "procedural_rule",
    }
