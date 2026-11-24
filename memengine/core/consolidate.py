"""Unified consolidation orchestrator: entity disambiguation, supersede, belief distillation & L0 rebuild."""
from __future__ import annotations

import time

from memengine.core.dreaming import dream_beliefs
from memengine.core.episodes import EpisodeLog
from memengine.core.facts import UnifiedFactLedger


def run_consolidation(
    engine,
    episodes: EpisodeLog,
    ledger: UnifiedFactLedger,
    *,
    user_id: str = "default",
    agent_id: str = "default",
    min_new_facts: int = 8,
    force: bool = False,
    timestamp: float | None = None,
) -> dict:
    """Run core consolidation + M4 evidence-backed belief dreaming into unified ``kind='belief'`` facts."""
    now = engine._now() if timestamp is None else float(timestamp)
    result = engine.consolidate(
        user_id=user_id,
        agent_id=agent_id,
        min_new_facts=min_new_facts,
        force=force,
        timestamp=now,
    )
    obs_ids = list(result.get("observation_ids") or ())
    if obs_ids:
        rows = engine.store.facts_by_ids(obs_ids, scope=(user_id, agent_id, None))
        for row in rows:
            ledger.attach(
                fact_id=row["id"],
                user_id=user_id,
                agent_id=agent_id,
                kind="belief",
                title="",
                content=row["plain"],
                verification_status="verified",
                salience_tag="normal",
                always_active=False,
                timestamp=now,
            )
        engine.store.commit()

    dream_res = dream_beliefs(
        engine,
        ledger,
        user_id=user_id,
        agent_id=agent_id,
        timestamp=now,
    )
    return {
        **result,
        "belief_ids": dream_res["belief_ids"],
        "belief_mode": dream_res["mode"],
        "belief_count": dream_res["count"],
    }


def rebuild_from_episodes(
    unified_engine,
    *,
    user_id: str = "default",
    agent_id: str = "default",
    session_id: str | None = None,
) -> dict:
    """Rebuild all derived facts and views cleanly from the L0 episode log (source of truth)."""
    records = list(
        unified_engine.episodes.iter_episodes(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )
    )
    replayed = 0
    for rec in records:
        if rec.role == "system" and (rec.metadata or {}).get("kind") == "rule":
            continue
        unified_engine.ingest_turn(
            rec.text,
            user_id=rec.user_id,
            agent_id=rec.agent_id,
            session_id=rec.session_id,
            role=rec.role,
            timestamp=rec.timestamp,
            mention_time=rec.timestamp,
            speaker=rec.speaker,
            metadata=rec.metadata,
        )
        replayed += 1
    return {"replayed_episodes": replayed, "source": "L0_jsonl_gz"}
