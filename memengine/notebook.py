"""Step 5 write operations and 2D rendering.

OVERWRITE supersedes one fact in the same display slot.
APPEND is a following line inside the block (single newline).
INSERT_ISOLATED keeps a blank line from the neighbor (double newline).
NOOP leaves the notebook unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone
import re

from memengine.store import DEFAULT_SCOPE

ACTIONS = {
    "TEMPORARY_OVERRIDE",
    "OVERWRITE_SAME_LINE",
    "APPEND_ADJACENT_LINE",
    "INSERT_ISOLATED_LINE",
    "DELETE_LINE",
    "NOOP",
}


def format_fact_line(timestamp: float, plain: str) -> str:
    moment = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
    return f"- [{moment.strftime('%Y-%m-%d')}] {plain.strip()}"


def line_body(formatted: str) -> str:
    one_line = " ".join(formatted.splitlines())
    return re.sub(r"^\s*[-*]\s*(?:\[\d{4}-\d{2}-\d{2}\]\s*)?", "", one_line, count=1).strip()


def render_block(lines: list[dict]) -> str:
    parts: list[str] = []
    for index, line in enumerate(lines):
        if index > 0 and (line.get("isolated") or lines[index - 1].get("isolated")):
            parts.append("")
        parts.append(line.get("formatted") or line.get("text") or "")
    return "\n".join(parts)


def _resolve_block(store, block_id: str, topic: str, timestamp: float, alias: dict,
                   *, scope, event_time, is_conflict) -> str | None:
    if block_id in alias:
        return alias[block_id]
    if not block_id or block_id == "NEW" or block_id.startswith("NEW:"):
        real = store.create_block(topic or "杂项", timestamp, scope=scope, event_time=event_time, is_conflict=is_conflict)
        if block_id.startswith("NEW:"):
            alias[block_id] = real
        return real
    return block_id if store.get_block(block_id, scope=scope) is not None else None


def apply_write_op(store, op: dict, *, topic: str, attribute: str, value: str, plain: str, timestamp: float,
                   alias: dict | None = None, fact_zkeys=None, scope=DEFAULT_SCOPE, role="user",
                   event_time=None, source_step_ids=None, source_turn_id=None, source_turn_ids=(),
                   epistemic_scope=None, valid_from=None, expires_at=None, chronicle=None) -> str | None:
    """Apply one layout operation. Returns the line id, or None for NOOP."""
    action = op.get("action")
    if action not in ACTIONS or action == "NOOP":
        return None
    alias = alias if alias is not None else {}
    line_id = op.get("target_line_id") or ""
    line_id = alias.get(line_id, line_id)
    if action == "DELETE_LINE":
        return line_id if store.delete_fact(line_id, ingest_time=timestamp, scope=scope) else None
    zkeys = [str(key) for key in (op.get("z_axis_keys") or []) if key]
    for key in fact_zkeys or []:
        key = str(key)
        if key and key not in zkeys:
            zkeys.append(key)
    is_conflict = op.get("is_conflict_or_override") is True
    surprise = (0.08 if topic == "时事杂讯" else 0.95 if is_conflict else
                {"OVERWRITE_SAME_LINE": 0.85, "INSERT_ISOLATED_LINE": 0.90, "APPEND_ADJACENT_LINE": 0.25, "TEMPORARY_OVERRIDE": 0.90}[action])
    formatted = op.get("formatted_line") or format_fact_line(timestamp, plain)
    formatted = " ".join(formatted.splitlines()).strip()
    plain = line_body(formatted)
    if not plain:
        return None
    isolated = action in {"INSERT_ISOLATED_LINE", "TEMPORARY_OVERRIDE"}

    if action == "OVERWRITE_SAME_LINE":
        return store.supersede_fact(line_id, formatted=formatted, plain=plain, attribute=attribute, value=str(value),
                                    surprise=surprise, layout=action, zkeys=zkeys, ingest_time=timestamp,
                                    current_step_ids=source_step_ids, current_turn_id=source_turn_id, scope=scope,
                                    role=role, event_time_start=event_time, is_conflict=is_conflict,
                                    source_turn_ids=source_turn_ids, chronicle=chronicle)

    block_id = _resolve_block(store, str(op.get("target_block_id") or "NEW"), topic, timestamp, alias,
                              scope=scope, event_time=event_time, is_conflict=is_conflict)
    if block_id is None:
        return None
    return store.insert_fact(
        block_id=block_id,
        formatted=formatted,
        plain=plain,
        attribute=attribute,
        value=str(value),
        surprise=surprise,
        layout=action,
        isolated=isolated,
        zkeys=zkeys,
        ingest_time=timestamp,
        source_step_ids=source_step_ids,
        source_turn_ids=[*source_turn_ids, *([source_turn_id] if source_turn_id is not None else [])],
        scope=scope,
        role=role,
        event_time_start=event_time,
        is_conflict=is_conflict,
        epistemic_scope=epistemic_scope, valid_from=valid_from, expires_at=expires_at, chronicle=chronicle,
    )
