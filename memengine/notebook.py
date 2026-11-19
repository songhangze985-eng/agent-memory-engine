"""Step 5 write operations and 2D rendering.

OVERWRITE replaces one line and archives the previous value.
APPEND is a following line inside the block (single newline).
INSERT_ISOLATED keeps a blank line from the neighbor (double newline).
NOOP leaves the notebook unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone

ACTIONS = {
    "OVERWRITE_SAME_LINE",
    "APPEND_ADJACENT_LINE",
    "INSERT_ISOLATED_LINE",
    "NOOP",
}


def format_fact_line(timestamp: float, plain: str) -> str:
    moment = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
    return f"- [{moment.strftime('%Y-%m-%d')}] {plain.strip()}"


def render_block(lines: list[dict]) -> str:
    parts: list[str] = []
    for index, line in enumerate(lines):
        if index > 0 and (line.get("isolated") or lines[index - 1].get("isolated")):
            parts.append("")
        parts.append(line.get("formatted") or line.get("text") or "")
    return "\n".join(parts)


def _resolve_block(store, block_id: str, topic: str, timestamp: float, alias: dict) -> str:
    if block_id in alias:
        return alias[block_id]
    if not block_id or block_id == "NEW" or str(block_id).startswith("NEW"):
        real = store.create_block(topic or "杂项", timestamp)
        if block_id:
            alias[block_id] = real
        return real
    if store.get_block(block_id) is None:
        real = store.create_block(topic or "杂项", timestamp)
        alias[block_id] = real
        return real
    return block_id


def apply_write_op(store, op: dict, *, topic: str, attribute: str, value: str, plain: str, timestamp: float, alias: dict | None = None) -> str | None:
    """Apply one layout operation. Returns the line id, or None for NOOP."""
    action = op.get("action")
    if action not in ACTIONS or action == "NOOP":
        return None
    alias = alias if alias is not None else {}
    zkeys = [str(key) for key in (op.get("z_axis_keys") or []) if key]
    try:
        surprise = float(op.get("surprise_score") if op.get("surprise_score") is not None else 0.5)
    except (TypeError, ValueError):
        surprise = 0.5
    formatted = op.get("formatted_line") or format_fact_line(timestamp, plain)
    isolated = action == "INSERT_ISOLATED_LINE"

    if action == "OVERWRITE_SAME_LINE":
        line_id = op.get("target_line_id") or ""
        current = store.get_line(line_id) if line_id else None
        if current is not None:
            store.archive_line(current, timestamp)
            store.update_line(
                line_id,
                formatted=formatted,
                plain=plain,
                attribute=attribute,
                value=str(value),
                surprise=surprise,
                raw_dist=surprise,
                layout=action,
                isolated=bool(current.get("isolated")),
                zkeys=zkeys,
                timestamp=timestamp,
            )
            return line_id
        action = "APPEND_ADJACENT_LINE"
        isolated = False

    block_id = _resolve_block(store, str(op.get("target_block_id") or "NEW"), topic, timestamp, alias)
    return store.insert_line(
        block_id=block_id,
        formatted=formatted,
        plain=plain,
        attribute=attribute,
        value=str(value),
        surprise=surprise,
        raw_dist=surprise,
        layout=action,
        isolated=isolated,
        zkeys=zkeys,
        timestamp=timestamp,
    )
