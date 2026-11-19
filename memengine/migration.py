"""One-time migration from the split notebook schema to the fact ledger."""

from collections import defaultdict
import json

from memengine.textutil import estimate_tokens, merge_step_ids, step_ids_in, strip_step_tags


def migrate_legacy(store):
    conn = store.conn
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table in ("blocks", "timeline"):
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for column in ("user_id", "agent_id", "session_id"):
            if column not in columns:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT NOT NULL DEFAULT 'default'")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_timeline_scope ON timeline(user_id,agent_id,session_id,layer,position)")
    if not tables.intersection({"ingest", "lines", "line_history", "line_zkeys"}):
        return

    def legacy_rows(table):
        return (dict(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid")) if table in tables else ()

    def scope(row):
        return tuple(row.get(name, "default") for name in ("user_id", "agent_id", "session_id"))

    conn.execute("DELETE FROM meta WHERE key LIKE 'history_tokens:%'")
    exact_sources, tagged_sources = defaultdict(list), defaultdict(list)
    for turn in legacy_rows("ingest"):
        stamp = turn.get("created_at")
        if stamp is None:
            stamp = turn.get("ts")
        if stamp is None:
            stamp = 0.0
        owner = scope(turn)
        turn_id = store.append_turn(turn["text"], stamp, scope=owner, role=turn.get("role", "user"),
                                    event_ts=turn.get("event_time"))
        exact_sources[(owner, stamp, strip_step_tags(turn["text"]))].append(turn_id)
        for step in step_ids_in(turn["text"]):
            tagged_sources[(owner, step)].append(turn_id)

    keys, history = defaultdict(list), defaultdict(list)
    for row in legacy_rows("line_zkeys"):
        keys[row["line_id"]].append(row["zkey"])
    for row in legacy_rows("line_history"):
        history[row["line_id"]].append(row)

    lines = list(legacy_rows("lines"))
    orphan_history = set(history) - {line["id"] for line in lines}
    if orphan_history:
        raise ValueError("Legacy archive contains revisions without their parent line; migration stopped")
    for line in lines:
        root = line["id"]
        owner = scope(line)
        active = bool(line.get("active", 1))
        revision = line.get("revision", 0)
        older = {row["revision"]: row for row in history[root] if row["revision"] < revision}
        versions = [(older[number], "superseded", number) for number in sorted(older)]
        versions.append((line, "active" if active else "deleted", revision))
        predecessor = None
        turns, steps = [], []
        for row, status, number in versions:
            stamp = (row.get("updated_at") if row is line else row.get("t_start"))
            if stamp is None:
                stamp = line.get("created_at") or 0.0
            current_steps = merge_step_ids(json.loads(row.get("source_step_ids") or "[]"),
                                           step_ids_in(row["plain"]), step_ids_in(row["formatted"]))
            verified = [ids[0] for step in current_steps if len(ids := tagged_sources[(owner, step)]) == 1]
            exact = exact_sources[(owner, stamp, strip_step_tags(row["plain"]))]
            if len(exact) == 1:
                verified.extend(exact)
            turns, steps = merge_step_ids(turns, verified), merge_step_ids(steps, current_steps)
            fact_id = root if predecessor is None else f"{root}#{number}"
            valid_until = row.get("valid_until")
            if valid_until is None:
                valid_until = row.get("t_end")
            if status != "active" and valid_until is None:
                valid_until = stamp
            data = store._version_data(fact_id=fact_id, block_id=line["block_id"], seq=line["seq"], scope=owner,
                                       formatted=row["formatted"], plain=row["plain"], attribute=row.get("attribute"),
                                       value=row.get("value"), surprise=row.get("surprise"), layout=row.get("layout"),
                                       isolated=row.get("isolated", line.get("isolated")), is_conflict=row.get("is_conflict", False),
                                       role=row.get("role", line.get("role", "user")), event_time_start=row.get("event_time"),
                                       ingest_time=stamp, source_turn_ids=turns, source_step_ids=steps, status=status,
                                       revision=number, supersedes_id=predecessor, root_fact_id=root,
                                       valid_until=valid_until, hit_count=row.get("hit_count", 0))
            store._insert_version(data, keys[root] if row is line else json.loads(row.get("zkeys") or "[]"), offline=True)
            predecessor = fact_id
    for prefix, table in (("l", "facts"), ("b", "blocks")):
        allocated = [int(row[0][1:]) for row in conn.execute(f"SELECT id FROM {table}")
                     if row[0].startswith(prefix) and row[0][1:].isdigit()]
        store.set_meta(f"seq_{prefix}", str(max([int(store.get_meta(f"seq_{prefix}") or 0), *allocated])))
    for old in ("line_zkeys", "line_history", "lines", "ingest"):
        if old in tables:
            conn.execute(f"DROP TABLE {old}")
