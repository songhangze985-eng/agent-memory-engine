"""Task workspaces stored alongside the ledger, with exact owner keys."""

import json


_KEY = "user_id=? AND agent_id=? AND session_id=? AND task_id=?"
_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS working_tasks (
        user_id TEXT NOT NULL, agent_id TEXT NOT NULL, session_id TEXT NOT NULL, task_id TEXT NOT NULL,
        goal TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('active','paused','committed','aborted')),
        checkpoint_json TEXT NOT NULL, created_at DOUBLE PRECISION NOT NULL,
        updated_at DOUBLE PRECISION NOT NULL, expires_at DOUBLE PRECISION,
        PRIMARY KEY (user_id,agent_id,session_id,task_id)
    )""",
    """CREATE TABLE IF NOT EXISTS working_items (
        user_id TEXT NOT NULL, agent_id TEXT NOT NULL, session_id TEXT NOT NULL, task_id TEXT NOT NULL,
        item_id TEXT NOT NULL,
        item_type TEXT NOT NULL CHECK(item_type IN ('tool_evidence','temp_fact','provisional_claim','open_question')),
        content TEXT NOT NULL, source_refs_json TEXT NOT NULL, verifier_ref_json TEXT,
        support_turn_id BIGINT, support_fact_id TEXT,
        verification_status TEXT NOT NULL CHECK(verification_status IN ('unverified','tool_verified','user_confirmed','rejected')),
        promoted_fact_id TEXT REFERENCES facts(id) ON DELETE SET NULL, created_at DOUBLE PRECISION NOT NULL,
        PRIMARY KEY (user_id,agent_id,session_id,task_id,item_id),
        FOREIGN KEY (user_id,agent_id,session_id,task_id)
            REFERENCES working_tasks(user_id,agent_id,session_id,task_id) ON DELETE CASCADE
    )""",
    """CREATE INDEX IF NOT EXISTS idx_working_items_recent
        ON working_items(user_id,agent_id,session_id,task_id,created_at,item_id)""",
    """CREATE INDEX IF NOT EXISTS idx_stale_session_sources
        ON facts(user_id,agent_id,session_id,origin_turn)
        WHERE status!='active' OR expires_at IS NOT NULL OR valid_from IS NOT NULL""",
)


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _row(row, *fields):
    result = dict(row)
    for name in fields:
        data = result.pop(name + "_json")
        result[name] = json.loads(data) if data is not None else None
    return result


class WorkspaceStore:
    def migrate_workspace(self, *, postgres=False):
        for statement in _SCHEMA:
            self.conn.execute(statement)
        if postgres:
            self.conn.execute("ALTER TABLE tool_calls ADD COLUMN IF NOT EXISTS task_id TEXT")
        elif "task_id" not in {row[1] for row in self.conn.execute("PRAGMA table_info(tool_calls)")}:
            self.conn.execute("ALTER TABLE tool_calls ADD COLUMN task_id TEXT")
        columns = {"support_turn_id": "BIGINT", "support_fact_id": "TEXT"}
        if not postgres:
            present = {row[1] for row in self.conn.execute("PRAGMA table_info(working_items)")}
            columns = {name: kind for name, kind in columns.items() if name not in present}
        for name, kind in columns.items():
            guard = "IF NOT EXISTS " if postgres else ""
            self.conn.execute(f"ALTER TABLE working_items ADD COLUMN {guard}{name} {kind}")

    def create_task(self, key, goal, now, expires_at):
        cursor = self.conn.execute(
            """INSERT INTO working_tasks (user_id,agent_id,session_id,task_id,goal,status,
                checkpoint_json,created_at,updated_at,expires_at) VALUES (?,?,?,?,?,'active','{}',?,?,?)
                ON CONFLICT (user_id,agent_id,session_id,task_id) DO NOTHING""",
            (*key, goal, now, now, expires_at))
        return cursor.rowcount == 1

    def task(self, key):
        row = self.conn.execute(f"SELECT * FROM working_tasks WHERE {_KEY}", key).fetchone()
        return _row(row, "checkpoint") if row is not None else None

    def task_items(self, key, *, limit=None, as_of=None):
        order = "ORDER BY created_at,item_id" if limit is None else "ORDER BY created_at DESC,item_id DESC LIMIT ?"
        args = key if limit is None else (*key, limit)
        if as_of is None:
            rows = self.conn.execute(f"SELECT * FROM working_items WHERE {_KEY} {order}", args)
        else:
            rows = self.conn.execute(f"""WITH selected AS (
                SELECT * FROM working_items WHERE {_KEY} {order}
            ) SELECT w.*, CASE
              WHEN w.support_fact_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM facts f WHERE f.id=w.support_fact_id AND f.status='active'
                  AND f.user_id=w.user_id AND f.agent_id=w.agent_id AND f.session_id=w.session_id
                  AND (f.expires_at IS NULL OR f.expires_at>?) AND (f.valid_from IS NULL OR f.valid_from<=?)
              ) THEN 0
              WHEN w.support_turn_id IS NOT NULL AND (NOT EXISTS (
                SELECT 1 FROM raw_turns t WHERE t.turn_id=w.support_turn_id AND t.user_id=w.user_id
                  AND t.agent_id=w.agent_id AND t.session_id=w.session_id
              ) OR EXISTS (
                SELECT 1 FROM facts f WHERE f.user_id=w.user_id AND f.agent_id=w.agent_id
                  AND f.session_id=w.session_id
                  AND (f.status!='active' OR f.expires_at IS NOT NULL OR f.valid_from IS NOT NULL)
                  AND (f.status!='active' OR f.expires_at<=? OR f.valid_from>?)
                  AND (f.origin_turn=w.support_turn_id OR (f.origin_turn IS NULL AND EXISTS (
                    SELECT 1 FROM json_each(f.source_turn_ids) j WHERE w.support_turn_id=j.value
                  )))
              )) THEN 0 ELSE 1 END AS support_current
              FROM selected w {order.split(' LIMIT ', 1)[0]}""", (*args, as_of, as_of, as_of, as_of))
        return [_row(row, "source_refs", "verifier_ref") for row in rows]

    def task_item(self, key, item_id):
        row = self.conn.execute(f"SELECT * FROM working_items WHERE {_KEY} AND item_id=?", (*key, item_id)).fetchone()
        return _row(row, "source_refs", "verifier_ref") if row is not None else None

    def set_task_state(self, key, status, checkpoint, now):
        self.conn.execute(
            f"UPDATE working_tasks SET status=?,checkpoint_json=?,updated_at=? WHERE {_KEY}",
            (status, _json(checkpoint), now, *key))

    def insert_task_item(self, key, item_id, kind, content, refs, now):
        self.conn.execute(
            """INSERT INTO working_items (user_id,agent_id,session_id,task_id,item_id,item_type,content,
               source_refs_json,verification_status,created_at) VALUES (?,?,?,?,?,?,?,?,'unverified',?)""",
            (*key, item_id, kind, content, _json(refs), now))

    def verify_task_item(self, key, item_id, status, verifier, support_turn, support_fact):
        self.conn.execute(
            f"""UPDATE working_items SET verification_status=?,verifier_ref_json=?,
                support_turn_id=?,support_fact_id=? WHERE {_KEY} AND item_id=?""",
            (status, _json(verifier), support_turn, support_fact, *key, item_id))

    def promote_task_item(self, key, item_id, fact_id):
        self.conn.execute(
            f"UPDATE working_items SET promoted_fact_id=? WHERE {_KEY} AND item_id=?",
            (fact_id, *key, item_id))
