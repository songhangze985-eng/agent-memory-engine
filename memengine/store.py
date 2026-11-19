"""SQLite persistence for lines, Z keys, timeline snapshots, and ingest log."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from memengine.textutil import ensure_step_tags, merge_step_ids, step_ids_in


_SCHEMA = """
CREATE TABLE IF NOT EXISTS blocks (
  id TEXT PRIMARY KEY,
  topic TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS lines (
  id TEXT PRIMARY KEY,
  block_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  formatted TEXT NOT NULL,
  plain TEXT NOT NULL,
  attribute TEXT,
  value TEXT,
  surprise REAL,
  raw_dist REAL,
  layout TEXT,
  isolated INTEGER NOT NULL DEFAULT 0,
  revision INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  active INTEGER NOT NULL DEFAULT 1,
  source_step_ids TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS line_zkeys (
  line_id TEXT NOT NULL,
  zkey TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS line_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  line_id TEXT NOT NULL,
  revision INTEGER NOT NULL,
  formatted TEXT NOT NULL,
  plain TEXT NOT NULL,
  attribute TEXT,
  value TEXT,
  zkeys TEXT NOT NULL,
  surprise REAL,
  layout TEXT,
  t_start REAL,
  t_end REAL,
  source_step_ids TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS timeline (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  layer INTEGER NOT NULL,
  position INTEGER NOT NULL,
  t_start REAL NOT NULL,
  t_end REAL NOT NULL,
  payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ingest (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  text TEXT NOT NULL,
  ts REAL
);
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_zkey ON line_zkeys(zkey);
CREATE INDEX IF NOT EXISTS idx_lines_block ON lines(block_id, seq);
"""


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def commit(self) -> None:
        self.conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else row["value"]

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def next_id(self, prefix: str) -> str:
        current = int(self.get_meta(f"seq_{prefix}") or "0") + 1
        self.set_meta(f"seq_{prefix}", str(current))
        return f"{prefix}{current}"

    def _parse_step_ids(self, raw) -> list[int]:
        if isinstance(raw, list):
            return merge_step_ids(raw)
        if not raw:
            return []
        try:
            return merge_step_ids(json.loads(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            return step_ids_in(str(raw))

    def _dump_step_ids(self, ids) -> str:
        return json.dumps(merge_step_ids(ids), ensure_ascii=False)

    def create_block(self, topic: str, timestamp: float) -> str:
        block_id = self.next_id("b")
        self.conn.execute(
            "INSERT INTO blocks(id, topic, created_at) VALUES(?, ?, ?)",
            (block_id, topic, float(timestamp)),
        )
        return block_id

    def get_block(self, block_id: str) -> dict | None:
        row = self.conn.execute("SELECT id, topic, created_at FROM blocks WHERE id = ?", (block_id,)).fetchone()
        return None if row is None else dict(row)

    def block_by_topic(self, topic: str) -> dict | None:
        row = self.conn.execute(
            "SELECT id, topic, created_at FROM blocks WHERE topic = ? ORDER BY created_at LIMIT 1",
            (topic,),
        ).fetchone()
        return None if row is None else dict(row)

    def insert_line(self, *, block_id, formatted, plain, attribute, value, surprise, raw_dist, layout, isolated, zkeys, timestamp, source_step_ids=None) -> str:
        line_id = self.next_id("l")
        seq = self.conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 AS n FROM lines WHERE block_id = ?", (block_id,)).fetchone()["n"]
        steps = merge_step_ids(source_step_ids, step_ids_in(plain or ""), step_ids_in(formatted or ""))
        self.conn.execute(
            """
            INSERT INTO lines(
              id, block_id, seq, formatted, plain, attribute, value, surprise, raw_dist,
              layout, isolated, revision, created_at, updated_at, active, source_step_ids
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, 1, ?)
            """,
            (
                line_id,
                block_id,
                seq,
                formatted,
                plain,
                attribute,
                value,
                surprise,
                raw_dist,
                layout,
                1 if isolated else 0,
                float(timestamp),
                float(timestamp),
                self._dump_step_ids(steps),
            ),
        )
        self._replace_zkeys(line_id, zkeys)
        return line_id

    def get_line(self, line_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM lines WHERE id = ? AND active = 1", (line_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["isolated"] = bool(data["isolated"])
        data["zkeys"] = self._zkeys_for(line_id)
        data["source_step_ids"] = self._parse_step_ids(data.get("source_step_ids"))
        return data

    def archive_line(self, line: dict, t_end: float) -> None:
        self.conn.execute(
            """
            INSERT INTO line_history(
              line_id, revision, formatted, plain, attribute, value, zkeys, surprise, layout, t_start, t_end, source_step_ids
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                line["id"],
                int(line["revision"]),
                line["formatted"],
                line["plain"],
                line.get("attribute"),
                line.get("value"),
                json.dumps(line.get("zkeys") or [], ensure_ascii=False),
                line.get("surprise"),
                line.get("layout"),
                line.get("updated_at"),
                float(t_end),
                self._dump_step_ids(line.get("source_step_ids") or step_ids_in(line.get("plain") or "")),
            ),
        )

    def update_line(self, line_id, *, formatted, plain, attribute, value, surprise, raw_dist, layout, isolated, zkeys, timestamp, source_step_ids=None) -> None:
        steps = merge_step_ids(source_step_ids, step_ids_in(plain or ""), step_ids_in(formatted or ""))
        self.conn.execute(
            """
            UPDATE lines
               SET formatted = ?, plain = ?, attribute = ?, value = ?, surprise = ?, raw_dist = ?,
                   layout = ?, isolated = ?, revision = revision + 1, updated_at = ?, source_step_ids = ?
             WHERE id = ?
            """,
            (
                formatted,
                plain,
                attribute,
                value,
                surprise,
                raw_dist,
                layout,
                1 if isolated else 0,
                float(timestamp),
                self._dump_step_ids(steps),
                line_id,
            ),
        )
        self._replace_zkeys(line_id, zkeys)

    def _replace_zkeys(self, line_id: str, zkeys) -> None:
        self.conn.execute("DELETE FROM line_zkeys WHERE line_id = ?", (line_id,))
        self.conn.executemany(
            "INSERT INTO line_zkeys(line_id, zkey) VALUES(?, ?)",
            [(line_id, key) for key in zkeys],
        )

    def _zkeys_for(self, line_id: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT zkey FROM line_zkeys WHERE line_id = ? ORDER BY zkey",
            (line_id,),
        ).fetchall()
        return [row["zkey"] for row in rows]

    def append_ingest(self, text: str, timestamp: float) -> None:
        self.conn.execute("INSERT INTO ingest(text, ts) VALUES(?, ?)", (text, float(timestamp)))

    def history_text(self) -> str:
        rows = self.conn.execute("SELECT text FROM ingest ORDER BY id").fetchall()
        return "\n".join(row["text"] for row in rows)

    def blocks_for_prompt(self) -> list[dict]:
        blocks = []
        for block in self.conn.execute("SELECT id, topic FROM blocks ORDER BY created_at, id"):
            lines = []
            rows = self.conn.execute(
                """
                SELECT * FROM lines
                 WHERE block_id = ? AND active = 1
                 ORDER BY seq
                """,
                (block["id"],),
            ).fetchall()
            for row in rows:
                lines.append(
                    {
                        "id": row["id"],
                        "text": row["plain"],
                        "formatted": row["formatted"],
                        "attribute": row["attribute"] or "",
                        "value": row["value"] or "",
                        "zkeys": self._zkeys_for(row["id"]),
                        "isolated": bool(row["isolated"]),
                        "layout": row["layout"],
                        "surprise": row["surprise"],
                        "revision": row["revision"],
                        "updated_at": row["updated_at"],
                        "source_step_ids": self._parse_step_ids(row["source_step_ids"] if "source_step_ids" in row.keys() else "[]"),
                    }
                )
            blocks.append({"id": block["id"], "topic": block["topic"], "lines": lines})
        return blocks

    def z_index(self) -> dict[str, list[dict]]:
        index: dict[str, list[dict]] = {}
        rows = self.conn.execute(
            """
            SELECT l.*, b.topic AS topic
              FROM lines l
              JOIN blocks b ON b.id = l.block_id
             WHERE l.active = 1
             ORDER BY l.updated_at, l.id
            """
        ).fetchall()
        for row in rows:
            zkeys = self._zkeys_for(row["id"])
            source_ids = self._parse_step_ids(row["source_step_ids"] if "source_step_ids" in row.keys() else "[]")
            source_ids = merge_step_ids(source_ids, step_ids_in(row["plain"] or ""), step_ids_in(row["formatted"] or ""))
            fact = {
                "id": row["id"],
                "text": ensure_step_tags(row["plain"], source_ids),
                "formatted": row["formatted"],
                "raw_dist": row["raw_dist"] if row["raw_dist"] is not None else 0.5,
                "surprise": row["surprise"] if row["surprise"] is not None else 0.5,
                "zkeys": zkeys,
                "attribute": row["attribute"] or "",
                "value": row["value"] or "",
                "revision": int(row["revision"]),
                "updated_at": row["updated_at"],
                "layout": row["layout"] or "",
                "isolated": bool(row["isolated"]),
                "block_id": row["block_id"],
                "topic": row["topic"],
                "historical": False,
                "source_step_ids": source_ids,
            }
            for key in zkeys:
                index.setdefault(key, []).append(dict(fact))
        return index

    def historical_versions(self, line_ids) -> list[dict]:
        ids = [line_id for line_id in line_ids if line_id]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"""
            SELECT * FROM line_history
             WHERE line_id IN ({placeholders})
             ORDER BY line_id, revision
            """,
            ids,
        ).fetchall()
        versions = []
        for row in rows:
            zkeys = json.loads(row["zkeys"] or "[]")
            keys = row.keys()
            source_ids = self._parse_step_ids(row["source_step_ids"] if "source_step_ids" in keys else "[]")
            source_ids = merge_step_ids(source_ids, step_ids_in(row["plain"] or ""), step_ids_in(row["formatted"] or ""))
            versions.append(
                {
                    "id": f"{row['line_id']}#{row['revision']}",
                    "line_id": row["line_id"],
                    "text": ensure_step_tags(row["plain"], source_ids),
                    "formatted": row["formatted"],
                    "raw_dist": 0.8,
                    "surprise": 0.8 if row["surprise"] is None else row["surprise"],
                    "zkeys": zkeys,
                    "attribute": row["attribute"] or "",
                    "value": row["value"] or "",
                    "revision": int(row["revision"]),
                    "updated_at": row["t_end"],
                    "layout": row["layout"] or "OVERWRITE_SAME_LINE",
                    "isolated": False,
                    "historical": True,
                    "source_step_ids": source_ids,
                }
            )
        return versions

    def replace_timeline(self, layer_no: int, nodes: list[dict]) -> None:
        self.conn.execute("DELETE FROM timeline WHERE layer = ?", (layer_no,))
        for position, node in enumerate(nodes):
            payload = {k: v for k, v in node.items() if k not in ("t_start", "t_end")}
            self.conn.execute(
                "INSERT INTO timeline(layer, position, t_start, t_end, payload) VALUES(?, ?, ?, ?, ?)",
                (
                    layer_no,
                    position,
                    float(node["t_start"]),
                    float(node["t_end"]),
                    json.dumps(payload, ensure_ascii=False),
                ),
            )

    def load_timeline(self, layer_no: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT t_start, t_end, payload FROM timeline WHERE layer = ? ORDER BY position, id",
            (layer_no,),
        ).fetchall()
        nodes = []
        for row in rows:
            payload = json.loads(row["payload"] or "{}")
            nodes.append({"t_start": row["t_start"], "t_end": row["t_end"], **payload})
        return nodes

    def active_snapshot(self) -> list[dict]:
        snap = []
        for block in self.blocks_for_prompt():
            rendered = []
            for line in block["lines"]:
                rendered.append(
                    {
                        "id": line["id"],
                        "plain": line["text"],
                        "attribute": line["attribute"],
                        "value": line["value"],
                        "zkeys": list(line["zkeys"]),
                        "revision": line["revision"],
                        "isolated": line["isolated"],
                        "layout": line["layout"],
                        "topic": block["topic"],
                        "formatted": line["formatted"],
                        "source_step_ids": list(line.get("source_step_ids") or step_ids_in(line.get("plain") or line.get("text") or "")),
                    }
                )
            if rendered:
                snap.append({"block_id": block["id"], "topic": block["topic"], "lines": rendered})
        return snap
