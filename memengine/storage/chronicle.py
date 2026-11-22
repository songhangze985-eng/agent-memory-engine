"""Portable ledger columns and scoped, bounded graph operations."""
from __future__ import annotations

import hashlib
import json
import re
from math import exp

import numpy as np

from memengine.chrono import DAY, date_iso, occurrence, resolve_temporal_spans
from memengine.lifecycle import allergy_slot
from memengine.linguistics import entity_name, index_text, stem_terms
from memengine.storage.base import IDS, scope_where
from memengine.textutil import merge_step_ids, normalize_entity, strip_step_tags
from memengine.versioning import extract_versions

FACT_COLUMNS = {
    "mention_time": "DOUBLE PRECISION NOT NULL DEFAULT 0",
    "occurrence_start": "DOUBLE PRECISION",
    "occurrence_end": "DOUBLE PRECISION",
    "time_precision": "TEXT NOT NULL DEFAULT 'turn'",
    "occurrence_spans": "TEXT NOT NULL DEFAULT '[]'",
    "speaker": "TEXT",
    "normalized_text": "TEXT",
    "canonical_entity_id": "TEXT",
    "attribute_slot": "TEXT",
    "origin_turn": "BIGINT",
    "is_stale": "INTEGER NOT NULL DEFAULT 0",
}
RAW_COLUMNS = {"speaker": "TEXT", "mention_time": "DOUBLE PRECISION"}
SCHEMA = (
    """CREATE TABLE IF NOT EXISTS entities (
        id TEXT PRIMARY KEY, user_id TEXT NOT NULL, agent_id TEXT NOT NULL,
        canonical_name TEXT NOT NULL, UNIQUE(user_id,agent_id,canonical_name))""",
    """CREATE TABLE IF NOT EXISTS entity_aliases (
        user_id TEXT NOT NULL, agent_id TEXT NOT NULL, alias TEXT NOT NULL,
        entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
        PRIMARY KEY(user_id,agent_id,alias,entity_id))""",
    """CREATE TABLE IF NOT EXISTS memory_links (
        src_id TEXT NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
        dst_id TEXT NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
        link_type TEXT NOT NULL CHECK(link_type IN ('entity','temporal','semantic','causal')),
        weight DOUBLE PRECISION NOT NULL CHECK(weight>0 AND weight<=1), meta TEXT NOT NULL DEFAULT '',
        PRIMARY KEY(src_id,dst_id,link_type), CHECK(src_id<>dst_id))""",
    "CREATE INDEX IF NOT EXISTS idx_memory_links_reverse ON memory_links(dst_id,src_id)",
    "CREATE INDEX IF NOT EXISTS idx_fact_origin ON facts(user_id,agent_id,session_id,origin_turn)",
    "CREATE INDEX IF NOT EXISTS idx_fact_occurrence ON facts(user_id,agent_id,occurrence_start,occurrence_end)",
    "CREATE INDEX IF NOT EXISTS idx_fact_slot ON facts(user_id,agent_id,canonical_entity_id,attribute_slot,status)",
)
_GENERIC = {"user", "assistant", "agent", "tool", "it", "i", "he", "she", "image", "great", "yeah", "thanks", "that", "what"}
_ASSERTION = re.compile(r"\b(?:uses?|runs?|upgrad(?:ed|e) to|switched? to|now on|moved to|currently lives? in)\b|改为|升级到|使用|搬到|现居", re.I)
_REPLY = re.compile(r"^(?:[^:]{1,32}:\s*)?(?:yes\b|no\b|exactly\b|that\b|it\b|they\b|indeed\b|对[，。]|是的|那件|这件)", re.I)
_CAUSAL = re.compile(r"\bbecause\b|\btherefore\b|\bled to\b|因为|导致", re.I)
_PREFERENCE = re.compile(r"^(?:[^:]{1,128}:\s*)?I (?:now |currently )?prefer [^.!?;\n]+? for "
                         r"(?:my |the )?([a-z]+(?: [a-z]+){0,2})(?=\s*[.!?;]|$)", re.I)


def dump(value):
    return json.dumps(value, ensure_ascii=False)


class Chronicle:
    def migrate_chronicle(self, *, postgres=False):
        if self.get_meta("chronicle_schema_version") == "1":
            return
        for table, columns in (("facts", FACT_COLUMNS), ("raw_turns", RAW_COLUMNS)):
            existing = () if postgres else {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")}
            for name, definition in columns.items():
                if name not in existing:
                    guard = "IF NOT EXISTS " if postgres else ""
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {guard}{name} {definition}")
        self.conn.execute("UPDATE facts SET mention_time=ingest_time WHERE mention_time=0")
        for row in self.conn.execute("SELECT id,plain FROM facts WHERE id IN (SELECT memory_id FROM exact_identifiers WHERE ident_type='version')").fetchall():
            self.index_versions(row["id"], row["plain"])
        for statement in SCHEMA:
            self.conn.execute(statement)
        self.set_meta("chronicle_schema_version", "1")

    def canonical_entity(self, speaker, scope):
        key = entity_name(speaker)
        if not key:
            raise ValueError("speaker must contain a name")
        owner = scope[:2]
        hits = self.conn.execute("""SELECT e.id,e.canonical_name FROM entity_aliases a JOIN entities e ON e.id=a.entity_id
            WHERE a.user_id=? AND a.agent_id=? AND a.alias=? ORDER BY e.id LIMIT 2""", (*owner, key)).fetchall()
        if len(hits) == 1:
            return hits[0]["id"], hits[0]["canonical_name"]
        parts = key.split()
        existing = None
        if len(parts) >= 2:
            early = self.conn.execute("SELECT id FROM entities WHERE user_id=? AND agent_id=? AND canonical_name IN (?,?)",
                                      (*owner, parts[0], parts[-1])).fetchall()
            if len(early) == 1:
                existing = early[0][0]
                self.conn.execute("UPDATE entities SET canonical_name=? WHERE id=?", (key, existing))
        ident = existing or hashlib.sha256(dump([*owner, key]).encode()).hexdigest()[:32]
        self.conn.execute("INSERT OR IGNORE INTO entities VALUES(?,?,?,?)", (ident, *owner, key))
        aliases = sorted({key, parts[0], parts[-1]})
        self.conn.executemany("INSERT OR IGNORE INTO entity_aliases VALUES(?,?,?,?)",
                              [(*owner, alias, ident) for alias in aliases])
        return ident, key

    def _alias_arguments(self, query, scope):
        words = re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", entity_name(query))
        choices = sorted({" ".join(words[i:i+n]) for i in range(len(words)) for n in (1, 2, 3) if i+n <= len(words)})
        return (*scope[:2], dump(choices))

    def aliases_for_query(self, query, scope):
        return self.recall_entity_keys(query, scope=scope)[1]

    def unique_entity_aliases(self, entity_id, scope):
        rows = self.conn.execute("""SELECT alias FROM entity_aliases WHERE user_id=? AND agent_id=?
            GROUP BY alias HAVING count(*)=1 AND min(entity_id)=? ORDER BY alias""", (*scope[:2], entity_id))
        return tuple(row["alias"] for row in rows)

    def populate_chronicle(self, data, zkeys, details):
        data["mention_time"] = data["ingest_time"]
        if not details:
            return list(zkeys)
        mention = details["mention_time"]
        speaker = details.get("speaker")
        quote = details.get("raw") or data["plain"]
        spans = resolve_temporal_spans(data["plain"], mention)
        if not spans and quote != data["plain"]:
            spans = resolve_temporal_spans(quote, mention)
        primary = occurrence(quote, spans)
        data.update(mention_time=mention, speaker=speaker,
                    origin_turn=details.get("origin_turn"), canonical_entity_id=details.get("canonical_entity_id"),
                    occurrence_start=primary.start if primary else None, occurrence_end=primary.end if primary else None,
                    time_precision=primary.precision if primary else "turn",
                    occurrence_spans=dump([{"start": item.start, "end": item.end, "precision": item.precision} for item in spans]),
                    normalized_text=index_text(data["plain"], speaker, spans))
        if speaker:
            body = re.sub(r"^" + re.escape(speaker) + r":\s*", "", data["plain"], count=1, flags=re.I)
            data["formatted"] = f"- [{date_iso(mention)} | {speaker}] {body}"
        keys = list(zkeys)
        subject = speaker or next((normalize_entity(key) for key in zkeys if key.startswith(("entity:", "实体:"))), None)
        if subject:
            ident, canonical = self.canonical_entity(subject, (data["user_id"], data["agent_id"], data["session_id"]))
            data["canonical_entity_id"] = ident
            keys.append(f"entity:{canonical}")
            alias = entity_name(subject)
            if alias != canonical:
                keys.append(f"entity:{alias}")
        versions = extract_versions(data["plain"])
        version = versions[0] if len(versions) == 1 and versions[0].qualifier_op == "==" else None
        if data.get("expires_at") is None and data["epistemic_scope"] == "world" and _ASSERTION.search(data["plain"]):
            if version and version.product_family != "default_pkg":
                data["attribute_slot"] = f"runtime_version:{version.product_family}"
            elif data.get("attribute", "").casefold() in {"location", "residence", "address", "workplace"}:
                data["attribute_slot"] = data["attribute"].casefold()
        if data.get("expires_at") is None and data["epistemic_scope"] == "world" and data["canonical_entity_id"]:
            if preference := _PREFERENCE.match(strip_step_tags(data["plain"])):
                slot = " ".join(preference[1].casefold().split())
                data["attribute_slot"] = f"preference:{slot}"
            if data["role"] == "user" and data.get("memory_type", "fact") == "fact":
                slot = allergy_slot(data["plain"], speaker)
                if slot:
                    data["attribute_slot"] = slot
        if "derived_from_ids" in details:
            self.validate_fact_ids(details["derived_from_ids"], (data["user_id"], data["agent_id"], None))
            data["derived_from_ids"] = dump(details["derived_from_ids"])
        if data.get("attribute_slot") and data.get("canonical_entity_id") and not data.get("supersedes_id"):
            self.revise_slot(data)
        return list(dict.fromkeys(keys))

    def revise_slot(self, data):
        allergy = data["attribute_slot"].startswith("allergy:")
        row = self.conn.execute("""SELECT id,root_fact_id,revision,ingest_time,source_turn_ids,source_step_ids,plain FROM facts
            WHERE user_id=? AND agent_id=? AND canonical_entity_id=? AND attribute_slot=? AND status='active'
            ORDER BY mention_time DESC,id DESC LIMIT 1""",
            tuple(data[key] for key in ("user_id", "agent_id", "canonical_entity_id", "attribute_slot"))).fetchone()
        if row is None:
            return
        if row["ingest_time"] > data["ingest_time"]:
            if allergy:
                data.update(status="superseded", valid_until=row["ingest_time"])
            return
        self.conn.execute("UPDATE facts SET status='superseded',valid_until=? WHERE id=?", (data["ingest_time"], row["id"]))
        self.invalidate_observations((data["user_id"], data["agent_id"]), [row["id"]], data["ingest_time"],
                                     compiled_line=row["plain"] if allergy else "")
        data.update(supersedes_id=row["id"], revision=row["revision"]+1, root_fact_id=row["root_fact_id"])
        if not allergy:
            for field in ("source_turn_ids", "source_step_ids"):
                data[field] = dump(merge_step_ids(json.loads(row[field]), json.loads(data[field])))

    def migrate_allergy_slots(self):
        if self.get_meta("allergy_slots_version") == "1":
            return
        rows = self.conn.execute("""SELECT id,plain,speaker,user_id,agent_id,canonical_entity_id,ingest_time,
                status,valid_until,root_fact_id,revision,supersedes_id FROM facts
            WHERE status='active' AND memory_type='fact' AND epistemic_scope='world' AND role='user'
              AND speaker IS NOT NULL AND canonical_entity_id IS NOT NULL AND expires_at IS NULL
              AND attribute_slot IS NULL AND revision=0 AND (lower(plain) LIKE ? OR plain LIKE ?)
            ORDER BY ingest_time,length(id),id""", ("%allerg%", "%过敏%")).fetchall()
        for row in rows:
            slot = allergy_slot(row["plain"], row["speaker"])
            if slot is None:
                continue
            data = {**dict(row), "attribute_slot": slot}
            self.revise_slot(data)
            self.conn.execute("""UPDATE facts SET attribute_slot=?,status=?,valid_until=?,root_fact_id=?,
                revision=?,supersedes_id=? WHERE id=?""",
                tuple(data[key] for key in ("attribute_slot", "status", "valid_until", "root_fact_id",
                                            "revision", "supersedes_id", "id")))
        self.set_meta("allergy_slots_version", "1")

    def index_versions(self, fact_id, text):
        rows = []
        for version in extract_versions(text):
            for depth in range(1, version.precision_depth + 1):
                key = version_key(version.numeric_tuple[:depth])
                rows.append((fact_id, version.literal, key, "version_precision"))
            if version.qualifier_op != "==":
                rows.append((fact_id, version.literal, "version-range:" + version.product_family, "version_precision"))
        if rows:
            self.conn.executemany("INSERT OR IGNORE INTO exact_identifiers VALUES(?,?,?,?)", rows)

    def version_candidates(self, wanted, *, scope, filters, include_cold=False, top_n=24):
        where, args = scope_where(scope, "f.")
        where, args = filters.sql(where, args)
        keys = {version_key(v.numeric_tuple[:min(2, v.precision_depth)]) for v in wanted}
        keys.update("version-range:" + v.product_family for v in wanted)
        rows = self.conn.execute(f"""SELECT f.id FROM exact_identifiers e JOIN facts f ON f.id=e.memory_id
            WHERE {where} AND {self._searchable(True, include_cold, filters)} AND e.ident_type='version_precision'
              AND e.ident_norm IN ({IDS}) GROUP BY f.id ORDER BY f.ingest_time DESC,f.id LIMIT ?""",
            (*args, dump(sorted(keys)), top_n))
        return [{"id": row["id"]} for row in rows]

    def occurrence_candidates(self, windows, *, scope, filters, include_cold=False, top_n=35):
        where, args = scope_where(scope, "f.")
        where, args = filters.sql(where, args)
        clauses = []
        params = []
        for window in windows[:3]:
            if window.start is None and window.end is None:
                continue
            clauses.append("(f.occurrence_start<=? AND f.occurrence_end>=?)")
            params.extend((window.end if window.end is not None else 253402300799,
                           window.start if window.start is not None else -62135596800))
        if not clauses:
            return []
        rows = self.conn.execute(f"""SELECT f.id FROM facts f WHERE {where}
            AND {self._searchable(True, include_cold, filters)} AND ({" OR ".join(clauses)})
            ORDER BY f.occurrence_start DESC, f.id LIMIT ?""", (*args, *params, top_n))
        return [{"id": row["id"]} for row in rows]

    def link_chronicle(self, data, zkeys, model, vector):
        fact_id, user, agent, session = (data[k] for k in ("id", "user_id", "agent_id", "session_id"))
        links = {}
        keys = {normalize_entity(key) for key in zkeys if key.startswith(("entity:", "实体:", "person:", "人物:"))}
        keys = sorted(keys - _GENERIC)
        if keys:
            rows = self.conn.execute(f"""WITH matched AS (
                SELECT f.id,f.session_id,f.mention_time,count(*) AS hits
                FROM fact_entities e JOIN facts f ON f.id=e.fact_id
                WHERE e.user_id=? AND e.agent_id=? AND e.entity_norm IN ({IDS})
                  AND e.key_type='entity' AND f.id<>? AND f.status='active' AND f.tier='hot'
                GROUP BY f.id
            ), spread AS (
                SELECT *, row_number() OVER(PARTITION BY session_id ORDER BY hits DESC,mention_time DESC,id) AS rn
                FROM matched
            ) SELECT id,hits FROM spread ORDER BY hits DESC,rn,mention_time DESC,id LIMIT 12""",
                (user, agent, dump(keys), fact_id)).fetchall()
            for row in rows[:3]:
                links[row["id"], "entity"] = (min(1, .6 + .15 * row["hits"]), "")
        else:
            rows = []
        semantic_ids = {row["id"] for row in rows}
        query = " ".join(sorted(stem_terms(data["plain"]) - _GENERIC)[:24])
        if query:
            semantic_ids.update(row["id"] for row in self.search_lexical(query, user_id=user, agent_id=agent,
                                include_superseded=False, top_n=12, ids_only=True))
        semantic_ids.discard(fact_id)
        if semantic_ids:
            semantic = []
            source_vector = np.frombuffer(vector, dtype="<f4")
            for row in self.conn.execute(f"""SELECT fact_id,vec FROM fact_vectors WHERE user_id=? AND agent_id=?
                    AND model_name=? AND dim=? AND fact_id IN ({IDS})""",
                    (user, agent, model, len(source_vector), dump(sorted(semantic_ids)))):
                candidate = (np.frombuffer(row["vec"], dtype="<f4") if isinstance(row["vec"], bytes)
                             else np.asarray(row["vec"], dtype="<f4"))
                score = min(1.0, float(np.dot(source_vector, candidate)))
                if score >= .26:
                    semantic.append((score, row["fact_id"]))
            for weight, ident in sorted(semantic, reverse=True)[:3]:
                links[ident, "semantic"] = (weight, "indexed-neighbors")
        event = data["occurrence_start"]
        if event is not None:
            rows = self.conn.execute("""SELECT id,occurrence_start FROM facts WHERE user_id=? AND agent_id=?
                AND occurrence_start BETWEEN ? AND ? AND id<>? AND status='active'
                ORDER BY abs(occurrence_start-?),id LIMIT 3""", (user, agent, event-7*DAY, event+7*DAY, fact_id, event))
            for row in rows:
                links[row["id"], "temporal"] = (exp(-abs(row["occurrence_start"]-event)/(3*DAY)), "")
        if data.get("supersedes_id"):
            links[data["supersedes_id"], "causal"] = (1, "revision")
        if data.get("origin_turn"):
            rows = self.conn.execute("""SELECT f.id,r.text FROM facts f JOIN raw_turns r ON r.turn_id=f.origin_turn
                WHERE f.user_id=? AND f.agent_id=? AND f.session_id=? AND f.origin_turn=?
                ORDER BY f.id LIMIT 2""", (user, agent, session, data["origin_turn"]-1))
            for row in rows:
                if "?" in row["text"] or "？" in row["text"] or _CAUSAL.search(data["plain"]) or _REPLY.search(data["plain"]):
                    links[row["id"], "causal"] = (.85, "adjacent")
        self.conn.executemany("INSERT OR IGNORE INTO memory_links VALUES(?,?,?,?,?)",
                              [(fact_id, other, kind, weight, meta) for (other, kind), (weight, meta) in links.items()])
        if data.get("canonical_entity_id") and data["epistemic_scope"] in ("world", "experience"):
            self.conn.execute("""UPDATE facts SET is_stale=1 WHERE user_id=? AND agent_id=? AND
                canonical_entity_id=? AND memory_type='observation' AND status='active'""",
                (user, agent, data["canonical_entity_id"]))

    def adjacent_links(self, seeds, *, scope, filters, include_cold=False, limit=16):
        if not seeds:
            return []
        where, args = scope_where(scope, "f.")
        where, args = filters.sql(where, args)
        searchable = self._searchable(True, include_cold, filters)
        seed_list = dump(sorted(seeds))
        rows = self.conn.execute(f"""WITH edge AS (
            SELECT src_id AS seed,dst_id AS target,weight,link_type FROM memory_links WHERE src_id IN ({IDS})
            UNION ALL
            SELECT dst_id,src_id,weight,link_type FROM memory_links WHERE dst_id IN ({IDS})
        ), chosen AS MATERIALIZED (
            SELECT e.*, f.canonical_entity_id, count(*) OVER(PARTITION BY e.seed) AS degree,
             row_number() OVER(PARTITION BY e.seed ORDER BY e.weight DESC,e.target,e.link_type) AS rn
            FROM edge e CROSS JOIN facts f ON f.id=e.target WHERE {where} AND {searchable}
        ), picked AS MATERIALIZED (
            SELECT * FROM chosen WHERE rn<=?
        ), targets AS (
            SELECT DISTINCT target FROM picked
        ), incidence AS (
            SELECT m.src_id AS node,m.dst_id AS peer FROM targets t JOIN memory_links m ON m.src_id=t.target
            UNION ALL
            SELECT m.dst_id,m.src_id FROM targets t JOIN memory_links m ON m.dst_id=t.target
        ), degrees AS (
            SELECT i.node,count(*) AS degree FROM incidence i
            CROSS JOIN facts f ON f.id=i.peer WHERE {where} AND {searchable} GROUP BY i.node
        ) SELECT p.seed,p.target,p.weight,p.link_type,p.degree,coalesce(d.degree,1) AS target_degree,
                 CASE WHEN s.canonical_entity_id=p.canonical_entity_id THEN 1 ELSE 0 END AS shared_speaker
          FROM picked p JOIN facts s ON s.id=p.seed LEFT JOIN degrees d ON d.node=p.target
          ORDER BY p.seed,p.rn""", (seed_list, seed_list, *args, limit, *args))
        return [dict(row) for row in rows]

    def recent_entity_facts(self, entity_ids, *, scope, after, limit=12, filters=None):
        if not entity_ids:
            return []
        where, args = scope_where(scope, "f.")
        if filters is not None:
            where, args = filters.sql(where, args)
        rows = self.conn.execute(f"""SELECT f.id FROM facts f WHERE {where} AND f.canonical_entity_id IN ({IDS})
            AND cast(substr(f.id,2) AS INTEGER)>? AND f.status='active' AND f.epistemic_scope IN ('world','experience')
            ORDER BY f.mention_time DESC,f.id LIMIT ?""", (*args, dump(sorted(entity_ids)), after, limit))
        return self.facts_by_ids([row["id"] for row in rows], scope=scope)


def version_key(digits):
    return "version-prefix:" + ".".join(map(str, digits))
