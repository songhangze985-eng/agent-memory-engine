"""Slot-level supersede tracking & relative occurrence date resolution (Milestone M2)."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from agentmem_core.core.text import stem_terms

_DAY = 86400.0
_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
    "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
    "a": "1", "an": "1", "once": "1", "twice": "2",
}
_NUM_WORD_PAT = "|".join(sorted(_NUM_WORDS, key=len, reverse=True))
_NUM_EXPR = rf"(?:\d+(?:\.\d+)?|{ _NUM_WORD_PAT })"

# Relative time offset patterns relative to mention_time
_REL_TODAY = re.compile(
    r"\b(?:today|tonight|this\s+morning|this\s+afternoon|this\s+evening|"
    r"just\s+(?:finished|attended|read|bought|got|started|returned|came\s+back|completed))\b|今天|今晚",
    re.I,
)
_REL_YESTERDAY = re.compile(r"\byesterday\b|昨天", re.I)
_REL_DAYS_AGO = re.compile(rf"\b({_NUM_EXPR})\s+days?\s+ago\b|(\d+)\s*天前", re.I)
_REL_WEEKS_AGO = re.compile(rf"\b(?:about\s+)?({_NUM_EXPR})\s+weeks?\s+ago\b|(\d+)\s*周前", re.I)
_REL_MONTHS_AGO = re.compile(rf"\b(?:about\s+)?(?:for\s+the\s+past\s+)?({_NUM_EXPR})\s+months?(?:\s+ago)?\b|(\d+)\s*个月前", re.I)
_REL_YEARS_AGO = re.compile(rf"\b(?:about\s+)?(?:for\s+the\s+past\s+)?({_NUM_EXPR})\s+years?(?:\s+ago)?\b|(\d+)\s*年前", re.I)
_REL_LAST_WEEK = re.compile(r"\blast\s+week\b|上周", re.I)
_REL_LAST_MONTH = re.compile(r"\blast\s+month\b|上个月", re.I)

# Explicit conversational state slot extractors (high precision — English + Chinese)
_SLOT_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("residence", re.compile(r"\b(?:moved to|relocated to|now live in|living in)\s+([A-Z][A-Za-z\s.-]{1,35}?)(?=[.,;!?\n]|$|\s+(?:and|with|for|since))", re.I)),
    ("residence", re.compile(r"(?:搬到|搬去|现居|住在)([\u4e00-\u9fa5A-Za-z0-9]{2,16})")),
    ("employer", re.compile(r"\b(?:started a new job at|switched jobs to|now work(?:ing)? at|working at)\s+([A-Z][A-Za-z0-9&\s.-]{1,35}?)(?=[.,;!?\n]|$|\s+(?:as|in|on|since))", re.I)),
    ("employer", re.compile(r"(?:入职|跳槽到|换工作到|在)([\u4e00-\u9fa5A-Za-z0-9]{2,16})(?:工作|上班)")),
    ("weight_lost", re.compile(rf"\blost\s+({_NUM_EXPR})\s*(pounds?|lbs?|kg|kilograms?|斤|公斤)\b", re.I)),
    ("coffee_limit", re.compile(rf"\b(?:limit(?:ed)?\s+(?:myself\s+)?to|cut\s+(?:back|down)\s+to|increased\s+(?:my\s+)?limit\s+to|up\s+to)\s+({_NUM_EXPR})\s+cups?\s+of\s+coffee\b", re.I)),
    ("favorite", re.compile(r"\bmy\s+(?:new\s+|current\s+)?favorite\s+([a-z]{3,20})\s+is\s+([A-Za-z0-9\s'&-]{2,40}?)(?=[.,;!?\n]|$)", re.I)),
    ("budget", re.compile(r"(?:预算(?:改为|调整为|是|为)|budget\s+(?:is|now|updated\s+to|changed\s+to))\s*\$?\s*(\d[\d,.]*)", re.I)),
    ("schedule", re.compile(r"\b(?:moved|rescheduled|changed)\s+(?:the\s+|my\s+)?([a-z\s]{3,24}?)\s+to\s+([A-Za-z0-9:\s,]{2,28}?)(?=[.,;!?\n]|$)", re.I)),
)

# High-precision personal countable-state update template ("watched 4 MCU films", "tried four different Korean restaurants")
_COUNT_STATE_RE = re.compile(
    rf"\b(?:i(?:'ve| have)|my)\s+(?:actually\s+|just\s+|recently\s+|now\s+|currently\s+)?"
    rf"(?:watched|tried|read|visited|finished|completed|adopted|collected|run|ran)\s+"
    rf"(?:about\s+|around\s+|over\s+|up\s+to\s+)?({_NUM_EXPR})\s+"
    rf"(?:different\s+|new\s+|more\s+|total\s+)?([a-zA-Z][a-zA-Z0-9-]*(?:\s+[a-zA-Z][a-zA-Z0-9-]*){{0,2}})",
    re.I,
)
_SWITCH_RE = re.compile(
    r"\b(?:switched|upgraded)\s+(?:from\s+[^.,;!?]{2,30}\s+)?to\s+(?:a\s+|an\s+|the\s+)?([A-Za-z0-9][A-Za-z0-9\s-]{2,32}?)(?=[.,;!?\n]|$|\s+(?:for|because|since|recently))",
    re.I,
)
_UNIT_GENERIC = frozenset({"one", "ones", "thing", "things", "time", "times", "day", "days", "week", "weeks", "month", "months", "year", "years", "minute", "minutes", "hour", "hours", "lot", "few", "couple"})


@dataclass(frozen=True)
class ExtractedSlot:
    slot_key: str
    slot_value: str
    numeric_value: float | None = None


def _parse_num(raw: str | None) -> float | None:
    if not raw:
        return None
    cleaned = raw.strip().casefold().replace(",", "")
    if cleaned in _NUM_WORDS:
        return float(_NUM_WORDS[cleaned])
    try:
        return float(cleaned)
    except ValueError:
        return None


def resolve_relative_occurrence(text: str, mention_ts: float | None) -> float | None:
    """Resolve relative event expressions (e.g. 'today', 'yesterday', '3 weeks ago') against ``mention_ts``."""
    if not text or mention_ts is None:
        return None
    ts = float(mention_ts)
    if _REL_YESTERDAY.search(text):
        return ts - 1.0 * _DAY
    if m := _REL_DAYS_AGO.search(text):
        n = _parse_num(m.group(1) or m.group(2))
        if n is not None and 0 < n <= 3650:
            return ts - n * _DAY
    if m := _REL_WEEKS_AGO.search(text):
        n = _parse_num(m.group(1) or m.group(2))
        if n is not None and 0 < n <= 520:
            return ts - n * 7.0 * _DAY
    if m := _REL_MONTHS_AGO.search(text):
        n = _parse_num(m.group(1) or m.group(2))
        if n is not None and 0 < n <= 240:
            return ts - n * 30.0 * _DAY
    if m := _REL_YEARS_AGO.search(text):
        n = _parse_num(m.group(1) or m.group(2))
        if n is not None and 0 < n <= 100:
            return ts - n * 365.0 * _DAY
    if _REL_LAST_WEEK.search(text):
        return ts - 7.0 * _DAY
    if _REL_LAST_MONTH.search(text):
        return ts - 30.0 * _DAY
    if _REL_TODAY.search(text):
        return ts
    return None


def _canonical_noun_head(phrase: str, full_text: str) -> str | None:
    """Produce a stable canonical slot suffix from a counted noun phrase (handling 'ones' via context)."""
    words = [w.casefold() for w in re.findall(r"[A-Za-z0-9]+", phrase)]
    useful = [w for w in words if w not in _UNIT_GENERIC and len(w) >= 2]
    if not useful:
        # Resolve anaphoric 'different ones' (e.g. 'Korean restaurants ... tried four different ones')
        ctx_stems = sorted(stem_terms(full_text))
        # Drop pure verb/filler stems to keep distinctive domain nouns
        ctx_useful = [s for s in ctx_stems if len(s) >= 4 and s not in {"tri", "differ", "recent", "actual", "think", "realli", "great", "good"}]
        if len(ctx_useful) >= 2:
            return "_".join(ctx_useful[:3])
        return None
    stems = sorted(stem_terms(" ".join(useful)))
    return "_".join(stems[:2]) if stems else None


def extract_state_slot(text: str, speaker: str | None = None) -> ExtractedSlot | None:
    """Extract canonical ``(slot_key, slot_value)`` for conversational state updates."""
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    subj = (speaker or "user").strip().casefold()

    for kind, pattern in _SLOT_PATTERNS:
        m = pattern.search(cleaned)
        if not m:
            continue
        if kind == "favorite":
            cat = m.group(1).strip().casefold()
            val = m.group(2).strip().casefold()
            return ExtractedSlot(f"{subj}:favorite:{cat}", val)
        if kind == "schedule":
            item = "_".join(sorted(stem_terms(m.group(1)))[:2]) or "event"
            val = m.group(2).strip().casefold()
            return ExtractedSlot(f"{subj}:schedule:{item}", val)
        if kind == "weight_lost":
            num = _parse_num(m.group(1))
            unit = m.group(2).casefold()
            return ExtractedSlot(f"{subj}:weight_lost", f"{num or m.group(1)} {unit}", numeric_value=num)
        if kind == "coffee_limit":
            num = _parse_num(m.group(1))
            return ExtractedSlot(f"{subj}:coffee_limit", str(num or m.group(1)), numeric_value=num)
        val = m.group(1).strip().casefold()
        return ExtractedSlot(f"{subj}:{kind}", val, numeric_value=_parse_num(val))

    # Countable personal progress/tally statements ("watched 4 MCU films", "tried four different ones")
    if m := _COUNT_STATE_RE.search(cleaned):
        num = _parse_num(m.group(1))
        noun = _canonical_noun_head(m.group(2), cleaned)
        if num is not None and noun:
            return ExtractedSlot(f"{subj}:count:{noun}", str(int(num) if num.is_integer() else num), numeric_value=num)

    # Explicit switch / upgrade statements ("switched to dark roast")
    if m := _SWITCH_RE.search(cleaned):
        val = m.group(1).strip().casefold()
        ctx = sorted(s for s in stem_terms(cleaned) if s not in stem_terms(val) and len(s) >= 4)[:2]
        if ctx:
            return ExtractedSlot(f"{subj}:switch:{'_'.join(ctx)}", val)

    return None


class SlotSupersedeTracker:
    """Tracks canonical state slots in SQLite/PostgreSQL and atomically supersedes stale values."""

    def __init__(self, store):
        self.store = store
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.store.conn.execute(
            """CREATE TABLE IF NOT EXISTS fact_slot_supersedes (
                fact_id TEXT PRIMARY KEY REFERENCES facts(id) ON DELETE CASCADE,
                user_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                speaker TEXT NOT NULL,
                slot_key TEXT NOT NULL,
                slot_value TEXT NOT NULL,
                numeric_value DOUBLE PRECISION,
                event_ts DOUBLE PRECISION NOT NULL,
                superseded_by TEXT,
                status TEXT NOT NULL DEFAULT 'active'
            )"""
        )
        self.store.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fss_scope_slot ON fact_slot_supersedes(user_id, agent_id, slot_key, status)"
        )
        self.store.conn.commit()

    def register_and_supersede(
        self,
        *,
        fact_id: str,
        text: str,
        user_id: str = "default",
        agent_id: str = "default",
        speaker: str | None = None,
        event_ts: float | None = None,
    ) -> dict | None:
        """Extract slot for ``fact_id``; if an older active fact shares ``slot_key`` with a new value, supersede it."""
        slot = extract_state_slot(text, speaker=speaker)
        if slot is None:
            return None
        ts = time.time() if event_ts is None else float(event_ts)
        sp = (speaker or "user").strip().casefold()

        # Find earlier active facts in the same slot
        existing = self.store.conn.execute(
            """SELECT fact_id, slot_value, numeric_value, event_ts
               FROM fact_slot_supersedes
               WHERE user_id=? AND agent_id=? AND slot_key=? AND status='active' AND fact_id<>?
               ORDER BY event_ts ASC""",
            (user_id, agent_id, slot.slot_key, fact_id),
        ).fetchall()

        superseded_ids: list[str] = []
        for row in existing:
            old_fid = row["fact_id"]
            old_ts = float(row["event_ts"])
            old_val = row["slot_value"]
            if old_ts <= ts and old_val != slot.slot_value:
                self.store.conn.execute(
                    "UPDATE fact_slot_supersedes SET status='superseded', superseded_by=? WHERE fact_id=?",
                    (fact_id, old_fid),
                )
                # Mark older fact superseded in the core unified facts table as well
                self.store.conn.execute(
                    "UPDATE facts SET status='superseded', valid_until=? WHERE id=? AND status='active'",
                    (ts, old_fid),
                )
                superseded_ids.append(old_fid)

        self.store.conn.execute(
            """INSERT INTO fact_slot_supersedes(
                   fact_id, user_id, agent_id, speaker, slot_key, slot_value, numeric_value, event_ts, superseded_by, status
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, 'active')
               ON CONFLICT (fact_id) DO UPDATE SET
                   slot_key=excluded.slot_key,
                   slot_value=excluded.slot_value,
                   numeric_value=excluded.numeric_value,
                   event_ts=excluded.event_ts,
                   status='active',
                   superseded_by=NULL""",
            (fact_id, user_id, agent_id, sp, slot.slot_key, slot.slot_value, slot.numeric_value, ts),
        )
        return {
            "slot_key": slot.slot_key,
            "slot_value": slot.slot_value,
            "numeric_value": slot.numeric_value,
            "superseded_ids": superseded_ids,
        }

    def superseded_fact_ids(self, *, user_id: str = "default", agent_id: str = "default") -> set[str]:
        rows = self.store.conn.execute(
            "SELECT fact_id FROM fact_slot_supersedes WHERE user_id=? AND agent_id=? AND status='superseded'",
            (user_id, agent_id),
        ).fetchall()
        return {row["fact_id"] for row in rows}

    def superseding_fact_ids(self, *, user_id: str = "default", agent_id: str = "default") -> set[str]:
        rows = self.store.conn.execute(
            """SELECT DISTINCT superseded_by FROM fact_slot_supersedes
               WHERE user_id=? AND agent_id=? AND superseded_by IS NOT NULL""",
            (user_id, agent_id),
        ).fetchall()
        return {row["superseded_by"] for row in rows if row["superseded_by"]}

    def chain_for_facts(self, fact_ids: list[str], *, user_id: str = "default", agent_id: str = "default") -> dict[str, list[dict]]:
        """Return ordered slot-revision chains ``slot_key -> [oldest..newest]`` touching any of ``fact_ids``."""
        ids = list(dict.fromkeys(fact_ids))
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        slot_rows = self.store.conn.execute(
            f"""SELECT DISTINCT slot_key FROM fact_slot_supersedes
                WHERE user_id=? AND agent_id=? AND fact_id IN ({placeholders})""",
            (user_id, agent_id, *ids),
        ).fetchall()
        chains: dict[str, list[dict]] = {}
        for srow in slot_rows:
            skey = srow["slot_key"]
            members = self.store.conn.execute(
                """SELECT s.*, f.plain, f.event_time_start, f.source_turn_ids
                   FROM fact_slot_supersedes s
                   JOIN facts f ON f.id = s.fact_id
                   WHERE s.user_id=? AND s.agent_id=? AND s.slot_key=?
                   ORDER BY s.event_ts ASC, s.fact_id ASC""",
                (user_id, agent_id, skey),
            ).fetchall()
            if len(members) >= 2:
                chains[skey] = [dict(m) for m in members]
        return chains
