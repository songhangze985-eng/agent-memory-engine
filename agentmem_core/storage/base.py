"""Storage boundaries shared by the ledger and retrieval operators."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import ContextManager, Protocol

DEFAULT_SCOPE = ("default", "default", "default")
SCOPES = frozenset({"world", "experience", "opinion", "observation", "procedural"})
IDS = "SELECT value FROM json_each(?)"


def scope_where(scope, alias=""):
    where = f"{alias}user_id = ? AND {alias}agent_id = ?"
    return (where, scope[:2]) if scope[2] is None else (where + f" AND {alias}session_id = ?", scope)


@dataclass(frozen=True)
class RecallFilter:
    memory_types: tuple[str, ...] | None = None
    time_start: float | None = None
    time_end: float | None = None
    session_provenance: bool = False
    scopes: tuple[str, ...] | None = None
    as_of: float | None = None
    historical_window: tuple[float | None, float | None] | None = None

    def sql(self, where, args):
        for column, values in (("memory_type", self.memory_types), ("epistemic_scope", self.scopes)):
            if values is not None:
                where += f" AND f.{column} IN ({IDS})"
                args += (json.dumps(values),)
        for op, value in ((">=", self.time_start), ("<", self.time_end)):
            if value is not None:
                where += f" AND f.event_time_start{op}?"
                args += (value,)
        if self.as_of is not None:
            where += " AND (f.valid_from IS NULL OR f.valid_from<=?) AND (f.expires_at IS NULL OR f.expires_at>?)"
            args += (self.as_of, self.as_of)
        if self.historical_window is not None:
            lower, upper = self.historical_window
            if lower is not None:
                where += " AND (f.expires_at IS NULL OR f.expires_at>?)"
                args += (lower,)
            if upper is not None:
                where += " AND (f.expires_at IS NULL OR f.valid_from<?)"
                args += (upper,)
        if self.session_provenance:
            where += """ AND NOT EXISTS (SELECT 1 FROM json_each(f.source_turn_ids) j
                LEFT JOIN raw_turns t ON t.turn_id=j.value AND t.user_id=f.user_id AND t.agent_id=f.agent_id
                AND t.session_id=f.session_id WHERE t.turn_id IS NULL)"""
        return where, args

    def matches(self, fact):
        stamp = fact["event_time_start"]
        start, end = fact.get("valid_from"), fact.get("expires_at")
        lower, upper = self.historical_window or (None, None)
        return ((self.memory_types is None or fact["memory_type"] in self.memory_types)
                and (self.scopes is None or fact["epistemic_scope"] in self.scopes)
                and (self.time_start is None or stamp >= self.time_start)
                and (self.time_end is None or stamp < self.time_end)
                and (self.as_of is None or ((start is None or start <= self.as_of) and (end is None or end > self.as_of)))
                and (end is None or lower is None or end > lower)
                and (end is None or upper is None or start < upper))


class MemoryStoreProtocol(Protocol):
    def transaction(self) -> ContextManager: ...
    def read_snapshot(self) -> ContextManager: ...
    def append_turn(self, text: str, timestamp=None, **options) -> int: ...
    def insert_fact(self, **options) -> str: ...
    def supersede_fact(self, old_fact_id: str, **options) -> str | None: ...
    def delete_fact(self, fact_id: str, **options) -> bool: ...
    def facts_by_ids(self, fact_ids, **options) -> list[dict]: ...
    def search_lexical(self, query, **options) -> list[dict]: ...
    def search_exact_identifiers(self, identifiers, **options) -> list[dict]: ...
    def search_entity_inverted(self, entities, **options) -> list[dict]: ...
    def search_temporal_range(self, **options) -> list[dict]: ...
    def search_vector_topk(self, query, **options) -> list[dict]: ...
    def fetch_raw_window(self, turn_ids, **options) -> list[dict]: ...
    def inactive_temporary_sources(self, turn_ids, *, scope, as_of, include_superseded=False) -> set[int]: ...
    def increment_hit_counts(self, fact_ids, accessed_at=None): ...
    def archive_cold(self, scope, **options) -> list[str]: ...
    def health_check(self) -> dict: ...
    def close(self): ...
