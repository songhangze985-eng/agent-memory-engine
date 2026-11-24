"""Bounded per-owner float32 and inverted entity-key caches."""
from collections import OrderedDict

import numpy as np

from agentmem_core.router import _entity_forms, _token_forms, entity_names, latin_tokens


class IncrementalVectorCache:
    def __init__(self, owners=16):
        self.owners = owners
        self.views = OrderedDict()

    def discard(self, owner, ids):
        if owner in self.views:
            view = self.views[owner]
            for fact_id in ids:
                view.entries.pop(fact_id, None)
            view.groups.clear()

    def view(self, owner):
        if owner not in self.views:
            self.views[owner] = VectorView()
            if len(self.views) > self.owners:
                self.views.popitem(last=False)
        self.views.move_to_end(owner)
        return self.views[owner]


class VectorView:
    def __init__(self):
        self.entries = {}
        self.groups = {}
        self.version = None

    def missing(self, key, generation, external_version):
        if external_version != self.version:
            self.entries.clear()
            self.groups.clear()
            self.version = external_version
        return key not in self.entries or self.entries[key][0] != generation

    def groups_for(self, rows):
        partitions, changes = {}, {}
        for fact_id, model, dim, generation, blob in rows:
            if blob is not None:
                self.entries[fact_id] = (generation, blob)
                changes.setdefault((model, dim), set()).add(fact_id)
            partitions.setdefault((model, dim), []).append(fact_id)
        result = []
        for key, ids in partitions.items():
            previous = self.groups.get(key)
            dim = key[1]
            count = len(ids)
            if previous is not None:
                prev_ids, backing, prev_len = previous
                if prev_ids == ids:
                    if key in changes:
                        for i, fact_id in enumerate(ids):
                            if fact_id in changes[key]:
                                backing[i] = np.frombuffer(self.entries[fact_id][1], dtype="<f4")
                    result.append((key, ids, backing[:count]))
                    continue
                # Fast path: monotonic append (e.g. steady ingestion without deleting middle rows)
                if count > prev_len and ids[:prev_len] == prev_ids and count <= backing.shape[0]:
                    if key in changes:
                        for i, fact_id in enumerate(prev_ids):
                            if fact_id in changes[key]:
                                backing[i] = np.frombuffer(self.entries[fact_id][1], dtype="<f4")
                    for i in range(prev_len, count):
                        backing[i] = np.frombuffer(self.entries[ids[i]][1], dtype="<f4")
                    self.groups[key] = (ids, backing, count)
                    result.append((key, ids, backing[:count]))
                    continue
            capacity = max(count, 64) if count <= 64 else max(count, int(count * 1.5))
            backing = np.empty((capacity, dim), dtype="<f4")
            for i, fact_id in enumerate(ids):
                backing[i] = np.frombuffer(self.entries[fact_id][1], dtype="<f4")
            self.groups[key] = (ids, backing, count)
            result.append((key, ids, backing[:count]))
        return result


class EntityKeyIndex:
    """Bounded per-scope inverted token/bigram index for O(query_tokens) entity matching.

    Mathematically guaranteed to return a superset of ``router.match_entities(query, all_keys)``:
    - Latin entity keys are indexed under all plural forms of their first token group.
      Since ``_name_matches`` requires every token group to intersect ``query_forms``,
      the first group MUST intersect ``query_forms``.
    - CJK / mixed entity keys (len >= 2) are indexed under their first 2-char prefix ``name[:2]``.
      Since ``_name_matches`` checks ``name in query``, ``name[:2]`` MUST occur as a 2-char
      substring of ``query``.
    """

    def __init__(self, max_scopes=32):
        self.max_scopes = max_scopes
        self._entries = OrderedDict()

    def clear(self):
        self._entries.clear()

    def get(self, scope_key, stamp):
        entry = self._entries.get(scope_key)
        if entry is None or entry[0] != stamp:
            return None
        self._entries.move_to_end(scope_key)
        return entry[1]

    def put(self, scope_key, stamp, raw_keys):
        latin_map: dict[str, list[str]] = {}
        cjk_map: dict[str, list[str]] = {}
        for key in raw_keys:
            name = entity_names(key)
            if not name:
                continue
            forms = _entity_forms(name)
            if forms is None:
                if len(name) >= 2:
                    cjk_map.setdefault(name[:2], []).append(key)
            elif forms:
                for token in forms[0]:
                    latin_map.setdefault(token, []).append(key)
        payload = (latin_map, cjk_map)
        self._entries[scope_key] = (stamp, payload)
        self._entries.move_to_end(scope_key)
        if len(self._entries) > self.max_scopes:
            self._entries.popitem(last=False)
        return payload

    @staticmethod
    def candidates(payload, query: str) -> list[str]:
        latin_map, cjk_map = payload
        hits: set[str] = set()
        if latin_map:
            query_forms = {form for token in latin_tokens(query) for form in _token_forms(token)}
            for form in query_forms:
                bucket = latin_map.get(form)
                if bucket:
                    hits.update(bucket)
        if cjk_map and len(query) >= 2:
            for index in range(len(query) - 1):
                bucket = cjk_map.get(query[index : index + 2])
                if bucket:
                    hits.update(bucket)
        return sorted(hits)
