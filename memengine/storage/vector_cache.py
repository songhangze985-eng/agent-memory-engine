"""Bounded per-owner float32 cache for immutable vector versions."""
from collections import OrderedDict

import numpy as np


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
            if previous is not None and previous[0] == ids:
                if key in changes:
                    for i, fact_id in enumerate(ids):
                        if fact_id in changes[key]:
                            previous[1][i] = np.frombuffer(self.entries[fact_id][1], dtype="<f4")
                result.append((key, ids, previous[1]))
                continue
            matrix = np.empty((len(ids), dim), dtype="<f4")
            for i, fact_id in enumerate(ids):
                matrix[i] = np.frombuffer(self.entries[fact_id][1], dtype="<f4")
            self.groups[key] = (ids, matrix)
            result.append((key, ids, matrix))
        return result
