"""Step 2, layer half: gap-buffered dual timeline.

Merge rule is the spec pseudocode: if ``now - last.t_end <= delta_t``,
extend ``t_end`` and ``dict.update`` the payload (latest snapshot wins).
Because ``t_end`` moves forward, a session that never goes idle stays one node.
"""

from __future__ import annotations


def record_timeline_buffered(layer: list, payload: dict, now: float, delta_t: float, enabled: bool = True) -> None:
    clean = {k: v for k, v in payload.items() if k not in ("t_start", "t_end")}
    if enabled and layer and (now - layer[-1]["t_end"] <= delta_t):
        layer[-1]["t_end"] = now
        layer[-1].update(clean)
        return
    layer.append({"t_start": now, "t_end": now, **clean})


class TimelineBook:
    def __init__(self, delta_t: float = 10.0, enabled: bool = True):
        self.delta_t = float(delta_t)
        self.enabled = bool(enabled)
        self.layer1: list[dict] = []
        self.layer2: list[dict] = []
        self.events = 0

    def record(self, layer_no: int, payload: dict, now: float) -> None:
        self.events += 1
        layer = self.layer1 if layer_no == 1 else self.layer2
        record_timeline_buffered(layer, payload, now, self.delta_t, self.enabled)

    @property
    def nodes(self) -> int:
        return len(self.layer1) + len(self.layer2)

    def reduction(self) -> float | None:
        if self.events <= 0:
            return None
        return 1.0 - (self.nodes / self.events)
