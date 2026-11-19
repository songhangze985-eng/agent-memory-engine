"""Dual timeline with bounded gap, duration and event count per node."""

from __future__ import annotations


def record_timeline_buffered(layer: list, payload: dict, now: float, delta_t: float,
                             enabled: bool = True, *, max_node_duration: float = 300.0,
                             max_events_per_node: int = 20) -> None:
    clean = {key: value for key, value in payload.items() if key not in {"t_start", "t_end", "event_count"}}
    last = layer[-1] if layer else None
    if (enabled and last and 0 <= now - last["t_end"] <= delta_t
            and now - last["t_start"] <= max_node_duration
            and last.get("event_count", 1) < max_events_per_node
            and not clean.get("origin_shifted")
            and clean.get("topic") == last.get("topic")):
        last.update(clean, t_end=now, event_count=last.get("event_count", 1) + 1)
        return
    layer.append({"t_start": now, "t_end": now, "event_count": 1, **clean})


class TimelineBook:
    def __init__(self, delta_t: float = 10.0, enabled: bool = True, *,
                 max_node_duration: float = 300.0, max_events_per_node: int = 20):
        self.delta_t = float(delta_t)
        self.enabled = bool(enabled)
        self.max_node_duration = float(max_node_duration)
        self.max_events_per_node = int(max_events_per_node)
        if self.max_node_duration < 0 or self.max_events_per_node < 1:
            raise ValueError("invalid timeline limits")
        self.layer1: list[dict] = []
        self.layer2: list[dict] = []
        self.events = 0

    def record(self, layer_no: int, payload: dict, now: float) -> None:
        self.events += 1
        layer = self.layer1 if layer_no == 1 else self.layer2
        self.buffer(layer, payload, now)

    def buffer(self, layer: list, payload: dict, now: float) -> None:
        record_timeline_buffered(layer, payload, now, self.delta_t, self.enabled,
                                 max_node_duration=self.max_node_duration,
                                 max_events_per_node=self.max_events_per_node)

    @property
    def nodes(self) -> int:
        return len(self.layer1) + len(self.layer2)

    def reduction(self) -> float | None:
        if self.events <= 0:
            return None
        return 1.0 - (self.nodes / self.events)
