"""Download and stratified-sample the official MemData category files."""

from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path

from eval.membench.catalog import CATEGORIES, RAW_BASE
from eval.membench.items import normalize_trajectory


def _iter_scenarios(payload):
    if isinstance(payload, list):
        yield "all", payload
        return
    if not isinstance(payload, dict):
        return
    for key, value in payload.items():
        if isinstance(value, list):
            yield str(key), value
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, list):
                    yield f"{key}/{sub_key}", sub_value


def _quota(total: int, slots: int) -> list[int]:
    if slots <= 0:
        return []
    base, extra = divmod(total, slots)
    return [base + (1 if index < extra else 0) for index in range(slots)]


def ensure_file(relative: str, data_dir: Path) -> Path:
    dest = data_dir / relative
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{RAW_BASE}/{relative}"
    request = urllib.request.Request(url, headers={"User-Agent": "memengine-membench"})
    with urllib.request.urlopen(request, timeout=180) as response:
        dest.write_bytes(response.read())
    return dest


def sample_category(category: str, files: list[str], per_category: int, rng: random.Random, data_dir: Path) -> list[dict]:
    quotas = _quota(per_category, len(files))
    chosen = []
    for relative, quota in zip(files, quotas):
        if quota <= 0:
            continue
        path = ensure_file(relative, data_dir)
        payload = json.loads(path.read_text(encoding="utf-8"))
        scenarios = list(_iter_scenarios(payload))
        if not scenarios:
            continue
        sub_quotas = _quota(quota, len(scenarios))
        subtype = Path(relative).stem
        for (scenario, rows), sub_quota in zip(scenarios, sub_quotas):
            if sub_quota <= 0 or not rows:
                continue
            pick = rows if len(rows) <= sub_quota else rng.sample(rows, sub_quota)
            for raw in pick:
                if not isinstance(raw, dict) or "QA" not in raw:
                    continue
                chosen.append(
                    normalize_trajectory(raw, category=category, subtype=subtype, scenario=scenario)
                )
        del payload
    return chosen


def load_sample(per_category: int, seed: int, data_dir: Path) -> list[dict]:
    rng = random.Random(seed)
    items = []
    for category, files in CATEGORIES.items():
        items.extend(sample_category(category, files, per_category, rng, data_dir))
    return items
