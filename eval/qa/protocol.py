"""Frozen evaluation protocol loader, deterministic dev/test splitting, provenance validator & multi-seed aggregator."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Callable, TypeVar

from agentmem_core.core.profiles import parse_simple_yaml

T = TypeVar("T")

DEFAULT_PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "protocol.yaml"

REQUIRED_PROVENANCE_FIELDS = (
    "engine",
    "extractor_model",
    "reader_model",
    "judge",
    "judge_prompt_version",
    "embedder",
    "reranker",
    "top_k",
    "token_budget",
    "seed",
    "split",
    "protocol_version",
    "protocol_sha256",
    "dataset",
    "dataset_sha256",
    "commit_hash",
    "timestamp_utc",
)


def load_protocol(path: str | Path | None = None) -> tuple[dict[str, Any], str]:
    """Load ``eval/protocol.yaml`` and return ``(parsed_dict, sha256_hex)``."""
    target = Path(path) if path is not None else DEFAULT_PROTOCOL_PATH
    raw_bytes = target.read_bytes()
    sha = hashlib.sha256(raw_bytes).hexdigest()
    parsed = parse_simple_yaml(raw_bytes.decode("utf-8"))
    return parsed, sha


def partition_by_split(
    rows: list[T],
    key_fn: Callable[[T], str],
    *,
    split: str = "full",
    dev_ratio: float = 0.25,
) -> list[T]:
    """Deterministically partition dataset rows into disjoint ``dev`` and ``test`` splits (or ``full``).

    Uses SHA-256 of each item's stable identifier so ``dev ∩ test = ∅`` regardless of ordering or seed.
    """
    norm = str(split or "full").strip().casefold()
    if norm == "full":
        return list(rows)
    if norm not in ("dev", "test"):
        raise ValueError(f"split must be 'dev', 'test', or 'full'; got {split!r}")
    cutoff = max(0.01, min(0.99, float(dev_ratio)))
    out: list[T] = []
    for row in rows:
        digest = hashlib.sha256(f"agentmem-split-v1:{key_fn(row)}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
        is_dev = bucket < cutoff
        if (norm == "dev" and is_dev) or (norm == "test" and not is_dev):
            out.append(row)
    return out


def validate_provenance(prov: dict[str, Any]) -> None:
    """Enforce G5 invariant: raise ValueError if any mandatory protocol metadata field is missing."""
    missing = [k for k in REQUIRED_PROVENANCE_FIELDS if k not in prov or prov[k] in (None, "")]
    if missing:
        raise ValueError(f"G5 provenance violation: missing required fields: {', '.join(missing)}")


def summarize_seed_interval(values: list[float]) -> dict[str, float]:
    """Return mean, min, max, sample std, and half-width 95% normal/t interval across seeds."""
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "ci95_half": 0.0}
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return {"mean": round(mean, 6), "min": round(mean, 6), "max": round(mean, 6), "std": 0.0, "ci95_half": 0.0}
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(max(0.0, var))
    # Conservative t-multiplier for small seed counts (df=2 -> 4.303, df=3 -> 3.182, df>=4 -> 2.776/1.96)
    t_mult = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 1.96)
    ci_half = t_mult * std / math.sqrt(n)
    return {
        "mean": round(mean, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "std": round(std, 6),
        "ci95_half": round(ci_half, 6),
    }


def aggregate_multi_seed_reports(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate multiple per-seed QA reports for one (benchmark, arm) pair with intervals."""
    if not seed_reports:
        raise ValueError("seed_reports must be non-empty")
    first = seed_reports[0]
    seeds = [int(r["provenance"]["seed"]) for r in seed_reports]
    overall_qa = [float(r["summary"]["overall"]["qa_accuracy"]) for r in seed_reports]
    overall_r10 = [float(r["summary"]["overall"]["turn_r10"]) for r in seed_reports if r["summary"]["overall"].get("turn_r10") is not None]
    overall_mrr = [float(r["summary"]["overall"]["turn_mrr10"]) for r in seed_reports if r["summary"]["overall"].get("turn_mrr10") is not None]
    overall_toks = [float(r["summary"]["overall"]["avg_injected_tokens"]) for r in seed_reports]
    overall_p95 = [float(r["summary"]["overall"]["p95_ms"]) for r in seed_reports]

    cats = list(first["summary"]["by_category"].keys())
    by_cat: dict[str, dict[str, Any]] = {}
    for cat in cats:
        cat_vals = [
            float(r["summary"]["by_category"][cat]["qa_accuracy"])
            for r in seed_reports
            if cat in r["summary"]["by_category"]
        ]
        by_cat[cat] = {
            "n": first["summary"]["by_category"][cat]["n"],
            "qa_accuracy": summarize_seed_interval(cat_vals),
        }

    return {
        "benchmark": first["benchmark"],
        "engine": first["provenance"]["engine"],
        "seeds": seeds,
        "provenance": {**first["provenance"], "seeds": seeds},
        "overall": {
            "n": first["summary"]["overall"]["n"],
            "qa_accuracy": summarize_seed_interval(overall_qa),
            "turn_r10": summarize_seed_interval(overall_r10) if overall_r10 else None,
            "turn_mrr10": summarize_seed_interval(overall_mrr) if overall_mrr else None,
            "avg_injected_tokens": summarize_seed_interval(overall_toks),
            "p95_ms": summarize_seed_interval(overall_p95),
        },
        "by_category": by_cat,
        "representative_attribution": first["summary"]["attribution"],
        "per_seed_overall_qa": {str(s): round(v, 6) for s, v in zip(seeds, overall_qa, strict=True)},
    }
