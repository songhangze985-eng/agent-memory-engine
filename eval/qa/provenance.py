"""Provenance metadata stamped automatically into every evaluation report."""
from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def file_sha256(path: Path | str | None) -> str | None:
    if path is None:
        return None
    target = Path(path)
    if not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dir_manifest_sha256(path: Path | str | None) -> str | None:
    if path is None:
        return None
    target = Path(path)
    if not target.is_dir():
        return None
    digest = hashlib.sha256()
    for child in sorted(p for p in target.rglob("*.json") if p.is_file()):
        digest.update(str(child.relative_to(target)).encode("utf-8"))
        digest.update(str(child.stat().st_size).encode("utf-8"))
    return digest.hexdigest()


def git_commit_hash(cwd: Path | None = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return f"{out}-dirty" if dirty else out
    except Exception:
        return "unknown"


def build_provenance(
    *,
    engine: str,
    embedder: str,
    reranker: str,
    llm: str,
    judge: str,
    protocol: str,
    granularity: str,
    seed: int,
    dataset: str,
    dataset_path: Path | str | None = None,
    dataset_sha256: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sha = dataset_sha256
    if sha is None and dataset_path is not None:
        p = Path(dataset_path)
        sha = file_sha256(p) if p.is_file() else dir_manifest_sha256(p)
    prov: dict[str, Any] = {
        "engine": engine,
        "embedder": embedder,
        "reranker": reranker,
        "llm": llm,
        "judge": judge,
        "protocol": protocol,
        "granularity": granularity,
        "commit_hash": git_commit_hash(),
        "seed": int(seed),
        "dataset": dataset,
        "dataset_sha256": sha or "synthetic/in-memory",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if extra:
        prov.update(extra)
    return prov


def render_provenance_markdown(prov: dict[str, Any]) -> str:
    lines = [
        "| Provenance Field | Value |",
        "| :--- | :--- |",
    ]
    for key in (
        "engine",
        "embedder",
        "reranker",
        "llm",
        "judge",
        "protocol",
        "granularity",
        "commit_hash",
        "seed",
        "dataset",
        "dataset_sha256",
        "timestamp_utc",
    ):
        if key in prov:
            lines.append(f"| `{key}` | `{prov[key]}` |")
    return "\n".join(lines)
