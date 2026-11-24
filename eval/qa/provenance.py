"""Provenance metadata stamped automatically and validated (G5) into every evaluation report."""
from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.qa.protocol import DEFAULT_PROTOCOL_PATH, load_protocol, validate_provenance


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
    extractor_model: str | None = None,
    reader_model: str | None = None,
    judge_prompt_version: str = "hindsight-aligned-v1",
    top_k: int = 10,
    token_budget: int = 2048,
    split: str = "full",
    protocol_path: Path | str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sha = dataset_sha256
    if sha is None and dataset_path is not None:
        p = Path(dataset_path)
        sha = file_sha256(p) if p.is_file() else dir_manifest_sha256(p)

    proto_ver = "1.0.0"
    proto_sha = "unfrozen"
    p_target = Path(protocol_path) if protocol_path is not None else DEFAULT_PROTOCOL_PATH
    if p_target.is_file():
        loaded, proto_sha = load_protocol(p_target)
        proto_ver = str(loaded.get("protocol_version") or "1.0.0")

    eff_reader = reader_model or llm
    eff_extractor = extractor_model or ("DeterministicRuleExtractor" if "MockLLMClient" in eff_reader else eff_reader)

    prov: dict[str, Any] = {
        "engine": engine,
        "extractor_model": eff_extractor,
        "reader_model": eff_reader,
        "llm": llm,
        "judge": judge,
        "judge_prompt_version": judge_prompt_version,
        "embedder": embedder,
        "reranker": reranker,
        "top_k": int(top_k),
        "token_budget": int(token_budget),
        "split": str(split),
        "protocol": protocol,
        "protocol_version": proto_ver,
        "protocol_sha256": proto_sha,
        "granularity": granularity,
        "commit_hash": git_commit_hash(),
        "seed": int(seed),
        "dataset": dataset,
        "dataset_sha256": sha or "synthetic/in-memory",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if extra:
        prov.update(extra)
    validate_provenance(prov)
    return prov


def render_provenance_markdown(prov: dict[str, Any]) -> str:
    lines = [
        "| Provenance Field | Value |",
        "| :--- | :--- |",
    ]
    for key in (
        "engine",
        "extractor_model",
        "reader_model",
        "judge",
        "judge_prompt_version",
        "embedder",
        "reranker",
        "top_k",
        "token_budget",
        "split",
        "protocol",
        "protocol_version",
        "protocol_sha256",
        "granularity",
        "commit_hash",
        "seed",
        "seeds",
        "dataset",
        "dataset_sha256",
        "timestamp_utc",
    ):
        if key in prov:
            lines.append(f"| `{key}` | `{prov[key]}` |")
    return "\n".join(lines)
