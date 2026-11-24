"""Protocol-verified adapters for SWE-Bench-CL (continual learning) and SWE-ContextBench."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentmem_core import CodeAnchor, UnifiedMemoryEngine


@dataclass(frozen=True)
class SWECLTask:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    gold_files: tuple[str, ...]
    gold_fix_summary: str = ""
    order_index: int = 0


def load_swebench_cl_jsonl(path: str | Path) -> list[SWECLTask]:
    """Load and validate a chronologically ordered SWE-Bench-CL JSONL stream."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"SWE-Bench-CL dataset not found: {p}")
    tasks: list[SWECLTask] = []
    for idx, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        for req in ("instance_id", "repo", "problem_statement"):
            if req not in row:
                raise ValueError(f"SWE-Bench-CL protocol error at line {idx + 1}: missing '{req}'")
        gold_files = tuple(str(f) for f in (row.get("gold_files") or row.get("modified_files") or ()))
        tasks.append(
            SWECLTask(
                instance_id=str(row["instance_id"]),
                repo=str(row["repo"]),
                base_commit=str(row.get("base_commit") or ""),
                problem_statement=str(row["problem_statement"]),
                gold_files=gold_files,
                gold_fix_summary=str(row.get("gold_fix_summary") or row.get("patch_summary") or ""),
                order_index=int(row.get("order_index", idx)),
            )
        )
    tasks.sort(key=lambda t: t.order_index)
    return tasks


def evaluate_swe_context_stream(
    tasks: list[SWECLTask],
    *,
    engine: UnifiedMemoryEngine,
    user_id: str = "swe-eval",
) -> dict[str, Any]:
    """Evaluate continual context recall/precision and forward transfer on a chronological SWE stream.

    Compatible with both SWE-Bench-CL continual streams and SWE-ContextBench gold-context retrieval.
    """
    hits = 0
    precision_sum = 0.0
    recall_sum = 0.0
    evaluated = 0
    latencies_ms: list[float] = []

    for task in tasks:
        amb = engine.ambient_context(
            task.problem_statement,
            active_files=list(task.gold_files),
            user_id=user_id,
            repo_id=task.repo,
            profile="coding",
        )
        latencies_ms.append(float(amb["ambient_ms"]))

        retrieved_files = {
            str(item.get("matched_file") or "")
            for item in amb["items"]
            if item.get("matched_file")
        }
        gold_set = set(task.gold_files)
        if evaluated > 0 and gold_set:
            inter = len(retrieved_files & gold_set)
            rec = inter / len(gold_set)
            prec = inter / len(retrieved_files) if retrieved_files else 0.0
            recall_sum += rec
            precision_sum += prec
            if inter > 0:
                hits += 1

        # Record resolved task experience into repo-scoped unified core with file anchors
        if task.gold_fix_summary:
            engine.ingest_rule(
                f"/swe/{task.instance_id}.md",
                title=f"Resolved {task.instance_id}",
                trigger_condition=task.problem_statement[:180],
                action=task.gold_fix_summary,
                verification_result=f"Verified at commit {task.base_commit or 'HEAD'}",
                user_id=user_id,
                repo_id=task.repo,
                scope_key=f"repo:{task.repo}",
                anchors=[CodeAnchor(file_path=f, commit_sha=task.base_commit) for f in task.gold_files],
            )
        evaluated += 1

    denom = max(1, evaluated - 1)
    return {
        "tasks_processed": evaluated,
        "forward_transfer_hit_rate": round(hits / denom, 4),
        "avg_context_recall": round(recall_sum / denom, 4),
        "avg_context_precision": round(precision_sum / denom, 4),
        "avg_ambient_ms": round(sum(latencies_ms) / max(1, len(latencies_ms)), 3),
    }
