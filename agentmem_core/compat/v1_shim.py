"""Deprecated /v1/* and /v2/vfs/* compatibility shims forwarding strictly to UnifiedMemoryEngine."""
from __future__ import annotations

import warnings
from typing import Any

from agentmem_core.vfs.models import normalize_node_path

V1_SUNSET_DATE = "2027-06-01"
V1_DEPRECATION_MSG = "The /v1/* HTTP routes are deprecated; migrate to unified /v2/memory/* routes."
V2_VFS_DEPRECATION_MSG = "The /v2/vfs/* HTTP routes are deprecated; migrate to unified /v2/memory/* routes."


def attach_v1_deprecation_middleware(app) -> None:
    """Mark all ``/v1/*`` (and unified-backed ``/v2/vfs/*``) responses with RFC 8594 headers."""

    @app.middleware("http")
    async def _v1_deprecation_shim(request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/v1/"):
            warnings.warn(V1_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
            response.headers["Deprecation"] = "true"
            response.headers["Sunset"] = V1_SUNSET_DATE
            response.headers["X-Deprecated-API"] = V1_DEPRECATION_MSG
        elif path.startswith("/v2/vfs/") and getattr(app.state, "vfs", None) is None:
            response.headers["Deprecation"] = "true"
            response.headers["Sunset"] = V1_SUNSET_DATE
            response.headers["X-Deprecated-API"] = V2_VFS_DEPRECATION_MSG
        return response


class UnifiedV1Shim:
    """Forwards legacy ``/v1/*`` calls onto the single ``UnifiedMemoryEngine`` core."""

    def __init__(self, unified_engine):
        self.unified = unified_engine

    def retain_turn(self, text: str, **kwargs: Any) -> dict:
        return self.unified.ingest_turn(text, **kwargs)

    def recall(self, query: str, **kwargs: Any) -> dict:
        return self.unified.recall(query, **kwargs)

    def build_context(self, query: str, **kwargs: Any) -> dict:
        return self.unified.build_context(query, **kwargs)

    def answer(self, query: str, **kwargs: Any) -> dict:
        return self.unified.answer(query, **kwargs)

    def reflect(self, query: str, **kwargs: Any) -> dict:
        return self.unified.reflect(query, **kwargs)

    def drilldown(self, turn_ids: list[int], window: int = 1, **kwargs: Any) -> list[dict]:
        return self.unified.drilldown(turn_ids, window=window, **kwargs)

    def history(self, target: str, **kwargs: Any) -> list[dict]:
        return self.unified.history(target, **kwargs)

    def forget(self, target: str, **kwargs: Any) -> bool:
        return self.unified.forget(target, **kwargs)


class UnifiedVFSShim:
    """Forwards legacy ``/v2/vfs/*`` calls onto ``UnifiedMemoryEngine`` (zero separate VFS tables)."""

    def __init__(self, unified_engine):
        self.unified = unified_engine

    def health_check(self) -> dict:
        base = dict(self.unified.store.health_check())
        base["shim"] = "unified-vfs-view"
        return base

    def append_transcript(
        self,
        text: str,
        *,
        user_id: str = "default",
        agent_id: str = "default",
        session_id: str = "default",
        role: str = "user",
        timestamp: float | None = None,
        speaker: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        res = self.unified.ingest_turn(
            text,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            role=role,
            timestamp=timestamp,
            speaker=speaker,
            metadata=metadata,
        )
        return {
            "transcript_id": res["transcript_id"],
            "session_id": session_id,
            "timestamp": timestamp or self.unified._engine._now(),
            "turn_id": res["turn_id"],
        }

    def write(
        self,
        *,
        path: str,
        title: str,
        content: str,
        summary: str = "",
        layer: str = "L2",
        confidence_score: float = 0.7,
        verification_status: str = "unverified",
        salience_tag: str = "normal",
        always_active: bool = False,
        source_transcript_ids: list[str] = (),
        expected_version: int | None = None,
        user_id: str = "default",
        agent_id: str = "default",
    ) -> dict:
        _ = layer
        res = self.unified.ingest_rule(
            path,
            title=title,
            trigger_condition=f"When consulting {path}",
            action=content,
            verification_result="Recorded via /v2/vfs/write shim" if verification_status == "verified" else "",
            index_summary=summary,
            user_id=user_id,
            agent_id=agent_id,
            confidence_score=confidence_score,
            verification_status=verification_status,
            salience_tag=salience_tag,
            always_active=always_active,
            source_transcript_ids=source_transcript_ids,
            expected_version=expected_version,
        )
        return {"written": bool(res.get("accepted")), "path": res["path"], "version": res.get("version"), "fact_id": res.get("fact_id")}

    def write_rule(self, **kwargs: Any) -> dict:
        kwargs.pop("layer", None)
        res = self.unified.ingest_rule(**kwargs)
        return {
            **res,
            "written": bool(res.get("accepted")),
        }

    def read_note(self, path: str, *, user_id: str = "default", agent_id: str = "default") -> dict | None:
        return self.unified.read_note(path, user_id=user_id, agent_id=agent_id)

    def tree_outline(self, path: str = "/", *, user_id: str = "default", agent_id: str = "default") -> str:
        return self.unified.tree_outline(path, user_id=user_id, agent_id=agent_id)

    def search(self, query: str, *, user_id: str = "default", agent_id: str = "default", top_k: int = 8, reference_time=None, **_ignored) -> list[dict]:
        res = self.unified.recall(query, user_id=user_id, agent_id=agent_id, top_k=top_k, reference_time=reference_time)
        return res["ranked"]

    def build_context(self, query: str, *, user_id: str = "default", agent_id: str = "default", token_budget: int = 2048, disclosure: str = "progressive", **_ignored) -> dict:
        ctx = self.unified.build_context(
            query,
            user_id=user_id,
            agent_id=agent_id,
            token_budget=token_budget,
            disclosure=disclosure,
            profile="coding",
        )
        preloaded = self.unified.vfs.preload_always_active(user_id=user_id, agent_id=agent_id)
        return {
            **ctx,
            "preloaded": [{"path": r["vfs_path"], "title": r["title"], "content": r["plain"]} for r in preloaded],
        }

    def compact_directory(self, dir_path: str, *, user_id: str = "default", agent_id: str = "default") -> dict:
        return self.unified.compact_directory(dir_path, user_id=user_id, agent_id=agent_id)

    def rollback(self, *, path: str | None = None, snapshot_id: str | None = None, dream_batch_id: str | None = None, user_id: str = "default", agent_id: str = "default") -> dict:
        if dream_batch_id:
            res = self.unified.rollback(batch_id=dream_batch_id, user_id=user_id, agent_id=agent_id)
            return {"restored": res["count"], **res}
        if path:
            _, norm = normalize_node_path(path)
            return self.unified.rollback(path=norm, snapshot_id=snapshot_id, user_id=user_id, agent_id=agent_id)
        raise ValueError("rollback requires either dream_batch_id or path")
