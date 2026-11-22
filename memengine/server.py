"""Optional FastAPI transport for Retain, Recall and Reflect."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from secrets import compare_digest
from threading import RLock
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from memengine.chrono import parse_session_clock
from memengine.engine import CognitiveMemoryEngine
from memengine.storage.base import RecallFilter
from memengine.tools import _timestamp
from memengine.vfs import OptimisticLockError, VFSMemoryEngine


class Owner(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    user_id: str = Field(default="default", min_length=1)
    agent_id: str = Field(default="default", min_length=1)
    session_id: str | None = None

    def scope(self):
        return self.user_id, self.agent_id, self.session_id

    def owner(self):
        return dict(user_id=self.user_id, agent_id=self.agent_id)

    def session_owner(self):
        return {**self.owner(), "session_id": self.session_id or "default"}


class Turn(BaseModel):
    model_config = Owner.model_config
    text: str = Field(min_length=1)
    role: str = "user"
    timestamp: float | None = None
    event_time: float | None = None
    mention_time: float | str | None = None
    speaker: str | None = Field(default=None, min_length=1, max_length=128)


class Buffer(Owner, Turn):
    pass


class Retain(Owner):
    text: str | None = None
    buffered_turn_id: int | None = Field(default=None, gt=0)
    role: str = "user"
    timestamp: float | None = None
    event_time: float | None = None
    mention_time: float | str | None = None
    speaker: str | None = Field(default=None, min_length=1, max_length=128)
    turns: list[Turn] | None = Field(default=None, max_length=1000)
    epistemic_scope: str | None = None
    memory_type: Literal["fact", "procedural"] = "fact"
    confidence: float = Field(default=1, ge=0, le=1)
    trigger: str = ""
    is_constraint: bool = False
    zkeys: list[str] = Field(default_factory=list)
    source_turn_ids: list[int] = Field(default_factory=list)
    ttl_seconds: float | None = Field(default=None, gt=0)
    expires_at: float | str | None = None
    valid_from: float | str | None = None


class Recall(Owner):
    query: str = Field(min_length=1)
    timestamp: float | None = None
    mode: Literal["single_query", "reconstructive"] = "single_query"
    scopes: list[str] | None = None
    time_start: float | str | None = None
    time_end: float | str | None = None
    include_cold: bool = False
    top_k: int | None = Field(default=None, ge=1, le=100)
    reference_time: float | str | None = None
    packing_mode: Literal["token_budget", "top_k"] | None = None
    token_budget: int = Field(default=2048, ge=0, le=32768)
    assemble: bool = False
    task_id: str | None = None
    include_short_term: bool = True


class ReasonRequest(Owner):
    query: str = Field(min_length=1)
    task_id: str | None = None
    reference_time: float | str | None = None
    token_budget: int = Field(default=2048, ge=0, le=32768)
    timestamp: float | None = None


class Maintenance(Owner):
    force: bool = False
    min_new_facts: int = Field(default=8, ge=0)
    archive: bool = True
    timestamp: float | None = None


class TaskStart(Owner):
    goal: str = Field(min_length=1)
    task_id: str | None = None
    ttl_sec: float | None = Field(default=86400, gt=0)


class TaskCheckpoint(Owner):
    state_dict: dict


class WorkItem(Owner):
    content: str = Field(min_length=1)
    item_type: Literal["tool_evidence", "temp_fact", "provisional_claim", "open_question"]
    source_refs: list[int | str | dict] = Field(default_factory=list, max_length=64)


class Verification(Owner):
    status: Literal["tool_verified", "user_confirmed", "rejected"]
    verifier_ref: int | str | dict | None = None


class Drilldown(Owner):
    turn_ids: list[int] = Field(max_length=100)
    window: int = Field(default=1, ge=0, le=20)


class Dispatch(Owner):
    name: str
    arguments: dict
    tool_call_id: str | None = None


class VFSTranscript(Owner):
    text: str = Field(min_length=1)
    role: str = "user"
    timestamp: float | None = None
    speaker: str | None = Field(default=None, min_length=1, max_length=128)
    metadata: dict = Field(default_factory=dict)


class VFSWrite(Owner):
    path: str = Field(min_length=1)
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    summary: str = ""
    layer: Literal["L1", "L2", "L3"] = "L2"
    confidence_score: float = Field(default=0.7, ge=0, le=1)
    verification_status: Literal["unverified", "verified"] = "unverified"
    salience_tag: Literal["normal", "preference", "taboo", "core_rule", "warning"] = "normal"
    always_active: bool = False
    source_transcript_ids: list[str] = Field(default_factory=list, max_length=128)
    expected_version: int | None = Field(default=None, ge=0)


class VFSWriteRule(Owner):
    path: str = Field(min_length=1)
    title: str = Field(min_length=1)
    trigger_condition: str = Field(min_length=1)
    action: str = Field(min_length=1)
    verification_result: str = ""
    notes: str = ""
    index_summary: str = ""
    layer: Literal["L1", "L2", "L3"] = "L2"
    confidence_score: float = Field(default=0.85, ge=0, le=1)
    verification_status: Literal["unverified", "verified"] = "verified"
    salience_tag: Literal["normal", "preference", "taboo", "core_rule", "warning"] = "core_rule"
    always_active: bool = False
    visible_in_context: bool = False
    reusable_pattern: bool = True
    future_pitfall_risk: bool = True
    source_transcript_ids: list[str] = Field(default_factory=list, max_length=128)
    expected_version: int | None = Field(default=None, ge=0)


class VFSPath(Owner):
    path: str = Field(default="/", min_length=1)
    layer: Literal["L1", "L2", "L3"] | None = None


class VFSSearch(Owner):
    query: str = Field(min_length=1)
    path_prefix: str | None = None
    verified_only: bool = False
    salience_tags: list[Literal["normal", "preference", "taboo", "core_rule", "warning"]] | None = None
    reference_time: float | str | None = None
    top_k: int = Field(default=8, ge=1, le=50)


class VFSContext(Owner):
    query: str = Field(min_length=1)
    token_budget: int = Field(default=2048, ge=64, le=32768)
    path_prefix: str | None = None
    verified_only: bool = False
    reference_time: float | str | None = None
    disclosure: Literal["progressive", "full"] = "progressive"
    include_tree_outline: bool = True
    include_l1_working: bool = True
    top_k: int = Field(default=8, ge=1, le=30)


class VFSCompact(Owner):
    dir_path: str = Field(min_length=1)


class VFSRollback(Owner):
    path: str | None = None
    snapshot_id: str | None = None
    dream_batch_id: str | None = None


def create_app(*, db_url=None, api_key=None, engine=None, vfs_db_url=None, vfs_engine=None):
    key = os.getenv("MEMENGINE_API_KEY") if api_key is None else api_key
    lock = RLock()

    def _get_vfs(app) -> VFSMemoryEngine:
        if app.state.vfs is None:
            target = vfs_db_url or os.getenv("MEMENGINE_VFS_DB_URL") or db_url
            app.state.vfs = VFSMemoryEngine(db_url=target)
            app.state.vfs_owned = True
        return app.state.vfs

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = engine or CognitiveMemoryEngine(db_url=db_url)
        app.state.vfs = vfs_engine
        app.state.vfs_owned = False
        try:
            yield
        finally:
            if engine is None:
                app.state.engine.store.close()
            if app.state.vfs is not None and app.state.vfs_owned:
                app.state.vfs.close()

    def authenticate(authorization: str | None = Header(default=None)):
        if key and not compare_digest((authorization or "").encode(), f"Bearer {key}".encode()):
            raise HTTPException(401, "invalid bearer credentials", headers={"WWW-Authenticate": "Bearer"})

    app = FastAPI(title="Agent Memory Engine", lifespan=lifespan, dependencies=[Depends(authenticate)])

    @app.exception_handler(ValueError)
    async def bad_input(request, error):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": str(error)}, status_code=422)

    @app.exception_handler(OptimisticLockError)
    async def lock_conflict(request, error):
        from fastapi.responses import JSONResponse
        return JSONResponse(
            {"detail": str(error), "path": error.path, "expected_version": error.expected_version, "actual_version": error.actual_version},
            status_code=409,
        )

    @app.exception_handler(FileNotFoundError)
    async def not_found(request, error):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": str(error)}, status_code=404)

    @app.get("/health")
    def health():
        with lock:
            return app.state.engine.store.health_check()

    @app.post("/v1/memories/retain", include_in_schema=False)
    @app.post("/v1/retain")
    def retain(body: Retain):
        scope = {**body.owner(), "session_id": body.session_id or "default"}
        with lock:
            service = app.state.engine
            if body.buffered_turn_id is not None:
                if body.model_fields_set - {"user_id", "agent_id", "session_id", "text", "buffered_turn_id"}:
                    raise ValueError("a retained buffered turn keeps its original metadata")
                return service.retain(body.text, buffered_turn_id=body.buffered_turn_id, **scope)
            if body.turns is not None:
                if body.text is not None or body.epistemic_scope is not None or body.expires_at is not None or body.ttl_seconds is not None:
                    raise ValueError("provide either turns or a single memory")
                return {"turns": [service.retain(**turn.model_dump(), **scope) for turn in body.turns]}
            if not body.text or not body.text.strip():
                raise ValueError("nonempty text is required")
            chronicle = dict(mention_time=body.mention_time, speaker=body.speaker)
            expiry = dict(ttl_seconds=body.ttl_seconds, expires_at=_timestamp(body.expires_at), valid_from=_timestamp(body.valid_from))
            if body.epistemic_scope or body.memory_type == "procedural" or body.source_turn_ids or body.zkeys or body.is_constraint:
                turns = body.source_turn_ids
                if not turns:
                    from memengine.foresight import interval
                    mention = parse_session_clock(body.mention_time)
                    stamp = (mention if mention is not None else service._now()) if body.timestamp is None else body.timestamp
                    interval(body.text, stamp, **expiry)
                    if body.memory_type == "procedural" and not body.trigger.strip():
                        raise ValueError("procedural memory requires a trigger condition")
                    with service.store.transaction():
                        turn = service.store.append_turn(body.text, stamp, scope=(scope["user_id"], scope["agent_id"], scope["session_id"]),
                                                         role=body.role, event_ts=body.event_time, speaker=body.speaker,
                                                         mention_time=mention)
                        memory_id = service.remember(body.text, **scope, source_turn_ids=[turn], timestamp=stamp,
                            memory_type=body.memory_type, epistemic_scope=body.epistemic_scope, trigger=body.trigger,
                            confidence=body.confidence, is_constraint=body.is_constraint, zkeys=body.zkeys, **expiry, **chronicle)
                    return {"id": memory_id, "turn_id": turn}
                return {"id": service.remember(body.text, **scope, source_turn_ids=turns, timestamp=body.timestamp,
                    memory_type=body.memory_type, epistemic_scope=body.epistemic_scope, trigger=body.trigger,
                    confidence=body.confidence, is_constraint=body.is_constraint, zkeys=body.zkeys, **expiry, **chronicle)}
            return service.retain(body.text, **scope, role=body.role, timestamp=body.timestamp, event_time=body.event_time, **expiry, **chronicle)

    @app.post("/v1/memories/recall", include_in_schema=False)
    @app.post("/v1/recall")
    def recall(body: Recall, response: Response):
        start, end = _timestamp(body.time_start), _timestamp(body.time_end)
        if start is not None and end is not None and start >= end:
            raise ValueError("time_start must precede time_end")
        filters = RecallFilter(time_start=start, time_end=end, session_provenance=body.session_id is not None)
        args = body.model_dump(exclude={"assemble", "time_start", "time_end", "task_id", "include_short_term"})
        if not {"token_budget", "packing_mode", "reference_time"} & body.model_fields_set:
            args["packing_mode"] = "top_k"
        with lock:
            service = app.state.engine
            result = (service.build_context(**args, filters=filters, task_id=body.task_id,
                                            include_short_term=body.include_short_term) if body.assemble else
                      service.recall(**args, filters=filters, read_only=True))
        response.headers["X-Retrieval-Ms"] = f"{result['retrieval_ms']:.3f}"
        return result

    @app.post("/v1/memories/reflect", include_in_schema=False)
    @app.post("/v1/reflect")
    def reflect(body: ReasonRequest):
        with lock:
            return app.state.engine.reflect(**body.model_dump())

    @app.post("/v1/answer")
    def answer(body: ReasonRequest):
        with lock:
            return app.state.engine.answer(**body.model_dump())

    @app.post("/v1/consolidate")
    def consolidate(body: Maintenance):
        with lock:
            return app.state.engine.maintain(**body.model_dump(exclude={"session_id"}))

    @app.post("/v1/turns/buffer")
    def buffer_turn(body: Buffer):
        with lock:
            return {"turn_id": app.state.engine.buffer_turn(
                **body.model_dump(exclude={"user_id", "agent_id", "session_id"}), **body.session_owner())}

    @app.post("/v1/tasks")
    def start_task(body: TaskStart):
        with lock:
            task_id = app.state.engine.start_task(body.goal, body.task_id, body.ttl_sec, **body.session_owner())
            return {"task_id": task_id}

    @app.post("/v1/tasks/{task_id}/workspace")
    def task_workspace(task_id: str, body: Owner):
        with lock:
            return app.state.engine.get_task_workspace(task_id, **body.session_owner())

    @app.post("/v1/tasks/{task_id}/items")
    def add_task_item(task_id: str, body: WorkItem):
        with lock:
            item = app.state.engine.record_task_evidence(
                task_id, body.content, body.item_type, body.source_refs, **body.session_owner())
            return {"item_id": item}

    @app.post("/v1/tasks/{task_id}/items/{item_id}/verify")
    def verify_item(task_id: str, item_id: str, body: Verification):
        with lock:
            app.state.engine.verify_task_item(task_id, item_id, body.status, body.verifier_ref, **body.session_owner())
            return {"verification_status": body.status}

    @app.post("/v1/tasks/{task_id}/checkpoint")
    def checkpoint_task(task_id: str, body: TaskCheckpoint):
        with lock:
            return {"checkpoint": app.state.engine.checkpoint_task(task_id, body.state_dict, **body.session_owner())}

    @app.post("/v1/tasks/{task_id}/resume")
    def resume_task(task_id: str, body: Owner):
        with lock:
            return {"checkpoint": app.state.engine.resume_task(task_id, **body.session_owner())}

    @app.post("/v1/tasks/{task_id}/commit")
    def commit_task(task_id: str, body: Owner):
        with lock:
            return {"promoted": app.state.engine.commit_task(task_id, **body.session_owner())}

    @app.post("/v1/memories/drilldown", include_in_schema=False)
    @app.post("/v1/drilldown")
    def drilldown(body: Drilldown):
        with lock:
            return app.state.engine.store.fetch_raw_window(body.turn_ids, body.window, **body.owner(), session_id=body.session_id)

    @app.get("/v1/memories/history/{target:path}", include_in_schema=False)
    @app.get("/v1/history/{target:path}")
    def history(target, user_id: str = "default", agent_id: str = "default", session_id: str | None = None):
        with lock:
            return app.state.engine.store.memory_history(target, scope=(user_id, agent_id, session_id))

    @app.delete("/v1/memories/{id}")
    def forget(id, hard: bool = False, user_id: str = "default", agent_id: str = "default", session_id: str | None = None):
        with lock:
            service = app.state.engine
            if session_id is not None and service.store.get_line(id, scope=(user_id, agent_id, session_id)) is None:
                return {"deleted": False}
            return {"deleted": service.forget(id, hard=hard, user_id=user_id, agent_id=agent_id)}

    @app.get("/v1/tools/schema")
    def schema():
        with lock:
            return app.state.engine.as_tools()

    @app.post("/v1/tools/dispatch")
    def dispatch(body: Dispatch):
        with lock:
            return app.state.engine.dispatch_tool(body.name, body.arguments, **body.owner(),
                                                  session_id=body.session_id, tool_call_id=body.tool_call_id)

    # ---- v2 VFS Memory Engine endpoints ----

    @app.get("/v2/vfs/health")
    def vfs_health():
        with lock:
            return _get_vfs(app).store.health_check()

    @app.post("/v2/vfs/transcripts")
    def vfs_append_transcript(body: VFSTranscript):
        with lock:
            rec = _get_vfs(app).append_transcript(
                body.text, role=body.role, timestamp=body.timestamp, speaker=body.speaker,
                metadata=body.metadata, **body.session_owner(),
            )
            return {"transcript_id": rec.transcript_id, "session_id": rec.session_id, "timestamp": rec.timestamp}

    @app.post("/v2/vfs/write")
    def vfs_write(body: VFSWrite):
        with lock:
            return _get_vfs(app).write(**body.model_dump(exclude={"user_id", "agent_id", "session_id"}), **body.owner())

    @app.post("/v2/vfs/write-rule")
    def vfs_write_rule(body: VFSWriteRule):
        with lock:
            return _get_vfs(app).write_rule(**body.model_dump(exclude={"user_id", "agent_id", "session_id"}), **body.owner())

    @app.post("/v2/vfs/read-note")
    def vfs_read_note(body: VFSPath):
        with lock:
            node = _get_vfs(app).read_note(body.path, **body.owner())
            if node is None:
                raise FileNotFoundError(f"VFS node not found: {body.path}")
            return node

    @app.post("/v2/vfs/tree")
    def vfs_tree(body: VFSPath):
        with lock:
            return {"path": body.path, "outline": _get_vfs(app).tree_outline(path=body.path, layer=body.layer or "L2", **body.owner())}

    @app.post("/v2/vfs/search")
    def vfs_search(body: VFSSearch):
        with lock:
            tags = tuple(body.salience_tags) if body.salience_tags else None
            return _get_vfs(app).search(
                body.query, path_prefix=body.path_prefix, verified_only=body.verified_only,
                salience_tags=tags, reference_time=body.reference_time, top_k=body.top_k, **body.owner(),
            )

    @app.post("/v2/vfs/context")
    def vfs_context(body: VFSContext):
        with lock:
            return _get_vfs(app).build_context(**body.model_dump(exclude={"user_id", "agent_id", "session_id"}), **body.owner())

    @app.post("/v2/vfs/compact")
    def vfs_compact(body: VFSCompact):
        with lock:
            return _get_vfs(app).compact_directory(body.dir_path, **body.owner())

    @app.post("/v2/vfs/rollback")
    def vfs_rollback(body: VFSRollback):
        with lock:
            vfs = _get_vfs(app)
            if body.dream_batch_id:
                return {"restored": vfs.rollback_batch(body.dream_batch_id, **body.owner())}
            if body.path:
                return vfs.rollback(body.path, snapshot_id=body.snapshot_id, **body.owner())
            raise ValueError("rollback requires either dream_batch_id or path")

    @app.get("/v2/vfs/tools/schema")
    def vfs_tools_schema():
        with lock:
            return _get_vfs(app).as_tools()

    @app.post("/v2/vfs/tools/dispatch")
    def vfs_tools_dispatch(body: Dispatch):
        with lock:
            return _get_vfs(app).as_tools(**body.owner()).dispatch_tool(body.name, body.arguments)

    return app
