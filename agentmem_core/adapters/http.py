"""Unified FastAPI router generated directly from UnifiedToolRegistry."""
from __future__ import annotations

from threading import RLock
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from agentmem_core.adapters.registry import TOOL_SPECS_BY_NAME, UnifiedToolRegistry
from agentmem_core.vfs.models import OptimisticLockError


class ToolEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    user_id: str = Field(default="default", min_length=1)
    agent_id: str = Field(default="default", min_length=1)
    session_id: str | None = None


def build_unified_router(get_unified_engine, lock: RLock) -> APIRouter:
    """Generate all ``/v2/memory/*`` endpoints directly from ``UnifiedToolRegistry``."""
    router = APIRouter()

    @router.get("/v2/memory/tools")
    def list_tools():
        return {
            "openai_tools": UnifiedToolRegistry(None).openai_definitions(),
            "mcp_tools": UnifiedToolRegistry(None).mcp_definitions(),
        }

    @router.post("/v2/memory/tools/{tool_name}")
    def invoke_tool(tool_name: str, body: ToolEnvelope):
        if tool_name not in TOOL_SPECS_BY_NAME:
            raise HTTPException(404, f"unknown tool: {tool_name}")
        raw = body.model_dump()
        user_id = raw.pop("user_id")
        agent_id = raw.pop("agent_id")
        session_id = raw.pop("session_id")
        with lock:
            reg = UnifiedToolRegistry(get_unified_engine(), user_id=user_id, agent_id=agent_id, session_id=session_id)
            try:
                return reg.dispatch(tool_name, raw)
            except OptimisticLockError as exc:
                raise HTTPException(409, str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc

    # Generate dedicated REST endpoints directly from the same ToolSpec table
    for spec in UnifiedToolRegistry.specs():
        if not spec.http_path:
            continue

        def _make_handler(tname: str):
            def _handler(body: ToolEnvelope) -> Any:
                raw = body.model_dump()
                user_id = raw.pop("user_id")
                agent_id = raw.pop("agent_id")
                session_id = raw.pop("session_id")
                with lock:
                    reg = UnifiedToolRegistry(get_unified_engine(), user_id=user_id, agent_id=agent_id, session_id=session_id)
                    try:
                        return reg.dispatch(tname, raw)
                    except OptimisticLockError as exc:
                        raise HTTPException(409, str(exc)) from exc
                    except ValueError as exc:
                        raise HTTPException(422, str(exc)) from exc

            _handler.__name__ = f"http_{tname}"
            return _handler

        router.add_api_route(spec.http_path, _make_handler(spec.name), methods=[spec.http_method])

    return router
