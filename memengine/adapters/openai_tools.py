"""OpenAI Chat Completions tool definitions generated from UnifiedToolRegistry."""
from __future__ import annotations

from memengine.adapters.registry import UnifiedToolRegistry


def openai_tool_definitions(engine, *, user_id: str = "default", agent_id: str = "default", session_id: str | None = None) -> list[dict]:
    return UnifiedToolRegistry(engine, user_id=user_id, agent_id=agent_id, session_id=session_id).openai_definitions()
