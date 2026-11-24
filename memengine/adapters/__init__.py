"""Adapters generated from the single UnifiedToolRegistry (HTTP, CLI, MCP, OpenAI)."""
from memengine.adapters.mcp import mcp_tool_definitions
from memengine.adapters.openai_tools import openai_tool_definitions
from memengine.adapters.registry import TOOL_SPECS_BY_NAME, ToolSpec, UnifiedToolRegistry


def __getattr__(name: str):
    if name == "build_unified_router":
        from memengine.adapters.http import build_unified_router

        return build_unified_router
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TOOL_SPECS_BY_NAME",
    "ToolSpec",
    "UnifiedToolRegistry",
    "build_unified_router",
    "mcp_tool_definitions",
    "openai_tool_definitions",
]
