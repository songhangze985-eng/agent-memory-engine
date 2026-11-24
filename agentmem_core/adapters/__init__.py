"""Adapters generated from the single UnifiedToolRegistry (HTTP, CLI, MCP, OpenAI)."""
from agentmem_core.adapters.mcp import mcp_tool_definitions
from agentmem_core.adapters.mcp_server import MCPServer, run_mcp_stdio
from agentmem_core.adapters.openai_tools import openai_tool_definitions
from agentmem_core.adapters.registry import TOOL_SPECS_BY_NAME, ToolSpec, UnifiedToolRegistry


def __getattr__(name: str):
    if name == "build_unified_router":
        from agentmem_core.adapters.http import build_unified_router

        return build_unified_router
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MCPServer",
    "TOOL_SPECS_BY_NAME",
    "ToolSpec",
    "UnifiedToolRegistry",
    "build_unified_router",
    "mcp_tool_definitions",
    "openai_tool_definitions",
    "run_mcp_stdio",
]
