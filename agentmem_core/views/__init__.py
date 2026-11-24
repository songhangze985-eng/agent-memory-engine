"""Derived views over the single unified fact layer (no duplicate fact storage)."""
from agentmem_core.views.profile import ProfileView
from agentmem_core.views.vfs import VFSView

__all__ = ["ProfileView", "VFSView"]
