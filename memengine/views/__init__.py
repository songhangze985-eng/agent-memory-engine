"""Derived views over the single unified fact layer (no duplicate fact storage)."""
from memengine.views.profile import ProfileView
from memengine.views.vfs import VFSView

__all__ = ["ProfileView", "VFSView"]
