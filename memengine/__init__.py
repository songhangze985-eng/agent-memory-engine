"""Deprecated compatibility shim: ``memengine`` has been renamed to ``agentmem_core`` (PyPI: ``agentmem-core``).

All top-level exports and nested submodule imports (e.g. ``from memengine.core import UnifiedMemoryEngine``
or ``python -m memengine.cli``) transparently forward to ``agentmem_core`` so existing code continues
working without breaking changes until v1.0.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import warnings

_DEPRECATION_MSG = (
    "The 'memengine' package has been renamed to 'agentmem_core' (PyPI: 'agentmem-core') "
    "to avoid collision with WWW'25 MemEngine. Migrate imports to 'agentmem_core'; "
    "the 'memengine' compatibility shim will be removed in v1.0."
)


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target_name: str, target_spec: importlib.machinery.ModuleSpec):
        self._target_name = target_name
        self._target_spec = target_spec

    def create_module(self, spec):
        target = importlib.import_module(self._target_name)
        sys.modules[spec.name] = target
        return target

    def exec_module(self, module):
        # Already executed when importing ``self._target_name``.
        return None

    def get_code(self, fullname: str):
        loader = self._target_spec.loader
        if loader is not None and hasattr(loader, "get_code"):
            return loader.get_code(self._target_name)
        return None

    def get_source(self, fullname: str):
        loader = self._target_spec.loader
        if loader is not None and hasattr(loader, "get_source"):
            return loader.get_source(self._target_name)
        return None

    def get_filename(self, fullname: str):
        loader = self._target_spec.loader
        if loader is not None and hasattr(loader, "get_filename"):
            return loader.get_filename(self._target_name)
        return self._target_spec.origin

    def is_package(self, fullname: str) -> bool:
        return self._target_spec.submodule_search_locations is not None


class _MemengineAliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None):
        if not fullname.startswith("memengine."):
            return None
        target_name = "agentmem_core." + fullname[len("memengine."):]
        try:
            target_spec = importlib.util.find_spec(target_name)
        except ModuleNotFoundError:
            return None
        if target_spec is None:
            return None
        loader = _AliasLoader(target_name, target_spec)
        spec = importlib.machinery.ModuleSpec(
            fullname,
            loader,
            origin=target_spec.origin,
            is_package=target_spec.submodule_search_locations is not None,
        )
        if target_spec.submodule_search_locations is not None:
            spec.submodule_search_locations = list(target_spec.submodule_search_locations)
        return spec


if not any(isinstance(finder, _MemengineAliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _MemengineAliasFinder())

warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)

import agentmem_core as _target  # noqa: E402

__all__ = list(getattr(_target, "__all__", ()))


def __getattr__(name: str):
    return getattr(_target, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_target)))
