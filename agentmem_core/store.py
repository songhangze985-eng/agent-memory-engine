"""Compatibility imports for the original SQLite store API."""
from agentmem_core.storage.base import DEFAULT_SCOPE, RecallFilter
from agentmem_core.storage.sqlite import SQLiteStore, key_type

Store = SQLiteStore
__all__ = ["Store", "SQLiteStore", "DEFAULT_SCOPE", "RecallFilter", "key_type"]
