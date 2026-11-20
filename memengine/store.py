"""Compatibility imports for the original SQLite store API."""
from memengine.storage.base import DEFAULT_SCOPE, RecallFilter
from memengine.storage.sqlite import SQLiteStore, key_type

Store = SQLiteStore
__all__ = ["Store", "SQLiteStore", "DEFAULT_SCOPE", "RecallFilter", "key_type"]
