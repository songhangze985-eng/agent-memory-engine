"""Resolve a local SQLite path, SQLite+LanceDB hybrid URI, or PostgreSQL DSN."""
import os
from urllib.parse import unquote

from memengine.storage.sqlite import SQLiteStore

_LANCE_SCHEMES = ("lancedb:///", "lance:///", "sqlite+lance:///")


def open_store(db_url=None, *, path=':memory:', embedder=None):
    target = str(db_url or (os.getenv('MEMENGINE_DB_URL') if str(path) == ':memory:' else None) or path)
    if target.startswith(('postgresql://', 'postgres://')):
        from memengine.storage.postgres import PostgresStore
        return PostgresStore(target, embedder=embedder)
    for scheme in _LANCE_SCHEMES:
        if target.startswith(scheme):
            from memengine.storage.lance import LanceHybridStore
            return LanceHybridStore(unquote(target.removeprefix(scheme)), embedder=embedder)
    if target.startswith('sqlite:///'):
        target = unquote(target.removeprefix('sqlite:///'))
    elif '://' in target:
        raise ValueError('unsupported database URL scheme')
    return SQLiteStore(target or ':memory:', embedder=embedder)
