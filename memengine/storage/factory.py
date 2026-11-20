"""Resolve a local SQLite path or an explicitly configured PostgreSQL DSN."""
import os
from urllib.parse import unquote

from memengine.storage.sqlite import SQLiteStore


def open_store(db_url=None, *, path=':memory:', embedder=None):
    target = str(db_url or (os.getenv('MEMENGINE_DB_URL') if str(path) == ':memory:' else None) or path)
    if target.startswith(('postgresql://', 'postgres://')):
        from memengine.storage.postgres import PostgresStore
        return PostgresStore(target, embedder=embedder)
    if target.startswith('sqlite:///'):
        target = unquote(target.removeprefix('sqlite:///'))
    elif '://' in target:
        raise ValueError('unsupported database URL scheme')
    return SQLiteStore(target or ':memory:', embedder=embedder)
