"""SQLite connection helper used by every persistence module.

WAL mode lets workers write while the UI reads without lock contention. Autocommit
isolation (``isolation_level=None``) gives us explicit transaction control; the
caller uses ``BEGIN`` / ``COMMIT`` only where batching is worth it.

A kill mid-``COMMIT`` rolls back to the last committed transaction — corruption
under WAL with these settings is not a practical concern.
"""

from __future__ import annotations

import os
import sqlite3


def open_wal_connection(path: str) -> sqlite3.Connection:
    """Open a SQLite connection in WAL mode with sensible defaults.

    Creates the parent directory if missing. Returns an autocommit connection
    safe to share across threads (callers still guard concurrent statements
    with their own lock).
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    conn = sqlite3.connect(
        path,
        check_same_thread=False,
        isolation_level=None,
        timeout=30.0,
    )
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn
