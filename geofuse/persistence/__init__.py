"""SQLite-backed persistence layer for GeoFuse.

Modules:
    sqlite_utils: shared connection helper (WAL mode, autocommit).
    job_store:    refresh-safe job tracking (Feature 0).
    job_executor: thread pool that runs job workers (Feature 0).
"""

from geofuse.persistence.sqlite_utils import open_wal_connection

__all__ = ["open_wal_connection"]
