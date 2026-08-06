"""SQLite-backed persistence layer for GeoFuse.

Modules:
    caches:       WAL connection helper, pano cache, NDVI tile cache.
    job_store:    refresh-safe job tracking across UI reruns.
    job_executor: thread pool that runs job workers.
"""

from geofuse.persistence.caches import open_wal_connection

__all__ = ["open_wal_connection"]
