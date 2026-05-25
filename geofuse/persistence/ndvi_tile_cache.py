"""Persistent NDVI tile cache shared across runs and processes.

Mirrors the structure of :mod:`geofuse.persistence.pano_cache`, but for
multi-megabyte raster tiles instead of two-float pano summaries. The tile
*bodies* sit on disk (SQLite blobs are the wrong tool at this size) and a
small SQLite WAL **index** at ``<cache_root>/_index.db`` tracks bytes,
tile counts, and last-access timestamps for LRU + size-cap eviction.

Key contract (mirrors :class:`PanoCache`):

* ``workspace_dir(resume_key)`` → returns (creating if needed) the per-key
  directory the engine writes ``tile_<idx>.tif`` files into.
* ``touch(resume_key)`` → recomputes ``bytes`` / ``n_tiles`` for the key
  from disk and bumps ``last_access``. Called after a successful mosaic.
* ``evict_lru(max_bytes)`` → drops oldest entries until the total drops
  under the cap, both from disk and from the index.

The engine uses ``workspace_dir`` as both the in-progress tile workspace
*and* the cross-run cache: a successful run no longer wipes the directory,
so the next run with the same parameters reuses every tile instantly. The
LRU + cap keep the cache bounded.
"""

from __future__ import annotations

import os
import shutil
import threading

from geofuse.persistence.sqlite_utils import open_wal_connection

# Default 5 GB. Roughly 500–1000 typical Sentinel-2 5 km tiles. Override per
# instance or via the ``GEOFUSE_NDVI_TILE_CACHE_BYTES`` env var.
DEFAULT_MAX_BYTES = 5 * 1024**3


class NdviTileCache:
    """Filesystem tile store with a SQLite WAL index for LRU eviction."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS entries (
        resume_key  TEXT PRIMARY KEY,
        bytes       INTEGER NOT NULL DEFAULT 0,
        n_tiles     INTEGER NOT NULL DEFAULT 0,
        last_access TEXT    NOT NULL DEFAULT (datetime('now'))
    );
    """

    def __init__(self, cache_root: str, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self.cache_root = cache_root
        self.max_bytes = int(max_bytes)
        os.makedirs(cache_root, exist_ok=True)
        # ``db_path`` is exposed so subprocesses can reopen the index on
        # the same file without sharing the parent's connection (mirrors
        # the PanoCache pattern).
        self.db_path = os.path.join(cache_root, "_index.db")
        self._lock = threading.RLock()
        self._conn = open_wal_connection(self.db_path)
        with self._lock:
            self._conn.executescript(self._SCHEMA)

    # ------------------------------------------------------------------
    # Engine API
    # ------------------------------------------------------------------

    def workspace_dir(self, resume_key: str) -> str:
        """Per-key tile directory; creates the parent on first call.

        Used by ``NDVIEngine._download_with_tiling`` as both the in-progress
        workspace (mid-run, tiles trickle in) and the persistent cache
        (post-run, the directory is preserved for the next same-key run).
        """
        if not resume_key:
            raise ValueError("resume_key must be a non-empty string.")
        path = os.path.join(self.cache_root, resume_key)
        os.makedirs(path, exist_ok=True)
        return path

    def touch(self, resume_key: str) -> None:
        """Recompute size + tile count for ``resume_key`` and bump access time.

        Called after a successful mosaic so LRU eviction uses the actual on-
        disk footprint. Safe to call even when the directory is empty (the
        index row is upserted with zeros).
        """
        if not resume_key:
            return
        path = os.path.join(self.cache_root, resume_key)
        total_bytes = 0
        n_tiles = 0
        if os.path.isdir(path):
            for fname in os.listdir(path):
                fpath = os.path.join(path, fname)
                if os.path.isfile(fpath):
                    try:
                        total_bytes += os.path.getsize(fpath)
                    except OSError:
                        continue
                    n_tiles += 1
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO entries "
                "(resume_key, bytes, n_tiles, last_access) "
                "VALUES (?, ?, ?, datetime('now'))",
                (resume_key, int(total_bytes), int(n_tiles)),
            )

    def evict_lru(self, max_bytes: int | None = None) -> tuple[int, int]:
        """Drop oldest entries until total cache size ≤ ``max_bytes``.

        Returns ``(n_entries_removed, bytes_freed)``. ``max_bytes=None`` uses
        the instance default. Safe to call after every run — a no-op when
        the cache is already under the cap.
        """
        cap = self.max_bytes if max_bytes is None else int(max_bytes)
        if cap <= 0:
            return (0, 0)
        with self._lock:
            rows = self._conn.execute(
                "SELECT resume_key, bytes FROM entries ORDER BY last_access ASC"
            ).fetchall()
        total = sum(int(b) for _, b in rows)
        if total <= cap:
            return (0, 0)
        n_removed = 0
        bytes_freed = 0
        for key, byte_count in rows:
            if total <= cap:
                break
            path = os.path.join(self.cache_root, key)
            shutil.rmtree(path, ignore_errors=True)
            with self._lock:
                self._conn.execute("DELETE FROM entries WHERE resume_key = ?", (key,))
            total -= int(byte_count)
            n_removed += 1
            bytes_freed += int(byte_count)
        return (n_removed, bytes_freed)

    def stats(self) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(bytes), 0) FROM entries"
            ).fetchone()
        return {"entries": int(row[0]), "bytes": int(row[1])}

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass
