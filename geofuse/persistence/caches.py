"""SQLite-backed stores: the shared connection helper and the content caches.

Three things that were three files. They share one storage decision — a WAL
connection opened by :func:`open_wal_connection` — and nothing else, so keeping
them together puts the pragma choices next to the two stores that depend on
them instead of one indirection away.

:func:`open_wal_connection` opens in WAL mode so workers can write while the UI
reads without lock contention, with autocommit isolation
(``isolation_level=None``) leaving transaction control to the caller; the
stores use ``BEGIN`` / ``COMMIT`` only where batching pays. A kill mid-COMMIT
rolls back to the last committed transaction — corruption under WAL with these
settings is not a practical concern.

:class:`PanoCache` maps a Street View panorama id to its two GVI floats,
globally across study areas, so any run benefits from every panorama previously
segmented. :class:`NdviTileCache` holds downloaded Earth Engine tiles under a
byte budget with LRU eviction. ``JobStore`` also builds on
``open_wal_connection`` and stays in its own module — it stores job state, not
content.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import threading


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


class PanoCache:
    """Dict-like SQLite store for ``pano_id → {"veg": float, "ter": float}``.

    Reads are served by an in-memory dict overlay that mirrors the SQLite
    table. The first read (or an explicit ``preload()``) loads every row
    into memory in one query — subsequent ``__contains__`` / ``__getitem__``
    calls cost a dict lookup instead of a SQLite round-trip. Writes go to
    SQLite first and then update the overlay so durability is preserved.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS panos (
        pano_id     TEXT PRIMARY KEY,
        veg         REAL NOT NULL,
        ter         REAL NOT NULL,
        computed_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """

    def __init__(self, db_path: str):
        self.db_path = db_path  # exposed for the GVI subprocess to reopen
        self._lock = threading.RLock()
        self._conn = open_wal_connection(db_path)
        with self._lock:
            self._conn.executescript(self._SCHEMA)
        # In-memory overlay: pano_id → (veg, ter). Populated lazily on first
        # access (or via explicit preload()) and kept in sync on every write.
        self._mem: dict[str, tuple[float, float]] = {}
        self._mem_loaded = False

    # ────────────────────────────────────────────────────────────
    # Preload — bulk-load every row into the overlay in one SELECT.
    # ────────────────────────────────────────────────────────────

    def preload(self) -> int:
        """Load every pano row into the in-memory overlay. Idempotent."""
        with self._lock:
            if self._mem_loaded:
                return len(self._mem)
            rows = self._conn.execute("SELECT pano_id, veg, ter FROM panos").fetchall()
            self._mem = {pid: (float(v), float(t)) for pid, v, t in rows}
            self._mem_loaded = True
            return len(self._mem)

    # ────────────────────────────────────────────────────────────
    # Dict protocol used by GVIEngine._process_one_point_async
    # ────────────────────────────────────────────────────────────

    def __contains__(self, pid: str) -> bool:
        if not self._mem_loaded:
            self.preload()
        return pid in self._mem

    def __getitem__(self, pid: str) -> dict:
        if not self._mem_loaded:
            self.preload()
        try:
            v, t = self._mem[pid]
        except KeyError:
            raise KeyError(pid) from None
        return {"veg": v, "ter": t}

    def __setitem__(self, pid: str, val: dict) -> None:
        veg = float(val["veg"])
        ter = float(val["ter"])
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO panos (pano_id, veg, ter, computed_at) "
                "VALUES (?, ?, ?, datetime('now'))",
                (pid, veg, ter),
            )
            if self._mem_loaded:
                self._mem[pid] = (veg, ter)

    def __len__(self) -> int:
        if self._mem_loaded:
            return len(self._mem)
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM panos").fetchone()
        return int(row[0]) if row else 0

    # ────────────────────────────────────────────────────────────
    # Optional helpers
    # ────────────────────────────────────────────────────────────

    def get(self, pid: str, default=None):
        try:
            return self[pid]
        except KeyError:
            return default

    def clear(self) -> None:
        """Drop every entry. Used by tests and the optional UI 'reset cache' control."""
        with self._lock:
            self._conn.execute("DELETE FROM panos")
            self._mem.clear()
            # Overlay remains "loaded" — it's just empty now.

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


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

    # ────────────────────────────────────────────────────────────
    # Engine API
    # ────────────────────────────────────────────────────────────

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
