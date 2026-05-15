"""SQLite-backed pano-id cache for GVI.

Replaces the in-memory ``dict[pano_id, {"veg": float, "ter": float}]``
(``st.session_state.master_cache``) that previously lived only for one
Streamlit-process lifetime.

The engine call sites in ``geofuse/gvi.py`` already use dict protocol:

* ``if pid in pano_cache:``          → :py:meth:`__contains__`
* ``cached = pano_cache[pid]``       → :py:meth:`__getitem__`
* ``pano_cache[pid] = {"veg": …}``  → :py:meth:`__setitem__`

so this class is a drop-in replacement. Values are two floats per pano-id,
keyed by the Street View / scraper panorama identifier. Storage is global
across study areas — any future run benefits from every previously-computed
panorama, regardless of which study area first produced it.
"""

from __future__ import annotations

import threading

from geofuse.persistence.sqlite_utils import open_wal_connection


class PanoCache:
    """Dict-like SQLite store for ``pano_id → {"veg": float, "ter": float}``."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS panos (
        pano_id     TEXT PRIMARY KEY,
        veg         REAL NOT NULL,
        ter         REAL NOT NULL,
        computed_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """

    def __init__(self, db_path: str):
        self._lock = threading.RLock()
        self._conn = open_wal_connection(db_path)
        with self._lock:
            self._conn.executescript(self._SCHEMA)

    # ------------------------------------------------------------------
    # Dict protocol used by GVIEngine._process_one_point_async
    # ------------------------------------------------------------------

    def __contains__(self, pid: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM panos WHERE pano_id = ?", (pid,)
            ).fetchone()
        return row is not None

    def __getitem__(self, pid: str) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT veg, ter FROM panos WHERE pano_id = ?", (pid,)
            ).fetchone()
        if row is None:
            raise KeyError(pid)
        return {"veg": row[0], "ter": row[1]}

    def __setitem__(self, pid: str, val: dict) -> None:
        veg = float(val["veg"])
        ter = float(val["ter"])
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO panos (pano_id, veg, ter, computed_at) "
                "VALUES (?, ?, ?, datetime('now'))",
                (pid, veg, ter),
            )

    def __len__(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM panos").fetchone()
        return int(row[0]) if row else 0

    # ------------------------------------------------------------------
    # Optional helpers
    # ------------------------------------------------------------------

    def get(self, pid: str, default=None):
        try:
            return self[pid]
        except KeyError:
            return default

    def clear(self) -> None:
        """Drop every entry. Used by tests and the optional UI 'reset cache' control."""
        with self._lock:
            self._conn.execute("DELETE FROM panos")

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass
