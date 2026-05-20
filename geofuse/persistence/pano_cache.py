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

    # ------------------------------------------------------------------
    # Preload — bulk-load every row into the overlay in one SELECT.
    # ------------------------------------------------------------------

    def preload(self) -> int:
        """Load every pano row into the in-memory overlay. Idempotent."""
        with self._lock:
            if self._mem_loaded:
                return len(self._mem)
            rows = self._conn.execute(
                "SELECT pano_id, veg, ter FROM panos"
            ).fetchall()
            self._mem = {pid: (float(v), float(t)) for pid, v, t in rows}
            self._mem_loaded = True
            return len(self._mem)

    # ------------------------------------------------------------------
    # Dict protocol used by GVIEngine._process_one_point_async
    # ------------------------------------------------------------------

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
            self._mem.clear()
            # Overlay remains "loaded" — it's just empty now.

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass
