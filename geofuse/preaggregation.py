"""Persisted, resumable pre-aggregation cache for the fusion optimizer.

Before optimization, every sample entity's metric values are aggregated across
every buffer radius in the ladder and every aggregation statistic (mean +
p10..p90 in 10 % steps), for each channel (``veg``, ``terrain``, ``ndvi``). The
results are written to a SQLite database so that each Optuna trial reads a single
``(channel, radius, stat)`` column via a fast indexed ``SELECT`` instead of
recomputing circular-buffer aggregations — and so the work survives cancels and
crashes and is reused by later runs with the same inputs.

Schema (one table per channel)::

    <channel>(radius INTEGER, entity_id INTEGER,
              mean REAL, p10 REAL, p20 REAL, ..., p90 REAL,
              PRIMARY KEY (radius, entity_id))

A ``meta`` key/value table records the data-config fingerprint, the radius
ladders, the stat set, the entity count, the schema version, and a completion
flag. A ``done_entities`` table records which entities are fully written, so a
resumed build skips them.

Layout choice: ``radius`` is a *row* key (not a column) so the column count per
table stays at 12 regardless of how fine the radius ladder is — this avoids
SQLite's per-table column cap on national-scale ladders while keeping
``WHERE radius = ?`` scans fast via the clustered primary key.

The aggregation math is format-aware: vector (point) metrics use a ``BallTree``
radius query; raster metrics use per-entity windowed reads with a circular mask.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from typing import Any

import numpy as np

from .persistence.sqlite_utils import open_wal_connection

SCHEMA_VERSION = 1

# Aggregation statistics stored per (entity, radius). ``mean`` plus the deciles
# p10..p90; ``median`` is served from ``p50``.
PERCENTILES: tuple[int, ...] = (10, 20, 30, 40, 50, 60, 70, 80, 90)
STAT_COLUMNS: tuple[str, ...] = ("mean",) + tuple(f"p{p}" for p in PERCENTILES)
CHANNELS: tuple[str, ...] = ("veg", "terrain", "ndvi")

_N_STATS = len(STAT_COLUMNS)
# Bound on the in-memory column cache (each entry is one float column for all
# entities at one (channel, radius, stat)). Keeps repeated trial lookups cheap
# without unbounded growth.
_COLUMN_CACHE_MAX = 1024


def compute_all_stats(values: np.ndarray) -> np.ndarray:
    """Return ``[mean, p10, p20, ..., p90]`` as float32; all-NaN if empty."""
    if values.size == 0:
        return np.full(_N_STATS, np.nan, dtype=np.float32)
    out = np.empty(_N_STATS, dtype=np.float32)
    out[0] = float(values.mean())
    out[1:] = np.percentile(values, PERCENTILES)
    return out


def stat_to_column(stat: str, percentile: int | None) -> str | None:
    """Map an Optuna (stat, percentile) choice to a stored column, or None.

    ``None`` means the value is not on the stored grid (e.g. an off-grid
    percentile), signalling the caller to fall back to on-the-fly aggregation.
    """
    if stat == "mean":
        return "mean"
    if stat == "median":
        return "p50"
    if stat == "percentile":
        try:
            p = int(percentile)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return f"p{p}" if p in PERCENTILES else None
    return None


# ---------------------------------------------------------------------------
# Format-aware per-batch aggregation
# ---------------------------------------------------------------------------


def build_vector_index(metric_gdf, metric_crs, value_col: str):
    """Build a ``BallTree`` over a point metric in ``metric_crs``.

    Returns ``(tree, values)`` where ``values`` are the metric values aligned
    with the tree's points. Raises if the value column has no non-NaN features.
    """
    from sklearn.neighbors import BallTree

    m = metric_gdf.to_crs(metric_crs)
    m = m[m[value_col].notna()]
    if len(m) == 0:
        raise ValueError(f"pre-aggregation: metric column '{value_col}' is all-NaN")
    xy = np.column_stack([m.geometry.x.to_numpy(), m.geometry.y.to_numpy()]).astype(
        np.float64
    )
    return BallTree(xy), m[value_col].to_numpy(dtype=np.float32)


def vector_batch_stats(
    tree: Any, values: np.ndarray, batch_xy: np.ndarray, radii_m: tuple[int, ...]
) -> np.ndarray:
    """Per-point disk stats from a vector metric: ``[n_points, n_radii, n_stats]``."""
    n = len(batch_xy)
    out = np.full((n, len(radii_m), _N_STATS), np.nan, dtype=np.float32)
    if n == 0:
        return out
    idx_arr, dist_arr = tree.query_radius(
        batch_xy, r=float(radii_m[-1]), return_distance=True
    )
    for b in range(n):
        neigh = idx_arr[b]
        if len(neigh) == 0:
            continue
        v_all = values[neigh]
        d_all = dist_arr[b]
        for ri, r in enumerate(radii_m):
            mask = d_all <= r
            if mask.any():
                out[b, ri, :] = compute_all_stats(v_all[mask])
    return out


def raster_pixel_size_m(transform, is_geographic: bool) -> float:
    """Approximate pixel size in metres (geographic CRS uses a coarse factor)."""
    px = abs(transform.a)
    return px * 111320.0 if is_geographic else px


def raster_batch_stats(
    metric_array: np.ndarray,
    transform: Any,
    pixel_size_m: float,
    batch_xy_raster_crs: np.ndarray,
    radii_m: tuple[int, ...],
) -> np.ndarray:
    """Per-point disk stats from a raster metric via windowed reads."""
    from rasterio.transform import rowcol

    n = len(batch_xy_raster_crs)
    out = np.full((n, len(radii_m), _N_STATS), np.nan, dtype=np.float32)
    if n == 0:
        return out

    h, w = metric_array.shape
    max_r_px = max(1, min(int(float(radii_m[-1]) / pixel_size_m), 10000))

    for i in range(n):
        x, y = batch_xy_raster_crs[i]
        row, col = rowcol(transform, x, y)
        rmin = max(row - max_r_px, 0)
        rmax = min(row + max_r_px + 1, h)
        cmin = max(col - max_r_px, 0)
        cmax = min(col + max_r_px + 1, w)
        if rmin >= rmax or cmin >= cmax:
            continue

        # ``metric_array`` may be an in-memory masked array or a LazyRasterArray
        # (windowed disk read); both return a masked window here.
        window = metric_array[rmin:rmax, cmin:cmax]
        rr = np.arange(rmin, rmax, dtype=np.float64)[:, None]
        cc = np.arange(cmin, cmax, dtype=np.float64)[None, :]
        dist_m = np.sqrt((rr - row) ** 2 + (cc - col) ** 2) * pixel_size_m

        data = np.asarray(np.ma.getdata(window), dtype=np.float64)
        base_valid = ~np.ma.getmaskarray(window) & ~np.isnan(data)

        for ri, r in enumerate(radii_m):
            mask = base_valid & (dist_m <= r)
            if mask.any():
                out[i, ri, :] = compute_all_stats(data[mask])
    return out


# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------


class PreAggregationCache:
    """SQLite-backed per-job pre-aggregation table with resume + fast lookup."""

    def __init__(
        self,
        db_path: str,
        *,
        gvi_radii: tuple[int, ...],
        ndvi_radii: tuple[int, ...],
        fingerprint: str,
    ) -> None:
        self.db_path = db_path
        self.gvi_radii = tuple(int(r) for r in gvi_radii)
        self.ndvi_radii = tuple(int(r) for r in ndvi_radii)
        self.fingerprint = fingerprint
        self._conn = open_wal_connection(db_path)
        self._lock = threading.Lock()
        self._col_cache: OrderedDict[tuple, Any] = OrderedDict()
        self._ensure_schema()

    # -- channel/radius helpers ------------------------------------------------
    def radii_for(self, channel: str) -> tuple[int, ...]:
        return self.ndvi_radii if channel == "ndvi" else self.gvi_radii

    # -- schema / meta ---------------------------------------------------------
    def _ensure_schema(self) -> None:
        with self._lock:
            cur = self._conn
            stat_cols = ", ".join(f"{c} REAL" for c in STAT_COLUMNS)
            for ch in CHANNELS:
                cur.execute(
                    f"CREATE TABLE IF NOT EXISTS {ch} ("
                    f"radius INTEGER NOT NULL, entity_id INTEGER NOT NULL, "
                    f"{stat_cols}, PRIMARY KEY (radius, entity_id))"
                )
            cur.execute(
                "CREATE TABLE IF NOT EXISTS done_entities (entity_id INTEGER PRIMARY KEY)"
            )
            cur.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
            )

    def _get_meta(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def matches_fingerprint(self) -> bool:
        """True if a prior build's fingerprint + ladders + stats match this config."""
        return (
            self._get_meta("schema_version") == str(SCHEMA_VERSION)
            and self._get_meta("fingerprint") == self.fingerprint
            and self._get_meta("gvi_radii") == json.dumps(list(self.gvi_radii))
            and self._get_meta("ndvi_radii") == json.dumps(list(self.ndvi_radii))
            and self._get_meta("stat_columns") == json.dumps(list(STAT_COLUMNS))
        )

    def is_complete(self) -> bool:
        return self.matches_fingerprint() and self._get_meta("complete") == "1"

    def reset(self) -> None:
        """Drop all rows + meta (used when the fingerprint changed)."""
        with self._lock:
            for ch in CHANNELS:
                self._conn.execute(f"DELETE FROM {ch}")
            self._conn.execute("DELETE FROM done_entities")
            self._conn.execute("DELETE FROM meta")
            self._col_cache.clear()

    def write_header(self, n_entities: int) -> None:
        with self._lock:
            self._set_meta("schema_version", str(SCHEMA_VERSION))
            self._set_meta("fingerprint", self.fingerprint)
            self._set_meta("gvi_radii", json.dumps(list(self.gvi_radii)))
            self._set_meta("ndvi_radii", json.dumps(list(self.ndvi_radii)))
            self._set_meta("stat_columns", json.dumps(list(STAT_COLUMNS)))
            self._set_meta("percentiles", json.dumps(list(PERCENTILES)))
            self._set_meta("n_entities", str(int(n_entities)))
            if self._get_meta("complete") is None:
                self._set_meta("complete", "0")

    def mark_complete(self) -> None:
        with self._lock:
            self._set_meta("complete", "1")

    # -- resume ----------------------------------------------------------------
    def pending_entities(self, entity_ids: list[int]) -> list[int]:
        """Return the subset of ``entity_ids`` not yet fully written, in order."""
        done = {
            int(r[0])
            for r in self._conn.execute(
                "SELECT entity_id FROM done_entities"
            ).fetchall()
        }
        return [e for e in entity_ids if int(e) not in done]

    def done_count(self) -> int:
        return int(
            self._conn.execute("SELECT COUNT(*) FROM done_entities").fetchone()[0]
        )

    # -- write -----------------------------------------------------------------
    def write_batch(
        self, entity_ids: list[int], channel_stats: dict[str, np.ndarray]
    ) -> None:
        """Persist one batch transactionally.

        ``channel_stats[channel]`` has shape ``[len(entity_ids), n_radii, n_stats]``
        where ``n_radii`` matches ``radii_for(channel)``.
        """
        placeholders = ", ".join(["?"] * (2 + _N_STATS))
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                for ch in CHANNELS:
                    arr = channel_stats[ch]
                    radii = self.radii_for(ch)
                    rows = []
                    for b, eid in enumerate(entity_ids):
                        for ri, r in enumerate(radii):
                            rows.append(
                                (int(r), int(eid), *(float(v) for v in arr[b, ri, :]))
                            )
                    self._conn.executemany(
                        f"INSERT OR REPLACE INTO {ch} "
                        f"(radius, entity_id, {', '.join(STAT_COLUMNS)}) "
                        f"VALUES ({placeholders})",
                        rows,
                    )
                self._conn.executemany(
                    "INSERT OR IGNORE INTO done_entities(entity_id) VALUES (?)",
                    [(int(e),) for e in entity_ids],
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            self._col_cache.clear()

    # -- read ------------------------------------------------------------------
    def lookup(
        self,
        entity_ids: np.ndarray,
        channel: str,
        radius: int,
        column: str,
    ) -> np.ndarray | None:
        """Return float32 values for ``entity_ids`` at ``(channel, radius, column)``.

        Returns ``None`` when the channel/column/radius is not stored (caller
        falls back). Missing entities come back as NaN. A small LRU caches the
        full column per ``(channel, radius, column)`` so repeated trial lookups
        avoid re-querying SQLite.
        """
        if channel not in CHANNELS or column not in STAT_COLUMNS:
            return None
        radius = int(radius)
        if radius not in self.radii_for(channel):
            return None

        key = (channel, radius, column)
        with self._lock:
            entry = self._col_cache.get(key)
            if entry is None:
                rows = self._conn.execute(
                    f"SELECT entity_id, {column} FROM {channel} WHERE radius = ?",
                    (radius,),
                ).fetchall()
                if not rows:
                    return None
                ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
                vals = np.fromiter(
                    (np.nan if r[1] is None else r[1] for r in rows),
                    dtype=np.float32,
                    count=len(rows),
                )
                # Store sorted-by-id so lookups vectorise via ``searchsorted``.
                order = np.argsort(ids, kind="stable")
                entry = (ids[order], vals[order])
                self._col_cache[key] = entry
                if len(self._col_cache) > _COLUMN_CACHE_MAX:
                    self._col_cache.popitem(last=False)
            else:
                self._col_cache.move_to_end(key)
            sorted_ids, sorted_vals = entry

        req = np.asarray(entity_ids).astype(np.int64, copy=False)
        out = np.full(req.shape[0], np.nan, dtype=np.float32)
        if sorted_ids.size:
            pos = np.searchsorted(sorted_ids, req)
            pos_clip = np.clip(pos, 0, sorted_ids.size - 1)
            valid = sorted_ids[pos_clip] == req
            out[valid] = sorted_vals[pos_clip[valid]]
        return out

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
