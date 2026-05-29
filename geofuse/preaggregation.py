"""Persisted, resumable pre-aggregation cache for the fusion optimizer.

Before optimization, every sample entity's metric values are aggregated across
every buffer radius in the ladder and every aggregation statistic (mean +
p10..p90 in 10 % steps), for each channel (``veg``, ``terrain``, ``ndvi``)
and — in longitudinal / mixed-effects mode — every wave. The results are
written to a SQLite database so that each Optuna trial reads a single
``(channel, wave, radius, stat)`` column via a fast indexed ``SELECT`` instead
of recomputing circular-buffer aggregations — and so the work survives cancels
and crashes and is reused by later runs with the same inputs.

Schema (one table per channel)::

    <channel>(wave INTEGER, radius INTEGER, entity_id INTEGER,
              mean REAL, p10 REAL, p20 REAL, ..., p90 REAL,
              PRIMARY KEY (wave, radius, entity_id))

Cross-sectional callers use a single implicit wave (index 0) and do not need
to pass wave info: ``write_batch`` and ``lookup`` default to that wave so
the legacy fusion code is unchanged. Longitudinal callers supply explicit
``wave_labels`` to the constructor and use ``write_channel_wave_batch`` /
``lookup(..., wave_index=...)`` / ``pending_entities_for(...)``.

Static-channel dedup: when a longitudinal run reuses one greenery file across
multiple waves for a channel, the runner calls
:meth:`PreAggregationCache.register_wave_aliases` to declare that those wave
indices share storage with a representative wave; only the representative
wave's rows are computed and written, and lookups for aliased waves
transparently resolve to the representative.

A ``meta`` key/value table records the data-config fingerprint, the radius
ladders, the stat set, the wave labels + aliases, the entity count, the
schema version, and a completion flag. A ``done_channel_wave_entity`` table
records which entities are fully written per ``(channel, wave)``, so a
resumed build skips them.

Layout choice: ``wave`` and ``radius`` are *row* keys (not columns) so the
column count per channel table stays at 12 regardless of how fine the radius
ladder or how many waves there are — this avoids SQLite's per-table column
cap on national-scale ladders while keeping ``WHERE wave = ? AND radius = ?``
scans fast via the clustered primary key.

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

# Bumped from 1 in the wave-axis extension: every per-channel table now has a
# ``wave`` PK column and per-entity completion is tracked per (channel, wave).
# Caches built under the previous schema fail the fingerprint check and are
# reset on the next run.
SCHEMA_VERSION = 2

# Cross-sectional callers use this single implicit wave; longitudinal callers
# pass an explicit ordered tuple of wave labels to the constructor.
DEFAULT_WAVE_LABELS: tuple[str, ...] = ("",)
DEFAULT_WAVE_INDEX = 0

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
        wave_labels: tuple[str, ...] = DEFAULT_WAVE_LABELS,
    ) -> None:
        self.db_path = db_path
        self.gvi_radii = tuple(int(r) for r in gvi_radii)
        self.ndvi_radii = tuple(int(r) for r in ndvi_radii)
        self.fingerprint = fingerprint
        self.wave_labels = tuple(wave_labels) or DEFAULT_WAVE_LABELS
        self._wave_index = {w: i for i, w in enumerate(self.wave_labels)}
        # Identity mapping by default — every wave is its own representative.
        # ``register_wave_aliases`` overwrites entries when a channel reuses a
        # file across multiple waves so only the representative is stored.
        self._wave_alias: dict[tuple[str, int], int] = {
            (ch, i): i for ch in CHANNELS for i in range(len(self.wave_labels))
        }
        self._conn = open_wal_connection(db_path)
        self._lock = threading.Lock()
        self._col_cache: OrderedDict[tuple, Any] = OrderedDict()
        self._ensure_schema()
        self._restore_aliases_from_meta()

    # -- channel/radius helpers ------------------------------------------------
    def radii_for(self, channel: str) -> tuple[int, ...]:
        return self.ndvi_radii if channel == "ndvi" else self.gvi_radii

    def wave_index_of(self, wave_label: str) -> int:
        try:
            return self._wave_index[wave_label]
        except KeyError as exc:
            raise KeyError(
                f"Unknown wave label {wave_label!r}; cache was built with "
                f"wave_labels={list(self.wave_labels)}."
            ) from exc

    def resolve_wave(self, channel: str, wave_index: int) -> int:
        """Return the representative wave index for ``(channel, wave_index)``."""
        return self._wave_alias.get((channel, int(wave_index)), int(wave_index))

    def representative_waves_for(self, channel: str) -> tuple[int, ...]:
        """Distinct representative wave indices for ``channel`` in label order."""
        seen: list[int] = []
        for i in range(len(self.wave_labels)):
            rep = self.resolve_wave(channel, i)
            if rep not in seen:
                seen.append(rep)
        return tuple(seen)

    # -- schema / meta ---------------------------------------------------------
    def _ensure_schema(self) -> None:
        with self._lock:
            cur = self._conn
            stat_cols = ", ".join(f"{c} REAL" for c in STAT_COLUMNS)
            for ch in CHANNELS:
                cur.execute(
                    f"CREATE TABLE IF NOT EXISTS {ch} ("
                    f"wave INTEGER NOT NULL, "
                    f"radius INTEGER NOT NULL, entity_id INTEGER NOT NULL, "
                    f"{stat_cols}, PRIMARY KEY (wave, radius, entity_id))"
                )
            cur.execute(
                "CREATE TABLE IF NOT EXISTS done_channel_wave_entity ("
                "channel TEXT NOT NULL, wave INTEGER NOT NULL, "
                "entity_id INTEGER NOT NULL, "
                "PRIMARY KEY (channel, wave, entity_id))"
            )
            cur.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
            )

    def _restore_aliases_from_meta(self) -> None:
        raw = self._get_meta("wave_aliases")
        if not raw:
            return
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return
        for ch, alias_map in payload.items():
            if ch not in CHANNELS:
                continue
            for w_str, rep in alias_map.items():
                self._wave_alias[(ch, int(w_str))] = int(rep)

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
            and self._get_meta("wave_labels") == json.dumps(list(self.wave_labels))
        )

    def is_complete(self) -> bool:
        return self.matches_fingerprint() and self._get_meta("complete") == "1"

    def reset(self) -> None:
        """Drop all rows + meta (used when the fingerprint changed)."""
        with self._lock:
            for ch in CHANNELS:
                self._conn.execute(f"DELETE FROM {ch}")
            self._conn.execute("DELETE FROM done_channel_wave_entity")
            self._conn.execute("DELETE FROM meta")
            self._col_cache.clear()
            # Restore identity alias map; aliases get re-registered by the
            # runner during the new build.
            self._wave_alias = {
                (ch, i): i
                for ch in CHANNELS
                for i in range(len(self.wave_labels))
            }

    def write_header(self, n_entities: int) -> None:
        with self._lock:
            self._set_meta("schema_version", str(SCHEMA_VERSION))
            self._set_meta("fingerprint", self.fingerprint)
            self._set_meta("gvi_radii", json.dumps(list(self.gvi_radii)))
            self._set_meta("ndvi_radii", json.dumps(list(self.ndvi_radii)))
            self._set_meta("stat_columns", json.dumps(list(STAT_COLUMNS)))
            self._set_meta("percentiles", json.dumps(list(PERCENTILES)))
            self._set_meta("wave_labels", json.dumps(list(self.wave_labels)))
            self._set_meta("n_entities", str(int(n_entities)))
            if self._get_meta("complete") is None:
                self._set_meta("complete", "0")

    def mark_complete(self) -> None:
        with self._lock:
            self._set_meta("complete", "1")

    # -- wave aliases ----------------------------------------------------------
    def register_wave_aliases(
        self, channel: str, alias_map: dict[int, int]
    ) -> None:
        """Declare that some wave indices for ``channel`` share storage.

        ``alias_map`` maps each wave index to its representative wave index;
        the representative is one of the entries (typically the lowest wave
        index of each unique-file group). The runner calls this once per
        channel after grouping waves by file fingerprint so only the
        representative wave is computed and persisted; lookups for aliased
        waves resolve through this mapping. Persisted to ``meta`` so resume
        across a process restart sees the same grouping.
        """
        if channel not in CHANNELS:
            raise ValueError(f"Unknown channel {channel!r}; expected one of {CHANNELS}.")
        n = len(self.wave_labels)
        for w, rep in alias_map.items():
            if not (0 <= int(w) < n) or not (0 <= int(rep) < n):
                raise ValueError(
                    f"alias entry {(w, rep)} out of range for "
                    f"wave_labels={list(self.wave_labels)}."
                )
            self._wave_alias[(channel, int(w))] = int(rep)
        # Persist the merged alias map (across all channels) to meta.
        merged: dict[str, dict[str, int]] = {ch: {} for ch in CHANNELS}
        for (ch, w), rep in self._wave_alias.items():
            if w != rep:
                merged[ch][str(w)] = rep
        with self._lock:
            self._set_meta(
                "wave_aliases",
                json.dumps({k: v for k, v in merged.items() if v}),
            )
            self._col_cache.clear()

    # -- resume ----------------------------------------------------------------
    def pending_entities_for(
        self, channel: str, wave_index: int, entity_ids: list[int]
    ) -> list[int]:
        """Subset of ``entity_ids`` not yet written for ``(channel, wave)``.

        Wave aliases are resolved first — aliased waves share completion
        rows with their representative, so a static-channel build never
        recomputes the same stats per wave.
        """
        rep = self.resolve_wave(channel, wave_index)
        done = {
            int(r[0])
            for r in self._conn.execute(
                "SELECT entity_id FROM done_channel_wave_entity "
                "WHERE channel = ? AND wave = ?",
                (channel, int(rep)),
            ).fetchall()
        }
        return [e for e in entity_ids if int(e) not in done]

    def pending_entities(self, entity_ids: list[int]) -> list[int]:
        """Entities not yet fully written across every (channel, representative wave).

        Legacy / cross-sectional convenience: an entity is "fully done" iff it
        appears in ``done_channel_wave_entity`` for every required
        ``(channel, representative_wave)`` pair. For a single-wave
        cross-sectional cache this is exactly "done across all three
        channels at wave 0" — the previous semantics.
        """
        required: list[tuple[str, int]] = []
        for ch in CHANNELS:
            for rep in self.representative_waves_for(ch):
                required.append((ch, rep))
        if not required:
            return list(entity_ids)
        # Build per-(channel, wave) done sets and intersect.
        done_per: list[set[int]] = []
        for ch, rep in required:
            rows = self._conn.execute(
                "SELECT entity_id FROM done_channel_wave_entity "
                "WHERE channel = ? AND wave = ?",
                (ch, int(rep)),
            ).fetchall()
            done_per.append({int(r[0]) for r in rows})
        fully_done: set[int] = set.intersection(*done_per) if done_per else set()
        return [e for e in entity_ids if int(e) not in fully_done]

    def done_count(self) -> int:
        """Count of entities fully done across every required (channel, wave)."""
        required: list[tuple[str, int]] = []
        for ch in CHANNELS:
            for rep in self.representative_waves_for(ch):
                required.append((ch, rep))
        if not required:
            return 0
        done_per: list[set[int]] = []
        for ch, rep in required:
            rows = self._conn.execute(
                "SELECT entity_id FROM done_channel_wave_entity "
                "WHERE channel = ? AND wave = ?",
                (ch, int(rep)),
            ).fetchall()
            done_per.append({int(r[0]) for r in rows})
        return len(set.intersection(*done_per)) if done_per else 0

    # -- write -----------------------------------------------------------------
    def write_channel_wave_batch(
        self,
        channel: str,
        wave_index: int,
        entity_ids: list[int],
        stats: np.ndarray,
    ) -> None:
        """Persist one channel × one wave's stats batch transactionally.

        ``stats`` has shape ``[len(entity_ids), n_radii, n_stats]`` where
        ``n_radii`` matches :meth:`radii_for`. The write is recorded against
        the representative wave for ``(channel, wave_index)`` so aliased
        waves transparently share storage.
        """
        if channel not in CHANNELS:
            raise ValueError(f"Unknown channel {channel!r}; expected one of {CHANNELS}.")
        rep = self.resolve_wave(channel, wave_index)
        radii = self.radii_for(channel)
        placeholders = ", ".join(["?"] * (3 + _N_STATS))
        rows = []
        for b, eid in enumerate(entity_ids):
            for ri, r in enumerate(radii):
                rows.append(
                    (
                        int(rep),
                        int(r),
                        int(eid),
                        *(float(v) for v in stats[b, ri, :]),
                    )
                )
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                self._conn.executemany(
                    f"INSERT OR REPLACE INTO {channel} "
                    f"(wave, radius, entity_id, {', '.join(STAT_COLUMNS)}) "
                    f"VALUES ({placeholders})",
                    rows,
                )
                self._conn.executemany(
                    "INSERT OR IGNORE INTO done_channel_wave_entity "
                    "(channel, wave, entity_id) VALUES (?, ?, ?)",
                    [(channel, int(rep), int(e)) for e in entity_ids],
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            # Invalidate any cached column reads for this (channel, wave).
            for key in list(self._col_cache.keys()):
                if key[0] == channel and key[1] == rep:
                    self._col_cache.pop(key, None)

    def write_batch(
        self, entity_ids: list[int], channel_stats: dict[str, np.ndarray]
    ) -> None:
        """Legacy convenience — writes every channel at the default wave (0).

        Kept for cross-sectional callers that compute all channels in one
        per-entity batch. Longitudinal callers use
        :meth:`write_channel_wave_batch` directly.
        """
        for ch in CHANNELS:
            self.write_channel_wave_batch(
                ch, DEFAULT_WAVE_INDEX, entity_ids, channel_stats[ch]
            )

    # -- read ------------------------------------------------------------------
    def lookup(
        self,
        entity_ids: np.ndarray,
        channel: str,
        radius: int,
        column: str,
        *,
        wave_index: int = DEFAULT_WAVE_INDEX,
    ) -> np.ndarray | None:
        """Return float32 values for ``entity_ids`` at the given cell.

        ``wave_index`` defaults to 0 so cross-sectional callers can omit it.
        Aliased waves resolve to their representative transparently. Returns
        ``None`` when the channel/column/radius is not stored (caller falls
        back). Missing entities come back as NaN. A small LRU caches the full
        column per ``(channel, wave, radius, column)`` so repeated trial
        lookups avoid re-querying SQLite.
        """
        if channel not in CHANNELS or column not in STAT_COLUMNS:
            return None
        radius = int(radius)
        if radius not in self.radii_for(channel):
            return None
        rep = self.resolve_wave(channel, wave_index)

        key = (channel, rep, radius, column)
        with self._lock:
            entry = self._col_cache.get(key)
            if entry is None:
                rows = self._conn.execute(
                    f"SELECT entity_id, {column} FROM {channel} "
                    f"WHERE wave = ? AND radius = ?",
                    (int(rep), radius),
                ).fetchall()
                if not rows:
                    return None
                ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
                vals = np.fromiter(
                    (np.nan if r[1] is None else r[1] for r in rows),
                    dtype=np.float32,
                    count=len(rows),
                )
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
