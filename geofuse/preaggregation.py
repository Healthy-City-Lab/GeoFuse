"""Persisted, resumable pre-aggregation cache for the fusion optimizer.

Before optimization, every sample entity's metric values are aggregated across
every buffer radius in the ladder and every aggregation statistic (mean +
p10..p90 in 10 % steps), for each channel (``veg``, ``terrain``, ``ndvi``)
and — in longitudinal / mixed-effects mode — every wave. The results are
written to a SQLite database so that each Optuna trial reads a single
``(channel, wave, radius, stat)`` column via a fast indexed ``SELECT`` instead
of recomputing circular-buffer aggregations — and so the work survives cancels
and crashes and is reused by later runs with the same inputs.

Storage. The bulk stat values live in dense per-``(channel, representative
wave)`` ``numpy.memmap`` files — shape ``(n_entities, n_radii, n_stats)``,
``float32``, ``NaN`` for an entity with no imagery in range. A dense array is
the natural container for a write-once / load-all payload: writes are a
``memcpy`` by entity ordinal and the load is an ``mmap``, not a parse, so the
on-disk size tracks the actual float32 footprint (no per-row key overhead).
Entity id ↔ ordinal is fixed once at build start and stored in a sidecar.

A small SQLite database alongside the memmaps keeps only the bookkeeping: a
``meta`` key/value table (fingerprint, ladders, stats, wave labels + aliases,
schema version, completion, coverage signature) and ``done_channel_wave_entity``
(which entities are fully written per ``(channel, wave)``), so a resumed build
skips finished entities. Both tables are tiny next to the value payload.

Values are greenery fractions rounded to four decimals. A complete cache is
loaded into an in-memory structure at engine start
(:mod:`geofuse.preaggr_memory`); per-trial lookups then read RAM, not disk.

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
The aggregation math is format-aware: vector (point) metrics use a ``BallTree``
radius query; raster metrics use per-entity windowed reads with a circular mask.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import threading
from typing import Any

import numpy as np

from .persistence.sqlite_utils import open_wal_connection

# Bumped when the on-disk greenery-cache layout changes so stale files are
# ignored rather than mis-read.
GREENERY_SCHEMA_VERSION = 1

# Version 2 added the ``wave`` PK column and per-(channel, wave) completion
# tracking. Version 3 moved raster aggregation onto buffered discs reprojected
# into the raster CRS for every entity type. Version 5 moved the bulk stat
# values from per-channel SQLite tables into dense per-(channel, wave) memmaps,
# leaving only meta + completion tracking in SQLite. Stored stats from earlier
# versions live in an incompatible layout, so they are reset rather than reused.
# Caches built under an older schema fail the fingerprint check and rebuild.
SCHEMA_VERSION = 5

# Cross-sectional callers use this single implicit wave; longitudinal callers
# pass an explicit ordered tuple of wave labels to the constructor.
DEFAULT_WAVE_LABELS: tuple[str, ...] = ("",)
DEFAULT_WAVE_INDEX = 0

# Aggregation statistics stored per (entity, radius): ``mean`` plus a compact
# percentile set (p10, p25, p75, p90 and p50 for the median). The full decile
# grid was mostly redundant for the search; this trims storage and the in-RAM
# footprint. ``median`` is served from ``p50``.
PERCENTILES: tuple[int, ...] = (10, 25, 50, 75, 90)
STAT_COLUMNS: tuple[str, ...] = ("mean",) + tuple(f"p{p}" for p in PERCENTILES)
CHANNELS: tuple[str, ...] = ("veg", "terrain", "ndvi")

_N_STATS = len(STAT_COLUMNS)
# Greenery metrics (GVI/NDVI) are fractions; four decimals is the useful
# resolution, so stored values are rounded there. This drops noise digits (so
# the on-disk REAL and the in-RAM float32 both represent the value cleanly) and
# makes the cache marginally more compressible.
_STORE_DECIMALS = 4


def compute_all_stats(values: np.ndarray) -> np.ndarray:
    """Return ``[mean, p10, p25, p50, p75, p90]`` as float32, rounded to four
    decimals; all-NaN if empty."""
    if values.size == 0:
        return np.full(_N_STATS, np.nan, dtype=np.float32)
    out = np.empty(_N_STATS, dtype=np.float32)
    out[0] = float(values.mean())
    out[1:] = np.percentile(values, PERCENTILES)
    return np.round(out, _STORE_DECIMALS)


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
# Buffered-disc construction and raster reduction
# ---------------------------------------------------------------------------
#
# Every raster aggregation in this module follows one rule: buffer the entity in
# the projected (metres) grid CRS, reproject the *buffer* into the raster's
# native CRS, and reduce the pixels it touches. The raster is never reprojected
# and no degree-to-metre conversion factor is involved, so the sampled footprint
# is a true metric disc for any raster CRS and any entity geometry type.

# Circle approximation for every buffered disc built here. One shared value so
# all paths reduce over the identical reference geometry.
BUFFER_QUAD_SEGS: int = 32


def origin_circle_templates(radii_eff: Any) -> dict:
    """Origin-centred buffer polygons, one per distinct effective radius.

    Point entities translate these instead of re-buffering: shapely lays a
    buffer's vertices out as centre + r·(cos θ, sin θ) over a fixed θ sequence,
    so a translated template is vertex-identical to buffering the point itself.
    """
    from shapely.geometry import Point as _Point

    return {
        float(r): _Point(0.0, 0.0).buffer(float(r), quad_segs=BUFFER_QUAD_SEGS)
        for r in {float(x) for x in radii_eff}
        if float(r) > 0.0
    }


def buffer_at(entity: Any, radius_eff: float, templates: dict | None = None) -> Any:
    """Buffer one entity at one effective radius, in the entity's own CRS."""
    r = float(radius_eff)
    if r <= 0.0:
        return entity
    if templates is not None and entity.geom_type == "Point":
        from shapely.affinity import translate as _translate

        tpl = templates.get(r)
        if tpl is not None:
            return _translate(tpl, entity.x, entity.y)
    return entity.buffer(r, quad_segs=BUFFER_QUAD_SEGS)


def reproject_geoms(geoms: list, xy_fn: Any) -> list:
    """Reproject a list of geometries with a single transformer call.

    ``shapely.transform`` over an array hands the transformation every
    coordinate at once, so one pyproj call covers the whole list regardless of
    geometry structure (holes and multi-parts included).
    """
    if xy_fn is None:
        return list(geoms)
    import shapely as _shapely

    arr = np.empty(len(geoms), dtype=object)
    arr[:] = geoms

    def _tf(coords: np.ndarray) -> np.ndarray:
        x, y = xy_fn(coords[:, 0], coords[:, 1])
        return np.column_stack(
            [np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)]
        )

    return list(_shapely.transform(arr, _tf))


def ring_index_grid(discs: list, out_shape: tuple, win_transform: Any) -> np.ndarray | None:
    """Per-pixel index of the smallest disc touching it (``len(discs)`` = none).

    ``discs`` must be ordered by ascending radius, which makes them strictly
    nested: a pixel touched by a smaller disc is necessarily touched by every
    larger one. Burning them largest-first in a single ``rasterize`` pass
    therefore leaves each pixel holding its smallest touching disc, and every
    per-radius mask is recovered as ``grid <= i`` — identical to masking each
    disc separately, with one rasterization instead of one per radius.
    """
    from rasterio.features import rasterize

    n = len(discs)
    shapes = [
        (g, i)
        for i, g in reversed(list(enumerate(discs)))
        if g is not None and not g.is_empty
    ]
    if not shapes:
        return None
    try:
        return rasterize(
            shapes,
            out_shape=out_shape,
            transform=win_transform,
            fill=n,
            all_touched=True,
            dtype=np.int32,
        )
    except Exception:
        return None


def stats_from_ring_grid(
    values: np.ndarray, ring_index: np.ndarray, n_radii: int
) -> np.ndarray:
    """Per-radius stats from ring indices, via one sort plus prefix slices.

    ``values`` and ``ring_index`` are the window's valid pixels, flattened and
    aligned. Disc *i* is every pixel whose ring index is ``<= i``, so sorting
    once by ring index makes each disc a prefix of the sorted values.
    """
    out = np.full((n_radii, _N_STATS), np.nan, dtype=np.float32)
    if values.size == 0:
        return out
    order = np.argsort(ring_index, kind="stable")
    rings_sorted = ring_index[order]
    vals_sorted = values[order]
    ends = np.searchsorted(rings_sorted, np.arange(n_radii), side="right")
    for i in range(n_radii):
        end = int(ends[i])
        if end > 0:
            out[i, :] = compute_all_stats(vals_sorted[:end])
    return out


def raster_disc_stats(
    discs_grid_crs: list,
    metric_array: Any,
    raster_transform: Any,
    *,
    xy_fn: Any = None,
) -> np.ndarray:
    """Per-radius stats for one entity's nested discs against a raster metric.

    ``discs_grid_crs`` are ascending-radius buffers in the projected grid CRS;
    they are reprojected into the raster's CRS in one call, the largest one
    sizes a single windowed read, and every radius is reduced from one ring
    grid over that window.
    """
    from rasterio.transform import rowcol
    from rasterio.windows import Window
    from rasterio.windows import transform as window_transform

    n_radii = len(discs_grid_crs)
    out = np.full((n_radii, _N_STATS), np.nan, dtype=np.float32)
    if n_radii == 0:
        return out
    discs = reproject_geoms(discs_grid_crs, xy_fn)
    largest = discs[-1]
    if largest is None or largest.is_empty:
        return out

    h, w = metric_array.shape
    minx, miny, maxx, maxy = largest.bounds
    r1, c1 = rowcol(raster_transform, minx, maxy)
    r2, c2 = rowcol(raster_transform, maxx, miny)
    rmin = max(0, min(int(r1), int(r2)))
    rmax = min(h, max(int(r1), int(r2)) + 1)
    cmin = max(0, min(int(c1), int(c2)))
    cmax = min(w, max(int(c1), int(c2)) + 1)
    if rmin >= rmax or cmin >= cmax:
        return out

    window = metric_array[rmin:rmax, cmin:cmax]
    data = np.asarray(np.ma.getdata(window), dtype=np.float64)
    base_valid = ~np.ma.getmaskarray(window) & ~np.isnan(data)
    if not base_valid.any():
        return out
    win_transform = window_transform(
        Window(cmin, rmin, cmax - cmin, rmax - rmin), raster_transform
    )
    ring = ring_index_grid(discs, window.shape, win_transform)
    if ring is None:
        return out
    return stats_from_ring_grid(data[base_valid], ring[base_valid], n_radii)


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


def build_vector_index_multi(metric_gdf, metric_crs, value_cols: list[str]):
    """Build one ``BallTree`` shared by several value columns of one metric.

    ``veg`` and ``terrain`` are two attribute columns of the same GVI points,
    so their geometry — and the expensive ``query_radius`` over it — is
    identical. Indexing once over the rows valid in *every* requested column
    lets a single query serve all of them. Returns ``(tree, {col: values})``
    with each column's values aligned to the tree's points. Raises if the
    shared valid set is empty.
    """
    from sklearn.neighbors import BallTree

    m = metric_gdf.to_crs(metric_crs)
    mask = np.ones(len(m), dtype=bool)
    for col in value_cols:
        mask &= m[col].notna().to_numpy()
    m = m[mask]
    if len(m) == 0:
        raise ValueError(
            f"pre-aggregation: no rows with all of {value_cols} present"
        )
    xy = np.column_stack([m.geometry.x.to_numpy(), m.geometry.y.to_numpy()]).astype(
        np.float64
    )
    values = {col: m[col].to_numpy(dtype=np.float32) for col in value_cols}
    return BallTree(xy), values


def vector_batch_stats_multi(
    tree: Any,
    values_by_col: dict[str, np.ndarray],
    batch_xy: np.ndarray,
    radii_m: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Per-column disk stats sharing one radius query: ``{col: [n, n_radii,
    n_stats]}``.

    One ``query_radius`` at the max radius and one distance sort per point
    serve every column — each column reduces the same ascending-radius
    prefixes over its own values, so the expensive neighbourhood work is done
    once for ``veg`` and ``terrain`` together.
    """
    cols = list(values_by_col)
    n = len(batch_xy)
    n_radii = len(radii_m)
    out = {c: np.full((n, n_radii, _N_STATS), np.nan, dtype=np.float32) for c in cols}
    if n == 0:
        return out
    radii_arr = np.asarray(radii_m, dtype=np.float64)
    idx_arr, dist_arr = tree.query_radius(
        batch_xy, r=float(radii_m[-1]), return_distance=True
    )
    for b in range(n):
        neigh = idx_arr[b]
        if len(neigh) == 0:
            continue
        order = np.argsort(dist_arr[b], kind="stable")
        d_sorted = dist_arr[b][order]
        neigh_sorted = neigh[order]
        ends = np.searchsorted(d_sorted, radii_arr, side="right")
        for c in cols:
            v_sorted = values_by_col[c][neigh_sorted]
            col_out = out[c]
            for ri in range(n_radii):
                end = int(ends[ri])
                if end > 0:
                    col_out[b, ri, :] = compute_all_stats(v_sorted[:end])
    return out


def vector_batch_stats(
    tree: Any, values: np.ndarray, batch_xy: np.ndarray, radii_m: tuple[int, ...]
) -> np.ndarray:
    """Per-point disk stats from a vector metric: ``[n_points, n_radii, n_stats]``.

    The radii are ascending, so each point's neighbourhood is sorted by
    distance once and every radius reduces a prefix of it — the same membership
    a per-radius distance mask selects, without re-scanning the full
    max-radius neighbourhood once per radius.
    """
    n = len(batch_xy)
    n_radii = len(radii_m)
    out = np.full((n, n_radii, _N_STATS), np.nan, dtype=np.float32)
    if n == 0:
        return out
    radii_arr = np.asarray(radii_m, dtype=np.float64)
    idx_arr, dist_arr = tree.query_radius(
        batch_xy, r=float(radii_m[-1]), return_distance=True
    )
    for b in range(n):
        neigh = idx_arr[b]
        if len(neigh) == 0:
            continue
        d_all = dist_arr[b]
        order = np.argsort(d_all, kind="stable")
        d_sorted = d_all[order]
        v_sorted = values[neigh][order]
        ends = np.searchsorted(d_sorted, radii_arr, side="right")
        for ri in range(n_radii):
            end = int(ends[ri])
            if end > 0:
                out[b, ri, :] = compute_all_stats(v_sorted[:end])
    return out


def detect_metric_grid_spacing(
    metric_gdf: Any,
    *,
    max_seed_attempts: int = 200,
    grid_tol_frac: float = 0.10,
    seed: int = 42,
) -> float | None:
    """Robustly estimate the sampling-grid spacing of a vector metric.

    Vector metric points (GVI sample grid, point-format NDVI, etc.) each
    represent a cell in the engine's underlying sampling grid; the
    aggregation path uses ``sqrt(2) / 2 × spacing`` as the half-diagonal
    of that cell so any cell touching the entity is counted (matching
    the raster ``all_touched=True`` semantic).

    Robustness — null-coverage areas (sky / water / out-of-area pixels)
    inflate mean / nearest-neighbour estimates, so this looks for any
    axis-aligned ``4 × 4`` block of points whose 16 positions all exist
    within ``grid_tol_frac × candidate_spacing`` of where they should
    be. The first such block's spacing is returned. If no block is
    found across ``max_seed_attempts`` random seeds, falls back to the
    median nearest-neighbour distance; if even that is unavailable
    (fewer than 2 points), returns ``None``.

    Coordinates must be in a projected (metric) CRS.
    """
    from sklearn.neighbors import NearestNeighbors

    if metric_gdf is None:
        return None
    try:
        n_pts = len(metric_gdf)
    except TypeError:
        return None
    if n_pts < 2:
        return None
    xy = np.column_stack(
        [
            metric_gdf.geometry.x.to_numpy(dtype=np.float64),
            metric_gdf.geometry.y.to_numpy(dtype=np.float64),
        ]
    )
    rng = np.random.default_rng(seed)

    if n_pts >= 16:
        nn = NearestNeighbors(n_neighbors=2).fit(xy)
        seed_idx = rng.choice(n_pts, size=min(max_seed_attempts, n_pts), replace=False)
        seed_dists, _ = nn.kneighbors(xy[seed_idx])
        radius_tree = NearestNeighbors().fit(xy)
        for k in range(len(seed_idx)):
            s = float(seed_dists[k, 1])
            if s <= 0.0 or not np.isfinite(s):
                continue
            tol = grid_tol_frac * s
            origin = xy[seed_idx[k]]
            target_pts = np.array(
                [origin + (i * s, j * s) for i in range(4) for j in range(4)]
            )
            hits = radius_tree.radius_neighbors(
                target_pts, radius=tol, return_distance=False
            )
            if all(len(h) > 0 for h in hits):
                return s

    nn = NearestNeighbors(n_neighbors=2).fit(xy)
    d, _ = nn.kneighbors(xy)
    return float(np.median(d[:, 1]))


def metric_cell_buffer_m(metric_gdf: Any) -> float:
    """Half-diagonal of the metric's cell — buffer to add for vector queries.

    Returns ``0.0`` when spacing cannot be detected, so callers degrade
    to today's strict ``intersects`` semantic instead of raising.
    """
    spacing = detect_metric_grid_spacing(metric_gdf)
    if spacing is None:
        return 0.0
    return float(spacing * (np.sqrt(2.0) / 2.0))


def vector_batch_geometry_stats(
    metric_gdf: Any,
    value_col: str,
    entity_geoms: list,
    radii_m: tuple[int, ...],
    *,
    cell_buffer_m: float = 0.0,
) -> np.ndarray:
    """Per-entity, per-radius, per-stat aggregation over ``entity.buffer(R)``.

    Returns ``[n_entities, n_radii, n_stats]``. ``entity_geoms`` and
    ``metric_gdf`` must share the same CRS (typically a metric UTM
    projection so the radii in metres are meaningful).

    For raster-equivalent semantics on vector data, pass
    ``cell_buffer_m = sqrt(2) / 2 × grid_spacing`` (see
    :func:`metric_cell_buffer_m`). The query then uses
    ``entity.buffer(R + cell_buffer_m)`` for every R (including R=0),
    so any vector cell whose footprint touches the entity gets counted —
    matching what the raster path does with ``all_touched=True``.
    """
    n = len(entity_geoms)
    out = np.full((n, len(radii_m), _N_STATS), np.nan, dtype=np.float32)
    if n == 0:
        return out
    values = metric_gdf[value_col].to_numpy(dtype=np.float32)
    if len(values) == 0:
        return out
    sindex = metric_gdf.sindex
    eff_radii = tuple(float(r) + float(cell_buffer_m) for r in radii_m)
    for ei, ent_geom in enumerate(entity_geoms):
        if ent_geom is None or ent_geom.is_empty:
            continue
        for ri, r_eff in enumerate(eff_radii):
            buf_geom = ent_geom if r_eff == 0.0 else ent_geom.buffer(r_eff)
            try:
                hits = np.asarray(
                    sindex.query(buf_geom, predicate="intersects"), dtype=np.int64
                )
            except TypeError:
                # Older geopandas: predicate kwarg not supported; fall back
                # to bbox + intersects loop.
                hits = np.asarray(
                    list(sindex.intersection(buf_geom.bounds)), dtype=np.int64
                )
                if len(hits):
                    geoms = metric_gdf.geometry.iloc[hits]
                    keep = geoms.intersects(buf_geom).to_numpy()
                    hits = hits[keep]
            if len(hits) == 0:
                continue
            vals = values[hits]
            valid = ~np.isnan(vals)
            if valid.any():
                out[ei, ri, :] = compute_all_stats(vals[valid])
    return out


def batch_geometry_stats(
    entity_geoms_in_grid_crs: list,
    channels_meta: list,
    *,
    cancel_check: Any = None,
) -> dict | None:
    """Per-channel pre-aggregation stats over ``entity.buffer(R)`` for a batch.

    Returns ``{channel_name: ndarray[n_entities, n_radii, n_stats]}``, or
    ``None`` if ``cancel_check`` (an optional zero-arg callable) starts
    returning truthy mid-batch — the caller treats ``None`` as "this
    batch was abandoned" and skips the write; the next resume picks the
    same entities up because they're still in the pending list.

    Loop structure (per the user-facing semantic that buffering is the
    expensive bit): for each entity, walk channels and their radii; cache
    each unique buffered geometry per ``(entity, effective_radius)`` so
    channels that share a radius — e.g. raster channels with the same R,
    or vector channels with the same ``cell_buffer_m`` — reuse one
    buffered polygon instead of rebuilding it per channel.

    ``entity_geoms_in_grid_crs`` are the entity geometries in the
    pre-aggregation grid CRS (a projected metres CRS picked by
    :func:`geofuse.crs_utils.select_grid_crs`). Each ``channels_meta``
    entry is a dict with:

      - ``name``: ``"veg"`` / ``"terrain"`` / ``"ndvi"`` (cache key).
      - ``kind``: ``"vector"`` or ``"raster"``.
      - ``radii``: tuple of radii (metres) to aggregate over.
      - ``cell_buffer_m``: extra buffer to add for vector metrics so
        cells touching the entity are counted (raster-equivalent
        semantic); ignored / 0 for raster channels.
      - Vector channels: ``gdf`` (in grid CRS), ``col``, ``sindex``,
        ``values`` (numpy float32 aligned with the gdf rows).
      - Raster channels: ``array``, ``raster_transform``, optionally
        ``to_raster_crs`` (a :class:`pyproj.Transformer` reprojecting
        the buffer from grid CRS to the raster's CRS — leave ``None``
        when they already match).
    """
    n = len(entity_geoms_in_grid_crs)
    output = {
        m["name"]: np.full((n, len(m["radii"]), _N_STATS), np.nan, dtype=np.float32)
        for m in channels_meta
    }
    if n == 0:
        return output

    # Pre-resolve per-channel data references once (avoids repeated dict
    # lookups in the hot inner loop). For both vector and raster channels,
    # ``xy_fn`` is the (x, y) reprojection from grid CRS to the metric's
    # native CRS — ``None`` when the metric is already in the grid CRS.
    resolved: list = []
    for meta in channels_meta:
        kind = meta["kind"]
        if kind == "vector":
            transformer = meta.get("to_metric_crs")
            xy_fn = None
            if transformer is not None:
                xy_fn = (
                    transformer.transform
                    if hasattr(transformer, "transform")
                    else transformer
                )
            resolved.append(
                {
                    "name": meta["name"],
                    "kind": "vector",
                    "radii": meta["radii"],
                    "cell": float(meta.get("cell_buffer_m", 0.0)),
                    "sindex": meta["sindex"],
                    "gdf": meta["gdf"],
                    "values": meta["values"],
                    "xy_fn": xy_fn,
                }
            )
        else:
            transformer = meta.get("to_raster_crs")
            xy_fn = None
            if transformer is not None:
                xy_fn = (
                    transformer.transform
                    if hasattr(transformer, "transform")
                    else transformer
                )
            resolved.append(
                {
                    "name": meta["name"],
                    "kind": "raster",
                    "radii": meta["radii"],
                    "cell": 0.0,
                    "array": meta["array"],
                    "raster_transform": meta["raster_transform"],
                    "xy_fn": xy_fn,
                    "shape": meta["array"].shape,
                }
            )

    # Origin-centred circle templates covering every effective radius any
    # channel asks for, so Point entities translate instead of re-buffering.
    templates = origin_circle_templates(
        [float(r) + m["cell"] for m in resolved for r in m["radii"]]
    )

    for entity_idx, entity in enumerate(entity_geoms_in_grid_crs):
        # Cancellation check fires once per entity (not per channel /
        # radius — the cost would dominate otherwise). For typical
        # entity counts of 10²–10⁴ this gives ms-to-sub-second cancel
        # latency without measurable per-call overhead.
        if cancel_check is not None and cancel_check():
            return None
        if entity is None or entity.is_empty:
            continue
        # Per-entity buffer cache keyed by effective buffer distance in
        # the grid CRS. Two channels sharing the same effective radius
        # reuse one buffered geometry.
        buffer_cache: dict = {}

        def _disc(r_eff: float, _entity=entity, _cache=buffer_cache):
            g = _cache.get(r_eff)
            if g is None:
                g = buffer_at(_entity, r_eff, templates)
                _cache[r_eff] = g
            return g

        for meta in resolved:
            ch = meta["name"]
            radii = meta["radii"]
            cell = meta["cell"]
            kind = meta["kind"]
            eff_radii = [float(r) + cell for r in radii]
            discs = [_disc(r_eff) for r_eff in eff_radii]

            if kind == "raster":
                output[ch][entity_idx, :, :] = raster_disc_stats(
                    discs,
                    meta["array"],
                    meta["raster_transform"],
                    xy_fn=meta["xy_fn"],
                )
                continue

            sindex = meta["sindex"]
            values = meta["values"]
            discs_metric = reproject_geoms(discs, meta["xy_fn"])
            for r_idx, buf_for_query in enumerate(discs_metric):
                try:
                    hits = np.asarray(
                        sindex.query(buf_for_query, predicate="intersects"),
                        dtype=np.int64,
                    )
                except TypeError:
                    hits = np.asarray(
                        list(sindex.intersection(buf_for_query.bounds)),
                        dtype=np.int64,
                    )
                    if len(hits):
                        geoms = meta["gdf"].geometry.iloc[hits]
                        keep = geoms.intersects(buf_for_query).to_numpy()
                        hits = hits[keep]
                if len(hits) == 0:
                    continue
                vals = values[hits]
                valid = ~np.isnan(vals)
                if valid.any():
                    output[ch][entity_idx, r_idx, :] = compute_all_stats(vals[valid])

    return output


def raster_batch_geometry_stats(
    metric_array: np.ndarray,
    raster_transform: Any,
    entity_geoms_buffer_crs: list,
    radii_m: tuple[int, ...],
    *,
    to_raster_crs: Any = None,
) -> np.ndarray:
    """Per-entity, per-radius, per-stat aggregation over ``entity.buffer(R)``.

    Returns ``[n_entities, n_radii, n_stats]``. Per entity, the largest
    buffer's bbox bounds a single windowed read; each radius then masks
    the same window via ``rasterio.features.geometry_mask`` with
    ``all_touched=True`` so pixels straddling the entity boundary are
    counted (matches the user-facing semantic that R=0 means "every
    pixel touching the geometry").

    ``entity_geoms_buffer_crs`` are in the CRS in which buffering happens
    — typically a projected (metres) UTM so the radii are accurate. When
    that CRS differs from the raster's, pass ``to_raster_crs`` (a
    :class:`pyproj.Transformer`, callable, or anything with a
    ``.transform(x, y)`` method); each buffered geometry is reprojected
    to the raster's CRS before the mask op, while the raster itself
    stays in its native CRS (no expensive raster reprojection).
    """
    n = len(entity_geoms_buffer_crs)
    out = np.full((n, len(radii_m), _N_STATS), np.nan, dtype=np.float32)
    if n == 0:
        return out

    if to_raster_crs is None:
        xy_fn = None
    elif hasattr(to_raster_crs, "transform"):
        xy_fn = to_raster_crs.transform
    elif callable(to_raster_crs):
        xy_fn = to_raster_crs
    else:
        raise TypeError(
            "to_raster_crs must be a pyproj.Transformer or callable; "
            f"got {type(to_raster_crs).__name__}."
        )

    templates = origin_circle_templates(radii_m)
    for ei, ent_geom in enumerate(entity_geoms_buffer_crs):
        if ent_geom is None or ent_geom.is_empty:
            continue
        discs = [buffer_at(ent_geom, float(r), templates) for r in radii_m]
        out[ei, :, :] = raster_disc_stats(
            discs, metric_array, raster_transform, xy_fn=xy_fn
        )
    return out


# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------


class GreeneryCache:
    """Reusable per-year greenery pre-aggregation store, resident in RAM.

    Each cache **unit** is one compact ``.npz`` in ``<cache_dir>/greenery/``
    holding the veg / terrain / ndvi stat arrays for the pixels a run
    references, keyed by a **stable global pixel id** (it encodes the pixel's
    grid row/col, so the same physical pixel has the same id in every job).

    A unit is identified by ``cfg_key = hash(veg/terrain/ndvi file identity,
    grid spacing, metric CRS, stat set)`` — nothing about the target — so the
    same greenery + grid is reused across jobs that change only the outcome,
    covariates, or target file. Years that resolve to the same greenery files
    share one unit automatically (dedup). The radius ladder lives *inside* the
    file rather than in the key, so a unit built to a wider ladder is reused for
    a job needing a subset of radii; a job needing a wider ladder rebuilds it.

    Storage is compact — only the referenced pixels, no NaN padding — and the
    whole unit is loaded into RAM once, so per-trial lookups are pure gathers
    shared read-only across worker threads.
    """

    _CHANNELS = ("veg", "terrain", "ndvi")

    def __init__(
        self,
        cache_dir: str,
        *,
        spacing_m: float,
        crs_key: str,
        stats: tuple[str, ...] = STAT_COLUMNS,
    ) -> None:
        self.dir = os.path.join(cache_dir, "greenery")
        os.makedirs(self.dir, exist_ok=True)
        self.spacing_m = float(spacing_m)
        self.crs_key = str(crs_key)
        self.stats = tuple(stats)
        self._stat_index = {s: i for i, s in enumerate(self.stats)}
        # cfg_key -> resident unit dict; wave_index -> cfg_key for lookups.
        self._units: dict[str, dict] = {}
        self._wave_unit: dict[int, str] = {}
        self.n_bytes = 0

    # ---------------------------------------------------------------- identity
    def unit_key(self, veg_id: str, terrain_id: str, ndvi_id: str) -> str:
        """Stable id for the greenery-file set + grid config (not the target)."""
        raw = "|".join(
            [
                str(veg_id),
                str(terrain_id),
                str(ndvi_id),
                f"sp={self.spacing_m:.6f}",
                f"crs={self.crs_key}",
                f"stats={','.join(self.stats)}",
                f"schema={GREENERY_SCHEMA_VERSION}",
            ]
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _path(self, cfg_key: str) -> str:
        return os.path.join(self.dir, f"greenery-{cfg_key}.npz")

    def bind_wave(self, wave_index: int, cfg_key: str) -> None:
        """Route lookups for ``wave_index`` to the unit ``cfg_key``."""
        self._wave_unit[int(wave_index)] = cfg_key

    def is_resident(self, cfg_key: str) -> bool:
        return cfg_key in self._units

    # ----------------------------------------------------------- reuse / build
    def open_unit(
        self,
        cfg_key: str,
        *,
        gvi_radii: tuple[int, ...],
        ndvi_radii: tuple[int, ...],
        required_ids: np.ndarray,
    ) -> tuple[tuple[int, ...], tuple[int, ...], np.ndarray]:
        """Load a reusable file if one covers the job's radii, else start fresh.

        Returns ``(effective_gvi_radii, effective_ndvi_radii, missing_ids)``:
        the ladder that new pixels must be computed at (the file's, when reused,
        so the unit stays uniform) and the referenced ids not yet stored.
        """
        gvi_radii = tuple(int(r) for r in gvi_radii)
        ndvi_radii = tuple(int(r) for r in ndvi_radii)
        required = np.asarray(required_ids, dtype=np.int64)

        if cfg_key not in self._units:
            loaded = self._try_load(self._path(cfg_key), gvi_radii, ndvi_radii)
            if loaded is not None:
                self._units[cfg_key] = loaded
        unit = self._units.get(cfg_key)
        if unit is not None:
            missing = self._setdiff(required, unit["ids"])
            return unit["gvi_radii"], unit["ndvi_radii"], missing

        # No reusable file: build at the job's ladder.
        self._units[cfg_key] = {
            "ids": np.empty(0, np.int64),
            "veg": np.empty((0, len(gvi_radii), len(self.stats)), np.float32),
            "terrain": np.empty((0, len(gvi_radii), len(self.stats)), np.float32),
            "ndvi": np.empty((0, len(ndvi_radii), len(self.stats)), np.float32),
            "gvi_radii": gvi_radii,
            "ndvi_radii": ndvi_radii,
            "dirty": False,
        }
        return gvi_radii, ndvi_radii, required

    def _try_load(
        self, path: str, job_gvi: tuple[int, ...], job_ndvi: tuple[int, ...]
    ) -> dict | None:
        if not os.path.isfile(path):
            return None
        try:
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
                if (
                    meta.get("schema") != GREENERY_SCHEMA_VERSION
                    or meta.get("stats") != list(self.stats)
                    or str(meta.get("crs")) != self.crs_key
                    or abs(float(meta.get("spacing", -1)) - self.spacing_m) > 1e-9
                ):
                    return None
                fgvi = tuple(int(r) for r in meta["gvi_radii"])
                fndvi = tuple(int(r) for r in meta["ndvi_radii"])
                # The stored ladder must cover the job's requested radii.
                if not (set(job_gvi) <= set(fgvi) and set(job_ndvi) <= set(fndvi)):
                    return None
                ids = np.ascontiguousarray(z["ids"], dtype=np.int64)
                veg = np.ascontiguousarray(z["veg"], dtype=np.float32)
                terrain = np.ascontiguousarray(z["terrain"], dtype=np.float32)
                ndvi = np.ascontiguousarray(z["ndvi"], dtype=np.float32)
        except Exception:
            return None
        unit = {
            "ids": ids,
            "veg": veg,
            "terrain": terrain,
            "ndvi": ndvi,
            "gvi_radii": fgvi,
            "ndvi_radii": fndvi,
            "dirty": False,
        }
        self._account(unit)
        return unit

    def commit_unit(
        self,
        cfg_key: str,
        new_ids: np.ndarray,
        veg: np.ndarray,
        terrain: np.ndarray,
        ndvi: np.ndarray,
        *,
        persist: bool = True,
    ) -> None:
        """Merge freshly-computed pixels into the unit, keep it sorted by id,
        and (optionally) write it to disk atomically."""
        unit = self._units[cfg_key]
        new_ids = np.asarray(new_ids, dtype=np.int64)
        if new_ids.size:
            unit["ids"] = np.concatenate([unit["ids"], new_ids])
            unit["veg"] = np.concatenate(
                [unit["veg"], np.asarray(veg, np.float32)], axis=0
            )
            unit["terrain"] = np.concatenate(
                [unit["terrain"], np.asarray(terrain, np.float32)], axis=0
            )
            unit["ndvi"] = np.concatenate(
                [unit["ndvi"], np.asarray(ndvi, np.float32)], axis=0
            )
            order = np.argsort(unit["ids"], kind="stable")
            unit["ids"] = np.ascontiguousarray(unit["ids"][order])
            for ch in self._CHANNELS:
                unit[ch] = np.ascontiguousarray(unit[ch][order])
            unit["dirty"] = True
        if persist and unit.get("dirty"):
            self._persist(cfg_key)
            unit["dirty"] = False
        self._account(unit)

    def _persist(self, cfg_key: str) -> None:
        unit = self._units[cfg_key]
        meta = {
            "schema": GREENERY_SCHEMA_VERSION,
            "gvi_radii": list(unit["gvi_radii"]),
            "ndvi_radii": list(unit["ndvi_radii"]),
            "stats": list(self.stats),
            "spacing": self.spacing_m,
            "crs": self.crs_key,
        }
        path = self._path(cfg_key)
        tmp = path + ".tmp"
        np.savez(
            tmp,
            ids=unit["ids"],
            veg=unit["veg"],
            terrain=unit["terrain"],
            ndvi=unit["ndvi"],
            meta=np.array(json.dumps(meta)),
        )
        os.replace(tmp + ".npz", path)

    # ------------------------------------------------------------------ lookup
    def lookup(
        self,
        entity_ids: np.ndarray,
        channel: str,
        radius: int,
        column: str,
        *,
        wave_index: int = DEFAULT_WAVE_INDEX,
    ) -> np.ndarray | None:
        """float32 values for ``entity_ids`` at one cell, or ``None`` if the
        cell is not stored (caller falls back). Missing ids come back NaN."""
        cfg = self._wave_unit.get(int(wave_index))
        unit = self._units.get(cfg) if cfg is not None else None
        if unit is None or channel not in self._CHANNELS:
            return None
        col_idx = self._stat_index.get(column)
        if col_idx is None:
            return None
        ladder = unit["ndvi_radii"] if channel == "ndvi" else unit["gvi_radii"]
        try:
            r_idx = ladder.index(int(radius))
        except ValueError:
            return None
        ids = unit["ids"]
        req = np.asarray(entity_ids).astype(np.int64, copy=False)
        out = np.full(req.shape[0], np.nan, dtype=np.float32)
        if not ids.size:
            return out
        pos = np.clip(np.searchsorted(ids, req), 0, ids.size - 1)
        valid = ids[pos] == req
        out[valid] = unit[channel][pos[valid], r_idx, col_idx]
        return out

    def resident_ids(self, wave_index: int) -> np.ndarray | None:
        cfg = self._wave_unit.get(int(wave_index))
        unit = self._units.get(cfg) if cfg is not None else None
        return None if unit is None else unit["ids"]

    # -------------------------------------------------------------- utilities
    @staticmethod
    def _setdiff(required: np.ndarray, present: np.ndarray) -> np.ndarray:
        if present.size == 0:
            return required
        return required[~np.isin(required, present)]

    def _account(self, unit: dict) -> None:
        self.n_bytes = sum(
            u["ids"].nbytes + sum(u[c].nbytes for c in self._CHANNELS)
            for u in self._units.values()
        )

    def close(self) -> None:
        self._units.clear()
        self._wave_unit.clear()
        self.n_bytes = 0
        gc.collect()


class PreAggregationCache:
    """Per-job pre-aggregation store: dense ``float32`` memmaps hold the stat
    values, a small SQLite database holds resume + meta, and lookup is an
    id-keyed gather into the memmap."""

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
        # Open value memmaps keyed by (channel, representative wave), plus the
        # fixed entity-id ↔ ordinal map that indexes their first axis.
        self._arrays: dict[tuple[str, int], np.memmap] = {}
        self._entity_ids: np.ndarray | None = None
        self._id_order: np.ndarray | None = None
        self._sorted_ids: np.ndarray | None = None
        self._n_entities: int = 0
        self._ensure_schema()
        self._restore_aliases_from_meta()
        self._load_entity_ids()

    # -- channel/radius helpers ------------------------------------------------
    def radii_for(self, channel: str) -> tuple[int, ...]:
        return self.ndvi_radii if channel == "ndvi" else self.gvi_radii

    # -- dense value storage (memmap) -----------------------------------------
    def _ids_path(self) -> str:
        return f"{self.db_path}.ids.npy"

    def _array_path(self, channel: str, rep: int) -> str:
        return f"{self.db_path}.{channel}.w{int(rep)}.f32"

    def _load_entity_ids(self) -> None:
        path = self._ids_path()
        if os.path.isfile(path):
            try:
                self._set_entity_ids(np.load(path))
            except Exception:
                pass

    def _set_entity_ids(self, ids: np.ndarray) -> None:
        ids = np.asarray(ids, dtype=np.int64)
        self._entity_ids = ids
        self._n_entities = int(ids.shape[0])
        # Sorted view + its permutation: a requested id searchsorts into the
        # sorted ids, and the permutation maps that position back to the row
        # ordinal in the dense array.
        self._id_order = np.argsort(ids, kind="stable")
        self._sorted_ids = ids[self._id_order]

    def _ords_for(self, req: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Dense-array row ordinals for ``req`` ids, plus an ``in-grid`` mask."""
        req = np.asarray(req, dtype=np.int64)
        if self._sorted_ids is None or self._sorted_ids.size == 0:
            return (
                np.zeros(req.shape[0], np.int64),
                np.zeros(req.shape[0], dtype=bool),
            )
        pos = np.searchsorted(self._sorted_ids, req)
        pos = np.clip(pos, 0, self._sorted_ids.size - 1)
        valid = self._sorted_ids[pos] == req
        assert self._id_order is not None
        return self._id_order[pos], valid

    def _get_array(
        self, channel: str, rep: int, *, create: bool = False
    ) -> "np.memmap | None":
        """The dense ``(n_entities, n_radii, n_stats)`` memmap for the cell.

        Returns ``None`` when the file does not exist and ``create`` is false
        (the cell was never written). A freshly created array is NaN-filled so
        an entity with no imagery in range reads back NaN, not 0.
        """
        key = (channel, int(rep))
        arr = self._arrays.get(key)
        if arr is not None:
            return arr
        path = self._array_path(channel, int(rep))
        shape = (self._n_entities, len(self.radii_for(channel)), _N_STATS)
        if os.path.isfile(path):
            arr = np.memmap(path, dtype=np.float32, mode="r+", shape=shape)
        elif create and self._n_entities > 0:
            arr = np.memmap(path, dtype=np.float32, mode="w+", shape=shape)
            arr[:] = np.nan
            arr.flush()
        else:
            return None
        self._arrays[key] = arr
        return arr

    def read_dense(
        self, channel: str, wave_index: int = DEFAULT_WAVE_INDEX
    ) -> "tuple[np.ndarray, np.memmap] | None":
        """``(entity_ids, values)`` for a whole ``(channel, wave)`` cell.

        ``entity_ids`` are in dense-array row order; ``values`` is the memmap
        itself (shape ``(n, n_radii, n_stats)``). ``None`` if never written.
        Used by :mod:`geofuse.preaggr_memory` to mirror the store into RAM.
        """
        rep = self.resolve_wave(channel, wave_index)
        arr = self._get_array(channel, rep)
        if arr is None or self._entity_ids is None:
            return None
        return self._entity_ids, arr

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
        # Only the bookkeeping lives in SQLite now — the stat values are in the
        # per-(channel, wave) memmaps.
        with self._lock:
            cur = self._conn
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

    def coverage_signature(self) -> str | None:
        """Grid+config signature identifying which (pixel, wave) ids this cache
        holds — used to reuse its coverage in the pre-aggregation gate. Set at
        build time; independent of the referenced-pixel subset, so it can be
        matched before the gate runs. ``None`` if never stored."""
        return self._get_meta("coverage_signature")

    def set_coverage_signature(self, signature: str) -> None:
        with self._lock:
            self._set_meta("coverage_signature", str(signature))
            self._conn.commit()

    def reset(self) -> None:
        """Drop all values + meta (used when the fingerprint changed)."""
        with self._lock:
            self._conn.execute("DELETE FROM done_channel_wave_entity")
            self._conn.execute("DELETE FROM meta")
            # Release and delete every dense memmap plus the ordinal map.
            for arr in self._arrays.values():
                try:
                    arr.flush()
                except Exception:
                    pass
            self._arrays.clear()
            for ch in CHANNELS:
                for rep in range(len(self.wave_labels)):
                    p = self._array_path(ch, rep)
                    if os.path.isfile(p):
                        try:
                            os.remove(p)
                        except OSError:
                            pass
            if os.path.isfile(self._ids_path()):
                try:
                    os.remove(self._ids_path())
                except OSError:
                    pass
            self._entity_ids = None
            self._id_order = None
            self._sorted_ids = None
            self._n_entities = 0
            # Restore identity alias map; aliases get re-registered by the
            # runner during the new build.
            self._wave_alias = {
                (ch, i): i for ch in CHANNELS for i in range(len(self.wave_labels))
            }

    def write_header(self, entity_ids: Any) -> None:
        """Record the build config and fix the entity id ↔ ordinal map.

        ``entity_ids`` is the full ordered set of grid entity ids the dense
        arrays are indexed by; every later write and lookup maps ids to rows
        through it. A resume keeps the order already on disk so existing rows
        stay addressable.
        """
        ids = np.asarray(entity_ids, dtype=np.int64)
        with self._lock:
            self._set_meta("schema_version", str(SCHEMA_VERSION))
            self._set_meta("fingerprint", self.fingerprint)
            self._set_meta("gvi_radii", json.dumps(list(self.gvi_radii)))
            self._set_meta("ndvi_radii", json.dumps(list(self.ndvi_radii)))
            self._set_meta("stat_columns", json.dumps(list(STAT_COLUMNS)))
            self._set_meta("percentiles", json.dumps(list(PERCENTILES)))
            self._set_meta("wave_labels", json.dumps(list(self.wave_labels)))
            self._set_meta("n_entities", str(int(ids.shape[0])))
            if self._get_meta("complete") is None:
                self._set_meta("complete", "0")
            # Keep any ordinal map already loaded from disk (resume); otherwise
            # fix it now from the passed ids and persist it.
            if self._entity_ids is None:
                self._set_entity_ids(ids)
                np.save(self._ids_path(), self._entity_ids)

    def mark_complete(self) -> None:
        with self._lock:
            self._set_meta("complete", "1")

    # -- wave aliases ----------------------------------------------------------
    def register_wave_aliases(self, channel: str, alias_map: dict[int, int]) -> None:
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
            raise ValueError(
                f"Unknown channel {channel!r}; expected one of {CHANNELS}."
            )
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
            raise ValueError(
                f"Unknown channel {channel!r}; expected one of {CHANNELS}."
            )
        if self._entity_ids is None:
            raise RuntimeError(
                "write_channel_wave_batch called before write_header fixed the "
                "entity ordinal map."
            )
        rep = self.resolve_wave(channel, wave_index)
        eids = np.asarray(entity_ids, dtype=np.int64)
        vals = np.asarray(stats, dtype=np.float32)
        ords, valid = self._ords_for(eids)
        with self._lock:
            arr = self._get_array(channel, rep, create=True)
            assert arr is not None
            if valid.any():
                arr[ords[valid]] = vals[valid]
            # Flush the values before recording completion, so a crash can
            # never leave an entity marked done with unwritten stats behind it.
            arr.flush()
            self._conn.execute("BEGIN")
            try:
                self._conn.executemany(
                    "INSERT OR IGNORE INTO done_channel_wave_entity "
                    "(channel, wave, entity_id) VALUES (?, ?, ?)",
                    [(channel, int(rep), int(e)) for e in eids[valid]],
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

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
        ``None`` when the ``(channel, wave)`` cell was never written (caller
        falls back). An in-grid entity with no imagery in range comes back as
        NaN; an id absent from the grid also comes back as NaN.
        """
        if channel not in CHANNELS or column not in STAT_COLUMNS:
            return None
        radius = int(radius)
        radii = self.radii_for(channel)
        if radius not in radii:
            return None
        rep = self.resolve_wave(channel, wave_index)
        with self._lock:
            arr = self._get_array(channel, rep)
        if arr is None:
            return None
        ridx = radii.index(radius)
        cidx = STAT_COLUMNS.index(column)

        req = np.asarray(entity_ids).astype(np.int64, copy=False)
        out = np.full(req.shape[0], np.nan, dtype=np.float32)
        if self._sorted_ids is None or not self._sorted_ids.size:
            return out
        ords, valid = self._ords_for(req)
        if valid.any():
            out[valid] = np.asarray(arr[ords[valid], ridx, cidx], dtype=np.float32)
        return out

    def close(self) -> None:
        for arr in self._arrays.values():
            try:
                arr.flush()
            except Exception:
                pass
        self._arrays.clear()
        try:
            self._conn.close()
        except Exception:
            pass
