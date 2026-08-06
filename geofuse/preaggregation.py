"""Reusable greenery pre-aggregation for the fusion optimizer.

Before optimization, every sample pixel's metric values are aggregated across
every buffer radius in the ladder and every aggregation statistic (mean plus a
compact percentile set) for each channel (``veg``, ``terrain``, ``ndvi``), so
each Optuna trial reads a stored value instead of recomputing circular-buffer
aggregations — and the work survives cancels/crashes and is reused by later
runs with the same inputs.

The store is :class:`GreeneryCache`: one compact ``.npz`` per greenery-file set
in ``<cache_dir>/greenery/``, keyed on ``(greenery-file identity, grid spacing,
metric CRS, stat set)`` — *not* the target — so a cache is reused across jobs
that change only the outcome, covariates, or target file. Pixels are keyed by a
stable global id (grid row/col), the radius ladder lives inside the file so a
wider cache serves a narrower job, and the whole unit loads into RAM once for
lock-free per-trial gathers. Longitudinal and cross-sectional runs share the
store and this module's aggregation functions; only the modelling differs.

The aggregation math is format-aware: vector (point) metrics use a ``BallTree``
radius query; raster metrics use per-entity windowed reads with a circular mask.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from typing import Any

import numpy as np

# Bumped when the on-disk greenery-cache layout changes so stale files are
# ignored rather than mis-read.
GREENERY_SCHEMA_VERSION = 1

# Cross-sectional callers use this single implicit wave; longitudinal callers
# pass an explicit ordered tuple of wave labels to the constructor.
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


# The percentile grid as fractions, and the interpolation ``np.percentile``
# applies between the two order statistics a fractional rank falls between.
# Both are reproduced here rather than called through ``np.percentile`` because
# that function's fixed per-call cost — argument validation, ``_ureduce``, and
# a ``np.unique`` over its own index list — dominated the pre-aggregation build:
# it is invoked once per (entity, radius, channel), tens of millions of times,
# on samples of a few thousand values each. Selecting the order statistics
# directly is exact against ``np.percentile`` and measured ~2x faster.
_PCT_FRACTIONS = np.asarray(PERCENTILES, dtype=np.float64) / 100.0


def _percentile_ranks(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(lower, upper, weight)`` — the ranks each percentile sits between."""
    virtual = _PCT_FRACTIONS * (n - 1)
    lower = virtual.astype(np.intp)
    return lower, np.minimum(lower + 1, n - 1), virtual - lower


def _interpolate(lower: np.ndarray, upper: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Blend two order statistics the way ``np.percentile``'s default does."""
    span = upper - lower
    return np.where(
        weight < 0.5, lower + span * weight, upper - span * (1.0 - weight)
    )


def compute_all_stats(values: np.ndarray) -> np.ndarray:
    """Return ``[mean, p10, p25, p50, p75, p90]`` as float32, rounded to four
    decimals; all-NaN if empty."""
    n = values.size
    if n == 0:
        return np.full(_N_STATS, np.nan, dtype=np.float32)
    out = np.empty(_N_STATS, dtype=np.float32)
    out[0] = float(values.mean())
    if n == 1:
        out[1:] = values[0]
    else:
        lower, upper, weight = _percentile_ranks(n)
        # Partitioning about the extremes as well as the wanted ranks puts the
        # maximum last, so one look there settles whether the sample holds a
        # NaN — the same test, and the same all-NaN answer, as ``np.percentile``.
        ranked = np.partition(values, np.concatenate(([0, -1], lower, upper)))
        out[1:] = (
            np.nan
            if np.isnan(ranked[-1])
            else _interpolate(ranked[lower], ranked[upper], weight)
        )
    return np.round(out, _STORE_DECIMALS)


def prefix_stats(values: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Stats for every nested prefix of ``values``: ``[len(ends), n_stats]``.

    ``ends`` is ascending and prefix *i* is ``values[:ends[i]]`` — the shape a
    radius ladder always takes once a neighbourhood is ordered by distance (or
    by ring), since a smaller disc's members are a prefix of a larger one's.
    Two consequences of that nesting are used here instead of reducing each
    radius from scratch:

    * every prefix mean is a slice of one cumulative sum, and
    * a prefix's sorted form is the previous prefix's sorted form merged with
      the block the radius added, so a stable sort over the concatenation
      absorbs the new block in a linear pass rather than re-sorting the whole.

    Values must be finite; the callers filter NaN out when they build the
    neighbourhood. (A NaN would sort last and leave the lower percentiles
    reading as if it weren't there.)
    """
    ends = np.asarray(ends, dtype=np.intp)
    out = np.full((ends.size, _N_STATS), np.nan, dtype=np.float32)
    filled = ends > 0
    if values.size == 0 or not filled.any():
        return out

    running_total = np.cumsum(values, dtype=np.float64)
    out[filled, 0] = running_total[ends[filled] - 1] / ends[filled]

    ranked = np.empty(0, dtype=values.dtype)
    grown = 0
    for i in range(ends.size):
        n = int(ends[i])
        if n == 0:
            continue
        if n > grown:
            block = np.sort(values[grown:n])
            ranked = (
                block
                if ranked.size == 0
                else np.sort(np.concatenate((ranked, block)), kind="stable")
            )
            grown = n
        if n == 1:
            out[i, 1:] = ranked[0]
            continue
        lower, upper, weight = _percentile_ranks(n)
        out[i, 1:] = _interpolate(ranked[lower], ranked[upper], weight)
    return np.round(out, _STORE_DECIMALS, out=out)


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


# ────────────────────────────────────────────────────────────────────
# Buffered-disc construction and raster reduction
# ────────────────────────────────────────────────────────────────────
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
    if values.size == 0:
        return np.full((n_radii, _N_STATS), np.nan, dtype=np.float32)
    order = np.argsort(ring_index, kind="stable")
    rings_sorted = ring_index[order]
    ends = np.searchsorted(rings_sorted, np.arange(n_radii), side="right")
    return prefix_stats(values[order], ends)


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


# ────────────────────────────────────────────────────────────────────
# Format-aware per-batch aggregation
# ────────────────────────────────────────────────────────────────────


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
            out[c][b] = prefix_stats(values_by_col[c][neigh_sorted], ends)
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
        ends = np.searchsorted(d_all[order], radii_arr, side="right")
        out[b] = prefix_stats(values[neigh][order], ends)
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


# ────────────────────────────────────────────────────────────────────
# Batch aggregation across all three channels
# ────────────────────────────────────────────────────────────────────
#
# One batch of entities reduced against veg, terrain and NDVI. The channels are
# described by an *aggregator* — a small tuple naming how that channel is
# sampled and holding whatever it is sampled through:
#
#   ("point",  BallTree, values)                 a point metric, entity is a point
#   ("geom",   gdf, value_column, cell_buffer)   a point metric, entity is a shape
#   ("raster", array, transform, xy_fn)          a raster metric, any entity
#
# ``aggregate_entity_batch`` is the single implementation of a batch, shared by
# the in-process path and by the pool workers below, so the two can never drift.


def channel_batch_stats(
    aggregator: tuple,
    radii_m: tuple[int, ...],
    *,
    point_xy: np.ndarray | None = None,
    geoms: list | None = None,
) -> np.ndarray:
    """Reduce one batch against one channel: ``[n_entities, n_radii, n_stats]``."""
    kind = aggregator[0]
    if kind == "point":
        return vector_batch_stats(aggregator[1], aggregator[2], point_xy, radii_m)
    if kind == "geom":
        return vector_batch_geometry_stats(
            aggregator[1], aggregator[2], geoms, radii_m, cell_buffer_m=aggregator[3]
        )
    return raster_batch_geometry_stats(
        aggregator[1], aggregator[2], geoms, radii_m, to_raster_crs=aggregator[3]
    )


def aggregate_entity_batch(
    state: dict,
    *,
    point_xy: np.ndarray | None = None,
    geoms: list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Aggregate one batch of entities across all three channels.

    ``state`` carries the resolved aggregators: ``veg`` / ``terrain`` / ``ndvi``
    plus the two radius ladders, and optionally ``gvi_shared`` — a
    ``(BallTree, {"veg": values, "terrain": values})`` pair used when veg and
    terrain are two attribute columns of one point layer, so a single radius
    query serves both instead of walking the same neighbourhood twice.

    Returns the three stat arrays plus the seconds spent on the GVI channels
    and on NDVI, which the caller accumulates for the parallel-efficiency log.
    """
    started = time.perf_counter()
    shared = state.get("gvi_shared")
    gvi_radii = state["gvi_radii"]
    if shared is not None:
        both = vector_batch_stats_multi(shared[0], shared[1], point_xy, gvi_radii)
        veg, terrain = both["veg"], both["terrain"]
    else:
        veg = channel_batch_stats(
            state["veg"], gvi_radii, point_xy=point_xy, geoms=geoms
        )
        terrain = channel_batch_stats(
            state["terrain"], gvi_radii, point_xy=point_xy, geoms=geoms
        )
    gvi_done = time.perf_counter()
    ndvi = channel_batch_stats(
        state["ndvi"], state["ndvi_radii"], point_xy=point_xy, geoms=geoms
    )
    return veg, terrain, ndvi, gvi_done - started, time.perf_counter() - gvi_done


# ────────────────────────────────────────────────────────────────────
# Process-pool workers
# ────────────────────────────────────────────────────────────────────
#
# Aggregation is a long chain of small numpy calls per entity, so it is bound by
# the GIL rather than by the CPU or the disk and a thread pool cannot speed it
# up (see :mod:`geofuse.parallel`). These workers run the same batches in their
# own interpreters instead.
#
# Nothing bulky crosses the pickle channel. The parent publishes the entity
# coordinates and each point metric's coordinates and values as ``.npy`` files;
# every worker memory-maps them, so one copy of the pages serves the whole pool.
# Raster channels carry only a path, which the worker opens for itself. A task
# is a pair of indices and its result is the reduced stats for that slice.

# This worker's resident state, built once by :func:`worker_init`.
_WORKER_STATE: dict = {}


def _transformer(from_crs: str | None, to_crs: str | None) -> Any:
    """The (x, y) reprojection between two CRS descriptions, or ``None``."""
    if not from_crs or not to_crs or from_crs == to_crs:
        return None
    from pyproj import Transformer

    return Transformer.from_crs(from_crs, to_crs, always_xy=True).transform


def _build_aggregator(spec: dict, shared_xy: Any) -> tuple:
    """Rebuild one channel's aggregator in a worker from its description."""
    from .parallel import attach_array

    if spec["kind"] == "raster":
        from .raster_sampling import LazyRasterArray

        array = LazyRasterArray(spec["path"], band=spec.get("band", 1))
        return (
            "raster",
            array,
            spec["transform"],
            _transformer(spec.get("from_crs"), spec.get("to_crs")),
        )
    if spec["kind"] == "raster_array":
        return (
            "raster",
            attach_array(spec["path"]),
            spec["transform"],
            _transformer(spec.get("from_crs"), spec.get("to_crs")),
        )
    from sklearn.neighbors import BallTree

    xy = shared_xy if spec.get("shared") else np.asarray(attach_array(spec["xy"]))
    return ("point", BallTree(xy), attach_array(spec["values"]))


def worker_init(spec: dict) -> None:
    """Pool initializer: attach the shared arrays and build this worker's state.

    Runs once per worker. The BallTree over a point metric is rebuilt here
    rather than pickled in — it is derived data, and rebuilding it from mapped
    coordinates measured cheaper than shipping a copy to every worker.
    """
    from sklearn.neighbors import BallTree

    from .parallel import attach_array

    _WORKER_STATE.clear()
    state: dict = {
        "entity_xy": attach_array(spec["entity_xy"]),
        "gvi_radii": tuple(spec["gvi_radii"]),
        "ndvi_radii": tuple(spec["ndvi_radii"]),
    }
    shared_xy = None
    if spec.get("gvi_shared_xy"):
        shared_xy = np.asarray(attach_array(spec["gvi_shared_xy"]))
        state["gvi_shared"] = (
            BallTree(shared_xy),
            {
                "veg": attach_array(spec["channels"]["veg"]["values"]),
                "terrain": attach_array(spec["channels"]["terrain"]["values"]),
            },
        )
    for name, chan in spec["channels"].items():
        if shared_xy is not None and name in ("veg", "terrain"):
            continue
        state[name] = _build_aggregator(chan, shared_xy)
    _WORKER_STATE.update(state)


def worker_release() -> None:
    """Drop this process's worker state, releasing the mapped scratch files.

    Windows refuses to delete a file that is still mapped or open, so the
    raster handles are closed explicitly rather than left to collection.
    """
    for value in _WORKER_STATE.values():
        for part in value if isinstance(value, tuple) else ():
            close_all = getattr(part, "close_all", None)
            if close_all is not None:
                close_all()
    _WORKER_STATE.clear()
    gc.collect()


def worker_aggregate(lo: int, hi: int) -> tuple:
    """Aggregate the entity slice ``[lo, hi)`` with this worker's state."""
    import shapely

    state = _WORKER_STATE
    xy = np.asarray(state["entity_xy"][lo:hi], dtype=np.float64)
    # Raster channels reduce over buffered geometry; the entities in this path
    # are grid-pixel centroids, so their points are rebuilt from the mapped
    # coordinates instead of travelling with the task.
    geoms = list(shapely.points(xy))
    veg, terrain, ndvi, gvi_s, ndvi_s = aggregate_entity_batch(
        state, point_xy=xy, geoms=geoms
    )
    return lo, hi, veg, terrain, ndvi, gvi_s, ndvi_s


# ────────────────────────────────────────────────────────────────────
# SQLite cache
# ────────────────────────────────────────────────────────────────────


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
        job_gvi = tuple(int(r) for r in gvi_radii)
        job_ndvi = tuple(int(r) for r in ndvi_radii)
        required = np.asarray(required_ids, dtype=np.int64)

        if cfg_key not in self._units:
            stored = self._peek_ladder(cfg_key)
            if (
                stored is not None
                and set(job_gvi) <= set(stored[0])
                and set(job_ndvi) <= set(stored[1])
            ):
                # The file's ladder already covers the requested radii — reuse
                # it (radii-subset reuse) and load its pixels into RAM.
                self._units[cfg_key] = self._load(cfg_key)
            else:
                # No file, or a wider radius is needed: build at the union of
                # the stored and requested ladders so a file's ladder only ever
                # grows and later subset jobs keep reusing it.
                if stored is not None:
                    eff_gvi = tuple(sorted(set(job_gvi) | set(stored[0])))
                    eff_ndvi = tuple(sorted(set(job_ndvi) | set(stored[1])))
                else:
                    eff_gvi, eff_ndvi = tuple(sorted(job_gvi)), tuple(sorted(job_ndvi))
                self._units[cfg_key] = {
                    "ids": np.empty(0, np.int64),
                    "veg": np.empty((0, len(eff_gvi), len(self.stats)), np.float32),
                    "terrain": np.empty(
                        (0, len(eff_gvi), len(self.stats)), np.float32
                    ),
                    "ndvi": np.empty((0, len(eff_ndvi), len(self.stats)), np.float32),
                    "gvi_radii": eff_gvi,
                    "ndvi_radii": eff_ndvi,
                    "dirty": False,
                }
        unit = self._units[cfg_key]
        missing = self._setdiff(required, unit["ids"])
        return unit["gvi_radii"], unit["ndvi_radii"], missing

    def _config_ok(self, meta: dict) -> bool:
        return (
            meta.get("schema") == GREENERY_SCHEMA_VERSION
            and meta.get("stats") == list(self.stats)
            and str(meta.get("crs")) == self.crs_key
            and abs(float(meta.get("spacing", -1)) - self.spacing_m) <= 1e-9
        )

    def _peek_ladder(
        self, cfg_key: str
    ) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        """The stored ``(gvi_radii, ndvi_radii)`` of a matching file, cheaply.

        Reads only the metadata entry of the ``.npz`` (not the arrays), so the
        reuse decision — does this file's ladder cover the job? — is made before
        loading anything. ``None`` when no compatible file exists.
        """
        path = self._path(cfg_key)
        if not os.path.isfile(path):
            return None
        try:
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
            if not self._config_ok(meta):
                return None
            return (
                tuple(int(r) for r in meta["gvi_radii"]),
                tuple(int(r) for r in meta["ndvi_radii"]),
            )
        except Exception:
            return None

    def _load(self, cfg_key: str) -> dict:
        with np.load(self._path(cfg_key), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            unit = {
                "ids": np.ascontiguousarray(z["ids"], dtype=np.int64),
                "veg": np.ascontiguousarray(z["veg"], dtype=np.float32),
                "terrain": np.ascontiguousarray(z["terrain"], dtype=np.float32),
                "ndvi": np.ascontiguousarray(z["ndvi"], dtype=np.float32),
                "gvi_radii": tuple(int(r) for r in meta["gvi_radii"]),
                "ndvi_radii": tuple(int(r) for r in meta["ndvi_radii"]),
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
