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

# Version 2 added the ``wave`` PK column and per-(channel, wave) completion
# tracking. Version 3 moved raster aggregation onto buffered discs reprojected
# into the raster CRS for every entity type; stored raster stats from earlier
# versions came from pixel-grid distance math and are not comparable, so they
# are reset rather than reused. Caches built under an older schema fail the
# fingerprint check and are rebuilt on the next run.
SCHEMA_VERSION = 3

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
# Bounds on the in-memory column cache (each entry is one id + value column for
# all entities at one (channel, wave, radius, stat)). The entry count keeps
# repeated trial lookups cheap; the byte ceiling is what actually binds on a
# large per-pixel grid, where a single column runs to megabytes.
_COLUMN_CACHE_MAX = 1024
_COLUMN_CACHE_MAX_BYTES = 256 * 1024 * 1024


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
        self._col_cache_bytes: int = 0
        # Memoized searchsorted positions per (requested-ids, channel, wave):
        # the fusion optimizer re-reads the same fold row set for every trial
        # and every (radius, stat) column of a channel wave shares one sorted
        # entity-id vector, so the indexer is computed once per fold subset.
        self._indexer_cache: OrderedDict[tuple, Any] = OrderedDict()
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
            self._col_cache_bytes = 0
            self._indexer_cache.clear()
            # Restore identity alias map; aliases get re-registered by the
            # runner during the new build.
            self._wave_alias = {
                (ch, i): i for ch in CHANNELS for i in range(len(self.wave_labels))
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
            self._col_cache.clear()
            self._col_cache_bytes = 0

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
            # Invalidate any cached column reads / row indexers for this
            # (channel, wave) — a write changes the stored entity-id set.
            for key in list(self._col_cache.keys()):
                if key[0] == channel and key[1] == rep:
                    dropped = self._col_cache.pop(key, None)
                    if dropped is not None:
                        self._col_cache_bytes -= int(dropped[0].nbytes) + int(
                            dropped[1].nbytes
                        )
            for key in list(self._indexer_cache.keys()):
                if key[0] == channel and key[1] == rep:
                    self._indexer_cache.pop(key, None)

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
                self._col_cache_bytes += int(ids.nbytes) + int(vals.nbytes)
                while self._col_cache and (
                    len(self._col_cache) > _COLUMN_CACHE_MAX
                    or self._col_cache_bytes > _COLUMN_CACHE_MAX_BYTES
                ):
                    _, evicted = self._col_cache.popitem(last=False)
                    self._col_cache_bytes -= int(evicted[0].nbytes) + int(
                        evicted[1].nbytes
                    )
            else:
                self._col_cache.move_to_end(key)
            sorted_ids, sorted_vals = entry

        req = np.asarray(entity_ids).astype(np.int64, copy=False)
        out = np.full(req.shape[0], np.nan, dtype=np.float32)
        if not sorted_ids.size:
            return out

        # Row indexer: positions of ``req`` inside the channel wave's sorted
        # entity ids. Callers re-read one fold's id vector for every trial and
        # every (radius, stat) cell, so the entry is keyed on that array's
        # identity and holds a reference to it — pinning the object means the
        # id cannot be reused by a different array while the entry lives, and
        # the stored array is re-checked on every hit.
        ikey = (channel, rep, id(req))
        indexer = None
        with self._lock:
            entry_idx = self._indexer_cache.get(ikey)
            if entry_idx is not None:
                held_req, held_ids, pos_clip, valid = entry_idx
                if held_req is req and held_ids is sorted_ids:
                    self._indexer_cache.move_to_end(ikey)
                    indexer = (pos_clip, valid)
        if indexer is None:
            pos = np.searchsorted(sorted_ids, req)
            pos_clip = np.clip(pos, 0, sorted_ids.size - 1)
            valid = sorted_ids[pos_clip] == req
            indexer = (pos_clip, valid)
            with self._lock:
                self._indexer_cache[ikey] = (req, sorted_ids, pos_clip, valid)
                while len(self._indexer_cache) > 64:
                    self._indexer_cache.popitem(last=False)
        pos_clip, valid = indexer
        out[valid] = sorted_vals[pos_clip[valid]]
        return out

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
