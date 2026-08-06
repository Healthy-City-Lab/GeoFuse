"""Metric sampling and circular-neighbourhood aggregation for the fusion engine.

Pure, engine-state-free helpers that :class:`geofuse.fusion.MetricFusionEngine`
delegates its point sampling and ring/disk aggregation to:

* **Point sampling** — :func:`nearest_metric_join` (nearest vector value per
  point) and :func:`sample_raster_values` (vectorised nearest-pixel read).
* **Radius grid** — :func:`radius_int_bounds` aligns an integer radius search to
  the user's min/max/step.
* **Ring / disk aggregation** — :func:`precompute_raster_ring_values` /
  :func:`precompute_vector_ring_values` bin each point's neighbourhood into
  concentric annuli once, then :func:`aggregate_from_ring_cache` (via
  :func:`ring_end_index` + :func:`aggregate_disk_from_rings`) reduces any disk
  radius to a statistic without re-sampling.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol

from . import metric_columns


def nearest_metric_join_multi(
    points_gdf: gpd.GeoDataFrame,
    metric_gdf: gpd.GeoDataFrame,
    value_cols: list[str],
    max_distance_m: float,
) -> dict[str, pd.Series]:
    """Nearest-neighbour join returning several of the metric's columns at once.

    Works around two ``gpd.sjoin_nearest`` pitfalls:

    1. In EPSG:4326 ``max_distance`` is read in degrees, not metres.
    2. Equidistant right-hand features yield several rows per source index,
       which breaks the follow-up column assignment.

    This helper reprojects both sides to a common UTM CRS so ``max_distance_m``
    is honoured, then drops duplicate left-index rows by keeping the first
    match. Each returned series is reindexed onto ``points_gdf.index`` so a
    simple ``points_gdf[col] = result`` assignment is always safe.

    Several columns share one join because the join is the expensive half and
    it depends only on geometry: ``veg`` and ``terrain`` are two attribute
    columns of the same street-view points, so asking for them together costs
    what one of them used to.
    """
    metric_crs = points_gdf.estimate_utm_crs()
    pts_m = points_gdf[["geometry"]].to_crs(metric_crs)
    src_m = metric_gdf[["geometry", *value_cols]].to_crs(metric_crs)
    joined = gpd.sjoin_nearest(
        pts_m,
        src_m,
        how="left",
        max_distance=max_distance_m,
    )
    # Collapse ties: keep the first match per source row.
    joined = joined[~joined.index.duplicated(keep="first")]
    out: dict[str, pd.Series] = {}
    for value_col in value_cols:
        col = value_col if value_col in joined.columns else f"{value_col}_right"
        out[value_col] = joined[col].reindex(points_gdf.index)
    return out


def nearest_metric_join(
    points_gdf: gpd.GeoDataFrame,
    metric_gdf: gpd.GeoDataFrame,
    value_col: str,
    max_distance_m: float,
) -> pd.Series:
    """Nearest-neighbour join in a metric CRS, returning one value per source row.

    Single-column form of :func:`nearest_metric_join_multi`.
    """
    return nearest_metric_join_multi(
        points_gdf, metric_gdf, [value_col], max_distance_m
    )[value_col]


def sample_raster_values(
    points_in_raster_crs: gpd.GeoDataFrame, raster: dict
) -> np.ndarray:
    """Vectorised nearest-pixel sample of a raster dict at point geometries.

    ``points_in_raster_crs`` is already in ``raster["crs"]``. Returns a
    ``float32`` array aligned with the input rows; points whose pixel falls
    outside the raster (or on a nodata pixel) are ``NaN``. Handles both an
    in-memory ``ndarray`` / masked array and a window-backed
    :class:`~geofuse.raster_sampling.LazyRasterArray` (national-scale rasters
    kept on disk), replacing a per-row ``iterrows`` loop that does not scale to
    the millions of pixel-centroids a per-pixel CGI grid produces.
    """
    geoms = points_in_raster_crs.geometry
    if bool((geoms.geom_type == "Point").all()):
        xs = geoms.x.to_numpy()
        ys = geoms.y.to_numpy()
    else:
        centroids = geoms.centroid
        xs = centroids.x.to_numpy()
        ys = centroids.y.to_numpy()

    data = raster["data"]
    h, w = int(data.shape[0]), int(data.shape[1])
    rows, cols = rowcol(raster["transform"], xs, ys)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    out = np.full(len(geoms), np.nan, dtype=np.float32)

    valid = np.flatnonzero((rows >= 0) & (rows < h) & (cols >= 0) & (cols < w))
    if valid.size == 0:
        return out
    vr = rows[valid]
    vc = cols[valid]

    # In-memory array (incl. masked): one fancy-indexed read.
    if isinstance(data, np.ndarray):
        sampled = data[vr, vc]
        if np.ma.isMaskedArray(sampled):
            sampled = sampled.filled(np.nan)
        out[valid] = sampled
        return out

    # Window-backed (lazy) raster: read a single bounding window when it is
    # small enough to materialise, else sample scattered points from disk.
    rmin, rmax = int(vr.min()), int(vr.max()) + 1
    cmin, cmax = int(vc.min()), int(vc.max()) + 1
    if (rmax - rmin) * (cmax - cmin) <= 64_000_000:
        window = data[rmin:rmax, cmin:cmax]
        sampled = window[vr - rmin, vc - cmin]
        if np.ma.isMaskedArray(sampled):
            sampled = sampled.filled(np.nan)
        out[valid] = sampled
        return out

    band = getattr(data, "band", 1)
    nodata = getattr(data, "nodata", None)
    with rasterio.open(data.path) as src:
        sampled = np.fromiter(
            (
                rec[0]
                for rec in src.sample(
                    np.column_stack([xs[valid], ys[valid]]), indexes=band
                )
            ),
            dtype=np.float32,
            count=valid.size,
        )
    if nodata is not None and np.isfinite(nodata):
        sampled[sampled == np.float32(nodata)] = np.nan
    out[valid] = sampled
    return out


def radius_int_bounds(
    r_min: float, r_max: float, r_step: float
) -> tuple[int, int, int]:
    """Align Optuna integer radius search to user min/max/step (metres)."""
    lo = max(1, int(round(r_min)))
    hi = int(round(r_max))
    step = max(1, int(round(r_step)))
    if lo > hi:
        lo, hi = hi, lo
    span = hi - lo
    hi_adj = lo + (span // step) * step
    if hi_adj < lo:
        hi_adj = lo
    return lo, hi_adj, step


def polygonal_parts(geom) -> list:
    """Flatten a geometry into its constituent polygon parts."""
    gt = geom.geom_type
    if gt == "Polygon":
        return [geom]
    if gt == "MultiPolygon":
        return list(geom.geoms)
    if gt == "GeometryCollection":
        parts: list = []
        for g in geom.geoms:
            parts.extend(polygonal_parts(g))
        return parts
    return []


def ring_end_index(radii: np.ndarray, radius_m: float) -> int:
    """Index of the outermost annulus whose outer edge is ``<= radius_m``."""
    r = int(round(float(radius_m)))
    hits = np.flatnonzero(radii == r)
    if hits.size:
        return int(hits[-1])
    return int(np.searchsorted(radii, r, side="right") - 1)


def aggregate_disk_from_rings(
    ring_arrays: list[np.ndarray],
    end_ring_idx: int,
    stat: str,
    percentile: int,
) -> float:
    """Reduce the concatenated annuli ``[0..end_ring_idx]`` to a statistic."""
    if end_ring_idx < 0:
        return np.nan
    parts = ring_arrays[: end_ring_idx + 1]
    nonempty = [p for p in parts if p.size > 0]
    if not nonempty:
        return np.nan
    vals = np.concatenate(nonempty)
    if stat == "mean":
        return float(np.mean(vals))
    if stat == "median":
        return float(np.median(vals))
    if stat == "percentile":
        return float(np.percentile(vals, percentile))
    return np.nan


def vector_metric_column(metric_data: gpd.GeoDataFrame, channel: str) -> str:
    """Resolve the numeric value column of a vector metric for ``channel``.

    Honours the ``metric_column`` hint a loader may have recorded, otherwise
    resolves strictly against the channel's own column names.
    """
    metric_col = metric_data.attrs.get("metric_column")
    if metric_col and metric_col in metric_data.columns:
        return str(metric_col)
    return metric_columns.resolve_channel_column(metric_data.columns, channel)


def _bin_into_rings(
    ring_row: list[np.ndarray],
    dists_or_d2: np.ndarray,
    values: np.ndarray,
    radii_edges: np.ndarray,
    n_rings: int,
) -> None:
    """Scatter ``values`` into ``ring_row`` by annulus, one sorting pass.

    ``radii_edges`` are the ring outer edges in the same (possibly squared)
    units as ``dists_or_d2``; ``searchsorted(..., side="left")`` reproduces the
    ``inner < d <= outer`` annulus semantics (ring 0 is ``d <= outer``).
    """
    ring_idx = np.searchsorted(radii_edges, dists_or_d2, side="left")
    keep = ring_idx < n_rings
    if not keep.any():
        return
    v = values[keep]
    ridx = ring_idx[keep]
    finite = ~np.isnan(v)
    if not finite.all():
        v = v[finite]
        ridx = ridx[finite]
    if not v.size:
        return
    order = np.argsort(ridx, kind="stable")
    v = v[order]
    ridx = ridx[order]
    bounds = np.searchsorted(ridx, np.arange(n_rings + 1))
    for k in range(n_rings):
        lo, hi = int(bounds[k]), int(bounds[k + 1])
        if hi > lo:
            ring_row[k] = v[lo:hi]


def precompute_raster_ring_values(
    metric_dict: dict,
    points_gdf: gpd.GeoDataFrame,
    radii_m: np.ndarray,
) -> list[list[np.ndarray]]:
    """Bin each point's raster neighbourhood into concentric annuli (per radius).

    Footprints come from the same buffered discs the pre-aggregation cache
    reduces over: the point is buffered in a projected (metres) CRS, the buffer
    is reprojected into the raster's CRS, and one nested-ring rasterization
    assigns every pixel the index of the smallest disc that touches it. Annulus
    *i* is then the pixels whose index is exactly *i*, so concatenating rings
    ``0..i`` reproduces disc *i* exactly.
    """
    from . import preaggregation

    n_pts = len(points_gdf)
    n_rings = len(radii_m)
    _empty = np.array([], dtype=np.float64)
    ring_values: list[list[np.ndarray]] = [
        [_empty] * n_rings for _ in range(n_pts)
    ]
    if n_pts == 0 or n_rings == 0:
        return ring_values

    metric_array = metric_dict["data"]
    transform = metric_dict["transform"]
    metric_crs = metric_dict["crs"]

    # Buffer in a projected CRS so the radii are metres, then reproject the
    # buffers (never the raster) into the raster's own CRS.
    from .crs_utils import crs_uses_metre_axes, estimate_metre_projected_crs_for_gdf

    if points_gdf.crs is not None and crs_uses_metre_axes(points_gdf.crs):
        buffer_crs = points_gdf.crs
    else:
        buffer_crs = estimate_metre_projected_crs_for_gdf(points_gdf)
    pts_buf_crs = points_gdf.to_crs(buffer_crs)
    if str(buffer_crs) != str(metric_crs):
        from pyproj import Transformer as _Transformer

        xy_fn = _Transformer.from_crs(buffer_crs, metric_crs, always_xy=True).transform
    else:
        xy_fn = None

    radii_list = [float(r) for r in np.asarray(radii_m, dtype=np.float64)]
    templates = preaggregation.origin_circle_templates(radii_list)

    from rasterio.transform import rowcol as _rowcol
    from rasterio.windows import Window as _Window
    from rasterio.windows import transform as _window_transform

    h, w = int(metric_array.shape[0]), int(metric_array.shape[1])
    for pos, geom in enumerate(pts_buf_crs.geometry.to_numpy()):
        if geom is None or geom.is_empty:
            continue
        discs = [preaggregation.buffer_at(geom, r, templates) for r in radii_list]
        discs = preaggregation.reproject_geoms(discs, xy_fn)
        largest = discs[-1]
        if largest is None or largest.is_empty:
            continue
        minx, miny, maxx, maxy = largest.bounds
        r1, c1 = _rowcol(transform, minx, maxy)
        r2, c2 = _rowcol(transform, maxx, miny)
        rmin = max(0, min(int(r1), int(r2)))
        rmax = min(h, max(int(r1), int(r2)) + 1)
        cmin = max(0, min(int(c1), int(c2)))
        cmax = min(w, max(int(c1), int(c2)) + 1)
        if rmin >= rmax or cmin >= cmax:
            continue

        window = metric_array[rmin:rmax, cmin:cmax]
        data = np.asarray(np.ma.getdata(window), dtype=np.float64)
        base_valid = ~np.ma.getmaskarray(window) & ~np.isnan(data)
        if not base_valid.any():
            continue
        win_transform = _window_transform(
            _Window(cmin, rmin, cmax - cmin, rmax - rmin), transform
        )
        ring = preaggregation.ring_index_grid(discs, window.shape, win_transform)
        if ring is None:
            continue
        idx = ring[base_valid]
        vals = data[base_valid]
        order = np.argsort(idx, kind="stable")
        idx_sorted = idx[order]
        vals_sorted = vals[order]
        bounds = np.searchsorted(idx_sorted, np.arange(n_rings + 1), side="left")
        row_out = ring_values[pos]
        for k in range(n_rings):
            lo, hi = int(bounds[k]), int(bounds[k + 1])
            if hi > lo:
                row_out[k] = vals_sorted[lo:hi]

    return ring_values


def precompute_vector_ring_values(
    metric_data: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame,
    radii_m: np.ndarray,
    metric_col: str,
) -> list[list[np.ndarray]]:
    """Bin each point's vector neighbourhood into concentric annuli (per radius).

    One distance query per point at the maximum radius — a ``BallTree`` radius
    query for point features, an r-tree candidate query + geometry distance
    otherwise — binned into annuli by ``searchsorted``, replacing the
    per-(point, ring) buffer + full spatial join of the previous
    implementation. Each feature lands in the annulus of its (nearest-point)
    distance, so a feature whose footprint spans several annuli contributes
    once rather than once per ring it touches.
    """
    n_pts = len(points_gdf)
    n_rings = len(radii_m)
    _empty = np.array([], dtype=np.float64)
    ring_values: list[list[np.ndarray]] = [
        [_empty] * n_rings for _ in range(n_pts)
    ]
    if n_pts == 0 or len(metric_data) == 0:
        return ring_values

    points_wgs84 = points_gdf.to_crs("EPSG:4326")
    centroid = points_wgs84.geometry.union_all().centroid
    lon, lat = centroid.x, centroid.y
    utm_zone = int((lon + 180) / 6) + 1
    utm_crs = f"EPSG:326{utm_zone:02d}" if lat >= 0 else f"EPSG:327{utm_zone:02d}"

    points_utm = points_gdf.to_crs(utm_crs)
    metric_utm = metric_data.to_crs(utm_crs)
    if metric_col in metric_utm.columns:
        metric_utm = metric_utm[metric_utm[metric_col].notna()]
    if len(metric_utm) == 0 or metric_col not in metric_utm.columns:
        return ring_values
    values = metric_utm[metric_col].to_numpy(dtype=np.float64)

    radii_arr = np.asarray(radii_m, dtype=np.float64)
    max_r = float(radii_arr[-1])

    if bool((metric_utm.geometry.geom_type == "Point").all()):
        from sklearn.neighbors import BallTree

        m_xy = np.column_stack(
            [metric_utm.geometry.x.to_numpy(), metric_utm.geometry.y.to_numpy()]
        )
        pt_xy = np.column_stack(
            [points_utm.geometry.x.to_numpy(), points_utm.geometry.y.to_numpy()]
        )
        tree = BallTree(m_xy)
        idx_arr, dist_arr = tree.query_radius(pt_xy, r=max_r, return_distance=True)
        for pos in range(n_pts):
            neigh = idx_arr[pos]
            if len(neigh):
                _bin_into_rings(
                    ring_values[pos],
                    dist_arr[pos],
                    values[neigh],
                    radii_arr,
                    n_rings,
                )
    else:
        sindex = metric_utm.sindex
        geoms = metric_utm.geometry
        pt_geoms = points_utm.geometry.to_numpy()
        for pos in range(n_pts):
            pt = pt_geoms[pos]
            try:
                hits = np.asarray(
                    sindex.query(pt.buffer(max_r), predicate="intersects"),
                    dtype=np.int64,
                )
            except TypeError:
                hits = np.asarray(
                    list(sindex.intersection(pt.buffer(max_r).bounds)),
                    dtype=np.int64,
                )
            if not len(hits):
                continue
            dists = geoms.iloc[hits].distance(pt).to_numpy(dtype=np.float64)
            _bin_into_rings(ring_values[pos], dists, values[hits], radii_arr, n_rings)

    return ring_values


def ring_cache_key(
    channel: str,
    fold_idx: int,
    subset: str,
    points_gdf: gpd.GeoDataFrame,
    radii: np.ndarray,
) -> tuple:
    """Cache key for a precomputed ring set (channel + fold subset + rows + radii)."""
    return (
        channel,
        fold_idx,
        subset,
        tuple(points_gdf.index),
        radii.tobytes(),
    )


def ring_prefix_stats(
    ring_rows: list[list[np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative per-ring ``(sums, counts)`` for the vectorized mean path.

    Ring arrays are NaN-free by construction, so a disk mean at any end ring
    is ``sums[:, end] / counts[:, end]`` — two array reads instead of a
    per-point concatenate + reduce.
    """
    n_pts = len(ring_rows)
    n_rings = len(ring_rows[0]) if n_pts else 0
    sums = np.zeros((n_pts, n_rings), dtype=np.float64)
    counts = np.zeros((n_pts, n_rings), dtype=np.int64)
    for i, rings in enumerate(ring_rows):
        for k, arr in enumerate(rings):
            if arr.size:
                sums[i, k] = float(arr.sum())
                counts[i, k] = arr.size
    np.cumsum(sums, axis=1, out=sums)
    np.cumsum(counts, axis=1, out=counts)
    return sums, counts


def aggregate_from_ring_cache(
    radii: np.ndarray,
    ring_rows: list[list[np.ndarray]],
    radius_m: float,
    stat: str,
    percentile: int,
    *,
    prefix: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Reduce every point's cached annuli to a per-point disk statistic.

    ``prefix`` (from :func:`ring_prefix_stats`) makes the ``mean`` stat a
    vectorized division; median / percentile keep the per-point reduce.
    """
    n = len(ring_rows)
    out = np.full(n, np.nan, dtype=np.float64)
    end_idx = ring_end_index(radii, radius_m)
    if end_idx < 0:
        return out
    if stat == "mean" and prefix is not None:
        sums, counts = prefix
        c = counts[:, end_idx]
        nz = c > 0
        out[nz] = sums[nz, end_idx] / c[nz]
        return out
    for pos in range(n):
        out[pos] = aggregate_disk_from_rings(ring_rows[pos], end_idx, stat, percentile)
    return out
