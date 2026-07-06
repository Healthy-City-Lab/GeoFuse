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


def nearest_metric_join(
    points_gdf: gpd.GeoDataFrame,
    metric_gdf: gpd.GeoDataFrame,
    value_col: str,
    max_distance_m: float,
) -> pd.Series:
    """Nearest-neighbour join in a metric CRS, returning one value per source row.

    ``gpd.sjoin_nearest`` has two pitfalls that produced
    ``ValueError: cannot reindex on an axis with duplicate labels`` in fusion:

    1. Run in EPSG:4326 (degrees), the ``max_distance`` parameter is interpreted
       in *degrees* rather than metres — effectively unbounded.
    2. When multiple right-hand features are exactly equidistant from a source
       point, the join returns *multiple rows for the same source index*. The
       subsequent column assignment then fails because pandas cannot reindex
       onto a duplicated axis.

    This helper reprojects both sides to a common UTM CRS so ``max_distance_m``
    is honoured, then drops duplicate left-index rows by keeping the first
    match. The returned series is reindexed onto ``points_gdf.index`` so a
    simple ``points_gdf[col] = result`` assignment is always safe.
    """
    metric_crs = points_gdf.estimate_utm_crs()
    pts_m = points_gdf[["geometry"]].to_crs(metric_crs)
    src_m = metric_gdf[["geometry", value_col]].to_crs(metric_crs)
    joined = gpd.sjoin_nearest(
        pts_m,
        src_m,
        how="left",
        max_distance=max_distance_m,
    )
    # Collapse ties: keep the first match per source row.
    joined = joined[~joined.index.duplicated(keep="first")]
    col = value_col if value_col in joined.columns else f"{value_col}_right"
    return joined[col].reindex(points_gdf.index)


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


def radius_int_bounds(r_min: float, r_max: float, r_step: float) -> tuple[int, int, int]:
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
    """Resolve the numeric value column of a vector metric for ``channel``."""
    metric_col = metric_data.attrs.get("metric_column")
    if metric_col and metric_col in metric_data.columns:
        return metric_col
    if channel == "veg":
        for col in ["veg", "gvi_veg", "gvi", "GVI", "value"]:
            if col in metric_data.columns:
                return col
    elif channel == "terrain":
        for col in ["terrain", "gvi_ter", "NDVI", "ndvi", "value"]:
            if col in metric_data.columns:
                return col
    else:
        for col in ["NDVI", "ndvi", "value"]:
            if col in metric_data.columns:
                return col
    numeric_cols = metric_data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [
        c for c in numeric_cols if c not in ["index_right", "index_left", "index"]
    ]
    if numeric_cols:
        return numeric_cols[0]
    raise ValueError(f"No numeric metric column for channel={channel}")


def precompute_raster_ring_values(
    metric_dict: dict,
    points_gdf: gpd.GeoDataFrame,
    radii_m: np.ndarray,
) -> list[list[np.ndarray]]:
    """Bin each point's raster neighbourhood into concentric annuli (per radius)."""
    n_pts = len(points_gdf)
    n_rings = len(radii_m)
    ring_values: list[list[np.ndarray]] = [
        [np.array([], dtype=np.float64) for _ in range(n_rings)] for _ in range(n_pts)
    ]

    metric_array = metric_dict["data"]
    transform = metric_dict["transform"]
    metric_crs = metric_dict["crs"]
    points_metric_crs = points_gdf.to_crs(metric_crs)

    pixel_size = abs(transform.a)
    if metric_crs.is_geographic:
        pixel_size_meters = pixel_size * 111320
    else:
        pixel_size_meters = pixel_size

    max_r_m = float(radii_m[-1])
    max_r_px = int(max_r_m / pixel_size_meters)
    max_r_px = max(1, min(max_r_px, 10000))

    idx_to_pos = {idx: pos for pos, idx in enumerate(points_gdf.index)}

    # Positional loop over pre-extracted coordinates (skips iterrows' per-row
    # Series construction).
    _xs = points_metric_crs.geometry.x.to_numpy()
    _ys = points_metric_crs.geometry.y.to_numpy()
    for idx, _px, _py in zip(points_metric_crs.index, _xs, _ys):
        pos = idx_to_pos[idx]
        row, col = rowcol(transform, _px, _py)

        rmin = max(row - max_r_px, 0)
        rmax = min(row + max_r_px + 1, metric_array.shape[0])
        cmin = max(col - max_r_px, 0)
        cmax = min(col + max_r_px + 1, metric_array.shape[1])
        if rmin >= rmax or cmin >= cmax:
            continue

        window = metric_array[rmin:rmax, cmin:cmax]
        rr = np.arange(rmin, rmax, dtype=np.float64)[:, None]
        cc = np.arange(cmin, cmax, dtype=np.float64)[None, :]
        dr = rr - float(row)
        dc = cc - float(col)
        dist_m = np.sqrt(dr * dr + dc * dc) * pixel_size_meters

        if hasattr(window, "mask"):
            base_valid = ~window.mask
            data = window.data
        else:
            base_valid = np.ones(window.shape, dtype=bool)
            data = window

        for k in range(n_rings):
            inner_m = 0.0 if k == 0 else float(radii_m[k - 1])
            outer_m = float(radii_m[k])
            if inner_m <= 0:
                ring_mask = dist_m <= outer_m
            else:
                ring_mask = (dist_m <= outer_m) & (dist_m > inner_m)
            valid = base_valid & ring_mask
            vals = np.asarray(data[valid], dtype=np.float64).ravel()
            vals = vals[~np.isnan(vals)]
            ring_values[pos][k] = vals

    return ring_values


def precompute_vector_ring_values(
    metric_data: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame,
    radii_m: np.ndarray,
    metric_col: str,
) -> list[list[np.ndarray]]:
    """Bin each point's vector neighbourhood into concentric annuli (per radius)."""
    n_pts = len(points_gdf)
    n_rings = len(radii_m)
    ring_values: list[list[np.ndarray]] = [
        [np.array([], dtype=np.float64) for _ in range(n_rings)] for _ in range(n_pts)
    ]

    points_wgs84 = points_gdf.to_crs("EPSG:4326")
    centroid = points_wgs84.geometry.union_all().centroid
    lon, lat = centroid.x, centroid.y
    utm_zone = int((lon + 180) / 6) + 1
    utm_crs = f"EPSG:326{utm_zone:02d}" if lat >= 0 else f"EPSG:327{utm_zone:02d}"

    points_utm = points_gdf.to_crs(utm_crs)
    metric_utm = metric_data.to_crs(utm_crs)

    idx_to_pos = {idx: pos for pos, idx in enumerate(points_gdf.index)}

    for idx, prow in points_utm.iterrows():
        pos = idx_to_pos[idx]
        pt = prow.geometry
        for k in range(n_rings):
            inner_m = 0.0 if k == 0 else float(radii_m[k - 1])
            outer_m = float(radii_m[k])
            buf_o = pt.buffer(outer_m)
            if inner_m <= 0:
                ring_poly = buf_o
            else:
                ring_poly = buf_o.difference(pt.buffer(inner_m))
            vals_list: list[np.ndarray] = []
            for poly in polygonal_parts(ring_poly):
                if poly.is_empty:
                    continue
                tmp = gpd.GeoDataFrame(geometry=[poly], crs=points_utm.crs)
                joined = gpd.sjoin(metric_utm, tmp, how="inner", predicate="intersects")
                if metric_col in joined.columns:
                    v = joined[metric_col].dropna().values.astype(np.float64)
                    if v.size:
                        vals_list.append(v)
            ring_values[pos][k] = (
                np.concatenate(vals_list) if vals_list else np.array([])
            )

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


def aggregate_from_ring_cache(
    radii: np.ndarray,
    ring_rows: list[list[np.ndarray]],
    radius_m: float,
    stat: str,
    percentile: int,
) -> np.ndarray:
    """Reduce every point's cached annuli to a per-point disk statistic."""
    n = len(ring_rows)
    out = np.full(n, np.nan, dtype=np.float64)
    end_idx = ring_end_index(radii, radius_m)
    if end_idx < 0:
        return out
    for pos in range(n):
        out[pos] = aggregate_disk_from_rings(ring_rows[pos], end_idx, stat, percentile)
    return out
