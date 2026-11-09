"""Metric sampling and buffer ladders for the fusion engine.

Pure, engine-state-free helpers that :class:`geofuse.fusion.MetricFusionEngine`
delegates to:

* **Point sampling** — :func:`nearest_metric_join` (nearest vector value per
  point) and :func:`sample_raster_values` (vectorised nearest-pixel read).
* **Radius grid** — :func:`radius_int_bounds` aligns an integer radius search to
  the user's min/max/step; :func:`ladder_radii` builds every buffer ladder.

Neighbourhood statistics come from the pre-aggregation cache
(:mod:`geofuse.preaggregation`).
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol

from . import metric_intake


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


#: Buffer-ladder spacings. ``linear`` steps by a fixed number of metres;
#: ``log`` places a chosen number of rungs geometrically between the bounds,
#: so every rung covers the same log distance.
LADDER_SPACINGS = ("linear", "log")


def ladder_radii(
    r_min: float,
    r_max: float,
    r_step: float,
    *,
    spacing: str = "linear",
    rungs: int | None = None,
) -> tuple[int, ...]:
    """Integer radii (m) of a buffer ladder, ascending and unique.

    ``linear`` is ``range(min, max + 1, step)`` after :func:`radius_int_bounds`.
    ``log`` takes ``rungs`` (at least 2) geometric points from ``min`` to
    ``max``, rounded to whole metres with duplicates dropped, so the count can
    come out lower than asked on a narrow range.
    """
    if spacing == "log":
        lo = max(1, int(round(min(r_min, r_max))))
        hi = max(lo, int(round(max(r_min, r_max))))
        n = max(2, int(rungs or 2))
        return tuple(sorted({int(round(r)) for r in np.geomspace(lo, hi, n)}))
    if spacing != "linear":
        raise ValueError(f"spacing must be one of {LADDER_SPACINGS}.")
    lo, hi, step = radius_int_bounds(r_min, r_max, r_step)
    return tuple(range(lo, hi + 1, step))


def vector_metric_column(metric_data: gpd.GeoDataFrame, channel: str) -> str:
    """Resolve the numeric value column of a vector metric for ``channel``.

    Honours the ``metric_column`` hint a loader may have recorded, otherwise
    resolves strictly against the channel's own column names.
    """
    metric_col = metric_data.attrs.get("metric_column")
    if metric_col and metric_col in metric_data.columns:
        return str(metric_col)
    return metric_intake.resolve_channel_column(metric_data.columns, channel)


