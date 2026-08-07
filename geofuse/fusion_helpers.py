"""Stateless helpers lifted out of :class:`~geofuse.fusion.MetricFusionEngine`.

Every function here was a method that never touched ``self`` — a free function
that happened to live on the class. Grouped by what they operate on: metric
files and their columns, entity collapse, split construction, and the pool /
cache-key bookkeeping the engine does around them.

Keeping them out of the engine is what makes them testable without building an
engine, and keeps the class to the methods that genuinely need its state.
"""

from __future__ import annotations

import hashlib
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol

from . import metric_intake, metric_sampling, parallel, preaggregation
from .crs_utils import (
    crs_uses_metre_axes,
    estimate_metre_projected_crs_for_gdf,
    normalize_geographic_gdf_to_wgs84,
    select_grid_crs_with_warning,
)
from .logger import get_logger
from .raster_sampling import LAZY_RASTER_THRESHOLD_BYTES, LazyRasterArray
from .vector_io import read_vector_aliased_column

logger = get_logger("FUSION")


def apply_circular_buffer_aggregation(
    points_gdf: gpd.GeoDataFrame,
    metric_data: gpd.GeoDataFrame | dict,
    radius_meters: float,
    stat: str,
    percentile: int = 50,
) -> np.ndarray:
    """
    Apply circular buffer aggregation to sample metric at point locations.

    This matches CGI.ipynb calculate_gm() logic:
    - Creates circular buffer around each point
    - Samples all metric values within buffer
    - Aggregates using specified statistic (mean/median/percentile)

    Args:
        points_gdf: GeoDataFrame with point geometries
        metric_data: Raster dict or vector GeoDataFrame to sample from
        radius_meters: Buffer radius in meters
        stat: Aggregation function ('mean', 'median', 'percentile')
        percentile: Percentile value if stat='percentile'

    Returns:
        Array of aggregated metric values for each point
    """
    result = np.full(len(points_gdf), np.nan)

    # Create mapping from original index to position for correct array indexing
    idx_to_pos = {idx: pos for pos, idx in enumerate(points_gdf.index)}

    if isinstance(metric_data, dict):  # Raster
        metric_array = metric_data["data"]
        transform = metric_data["transform"]
        metric_crs = metric_data["crs"]

        # Same footprint the pre-aggregation cache reduces over: buffer the
        # entity in a projected (metres) CRS and reproject the buffer into
        # the raster's CRS, so the sampled disc is metrically true whatever
        # the raster's CRS.
        from rasterio.windows import Window as _Window
        from rasterio.windows import transform as _window_transform

        if points_gdf.crs is not None and crs_uses_metre_axes(points_gdf.crs):
            buffer_crs = points_gdf.crs
        else:
            buffer_crs = estimate_metre_projected_crs_for_gdf(points_gdf)
        pts_buf_crs = points_gdf.to_crs(buffer_crs)
        if str(buffer_crs) != str(metric_crs):
            from pyproj import Transformer as _Transformer

            xy_fn = _Transformer.from_crs(
                buffer_crs, metric_crs, always_xy=True
            ).transform
        else:
            xy_fn = None

        templates = preaggregation.origin_circle_templates([float(radius_meters)])
        h, w = int(metric_array.shape[0]), int(metric_array.shape[1])
        for idx, geom in zip(pts_buf_crs.index, pts_buf_crs.geometry.to_numpy()):
            if geom is None or geom.is_empty:
                continue
            disc = preaggregation.reproject_geoms(
                [preaggregation.buffer_at(geom, float(radius_meters), templates)],
                xy_fn,
            )[0]
            if disc is None or disc.is_empty:
                continue
            minx, miny, maxx, maxy = disc.bounds
            r1, c1 = rowcol(transform, minx, maxy)
            r2, c2 = rowcol(transform, maxx, miny)
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
            ring = preaggregation.ring_index_grid([disc], window.shape, win_transform)
            if ring is None:
                continue
            values = data[base_valid & (ring == 0)]

            if len(values) > 0:
                pos = idx_to_pos[idx]
                if stat == "mean":
                    result[pos] = np.mean(values)
                elif stat == "median":
                    result[pos] = np.median(values)
                elif stat == "percentile":
                    result[pos] = np.percentile(values, percentile)

    else:  # Vector GeoDataFrame
        # For vector data, use spatial join with buffered points
        # Convert both to a common metric CRS for accurate buffering

        # Determine appropriate UTM zone from point data
        # Convert to WGS84 first to get proper geographic coordinates (lon/lat in degrees)
        points_wgs84 = points_gdf.to_crs("EPSG:4326")
        centroid = points_wgs84.geometry.union_all().centroid
        lon = centroid.x
        lat = centroid.y

        # Validate we actually have geographic coordinates (lon should be -180 to 180)
        if not (-180 <= lon <= 180):
            raise ValueError(
                f"Invalid longitude {lon} after WGS84 conversion. "
                f"Input CRS was {points_gdf.crs}, conversion may have failed."
            )

        # Calculate UTM zone from geographic coordinates
        utm_zone = int((lon + 180) / 6) + 1
        # Determine hemisphere
        if lat >= 0:
            utm_crs = f"EPSG:326{utm_zone:02d}"  # Northern hemisphere
        else:
            utm_crs = f"EPSG:327{utm_zone:02d}"  # Southern hemisphere

        # Convert to UTM for buffering
        points_utm = points_gdf.to_crs(utm_crs)
        metric_utm = metric_data.to_crs(utm_crs)

        # Create buffers in UTM (meters)
        points_buffered = points_utm.copy()
        points_buffered["geometry"] = points_utm.buffer(radius_meters)

        # Spatial join to find metrics within buffers
        joined = gpd.sjoin(
            points_buffered, metric_utm, how="left", predicate="intersects"
        )

        # Detect metric column
        metric_col = metric_data.attrs.get("metric_column")
        if metric_col is None or metric_col not in metric_data.columns:
            numeric_cols = metric_data.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            # Filter out geometry-related columns
            numeric_cols = [
                c
                for c in numeric_cols
                if c not in ["index_right", "index_left", "index"]
            ]
            if numeric_cols:
                metric_col = numeric_cols[0]
            else:
                logger.warning(
                    f"No numeric columns found in metric data. Columns: {metric_data.columns.tolist()}"
                )
                return result  # Return all NaN

        logger.info(
            f"Buffer aggregation using column '{metric_col}' from {metric_data.columns.tolist()}, radius={radius_meters}m, stat={stat}"
        )

        # Check if any data was joined
        non_null_joins = (
            joined[metric_col].notna().sum() if metric_col in joined.columns else 0
        )
        logger.info(
            f"Spatial join: {len(joined)} total rows, {non_null_joins} with valid metric values"
        )

        logger.debug(
            f"Using metric column: {metric_col} from {metric_data.columns.tolist()}"
        )
        logger.debug(f"Joined shape: {joined.shape}, Points shape: {len(points_gdf)}")

        # Aggregate by original point index — one vectorized groupby over the
        # joined frame instead of a per-point boolean scan (which was O(n²)).
        if metric_col in joined.columns:
            grouped = joined[metric_col].dropna().groupby(level=0)
            if stat == "median":
                agg = grouped.median()
            elif stat == "percentile":
                agg = grouped.quantile(percentile / 100.0)
            else:
                agg = grouped.mean()
            for idx, val in agg.items():
                pos = idx_to_pos.get(idx)
                if pos is not None:
                    result[pos] = val

        # Log summary of results
        valid_count = np.sum(~np.isnan(result))
        logger.info(
            f"Buffer aggregation result: {valid_count}/{len(result)} points have valid values"
        )

    return result


def pool_spec(
    positions: np.ndarray,
    *,
    preps: dict,
    shared_xy: np.ndarray | None,
    gvi_radii: tuple[int, ...],
    ndvi_radii: tuple[int, ...],
    utm_crs: Any,
    point_xy_utm: np.ndarray,
    share_dir: str,
) -> dict:
    """Describe the whole job to a worker, publishing its bulk arrays."""
    crs_key = str(utm_crs)

    def _channel_spec(name: str, prep: tuple) -> dict:
        if prep[0] == "raster":
            array = prep[1]
            common = {
                "transform": prep[2],
                "from_crs": crs_key,
                "to_crs": str(prep[3]),
            }
            path = getattr(array, "path", None)
            if path is not None:
                # Already a windowed reader over a file; the worker opens
                # its own handle rather than sharing this one, since GDAL
                # datasets are not safe across processes.
                return {
                    "kind": "raster",
                    "path": path,
                    "band": getattr(array, "band", 1),
                    **common,
                }
            return {
                "kind": "raster_array",
                "path": parallel.publish_array(
                    share_dir,
                    f"{name}-raster",
                    (
                        np.ma.filled(array, np.nan)
                        if np.issubdtype(array.dtype, np.floating)
                        else np.ma.filled(array.astype(np.float32), np.nan)
                    ),
                ),
                **common,
            }
        spec = {
            "kind": "points",
            "values": parallel.publish_array(share_dir, f"{name}-values", prep[2]),
        }
        if shared_xy is not None and name in ("veg", "terrain"):
            spec["shared"] = True
        else:
            spec["xy"] = parallel.publish_array(share_dir, f"{name}-xy", prep[1])
        return spec

    spec = {
        "entity_xy": parallel.publish_array(
            share_dir, "entity-xy", point_xy_utm[positions]
        ),
        "gvi_radii": tuple(gvi_radii),
        "ndvi_radii": tuple(ndvi_radii),
        "channels": {n: _channel_spec(n, p) for n, p in preps.items()},
    }
    if shared_xy is not None:
        spec["gvi_shared_xy"] = parallel.publish_array(
            share_dir, "gvi-shared-xy", shared_xy
        )
    return spec


def area_balanced_split_within_bin(
    bin_df: pd.DataFrame,
    area_col: str,
    target_fractions: dict[str, float],
    random_state: int,
) -> dict[str, list]:
    """Allocate one outcome bin's polygons to splits with greedy area balance.

    Shuffles the polygons inside the bin, then for each one picks the
    split with the largest remaining area deficit (in fraction-of-bin
    units). Returns a dict mapping split name to a list of polygon
    row indices.
    """
    if len(bin_df) == 0:
        return {name: [] for name in target_fractions}
    total_area = float(bin_df[area_col].sum())
    if total_area <= 0:
        # Fall back to count-balanced when areas are missing.
        rng = np.random.default_rng(random_state)
        shuffled = rng.permutation(bin_df.index.to_numpy())
        n = len(shuffled)
        order = list(target_fractions)
        counts = {name: int(round(target_fractions[name] * n)) for name in order}
        # Reconcile rounding to total n.
        diff = n - sum(counts.values())
        counts[order[0]] += diff
        out: dict[str, list] = {name: [] for name in order}
        i = 0
        for name in order:
            out[name] = list(shuffled[i : i + counts[name]])
            i += counts[name]
        return out

    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(bin_df.index.to_numpy())
    target_area = {
        name: target_fractions[name] * total_area for name in target_fractions
    }
    used_area = dict.fromkeys(target_fractions, 0.0)
    assignments: dict[str, list] = {name: [] for name in target_fractions}
    for idx in shuffled:
        area = float(bin_df.at[idx, area_col])
        deficits = {
            name: target_area[name] - used_area[name] for name in target_fractions
        }
        choice = max(deficits, key=deficits.get)
        assignments[choice].append(idx)
        used_area[choice] += area
    return assignments


def load_metric_file(filepath: str, channel: str) -> gpd.GeoDataFrame | dict:
    """Load one channel's metric from GeoJSON or GeoTIFF."""
    if filepath.endswith((".tif", ".tiff")):
        with rasterio.open(filepath) as src:
            transform, crs, bounds = src.transform, src.crs, src.bounds
            itemsize = np.dtype(src.dtypes[0]).itemsize
            est_bytes = src.width * src.height * itemsize
            data = (
                src.read(1, masked=True)
                if est_bytes <= LAZY_RASTER_THRESHOLD_BYTES
                else None
            )
        if data is None:
            # Too large to hold in RAM (national-scale): read windows from
            # disk on demand instead. Each thread gets its own handle.
            data = LazyRasterArray(filepath, band=1)
            logger.info(
                f"Metric raster ~{est_bytes / (1024**2):.0f} MB exceeds the "
                f"in-memory threshold; reading windows lazily from {filepath}"
            )
        return {
            "data": data,
            "transform": transform,
            "crs": crs,
            "bounds": bounds,
        }
    else:
        # Only the channel's own column is sampled, so the rest of the
        # file's schema never needs to reach memory.
        gdf, metric_col = read_vector_aliased_column(
            filepath,
            metric_intake.channel_columns(channel),
            description=f"{channel} metric value",
        )
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        gdf = normalize_geographic_gdf_to_wgs84(gdf)
        gdf.attrs["metric_column"] = metric_col
        return gdf


def resolved_state(
    preps: dict,
    shared_xy: np.ndarray | None,
    *,
    gvi_radii: tuple[int, ...],
    ndvi_radii: tuple[int, ...],
    utm_crs: Any,
) -> dict:
    """In-process aggregator state for :func:`preaggregation.aggregate_entity_batch`."""
    from sklearn.neighbors import BallTree

    def _aggregator(prep: tuple) -> tuple:
        if prep[0] == "raster":
            to_raster = None
            if str(prep[3]) != str(utm_crs):
                from pyproj import Transformer

                to_raster = Transformer.from_crs(
                    utm_crs, prep[3], always_xy=True
                ).transform
            return ("raster", prep[1], prep[2], to_raster)
        if prep[0] == "point":
            return ("point", BallTree(prep[1]), prep[2])
        return prep

    state = {"gvi_radii": gvi_radii, "ndvi_radii": ndvi_radii}
    if shared_xy is not None:
        state["gvi_shared"] = (
            BallTree(shared_xy),
            {"veg": preps["veg"][2], "terrain": preps["terrain"][2]},
        )
    else:
        state["veg"] = _aggregator(preps["veg"])
        state["terrain"] = _aggregator(preps["terrain"])
    state["ndvi"] = _aggregator(preps["ndvi"])
    return state


def downcast_fusion_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Narrow a fusion/panel frame's column dtypes in place-ish.

    ``wave`` (year strings) -> category; float64 -> float32; int64 -> the
    smallest signed int that holds the column. Geometry and already-narrow
    columns are left untouched. Returns the same frame.
    """
    if df is None or len(df) == 0:
        return df
    for col in df.columns:
        if col == "geometry":
            continue
        try:
            s = df[col]
            dt = s.dtype
            if col == "wave" and dt == object:
                df[col] = s.astype("category")
            elif dt == np.float64:
                df[col] = s.astype(np.float32)
            elif dt == np.int64:
                lo, hi = int(s.min()), int(s.max())
                for t in (np.int16, np.int32):
                    if np.iinfo(t).min <= lo and hi <= np.iinfo(t).max:
                        df[col] = s.astype(t)
                        break
        except Exception:
            continue  # leave the column as-is on any edge case
    return df


def align_channel_axes(
    arrays: list[tuple[np.ndarray, bool]], static: dict, n_rows: int
) -> list[np.ndarray]:
    """Put every channel on one axis, expanding deduplicated ones if needed.

    All three channels normally resolve the same way, but an off-grid cell
    can send one of them down the row-wise fallback; expanding the others
    keeps the composite's inputs row-aligned.
    """
    if not arrays:
        return []
    if all(on_uniq for _, on_uniq in arrays):
        return [a for a, _ in arrays]
    inverse = static.get("uniq_inverse")
    out: list[np.ndarray] = []
    for arr, on_uniq in arrays:
        if on_uniq and inverse is not None:
            out.append(np.asarray(arr)[inverse])
        elif on_uniq:
            out.append(np.full(n_rows, np.nan, dtype=np.float32))
        else:
            out.append(arr)
    return out


def spatial_cache_key(
    coords_xy: np.ndarray, target: np.ndarray, cov: np.ndarray | None
) -> bytes:
    """Content fingerprint of a scored split for memoizing its spatial basis.

    Keyed on the coordinates and outcome (both stable across Optuna trials)
    so studies sharing a fold reuse the basis, while bootstrap resamples —
    which change row membership — get a fresh one without manual cache
    invalidation. Digests the raw bytes so two splits holding the same
    values in a different order cannot share an entry.
    """
    h = hashlib.blake2b(digest_size=16)
    for arr in (coords_xy, target, cov):
        if arr is None:
            h.update(b"\x00")
            continue
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
        h.update(repr(a.shape).encode())
        h.update(a.tobytes())
    return h.digest()


def entity_collapse_mask(
    data: pd.DataFrame, catchment_radius: float | None
) -> np.ndarray | None:
    """Row mask for the per-trial catchment collapse, or ``None``.

    Returns ``None`` for polygon / raster targets (no radius mask — every
    in-footprint pixel contributes, the original behavior). For point /
    line targets, keeps pixels within ``catchment_radius`` of their
    entity plus each entity's nearest pixel, so no entity drops out at
    small radii.
    """
    if "_catchment_dist" not in data.columns or catchment_radius is None:
        return None
    dist = data["_catchment_dist"].to_numpy(dtype=np.float64)
    if "_is_nearest" in data.columns:
        near = data["_is_nearest"].to_numpy(dtype=bool)
    else:
        near = np.zeros(len(data), dtype=bool)
    return (dist <= float(catchment_radius)) | near


def stripe_to_test_blocks(
    ordered_blocks: np.ndarray, test_fraction: float, rng: np.random.Generator
) -> set:
    """Pick ~``test_fraction`` of blocks, spread evenly across their order.

    ``ordered_blocks`` is the space-filling block sequence; selecting evenly
    spaced positions (with a seed-jittered start) spreads the held-out
    blocks across the full extent rather than into one contiguous corner.
    """
    n = len(ordered_blocks)
    if n == 0:
        return set()
    n_pick = max(1, int(round(test_fraction * n)))
    n_pick = min(n_pick, n)
    step = n / n_pick
    start = int(rng.integers(0, max(1, int(np.floor(step)))))
    idxs = (np.floor(np.arange(n_pick) * step).astype(int) + start) % n
    return set(ordered_blocks[np.unique(idxs)].tolist())


def catchment_radius(
    veg_radius: float,
    terrain_radius: float,
    ndvi_radius: float,
    active_channel: str | None = None,
) -> float:
    """Per-trial catchment radius for the point/line collapse.

    ``active_channel``'s radius for a standalone study (only that channel
    contributes to the score), or the largest of the three for the combined
    CGI run (every channel feeds the per-pixel composite).
    """
    ch = active_channel or "cgi"
    if ch == "veg":
        return float(veg_radius)
    if ch == "terrain":
        return float(terrain_radius)
    if ch == "ndvi":
        return float(ndvi_radius)
    return float(max(veg_radius, terrain_radius, ndvi_radius))


def split_control_matrix(
    c_arr: np.ndarray | None, n_cov_cols: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Split a resampled ``[covariates | smooth]`` matrix into its two parts.

    The reporting resamplers draw rows of one control matrix so covariates
    and smooth stay row-aligned with the observation; the scorer wants them
    separately (it residualizes the exposure on the smooth alone under
    ``spatial_plus``, and expands only the covariates into the
    residualization basis).
    """
    if c_arr is None:
        return None, None
    cov_part = c_arr[:, :n_cov_cols] if n_cov_cols else None
    sb_part = c_arr[:, n_cov_cols:] if c_arr.shape[1] > n_cov_cols else None
    return cov_part, sb_part


def grid_metric_crs(preaggr_gdf: gpd.GeoDataFrame, log) -> Any:
    """The metric CRS the pre-aggregation samples in.

    For a gridded (point / polygon) target the pixels are already built in
    the grid's metric CRS, so that same CRS is reused — this keeps the
    greenery-cache key identical between the coverage probe and the build.
    A raster target has no grid, so a distortion-minimising CRS is picked.
    """
    if preaggr_gdf.crs is not None and "_global_pid" in preaggr_gdf.columns:
        return preaggr_gdf.crs
    crs, _dist, _name = select_grid_crs_with_warning(
        preaggr_gdf, log, role="Pre-aggregation CRS"
    )
    return crs


def attach_cache_for_coverage(entity_gdf: gpd.GeoDataFrame) -> bool:
    """Reserved: derive the coverage gate from a reusable greenery cache.

    Disabled for now. The coverage sampler reports the nearest-feature
    value while the cache stores the max-radius mean, so reading coverage
    from the cache would shift the channel-normalisation bounds between a
    fresh run and a reuse run. Re-enabling it requires computing the
    channel scale from cache statistics on both paths.
    """
    return False


def fusion_param_key(channel_mode: str, weights: dict) -> tuple:
    """Hashable key over the params that determine the composite."""
    items = tuple(
        sorted((str(k), v) for k, v in weights.items() if not str(k).startswith("__"))
    )
    return (channel_mode, items)


def metric_value_column(metric_gdf: gpd.GeoDataFrame, channel: str) -> str:
    return metric_sampling.vector_metric_column(metric_gdf, channel)
