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

from . import metric_intake, parallel, preaggregation
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
            logger(
                "INFO",
                f"Metric raster ~{est_bytes / (1024**2):.0f} MB exceeds the "
                f"in-memory threshold; reading windows lazily from {filepath}",
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
            if col == "wave" and dt == np.dtype(object):
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


def random_test_blocks(
    blocks: np.ndarray,
    sizes: np.ndarray,
    test_fraction: float,
    rng: np.random.Generator,
) -> set:
    """Draw whole blocks at random until they hold ~``test_fraction`` of the groups.

    ``sizes`` is the number of groups in each of ``blocks``. Blocks are taken
    in a random order, and the block that crosses the target is kept when that
    lands closer to it than stopping short. At least one block is held out
    and at least one is left for training. Unlike
    :func:`stripe_to_test_blocks`, whose seed only shifts the stripe's start,
    every seed can draw another test set.
    """
    n = len(blocks)
    if n == 0:
        return set()
    order = rng.permutation(n)
    cum = np.cumsum(np.asarray(sizes, dtype=np.float64)[order])
    target = test_fraction * cum[-1]
    k = min(int(np.searchsorted(cum, target)), n - 1)
    before = cum[k - 1] if k > 0 else 0.0
    take = k + 1 if abs(cum[k] - target) <= abs(before - target) else k
    take = min(max(take, 1), max(n - 1, 1))
    return set(np.asarray(blocks)[order[:take]].tolist())


def split_control_matrix(
    c_arr: np.ndarray | None, n_cov_cols: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Split a resampled ``[covariates | smooth]`` matrix into its two parts.

    The reporting resamplers draw rows of one control matrix so covariates
    and smooth stay row-aligned with the observation; the scorer wants them
    separately (it expands only the covariates into the residualization
    basis, then appends the smooth).
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


def fusion_param_key(channel_mode: str, weights: dict) -> tuple:
    """Hashable key over the params that determine the composite."""
    items = tuple(
        sorted((str(k), v) for k, v in weights.items() if not str(k).startswith("__"))
    )
    return (channel_mode, items)


