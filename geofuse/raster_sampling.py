"""Sample a raster at vector features (points with optional buffer, polygons).

Promoted out of the fusion engine's ``_apply_circular_buffer_aggregation`` so
the NDVI engine — and any future consumer — can attach raster values to
arbitrary input features without depending on fusion's internals.

Output shape: the input GeoDataFrame plus one numeric column (default name
``value``). Each row's value is the raster's pixel reading at the feature
(exact pixel for unbuffered points, zonal stat for polygons / buffered
points). Geometry and CRS are preserved.

Two stat families are supported:

* Exact-pixel reads (``radius_m == 0`` and geometry is a Point) → take the
  pixel under the point.
* Zonal aggregates (``radius_m > 0`` or polygon features) → mask raster
  values within the geometry and reduce via ``mean`` / ``median`` /
  ``min`` / ``max`` / ``std`` / ``count``.

The raster is opened once and each feature is read through a small window,
so memory use is proportional to one feature's worth of pixels.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window
from rasterio.windows import from_bounds as window_from_bounds
from shapely.geometry import Point

from .crs_utils import metres_per_degree_at_lat


class LazyRasterArray:
    """A 2D, array-like view over one raster band that reads windows from disk.

    Lets large (national-scale) metric rasters be sampled without holding the
    whole band in memory. Supports ``.shape`` / ``.dtype`` plus:

    * integer indexing ``arr[row, col]`` → a single masked pixel value, and
    * contiguous slice indexing ``arr[r0:r1, c0:c1]`` → a masked 2D window,

    matching how the fusion engine reads in-memory ``src.read(masked=True)``
    arrays, so consumers work unchanged.

    Concurrency: GDAL datasets are **not** safe for concurrent reads on a single
    handle, so each thread gets its **own** read-only handle (``threading.local``).
    Multiple read-only handles to the same file do not conflict, which lets the
    parallel pre-aggregation build sample windows from many threads at once.
    """

    def __init__(self, path: str, band: int = 1) -> None:
        self.path = str(path)
        self.band = int(band)
        with rasterio.open(self.path) as src:
            self.shape = (int(src.height), int(src.width))
            self.dtype = np.dtype(src.dtypes[self.band - 1])
            self.nodata = src.nodata
        self._local = threading.local()
        # Every handle any thread opens, so ``close_all`` can release the file
        # deterministically instead of waiting for each worker thread to be
        # collected — Windows keeps the lock until then.
        self._handles: list = []
        self._handles_lock = threading.Lock()

    @property
    def ndim(self) -> int:
        return 2

    def _src(self) -> rasterio.io.DatasetReader:
        src = getattr(self._local, "src", None)
        if src is None or src.closed:
            src = rasterio.open(self.path)
            self._local.src = src
            with self._handles_lock:
                self._handles.append(src)
        return src

    def __getitem__(self, key):
        rk, ck = key
        h, w = self.shape
        if isinstance(rk, slice) or isinstance(ck, slice):
            r0 = 0 if rk.start is None else max(0, int(rk.start))
            r1 = h if rk.stop is None else min(h, int(rk.stop))
            c0 = 0 if ck.start is None else max(0, int(ck.start))
            c1 = w if ck.stop is None else min(w, int(ck.stop))
            if r1 <= r0 or c1 <= c0:
                return np.ma.masked_array(
                    np.empty((max(0, r1 - r0), max(0, c1 - c0)), dtype=self.dtype),
                    mask=True,
                )
            win = Window(c0, r0, c1 - c0, r1 - r0)
            return self._src().read(self.band, window=win, masked=True)
        r, c = int(rk), int(ck)
        return self._src().read(self.band, window=Window(c, r, 1, 1), masked=True)[0, 0]

    def close(self) -> None:
        """Close the calling thread's handle (others close on thread teardown)."""
        src = getattr(self._local, "src", None)
        if src is not None and not src.closed:
            src.close()
            self._local.src = None

    def close_all(self) -> None:
        """Close every handle opened for this raster, on any thread."""
        with self._handles_lock:
            handles, self._handles = self._handles, []
        for src in handles:
            try:
                if not src.closed:
                    src.close()
            except Exception:
                pass
        self._local = threading.local()


_STAT_FUNCS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": lambda a: float(np.nanmean(a)),
    "median": lambda a: float(np.nanmedian(a)),
    "min": lambda a: float(np.nanmin(a)),
    "max": lambda a: float(np.nanmax(a)),
    "std": lambda a: float(np.nanstd(a)),
    "count": lambda a: float(np.count_nonzero(~np.isnan(a))),
}


def _buffer_geom_in_metres(point: Point, point_lat: float, radius_m: float):
    """Approximate circular buffer in degrees using metres-per-degree at ``point_lat``.

    Faster than reprojecting to a metric CRS per feature and accurate enough
    for the small-radius use case (10s to 100s of metres) the NDVI tab
    typically asks for. For polygons / large radii callers should pre-
    reproject to a metric CRS and pass the already-projected geometry.
    """
    m_per_deg_lon, m_per_deg_lat = metres_per_degree_at_lat(point_lat)
    return point.buffer(
        radius_m / max(m_per_deg_lon, m_per_deg_lat),
        resolution=16,
    )


def sample_raster_at_features(
    raster_path: str,
    features_gdf: gpd.GeoDataFrame,
    *,
    band: int = 1,
    radius_m: float = 0.0,
    stat: str = "mean",
    value_column: str = "value",
    count_column: str | None = None,
    nodata_sentinels: tuple[float, ...] = (-9999.0,),
) -> gpd.GeoDataFrame:
    """Attach raster values to each feature in ``features_gdf``.

    Returns a copy of ``features_gdf`` with ``value_column`` added (and
    optionally ``count_column`` recording how many valid pixels contributed
    to each aggregate — useful as a confidence signal).

    ``radius_m == 0`` with point features → exact-pixel read. Otherwise the
    feature is rasterised into the window and the masked pixels are reduced
    via ``stat``. Features that fall outside the raster, or end up with
    only nodata pixels, get ``NaN``.
    """
    if stat not in _STAT_FUNCS:
        raise ValueError(
            f"Unknown stat '{stat}'. Expected one of {sorted(_STAT_FUNCS)}."
        )
    reducer = _STAT_FUNCS[stat]
    out = features_gdf.copy()
    out[value_column] = np.nan
    if count_column:
        out[count_column] = 0

    if features_gdf.empty:
        return out

    with rasterio.open(raster_path) as src:
        # Reproject features to raster CRS up front — cheap one-shot vs.
        # per-feature transforms.
        if src.crs is None:
            feat_in_src = features_gdf
        elif features_gdf.crs is None or features_gdf.crs == src.crs:
            feat_in_src = features_gdf
        else:
            feat_in_src = features_gdf.to_crs(src.crs)

        src_nodata = src.nodata
        is_geographic = bool(getattr(src.crs, "is_geographic", False))

        for fid, row in feat_in_src.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            # Resolve the working geometry for this feature.
            if geom.geom_type == "Point":
                if radius_m and radius_m > 0:
                    if is_geographic:
                        work_geom = _buffer_geom_in_metres(geom, geom.y, radius_m)
                    else:
                        work_geom = geom.buffer(radius_m, resolution=16)
                else:
                    # Exact-pixel sample.
                    try:
                        r, c = src.index(geom.x, geom.y)
                        if r < 0 or c < 0 or r >= src.height or c >= src.width:
                            continue
                        val = float(
                            src.read(band, window=rasterio.windows.Window(c, r, 1, 1))[
                                0, 0
                            ]
                        )
                        if (
                            math.isnan(val)
                            or val in nodata_sentinels
                            or (src_nodata is not None and val == src_nodata)
                        ):
                            continue
                        out.at[fid, value_column] = val
                        if count_column:
                            out.at[fid, count_column] = 1
                    except Exception:
                        continue
                    continue
            else:
                work_geom = geom

            # Zonal aggregate path — compute the window, mask, reduce.
            try:
                minx, miny, maxx, maxy = work_geom.bounds
                win = (
                    window_from_bounds(minx, miny, maxx, maxy, transform=src.transform)
                    .round_offsets()
                    .round_lengths()
                )
                if win.width <= 0 or win.height <= 0:
                    continue
                # Clamp to raster bounds — rasterio is permissive but the
                # downstream mask call wants a positive window.
                col_off = max(0, int(win.col_off))
                row_off = max(0, int(win.row_off))
                width = min(int(win.width), src.width - col_off)
                height = min(int(win.height), src.height - row_off)
                if width <= 0 or height <= 0:
                    continue
                win = rasterio.windows.Window(col_off, row_off, width, height)
                arr = src.read(band, window=win).astype(np.float64)
                if arr.size == 0:
                    continue

                win_transform = rasterio.windows.transform(win, src.transform)
                mask = geometry_mask(
                    [work_geom],
                    transform=win_transform,
                    out_shape=arr.shape,
                    invert=True,
                )
                arr[~mask] = np.nan
                if src_nodata is not None:
                    arr[arr == src_nodata] = np.nan
                for sentinel in nodata_sentinels:
                    arr[arr == sentinel] = np.nan
                valid = arr[~np.isnan(arr)]
                if valid.size == 0:
                    continue
                out.at[fid, value_column] = reducer(arr)
                if count_column:
                    out.at[fid, count_column] = int(valid.size)
            except Exception:
                continue

    return out
