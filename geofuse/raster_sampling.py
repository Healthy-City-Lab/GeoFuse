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

# Metric rasters larger than this (uncompressed band bytes) are read lazily in
# windows from disk instead of loaded whole — keeps national-scale rasters off
# the heap. Smaller rasters stay in RAM (faster per-point reads).
LAZY_RASTER_THRESHOLD_BYTES = 256 * 1024 * 1024


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


