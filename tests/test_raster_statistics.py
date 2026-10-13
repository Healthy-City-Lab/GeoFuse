"""GeoTIFF outputs store their exact statistics.

Each band carries GDAL's ``STATISTICS_*`` tags: the minimum, maximum, mean,
standard deviation and valid percentage of the pixels that are neither nodata
nor non-finite. A file's statistics are computed from the blocks stored on
disk only, so they cost what the data costs, not what the bounding box does.
"""

import os
import sys
import tempfile
import unittest
import warnings
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import rasterio
from rasterio.transform import from_origin

from geofuse import crs_utils
from geofuse import ndvi as ndvi_mod
from geofuse.crs_utils import stream_mosaic_to_geotiff, write_band_statistics

CRS = "EPSG:32611"
RES = 30.0


def _field(rng, h, w):
    """NDVI-like values with a sentinel gap and a NaN gap."""
    yy, xx = np.mgrid[0:h, 0:w]
    v = 0.3 + 0.4 * np.sin(xx / 9.0) * np.cos(yy / 11.0)
    v = v + rng.normal(0.0, 0.02, (h, w))
    v[h // 4 : h // 4 + 6, w // 3 : w // 3 + 7] = -9999.0
    v[h // 2 : h // 2 + 3, :5] = np.nan
    return v


def _write(path, stack, x0=500000.0, y0=5600000.0, **profile):
    """A GeoTIFF as Earth Engine delivers it unless ``profile`` says otherwise."""
    stack = stack[None] if stack.ndim == 2 else stack
    base = {
        "driver": "GTiff",
        "height": stack.shape[1],
        "width": stack.shape[2],
        "count": stack.shape[0],
        "dtype": "float64",
        "crs": CRS,
        "transform": from_origin(x0, y0, RES, RES),
        "compress": "DEFLATE",
    }
    base.update(profile)
    with rasterio.open(path, "w", **base) as dst:
        dst.write(stack.astype(base["dtype"]))


def _valid(values, nodata=-9999.0):
    v = np.asarray(values, dtype=np.float64).ravel()
    return v[np.isfinite(v) & (v != nodata)]


class _Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.rng = np.random.default_rng(17)

    def tearDown(self):
        self.tmp.cleanup()

    def assertStored(self, path, band, values, nodata=-9999.0):
        valid = _valid(values, nodata)
        with rasterio.open(path) as src:
            tags = src.tags(band)
            size = src.width * src.height
        self.assertEqual(float(tags["STATISTICS_MINIMUM"]), valid.min())
        self.assertEqual(float(tags["STATISTICS_MAXIMUM"]), valid.max())
        for key, value in (("MEAN", valid.mean()), ("STDDEV", valid.std())):
            self.assertAlmostEqual(float(tags[f"STATISTICS_{key}"]), value, places=10)
        self.assertAlmostEqual(
            float(tags["STATISTICS_VALID_PERCENT"]), 100.0 * valid.size / size
        )


class WriteBandStatisticsTests(_Case):
    def test_sentinel_and_nan_left_out_of_an_undeclared_file(self):
        data = _field(self.rng, 300, 280)
        path = os.path.join(self.dir, "ee.tif")
        _write(path, data)

        (tags,) = write_band_statistics(path, nodata=-9999)

        with rasterio.open(path) as src:
            self.assertIsNone(src.nodata)
            np.testing.assert_array_equal(src.read(1), data)
        self.assertStored(path, 1, data)
        self.assertEqual(float(tags["STATISTICS_MINIMUM"]), _valid(data).min())
        self.assertEqual(sorted(os.listdir(self.dir)), ["ee.tif"])

    def test_declared_nodata_wins(self):
        data = _field(self.rng, 60, 60)
        data[:10] = 0.0
        path = os.path.join(self.dir, "zero.tif")
        _write(path, data, nodata=0.0)

        write_band_statistics(path, nodata=-9999)

        with rasterio.open(path) as src:
            self.assertEqual(src.nodata, 0.0)
        self.assertStored(path, 1, data, nodata=0.0)

    def test_each_band_has_its_own(self):
        stack = np.stack([_field(self.rng, 80, 90) + k for k in range(3)])
        path = os.path.join(self.dir, "stack.tif")
        _write(path, stack)

        write_band_statistics(path, nodata=-9999)

        for band in (1, 2, 3):
            self.assertStored(path, band, stack[band - 1])

    def test_band_without_data_gets_no_tags(self):
        path = os.path.join(self.dir, "empty.tif")
        _write(path, np.full((40, 40), -9999.0))

        self.assertEqual(write_band_statistics(path, nodata=-9999), [{}])
        with rasterio.open(path) as src:
            self.assertNotIn("STATISTICS_MINIMUM", src.tags(1))

    def test_sparse_mosaic_reads_stored_blocks_only(self):
        """Two small clusters far apart: a few blocks in a large extent."""
        tiles = []
        for i, (x0, y0) in enumerate(
            [(500000.0, 5600000.0), (500000.0 + 6000 * RES, 5600000.0 - 3000 * RES)]
        ):
            path = os.path.join(self.dir, f"tile_{i}.tif")
            _write(path, _field(self.rng, 120, 150), x0, y0)
            tiles.append(path)
        mosaic = os.path.join(self.dir, "mosaic.tif")
        stream_mosaic_to_geotiff(
            tiles,
            mosaic,
            nodata=-9999,
            build_overviews=False,
            compress=True,
            dst_dtype="float32",
        )

        seen: list[bool] = []
        real = crs_utils._block_stored

        def counting(*args):
            seen.append(real(*args))
            return seen[-1]

        with mock.patch.object(crs_utils, "_block_stored", side_effect=counting):
            write_band_statistics(mosaic)
        self.assertGreater(sum(seen), 0)
        self.assertLess(sum(seen), len(seen) // 20)

        with rasterio.open(mosaic) as src:
            full = src.read(1)
        self.assertStored(mosaic, 1, full)

        # GDAL's own statistics call returns the stored tags.
        with rasterio.open(mosaic) as src, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gdal = src.statistics(1, approx=True)
        valid = _valid(full)
        self.assertEqual((gdal.min, gdal.max), (valid.min(), valid.max()))


class CacheTileStatisticsTests(_Case):
    def test_extracted_band_stores_the_statistics_of_its_pixels(self):
        stack = np.stack([_field(self.rng, 70, 70) * (k + 1) for k in range(2)])
        src = os.path.join(self.dir, "ee.tif")
        _write(src, stack)
        dst = os.path.join(self.dir, "tile_0.tif")

        ndvi_mod._extract_band(src, 2, dst, dtype="float32")

        with rasterio.open(dst) as out:
            self.assertIsNone(out.nodata)
        self.assertStored(dst, 1, stack[1].astype(np.float32))

    def test_salvaged_tile_stores_the_statistics_of_its_pixels(self):
        values = _field(self.rng, 50, 40).astype(np.float32)
        values[np.isnan(values)] = -9999.0
        path = os.path.join(self.dir, "tile_3.tif")

        ndvi_mod._write_tile(path, 500000.0, 5600000.0, values, RES, CRS)

        with rasterio.open(path) as out:
            self.assertIsNone(out.nodata)
            np.testing.assert_array_equal(out.read(1), values)
        self.assertStored(path, 1, values)


if __name__ == "__main__":
    unittest.main()
