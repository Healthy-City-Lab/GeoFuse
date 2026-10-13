"""NDVI cache tiles are compact without changing any output.

Earth Engine returns NDVI as float64, but every mosaic the engine builds from the
tile cache is float32. Storing cache tiles as float32 with DEFLATE and the
floating-point predictor therefore loses nothing that reaches an output, and a
tile takes under half the bytes.
"""

import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import rasterio
from rasterio.transform import from_origin

from geofuse import ndvi as ndvi_mod
from geofuse.crs_utils import stream_mosaic_to_geotiff

CRS = "EPSG:32611"
RES = 10.0


def _ndvi_field(rng, h, w):
    """Smooth NDVI surface with full float64 mantissas and a cloud gap."""
    yy, xx = np.mgrid[0:h, 0:w]
    field = 0.35 + 0.3 * np.sin(xx / 17.0) * np.cos(yy / 23.0)
    field = field + rng.normal(0.0, 0.02, (h, w))
    field[h // 3 : h // 3 + 5, w // 4 : w // 4 + 9] = -9999.0
    return field.astype(np.float64)


def _write_ee_tile(path, data, x0, y0):
    """A tile encoded the way Earth Engine delivers it: float64, DEFLATE,
    no predictor, no nodata tag."""
    count = 1 if data.ndim == 2 else data.shape[0]
    stack = data[None] if data.ndim == 2 else data
    profile = {
        "driver": "GTiff",
        "height": stack.shape[1],
        "width": stack.shape[2],
        "count": count,
        "dtype": "float64",
        "crs": CRS,
        "transform": from_origin(x0, y0, RES, RES),
        "compress": "DEFLATE",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(stack)


class ExtractBandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.rng = np.random.default_rng(3)

    def tearDown(self):
        self.tmp.cleanup()

    def test_float32_tile_is_the_cast_and_smaller(self):
        data = _ndvi_field(self.rng, 300, 300)
        src = os.path.join(self.dir, "ee.tif")
        dst = os.path.join(self.dir, "tile_0.tif")
        _write_ee_tile(src, data, 500000.0, 5600000.0)

        ndvi_mod._extract_band(src, 1, dst, dtype="float32")

        with rasterio.open(dst) as out, rasterio.open(src) as ref:
            self.assertEqual(out.dtypes[0], "float32")
            self.assertEqual(out.crs, ref.crs)
            self.assertEqual(out.transform, ref.transform)
            self.assertEqual(out.profile.get("compress"), "deflate")
            np.testing.assert_array_equal(out.read(1), data.astype(np.float32))
        self.assertLess(os.path.getsize(dst), 0.6 * os.path.getsize(src))
        self.assertFalse(os.path.exists(dst + ".part"))

    def test_source_dtype_kept_by_default(self):
        data = _ndvi_field(self.rng, 64, 64)
        src = os.path.join(self.dir, "ee.tif")
        dst = os.path.join(self.dir, "out.tif")
        _write_ee_tile(src, data, 0.0, 640.0)

        ndvi_mod._extract_band(src, 1, dst)

        with rasterio.open(dst) as out:
            self.assertEqual(out.dtypes[0], "float64")
            np.testing.assert_array_equal(out.read(1), data)

    def test_picks_the_requested_band(self):
        stack = np.stack([_ndvi_field(self.rng, 50, 60) for _ in range(3)])
        src = os.path.join(self.dir, "ee.tif")
        _write_ee_tile(src, stack, 0.0, 500.0)

        for band in (1, 2, 3):
            dst = os.path.join(self.dir, f"b{band}.tif")
            ndvi_mod._extract_band(src, band, dst, dtype="float32")
            with rasterio.open(dst) as out:
                self.assertEqual(out.count, 1)
                np.testing.assert_array_equal(
                    out.read(1), stack[band - 1].astype(np.float32)
                )


class MosaicIdentityTests(unittest.TestCase):
    """The mosaic of compact tiles equals the mosaic of Earth Engine's tiles."""

    def test_mosaic_identical(self):
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as d:
            originals, compact = [], []
            for i, (col, row) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
                x0 = 400000.0 + col * 120 * RES
                y0 = 5500000.0 - row * 100 * RES
                path = os.path.join(d, f"ee_{i}.tif")
                _write_ee_tile(path, _ndvi_field(rng, 100, 120), x0, y0)
                originals.append(path)
                small = os.path.join(d, f"tile_{i}.tif")
                ndvi_mod._extract_band(path, 1, small, dtype="float32")
                compact.append(small)

            out_a = os.path.join(d, "mosaic_ee.tif")
            out_b = os.path.join(d, "mosaic_compact.tif")
            for tiles, out in ((originals, out_a), (compact, out_b)):
                stream_mosaic_to_geotiff(
                    tiles,
                    out,
                    nodata=-9999,
                    build_overviews=False,
                    compress=True,
                    dst_dtype="float32",
                )
            with rasterio.open(out_a) as a, rasterio.open(out_b) as b:
                self.assertEqual(a.transform, b.transform)
                np.testing.assert_array_equal(a.read(1), b.read(1))


class _Image:
    """Stand-in for an ``ee.Image`` in Earth Engine's float64 encoding."""

    dtype = "float64"


class DownloadOneTileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.engine = ndvi_mod.NDVIEngine.__new__(ndvi_mod.NDVIEngine)
        self.requested = []

    def tearDown(self):
        self.tmp.cleanup()

    def _fake_export(self, image, region, out_path, **_kw):
        self.requested.append(image.dtype)
        data = _ndvi_field(np.random.default_rng(5), 40, 40)
        _write_ee_tile(out_path, data, 0.0, 400.0)

    def _run(self):
        spec = {"tile_idx": 7, "cluster_id": 0, "tile_geom_4326": None}
        return self.engine._download_one_tile(
            spec,
            {"median": _Image()},
            ["median"],
            {"median": self.dir},
            CRS,
            [10.0, 0, 0, 0, -10.0, 0],
            None,
        )

    def test_download_as_is_and_compact_tile(self):
        self.engine._export_region_adaptive = self._fake_export
        res = self._run()

        self.assertIsNone(res["error"])
        self.assertEqual(self.requested, ["float64"])
        self.assertEqual(os.listdir(self.dir), ["tile_7.tif"])
        with rasterio.open(res["tile_files"]["median"]) as t:
            self.assertEqual(t.dtypes[0], "float32")

    def test_failure_leaves_no_files(self):
        def _boom(*_a, **_kw):
            raise RuntimeError("HTTP 500")

        self.engine._export_region_adaptive = _boom
        res = self._run()

        self.assertEqual(res["tile_files"], {})
        self.assertIn("HTTP 500", res["error"])
        self.assertEqual(os.listdir(self.dir), [])


if __name__ == "__main__":
    unittest.main()
