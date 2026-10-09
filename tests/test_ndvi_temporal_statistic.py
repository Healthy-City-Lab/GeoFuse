"""NDVI composites take a temporal statistic: median (default), mean or max.

Earth Engine is replaced by a small fake: the image collection is an analytic
NDVI stack over planar coordinates with cloud gaps (NaN), its ``median`` /
``mean`` / ``max`` are numpy's NaN-aware reductions, and the export writes the
requested image onto the engine's snap grid. That drives the real engine end
to end — single-tile and tiled paths, band stacking and splitting, the
per-statistic tile cache, resume keys and sidecars — and every output is
checked pixel by pixel against numpy on the same stack.
"""

import json
import math
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.transform import from_origin
import shapely
from shapely.geometry import box
from shapely.ops import transform as shp_transform

from geofuse import ndvi
from geofuse.persistence.caches import NdviTileCache
from geofuse.vector_io import geometry_sha256

GRID = CRS.from_epsg(32611)
RES = 20
CRS_OVERRIDE = ("EPSG:32611", GRID, 0.0, "UTM zone 11N")
_TO_GRID = Transformer.from_crs("EPSG:4326", GRID, always_xy=True).transform
N_IMAGES = 5


def _aoi(km: float):
    """A square study area of side ``km`` near Calgary, in EPSG:4326."""
    lon, lat = -114.07, 51.04
    dlon = km * 1000 / (111_320 * math.cos(math.radians(lat)))
    dlat = km * 1000 / 110_950
    return gpd.GeoDataFrame(
        geometry=[box(lon, lat, lon + dlon, lat + dlat)], crs="EPSG:4326"
    )


# ────────────────────────────────────────────────────────────────────
# Fake Earth Engine
# ────────────────────────────────────────────────────────────────────


def _stack(X, Y):
    """``N_IMAGES`` cloud-masked NDVI images at planar ``(X, Y)``."""
    k = np.arange(N_IMAGES)[:, None, None]
    v = 0.35 + 0.25 * np.sin(X / 310.0 + k) * np.cos(Y / 270.0 - 0.5 * k) + 0.01 * k
    cloudy = np.sin(X / 97.0 + 2 * k) * np.cos(Y / 83.0 + k) > 0.55
    v = np.where(cloudy, np.nan, v)
    never_clear = (np.mod(X, 600) < 60) & (np.mod(Y, 600) < 60)
    return np.where(never_clear[None], np.nan, v)


def _reduce(stat, X, Y):
    """NaN-aware statistic over the stack; NaN where no image is clear."""
    masked = np.ma.masked_invalid(_stack(X, Y))
    reducer = {"median": np.ma.median, "mean": np.ma.mean, "max": np.ma.max}[stat]
    return np.ma.filled(reducer(masked, axis=0).astype(np.float64), np.nan)


def _expected(stat, X, Y):
    """The composite the engine should write: NaN-aware statistic, sentinel."""
    out = _reduce(stat, X, Y)
    return np.where(np.isnan(out), -9999.0, out)


class FakeImage:
    """The ``ee.Image`` calls the engine makes, evaluated lazily at (X, Y)."""

    def __init__(self, fn, bands=("NDVI",), dtype="float64"):
        self.fn, self.bands, self.dtype = fn, tuple(bands), dtype

    def evaluate(self, X, Y):
        out = np.asarray(self.fn(X, Y), dtype=np.float64)
        return out[None] if out.ndim == 2 else out

    def rename(self, *names):
        return FakeImage(self.fn, names, self.dtype)

    def unmask(self, value):
        fn = self.fn
        return FakeImage(
            lambda X, Y: np.where(np.isnan(fn(X, Y)), value, fn(X, Y)),
            self.bands,
            self.dtype,
        )

    def clip(self, _region):
        return self

    def toFloat(self):
        return FakeImage(self.fn, self.bands, "float32")

    @staticmethod
    def cat(images):
        return FakeImage(
            lambda X, Y: np.concatenate([im.evaluate(X, Y) for im in images]),
            sum((im.bands for im in images), ()),
            images[0].dtype,
        )


class FakeCollection:
    def select(self, _band):
        return self

    def median(self):
        return FakeImage(lambda X, Y: _reduce("median", X, Y))

    def mean(self):
        return FakeImage(lambda X, Y: _reduce("mean", X, Y))

    def max(self):
        return FakeImage(lambda X, Y: _reduce("max", X, Y))


class FakeEarthEngine:
    """Patches the engine's Earth Engine seams and records every export.

    Like Earth Engine, a float32 request writes ``-inf`` where a pixel centre
    falls outside the export region, while float64 writes the data there.
    Neighbouring tiles overlap by a pixel, so a float32 request would put
    ``-inf`` into the mosaic.
    """

    def __init__(self, aoi):
        self.aoi = aoi
        self.exports: list[tuple[tuple[str, ...], str]] = []

    def export(self, image, filename, *, crs, crs_transform, region, timeout=300):
        self.exports.append((image.bands, image.dtype))
        region_grid = shp_transform(_TO_GRID, region)
        minx, miny, maxx, maxy = region_grid.bounds
        res = float(crs_transform[0])
        x0, x1 = math.floor(minx / res) * res, math.ceil(maxx / res) * res
        y0, y1 = math.floor(miny / res) * res, math.ceil(maxy / res) * res
        w, h = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
        X, Y = np.meshgrid(
            x0 + (np.arange(w) + 0.5) * res, y1 - (np.arange(h) + 0.5) * res
        )
        data = image.evaluate(X, Y).astype(image.dtype)
        if image.dtype == "float32":
            data[:, ~shapely.contains_xy(region_grid, X, Y)] = -np.inf
        profile = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": data.shape[0],
            "dtype": image.dtype,
            "crs": GRID,
            "transform": from_origin(x0, y1, res, res),
            "compress": "DEFLATE",
        }
        with rasterio.open(filename, "w", **profile) as dst:
            dst.write(data)

    def __enter__(self):
        chain = mock.MagicMock()
        chain.filterBounds.return_value = chain
        chain.filterDate.return_value = chain
        chain.filter.return_value = chain
        chain.size.return_value.getInfo.return_value = N_IMAGES
        self._patches = [
            mock.patch.object(ndvi, "ee"),
            mock.patch.object(ndvi, "shapely_to_ee_geometry", side_effect=lambda g: g),
            mock.patch.object(ndvi, "_export_ee_image_to_tif", side_effect=self.export),
        ]
        fake_ee = self._patches[0].start()
        for p in self._patches[1:]:
            p.start()
        fake_ee.ImageCollection.return_value = chain
        fake_ee.FeatureCollection.return_value.geometry.return_value = (
            self.aoi.geometry.iloc[0]
        )
        fake_ee.Image.cat.side_effect = FakeImage.cat
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()


def _engine(cache_dir):
    eng = ndvi.NDVIEngine.__new__(ndvi.NDVIEngine)
    eng.tile_cache = NdviTileCache(cache_dir)
    eng.get_collection = lambda aoi, s, e, cmax=10: FakeCollection()
    return eng


def _run(eng, aoi, folder, reducers, **kw):
    kw.setdefault("write_geojson", False)
    return eng.download_and_process(
        geometry=aoi,
        start_date="2023-06-01",
        end_date="2023-09-30",
        output_name="site",
        resolution=RES,
        folder=folder,
        crs_override=CRS_OVERRIDE,
        temporal_reducers=reducers,
        **kw,
    )


def _read_with_centres(path):
    with rasterio.open(path) as src:
        arr = src.read(1)
        t = src.transform
        dtype = src.dtypes[0]
    cols, rows = np.meshgrid(np.arange(arr.shape[1]), np.arange(arr.shape[0]))
    X = t.c + (cols + 0.5) * t.a
    Y = t.f + (rows + 0.5) * t.e
    return arr, X, Y, dtype


# ────────────────────────────────────────────────────────────────────
# Names, keys and result combination
# ────────────────────────────────────────────────────────────────────


class TemporalReducerNameTests(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(ndvi.normalize_temporal_reducers(), ("median",))
        self.assertEqual(
            ndvi.normalize_temporal_reducers(["max", "MEDIAN", "max", " mean "]),
            ("median", "mean", "max"),
        )
        self.assertEqual(ndvi.normalize_temporal_reducers("max"), ("max",))
        with self.assertRaises(ValueError):
            ndvi.normalize_temporal_reducers(["p90"])
        with self.assertRaises(ValueError):
            ndvi.normalize_temporal_reducers([])

    def test_output_names(self):
        self.assertEqual(ndvi.ndvi_output_name("a_2015", "median"), "a_2015")
        self.assertEqual(ndvi.ndvi_output_name("a_2015", "max"), "a_2015_max")

    def test_resume_keys(self):
        """Median keys are unchanged; every statistic gets its own key."""
        import hashlib

        aoi = _aoi(1.0)
        args = (aoi, "2023-06-01", "2023-09-30", 10, 10, "COPERNICUS/S2_SR_HARMONIZED")
        h = hashlib.sha256()
        h.update(b"planar_v1|")
        h.update(geometry_sha256(aoi).encode())
        h.update(b"|2023-06-01|2023-09-30|10|10|")
        h.update(b"COPERNICUS/S2_SR_HARMONIZED")
        before = h.hexdigest()[:16]

        keys = {
            r: ndvi._compute_resume_key(*args, temporal_reducer=r)
            for r in ndvi.TEMPORAL_REDUCERS
        }
        self.assertEqual(ndvi._compute_resume_key(*args), before)
        self.assertEqual(keys["median"], before)
        self.assertEqual(len(set(keys.values())), 3)


class CombineResultsTests(unittest.TestCase):
    ok = {"status": "success", "tif": "a.tif"}

    def test_single_statistic_passes_through(self):
        err = {"status": "error", "message": "Mosaic failed: x"}
        out = ndvi._combine_reducer_results({"median": err})
        self.assertEqual(out["message"], "Mosaic failed: x")
        self.assertEqual(set(out["reducers"]), {"median"})

    def test_any_failure_fails_the_run_and_names_it(self):
        out = ndvi._combine_reducer_results(
            {"median": self.ok, "max": {"status": "error", "message": "boom"}}
        )
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["message"], "max: boom")
        self.assertEqual(out["tif"], "a.tif")

    def test_cancel_wins(self):
        out = ndvi._combine_reducer_results(
            {
                "median": {"status": "error", "message": "x"},
                "max": {"status": "cancelled", "message": "Cancelled by user"},
            }
        )
        self.assertEqual(out["status"], "cancelled")


# ────────────────────────────────────────────────────────────────────
# End to end through the engine
# ────────────────────────────────────────────────────────────────────


class _EngineCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = os.path.join(self.tmp.name, "cache")
        self.out = os.path.join(self.tmp.name, "out")
        self._engines = []

    def tearDown(self):
        for eng in self._engines:
            eng.tile_cache.close()
        self.tmp.cleanup()

    def engine(self):
        eng = _engine(self.cache)
        self._engines.append(eng)
        return eng

    def assertComposite(self, path, stat):
        arr, X, Y, dtype = _read_with_centres(path)
        np.testing.assert_array_equal(arr, _expected(stat, X, Y).astype(dtype))
        self.assertTrue((arr == -9999).any(), "sentinel pixels expected")
        self.assertTrue((arr != -9999).any())

    def sidecar(self, name):
        with open(os.path.join(self.out, f"{name}_ndvi.json")) as f:
            return json.load(f)


class SingleTileTests(_EngineCase):
    def test_one_export_three_files(self):
        aoi = _aoi(0.6)
        with FakeEarthEngine(aoi) as fake:
            res = _run(
                self.engine(),
                aoi,
                self.out,
                ["max", "median", "mean"],
                write_geopackage=True,
            )

        self.assertEqual(res["status"], "success")
        self.assertEqual(fake.exports, [(("median", "mean", "max"), "float64")])
        self.assertEqual(set(res["reducers"]), {"median", "mean", "max"})
        self.assertEqual(res["tif"], os.path.join(self.out, "site_ndvi.tif"))
        for stat, name in (("median", "site"), ("mean", "site_mean"), ("max", "site_max")):
            tif = os.path.join(self.out, f"{name}_ndvi.tif")
            self.assertEqual(res["reducers"][stat]["tif"], tif)
            self.assertComposite(tif, stat)
            self.assertTrue(os.path.isfile(os.path.join(self.out, f"{name}_ndvi.gpkg")))
            self.assertEqual(self.sidecar(name)["temporal_reducer"], stat)
        self.assertEqual(sorted(os.listdir(self.out)), sorted(
            f"{n}_ndvi.{ext}"
            for n in ("site", "site_mean", "site_max")
            for ext in ("tif", "gpkg", "json")
        ))

    def test_matches_a_single_statistic_run(self):
        """A statistic's file holds what a run of that statistic alone writes."""
        aoi = _aoi(0.6)
        alone = os.path.join(self.tmp.name, "alone")
        with FakeEarthEngine(aoi):
            _run(self.engine(), aoi, alone, "median")
            _run(self.engine(), aoi, self.out, ["median", "max"])
        with rasterio.open(os.path.join(alone, "site_ndvi.tif")) as a, rasterio.open(
            os.path.join(self.out, "site_ndvi.tif")
        ) as b:
            self.assertEqual(a.dtypes, b.dtypes)
            self.assertEqual(a.transform, b.transform)
            np.testing.assert_array_equal(a.read(1), b.read(1))


class TiledCacheTests(_EngineCase):
    """Per-statistic workspaces; one request per tile for what is missing."""

    def setUp(self):
        super().setUp()
        self.aoi = _aoi(2.2)

    def _run(self, reducers, progress=None):
        with FakeEarthEngine(self.aoi) as fake:
            res = _run(
                self.engine(),
                self.aoi,
                self.out,
                reducers,
                max_tile_size_km=1,
                ndvi_progress_callback=progress,
            )
        return res, fake.exports

    def _workspaces(self):
        return sorted(
            d
            for d in os.listdir(self.cache)
            if os.path.isdir(os.path.join(self.cache, d))
        )

    def test_shared_download_then_partial_and_full_reuse(self):
        seen: list[float] = []
        res, exports = self._run(
            ["median", "max"], progress=lambda d: seen.append(d["sub_progress"])
        )
        self.assertEqual(res["status"], "success")
        n_tiles = res["meta"]["tiles_total"]
        self.assertGreater(n_tiles, 3)
        self.assertEqual(exports, [(("median", "max"), "float64")] * n_tiles)
        self.assertEqual(seen, sorted(seen))
        self.assertLessEqual(max(seen), 0.99)
        self.assertComposite(os.path.join(self.out, "site_ndvi.tif"), "median")
        self.assertComposite(os.path.join(self.out, "site_max_ndvi.tif"), "max")

        keys = {r: self.sidecar(n)["resume_key"] for r, n in (("median", "site"), ("max", "site_max"))}
        self.assertEqual(self._workspaces(), sorted(keys.values()))
        for key in keys.values():
            tiles = os.listdir(os.path.join(self.cache, key))
            self.assertEqual(len(tiles), n_tiles)
            with rasterio.open(os.path.join(self.cache, key, tiles[0])) as t:
                self.assertEqual((t.count, t.dtypes[0]), (1, "float32"))
        self.assertEqual(self.engine().tile_cache.stats()["entries"], 2)

        # Adding mean downloads mean only, one single-band request per tile.
        res, exports = self._run(["median", "mean", "max"])
        self.assertEqual(res["status"], "success")
        self.assertEqual(exports, [(("NDVI",), "float64")] * n_tiles)
        self.assertComposite(os.path.join(self.out, "site_mean_ndvi.tif"), "mean")
        self.assertEqual(res["reducers"]["max"]["meta"]["tiles_resumed"], n_tiles)
        self.assertEqual(res["reducers"]["mean"]["meta"]["tiles_resumed"], 0)

        # Everything cached: no request at all.
        res, exports = self._run(["max"])
        self.assertEqual((res["status"], exports), ("success", []))
        self.assertComposite(os.path.join(self.out, "site_max_ndvi.tif"), "max")

    def test_float64_tiles_from_before_are_reused(self):
        res, _ = self._run("median")
        key = res["meta"]["resume_key"]
        ws = os.path.join(self.cache, key)
        # Rewrite the cached tiles in Earth Engine's float64 encoding, as
        # caches written before compact tiles hold them.
        for name in os.listdir(ws):
            path = os.path.join(ws, name)
            with rasterio.open(path) as src:
                data = src.read(1).astype("float64")
                profile = src.profile
            profile.update(dtype="float64", predictor=1)
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(data, 1)
        first = _read_with_centres(os.path.join(self.out, "site_ndvi.tif"))[0]

        res, exports = self._run(["median", "max"])

        self.assertEqual(exports, [(("NDVI",), "float64")] * res["meta"]["tiles_total"])
        np.testing.assert_array_equal(
            _read_with_centres(os.path.join(self.out, "site_ndvi.tif"))[0], first
        )
        self.assertComposite(os.path.join(self.out, "site_max_ndvi.tif"), "max")

    def test_failed_tile_is_a_gap_for_every_statistic_it_carried(self):
        with FakeEarthEngine(self.aoi) as fake:
            real = fake.export

            def flaky(image, filename, **kw):
                if os.path.basename(filename).startswith("tile_0."):
                    raise RuntimeError("HTTP 500")
                real(image, filename, **kw)

            ndvi._export_ee_image_to_tif.side_effect = flaky
            with mock.patch("geofuse.jobs.time.sleep"):
                res = _run(
                    self.engine(),
                    self.aoi,
                    self.out,
                    ["median", "max"],
                    max_tile_size_km=1,
                )
        self.assertEqual(res["status"], "success")
        for stat in ("median", "max"):
            meta = res["reducers"][stat]["meta"]
            self.assertEqual(meta["tiles_failed"], 1)
            self.assertEqual(len(meta["failed_tile_refs"]), 1)


if __name__ == "__main__":
    unittest.main()
