"""NDVI acquisition options: water mask, cloud handling, sensors, reflectance, 2012.

Every option is the user's choice and defaults to the behaviour before it
existed: water stays in the composite, cloudy scenes are dropped and cloudy
pixels masked, Landsat merges sensors 5 / 7 / 8 / 9, reflectance is surface
reflectance, and a 2012 window is queried as asked. The engine runs end to end
on a numeric fake of Earth Engine (``_fake_earth_engine``), and outputs are
checked pixel by pixel against NDVI computed by hand from the scene bands.
"""

import json
import math
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (ROOT, os.path.dirname(__file__)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import geopandas as gpd
import numpy as np
import rasterio
from _fake_earth_engine import GRID, FakeEarthEngine, Image
from pyproj import Transformer
from shapely.geometry import box

from geofuse import ndvi
from geofuse.persistence.caches import NdviTileCache

E = ndvi.NDVIEngine
CRS_OVERRIDE = ("EPSG:32611", GRID, 0.0, "UTM zone 11N")
RES = 20


def _aoi(km=0.6):
    lon, lat = -114.07, 51.04
    dlon = km * 1000 / (111_320 * math.cos(math.radians(lat)))
    dlat = km * 1000 / 110_950
    return gpd.GeoDataFrame(
        geometry=[box(lon, lat, lon + dlon, lat + dlat)], crs="EPSG:4326"
    )


AOI = _aoi()
_X0, _, _X1, _ = Transformer.from_crs(
    "EPSG:4326", GRID, always_xy=True
).transform_bounds(*AOI.total_bounds)
THIRD_1 = _X0 + (_X1 - _X0) / 3
THIRD_2 = _X0 + 2 * (_X1 - _X0) / 3


def _west(X, Y):
    return X < THIRD_1


# ────────────────────────────────────────────────────────────────────
# Scenes, built from reflectances so NDVI is known
# ────────────────────────────────────────────────────────────────────


def _ndvi(red, nir):
    return (nir - red) / (nir + red)


def landsat_sr(date, cloud, red, nir, *, tm=False, qa=0):
    dn = lambda r: (r + 0.2) / 0.0000275  # noqa: E731
    bands = (
        {"SR_B3": dn(red), "SR_B4": dn(nir)}
        if tm
        else {"SR_B4": dn(red), "SR_B5": dn(nir)}
    )
    bands["QA_PIXEL"] = qa
    return Image(bands, {"date": date, "CLOUD_COVER": cloud})


def landsat_toa(date, cloud, red, nir, *, tm=False, qa=0):
    bands = {"B3": red, "B4": nir} if tm else {"B4": red, "B5": nir}
    bands["QA_PIXEL"] = qa
    return Image(bands, {"date": date, "CLOUD_COVER": cloud})


def s2(date, cloud, red, nir, **masks):
    bands = {"B4": red / 1e-4, "B8": nir / 1e-4, **masks}
    return Image(bands, {"date": date, "CLOUDY_PIXEL_PERCENTAGE": cloud})


CLOUD_BIT = lambda X, Y: np.where(_west(X, Y), 1 << 3, 0)  # noqa: E731


def _tm_harmonised(red, nir):
    rs, ri = E._ETM_TO_OLI["red"]
    ns, ni = E._ETM_TO_OLI["nir"]
    return _ndvi(red * rs + ri, nir * ns + ni)


# ────────────────────────────────────────────────────────────────────
# Harness
# ────────────────────────────────────────────────────────────────────


class _Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.tmp.name, "out")
        self.engines = []

    def tearDown(self):
        for eng in self.engines:
            eng.tile_cache.close()
        self.tmp.cleanup()

    def run_engine(self, collections, images=None, *, start, end, name="site", **kw):
        eng = E.__new__(E)
        eng.tile_cache = NdviTileCache(os.path.join(self.tmp.name, "cache"))
        self.engines.append(eng)
        kw.setdefault("write_geojson", False)
        kw.setdefault("coverage_rescue", False)
        with FakeEarthEngine(AOI, collections, images) as fake:
            res = eng.download_and_process(
                geometry=AOI,
                start_date=start,
                end_date=end,
                output_name=name,
                resolution=RES,
                folder=self.out,
                crs_override=CRS_OVERRIDE,
                **kw,
            )
        self.fake = fake
        return res

    def raster(self, name="site"):
        with rasterio.open(os.path.join(self.out, f"{name}_ndvi.tif")) as src:
            arr = src.read(1).astype(np.float64)
            t = src.transform
        cols, rows = np.meshgrid(np.arange(arr.shape[1]), np.arange(arr.shape[0]))
        return arr, t.c + (cols + 0.5) * t.a, t.f + (rows + 0.5) * t.e

    def sidecar(self, name="site"):
        with open(os.path.join(self.out, f"{name}_ndvi.json")) as f:
            return json.load(f)


LANDSAT_SR = {
    E.LANDSAT8_COLLECTION_ID: [
        landsat_sr("2015-07-01", 5, 0.05, 0.40),
        landsat_sr("2015-07-17", 8, 0.05, 0.30, qa=CLOUD_BIT),
        landsat_sr("2015-08-02", 90, 0.10, 0.12),
    ]
}


class DefaultsTests(_Case):
    """Every option at its default reproduces the behaviour before G12."""

    def test_default_landsat_composite(self):
        res = self.run_engine(
            LANDSAT_SR,
            start="2015-06-01",
            end="2015-09-30",
            satellite="landsat",
            temporal_reducers="mean",
        )
        self.assertEqual(res["status"], "success", res.get("message"))
        arr, X, Y = self.raster("site_mean")
        a, b = _ndvi(0.05, 0.40), _ndvi(0.05, 0.30)
        expected = np.where(_west(X, Y), a, (a + b) / 2)
        np.testing.assert_allclose(arr, expected, rtol=1e-6)
        side = self.sidecar("site_mean")
        self.assertEqual(
            (
                side["water_mask"],
                side["cloud_handling"],
                side["reflectance"],
                side["year_substituted"],
            ),
            (False, "scene_filter", "surface", None),
        )
        self.assertEqual(side["ee_collection"], E.LANDSAT_MERGED_LABEL)
        self.assertTrue(all("T1_L2" in c for c in self.fake.queried))

    def test_default_resume_key_is_unchanged(self):
        args = (AOI, "2015-06-01", "2015-09-30", 10, 30, E.LANDSAT_MERGED_LABEL)
        self.assertEqual(
            ndvi._compute_resume_key(*args),
            ndvi._compute_resume_key(*args, temporal_reducer="median", variant=()),
        )


class WaterMaskTests(_Case):
    """Acceptance 2: occurrence ≥ 50 % masked, no record kept."""

    WATER = {
        ndvi.WATER_DATASET: Image(
            {
                "occurrence": lambda X, Y: np.select(
                    [X < THIRD_1, X < THIRD_2, Y > np.median(Y)],
                    [80.0, np.nan, 50.0],
                    20.0,
                )
            }
        )
    }

    def test_water_is_masked_only_when_asked(self):
        col = {E.LANDSAT8_COLLECTION_ID: [landsat_sr("2015-07-01", 5, 0.05, 0.40)]}
        for water in (False, True):
            res = self.run_engine(
                col,
                self.WATER,
                start="2015-06-01",
                end="2015-09-30",
                satellite="landsat",
                water_mask=water,
                name=f"w{int(water)}",
            )
            self.assertEqual(res["status"], "success", res.get("message"))
            arr, X, Y = self.raster(f"w{int(water)}")
            ndvi_v = _ndvi(0.05, 0.40)
            if not water:
                np.testing.assert_allclose(arr, ndvi_v, rtol=1e-6)
                continue
            east = X >= THIRD_2
            north = Y > np.median(Y)
            masked = (X < THIRD_1) | (east & north)
            self.assertTrue((arr[masked] == -9999).all())
            np.testing.assert_allclose(arr[~masked], ndvi_v, rtol=1e-6)
            self.assertTrue(((X >= THIRD_1) & (X < THIRD_2)).any())
            self.assertEqual(self.sidecar("w1")["water_occurrence_max"], 50)


class CloudHandlingTests(_Case):
    """Acceptance 3, plus the scene-only mode."""

    def _mean(self, scenes, mode, name):
        res = self.run_engine(
            {E.LANDSAT8_COLLECTION_ID: scenes},
            start="2015-06-01",
            end="2015-09-30",
            satellite="landsat",
            cloud_handling=mode,
            temporal_reducers="mean",
            cloud_max=20,
            name=name,
        )
        self.assertEqual(res["status"], "success", res.get("message"))
        return self.raster(f"{name}_mean")

    def test_pixel_only_keeps_cloudy_scenes_and_masks_cloudy_pixels(self):
        scenes = [
            landsat_sr("2015-07-01", 5, 0.05, 0.40),
            landsat_sr("2015-07-17", 90, 0.05, 0.20, qa=CLOUD_BIT),
        ]
        a, b = _ndvi(0.05, 0.40), _ndvi(0.05, 0.20)
        arr, X, Y = self._mean(scenes, "scene_filter", "sf")
        np.testing.assert_allclose(arr, a, rtol=1e-6)
        arr, X, Y = self._mean(scenes, "pixel_only", "po")
        np.testing.assert_allclose(
            arr, np.where(_west(X, Y), a, (a + b) / 2), rtol=1e-6
        )

    def test_scene_only_keeps_the_cloudy_pixels_of_kept_scenes(self):
        scenes = [
            landsat_sr("2015-07-01", 5, 0.05, 0.40, qa=CLOUD_BIT),
            landsat_sr("2015-07-17", 8, 0.05, 0.25),
            landsat_sr("2015-08-02", 90, 0.10, 0.12),
        ]
        a, c = _ndvi(0.05, 0.40), _ndvi(0.05, 0.25)
        arr, X, Y = self._mean(scenes, "scene_filter", "sf")
        np.testing.assert_allclose(
            arr, np.where(_west(X, Y), c, (a + c) / 2), rtol=1e-6
        )
        arr, X, Y = self._mean(scenes, "scene_only", "so")
        np.testing.assert_allclose(arr, (a + c) / 2, rtol=1e-6)

    def test_pixel_only_never_reports_every_scene_as_too_cloudy(self):
        scenes = [landsat_sr("2015-07-01", 95, 0.05, 0.40)]
        res = self.run_engine(
            {E.LANDSAT8_COLLECTION_ID: scenes},
            start="2015-06-01",
            end="2015-09-30",
            satellite="landsat",
            cloud_max=10,
            name="sf",
        )
        self.assertEqual(res["status"], "error")
        self.assertIn("exceeded the 10% cloud threshold", res["message"])
        res = self.run_engine(
            {E.LANDSAT8_COLLECTION_ID: scenes},
            start="2015-06-01",
            end="2015-09-30",
            satellite="landsat",
            cloud_max=10,
            cloud_handling="pixel_only",
            name="po",
        )
        self.assertEqual(res["status"], "success", res.get("message"))

    def test_sentinel2_masks_follow_the_same_modes(self):
        scl = lambda X, Y: np.where(_west(X, Y), 9, 4)  # noqa: E731 - 9 = cloud
        col = {E.EE_COLLECTION_ID: [s2("2020-07-01", 5, 0.05, 0.40, SCL=scl)]}
        for mode, west_masked in (("scene_filter", True), ("scene_only", False)):
            res = self.run_engine(
                col,
                start="2020-06-01",
                end="2020-09-30",
                cloud_handling=mode,
                name=mode,
            )
            self.assertEqual(res["status"], "success", res.get("message"))
            arr, X, Y = self.raster(mode)
            west = _west(X, Y)
            self.assertEqual(bool((arr[west] == -9999).all()), west_masked)
            np.testing.assert_allclose(arr[~west], _ndvi(0.05, 0.40), rtol=1e-6)


class SensorSetTests(_Case):
    """Acceptance 4: no striped Landsat 7, no Landsat 9; pre-2003 Landsat 7 kept."""

    COLLECTIONS = {
        E.LANDSAT5_COLLECTION_ID: [landsat_sr("2002-07-10", 5, 0.05, 0.30, tm=True)],
        E.LANDSAT7_COLLECTION_ID: [
            landsat_sr("2002-07-20", 5, 0.05, 0.35, tm=True),
            landsat_sr("2005-07-20", 5, 0.05, 0.10, tm=True),
        ],
        E.LANDSAT8_COLLECTION_ID: [landsat_sr("2015-07-01", 5, 0.05, 0.40)],
        E.LANDSAT9_COLLECTION_ID: [landsat_sr("2022-07-01", 5, 0.05, 0.45)],
    }

    def test_collections_kept_by_each_set(self):
        eng = E.__new__(E)
        with FakeEarthEngine(AOI, self.COLLECTIONS):
            n_all = eng._landsat_base(AOI, "2001-01-01", "2023-12-31").size().getInfo()
            n_clean = (
                eng._landsat_base(
                    AOI, "2001-01-01", "2023-12-31", sensor_set="no_slc_off"
                )
                .size()
                .getInfo()
            )
        self.assertEqual((n_all, n_clean), (5, 3))

    def test_a_2002_composite_uses_tm_and_pre_slc_etm(self):
        res = self.run_engine(
            self.COLLECTIONS,
            start="2002-06-01",
            end="2002-09-30",
            satellite="landsat_no_slc_off",
            temporal_reducers="mean",
        )
        self.assertEqual(res["status"], "success", res.get("message"))
        arr, _, _ = self.raster("site_mean")
        expected = (_tm_harmonised(0.05, 0.30) + _tm_harmonised(0.05, 0.35)) / 2
        np.testing.assert_allclose(arr, expected, rtol=1e-6)
        self.assertNotIn(E.LANDSAT9_COLLECTION_ID, self.fake.queried)
        self.assertEqual(
            self.sidecar("site_mean")["ee_collection"],
            "LANDSAT/LT05+LE07(pre-2003-06)+LC08/C02/T1_L2",
        )

    def test_slc_off_years_have_no_scenes(self):
        res = self.run_engine(
            self.COLLECTIONS,
            start="2005-06-01",
            end="2005-09-30",
            satellite="landsat_no_slc_off",
        )
        self.assertEqual(res["status"], "error")
        res = self.run_engine(
            self.COLLECTIONS,
            start="2005-06-01",
            end="2005-09-30",
            satellite="landsat",
            name="all",
        )
        self.assertEqual(res["status"], "success", res.get("message"))


class ReflectanceTests(_Case):
    """Acceptance 5: TOA collections and bands, no harmonisation."""

    def test_landsat_toa(self):
        cols = {
            E.LANDSAT_TOA_COLLECTION_IDS[0]: [
                landsat_toa("2009-07-01", 5, 0.06, 0.32, tm=True)
            ],
            E.LANDSAT5_COLLECTION_ID: [
                landsat_sr("2009-07-01", 5, 0.20, 0.25, tm=True)
            ],
        }
        res = self.run_engine(
            cols,
            start="2009-06-01",
            end="2009-09-30",
            satellite="landsat",
            reflectance="toa",
        )
        self.assertEqual(res["status"], "success", res.get("message"))
        arr, _, _ = self.raster()
        np.testing.assert_allclose(arr, _ndvi(0.06, 0.32), rtol=1e-6)
        self.assertTrue(all("T1_TOA" in c for c in self.fake.queried))
        self.assertEqual(
            self.sidecar()["ee_collection"], "LANDSAT/LT05+LE07+LC08+LC09/C02/T1_TOA"
        )
        self.assertEqual(self.sidecar()["reflectance"], "toa")

    def test_sentinel2_toa_masks_with_qa60(self):
        qa60 = lambda X, Y: np.where(_west(X, Y), 1 << 10, 0)  # noqa: E731
        cols = {E.S2_TOA_COLLECTION_ID: [s2("2020-07-01", 5, 0.07, 0.33, QA60=qa60)]}
        res = self.run_engine(
            cols, start="2020-06-01", end="2020-09-30", reflectance="toa"
        )
        self.assertEqual(res["status"], "success", res.get("message"))
        arr, X, Y = self.raster()
        west = _west(X, Y)
        self.assertTrue((arr[west] == -9999).all())
        np.testing.assert_allclose(arr[~west], _ndvi(0.07, 0.33), rtol=1e-6)
        self.assertEqual(
            self.fake.queried, [E.S2_TOA_COLLECTION_ID] * len(self.fake.queried)
        )


class Year2012Tests(_Case):
    """Acceptance 6: a 2012 window is filled from the same days of 2013."""

    COLLECTIONS = {E.LANDSAT8_COLLECTION_ID: [landsat_sr("2013-07-01", 5, 0.05, 0.40)]}

    def test_substitution(self):
        res = self.run_engine(
            self.COLLECTIONS,
            start="2012-06-01",
            end="2012-09-30",
            satellite="landsat",
        )
        self.assertEqual(res["status"], "error")
        res = self.run_engine(
            self.COLLECTIONS,
            start="2012-06-01",
            end="2012-09-30",
            satellite="landsat",
            substitute_2012=True,
            name="sub",
        )
        self.assertEqual(res["status"], "success", res.get("message"))
        side = self.sidecar("sub")
        self.assertEqual(
            (side["start_date"], side["end_date"]), ("2012-06-01", "2012-09-30")
        )
        self.assertEqual(
            (side["used_start_date"], side["used_end_date"]),
            ("2013-06-01", "2013-09-30"),
        )
        self.assertEqual(side["year_substituted"], "2012 → 2013")

    def test_partial_and_leap_windows(self):
        self.assertEqual(
            ndvi.substitute_year("2011-12-01", "2012-02-01"),
            ("2011-12-01", "2012-02-01", False),
        )
        self.assertEqual(
            ndvi.substitute_year("2012-02-01", "2012-03-01"),
            ("2013-02-01", "2013-03-01", True),
        )
        self.assertEqual(
            ndvi.substitute_year("2015-02-01", "2015-03-01"),
            ("2015-02-01", "2015-03-01", False),
        )

    def test_no_slc_off_2012_suggests_the_substitution(self):
        res = self.run_engine(
            self.COLLECTIONS,
            start="2012-06-01",
            end="2012-09-30",
            satellite="landsat_no_slc_off",
        )
        self.assertIn("fill 2012 with 2013", res["message"])


class SceneCountTests(_Case):
    """The sidecar states which sensors and how many scenes made a file."""

    def counts(self, **kw):
        side = self.sidecar(kw.pop("name", "site"))
        return side["n_images"], side["n_images_after_cloud_filter"]

    def test_scene_filter_and_every_scene(self):
        for mode, filtered in (("scene_filter", 2), ("pixel_only", None)):
            res = self.run_engine(
                LANDSAT_SR,
                start="2015-06-01",
                end="2015-09-30",
                satellite="landsat",
                cloud_handling=mode,
                name=mode,
            )
            self.assertEqual(res["status"], "success", res.get("message"))
            self.assertEqual(self.counts(name=mode), (3, filtered))
            side = self.sidecar(mode)
            self.assertEqual(side["satellite"], "landsat")
            self.assertEqual(side["ee_collection"], E.LANDSAT_MERGED_LABEL)

    def test_a_widened_window_counts_its_own_scenes(self):
        col = {
            E.LANDSAT8_COLLECTION_ID: [
                landsat_sr("2015-07-10", 5, 0.05, 0.40),
                landsat_sr("2015-06-20", 5, 0.05, 0.40),
                landsat_sr("2015-08-05", 5, 0.05, 0.40),
                landsat_sr("2015-08-10", 95, 0.05, 0.40),
            ]
        }
        res = self.run_engine(
            col,
            start="2015-07-01",
            end="2015-07-31",
            satellite="landsat",
            coverage_rescue=True,
        )
        self.assertEqual(res["status"], "success", res.get("message"))
        self.assertTrue(self.sidecar()["coverage_widened"])
        self.assertEqual(self.counts(), (4, 3))


class ResumeKeyTests(_Case):
    """Acceptance 7: every non-default option gets its own cache entry."""

    def test_each_option_changes_the_key(self):
        col = {
            E.LANDSAT8_COLLECTION_ID: [landsat_sr("2015-07-01", 5, 0.05, 0.40)],
            E.LANDSAT_TOA_COLLECTION_IDS[2]: [landsat_toa("2015-07-01", 5, 0.05, 0.40)],
        }
        water = {ndvi.WATER_DATASET: Image({"occurrence": 0.0})}
        keys = {}
        for name, kw in (
            ("default", {}),
            ("water", {"water_mask": True}),
            ("pixel", {"cloud_handling": "pixel_only"}),
            ("scene", {"cloud_handling": "scene_only"}),
            ("toa", {"reflectance": "toa"}),
            ("clean", {"satellite": "landsat_no_slc_off"}),
        ):
            kw.setdefault("satellite", "landsat")
            res = self.run_engine(
                col, water, start="2015-06-01", end="2015-09-30", name=name, **kw
            )
            self.assertEqual(res["status"], "success", res.get("message"))
            keys[name] = self.sidecar(name)["resume_key"]
        self.assertEqual(len(set(keys.values())), len(keys))
        self.assertEqual(
            keys["default"],
            ndvi._compute_resume_key(
                AOI.to_crs(4326),
                "2015-06-01",
                "2015-09-30",
                10,
                RES,
                E.LANDSAT_MERGED_LABEL,
            ),
        )

    def test_unknown_option_values_are_refused(self):
        eng = E.__new__(E)
        for kw in (
            {"cloud_handling": "none"},
            {"reflectance": "dn"},
            {"satellite": "modis"},
        ):
            with self.assertRaises(ValueError):
                eng.download_and_process(AOI, "2015-06-01", "2015-09-30", "x", **kw)


if __name__ == "__main__":
    unittest.main()
