"""A tiled NDVI run reuses tiles cached by a run over a larger study area.

Every tiled workspace is recorded (content key, export grid, study area).
A later run whose outputs have the same content key takes each tile from such
a donor when the donor covers every pixel the tile needs, and asks Earth
Engine only for the rest. The engine runs on the numeric fake of Earth Engine;
each reusing run is compared pixel by pixel with a fresh run of the same area.
"""

import math
import os
import shutil
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
import shapely
from _fake_earth_engine import GRID, FakeEarthEngine
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform as shapely_transform
from test_ndvi_acquisition import CLOUD_BIT, CRS_OVERRIDE, RES, E, landsat_sr
from test_ndvi_formula import Field

from geofuse import ndvi
from geofuse.persistence.caches import NdviTileCache

LON, LAT = -114.07, 51.04
NIR = Field(lambda X, Y: 0.2 + 0.2 * np.sin(X / 90.0) * np.cos(Y / 70.0))
SCENES = {
    E.LANDSAT8_COLLECTION_ID: [
        landsat_sr("2015-07-01", 5, 0.05, NIR, qa=CLOUD_BIT),
        landsat_sr("2015-07-17", 5, 0.05, NIR + 0.1),
    ]
}
CHAINS = [
    "fm: focal:mean:60, temporal:mean, water_mask",
    "tm: temporal:max",
]


def _square(x_km: float, y_km: float, size_km: float) -> gpd.GeoDataFrame:
    """A square ``size_km`` wide whose south-west corner is ``(x_km, y_km)``
    from the reference corner."""
    dlon = 1000 / (111_320 * math.cos(math.radians(LAT)))
    dlat = 1000 / 110_950
    x0, y0 = LON + x_km * dlon, LAT + y_km * dlat
    return gpd.GeoDataFrame(
        geometry=[box(x0, y0, x0 + size_km * dlon, y0 + size_km * dlat)],
        crs="EPSG:4326",
    )


class SalvageTests(unittest.TestCase):
    def setUp(self):
        from _fake_earth_engine import Image

        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.cache = os.path.join(self.tmp.name, "cache")
        self.engines = []
        self.water = {ndvi.WATER_DATASET: Image({"occurrence": 0.0})}

    def tearDown(self):
        for eng in self.engines:
            eng.tile_cache.close()
        self.tmp.cleanup()

    def run_area(self, area, name, *, cache=None, start="2015-06-01", **kw):
        eng = E.__new__(E)
        eng.tile_cache = NdviTileCache(cache or self.cache)
        self.engines.append(eng)
        out = os.path.join(self.tmp.name, name)
        with FakeEarthEngine(area, SCENES, self.water) as fake:
            res = eng.download_and_process(
                geometry=area,
                start_date=start,
                end_date="2015-09-30",
                output_name="x",
                resolution=RES,
                folder=out,
                crs_override=CRS_OVERRIDE,
                satellite="landsat",
                coverage_rescue=False,
                write_geojson=False,
                max_tile_size_km=0.4,
                chains=CHAINS,
                **kw,
            )
        self.assertEqual(res["status"], "success", res.get("message"))
        return res, fake.exports, out

    def assert_same_inside(self, area, out_a, out_b):
        """Both runs' rasters agree at every pixel whose centre is in ``area``."""
        to_grid = Transformer.from_crs("EPSG:4326", GRID, always_xy=True).transform
        poly = shapely_transform(to_grid, area.geometry.iloc[0])
        for chain in ("fm", "tm"):
            grids = []
            for out in (out_a, out_b):
                with rasterio.open(os.path.join(out, f"x_{chain}_ndvi.tif")) as src:
                    arr = src.read(1).astype(float)
                    t = src.transform
                h, w = arr.shape
                cols, rows = np.meshgrid(np.arange(w), np.arange(h))
                X, Y = t.c + (cols + 0.5) * t.a, t.f + (rows + 0.5) * t.e
                inside = shapely.contains_xy(poly, X, Y)
                grids.append(dict(zip(zip(X[inside], Y[inside]), arr[inside])))
            self.assertEqual(set(grids[0]), set(grids[1]), chain)
            a = np.array([grids[0][k] for k in sorted(grids[0])])
            b = np.array([grids[1][k] for k in sorted(grids[0])])
            np.testing.assert_array_equal(a, b, chain)
            self.assertGreater(len(a), 100)

    def fresh(self, area, name):
        return self.run_area(area, name, cache=os.path.join(self.tmp.name, name + "c"))

    def test_a_smaller_area_takes_every_tile_from_the_larger_run(self):
        self.run_area(_square(0, 0, 1.2), "big")
        small = _square(0.3, 0.3, 0.5)
        res, exports, out = self.run_area(small, "small")
        self.assertEqual(exports, [])
        meta = res["reducers"]["fm"]["meta"]
        self.assertEqual(meta["tiles_salvaged"], meta["tiles_total"])
        self.assertGreater(meta["tiles_total"], 1)
        _, fresh_exports, fresh = self.fresh(small, "fresh")
        self.assertTrue(fresh_exports)
        self.assert_same_inside(small, out, fresh)

    def test_a_donor_with_missing_tiles_leaves_them_to_earth_engine(self):
        self.run_area(_square(0, 0, 1.2), "big")
        for name in os.listdir(self.cache):
            folder = os.path.join(self.cache, name)
            if os.path.isdir(folder) and not name.startswith("_"):
                for tile in ("tile_0.tif",):
                    if os.path.exists(os.path.join(folder, tile)):
                        os.remove(os.path.join(folder, tile))
        small = _square(0.3, 0.3, 0.5)
        res, exports, out = self.run_area(small, "small")
        meta = res["reducers"]["tm"]["meta"]
        self.assertTrue(exports)
        self.assertLess(len(exports), meta["tiles_total"])
        self.assertEqual(meta["tiles_salvaged"] + len(exports), meta["tiles_total"])
        _, _, fresh = self.fresh(small, "fresh")
        self.assert_same_inside(small, out, fresh)

    def test_an_area_reaching_beyond_the_donor_computes_the_rest(self):
        self.run_area(_square(0, 0, 1.2), "big")
        edge = _square(0.8, 0.3, 0.8)
        res, exports, out = self.run_area(edge, "edge")
        meta = res["reducers"]["fm"]["meta"]
        self.assertTrue(exports)
        self.assertGreater(meta["tiles_salvaged"], 0)
        _, _, fresh = self.fresh(edge, "fresh")
        self.assert_same_inside(edge, out, fresh)

    def test_other_content_is_not_reused(self):
        self.run_area(_square(0, 0, 1.2), "big")
        res, exports, _ = self.run_area(
            _square(0.3, 0.3, 0.5), "small", start="2015-07-01"
        )
        self.assertEqual(res["reducers"]["fm"]["meta"]["tiles_salvaged"], 0)
        self.assertEqual(len(exports), res["meta"]["tiles_total"])

    def test_workspaces_cached_before_records_are_adopted(self):
        big = _square(0, 0, 1.2)
        self.run_area(big, "big")
        shutil.rmtree(os.path.join(self.cache, ndvi.RECORDS_DIR))
        shutil.rmtree(os.path.join(self.cache, ndvi.AREAS_DIR))
        args = dict(
            cloud_max=10,
            resolution=RES,
            max_tile_size_km=0.4,
            satellite="landsat",
            crs_override=CRS_OVERRIDE,
            chains=CHAINS,
        )
        eng = self.engines[0]
        self.assertEqual(
            eng.record_workspaces(big, "2015-06-01", "2015-09-30", "x", **args),
            {"fm": "recorded", "tm": "recorded"},
        )
        self.assertEqual(
            eng.record_workspaces(big, "2015-06-01", "2015-09-30", "x", **args),
            {"fm": "already recorded", "tm": "already recorded"},
        )
        other = eng.record_workspaces(big, "2016-06-01", "2016-09-30", "x", **args)
        self.assertEqual(other, {"fm": "not cached", "tm": "not cached"})
        _, exports, _ = self.run_area(_square(0.3, 0.3, 0.5), "small")
        self.assertEqual(exports, [])

    def test_content_keys_ignore_only_the_area(self):
        fields = ("2015-06-01", "2015-09-30", 10, RES, E.LANDSAT_MERGED_LABEL)
        base = ndvi._content_key(*fields)
        self.assertEqual(base, ndvi._content_key(*fields, temporal_reducer="median"))
        for kw in (
            {"temporal_reducer": "max"},
            {"variant": ("pixel_only",)},
        ):
            self.assertNotEqual(ndvi._content_key(*fields, **kw), base)
        self.assertNotEqual(ndvi._content_key("2015-06-02", *fields[1:]), base)


if __name__ == "__main__":
    unittest.main()
