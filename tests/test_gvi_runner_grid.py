"""A GVI job builds its own sampling grid when it is not handed one.

The submitting process (the GVI tab, or the command line) ships only the study
area; ``run_gvi`` builds the points from ``run_args["step"]`` and
``run_args["buffer"]`` — the clustered grid for polygons and buffered points,
the points themselves otherwise — exactly as the tab used to before submitting.
A grid that is handed in is used as is.
"""

import os
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import geopandas as gpd
from shapely.geometry import Point, box

import geofuse  # noqa: F401  (import order: torch before GeoPandas)
from geofuse.core import generate_clustered_grid, gvi_sampling_points
from geofuse.jobs import runners


class _Ctx:
    def progress(self, **_kw):
        pass

    def heartbeat(self):
        pass

    def is_cancelled(self):
        return False


class _Engine:
    """Records the points it is given and samples each of them."""

    def run_analysis(self, gdf, *, result_callback, **_kw):
        self.points = gdf
        for i, geom in enumerate(gdf.geometry):
            result_callback(
                {
                    "orig_index": i,
                    "geometry": geom,
                    "gvi_veg": 0.3,
                    "gvi_ter": 0.1,
                    "pano_id": f"p{i}",
                    "pano_date": "2020-07",
                    "lat": geom.y,
                    "lon": geom.x,
                    "row": 0,
                    "col": i,
                    "cluster_id": 0,
                }
            )


class RunnerGridTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.engine = _Engine()
        p = mock.patch.object(runners, "_get_gvi_engine", return_value=self.engine)
        p.start()
        self.addCleanup(p.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def run_gvi(self, raw, gtype, buffer_m, processed=None):
        data = {
            "raw": raw,
            "type": gtype,
            "processed": processed,
            "meta": None,
            "accumulated": [],
            "cache_ref": {},
        }
        runners.run_gvi(
            _Ctx(),
            fname="site.gpkg",
            dataset_data=data,
            init_args={"model_path": "m.pth"},
            run_args={
                "step": 50,
                "buffer": buffer_m,
                "save_panos": False,
                "save_masks": False,
            },
            output_dir=self.tmp.name,
            save_geotiff=False,
            save_geojson=False,
            gpu_lock=threading.Lock(),
        )
        return data

    def test_polygons_get_the_clustered_grid(self):
        lon, lat = -114.07, 51.04
        raw = gpd.GeoDataFrame(
            geometry=[box(lon, lat, lon + 0.004, lat + 0.003)], crs="EPSG:4326"
        )
        data = self.run_gvi(raw, "poly", 0)
        expected, meta = generate_clustered_grid(raw, buffer_m=0.0, step_m=50.0)
        self.assertEqual(len(self.engine.points), len(expected))
        self.assertEqual(data["meta"]["step_m"], meta["step_m"])

    def test_bare_points_are_sampled_where_they_are(self):
        raw = gpd.GeoDataFrame(geometry=[Point(-114.07, 51.04)], crs="EPSG:4326")
        data = self.run_gvi(raw, "point", 0)
        self.assertIsNone(data["meta"])
        self.assertEqual(list(self.engine.points.geometry), list(raw.geometry))
        buffered = self.run_gvi(raw, "point", 100)
        self.assertGreater(len(self.engine.points), 1)
        self.assertIsNotNone(buffered["meta"])

    def test_a_grid_handed_in_is_used(self):
        raw = gpd.GeoDataFrame(geometry=[Point(-114.07, 51.04)], crs="EPSG:4326")
        given = gpd.GeoDataFrame(
            geometry=[Point(-114.0, 51.0), Point(-114.1, 51.1)], crs="EPSG:4326"
        )
        self.run_gvi(raw, "point", 100, processed=given)
        self.assertEqual(len(self.engine.points), 2)

    def test_no_panorama_anywhere_finishes_without_outputs(self):
        class _Empty:
            def run_analysis(self, gdf, **_kw):
                pass

        raw = gpd.GeoDataFrame(geometry=[Point(-114.07, 51.04)], crs="EPSG:4326")
        with mock.patch.object(runners, "_get_gvi_engine", return_value=_Empty()):
            out = runners.run_gvi(
                _Ctx(),
                fname="site.gpkg",
                dataset_data={
                    "raw": raw,
                    "type": "point",
                    "processed": None,
                    "accumulated": [],
                    "cache_ref": {},
                },
                init_args={"model_path": "m.pth"},
                run_args={
                    "step": 50,
                    "buffer": 0,
                    "save_panos": False,
                    "save_masks": False,
                },
                output_dir=self.tmp.name,
                save_geotiff=False,
                save_geojson=False,
                gpu_lock=threading.Lock(),
            )
        self.assertEqual(out, {"output_paths": []})
        self.assertEqual(os.listdir(self.tmp.name), [])

    def test_the_helper_matches_the_tab_s_rule(self):
        raw = gpd.GeoDataFrame(geometry=[Point(-114.07, 51.04)], crs="EPSG:4326")
        points, meta = gvi_sampling_points(raw, False, 0, 50)
        self.assertIsNone(meta)
        self.assertIsNot(points, raw)


if __name__ == "__main__":
    unittest.main()
