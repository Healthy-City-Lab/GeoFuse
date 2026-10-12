"""NDVI job runners pass the temporal statistics through and list every file.

The engine is replaced by a stub that records its arguments and writes the
files a real download would, so the tests pin the runner contract: the chosen
statistics reach ``download_and_process``, and the job's output paths name each
statistic's files (median unsuffixed, the others ``_<stat>_ndvi``).
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import geopandas as gpd
from shapely.geometry import Point

from geofuse.jobs import runners
from geofuse.ndvi import ndvi_output_name, normalize_temporal_reducers


class _Ctx:
    def progress(self, **_kw):
        pass

    def heartbeat(self):
        pass

    def is_cancelled(self):
        return False


class _StubEngine:
    calls: list[dict] = []
    projects: list = []

    def __init__(self, project_id=None, **_kw):
        _StubEngine.projects.append(project_id)

    @staticmethod
    def compute_export_crs(_gdf):
        return ("EPSG:32611", None, 0.0, "UTM zone 11N")

    def download_and_process(self, **kw):
        _StubEngine.calls.append(kw)
        for r in normalize_temporal_reducers(kw["temporal_reducers"]):
            stem = os.path.join(kw["folder"], ndvi_output_name(kw["output_name"], r))
            for suffix in ("_ndvi.tif", "_ndvi.json"):
                with open(stem + suffix, "w") as f:
                    f.write("x")
        return {"status": "success"}


def _points(years):
    return gpd.GeoDataFrame(
        {"year": years},
        geometry=[Point(-114.07 + 0.01 * i, 51.04) for i in range(len(years))],
        crs="EPSG:4326",
    )


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        _StubEngine.calls = []
        _StubEngine.projects = []
        patcher = mock.patch.object(runners, "NDVIEngine", _StubEngine)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def _common(self):
        return dict(
            cloud_pct=10,
            resolution=10,
            buffer_m=0,
            output_dir=self.dir,
            save_geotiff=True,
            save_geojson=False,
        )

    def test_range_job(self):
        out = runners.run_ndvi(
            _Ctx(),
            fname="site.geojson",
            dataset_data={"raw": _points([2020])},
            start_date="2023-06-01",
            end_date="2023-09-30",
            output_name="site",
            temporal_reducers=["max", "median"],
            **self._common(),
        )
        self.assertEqual(_StubEngine.calls[0]["temporal_reducers"], ["max", "median"])
        names = [os.path.basename(p) for p in out["output_paths"]]
        self.assertEqual(
            names,
            ["site_ndvi.tif", "site_ndvi.json", "site_max_ndvi.tif", "site_max_ndvi.json"],
        )

    def test_default_is_median_only(self):
        out = runners.run_ndvi(
            _Ctx(),
            fname="site.geojson",
            dataset_data={"raw": _points([2020])},
            start_date="2023-06-01",
            end_date="2023-09-30",
            output_name="site",
            **self._common(),
        )
        self.assertEqual(
            tuple(_StubEngine.calls[0]["temporal_reducers"]), ("median",)
        )
        self.assertEqual(
            [os.path.basename(p) for p in out["output_paths"]],
            ["site_ndvi.tif", "site_ndvi.json"],
        )

    def test_acquisition_options_reach_the_engine_in_every_mode(self):
        opts = dict(
            water_mask=True,
            cloud_handling="pixel_only",
            reflectance="toa",
            substitute_2012=True,
            satellite="landsat_no_slc_off",
        )
        runners.run_ndvi(
            _Ctx(),
            fname="site.geojson",
            dataset_data={"raw": _points([2012])},
            start_date="2012-06-01",
            end_date="2012-09-30",
            output_name="site",
            **opts,
            **self._common(),
        )
        runners.run_ndvi_column(
            _Ctx(),
            fname="site.geojson",
            dataset_data={"raw": _points([2012])},
            date_column="year",
            season_start_month=5,
            season_end_month=9,
            **opts,
            **self._common(),
        )
        for call in _StubEngine.calls:
            self.assertEqual({k: call[k] for k in opts}, opts)

    def test_acquisition_defaults(self):
        runners.run_ndvi(
            _Ctx(),
            fname="site.geojson",
            dataset_data={"raw": _points([2020])},
            start_date="2023-06-01",
            end_date="2023-09-30",
            output_name="site",
            **self._common(),
        )
        call = _StubEngine.calls[0]
        self.assertEqual(
            (call["water_mask"], call["cloud_handling"], call["reflectance"]),
            (False, "scene_filter", "surface"),
        )
        self.assertFalse(call["substitute_2012"])

    def test_earth_engine_project_reaches_the_engine(self):
        runners.run_ndvi(
            _Ctx(),
            fname="site.geojson",
            dataset_data={"raw": _points([2020])},
            start_date="2023-06-01",
            end_date="2023-09-30",
            output_name="site",
            ee_project="my-project",
            **self._common(),
        )
        runners.run_ndvi_column(
            _Ctx(),
            fname="site.geojson",
            dataset_data={"raw": _points([2020])},
            date_column="year",
            season_start_month=5,
            season_end_month=9,
            **self._common(),
        )
        self.assertEqual(_StubEngine.projects, ["my-project", None])

    def test_column_job_per_year_and_statistic(self):
        out = runners.run_ndvi_column(
            _Ctx(),
            fname="site.geojson",
            dataset_data={"raw": _points([2015, 2018])},
            date_column="year",
            season_start_month=5,
            season_end_month=9,
            temporal_reducers=["mean"],
            **self._common(),
        )
        self.assertEqual([c["temporal_reducers"] for c in _StubEngine.calls], [["mean"]] * 2)
        names = [os.path.basename(p) for p in out["output_paths"][1:]]
        self.assertEqual(
            names,
            [
                "site_2015_mean_ndvi.tif",
                "site_2015_mean_ndvi.json",
                "site_2018_mean_ndvi.tif",
                "site_2018_mean_ndvi.json",
            ],
        )


if __name__ == "__main__":
    unittest.main()
