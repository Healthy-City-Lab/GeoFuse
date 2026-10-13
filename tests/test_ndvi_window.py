"""NDVI windows: Earth Engine's end date is exclusive, users' end dates are not.

``filterDate(start, end)`` keeps images dated ``start <= date < end``. The
engine keeps that reading by default and, with ``end_inclusive=True``, sends
the day after a given last day, so a window picked as "through 30 September"
contains 30 September. The 2012 fill, the satellite pick and coverage rescue
all work on that half-open window. The engine runs end to end on the numeric
fake of Earth Engine, whose ``filterDate`` is half-open like the real one.
"""

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (ROOT, os.path.dirname(__file__)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json
from unittest import mock

import numpy as np
import rasterio
from _fake_earth_engine import FakeEarthEngine
from test_ndvi_acquisition import AOI, CRS_OVERRIDE, RES, _Case, _ndvi, landsat_sr

from geofuse import ndvi
from geofuse.jobs import runners
from geofuse.persistence.caches import NdviTileCache

E = ndvi.NDVIEngine

SEPT_30 = landsat_sr("2015-09-30", 5, 0.05, 0.40)
OCT_1 = landsat_sr("2015-10-01", 5, 0.05, 0.10)
JULY = landsat_sr("2015-07-01", 5, 0.05, 0.20)


# ────────────────────────────────────────────────────────────────────
# Engine
# ────────────────────────────────────────────────────────────────────


class EngineWindowTests(_Case):
    def _mean(self, scenes, *, start, end, name, **kw):
        res = self.run_engine(
            {E.LANDSAT8_COLLECTION_ID: scenes},
            start=start,
            end=end,
            name=name,
            satellite="landsat",
            temporal_reducers="mean",
            **kw,
        )
        self.assertEqual(res["status"], "success", res.get("message"))
        arr, _, _ = self.raster(f"{name}_mean")
        return arr

    def test_inclusive_end_keeps_the_last_day_and_not_the_next(self):
        arr = self._mean(
            [JULY, SEPT_30, OCT_1],
            start="2015-05-01",
            end="2015-09-30",
            name="incl",
            end_inclusive=True,
        )
        expected = (_ndvi(0.05, 0.20) + _ndvi(0.05, 0.40)) / 2
        np.testing.assert_allclose(arr, expected, rtol=1e-6)

    def test_default_end_stays_exclusive(self):
        arr = self._mean(
            [JULY, SEPT_30, OCT_1],
            start="2015-05-01",
            end="2015-09-30",
            name="excl",
        )
        np.testing.assert_allclose(arr, _ndvi(0.05, 0.20), rtol=1e-6)
        side = self.sidecar("excl_mean")
        self.assertEqual(side["end_date_exclusive"], "2015-09-30")
        self.assertEqual(
            side["resume_key"],
            ndvi._compute_resume_key(
                AOI.to_crs(4326),
                "2015-05-01",
                "2015-09-30",
                10,
                RES,
                E.LANDSAT_MERGED_LABEL,
                temporal_reducer="mean",
            ),
        )

    def test_sidecar_records_the_given_and_the_exclusive_end(self):
        self._mean(
            [JULY, SEPT_30],
            start="2015-05-01",
            end="2015-09-30",
            name="side",
            end_inclusive=True,
        )
        side = self.sidecar("side_mean")
        self.assertEqual(
            (side["start_date"], side["end_date"], side["end_date_exclusive"]),
            ("2015-05-01", "2015-09-30", "2015-10-01"),
        )
        self.assertEqual(
            (side["used_start_date"], side["used_end_date"]),
            ("2015-05-01", "2015-10-01"),
        )
        self.assertEqual(
            side["resume_key"],
            ndvi._compute_resume_key(
                AOI.to_crs(4326),
                "2015-05-01",
                "2015-10-01",
                10,
                RES,
                E.LANDSAT_MERGED_LABEL,
                temporal_reducer="mean",
            ),
        )

    def test_whole_year_2012_is_filled_from_the_whole_of_2013(self):
        scenes = [
            landsat_sr("2013-07-01", 5, 0.05, 0.20),
            landsat_sr("2013-12-31", 5, 0.05, 0.40),
            landsat_sr("2014-01-01", 5, 0.05, 0.10),
        ]
        arr = self._mean(
            scenes,
            start="2012-01-01",
            end="2012-12-31",
            name="y2012",
            end_inclusive=True,
            substitute_2012=True,
        )
        np.testing.assert_allclose(
            arr, (_ndvi(0.05, 0.20) + _ndvi(0.05, 0.40)) / 2, rtol=1e-6
        )
        side = self.sidecar("y2012_mean")
        self.assertEqual(side["year_substituted"], "2012 → 2013")
        self.assertEqual(
            (side["used_start_date"], side["used_end_date"]),
            ("2013-01-01", "2014-01-01"),
        )

    def test_coverage_rescue_widens_the_corrected_window(self):
        scenes = [
            JULY,
            landsat_sr("2015-11-30", 5, 0.05, 0.30),
            landsat_sr("2015-12-01", 5, 0.05, 0.30),
        ]
        res = self.run_engine(
            {E.LANDSAT8_COLLECTION_ID: scenes},
            start="2015-06-01",
            end="2015-09-30",
            satellite="landsat",
            coverage_rescue=True,
            end_inclusive=True,
            name="rescue",
        )
        self.assertEqual(res["status"], "success", res.get("message"))
        side = self.sidecar("rescue")
        # [06-01, 10-01) spans 122 days, so each side gains 61.
        self.assertTrue(side["coverage_widened"])
        self.assertEqual(side["n_cloud_filtered_images"], 2)
        self.assertEqual(
            (side["used_start_date"], side["used_end_date"]),
            ("2015-04-01", "2015-12-01"),
        )

    def test_auto_satellite_reads_the_end_as_exclusive(self):
        eng = E.__new__(E)
        start = E.S2_COLLECTION_START
        self.assertEqual(eng._pick_satellite("2016-06-01", start, "auto"), "landsat")
        self.assertEqual(
            eng._pick_satellite("2016-06-01", ndvi._shift_day(start, 1), "auto"),
            "sentinel2",
        )


# ────────────────────────────────────────────────────────────────────
# Runners: the NDVI tab's dates are inclusive
# ────────────────────────────────────────────────────────────────────


class _Ctx:
    def progress(self, **_kw):
        pass

    def heartbeat(self):
        pass

    def is_cancelled(self):
        return False


class RunnerWindowTests(_Case):
    def setUp(self):
        super().setUp()
        case = self

        class _Engine(E):
            def __init__(self, *_a, **_kw):
                self.tile_cache = NdviTileCache(os.path.join(case.tmp.name, "cache"))
                case.engines.append(self)

            @staticmethod
            def compute_export_crs(_gdf):
                return CRS_OVERRIDE

        for p in (
            mock.patch.object(runners, "NDVIEngine", _Engine),
            mock.patch.object(ndvi, "_select_export_crs", return_value=CRS_OVERRIDE),
        ):
            p.start()
            self.addCleanup(p.stop)

    def _kw(self, year):
        return {
            "fname": "site.geojson",
            "dataset_data": {"raw": AOI.assign(year=[year])},
            "cloud_pct": 10,
            "resolution": RES,
            "buffer_m": 0,
            "output_dir": self.out,
            "save_geotiff": True,
            "save_geojson": False,
            "satellite": "landsat",
            "coverage_rescue": False,
            "temporal_reducers": ["mean"],
        }

    def _column(self, scenes, year, first, last):
        with FakeEarthEngine(AOI, {E.LANDSAT8_COLLECTION_ID: scenes}):
            runners.run_ndvi_column(
                _Ctx(),
                date_column="year",
                season_start_month=first,
                season_end_month=last,
                **self._kw(year),
            )
        return self._read(
            os.path.join(self.out, "site_temporal_ndvi", f"site_{year}_mean")
        )

    def _read(self, stem):
        with rasterio.open(stem + "_ndvi.tif") as src:
            arr = src.read(1).astype(np.float64)
        with open(stem + "_ndvi.json") as f:
            return arr, json.load(f)

    def test_column_season_keeps_the_last_day_of_its_end_month(self):
        arr, side = self._column([JULY, SEPT_30, OCT_1], 2015, 5, 9)
        np.testing.assert_allclose(
            arr, (_ndvi(0.05, 0.20) + _ndvi(0.05, 0.40)) / 2, rtol=1e-6
        )
        self.assertEqual(
            (side["start_date"], side["end_date"], side["end_date_exclusive"]),
            ("2015-05-01", "2015-09-30", "2015-10-01"),
        )

    def test_column_january_to_december_ends_on_the_next_new_year(self):
        scenes = [
            JULY,
            landsat_sr("2015-12-31", 5, 0.05, 0.40),
            landsat_sr("2016-01-01", 5, 0.05, 0.10),
        ]
        arr, side = self._column(scenes, 2015, 1, 12)
        np.testing.assert_allclose(
            arr, (_ndvi(0.05, 0.20) + _ndvi(0.05, 0.40)) / 2, rtol=1e-6
        )
        self.assertEqual(
            (side["end_date"], side["used_end_date"]), ("2015-12-31", "2016-01-01")
        )

    def test_range_keeps_the_picked_end_and_not_the_next_day(self):
        with FakeEarthEngine(AOI, {E.LANDSAT8_COLLECTION_ID: [JULY, SEPT_30, OCT_1]}):
            runners.run_ndvi(
                _Ctx(),
                start_date="2015-05-01",
                end_date="2015-09-30",
                output_name="site",
                **self._kw(2015),
            )
        arr, side = self._read(os.path.join(self.out, "site_mean"))
        np.testing.assert_allclose(
            arr, (_ndvi(0.05, 0.20) + _ndvi(0.05, 0.40)) / 2, rtol=1e-6
        )
        self.assertEqual(
            (side["end_date"], side["end_date_exclusive"]), ("2015-09-30", "2015-10-01")
        )


class FusionAutoDownloadTests(unittest.TestCase):
    def test_auto_download_sends_an_inclusive_end(self):
        from geofuse.fusion import MetricFusionEngine

        calls = []

        class _Stub:
            def __init__(self, *_a, **_kw):
                pass

            def download_and_process(self, **kw):
                calls.append(kw)
                return {"status": "success", "tif": "site_ndvi.tif"}

        eng = MetricFusionEngine.__new__(MetricFusionEngine)
        eng.buffered_extent = AOI
        eng.target_file = "site.gpkg"
        eng.cache_dir = "unused"
        eng._ndvi_export_resolution_m = 10.0
        with (
            mock.patch.object(ndvi, "NDVIEngine", _Stub),
            mock.patch.object(eng, "_get_cache_filename", return_value="missing.tif"),
        ):
            eng._auto_download_ndvi("2023-06-01", "2023-09-30", cache=False)
        self.assertEqual(
            (calls[0]["end_date"], calls[0]["end_inclusive"]), ("2023-09-30", True)
        )


# ────────────────────────────────────────────────────────────────────
# Window helpers
# ────────────────────────────────────────────────────────────────────


class HelperTests(unittest.TestCase):
    def test_substitute_year_reads_a_half_open_window(self):
        self.assertEqual(
            ndvi.substitute_year("2012-01-01", "2013-01-01"),
            ("2013-01-01", "2014-01-01", True),
        )
        self.assertEqual(
            ndvi.substitute_year("2012-05-01", "2012-10-01"),
            ("2013-05-01", "2013-10-01", True),
        )
        self.assertEqual(
            ndvi.substitute_year("2012-06-01", "2013-01-02"),
            ("2012-06-01", "2013-01-02", False),
        )

    def test_overlap_ignores_the_exclusive_end(self):
        self.assertFalse(ndvi._overlaps_year("2011-06-01", "2012-01-01", 2012))
        self.assertTrue(ndvi._overlaps_year("2011-06-01", "2012-01-02", 2012))

    def test_window_text_names_the_last_day(self):
        self.assertEqual(
            ndvi._window_text("2015-05-01", "2015-10-01"), "2015-05-01 → 2015-09-30"
        )
        self.assertEqual(ndvi._shift_day("2015-12-31", 1), "2016-01-01")


if __name__ == "__main__":
    unittest.main()
