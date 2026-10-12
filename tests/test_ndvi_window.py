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

import numpy as np
from test_ndvi_acquisition import AOI, RES, _Case, _ndvi, landsat_sr

from geofuse import ndvi

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
            {E.LANDSAT8_COLLECTION_ID: scenes}, start=start, end=end, name=name,
            satellite="landsat", temporal_reducers="mean", **kw,
        )
        self.assertEqual(res["status"], "success", res.get("message"))
        arr, _, _ = self.raster(f"{name}_mean")
        return arr

    def test_inclusive_end_keeps_the_last_day_and_not_the_next(self):
        arr = self._mean(
            [JULY, SEPT_30, OCT_1], start="2015-05-01", end="2015-09-30",
            name="incl", end_inclusive=True,
        )
        expected = (_ndvi(0.05, 0.20) + _ndvi(0.05, 0.40)) / 2
        np.testing.assert_allclose(arr, expected, rtol=1e-6)

    def test_default_end_stays_exclusive(self):
        arr = self._mean(
            [JULY, SEPT_30, OCT_1], start="2015-05-01", end="2015-09-30", name="excl",
        )
        np.testing.assert_allclose(arr, _ndvi(0.05, 0.20), rtol=1e-6)
        side = self.sidecar("excl_mean")
        self.assertEqual(side["end_date_exclusive"], "2015-09-30")
        self.assertEqual(
            side["resume_key"],
            ndvi._compute_resume_key(
                AOI.to_crs(4326), "2015-05-01", "2015-09-30", 10, RES,
                E.LANDSAT_MERGED_LABEL, temporal_reducer="mean",
            ),
        )

    def test_sidecar_records_the_given_and_the_exclusive_end(self):
        self._mean(
            [JULY, SEPT_30], start="2015-05-01", end="2015-09-30", name="side",
            end_inclusive=True,
        )
        side = self.sidecar("side_mean")
        self.assertEqual(
            (side["start_date"], side["end_date"], side["end_date_exclusive"]),
            ("2015-05-01", "2015-09-30", "2015-10-01"),
        )
        self.assertEqual(
            (side["used_start_date"], side["used_end_date"]), ("2015-05-01", "2015-10-01")
        )
        self.assertEqual(
            side["resume_key"],
            ndvi._compute_resume_key(
                AOI.to_crs(4326), "2015-05-01", "2015-10-01", 10, RES,
                E.LANDSAT_MERGED_LABEL, temporal_reducer="mean",
            ),
        )

    def test_whole_year_2012_is_filled_from_the_whole_of_2013(self):
        scenes = [
            landsat_sr("2013-07-01", 5, 0.05, 0.20),
            landsat_sr("2013-12-31", 5, 0.05, 0.40),
            landsat_sr("2014-01-01", 5, 0.05, 0.10),
        ]
        arr = self._mean(
            scenes, start="2012-01-01", end="2012-12-31", name="y2012",
            end_inclusive=True, substitute_2012=True,
        )
        np.testing.assert_allclose(
            arr, (_ndvi(0.05, 0.20) + _ndvi(0.05, 0.40)) / 2, rtol=1e-6
        )
        side = self.sidecar("y2012_mean")
        self.assertEqual(side["year_substituted"], "2012 → 2013")
        self.assertEqual(
            (side["used_start_date"], side["used_end_date"]), ("2013-01-01", "2014-01-01")
        )

    def test_coverage_rescue_widens_the_corrected_window(self):
        scenes = [
            JULY,
            landsat_sr("2015-11-30", 5, 0.05, 0.30),
            landsat_sr("2015-12-01", 5, 0.05, 0.30),
        ]
        res = self.run_engine(
            {E.LANDSAT8_COLLECTION_ID: scenes}, start="2015-06-01", end="2015-09-30",
            satellite="landsat", coverage_rescue=True, end_inclusive=True, name="rescue",
        )
        self.assertEqual(res["status"], "success", res.get("message"))
        side = self.sidecar("rescue")
        # [06-01, 10-01) spans 122 days, so each side gains 61.
        self.assertTrue(side["coverage_widened"])
        self.assertEqual(side["n_cloud_filtered_images"], 2)
        self.assertEqual(
            (side["used_start_date"], side["used_end_date"]), ("2015-04-01", "2015-12-01")
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
