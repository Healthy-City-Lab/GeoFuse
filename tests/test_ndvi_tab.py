"""The NDVI tab records and replays the download options of each job.

Each date mode of each dataset has its own temporal-statistic picker; the
satellite and coverage-rescue settings apply to the whole run. The options are
stored on the job record, so a restart replays them, and jobs recorded before
the options existed replay with the defaults (median, auto, rescue on).
"""

import os
import sys
import tempfile
import textwrap
import unittest
from datetime import date

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (os.path.join(ROOT, "ui"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import geopandas as gpd
from shapely.geometry import Point

from tabs import ndvi as ndvi_tab


class _Store:
    def __init__(self):
        self.submitted = []

    def submit(self, *, type, name, params):
        self.submitted.append((type, name, params))
        return type


class _Executor:
    def __init__(self):
        self.calls = []

    def submit_ndvi_subprocess(self, record, **kw):
        self.calls.append(("range", kw))

    def submit_ndvi_column_subprocess(self, record, **kw):
        self.calls.append(("column", kw))


def _raw():
    return gpd.GeoDataFrame(
        {"year": [2015]}, geometry=[Point(-114.07, 51.04)], crs="EPSG:4326"
    )


class ResubmitTests(unittest.TestCase):
    def _resubmit(self, **params):
        store, executor = _Store(), _Executor()
        ndvi_tab._ndvi_resubmit_from_params(
            store, executor, "out", "job1", params, _raw()
        )
        return store.submitted[0][2], executor.calls[0]

    def test_range_job_replays_its_options(self):
        params, (kind, kw) = self._resubmit(
            fname="site.geojson",
            mode="range",
            start_date="2023-06-01",
            end_date="2023-09-30",
            output_name="site_20230601_20230930",
            temporal_reducers=["median", "max"],
            satellite="landsat",
            coverage_rescue=False,
        )
        self.assertEqual(kind, "range")
        self.assertEqual(kw["temporal_reducers"], ["median", "max"])
        self.assertEqual(kw["satellite"], "landsat")
        self.assertFalse(kw["coverage_rescue"])
        self.assertEqual(params["temporal_reducers"], ["median", "max"])

    def test_column_job_replays_its_options(self):
        _params, (kind, kw) = self._resubmit(
            fname="site.geojson",
            mode="column",
            date_column="year",
            temporal_reducers=["mean"],
        )
        self.assertEqual(kind, "column")
        self.assertEqual(kw["temporal_reducers"], ["mean"])
        self.assertEqual((kw["satellite"], kw["coverage_rescue"]), ("auto", True))

    def test_old_job_gets_the_defaults(self):
        _params, (_kind, kw) = self._resubmit(
            fname="site.geojson",
            mode="specific",
            target_date="2023-07-15",
            window_days=30,
        )
        self.assertEqual(kw["temporal_reducers"], ["median"])
        self.assertEqual((kw["satellite"], kw["coverage_rescue"]), ("auto", True))

    def test_summary_names_the_options(self):
        lines = ndvi_tab._ndvi_restart_summary_lines(
            {"mode": "range", "temporal_reducers": ["median", "max"]}
        )
        self.assertIn(
            "**Temporal statistic:** Median, Maximum · "
            "**Satellite:** Auto (Sentinel-2 from 2017-03-28) · "
            "**Reflectance:** Surface reflectance",
            lines,
        )
        self.assertIn(
            "**Clouds:** Drop cloudy scenes, mask cloudy pixels · "
            "**Water mask:** off · **Coverage rescue:** on · "
            "**2012 from 2013:** off",
            lines,
        )

    def test_acquisition_options_replay(self):
        _params, (_kind, kw) = self._resubmit(
            fname="site.geojson",
            mode="range",
            start_date="2012-06-01",
            end_date="2012-09-30",
            water_mask=True,
            cloud_handling="pixel_only",
            reflectance="toa",
            substitute_2012=True,
            satellite="landsat_no_slc_off",
        )
        self.assertEqual(
            {k: kw[k] for k in ("water_mask", "cloud_handling", "reflectance", "substitute_2012", "satellite")},
            {
                "water_mask": True,
                "cloud_handling": "pixel_only",
                "reflectance": "toa",
                "substitute_2012": True,
                "satellite": "landsat_no_slc_off",
            },
        )

    def test_old_job_gets_acquisition_defaults(self):
        _params, (_kind, kw) = self._resubmit(
            fname="site.geojson", mode="range", start_date="2023-06-01", end_date="2023-09-30"
        )
        self.assertEqual(
            (kw["water_mask"], kw["cloud_handling"], kw["reflectance"], kw["substitute_2012"]),
            (False, "scene_filter", "surface", False),
        )


class DateConfigPickerTests(unittest.TestCase):
    """One picker per date mode; each edits only its own mode."""

    APP = textwrap.dedent(
        """
        import os, sys
        from datetime import date
        ROOT = {root!r}
        for p in (os.path.join(ROOT, "ui"), ROOT):
            if p not in sys.path:
                sys.path.insert(0, p)
        import geopandas as gpd
        import streamlit as st
        from shapely.geometry import Point
        from tabs import ndvi as ndvi_tab

        if "ndvi_datasets" not in st.session_state:
            raw = gpd.GeoDataFrame(
                {{"year": [2015, 2018]}},
                geometry=[Point(-114.07, 51.04), Point(-114.06, 51.05)],
                crs="EPSG:4326",
            )
            st.session_state.ndvi_datasets = {{"site.geojson": {{"raw": raw, "type": "input"}}}}
            st.session_state.ndvi_date_configs = {{
                "site.geojson": {{
                    "use_ranges": True,
                    "use_specific": True,
                    "use_column": True,
                    "ranges": [(date(2023, 6, 1), date(2023, 9, 30))],
                    "specific_dates": [date(2023, 7, 15)],
                    "window_days_specific": 30,
                    "season_start_month": 6,
                    "season_end_month": 9,
                }}
            }}
        ndvi_tab._render_ndvi_date_config()
        """
    )

    def test_pickers(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "app.py")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.APP.format(root=ROOT))
            at = AppTest.from_file(path, default_timeout=60)
            at.run()
            self.assertFalse(at.exception)

            keys = ("ranges", "specific", "col")
            for k in keys:
                picker = at.multiselect(key=f"ndvi_stats_{k}_site.geojson")
                self.assertEqual(picker.label, "Temporal statistic")
                self.assertEqual(picker.value, ["median"])

            at.multiselect(key="ndvi_stats_col_site.geojson").set_value(
                ["mean", "max"]
            )
            at.run()
            cfg = at.session_state["ndvi_date_configs"]["site.geojson"]
            self.assertEqual(cfg["stats_column"], ["mean", "max"])
            self.assertEqual(cfg["stats_ranges"], ["median"])
            self.assertEqual(cfg["stats_specific"], ["median"])

            at.multiselect(key="ndvi_stats_ranges_site.geojson").set_value([])
            at.run()
            self.assertIn(
                "Select at least one temporal statistic.",
                [w.value for w in at.warning],
            )


class OneDayRangeTests(unittest.TestCase):
    """End dates are included, so a range may start and end on the same day."""

    APP = DateConfigPickerTests.APP.replace(
        '"use_specific": True', '"use_specific": False'
    ).replace("date(2023, 6, 1), date(2023, 9, 30)", "date(2023, 7, 1), date(2023, 7, 1)")
    WARNING = "End date must be on or after the start date."

    def test_same_day_is_valid_and_a_reversed_range_is_not(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "app.py")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.APP.format(root=ROOT))
            at = AppTest.from_file(path, default_timeout=60)
            at.run()
            self.assertFalse(at.exception)
            self.assertFalse(any(self.WARNING in m.value for m in at.markdown))

            at.date_input(key="ndvi_re_site.geojson_0").set_value(date(2023, 6, 30))
            at.run()
            self.assertTrue(any(self.WARNING in m.value for m in at.markdown))


class SettingsBoxTests(unittest.TestCase):
    """Download settings: defaults are today's behaviour, every option is the user's."""

    APP = textwrap.dedent(
        """
        import os, sys
        ROOT = {root!r}
        for p in (os.path.join(ROOT, "ui"), ROOT):
            if p not in sys.path:
                sys.path.insert(0, p)
        import streamlit as st
        from tabs import ndvi as ndvi_tab
        if "ndvi_datasets" not in st.session_state:
            st.session_state.ndvi_datasets = {{}}
        ndvi_tab._render_ndvi_settings_map()
        """
    )

    def test_defaults_and_cloud_threshold(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "app.py")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.APP.format(root=ROOT))
            at = AppTest.from_file(path, default_timeout=60)
            at.run()
            self.assertFalse(at.exception)
            self.assertEqual(at.selectbox(key="ndvi_satellite").value, "auto")
            self.assertEqual(at.selectbox(key="ndvi_reflectance").value, "surface")
            self.assertEqual(at.selectbox(key="ndvi_cloud_handling").value, "scene_filter")
            self.assertFalse(at.checkbox(key="ndvi_water_mask").value)
            self.assertFalse(at.checkbox(key="ndvi_substitute_2012").value)
            self.assertTrue(at.checkbox(key="ndvi_coverage_rescue").value)
            self.assertFalse(at.slider(key="ndvi_cloud").disabled)
            self.assertEqual(
                at.selectbox(key="ndvi_satellite").options,
                [
                    "Auto (Sentinel-2 from 2017-03-28)",
                    "Sentinel-2",
                    "Landsat (5 / 7 / 8 / 9)",
                    "Landsat 5 / 7 / 8, no SLC-off",
                ],
            )

            at.selectbox(key="ndvi_cloud_handling").set_value("pixel_only")
            at.checkbox(key="ndvi_water_mask").check()
            at.run()
            self.assertTrue(at.slider(key="ndvi_cloud").disabled)
            self.assertTrue(at.session_state["ndvi_water_mask"])
            at.selectbox(key="ndvi_cloud_handling").set_value("scene_only")
            at.run()
            self.assertFalse(at.slider(key="ndvi_cloud").disabled)


if __name__ == "__main__":
    unittest.main()
