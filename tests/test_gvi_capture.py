"""GVI capture selection: year in output names, capture metadata, month window.

Without a month window the capture chosen is exactly the one chosen before
windows existed (checked against a copy of the old rule on random capture
lists). With a window, ``exclude`` never returns a capture outside it,
``prefer`` falls back to one only when nothing inside fits, and same-year ties
go to the month nearest the window's middle. Specific-year runs carry the year
in their file names, and the sidecar records the capture rule and what was
sampled.
"""

import asyncio
import json
import os
import random
import sys
import tempfile
import textwrap
import threading
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (os.path.join(ROOT, "ui"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import geofuse  # noqa: F401  (import order: torch before GeoPandas)
import geopandas as gpd
from shapely.geometry import Point

from geofuse import streetview as sv
from geofuse.gvi import CAPTURE_STATUS_KEY, NO_CAPTURE_IN_WINDOW, GVIEngine
from geofuse.jobs import runners

C = sv.PanoCapture


def _pano(captures, date=None):
    return sv.StreetViewPanorama(
        id=captures[0].id if captures else "own",
        lat=51.0,
        lon=-114.0,
        tile_size=sv.Size(512, 512),
        image_sizes=[sv.Size(512, 256)],
        date=date,
        captures=list(captures),
    )


def _old_select(pano, target_year, max_year_diff=None):
    """The rule before month windows, verbatim."""
    caps = pano.captures
    if not caps:
        if max_year_diff is not None:
            return None
        if pano.date is not None:
            return C(pano.id, pano.date[0], pano.date[1])
        return C(pano.id, None, None)
    candidates = caps
    if max_year_diff is not None:
        candidates = [
            c
            for c in caps
            if c.year is not None and abs(c.year - target_year) <= max_year_diff
        ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda c: (abs(c.year - target_year), -c.year, -(c.month or 0)),
    )


class NoWindowTests(unittest.TestCase):
    def test_same_choice_as_before_on_random_lists(self):
        rng = random.Random(7)
        for _ in range(2000):
            n = rng.randint(0, 6)
            caps = [
                C(f"p{i}", rng.randint(2008, 2023), rng.choice([None] + list(range(1, 13))))
                for i in range(n)
            ]
            date = rng.choice([None, (2019, 7)])
            pano = _pano(caps, date)
            target = rng.randint(2006, 2025)
            diff = rng.choice([None, 0, 1, 2, 5])
            self.assertEqual(pano.select_capture(target, diff), _old_select(pano, target, diff))


class MonthWindowTests(unittest.TestCase):
    CAPS = [
        C("nov17", 2017, 11),
        C("aug17", 2017, 8),
        C("jun17", 2017, 6),
        C("feb18", 2018, 2),
        C("jul15", 2015, 7),
        C("nomonth16", 2016, None),
    ]

    def test_old_rule_prefers_the_latest_month(self):
        self.assertEqual(_pano(self.CAPS).select_capture(2017).id, "nov17")

    def test_same_year_tie_goes_to_the_middle_of_the_window(self):
        pick = _pano(self.CAPS).select_capture(2017, month_window=(5, 10))
        self.assertEqual(pick.id, "aug17")  # middle of May-Oct is 7.5

    def test_exclude_never_returns_outside_or_monthless(self):
        rng = random.Random(3)
        for _ in range(500):
            caps = [
                C(f"p{i}", rng.randint(2010, 2020), rng.choice([None] + list(range(1, 13))))
                for i in range(rng.randint(1, 6))
            ]
            window = (rng.randint(1, 12), rng.randint(1, 12))
            pick = _pano(caps).select_capture(
                rng.randint(2010, 2020), rng.choice([None, 1, 3]), month_window=window
            )
            if pick is not None:
                self.assertTrue(sv.in_month_window(pick.month, window))

    def test_prefer_falls_back_only_when_nothing_fits(self):
        pano = _pano(self.CAPS)
        # 2018 ± 0: only February 2018 is in the year window; outside May-Oct.
        self.assertIsNone(pano.select_capture(2018, 0, month_window=(5, 10)))
        self.assertEqual(
            pano.select_capture(2018, 0, month_window=(5, 10), month_policy="prefer").id,
            "feb18",
        )
        # 2018 ± 1: summer 2017 is inside the window and wins over February 2018.
        self.assertEqual(
            pano.select_capture(2018, 1, month_window=(5, 10), month_policy="prefer").id,
            "aug17",
        )

    def test_a_window_can_wrap_over_the_new_year(self):
        pick = _pano(self.CAPS).select_capture(2018, 1, month_window=(11, 3))
        self.assertEqual(pick.id, "feb18")
        self.assertTrue(sv.in_month_window(1, (11, 3)))
        self.assertFalse(sv.in_month_window(6, (11, 3)))

    def test_most_recent_in_window_when_no_year_is_targeted(self):
        self.assertEqual(_pano(self.CAPS).select_capture(None, month_window=(5, 10)).id, "aug17")
        self.assertEqual(_pano(self.CAPS).select_capture(None).id, "feb18")

    def test_no_capture_list_checks_the_own_date(self):
        own_summer = _pano([], date=(2019, 7))
        own_winter = _pano([], date=(2019, 1))
        self.assertEqual(own_summer.select_capture(None, month_window=(5, 10)).month, 7)
        self.assertIsNone(own_winter.select_capture(None, month_window=(5, 10)))
        self.assertEqual(
            own_winter.select_capture(None, month_window=(5, 10), month_policy="prefer").month,
            1,
        )


class EngineMarksWindowMissesTests(unittest.TestCase):
    def _resolve(self, pano, **kw):
        engine = GVIEngine.__new__(GVIEngine)
        gdf = gpd.GeoDataFrame(geometry=[Point(-114.0, 51.0)], crs="EPSG:4326")
        pt = {"orig_index": 0, "lat": 51.0, "lon": -114.0, "geometry": gdf.geometry[0],
              "row": 0, "col": 0}
        with mock.patch.object(sv, "find_panorama_async", mock.AsyncMock(return_value=pano)):
            return asyncio.run(
                engine._search_and_resolve(pt, None, gdf, set(), {}, None, **kw)
            )

    def test_a_point_with_no_capture_in_the_window_is_marked(self):
        pano = _pano([C("dec19", 2019, 12), C("jan18", 2018, 1)])
        out = self._resolve(pano, target_year=None, max_year_diff=None, month_window=(5, 10))
        self.assertEqual(out[0], "done")
        self.assertEqual(out[1][CAPTURE_STATUS_KEY], NO_CAPTURE_IN_WINDOW)

    def test_an_in_window_capture_is_segmented(self):
        pano = _pano([C("dec19", 2019, 12), C("jul18", 2018, 7)])
        out = self._resolve(pano, target_year=None, max_year_diff=None, month_window=(5, 10))
        self.assertEqual(out[0], "segment")
        self.assertEqual(out[2], "jul18")
        self.assertEqual(out[3], "2018-07")

    def test_without_a_window_the_recent_pano_is_kept(self):
        pano = _pano([C("dec19", 2019, 12), C("jul18", 2018, 7)])
        out = self._resolve(pano, target_year=None, max_year_diff=None)
        self.assertEqual(out[2], "dec19")


class _Ctx:
    def progress(self, **_kw):
        pass

    def heartbeat(self):
        pass

    def is_cancelled(self):
        return False


class _StubEngine:
    """Feeds two sampled points, one point with no in-window capture and one miss."""

    def run_analysis(self, gdf, *, result_callback, target_year=None, month_window=None,
                     month_policy="exclude", **_kw):
        self.kw = {"target_year": target_year, "month_window": month_window,
                   "month_policy": month_policy}
        for i, (veg, date, status) in enumerate(
            [(0.3, "2017-07", None), (0.4, "2017-08", None),
             (None, None, NO_CAPTURE_IN_WINDOW), (None, None, None)]
        ):
            res = {"orig_index": i, "geometry": Point(-114.0 + i * 1e-3, 51.0),
                   "gvi_veg": veg, "gvi_ter": veg, "pano_id": None if veg is None else f"p{i}",
                   "pano_date": date, "lat": 51.0, "lon": -114.0, "row": 0, "col": i,
                   "cluster_id": 0}
            if status:
                res[CAPTURE_STATUS_KEY] = status
            result_callback(res)


class RunnerOutputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.stub = _StubEngine()
        p = mock.patch.object(runners, "_get_gvi_engine", return_value=self.stub)
        p.start()
        self.addCleanup(p.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, target_year, month_window=None):
        pts = gpd.GeoDataFrame(geometry=[Point(-114.0, 51.0)], crs="EPSG:4326")
        return runners.run_gvi(
            _Ctx(),
            fname="site.geojson",
            dataset_data={"accumulated": [], "processed": pts, "meta": {"step_m": 50},
                          "cache_ref": {}},
            init_args={"model_path": "m.pth"},
            run_args={"step": 50, "save_panos": False, "save_masks": False,
                      "target_year": target_year, "max_year_diff": 2,
                      "month_window": month_window, "month_policy": "exclude"},
            output_dir=self.tmp.name,
            save_geotiff=False,
            save_geojson=False,
            gpu_lock=threading.Lock(),
        )

    def test_two_specific_year_runs_share_a_folder(self):
        self._run(2015)
        self._run(2018)
        names = sorted(os.listdir(self.tmp.name))
        self.assertEqual(
            names,
            ["site_2015_gvi.gpkg", "site_2015_gvi.json", "site_2018_gvi.gpkg", "site_2018_gvi.json"],
        )

    def test_most_recent_keeps_its_name(self):
        self._run(None)
        self.assertEqual(sorted(os.listdir(self.tmp.name)), ["site_gvi.gpkg", "site_gvi.json"])

    def test_sidecar_records_the_capture_rule_and_counts(self):
        self._run(2017, month_window=[5, 10])
        self.assertEqual(self.stub.kw["month_window"], (5, 10))
        with open(os.path.join(self.tmp.name, "site_2017_gvi.json")) as f:
            side = json.load(f)
        self.assertEqual(side["capture_mode"], "year")
        self.assertEqual(side["target_year"], 2017)
        self.assertEqual(side["max_year_diff"], 2)
        self.assertEqual(side["month_window"], [5, 10])
        self.assertEqual(side["month_policy"], "exclude")
        self.assertEqual((side["n_points"], side["n_sampled"]), (4, 2))
        self.assertEqual(side["n_no_capture_in_window"], 1)
        self.assertEqual(side["capture_years"], {"2017": 2})
        self.assertEqual(side["capture_months"], {"7": 1, "8": 1})
        self.assertEqual(sum(side["capture_years"].values()), side["n_sampled"])
        gpkg = gpd.read_file(os.path.join(self.tmp.name, "site_2017_gvi.gpkg"))
        self.assertNotIn(CAPTURE_STATUS_KEY, gpkg.columns)


class CaptureConfigWidgetTests(unittest.TestCase):
    APP = textwrap.dedent(
        """
        import os, sys
        ROOT = {root!r}
        for p in (os.path.join(ROOT, "ui"), ROOT):
            if p not in sys.path:
                sys.path.insert(0, p)
        import geopandas as gpd
        import streamlit as st
        from shapely.geometry import Point
        from tabs import gvi as gvi_tab
        if "datasets" not in st.session_state:
            raw = gpd.GeoDataFrame({{"year": [2017]}}, geometry=[Point(-114.0, 51.0)], crs="EPSG:4326")
            st.session_state.datasets = {{"site.geojson": {{"raw": raw, "type": "point"}}}}
            st.session_state.gvi_date_configs = {{}}
        gvi_tab._render_gvi_date_config()
        """
    )

    def test_month_window_controls(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "app.py")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.APP.format(root=ROOT))
            at = AppTest.from_file(path, default_timeout=60)
            at.run()
            self.assertFalse(at.exception)
            box = at.checkbox(key="gvi_usemonths_site.geojson")
            self.assertEqual(box.label, "Limit capture months")
            self.assertFalse(box.value)
            box.check()
            at.run()
            self.assertEqual(at.selectbox(key="gvi_month_start_site.geojson").value, 5)
            self.assertEqual(at.selectbox(key="gvi_month_end_site.geojson").value, 10)
            at.radio(key="gvi_month_policy_site.geojson").set_value("prefer")
            at.run()
            cfg = at.session_state["gvi_date_configs"]["site.geojson"]
            self.assertTrue(cfg["use_months"])
            self.assertEqual((cfg["month_start"], cfg["month_end"], cfg["month_policy"]), (5, 10, "prefer"))


class RestartReplayTests(unittest.TestCase):
    def test_summary_and_replay(self):
        from tabs import gvi as gvi_tab

        p = {"fname": "site.geojson", "month_window": [5, 10], "month_policy": "prefer",
             "target_year": 2017}
        self.assertIn(
            "**Capture months:** May–Oct (outside as fallback)",
            gvi_tab._gvi_restart_summary_lines(p),
        )
        self.assertEqual(
            gvi_tab._capture_args(p), {"month_window": [5, 10], "month_policy": "prefer"}
        )
        self.assertEqual(
            gvi_tab._capture_args({}), {"month_window": None, "month_policy": "exclude"}
        )
        self.assertIn("**Capture months:** any", gvi_tab._gvi_restart_summary_lines({}))


if __name__ == "__main__":
    unittest.main()
