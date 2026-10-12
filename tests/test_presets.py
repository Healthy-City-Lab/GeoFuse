"""Presets for the NDVI and GVI settings.

Engine side (``geofuse.presets``): documents validate against a per-engine
schema, a preset for the other tab is refused with the tab named, unknown
settings are dropped with a warning, bad values are refused with the key named,
credentials never travel, the built-ins validate, and save -> load reproduces
every setting. UI side (AppTest): choosing a preset sets the widgets and every
dataset's method settings, editing one shows "Custom", Save writes no API key,
and a run records a preset only after one was actually used.
"""

import json
import os
import sys
import tempfile
import textwrap
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (os.path.join(ROOT, "ui"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from geofuse import presets


def _doc(engine="ndvi", **settings):
    return {
        "format": "geofuse-preset",
        "version": 1,
        "engine": engine,
        "name": "Test",
        "description": "",
        "settings": settings,
    }


class ValidationTests(unittest.TestCase):
    def test_other_engine_is_refused_with_the_tab_named(self):
        with self.assertRaisesRegex(presets.PresetError, "This is a GVI preset — load it in the GVI tab"):
            presets.validate(_doc("gvi", buffer_m=0), "ndvi")
        with self.assertRaisesRegex(presets.PresetError, "This is a NDVI preset — load it in the NDVI tab"):
            presets.validate(_doc("ndvi", buffer_m=0), "gvi")

    def test_unknown_settings_warn_and_are_ignored(self):
        settings, warnings = presets.validate(_doc(buffer_m=50, colour="green"), "ndvi")
        self.assertEqual(settings, {"buffer_m": 50})
        self.assertEqual(warnings, ["Unknown setting 'colour' ignored."])

    def test_bad_values_name_the_key(self):
        for settings, key in (
            ({"cloud_max": 120}, "cloud_max"),
            ({"cloud_max": "10"}, "cloud_max"),
            ({"water_mask": 1}, "water_mask"),
            ({"satellite": "modis"}, "satellite"),
            ({"statistics_ranges": []}, "statistics_ranges"),
            ({"statistics_ranges": ["p90"]}, "statistics_ranges"),
        ):
            with self.assertRaisesRegex(presets.PresetError, f"'{key}'"):
                presets.validate(_doc(**settings), "ndvi")
        with self.assertRaisesRegex(presets.PresetError, "season_start_month"):
            presets.validate(_doc(season_start_month=9, season_end_month=5), "ndvi")

    def test_not_a_preset(self):
        for doc in ([], {"format": "x"}, {"format": "geofuse-preset", "version": 9}):
            with self.assertRaises(presets.PresetError):
                presets.validate(doc, "ndvi")

    def test_credentials_never_travel(self):
        with self.assertRaisesRegex(presets.PresetError, "credential"):
            presets.validate({**_doc("gvi", buffer_m=0), "api_key": "x"}, "gvi")
        with self.assertRaisesRegex(presets.PresetError, "credential"):
            presets.validate(_doc("gvi", download_mode="api", google_api_key="x"), "gvi")
        leaked = "AIza" + "B" * 35
        with self.assertRaisesRegex(presets.PresetError, "credential"):
            presets.validate({**_doc("gvi", buffer_m=0), "description": leaked}, "gvi")

    def test_builtins_validate(self):
        for engine in presets.ENGINES:
            entries = [e for e in presets.list_presets(engine) if e.builtin]
            self.assertGreaterEqual(len(entries), 2)
            self.assertEqual(entries[0].name, "Default")
            for entry in entries:
                presets.validate(presets.read_preset(entry.path), engine)
                self.assertNotRegex(entry.description, "(?i)canue|clsa")

    def test_the_landsat_preset_matches_its_description(self):
        s = presets.load_settings("ndvi", "Landsat TOA, all scenes, water-masked")
        self.assertEqual(
            (s["satellite"], s["reflectance"], s["cloud_handling"], s["water_mask"]),
            ("landsat_no_slc_off", "toa", "pixel_only", True),
        )
        self.assertEqual((s["substitute_2012"], s["coverage_rescue"], s["resolution_m"]), (True, False, 30))
        self.assertEqual(s["statistics_column"], ["mean", "max"])


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lib = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_save_then_load_round_trip(self):
        settings = presets.load_settings("ndvi", "Landsat TOA, all scenes, water-masked")
        doc = presets.make_preset("ndvi", "My Landsat", "for P1", settings)
        path = presets.save_user_preset(doc, self.lib)
        self.assertEqual(path.parent.name, "ndvi")
        self.assertEqual(presets.load_settings("ndvi", "My Landsat", self.lib), settings)
        names = [e.name for e in presets.list_presets("ndvi", self.lib)]
        self.assertEqual(names[-1], "My Landsat")
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["format"], "geofuse-preset")

    def test_name_clashes(self):
        doc = presets.make_preset("gvi", "Mine", "", {"buffer_m": 50})
        presets.save_user_preset(doc, self.lib)
        with self.assertRaises(presets.PresetExists):
            presets.save_user_preset(doc, self.lib)
        doc2 = presets.make_preset("gvi", "Mine", "", {"buffer_m": 100})
        presets.save_user_preset(doc2, self.lib, overwrite=True)
        self.assertEqual(presets.load_settings("gvi", "Mine", self.lib), {"buffer_m": 100})
        with self.assertRaisesRegex(presets.PresetError, "built-in"):
            presets.save_user_preset(presets.make_preset("gvi", "Default", "", {}), self.lib)

    def test_unreadable_files(self):
        bad = os.path.join(self.lib, "bad.json")
        with open(bad, "w") as f:
            f.write("{not json")
        with self.assertRaisesRegex(presets.PresetError, "not valid JSON"):
            presets.read_preset(bad)


_APP_HEADER = """
import os, sys
from pathlib import Path
ROOT = {root!r}
for p in (os.path.join(ROOT, "ui"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
import geopandas as gpd
import streamlit as st
from shapely.geometry import Point
from geofuse import presets
presets.USER_DIR = Path({lib!r})
"""


class NdviPresetAppTests(unittest.TestCase):
    APP = textwrap.dedent(
        _APP_HEADER
        + """
from tabs import ndvi as ndvi_tab
from datetime import date
if "ndvi_datasets" not in st.session_state:
    raw = gpd.GeoDataFrame({{"year": [2015]}}, geometry=[Point(-114.0, 51.0)], crs="EPSG:4326")
    st.session_state.ndvi_datasets = {{"site.geojson": {{"raw": raw, "type": "input"}}}}
    st.session_state.ndvi_date_configs = {{}}
ndvi_tab._render_ndvi_settings_map()
ndvi_tab._render_ndvi_date_config()
"""
    )

    def _app(self, lib):
        from streamlit.testing.v1 import AppTest

        path = os.path.join(lib, "app.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.APP.format(root=ROOT, lib=lib))
        at = AppTest.from_file(path, default_timeout=60)
        at.run()
        self.assertFalse(at.exception)
        return at

    def test_choose_edit_save_and_load(self):
        with tempfile.TemporaryDirectory() as lib:
            at = self._app(lib)
            sel = at.selectbox(key="ndvi_preset_select")
            self.assertEqual(sel.value, "Default")
            self.assertIn("Landsat TOA, all scenes, water-masked", sel.options)

            sel.set_value("Landsat TOA, all scenes, water-masked")
            at.run()
            self.assertFalse(at.exception)
            self.assertEqual(at.selectbox(key="ndvi_satellite").value, "landsat_no_slc_off")
            self.assertEqual(at.selectbox(key="ndvi_reflectance").value, "toa")
            self.assertEqual(at.selectbox(key="ndvi_cloud_handling").value, "pixel_only")
            self.assertEqual(at.number_input(key="ndvi_res").value, 30)
            self.assertTrue(at.checkbox(key="ndvi_water_mask").value)
            self.assertTrue(at.checkbox(key="ndvi_substitute_2012").value)
            self.assertFalse(at.checkbox(key="ndvi_coverage_rescue").value)
            self.assertTrue(at.slider(key="ndvi_cloud").disabled)
            self.assertEqual(at.multiselect(key="ndvi_stats_ranges_site.geojson").value, ["mean", "max"])
            cfg = at.session_state["ndvi_date_configs"]["site.geojson"]
            self.assertEqual((cfg["stats_column"], cfg["season_start_month"]), (["mean", "max"], 5))
            self.assertEqual(at.session_state["_ndvi_preset_in_force"], "Landsat TOA, all scenes, water-masked")
            self.assertEqual(
                at.selectbox(key="ndvi_preset_select").value, "Landsat TOA, all scenes, water-masked"
            )

            at.checkbox(key="ndvi_water_mask").uncheck()
            at.run()
            self.assertEqual(at.selectbox(key="ndvi_preset_select").value, "Custom")
            self.assertIsNone(at.session_state["_ndvi_preset_in_force"])

            at.text_input(key="ndvi_preset_name").input("No water mask")
            at.button(key="ndvi_preset_save").click()
            at.run()
            saved = presets.load_settings("ndvi", "No water mask", lib)
            self.assertFalse(saved["water_mask"])
            self.assertEqual(saved["reflectance"], "toa")
            self.assertEqual(at.selectbox(key="ndvi_preset_select").value, "No water mask")

            # Loading a GVI file in this tab is refused and changes nothing.
            gvi_file = os.path.join(lib, "gvi.json")
            presets.write_preset(gvi_file, presets.make_preset("gvi", "G", "", {"buffer_m": 0}))
            at.session_state["ndvi_preset_manual"] = True
            at.run()
            at.text_input(key="ndvi_preset_manual_path").input(gvi_file)
            at.run()
            self.assertIn("This is a GVI preset — load it in the GVI tab", " ".join(e.value for e in at.error))
            self.assertEqual(at.selectbox(key="ndvi_preset_select").value, "No water mask")

    def test_no_preset_means_no_provenance(self):
        with tempfile.TemporaryDirectory() as lib:
            at = self._app(lib)
            self.assertEqual(at.selectbox(key="ndvi_preset_select").value, "Default")
            self.assertIsNone(at.session_state["_ndvi_preset_in_force"])


class GviPresetAppTests(unittest.TestCase):
    APP = textwrap.dedent(
        _APP_HEADER
        + """
from tabs import gvi as gvi_tab
if "datasets" not in st.session_state:
    raw = gpd.GeoDataFrame({{"year": [2017]}}, geometry=[Point(-114.0, 51.0)], crs="EPSG:4326")
    st.session_state.datasets = {{"site.geojson": {{"raw": raw, "type": "point"}}}}
    st.session_state.gvi_date_configs = {{}}
    st.session_state.gvi_download_mode = "API (Street View)"
    st.session_state.gvi_google_api_key = "AIza" + "C" * 35
gvi_tab._render_gvi_settings_map()
gvi_tab._render_gvi_date_config()
"""
    )

    def test_historical_preset_and_no_key_in_saves(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as lib:
            path = os.path.join(lib, "app.py")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.APP.format(root=ROOT, lib=lib))
            at = AppTest.from_file(path, default_timeout=60)
            at.run()
            self.assertFalse(at.exception)
            self.assertEqual(at.selectbox(key="gvi_preset_select").value, "Custom")

            at.selectbox(key="gvi_preset_select").set_value("Historical capture ± 2 years")
            at.run()
            self.assertFalse(at.exception)
            from tabs import gvi as gvi_tab

            self.assertEqual(at.radio(key="gvi_mode_site.geojson").value, gvi_tab._GVI_MODE_YEAR)
            self.assertTrue(at.checkbox(key="gvi_usemd_site.geojson").value)
            self.assertEqual(at.number_input(key="gvi_md_site.geojson").value, 2)
            self.assertEqual(at.radio(key="gvi_download_mode").value, "Package (Scraper)")
            self.assertEqual(at.selectbox(key="gvi_preset_select").value, "Historical capture ± 2 years")

            at.number_input(key="gvi_md_site.geojson").set_value(3)
            at.run()
            self.assertEqual(at.selectbox(key="gvi_preset_select").value, "Custom")

            at.radio(key="gvi_download_mode").set_value("API (Street View)")
            at.run()
            at.text_input(key="gvi_preset_name").input("Mine")
            at.button(key="gvi_preset_save").click()
            at.run()
            text = open(os.path.join(lib, "gvi", "mine.json"), encoding="utf-8").read()
            self.assertNotIn("AIza", text)
            self.assertEqual(json.loads(text)["settings"]["download_mode"], "api")
            self.assertEqual(json.loads(text)["settings"]["max_year_difference"], 3)


class ProvenanceTests(unittest.TestCase):
    def test_sidecar_records_a_preset_only_when_used(self):
        from geofuse import ndvi
        from geofuse.jobs import runners

        with tempfile.TemporaryDirectory() as d:
            kw = dict(grid_crs=None, export_crs="EPSG:32611", export_crs_name="UTM",
                      export_distortion=0.0, n_clusters=1, tiles_total=1, tiles_succeeded=1,
                      start_date="2020-06-01", end_date="2020-09-30", cloud_max=10,
                      resolution_m=10, max_tile_size_km=5, ee_collection="X")
            with open(ndvi._write_ndvi_sidecar(d, "a", **kw)) as f:
                self.assertNotIn("preset", json.load(f))
            with open(ndvi._write_ndvi_sidecar(d, "b", preset="Sentinel-2 10 m", **kw)) as f:
                self.assertEqual(json.load(f)["preset"], "Sentinel-2 10 m")
        self.assertNotIn("preset", runners._capture_settings("recent", None, {}))
        self.assertEqual(
            runners._capture_settings("year", 2017, {"preset": "P"})["preset"], "P"
        )


if __name__ == "__main__":
    unittest.main()
