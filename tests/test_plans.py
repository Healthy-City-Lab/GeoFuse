"""Run-file steps expand into the jobs the tabs would submit.

``expand`` lays a run file out as jobs — one per NDVI date range, specific
date and year column per study-area unit, one per GVI unit, one per fusion
step — with the files each will write, so a later step's ``@engine.step``
reference resolves to them. ``build`` then produces the job record and the
executor call. The parity tests click each tab's Run button against a stub
store and executor and compare what the tab submitted with ``build``.
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

from geofuse import runfile
from geofuse.jobs import plans
from geofuse.jobs.plans import PlanError
from geofuse.vector_io import file_size_mtime_fingerprint, geometry_sha256

TODAY = date(2026, 10, 12)


def _points(path, n=4, **columns):
    gdf = gpd.GeoDataFrame(
        columns,
        geometry=[Point(-114.07 + 0.01 * i, 51.04 + 0.01 * i) for i in range(n)],
        crs="EPSG:4326",
    )
    gdf.to_file(path, driver="GPKG", layer="layer1")
    return path


class _Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        p = lambda name: os.path.join(self.dir, name)  # noqa: E731
        self.city = _points(p("city.gpkg"), year=[2015, 2015, 2018, 2018])
        self.town = _points(p("town.gpkg"), n=2, year=[2018, 2018])
        self.cohort = _points(
            p("cohort.gpkg"),
            cesd=[1.0, 2.0, 3.0, 4.0],
            grip=[1.0, 1.5, 2.0, 2.5],
            age=[50, 60, 70, 80],
            visit=["2015-06-01", "2015-07-01", "2018-06-01", "2018-07-01"],
        )
        for name in ("ndvi_2015.tif", "ndvi_2018.tif", "gvi.gpkg"):
            open(p(name), "w").close()

    def tearDown(self):
        self.tmp.cleanup()

    def expand(self, text, overrides=()):
        path = os.path.join(self.dir, "run.toml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(text))
        return plans.expand(runfile.load(path, overrides), today=TODAY)


class ExpandTests(_Case):
    def test_ndvi_date_modes_and_units(self):
        specs = self.expand(
            """
            [[ndvi]]
            name = "areas"
            study_area = ["city.gpkg", "town.gpkg"]
            ranges = [[2019-05-01, 2019-09-30]]
            specific_dates = [2026-10-01]
            year_column = "year"
            statistics_column = ["mean", "max"]
            """
        )
        self.assertEqual(
            [(s.kind, s.name, s.options.get("output_name")) for s in specs],
            [
                ("ndvi", "city", "city_20190501_20190930"),
                ("ndvi", "city", "city_20261001"),
                ("ndvi_column", "city", None),
                ("ndvi", "town", "town_20190501_20190930"),
                ("ndvi", "town", "town_20261001"),
                ("ndvi_column", "town", None),
            ],
        )
        specific = specs[1].options
        self.assertEqual(
            (specific["start_date"], specific["end_date"]), ("2026-09-01", "2026-10-12")
        )
        years = specs[2].outputs["years"]
        self.assertEqual(sorted(years), ["max", "mean"])
        self.assertEqual(
            os.path.basename(years["max"]["2018"]), "city_2018_max_ndvi.tif"
        )
        self.assertEqual(specs[2].resources, ["earthengine"])

    def test_merged_units_take_the_step_name(self):
        specs = self.expand(
            """
            [[gvi]]
            name = "both towns"
            study_area = ["city.gpkg", "town.gpkg"]
            merge = true
            capture_mode = "year"
            target_year = 2030
            """.replace(
                '"both towns"', '"both-towns"'
            )
        )
        (spec,) = specs
        self.assertTrue(spec.merged)
        self.assertEqual(spec.group, "both-towns")
        self.assertEqual(spec.options["target_year"], 2026)  # clamped to this year
        self.assertEqual(
            os.path.basename(spec.outputs["files"]), "both-towns_2026_gvi.gpkg"
        )

    def test_fusion_references_resolve_to_per_year_files(self):
        specs = self.expand(
            """
            [[ndvi]]
            name = "city"
            study_area = "city.gpkg"
            year_column = "year"
            statistics_column = ["mean", "max"]

            [[gvi]]
            name = "city"
            study_area = "city.gpkg"
            capture_mode = "column"
            year_column = "year"

            [[fusion]]
            name = "cesd"
            target = "cohort.gpkg"
            outcomes = ["cesd"]
            mode = "year-aware"
            date_column = "visit"
            ndvi = "@ndvi.city[max]"
            gvi = "@gvi.city"
            """
        )
        fusion = specs[-1]
        self.assertEqual(fusion.options["waves_found"], ["2015", "2018"])
        self.assertEqual(
            {y: os.path.basename(p) for y, p in fusion.options["ndvi_files"].items()},
            {"2015": "city_2015_max_ndvi.tif", "2018": "city_2018_max_ndvi.tif"},
        )
        self.assertEqual(fusion.depends_on, ["ndvi.city", "gvi.city"])
        self.assertEqual(fusion.options["layer"], "layer1")

    def test_plan_errors_name_the_problem(self):
        base = """
            [[ndvi]]
            name = "city"
            study_area = "city.gpkg"
            year_column = "year"
            statistics_column = ["mean", "max"]

            [[fusion]]
            target = "cohort.gpkg"
            outcomes = ["cesd"]
            mode = "year-aware"
            date_column = "visit"
            ndvi = "@ndvi.city"
            gvi = {2015 = "gvi.gpkg", 2018 = "gvi.gpkg"}
            """
        with self.assertRaisesRegex(PlanError, r"several statistics .*\[mean\]"):
            self.expand(base)
        with self.assertRaisesRegex(PlanError, "no GVI file for 2018"):
            self.expand(
                base.replace("@ndvi.city", "@ndvi.city[max]"),
                ['fusion.gvi={2015 = "gvi.gpkg"}'],
            )
        with self.assertRaisesRegex(PlanError, r"no column\(s\) \['bmi'\]"):
            self.expand(
                base.replace("@ndvi.city", "@ndvi.city[max]"),
                ["fusion.covariates=['bmi']"],
            )
        with self.assertRaisesRegex(PlanError, "has no column 'yr'"):
            self.expand(base, ["ndvi.city.year_column=yr"])

    def test_filename_lists_match_years(self):
        (fusion,) = self.expand(
            """
            [[fusion]]
            target = "cohort.gpkg"
            outcomes = ["cesd"]
            mode = "year-aware"
            date_column = "visit"
            ndvi = ["ndvi_2015.tif", "ndvi_2018.tif"]
            gvi = {2015 = "gvi.gpkg", 2018 = "gvi.gpkg"}
            """
        )
        self.assertEqual(sorted(fusion.options["ndvi_files"]), ["2015", "2018"])

    def test_specs_round_trip_as_json(self):
        import json

        specs = self.expand(
            '[[ndvi]]\nstudy_area = "city.gpkg"\nranges = [[2019-05-01, 2019-09-30]]\n'
        )
        again = plans.JobSpec.from_dict(json.loads(json.dumps(specs[0].to_dict())))
        self.assertEqual(again, specs[0])


class BuildTests(_Case):
    def test_year_aware_fusion_carries_the_tab_s_spec(self):
        (fusion,) = self.expand(
            """
            [[fusion]]
            target = "cohort.gpkg"
            outcomes = ["cesd"]
            negative_controls = ["grip"]
            covariates = ["age"]
            mode = "year-aware"
            date_column = "visit"
            ndvi = {2015 = "ndvi_2015.tif", 2018 = "ndvi_2018.tif"}
            gvi = {2015 = "gvi.gpkg", 2018 = "gvi.gpkg"}
            spatial_split = false
            spatial_block_size_m = 500
            """
        )
        sub = plans.build(fusion)
        run = sub.kwargs["run_kwargs"]
        lon = run["longitudinal_spec_payload"]
        self.assertEqual(lon["entity_id_col"], "_gf_entity")
        self.assertTrue(lon["derive_wave_from_date"])
        self.assertEqual(lon["wave_labels"], ["2015", "2018"])
        self.assertIsNone(run["veg_path"])
        self.assertEqual(run["negative_controls"], ["grip"])
        self.assertIsNone(run["spatial_block_size_m"])  # blocks are off
        fps = sub.params["longitudinal_spec_payload"]["__file_fingerprints__"]
        ndvi_2015 = os.path.join(self.dir, "ndvi_2015.tif")
        self.assertEqual(fps["ndvi"]["2015"], file_size_mtime_fingerprint(ndvi_2015))

    def test_earth_engine_project_goes_to_the_executor_only(self):
        (spec,) = self.expand(
            '[[ndvi]]\nstudy_area = "city.gpkg"\nranges = [[2019-05-01, 2019-09-30]]\n'
        )
        sub = plans.build(spec, ee_project="my-project")
        self.assertEqual(sub.kwargs["ee_project"], "my-project")
        self.assertNotIn("ee_project", sub.params)
        self.assertNotIn("ee_project", plans.build(spec).kwargs)

    def test_options_left_at_run_fusion_defaults_are_not_passed(self):
        (fusion,) = self.expand(
            """
            [[fusion]]
            target = "cohort.gpkg"
            outcomes = ["cesd"]
            ndvi = "ndvi_2015.tif"
            gvi = "gvi.gpkg"
            report_paired_bootstrap = 500
            """
        )
        run = plans.build(fusion).kwargs["run_kwargs"]
        self.assertEqual(run["report_paired_bootstrap"], 500)
        self.assertNotIn("report_effects_bootstrap", run)


class ShardTests(_Case):
    RUN = """
        [[gvi]]
        name = "city"
        study_area = "city.gpkg"
        capture_mode = "column"
        year_column = "year"
        shards = 3

        [[fusion]]
        name = "cesd"
        target = "cohort.gpkg"
        outcomes = ["cesd"]
        mode = "year-aware"
        date_column = "visit"
        ndvi = ["ndvi_2015.tif", "ndvi_2018.tif"]
        gvi = "@gvi.city"
        """

    def test_a_sharded_step_becomes_shard_jobs_and_a_merge(self):
        specs = self.expand(self.RUN)
        self.assertEqual(
            [(s.step, s.kind, s.options.get("shard")) for s in specs],
            [
                ("gvi.city:shards", "gvi_column", [0, 3]),
                ("gvi.city:shards", "gvi_column", [1, 3]),
                ("gvi.city:shards", "gvi_column", [2, 3]),
                ("gvi.city", "gvi_merge", None),
                ("fusion.cesd", "fusion", None),
            ],
        )
        merge, fusion = specs[3], specs[4]
        self.assertEqual(merge.depends_on, ["gvi.city:shards"])
        self.assertEqual(merge.resources, [])
        self.assertEqual(sorted(fusion.options["gvi_files"]), ["2015", "2018"])

    def test_the_merge_waits_for_the_shards_and_fusion_for_the_merge(self):
        from geofuse.jobs.state import StateDir
        from geofuse.jobs.supervisor import submit_specs

        state = StateDir(os.path.join(self.dir, "state"))
        _, ids = submit_specs(state, self.expand(self.RUN), run_name="t")
        self.assertEqual(state.job(ids[3]).spec["depends_on"], ids[:3])
        self.assertEqual(state.job(ids[4]).spec["depends_on"], [ids[3]])

    def test_shards_and_the_merge_build_their_calls(self):
        specs = self.expand(self.RUN)
        shard = plans.build(specs[1])
        self.assertEqual(shard.kwargs["run_args"]["shard"], [1, 3])
        self.assertEqual(shard.params["shard"], [1, 3])
        merge = plans.build(specs[3])
        self.assertEqual(merge.method, "submit_gvi_merge_subprocess")
        self.assertEqual(merge.kwargs["shards"], 3)
        self.assertEqual(merge.kwargs["date_column"], "year")
        self.assertNotIn("shard", merge.kwargs["run_args"])

    def test_shard_counts_are_checked(self):
        one = self.expand(self.RUN.replace("shards = 3", "shards = 1"))
        self.assertEqual([s.kind for s in one], ["gvi_column", "fusion"])
        for value in ("0", '"2"', "true"):
            with self.assertRaisesRegex(runfile.RunFileError, "shards"):
                self.expand(self.RUN.replace("shards = 3", f"shards = {value}"))


# ────────────────────────────────────────────────────────────────────
# Parity with the tabs
# ────────────────────────────────────────────────────────────────────

_STUB = """
class Store:
    def __init__(self):
        self.submitted = []
    def submit(self, type, params, name=""):
        rec = types.SimpleNamespace(id=f"j{len(self.submitted)}", type=type,
                                    name=name, params=params)
        self.submitted.append(rec)
        return rec
    def list_active(self):
        return []
    def list_all(self):
        return []
    def get(self, _id):
        return None
    def health(self):
        return {"active": 0, "stuck": 0, "errored_last_hour": 0}

class Executor:
    def __init__(self):
        self.calls = []
    def __getattr__(self, method):
        if not method.startswith("submit_"):
            raise AttributeError(method)
        def call(record, *args, **kwargs):
            self.calls.append((method, record, args, kwargs))
        return call

import types
STORE, EXECUTOR = Store(), Executor()
from geofuse.persistence import caches
PANO = types.SimpleNamespace(db_path=caches.pano_db())
"""

_APP = """
import os, sys
for p in ({root!r}, os.path.join({root!r}, "ui"), {stub_dir!r}):
    if p not in sys.path:
        sys.path.insert(0, p)
import streamlit as st
import services
import gf_parity_stub as stub
services.get_job_store = lambda: stub.STORE
services.get_job_executor = lambda: stub.EXECUTOR
services.get_pano_cache = lambda: stub.PANO
from tabs import {tab}
if "_seeded" not in st.session_state:
    st.session_state["_seeded"] = True
    for k, v in {seed!r}.items():
        st.session_state[k] = v
{call}
"""


class TabParityTests(_Case):
    def setUp(self):
        super().setUp()
        self.out = os.path.join(self.dir, "out")
        with open(os.path.join(self.dir, "gf_parity_stub.py"), "w") as f:
            f.write(_STUB)
        sys.modules.pop("gf_parity_stub", None)

    def click(self, tab, seed, call, button, numbers=None, selects=None, submits=1):
        from streamlit.testing.v1 import AppTest

        path = os.path.join(self.dir, "app.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                _APP.format(root=ROOT, stub_dir=self.dir, tab=tab, seed=seed, call=call)
            )
        at = AppTest.from_file(path, default_timeout=120)
        at.run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        for key, value in (numbers or {}).items():
            at.number_input(key=key).set_value(value)
        for key, value in (selects or {}).items():
            at.selectbox(key=key).set_value(value)
        if numbers or selects:
            at.run()
        at.button(key=button).click()
        at.run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        stub = sys.modules["gf_parity_stub"]
        self.assertEqual(len(stub.STORE.submitted), submits)
        if not submits:
            return at
        (rec,) = stub.STORE.submitted
        ((method, _r, args, kwargs),) = stub.EXECUTOR.calls
        return rec, method, args, kwargs

    def assert_same(self, gui, cli: plans.Submission):
        rec, method, args, kwargs = gui
        self.assertEqual((rec.type, rec.name), (cli.record_type, cli.record_name))
        self.assertEqual(method, cli.method)
        self.assertEqual(rec.params, cli.params)
        if args:
            kwargs = {"run_kwargs": args[0]}
        gui_data = kwargs.pop("dataset_data", None)
        cli_kwargs = dict(cli.kwargs)
        cli_data = cli_kwargs.pop("dataset_data", None)
        self.assertEqual(kwargs, cli_kwargs)
        if gui_data is not None:
            gui_data = {k: v for k, v in gui_data.items() if k != "cache_ref"}
            self.assertEqual(sorted(gui_data), sorted(cli_data))
            for key in ("raw", "processed"):
                if gui_data.get(key) is not None:
                    self.assertEqual(
                        geometry_sha256(gui_data[key]), geometry_sha256(cli_data[key])
                    )

    def test_ndvi_tab(self):
        gui = self.click(
            "ndvi",
            {"ndvi_picked_paths": [self.city]},
            f"ndvi.render({self.out!r})",
            "ndvi_run_btn",
        )
        (spec,) = self.expand(
            f"""
            output_dir = {self.out!r}
            [[ndvi]]
            study_area = "city.gpkg"
            ranges = [[2023-06-01, 2023-09-30]]
            """
        )
        self.assert_same(gui, plans.build(spec))

    def test_gvi_tab(self):
        gui = self.click(
            "gvi",
            {"gvi_picked_paths": [self.city]},
            f"gvi.render({self.out!r}, {ROOT!r})",
            "gvi_run_analysis",
        )
        (spec,) = self.expand(
            f"""
            output_dir = {self.out!r}
            [[gvi]]
            study_area = "city.gpkg"
            """
        )
        self.assert_same(gui, plans.build(spec))

    def test_gvi_tab_throughput_settings(self):
        sizing = {"download_concurrency": 8, "batch_size": 4, "batch_wait_ms": 50}
        gui = self.click(
            "gvi",
            {"gvi_picked_paths": [self.city]},
            f"gvi.render({self.out!r}, {ROOT!r})",
            "gvi_run_analysis",
            numbers={f"gvi_{k}": v for k, v in sizing.items()},
            selects={"gvi_precision": "fp32"},
        )
        (spec,) = self.expand(
            f"""
            output_dir = {self.out!r}
            [[gvi]]
            study_area = "city.gpkg"
            download_concurrency = 8
            batch_size = 4
            batch_wait_ms = 50
            precision = "fp32"
            """
        )
        sizing["precision"] = "fp32"
        self.assertEqual({k: gui[3]["run_args"][k] for k in sizing}, sizing)
        self.assert_same(gui, plans.build(spec))

    def test_gvi_tab_falls_back_to_the_stored_street_view_key(self):
        from geofuse import credentials

        self.addCleanup(setattr, credentials, "get", credentials.get)
        gui = self.click(
            "gvi",
            {"gvi_picked_paths": [self.city], "gvi_download_mode": "API (Street View)"},
            "import geofuse.credentials as c\n"
            "c.get = lambda name: 'AIzaSTORED'\n"
            f"gvi.render({self.out!r}, {ROOT!r})",
            "gvi_run_analysis",
        )
        (spec,) = self.expand(
            f"""
            output_dir = {self.out!r}
            [[gvi]]
            study_area = "city.gpkg"
            download_mode = "api"
            """
        )
        self.assertEqual(gui[3]["init_args"]["api_key"], "AIzaSTORED")
        self.assert_same(gui, plans.build(spec, api_key="AIzaSTORED"))

    def test_gvi_tab_api_mode_without_a_key_submits_nothing(self):
        from geofuse import credentials

        self.addCleanup(setattr, credentials, "get", credentials.get)
        at = self.click(
            "gvi",
            {"gvi_picked_paths": [self.city], "gvi_download_mode": "API (Street View)"},
            "import geofuse.credentials as c\n"
            "c.get = lambda name: None\n"
            f"gvi.render({self.out!r}, {ROOT!r})",
            "gvi_run_analysis",
            submits=0,
        )
        self.assertTrue(any("needs a key" in e.value for e in at.error))
        (spec,) = self.expand(
            f"""
            output_dir = {self.out!r}
            [[gvi]]
            study_area = "city.gpkg"
            download_mode = "api"
            """
        )
        with self.assertRaises(plans.PlanError):
            plans.build(spec)

    def test_fusion_tab(self):
        seed = {
            "fusion_target_paths": [self.cohort],
            "fusion_target_upload_sig": (self.cohort, os.path.getsize(self.cohort)),
            "fusion_outcome_columns": ["cesd"],
            "fusion_negative_controls": ["grip"],
            "fusion_covariate_columns": ["age"],
            "fusion_run_mode": "Cross-sectional",
            "fusion_ndvi_paths": [os.path.join(self.dir, "ndvi_2015.tif")],
            "fusion_gvi_paths": [os.path.join(self.dir, "gvi.gpkg")],
        }
        gui = self.click(
            "fusion", seed, f"fusion.render({self.out!r})", "fusion_form_run_submit"
        )
        (spec,) = self.expand(
            f"""
            output_dir = {self.out!r}
            [[fusion]]
            target = "cohort.gpkg"
            outcomes = ["cesd"]
            negative_controls = ["grip"]
            covariates = ["age"]
            ndvi = "ndvi_2015.tif"
            gvi = "gvi.gpkg"
            """
        )
        self.assert_same(gui, plans.build(spec))


if __name__ == "__main__":
    unittest.main()
