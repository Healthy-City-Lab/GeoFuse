"""Run files: TOML / JSON documents of NDVI, GVI and fusion steps.

Settings resolve default < preset < [defaults.<engine>] < step < command line;
unknown keys, bad values and credentials are refused with the key named;
paths resolve against the run file's folder; references to earlier steps add
dependencies and order the steps; year-to-file matching follows the fusion
tab's rule.
"""

import json
import os
import sys
import tempfile
import textwrap
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from geofuse import runfile, settings
from geofuse.runfile import Override, Reference, RunFileError

PIPELINE = """
format = "geofuse-run"
version = 1
output_dir = "out"

[defaults.ndvi]
preset = "Landsat TOA, all scenes, water-masked"
cloud_max = 30

[[fusion]]
name = "cesd"
target = "data/cohort.gpkg"
outcomes = ["cesd"]
covariates = ["age"]
categorical = ["sex"]
mode = "year-aware"
date_column = "interview_date"
ndvi = "@ndvi.city[max]"
gvi = "@gvi.city"

[[ndvi]]
name = "city"
study_area = "data/city.gpkg"
year_column = "year"
ranges = [[2019-05-01, 2019-09-30]]
cloud_max = 20

[[gvi]]
name = "city"
study_area = ["data/a.gpkg", "data/b.gpkg"]
merge = true
capture_mode = "column"
year_column = "year"
"""


class _Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, text, name="run.toml"):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(text))
        return path

    def load(self, text, overrides=()):
        return runfile.load(self.write(text), overrides)


class LoadTests(_Case):
    def test_pipeline_resolves_paths_order_and_dependencies(self):
        run = self.load(PIPELINE)
        self.assertEqual(
            [s.key for s in run.steps], ["ndvi.city", "gvi.city", "fusion.cesd"]
        )
        self.assertEqual(run.name, "run")
        here = os.path.realpath(self.dir)
        self.assertEqual(run.output_dir, os.path.join(here, "out"))
        ndvi = run.step("ndvi.city")
        self.assertEqual(
            ndvi.inputs["study_area"],
            [os.path.join(os.path.realpath(self.dir), "data", "city.gpkg")],
        )
        self.assertEqual(ndvi.inputs["ranges"], [["2019-05-01", "2019-09-30"]])
        fusion = run.step("fusion.cesd")
        self.assertEqual(fusion.depends_on, ["ndvi.city", "gvi.city"])
        self.assertEqual(fusion.inputs["ndvi"], Reference("ndvi", "city", "max"))
        self.assertEqual(fusion.inputs["covariates"], ["age", "sex"])
        self.assertTrue(run.step("gvi.city").inputs["merge"])

    def test_settings_precedence(self):
        run = self.load(
            PIPELINE,
            [
                Override(("ndvi",), "water_mask", False),
                Override(("ndvi", "city"), "resolution_m", 60),
            ],
        )
        s = run.step("ndvi.city").settings
        self.assertEqual(s["reflectance"], "toa")  # preset
        self.assertEqual(s["cloud_max"], 20)  # step beats [defaults.ndvi]
        self.assertFalse(s["water_mask"])  # command line beats preset
        self.assertEqual(s["resolution_m"], 60)
        self.assertEqual(s["season_start_month"], 5)  # preset
        self.assertEqual(
            run.step("ndvi.city").preset, "Landsat TOA, all scenes, water-masked"
        )
        gvi = run.step("gvi.city").settings
        default = settings.GVI["grid_resolution_m"].default
        self.assertEqual(gvi["grid_resolution_m"], default)

    def test_json_run_files_read_the_same(self):
        doc = {
            "format": "geofuse-run",
            "version": 1,
            "ndvi": [
                {
                    "name": "a",
                    "study_area": "a.gpkg",
                    "specific_dates": ["2020-07-01"],
                    "statistics_specific": ["max", "mean"],
                }
            ],
        }
        path = os.path.join(self.dir, "run.json")
        with open(path, "w") as f:
            json.dump(doc, f)
        step = runfile.load(path).steps[0]
        self.assertEqual(step.settings["statistics_specific"], ["mean", "max"])
        self.assertEqual(step.inputs["specific_dates"], ["2020-07-01"])

    def test_star_overrides_reach_every_engine_that_has_the_key(self):
        run = self.load(PIPELINE, [Override(("*",), "buffer_m", 500)])
        self.assertEqual(run.step("ndvi.city").settings["buffer_m"], 500)
        self.assertEqual(run.step("gvi.city").settings["buffer_m"], 500)
        with self.assertRaisesRegex(RunFileError, "no matching step"):
            self.load(PIPELINE, [Override(("*",), "sweep_split", 3)])

    def test_longitudinal_runs_default_to_their_own_metric(self):
        text = """
        [[fusion]]
        target = "t.gpkg"
        outcomes = ["y"]
        mode = "longitudinal"
        entity_column = "id"
        date_column = "date"
        ndvi = {2015 = "n15.tif", 2018 = "n18.tif"}
        gvi = ["g_2015.gpkg", "g_2018.gpkg"]
        """
        step = self.load(text).steps[0]
        self.assertEqual(step.settings["objective_metric"], "mixedlm_tstat")
        self.assertEqual(sorted(step.inputs["ndvi"]), ["2015", "2018"])
        with self.assertRaisesRegex(RunFileError, "longitudinal run scores with"):
            self.load(text, ["fusion.objective_metric=spearman"])


class RefusalTests(_Case):
    def assert_refused(self, text, pattern, overrides=()):
        with self.assertRaisesRegex(RunFileError, pattern):
            self.load(text, overrides)

    def test_unknown_keys_are_named(self):
        typo = PIPELINE.replace("cloud_max = 20", "cloud_maxx = 20")
        self.assert_refused(typo, "cloud_maxx")
        self.assert_refused("colour = 1\n" + PIPELINE, "colour")
        self.assert_refused(PIPELINE, "sweep_split", ["fusion.sweep_split=4"])

    def test_bad_values_name_the_step_and_key(self):
        too_cloudy = PIPELINE.replace("cloud_max = 20", "cloud_max = 200")
        self.assert_refused(too_cloudy, r"city: 'cloud_max'")
        self.assert_refused(PIPELINE, "'test_size'", ["fusion.cesd.test_size=0.9"])
        self.assert_refused(
            PIPELINE.replace(
                "[[2019-05-01, 2019-09-30]]", "[[2019-09-30, 2019-05-01]]"
            ),
            "ends before it starts",
        )

    def test_credentials_are_refused(self):
        keyed = PIPELINE + '\n[[gvi]]\nname="x"\napi_key="k"\n'
        self.assert_refused(keyed, "credential")
        leaked = '"AIza' + "C" * 35 + '"'
        self.assert_refused(PIPELINE.replace('"data/a.gpkg"', leaked), "credential")

    def test_chains_are_checked_with_the_chain_named(self):
        bad = 'cloud_max = 20\nchains = ["g7: temporal:mean, temporal:max"]'
        refused = PIPELINE.replace("cloud_max = 20", bad)
        self.assert_refused(refused, r"city: 'chains_ranges': Chain 'g7'")
        shared = 'cloud_max = 30\nchains = ["a: temporal:max"]'
        run = self.load(
            PIPELINE.replace("cloud_max = 30", shared),
            ["ndvi.city.chains_column=['b: temporal:mean']"],
        )
        s = run.step("ndvi.city").settings
        a = {"name": "a", "steps": ["temporal:max"]}
        b = {"name": "b", "steps": ["temporal:mean"]}
        self.assertEqual((s["chains_ranges"], s["chains_column"]), ([a], [b]))

    def test_references_and_dependencies(self):
        self.assert_refused(PIPELINE.replace("@gvi.city", "@gvi.town"), "names no step")
        crossed = PIPELINE.replace('ndvi = "@ndvi.city[max]"', 'ndvi = "@gvi.city"')
        self.assert_refused(crossed, "must reference")
        self.assert_refused(PIPELINE.replace("[max]", "[max-1]"), "not a step ref")
        named = PIPELINE.replace('"@gvi.city"', '"@gvi.city[max]"')
        self.assert_refused(named, "only NDVI references")
        area = 'study_area = "data/city.gpkg"'
        cyclic = PIPELINE.replace(
            f'name = "city"\n{area}',
            f'name = "city"\ndepends_on = ["fusion.cesd"]\n{area}',
        )
        self.assert_refused(cyclic, "depend on each other")

    def test_fusion_inputs(self):
        panel = PIPELINE.replace('mode = "year-aware"', 'mode = "panel"')
        self.assert_refused(panel, "'mode'")
        self.assert_refused(
            PIPELINE.replace('date_column = "interview_date"\n', ""),
            "needs 'date_column'",
        )
        cross = PIPELINE.replace('mode = "year-aware"', 'mode = "cross-sectional"')
        self.assert_refused(cross, "per-year files", ['fusion.ndvi={2019 = "x.tif"}'])
        no_gvi = PIPELINE.replace('gvi = "@gvi.city"\n', "")
        self.assert_refused(no_gvi, "'gvi' is required")

    def test_comparators_are_kept_apart_from_the_other_columns(self):
        with_cmp = PIPELINE.replace(
            'covariates = ["age"]',
            'covariates = ["age"]\ncomparators = ["ndvi_500m", "ndvi_1km"]',
        )
        step = self.load(with_cmp).step("fusion.cesd")
        self.assertEqual(step.inputs["comparators"], ["ndvi_500m", "ndvi_1km"])
        self.assertEqual(step.inputs["comparators_primary"], [])
        for clash, role in (("age", "covariate"), ("cesd", "outcome")):
            self.assert_refused(
                with_cmp.replace('"ndvi_1km"]', f'"{clash}"]'),
                rf"{role}: \['{clash}'\]",
            )
        primary = with_cmp.replace(
            'outcomes = ["cesd"]',
            'outcomes = ["cesd"]\ncomparators_primary = ["ndvi_250m"]',
        )
        self.assert_refused(primary, r"non-comparators: \['ndvi_250m'\]")

    def test_gvi_column_mode_needs_its_column(self):
        blank = ["gvi.city.year_column=''"]
        self.assert_refused(PIPELINE, "needs 'year_column'", blank)


class OverrideParsingTests(unittest.TestCase):
    def test_values_are_read_as_toml_or_text(self):
        o = runfile.parse_override
        self.assertEqual(o("ndvi.cloud_max=20"), Override(("ndvi",), "cloud_max", 20))
        self.assertEqual(o("ndvi.a.water_mask=true").value, True)
        self.assertEqual(o("fusion.outcomes=['y', 'z']").value, ["y", "z"])
        self.assertEqual(o("ndvi.a.specific_dates=[2020-07-01]").value, ["2020-07-01"])
        self.assertEqual(o("fusion.objective_metric=spearman").value, "spearman")
        self.assertEqual(o("output_dir=C:/runs/a").target, ())

    def test_malformed_keys(self):
        for spec in ("cloud_max", "=3", "modis.cloud_max=3", "ndvi.a.b.c=1"):
            with self.assertRaises(RunFileError):
                runfile.parse_override(spec)


class YearMatchTests(unittest.TestCase):
    def test_one_to_one_matches_only(self):
        files = [
            "n_2015.tif",
            "n2016.tif",
            "n_2017_2018.tif",
            "a_2019.tif",
            "b_2019.tif",
        ]
        years = ["2015", "2016", "2017", "2018", "2019"]
        self.assertEqual(
            runfile.match_years_to_filenames(files, years), {"n_2015.tif": "2015"}
        )

    def test_the_fusion_tab_uses_the_same_rule(self):
        sys.path.insert(0, os.path.join(ROOT, "ui"))
        from tabs import fusion

        files, years = ["x_2011.tif", "y_20115.tif"], ["2011", "20115"]
        self.assertEqual(
            fusion._match_waves_to_filenames(files, years),
            runfile.match_years_to_filenames(files, years),
        )


if __name__ == "__main__":
    unittest.main()
