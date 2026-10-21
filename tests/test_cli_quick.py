"""``geofuse ndvi | gvi | fusion`` without a run file.

Each builds the one-step run file it stands for: the saved file plans to the
same jobs, only the engine's own settings are flags, a fusion step
takes one file, files matched to years, ``YEAR=FILE`` pairs or one file per
wave, and the default action submits.
"""

import contextlib
import io
import os
import sys
import tempfile
import tomllib
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import geopandas as gpd
from shapely.geometry import Point, box

from geofuse import runfile
from geofuse.cli import commands as cli
from geofuse.cli import quick
from geofuse.jobs.plans import expand

NDVI = "ndvi city.gpkg --range 2019-05-01:2019-09-30 --date 2020-07-15"


class _Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.here = os.getcwd()
        os.chdir(self.dir)
        area = box(-114.08, 51.04, -114.07, 51.05)
        gpd.GeoDataFrame({"year": [2019]}, geometry=[area], crs="EPSG:4326").to_file(
            "city.gpkg", driver="GPKG"
        )
        people = {"cesd": [3.0, 5.0], "age": [60, 70], "sex": ["f", "m"]}
        points = [Point(-114.07, 51.04), Point(-114.06, 51.05)]
        gpd.GeoDataFrame(
            {**people, "year": [2019, 2021]}, geometry=points, crs="EPSG:4326"
        ).to_file("cohort.gpkg", driver="GPKG")
        for name in ("ndvi_2019.tif", "ndvi_2021.tif", "gvi.gpkg"):
            open(name, "w").close()

    def tearDown(self):
        os.chdir(self.here)
        self.tmp.cleanup()

    def geofuse(self, line: str):
        out = io.StringIO()
        state = ["--state-dir", os.path.join(self.dir, "state")]
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = cli.main(state + line.split())
        return code, out.getvalue()

    def document(self, line: str) -> dict:
        return quick.document(cli.build_parser().parse_args(line.split()))


class ShortcutTests(_Case):
    def test_the_saved_run_file_plans_to_the_same_jobs(self):
        flags = " --cloud-max 20 --no-water-mask"
        code, out = self.geofuse(NDVI + flags + " --plan --save run.toml")
        self.assertEqual(code, 0, out)
        self.assertIn("cloud_max=20", out)
        doc = self.document(NDVI + flags)
        direct = expand(runfile.from_document(doc, base_dir=self.dir))
        saved = expand(runfile.load("run.toml"))
        self.assertEqual([s.to_dict() for s in direct], [s.to_dict() for s in saved])
        self.assertEqual(len(saved), 2)
        self.assertEqual(saved[0].settings["cloud_max"], 20)
        self.assertIs(saved[0].settings["water_mask"], False)

    def test_settings_and_inputs_are_checked(self):
        with self.assertRaises(SystemExit) as refused:
            self.geofuse("gvi city.gpkg --cloud-max 3 --plan")
        self.assertEqual(refused.exception.code, 2)
        code, out = self.geofuse("ndvi city.gpkg --range 2019 --plan")
        self.assertIn("START:END", out)
        code, out = self.geofuse("gvi city.gpkg --capture-mode column --plan")
        self.assertIn("year_column", out)
        line = "gvi city.gpkg --capture-mode column --year-column year --plan"
        self.assertEqual(self.geofuse(line)[0], 0)

    def test_fusion_channels_take_files_years_or_pairs(self):
        doc = self.document(
            "fusion --target cohort.gpkg --outcome cesd --covariate age "
            "--categorical sex --mode year-aware --date-column year "
            "--ndvi 2019=ndvi_2019.tif 2021=ndvi_2021.tif --gvi gvi.gpkg "
            "--test-size 0.3"
        )
        (step,) = doc["fusion"]
        self.assertEqual(sorted(step["ndvi"]), ["2019", "2021"])
        self.assertTrue(step["gvi"].endswith("gvi.gpkg"))
        self.assertEqual(step["categorical"], ["sex"])
        self.assertEqual(step["test_size"], 0.3)
        text = quick.dumps(doc)
        self.assertIn("[[fusion]]", text)
        self.assertEqual(tomllib.loads(text)["fusion"][0]["ndvi"], step["ndvi"])
        with self.assertRaisesRegex(cli.CliError, "every file as YEAR=FILE"):
            self.document(
                "fusion --target cohort.gpkg --ndvi 2019=a.tif b.tif --gvi g.gpkg"
            )
        waves = self.document(
            "fusion --wave w1=a.gpkg --wave b.gpkg --ndvi n.tif --gvi g.gpkg"
        )
        labels = [w["label"] for w in waves["fusion"][0]["waves"]]
        self.assertEqual(labels, ["w1", "wave2"])
        self.assertIn("[[fusion.waves]]", quick.dumps(waves))

    def test_fusion_comparators_are_repeatable_flags(self):
        doc = self.document(
            "fusion --target cohort.gpkg --outcome cesd --comparator ndvi_500m "
            "--comparator ndvi_1km --comparator-primary ndvi_500m "
            "--ndvi n.tif --gvi g.gpkg"
        )
        (step,) = doc["fusion"]
        self.assertEqual(step["comparators"], ["ndvi_500m", "ndvi_1km"])
        self.assertEqual(step["comparators_primary"], ["ndvi_500m"])

    def test_the_default_action_submits(self):
        with mock.patch.object(quick, "submit", return_value=0) as submitted:
            code, _ = self.geofuse("ndvi city.gpkg --range 2019-05-01:2019-09-30")
        self.assertEqual(code, 0)
        _args, run, specs = submitted.call_args.args
        self.assertEqual(run.name, "city")
        self.assertEqual(len(specs), 1)


if __name__ == "__main__":
    unittest.main()
