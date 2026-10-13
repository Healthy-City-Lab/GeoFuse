"""Job arrays: ``--level``, ``--group`` and ``--array-task``.

Jobs carry a dependency level and a resource group; ``plan`` numbers them,
and a selection by level and group, then by index, names exactly one job.
"""

import contextlib
import io
import os
import sys
import tempfile
import textwrap
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import geopandas as gpd
from shapely.geometry import box

from geofuse.cli import commands as cli

RUN = """
output_dir = "out"

[[ndvi]]
name = "box"
study_area = "area.gpkg"
ranges = [[2023-07-01, 2023-08-31]]

[[gvi]]
name = "box"
study_area = "area.gpkg"
shards = 2
"""


class _Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        area = box(-114.07, 51.045, -114.066, 51.0477)
        gpd.GeoDataFrame(geometry=[area], crs="EPSG:4326").to_file(
            os.path.join(self.dir, "area.gpkg"), driver="GPKG"
        )
        self.run = os.path.join(self.dir, "run.toml")
        with open(self.run, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(RUN))
        self.state = os.path.join(self.dir, "state")

    def tearDown(self):
        self.tmp.cleanup()

    def geofuse(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = cli.main(["--state-dir", self.state, *argv])
        return code, out.getvalue()

    def picked(self, *flags):
        args = cli.build_parser().parse_args(["run", self.run, *flags])
        return cli._load(args)[1]


class ArrayTests(_Case):
    def test_plan_numbers_jobs_with_their_level_and_group(self):
        code, out = self.geofuse("plan", self.run)
        self.assertEqual(code, 0, out)
        self.assertIn("(level 0, light)", out)
        self.assertIn("#1   gvi", out)
        self.assertIn("(level 0, gpu)", out)
        self.assertIn("gvi_merge    area  (level 1, light)", out)

    def test_level_group_and_task_name_one_job(self):
        (shard,) = self.picked(*"--level 0 --group gpu --array-task 1".split())
        self.assertEqual(shard.options["shard"], [1, 2])
        (merge,) = self.picked("--level", "1")
        self.assertEqual(merge.kind, "gvi_merge")
        self.assertEqual(len(self.picked("--level", "0")), 3)
        too_far = "--group gpu --array-task 5".split()
        code, out = self.geofuse("plan", self.run, *too_far)
        self.assertEqual(code, 2)
        self.assertIn("0..1", out)
        code, out = self.geofuse("plan", self.run, "--level", "4")
        self.assertIn("No job matches", out)


if __name__ == "__main__":
    unittest.main()
