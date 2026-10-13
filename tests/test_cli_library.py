"""``geofuse preset`` and ``geofuse cache``.

Presets: the tabs' built-ins and user library, shown as run-file lines, saved
from flags or a run-file step (settings of another engine refused), imported
and removed (built-ins kept). Caches: listed with each fusion cache a
command-line job wrote, purged only after a yes, and not while a job using
the cache runs.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from geofuse import presets
from geofuse.cli import commands as cli
from geofuse.jobs.state import StateDir
from geofuse.persistence import caches


class _Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.state_dir = os.path.join(self.dir, "state")
        library = os.path.join(self.dir, "presets")
        patches = [
            mock.patch.object(presets, "USER_DIR", Path(library)),
            mock.patch.object(caches, "cache_root", lambda: self.dir),
            mock.patch.object(caches, "GUI_OUTPUT_DIR", os.path.join(self.dir, "gui")),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def geofuse(self, *argv, stdin=""):
        out = io.StringIO()
        with (
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(out),
            mock.patch.object(sys, "stdin", io.StringIO(stdin)),
        ):
            code = cli.main(["--state-dir", self.state_dir, *argv])
        return code, out.getvalue()


class PresetTests(_Case):
    def test_built_ins_list_and_show_as_run_file_lines(self):
        code, out = self.geofuse("preset", "list")
        self.assertEqual(code, 0)
        self.assertIn("Sentinel-2 10 m", out)
        self.assertIn("Historical capture", out)
        code, out = self.geofuse("preset", "show", "ndvi", "Sentinel-2 10 m")
        self.assertIn('preset = "Sentinel-2 10 m"', out)
        self.assertIn("resolution_m = 10", out)
        code, out = self.geofuse("preset", "show", "ndvi", "Nope")
        self.assertEqual(code, 2)
        self.assertIn("known: Default", out)

    def test_save_from_flags_and_from_a_run_file_step(self):
        code, out = self.geofuse(
            "preset",
            "save",
            "ndvi",
            "Cloudy",
            "--cloud-max",
            "60",
            "--description",
            "lenient",
        )
        self.assertEqual(code, 0, out)
        self.assertEqual(presets.load_settings("ndvi", "Cloudy"), {"cloud_max": 60})
        code, out = self.geofuse("preset", "save", "ndvi", "Cloudy", "--cloud-max", "5")
        self.assertEqual(code, 2)
        self.assertIn("--overwrite", out)
        code, out = self.geofuse("preset", "save", "ndvi", "X", "--test-size", "0.3")
        self.assertIn("--test-size is not a NDVI setting", out)

        run = os.path.join(self.dir, "run.toml")
        with open(run, "w", encoding="utf-8") as f:
            f.write(
                textwrap.dedent(
                    """
                [[ndvi]]
                name = "city"
                study_area = "a.gpkg"
                ranges = [[2020-06-01, 2020-08-31]]
                satellite = "sentinel2"
                water_mask = true
                """
                )
            )
        open(os.path.join(self.dir, "a.gpkg"), "w").close()
        code, out = self.geofuse(
            "preset", "save", "ndvi", "City", "--from", run, "--step", "city"
        )
        self.assertEqual(code, 0, out)
        self.assertEqual(
            presets.load_settings("ndvi", "City"),
            {"satellite": "sentinel2", "water_mask": True},
        )

    def test_import_and_remove(self):
        doc = presets.make_preset("gvi", "Fine grid", "", {"grid_resolution_m": 20})
        path = os.path.join(self.dir, "shared.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f)
        self.assertEqual(self.geofuse("preset", "import", path)[0], 0)
        settings = presets.load_settings("gvi", "Fine grid")
        self.assertEqual(settings, {"grid_resolution_m": 20})
        self.assertEqual(self.geofuse("preset", "remove", "gvi", "Fine grid")[0], 0)
        self.assertIsNone(presets.find_preset("gvi", "Fine grid"))
        code, out = self.geofuse("preset", "remove", "gvi", "Default")
        self.assertEqual(code, 2)
        self.assertIn("built-in", out)


class CacheTests(_Case):
    def setUp(self):
        super().setUp()
        self.out = os.path.join(self.dir, "out")
        greenery = os.path.join(caches.fusion_cache_dir(self.out), "greenery")
        os.makedirs(greenery)
        with open(os.path.join(greenery, "greenery-1.npz"), "wb") as f:
            f.write(b"z" * 4096)
        self.state = StateDir(self.state_dir)
        self.job = self.state.create_job(
            {
                "step": "fusion.fit",
                "kind": "fusion",
                "name": "fit",
                "output_dir": self.out,
            },
            run_id="r1",
            depends_on=[],
        )

    def test_list_shows_each_command_line_jobs_fusion_cache(self):
        code, out = self.geofuse("cache", "list")
        self.assertEqual(code, 0)
        self.assertIn("4.0 KB · 1 files", out)
        self.assertIn(caches.fusion_cache_dir(self.out), out)
        self.assertIn("GVI panorama cache", out)

    def test_purge_asks_and_waits_for_running_jobs(self):
        code, out = self.geofuse("cache", "purge", "fusion", stdin="n\n")
        self.assertEqual(code, 2)
        self.assertIn("--yes", out)
        self.state.write_status(self.job, {"status": "running"})
        self.state.claim(self.job, "client")
        with mock.patch.object(StateDir, "effective_status", lambda s, j, st, a: st):
            code, out = self.geofuse("cache", "purge", "fusion", "--yes")
        self.assertEqual(code, 2)
        self.assertIn("fit", out)
        self.state.write_status(self.job, {"status": "completed"})
        code, out = self.geofuse("cache", "purge", "all", "--yes")
        self.assertEqual(code, 0, out)
        self.assertIn("Removed 1 files", out)
        self.assertEqual(caches.greenery_cache_files(self.out), [])


if __name__ == "__main__":
    unittest.main()
