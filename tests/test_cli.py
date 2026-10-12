"""The ``geofuse`` command line.

Settings become flags that override every step having them; ``plan`` lays a
run file out without running it and exits 2 on bad input; templates load as
run files; the job commands act on a state folder; ``run`` drives a
foreground supervisor to the end; and ``submit`` starts a detached shared
supervisor that a stop request ends.
"""

import contextlib
import io
import os
import sys
import tempfile
import textwrap
import threading
import time
import tomllib
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import geopandas as gpd
from shapely.geometry import Point

from geofuse import runfile
from geofuse.cli import commands as cli
from geofuse.cli.templates import template
from geofuse.jobs.plans import Submission
from geofuse.jobs.state import StateDir
from geofuse.persistence.job_store import JobStore

RUN = """
[[ndvi]]
name = "city"
study_area = "city.gpkg"
ranges = [[2019-05-01, 2019-09-30], [2020-05-01, 2020-09-30]]
cloud_max = 30
"""


def _geofuse(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class _Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.state_dir = os.path.join(self.dir, "state")
        gpd.GeoDataFrame(
            {"year": [2019]}, geometry=[Point(-114.07, 51.04)], crs="EPSG:4326"
        ).to_file(os.path.join(self.dir, "city.gpkg"), driver="GPKG")
        self.run_file = os.path.join(self.dir, "run.toml")
        with open(self.run_file, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(RUN))
        self.config = os.path.join(self.dir, "config.toml")

    def tearDown(self):
        self.tmp.cleanup()

    def geofuse(self, *argv):
        return _geofuse("--state-dir", self.state_dir, "--config", self.config, *argv)


class ParserTests(unittest.TestCase):
    def test_every_setting_is_a_flag(self):
        args = cli.build_parser().parse_args(
            ["plan", "r.toml", "--cloud-max", "20", "--no-water-mask",
             "--statistics-ranges", "mean", "max", "--objective-metric", "spearman"]
        )
        overrides = {o.key: o for o in cli._overrides(args)}
        self.assertEqual(overrides["cloud_max"].value, 20)
        self.assertIs(overrides["water_mask"].value, False)
        self.assertEqual(overrides["statistics_ranges"].value, ["mean", "max"])
        self.assertEqual(overrides["objective_metric"].target, ("*",))

    def test_help_lists_the_setting_flags(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as done:
            cli.main(["plan", "--help"])
        self.assertEqual(done.exception.code, 0)
        self.assertIn("cloud cover (%)", out.getvalue())

    def test_set_overrides_come_after_flags(self):
        args = cli.build_parser().parse_args(
            ["plan", "r.toml", "--cloud-max", "20", "--set", "ndvi.city.cloud_max=5"]
        )
        keys = [(o.target, o.value) for o in cli._overrides(args)]
        self.assertEqual(keys, [(("*",), 20), (("ndvi", "city"), 5)])


class PlanAndConfigTests(_Case):
    def test_plan_lists_jobs_and_resolved_settings(self):
        code, out, _ = self.geofuse("plan", self.run_file, "--resolution-m", "30")
        self.assertEqual(code, 0)
        self.assertIn("cloud_max=30", out)
        self.assertIn("resolution_m=30", out)
        self.assertIn("city_20190501_20190930", out)
        self.assertIn("2 job(s).", out)

    def test_bad_input_exits_2_with_the_reason(self):
        code, _, err = self.geofuse("plan", self.run_file, "--set", "ndvi.cloud_maxx=3")
        self.assertEqual(code, 2)
        self.assertIn("cloud_maxx", err)
        code, _, err = self.geofuse("plan", self.run_file, "--only", "gvi.city")
        self.assertEqual(code, 2)
        self.assertIn("matches no step", err)

    def test_templates_are_valid_run_files(self):
        for kind in ("pipeline", "ndvi", "gvi", "fusion"):
            text = template(kind)
            doc = tomllib.loads(text)
            run = runfile.from_document(doc, base_dir=self.dir)
            self.assertTrue(run.steps, kind)

    def test_config_set_and_show(self):
        self.assertEqual(self.geofuse("config", "set", "limits.gpu", "2")[0], 0)
        code, out, _ = self.geofuse("config", "show")
        self.assertEqual(code, 0)
        self.assertIn("gpu=2", out)
        code, _, err = self.geofuse("config", "set", "limits.gpus", "2")
        self.assertEqual(code, 2)
        self.assertIn("limits.gpus", err)


class JobCommandTests(_Case):
    def submit_only(self):
        from geofuse.jobs.plans import expand
        from geofuse.jobs.supervisor import submit_specs

        run = runfile.load(self.run_file)
        state = StateDir(self.state_dir)
        return state, submit_specs(state, expand(run), run_name="t")[1]

    def test_status_cancel_pause_and_rm(self):
        state, ids = self.submit_only()
        code, out, _ = self.geofuse("status")
        self.assertEqual(code, 0)
        self.assertIn(ids[0], out)
        self.assertIn("not running", out)
        self.assertEqual(self.geofuse("cancel", ids[0][:5])[0], 0)
        self.assertEqual(state.job(ids[0]).state, "cancelled")
        self.geofuse("pause", ids[1])
        self.assertTrue(state.read_control(ids[1])["pause"])
        code, out, _ = self.geofuse("rm", "--finished")
        self.assertIn(f"{ids[0]}: removed", out)
        self.assertEqual([j.id for j in state.jobs()], [ids[1]])
        self.assertEqual(self.geofuse("logs", "nope")[0], 2)

    def test_run_drives_a_foreground_supervisor_to_the_end(self):
        class _Executor:
            def __init__(self, store, max_workers=4):
                self.store = store

            def submit_ndvi_subprocess(self, record, **_kw):
                def work():
                    self.store.transition(record.id, "running")
                    time.sleep(0.1)
                    self.store.transition(
                        record.id, "completed", output_paths=["x.tif"]
                    )

                threading.Thread(target=work, daemon=True).start()

            def shutdown(self, wait=True):
                pass

        def build(spec, **_kw):
            return Submission("ndvi", spec.name, {}, "submit_ndvi_subprocess", {})

        with mock.patch(
            "geofuse.persistence.job_executor.JobExecutor", _Executor
        ), mock.patch("geofuse.jobs.plans.build", build):
            code, out, _ = self.geofuse("run", self.run_file)
        self.assertEqual(code, 0, out)
        jobs = StateDir(self.state_dir).jobs()
        self.assertEqual([j.state for j in jobs], ["completed", "completed"])
        self.assertIn("completed", out)


class DaemonTests(_Case):
    def test_a_detached_supervisor_starts_and_stops_on_request(self):
        from geofuse.jobs.supervisor import start_daemon

        state = StateDir(self.state_dir)
        record = start_daemon(state, wait_s=60)
        try:
            self.assertEqual(record["mode"], "daemon")
            self.assertNotEqual(record["pid"], os.getpid())
            self.assertEqual(start_daemon(state)["id"], record["id"])
        finally:
            state.request_stop(record["id"])
            end = time.monotonic() + 60
            while time.monotonic() < end and state.live_supervisors():
                time.sleep(0.5)
            # The record goes just before the process ends; wait for the
            # process too, so its log file is closed before cleanup.
            import psutil

            try:
                psutil.Process(record["pid"]).wait(timeout=30)
            except psutil.NoSuchProcess:
                pass
        self.assertEqual(state.live_supervisors(), set())


if __name__ == "__main__":
    unittest.main()
