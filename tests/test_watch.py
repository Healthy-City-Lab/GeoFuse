"""The ``geofuse watch`` monitor (``geofuse.cli.watch``).

Driven headless through Textual's pilot against a state folder: rows show
newest first with the time left; the detail pane shows stages, changed
settings and outputs; the log pane follows the selected job; keys pause,
cancel (after confirming), toggle finished jobs, filter and move to a remote's
mirrored jobs, whose stale link the title states; quitting leaves the jobs as
they were.
"""

import asyncio
import os
import socket
import sys
import tempfile
import time
import unittest
from datetime import UTC, datetime, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from textual.widgets import DataTable, RichLog, Static

from geofuse.cli.watch import LocalSource, WatchApp, details, eta
from geofuse.jobs.remote import Mirror, MirrorSource
from geofuse.jobs.state import StateDir


def _ago(seconds: float) -> str:
    stamp = datetime.now(UTC) - timedelta(seconds=seconds)
    return stamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class _Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = StateDir(self.tmp.name)
        self.state.write_supervisor(
            "s1", {"pid": os.getpid(), "host": socket.gethostname(), "mode": "daemon"}
        )
        self.running = self.job("fusion", "fit", {"n_trials": 40, "test_size": 0.25})
        self.state.claim(self.running, "s1")
        ledger = {
            "stages": [
                {"key": "preaggr", "label": "Pre-aggregation", "status": "done"},
                {
                    "key": "sweep",
                    "label": "Sweep",
                    "status": "running",
                    "progress": 0.5,
                    "message": "form 2 of 3",
                },
                {"key": "posterior", "label": "Posterior", "status": "pending"},
            ]
        }
        self.state.write_status(
            self.running,
            {
                "status": "running",
                "progress": 0.5,
                "started_at": _ago(60),
                "text": "Sweeping",
                "stage_ledger": ledger,
                "owner": "s1",
            },
        )
        self.log = self.state.log_path(self.running)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.log.write_text("first line\nsecond line\n", encoding="utf-8")
        self.queued = self.job("ndvi", "city")
        self.done = self.job("gvi", "streets")
        self.state.claim(self.done, "s1")
        self.state.write_status(
            self.done, {"status": "completed", "progress": 1.0, "outputs": ["a.gpkg"]}
        )

    def tearDown(self):
        self.tmp.cleanup()

    def job(self, kind, name, settings=None):
        time.sleep(0.01)
        spec = {
            "step": f"{kind}.{name}",
            "kind": kind,
            "name": name,
            "settings": settings or {},
            "output_dir": "out",
        }
        return self.state.create_job(spec, run_id="r1", depends_on=[])


class CellTests(_Case):
    def test_time_left_follows_progress(self):
        job = self.state.job(self.running)
        self.assertEqual(eta(job), "1m")
        self.assertEqual(eta(self.state.job(self.queued)), "")

    def test_details_show_stages_changed_settings_and_outputs(self):
        text = details(self.state.job(self.running)).plain
        for part in (
            "Pre-aggregation",
            "Sweep  50%  form 2 of 3",
            "Posterior",
            "n_trials = 40",
            "none yet; written to out",
        ):
            self.assertIn(part, text)
        self.assertNotIn("test_size", text)
        self.assertIn("a.gpkg", details(self.state.job(self.done)).plain)


class MonitorTests(_Case):
    def test_keys_act_on_the_selected_job(self):
        asyncio.run(self._scenario())

    async def _scenario(self):
        app = WatchApp(LocalSource(self.state), refresh_s=0.1)
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause(0.3)
            table = app.query_one(DataTable)
            self.assertEqual(app._order, [self.done, self.queued, self.running])
            self.assertEqual(table.row_count, 3)

            await pilot.press("down", "down")
            await pilot.pause(0.3)
            self.assertEqual(app.selected, self.running)
            body = str(app.query_one("#details-body", Static).render())
            self.assertIn("Pre-aggregation", body)
            log = app.query_one(RichLog)
            self.assertEqual(len(log.lines), 2)
            with open(self.log, "a", encoding="utf-8") as f:
                f.write("third line\npartial")
            await pilot.pause(0.4)
            self.assertEqual(len(log.lines), 3)

            await pilot.press("p")
            await pilot.pause(0.2)
            self.assertTrue(self.state.read_control(self.running)["pause"])

            await pilot.press("up")
            await pilot.pause(0.2)
            self.assertEqual(app.selected, self.queued)
            await pilot.press("c")
            await pilot.pause(0.2)
            await pilot.press("n")
            await pilot.pause(0.2)
            self.assertEqual(self.state.job(self.queued).state, "queued")
            await pilot.press("c")
            await pilot.pause(0.2)
            await pilot.press("y")
            await pilot.pause(0.3)
            self.assertEqual(self.state.job(self.queued).state, "cancelled")

            await pilot.press("a")
            await pilot.pause(0.3)
            self.assertEqual(app._order, [self.running])
            await pilot.press("a", "slash", "s", "t", "r", "enter")
            await pilot.pause(0.3)
            self.assertEqual(app._order, [self.done])

            await pilot.press("q")
        self.assertEqual(self.state.job(self.running).state, "running")

    def test_the_source_key_moves_to_a_remote(self):
        asyncio.run(self._remote_scenario())

    async def _remote_scenario(self):
        mirror = Mirror(self.state, "lab")
        remote_job = {
            "id": "feedbeef",
            "spec": {"kind": "gvi", "name": "remote-streets", "step": "gvi.x"},
            "status": {"status": "running", "progress": 0.25},
        }
        mirror.write_snapshot({"jobs": [remote_job], "supervisor": None})
        mirror.write_link("stopped", host="me@lab.example.org")
        local = LocalSource(self.state)
        app = WatchApp(
            local, sources=[local, MirrorSource(self.state, "lab")], refresh_s=0.1
        )
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause(0.3)
            self.assertEqual(len(app._order), 3)
            await pilot.press("s")
            await pilot.pause(0.3)
            self.assertEqual(app._order, ["feedbeef"])
            self.assertIn("lab · me@lab.example.org · stopped · stale", app.sub_title)
            self.assertIn("jobs as of", app.sub_title)
            await pilot.press("s")
            await pilot.pause(0.3)
            self.assertEqual(len(app._order), 3)
            await pilot.press("q")


if __name__ == "__main__":
    unittest.main()
