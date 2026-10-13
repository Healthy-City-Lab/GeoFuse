"""The GUI's Jobs tab (``ui/tabs/jobs.py``), driven through Streamlit's AppTest.

Local command-line jobs list newest first; the selected job shows its stages,
changed settings, outputs and log; Pause and a confirmed Cancel reach the job's
control file; Re-run queues it again; Load results puts a fusion run's bundle
into the results view; the filter narrows the list; a remote's mirrored jobs
show with a warning when its link is down, and nothing is sent over it then.
"""

import json
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

from geofuse.jobs.remote import Mirror
from geofuse.jobs.state import StateDir

_APP = """
import os, sys
for p in ({root!r}, os.path.join({root!r}, "ui")):
    if p not in sys.path:
        sys.path.insert(0, p)
from geofuse import userconfig
userconfig.load = lambda path=None: userconfig.Config(
    state_dir={state!r}, remotes={{"lab": {{"host": "lab.example.org"}}}}
)
import geofuse.jobs.supervisor as supervisor
supervisor.start_daemon = lambda state, **kw: {{}}
from tabs import jobs
jobs.render()
"""


def _ago(seconds: float) -> str:
    stamp = datetime.now(UTC) - timedelta(seconds=seconds)
    return stamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class JobsTabTests(unittest.TestCase):
    def setUp(self):
        from geofuse import userconfig
        from geofuse.jobs import supervisor

        # The app script replaces these in this process; put them back after.
        self.addCleanup(setattr, userconfig, "load", userconfig.load)
        self.addCleanup(setattr, supervisor, "start_daemon", supervisor.start_daemon)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.tmp.name
        self.state = StateDir(os.path.join(root, "state"))
        self.state.write_supervisor(
            "s1", {"pid": os.getpid(), "host": socket.gethostname(), "mode": "daemon"}
        )
        self.running = self.job("fusion", "fit", {"n_trials": 40})
        self.state.claim(self.running, "s1")
        ledger = {"stages": [
            {"key": "preaggr", "label": "Pre-aggregation", "status": "done"},
            {"key": "sweep", "label": "Sweep", "status": "running", "progress": 0.5},
        ]}
        self.state.write_status(
            self.running,
            {"status": "running", "progress": 0.5, "started_at": _ago(60),
             "text": "Sweeping", "stage_ledger": ledger, "owner": "s1"},
        )
        self.state.logs_dir.mkdir(parents=True, exist_ok=True)
        self.state.log_path(self.running).write_bytes(b"first line\nsecond line\n")
        self.done = self.job("fusion", "cesd")
        out = os.path.join(root, "out", "fusion", "run1")
        os.makedirs(out)
        with open(os.path.join(out, "results_bundle.json"), "w") as f:
            json.dump({"outcome": "cesd"}, f)
        self.state.claim(self.done, "s1")
        self.state.write_status(
            self.done, {"status": "completed", "progress": 1.0, "outputs": [out]}
        )
        self.queued = self.job("ndvi", "streets")
        mirror = Mirror(self.state, "lab")
        mirror.write_snapshot({"jobs": [{
            "id": "feedbeef",
            "spec": {"kind": "gvi", "name": "remote-gvi", "step": "gvi.x"},
            "status": {"status": "completed", "progress": 1.0, "outputs": ["/x"]},
        }]})
        mirror.write_link("stopped", host="me@lab.example.org")

        path = os.path.join(root, "app.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(_APP.format(root=ROOT, state=str(self.state.root)))
        from streamlit.testing.v1 import AppTest

        self.at = AppTest.from_file(path, default_timeout=60)
        self.at.run()
        self.assertFalse(self.at.exception, [e.value for e in self.at.exception])

    def job(self, kind, name, settings=None):
        time.sleep(0.01)
        spec = {"step": f"{kind}.{name}", "kind": kind, "name": name,
                "settings": settings or {}, "output_dir": "out", "sources": [],
                "merged": False, "group": None, "options": {}, "preset": None}
        return self.state.create_job(spec, run_id="r1", depends_on=[])

    def text(self) -> str:
        parts = [m.value for m in self.at.markdown] + [c.value for c in self.at.code]
        parts += [w.value for w in self.at.warning] + [c.value for c in self.at.caption]
        return "\n".join(map(str, parts))

    def select(self, job_id):
        self.at.selectbox(key="job_selected").set_value(job_id).run()

    def test_local_jobs_and_the_selected_job(self):
        table = self.at.dataframe[0].value
        self.assertEqual(list(table["ID"]), [self.queued, self.done, self.running])
        self.select(self.running)
        text = self.text()
        for part in ("Pre-aggregation", "Sweep · 50%", "n_trials = 40", "second line"):
            self.assertIn(part, text)
        self.at.button(key="job_pause").click().run()
        self.assertTrue(self.state.read_control(self.running)["pause"])
        self.assertIn(f"Cancel **fit** ({self.running})?", self.text())
        self.assertNotIn("cancel", self.state.read_control(self.running))
        self.at.button(key="job_cancel_confirm").click().run()
        self.assertTrue(self.state.read_control(self.running)["cancel"])

    def test_rerun_and_load_results(self):
        self.select(self.done)
        files = list(self.at.dataframe[1].value["File"])
        self.assertTrue(files[0].endswith("results_bundle.json"), files)
        self.at.button(key="job_load_results").click().run()
        self.assertEqual(self.at.session_state["fusion_results"], {"outcome": "cesd"})
        before = set(self.state.job_ids())
        self.at.button(key="job_rerun").click().run()
        (new,) = set(self.state.job_ids()) - before
        self.assertEqual(self.state.read_spec(new)["restart_of"], self.done)

    def test_filter_and_finished_toggle(self):
        self.at.text_input(key="job_filter").set_value("streets").run()
        self.assertEqual(list(self.at.dataframe[0].value["ID"]), [self.queued])
        self.at.text_input(key="job_filter").set_value("").run()
        self.at.checkbox(key="job_show_finished").uncheck().run()
        ids = list(self.at.dataframe[0].value["ID"])
        self.assertEqual(ids, [self.queued, self.running])

    def test_a_remote_with_its_link_down(self):
        self.at.selectbox(key="job_source").set_value("lab").run()
        self.assertEqual(list(self.at.dataframe[0].value["ID"]), ["feedbeef"])
        self.assertIn("geofuse remote attach lab", self.text())
        self.at.button(key="job_fetch").click().run()
        toasts = " ".join(t.value for t in self.at.toast)
        self.assertIn("not sent", toasts)
        self.assertFalse(Mirror(self.state, "lab").requests())


if __name__ == "__main__":
    unittest.main()
