"""The command line's job state folder (``geofuse.jobs.state``).

A job is a folder: its spec written once, its status written by the
supervisor that claimed it, and control requests written by any client. A job
whose supervisor's heartbeat has stopped reads as interrupted; files are
replaced in one step and unreadable ones read as missing.
"""

import json
import os
import socket
import sys
import tempfile
import time
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from geofuse.jobs import state
from geofuse.jobs.state import StateDir


class _Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = StateDir(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def new_job(self, **kw):
        spec = {"step": "ndvi.city", "kind": "ndvi", "name": "city"}
        return self.state.create_job(spec, run_id="r1", depends_on=[], **kw)


class JobFileTests(_Case):
    def test_a_new_job_is_queued_with_its_spec(self):
        job_id = self.new_job()
        job = self.state.job(job_id)
        self.assertEqual(job.state, "queued")
        self.assertEqual(job.name, "city")
        self.assertEqual(job.spec["run_id"], "r1")
        self.assertEqual(self.state.find(job_id[:3]), [job_id])

    def test_one_supervisor_claims_a_job(self):
        job_id = self.new_job()
        self.assertTrue(self.state.claim(job_id, "s1"))
        self.assertTrue(self.state.claim(job_id, "s1"))
        self.assertFalse(self.state.claim(job_id, "s2"))
        self.assertEqual(self.state.owner(job_id), "s1")

    def test_control_requests_accumulate(self):
        job_id = self.new_job()
        self.state.request(job_id, pause=True)
        self.state.request(job_id, cancel=True)
        control = self.state.read_control(job_id)
        self.assertTrue(control["pause"] and control["cancel"])
        self.state.request(job_id, pause=False)
        self.assertFalse(self.state.read_control(job_id)["pause"])

    def test_jobs_list_oldest_first_and_remove(self):
        first = self.new_job()
        time.sleep(0.01)
        second = self.new_job()
        self.assertEqual([j.id for j in self.state.jobs()], [first, second])
        self.state.remove_job(first)
        self.assertEqual([j.id for j in self.state.jobs()], [second])

    def test_unreadable_files_read_as_missing(self):
        job_id = self.new_job()
        with open(self.state.job_dir(job_id) / "status.json", "w") as f:
            f.write("{half")
        self.assertEqual(self.state.read_status(job_id), {})
        self.assertIsNone(state.read_json(self.state.root / "nope.json"))

    def test_writes_replace_the_file_whole(self):
        path = self.state.root / "x.json"
        state.write_json(path, {"a": 1})
        state.write_json(path, {"a": 2})
        self.assertEqual(json.loads(path.read_text()), {"a": 2})
        self.assertEqual([p.name for p in self.state.root.iterdir()], ["x.json"])


class SupervisorLivenessTests(_Case):
    def supervise(self, sup_id, *, pid=None, beat_age=0.0):
        record = {
            "pid": pid or os.getpid(),
            "host": socket.gethostname(),
            "mode": "daemon",
        }
        self.state.write_supervisor(sup_id, record)
        if beat_age:
            path = self.state.supervisors_dir / f"{sup_id}.json"
            data = json.loads(path.read_text())
            data["heartbeat"] = "2000-01-01T00:00:00.000000Z"
            path.write_text(json.dumps(data))

    def test_a_fresh_heartbeat_keeps_jobs_running(self):
        job_id = self.new_job()
        self.supervise("s1")
        self.state.claim(job_id, "s1")
        self.state.write_status(job_id, {"status": "running"})
        self.assertEqual(self.state.job(job_id).state, "running")
        self.assertEqual(self.state.daemon()["id"], "s1")

    def test_a_stopped_supervisor_leaves_its_jobs_interrupted(self):
        job_id = self.new_job()
        self.supervise("s1", beat_age=3600)
        self.state.claim(job_id, "s1")
        self.state.write_status(job_id, {"status": "running"})
        job = self.state.job(job_id)
        self.assertEqual(job.state, "interrupted")
        self.assertIn("supervisor stopped", job.status["error"])
        self.assertIsNone(self.state.daemon())

    def test_an_unclaimed_queued_job_is_still_queued(self):
        job_id = self.new_job()
        self.assertEqual(self.state.job(job_id).state, "queued")

    def test_a_dead_process_on_this_host_is_not_alive(self):
        self.supervise("s1", pid=2**30)
        self.assertEqual(self.state.live_supervisors(), set())

    def test_finished_jobs_keep_their_status(self):
        job_id = self.new_job()
        self.state.claim(job_id, "gone")
        self.state.write_status(job_id, {"status": "completed"})
        self.assertEqual(self.state.job(job_id).state, "completed")


if __name__ == "__main__":
    unittest.main()
