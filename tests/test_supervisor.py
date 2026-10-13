"""The command line's supervisor (``geofuse.jobs.supervisor``).

A stub executor stands in for the spawn processes: each job is a thread that
moves its record through the store the way the real watcher does, honouring
pause and cancel. The supervisor claims jobs, waits for their dependencies,
caps what runs by resource class, applies control requests, mirrors status
files, and skips jobs whose dependency failed.
"""

import os
import sys
import tempfile
import threading
import time
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from geofuse.jobs.plans import JobSpec, Submission
from geofuse.jobs.state import StateDir
from geofuse.jobs.supervisor import Supervisor, submit_specs
from geofuse.persistence.job_store import JobStore


class _Executor:
    """Runs each job on a thread: ``ticks`` steps of ``tick_s``, then done."""

    def __init__(self):
        self.store = JobStore(":memory:")
        self.running = 0
        self.peak = 0
        self._lock = threading.Lock()

    def submit_ndvi_subprocess(self, record, *, ticks=4, tick_s=0.05, fail=False):
        def work():
            with self._lock:
                self.running += 1
                self.peak = max(self.peak, self.running)
            self.store.transition(record.id, "running")
            try:
                done = 0
                while done < ticks:
                    if record.cancel_event.is_set():
                        self.store.transition(record.id, "cancelled")
                        return
                    if record.pause_event.is_set():
                        time.sleep(0.02)
                        continue
                    time.sleep(tick_s)
                    done += 1
                    self.store.update_progress(
                        record.id, progress=done / ticks, status_text=f"{done}/{ticks}"
                    )
                if fail:
                    self.store.transition(record.id, "error", error="boom")
                else:
                    self.store.transition(
                        record.id, "completed", output_paths=[f"{record.id}.tif"]
                    )
            finally:
                with self._lock:
                    self.running -= 1

        threading.Thread(target=work, daemon=True).start()

    def shutdown(self, wait=True):
        pass


def _spec(step, name, *, depends=(), resources=("earthengine",), **options):
    return JobSpec(
        step=step,
        kind="ndvi",
        name=name,
        output_dir=tempfile.gettempdir(),
        sources=[],
        merged=False,
        group=None,
        options=options,
        settings={},
        preset=None,
        depends_on=list(depends),
        resources=list(resources),
    )


def _build(spec, **_kw):
    if spec.options.get("broken"):
        raise ValueError("cannot read the study area")
    return Submission(
        "ndvi",
        spec.name,
        {},
        "submit_ndvi_subprocess",
        {k: v for k, v in spec.options.items() if k in ("ticks", "tick_s", "fail")},
    )


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = StateDir(self.tmp.name)
        self.executor = _Executor()

    def tearDown(self):
        self.executor.store.shutdown()
        self.tmp.cleanup()

    def supervisor(self, mode="foreground", **kw):
        return Supervisor(
            self.state,
            mode=mode,
            executor=self.executor,
            build=_build,
            poll_s=0.02,
            supervisor_id=kw.pop("supervisor_id", "fg1"),
            **kw,
        )

    def submit(self, specs, owner="fg1"):
        return submit_specs(self.state, specs, run_name="t", owner=owner)[1]

    def run_in_thread(self, sup):
        result = {}
        t = threading.Thread(target=lambda: result.setdefault("code", sup.run()))
        t.start()
        return t, result

    def test_dependencies_run_in_order_and_statuses_are_written(self):
        ids = self.submit(
            [_spec("ndvi.a", "a"), _spec("fusion.b", "b", depends=["ndvi.a"])]
        )
        sup = self.supervisor()
        self.assertEqual(sup.run(), 0)
        first, second = (self.state.job(i) for i in ids)
        self.assertEqual((first.state, second.state), ("completed", "completed"))
        self.assertLessEqual(first.status["completed_at"], second.status["started_at"])
        self.assertEqual(first.status["outputs"], [f"{ids[0]}.tif"])
        self.assertEqual(first.status["progress"], 1.0)
        self.assertEqual(second.spec["depends_on"], [ids[0]])

    def test_a_failed_dependency_skips_what_waits_on_it(self):
        ids = self.submit(
            [
                _spec("ndvi.a", "a", fail=True),
                _spec("fusion.b", "b", depends=["ndvi.a"]),
                _spec("ndvi.c", "c"),
            ]
        )
        self.assertEqual(self.supervisor().run(), 1)
        states = [self.state.job(i).state for i in ids]
        self.assertEqual(states, ["error", "skipped", "completed"])
        self.assertIn(ids[0], self.state.job(ids[1]).status["error"])

    def test_resource_classes_cap_what_runs_at_once(self):
        self.submit([_spec("ndvi.a", f"a{i}", ticks=6) for i in range(5)])
        sup = self.supervisor(limits={"earthengine": 2, "max_parallel": 4})
        self.assertEqual(sup.run(), 0)
        self.assertEqual(self.executor.peak, 2)

    def test_pause_resume_and_cancel_through_control_files(self):
        (job_id,) = self.submit([_spec("ndvi.a", "a", ticks=3000, tick_s=0.01)])
        sup = self.supervisor()
        t, result = self.run_in_thread(sup)

        def wait_for(pred, timeout=10.0):
            end = time.monotonic() + timeout
            while time.monotonic() < end:
                if pred(self.state.job(job_id)):
                    return True
                time.sleep(0.05)
            return False

        self.assertTrue(
            wait_for(lambda j: j.state == "running" and j.status.get("progress", 0))
        )
        self.state.request(job_id, pause=True)
        self.assertTrue(wait_for(lambda j: j.status.get("paused")))
        self.state.request(job_id, pause=False)
        self.assertTrue(wait_for(lambda j: not j.status.get("paused")))
        self.state.request(job_id, cancel=True)
        t.join(15)
        self.assertEqual(self.state.job(job_id).state, "cancelled")
        self.assertEqual(result["code"], 1)

    def test_a_job_that_cannot_be_built_fails_alone(self):
        ids = self.submit([_spec("ndvi.a", "a", broken=True), _spec("ndvi.b", "b")])
        self.assertEqual(self.supervisor().run(), 1)
        bad, good = (self.state.job(i) for i in ids)
        self.assertEqual((bad.state, good.state), ("error", "completed"))
        self.assertIn("cannot read the study area", bad.status["error"])
        self.assertIn("Traceback", bad.status["error_detail"])

    def test_the_daemon_takes_unreserved_jobs_and_a_foreground_run_its_own(self):
        shared = self.submit([_spec("ndvi.a", "shared")], owner=None)
        own = self.submit([_spec("ndvi.b", "own")], owner="fg1")
        fg = self.supervisor()
        self.assertEqual(fg.run(), 0)
        self.assertEqual(self.state.job(own[0]).state, "completed")
        self.assertEqual(self.state.job(shared[0]).state, "queued")
        daemon = self.supervisor(
            mode="daemon", supervisor_id="d1", limits={"idle_minutes": 1}
        )
        t = threading.Thread(target=daemon.run)
        t.start()
        end = time.monotonic() + 10
        while time.monotonic() < end and self.state.job(shared[0]).state != "completed":
            time.sleep(0.05)
        daemon.stop()
        t.join(10)
        self.assertEqual(self.state.job(shared[0]).state, "completed")
        self.assertEqual(self.state.live_supervisors(), set())

    def test_held_jobs_wait_and_cancel_all_cancels_queued_ones(self):
        ids = self.submit([_spec("ndvi.a", "a", ticks=100, tick_s=0.02)])
        self.state.request(ids[0], pause=True)
        sup = self.supervisor()
        t, result = self.run_in_thread(sup)
        end = time.monotonic() + 5
        while time.monotonic() < end:
            if self.state.job(ids[0]).status.get("text") == "Held":
                break
            time.sleep(0.05)
        self.assertEqual(self.state.job(ids[0]).state, "queued")
        sup.cancel_all()
        t.join(10)
        self.assertEqual(self.state.job(ids[0]).state, "cancelled")


if __name__ == "__main__":
    unittest.main()
