"""Resource slots shared by every GeoFuse process (``geofuse.jobs.locks``).

A job holds one slot per resource class while it runs: a second job needing a
full class waits and says so, a class with two slots admits two jobs, a job
cancelled while waiting holds nothing, and a holder that dies frees its slot.
The child entry points run their runner inside the slots.
"""

import os
import queue
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from geofuse.jobs import locks
from geofuse.jobs import subprocess_runner as child

LIMITS = {"gpu": 1, "streetview": 1, "earthengine": 2, "fusion": 1}


class _Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def hold(self, job_type, **kw):
        kw.setdefault("poll_s", 0.05)
        return locks.hold(job_type, folder=self.folder, limits=LIMITS, **kw)


class SlotTests(_Case):
    def test_a_second_job_waits_for_the_only_slot(self):
        said, got = [], []

        def second():
            with self.hold("fusion", waiting=said.append) as ok:
                got.append((ok, time.monotonic()))

        with self.hold("fusion") as ok:
            self.assertTrue(ok)
            waiter = threading.Thread(target=second)
            waiter.start()
            time.sleep(0.3)
            self.assertEqual(got, [])
            released = time.monotonic()
        waiter.join(10)
        self.assertTrue(got[0][0])
        self.assertGreaterEqual(got[0][1], released)
        self.assertEqual(len(said), 1)
        self.assertIn("fusion slot", said[0])

    def test_two_slots_admit_two_jobs(self):
        said = []
        with (
            self.hold("ndvi") as first,
            self.hold("ndvi_column", waiting=said.append) as second,
        ):
            self.assertTrue(first and second)
        self.assertEqual(said, [])

    def test_gvi_holds_the_gpu_and_street_view_but_not_earth_engine(self):
        with self.hold("gvi"):
            with self.hold("ndvi") as ok:
                self.assertTrue(ok)
            with self.hold("gvi_column", cancelled=lambda: True) as ok:
                self.assertFalse(ok)

    def test_a_job_cancelled_while_waiting_holds_nothing(self):
        ticks = []
        with self.hold("gvi"):
            with self.hold(
                "gvi", cancelled=lambda: len(ticks) >= 2, tick=lambda: ticks.append(1)
            ) as ok:
                self.assertFalse(ok)
        with self.hold("gvi", cancelled=lambda: True) as ok:
            self.assertTrue(ok)

    def test_unknown_job_types_need_no_slot(self):
        with self.hold("fusion"), self.hold("other") as ok:
            self.assertTrue(ok)

    def test_a_dead_holder_frees_its_slot(self):
        flag = os.path.join(self.folder, "held")
        script = textwrap.dedent(
            f"""
            import sys, time
            sys.path.insert(0, {ROOT!r})
            from geofuse.jobs.locks import hold
            with hold("fusion", folder={self.folder!r}, limits={LIMITS!r}):
                open({flag!r}, "w").close()
                time.sleep(120)
            """
        )
        proc = subprocess.Popen([sys.executable, "-c", script])
        try:
            end = time.monotonic() + 60
            while not os.path.exists(flag) and time.monotonic() < end:
                time.sleep(0.1)
            self.assertTrue(os.path.exists(flag))
            with self.hold("fusion", cancelled=lambda: True) as ok:
                self.assertFalse(ok)
        finally:
            proc.kill()
            proc.wait(30)
        with self.hold("fusion", cancelled=lambda: True) as ok:
            self.assertTrue(ok)


class _Ctx:
    def __init__(self, cancelled=False):
        self.cancelled = cancelled
        self.texts = []

    def is_cancelled(self):
        return self.cancelled

    def progress(self, value=None, status_text=None, **_):
        self.texts.append(status_text)

    def heartbeat(self):
        pass


class ChildEntryTests(_Case):
    def setUp(self):
        super().setUp()
        patches = [
            mock.patch.object(locks, "lock_dir", lambda: locks.Path(self.folder)),
            mock.patch.object(locks, "_limits", lambda: dict(LIMITS)),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_the_runner_runs_inside_the_slots(self):
        events, seen = queue.Queue(), []

        def runner(ctx, **kw):
            with self.hold("fusion", cancelled=lambda: True) as free:
                seen.append(free)
            return {"output_paths": [kw["out"]]}

        child._run_held("fusion", _Ctx(), events, runner, out="a.gpkg")
        self.assertEqual(seen, [False])
        self.assertEqual(events.get_nowait(), (child.MSG_COMPLETE, ["a.gpkg"]))

    def test_a_job_cancelled_while_waiting_completes_with_nothing(self):
        events, ctx = queue.Queue(), _Ctx(cancelled=True)
        ran = []
        with self.hold("gvi"):
            child._run_held("gvi", ctx, events, lambda c, **kw: ran.append(1))
        self.assertEqual(ran, [])
        self.assertEqual(events.get_nowait(), (child.MSG_COMPLETE, []))


if __name__ == "__main__":
    unittest.main()
