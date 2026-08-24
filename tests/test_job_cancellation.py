"""What has to happen when a job is stopped, and when one dies on its own.

Cancellation here is cooperative: the parent raises a flag and the child puts
itself down at the next point its current phase checks. That is the right
default and it is not sufficient on its own, because a phase can be inside a
call that cannot answer — a NUTS run, an exact MixedLM refit. So the parent
also stops waiting, and a child that has stopped answering is killed rather
than left to hold a "stopping" job open indefinitely.

The second half is the tree. A job child owns a process pool of its own, and
killing a process on Windows does not touch its children. What reaps them is
the parent guard every pool worker installs, so these tests kill a child and
check that its workers went with it rather than checking that it exited.
"""

import multiprocessing as mp
import os
import queue
import subprocess
import sys
import time
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from geofuse.jobs import subprocess_runner as sr
from geofuse.persistence.job_executor import JobExecutor


class _FakeProc:
    """Process handle that reports itself alive, like one ignoring a cancel."""

    exitcode = None

    def is_alive(self) -> bool:
        return True


class _Store:
    """Swallows the parent-side effects; these tests only read the verdict."""

    def __getattr__(self, _name):
        return lambda *a, **k: None


class TestAStoppedChildIsNotWaitedOnForever(unittest.TestCase):
    def test_a_child_that_will_not_wind_down_is_declared_cancelled(self):
        q: queue.Queue = queue.Queue()
        cancel = mp.Event()
        cancel.set()
        t0 = time.monotonic()
        status, payload = sr.drain_events_until_done(
            "job", _Store(), q, cancel,
            process_handle=_FakeProc(), poll_timeout=0.05, cancel_grace_s=0.5,
        )
        self.assertEqual((status, payload), ("cancelled", None))
        self.assertLess(time.monotonic() - t0, 5.0)

    def test_a_child_that_finishes_inside_the_grace_still_completes(self):
        q: queue.Queue = queue.Queue()
        q.put((sr.MSG_COMPLETE, ["out.json"]))
        cancel = mp.Event()
        cancel.set()
        status, payload = sr.drain_events_until_done(
            "job", _Store(), q, cancel,
            process_handle=_FakeProc(), poll_timeout=0.05, cancel_grace_s=5.0,
        )
        self.assertEqual((status, payload), ("completed", ["out.json"]))

    def test_an_uncancelled_job_is_never_hurried(self):
        q: queue.Queue = queue.Queue()
        cancel = mp.Event()

        def _finish():
            time.sleep(0.4)
            q.put((sr.MSG_COMPLETE, []))

        import threading
        threading.Thread(target=_finish, daemon=True).start()
        status, _ = sr.drain_events_until_done(
            "job", _Store(), q, cancel,
            process_handle=_FakeProc(), poll_timeout=0.05, cancel_grace_s=0.1,
        )
        self.assertEqual(status, "completed")


class TestTeardownSurvivesAChildThatNeverRan(unittest.TestCase):
    def test_a_handle_that_was_never_started_is_left_alone(self):
        # A spawn that raised leaves a handle that cannot even be joined, and
        # the watcher's cleanup runs over it just the same.
        proc = mp.get_context("spawn").Process(target=print, args=("never runs",))
        JobExecutor._stop_process(proc, grace_s=0.0)
        self.assertIsNone(proc.exitcode)


class TestKillingAChildTakesItsPoolWithIt(unittest.TestCase):
    """The escalation has to clear the tree, not just the process it signals.

    Driven by ``_pool_teardown_probe.py`` rather than run here: a spawn worker
    re-executes its parent's ``__main__``, so a pool started from inside a test
    runner behaves differently depending on which tests ran before it.
    """

    @staticmethod
    def _run_probe(*args):
        probe = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "_pool_teardown_probe.py")
        done = subprocess.run([sys.executable, probe, *args],
                              capture_output=True, text=True, timeout=300)
        report = "\n".join(
            x for x in (done.stdout.strip(), done.stderr.strip()) if x)
        return done.returncode, report

    def test_the_workers_do_not_outlive_the_child(self):
        code, report = self._run_probe()
        self.assertEqual(code, 0, report)

    def test_a_pool_without_the_guard_does_leak(self):
        # Negative control. Without it the check above passes for any reason at
        # all, including a probe that has stopped looking properly — an earlier
        # version reported a clean tree because it re-walked from the top, and
        # Windows leaves an orphan pointing at a pid that no longer exists.
        code, report = self._run_probe("--bare-pool")
        self.assertEqual(code, 4, report)
        self.assertIn("outlived the child", report)


if __name__ == "__main__":
    unittest.main()
