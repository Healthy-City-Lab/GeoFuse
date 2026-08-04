"""Tests for per-stage wall-clock timing and the concurrency reporter.

The timing feeds tuning decisions (which phase dominates, whether a worker pool
is the ceiling), so it has to survive persistence, tolerate records written
before stages carried timestamps, and not charge a retry for the attempt it
replaced.
"""

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (os.path.join(ROOT, "ui"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from geofuse.jobs.stage_ledger import StageLedger


class TestStageTiming(unittest.TestCase):
    def _ledger(self):
        return StageLedger.from_steps(
            [("load", "Load"), ("preagg", "Pre-aggregate"), ("opt", "Search")]
        )

    def test_set_status_times_a_stage(self):
        # The runner sets status directly rather than via mark_*, so that path
        # is the one that must record the clock.
        led = self._ledger()
        led.set_status("load", "running")
        led.set_status("load", "done", progress=1.0)
        s = led.get("load")
        self.assertIsNotNone(s.started_at)
        self.assertIsNotNone(s.ended_at)
        self.assertIsNotNone(s.duration_s)

    def test_progress_without_running_still_starts_the_clock(self):
        led = self._ledger()
        led.mark_progress("opt", 0.5)
        self.assertIsNotNone(led.get("opt").started_at)

    def test_a_running_stage_counts_up(self):
        led = self._ledger()
        led.mark_running("opt")
        self.assertIsNone(led.get("opt").ended_at)
        self.assertGreaterEqual(led.get("opt").duration_s, 0.0)

    def test_skipped_stage_has_no_duration(self):
        led = self._ledger()
        led.mark_skipped("preagg")
        self.assertIsNone(led.get("preagg").duration_s)

    def test_report_shares_sum_to_one_and_sort_descending(self):
        led = self._ledger()
        for key, start, end in (
            ("load", 1000.0, 1010.0),
            ("preagg", 1010.0, 1110.0),
            ("opt", 1110.0, 1160.0),
        ):
            s = led.get(key)
            s.status, s.started_at, s.ended_at = "done", start, end
        rows = led.timing_report()
        self.assertEqual([r[0] for r in rows], ["preagg", "opt", "load"])
        self.assertAlmostEqual(sum(r[3] for r in rows), 1.0, places=9)
        self.assertAlmostEqual(rows[0][3], 100.0 / 160.0, places=9)

    def test_timings_survive_persist_and_reload(self):
        led = self._ledger()
        s = led.get("preagg")
        s.status, s.started_at, s.ended_at = "done", 500.0, 560.0
        reloaded = StageLedger.from_dict(led.to_dict())
        self.assertEqual(reloaded.get("preagg").duration_s, 60.0)

    def test_a_record_without_timestamps_still_loads(self):
        legacy = {
            "stages": [
                {"key": "load", "label": "Load", "status": "done", "progress": 1.0}
            ]
        }
        led = StageLedger.from_dict(legacy)
        self.assertIsNone(led.get("load").duration_s)
        self.assertEqual(led.timing_report(), [])

    def test_reset_unfinished_clears_the_abandoned_clock(self):
        led = self._ledger()
        led.mark_running("opt")
        led.reset_unfinished()
        s = led.get("opt")
        self.assertEqual(s.status, "pending")
        self.assertIsNone(s.started_at)
        self.assertIsNone(s.duration_s)

    def test_reset_keeps_finished_stage_timings(self):
        led = self._ledger()
        s = led.get("load")
        s.status, s.started_at, s.ended_at = "done", 1.0, 3.0
        led.mark_running("opt")
        led.reset_unfinished()
        self.assertEqual(led.get("load").duration_s, 2.0)


class TestParallelEfficiencyReport(unittest.TestCase):
    def _run(self, *, wall_s, busy_s, workers):
        from geofuse.fusion import _log_parallel_efficiency

        out = []
        _log_parallel_efficiency(
            lambda lvl, m: out.append(m),
            "phase",
            wall_s=wall_s,
            busy_s=busy_s,
            workers=workers,
        )
        return out[0] if out else ""

    def test_saturated_small_pool_is_called_pool_bound(self):
        # The real shape of pre-aggregation: ~7.7 of 8 workers busy.
        msg = self._run(wall_s=2563.0, busy_s=19862.0, workers=8)
        self.assertIn("pool-bound", msg)
        self.assertIn("of 8 worker(s)", msg)

    def test_idle_pool_is_called_work_bound(self):
        msg = self._run(wall_s=1000.0, busy_s=1500.0, workers=4)
        self.assertIn("work-bound", msg)

    def test_a_full_pool_on_every_core_is_not_flagged(self):
        cores = os.cpu_count() or 1
        msg = self._run(wall_s=100.0, busy_s=100.0 * cores, workers=cores)
        self.assertNotIn("pool-bound", msg)
        self.assertNotIn("work-bound", msg)

    def test_degenerate_input_logs_nothing(self):
        self.assertEqual(self._run(wall_s=0.0, busy_s=10.0, workers=4), "")
        self.assertEqual(self._run(wall_s=10.0, busy_s=0.0, workers=4), "")


if __name__ == "__main__":
    unittest.main()
