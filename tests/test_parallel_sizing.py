"""Worker sizing and the parallel cluster bootstrap.

Pool size must follow the host rather than a constant, and must never change a
result: the bootstrap draws its replicates before fitting any of them, so the
CI is identical whatever the pool size turns out to be on the machine that runs
it.
"""

import os
import sys
import unittest
import warnings

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from geofuse import parallel


class TestWorkerCount(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("GEOFUSE_WORKERS", None)

    def tearDown(self):
        os.environ.pop("GEOFUSE_WORKERS", None)
        if self._saved is not None:
            os.environ["GEOFUSE_WORKERS"] = self._saved

    def test_scales_with_the_host_but_leaves_room_for_blas(self):
        cores = parallel.cpu_budget()
        n = parallel.worker_count()
        self.assertGreaterEqual(n, 1)
        self.assertLessEqual(n, cores)
        if cores > 2:
            # A third of the cores — throughput peaks there because each task
            # is already multi-threaded inside.
            self.assertEqual(n, max(2, cores // 3))

    def test_never_returns_less_than_one(self):
        self.assertGreaterEqual(parallel.worker_count(), 1)

    def test_cap_bounds_the_pool(self):
        self.assertLessEqual(parallel.worker_count(cap=2), 2)

    def test_env_override_wins(self):
        os.environ["GEOFUSE_WORKERS"] = "3"
        self.assertEqual(parallel.worker_count(), 3)
        # A cap still bounds an override — a phase cannot use more threads
        # than it has independent work for.
        self.assertEqual(parallel.worker_count(cap=2), 2)

    def test_garbage_override_falls_back_to_the_host(self):
        os.environ.pop("GEOFUSE_WORKERS", None)
        default = parallel.worker_count()
        for bad in ("", "  ", "abc", "0", "-4"):
            os.environ["GEOFUSE_WORKERS"] = bad
            self.assertEqual(parallel.worker_count(), default, bad)

    def test_pool_never_exceeds_the_work(self):
        self.assertEqual(parallel.workers_for(0), 1)
        self.assertEqual(parallel.workers_for(1), 1)
        self.assertLessEqual(parallel.workers_for(3), 3)


class TestClusterBootstrapIsPoolInvariant(unittest.TestCase):
    """The CI must not depend on how many threads happened to be available."""

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        rng = np.random.default_rng(11)
        n_ent, waves = 90, 3
        cls.eid = np.repeat(np.arange(n_ent), waves)
        n = len(cls.eid)
        cls.t = np.tile(np.arange(waves, dtype=float), n_ent)
        cls.wave = np.tile(np.arange(waves), n_ent)
        u0 = rng.normal(0, 1.0, n_ent)[cls.eid]
        cls.g = rng.normal(0, 1, n)
        cls.cov = rng.normal(0, 1, (n, 2))
        cls.y = (
            1.0
            + u0
            + 0.4 * cls.g
            + 0.2 * cls.t
            + cls.cov @ np.array([0.5, -0.3])
            + rng.normal(0, 1.0, n)
        )

    def _ci(self, workers):
        from geofuse import mixed_effects_scoring as mes

        return mes.cluster_bootstrap_metric_ci(
            "mixedlm_coef",
            self.y,
            self.g,
            self.eid,
            self.t,
            covariates=self.cov,
            include_time_fixed=True,
            random_slope=False,
            wave_index=self.wave,
            n_bootstrap=8,
            ci_level=0.95,
            seed=42,
            workers=workers,
        )

    def test_threaded_matches_serial_exactly(self):
        serial = self._ci(1)
        threaded = self._ci(4)
        for key in ("observed", "mean", "lower", "upper", "n_boot", "n"):
            a, b = serial.get(key), threaded.get(key)
            if a is None or b is None:
                self.assertEqual(a, b, key)
            else:
                self.assertAlmostEqual(float(a), float(b), places=12, msg=key)

    def test_default_is_serial(self):
        # Threading the exact MixedLM refits measured 0.62x of serial, so the
        # default must stay sequential until a process pool replaces it.
        serial = self._ci(1)
        auto = self._ci(None)
        self.assertAlmostEqual(float(serial["lower"]), float(auto["lower"]), places=12)
        self.assertAlmostEqual(float(serial["upper"]), float(auto["upper"]), places=12)

    def test_a_real_interval_came_back(self):
        r = self._ci(4)
        self.assertEqual(r["method"], "cluster_bootstrap")
        self.assertGreater(r["n_boot"], 0)
        self.assertLessEqual(float(r["lower"]), float(r["upper"]))


if __name__ == "__main__":
    unittest.main()
