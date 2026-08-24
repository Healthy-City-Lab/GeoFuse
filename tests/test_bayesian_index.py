"""Validation gates for the sweep + posterior discovery engine.

These are the checks that decide whether a reported discovery means anything:
a planted signal has to be recovered, the parsimony tiebreak must not quietly
swap which channels are active, a radius picked at the edge of the search range
has to be flagged, and the whole procedure re-run on a permuted outcome must
not keep firing. A run that passes the smoke test but fails these is producing
confident noise.

The sweeps here run with a single worker on purpose. A spawned process pool
cannot re-import ``__main__`` once another test in the same interpreter has
replaced it, which made these results depend on test ordering rather than on
the code.
"""

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from geofuse import JobCancelled
from geofuse import bayesian_index as bi

RADII = np.array([200.0, 400.0, 600.0, 800.0])
STATS = ["mean", "p50", "p90"]


def _planted(n=1500, channel=1, radius=2, stat=1, strength=0.6, seed=0):
    """Synthetic cube whose outcome depends on exactly one (channel, radius, stat)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2, len(RADII), len(STATS)))
    y = strength * X[:, channel, radius, stat] + rng.normal(size=n)
    return bi.prep(X, y, None)[::-1]  # (Xr, yr)


class TestRecoversAPlantedSignal(unittest.TestCase):
    def test_the_sweep_finds_the_column_the_outcome_was_built_from(self):
        Xr, yr = _planted()
        res = bi.sweep(
            Xr, RADII, STATS, yr,
            channels=("a", "b"), channel_index=(0, 1), splits=6, workers=1,
        )
        self.assertEqual(res.picked[1], (600, "p50"))

    def test_the_weight_lands_on_the_channel_carrying_the_signal(self):
        Xr, yr = _planted()
        res = bi.sweep(
            Xr, RADII, STATS, yr,
            channels=("a", "b"), channel_index=(0, 1), splits=6, workers=1,
        )
        Z = Xr.reshape(len(Xr), -1)[:, list(res.columns)]
        kept, w = bi.simplex_fit(Z.T @ Z, Z.T @ yr)
        self.assertIn(1, kept)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=9)
        self.assertTrue(np.all(w >= 0.0))
        # The noise channel cannot carry the majority of a simplex weight.
        share = dict(zip(kept, w))
        self.assertGreater(share.get(1, 0.0), share.get(0, 0.0))


class TestParsimonyTiebreak(unittest.TestCase):
    """One-SE ranking must not change which channels are active.

    Ranking across channel sets let the surviving channel flip, which reverses
    the sign of the reported effect while claiming to be the same model.
    """

    def test_the_one_se_pick_keeps_the_same_active_channels(self):
        Xr, yr = _planted()
        res = bi.sweep(
            Xr, RADII, STATS, yr,
            channels=("a", "b"), channel_index=(0, 1), splits=6, workers=1,
        )
        self.assertEqual(len(res.one_se_columns), len(res.columns))
        n_radii, n_stats = Xr.shape[2], Xr.shape[3]
        picked_ch = [c // (n_radii * n_stats) for c in res.columns]
        one_se_ch = [c // (n_radii * n_stats) for c in res.one_se_columns]
        self.assertEqual(picked_ch, one_se_ch)

    def test_the_one_se_pick_is_no_larger_than_the_winner(self):
        Xr, yr = _planted()
        res = bi.sweep(
            Xr, RADII, STATS, yr,
            channels=("a", "b"), channel_index=(0, 1), splits=6, workers=1,
        )
        n_radii, n_stats = Xr.shape[2], Xr.shape[3]
        one_se = bi._decode(res.one_se_columns, n_radii, n_stats, RADII, STATS)
        self.assertLessEqual(
            sum(r for r, _ in one_se), sum(r for r, _ in res.picked)
        )


class TestBoundaryIsFlagged(unittest.TestCase):
    """A radius at the edge of the range means the optimum may be outside it."""

    def test_a_signal_at_the_largest_radius_raises_the_flag(self):
        Xr, yr = _planted(radius=len(RADII) - 1, strength=1.2)
        res = bi.sweep(
            Xr, RADII, STATS, yr,
            channels=("a", "b"), channel_index=(0, 1), splits=6, workers=1,
        )
        self.assertEqual(res.picked[1][0], 800)
        self.assertIn("b", res.boundary_hit)

    def test_an_interior_signal_does_not(self):
        Xr, yr = _planted(radius=2)
        res = bi.sweep(
            Xr, RADII, STATS, yr,
            channels=("a", "b"), channel_index=(0, 1), splits=6, workers=1,
        )
        self.assertNotIn("b", res.boundary_hit)


class TestNullCalibration(unittest.TestCase):
    """On a permuted outcome the posterior interval must mostly cover zero."""

    def test_a_shuffled_outcome_rarely_produces_an_interval_excluding_zero(self):
        Xr, yr = _planted(n=600)
        res = bi.sweep(
            Xr, RADII, STATS, yr,
            channels=("a", "b"), channel_index=(0, 1), splits=4, workers=1,
        )
        E = Xr.reshape(len(Xr), -1)[:, list(res.columns)]
        out = bi.null_calibration(E, yr, form=res.form, n=8, workers=1)
        self.assertEqual(out["runs"], 8)
        # 8 runs is too few to pin 5 %, but a procedure firing on most of them
        # is broken, not merely noisy.
        self.assertLessEqual(out["rate"], 0.5)


def _probe(task):
    """Report the worker's identity and its BLAS thread limit."""
    return task, os.getpid(), os.environ.get("OMP_NUM_THREADS")


class TestPooledMapIsPinned(unittest.TestCase):
    """The pool's workers must inherit the thread limits as they are created.

    Unpinned, numpy and OpenBLAS reserve ~3.0 GB of commit per worker against a
    budget that assumes ~0.11 GB, and a pool sized on that budget overdraws the
    host until unrelated allocations fail. Nothing about the returned numbers
    shows this, so it needs its own gate.
    """

    def test_every_worker_runs_with_one_blas_thread(self):
        got = bi._map(_probe, list(range(8)), 3)
        self.assertEqual({omp for _, _, omp in got}, {"1"})
        self.assertGreater(len({pid for _, pid, _ in got}), 1)

    def test_results_come_back_in_task_order(self):
        got = bi._map(_probe, list(range(8)), 3)
        self.assertEqual([task for task, _, _ in got], list(range(8)))

    def test_the_parent_environment_is_left_alone(self):
        before = os.environ.get("OMP_NUM_THREADS")
        bi._map(_probe, list(range(4)), 2)
        self.assertEqual(os.environ.get("OMP_NUM_THREADS"), before)


class TestWeightsAreOnePerChannel(unittest.TestCase):
    """A fit's weight vector must be readable without knowing which fit it was.

    The simplex fit drops channels it cannot use, so the surviving set differs
    from fit to fit. Reporting only the survivors makes position *i* mean a
    different channel in each vector, and the discovery loop averages weights
    across dozens of them.
    """

    @staticmethod
    def _two_channels(sign):
        rng = np.random.default_rng(11)
        E = rng.normal(size=(400, 2))
        y = -1.2 * E[:, 0] + sign * 0.9 * E[:, 1] + 0.2 * rng.normal(size=400)
        return E, y

    def test_a_dropped_channel_is_reported_as_a_zero(self):
        E, y = self._two_channels(1.0)
        _, params = bi.build_index(E, y, "linear")
        w = np.asarray(params["weights"])
        self.assertEqual(len(w), E.shape[1])
        self.assertAlmostEqual(float(w.sum()), 1.0, places=9)
        self.assertLess(len(params["kept"]), E.shape[1])
        self.assertAlmostEqual(float(w[0]), 0.0, places=12)

    def test_the_index_still_matches_the_weights_it_reports(self):
        E, y = self._two_channels(1.0)
        apply_fn, params = bi.build_index(E, y, "linear")
        self.assertTrue(
            np.allclose(apply_fn(E), E @ np.asarray(params["weights"])))

    def test_fits_with_different_active_sets_stack(self):
        rows = [
            np.asarray(bi.build_index(*self._two_channels(s), "linear")[1]["weights"])
            for s in (1.0, -1.0, 1.0, -1.0)
        ]
        self.assertEqual(np.asarray(rows).shape, (4, 2))
        self.assertEqual(len(np.mean(rows, axis=0)), 2)

    def test_the_discovery_loop_averages_them(self):
        Xr, yr = _planted(n=500)
        out = bi.repeated_discovery(
            Xr, RADII, STATS, yr, channels=("a", "b"), channel_index=(0, 1),
            reps=2, shuffles=3, workers=1,
        )
        self.assertEqual(len(out["per_form"]["linear"]["weights_mean"]), 2)
        self.assertEqual(len(out["per_form"]["linear"]["weights_sd"]), 2)


class TestCancelStopsTheSearch(unittest.TestCase):
    """Every phase long enough to need a pool has to answer the cancel flag.

    Read only between phases, the flag leaves a stopped job running for as long
    as the phase it landed in, which for a full grid is the whole search.
    """

    def test_the_sweep_gives_up_when_the_flag_is_set(self):
        Xr, yr = _planted(n=400)
        with self.assertRaises(JobCancelled):
            bi.sweep(
                Xr, RADII, STATS, yr, channels=("a", "b"), channel_index=(0, 1),
                splits=50, workers=1, cancel_check=lambda: True,
            )

    def test_discovery_and_gain_give_up_too(self):
        Xr, yr = _planted(n=400)
        with self.assertRaises(JobCancelled):
            bi.repeated_discovery(
                Xr, RADII, STATS, yr, channels=("a", "b"), channel_index=(0, 1),
                reps=2, shuffles=4, workers=1, cancel_check=lambda: True,
            )
        with self.assertRaises(JobCancelled):
            bi.holdout_gain(
                Xr, yr, channels=("a", "b"), channel_index=(0, 1),
                splits=4, perm=4, workers=1, cancel_check=lambda: True,
            )

    def test_an_unset_flag_changes_nothing(self):
        Xr, yr = _planted(n=400)
        res = bi.sweep(
            Xr, RADII, STATS, yr, channels=("a", "b"), channel_index=(0, 1),
            splits=6, workers=1, cancel_check=lambda: False,
        )
        self.assertEqual(res.picked[1], (int(RADII[2]), STATS[1]))


class TestSimplexFit(unittest.TestCase):
    def test_weights_stay_on_the_simplex_for_a_correlated_design(self):
        rng = np.random.default_rng(3)
        base = rng.normal(size=(400, 1))
        Z = np.hstack([base + 0.1 * rng.normal(size=(400, 1)) for _ in range(3)])
        y = (Z @ np.array([0.5, 0.3, 0.2])) + rng.normal(size=400) * 0.1
        kept, w = bi.simplex_fit(Z.T @ Z, Z.T @ y)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=9)
        self.assertTrue(np.all(w >= -1e-12))
        self.assertEqual(len(kept), len(w))


if __name__ == "__main__":
    unittest.main()
