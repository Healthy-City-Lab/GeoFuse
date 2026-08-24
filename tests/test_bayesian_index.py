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
        # Channel 0 is the stronger association (-1.2 against +0.9) and it is
        # protective, so a direction-free fit keeps it and drops channel 1.
        E, y = self._two_channels(1.0)
        _, params = bi.build_index(E, y, "linear")
        w = np.asarray(params["weights"])
        self.assertEqual(len(w), E.shape[1])
        self.assertAlmostEqual(float(w.sum()), 1.0, places=9)
        self.assertLess(len(params["kept"]), E.shape[1])
        self.assertAlmostEqual(float(w[1]), 0.0, places=12)

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


class TestDirectionIsNotAssumed(unittest.TestCase):
    """A protective exposure must be found as readily as a harmful one.

    The toolbox is pointed at outcomes that rise with greenery and at outcomes
    that fall with it. A fit that maximises the *signed* association walks away
    from the protective channel and hands the whole weight to whichever channel
    happened to correlate upward, which is usually the one carrying no signal.
    Nothing downstream reveals this: the sweep scores |t|, so the wrong weights
    still come back with a plausible-looking held-out number.
    """

    @staticmethod
    def _one_signal(sign, seed=1):
        rng = np.random.default_rng(seed)
        n = 3000
        signal = rng.normal(size=n)
        noise = rng.normal(size=n)
        y = sign * 0.5 * signal + 0.02 * noise + 0.5 * rng.normal(size=n)
        Z = np.column_stack([signal, noise])
        Z = (Z - Z.mean(0)) / Z.std(0)
        return Z, (y - y.mean()) / y.std()

    def test_the_protective_channel_carries_the_weight(self):
        Z, y = self._one_signal(-1.0)
        kept, w = bi.simplex_fit(Z.T @ Z, Z.T @ y)
        self.assertIn(0, kept)
        self.assertGreater(float(dict(zip(kept, w)).get(0, 0.0)), 0.9)

    def test_flipping_the_outcome_gives_the_same_weights_and_score(self):
        # The exact symmetry: one dataset, the outcome negated. Anything the
        # fit does differently between these two is a direction assumption.
        Z, y = self._one_signal(-1.0)
        out = []
        for target in (y, -y):
            kept, w = bi.simplex_fit(Z.T @ Z, Z.T @ target)
            out.append((kept, w, bi._tstat(Z[:, kept] @ w, target)))
        self.assertEqual(list(out[0][0]), list(out[1][0]))
        self.assertTrue(np.allclose(out[0][1], out[1][1]))
        self.assertAlmostEqual(out[0][2], out[1][2], places=9)

    def test_the_sweep_finds_a_protective_planted_column(self):
        rng = np.random.default_rng(4)
        n = 1500
        X = rng.normal(size=(n, 2, len(RADII), len(STATS)))
        y = -0.6 * X[:, 1, 2, 1] + rng.normal(size=n)
        yr, Xr = bi.prep(X, y, None)
        res = bi.sweep(
            Xr, RADII, STATS, yr,
            channels=("a", "b"), channel_index=(0, 1), splits=6, workers=1,
        )
        self.assertEqual(res.picked[1], (600, "p50"))


def _humped(n=2000, peak=400.0, width=0.6, stat="p10", sign=-1.0, seed=7):
    """A grid whose fidelity peaks at an interior radius, not at the smallest.

    An exponential decay kernel cannot express this shape at all, so it is the
    case that separates a peaked radius kernel from a monotone one.
    """
    rng = np.random.default_rng(seed)
    radii = np.array([50.0, 100.0, 200.0, 400.0, 800.0, 1600.0])
    stats = ["mean", "p10", "p50", "p90"]
    fidelity = np.exp(-0.5 * ((np.log(radii) - np.log(peak)) / width) ** 2)
    latent = rng.normal(size=(n, 2))
    X = np.empty((n, 2, len(radii), len(stats)))
    for c in range(2):
        for r in range(len(radii)):
            for s in range(len(stats)):
                f = fidelity[r] * (1.0 if stats[s] == stat else 0.35)
                X[:, c, r, s] = (f * latent[:, c]
                                 + np.sqrt(1 - f ** 2) * rng.normal(size=n))
    # Only channel 0 drives the outcome; channel 1 is there to be rejected.
    y = sign * 0.35 * latent[:, 0] + rng.normal(size=n)
    yr, Xr = bi.prep(X, y, None)
    return Xr, yr, radii, stats


class TestTheGridIsFittedNotPicked(unittest.TestCase):
    """Radius and aggregator are model parameters, and must behave like ones.

    The sweep picks a cell; the posterior estimates a blend over the whole
    grid. The point of the second is that its interval carries the uncertainty
    the first hides, so these gates check both that the blend lands in the
    right place and that it says so when it cannot.
    """

    @classmethod
    def setUpClass(cls):
        Xr, yr, radii, stats = _humped()
        mcmc = bi.fit(Xr, yr, form="linear", radii=radii, stats=stats,
                      radius_kernel="lognormal", aggregator="dirichlet",
                      draws=400, warmup=400, chains=2, seed=1)
        cls.post = bi.posterior_from(
            mcmc, channels=("a", "b"), picked=(), form="linear",
            radii=radii, stats=stats, radius_kernel="lognormal")
        cls.out = cls.post.summary()

    def test_the_peak_lands_at_the_interior_radius_the_data_was_built_around(self):
        peak = self.out["peak_radius_mean"][0]
        self.assertGreater(peak, 250.0)
        self.assertLess(peak, 650.0)

    def test_the_radius_profile_rises_and_then_falls(self):
        prof = np.asarray(self.out["radius_profile"][0])
        top = int(np.argmax(prof))
        self.assertGreater(top, 0)
        self.assertLess(top, len(prof) - 1)

    def test_the_aggregator_blend_concentrates_on_the_informative_statistic(self):
        blend = dict(zip(self.out["stats"], self.out["aggregator_mean"][0]))
        self.assertEqual(max(blend, key=blend.get), "p10")

    def test_a_channel_with_no_signal_reports_an_unidentified_scale(self):
        # The tell is the prior comparison, not the point estimate: an
        # unidentified kernel still returns a peak radius, and it looks like a
        # finding until it is put next to the prior it came from.
        ratios = self.out["peak_radius_width_ratio"]
        self.assertLess(ratios[0], 0.5)
        self.assertGreater(ratios[1], 0.5)
        # The gap is the usable signal: the driving channel's scale interval is
        # a fraction of the prior's, the passenger's is most of it.
        self.assertGreater(ratios[1], 2.0 * ratios[0])

    def test_the_effect_keeps_its_protective_sign(self):
        self.assertLess(self.out["beta_mean"], 0.0)
        self.assertLess(self.out["beta_ci_high"], 0.0)

    def test_the_weight_goes_to_the_channel_driving_the_outcome(self):
        self.assertGreater(self.out["weight_mean"][0],
                           self.out["weight_mean"][1])

    def test_the_projection_names_a_cell_the_composite_can_carry(self):
        radius, stat = self.post.projected_pick()[0]
        self.assertIn(radius, [int(r) for r in self.out["radii"]])
        self.assertIn(stat, self.out["stats"])

    def test_partial_r2_is_reported_as_a_share(self):
        self.assertGreater(self.out["partial_r2_mean"], 0.0)
        self.assertLess(self.out["partial_r2_mean"], 1.0)


class TestOffLadderCellsAreExcluded(unittest.TestCase):
    """Channels search different ladders, and the masked rungs are NaN.

    A masked rung must take no posterior weight at all: it is not a rung the
    channel was ever measured at, so mass there would be the model averaging in
    a cell that does not exist.
    """

    def test_a_masked_rung_takes_no_weight(self):
        Xr, yr, radii, stats = _humped(n=800)
        mask = np.ones((2, len(radii)), dtype=bool)
        mask[1, :2] = False
        Xr = Xr.copy()
        Xr[:, 1, :2, :] = np.nan
        mcmc = bi.fit(Xr, yr, form="linear", radii=radii, stats=stats,
                      radius_mask=mask, draws=200, warmup=200, chains=2, seed=2)
        post = bi.posterior_from(
            mcmc, channels=("a", "b"), picked=(), form="linear",
            radii=radii, stats=stats, radius_kernel="lognormal")
        prof = np.asarray(post.summary()["radius_profile"])
        self.assertTrue(np.allclose(prof[1, :2], 0.0))
        self.assertAlmostEqual(float(prof[1].sum()), 1.0, places=5)


class TestSingleChannelStudy(unittest.TestCase):
    """A standalone study fits one channel, and its weight vector is constant.

    ``Dirichlet`` over one component always returns ``[1.0]``, so that
    parameter has no between-chain spread and its R-hat is 0/0. A NaN there
    propagates through the reported maximum and the sampler-health gate stops
    firing -- silently, because a missing diagnostic renders the same as a
    healthy one.
    """

    @classmethod
    def setUpClass(cls):
        Xr, yr, radii, stats = _humped(n=800)
        mcmc = bi.fit(Xr[:, :1], yr, form="linear", radii=radii, stats=stats,
                      draws=250, warmup=250, chains=2, seed=5)
        cls.out = bi.posterior_from(
            mcmc, channels=("ndvi",), picked=(), form="linear",
            radii=radii, stats=stats, radius_kernel="lognormal").summary()

    def test_the_convergence_diagnostics_are_reported_as_numbers(self):
        self.assertTrue(np.isfinite(self.out["rhat_max"]))
        self.assertTrue(np.isfinite(self.out["ess_min"]))
        self.assertGreater(self.out["ess_min"], 0.0)

    def test_the_only_channel_holds_the_whole_weight(self):
        self.assertAlmostEqual(float(self.out["weight_mean"][0]), 1.0, places=6)

    def test_the_scale_and_aggregator_are_still_estimated(self):
        self.assertGreater(self.out["peak_radius_mean"][0], 150.0)
        self.assertLess(self.out["peak_radius_mean"][0], 900.0)
        blend = dict(zip(self.out["stats"], self.out["aggregator_mean"][0]))
        self.assertEqual(max(blend, key=blend.get), "p10")


class TestCollinearAggregatorsDoNotManufactureAPick(unittest.TestCase):
    """Percentiles of one buffer are six views of a single distribution.

    They separate only where that distribution's *shape* varies between people
    independently of its level. Where it does not, every aggregator is the same
    column shifted by a constant, the blend is flat, and the argmax of a flat
    vector is a coin flip -- one that would otherwise be shipped to the
    composite as though the data had chosen it.
    """

    @staticmethod
    def _flat_blend(k=6):
        rng = np.random.default_rng(0)
        draws = rng.dirichlet(np.ones(k), size=4000)
        return draws

    def _posterior(self, aggregator_draws, stats):
        n = len(aggregator_draws)
        return bi.IndexPosterior(
            weights=np.full((n, 1), 1.0), beta=np.zeros(n), powers=None,
            channels=("a",), picked=(), form="linear",
            rhat_max=1.0, ess_min=100.0, divergences=0,
            radius_weights=np.tile(
                np.array([[0.1, 0.8, 0.1]]), (n, 1, 1)),
            aggregator_weights=aggregator_draws[:, None, :],
            radii=(100, 500, 1000), stats=tuple(stats),
        )

    def test_a_flat_blend_reports_nothing_as_informative(self):
        stats = ["mean", "p10", "p25", "p50", "p75", "p90"]
        post = self._posterior(self._flat_blend(), stats)
        self.assertEqual(post.informative_aggregators(), [[]])

    def test_a_flat_blend_falls_back_to_the_mean(self):
        stats = ["mean", "p10", "p25", "p50", "p75", "p90"]
        post = self._posterior(self._flat_blend(), stats)
        self.assertEqual(post.projected_pick(), ((500, "mean"),))

    def test_a_concentrated_blend_keeps_its_own_statistic(self):
        stats = ["mean", "p10", "p25", "p50", "p75", "p90"]
        rng = np.random.default_rng(1)
        conc = rng.dirichlet(np.array([1.0, 60.0, 1.0, 1.0, 1.0, 1.0]), size=4000)
        post = self._posterior(conc, stats)
        self.assertIn("p10", post.informative_aggregators()[0])
        self.assertEqual(post.projected_pick(), ((500, "p10"),))

    def test_a_component_ruled_out_from_below_also_counts(self):
        # "Certainly not p90" is a finding, not an absence of one.
        stats = ["mean", "p10", "p25", "p50", "p75", "p90"]
        rng = np.random.default_rng(2)
        conc = rng.dirichlet(
            np.array([20.0, 20.0, 20.0, 20.0, 20.0, 0.05]), size=4000)
        post = self._posterior(conc, stats)
        self.assertIn("p90", post.informative_aggregators()[0])


class TestPriorIntervals(unittest.TestCase):
    """Every reported interval needs the prior it was drawn from beside it."""

    def test_a_symmetric_dirichlet_marginal_matches_its_beta(self):
        from scipy.stats import beta as _beta
        for k in (2, 4, 6):
            lo, hi = bi._dirichlet_prior_ci(k)
            self.assertAlmostEqual(lo, float(_beta.ppf(0.025, 1, k - 1)), places=9)
            self.assertAlmostEqual(hi, float(_beta.ppf(0.975, 1, k - 1)), places=9)

    def test_an_interval_as_wide_as_its_prior_ratios_to_one(self):
        self.assertAlmostEqual(bi._width_ratio(0.0, 1.0, 0.0, 1.0), 1.0)
        self.assertAlmostEqual(bi._width_ratio(0.2, 0.4, 0.0, 1.0), 0.2)

    def test_the_radius_prior_follows_the_ladder_it_was_given(self):
        near, _ = bi._radius_prior([50.0, 100.0, 200.0])
        far, _ = bi._radius_prior([500.0, 1000.0, 2000.0])
        self.assertLess(near, far)


if __name__ == "__main__":
    unittest.main()
