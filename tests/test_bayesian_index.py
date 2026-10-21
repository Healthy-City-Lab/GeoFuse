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
            Xr,
            RADII,
            STATS,
            yr,
            channels=("a", "b"),
            channel_index=(0, 1),
            splits=6,
            workers=1,
        )
        self.assertEqual(res.picked[1], (600, "p50"))

    def test_the_weight_lands_on_the_channel_carrying_the_signal(self):
        Xr, yr = _planted()
        res = bi.sweep(
            Xr,
            RADII,
            STATS,
            yr,
            channels=("a", "b"),
            channel_index=(0, 1),
            splits=6,
            workers=1,
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
            Xr,
            RADII,
            STATS,
            yr,
            channels=("a", "b"),
            channel_index=(0, 1),
            splits=6,
            workers=1,
        )
        self.assertEqual(len(res.one_se_columns), len(res.columns))
        n_radii, n_stats = Xr.shape[2], Xr.shape[3]
        picked_ch = [c // (n_radii * n_stats) for c in res.columns]
        one_se_ch = [c // (n_radii * n_stats) for c in res.one_se_columns]
        self.assertEqual(picked_ch, one_se_ch)

    def test_the_one_se_pick_is_no_larger_than_the_winner(self):
        Xr, yr = _planted()
        res = bi.sweep(
            Xr,
            RADII,
            STATS,
            yr,
            channels=("a", "b"),
            channel_index=(0, 1),
            splits=6,
            workers=1,
        )
        n_radii, n_stats = Xr.shape[2], Xr.shape[3]
        one_se = bi._decode(res.one_se_columns, n_radii, n_stats, RADII, STATS)
        self.assertLessEqual(sum(r for r, _ in one_se), sum(r for r, _ in res.picked))


class TestBoundaryIsFlagged(unittest.TestCase):
    """A radius at the edge of the range means the optimum may be outside it."""

    def test_a_signal_at_the_largest_radius_raises_the_flag(self):
        Xr, yr = _planted(radius=len(RADII) - 1, strength=1.2)
        res = bi.sweep(
            Xr,
            RADII,
            STATS,
            yr,
            channels=("a", "b"),
            channel_index=(0, 1),
            splits=6,
            workers=1,
        )
        self.assertEqual(res.picked[1][0], 800)
        self.assertIn("b", res.boundary_hit)

    def test_an_interior_signal_does_not(self):
        Xr, yr = _planted(radius=2)
        res = bi.sweep(
            Xr,
            RADII,
            STATS,
            yr,
            channels=("a", "b"),
            channel_index=(0, 1),
            splits=6,
            workers=1,
        )
        self.assertNotIn("b", res.boundary_hit)


class TestNullCalibration(unittest.TestCase):
    """On a permuted outcome the posterior interval must mostly cover zero."""

    def test_a_shuffled_outcome_rarely_produces_an_interval_excluding_zero(self):
        Xr, yr = _planted(n=600)
        res = bi.sweep(
            Xr,
            RADII,
            STATS,
            yr,
            channels=("a", "b"),
            channel_index=(0, 1),
            splits=4,
            workers=1,
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
        _, params = bi.build_index(E, y)
        w = np.asarray(params["weights"])
        self.assertEqual(len(w), E.shape[1])
        self.assertAlmostEqual(float(w.sum()), 1.0, places=9)
        self.assertLess(len(params["kept"]), E.shape[1])
        self.assertAlmostEqual(float(w[1]), 0.0, places=12)

    def test_the_index_still_matches_the_weights_it_reports(self):
        E, y = self._two_channels(1.0)
        apply_fn, params = bi.build_index(E, y)
        self.assertTrue(np.allclose(apply_fn(E), E @ np.asarray(params["weights"])))

    def test_fits_with_different_active_sets_stack(self):
        rows = [
            np.asarray(bi.build_index(*self._two_channels(s))[1]["weights"])
            for s in (1.0, -1.0, 1.0, -1.0)
        ]
        self.assertEqual(np.asarray(rows).shape, (4, 2))
        self.assertEqual(len(np.mean(rows, axis=0)), 2)

    def test_the_discovery_loop_averages_them(self):
        Xr, yr = _planted(n=500)
        out = bi.repeated_discovery(
            Xr,
            RADII,
            STATS,
            yr,
            channels=("a", "b"),
            channel_index=(0, 1),
            reps=2,
            shuffles=3,
            workers=1,
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
                Xr,
                RADII,
                STATS,
                yr,
                channels=("a", "b"),
                channel_index=(0, 1),
                splits=50,
                workers=1,
                cancel_check=lambda: True,
            )

    def test_discovery_and_gain_give_up_too(self):
        Xr, yr = _planted(n=400)
        with self.assertRaises(JobCancelled):
            bi.repeated_discovery(
                Xr,
                RADII,
                STATS,
                yr,
                channels=("a", "b"),
                channel_index=(0, 1),
                reps=2,
                shuffles=4,
                workers=1,
                cancel_check=lambda: True,
            )
        with self.assertRaises(JobCancelled):
            bi.holdout_gain(
                Xr,
                yr,
                channels=("a", "b"),
                channel_index=(0, 1),
                splits=4,
                workers=1,
                cancel_check=lambda: True,
            )

    def test_an_unset_flag_changes_nothing(self):
        Xr, yr = _planted(n=400)
        res = bi.sweep(
            Xr,
            RADII,
            STATS,
            yr,
            channels=("a", "b"),
            channel_index=(0, 1),
            splits=6,
            workers=1,
            cancel_check=lambda: False,
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
            Xr,
            RADII,
            STATS,
            yr,
            channels=("a", "b"),
            channel_index=(0, 1),
            splits=6,
            workers=1,
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
                X[:, c, r, s] = f * latent[:, c] + np.sqrt(1 - f**2) * rng.normal(
                    size=n
                )
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
        mcmc = bi.fit(
            Xr,
            yr,
            form="linear",
            radii=radii,
            stats=stats,
            radius_kernel="lognormal",
            aggregator="dirichlet",
            draws=400,
            warmup=400,
            chains=2,
            seed=1,
        )
        cls.post = bi.posterior_from(
            mcmc,
            channels=("a", "b"),
            picked=(),
            form="linear",
            radii=radii,
            stats=stats,
            radius_kernel="lognormal",
        )
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
        self.assertGreater(self.out["weight_mean"][0], self.out["weight_mean"][1])

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
        mcmc = bi.fit(
            Xr,
            yr,
            form="linear",
            radii=radii,
            stats=stats,
            radius_mask=mask,
            draws=200,
            warmup=200,
            chains=2,
            seed=2,
        )
        post = bi.posterior_from(
            mcmc,
            channels=("a", "b"),
            picked=(),
            form="linear",
            radii=radii,
            stats=stats,
            radius_kernel="lognormal",
        )
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
        mcmc = bi.fit(
            Xr[:, :1],
            yr,
            form="linear",
            radii=radii,
            stats=stats,
            draws=250,
            warmup=250,
            chains=2,
            seed=5,
        )
        cls.out = bi.posterior_from(
            mcmc,
            channels=("ndvi",),
            picked=(),
            form="linear",
            radii=radii,
            stats=stats,
            radius_kernel="lognormal",
        ).summary()

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
            weights=np.full((n, 1), 1.0),
            beta=np.zeros(n),
            channels=("a",),
            picked=(),
            form="linear",
            rhat_max=1.0,
            ess_min=100.0,
            divergences=0,
            radius_weights=np.tile(np.array([[0.1, 0.8, 0.1]]), (n, 1, 1)),
            aggregator_weights=aggregator_draws[:, None, :],
            radii=(100, 500, 1000),
            stats=tuple(stats),
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
        conc = rng.dirichlet(np.array([20.0, 20.0, 20.0, 20.0, 20.0, 0.05]), size=4000)
        post = self._posterior(conc, stats)
        self.assertIn("p90", post.informative_aggregators()[0])


class TestPartialCoverageDoesNotPoisonTheReport(unittest.TestCase):
    """Street-view coverage is missing for a few addresses at small radii.

    Two separate failures follow from that on real data, and neither shows up
    as an error -- the run completes and reports NaN where a number belongs.
    Standardising a column that holds one NaN yields a NaN column, so a handful
    of uncovered entities cost every entity that candidate; and an index built
    from the surviving channels still multiplied the dropped column by zero,
    which is NaN rather than nothing.
    """

    @staticmethod
    def _cube(n=600, gap_rows=5):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(n, 2, len(RADII), len(STATS)))
        y = 0.6 * X[:, 1, 2, 1] + rng.normal(size=n)
        # A few entities have no coverage for channel 0 at the smallest radius,
        # exactly as an address with no panorama inside 200 m does.
        X[:gap_rows, 0, 0, :] = np.nan
        return X, y

    def test_a_few_uncovered_entities_do_not_kill_a_column(self):
        X, y = self._cube()
        yr, Xr = bi.prep(X, y, None)
        flat = Xr.reshape(len(Xr), -1)
        self.assertTrue(np.isfinite(flat).all())
        self.assertEqual(len(Xr), len(X) - 5)
        self.assertEqual(len(yr), len(Xr))

    def test_a_structurally_absent_cell_drops_the_column_not_the_rows(self):
        # A channel simply not measured at a radius is NaN for everyone. That
        # is a column to leave out, not a reason to drop the whole cohort.
        X, y = self._cube(gap_rows=0)
        X[:, 0, 3, :] = np.nan
        yr, Xr = bi.prep(X, y, None)
        self.assertEqual(len(Xr), len(X))
        self.assertTrue(np.isnan(Xr[:, 0, 3, :]).all())
        keep = np.ones(Xr.shape, dtype=bool)
        keep[:, 0, 3, :] = False
        self.assertTrue(np.isfinite(Xr[keep]).all())

    def test_the_scan_ignores_channels_the_study_does_not_use(self):
        X, y = self._cube(gap_rows=0)
        X[:7, 0, 1, :] = np.nan
        _, Xr = bi.prep(X, y, None, channel_index=[1])
        self.assertEqual(len(Xr), len(X))

    def test_an_index_ignores_a_channel_it_dropped(self):
        # The weight is zero, so the column must not reach the arithmetic at
        # all -- ``0 * nan`` is what turned every reported t into NaN.
        rng = np.random.default_rng(6)
        E = rng.normal(size=(300, 2))
        y = 1.5 * E[:, 0] + 0.05 * rng.normal(size=300)
        apply_fn, params = bi.build_index(E, y)
        self.assertAlmostEqual(float(np.asarray(params["weights"])[1]), 0.0, places=9)
        E_gap = E.copy()
        E_gap[:4, 1] = np.nan
        self.assertTrue(np.isfinite(apply_fn(E_gap)).all())
        self.assertTrue(np.isfinite(bi._tstat(apply_fn(E_gap), y)))

    def test_the_discovery_loop_reports_numbers_not_nan(self):
        X, y = self._cube()
        yr, Xr = bi.prep(X, y, None)
        out = bi.repeated_discovery(
            Xr,
            RADII,
            STATS,
            yr,
            channels=("a", "b"),
            channel_index=(0, 1),
            reps=2,
            shuffles=3,
            workers=1,
        )
        lin = out["per_form"]["linear"]
        for key in ("train_t", "test_t", "shrinkage"):
            self.assertTrue(np.isfinite(lin[key]), f"{key} is {lin[key]}")

    def test_an_undefined_gain_does_not_become_a_significant_p_value(self):
        # Splits whose gain is undefined carry no statistic, so a test built
        # from them must report nothing rather than a p-value.
        import geofuse.bayesian_index as _bi

        real_map = _bi._map

        def fake_map(fn, tasks, n_workers, cancel_check=None):
            out = real_map(fn, tasks, 1, cancel_check)
            for r in out:
                if isinstance(r, dict):
                    r["gain"] = float("nan")
            return out

        X, y = self._cube()
        yr, Xr = bi.prep(X, y, None)
        _bi._map = fake_map
        try:
            gain = bi.holdout_gain(
                Xr,
                yr,
                channels=("a", "b"),
                channel_index=(0, 1),
                splits=3,
                workers=1,
            )
        finally:
            _bi._map = real_map
        self.assertIsNone(gain["gain_p"])

    def test_the_gain_p_value_is_the_corrected_resampled_t(self):
        from scipy import stats

        import geofuse.bayesian_index as _bi

        real_map = _bi._map
        planted = iter([1.0, 2.0, 3.0])

        def fake_map(fn, tasks, n_workers, cancel_check=None):
            out = real_map(fn, tasks, 1, cancel_check)
            for r in out:
                if isinstance(r, dict) and "gain" in r:
                    r["gain"] = next(planted)
            return out

        X, y = self._cube()
        yr, Xr = bi.prep(X, y, None)
        _bi._map = fake_map
        try:
            gain = bi.holdout_gain(
                Xr, yr, channels=("a", "b"), channel_index=(0, 1), splits=3, workers=1
            )
        finally:
            _bi._map = real_map
        # mean 2, variance 1, inflation 1/3 + 0.25/0.75.
        t = 2.0 / np.sqrt(2.0 / 3.0)
        self.assertAlmostEqual(gain["gain_t"], t)
        self.assertAlmostEqual(gain["gain_p"], float(stats.t.sf(t, 2)))

    def test_the_gain_comparison_reports_a_composite_score(self):
        X, y = self._cube()
        yr, Xr = bi.prep(X, y, None)
        gain = bi.holdout_gain(
            Xr,
            yr,
            channels=("a", "b"),
            channel_index=(0, 1),
            splits=3,
            workers=1,
        )
        for key in ("cgi", "best_single", "gain"):
            self.assertTrue(np.isfinite(gain[key]), f"{key} is {gain[key]}")


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


class TestRankSafeResidualisation(unittest.TestCase):
    """A degenerate covariate column must remove nothing but its own span.

    Reduced QR hands back one orthonormal column per input column whatever the
    rank, so an all-zero dummy or a full one-hot set beside an intercept adds
    arbitrary directions to the projection and strips real variation from the
    outcome and from every exposure column.
    """

    @staticmethod
    def _data(n=400, seed=11):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 2, len(RADII), len(STATS)))
        y = 0.4 * X[:, 0, 1, 0] + rng.normal(size=n)
        group = rng.integers(0, 3, size=n)
        onehot = np.eye(3)[group]
        age = rng.normal(size=n)
        return X, y, onehot, age

    @staticmethod
    def _old_qr_prep(X, y, cov):
        q, _ = np.linalg.qr(cov)
        yr = y - q @ (q.T @ y)
        yr = (yr - yr.mean()) / (yr.std() + 1e-12)
        flat = X.reshape(len(X), -1)
        flat = flat - q @ (q.T @ flat)
        flat = (flat - flat.mean(0)) / (flat.std(0) + 1e-12)
        return yr, flat.reshape(X.shape)

    def _assert_same(self, a, b):
        for u, v in zip(a, b):
            np.testing.assert_allclose(u, v, atol=1e-10, rtol=0)

    def test_an_all_zero_column_changes_nothing(self):
        X, y, onehot, age = self._data()
        cov = np.column_stack([age, onehot[:, 1:]])
        with_zero = np.column_stack([cov, np.zeros(len(y))])
        self._assert_same(bi.prep(X, y, with_zero), bi.prep(X, y, cov))

    def test_a_full_dummy_set_with_an_intercept_equals_drop_first(self):
        X, y, onehot, age = self._data()
        full = np.column_stack([np.ones(len(y)), age, onehot])
        drop_first = np.column_stack([age, onehot[:, 1:]])
        self._assert_same(bi.prep(X, y, full), bi.prep(X, y, drop_first))

    def test_a_full_rank_design_with_an_intercept_matches_the_qr_path(self):
        X, y, onehot, age = self._data()
        cov = np.column_stack([np.ones(len(y)), age, onehot[:, 1:]])
        self._assert_same(bi.prep(X, y, cov), self._old_qr_prep(X, y, cov))

    def test_the_intercept_is_always_projected_out(self):
        # Without an intercept in the basis the covariate projection is not the
        # Frisch-Waugh partial; centring afterwards does not make it one.
        X, y, onehot, age = self._data()
        drop_first = np.column_stack([age + 5.0, onehot[:, 1:]])
        with_one = np.column_stack([np.ones(len(y)), drop_first])
        self._assert_same(bi.prep(X, y, drop_first), bi.prep(X, y, with_one))

    def test_the_deficit_is_reported_and_logged(self):
        X, y, onehot, age = self._data()
        full = np.column_stack([np.ones(len(y)), age, onehot])
        with self.assertLogs("geofuse.bayesian_index", level="WARNING") as cm:
            *_, info = bi.prep(X, y, full, return_info=True)
        self.assertIn("1 redundant direction", cm.output[0])
        self.assertEqual(info["dropped_directions"], 1)
        self.assertEqual(info["covariate_rank"], 4)
        self.assertEqual(info["residual_df"], len(y) - 4)

    def test_a_clean_design_reports_no_deficit(self):
        X, y, onehot, age = self._data()
        *_, info = bi.prep(
            X, y, np.column_stack([age, onehot[:, 1:]]), return_info=True
        )
        self.assertEqual(info["dropped_directions"], 0)
        self.assertEqual(info["covariate_rank"], 4)

    def test_col_basis_keeps_only_the_real_span(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(50, 2))
        Z = np.column_stack([a, a[:, 0] + a[:, 1], np.zeros(50)])
        q = bi.col_basis(Z)
        self.assertEqual(q.shape, (50, 2))
        np.testing.assert_allclose(q.T @ q, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(q @ (q.T @ Z), Z, atol=1e-10)


class TestHonestFormSelection(unittest.TestCase):
    """The form must be chosen without looking at the rows that score it.

    Picking the better form on the held-out rows is a winner's curse over
    forms; comparing synergy against the linear grid's own maximum stacks the
    comparison in linear's favour. Both are what these tests rule out.
    """

    @staticmethod
    def _cube(truth, n=1500, seed=0):
        """Linear truth on the channels; synergy truth inside its family.

        The synergy form's interaction is the geometric mean of the raw
        channels, so its truth is one: mostly the interaction of two positive
        channels. Returns ``(Xr, yr, design)``.
        """
        rng = np.random.default_rng(seed)
        X = rng.lognormal(-1.0, 0.6, size=(n, 2, 1, 1))
        a, b = X[:, 0, 0, 0], X[:, 1, 0, 0]
        if truth == "synergy":
            t = 0.1 * a / a.std() + 0.1 * b / b.std() + 0.8 * np.sqrt(a * b)
            y = 0.8 * (t - t.mean()) / t.std() + rng.normal(size=n)
        else:
            y = 0.25 * a / a.std() + 0.25 * b / b.std() + rng.normal(size=n)
        yr, Xr = bi.prep(X, y, None)
        return Xr, yr, bi.SynergyDesign.build(X)

    def _sweep(self, Xr, yr, design, **kw):
        return bi.sweep(
            Xr,
            np.array([500.0]),
            ["mean"],
            yr,
            channels=("a", "b"),
            channel_index=(0, 1),
            splits=8,
            workers=1,
            design=design,
            **kw,
        )

    def test_a_null_outcome_shows_no_systematic_gain(self):
        rng = np.random.default_rng(21)
        X = rng.lognormal(size=(800, 2, 2, 2))
        yr, Xr = bi.prep(X, rng.normal(size=800), None)
        gain = bi.holdout_gain(
            Xr,
            yr,
            channels=("a", "b"),
            channel_index=(0, 1),
            splits=30,
            workers=1,
            design=bi.SynergyDesign.build(X),
        )
        self.assertLess(abs(gain["gain"]), 0.5)
        self.assertGreater(gain["gain_p"], 0.01)
        self.assertLessEqual(gain["cgi"], gain["cgi_max_over_forms_optimistic"])
        self.assertEqual(sum(gain["form_counts"].values()), 30)
        self.assertEqual(set(gain["cgi_by_form"]), {"linear", "synergy"})

    def test_linear_truth_does_not_pull_in_synergy(self):
        forms = [self._sweep(*self._cube("linear", seed=s)).form for s in range(6)]
        self.assertLessEqual(forms.count("synergy"), 2, forms)

    def test_synergy_truth_is_recognised(self):
        forms = [self._sweep(*self._cube("synergy", seed=s)).form for s in range(4)]
        self.assertEqual(forms.count("synergy"), 4, forms)

    def test_the_forms_are_compared_on_fresh_splits(self):
        res = self._sweep(*self._cube("linear"))
        self.assertEqual(set(res.form_scores), {"linear", "synergy"})
        self.assertEqual(res.score, res.form_scores[res.form])
        self.assertNotAlmostEqual(res.form_scores["linear"], res.selection_score)

    def test_only_requested_forms_compete(self):
        res = self._sweep(*self._cube("linear"), forms=("synergy",))
        self.assertEqual(res.form, "synergy")
        self.assertEqual(set(res.form_scores), {"synergy"})

    def test_a_single_channel_is_always_linear(self):
        self.assertEqual(bi.eligible_forms(("synergy",), 1), ("linear",))
        self.assertEqual(bi.eligible_forms(bi.FORMS, 2), bi.FORMS)

    def test_synergy_needs_the_raw_cell_values(self):
        self.assertEqual(bi.eligible_forms(bi.FORMS, 2, False), ("linear",))
        Xr, yr, _ = self._cube("synergy")
        self.assertEqual(self._sweep(Xr, yr, None).form, "linear")

    def test_the_main_terms_are_the_tensor_columns(self):
        Xr, _, design = self._cube("synergy")
        T, sd = design.columns((0, 1))
        # The design stores its cells as float32.
        np.testing.assert_allclose(T[:, :2], Xr.reshape(len(Xr), -1), atol=1e-5)
        self.assertEqual(T.shape[1], 3)
        self.assertTrue((sd > 0).all())

    def test_discovery_chooses_on_training_rows(self):
        Xr, yr, design = self._cube("synergy", n=1200)
        out = bi.repeated_discovery(
            Xr,
            np.array([500.0]),
            ["mean"],
            yr,
            channels=("a", "b"),
            channel_index=(0, 1),
            reps=1,
            shuffles=4,
            workers=1,
            design=design,
        )
        self.assertEqual(sum(out["form_counts"].values()), out["n_results"])
        self.assertGreaterEqual(out["form_counts"]["synergy"], 3)
        self.assertTrue(np.isfinite(out["chosen_test_t"]))

    def test_choose_form_breaks_ties_towards_the_first_form(self):
        Xr, yr, _ = self._cube("linear")
        E = Xr.reshape(len(Xr), -1)
        self.assertEqual(bi.choose_form({"linear": E}, yr, seed=0), ("linear", {}))
        form, _ = bi.choose_form({"linear": E, "synergy": E}, yr, seed=0)
        self.assertEqual(form, "linear")


class TestDistanceDecay(unittest.TestCase):
    """A blend of whole-disc means is one radial weight; report where it sits.

    For one rung r*, the weight is uniform over the disc, so half of it lies
    within r*/sqrt(2) and 90 % within sqrt(0.9) r* -- not at r*, which is what a
    kernel peak would suggest.
    """

    @staticmethod
    def _q(k, radii, **kw):
        k = np.asarray(k, dtype=float)[None, :]
        out = bi.distance_quantiles(k, radii, **kw)
        return float(out[0.5][0]), float(out[0.9][0])

    def test_all_mass_on_one_rung(self):
        r50, r90 = self._q([0.0, 1.0, 0.0], [100.0, 300.0, 900.0])
        self.assertAlmostEqual(r50, 300.0 / np.sqrt(2.0), places=9)
        self.assertAlmostEqual(r90, np.sqrt(0.9) * 300.0, places=9)

    def test_two_rungs_match_the_closed_form(self):
        r1, r2 = 100.0, 400.0
        # Median beyond the inner rung: k1 + k2 D²/r2² = q.
        r50, r90 = self._q([0.3, 0.7], [r1, r2])
        self.assertAlmostEqual(r50, r2 * np.sqrt(0.2 / 0.7), places=9)
        self.assertAlmostEqual(r90, r2 * np.sqrt(0.6 / 0.7), places=9)
        # Median inside the inner rung: D²(k1/r1² + k2/r2²) = q.
        r50, r90 = self._q([0.8, 0.2], [r1, r2])
        self.assertAlmostEqual(
            r50, np.sqrt(0.5 / (0.8 / r1**2 + 0.2 / r2**2)), places=9
        )
        self.assertAlmostEqual(r90, r2 * np.sqrt(0.1 / 0.2), places=9)

    def test_unequal_sd_moves_the_median_the_expected_way(self):
        k = np.array([[[0.5, 0.5]]])
        radii = [100.0, 400.0]
        even = bi.distance_quantiles(bi.raw_rung_weights(k, [[1.0, 1.0]]), radii)
        # A smaller SD on the inner rung means more raw weight per unit of its
        # kernel weight, so more of the influence sits close in.
        tight = bi.distance_quantiles(bi.raw_rung_weights(k, [[0.5, 1.0]]), radii)
        wide = bi.distance_quantiles(bi.raw_rung_weights(k, [[2.0, 1.0]]), radii)
        self.assertLess(tight[0.5][0, 0], even[0.5][0, 0])
        self.assertGreater(wide[0.5][0, 0], even[0.5][0, 0])

    def test_vectorised_over_draws_and_channels(self):
        rng = np.random.default_rng(2)
        radii = [50.0, 150.0, 400.0, 1000.0]
        k = rng.dirichlet(np.ones(4), size=(7, 3))
        both = bi.distance_quantiles(k, radii)
        for d in range(7):
            for c in range(3):
                r50, r90 = self._q(k[d, c], radii)
                self.assertAlmostEqual(both[0.5][d, c], r50, places=9)
                self.assertAlmostEqual(both[0.9][d, c], r90, places=9)
        self.assertEqual(both[0.5].shape, (7, 3))

    def test_a_masked_rung_with_no_sd_still_works(self):
        k = np.array([[[0.0, 1.0, 0.0]]])
        raw = bi.raw_rung_weights(k, [[np.nan, 2.0, np.nan]])
        np.testing.assert_allclose(raw, k)
        r50 = bi.distance_quantiles(raw, [100.0, 200.0, 400.0])[0.5]
        self.assertAlmostEqual(float(r50[0, 0]), 200.0 / np.sqrt(2.0), places=9)

    def test_a_point_rung_is_mass_at_the_entity(self):
        r50, r90 = self._q([0.6, 0.4], [0.0, 100.0])
        self.assertEqual(r50, 0.0)
        self.assertAlmostEqual(r90, np.sqrt(0.3 / (0.4 / 100.0**2)), places=9)

    def test_observed_areas_replace_the_uniform_density(self):
        radii = [100.0, 400.0]
        uniform = self._q([0.0, 1.0], radii)
        scaled = self._q([0.0, 1.0], radii, areas=[[5.0 * 100**2, 5.0 * 400**2]])
        np.testing.assert_allclose(uniform, scaled)
        # Half the points already inside 100 m: the median distance is 100 m.
        dense = self._q([0.0, 1.0], radii, areas=[[100.0, 200.0]])
        self.assertAlmostEqual(dense[0], 100.0, places=9)

    def test_the_implied_curve_never_rises_even_for_a_humped_kernel(self):
        k = np.array([[[0.1, 0.7, 0.2]]])
        grid, curve = bi.implied_weight_curve(k, [100.0, 300.0, 900.0])
        self.assertEqual(curve[0, 0, 0], 1.0)
        self.assertTrue(np.all(np.diff(curve[0, 0]) <= 1e-12))
        self.assertEqual(grid[-1], 900.0)

    def test_the_posterior_summary_reports_r50_and_labels_its_basis(self):
        rng = np.random.default_rng(5)
        draws, radii = 50, (100, 300, 900)
        post = bi.IndexPosterior(
            weights=rng.dirichlet(np.ones(2), size=draws),
            beta=rng.normal(size=draws),
            channels=("ndvi", "gvi"),
            picked=((300, "mean"),) * 2,
            form="linear",
            rhat_max=1.0,
            ess_min=400.0,
            divergences=0,
            radius_weights=np.broadcast_to([0.0, 1.0, 0.0], (draws, 2, 3)).copy(),
            aggregator_weights=rng.dirichlet(np.ones(2), size=(draws, 2)),
            radii=radii,
            stats=("mean", "p50"),
            radius_kernel="dirichlet",
            rung_sd=np.ones((2, 3)),
        )
        s = post.summary()
        np.testing.assert_allclose(s["r50_mean"], [300 / np.sqrt(2)] * 2)
        np.testing.assert_allclose(s["r90_ci_high"], [np.sqrt(0.9) * 300] * 2)
        self.assertEqual(s["distance_scale"], "raw")
        self.assertEqual(s["distance_basis"], "mean-equivalent (approximate)")
        self.assertEqual(len(s["implied_weight_curve"]["weight"]), 2)

    def test_prep_reports_the_sd_it_divided_by(self):
        rng = np.random.default_rng(6)
        X = rng.normal(size=(300, 2, 2, 1)) * np.array([1.0, 3.0])[None, None, :, None]
        *_, info = bi.prep(X, rng.normal(size=300), None, return_info=True)
        self.assertEqual(info["column_sd"].shape, (2, 2, 1))
        self.assertGreater(info["column_sd"][0, 1, 0], 2.5 * info["column_sd"][0, 0, 0])


class TestSynergyPosteriorSamples(unittest.TestCase):
    """The synergy grid fit must sample, not freeze or diverge.

    Its interactions move with the kernel and aggregator, and a geometric mean
    has an infinite slope where a clipped channel reaches 0; the channels here
    are centred on 0, so half of every interaction sits on that edge.
    """

    def test_single_short_chains_recover_the_linear_fits_effect(self):
        rng = np.random.default_rng(0)
        n = 800
        X = rng.normal(size=(n, 2, len(RADII), len(STATS)))
        design = bi.SynergyDesign.build(X)
        y = 0.23 * X[:, 0, 1, 0] + rng.normal(size=n)
        yr, Xr = bi.prep(X, y, None)
        lin = bi.fit(
            Xr,
            yr,
            form="linear",
            radii=RADII,
            stats=STATS,
            draws=200,
            warmup=200,
            chains=1,
            seed=1,
        )
        b_lin = float(np.asarray(lin.get_samples()["beta"]).mean())
        for seed in (1, 3, 5, 7):
            m = bi.fit(
                Xr,
                yr,
                form="synergy",
                radii=RADII,
                stats=STATS,
                draws=200,
                warmup=200,
                chains=1,
                seed=seed,
                design=design,
                design_channels=(0, 1),
            )
            beta = np.asarray(m.get_samples()["beta"])
            self.assertGreater(float(beta.std()), 0.01, f"seed {seed} froze")
            self.assertLess(abs(float(beta.mean()) - b_lin), 0.1, f"seed {seed}")

    def test_the_grid_fit_does_not_diverge(self):
        rng = np.random.default_rng(0)
        n = 800
        X = rng.normal(size=(n, 2, len(RADII), 2))
        y = 0.23 * X[:, 0, 1, 0] + rng.normal(size=n)
        yr, Xr = bi.prep(X, y, None)
        m = bi.fit(
            Xr,
            yr,
            form="synergy",
            radii=RADII,
            stats=["mean", "p50"],
            draws=300,
            warmup=300,
            chains=2,
            seed=1,
            design=bi.SynergyDesign.build(X),
            design_channels=(0, 1),
        )
        self.assertLess(int(np.sum(m.get_extra_fields()["diverging"])), 10)

    def test_it_refuses_to_run_without_the_raw_cell_values(self):
        rng = np.random.default_rng(0)
        _, Xr = bi.prep(rng.normal(size=(50, 2, 2, 1)), rng.normal(size=50), None)
        with self.assertRaises(ValueError):
            bi.fit(Xr, rng.normal(size=50), form="synergy", radii=RADII[:2])


class _StubMCMC:
    def __init__(self, beta):
        self._beta = beta

    def get_samples(self):
        return {"beta": self._beta}


class TestNullCalibrationPrecision(unittest.TestCase):
    """A false-positive rate is quoted with its uncertainty, or not at all.

    The NUTS refits are stubbed: what is under test is the bookkeeping around
    them, and a real refit per permutation would make the suite take minutes.
    """

    def setUp(self):
        self._real_fit = bi.fit
        self.forms_seen = []

        def stub(E, y, *, form="linear", seed=0, **kw):
            self.forms_seen.append(form)
            centre = 1.0 if seed % 4 == 0 else 0.0
            return _StubMCMC(np.random.default_rng(seed).normal(centre, 0.1, 200))

        bi.fit = stub

    def tearDown(self):
        bi.fit = self._real_fit

    @staticmethod
    def _null_grid(n=600, seed=8):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 2, 1, 1))
        return bi.prep(X, rng.normal(size=n), None)[::-1]

    def test_the_interval_is_the_exact_binomial_one(self):
        from scipy.stats import binomtest

        for k, n in ((0, 16), (1, 16), (10, 200), (16, 16)):
            ci = binomtest(k, n).proportion_ci(method="exact")
            lo, hi = bi.clopper_pearson(k, n)
            self.assertAlmostEqual(lo, ci.low, places=10)
            self.assertAlmostEqual(hi, ci.high, places=10)

    def test_zero_of_sixteen_still_allows_a_twenty_percent_rate(self):
        self.assertAlmostEqual(bi.clopper_pearson(0, 16)[1], 0.206, places=3)

    def test_sixteen_runs_are_flagged_and_logged(self):
        Xr, yr = self._null_grid()
        with self.assertLogs("geofuse.bayesian_index", level="WARNING") as cm:
            out = bi.null_calibration(Xr, yr, n=16, workers=1)
        self.assertTrue(out["imprecise"])
        self.assertIn("too imprecise to support a calibration claim", out["note"])
        self.assertIn("too imprecise", cm.output[0])
        self.assertEqual(out["excluded_zero"], 4)
        self.assertEqual(
            (out["rate_ci_low"], out["rate_ci_high"]), bi.clopper_pearson(4, 16)
        )

    def test_a_publication_run_is_not_flagged(self):
        Xr, yr = self._null_grid()
        out = bi.null_calibration(Xr, yr, n=bi.NULL_RUNS_PUBLICATION, workers=1)
        self.assertFalse(out["imprecise"])
        self.assertNotIn("note", out)

    def test_the_form_is_fixed_unless_reselection_is_asked_for(self):
        Xr, yr = self._null_grid()
        out = bi.null_calibration(Xr, yr, form="synergy", n=6, workers=1)
        self.assertEqual(set(self.forms_seen), {"synergy"})
        self.assertNotIn("form_counts", out)

    def test_reselection_varies_the_form_across_null_permutations(self):
        Xr, yr = self._null_grid()
        E = Xr.reshape(len(Xr), -1)
        out = bi.null_calibration(
            Xr,
            yr,
            n=16,
            workers=1,
            reselect_form=True,
            form_E={"linear": E, "synergy": np.column_stack([E, E[:, 0] * E[:, 1]])},
        )
        self.assertEqual(sum(out["form_counts"].values()), 16)
        self.assertGreater(min(out["form_counts"].values()), 0, out["form_counts"])
        self.assertEqual(len(set(self.forms_seen)), 2)

    def test_reselection_needs_the_picked_columns(self):
        Xr, yr = self._null_grid()
        with self.assertRaises(ValueError):
            bi.null_calibration(Xr, yr, n=2, workers=1, reselect_form=True)


class TestExposureResponseProjectionIsRankSafe(unittest.TestCase):
    def test_a_duplicated_linear_column_removes_only_its_span(self):
        from geofuse.exposure_response import _orthogonalise

        rng = np.random.default_rng(4)
        n = 200
        x = rng.normal(size=n)
        block = rng.normal(size=(n, 3))
        clean = np.column_stack([np.ones(n), x])
        dup = np.column_stack([np.ones(n), x, 2.0 * x])
        r_clean, keep_clean = _orthogonalise(block, clean)
        r_dup, keep_dup = _orthogonalise(block, dup)
        np.testing.assert_array_equal(keep_clean, keep_dup)
        np.testing.assert_allclose(r_dup, r_clean, atol=1e-10)


class TestPanelPosterior(unittest.TestCase):
    def test_the_entity_intercept_widens_the_effect_interval(self):
        # Exposure and a strong intercept both vary by entity, so treating the
        # waves as independent rows understates the uncertainty on beta.
        rng = np.random.default_rng(31)
        n_ent, waves = 120, 8
        ent = np.repeat(np.arange(n_ent), waves)
        E = np.repeat(rng.normal(size=(n_ent, 2)), waves, axis=0)
        E += 0.1 * rng.normal(size=E.shape)
        y = np.repeat(rng.normal(size=n_ent), waves) * 2.0 + rng.normal(size=len(ent))
        y = (y - y.mean()) / y.std()
        kw = {"form": "linear", "draws": 200, "warmup": 200, "chains": 1, "seed": 1}
        plain = bi.fit(E, y, **kw).get_samples()
        panel = bi.fit(E, y, entities=ent, **kw).get_samples()
        self.assertIn("sigma_u", panel)
        self.assertGreater(
            float(np.std(panel["beta"])), 1.5 * float(np.std(plain["beta"]))
        )

    def test_one_row_per_entity_is_the_plain_fit_draw_for_draw(self):
        # Singleton groups cannot separate the entity variance from the row
        # variance, so the intercept is left out rather than freezing chains.
        rng = np.random.default_rng(32)
        n = 300
        E = rng.normal(size=(n, 2))
        y = 0.3 * E[:, 0] + rng.normal(size=n)
        y = (y - y.mean()) / y.std()
        kw = {"form": "linear", "draws": 800, "warmup": 600, "chains": 2, "seed": 1}
        plain = bi.fit(E, y, **kw)
        with self.assertLogs("geofuse.bayesian_index", "WARNING"):
            single = bi.fit(E, y, entities=np.arange(n).astype(str), **kw)
        self.assertNotIn("sigma_u", single.get_samples())
        np.testing.assert_array_equal(
            plain.get_samples()["beta"], single.get_samples()["beta"]
        )
        post = bi.posterior_from(
            single,
            channels=("a", "b"),
            picked=((1, "mean"), (1, "mean")),
            form="linear",
        )
        self.assertLessEqual(post.rhat_max, 1.01)
        self.assertTrue(post.summary()["converged"])

    def test_panel_entities_are_kept_and_singletons_dropped(self):
        panel = np.array([0, 0, 1, 2, 2])
        self.assertIs(bi.panel_entities(panel), panel)
        self.assertIsNone(bi.panel_entities(None))
        with self.assertLogs("geofuse.bayesian_index", "WARNING"):
            self.assertIsNone(bi.panel_entities(np.arange(5)))
        self.assertIsNone(bi.panel_entities(np.arange(5), warn=False))


class TestPanelPermutation(unittest.TestCase):
    def test_rows_are_permuted_as_before_without_entities(self):
        y = np.arange(50.0)
        expected = np.random.default_rng(4).permutation(y)
        np.testing.assert_array_equal(bi._permuted(y, 4), expected)

    def test_whole_histories_move_between_entities_of_one_wave_pattern(self):
        # Entities 0-2 have waves (1, 2); entities 3-4 only wave 1.
        ent = np.array([0, 0, 1, 1, 2, 2, 3, 4])
        wave = np.array([1, 2, 1, 2, 1, 2, 1, 1])
        y = np.array([10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 50.0])
        histories = {(10.0, 11.0), (20.0, 21.0), (30.0, 31.0)}
        for seed in range(10):
            out = bi._permuted(y, seed, ent, wave)
            for e in range(3):
                self.assertIn(tuple(out[ent == e]), histories)
            self.assertEqual(sorted(out[ent >= 3]), [40.0, 50.0])
            self.assertEqual(sorted(out), sorted(y))


class TestGroupedInnerSplits(unittest.TestCase):
    def test_row_splits_are_unchanged_without_groups(self):
        expected = np.random.default_rng(7).random(500) < 0.25
        np.testing.assert_array_equal(bi._holdout(500, 7, 0.25), expected)

    def test_a_group_never_sits_on_both_sides(self):
        groups = np.repeat(np.arange(120), 3)
        for seed in range(20):
            te = bi._holdout(len(groups), seed, 0.25, groups)
            self.assertTrue(te.any() and (~te).any())
            for g in np.unique(groups):
                self.assertEqual(len(set(te[groups == g])), 1)

    def test_the_sweep_runs_on_grouped_splits(self):
        rng = np.random.default_rng(3)
        n_ent, waves = 150, 3
        groups = np.repeat(np.arange(n_ent), waves)
        X = rng.normal(size=(n_ent * waves, 2, 2, 1))
        y = X[:, 0, 1, 0] + rng.normal(size=len(groups))
        yr, Xr = bi.prep(X, y, None)
        res = bi.sweep(
            Xr,
            [100.0, 300.0],
            ["mean"],
            yr,
            channels=["a", "b"],
            channel_index=[0, 1],
            splits=4,
            workers=1,
            groups=groups,
        )
        self.assertEqual(res.picked[0], (300, "mean"))


def _linear_grid(n, seed=1, effect=0.15):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(n, 2, 4, 3))
    E = np.cumsum(E, axis=2) / np.sqrt(np.arange(1, 5))[None, None, :, None]
    y = effect * E[:, 0, 1, 0] + rng.normal(size=n)
    return E, (y - y.mean()) / y.std()


_GRID = {"radii": [100, 200, 300, 400], "stats": ("a", "b", "c")}


class TestCrossProductLikelihood(unittest.TestCase):
    """The linear form's likelihood from the columns' cross-products.

    It is the row likelihood term for term, so the posterior is the same; the
    draws themselves differ, since float32 sums in another order send the
    trajectories apart, so the comparison is on posterior summaries.
    """

    def _close(self, a, b, key):
        sa, sb = np.asarray(a[key]), np.asarray(b[key])
        sd = np.maximum(sa.std(axis=0), 1e-6)
        self.assertTrue(
            np.all(np.abs(sa.mean(axis=0) - sb.mean(axis=0)) < 0.3 * sd), key
        )

    def test_the_grid_posterior_is_the_row_posterior(self):
        E, y = _linear_grid(3000)
        kw = dict(_GRID, draws=400, warmup=400, chains=2, seed=3)
        rows = bi.fit(E, y, likelihood="rows", **kw).get_samples()
        cross = bi.fit(E, y, **kw).get_samples()
        for key in ("beta", "w", "mu", "a", "sigma"):
            self._close(rows, cross, key)
        for q in (2.5, 97.5):
            self.assertAlmostEqual(
                np.percentile(rows["beta"], q), np.percentile(cross["beta"], q), 2
            )

    def test_fixed_columns_too(self):
        E, y = _linear_grid(3000)
        X = E[:, :, 1, 0]
        kw = {"draws": 400, "warmup": 400, "chains": 2, "seed": 3}
        rows = bi.fit(X, y, likelihood="rows", **kw).get_samples()
        cross = bi.fit(X, y, **kw).get_samples()
        for key in ("beta", "w", "sigma"):
            self._close(rows, cross, key)

    def test_the_cross_products_replace_the_rows(self):
        E, y = _linear_grid(2000)
        cross = bi.linear_cross_products(E, y)
        self.assertEqual(cross["S"].shape, (24, 24))
        del cross["Xc"]
        kw = dict(_GRID, draws=200, warmup=200, chains=1, seed=5)
        given = bi.fit(None, None, cross_products=cross, **kw).get_samples()
        own = bi.fit(E, y, **kw).get_samples()
        np.testing.assert_allclose(given["beta"], own["beta"], rtol=1e-5)

    def test_panels_and_synergy_keep_the_row_likelihood(self):
        E, y = _linear_grid(600)
        ent = np.repeat(np.arange(200), 3)
        kw = dict(_GRID, entities=ent, draws=100, warmup=100, chains=1, seed=2)
        auto = bi.fit(E, y, **kw).get_samples()
        rows = bi.fit(E, y, likelihood="rows", **kw).get_samples()
        self.assertIn("sigma_u", auto)
        np.testing.assert_array_equal(auto["beta"], rows["beta"])
        with self.assertRaises(ValueError):
            bi.fit(None, None, cross_products=bi.linear_cross_products(E, y), **kw)

    @unittest.skipUnless(os.environ.get("GEOFUSE_SLOW_TESTS"), "timing check")
    def test_the_cost_does_not_grow_with_n(self):
        import time

        kw = dict(_GRID, draws=300, warmup=300, chains=2, seed=1)
        took = {}
        for n in (2000, 20000):
            E, y = _linear_grid(n)
            t = time.perf_counter()
            bi.fit(E, y, **kw)
            took[n] = time.perf_counter() - t
        self.assertLess(took[20000], 1.5 * took[2000])


class TestNullRefits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.E, cls.y = _linear_grid(3000)
        cls.post, cls.grid, cls.kwargs = bi.grid_posterior(
            cls.E,
            cls.y,
            channels=("x1", "x2"),
            channel_index=(0, 1),
            radius_idx=[range(4), range(4)],
            form="linear",
            picked=((200, "a"), (200, "a")),
            draws=300,
            warmup=300,
            chains=2,
            seed=1,
            **_GRID,
        )

    def test_the_main_fit_hands_over_its_adapted_sampler(self):
        warm = self.post.warm_start
        self.assertGreater(warm["step_size"], 0.0)
        self.assertEqual(warm["warmup"], bi.NULL_WARMUP_WARM)
        self.assertTrue(warm["adapt_mass_matrix"])
        self.assertEqual(set(warm["init"]), {"a", "beta", "kw", "mu", "sigma", "w"})
        (block,) = warm["inverse_mass_matrix"].values()
        self.assertEqual(block.ndim, 1)

    def test_cross_product_refits_match_the_row_refits(self):
        drop = ("design", "design_channels")
        kwargs = {k: v for k, v in self.kwargs.items() if k not in drop}
        for seed in range(3000, 3004):
            yp = bi._permuted(self.y, seed)
            cross = bi.linear_cross_products(self.grid, yp)
            path = bi._write_memmap(cross["S"], ".test-cross")
            try:
                fast = bi._null_fit_linear(
                    (
                        path,
                        cross["S"].shape[0],
                        cross["shape"],
                        cross["cy"],
                        cross["yy"],
                        "linear",
                        seed,
                        kwargs,
                    )
                )
            finally:
                os.unlink(path)
            path = bi._write_memmap(self.grid, ".test-grid")
            try:
                slow = bi._null_fit(
                    (
                        path,
                        self.grid.shape,
                        self.y,
                        "linear",
                        seed,
                        dict(kwargs, likelihood="rows"),
                        None,
                        (None, None, None),
                    )
                )
            finally:
                os.unlink(path)
            self.assertEqual(fast, slow, seed)

    def test_the_calibration_records_how_it_refitted_and_logs_progress(self):
        with self.assertLogs("geofuse.bayesian_index", "INFO") as logs:
            out = bi.null_calibration(
                self.grid,
                self.y,
                form="linear",
                n=4,
                workers=1,
                warm_start=self.post.warm_start,
                **self.kwargs,
            )
        self.assertEqual(out["likelihood"], "cross_products")
        self.assertEqual(out["warm_start"]["warmup"], bi.NULL_WARMUP_WARM)
        self.assertGreater(out["elapsed_s"], 0.0)
        self.assertTrue(any("4 / 4 refits" in m for m in logs.output), logs.output)

    def test_a_warm_refit_mixes_like_a_cold_one(self):
        drop = ("design", "design_channels")
        kwargs = {k: v for k, v in self.kwargs.items() if k not in drop}
        yp = bi._permuted(self.y, 3005)
        kw = dict(kwargs, draws=300, warmup=300, chains=4, seed=3005)
        summaries = {}
        warm = {"warm_start": self.post.warm_start}
        for label, extra in (("cold", {}), ("warm", warm)):
            m = bi.fit(self.grid, yp, **kw, **extra)
            post = bi.posterior_from(
                m,
                channels=("x1", "x2"),
                picked=((200, "a"), (200, "a")),
                form="linear",
                radius_kernel="lognormal",
                **_GRID,
            )
            lo, hi = np.percentile(np.asarray(m.get_samples()["beta"]), [2.5, 97.5])
            summaries[label] = (post.rhat_max, hi - lo)
        self.assertLess(summaries["warm"][0], 1.05)
        self.assertAlmostEqual(
            summaries["warm"][1], summaries["cold"][1], delta=0.2 * summaries["cold"][1]
        )


# Run in a fresh interpreter: XLA fixes its CPU device count the first time a
# process starts JAX, so it cannot be tested inside one that already has.
_CHAINS_SCRIPT = """
import json, logging, sys, warnings
sys.path.insert(0, {root!r})
import numpy as np
from geofuse import bayesian_index as bi
if {start_jax}:
    import jax.numpy as jnp
    jnp.zeros(1)
records = []
handler = logging.Handler()
handler.emit = lambda r: records.append(r.getMessage())
logging.getLogger("geofuse.bayesian_index").addHandler(handler)
logging.getLogger("geofuse.bayesian_index").setLevel(logging.INFO)
rng = np.random.default_rng(5)
ent = np.repeat(np.arange(100), 3)
E = rng.normal(size=(len(ent), 2))
y = 0.3 * E[:, 0] + rng.normal(size=len(ent))
kw = dict(entities=ent, draws=100, warmup=100, chains=4, seed=3)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    a = np.asarray(bi.fit(E, y, **kw).get_samples()["beta"])
    b = np.asarray(bi.fit(E, y, **kw).get_samples()["beta"])
import jax
print(json.dumps({{
    "devices": jax.local_device_count(),
    "same": bool(np.array_equal(a, b)),
    "device_warnings": [str(w.message) for w in caught if "devices" in str(w.message)],
    "logs": records,
}}))
"""


def _chains_run(start_jax: bool) -> dict:
    import json
    import subprocess

    script = _CHAINS_SCRIPT.format(root=ROOT, start_jax=start_jax)
    env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


class TestParallelChains(unittest.TestCase):
    def test_an_entity_fit_runs_its_chains_in_parallel(self):
        # The entity block touches JAX first; the devices must be asked before.
        out = _chains_run(start_jax=False)
        self.assertGreaterEqual(out["devices"], 4)
        self.assertEqual(out["device_warnings"], [])
        self.assertTrue(out["same"])

    def test_a_process_that_started_jax_runs_them_in_turn_without_a_warning(self):
        out = _chains_run(start_jax=True)
        self.assertEqual(out["devices"], 1)
        self.assertEqual(out["device_warnings"], [])
        self.assertTrue(any("one after another" in m for m in out["logs"]))
        self.assertTrue(out["same"])

    def test_a_requested_device_count_is_never_lowered(self):
        saved = os.environ.get("XLA_FLAGS")

        def restore():
            if saved is None:
                os.environ.pop("XLA_FLAGS", None)
            else:
                os.environ["XLA_FLAGS"] = saved

        self.addCleanup(restore)
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8 --x=1"
        bi.reserve_host_devices(4)
        self.assertIn("device_count=8", os.environ["XLA_FLAGS"])
        bi.reserve_host_devices(12)
        self.assertIn("device_count=12", os.environ["XLA_FLAGS"])
        self.assertIn("--x=1", os.environ["XLA_FLAGS"])


if __name__ == "__main__":
    unittest.main()
