"""The discovery diagnostics panel must render every shape the engine emits.

The panel reads a bundle assembled from several optional stages, so most of its
fields can legitimately be absent — a run with the gain comparison disabled, a
standalone study with one channel, a synergy fit whose weight vector is longer
than its channel list. Each of those used to be a lookup that raised only when
a user opened the results tab, long after the run finished.
"""

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (os.path.join(ROOT, "ui"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from tabs import fusion as fusion_tab


class _Recorder:
    """Collects what the panel drew instead of talking to a Streamlit runtime."""

    def __init__(self):
        self.calls: list[tuple[str, object]] = []

    def __getattr__(self, name):
        def call(*a, **k):
            self.calls.append((name, a[0] if a else None))
            return self

        return call

    def columns(self, n):
        return [self for _ in range(n if isinstance(n, int) else len(n))]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def text(self) -> str:
        return " ".join(str(v) for _n, v in self.calls)


def _summary(**over) -> dict:
    s = {
        "picked": [[400, "mean"], [800, "p50"]],
        "channels": ["ndvi", "gvi"],
        "weight_labels": ["ndvi", "gvi"],
        "form": "linear",
        "form_scores": {"linear": 2.4, "synergy": 2.1},
        "sweep_score": 2.4,
        "one_se_picked": [[400, "mean"], [800, "p50"]],
        "boundary_hit": [],
        "n_candidates": 324,
        "distinct_split_winners": 9,
        "weight_mean": [0.6, 0.4],
        "weight_ci_low": [0.3, 0.1],
        "weight_ci_high": [0.9, 0.7],
        "powers": None,
        "beta_mean": 0.11,
        "beta_ci_low": 0.02,
        "beta_ci_high": 0.2,
        "p_direction": 0.99,
        "rhat_max": 1.001,
        "ess_min": 1200.0,
        "divergences": 0,
        "discovery": {
            "n_results": 60,
            "reps": 5,
            "shuffles": 12,
            "distinct_picks": 4,
            "pick_counts": {"a": 40, "b": 20},
            "per_form": {
                "linear": {"train_t": 3.1, "test_t": 2.4, "shrinkage": 0.7},
                "synergy": {"train_t": 3.6, "test_t": 2.1, "shrinkage": 1.5},
            },
        },
        "holdout_gain": {
            "cgi": 2.4,
            "best_single": 2.2,
            "gain": 0.2,
            "gain_p": 0.08,
            "gain_won": 13,
            "splits": 20,
            "standalone_ndvi": 2.2,
            "standalone_gvi": 1.9,
        },
        "null_calibration": {"runs": 16, "excluded_zero": 1, "rate": 0.0625},
        "elapsed_s": 412.0,
    }
    s.update(over)
    return s


class TestPanelRenders(unittest.TestCase):
    def setUp(self):
        self._real_st = fusion_tab.st
        self.rec = _Recorder()
        fusion_tab.st = self.rec

    def tearDown(self):
        fusion_tab.st = self._real_st

    def _render(self, summary):
        fusion_tab._render_posterior_diagnostics(summary, "DCOR")
        return self.rec.text()

    def test_a_full_bundle_reports_the_pick_and_the_checks(self):
        out = self._render(_summary())
        for expected in (
            "Sweep",
            "Posterior weights",
            "Does the discovery reproduce?",
            "Composite vs standalone channels",
            "Null calibration",
        ):
            self.assertIn(expected, out)

    def test_an_empty_bundle_draws_nothing(self):
        self._render({})
        self.assertEqual(self.rec.calls, [])

    def test_optional_stages_can_all_be_missing(self):
        out = self._render(
            _summary(discovery={}, holdout_gain={}, null_calibration={})
        )
        self.assertIn("Sweep", out)
        self.assertNotIn("Null calibration", out)

    def test_a_boundary_pick_is_warned_about(self):
        self._render(_summary(boundary_hit=["gvi"], picked=[[400, "mean"], [1000, "p50"]]))
        warnings = [v for n, v in self.rec.calls if n == "warning"]
        self.assertTrue(any("edge of the searched range" in str(w) for w in warnings))

    def test_a_bad_sampler_is_warned_about(self):
        self._render(_summary(rhat_max=1.4, divergences=37))
        warnings = [v for n, v in self.rec.calls if n == "warning"]
        self.assertTrue(any("did not converge" in str(w) for w in warnings))

    def test_an_uncalibrated_null_is_warned_about(self):
        self._render(
            _summary(null_calibration={"runs": 16, "excluded_zero": 9, "rate": 0.5625})
        )
        warnings = [v for n, v in self.rec.calls if n == "warning"]
        self.assertTrue(any("permuted" in str(w) for w in warnings))

    def test_synergy_pair_weights_get_their_own_label(self):
        out = self._render(
            _summary(
                form="synergy",
                weight_labels=["ndvi", "gvi", "ndvi x gvi"],
                weight_mean=[0.5, 0.3, 0.2],
                weight_ci_low=[0.2, 0.1, 0.0],
                weight_ci_high=[0.8, 0.6, 0.5],
                powers=[0.7, 0.9],
            )
        )
        self.assertIn("Posterior weights", out)

    def test_a_single_channel_standalone_renders(self):
        out = self._render(
            _summary(
                picked=[[600, "p90"]],
                channels=["ndvi"],
                weight_labels=["ndvi"],
                weight_mean=[1.0],
                weight_ci_low=[1.0],
                weight_ci_high=[1.0],
                holdout_gain={},
            )
        )
        self.assertIn("Sweep", out)


def _grid(**over) -> dict:
    """A bundle from a posterior that fitted the grid rather than a picked cell."""
    s = _summary()
    s.update({
        "radius_kernel": "lognormal",
        "radii": [100, 250, 500, 1000],
        "radius_profile": [[0.05, 0.20, 0.70, 0.05], [0.25, 0.25, 0.25, 0.25]],
        "projected_pick": [[500, "p10"], [250, "mean"]],
        "peak_radius_mean": [480.0, 390.0],
        "peak_radius_ci_low": [410.0, 105.0],
        "peak_radius_ci_high": [560.0, 980.0],
        "peak_radius_prior_ci": [90.0, 1100.0],
        "peak_radius_width_ratio": [0.15, 0.87],
        "stats": ["mean", "p10", "p50", "p90"],
        "aggregator_mean": [[0.1, 0.7, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25]],
        "aggregator_ci_low": [[0.02, 0.5, 0.02, 0.02], [0.05, 0.05, 0.05, 0.05]],
        "aggregator_ci_high": [[0.3, 0.88, 0.3, 0.3], [0.6, 0.6, 0.6, 0.6]],
        "aggregator_prior_ci": [0.008, 0.63],
        "aggregator_width_ratio": [[0.45, 0.61, 0.45, 0.45], [0.88, 0.88, 0.88, 0.88]],
        "aggregator_uniform": 0.25,
        "aggregator_informative": [["p10"], []],
        "partial_r2_mean": 0.0012,
        "weight_prior_ci": [0.025, 0.975],
        "weight_width_ratio": [0.63, 0.63],
    })
    s.update(over)
    return s


class TestFittedGridPanel(unittest.TestCase):
    """The scale and aggregator sections, and the prior comparison they carry.

    A posterior as wide as its prior is the model saying the data was silent.
    That has to reach the page as a warning rather than as a confident radius,
    because the point estimate looks identical either way.
    """

    def setUp(self):
        self._real_st = fusion_tab.st
        self.rec = _Recorder()
        fusion_tab.st = self.rec

    def tearDown(self):
        fusion_tab.st = self._real_st

    def _render(self, summary):
        fusion_tab._render_posterior_diagnostics(summary, "DCOR")
        return self.rec.text()

    def test_the_fitted_scale_and_aggregator_are_drawn(self):
        out = self._render(_grid())
        self.assertIn("Fitted spatial scale and aggregator", out)
        self.assertIn("480", out)
        self.assertIn("p10", out)

    def test_an_unidentified_scale_is_warned_about(self):
        out = self._render(_grid())
        self.assertIn("keeps most of its prior width", out)
        self.assertIn("GVI", out)

    def test_an_identified_scale_alone_raises_no_warning(self):
        out = self._render(_grid(peak_radius_width_ratio=[0.15, 0.2]))
        self.assertNotIn("keeps most of its prior width", out)

    def test_the_projection_shipped_to_the_composite_is_named(self):
        out = self._render(_grid())
        self.assertIn("Shipped to the composite", out)
        self.assertIn("500 m p10", out)

    def test_partial_r2_reaches_the_page(self):
        # The recorder captures a metric's label, not its value, so this gates
        # that the tile is drawn at all -- which is what regressed when the
        # beta row was widened to make room for it.
        self.assertIn("Partial R", self._render(_grid()))

    def test_a_bundle_without_the_grid_skips_the_section(self):
        out = self._render(_summary())
        self.assertNotIn("Fitted spatial scale and aggregator", out)

    def test_an_undistinguishable_aggregator_is_called_out(self):
        out = self._render(_grid())
        self.assertIn("No aggregator is distinguishable", out)
        self.assertIn("falls back to the mean", out)

    def test_a_channel_whose_aggregator_separated_is_named(self):
        out = self._render(_grid())
        self.assertIn("Aggregators the data separated", out)
        self.assertIn("NDVI: p10", out)

    def test_all_channels_informative_raises_no_fallback_notice(self):
        out = self._render(_grid(aggregator_informative=[["p10"], ["p90"]]))
        self.assertNotIn("No aggregator is distinguishable", out)

    def test_a_scale_without_an_aggregator_still_renders(self):
        out = self._render(_grid(aggregator_mean=None))
        self.assertIn("Fitted spatial scale and aggregator", out)


class TestRadiusProfileChart(unittest.TestCase):
    """The ladder is numeric; its axis and its bars have to respect that."""

    def setUp(self):
        self._real_st = fusion_tab.st
        self.rec = _Recorder()
        fusion_tab.st = self.rec

    def tearDown(self):
        fusion_tab.st = self._real_st

    def _figure(self, summary):
        fusion_tab._render_posterior_diagnostics(summary, "DCOR")
        for name, val in self.rec.calls:
            if name == "plotly_chart" and getattr(val, "data", None):
                if any(getattr(t, "name", None) for t in val.data):
                    return val
        return None

    def test_the_axis_follows_the_ladder_not_the_label_text(self):
        # "1000 m" sorts before "150 m" as text, which puts the widest buffer in
        # the middle of the axis and turns any profile into a sawtooth.
        fig = self._figure(_grid(radii=[100, 150, 500, 1000],
                                 radius_profile=[[0.1, 0.2, 0.6, 0.1],
                                                 [0.25, 0.25, 0.25, 0.25]]))
        self.assertIsNotNone(fig)
        self.assertEqual(
            list(fig.layout.xaxis.categoryarray),
            ["100 m", "150 m", "500 m", "1000 m"],
        )

    def test_the_channels_sit_side_by_side_rather_than_stacked(self):
        # Each channel's profile sums to 1 on its own, so a stacked bar adds two
        # unrelated distributions into a height that means nothing.
        fig = self._figure(_grid())
        self.assertEqual(fig.layout.barmode, "group")

    def test_every_channel_gets_its_own_series(self):
        fig = self._figure(_grid())
        self.assertEqual({t.name for t in fig.data}, {"NDVI", "GVI"})


class TestSweepAndPosteriorDisagreement(unittest.TestCase):
    """The sweep's cell and the posterior's projection are different objects.

    Showing the sweep table alone made the params panel look inconsistent with
    the diagnostics, when in fact the posterior is meant to override it.
    """

    def setUp(self):
        self._real_st = fusion_tab.st
        self.rec = _Recorder()
        fusion_tab.st = self.rec

    def tearDown(self):
        fusion_tab.st = self._real_st

    def _render(self, summary):
        fusion_tab._render_posterior_diagnostics(summary, "DCOR")
        return self.rec.text()

    def test_a_disagreement_is_called_out_with_what_actually_ships(self):
        out = self._render(_grid(picked=[[600, "p25"], [600, "p10"]],
                                 projected_pick=[[500, "p10"], [500, "p10"]]))
        self.assertIn("not what the study was built from", out)
        self.assertIn("500 m p10", out)

    def test_agreement_stays_quiet(self):
        out = self._render(_grid(picked=[[500, "p10"], [250, "mean"]],
                                 projected_pick=[[500, "p10"], [250, "mean"]]))
        self.assertNotIn("not what the study was built from", out)


class TestAggregatorVerdictForOlderBundles(unittest.TestCase):
    """A bundle written before the verdict was stored still carries its inputs."""

    def setUp(self):
        self._real_st = fusion_tab.st
        self.rec = _Recorder()
        fusion_tab.st = self.rec

    def tearDown(self):
        fusion_tab.st = self._real_st

    def _render(self, summary):
        fusion_tab._render_posterior_diagnostics(summary, "DCOR")
        return self.rec.text()

    def test_a_flat_blend_is_recognised_without_the_stored_field(self):
        s = _grid()
        s.pop("aggregator_informative")
        s["aggregator_ci_low"] = [[0.02, 0.02, 0.02, 0.02]] * 2
        s["aggregator_ci_high"] = [[0.51, 0.51, 0.51, 0.51]] * 2
        self.assertIn("No aggregator is distinguishable", self._render(s))

    def test_a_separated_blend_is_recognised_without_the_stored_field(self):
        s = _grid()
        s.pop("aggregator_informative")
        s["aggregator_ci_low"] = [[0.30, 0.02, 0.02, 0.02]] * 2
        s["aggregator_ci_high"] = [[0.80, 0.20, 0.20, 0.20]] * 2
        out = self._render(s)
        self.assertIn("Aggregators the data separated", out)
        self.assertNotIn("No aggregator is distinguishable", out)


class TestFormChoiceIsReported(unittest.TestCase):
    """Where the form was chosen, and what choosing it on test rows would read."""

    def setUp(self):
        self._real_st = fusion_tab.st
        self.rec = _Recorder()
        fusion_tab.st = self.rec

    def tearDown(self):
        fusion_tab.st = self._real_st

    def _render(self, summary):
        fusion_tab._render_posterior_diagnostics(summary, "DCOR")
        return self.rec.text()

    def test_the_gain_panel_names_the_training_choice_and_the_optimistic_max(self):
        s = _summary()
        s["holdout_gain"].update(
            form_counts={"linear": 15, "synergy": 5},
            cgi_max_over_forms_optimistic=2.6,
        )
        out = self._render(s)
        self.assertIn("`linear` 15, `synergy` 5", out)
        self.assertIn("2.600 - optimistic", out)

    def test_the_discovery_caption_counts_the_chosen_forms(self):
        s = _summary()
        s["discovery"].update(form_counts={"linear": 50, "synergy": 10},
                              chosen_test_t=2.35)
        out = self._render(s)
        self.assertIn("inner splits of its training rows", out)
        self.assertIn("2.350", out)

    def test_the_sweep_names_its_selection_maximum(self):
        out = self._render(_summary(sweep_selection_score=2.9))
        self.assertIn("grid maximum the columns were picked on was 2.900", out)

    def test_older_bundles_without_the_new_keys_still_render(self):
        out = self._render(_summary())
        self.assertNotIn("optimistic, because", out)
        self.assertNotIn("inner splits of its training rows", out)


class _MetricRecorder(_Recorder):
    """Also keeps each metric's value, which the base recorder drops."""

    def metric(self, label, value=None, *a, **k):
        self.calls.append(("metric", f"{label} = {value}"))
        return self


class TestNullCalibrationPanel(unittest.TestCase):
    def setUp(self):
        self._real_st = fusion_tab.st
        self.rec = _MetricRecorder()
        fusion_tab.st = self.rec

    def tearDown(self):
        fusion_tab.st = self._real_st

    def _render(self, null):
        fusion_tab._render_posterior_diagnostics(
            _summary(null_calibration=null), "DCOR")
        return self.rec.text()

    def test_the_rate_carries_its_exact_interval(self):
        out = self._render({"runs": 200, "excluded_zero": 10, "rate": 0.05,
                            "rate_ci_low": 0.0242, "rate_ci_high": 0.0901,
                            "imprecise": False})
        self.assertIn("[2.4%, 9.0%]", out)
        self.assertNotIn("too few to support a calibration claim", out)

    def test_a_short_run_is_called_a_quick_check(self):
        out = self._render({"runs": 16, "excluded_zero": 0, "rate": 0.0,
                            "rate_ci_low": 0.0, "rate_ci_high": 0.206,
                            "imprecise": True})
        self.assertIn("too few to support a calibration claim", out)

    def test_a_reselected_form_is_reported(self):
        out = self._render({"runs": 16, "excluded_zero": 1, "rate": 0.0625,
                            "form_counts": {"linear": 11, "synergy": 5}})
        self.assertIn("covers the form choice as well", out)


class TestCovariateAxisOrdering(unittest.TestCase):
    """Numerics first, then each categorical's levels in value order.

    Ranking every term by partial R² scatters a categorical's levels across the
    axis, which is precisely the arrangement that hides a gradient across them.
    """

    def _frame(self):
        import pandas as pd

        return pd.DataFrame([
            {"covariate": "SDC_MRTL=10.0", "coef": 0.4, "partial_r2": 0.002},
            {"covariate": "AGE", "coef": -0.05, "partial_r2": 0.010},
            {"covariate": "SDC_MRTL=2.0", "coef": 0.1, "partial_r2": 0.001},
            {"covariate": "STRESS", "coef": -0.03, "partial_r2": 0.050},
            {"covariate": "SDC_MRTL=3.0", "coef": 0.2, "partial_r2": 0.004},
        ])

    def test_numeric_terms_come_first_ranked_by_partial_r2(self):
        order = fusion_tab._covariate_term_order(self._frame())
        self.assertEqual(order[:2], ["STRESS", "AGE"])

    def test_levels_are_ordered_by_value_not_by_text(self):
        order = fusion_tab._covariate_term_order(self._frame())
        self.assertEqual(
            order[2:], ["SDC_MRTL=2.0", "SDC_MRTL=3.0", "SDC_MRTL=10.0"]
        )

    def test_a_term_is_split_into_variable_and_level(self):
        self.assertEqual(fusion_tab._split_term("SEX=M"), ("SEX", "M"))
        self.assertEqual(fusion_tab._split_term("AGE"), ("AGE", None))


class TestCovariateTrendLine(unittest.TestCase):
    def test_a_cubic_needs_more_than_four_levels_to_stay_a_summary(self):
        four = [(str(i), float(i)) for i in range(1, 5)]
        self.assertEqual(fusion_tab._trend_points(four)[2], 2)
        six = [(str(i), float(i)) for i in range(1, 7)]
        self.assertEqual(fusion_tab._trend_points(six)[2], 3)

    def test_too_few_levels_produce_no_line(self):
        self.assertIsNone(fusion_tab._trend_points([("1", 0.5), ("2", 0.7)]))

    def test_the_fit_follows_the_level_values_and_their_spacing(self):
        pts = [("1", 1.0), ("2", 2.0), ("5", 5.0), ("9", 9.0), ("12", 12.0),
               ("20", 20.0)]
        labels, fitted, _deg = fusion_tab._trend_points(pts)
        self.assertEqual(labels, ["1", "2", "5", "9", "12", "20"])
        # A straight line through y == x must be recovered, not bent by
        # treating the unequal level gaps as if they were evenly spaced.
        self.assertTrue(
            np.allclose(fitted, [1.0, 2.0, 5.0, 9.0, 12.0, 20.0], atol=1e-6)
        )

    def test_non_numeric_levels_fall_back_to_position(self):
        pts = [("a", 1.0), ("b", 2.0), ("c", 3.0), ("d", 4.0), ("e", 5.0)]
        labels, fitted, _ = fusion_tab._trend_points(pts)
        self.assertEqual(labels, ["a", "b", "c", "d", "e"])
        self.assertTrue(np.allclose(fitted, [1, 2, 3, 4, 5], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
