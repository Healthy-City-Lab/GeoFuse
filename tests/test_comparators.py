"""Comparator exposures are scored exactly as the composite is, untuned.

A comparator is a finished exposure column of the target. Pinned to
``comparator_mode``, the engine takes the column's per-entity value as its
composite, so every scoring path — held-out score and bootstrap, effects,
subset slices, paired difference, negative controls — runs unchanged. One
synthetic study, prepared twice: engine A computes the composite and a single
channel's composite at fixed params; engine B carries both as comparator
columns, plus a copy with gaps. Pinned here:

- comparators leave the composite and the split untouched;
- a comparator equal to the frozen composite scores like it, held-out score,
  interval and p-value alike, with a paired difference of zero;
- a comparator equal to a single channel's composite scores like that channel;
- missing values cost only that comparator's statistics, and ``n`` says so;
- its negative-control rows are the transfer test on the same vector;
- the Holm correction runs over the primary comparators only;
- a comparator that is also a covariate is refused.
"""

import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (os.path.dirname(os.path.abspath(__file__)), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import geopandas as gpd
import numpy as np

from geofuse.fusion import MetricFusionEngine, comparator_mode
from geofuse.jobs.fusion_outputs import _comparator_family
from test_fusion_run import _write_study

PARAMS = {
    "veg_weight": 30,
    "terrain_weight": 30,
    "ndvi_weight": 40,
    "veg_radius": 200,
    "terrain_radius": 200,
    "ndvi_radius": 200,
    "veg_stat": "mean",
    "veg_percentile": 50,
    "terrain_stat": "mean",
    "terrain_percentile": 50,
    "ndvi_stat": "mean",
    "ndvi_percentile": 50,
}
METRIC = "r2"


def _engine(folder, target, cache, comparators=()):
    eng = MetricFusionEngine(
        target_file=os.path.join(folder, target),
        target_feature="outcome",
        buffer_meters=300,
        gvi_buffer_min_m=100,
        gvi_buffer_max_m=300,
        gvi_buffer_step_m=100,
        ndvi_buffer_min_m=100,
        ndvi_buffer_max_m=300,
        ndvi_buffer_step_m=100,
        cache_dir=os.path.join(folder, cache),
        cgi_grid_spacing_m=100,
        covariate_columns=["age"],
        negative_control_columns=["grip"],
        comparator_columns=list(comparators),
    )
    eng.load_target()
    eng.load_metrics(
        veg_file=os.path.join(folder, "gvi.gpkg"),
        ndvi_file=os.path.join(folder, "ndvi_2015.tif"),
        cache_metrics=False,
    )
    df = eng.prepare_fusion_data()
    eng.precompute_aggregations()
    eng.split_data(test_size=0.25, random_state=42, fusion_df=df)
    return eng


def _as(eng, mode, fn):
    previous = eng._active_greenery_channel
    eng._active_greenery_channel = mode
    try:
        return fn()
    finally:
        eng._active_greenery_channel = previous


def _cmp(name):
    return {"comparator": name}


class TestComparatorScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = tempfile.mkdtemp(prefix="geofuse-comparators-")
        _write_study(folder)
        a = _engine(folder, "cohort.gpkg", "cache_a")
        cgi = a.apply_fusion(weights=dict(PARAMS)).set_index("polygon_id")
        ndvi = _as(a, "ndvi", lambda: a.apply_fusion(weights=dict(PARAMS)))
        ndvi = ndvi.set_index("polygon_id")

        cohort = gpd.read_file(os.path.join(folder, "cohort.gpkg"))
        rows = np.arange(len(cohort))
        cohort["cgi_copy"] = cgi["composite"].reindex(rows).to_numpy()
        cohort["ndvi_copy"] = ndvi["composite"].reindex(rows).to_numpy()
        patchy = cohort["ndvi_copy"].to_numpy().copy()
        patchy[::5] = np.nan
        cohort["gappy_copy"] = patchy
        cohort.to_file(os.path.join(folder, "cohort_b.gpkg"), driver="GPKG")
        cls.a = a
        cls.b = _engine(
            folder,
            "cohort_b.gpkg",
            "cache_b",
            comparators=("cgi_copy", "ndvi_copy", "gappy_copy"),
        )
        cls.gappy_rows = set(np.flatnonzero(np.isnan(patchy)).tolist())

    def test_comparators_leave_the_composite_and_the_split_alone(self):
        for part in ("test_data", "train_val_data"):
            self.assertEqual(
                set(getattr(self.a, part)["polygon_id"]),
                set(getattr(self.b, part)["polygon_id"]),
            )
        self.assertAlmostEqual(
            self.a.evaluate_on_test(params=dict(PARAMS), metric=METRIC)["test_score"],
            self.b.evaluate_on_test(params=dict(PARAMS), metric=METRIC)["test_score"],
            places=10,
        )

    def test_a_copy_of_the_composite_scores_like_it(self):
        b, mode = self.b, comparator_mode("cgi_copy")
        cgi = b.evaluate_on_test(params=dict(PARAMS), metric=METRIC)
        cmp = _as(
            b, mode, lambda: b.evaluate_on_test(params=_cmp("cgi_copy"), metric=METRIC)
        )
        self.assertAlmostEqual(cmp["test_score"], cgi["test_score"], places=5)
        ci_kw = dict(n_bootstrap=200, ci_level=0.95, method="percentile", seed=42)
        cgi_ci = b.bootstrap_test_score_ci(dict(PARAMS), METRIC, **ci_kw)
        cmp_ci = _as(
            b,
            mode,
            lambda: b.bootstrap_test_score_ci(_cmp("cgi_copy"), METRIC, **ci_kw),
        )
        self.assertAlmostEqual(cmp_ci["lower"], cgi_ci["lower"], places=5)
        self.assertAlmostEqual(cmp_ci["upper"], cgi_ci["upper"], places=5)
        eff_kw = dict(n_bootstrap=100, n_perm=99, seed=42)
        cgi_p = b.evaluate_effects(dict(PARAMS), METRIC, **eff_kw)["test"]["p_value"]
        cmp_p = _as(
            b,
            mode,
            lambda: b.evaluate_effects(_cmp("cgi_copy"), METRIC, **eff_kw),
        )["test"]["p_value"]
        self.assertAlmostEqual(cmp_p, cgi_p, places=6)

        paired = b.paired_objective_difference(
            dict(PARAMS), _cmp("cgi_copy"), mode, METRIC, n_bootstrap=100, seed=42
        )
        self.assertAlmostEqual(paired["observed_diff"], 0.0, places=5)
        self.assertAlmostEqual(paired["lower"], 0.0, places=5)
        self.assertAlmostEqual(paired["upper"], 0.0, places=5)

    def test_a_copy_of_a_single_channel_scores_like_it(self):
        b = self.b
        channel = _as(
            b, "ndvi", lambda: b.evaluate_on_test(params=dict(PARAMS), metric=METRIC)
        )
        cmp = _as(
            b,
            comparator_mode("ndvi_copy"),
            lambda: b.evaluate_on_test(params=_cmp("ndvi_copy"), metric=METRIC),
        )
        self.assertAlmostEqual(cmp["test_score"], channel["test_score"], places=5)

    def test_missing_values_cost_only_that_comparator(self):
        b = self.b

        def counts(name):
            def run():
                eff = b.evaluate_effects(_cmp(name), METRIC, n_bootstrap=50, n_perm=0)
                slopes = b.compute_partial_slopes(_cmp(name))
                return eff, slopes

            return _as(b, comparator_mode(name), run)

        full_eff, full_slopes = counts("ndvi_copy")
        gap_eff, gap_slopes = counts("gappy_copy")
        test_ids = set(b.test_data["polygon_id"])
        missing_test = len(self.gappy_rows & test_ids)
        self.assertGreater(missing_test, 0)
        self.assertEqual(gap_eff["test"]["n"], full_eff["test"]["n"] - missing_test)
        self.assertEqual(
            gap_slopes["test"]["n"], full_slopes["test"]["n"] - missing_test
        )
        studied = set(b.apply_fusion(weights=dict(PARAMS))["polygon_id"])
        self.assertEqual(
            gap_eff["all"]["n"], full_eff["all"]["n"] - len(self.gappy_rows & studied)
        )
        # The composite itself keeps every entity.
        cgi = b.evaluate_effects(dict(PARAMS), METRIC, n_bootstrap=50, n_perm=0)
        self.assertEqual(cgi["all"]["n"], full_eff["all"]["n"])

    def test_negative_controls_are_the_transfer_test_on_the_same_vector(self):
        b = self.b
        cgi = b.compute_negative_controls(dict(PARAMS), n_boot=100, seed=42)
        cmp = _as(
            b,
            comparator_mode("cgi_copy"),
            lambda: b.compute_negative_controls(_cmp("cgi_copy"), n_boot=100, seed=42),
        )
        for split in ("test", "train_val"):
            ours, theirs = cmp["splits"][split], cgi["splits"][split]
            self.assertAlmostEqual(
                ours["target"]["beta"], theirs["target"]["beta"], places=5
            )
            for key in ("beta", "delta", "delta_ci_low", "delta_ci_high"):
                self.assertAlmostEqual(
                    ours["controls"]["grip"][key],
                    theirs["controls"]["grip"][key],
                    places=5,
                    msg=key,
                )


class TestComparatorFamily(unittest.TestCase):
    def test_holm_runs_over_the_primary_comparators_only(self):
        comparators = {
            "a": {"paired": {"comparator": "a", "primary": True, "p_value": 0.01}},
            "b": {"paired": {"comparator": "b", "primary": True, "p_value": 0.04}},
            "c": {"paired": {"comparator": "c", "primary": False, "p_value": 0.001}},
            "d": {"paired": None},
        }
        rows = {r["comparator"]: r for r in _comparator_family(comparators)}
        self.assertEqual(sorted(rows), ["a", "b", "c"])
        self.assertAlmostEqual(rows["a"]["p_value_holm"], 0.02)
        self.assertAlmostEqual(rows["b"]["p_value_holm"], 0.04)
        self.assertEqual({rows[k]["family_size"] for k in rows}, {2})
        self.assertIsNone(rows["c"]["p_value_holm"])
        self.assertTrue(rows["c"]["descriptive"])
        self.assertFalse(rows["a"]["descriptive"])


class TestRefusals(unittest.TestCase):
    def test_a_comparator_that_is_also_a_covariate_or_control_is_refused(self):
        for kw, name in (
            ({"covariate_columns": ["age"]}, "age"),
            ({"negative_control_columns": ["grip"]}, "grip"),
            ({"target_feature": "outcome"}, "outcome"),
        ):
            with self.assertRaises(ValueError) as ctx:
                MetricFusionEngine(
                    target_file="dummy.geojson",
                    cache_dir=tempfile.mkdtemp(),
                    comparator_columns=[name],
                    **kw,
                )
            self.assertIn(name, str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
