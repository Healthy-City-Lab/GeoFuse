"""What a run reports about itself: the smooth's record and the split stage.

The summary of a spatially adjusted run names the paired-test rule in force
and the number of spatial clusters. The repeated-split stage is ``done`` only
when every split produced results: one that stops short is marked failed with
a WARN and its report says ``complete: False``, and a cancel during the stage
ends the run as cancelled rather than as a finished stage.

The resampled statistics under a spatial smooth agree with the per-replicate
fits on a design built to strain the rank rules: many clusters too small for
their smooth, which resamples empty, and categories of one to three members,
which resamples drop.
"""

import os
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import geopandas  # noqa: F401  (pyproj before torchvision, see vision.py)
import numpy as np

from geofuse import JobCancelled
from geofuse import negative_controls as nc
from geofuse import objective_scoring as osc
from geofuse import resampling as rs
from geofuse import spatial_basis as sb
from geofuse.fusion import MetricFusionEngine
from geofuse.jobs import runners
from geofuse.jobs.stage_ledger import DONE, FAILED


def _engine(method, rule="own"):
    return MetricFusionEngine(
        target_file="dummy.geojson",
        cache_dir=tempfile.mkdtemp(),
        spatial_adjust_method=method,
        spatial_paired_smooth=rule,
    )


class SummaryRecordTests(unittest.TestCase):
    def test_the_record_names_the_rule_and_the_clusters(self):
        rng = np.random.default_rng(0)
        xy = np.vstack([rng.normal(size=(200, 2)) * 2_000 + c for c in (0, 1e6)])
        y = rng.normal(size=len(xy))
        for method, rule, want in (
            ("spatial_plus", "own", "own"),
            ("spatial_plus", "max", "max"),
            ("ks_aic", "own", "outcome"),
        ):
            eng = _engine(method, rule)
            eng._spatial_selection(xy, y, None, y)
            report = eng.spatial_selection_report()
            self.assertEqual(report["paired_rule"], want)
            self.assertEqual(report["clusters"], 2)

    def test_the_record_outlives_a_re_split(self):
        rng = np.random.default_rng(1)
        xy = np.vstack([rng.normal(size=(200, 2)) * 2_000 + c for c in (0, 1e6)])
        y = rng.normal(size=len(xy))
        for scales in (None, [3_000.0]):
            eng = MetricFusionEngine(
                target_file="dummy.geojson",
                cache_dir=tempfile.mkdtemp(),
                spatial_adjust_method="spatial_plus",
                spatial_adjust_scale_m=scales,
            )
            eng._spatial_selection(xy, y, None, y)
            before = eng.spatial_selection_report()
            # What ``split_data`` resets when the first split is restored.
            eng._spatial_basis_cache = {}
            eng._spatial_adjust_summary = None
            after = eng.spatial_selection_report()
            self.assertEqual(after["clusters"], 2)
            self.assertEqual(after, before)

    def test_unadjusted_runs_have_no_record(self):
        self.assertIsNone(_engine("none").spatial_selection_report())


class SplitStageTests(unittest.TestCase):
    def _run(self, repeated_rows, cancelled=False):
        stages = []
        engine = types.SimpleNamespace(
            compute_partial_slopes=lambda params: {
                "test": {"beta": 0.1, "ci_low": 0.0, "ci_high": 0.2, "t": 2.0, "df": 50}
            },
            split_data=lambda **kw: None,
            _active_greenery_channel="cgi",
            best_params=None,
        )
        ctx = types.SimpleNamespace(
            is_cancelled=lambda: cancelled, progress=lambda **kw: None
        )
        with (
            mock.patch.object(runners._outer, "repeat", return_value=repeated_rows),
            mock.patch.object(
                runners._outer, "split_info", return_value={"n_test": 10}
            ),
        ):
            report = runners._run_outer_splits(
                engine,
                k_splits=3,
                split_seed=1,
                split_kwargs={},
                fit_kwargs={},
                headline_params={},
                metric="r2",
                first_split={},
                standalones=[],
                comparators=[],
                comparators_primary=[],
                outer_comparators="primary",
                n_effects_boot=10,
                n_perm=10,
                n_paired=10,
                stage_key="outer_splits",
                stage=lambda key, status, message="": stages.append(status),
                stage_progress=lambda *a, **k: None,
                ctx=ctx,
                label="t",
            )
        return report, stages

    @staticmethod
    def _rows(split):
        return [
            {
                "quantity": "held_out_beta",
                "exposure": "cgi",
                "estimate": 0.1,
                "ci_low": 0.0,
                "ci_high": 0.2,
                "p_value": 0.04,
                "split": split,
            }
        ]

    def test_every_split_done(self):
        report, stages = self._run(self._rows(1) + self._rows(2))
        self.assertEqual(stages[-1], DONE)
        self.assertTrue(report["complete"])

    def test_missing_splits_fail_the_stage(self):
        with mock.patch.object(runners, "_log_fusion") as log:
            report, stages = self._run([])
        self.assertEqual(stages[-1], FAILED)
        self.assertFalse(report["complete"])
        warns = [c for c in log.call_args_list if c.args[0] == "WARN"]
        self.assertTrue(any("1 of 3" in c.args[1] for c in warns))

    def test_a_cancel_is_not_a_finished_stage(self):
        with self.assertRaises(JobCancelled):
            self._run(self._rows(1), cancelled=True)


def _strained(seed):
    """Large clusters, forty of 4–11 entities saturated by a df-10 smooth, and
    six categories of one to three members, with a smooth confounder."""
    rng = np.random.default_rng(seed)
    parts = [
        rng.random(2) * 2e6 + rng.normal(size=(int(rng.integers(150, 400)), 2)) * 4e3
        for _ in range(4)
    ]
    parts += [
        rng.random(2) * 2e6 + rng.normal(size=(int(rng.integers(4, 12)), 2)) * 1.5e3
        for _ in range(40)
    ]
    xy = np.vstack(parts)
    n = len(xy)
    age = rng.uniform(45, 90, n)
    cat = rng.choice(6, n)
    cat[rng.choice(n, 12, replace=False)] = [6, 7, 8, 8, 9, 9, 10, 10, 10, 11, 11, 11]
    cov = np.column_stack([age, age**2, np.eye(12)[cat][:, 1:]])
    u = np.sin(xy[:, 0] / 4e4) + np.cos(xy[:, 1] / 5e4)
    x, other = 0.6 * u + rng.normal(size=n), 0.5 * u + rng.normal(size=n)
    target = 0.05 * x + 0.4 * u + rng.normal(size=n)
    control = 0.3 * u + rng.normal(size=n)
    smooth = sb.build_block_basis(xy, max_df=10).design(10)
    return rng, cov, smooth, x, other, target, control


class StrainedResamplingTests(unittest.TestCase):
    def test_replicate_scores_match_the_per_replicate_fits(self):
        rng, cov, smooth, x, _, y, _ = _strained(0)
        n = len(y)
        idx = rng.integers(0, n, size=(40, n))
        dense = np.column_stack([cov, smooth])
        want = np.array([osc.score("r2", y[i], x[i], dense[i]) for i in idx])
        w = rs.row_weights(idx, n)
        got = rs.incremental_r2(
            rs.BlockDesign(n, cov, smooth).cross(np.column_stack([y, x]), w),
            rs.weighted_sst(y, w),
            rs.weighted_sst(x, w),
        )
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-10)

    def test_contrast_intervals_match_the_dense_path(self):
        _, cov, smooth, x, other, target, control = _strained(1)
        dense = np.column_stack([cov, smooth])
        pairs = (
            (
                nc.specificity_contrast(
                    x, other, target, control, cov,
                    extra={"exposure": smooth, "other": smooth}, n_boot=300, seed=7,
                ),
                nc.specificity_contrast(
                    x, other, target, control, dense, n_boot=300, seed=7
                ),
            ),
            (
                nc.paired_contrast(
                    x, target, control, cov, extra={"exposure": smooth},
                    n_boot=300, seed=7,
                ),
                nc.paired_contrast(x, target, control, dense, n_boot=300, seed=7),
            ),
        )
        for blocks, plain in pairs:
            for key, value in plain.items():
                if isinstance(value, float):
                    tol = 1e-8 * max(1.0, abs(value))
                    self.assertAlmostEqual(blocks[key], value, delta=tol, msg=key)


if __name__ == "__main__":
    unittest.main()
