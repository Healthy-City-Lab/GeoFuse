"""A spatial smooth defined by a distance (``spatial_adjust_scale_m``).

Pinned here:

- clusters: a sparse chain joins two tight clusters into one k-NN cluster,
  which the df mode reports as chained (and warns about); the gap rule of the
  scale mode splits them, and every cluster below the smallest size joins its
  nearest larger one, so no entity sits alone under its own intercept;
- knots: two clusters of one density and extents L and 3L get knots in a
  ratio near 9; a planted field four scales long is absorbed, one half a
  scale long mostly not;
- one shared smooth: the slope is an OLS on the covariates and that smooth,
  ``spatial_plus`` and ``ks_aic`` and every paired rule give the same numbers,
  and a paired difference is the difference of the two printed scores;
- the ladder: another scale's rows equal a run at that scale with the primary
  scale's clusters, the ``none`` rows an unadjusted run, and the summary
  groups per scale with only the primary scale in the Holm families;
- settings, the engine's checks and the run's scale list.
"""

import os
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (os.path.dirname(os.path.abspath(__file__)), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import geopandas  # noqa: F401  (pyproj before torchvision, see vision.py)
import numpy as np

import test_paired_smooth as paired_fixture
import test_spatial_reporting as reporting_fixture
from geofuse import settings
from geofuse import spatial_basis as sb
from geofuse.fusion import MetricFusionEngine, _absorbed_share
from geofuse.jobs import outer_splits
from geofuse.jobs.runners import _spatial_scale_list


def _square(rng, n, side, origin=(0.0, 0.0)):
    return rng.random((n, 2)) * side + np.asarray(origin)


def _scaled(eng, scales, eps=None, min_component=sb.DEFAULT_MIN_COMPONENT):
    """A stand-in engine of the reporting fixtures, put in scale mode."""
    eng.spatial_adjust_scales = tuple(float(s) for s in scales)
    eng._spatial_scale = eng.spatial_adjust_scales[0]
    eng.spatial_adjust_min_component = min_component
    eng._spatial_scale_summaries = {}
    eng._scale_eps = 2.0 * eng.spatial_adjust_scales[0] if eps is None else eps
    for name in ("_scale_selection", "spatial_scale"):
        setattr(eng, name, types.MethodType(getattr(MetricFusionEngine, name), eng))
    return eng


# ────────────────────────────────────────────────────────────────────
# Clusters and knots
# ────────────────────────────────────────────────────────────────────


def _chained(rng):
    """Two tight clusters 150 km apart joined by points every 15 km, and three
    separate small towns."""
    a = _square(rng, 400, 2_000.0)
    b = _square(rng, 400, 2_000.0, (150_000.0, 0.0))
    chain = np.column_stack([np.arange(15_000.0, 150_000.0, 15_000.0), np.zeros(9)])
    towns = [_square(rng, 300, 2_000.0, (0.0, y)) for y in (400e3, 800e3, 1200e3)]
    return np.vstack([a, b, chain, *towns])


class ClusterTests(unittest.TestCase):
    def test_a_chain_is_reported_in_df_mode_and_split_by_the_gap(self):
        xy = _chained(np.random.default_rng(1))
        df_mode = sb.build_block_basis(xy, max_df=5)
        self.assertIsNotNone(df_mode.summary["chained"])
        sizes = [c["n"] for c in df_mode.summary["components"]]
        self.assertIn(809, sizes)

        scaled = sb.build_block_basis(xy, scale_m=5_000.0, min_component=30)
        sizes = sorted(c["n"] for c in scaled.summary["components"])
        self.assertEqual(sizes, [300, 300, 300, 400, 400])
        self.assertEqual(scaled.summary["merged_components"], 9)
        self.assertEqual(scaled.summary["merged_entities"], 9)
        self.assertEqual(scaled.summary["eps_used"], 10_000.0)

    def test_df_mode_warns_once_about_a_chain(self):
        xy = _chained(np.random.default_rng(1))
        y = np.random.default_rng(2).normal(size=len(xy))
        eng = MetricFusionEngine(
            target_file="t.geojson",
            cache_dir=tempfile.mkdtemp(),
            spatial_adjust_method="ks_aic",
            spatial_adjust_df=[0, 2],
        )
        with mock.patch("geofuse.fusion._log") as log:
            eng._spatial_selection(xy, y, None, y)
            eng._spatial_selection(xy[:-1], y[:-1], None, y[:-1])
        warns = [c.args[1] for c in log.call_args_list if c.args[0] == "WARN"]
        chained = [w for w in warns if "spatial_adjust_scale_m" in w]
        self.assertEqual(len(chained), 1)

    def test_small_clusters_join_a_larger_one(self):
        rng = np.random.default_rng(3)
        xy = np.vstack(
            [
                _square(rng, 200, 3_000.0),
                _square(rng, 12, 500.0, (20_000.0, 0.0)),
                _square(rng, 1, 1.0, (0.0, 30_000.0)),
            ]
        )
        basis = sb.build_block_basis(xy, scale_m=2_000.0, min_component=30)
        self.assertEqual(basis.summary["n_components"], 1)
        self.assertEqual(basis.summary["merged_entities"], 13)
        # One cluster: no intercept column besides the global one.
        intercepts = [
            j
            for j in range(basis.linear.shape[1])
            if set(np.unique(basis.linear[:, j])) <= {0.0, 1.0}
        ]
        self.assertEqual(intercepts, [])
        # Merged entities carry the cluster's level only: zero trend and radial.
        moved = np.arange(200, 213)
        self.assertTrue(np.all(basis.linear[moved] == 0.0))
        self.assertTrue(np.all(basis.radial[moved] == 0.0))

    def test_knots_grow_with_the_area(self):
        rng = np.random.default_rng(4)
        small = _square(rng, 400, 10_000.0)
        large = _square(rng, 3_600, 30_000.0, (200_000.0, 0.0))
        basis = sb.build_block_basis(
            np.vstack([small, large]), scale_m=2_000.0, min_component=30
        )
        k_small, k_large = (c["knots"] for c in basis.summary["components"])
        self.assertGreater(k_small, 10)
        self.assertLess(abs(k_large / k_small / 9.0 - 1.0), 0.2)

    def test_a_field_coarser_than_the_scale_is_absorbed(self):
        rng = np.random.default_rng(5)
        xy = _square(rng, 3_000, 40_000.0)
        scale = 4_000.0
        smooth = sb.build_block_basis(xy, scale_m=scale, min_component=30).design(1)
        for wavelength, low, high in ((4 * scale, 0.9, 1.0), (scale / 2, 0.0, 0.5)):
            field = np.sin(2 * np.pi * xy[:, 0] / wavelength + 0.3) + np.cos(
                2 * np.pi * xy[:, 1] / wavelength
            )
            with self.subTest(wavelength=wavelength):
                share = _absorbed_share(field, smooth)
                self.assertGreaterEqual(share, low)
                self.assertLessEqual(share, high)

    def test_a_large_smooth_is_flagged(self):
        rng = np.random.default_rng(6)
        xy = _square(rng, 3_000, 40_000.0)
        with mock.patch.object(sb, "SCALE_WARN_BLOCK", 20):
            basis = sb.build_block_basis(xy, scale_m=4_000.0)
        self.assertIn("4 km", basis.summary["size_warning"])


# ────────────────────────────────────────────────────────────────────
# One shared smooth
# ────────────────────────────────────────────────────────────────────


def _reporting(method="spatial_plus", scales=(3_000.0,), eps=None, **kw):
    eng = reporting_fixture._engine(method, **kw)
    return _scaled(eng, scales, eps=eps)


class SharedSmoothTests(unittest.TestCase):
    def test_the_slope_is_an_ols_on_the_covariates_and_the_smooth(self):
        eng = _reporting()
        got = eng.compute_partial_slopes({})["test"]
        d, t = eng.data, eng.test
        smooth = sb.build_block_basis(
            d["xy"][t], eps=6_000.0, scale_m=3_000.0, min_component=30
        ).design(1)
        controls = np.column_stack([d["age"][t], smooth])
        want = reporting_fixture._ols_beta(d["x"][t], d["y"][t], controls)
        self.assertAlmostEqual(got["beta"], want, delta=1e-8)
        record = got["spatial_adjustment"]
        self.assertEqual(record["scale_m"], 3_000.0)
        self.assertEqual(record["df"], smooth.shape[1])
        self.assertAlmostEqual(
            record["absorbed"], _absorbed_share(d["x"][t], smooth), places=6
        )

    def test_spatial_plus_and_ks_aic_give_the_same_numbers(self):
        plus, ks = _reporting("spatial_plus"), _reporting("ks_aic")
        exposures = {"bench": ("comparator:bench", {})}
        pairs = [
            (e.compute_partial_slopes({})["test"]["beta"] for e in (plus, ks)),
            (
                e.compute_negative_controls({}, n_boot=30)["splits"]["test"][
                    "controls"
                ]["grip"]["beta"]
                for e in (plus, ks)
            ),
        ]
        for a, b in pairs:
            self.assertAlmostEqual(a, b, places=12)
        rows = [
            e.compute_specificity_contrast({}, exposures, n_boot=30, seed=1)["rows"][0]
            for e in (plus, ks)
        ]
        for key in ("delta_d", "ci_low", "ci_high", "p_value"):
            self.assertAlmostEqual(rows[0][key], rows[1][key], places=12, msg=key)

    def test_every_paired_rule_gives_the_difference_of_the_printed_scores(self):
        outs = []
        for method, rule in (
            ("spatial_plus", "own"),
            ("spatial_plus", "cgi"),
            ("spatial_plus", "max"),
            ("spatial_plus", "outcome"),
            ("ks_aic", "own"),
        ):
            eng = _scaled(paired_fixture._engine(method, rule), (3_000.0,))
            out = paired_fixture._paired(eng)
            self.assertEqual(out["spatial_adjustment"]["rule"], "shared")
            d = eng.data
            cgi = paired_fixture._held_out_score(eng, d["cgi"])
            other = paired_fixture._held_out_score(eng, d["other"])
            self.assertAlmostEqual(out["observed_diff"], cgi - other, places=10)
            outs.append(out)
        for out in outs[1:]:
            for key in ("observed_diff", "lower", "upper", "p_value"):
                self.assertAlmostEqual(out[key], outs[0][key], places=10, msg=key)


# ────────────────────────────────────────────────────────────────────
# The scale ladder
# ────────────────────────────────────────────────────────────────────


def _by_key(rows):
    return {(r["quantity"], r["exposure"]): r for r in rows}


class LadderTests(unittest.TestCase):
    def test_each_scale_equals_a_run_at_that_scale(self):
        eng = _reporting(scales=(3_000.0, 6_000.0))
        rows = outer_splits.scale_rows(eng, {}, [6_000.0, None], n_boot=20)
        at = {
            s: _by_key(r for r in rows if r["spatial_scale_m"] == s)
            for s in ("6000", "none")
        }
        alone = _reporting(scales=(6_000.0,), eps=6_000.0)
        plain = reporting_fixture._engine("none")
        for text, other in (("6000", alone), ("none", plain)):
            want = _by_key(
                outer_splits.split_rows(
                    effects_test=None,
                    slope_test=other.compute_partial_slopes({})["test"],
                    channel_paired=None,
                    comparator_paired=[],
                    negative_controls=other.compute_negative_controls({}, n_boot=20),
                )
            )
            self.assertEqual(set(at[text]), set(want))
            for key, row in want.items():
                for field in ("estimate", "ci_low", "ci_high", "p_value"):
                    self.assertAlmostEqual(
                        at[text][key][field], row[field], delta=1e-10, msg=key
                    )
        # The primary scale is back in place afterwards.
        self.assertEqual(eng._spatial_scale, 3_000.0)
        self.assertEqual(eng.spatial_adjust_method, "spatial_plus")

    def test_the_summary_groups_per_scale_and_holm_takes_the_primary(self):
        rows = []
        for k in range(4):
            for scale, shift in (("3000", 0.0), ("6000", 0.1), ("none", 0.2)):
                for name in ("a", "b"):
                    rows.append(
                        {
                            "quantity": "cgi_minus_comparator"
                            if scale == "3000"
                            else "held_out_beta",
                            "exposure": name,
                            "estimate": 0.1 + shift + 0.01 * k,
                            "ci_low": None,
                            "ci_high": None,
                            "p_value": 0.01,
                            "spatial_scale_m": scale,
                            "split": k,
                            "test_share": 0.2,
                        }
                    )
        summary = outer_splits.summarise(rows, primary_scale=3_000.0)
        beta = [e for e in summary if e["quantity"] == "held_out_beta"]
        self.assertEqual(
            [(e["exposure"], e["spatial_scale_m"]) for e in beta],
            [("a", "6000"), ("a", "none"), ("b", "6000"), ("b", "none")],
        )
        primary = [e for e in summary if e["quantity"] == "cgi_minus_comparator"]
        self.assertTrue(all(e["corrected_p_holm"] is not None for e in primary))

    def test_the_label_names_the_scale(self):
        record = {"method": "ks_aic", "df": 40, "scale_m": 5_000.0}
        self.assertEqual(outer_splits.spatial_label(record), "ks_aic at 5 km")
        self.assertEqual(outer_splits.spatial_columns(record)["spatial_scale_m"], "5000")


# ────────────────────────────────────────────────────────────────────
# Settings and checks
# ────────────────────────────────────────────────────────────────────


class SettingsTests(unittest.TestCase):
    def test_scales_keep_their_order(self):
        s = settings.SCHEMAS["fusion"]["spatial_adjust_scale_m"]
        self.assertEqual(s.check("x", "5000, 20000, 3000"), [5000, 20000, 3000])
        self.assertIsNone(settings.defaults("fusion")["spatial_adjust_scale_m"])
        self.assertIsNone(settings.defaults("fusion")["spatial_adjust_min_component"])
        self.assertEqual(_spatial_scale_list([5000, 3000, 5000]), [5000.0, 3000.0])
        self.assertEqual(_spatial_scale_list(None), [])

    def test_the_engine_checks_and_defaults(self):
        def engine(**kw):
            return MetricFusionEngine(
                target_file="t.geojson", cache_dir=tempfile.mkdtemp(), **kw
            )

        with self.assertRaises(ValueError):
            engine(spatial_adjust_scale_m=5000)
        eng = engine(spatial_adjust_method="ks_aic", spatial_adjust_scale_m=[5e3, 2e4])
        self.assertEqual(eng.spatial_adjust_scales, (5e3, 2e4))
        self.assertEqual(eng.spatial_adjust_min_component, 30)
        self.assertEqual(eng._scale_eps, 10_000.0)
        report = eng.spatial_selection_report()
        self.assertEqual(report["selection"], "fixed scale")
        self.assertEqual(report["paired_rule"], "shared")
        self.assertIsNone(engine(spatial_adjust_method="ks_aic").spatial_adjust_min_component)


if __name__ == "__main__":
    unittest.main()
