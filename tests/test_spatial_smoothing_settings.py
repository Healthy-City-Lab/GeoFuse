"""Smoothing settings come from the job.

The spatial smooth compares a list of candidate df (default
``0, 1, 2, 3, 4, 6, 8, 11, 15, 20``) instead of every df up to a maximum, by
AIC or BIC, with the knot and neighbour counts of the basis as settings; the
older ``spatial_adjust_max_df`` N means the list ``0 … N`` and reproduces
earlier runs. Every selection records its criterion profile, and the largest
candidate winning sets ``at_cap`` and logs a WARN. The covariate splines and
the exposure-response spline take their df from the job as well.
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import geopandas  # noqa: F401  (pyproj before torchvision, see vision.py)
import numpy as np

from geofuse import exposure_response, objective_scoring, settings
from geofuse import spatial_basis as sb
from geofuse.cli import commands as cli
from geofuse.fusion import MetricFusionEngine
from geofuse.jobs import runners


def _field(n=600, seed=0, clusters=3):
    """Points in ``clusters`` separated squares; ``y`` a smooth field + noise."""
    rng = np.random.default_rng(seed)
    parts = []
    for c in range(clusters):
        parts.append(rng.random((n // clusters, 2)) * 5_000.0 + c * 50_000.0)
    xy = np.vstack(parts)
    wave = 2 * np.pi / 4_000.0
    u = np.sin(wave * xy[:, 0]) * np.cos(wave * xy[:, 1])
    y = 2.0 * u + 0.3 * rng.normal(size=len(xy))
    return xy, y, rng


def _engine(**kw):
    return MetricFusionEngine(
        target_file="dummy.geojson",
        cache_dir=tempfile.mkdtemp(),
        spatial_adjust_method=kw.pop("spatial_adjust_method", "spatial_plus"),
        **kw,
    )


class LegacyAliasTests(unittest.TestCase):
    def test_max_df_is_the_list_zero_to_n(self):
        old = _engine(spatial_adjust_max_df=10)
        new = _engine(spatial_adjust_df=list(range(11)))
        self.assertEqual(old.spatial_adjust_df, new.spatial_adjust_df)
        xy, y, rng = _field()
        x = y + rng.normal(size=len(y))
        for method in ("spatial_plus", "ks_aic"):
            a = _engine(spatial_adjust_method=method, spatial_adjust_max_df=10)
            b = _engine(spatial_adjust_method=method, spatial_adjust_df=range(11))
            ra = a._spatial_selection(xy, y, None, x)
            rb = b._spatial_selection(xy, y, None, x)
            self.assertEqual(ra["df"], rb["df"])
            self.assertEqual(ra["ic"], rb["ic"])
            np.testing.assert_array_equal(ra["columns"], rb["columns"])

    def test_the_selection_functions_default_to_every_level(self):
        xy, y, _ = _field()
        basis = sb.build_block_basis(xy, max_df=10)
        self.assertEqual(
            sb.select_df_aic(y, basis)[0],
            sb.select_df_aic(y, basis, candidates=range(11))[0],
        )

    def test_both_forms_are_refused(self):
        with self.assertRaisesRegex(ValueError, "not both"):
            _engine(spatial_adjust_max_df=5, spatial_adjust_df=[0, 5])
        values = settings.defaults("fusion")
        values.update(spatial_adjust_max_df=5, spatial_adjust_df=[0, 5])
        with self.assertRaisesRegex(settings.SettingError, "not both"):
            settings.check("fusion", values)

    def test_engine_default_is_the_log_spaced_list(self):
        self.assertEqual(
            _engine().spatial_adjust_df, (0, 1, 2, 3, 4, 6, 8, 11, 15, 20)
        )
        self.assertEqual(sb.DEFAULT_DF_CANDIDATES, settings.SPATIAL_DF_CANDIDATES)
        self.assertEqual(sb.CRITERIA, settings.SPATIAL_CRITERIA)

    def test_schema_defaults_are_the_engines(self):
        f = settings.FUSION
        for key, value in (
            ("spatial_adjust_criterion", sb.CRITERIA[0]),
            ("spatial_adjust_neighbours", sb.DEFAULT_NEIGHBOURS),
            ("spatial_adjust_knots_min", sb.DEFAULT_KNOTS_MIN),
            ("spatial_adjust_knots_max", sb.DEFAULT_KNOTS_MAX),
            ("residualize_spline_df", objective_scoring.SPLINE_DF),
            ("residualize_spline_min_unique", objective_scoring.SPLINE_MIN_UNIQUE),
            ("exposure_response_spline_df", exposure_response.SPLINE_DF),
        ):
            self.assertEqual(f[key].default, value, key)
        unchanged = runners._non_default_options({"spatial_adjust_neighbours": 10})
        self.assertEqual(unchanged, {})

    def test_the_fingerprint_keeps_older_runs_hash(self):
        base = dict(
            buffer_meters=1000.0,
            gvi_buffer_min_m=100.0,
            gvi_buffer_max_m=1000.0,
            gvi_buffer_step_m=100.0,
            ndvi_buffer_min_m=100.0,
            ndvi_buffer_max_m=1000.0,
            ndvi_buffer_step_m=100.0,
            cgi_formula="weighted_average",
            covariate_columns=[],
            whole_grid_scaling=True,
            cgi_grid_spacing_m=50,
            area_balanced_split=False,
            test_size=0.25,
            spatial_adjust_method="ks_aic",
        )
        fp = runners._fusion_config_fingerprint
        self.assertEqual(
            fp(**base, spatial_adjust_max_df=10),
            fp(**base, spatial_adjust_df=list(range(11))),
        )
        self.assertNotEqual(fp(**base, spatial_adjust_max_df=10), fp(**base))
        self.assertNotEqual(
            fp(**base), fp(**base, spatial_options={"spatial_adjust_criterion": "bic"})
        )


class CandidateListTests(unittest.TestCase):
    def setUp(self):
        self.xy, self.y, _ = _field()
        self.basis = sb.build_block_basis(self.xy, max_df=20)

    def test_only_listed_df_are_compared(self):
        for pick in (sb.select_df_aic, None):
            if pick is None:
                pre = sb.precompute_df_selection(self.basis)
                df, _, ic = sb.select_df_aic_fast(
                    self.y,
                    self.basis,
                    pre,
                    candidates=[0, 3, 7],
                    return_profile=True,
                )
            else:
                df, _, ic = pick(
                    self.y, self.basis, candidates=[0, 3, 7], return_profile=True
                )
            self.assertEqual(list(ic), ["none", "0", "3", "7"])
            self.assertIn(df, (None, 0, 3, 7))

    def test_fast_and_exact_selection_agree_on_a_list(self):
        pre = sb.precompute_df_selection(self.basis)
        cands = list(settings.SPATIAL_DF_CANDIDATES)
        exact = sb.select_df_aic(
            self.y, self.basis, candidates=cands, return_profile=True
        )
        fast = sb.select_df_aic_fast(
            self.y, self.basis, pre, candidates=cands, return_profile=True
        )
        self.assertEqual(exact[0], fast[0])
        for k in exact[2]:
            self.assertAlmostEqual(exact[2][k], fast[2][k], places=6)

    def test_a_df_above_the_basis_is_capped(self):
        small = sb.build_block_basis(self.xy, max_df=4)
        self.assertEqual(sb.candidate_dfs(small, [0, 2, 8, 20]), [0, 2, 4])

    def test_bic_picks_no_more_than_aic(self):
        for seed in range(5):
            xy, y, _ = _field(seed=seed)
            basis = sb.build_block_basis(xy, max_df=20)
            cands = settings.SPATIAL_DF_CANDIDATES
            aic = sb.select_df_aic(y, basis, candidates=cands, criterion="aic")[0]
            bic = sb.select_df_aic(y, basis, candidates=cands, criterion="bic")[0]
            self.assertLessEqual(-1 if bic is None else bic, -1 if aic is None else aic)

    def test_an_unknown_criterion_is_refused(self):
        with self.assertRaisesRegex(ValueError, "criterion"):
            sb.select_df_aic(self.y, self.basis, criterion="cv")


class BasisSettingsTests(unittest.TestCase):
    def test_neighbours_reach_the_component_graph(self):
        rng = np.random.default_rng(3)
        # Pairs of points far apart: 1 neighbour cuts them into pairs.
        centres = rng.random((40, 2)) * 1e6
        xy = np.vstack([centres, centres + 1.0])
        few = sb.build_block_basis(xy, neighbours=1)
        many = sb.build_block_basis(xy, neighbours=30)
        self.assertGreater(few.summary["n_components"], many.summary["n_components"])
        self.assertEqual(few.summary["neighbours"], 1)

    def test_knot_settings_reach_each_component(self):
        xy, _, _ = _field(n=900, clusters=1)
        lo = sb.build_block_basis(xy, max_df=20, knots_min=10, knots_max=12)
        hi = sb.build_block_basis(xy, max_df=20, knots_min=60, knots_max=150)
        # A component holds K − 3 radial functions for K knots (the thin-plate
        # constraint leaves the three linear directions to the linear terms).
        self.assertEqual(lo.max_df, 9)
        self.assertEqual(hi.max_df, 20)

    def test_the_engine_passes_them_on(self):
        eng = _engine(
            spatial_adjust_neighbours=1,
            spatial_adjust_knots_min=10,
            spatial_adjust_knots_max=12,
        )
        rng = np.random.default_rng(4)
        centres = rng.random((40, 2)) * 1e6
        xy = np.vstack([centres, centres + 1.0, centres + 2.0, centres + 3.0])
        y = rng.normal(size=len(xy))
        with mock.patch.object(
            sb, "build_block_basis", wraps=sb.build_block_basis
        ) as built:
            eng._spatial_selection(xy, y, None, y)
        kw = built.call_args.kwargs
        self.assertEqual(
            (kw["neighbours"], kw["knots_min"], kw["knots_max"], kw["max_df"]),
            (1, 10, 12, 20),
        )


class AtCapTests(unittest.TestCase):
    def test_the_largest_candidate_winning_is_flagged_and_warned(self):
        xy, y, _ = _field(n=900)
        eng = _engine(spatial_adjust_df=[0, 20])
        with mock.patch("geofuse.fusion._log") as log:
            chosen = eng._spatial_selection(xy, y, None, y)
            eng._spatial_selection(xy, y, None, y)
        self.assertEqual(chosen["df"], 20)
        self.assertTrue(chosen["at_cap"])
        warns = [c for c in log.call_args_list if c.args[0] == "WARN"]
        self.assertEqual(len(warns), 1)
        self.assertIn("largest candidate", warns[0].args[1])
        report = eng.spatial_selection_report()
        self.assertEqual(report["candidates"], [0, 20])
        self.assertEqual((report["selections"], report["selections_at_cap"]), (2, 2))
        self.assertTrue(report["at_cap"])

    def test_no_flag_below_the_cap(self):
        rng = np.random.default_rng(5)
        xy = rng.random((400, 2)) * 5_000.0
        y = rng.normal(size=400)
        eng = _engine()
        with mock.patch("geofuse.fusion._log") as log:
            chosen = eng._spatial_selection(xy, y, None, y)
        self.assertFalse(chosen["at_cap"])
        self.assertFalse([c for c in log.call_args_list if c.args[0] == "WARN"])

    def test_reporting_records_carry_the_profile(self):
        xy, y, _ = _field()
        eng = _engine()
        rows = np.ones(len(y), dtype=bool)
        _cols, record = eng._reporting_smooth(xy, rows, y, None, y)
        self.assertEqual(set(record), {"method", "df", "at_cap", "ic"})
        self.assertEqual(list(record["ic"])[0], "none")
        from geofuse.jobs.outer_splits import spatial_columns

        flat = spatial_columns(record)
        self.assertNotIn("spatial_ic", flat)
        self.assertIn("spatial_at_cap", flat)


class SplineSettingsTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(6)
        self.X = np.column_stack([rng.normal(size=300), rng.integers(0, 5, 300)])

    def test_covariate_spline_width_follows_the_df(self):
        default = objective_scoring._expand_covariate_basis(self.X, "spline")
        same = objective_scoring._expand_covariate_basis(
            self.X, "spline", spline_df=4, spline_min_unique=7
        )
        np.testing.assert_array_equal(default, same)
        wide = objective_scoring._expand_covariate_basis(self.X, "spline", spline_df=6)
        self.assertEqual(default.shape[1], 4 + 1)
        self.assertEqual(wide.shape[1], 6 + 1)
        # Five distinct codes: a spline once the threshold drops to 5.
        coded = objective_scoring._expand_covariate_basis(
            self.X, "spline", spline_min_unique=5
        )
        self.assertEqual(coded.shape[1], 4 + 4)

    def test_the_score_uses_them(self):
        rng = np.random.default_rng(7)
        t = rng.normal(size=300)
        c = rng.normal(size=300) + 0.2 * t
        kw = dict(residualize_method="spline")
        a = objective_scoring.score("r2", t, c, self.X, **kw)
        b = objective_scoring.score("r2", t, c, self.X, **kw, spline_df=4)
        d = objective_scoring.score("r2", t, c, self.X, **kw, spline_df=8)
        self.assertEqual(a, b)
        self.assertNotEqual(a, d)

    def test_the_engine_carries_them(self):
        eng = _engine(
            residualize_method="spline",
            residualize_spline_df=6,
            residualize_spline_min_unique=3,
            exposure_response_spline_df=5,
        )
        self.assertEqual(
            eng._residualize_options,
            {"residualize_method": "spline", "spline_df": 6, "spline_min_unique": 3},
        )
        self.assertEqual(eng.exposure_response_spline_df, 5)
        self.assertEqual(exposure_response.SPLINE_DF, 4)


class SettingsRoundTripTests(unittest.TestCase):
    def test_the_df_list_reads_text_and_lists(self):
        check = settings.FUSION["spatial_adjust_df"].check
        self.assertEqual(check("k", "20, 0,3, 3"), [0, 3, 20])
        self.assertEqual(check("k", [4, 1]), [1, 4])
        for bad in ("a, b", [], [-1], [51]):
            with self.assertRaises(settings.SettingError):
                check("k", bad)

    def test_knots_bound_the_df(self):
        with self.assertRaisesRegex(settings.SettingError, "knots_max"):
            settings.check(
                "fusion", {"spatial_adjust_df": [0, 30], "spatial_adjust_knots_max": 20}
            )
        with self.assertRaisesRegex(settings.SettingError, "knots_min"):
            settings.check(
                "fusion",
                {"spatial_adjust_knots_min": 50, "spatial_adjust_knots_max": 40},
            )

    def test_cli_flags(self):
        args = cli.build_parser().parse_args(
            [
                "plan",
                "r.toml",
                "--spatial-adjust-df",
                "0",
                "4",
                "12",
                "--spatial-adjust-criterion",
                "bic",
                "--spatial-paired-smooth",
                "cgi",
                "--residualize-spline-df",
                "5",
            ]
        )
        overrides = {o.key: o.value for o in cli._overrides(args)}
        self.assertEqual(overrides["spatial_adjust_df"], [0, 4, 12])
        self.assertEqual(overrides["spatial_adjust_criterion"], "bic")
        self.assertEqual(overrides["spatial_paired_smooth"], "cgi")
        self.assertEqual(overrides["residualize_spline_df"], 5)

    def test_the_older_maximum_resolves_to_a_list(self):
        values = settings.check("fusion", {"spatial_adjust_max_df": 3})
        self.assertEqual(settings.spatial_df_candidates(values), [0, 1, 2, 3])
        self.assertEqual(
            settings.spatial_df_candidates(settings.defaults("fusion")),
            list(settings.SPATIAL_DF_CANDIDATES),
        )


if __name__ == "__main__":
    unittest.main()
