"""Spatial adjustment reaches the slopes, negative controls and specificity.

With ``spatial_adjust_method`` other than ``none`` the reported slopes adjust
for the same kind of coordinate smooth the scores do. Pinned here:

- ``extra`` columns in the paired statistics: an exposure's join both of its
  slopes, an outcome's only the slopes on it, and unit weights reproduce
  :func:`negative_controls.partial_slope` on each slope's own design;
- a smooth confounder driving exposure, target and control (no effect): under
  ``spatial_plus`` the held-out slope and the control slope shrink towards 0,
  and the slope equals an OLS on the covariates plus the same smooth columns;
- a true within-area effect survives the adjustment;
- with unit weights the specificity contrast's four slopes are the adjusted
  partial slopes (each exposure's own smooth under ``spatial_plus``, each
  outcome's under ``ks_aic``), and every slope and row records its smooth;
- ``none`` adds no record anywhere;
- the records become text columns on the slope rows; in a spatially adjusted
  run with two held-out splits every split's slope rows carry them, the
  corrected t recomputes from those rows, and every table names the smooth,
  while an unadjusted run has no such column;
- the reported direction conditions on the score's smooth;
- ``geofuse results`` names the adjustment and each row's smooth.
"""

import csv
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (os.path.dirname(os.path.abspath(__file__)), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import geopandas as gpd  # noqa: F401
import numpy as np
import pandas as pd

from geofuse import negative_controls as nc
from geofuse.jobs import outer_splits
from geofuse.statistical_testing import corrected_resampled_t

# ────────────────────────────────────────────────────────────────────
# The statistics' extra columns
# ────────────────────────────────────────────────────────────────────


def _data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    x, other = rng.normal(size=n), rng.normal(size=n)
    target = 0.3 * x + rng.normal(size=n)
    control = 0.1 * x + rng.normal(size=n)
    cov = rng.normal(size=(n, 2))
    blocks = {k: rng.normal(size=(n, 3)) for k in ("a", "b", "t", "c")}
    return x, other, target, control, cov, blocks


def _beta(x, y, *design):
    return nc.partial_slope(x, y, np.column_stack(design))["beta"]


class TestExtraColumns(unittest.TestCase):
    def test_an_exposures_columns_are_the_covariates_of_both_slopes(self):
        x, _, target, control, cov, b = _data()
        got = nc.paired_contrast(
            x, target, control, cov, extra={"exposure": b["a"]}, n_boot=50, seed=3
        )
        want = nc.paired_contrast(
            x, target, control, np.column_stack([cov, b["a"]]), n_boot=50, seed=3
        )
        self.assertEqual(set(got), set(want))
        for key, value in want.items():
            if isinstance(value, float):
                self.assertAlmostEqual(got[key], value, places=10, msg=key)
            else:
                self.assertEqual(got[key], value, key)

    def test_an_outcomes_columns_join_only_its_own_slope(self):
        x, _, target, control, cov, b = _data()
        got = nc.paired_contrast(
            x, target, control, cov, extra={"target": b["t"], "control": b["c"]}
        )
        d = abs(_beta(x, target, cov, b["t"])) - abs(_beta(x, control, cov, b["c"]))
        self.assertAlmostEqual(got["delta"], d, places=10)

    def test_specificity_slopes_use_each_slopes_own_design(self):
        x, other, target, control, cov, b = _data()
        cases = {
            "by exposure": (
                {"exposure": b["a"], "other": b["b"]},
                [(x, target, b["a"]), (x, control, b["a"])]
                + [(other, target, b["b"]), (other, control, b["b"])],
            ),
            "by outcome": (
                {"target": b["t"], "control": b["c"]},
                [(x, target, b["t"]), (x, control, b["c"])]
                + [(other, target, b["t"]), (other, control, b["c"])],
            ),
        }
        keys = ("beta_target", "beta_control", "beta_target_other")
        keys += ("beta_control_other",)
        for name, (extra, slopes) in cases.items():
            with self.subTest(name):
                got = nc.specificity_contrast(
                    x, other, target, control, cov, extra=extra, n_boot=0
                )
                for key, (e, y, block) in zip(keys, slopes, strict=True):
                    self.assertAlmostEqual(got[key], _beta(e, y, cov, block), places=10)

    def test_the_same_exposure_and_columns_differ_by_zero_on_every_resample(self):
        x, _, target, control, cov, b = _data(n=300)
        got = nc.specificity_contrast(
            x,
            x.copy(),
            target,
            control,
            cov,
            extra={"exposure": b["a"], "other": b["a"].copy()},
            n_boot=200,
            seed=1,
        )
        self.assertEqual((got["delta"], got["ci_low"], got["ci_high"]), (0, 0, 0))

    def test_transfer_test_hands_each_slope_its_columns(self):
        x, _, target, control, cov, b = _data()
        out = nc.transfer_test(
            x,
            target,
            {"grip": control},
            cov,
            extra={"target": b["t"], "controls": {"grip": b["c"]}},
            n_boot=0,
        )
        self.assertAlmostEqual(
            out["target"]["beta"], _beta(x, target, cov, b["t"]), places=12
        )
        self.assertAlmostEqual(
            out["controls"]["grip"]["beta"], _beta(x, control, cov, b["c"]), places=12
        )

    def test_a_misspelt_role_raises(self):
        x, _, target, control, cov, b = _data(n=50)
        with self.assertRaises(ValueError):
            nc.paired_contrast(x, target, control, cov, extra={"outcome": b["t"]})
        with self.assertRaises(ValueError):
            nc.transfer_test(x, target, {"g": control}, cov, extra={"other": b["t"]})


# ────────────────────────────────────────────────────────────────────
# The engine's reporting methods over a smooth confounder
# ────────────────────────────────────────────────────────────────────

_METHODS = (
    "_reporting_covariate_matrix",
    "_reporting_raw_column",
    "_reporting_clusters",
    "_test_entity_mask",
    "_whole_data_coords",
    "_spatial_smooth",
    "_spatial_selection",
    "_note_basis_summary",
    "_spatial_candidates",
    "_selection_record",
    "_count_selection",
    "_reporting_smooth",
    "_slope_smooths",
    "compute_partial_slopes",
    "compute_negative_controls",
    "compute_specificity_contrast",
)


def _field(n_ent, rng, effect=0.0):
    """Entities on a 10 km square; ``u`` a smooth field (12 km waves, unit SD)
    driving the exposure, the target and the control; ``g`` the exposure's
    own part, on which the target loads by ``effect``."""
    xy = rng.random((n_ent, 2)) * 10_000.0
    wave = 2 * np.pi / 12_000.0
    u = np.sin(wave * xy[:, 0]) + np.cos(wave * xy[:, 1])
    u = (u - u.mean()) / u.std()
    g = rng.normal(size=n_ent)
    exposure = u + 0.7 * g
    target = effect * g + u + 0.5 * rng.normal(size=n_ent)
    control = u + 0.5 * rng.normal(size=n_ent)
    bench = u + 0.2 * g + 0.5 * rng.normal(size=n_ent)
    return xy, exposure, target, control, bench


def _engine(method="spatial_plus", n_ent=600, n_test=300, seed=11, effect=0.0):
    """The engine's reporting methods over entities whose frames the test sets:
    the CGI, a comparator ``bench`` (frame in another order) and one control."""
    from geofuse.fusion import MetricFusionEngine as E
    from geofuse.fusion import comparator_mode

    rng = np.random.default_rng(seed)
    xy, exposure, target, control, bench = _field(n_ent, rng, effect)
    age = rng.normal(size=n_ent)
    pid = np.repeat(np.arange(n_ent), 2)
    full = pd.DataFrame(
        {
            "polygon_id": pid,
            "target": target[pid],
            "age": age[pid],
            "grip": control[pid],
            "_cx": xy[pid, 0],
            "_cy": xy[pid, 1],
        }
    )
    ids = np.arange(n_ent)

    def frame(values, order=ids):
        return pd.DataFrame(
            {
                "polygon_id": ids[order],
                "target": target[order],
                "composite": values[order],
            }
        )

    frames = {
        "cgi": frame(exposure),
        comparator_mode("bench"): frame(bench, rng.permutation(n_ent)),
    }
    eng = types.SimpleNamespace(
        negative_control_columns=["grip"],
        is_panel=False,
        covariate_columns=["age"],
        _covariate_dummy_map={},
        test_data=full[full["polygon_id"] < n_test],
        _active_greenery_channel="cgi",
        spatial_adjust_method=method,
        spatial_adjust_max_df=10,
        spatial_adjust_df=tuple(range(11)),
        spatial_adjust_criterion="aic",
        spatial_adjust_neighbours=10,
        spatial_adjust_knots_min=35,
        spatial_adjust_knots_max=150,
        spatial_adjust_eps_m=None,
        spatial_adjust_min_component=None,
        _spatial_scale=None,
        _spatial_basis_cache={},
        _spatial_adjust_summary=None,
        _spatial_run_summary=None,
        _spatial_selections={"n": 0, "at_cap": 0},
    )
    eng.apply_fusion = lambda weights=None: frames[eng._active_greenery_channel]
    eng._full_data_frame = lambda: full
    for name in _METHODS:
        setattr(eng, name, types.MethodType(getattr(E, name), eng))
    eng._finite_rows = E._finite_rows
    eng._with_smooth = E._with_smooth
    eng.data = {"xy": xy, "x": exposure, "y": target, "c": control, "age": age}
    eng.data["bench"] = bench
    eng.test = np.arange(n_ent) < n_test
    return eng


def _ols_beta(x, y, controls):
    """SD-unit slope of ``y`` on ``x`` from an OLS on ``[1, x, controls]``."""
    design = np.column_stack([np.ones(len(x)), x, controls])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    rest = np.column_stack([np.ones(len(x)), controls])

    def resid(v):
        return v - rest @ np.linalg.lstsq(rest, v, rcond=None)[0]

    return coef[1] * resid(x).std() / resid(y).std()


def _held_out_beta(eng) -> float:
    return eng.compute_partial_slopes({})["test"]["beta"]


def _control_beta(eng) -> float:
    report = eng.compute_negative_controls({}, n_boot=0)
    return report["splits"]["test"]["controls"]["grip"]["beta"]


class TestSmoothConfounder(unittest.TestCase):
    def test_the_slopes_shrink_towards_zero(self):
        plain = _engine("none")
        adjusted = _engine("spatial_plus")
        for name, slope in (("held-out", _held_out_beta), ("control", _control_beta)):
            with self.subTest(name):
                before, after = slope(plain), slope(adjusted)
                self.assertGreater(before, 0.5)
                self.assertLess(abs(after), 0.25 * before)

    def test_the_slope_is_an_ols_on_the_covariates_and_the_same_smooth(self):
        for method in ("spatial_plus", "ks_aic"):
            with self.subTest(method):
                eng = _engine(method)
                got = eng.compute_partial_slopes({})["test"]
                d, t = eng.data, eng.test
                cov = d["age"].reshape(-1, 1)
                smooth, record = eng._reporting_smooth(d["xy"], t, d["y"], cov, d["x"])
                self.assertIsNotNone(smooth)
                controls = np.column_stack([cov[t], smooth])
                want = _ols_beta(d["x"][t], d["y"][t], controls)
                self.assertAlmostEqual(got["beta"], want, delta=1e-8)
                self.assertEqual(got["spatial_adjustment"], record)
                self.assertEqual(record["method"], method)
                self.assertIsInstance(record["df"], int)

    def test_a_true_within_area_effect_survives(self):
        got = _engine("spatial_plus", effect=0.3).compute_partial_slopes({})["test"]
        self.assertGreater(got["ci_low"], 0.0)

    def test_every_split_and_every_outcome_records_its_smooth(self):
        eng = _engine("ks_aic")
        slopes = eng.compute_partial_slopes({})
        self.assertEqual(set(slopes), {"test", "train_val", "all"})
        for entry in slopes.values():
            self.assertEqual(entry["spatial_adjustment"]["method"], "ks_aic")
        report = eng.compute_negative_controls({}, metric="r2", n_boot=50)
        for res in report["splits"].values():
            self.assertIn("spatial_adjustment", res["target"])
            self.assertIn("spatial_adjustment", res["controls"]["grip"])
            self.assertIsNotNone(res["controls"]["grip"]["objective"])

    def test_without_adjustment_nothing_is_recorded(self):
        eng = _engine("none")
        self.assertNotIn("spatial_adjustment", eng.compute_partial_slopes({})["test"])
        report = eng.compute_negative_controls({}, n_boot=0)
        self.assertNotIn("spatial_adjustment", report["splits"]["test"]["target"])
        out = eng.compute_specificity_contrast(
            {}, {"bench": ("comparator:bench", {})}, n_boot=0
        )
        self.assertNotIn("spatial_adjustment", out["rows"][0])
        self.assertNotIn("spatial_adjustment", out["slopes"]["bench"]["target"])


class TestSpecificityAdjusted(unittest.TestCase):
    exposures = {"bench": ("comparator:bench", {})}

    def test_spatial_plus_gives_each_exposure_its_own_smooth(self):
        eng = _engine("spatial_plus")
        out = eng.compute_specificity_contrast({}, self.exposures, n_boot=0)
        (row,) = out["rows"]
        cgi = eng.compute_partial_slopes({})["test"]
        own = out["slopes"]["bench"]
        self.assertAlmostEqual(row["beta_target_cgi"], cgi["beta"], places=10)
        self.assertAlmostEqual(
            row["beta_target_exposure"], own["target"]["beta"], places=10
        )
        self.assertAlmostEqual(
            row["beta_control_exposure"], own["controls"]["grip"]["beta"], places=10
        )
        controls = eng.compute_negative_controls({}, n_boot=0)["splits"]["test"]
        self.assertAlmostEqual(
            row["beta_control_cgi"], controls["controls"]["grip"]["beta"], places=10
        )
        record = row["spatial_adjustment"]
        self.assertEqual(record["method"], "spatial_plus")
        self.assertEqual(record["df_cgi"], cgi["spatial_adjustment"]["df"])
        self.assertEqual(
            record["df_exposure"], own["target"]["spatial_adjustment"]["df"]
        )

    def test_ks_aic_gives_each_outcome_its_own_smooth(self):
        eng = _engine("ks_aic")
        out = eng.compute_specificity_contrast({}, self.exposures, n_boot=0)
        (row,) = out["rows"]
        d, t = eng.data, eng.test
        cov = d["age"].reshape(-1, 1)
        s_t, r_t = eng._reporting_smooth(d["xy"], t, d["y"], cov, d["x"])
        s_c, r_c = eng._reporting_smooth(d["xy"], t, d["c"], cov, d["x"])
        for key, x, y, s in (
            ("beta_target_cgi", d["x"], d["y"], s_t),
            ("beta_control_cgi", d["x"], d["c"], s_c),
            ("beta_target_exposure", d["bench"], d["y"], s_t),
            ("beta_control_exposure", d["bench"], d["c"], s_c),
        ):
            want = _ols_beta(x[t], y[t], np.column_stack([cov[t], s]))
            self.assertAlmostEqual(row[key], want, delta=1e-8, msg=key)
        self.assertEqual(
            row["spatial_adjustment"],
            {"method": "ks_aic", "df_target": r_t["df"], "df_control": r_c["df"]},
        )

    def test_the_cgi_against_itself_differs_by_zero(self):
        out = _engine("spatial_plus").compute_specificity_contrast(
            {}, {"same": ("cgi", {})}, n_boot=100
        )
        (row,) = out["rows"]
        self.assertEqual((row["delta_d"], row["ci_low"], row["ci_high"]), (0, 0, 0))


# ────────────────────────────────────────────────────────────────────
# Rows and tables
# ────────────────────────────────────────────────────────────────────


class TestRecords(unittest.TestCase):
    def test_columns_and_labels(self):
        one = {"method": "spatial_plus", "df": 3}
        pair = {"method": "ks_aic", "df_target": 2, "df_control": None}
        self.assertEqual(
            outer_splits.spatial_columns(one),
            {"spatial_method": "spatial_plus", "spatial_df": "3"},
        )
        self.assertEqual(
            outer_splits.spatial_columns(pair),
            {
                "spatial_method": "ks_aic",
                "spatial_df_target": "2",
                "spatial_df_control": "none",
            },
        )
        self.assertEqual(outer_splits.spatial_columns(None), {})
        self.assertEqual(outer_splits.spatial_label(one), "spatial_plus df 3")
        self.assertEqual(
            outer_splits.spatial_label(pair), "ks_aic df target 2, control none"
        )

    def test_slope_rows_carry_the_record_and_the_summary_the_method(self):
        record = {"method": "spatial_plus", "df": 4}
        slope = {"beta": 0.1, "ci_low": 0.0, "ci_high": 0.2, "t": 2.0, "df": 300}
        rows = []
        for k in range(3):
            for r in outer_splits.split_rows(
                effects_test={"score": 0.01},
                slope_test={**slope, "spatial_adjustment": record},
                channel_paired=None,
                comparator_paired=[],
                negative_controls=None,
            ):
                r.update(split=k, test_share=0.2)
                rows.append(r)
        beta = [r for r in rows if r["quantity"] == "held_out_beta"]
        self.assertTrue(all(r["spatial_df"] == "4" for r in beta))
        score = [r for r in rows if r["quantity"] == "held_out_score"]
        self.assertNotIn("spatial_method", score[0])
        summary = {e["quantity"]: e for e in outer_splits.summarise(rows)}
        self.assertEqual(summary["held_out_beta"]["spatial_method"], "spatial_plus")
        self.assertNotIn("spatial_method", summary["held_out_score"])


# ────────────────────────────────────────────────────────────────────
# A whole spatially adjusted run with two held-out splits
# ────────────────────────────────────────────────────────────────────

_INPUTS = """
    mode = "cross-sectional"
    ndvi = "ndvi_2015.tif"
    gvi = "gvi.gpkg"
    """

_SLOPES = (
    "held_out_beta",
    "negative_control_beta",
    "exposure_beta",
    "exposure_negative_control_beta",
    "specificity_difference",
)


class TestAdjustedRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_fusion_run import _write_study, run_study

        cls.folder = tempfile.mkdtemp(prefix="geofuse-spatial-")
        _write_study(cls.folder)
        cls.ctx, cls.bundle = run_study(
            cls.folder,
            _INPUTS,
            spatial_adjust_method="spatial_plus",
            outer_splits=2,
            specificity_contrast=True,
            comparators=["ndvi_here", "noise", "patchy"],
            comparators_primary=["ndvi_here", "patchy"],
        )
        cls.report = cls.bundle["report_dir"]

    def _table(self, name):
        with open(os.path.join(self.report, name), encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_the_run_completes(self):
        stages = {s["key"]: s["status"] for s in self.ctx.ledger["stages"]}
        self.assertNotIn("failed", stages.values(), stages)

    def test_every_splits_slope_rows_carry_the_smooth(self):
        rows = self.bundle["outer_splits"]["splits"]
        for k in (0, 1):
            split = [r for r in rows if r["split"] == k]
            quantities = {r["quantity"] for r in split}
            for quantity in _SLOPES:
                with self.subTest(split=k, quantity=quantity):
                    self.assertIn(quantity, quantities)
                    for r in split:
                        if r["quantity"] == quantity:
                            self.assertEqual(r["spatial_method"], "spatial_plus")
        diffs = [r for r in rows if r["quantity"] == "specificity_difference"]
        self.assertTrue(all("spatial_df_cgi" in r for r in diffs))

    def test_the_corrected_t_recomputes_from_the_adjusted_rows(self):
        outer = self.bundle["outer_splits"]
        rows = [r for r in outer["splits"] if r["quantity"] == "held_out_beta"]
        (entry,) = [e for e in outer["summary"] if e["quantity"] == "held_out_beta"]
        self.assertEqual(entry["spatial_method"], "spatial_plus")
        want = corrected_resampled_t(
            [r["estimate"] for r in rows],
            float(np.mean([r["test_share"] for r in rows])),
            alternative="two-sided",
        )
        self.assertAlmostEqual(entry["corrected_p"], want["p_value"])
        first = next(r for r in rows if r["split"] == 0)
        reference = self.bundle["comparator_reference"]["test"]
        self.assertAlmostEqual(first["estimate"], reference["beta"])
        self.assertEqual(
            first["spatial_df"], str(reference["spatial_adjustment"]["df"])
        )

    def test_the_tables_name_the_smooth(self):
        for name, column in (
            ("negative_controls.csv", "spatial_df"),
            ("comparators.csv", "spatial_df"),
            ("specificity_contrast.csv", "spatial_df_exposure"),
            ("outer_splits.csv", "spatial_df"),
        ):
            with self.subTest(name):
                table = self._table(name)
                self.assertIn(column, table[0])
                methods = {r["spatial_method"] for r in table if r["spatial_method"]}
                self.assertEqual(methods, {"spatial_plus"})


    def test_the_covariates_only_reports_say_so(self):
        for name in ("exposure_response.csv", "covariate_impact.csv"):
            with self.subTest(name):
                table = self._table(name)
                labels = {r["spatial_adjustment"] for r in table}
                self.assertEqual(labels, {"not applied"})
        report = self.bundle["exposure_response"]
        self.assertEqual(report["spatial_adjustment"], "not applied")

    def test_every_split_has_per_sd0_rows_with_the_slopes_p(self):
        rows = self.bundle["outer_splits"]["splits"]
        for k in (0, 1):
            split = {
                r["quantity"]: r
                for r in rows
                if r["split"] == k
                and r["exposure"] == "cgi"
                and not r.get("control")
                and r.get("weighting") is None
            }
            self.assertIn("held_out_per_sd0", split)
            self.assertAlmostEqual(
                split["held_out_per_sd0"]["p_value"],
                split["held_out_beta"]["p_value"],
                delta=1e-10,
            )


class TestUnadjustedRun(unittest.TestCase):
    def test_no_spatial_column_anywhere(self):
        from test_fusion_run import _run

        _, bundle = _run("cross-sectional with comparators")
        for name in (
            "negative_controls.csv",
            "comparators.csv",
            "exposure_response.csv",
            "covariate_impact.csv",
        ):
            path = os.path.join(bundle["report_dir"], name)
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                header = next(csv.reader(f))
            self.assertFalse([c for c in header if c.startswith("spatial_")], name)
        raw = (bundle.get("raw_effects") or {}).get("exposures") or {}
        for block in raw.values():
            self.assertNotIn("per_sd0", block.get("target") or {})


# ────────────────────────────────────────────────────────────────────
# The direction
# ────────────────────────────────────────────────────────────────────


class TestDirection(unittest.TestCase):
    def test_the_direction_conditions_on_the_scores_smooth(self):
        from geofuse.jobs import fusion_outputs

        rng = np.random.default_rng(5)
        u, g = rng.normal(size=400), rng.normal(size=400)
        held_out = {
            "targets": u - 0.5 * g,
            "predictions": u + g,
            "covariates": None,
            "spatial_basis": u.reshape(-1, 1),
        }
        signs = {}
        for method in ("none", "spatial_plus", "ks_aic"):
            eng = types.SimpleNamespace(
                evaluate_on_test=lambda **kw: dict(held_out),
                is_longitudinal=False,
                spatial_adjust_method=method,
                residualize_method="linear",
            )
            signs[method] = fusion_outputs._direction_sign(eng, {}, "r2")
        self.assertEqual(signs, {"none": 1, "spatial_plus": -1, "ks_aic": -1})


# ────────────────────────────────────────────────────────────────────
# The command line
# ────────────────────────────────────────────────────────────────────


class TestCommandLine(unittest.TestCase):
    def test_the_summary_names_the_adjustment(self):
        from geofuse.cli import results

        record = {"method": "spatial_plus", "df": 3}
        bundle = {
            "test_results": {"spatial_adjustment": {"method": "spatial_plus"}},
            "negative_controls": {
                "splits": {"test": {"target": {"spatial_adjustment": record}}}
            },
            "specificity_contrast": {
                "rows": [
                    {
                        "exposure": "g13",
                        "kind": "comparator",
                        "control": "grip",
                        "d_cgi": 0.02,
                        "d_exposure": -0.01,
                        "delta_d": 0.03,
                        "ci_low": 0.01,
                        "ci_high": 0.05,
                        "p_value": 0.01,
                        "n": 300,
                        "spatial_adjustment": {
                            "method": "spatial_plus",
                            "df_cgi": 3,
                            "df_exposure": 2,
                        },
                    }
                ]
            },
        }
        text = "\n".join(results.summary("y", bundle, Path(".")))
        self.assertIn(
            "Spatial adjust   spatial_plus smooth in the scores, slopes, negative "
            "controls and specificity (held-out β: df 3)",
            text,
        )
        self.assertIn("spatial_plus df cgi 3, exposure 2", text)
        old = {"test_results": {"spatial_adjustment": {"method": "ks_aic"}}}
        lines = results.summary("y", old, Path("."))
        self.assertIn("  Spatial adjust   ks_aic smooth in the scores", lines)


if __name__ == "__main__":
    unittest.main()
