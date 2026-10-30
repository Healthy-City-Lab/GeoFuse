"""Specificity contrast: is the CGI less non-specific than another exposure.

With ``D = |β_target| − |β_control|`` per exposure, ``ΔD = D_cgi − D_e > 0``
says the CGI is the more specific of the two. Pinned here:

- unit weights reproduce the four partial slopes on the common rows;
- a benchmark loading on a confounder that drives target and control gives
  ``ΔD > 0`` with an interval above zero, and swapping the roles negates it;
- an exposure equal to the CGI gives ``ΔD = 0`` on every resample;
- the engine compares on the held-out rows, aligns each exposure by entity,
  reports each exposure's own slopes, and skips panels;
- the primary comparators form one Holm family per control, the comparison
  channel none;
- on repeated splits there is one row per split × exposure × control, and the
  corrected p-value is the one recomputed from those rows;
- a run with the option off writes nothing new; a run with it on writes the
  table, the bundle key and the across-split rows; on without controls it
  warns, writes no rows and completes.
"""

import csv
import json
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

import geopandas as gpd  # noqa: F401
import numpy as np
import pandas as pd

from geofuse import negative_controls as nc
from geofuse.jobs import fusion_outputs, outer_splits, runners
from geofuse.statistical_testing import corrected_resampled_t


def _confounded(n=4000, seed=0):
    """A confounder U drives target and control; the benchmark loads on U,
    the CGI carries the pathway G with little U."""
    rng = np.random.default_rng(seed)
    u, g = rng.normal(size=n), rng.normal(size=n)
    target = 0.3 * g + 0.3 * u + rng.normal(size=n)
    control = 0.3 * u + rng.normal(size=n)
    cgi = g + 0.1 * u + 0.3 * rng.normal(size=n)
    bench = 0.5 * g + u + 0.3 * rng.normal(size=n)
    cov = rng.normal(size=(n, 2))
    return cgi, bench, target, control, cov


class TestContrast(unittest.TestCase):
    def test_unit_weights_reproduce_the_four_partial_slopes(self):
        cgi, bench, target, control, cov = _confounded(n=600)
        control[:40] = np.nan
        got = nc.specificity_contrast(cgi, bench, target, control, cov, n_boot=0)
        rows = np.isfinite(control)
        self.assertEqual(got["n"], int(rows.sum()))
        for key, x, y in (
            ("beta_target", cgi, target),
            ("beta_control", cgi, control),
            ("beta_target_other", bench, target),
            ("beta_control_other", bench, control),
        ):
            want = nc.partial_slope(x[rows], y[rows], cov[rows])["beta"]
            self.assertAlmostEqual(got[key], want, places=10)
        self.assertAlmostEqual(got["delta"], got["d"] - got["d_other"], places=12)

    def test_a_confounded_benchmark_is_less_specific(self):
        cgi, bench, target, control, cov = _confounded()
        got = nc.specificity_contrast(
            cgi, bench, target, control, cov, n_boot=600, seed=1
        )
        self.assertGreater(got["delta"], 0.0)
        self.assertGreater(got["ci_low"], 0.0)
        self.assertLess(got["p_value"], 0.01)
        self.assertLess(got["d_other"], got["d"])
        swapped = nc.specificity_contrast(
            bench, cgi, target, control, cov, n_boot=600, seed=1
        )
        self.assertAlmostEqual(swapped["delta"], -got["delta"], places=12)
        self.assertLess(swapped["ci_high"], 0.0)
        self.assertGreater(swapped["p_value"], 0.99)

    def test_the_same_exposure_differs_by_zero_on_every_resample(self):
        cgi, _, target, control, cov = _confounded(n=500)
        got = nc.specificity_contrast(
            cgi, cgi.copy(), target, control, cov, n_boot=300, seed=2
        )
        self.assertEqual(got["delta"], 0.0)
        self.assertEqual((got["ci_low"], got["ci_high"]), (0.0, 0.0))
        self.assertEqual(got["p_value"], 1.0)

    def test_whole_clusters_are_resampled_on_request(self):
        cgi, bench, target, control, cov = _confounded(n=800)
        labels = np.arange(800) % 40
        got = nc.specificity_contrast(
            cgi, bench, target, control, cov, n_boot=100, clusters=labels
        )
        self.assertEqual(got["clusters"], 40)

    def test_too_few_rows_give_empty_values(self):
        got = nc.specificity_contrast([1.0, 2.0], [2.0, 1.0], [1.0, 3.0], [0.0, 1.0])
        self.assertTrue(np.isnan(got["delta"]))
        self.assertEqual(got["n"], 2)


def _engine_stub(n_ent=400, n_test=120, seed=7, controls=("grip", "height")):
    """The engine's reporting helpers over a frame the test controls: the CGI,
    a confounded comparator ``bench`` (rows in another order) and a channel."""
    from geofuse.fusion import MetricFusionEngine as E
    from geofuse.fusion import comparator_mode

    rng = np.random.default_rng(seed)
    u, g = rng.normal(size=n_ent), rng.normal(size=n_ent)
    target = 0.3 * g + 0.3 * u + rng.normal(size=n_ent)
    grip = 0.3 * u + rng.normal(size=n_ent)
    grip[:12] = np.nan
    height = rng.normal(size=n_ent)
    age = rng.normal(size=n_ent)
    pid = np.repeat(np.arange(n_ent), 2)
    full = pd.DataFrame(
        {
            "polygon_id": pid,
            "target": target[pid],
            "age": age[pid],
            "grip": grip[pid],
            "height": height[pid],
            "fsa": (pid // 10).astype(str),
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

    bench = 0.5 * g + u + 0.3 * rng.normal(size=n_ent)
    frames = {
        "cgi": frame(g + 0.1 * u + 0.3 * rng.normal(size=n_ent)),
        comparator_mode("bench"): frame(bench, rng.permutation(n_ent)),
        "ndvi": frame(g + 0.5 * rng.normal(size=n_ent)),
    }
    eng = types.SimpleNamespace(
        negative_control_columns=list(controls),
        is_panel=False,
        covariate_columns=["age"],
        _covariate_dummy_map={},
        test_data=full[full["polygon_id"] < n_test],
        _active_greenery_channel="cgi",
        spatial_adjust_method="none",
    )
    eng.apply_fusion = lambda weights=None: frames[eng._active_greenery_channel]
    eng._full_data_frame = lambda: full
    for name in (
        "_reporting_covariate_matrix",
        "_reporting_raw_column",
        "_reporting_clusters",
        "_test_entity_mask",
        "_whole_data_coords",
        "_slope_smooths",
        "compute_specificity_contrast",
    ):
        setattr(eng, name, types.MethodType(getattr(E, name), eng))
    eng._with_smooth = E._with_smooth
    eng.frames = frames
    eng.entity = {"target": target, "age": age, "bench": bench}
    return eng


class TestEngineContrast(unittest.TestCase):
    def test_every_exposure_and_control_on_the_held_out_rows(self):
        eng = _engine_stub()
        exposures = {"bench": ("comparator:bench", {}), "ndvi": ("ndvi", {})}
        out = eng.compute_specificity_contrast({}, exposures, n_boot=200)
        self.assertEqual(out["split"], "test")
        self.assertEqual(out["n_test"], 120)
        self.assertEqual(
            [(r["exposure"], r["control"]) for r in out["rows"]],
            [
                ("bench", "grip"),
                ("bench", "height"),
                ("ndvi", "grip"),
                ("ndvi", "height"),
            ],
        )
        grip = next(r for r in out["rows"] if r["control"] == "grip")
        self.assertEqual(grip["n"], 120 - 12)
        self.assertEqual(eng._active_greenery_channel, "cgi")
        slopes = out["slopes"]["bench"]
        self.assertEqual(slopes["target"]["n"], 120)
        self.assertEqual(slopes["controls"]["grip"]["n"], 108)

    def test_each_exposure_is_aligned_by_entity(self):
        eng = _engine_stub()
        out = eng.compute_specificity_contrast(
            {}, {"bench": ("comparator:bench", {})}, n_boot=0
        )
        e = eng.entity
        want = nc.partial_slope(e["bench"][:120], e["target"][:120], e["age"][:120])
        got = out["slopes"]["bench"]["target"]
        self.assertAlmostEqual(got["beta"], want["beta"], places=10)
        row = next(r for r in out["rows"] if r["control"] == "height")
        self.assertAlmostEqual(row["beta_target_exposure"], want["beta"], places=10)

    def test_the_cgi_against_itself_differs_by_zero(self):
        out = _engine_stub().compute_specificity_contrast(
            {}, {"same": ("cgi", {})}, n_boot=200
        )
        for r in out["rows"]:
            self.assertEqual(r["delta_d"], 0.0)
            self.assertEqual((r["ci_low"], r["ci_high"]), (0.0, 0.0))

    def test_without_controls_or_exposures_there_is_nothing(self):
        self.assertIsNone(
            _engine_stub(controls=()).compute_specificity_contrast(
                {}, {"same": ("cgi", {})}
            )
        )
        self.assertIsNone(_engine_stub().compute_specificity_contrast({}, {}))

    def test_a_panel_is_skipped(self):
        eng = _engine_stub()
        eng.is_panel = True
        out = eng.compute_specificity_contrast({}, {"same": ("cgi", {})})
        self.assertEqual(out["skipped"], "longitudinal")


class TestReport(unittest.TestCase):
    def test_holm_runs_over_the_primary_comparators_per_control(self):
        eng = _engine_stub()
        exposures = fusion_outputs.specificity_exposures(
            ["bench", "again", "aside"], ["bench", "again"], "ndvi", {}
        )
        eng.frames["comparator:again"] = eng.frames["comparator:bench"]
        eng.frames["comparator:aside"] = eng.frames["comparator:bench"]
        out = fusion_outputs._specificity_report(
            eng, {}, exposures, 200, label="t", log=None
        )
        rows = {(r["exposure"], r["control"]): r for r in out["rows"]}
        for control in ("grip", "height"):
            family = [rows[("bench", control)], rows[("again", control)]]
            ps = sorted(r["p_value"] for r in family)
            self.assertAlmostEqual(
                min(r["p_value_holm"] for r in family), min(1.0, 2 * ps[0])
            )
            self.assertIsNone(rows[("aside", control)]["p_value_holm"])
            self.assertIsNone(rows[("ndvi", control)]["p_value_holm"])
        self.assertEqual(rows[("ndvi", "grip")]["kind"], "channel")
        self.assertIs(rows[("aside", "grip")]["primary"], False)

    def test_a_channel_named_like_a_comparator_keeps_both(self):
        exposures = fusion_outputs.specificity_exposures(["ndvi"], [], "ndvi", {})
        self.assertEqual(list(exposures), ["ndvi", "channel:ndvi"])
        self.assertIs(exposures["ndvi"]["primary"], True)


def _split_report(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    info = {
        "g13": {"kind": "comparator", "primary": True},
        "ndvi": {"kind": "channel", "primary": None},
    }
    slope = {"beta": 0.05, "ci_low": 0.0, "ci_high": 0.1, "t": 2.0, "df": 200}
    return {
        "controls": ["grip"],
        "exposure_info": info,
        "slopes": {e: {"target": slope, "controls": {"grip": slope}} for e in info},
        "rows": [
            {
                "exposure": e,
                "control": "grip",
                "delta_d": float(rng.normal(0.03, 0.01)),
                "ci_low": -0.01,
                "ci_high": 0.07,
                "p_value": 0.04,
            }
            for e in info
        ],
    }


class TestAcrossSplits(unittest.TestCase):
    def test_one_row_per_exposure_and_control_with_the_corrected_p(self):
        rows = []
        for k in range(4):
            for r in outer_splits.specificity_rows(_split_report(k)):
                r.update(split=k, test_share=0.25)
                rows.append(r)
        diffs = [r for r in rows if r["quantity"] == "specificity_difference"]
        self.assertEqual(len(diffs), 4 * 2)
        self.assertTrue(all(r["control"] == "grip" for r in diffs))
        betas = [r for r in rows if r["quantity"] == "exposure_beta"]
        self.assertTrue(all(r["control"] is None for r in betas))
        summary = outer_splits.summarise(rows)
        entry = next(
            e
            for e in summary
            if (e["quantity"], e["exposure"], e.get("control"))
            == ("specificity_difference", "g13", "grip")
        )
        values = [r["estimate"] for r in diffs if r["exposure"] == "g13"]
        want = corrected_resampled_t(values, 0.25, alternative="greater")
        self.assertAlmostEqual(entry["corrected_p"], want["p_value"])
        self.assertAlmostEqual(entry["corrected_p_holm"], want["p_value"])
        self.assertEqual(entry["kind"], "comparator")
        channel = next(
            e
            for e in summary
            if (e["quantity"], e["exposure"]) == ("specificity_difference", "ndvi")
        )
        self.assertIsNone(channel["corrected_p_holm"])

    def test_rows_without_a_control_have_no_control_column(self):
        rows = outer_splits.split_rows(
            effects_test={"score": 0.01},
            slope_test=None,
            channel_paired=None,
            comparator_paired=[],
            negative_controls=None,
        )
        self.assertNotIn("control", rows[0])
        entry = outer_splits.summarise([{**rows[0], "split": 0, "test_share": 0.2}])
        self.assertNotIn("control", entry[0])


# ────────────────────────────────────────────────────────────────────
# Whole runs
# ────────────────────────────────────────────────────────────────────

_INPUTS = """
    mode = "cross-sectional"
    ndvi = "ndvi_2015.tif"
    gvi = "gvi.gpkg"
    """

_COMPARATORS = {
    "comparators": ["ndvi_here", "noise", "patchy"],
    "comparators_primary": ["ndvi_here", "patchy"],
}


class TestOffRun(unittest.TestCase):
    def test_a_run_without_the_option_writes_nothing_new(self):
        from test_fusion_run import _run

        _, bundle = _run("cross-sectional with comparators")
        self.assertNotIn("specificity_contrast", bundle)
        report = bundle["report_dir"]
        self.assertFalse(
            os.path.exists(os.path.join(report, "specificity_contrast.csv"))
        )
        with open(os.path.join(report, "run_config.json"), encoding="utf-8") as f:
            self.assertNotIn("specificity_contrast", json.load(f))
        with open(os.path.join(report, "results_summary.json"), encoding="utf-8") as f:
            self.assertNotIn("specificity_contrast", json.load(f))


class TestOnRun(unittest.TestCase):
    """The option on, with comparators, standalones and two held-out splits."""

    @classmethod
    def setUpClass(cls):
        from test_fusion_run import _write_study, run_study

        cls.folder = tempfile.mkdtemp(prefix="geofuse-specificity-")
        _write_study(cls.folder)
        cls.ctx, cls.bundle = run_study(
            cls.folder,
            _INPUTS,
            outer_splits=2,
            specificity_contrast=True,
            **_COMPARATORS,
        )
        cls.report = cls.bundle["specificity_contrast"]

    def test_the_first_split_compares_the_primary_comparators_and_the_channel(self):
        rows = self.report["rows"]
        channel = self.bundle["cgi_vs_standalone_paired"]["standalone_channel"]
        self.assertEqual(
            [(r["exposure"], r["kind"]) for r in rows],
            [
                ("ndvi_here", "comparator"),
                ("patchy", "comparator"),
                (channel, "channel"),
            ],
        )
        self.assertTrue(all(r["control"] == "grip" for r in rows))
        n_test = self.report["n_test"]
        for r in rows:
            self.assertLessEqual(r["n"], n_test)
            self.assertAlmostEqual(r["delta_d"], r["d_cgi"] - r["d_exposure"])
            self.assertLessEqual(r["ci_low"], r["ci_high"])

    def test_the_table_and_the_manifest_carry_it(self):
        out = self.bundle["report_dir"]
        with open(os.path.join(out, "specificity_contrast.csv"), encoding="utf-8") as f:
            table = list(csv.DictReader(f))
        self.assertEqual(len(table), 3)
        self.assertIn("delta_d", table[0])
        with open(os.path.join(out, "results_summary.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(len(manifest["specificity_contrast"]["rows"]), 3)
        with open(os.path.join(out, "run_config.json"), encoding="utf-8") as f:
            self.assertIs(json.load(f)["specificity_contrast"], True)

    def test_every_split_has_a_row_per_exposure_and_control(self):
        outer = self.bundle["outer_splits"]
        for k in (0, 1):
            diffs = [
                r
                for r in outer["splits"]
                if r["split"] == k and r["quantity"] == "specificity_difference"
            ]
            kinds = sorted(r["kind"] for r in diffs)
            self.assertEqual(kinds, ["channel", "comparator", "comparator"])
            self.assertTrue(all(r["control"] == "grip" for r in diffs))
        first = [
            r["estimate"]
            for r in outer["splits"]
            if r["split"] == 0 and r["quantity"] == "specificity_difference"
        ]
        self.assertEqual(first, [r["delta_d"] for r in self.report["rows"]])
        for entry in outer["summary"]:
            if entry["quantity"] != "specificity_difference":
                continue
            values = [
                r["estimate"]
                for r in outer["splits"]
                if (r["quantity"], r["exposure"], r.get("control"))
                == ("specificity_difference", entry["exposure"], entry["control"])
            ]
            shares = [
                r["test_share"]
                for r in outer["splits"]
                if r["quantity"] == "specificity_difference"
                and r["exposure"] == entry["exposure"]
            ]
            want = corrected_resampled_t(
                values, float(np.mean(shares)), alternative="greater"
            )
            if want["p_value"] is None:
                self.assertIsNone(entry["corrected_p"])
            else:
                self.assertAlmostEqual(entry["corrected_p"], want["p_value"])
        old = [r for r in outer["splits"] if r["quantity"] == "held_out_beta"]
        self.assertTrue(all(r.get("control") is None for r in old))


class TestOnWithoutControls(unittest.TestCase):
    def test_it_warns_writes_no_rows_and_completes(self):
        from test_fusion_run import _write_study, run_study

        folder = tempfile.mkdtemp(prefix="geofuse-specificity-")
        _write_study(folder)
        logged = []
        real = runners._log_fusion

        def record(level, message, *args, **kwargs):
            logged.append((level, message))
            return real(level, message, *args, **kwargs)

        with mock.patch.object(runners, "_log_fusion", side_effect=record):
            ctx, bundle = run_study(
                folder, _INPUTS, negative_controls=[], specificity_contrast=True
            )
        warns = [m for lvl, m in logged if lvl == "WARN" and "Specificity" in m]
        self.assertEqual(len(warns), 1, warns)
        self.assertIsNone(bundle["specificity_contrast"])
        stages = {s["status"] for s in ctx.ledger["stages"]}
        self.assertNotIn("failed", stages)
        self.assertFalse(
            os.path.exists(
                os.path.join(bundle["report_dir"], "specificity_contrast.csv")
            )
        )


if __name__ == "__main__":
    unittest.main()
