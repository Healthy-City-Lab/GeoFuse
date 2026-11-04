"""Raw-unit effect sizes: per IQR and per 5th–95th percentile, every split.

Pinned here:

- ``raw_slope`` is the OLS coefficient of the outcome on ``exposure / scale``
  with the covariates (and a smooth's columns): a planted 0.5 per unit gives
  0.5 × IQR per IQR, a comparator twice the exposure gives the same per-IQR
  contrast and half the per-unit slope, and its t is the per-SD slope's;
- a run reports, for the CGI, every standalone and every comparator, the
  held-out slope on the target and on each control per IQR and per p5–p95,
  with the scale set on the split's train+val rows, on every split;
- ``fixed_iqr`` replaces an exposure's IQR, and the run record names it only
  when set;
- the CGI's per-IQR slope carries the same test as its per-SD slope, which
  the new rows leave as it was.
"""

import csv
import json
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

from geofuse import negative_controls as nc

INPUTS = """
    mode = "cross-sectional"
    ndvi = "ndvi_2015.tif"
    gvi = "gvi.gpkg"
    """

RAW = {
    "held_out_per_iqr",
    "held_out_per_p5_p95",
    "exposure_per_iqr",
    "exposure_per_p5_p95",
    "exposure_negative_control_per_iqr",
    "exposure_negative_control_per_p5_p95",
}


class TestRawSlope(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.n = 4000
        self.x = rng.gamma(2.0, 0.1, self.n)
        self.cov = rng.normal(size=(self.n, 2))
        self.y = 0.5 * self.x + self.cov @ [0.3, -0.2] + rng.normal(0, 0.05, self.n)
        q25, q75 = np.quantile(self.x, [0.25, 0.75])
        self.iqr = q75 - q25

    def test_a_planted_slope_is_recovered_per_iqr(self):
        r = nc.raw_slope(self.x, self.y, self.cov, scale=self.iqr)
        self.assertAlmostEqual(r["estimate"], 0.5 * self.iqr, delta=3 * r["se"])
        self.assertLess(r["ci_low"], 0.5 * self.iqr)
        self.assertGreater(r["ci_high"], 0.5 * self.iqr)
        self.assertEqual(r["scale"], self.iqr)

    def test_a_rescaled_exposure_gives_the_same_contrast(self):
        a = nc.raw_slope(self.x, self.y, self.cov, scale=self.iqr)
        b = nc.raw_slope(2 * self.x, self.y, self.cov, scale=2 * self.iqr)
        self.assertAlmostEqual(a["estimate"], b["estimate"], places=12)
        self.assertAlmostEqual(a["ci_low"], b["ci_low"], places=12)
        unit_a = nc.raw_slope(self.x, self.y, self.cov)
        unit_b = nc.raw_slope(2 * self.x, self.y, self.cov)
        self.assertAlmostEqual(unit_b["estimate"], unit_a["estimate"] / 2, places=12)

    def test_it_is_ols_on_the_scaled_exposure_with_a_smooth(self):
        from geofuse import spatial_basis

        rng = np.random.default_rng(9)
        xy = rng.uniform(0, 20_000, (self.n, 2))
        basis = spatial_basis.build_block_basis(xy, max_df=8)
        smooth = basis.design(8)
        controls = np.column_stack([self.cov, smooth])
        r = nc.raw_slope(self.x, self.y, controls, scale=self.iqr)
        design = np.column_stack([np.ones(self.n), self.x / self.iqr, controls])
        coef, *_ = np.linalg.lstsq(design, self.y, rcond=None)
        self.assertAlmostEqual(r["estimate"], coef[1], delta=1e-8)
        resid = self.y - design @ coef
        dof = self.n - np.linalg.matrix_rank(design)
        cov_b = (resid @ resid / dof) * np.linalg.pinv(design.T @ design)
        self.assertAlmostEqual(r["se"], np.sqrt(cov_b[1, 1]), delta=1e-8)
        self.assertEqual(r["df"], dof)

    def test_its_t_is_the_per_sd_slopes(self):
        r = nc.raw_slope(self.x, self.y, self.cov, scale=self.iqr)
        sd = nc.partial_slope(self.x, self.y, self.cov)
        self.assertAlmostEqual(r["t"], sd["t"], places=8)

    def test_a_missing_scale_gives_no_estimate(self):
        r = nc.raw_slope(self.x, self.y, self.cov, scale=float("nan"))
        self.assertTrue(np.isnan(r["estimate"]))


def _study() -> str:
    """The repeated-split study with the smooth field at each home as a
    comparator (the outcome is 0.8 per unit of it) and that field doubled."""
    import rasterio
    from test_fusion_run import _PX, _X0, _Y0
    from test_outer_splits import _write_smooth_study

    folder = tempfile.mkdtemp(prefix="geofuse-raw-")
    _write_smooth_study(folder)
    with rasterio.open(os.path.join(folder, "ndvi_2015.tif")) as src:
        field = src.read(1).astype(np.float64)
    path = os.path.join(folder, "cohort.gpkg")
    cohort = gpd.read_file(path)
    col = ((cohort.geometry.x - (_X0 - 0.01)) / _PX).astype(int)
    row = (((_Y0 + 0.04) - cohort.geometry.y) / _PX).astype(int)
    cohort["field_here"] = field[row, col]
    cohort["field_twice"] = 2.0 * cohort["field_here"]
    cohort.to_file(path, driver="GPKG")
    return folder


COMPARATORS = {
    "comparators": ["field_here", "field_twice", "noise"],
    "comparators_primary": ["field_here", "field_twice"],
}


class TestRawEffectsRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_fusion_run import run_study

        cls.folder = _study()
        cls.bundle = run_study(cls.folder, INPUTS, outer_splits=2, **COMPARATORS)[1]
        cls.outer = cls.bundle["outer_splits"]
        cls.fixed = run_study(
            cls.folder, INPUTS, fixed_iqr={"field_here": 0.1}, **COMPARATORS
        )[1]

    def _rows(self, quantity, exposure, control=None):
        return [
            r
            for r in self.outer["splits"]
            if (r["quantity"], r["exposure"], r.get("control"))
            == (quantity, exposure, control)
        ]

    def test_every_split_reports_every_exposure(self):
        for split in (0, 1):
            rows = [
                r
                for r in self.outer["splits"]
                if r["split"] == split and r["quantity"] in RAW
            ]
            exposures = {r["exposure"] for r in rows}
            self.assertTrue(
                {"cgi", "ndvi", "gvi", "field_here", "field_twice"} <= exposures,
                exposures,
            )
            controls = {r["control"] for r in rows if r["control"]}
            self.assertEqual(controls, {"grip"})
            for r in rows:
                self.assertIsNotNone(r["estimate"])
                self.assertGreater(r["scale"], 0)
                self.assertLessEqual(r["ci_low"], r["estimate"])

    def test_the_planted_effect_is_recovered_per_iqr(self):
        scales = {
            (d["split"], d["exposure"]): d
            for d in self.outer["distributions"]
            if d["layer"] is None
        }
        for r in self._rows("exposure_per_iqr", "field_here"):
            iqr = scales[(r["split"], "field_here")]["iqr"]
            self.assertAlmostEqual(r["scale"], iqr)
            se = (r["ci_high"] - r["ci_low"]) / (2 * 1.97)
            self.assertAlmostEqual(r["estimate"], 0.8 * iqr, delta=3 * se)

    def test_a_doubled_comparator_gives_the_same_contrast(self):
        for q in ("exposure_per_iqr", "exposure_per_p5_p95"):
            here = {r["split"]: r for r in self._rows(q, "field_here")}
            twice = {r["split"]: r for r in self._rows(q, "field_twice")}
            self.assertEqual(set(here), {0, 1})
            for split, r in here.items():
                self.assertAlmostEqual(twice[split]["estimate"], r["estimate"], 9)
                self.assertAlmostEqual(twice[split]["scale"], 2 * r["scale"], 9)

    def test_the_summary_tests_each_raw_quantity(self):
        entries = [e for e in self.outer["summary"] if e["quantity"] in RAW]
        self.assertTrue(entries)
        for e in entries:
            self.assertEqual(e["alternative"], "two-sided")
            self.assertEqual(e["n_splits"], 2)
            self.assertIsNotNone(e["corrected_p"])

    def test_the_cgi_layers_are_described(self):
        layers = [
            d for d in self.outer["distributions"] if d["kind"] == "layer"
        ]
        self.assertEqual({d["split"] for d in layers}, {0, 1})
        self.assertEqual({d["layer"] for d in layers}, {"ndvi", "gvi"})
        for d in layers:
            self.assertGreaterEqual(d["iqr"], 0)
            self.assertLessEqual(d["p5"], d["median"])

    def test_the_tables_are_written(self):
        report = self.bundle["report_dir"]
        with open(os.path.join(report, "raw_effects.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual({r["exposure"] for r in rows} >= {"cgi", "noise"}, True)
        with open(os.path.join(report, "exposure_scales.csv"), encoding="utf-8") as f:
            scales = list(csv.DictReader(f))
        self.assertEqual({r["split"] for r in scales}, {"0", "1"})
        with open(os.path.join(report, "results_summary.json"), encoding="utf-8") as f:
            self.assertIn("raw_effects", json.load(f))

    def test_the_raw_and_per_sd_slopes_share_their_test(self):
        beta = {r["split"]: r for r in self._rows("held_out_beta", "cgi")}
        raw = {r["split"]: r for r in self._rows("held_out_per_iqr", "cgi")}
        self.assertEqual(set(beta), {0, 1})
        for split, r in beta.items():
            self.assertNotIn("scale", r)
            self.assertAlmostEqual(raw[split]["p_value"], r["p_value"], places=10)
            self.assertEqual(np.sign(raw[split]["estimate"]), np.sign(r["estimate"]))

    def test_a_fixed_iqr_replaces_the_train_val_iqr(self):
        exposures = self.fixed["raw_effects"]["exposures"]
        self.assertEqual(exposures["field_here"]["iqr_source"], "fixed")
        self.assertEqual(exposures["field_here"]["target"]["per_iqr"]["scale"], 0.1)
        self.assertEqual(exposures["field_twice"]["iqr_source"], "train_val")
        with open(
            os.path.join(self.fixed["report_dir"], "run_config.json"), encoding="utf-8"
        ) as f:
            self.assertEqual(json.load(f)["fixed_iqr"], {"field_here": 0.1})
        with open(
            os.path.join(self.bundle["report_dir"], "run_config.json"), encoding="utf-8"
        ) as f:
            self.assertNotIn("fixed_iqr", json.load(f))


    def test_the_command_line_prints_the_raw_rows(self):
        from pathlib import Path

        from geofuse.cli import results

        lines = results.summary("outcome", self.bundle, Path(self.folder))
        text = "\n".join(lines)
        self.assertIn("Raw units", text)
        self.assertIn("CGI layers", text)
        self.assertIn("per IQR, field_here", text)

    def test_the_results_panel_draws_the_raw_rows(self):
        ui = os.path.join(ROOT, "ui")
        if ui not in sys.path:
            sys.path.insert(0, ui)
        from tabs import fusion as fusion_tab
        from test_posterior_panel import _Recorder

        real, rec = fusion_tab.st, _Recorder()
        fusion_tab.st = rec
        try:
            fusion_tab._render_raw_effects(self.bundle)
            fusion_tab._render_raw_effects({})
        finally:
            fusion_tab.st = real
        frame = next(v for n, v in rec.calls if n == "dataframe")
        self.assertIn("Change", frame.columns)
        self.assertIn("field_here", set(frame["Exposure"]))
        self.assertIn("Raw-unit effect sizes", rec.text())

if __name__ == "__main__":
    unittest.main()
