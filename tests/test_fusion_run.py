"""A small fusion job runs from its run file to its results.

The runner chains many reporting steps after the fits, and a defect in any of
them used to surface only at the end of a real run. Here one synthetic study
goes through ``plans.build`` and ``run_fusion`` in this process, with the
standalone studies on, at the smallest budgets the settings allow. Pinned
here:

- the run completes and writes its results;
- every CGI-vs-standalone row carries the held-out interval and p-value, and
  its Holm-corrected p-value;
- a missing held-out half leaves that inference empty instead of failing;
- a year-aware run (a greenery file per measurement year, one row per entity)
  is not a panel: with every visit in one year its posterior, splits and
  negative controls are the cross-sectional run's.
"""

import os
import sys
import tempfile
import textwrap
import types
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from geofuse import runfile
from geofuse.fusion import MetricFusionEngine
from geofuse.jobs import plans, runners

_X0, _Y0, _PX = -114.10, 51.00, 0.0004

_SETTINGS = """
objective_metric = "r2"
gvi_buffer_min_m = 100
gvi_buffer_max_m = 300
gvi_buffer_step_m = 100
ndvi_buffer_min_m = 100
ndvi_buffer_max_m = 300
ndvi_buffer_step_m = 100
sweep_splits = 5
discovery_reps = 0
gain_splits = 0
null_calibration_runs = 0
posterior_draws = 100
posterior_warmup = 100
posterior_chains = 2
run_standalones = true
check_collinearity = false
spatial_split = false
report_ci_bootstrap = 100
report_effects_bootstrap = 100
report_effects_permutations = 0
report_paired_bootstrap = 100
"""


def _write_study(folder: str, n: int = 160) -> None:
    """A cohort whose outcome follows the NDVI at its point, every visit in
    2015, per-year NDVI rasters and street-view points."""
    rng = np.random.default_rng(3)
    xs = _X0 + rng.random(n) * 0.03
    ys = _Y0 + rng.random(n) * 0.02
    ndvi = rng.random((160, 160)).astype("float32")
    for year, shift in ((2015, 0.0), (2018, 0.02)):
        with rasterio.open(
            os.path.join(folder, f"ndvi_{year}.tif"),
            "w",
            driver="GTiff",
            height=160,
            width=160,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(_X0 - 0.01, _Y0 + 0.04, _PX, _PX),
        ) as dst:
            dst.write(ndvi + shift, 1)
    col = ((xs - (_X0 - 0.01)) / _PX).astype(int)
    row = (((_Y0 + 0.04) - ys) / _PX).astype(int)
    gpd.GeoDataFrame(
        {
            "outcome": 0.8 * ndvi[row, col] + rng.normal(0, 0.3, n),
            "grip": rng.normal(size=n),
            "age": rng.normal(60, 8, n),
            "sex": rng.choice(["f", "m"], n),
            "visit": ["2015-06-15"] * n,
        },
        geometry=[Point(a, b) for a, b in zip(xs, ys)],
        crs="EPSG:4326",
    ).to_file(os.path.join(folder, "cohort.gpkg"), driver="GPKG")
    pts = [
        Point(_X0 - 0.005 + rng.random() * 0.04, _Y0 - 0.005 + rng.random() * 0.03)
        for _ in range(900)
    ]
    gpd.GeoDataFrame(
        {"gvi_veg": rng.random(900), "gvi_ter": rng.random(900)},
        geometry=pts,
        crs="EPSG:4326",
    ).to_file(os.path.join(folder, "gvi.gpkg"), driver="GPKG")


class _Context:
    """The parts of a job context the fusion runner calls."""

    job_id = "test-fusion-run"

    def __init__(self):
        self.extra: dict = {}
        self.ledger: dict = {}

    def progress(self, *args, **kwargs):
        pass

    def is_cancelled(self):
        return False

    def update_stage_ledger(self, ledger):
        self.ledger = ledger

    def set_extra(self, **kwargs):
        self.extra.update(kwargs)

    def heartbeat(self, *args, **kwargs):
        pass

    def wait_while_paused(self, *args, **kwargs):
        pass


def run_study(folder: str, inputs: str) -> tuple[_Context, dict]:
    """Run one fusion step over the study in ``folder``; ``inputs`` names the
    mode and the greenery files. Returns the context and the outcome's bundle."""
    doc = (
        f"output_dir = {os.path.join(folder, 'out')!r}\n"
        "[[fusion]]\n"
        'name = "t"\n'
        'target = "cohort.gpkg"\n'
        'outcomes = ["outcome"]\n'
        'negative_controls = ["grip"]\n'
        'covariates = ["age"]\n'
        'categorical = ["sex"]\n' + textwrap.dedent(inputs) + _SETTINGS
    )
    path = os.path.join(folder, "run.toml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    (spec,) = plans.expand(runfile.load(path, ()))
    ctx = _Context()
    runners.run_fusion(ctx, **plans.build(spec).kwargs["run_kwargs"])
    return ctx, ctx.extra["results"]["by_target"]["outcome"]


_RUNS: dict = {}


def _run(mode: str) -> tuple[_Context, dict]:
    """The study run once per mode for the whole module."""
    if mode not in _RUNS:
        if "folder" not in _RUNS:
            _RUNS["folder"] = tempfile.mkdtemp(prefix="geofuse-run-")
            _write_study(_RUNS["folder"])
        inputs = {
            "cross-sectional": """
                mode = "cross-sectional"
                ndvi = "ndvi_2015.tif"
                gvi = "gvi.gpkg"
                """,
            "year-aware": """
                mode = "year-aware"
                date_column = "visit"
                ndvi = {2015 = "ndvi_2015.tif"}
                gvi = {2015 = "gvi.gpkg"}
                """,
        }[mode]
        _RUNS[mode] = run_study(_RUNS["folder"], inputs)
    return _RUNS[mode]


class TestCrossSectionalRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx, cls.bundle = _run("cross-sectional")

    def test_the_run_completes_and_writes_its_results(self):
        stages = {s["key"]: s["status"] for s in self.ctx.ledger["stages"]}
        self.assertNotIn("failed", stages.values(), stages)
        summary = os.path.join(self.bundle["report_dir"], "results_summary.json")
        self.assertTrue(os.path.exists(summary))

    def test_every_paired_row_carries_its_held_out_inference(self):
        family = self.bundle["cgi_vs_standalone_paired_family"]
        self.assertEqual({d["standalone_channel"] for d in family}, {"ndvi", "gvi"})
        for d in family:
            self.assertLessEqual(d["lower"], d["upper"])
            self.assertTrue(0.0 < d["p_value"] <= 1.0)
            self.assertGreaterEqual(d["p_value_holm"], d["p_value"])
            self.assertLess(d["inference_n"], d["n"])
        self.assertIn(self.bundle["cgi_vs_standalone_paired"], family)


class TestYearAwareRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx, cls.bundle = _run("year-aware")
        cls.cross = _run("cross-sectional")[1]

    def test_the_posterior_is_the_cross_sectional_one(self):
        year, cross = self.bundle["discovery_summary"], self.cross["discovery_summary"]
        self.assertEqual(year["converged"], cross["converged"])
        for key in ("beta_mean", "beta_ci_low", "beta_ci_high", "rhat_max"):
            self.assertAlmostEqual(year[key], cross[key], places=6, msg=key)
        np.testing.assert_allclose(year["weight_mean"], cross["weight_mean"])

    def test_negative_controls_are_reported_as_in_the_cross_sectional_run(self):
        year = self.bundle["negative_controls"]
        cross = self.cross["negative_controls"]
        self.assertNotIn("skipped", year)
        for split in ("test", "train_val"):
            y_split, c_split = year["splits"][split], cross["splits"][split]
            self.assertEqual(y_split["target"]["n"], c_split["target"]["n"])
            self.assertAlmostEqual(
                y_split["target"]["beta"], c_split["target"]["beta"], places=6
            )
            y_grip, c_grip = y_split["controls"]["grip"], c_split["controls"]["grip"]
            for key in ("beta", "delta", "delta_ci_low", "delta_ci_high"):
                self.assertAlmostEqual(y_grip[key], c_grip[key], places=6, msg=key)


class TestPanelRule(unittest.TestCase):
    def _state(self, longitudinal, panel):
        return types.SimpleNamespace(is_longitudinal=longitudinal, _panel=panel)

    def test_only_repeated_entities_make_a_panel(self):
        is_panel = MetricFusionEngine.is_panel.fget
        self.assertFalse(is_panel(self._state(False, None)))
        self.assertFalse(is_panel(self._state(True, False)))
        self.assertTrue(is_panel(self._state(True, True)))
        # Before the prep has seen the rows, a longitudinal run counts as one.
        self.assertTrue(is_panel(self._state(True, None)))


class TestPairedInferenceWithoutAHeldOutHalf(unittest.TestCase):
    def _engine(self, held):
        whole = {
            "observed_diff": 0.02,
            "cgi_score": 0.05,
            "standalone_score": 0.03,
            "standalone_channel": "ndvi",
            "n": 200,
            "favors_cgi": True,
        }

        def paired_on(*args, held_out=False, **kwargs):
            return held if held_out else dict(whole)

        return types.SimpleNamespace(_paired_difference_on=paired_on)

    def test_the_held_out_interval_and_p_value_are_kept(self):
        held = {
            "observed_diff": 0.01,
            "n": 50,
            "lower": -0.02,
            "upper": 0.04,
            "p_value": 0.3,
        }
        out = MetricFusionEngine.paired_objective_difference(
            self._engine(held), {}, {}, "ndvi", "r2"
        )
        self.assertEqual(out["observed_diff"], 0.02)
        self.assertEqual((out["lower"], out["upper"]), (-0.02, 0.04))
        self.assertEqual(out["p_value"], 0.3)
        self.assertEqual(out["inference_n"], 50)

    def test_no_held_out_half_leaves_the_inference_empty(self):
        out = MetricFusionEngine.paired_objective_difference(
            self._engine(None), {}, {}, "ndvi", "r2"
        )
        self.assertEqual(out["observed_diff"], 0.02)
        for key in ("lower", "upper", "p_value", "inference_n"):
            self.assertIsNone(out[key], key)

    def test_the_family_summary_tolerates_a_missing_p_value(self):
        family = [
            {"standalone_channel": "ndvi", "observed_diff": 0.02, "p_value": None},
            {"standalone_channel": "gvi", "observed_diff": 0.01, "p_value": 0.04},
        ]
        head = runners._paired_family_headline(family, best_channel="gvi")
        self.assertIs(head, family[1])
        self.assertIsNone(family[0]["p_value_holm"])
        self.assertAlmostEqual(family[1]["p_value_holm"], 0.04)
        self.assertEqual({d["family_size"] for d in family}, {2})
        # Without an AIC/BIC verdict the largest difference heads the family.
        self.assertIs(runners._paired_family_headline(family, None), family[0])
        self.assertEqual(runners._fmt_or_na(None, ".4g"), "n/a")
        self.assertEqual(runners._fmt_or_na(float("nan"), ".4f"), "n/a")
        self.assertEqual(runners._fmt_or_na(0.04, ".2f"), "0.04")


if __name__ == "__main__":
    unittest.main()
