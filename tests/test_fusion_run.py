"""A small fusion job runs from its run file to its results.

The runner chains many reporting steps after the fits, and a defect in any of
them used to surface only at the end of a real run. Here one synthetic study
goes through ``plans.build`` and ``run_fusion`` in this process, with the
standalone studies on, at the smallest budgets the settings allow. Pinned
here:

- the run completes and writes its results;
- every CGI-vs-standalone row carries the held-out difference, interval and
  p-value, and its Holm-corrected p-value; the estimate lies inside its own
  interval, and the whole-data difference is a separate field;
- the headline channel is the one scoring best on the train+val pool, so the
  test outcomes never choose it, and its verdict is the held-out test;
- a missing held-out half leaves that inference empty instead of failing;
- a year-aware run (a greenery file per measurement year, one row per entity)
  is not a panel: with every visit in one year its posterior, splits and
  negative controls are the cross-sectional run's;
- comparator exposures are scored on the same rows without touching the
  composite, and written to their own tables; every entity's split is recorded.
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
from geofuse.fusion import PAIRED_HEADLINE_KEYS, MetricFusionEngine
from geofuse.jobs import fusion_outputs, plans, runners

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
    2015, per-year NDVI rasters and street-view points. Three finished exposure
    columns ride along: the NDVI at the point, noise, and the NDVI with gaps."""
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
    here = ndvi[row, col].astype(np.float64)
    patchy = here.copy()
    patchy[::7] = np.nan
    gpd.GeoDataFrame(
        {
            "outcome": 0.8 * ndvi[row, col] + rng.normal(0, 0.3, n),
            "grip": rng.normal(size=n),
            "age": rng.normal(60, 8, n),
            "sex": rng.choice(["f", "m"], n),
            "visit": ["2015-06-15"] * n,
            "ndvi_here": here,
            "noise": rng.normal(size=n),
            "patchy": patchy,
        },
        geometry=[Point(a, b) for a, b in zip(xs, ys, strict=True)],
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


def run_study(folder: str, inputs: str, **extra) -> tuple[_Context, dict]:
    """Run one fusion step over the study in ``folder``; ``inputs`` names the
    mode and the greenery files, ``extra`` adds ``run_fusion`` arguments.
    Returns the context and the outcome's bundle."""
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
    runners.run_fusion(ctx, **{**plans.build(spec).kwargs["run_kwargs"], **extra})
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
        }[mode.split(" with ")[0]]
        extra = (
            {
                "comparators": ["ndvi_here", "noise", "patchy"],
                "comparators_primary": ["ndvi_here", "patchy"],
            }
            if mode.endswith("with comparators")
            else {}
        )
        _RUNS[mode] = run_study(_RUNS["folder"], inputs, **extra)
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
            self.assertLessEqual(d["lower"], d["observed_diff"])
            self.assertLessEqual(d["observed_diff"], d["upper"])
            self.assertTrue(0.0 < d["p_value"] <= 1.0)
            self.assertGreaterEqual(d["p_value_holm"], d["p_value"])
            self.assertEqual(d["inference_n"], d["n"])
            self.assertLess(d["n"], d["n_all"])
            self.assertIsNotNone(d["observed_diff_all"])
            self.assertIn(d["verdict"], fusion_outputs.PAIRED_VERDICTS)
        self.assertIn(self.bundle["cgi_vs_standalone_paired"], family)

    def test_the_headline_channel_is_chosen_on_the_train_val_pool(self):
        pool = {
            ch: b["subset_scores"]["train"]["score"]
            for ch, b in self.bundle["standalones"].items()
        }
        head = self.bundle["cgi_vs_standalone_paired"]
        self.assertEqual(head["standalone_channel"], max(pool, key=pool.get))
        self.assertEqual(head["selected_on"], "train_val")
        aic_bic = self.bundle["cgi_vs_standalone_aic_bic"]
        self.assertEqual(aic_bic["best_channel"], head["standalone_channel"])
        self.assertTrue(aic_bic["in_sample"])


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


class TestComparatorRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx, cls.bundle = _run("year-aware with comparators")
        cls.plain = _run("year-aware")[1]
        cls.report = cls.bundle["report_dir"]

    def _csv(self, name):
        import pandas as pd

        return pd.read_csv(os.path.join(self.report, name))

    def test_the_composite_is_untouched_by_comparators(self):
        with_cmp, without = self.bundle, self.plain
        self.assertEqual(without["comparators"], {})
        self.assertFalse(
            os.path.exists(os.path.join(without["report_dir"], "comparators.csv"))
        )
        self.assertEqual(
            with_cmp["test_results"]["test_score"],
            without["test_results"]["test_score"],
        )
        for key in ("beta_mean", "beta_ci_low", "beta_ci_high", "weight_mean"):
            self.assertEqual(
                with_cmp["discovery_summary"][key],
                without["discovery_summary"][key],
                key,
            )

    def test_every_comparator_is_scored_and_gaps_cost_only_its_rows(self):
        cmp = self.bundle["comparators"]
        self.assertEqual(sorted(cmp), ["ndvi_here", "noise", "patchy"])
        for name, c in cmp.items():
            self.assertEqual(c["status"], "ok", name)
            self.assertIsNotNone(c["test"]["score"], name)
            self.assertIn("negative_controls", c)
        full, gappy = cmp["ndvi_here"]["n"], cmp["patchy"]["n"]
        for split in ("test", "train_val", "all"):
            self.assertLess(gappy[split], full[split], split)
        self.assertEqual(
            full["test"], self.bundle["cgi_effects"]["test"]["n"], "same held-out rows"
        )

    def test_holm_runs_over_the_primary_comparators(self):
        rows = {r["comparator"]: r for r in self.bundle["cgi_vs_comparator_paired"]}
        self.assertTrue(rows["noise"]["descriptive"])
        self.assertIsNone(rows["noise"]["p_value_holm"])
        for name in ("ndvi_here", "patchy"):
            self.assertEqual(rows[name]["family_size"], 2)
            self.assertGreaterEqual(rows[name]["p_value_holm"], rows[name]["p_value"])

    def test_the_tables_carry_the_comparators(self):
        table = self._csv("comparators.csv")
        self.assertEqual(
            list(table["exposure"].unique()), ["cgi", "ndvi_here", "noise", "patchy"]
        )
        self.assertEqual(len(table), 12)
        self.assertEqual(len(self._csv("cgi_vs_comparator_paired.csv")), 3)
        nc = self._csv("negative_controls.csv")
        self.assertEqual(set(nc["exposure"]), {"cgi", "standalone", "comparator"})
        self.assertEqual(
            set(nc.loc[nc["exposure"] == "comparator", "study"]),
            {"ndvi_here", "noise", "patchy"},
        )
        import json

        with open(
            os.path.join(self.report, "results_summary.json"), encoding="utf-8"
        ) as f:
            manifest = json.load(f)
        self.assertEqual(
            sorted(manifest["comparators"]), ["ndvi_here", "noise", "patchy"]
        )
        self.assertEqual(len(manifest["cgi_vs_comparator_paired"]), 3)

    def test_every_entity_names_its_split(self):
        import pandas as pd

        path = self.bundle["entity_table_path"]
        if path.endswith(".gpkg"):
            table = gpd.read_file(path, layer="cgi")
        else:
            table = pd.read_csv(path)
        counts = table["split"].value_counts()
        self.assertEqual(int(counts["test"]), self.bundle["cgi_effects"]["test"]["n"])
        self.assertEqual(
            int(counts["train_val"]), self.bundle["cgi_effects"]["train_val"]["n"]
        )


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

    def test_the_headline_is_the_held_out_difference(self):
        held = {
            "observed_diff": 0.01,
            "cgi_score": 0.04,
            "standalone_score": 0.03,
            "n": 50,
            "lower": -0.02,
            "upper": 0.04,
            "p_value": 0.3,
        }
        out = MetricFusionEngine.paired_objective_difference(
            self._engine(held), {}, {}, "ndvi", "r2"
        )
        self.assertEqual(out["observed_diff"], 0.01)
        self.assertEqual((out["cgi_score"], out["standalone_score"]), (0.04, 0.03))
        self.assertEqual((out["lower"], out["upper"]), (-0.02, 0.04))
        self.assertEqual(out["p_value"], 0.3)
        self.assertEqual((out["n"], out["inference_n"]), (50, 50))
        self.assertEqual(out["observed_diff_all"], 0.02)
        self.assertEqual(
            (out["cgi_score_all"], out["standalone_score_all"], out["n_all"]),
            (0.05, 0.03, 200),
        )

    def test_no_held_out_half_leaves_the_headline_empty(self):
        out = MetricFusionEngine.paired_objective_difference(
            self._engine(None), {}, {}, "ndvi", "r2"
        )
        self.assertEqual(out["observed_diff_all"], 0.02)
        self.assertEqual(out["standalone_channel"], "ndvi")
        for key in (*PAIRED_HEADLINE_KEYS, "inference_n"):
            self.assertIsNone(out[key], key)
        self.assertEqual(fusion_outputs.paired_verdict(out), "no inference")

    def test_the_family_summary_tolerates_a_missing_p_value(self):
        family = [
            {"standalone_channel": "ndvi", "observed_diff": 0.02, "p_value": None},
            {"standalone_channel": "gvi", "observed_diff": 0.01, "p_value": 0.04},
        ]
        head = runners._paired_family_headline(family, "gvi")
        self.assertIs(head, family[1])
        self.assertIsNone(family[0]["p_value_holm"])
        self.assertAlmostEqual(family[1]["p_value_holm"], 0.04)
        self.assertEqual({d["family_size"] for d in family}, {2})
        self.assertEqual(family[0]["verdict"], "no inference")
        self.assertEqual(family[1]["verdict"], "CGI better")
        self.assertEqual(runners._fmt_or_na(None, ".4g"), "n/a")
        self.assertEqual(runners._fmt_or_na(float("nan"), ".4f"), "n/a")
        self.assertEqual(runners._fmt_or_na(0.04, ".2f"), "0.04")

    def test_a_channel_without_a_row_gives_no_headline(self):
        family = [{"standalone_channel": "ndvi", "observed_diff": 0.02}]
        # The largest difference must not stand in: it was read off the test rows.
        self.assertIsNone(runners._paired_family_headline(family, "gvi"))
        self.assertIsNone(runners._paired_family_headline(family, None))


class TestComparisonChannel(unittest.TestCase):
    """The CGI's comparison channel and the verdict read from its row."""

    @staticmethod
    def _bundle(train, test):
        return {
            ch: {
                "subset_scores": {
                    "train": {"score": train[ch]},
                    "test": {"score": test[ch]},
                    "all": {"score": test[ch]},
                }
            }
            for ch in train
        }

    def test_the_channel_never_depends_on_the_test_rows(self):
        train = {"ndvi": 0.030, "gvi": 0.020}
        for test in ({"ndvi": 0.01, "gvi": 0.05}, {"ndvi": 0.05, "gvi": 0.01}):
            ch, scores = fusion_outputs.comparison_channel(
                self._bundle(train, test), ["ndvi", "gvi"], "r2"
            )
            self.assertEqual(ch, "ndvi")
            self.assertEqual(scores, train)

    def test_a_lower_is_better_objective_picks_the_smallest_score(self):
        bundle = self._bundle({"ndvi": 0.9, "gvi": 0.7}, {"ndvi": 0, "gvi": 0})
        ch, _ = fusion_outputs.comparison_channel(bundle, ["ndvi", "gvi"], "nrmse")
        self.assertEqual(ch, "gvi")

    def test_no_pool_score_means_no_channel(self):
        bundle = {"ndvi": {"subset_scores": {"train": {"score": float("nan")}}}}
        self.assertEqual(
            fusion_outputs.comparison_channel(bundle, ["ndvi", "gvi"], "r2"),
            (None, {}),
        )

    def test_a_composite_equal_to_its_channel_reads_as_no_difference(self):
        # Identical composites: every replicate's difference is 0, so the
        # one-sided bootstrap p is 1 and the interval is [0, 0].
        row = {"observed_diff": 0.0, "lower": 0.0, "upper": 0.0, "p_value": 1.0}
        self.assertEqual(fusion_outputs.paired_verdict(row), "no difference")

    def test_the_verdict_reads_the_held_out_test(self):
        verdict = fusion_outputs.paired_verdict
        base = {"observed_diff": 0.02, "lower": 0.001, "upper": 0.04}
        self.assertEqual(verdict({**base, "p_value": 0.01}), "CGI better")
        # The Holm p-value decides when there is one.
        self.assertEqual(
            verdict({**base, "p_value": 0.01, "p_value_holm": 0.08}), "no difference"
        )
        below = {"observed_diff": -0.02, "lower": -0.04, "upper": -0.001}
        self.assertEqual(verdict({**below, "p_value": 0.99}), "standalone better")


if __name__ == "__main__":
    unittest.main()
