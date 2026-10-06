"""The CSVs a fusion job leaves on disk are read without the app to explain them.

A column header is the only unit label these files carry, so a header that names
the wrong quantity is not cosmetic — it is the whole statement. The sweep's row
holds a t-statistic, not a value of the objective, and the discovery export must
name the cell the composite was built from, not the sweep's candidate.
"""

import csv
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from geofuse.jobs.fusion_outputs import _write_fusion_outputs


def _study(**over) -> dict:
    b = {
        "objective_metric": "r2",
        "averaged_params": {"ndvi_radius": 500, "__selection_method__": "x"},
        "test_results": {"test_score": 0.001, "test_ci": {"status": "ok"}},
        "subset_scores": {
            "train": {"score": 0.0003, "score_raw": 0.004, "n": 100},
            "val": {"score": 1.107, "score_raw": None, "n": 100},
            "test": {"score": 0.0010, "score_raw": 0.006, "n": 50},
            "all": {"score": 0.0004, "score_raw": 0.006, "n": 150},
        },
        "discovery_summary": {
            "channels": ["ndvi", "gvi"],
            "weight_labels": ["ndvi", "gvi"],
            "picked": [[600, "p25"], [600, "p10"]],
            "projected_pick": [[500, "p10"], [500, "mean"]],
            "radius_profile": [[0.5, 0.5], [0.5, 0.5]],
            "peak_radius_mean": [613.0, 609.0],
            "peak_radius_width_vs_prior": [0.97, 0.92],
            "peak_radius_width_ratio": [0.97, 0.92],
            "aggregator_informative": [["p10"], []],
            "weight_mean": [0.49, 0.51],
            "weight_ci_low": [0.02, 0.03],
            "weight_ci_high": [0.97, 0.98],
            "form": "linear",
            "sweep_score": 1.107,
            "holdout_gain": {},
            "discovery": {},
        },
        "direction_sign": -1,
    }
    b.update(over)
    return b


def _write(**over):
    out = tempfile.mkdtemp(prefix="geofuse-export-test-")
    _write_fusion_outputs(
        output_dir=out, label="Y", multi_outcome=False, objective_metric="r2",
        formula_name="weighted_average_gvi", cgi_bundle=_study(**over),
        standalones_bundle={}, aic_bic=None, covariate_impact=None,
        collinearity_report=None, run_config_record=None,
        log=lambda *a, **k: None,
    )
    return out


def _rows(out, name):
    with open(os.path.join(out, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestScoresCsvLabelsItsUnits(unittest.TestCase):
    def test_the_sweep_row_is_not_labelled_with_the_objective(self):
        # It holds a mean held-out |t| near 1 next to R² values near 0.0005;
        # sharing a metric name makes the column impossible to read or plot.
        rows = {r["subset"]: r for r in _rows(_write(), "scores.csv")}
        self.assertEqual(rows["val"]["metric"], "sweep_holdout_abs_t")
        self.assertNotEqual(rows["val"]["metric"], rows["test"]["metric"])

    def test_the_objective_rows_keep_the_objective_name(self):
        rows = {r["subset"]: r for r in _rows(_write(), "scores.csv")}
        for subset in ("train", "test", "all"):
            self.assertEqual(rows[subset]["metric"], "r2")


class TestDiscoveryCsvReportsWhatWasBuilt(unittest.TestCase):
    def test_the_cell_column_is_the_posterior_projection(self):
        rows = _rows(_write(), "discovery.csv")
        self.assertEqual([r["radius_m"] for r in rows], ["500", "500"])
        self.assertEqual([r["aggregator"] for r in rows], ["p10", "mean"])

    def test_the_sweep_cell_is_kept_in_its_own_columns(self):
        rows = _rows(_write(), "discovery.csv")
        self.assertEqual([r["sweep_radius_m"] for r in rows], ["600", "600"])
        self.assertEqual([r["sweep_aggregator"] for r in rows], ["p25", "p10"])

    def test_the_identifiability_numbers_travel_with_the_estimate(self):
        # A radius without the width comparison beside it reads as a finding
        # whether or not the data moved the prior at all.
        rows = _rows(_write(), "discovery.csv")
        self.assertEqual(rows[0]["peak_radius_width_vs_prior"], "0.97")
        self.assertEqual(rows[0]["aggregators_separated"], "p10")
        self.assertEqual(rows[1]["aggregators_separated"], "")

    def test_the_distances_of_influence_are_exported(self):
        summ = _study()["discovery_summary"]
        summ.update(r50_mean=[300.0, 410.0], r50_ci_low=[200.0, 350.0],
                    r50_ci_high=[380.0, 460.0], r90_mean=[520.0, 700.0],
                    r90_ci_low=[400.0, 640.0], r90_ci_high=[600.0, 760.0])
        rows = _rows(_write(discovery_summary=summ), "discovery.csv")
        self.assertEqual([r["r50_m"] for r in rows], ["300.0", "410.0"])
        self.assertEqual(rows[1]["r90_ci_high"], "760.0")

    def test_a_run_without_a_fitted_grid_falls_back_to_the_sweep_cell(self):
        summ = _study()["discovery_summary"]
        for key in ("projected_pick", "radius_profile", "peak_radius_mean",
                    "aggregator_informative"):
            summ.pop(key)
        rows = _rows(_write(discovery_summary=summ), "discovery.csv")
        self.assertEqual([r["radius_m"] for r in rows], ["600", "600"])
        self.assertEqual([r["peak_radius_m"] for r in rows], ["", ""])


if __name__ == "__main__":
    unittest.main()
