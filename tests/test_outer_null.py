"""The decision report of the repeated splits.

Pinned here:

- ``outer_report = "decision"`` gives the full report's held-out slopes,
  paired estimates and corrected p-values, without the per-split intervals
  the resampling gives;
- the run record names the report only when it is not the full one.
"""

import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (os.path.dirname(os.path.abspath(__file__)), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import geopandas as gpd  # noqa: F401

INPUTS = """
    mode = "cross-sectional"
    ndvi = "ndvi_2015.tif"
    gvi = "gvi.gpkg"
    """

COMPARATORS = {
    "comparators": ["ndvi_here", "noise", "patchy"],
    "comparators_primary": ["ndvi_here", "patchy"],
}


def _study() -> str:
    from test_outer_splits import _write_smooth_study

    folder = tempfile.mkdtemp(prefix="geofuse-null-")
    _write_smooth_study(folder)
    return folder


def _by(rows, *keys):
    return {tuple(r.get(k) for k in keys): r for r in rows}


class TestDecisionReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_fusion_run import run_study

        folder = _study()
        cls.full = run_study(folder, INPUTS, outer_splits=3, **COMPARATORS)[1]
        cls.decision = run_study(
            folder, INPUTS, outer_splits=3, outer_report="decision", **COMPARATORS
        )[1]

    def test_slopes_and_corrected_p_are_the_full_reports(self):
        full = _by(self.full["outer_splits"]["splits"], "split", "quantity", "exposure")
        dec = _by(
            self.decision["outer_splits"]["splits"], "split", "quantity", "exposure"
        )
        for key, row in full.items():
            if key[1] not in ("held_out_beta", "negative_control_beta"):
                continue
            for k in ("estimate", "ci_low", "ci_high", "p_value"):
                self.assertAlmostEqual(dec[key][k], row[k], places=10, msg=(key, k))
        fs = _by(self.full["outer_splits"]["summary"], "quantity", "exposure")
        ds = _by(self.decision["outer_splits"]["summary"], "quantity", "exposure")
        for key, entry in fs.items():
            if entry.get("corrected_p") is None:
                continue
            self.assertAlmostEqual(ds[key]["median"], entry["median"], places=10)
            self.assertAlmostEqual(
                ds[key]["corrected_p"], entry["corrected_p"], places=10, msg=key
            )

    def test_later_splits_keep_estimates_without_resampling(self):
        later = [r for r in self.decision["outer_splits"]["splits"] if r["split"] > 0]
        paired = [
            r
            for r in later
            if r["quantity"] in ("cgi_minus_channel", "cgi_minus_comparator")
        ]
        self.assertTrue(paired)
        for r in paired:
            self.assertIsNotNone(r["estimate"])
            self.assertIsNone(r["ci_low"])
            self.assertIsNone(r["p_value"])
        score = [r for r in later if r["quantity"] == "held_out_score"]
        self.assertTrue(score)
        for r in score:
            self.assertIsNotNone(r["estimate"])
            self.assertIsNone(r["ci_low"])
        self.assertEqual(self.decision["outer_splits"]["report"], "decision")

    def test_the_record_names_the_report_only_when_not_full(self):
        with open(
            os.path.join(self.full["report_dir"], "run_config.json"), encoding="utf-8"
        ) as f:
            full = json.load(f)
        with open(
            os.path.join(self.decision["report_dir"], "run_config.json"),
            encoding="utf-8",
        ) as f:
            decision = json.load(f)
        self.assertNotIn("outer_report", full)
        self.assertNotIn("outer_null_runs", full)
        self.assertEqual(decision["outer_report"], "decision")
        self.assertNotEqual(full["config_hash"], decision["config_hash"])


if __name__ == "__main__":
    unittest.main()
