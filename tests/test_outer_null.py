"""The decision report of the repeated splits and its null calibration.

Pinned here:

- ``outer_report = "decision"`` gives the full report's held-out slopes,
  paired estimates and corrected p-values, without the per-split intervals
  the resampling gives;
- null outcomes go through the whole rule (search and every split) inside
  one job, free and spatial, with one row per null and tested quantity, a
  rate per kind with its exact interval, the variogram match table and the
  tables written;
- a rerun reuses the finished nulls, and a pool of workers gives the nulls the
  job's own process gives;
- the summary places the real outcome among the nulls, and the sequential
  rule keeps only the nulls it decided on;
- the spatial null is grouped by any target column (text or a numeric code
  of the same areas give the same nulls, a categorical covariate works), or
  in an unadjusted study by the gap clusters of ``spatial_adjust_eps_m``; a
  missing column is refused when the run is planned;
- without nulls the run record carries none of the new settings, and the
  results panel draws the rates only when there are some.
"""

import csv
import json
import os
import shutil
import sys
import tempfile
import textwrap
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (os.path.dirname(os.path.abspath(__file__)), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import geopandas as gpd
import numpy as np

from geofuse.jobs import outer_null

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


class TestNullCalibration(unittest.TestCase):
    """Two nulls of each kind over two splits, in the job's process, then
    rerun, then through a pool of two workers."""

    SETTINGS = {
        "outer_splits": 2,
        "outer_null_runs": 2,
        "outer_null_kind": "both",
        "outer_null_workers": 1,
    }

    @classmethod
    def setUpClass(cls):
        from test_fusion_run import run_study

        cls.folder = _study()
        cls.bundle = run_study(cls.folder, INPUTS, **cls.SETTINGS)[1]
        cls.null = cls.bundle["outer_null"]
        cls.resume = os.path.join(cls.folder, "out", "fusion_cache", "outer_null")
        (cls.resume_file,) = [
            f for f in os.listdir(cls.resume) if f.endswith(".jsonl")
        ]
        cls.rerun = run_study(cls.folder, INPUTS, **cls.SETTINGS)[1]["outer_null"]
        shutil.rmtree(cls.resume)
        cls.pooled = run_study(
            cls.folder, INPUTS, **{**cls.SETTINGS, "outer_null_workers": 2}
        )[1]["outer_null"]

    def test_every_null_reports_every_tested_quantity(self):
        self.assertEqual(self.null["kinds"], ["spatial", "free"])
        self.assertTrue(self.null["complete"])
        nulls = {(r["null_kind"], r["null"]) for r in self.null["rows"]}
        self.assertEqual(nulls, {(k, i) for k in ("spatial", "free") for i in (1, 2)})
        quantities = {r["quantity"] for r in self.null["rows"]}
        self.assertIn("held_out_beta", quantities)
        self.assertIn("cgi_minus_channel", quantities)
        for r in self.null["rows"]:
            self.assertEqual(r["n_splits"], 2)
            p = r["corrected_p"]
            self.assertEqual(r["hit"], p is not None and p < 0.05)
            if r["quantity"] == "held_out_beta":
                self.assertIsNotNone(p)
        spatial = [r for r in self.null["rows"] if r["null_kind"] == "spatial"]
        self.assertTrue(all(r["surrogate_k"] for r in spatial))

    def test_the_rate_has_its_exact_interval(self):
        head = [
            e
            for e in self.null["summary"]
            if (e["quantity"], e["exposure"]) == ("held_out_beta", "cgi")
        ]
        self.assertEqual({e["null_kind"] for e in head}, {"spatial", "free"})
        for e in head:
            self.assertEqual(e["finished"], 2)
            self.assertAlmostEqual(e["rate"], e["hits"] / 2)
            self.assertLessEqual(e["ci_low"], e["rate"])
            self.assertGreaterEqual(e["ci_high"], e["rate"])
            self.assertIsNotNone(e["observed_corrected_p"])
            self.assertIn(e["calibrated_p"], (1 / 3, 2 / 3, 1.0))

    def test_the_variogram_table_and_files_are_written(self):
        table = self.null["variogram"]
        self.assertTrue(table)
        self.assertTrue(all(r["n_surrogates"] == 2 for r in table if r["n_pairs"]))
        report = self.bundle["report_dir"]
        for name in ("null_outer.csv", "null_summary.csv", "null_variogram.csv"):
            self.assertTrue(os.path.isfile(os.path.join(report, name)), name)
        with open(os.path.join(report, "null_summary.csv"), encoding="utf-8") as f:
            self.assertIn("calibrated_p", next(csv.reader(f)))
        with open(os.path.join(report, "results_summary.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["outer_null"]["kinds"], ["spatial", "free"])
        self.assertNotIn("records", manifest["outer_null"])

    def test_a_rerun_reuses_the_finished_nulls(self):
        with open(os.path.join(self.resume, self.resume_file), encoding="utf-8") as f:
            self.assertEqual(len(f.readlines()), 4)
        self.assertEqual(self.rerun["rows"], self.null["rows"])

    def test_a_pool_gives_the_same_nulls(self):
        self.assertEqual(self.pooled["workers"], 2)
        first = _by(self.null["rows"], "null_kind", "null", "quantity", "exposure")
        pooled = _by(self.pooled["rows"], "null_kind", "null", "quantity", "exposure")
        self.assertEqual(set(first), set(pooled))
        for key, row in first.items():
            self.assertAlmostEqual(pooled[key]["median"], row["median"], places=8)
            self.assertAlmostEqual(
                pooled[key]["corrected_p"], row["corrected_p"], places=8
            )



def _record(index, p, kind="free", median=0.1):
    return {
        "kind": kind,
        "index": index,
        "n_splits": 10,
        "entries": [
            {
                "quantity": "held_out_beta",
                "exposure": "cgi",
                "control": None,
                "median": median,
                "corrected_t": 1.0,
                "corrected_p": p,
            }
        ],
    }


class TestSummary(unittest.TestCase):
    def test_the_real_outcome_is_placed_among_the_nulls(self):
        records = [_record(i, p) for i, p in enumerate([0.01, 0.2, 0.5, 0.9])]
        observed = [
            {
                "quantity": "held_out_beta",
                "exposure": "cgi",
                "alternative": "two-sided",
                "corrected_p": 0.3,
            }
        ]
        (row,) = outer_null.null_summary(records, observed, {"free": "ran all"})
        self.assertEqual((row["finished"], row["hits"]), (4, 1))
        self.assertAlmostEqual(row["rate"], 0.25)
        self.assertAlmostEqual(row["calibrated_p"], (1 + 2) / (1 + 4))
        self.assertEqual(row["decision"], "ran all")

    def test_the_rule_decides_once_the_first_nulls_are_done(self):
        rule = {"after": 3, "low": 0, "high": 2}
        done = [_record(0, 0.5), _record(2, 0.5)]
        self.assertIsNone(outer_null._decide(done, rule))
        done.append(_record(1, 0.5))
        self.assertEqual(outer_null._decide(done, rule), "not inflated")
        hits = [_record(i, 0.01) for i in range(3)]
        self.assertEqual(outer_null._decide(hits, rule), "inflated")
        mixed = [_record(0, 0.01), _record(1, 0.5), _record(2, 0.5)]
        self.assertIsNone(outer_null._decide(mixed, rule))
        self.assertIsNone(outer_null._decide(hits, {}))



class TestResultsPanel(unittest.TestCase):
    """The Fusion tab draws the null rates, and nothing without them."""

    def setUp(self):
        ui = os.path.join(ROOT, "ui")
        if ui not in sys.path:
            sys.path.insert(0, ui)
        from tabs import fusion as fusion_tab
        from test_posterior_panel import _Recorder

        self.tab = fusion_tab
        self._real_st = fusion_tab.st
        self.rec = _Recorder()
        fusion_tab.st = self.rec

    def tearDown(self):
        self.tab.st = self._real_st

    def test_the_rate_is_drawn(self):
        records = [_record(i, p) for i, p in enumerate([0.01, 0.2, 0.5])]
        view = {
            "outer_null": {
                "summary": outer_null.null_summary(records, [], {}),
                "groups": {"source": "gap clusters", "n_groups": 4},
                "complete": True,
            }
        }
        self.tab._render_outer_null(view)
        text = self.rec.text()
        self.assertIn("Null calibration of the held-out decision", text)
        self.assertIn("gap clusters", text)
        frame = next(v for n, v in self.rec.calls if n == "dataframe")
        self.assertEqual(frame.iloc[0]["Rejected"], "1/3")
        self.assertEqual(frame.iloc[0]["Null"], "Freely permuted")

    def test_nothing_without_nulls(self):
        self.tab._render_outer_null({})
        self.assertEqual(self.rec.calls, [])


def _area_study() -> str:
    """The smooth study with six areas as text and as a numeric code, neither a
    covariate."""
    folder = _study()
    path = os.path.join(folder, "cohort.gpkg")
    cohort = gpd.read_file(path)
    rank = cohort.geometry.x.rank(method="first").to_numpy()
    code = np.minimum((6 * (rank - 1) / len(cohort)).astype(int), 5) + 1
    cohort["area"] = [f"area {c}" for c in code]
    cohort["area_code"] = code
    cohort.to_file(path, driver="GPKG")
    return folder


NULL_SETTINGS = """
    outer_splits = 2
    outer_null_runs = 1
    outer_null_kind = "spatial"
    outer_null_workers = 1
    """


class TestNullGroups(unittest.TestCase):
    """The spatial null grouped by a target column that is not a covariate,
    by a categorical covariate, and by gap clusters in an unadjusted study."""

    @classmethod
    def setUpClass(cls):
        from test_fusion_run import run_study

        cls.folder = _area_study()

        def run(extra: str):
            bundle = run_study(cls.folder, INPUTS + NULL_SETTINGS + extra)[1]
            return bundle["outer_null"]

        cls.area = run('outer_null_group = "area"\n')
        cls.code = run('outer_null_group = "area_code"\n')
        cls.sex = run('outer_null_group = "sex"\n')
        cls.gaps = run("spatial_adjust_eps_m = 150\n")

    def test_a_text_area_column_groups_the_null(self):
        groups = self.area["groups"]
        self.assertEqual(groups["source"], "column area")
        self.assertEqual(groups["n_groups"], 6)
        self.assertEqual(groups["n_missing"], 0)
        self.assertTrue(self.area["complete"])

    def test_text_labels_and_a_numeric_code_give_the_same_nulls(self):
        a = _by(self.area["rows"], "null_kind", "null", "quantity", "exposure")
        b = _by(self.code["rows"], "null_kind", "null", "quantity", "exposure")
        self.assertEqual(set(a), set(b))
        for key, row in a.items():
            self.assertEqual(b[key]["median"], row["median"], key)
            self.assertEqual(b[key]["corrected_p"], row["corrected_p"], key)

    def test_a_categorical_covariate_groups_the_null(self):
        groups = self.sex["groups"]
        self.assertEqual(groups["source"], "column sex")
        self.assertEqual(groups["n_groups"], 2)
        self.assertTrue(self.sex["rows"])
        self.assertTrue(self.sex["complete"])

    def test_an_unadjusted_study_uses_gap_clusters(self):
        self.assertEqual(self.gaps["groups"]["source"], "gap clusters at 150 m")
        self.assertGreater(self.gaps["groups"]["n_groups"], 1)
        self.assertTrue(self.gaps["complete"])
        self.assertEqual({r["null"] for r in self.gaps["rows"]}, {1})

    def test_a_missing_column_is_refused_when_planned(self):
        from geofuse import runfile
        from geofuse.jobs import plans

        doc = (
            f"output_dir = {os.path.join(self.folder, 'out')!r}\n"
            "[[fusion]]\n"
            'name = "t"\n'
            'target = "cohort.gpkg"\n'
            'outcomes = ["outcome"]\n' + textwrap.dedent(INPUTS + NULL_SETTINGS)
            + 'outer_null_group = "no_such_column"\n'
        )
        path = os.path.join(self.folder, "missing.toml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc)
        with self.assertRaisesRegex(plans.PlanError, "no_such_column"):
            plans.expand(runfile.load(path, ()))


class TestGroupRecord(unittest.TestCase):
    def test_sizes_and_warnings(self):
        rec = outer_null.group_record(np.array(["a"] * 8 + ["b"] * 2), "column x")
        self.assertEqual((rec["n_groups"], rec["smallest"]), (2, 2))
        self.assertAlmostEqual(rec["largest_share"], 0.8)
        text = outer_null.group_warning(rec)
        self.assertIn("only 2 groups", text)
        self.assertIn("80%", text)
        even = outer_null.group_record(np.repeat(np.arange(8), 5), "gaps")
        self.assertIsNone(outer_null.group_warning(even))

    def test_missing_labels_form_one_counted_group(self):
        import pandas as pd

        frame = pd.DataFrame(
            {
                "polygon_id": [0, 1, 2, 3],
                "target": [1.0, 2.0, 3.0, 4.0],
                "_cx": [0.0, 1.0, 2.0, 3.0],
                "_cy": [0.0, 0.0, 0.0, 0.0],
                "area": ["n", None, "s", np.nan],
            }
        )
        engine = type("E", (), {"_coverage_keep_index": None})()
        ent = outer_null.Entities(engine, frame, "area")
        self.assertEqual(ent.n_missing_group, 2)
        missing = outer_null.MISSING_GROUP
        self.assertEqual(list(ent.group), ["n", missing, "s", missing])

if __name__ == "__main__":
    unittest.main()
