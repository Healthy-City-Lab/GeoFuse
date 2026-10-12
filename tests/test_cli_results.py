"""``geofuse results`` and ``geofuse export``.

A fusion run is found by job id or by its results folder (paths a GUI run
recorded relative to its own folder are found again); the summary carries the
held-out score and interval, direction, each channel alone, the paired
difference, the AIC / BIC verdict and the weights; the per-entity table
exports as GeoPackage, CSV (no geometry) or GeoJSON (WGS 84).
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from geofuse.cli import commands as cli
from geofuse.jobs.state import StateDir


def _bundle(artifacts: str) -> dict:
    rel = os.path.join("output_results", "fusion", os.path.basename(artifacts))
    return {
        "mode": "single",
        "ordered_labels": ["mood"],
        "objective_metric": "spearman",
        "cgi_formula": "weighted_average",
        "cgi_effects": {
            "test": {"score": 0.061, "p_value": 0.001, "n": 400},
            "all": {"score": 0.047, "lower": 0.035, "upper": 0.065},
        },
        "test_results": {"test_ci": {"lower": 0.046, "upper": 0.089, "n": 400}},
        "direction_sign": -1,
        "standalones": {"ndvi": {"test_results": {"test_score": 0.062}}},
        "cgi_vs_standalone_paired_family": [
            {"standalone_channel": "ndvi", "observed_diff": -0.015,
             "lower": -0.029, "upper": 0.001, "p_value_holm": 0.97}
        ],
        "cgi_vs_standalone_aic_bic": {
            "ok": True, "best_channel": "ndvi", "verdict": "not justified",
            "delta_aic": -2.3, "delta_bic": -15.6,
        },
        "averaged_params": {"ndvi_weight": 20, "gvi_weight": 80,
                            "ndvi_radius": 700, "gvi_radius": 400, "__x__": 1},
        "report_dir": os.path.join(rel, "study_results"),
        "entity_table_path": os.path.join(rel, "composite_entities.gpkg"),
    }


class _Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.artifacts = os.path.join(self.dir, "fusion", "20260101T000000__abcd1234")
        os.makedirs(self.artifacts)
        with open(os.path.join(self.artifacts, "results_bundle.json"), "w") as f:
            json.dump(_bundle(self.artifacts), f)
        gpd.GeoDataFrame(
            {"polygon_id": [0, 1], "cgi": [0.4, 0.7]},
            geometry=[Point(500000, 5650000), Point(501000, 5651000)],
            crs="EPSG:32612",
        ).to_file(
            os.path.join(self.artifacts, "composite_entities.gpkg"),
            driver="GPKG",
            layer="cgi",
        )
        self.state_dir = os.path.join(self.dir, "state")

    def tearDown(self):
        self.tmp.cleanup()

    def geofuse(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = cli.main(["--state-dir", self.state_dir, *argv])
        return code, out.getvalue()


class ResultsTests(_Case):
    def test_the_headline_of_a_results_folder(self):
        code, out = self.geofuse("results", self.artifacts)
        self.assertEqual(code, 0, out)
        for part in (
            "Outcome mood · objective spearman · formula weighted_average",
            "0.0610  [0.0460, 0.0890]  n=400  p=0.001",
            "negative: more greenery, lower outcome",
            "NDVI alone       0.0620",
            "CGI − NDVI       -0.0150  [-0.0290, 0.0010]  Holm p=0.97",
            "CGI vs NDVI alone: not justified (ΔAIC -2.3, ΔBIC -15.6)",
            "NDVI 20, GVI 80 (of 100)",
            "NDVI 700 m, GVI 400 m",
            os.path.join(self.artifacts, "study_results"),
        ):
            self.assertIn(part, out)

    def test_a_job_id_finds_its_bundle_and_unfinished_runs_say_so(self):
        state = StateDir(self.state_dir)
        job = state.create_job(
            {"step": "fusion.fit", "kind": "fusion", "name": "fit"},
            run_id="r1",
            depends_on=[],
        )
        code, out = self.geofuse("results", job[:4])
        self.assertEqual(code, 2)
        self.assertIn("no results yet", out)
        bundle = os.path.join(self.artifacts, "results_bundle.json")
        state.claim(job, "client")
        state.write_status(job, {"status": "completed", "outputs": [bundle]})
        code, out = self.geofuse("results", job, "--json")
        self.assertEqual(code, 0, out)
        self.assertEqual(json.loads(out)["mood"]["cgi_formula"], "weighted_average")
        empty = os.path.join(self.dir, "fusion", "unfinished")
        os.makedirs(empty)
        code, out = self.geofuse("results", empty)
        self.assertIn("did not finish", out)


class ExportTests(_Case):
    def test_the_table_exports_in_each_format(self):
        csv = os.path.join(self.dir, "x", "cgi.csv")
        self.assertEqual(self.geofuse("export", self.artifacts, "-o", csv)[0], 0)
        self.assertEqual(list(pd.read_csv(csv).columns), ["polygon_id", "cgi"])
        geojson = os.path.join(self.dir, "cgi.geojson")
        self.assertEqual(self.geofuse("export", self.artifacts, "-o", geojson)[0], 0)
        self.assertEqual(gpd.read_file(geojson).crs.to_epsg(), 4326)
        gpkg = os.path.join(self.dir, "cgi.gpkg")
        self.assertEqual(self.geofuse("export", self.artifacts, "-o", gpkg)[0], 0)
        self.assertEqual(gpd.read_file(gpkg, layer="cgi").crs.to_epsg(), 32612)
        code, out = self.geofuse("export", self.artifacts)
        self.assertTrue(out.strip().endswith("composite_entities.gpkg"))
        code, out = self.geofuse("export", self.artifacts, "-o", "cgi.xlsx")
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
