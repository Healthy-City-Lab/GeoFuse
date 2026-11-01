"""What a run reports about itself: the smooth's record.

The summary of a spatially adjusted run names the paired-test rule in force
and the number of spatial clusters.
"""

import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import geopandas  # noqa: F401  (pyproj before torchvision, see vision.py)
import numpy as np

from geofuse.fusion import MetricFusionEngine


def _engine(method, rule="own"):
    return MetricFusionEngine(
        target_file="dummy.geojson",
        cache_dir=tempfile.mkdtemp(),
        spatial_adjust_method=method,
        spatial_paired_smooth=rule,
    )


class SummaryRecordTests(unittest.TestCase):
    def test_the_record_names_the_rule_and_the_clusters(self):
        rng = np.random.default_rng(0)
        xy = np.vstack([rng.normal(size=(200, 2)) * 2_000 + c for c in (0, 1e6)])
        y = rng.normal(size=len(xy))
        for method, rule, want in (
            ("spatial_plus", "own", "own"),
            ("spatial_plus", "max", "max"),
            ("ks_aic", "own", "outcome"),
        ):
            eng = _engine(method, rule)
            eng._spatial_selection(xy, y, None, y)
            report = eng.spatial_selection_report()
            self.assertEqual(report["paired_rule"], want)
            self.assertEqual(report["clusters"], 2)

    def test_unadjusted_runs_have_no_record(self):
        self.assertIsNone(_engine("none").spatial_selection_report())


if __name__ == "__main__":
    unittest.main()
