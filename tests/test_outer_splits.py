"""Corrected resampled t-test across repeated held-out splits.

Under a null outcome it rejects at about its level over ten splits, while a
naive t over the same splits rejects several times as often.
"""

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (os.path.dirname(os.path.abspath(__file__)), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import geopandas as gpd  # noqa: F401
import numpy as np
from scipy import stats

from geofuse.statistical_testing import corrected_resampled_t


def _null_slopes(rng, n=1000, k=10, p=5, test_fraction=0.25):
    """Held-out slopes of the column picked on each split's training rows, for
    an outcome unrelated to every column."""
    cov = 0.6 * np.ones((p, p)) + 0.4 * np.eye(p)
    x = rng.normal(size=(n, p)) @ np.linalg.cholesky(cov).T
    y = rng.normal(size=n)
    m = int(test_fraction * n)
    slopes = []
    for _ in range(k):
        idx = rng.permutation(n)
        test, train = idx[:m], idx[m:]
        r = [abs(np.corrcoef(x[train, j], y[train])[0, 1]) for j in range(p)]
        j = int(np.argmax(r))
        slopes.append(np.polyfit(x[test, j], y[test], 1)[0])
    return np.array(slopes)


class TestCorrectedResampledT(unittest.TestCase):
    def test_the_null_rejection_rate_is_near_its_level(self):
        rng = np.random.default_rng(1)
        corrected = naive = 0
        runs = 400
        for _ in range(runs):
            b = _null_slopes(rng)
            corrected += corrected_resampled_t(b, 0.25)["p_value"] < 0.05
            t = b.mean() / (b.std(ddof=1) / np.sqrt(b.size))
            naive += 2 * stats.t.sf(abs(t), b.size - 1) < 0.05
        self.assertLessEqual(corrected / runs, 0.08)
        self.assertGreater(naive / runs, 2 * corrected / runs)

    def test_a_planted_effect_is_found(self):
        b = np.array([0.031, 0.028, 0.035, 0.026, 0.030, 0.033, 0.029, 0.027])
        test = corrected_resampled_t(b, 0.25, alternative="greater")
        self.assertLess(test["p_value"], 0.01)
        self.assertEqual(test["k"], 8)

    def test_too_few_values_give_no_test(self):
        self.assertIsNone(corrected_resampled_t([0.1], 0.25)["p_value"])
        self.assertIsNone(corrected_resampled_t([0.1, None, 0.1], 0.25)["t"])
        with self.assertRaises(ValueError):
            corrected_resampled_t([0.1, 0.2], 0.25, alternative="less")


if __name__ == "__main__":
    unittest.main()
