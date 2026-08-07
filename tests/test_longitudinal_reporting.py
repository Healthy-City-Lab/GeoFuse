"""Reporting-path regressions for the longitudinal mixed-effects results.

Two failure modes that both surface in the results UI as blank or "nan" tiles:

1. Wave indicators entered twice. ``_expand_period_controls`` folds the wave
   dummies into ``covariate_columns``, so every control matrix already carries
   them; a reporting helper that *also* passes ``wave_index`` makes the scorer
   one-hot the same levels a second time and the fixed-effects design goes
   exactly singular.
2. A design whose columns span very different magnitudes stalling lbfgs even
   though the model is identified.
"""

import os
import sys
import unittest
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geofuse import mixed_effects_scoring as mes  # noqa: E402
from geofuse.fusion import MetricFusionEngine  # noqa: E402

warnings.simplefilter("ignore")


def _panel(seed: int = 0, n_entities: int = 300, n_waves: int = 4):
    """Balanced panel with staggered entry, so wave and elapsed time differ."""
    rng = np.random.default_rng(seed)
    entity, wave_idx = [], []
    for e in range(n_entities):
        start = int(rng.integers(0, 4))
        for k in range(n_waves):
            entity.append(e)
            wave_idx.append(start + k)
    eid = np.asarray(entity)
    wi = np.asarray(wave_idx)
    t = (wi - pd.Series(wi).groupby(eid).transform("min").to_numpy()).astype(float)
    wave = np.asarray([f"{2011 + w}" for w in wi])
    g = rng.normal(size=len(eid))
    y = 0.4 * g + 0.1 * t + rng.normal(size=len(eid)) + rng.normal(size=n_entities)[eid]
    wave_dummies = (
        pd.get_dummies(pd.Series(wave), drop_first=True).astype(float).to_numpy()
    )
    return y, g, eid, t, wave, wave_dummies


class TestWaveIndicatorsEnteredOnce(unittest.TestCase):
    def test_duplicate_wave_indicators_break_the_fit(self):
        """The failure this guards against, stated directly."""
        y, g, eid, t, wave, wave_d = _panel()
        both = mes.score_mixedlm(
            "mixedlm_tstat", y, g, eid, t, wave_d, wave_index=wave, nan_on_fail=True
        )
        self.assertTrue(np.isnan(both))

        once = mes.score_mixedlm(
            "mixedlm_tstat", y, g, eid, t, wave_d, nan_on_fail=True
        )
        self.assertTrue(np.isfinite(once))
        self.assertGreater(once, 2.0)

    def test_engine_withholds_wave_labels_once_they_are_controls(self):
        engine = MetricFusionEngine.__new__(MetricFusionEngine)
        engine.longitudinal_spec = object()  # ``is_longitudinal`` reads this
        engine._wave_control_columns = []
        df = pd.DataFrame({"wave": ["2011", "2012", "2013"]})

        labels = engine._wave_labels_for_scoring(df)
        self.assertIsNotNone(labels)
        self.assertEqual(list(labels), ["2011", "2012", "2013"])
        self.assertIsNotNone(engine._wave_labels_for_scoring_array(labels))

        engine._wave_control_columns = ["wave=2012", "wave=2013"]
        self.assertIsNone(engine._wave_labels_for_scoring(df))
        self.assertIsNone(engine._wave_labels_for_scoring_array(labels))


class TestRescaledRetry(unittest.TestCase):
    def test_column_scaling_leaves_the_fit_unchanged(self):
        y, g, eid, t, _wave, _wd = _panel(seed=1, n_entities=200)
        X = np.column_stack([np.ones(len(y)), g, t])
        Z = np.column_stack([np.ones(len(y)), t])

        plain = mes._fit_mixedlm(y, X, eid, Z)
        self.assertIsNotNone(plain)
        # What the retry does: fit the scaled design, report it back unscaled.
        scales = np.array([1.0, 1e6, 1.0])
        rescaled = mes._RescaledFit(
            mes._fit_mixedlm_once(y, X / scales, eid, Z, reml=True), scales
        )

        for attr in ("fe_params", "bse_fe"):
            np.testing.assert_allclose(
                np.asarray(getattr(plain, attr)),
                np.asarray(getattr(rescaled, attr)),
                rtol=1e-6,
                atol=1e-9,
            )
        self.assertAlmostEqual(float(plain.scale), float(rescaled.scale), places=6)
        # Attributes the wrapper does not rewrite still reach the caller.
        self.assertAlmostEqual(
            mes._marginal_r2_from_fit(plain, X, Z),
            mes._marginal_r2_from_fit(rescaled, X, Z),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
