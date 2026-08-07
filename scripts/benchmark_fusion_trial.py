"""Benchmark the fusion engine's per-trial scoring hot paths.

Times the pieces one Optuna trial pays per scored subset -- the partial
distance correlation (stock ``dcor`` vs the cached-side scorer) and the
pixel-to-entity collapse (factorizing legacy vs precomputed codes) -- across
entity counts and pixel-row counts, so future changes have a reference.

Run with the project interpreter::

    python scripts/benchmark_fusion_trial.py
"""

import os
import sys
import time
from collections import OrderedDict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from geofuse import pdcor
from geofuse.fusion import MetricFusionEngine


def _time(fn, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_pdcor():
    import dcor

    rng = np.random.default_rng(0)
    print("partial distance correlation -- one score call")
    print(f"{'n':>7} {'stock dcor':>12} {'cached side':>12} {'speedup':>8}")
    for n in (1_000, 4_000, 8_000):
        t = rng.normal(size=n)
        c = rng.normal(size=n)
        z = rng.normal(size=(n, 4))
        t2 = t.reshape(-1, 1)
        c2 = c.reshape(-1, 1)

        stock = _time(lambda: dcor.partial_distance_correlation(t2, c2, z))
        cache: OrderedDict = OrderedDict()
        pdcor.partial_distance_correlation_cached(t2, c2, z, cache)  # warm side
        cached = _time(
            lambda: pdcor.partial_distance_correlation_cached(t2, c2, z, cache)
        )
        print(
            f"{n:>7,} {stock * 1e3:>10.1f}ms {cached * 1e3:>10.1f}ms "
            f"{stock / cached:>7.1f}x"
        )


def bench_collapse():
    rng = np.random.default_rng(1)
    print("\npixel-to-entity mean collapse -- one trial's train+val collapses")
    print(
        f"{'pixel rows':>11} {'legacy factorize':>17} {'cached codes':>13} "
        f"{'speedup':>8}"
    )
    for n_pixels in (10_000, 200_000):
        n_entities = max(50, n_pixels // 200)
        pid = np.array(
            [f"e{p:05d}|w1" for p in rng.integers(0, n_entities, size=n_pixels)]
        )
        values = rng.normal(size=n_pixels)
        legacy = _time(
            lambda: MetricFusionEngine._collapse_to_entities(values, pid, None, "mean")
        )
        codes, uniq = pd.factorize(pid, sort=True)
        codes = codes.astype(np.int64)
        fast = _time(
            lambda: MetricFusionEngine._collapse_mean_from_codes(
                values, codes, len(uniq), None
            )
        )
        print(
            f"{n_pixels:>11,} {legacy * 1e3:>15.1f}ms {fast * 1e3:>11.1f}ms "
            f"{legacy / fast:>7.1f}x"
        )


if __name__ == "__main__":
    bench_pdcor()
    bench_collapse()
