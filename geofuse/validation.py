"""Plasmode recovery: does the index find a CGI planted in your own exposure?

A pipeline that recovers the right radius, aggregator and weights on independent
synthetic columns can still fail on real exposures, whose columns are strongly
correlated across radii, statistics and channels. A plasmode keeps the real
exposure tensor and the real covariates, plants a known CGI into a simulated
outcome, and runs the same sweep and posterior as
:meth:`MetricFusionEngine.fit_bayesian_index` on it — many times, under
several truths:

- ``S0`` null (no effect): how often the effect interval excludes zero;
- ``S1`` one channel at a small radius, mean statistic;
- ``S2`` two channels, weights 0.7 / 0.3, at different radii;
- ``S3`` a synergy truth (powers and a pairwise product);
- ``S4`` a truth between rungs (an even blend of two adjacent rungs).

The outcome is ``y* = β · z(CGI_true) + covariate part + ε`` with ``β`` set so
the planted index explains a chosen partial R² (about 0.001 is realistic in
this literature). The CGI is built on the residualised, standardised tensor —
the scale the model itself works on — so the true R50 is computed through the
same per-column SDs the posterior's is.

Reported per scenario and effect size: the rate at which the effect interval
excludes zero (the false-positive rate under ``S0``, power otherwise) with an
exact interval, effect bias and interval coverage, R50 absolute log error and
coverage, the aggregator blend's total-variation error, channel-weight coverage
(interior true weights only — 0 and 1 lie on the simplex boundary) and absolute
error, and how often the form is recovered.

Each replicate is one process (numpyro recompiles per fit), the tensor is
memory-mapped rather than pickled, seeds come from ``SeedSequence`` and
``crc32`` (never Python's per-process ``hash``), and nothing inside a replicate
opens a pool of its own.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import zlib
from dataclasses import asdict, dataclass

import numpy as np

from . import bayesian_index as bi
from . import parallel

#: Light sampler and sweep settings, declared in every result.
LIGHT_SETTINGS = {"draws": 300, "warmup": 300, "chains": 2, "sweep_splits": 10}


@dataclass
class Truth:
    """One planted CGI: per-channel kernel and aggregator, weights, form."""

    name: str
    weights: tuple                 # per channel, then per pair for synergy
    kernel: tuple                  # (channel, rung) rung weights, rows sum to 1
    aggregator: tuple              # (channel, stat) blend, rows sum to 1
    form: str = "linear"
    powers: tuple | None = None
    null: bool = False


def _one_hot(n, i) -> list[float]:
    out = [0.0] * n
    out[i] = 1.0
    return out


def default_scenarios(n_channels, radii, stats, radius_idx=None) -> list[Truth]:
    """S0-S4 for this ladder. S2 and S3 need two channels and are skipped
    otherwise; S4 needs three rungs on the first channel."""
    nr, ns = len(radii), len(stats)
    if radius_idx is None:
        radius_idx = [list(range(nr))] * n_channels
    s_mean = list(stats).index("mean") if "mean" in stats else 0
    agg = [_one_hot(ns, s_mean) for _ in range(n_channels)]
    ladder0 = list(radius_idx[0])
    small = ladder0[min(1, len(ladder0) - 1)]

    def kern(rungs_by_channel):
        return tuple(tuple(row) for row in rungs_by_channel)

    def solo(c, rung):
        rows = [_one_hot(nr, radius_idx[k][0]) for k in range(n_channels)]
        rows[c] = _one_hot(nr, rung)
        return rows

    w_first = tuple([1.0] + [0.0] * (n_channels - 1))
    out = [
        Truth("S0", w_first, kern(solo(0, small)), kern(agg), null=True),
        Truth("S1", w_first, kern(solo(0, small)), kern(agg)),
    ]
    if n_channels >= 2:
        ladder1 = list(radius_idx[1])
        far = ladder1[max(len(ladder1) - 2, 0)]
        rows = solo(0, small)
        rows[1] = _one_hot(nr, far)
        w2 = tuple([0.7, 0.3] + [0.0] * (n_channels - 2))
        out.append(Truth("S2", w2, kern(rows), kern(agg)))
        n_pairs = n_channels * (n_channels - 1) // 2
        w3 = [0.2, 0.2] + [0.0] * (n_channels - 2) + [0.6] + [0.0] * (n_pairs - 1)
        out.append(Truth("S3", tuple(w3), kern(rows), kern(agg), form="synergy",
                         powers=tuple([0.6] * n_channels)))
    if len(ladder0) >= 3:
        rows = solo(0, ladder0[1])
        rows[0] = [0.0] * nr
        rows[0][ladder0[1]] = rows[0][ladder0[2]] = 0.5
        out.append(Truth("S4", w_first, kern(rows), kern(agg)))
    return out


def planted_index(Xr, truth: Truth, channel_index) -> np.ndarray:
    """The truth's CGI on a prepped tensor, standardised.

    Mirrors the model's contraction: each channel is the kernel- and
    aggregator-weighted sum of its columns, standardised, then combined by the
    truth's form.
    """
    grid = np.nan_to_num(np.asarray(Xr)[:, list(channel_index)])
    k, a = np.asarray(truth.kernel), np.asarray(truth.aggregator)
    v = np.einsum("ncrs,cr,cs->nc", grid, k, a)
    v = (v - v.mean(0)) / np.maximum(v.std(0), 1e-12)
    w = np.asarray(truth.weights, dtype=np.float64)
    if truth.form == "synergy":
        e = bi.SynergyFit(w, np.asarray(truth.powers), v.min(0), v.max(0)).apply(v)
    else:
        e = v @ w[: v.shape[1]]
    return (e - e.mean()) / np.maximum(e.std(), 1e-12)


def true_r50(truth: Truth, radii, column_sd, channel_index, stats) -> np.ndarray:
    """R50 per channel on the raw scale, through the same SDs as the posterior."""
    rung_sd = bi.rung_sd_from(column_sd, channel_index, stats)
    k = bi.raw_rung_weights(np.asarray(truth.kernel)[None], rung_sd)
    return bi.distance_quantiles(k, radii, q=(0.5,))[0.5][0]


def _seed(seed: int, name: str, pr2: float, rep: int) -> int:
    seq = np.random.SeedSequence(
        [int(seed), zlib.crc32(name.encode()), int(round(pr2 * 1e7)), int(rep)])
    return int(seq.generate_state(1)[0] % (2 ** 31 - 1))


def _write_memmap(arr) -> str:
    fd, path = tempfile.mkstemp(suffix=".geofuse-plasmode")
    os.close(fd)
    mm = np.memmap(path, dtype=np.float64, mode="w+", shape=arr.shape)
    mm[:] = arr
    mm.flush()
    del mm
    return path


def _replicate(task) -> dict:
    """One plasmode replicate: simulate, prep, sweep, grid posterior, score."""
    (x_path, x_shape, cov_path, cov_shape, z, gamma, truth, pr2, seed, cfg) = task
    X = np.asarray(np.memmap(x_path, dtype=np.float64, mode="r", shape=x_shape))
    cov = (None if cov_path is None else
           np.asarray(np.memmap(cov_path, dtype=np.float64, mode="r", shape=cov_shape)))
    rng = np.random.default_rng(seed)
    beta = 0.0 if truth.null else math.sqrt(pr2 / (1.0 - pr2))
    y = beta * z + rng.normal(size=len(z))
    if cov is not None:
        y = y + cov @ gamma

    yr, Xr, info = bi.prep(X, y, cov, channel_index=cfg["channel_index"],
                           return_info=True)
    res = bi.sweep(Xr, cfg["radii"], cfg["stats"], yr, channels=cfg["channels"],
                   channel_index=cfg["channel_index"], radius_idx=cfg["radius_idx"],
                   forms=cfg["forms"], splits=cfg["sweep_splits"], seed=seed,
                   workers=1)
    post, _, _ = bi.grid_posterior(
        Xr, yr, channels=cfg["channels"], channel_index=cfg["channel_index"],
        radii=cfg["radii"], stats=cfg["stats"], radius_idx=cfg["radius_idx"],
        form=res.form, picked=res.picked, radius_kernel=cfg["radius_kernel"],
        aggregator=cfg["aggregator"], draws=cfg["draws"], warmup=cfg["warmup"],
        chains=cfg["chains"], seed=seed, column_sd=info["column_sd"],
    )
    s = post.summary()
    b_true = math.sqrt(pr2) if not truth.null else 0.0
    rec = {
        "scenario": truth.name, "partial_r2": pr2, "seed": seed,
        "form_true": truth.form, "form": res.form,
        "beta_true": b_true, "beta_mean": s["beta_mean"],
        "beta_ci_low": s["beta_ci_low"], "beta_ci_high": s["beta_ci_high"],
        "excluded_zero": bool(s["beta_ci_low"] > 0 or s["beta_ci_high"] < 0),
        "rhat_max": s.get("rhat_max"), "divergences": s.get("divergences"),
        "channels": {},
    }
    w_true = np.asarray(truth.weights)
    w_lo, w_hi = s["weight_ci_low"], s["weight_ci_high"]
    if not truth.null and len(w_lo) == len(w_true):
        # A weight of exactly 0 or 1 sits on the simplex boundary, which a
        # continuous interval never contains, so coverage is read only for
        # interior true weights; the absolute error covers every one.
        rec["weight_covered"] = [bool(lo <= wt <= hi)
                                 for lo, wt, hi in zip(w_lo, w_true, w_hi)
                                 if 0.0 < wt < 1.0]
        rec["weight_abs_error"] = float(np.mean(
            np.abs(np.asarray(s["weight_mean"]) - w_true)))
    r50_true = true_r50(truth, cfg["radii"], info["column_sd"],
                        cfg["channel_index"], cfg["stats"])
    agg_true = np.asarray(truth.aggregator)
    for c, ch in enumerate(cfg["channels"]):
        if truth.null or w_true[c] <= 0:
            continue
        row = {"r50_true": float(r50_true[c])}
        if "r50_mean" in s:
            row.update(r50_mean=s["r50_mean"][c], r50_ci_low=s["r50_ci_low"][c],
                       r50_ci_high=s["r50_ci_high"][c])
        if "aggregator_mean" in s:
            row["aggregator_tv"] = 0.5 * float(
                np.abs(np.asarray(s["aggregator_mean"][c]) - agg_true[c]).sum())
        rec["channels"][ch] = row
    return rec


def _rate_rows(name, pr2, metric, hits, n):
    lo, hi = bi.clopper_pearson(int(hits), int(n))
    return {"scenario": name, "partial_r2": pr2, "metric": metric,
            "value": hits / n if n else float("nan"), "ci_low": lo, "ci_high": hi,
            "n": int(n)}


def summarise(records: list[dict]) -> list[dict]:
    """Tidy ``scenario x partial_r2 x metric`` rows from the replicate records."""
    rows: list[dict] = []
    keys = sorted({(r["scenario"], r["partial_r2"]) for r in records})
    for name, pr2 in keys:
        rs = [r for r in records if r["scenario"] == name and r["partial_r2"] == pr2]
        n = len(rs)
        hits = sum(r["excluded_zero"] for r in rs)
        null = all(r["beta_true"] == 0.0 for r in rs)
        rows.append(_rate_rows(name, pr2, "false_positive_rate" if null else "power",
                               hits, n))

        def mean_row(metric, values):
            vals = [v for v in values if v is not None and np.isfinite(v)]
            rows.append({"scenario": name, "partial_r2": pr2, "metric": metric,
                         "value": float(np.mean(vals)) if vals else float("nan"),
                         "ci_low": None, "ci_high": None, "n": len(vals)})

        if not null:
            mean_row("beta_bias", [r["beta_mean"] - r["beta_true"] for r in rs])
            mean_row("beta_coverage", [float(r["beta_ci_low"] <= r["beta_true"]
                                             <= r["beta_ci_high"]) for r in rs])
            chans = [c for r in rs for c in r["channels"].values()]
            mean_row("r50_abs_log_error", [
                abs(math.log(c["r50_mean"] / c["r50_true"])) for c in chans
                if c.get("r50_mean") and c.get("r50_true")])
            mean_row("r50_coverage", [
                float(c["r50_ci_low"] <= c["r50_true"] <= c["r50_ci_high"])
                for c in chans if "r50_ci_low" in c])
            mean_row("aggregator_tv_error", [c.get("aggregator_tv") for c in chans])
            mean_row("weight_coverage", [
                float(v) for r in rs for v in r.get("weight_covered", [])])
            mean_row("weight_abs_error", [r.get("weight_abs_error") for r in rs])
            mean_row("form_recovery", [float(r["form"] == r["form_true"]) for r in rs])
    return rows


def plasmode_recovery(X, covariates=None, *, channels, radii, stats,
                      channel_index=None, radius_idx=None, scenarios=None,
                      reps=200, partial_r2=(0.0005, 0.001, 0.002), forms=bi.FORMS,
                      pipeline_kwargs=None, seed=0, workers=None,
                      cancel_check=None) -> dict:
    """Plant each scenario's CGI in ``X`` and measure what the pipeline recovers.

    ``X`` is the real ``(entity, channel, radius, stat)`` tensor and
    ``covariates`` the real covariate matrix, both unchanged. ``reps`` runs per
    scenario and effect size; ``S0`` runs once, at no effect. ``pipeline_kwargs``
    overrides :data:`LIGHT_SETTINGS` and may set ``radius_kernel`` /
    ``aggregator``. Returns ``{"settings", "table", "replicates"}``; the
    settings record every light setting used.
    """
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    n_channels = len(channels)
    if channel_index is None:
        channel_index = list(range(n_channels))
    if radius_idx is None:
        radius_idx = [list(range(len(radii)))] * n_channels
    cfg = {**LIGHT_SETTINGS, "radius_kernel": "lognormal", "aggregator": "dirichlet",
           **(pipeline_kwargs or {})}
    cfg.update(channels=list(channels), channel_index=list(channel_index),
               radii=np.asarray(radii, dtype=np.float64), stats=list(stats),
               radius_idx=[list(r) for r in radius_idx], forms=tuple(forms))
    truths = scenarios or default_scenarios(n_channels, radii, stats, radius_idx)

    cov = None if covariates is None else np.ascontiguousarray(
        np.asarray(covariates, dtype=np.float64).reshape(len(X), -1))
    _, Xr0 = bi.prep(X, np.zeros(len(X)), cov, channel_index=channel_index)
    if len(Xr0) != len(X):
        raise ValueError("plasmode_recovery needs complete rows; drop entities "
                         "with partial coverage first.")
    gamma = None
    if cov is not None:
        g = np.random.default_rng(seed).normal(size=cov.shape[1])
        part = cov @ g
        gamma = g * (0.5 / max(float(part.std()), 1e-12))

    paths = [_write_memmap(X)]
    try:
        cov_path = None
        if cov is not None:
            paths.append(_write_memmap(cov))
            cov_path = paths[1]
        tasks = []
        for truth in truths:
            z = planted_index(Xr0, truth, channel_index)
            for pr2 in ((0.0,) if truth.null else tuple(partial_r2)):
                for rep in range(int(reps)):
                    tasks.append((paths[0], X.shape, cov_path,
                                  None if cov is None else cov.shape, z, gamma,
                                  truth, float(pr2),
                                  _seed(seed, truth.name, float(pr2), rep), cfg))
        n_workers = workers or parallel.process_worker_count(len(tasks))
        records = bi._map(_replicate, tasks, n_workers, cancel_check)
    finally:
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    settings = {k: v for k, v in cfg.items() if k != "radii"}
    settings.update(radii=[float(r) for r in cfg["radii"]], reps=int(reps),
                    partial_r2=list(partial_r2), seed=int(seed), n=int(len(X)),
                    scenarios=[asdict(t) for t in truths])
    return {"settings": settings, "table": summarise(records), "replicates": records}


def write_plasmode_report(result: dict, out_dir: str) -> list[str]:
    """``plasmode_summary.csv`` (the tidy table) and ``plasmode_results.json``."""
    import csv

    os.makedirs(out_dir, exist_ok=True)
    table_path = os.path.join(out_dir, "plasmode_summary.csv")
    with open(table_path, "w", newline="", encoding="utf-8") as f:
        cols = ["scenario", "partial_r2", "metric", "value", "ci_low", "ci_high", "n"]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(result["table"])
    json_path = os.path.join(out_dir, "plasmode_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, default=str, indent=1)
    return [table_path, json_path]
