"""Post-hoc multi-metric reporting for mixed-effects fusion runs.

After the Optuna study completes, this module re-scores every robust trial
(BH-FDR filtered) and every top-``top_percent`` trial on the held-out test
set, computing all four ``mixedlm_*`` scorer metrics per trial regardless of
which one was optimised. The final composite — using the averaged
top-``top_percent`` parameters that ``generate_composite_greenery_map`` picks
— is also scored once. Per-trial values land in ``mixedlm_metrics.csv``
alongside a per-(pool, metric) summary row carrying the sample mean + 95 %
CI across the trial pool (1.96·SE for n>=30, the 2.5/97.5 quantiles
otherwise).

Output schema (``<study_results>/mixedlm_metrics.csv``)::

    pool, trial_id, mixedlm_tstat, mixedlm_marginal_r2, mixedlm_lr, mixedlm_coef

with one row per (pool, trial) plus one summary row per (pool, summary kind)
where ``trial_id`` carries ``__mean__`` / ``__ci_lo__`` / ``__ci_hi__`` /
``__n__``.
"""

from __future__ import annotations

import csv
import os
from typing import Any

import numpy as np

from . import mixed_effects_scoring


def _formula_average_params(engine, top_trials: list[Any]) -> dict | None:
    """Average params across ``top_trials`` using the engine's CGI formula.

    Mirrors the averaging step inside
    ``MetricFusionEngine.generate_composite_greenery_map`` so the "final"
    pool reflects the same parameter set that the composite raster used. In
    standalone single-metric mode (no CGI weights / powers) this returns
    the best trial's params unchanged.
    """
    if not top_trials:
        return None
    from . import cgi_formulas
    from statistics import mode as _mode

    channel_mode = engine._active_greenery_channel
    if channel_mode != "cgi":
        # Standalone studies have no CGI weights/powers to average — the
        # composite raster (if any) uses the single best trial's params.
        return dict(top_trials[0].params)

    formula = cgi_formulas.get_formula(engine.cgi_formula)
    out: dict[str, Any] = {}

    # Numeric params: per-key arithmetic mean. The formula registry tells
    # us which keys are weights vs powers; weights get rounded back to int
    # (the original sampling type) and renormalised so their sum matches
    # the formula's weight scale (100 for weighted_average, 1.0 for
    # synergy).
    weight_keys = list(formula.weight_keys)
    power_keys = list(formula.power_keys)
    radius_keys = ["veg_radius", "terrain_radius", "ndvi_radius"]
    pct_keys = ["streetview_percentile", "ndvi_percentile"]

    def _mean_of(key: str) -> float | None:
        vals = [t.params.get(key) for t in top_trials if key in t.params]
        if not vals:
            return None
        return float(np.mean([float(v) for v in vals]))

    for key in weight_keys + power_keys + radius_keys + pct_keys:
        v = _mean_of(key)
        if v is None:
            continue
        if key in weight_keys or key in radius_keys or key in pct_keys:
            out[key] = int(round(v))
        else:
            out[key] = v

    # Renormalise weights so they sum to 100 — the integer scale every
    # registered formula uses. The compute functions self-renormalise too,
    # so this is mostly for readability when the CSV is inspected.
    weight_vals = [out[k] for k in weight_keys if k in out]
    if weight_vals:
        total = sum(weight_vals)
        if total > 0:
            for k in weight_keys:
                if k in out:
                    out[k] = int(round(out[k] * 100.0 / total))

    # Categorical params: pick the most common value across the top trials.
    for key in ("streetview_stat", "ndvi_stat"):
        cats = [t.params.get(key) for t in top_trials if key in t.params]
        if cats:
            try:
                out[key] = _mode(cats)
            except Exception:
                out[key] = cats[0]
    return out


def _pool_summary_rows(
    pool: str, trial_metric_values: list[dict[str, float]]
) -> list[dict[str, Any]]:
    """One-row-per-metric summary (mean + 95 % CI) for a pool of trials."""
    if not trial_metric_values:
        return []
    metrics = sorted(mixed_effects_scoring.MIXEDLM_METRICS)
    n = len(trial_metric_values)
    rows: list[dict[str, Any]] = []
    base = {m: float("nan") for m in metrics}

    def _summary(kind: str, values: dict[str, float]) -> dict[str, Any]:
        row: dict[str, Any] = {"pool": pool, "trial_id": kind, **base}
        for m, v in values.items():
            row[m] = v
        return row

    means = {m: float(np.mean([r[m] for r in trial_metric_values])) for m in metrics}
    rows.append(_summary("__mean__", means))
    if n >= 30:
        # Normal-approx CI: 1.96 · SE.
        ses = {
            m: float(np.std([r[m] for r in trial_metric_values], ddof=1) / np.sqrt(n))
            for m in metrics
        }
        rows.append(_summary("__ci_lo__", {m: means[m] - 1.96 * ses[m] for m in metrics}))
        rows.append(_summary("__ci_hi__", {m: means[m] + 1.96 * ses[m] for m in metrics}))
    else:
        # Quantile CI: empirical 2.5 / 97.5 percentiles (more robust for
        # the small-n pools we typically see).
        qlo = {
            m: float(np.percentile([r[m] for r in trial_metric_values], 2.5))
            for m in metrics
        }
        qhi = {
            m: float(np.percentile([r[m] for r in trial_metric_values], 97.5))
            for m in metrics
        }
        rows.append(_summary("__ci_lo__", qlo))
        rows.append(_summary("__ci_hi__", qhi))
    rows.append(_summary("__n__", {m: float(n) for m in metrics}))
    return rows


def compute_post_metrics(
    engine,
    output_dir: str,
    *,
    top_percent: float = 0.2,
    min_robust: int = 10,
    csv_basename: str = "mixedlm_metrics.csv",
    log: Any = None,
) -> str | None:
    """Score every robust + top-``top_percent`` trial + the final composite
    parameters on the held-out test set and write ``mixedlm_metrics.csv``.

    No-op (returns ``None``) for cross-sectional runs or when the engine has
    no completed trials. ``log`` is an optional ``_log``-style callable from
    the runner so progress messages route into the per-job log.
    """
    if not engine.is_longitudinal:
        return None
    if engine.study is None or engine.test_data is None:
        return None

    def _say(level: str, msg: str) -> None:
        if log is not None:
            log(level, msg)

    robust_trials = engine.get_robust_trials(
        method="auto", p_threshold=0.05, tolerance=0.1, min_trials=min_robust
    )
    if not robust_trials:
        _say("WARN", "Post-hoc MixedLM metrics: no robust trials; skipping.")
        return None

    n_top = max(1, int(len(robust_trials) * top_percent))
    direction_max = engine.study.direction.name == "MAXIMIZE"
    top_trials = sorted(
        robust_trials, key=lambda t: t.value, reverse=direction_max
    )[:n_top]

    final_params = _formula_average_params(engine, top_trials)

    pools: list[tuple[str, list[Any]]] = [
        ("robust", robust_trials),
        (f"top_{int(top_percent * 100)}pct", top_trials),
    ]
    if final_params is not None:
        pools.append(("final", [final_params]))

    rows: list[dict[str, Any]] = []
    for pool_name, items in pools:
        per_trial_values: list[dict[str, float]] = []
        for entry in items:
            if hasattr(entry, "params"):
                params = entry.params
                trial_id: Any = entry.number
            else:
                params = entry
                trial_id = "final_composite"
            try:
                out = engine.evaluate_on_test(
                    params=params,
                    metric=engine.longitudinal_spec.scoring_metric,
                    return_all_mixedlm=True,
                )
                metrics = out.get("mixedlm_metrics") or {}
            except Exception as exc:
                _say(
                    "WARN",
                    f"Post-hoc score failed for {pool_name} trial {trial_id}: {exc}",
                )
                metrics = {m: float("nan") for m in mixed_effects_scoring.MIXEDLM_METRICS}
            row = {"pool": pool_name, "trial_id": trial_id, **metrics}
            rows.append(row)
            if all(m in metrics for m in mixed_effects_scoring.MIXEDLM_METRICS):
                per_trial_values.append(
                    {m: float(metrics[m]) for m in mixed_effects_scoring.MIXEDLM_METRICS}
                )
        # Only summarise pools with more than one entry (the ``final`` pool
        # is a single point estimate by design — no spread).
        if len(per_trial_values) > 1:
            rows.extend(_pool_summary_rows(pool_name, per_trial_values))

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, csv_basename)
    fields = ["pool", "trial_id"] + sorted(mixed_effects_scoring.MIXEDLM_METRICS)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    _say("OK", f"Post-hoc MixedLM metrics written: {csv_path}")
    return csv_path
