"""Post-hoc multi-metric reporting for mixed-effects fusion runs.

After the stability search selects a channel mix, this module re-scores the
winning cell's trials and the final averaged composite on the held-out test
set, computing all four ``mixedlm_*`` scorer metrics per trial regardless of
which one was optimized. Per-trial values land in ``mixedlm_metrics.csv``
alongside a per-(pool, metric) summary row carrying the sample mean + 95 %
CI across the pool (1.96·SE for n>=30, the 2.5/97.5 quantiles otherwise).

The trials come from the per-trial ``__trial_history__`` table the stability
search records on its winning-params dict, so no Optuna study is needed.

Output schema (``<study_results>/mixedlm_metrics.csv``)::

    pool, trial_id, mixedlm_tstat, mixedlm_marginal_r2, mixedlm_lr, mixedlm_coef, n_trials

with one row per (pool, trial) plus one summary row per (pool, summary kind)
where ``trial_id`` carries ``__mean__`` / ``__ci_lo__`` / ``__ci_hi__``. The
pool's trial count lives in its own ``n_trials`` column so the metric columns
hold only metric values.
"""

from __future__ import annotations

import csv
import os
from typing import Any

import numpy as np

from . import mixed_effects_scoring

# Bookkeeping columns on each ``__trial_history__`` row that are not engine
# parameters and so must be stripped before re-scoring.
_HISTORY_META_KEYS = {"bootstrap", "oob_score", "cell", "in_top_k"}


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
        rows.append(
            _summary("__ci_lo__", {m: means[m] - 1.96 * ses[m] for m in metrics})
        )
        rows.append(
            _summary("__ci_hi__", {m: means[m] + 1.96 * ses[m] for m in metrics})
        )
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
    return rows


def compute_post_metrics(
    engine,
    output_dir: str,
    *,
    winning_params: dict,
    max_trials: int = 200,
    csv_basename: str = "mixedlm_metrics.csv",
    log: Any = None,
) -> str | None:
    """Re-score the stability winner's trials + the final composite on the
    held-out test set with all four ``mixedlm_*`` metrics and write
    ``mixedlm_metrics.csv``.

    Pools:

    * ``winning_cell`` — every trial whose snapped weight cell matches the
      selected channel mix, reconstructed from the per-trial
      ``__trial_history__`` the stability search recorded. Capped at
      ``max_trials`` (highest out-of-bag score first) to bound MixedLM fits.
    * ``final`` — the averaged winning parameters the composite was built from.

    No-op (returns ``None``) for cross-sectional runs, when there's no test
    split, or when the run carries no trial history. ``log`` is an optional
    ``_log``-style callable from the runner so progress routes into the job log.
    """
    if not engine.is_longitudinal or engine.test_data is None:
        return None
    winning_params = winning_params or {}
    history = winning_params.get("__trial_history__") or []
    if not history:
        return None

    def _say(level: str, msg: str) -> None:
        if log is not None:
            log(level, msg)

    winning_cell = list(winning_params.get("__winning_cell__") or [])
    higher = bool(winning_params.get("__higher_is_better__", True))

    cell_rows = [r for r in history if list(r.get("cell") or []) == winning_cell]
    cell_rows.sort(
        key=lambda r: (
            r.get("oob_score")
            if r.get("oob_score") is not None
            else float("-inf" if higher else "inf")
        ),
        reverse=higher,
    )
    if max_trials and len(cell_rows) > max_trials:
        cell_rows = cell_rows[:max_trials]

    final_clean = {k: v for k, v in winning_params.items() if not k.startswith("__")}

    pools: list[tuple[str, list[dict]]] = [("winning_cell", cell_rows)]
    if final_clean:
        pools.append(("final", [final_clean]))

    metric = engine.longitudinal_spec.scoring_metric
    rows: list[dict[str, Any]] = []
    pool_counts: dict[str, int] = {}
    for pool_name, items in pools:
        per_trial_values: list[dict[str, float]] = []
        for idx, entry in enumerate(items):
            if pool_name == "final":
                params = entry
                trial_id: Any = "final_composite"
            else:
                params = {k: v for k, v in entry.items() if k not in _HISTORY_META_KEYS}
                trial_id = idx
            try:
                out = engine.evaluate_on_test(
                    params=params,
                    metric=metric,
                    return_all_mixedlm=True,
                )
                metrics = out.get("mixedlm_metrics") or {}
            except Exception as exc:
                _say(
                    "WARN",
                    f"Post-hoc score failed for {pool_name} trial {trial_id}: {exc}",
                )
                metrics = {
                    m: float("nan") for m in mixed_effects_scoring.MIXEDLM_METRICS
                }
            rows.append({"pool": pool_name, "trial_id": trial_id, **metrics})
            if all(m in metrics for m in mixed_effects_scoring.MIXEDLM_METRICS):
                per_trial_values.append(
                    {
                        m: float(metrics[m])
                        for m in mixed_effects_scoring.MIXEDLM_METRICS
                    }
                )
        pool_counts[pool_name] = len(per_trial_values)
        # Only summarise pools with more than one entry (the ``final`` pool
        # is a single point estimate by design — no spread).
        if len(per_trial_values) > 1:
            rows.extend(_pool_summary_rows(pool_name, per_trial_values))

    if not rows:
        _say("WARN", "Post-hoc MixedLM metrics: no scorable trials; skipping.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, csv_basename)
    # ``n_trials`` is its own column so the metric columns hold only metric
    # values.
    fields = (
        ["pool", "trial_id"]
        + sorted(mixed_effects_scoring.MIXEDLM_METRICS)
        + ["n_trials"]
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k, "") for k in fields}
            row["n_trials"] = pool_counts.get(r.get("pool"), "")
            w.writerow(row)

    _say("OK", f"Post-hoc MixedLM metrics written: {csv_path}")
    return csv_path
