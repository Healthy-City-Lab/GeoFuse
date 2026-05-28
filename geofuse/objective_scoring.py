"""Covariate-aware objective scoring for the fusion engine.

When the user supplies covariates (additional outcome predictors that should
be controlled for), the optimizer's score becomes the **greenery term's
partial contribution** to predicting the outcome — not the full-model fit.
A dominant covariate must not be allowed to let the optimizer ignore the CGI
parameters; the run is meant to maximize how much the CGI itself explains
*over and above* the covariates.

Per-metric semantics
--------------------

``pearson`` / ``spearman``
    Partial correlation: residualize both target and CGI on the covariates
    (Spearman residualizes the *ranks*), then take the standard correlation
    coefficient. The p-value is the residual correlation's p-value, accepted
    as a conventional approximation — the dof loss from residualization is
    small for the sample sizes the fusion engine typically sees.
``r2``
    Incremental (partial) R²: ``R²(target ~ covariates + cgi) − R²(target ~
    covariates)``. Negative values are possible (CGI hurts the fit) and are
    returned as-is so Optuna can rank them.
``rmse``
    Full-model RMSE: ``rmse(target, OLS(target ~ covariates + cgi))``. Does
    not isolate CGI as cleanly as the partial-correlation / incremental-R²
    paths; surfaced as a documented limitation in the UI tooltip.
``mutual_info``
    Plain MI between target and CGI; **covariates are ignored** for this
    metric. Conditional MI is hard and lossy to estimate from binned data, so
    the UI tooltip explains that the MI score is a direct (non-controlled)
    relationship — pick a different metric to control for covariates.

When no covariates are supplied, every metric collapses to its no-covariate
baseline — identical to the engine's previous ``_calculate_metric`` (matched
sign, magnitude, and return type) so legacy studies replay unchanged.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, pearsonr, rankdata
from sklearn.metrics import mean_squared_error, mutual_info_score, r2_score

# Public set of metric names this module knows how to score; engine-level
# validation should compare against it before calling :func:`score`.
SUPPORTED_METRICS: frozenset[str] = frozenset(
    {"pearson", "spearman", "r2", "rmse", "mutual_info"}
)

# Metrics whose return value is "higher is better" — useful when the caller
# needs a uniform direction for ranking. ``rmse`` is the only one where lower
# is better.
HIGHER_IS_BETTER: frozenset[str] = frozenset(
    {"pearson", "spearman", "r2", "mutual_info"}
)

# Metrics for which a meaningful p-value is produced — ``r2`` / ``rmse`` /
# ``mutual_info`` return ``1.0`` as a sentinel so callers don't have to special-
# case them at the call site.
HAS_PVALUE: frozenset[str] = frozenset({"pearson", "spearman"})

# Metrics for which the covariate-adjusted score *ignores* covariates. The UI
# uses this list to render a tooltip on the covariate multi-select.
COVARIATE_IGNORED: frozenset[str] = frozenset({"mutual_info"})

# Worst score returned when the inputs are degenerate (all-NaN, constant, < 3
# rows). Picked to match the engine's previous behaviour.
_DEGENERATE_SCORE: dict[str, float] = {
    "pearson": 0.0,
    "spearman": 0.0,
    "r2": 0.0,
    "rmse": float("inf"),
    "mutual_info": 0.0,
}


def _validate_metric(metric: str) -> None:
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unknown metric '{metric}'. Expected one of {sorted(SUPPORTED_METRICS)}."
        )


def _coerce_covariates(covariates: np.ndarray | None) -> np.ndarray | None:
    """Return a 2-D float covariate matrix, or ``None`` if no covariates."""
    if covariates is None:
        return None
    cov = np.asarray(covariates, dtype=np.float64)
    if cov.size == 0:
        return None
    if cov.ndim == 1:
        cov = cov.reshape(-1, 1)
    return cov


def _drop_nan_rows(
    target: np.ndarray, cgi: np.ndarray, cov: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Mask rows where any of ``target`` / ``cgi`` / any covariate is non-finite."""
    mask = np.isfinite(target) & np.isfinite(cgi)
    if cov is not None:
        mask &= np.isfinite(cov).all(axis=1)
    if mask.all():
        return target, cgi, cov
    return (
        target[mask],
        cgi[mask],
        None if cov is None else cov[mask],
    )


def _residualize(y: np.ndarray, X: np.ndarray | None) -> np.ndarray:
    """OLS residuals of ``y ~ X`` (intercept added). ``X=None`` returns ``y``."""
    if X is None or X.shape[1] == 0:
        return y
    Xc = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    return y - Xc @ beta


def _ols_r2(y: np.ndarray, X: np.ndarray | None) -> float:
    """In-sample R² of ``OLS(y ~ X)`` (intercept added)."""
    ymean = float(y.mean())
    ss_tot = float(((y - ymean) ** 2).sum())
    if ss_tot == 0:
        return 0.0
    if X is None or X.shape[1] == 0:
        # Reduced model = intercept only → predicts ymean → R² = 0.
        return 0.0
    Xc = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    yhat = Xc @ beta
    return 1.0 - float(((y - yhat) ** 2).sum()) / ss_tot


def _ols_rmse(y: np.ndarray, X: np.ndarray) -> float:
    Xc = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    yhat = Xc @ beta
    return float(np.sqrt(((y - yhat) ** 2).mean()))


def score(
    metric: str,
    target: np.ndarray,
    cgi: np.ndarray,
    covariates: np.ndarray | None = None,
    *,
    return_pvalue: bool = False,
) -> float | tuple[float, float]:
    """Score CGI's predictive power for ``target``, controlling for ``covariates``.

    ``target`` and ``cgi`` are 1-D arrays of equal length. ``covariates`` is
    either ``None`` (no control variables) or a 2-D ``(n_samples, n_covs)``
    array. NaN rows are dropped jointly. Returns a single float by default; if
    ``return_pvalue=True`` returns ``(score, pvalue)`` with ``pvalue=1.0`` as a
    sentinel for metrics that don't produce one.
    """
    _validate_metric(metric)
    t = np.asarray(target, dtype=np.float64)
    c = np.asarray(cgi, dtype=np.float64)
    if t.shape != c.shape:
        raise ValueError(
            f"target and cgi must have the same shape, got {t.shape} vs {c.shape}."
        )

    cov = _coerce_covariates(covariates)
    if metric in COVARIATE_IGNORED:
        # MI ignores covariates by design (documented).
        cov = None
    t, c, cov = _drop_nan_rows(t, c, cov)

    if len(t) < 3 or float(np.var(t)) == 0 or float(np.var(c)) == 0:
        s = _DEGENERATE_SCORE[metric]
        return (s, 1.0) if return_pvalue else s

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=ConstantInputWarning)

        if metric in ("pearson", "spearman"):
            if metric == "spearman":
                tt = rankdata(t)
                cc = rankdata(c)
                if cov is not None:
                    cov_use = np.column_stack(
                        [rankdata(cov[:, j]) for j in range(cov.shape[1])]
                    )
                else:
                    cov_use = None
            else:
                tt, cc, cov_use = t, c, cov

            tr = _residualize(tt, cov_use)
            cr = _residualize(cc, cov_use)
            if float(np.var(tr)) == 0 or float(np.var(cr)) == 0:
                return (0.0, 1.0) if return_pvalue else 0.0
            # After residualization Pearson is the appropriate test for both
            # original metric choices (Spearman is just Pearson on ranks).
            corr, pval = pearsonr(tr, cr)
            if np.isnan(corr):
                return (0.0, 1.0) if return_pvalue else 0.0
            s = float(abs(corr))
            return (s, float(pval)) if return_pvalue else s

        if metric == "r2":
            if cov is None:
                # Legacy path: r2 of cgi treated as a direct prediction of
                # target — preserves the previous engine behaviour exactly.
                s = float(r2_score(t, c))
            else:
                X_red = cov
                X_full = np.column_stack([cov, c])
                s = _ols_r2(t, X_full) - _ols_r2(t, X_red)
            if np.isnan(s):
                s = 0.0
            return (s, 1.0) if return_pvalue else s

        if metric == "rmse":
            if cov is None:
                s = float(np.sqrt(mean_squared_error(t, c)))
            else:
                X = np.column_stack([cov, c])
                s = _ols_rmse(t, X)
            return (s, 1.0) if return_pvalue else s

        if metric == "mutual_info":
            try:
                tb = pd.qcut(t, 10, labels=False, duplicates="drop")
                cb = pd.qcut(c, 10, labels=False, duplicates="drop")
                s = float(mutual_info_score(tb, cb))
            except (ValueError, TypeError):
                s = 0.0
            return (s, 1.0) if return_pvalue else s

    # Unreachable thanks to _validate_metric above.
    raise AssertionError(f"unreachable metric '{metric}'")
