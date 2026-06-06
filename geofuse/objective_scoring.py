"""Covariate-aware objective scoring for the fusion engine.

When the user supplies covariates (additional outcome predictors that should
be controlled for), the optimizer's score becomes the **greenery term's
partial contribution** to predicting the outcome — not the full-model fit.
A dominant covariate must not be allowed to let the optimizer ignore the CGI
parameters; the run is meant to maximize how much the CGI itself explains
*over and above* the covariates.

Per-metric semantics
--------------------

``distance_corr`` (default)
    Distance correlation between target and CGI. Detects nonlinear as well as
    linear association and is unsigned (always in ``[0, 1]``). With covariates
    both target and CGI are OLS-residualized first, then distance-correlated,
    so the score is the greenery term's partial (linear-controlled) nonlinear
    association. Direction is reported separately via :func:`relationship_sign`.
``spearman``
    Partial rank correlation magnitude: residualize the *ranks* of target and
    CGI on the (ranked) covariates, then take ``|Pearson|`` of the residuals.
``r2``
    Incremental (partial) R²: ``R²(target ~ covariates + cgi) − R²(target ~
    covariates)``. Negative values are possible (CGI hurts the fit) and are
    returned as-is so Optuna can rank them.
``nrmse``
    Min-max normalized RMSE: target and CGI are each scaled to ``[0, 1]``
    (residualized first when covariates are present), then the RMSE of the
    pattern alignment is taken. Dimensionless and scale-free — it measures how
    well the CGI *pattern* tracks the outcome *pattern*, not unit translation.
    Lower is better.
``mutual_info``
    Plain MI between target and CGI; **covariates are ignored** for this
    metric. Conditional MI is hard and lossy to estimate from binned data, so
    the UI tooltip explains that the MI score is a direct (non-controlled)
    relationship — pick a different metric to control for covariates.

When no covariates are supplied, every metric collapses to its no-covariate
baseline.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, pearsonr, rankdata
from sklearn.metrics import mutual_info_score, r2_score

# Public set of metric names this module knows how to score; engine-level
# validation should compare against it before calling :func:`score`.
SUPPORTED_METRICS: frozenset[str] = frozenset(
    {"distance_corr", "spearman", "r2", "nrmse", "mutual_info"}
)

# The default cross-sectional objective.
DEFAULT_METRIC: str = "distance_corr"

# Metrics whose return value is "higher is better" — useful when the caller
# needs a uniform direction for ranking. ``nrmse`` is the only one where lower
# is better.
HIGHER_IS_BETTER: frozenset[str] = frozenset(
    {"distance_corr", "spearman", "r2", "mutual_info"}
)

# Metrics for which the covariate-adjusted score *ignores* covariates. The UI
# uses this list to render a tooltip on the covariate multi-select.
COVARIATE_IGNORED: frozenset[str] = frozenset({"mutual_info"})

# Worst score returned when the inputs are degenerate (all-NaN, constant, < 3
# rows).
_DEGENERATE_SCORE: dict[str, float] = {
    "distance_corr": 0.0,
    "spearman": 0.0,
    "r2": 0.0,
    "nrmse": float("inf"),
    "mutual_info": 0.0,
}

# Above this many rows the distance-correlation O(n²) distance matrices are
# capped via a deterministic subsample. Collapsed polygon / entity counts are
# normally a few hundred, so this is a safety rail rather than a routine path.
_DCOR_N_CAP: int = 2000


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


def _minmax01(v: np.ndarray) -> np.ndarray:
    """Scale ``v`` to ``[0, 1]``. A constant vector maps to all-zeros."""
    lo = float(np.min(v))
    hi = float(np.max(v))
    span = hi - lo
    if span <= 0:
        return np.zeros_like(v)
    return (v - lo) / span


# ---------------------------------------------------------------------------
# Distance correlation
# ---------------------------------------------------------------------------


def _double_center(D: np.ndarray) -> np.ndarray:
    """Double-center a distance matrix (row + column means subtracted, grand added)."""
    row_mean = D.mean(axis=0, keepdims=True)
    col_mean = D.mean(axis=1, keepdims=True)
    return D - row_mean - col_mean + D.mean()


def distance_correlation(
    x: np.ndarray, y: np.ndarray, *, seed: int = 0
) -> float:
    """Distance correlation between two 1-D arrays (biased estimator, ``[0, 1]``).

    Uses the Székely–Rizzo double-centred distance matrices:
    ``dCov² = mean(A ⊙ B)``, ``dVar = mean(A ⊙ A)``, and
    ``dCor = sqrt(dCov² / sqrt(dVarx · dVary))``. Returns ``0.0`` when either
    input is constant (no spread → no distances). For ``n > _DCOR_N_CAP`` a
    deterministic subsample bounds the O(n²) memory.
    """
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    if xa.shape != ya.shape:
        raise ValueError(
            f"x and y must have the same shape; got {xa.shape} vs {ya.shape}."
        )
    n = len(xa)
    if n < 3:
        return 0.0
    if n > _DCOR_N_CAP:
        idx = np.random.default_rng(seed).choice(n, size=_DCOR_N_CAP, replace=False)
        xa = xa[idx]
        ya = ya[idx]
    if float(np.var(xa)) == 0 or float(np.var(ya)) == 0:
        return 0.0

    a = np.abs(xa[:, None] - xa[None, :])
    b = np.abs(ya[:, None] - ya[None, :])
    A = _double_center(a)
    B = _double_center(b)
    dcov2 = float(np.mean(A * B))
    dvar_x = float(np.mean(A * A))
    dvar_y = float(np.mean(B * B))
    denom = np.sqrt(dvar_x * dvar_y)
    if denom <= 0 or dcov2 <= 0:
        return 0.0
    return float(np.sqrt(dcov2 / denom))


# ---------------------------------------------------------------------------
# Direction reporting
# ---------------------------------------------------------------------------


def relationship_sign(
    target: np.ndarray,
    cgi: np.ndarray,
    covariates: np.ndarray | None = None,
) -> int:
    """Sign (``+1`` / ``-1``) of the greenery↔outcome relationship.

    Distance correlation is unsigned, so the report needs a separate direction
    indicator. Uses the sign of the partial Spearman correlation (rank
    residuals on ranked covariates). Returns ``+1`` when higher greenery tracks
    higher outcome, ``-1`` otherwise, and ``+1`` as a neutral default for
    degenerate inputs.
    """
    t = np.asarray(target, dtype=np.float64).ravel()
    c = np.asarray(cgi, dtype=np.float64).ravel()
    cov = _coerce_covariates(covariates)
    t, c, cov = _drop_nan_rows(t, c, cov)
    if len(t) < 3 or float(np.var(t)) == 0 or float(np.var(c)) == 0:
        return 1
    tt = rankdata(t)
    cc = rankdata(c)
    if cov is not None:
        cov_use = np.column_stack([rankdata(cov[:, j]) for j in range(cov.shape[1])])
    else:
        cov_use = None
    tr = _residualize(tt, cov_use)
    cr = _residualize(cc, cov_use)
    if float(np.var(tr)) == 0 or float(np.var(cr)) == 0:
        return 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = float(np.corrcoef(tr, cr)[0, 1])
    if not np.isfinite(corr) or corr == 0:
        return 1
    return 1 if corr > 0 else -1


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


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
    ``return_pvalue=True`` returns ``(score, 1.0)`` — the ``1.0`` is a retained
    sentinel (no metric feeds a p-value gate anymore; robustness comes from
    bootstrap stability selection).
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

        if metric == "distance_corr":
            tr = _residualize(t, cov)
            cr = _residualize(c, cov)
            if float(np.var(tr)) == 0 or float(np.var(cr)) == 0:
                return (0.0, 1.0) if return_pvalue else 0.0
            s = distance_correlation(tr, cr)
            return (s, 1.0) if return_pvalue else s

        if metric == "spearman":
            tt = rankdata(t)
            cc = rankdata(c)
            if cov is not None:
                cov_use = np.column_stack(
                    [rankdata(cov[:, j]) for j in range(cov.shape[1])]
                )
            else:
                cov_use = None
            tr = _residualize(tt, cov_use)
            cr = _residualize(cc, cov_use)
            if float(np.var(tr)) == 0 or float(np.var(cr)) == 0:
                return (0.0, 1.0) if return_pvalue else 0.0
            corr, _pval = pearsonr(tr, cr)
            if np.isnan(corr):
                return (0.0, 1.0) if return_pvalue else 0.0
            s = float(abs(corr))
            return (s, 1.0) if return_pvalue else s

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

        if metric == "nrmse":
            # Min-max normalized RMSE: scale-free pattern error. Residualize on
            # covariates first (matching distance_corr) so the score reflects
            # the greenery term's partial contribution.
            tr = _residualize(t, cov) if cov is not None else t
            cr = _residualize(c, cov) if cov is not None else c
            t01 = _minmax01(tr)
            c01 = _minmax01(cr)
            s = float(np.sqrt(np.mean((t01 - c01) ** 2)))
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


# ---------------------------------------------------------------------------
# Penalized model comparison (CGI vs best standalone channel)
# ---------------------------------------------------------------------------


def _verdict_from_delta_bic(delta_bic: float) -> str:
    """Map ΔBIC (reduced − full, positive favours the full CGI model) to a verdict.

    Follows the conventional Kass–Raftery strength bands on the BIC difference.
    """
    if not np.isfinite(delta_bic):
        return "inconclusive"
    if delta_bic > 10:
        return "justified (very strong)"
    if delta_bic > 6:
        return "justified (strong)"
    if delta_bic > 2:
        return "justified (positive)"
    if delta_bic < -2:
        return "not justified"
    return "negligible difference"


def compare_models_aic_bic(
    target: np.ndarray,
    channel_matrix: np.ndarray,
    channel_names: list[str],
    best_channel_idx: int,
    covariates: np.ndarray | None = None,
) -> dict:
    """AIC/BIC comparison of a full multi-channel model vs the best single channel.

    Fits two OLS models on ``target``:

    * **full** — ``target ~ all channels + covariates``
    * **reduced** — ``target ~ best single channel + covariates``

    and returns their AIC/BIC plus ``delta_aic`` / ``delta_bic`` (reduced minus
    full, so a *positive* delta means the extra channels improve the penalized
    fit enough to justify their complexity) and a ``verdict`` keyed off ΔBIC.

    The full model is naturally penalized for its extra predictors, so this
    answers "do the other channels earn their keep over the best one alone?".
    """
    import statsmodels.api as sm

    t = np.asarray(target, dtype=np.float64).ravel()
    X = np.asarray(channel_matrix, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"channel_matrix must be 2-D; got shape {X.shape}.")
    if X.shape[1] != len(channel_names):
        raise ValueError(
            f"channel_names ({len(channel_names)}) must match channel columns "
            f"({X.shape[1]})."
        )
    if not (0 <= best_channel_idx < X.shape[1]):
        raise ValueError(f"best_channel_idx {best_channel_idx} out of range.")

    cov = _coerce_covariates(covariates)
    mask = np.isfinite(t) & np.isfinite(X).all(axis=1)
    if cov is not None:
        mask &= np.isfinite(cov).all(axis=1)
    t = t[mask]
    X = X[mask]
    cov = None if cov is None else cov[mask]

    if len(t) < (X.shape[1] + (0 if cov is None else cov.shape[1]) + 3):
        return {
            "ok": False,
            "reason": "too few rows for a stable OLS comparison",
            "channel_names": list(channel_names),
            "best_channel": channel_names[best_channel_idx],
        }

    def _design(cols: np.ndarray) -> np.ndarray:
        parts = [cols]
        if cov is not None:
            parts.append(cov)
        return sm.add_constant(np.column_stack(parts), has_constant="add")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = sm.OLS(t, _design(X)).fit()
        reduced = sm.OLS(
            t, _design(X[:, [best_channel_idx]])
        ).fit()

    delta_aic = float(reduced.aic - full.aic)
    delta_bic = float(reduced.bic - full.bic)
    return {
        "ok": True,
        "channel_names": list(channel_names),
        "best_channel": channel_names[best_channel_idx],
        "aic_full": float(full.aic),
        "bic_full": float(full.bic),
        "aic_reduced": float(reduced.aic),
        "bic_reduced": float(reduced.bic),
        "delta_aic": delta_aic,
        "delta_bic": delta_bic,
        "n": int(len(t)),
        "verdict": _verdict_from_delta_bic(delta_bic),
    }
