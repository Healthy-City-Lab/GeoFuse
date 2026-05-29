"""Mixed-effects scoring for the fusion engine's longitudinal mode.

Scores the greenery fixed-effect's contribution to a linear mixed model
that accounts for within-entity correlation through a random intercept
(and optionally a random slope on ``years_since_baseline``) per entity.

The fitted model is::

    outcome ~ 1 + greenery + [covariates] + [years_since_baseline]
              + (1 + [years_since_baseline] | entity_id)

``include_time_fixed`` toggles the fixed-effect time term; ``random_slope``
toggles the random slope on time. The greenery coefficient's t-statistic /
sign / contribution to the model is what every metric here measures.

Per-metric semantics
--------------------

``mixedlm_tstat``
    ``|t-statistic|`` of the greenery fixed-effect coefficient. The
    longitudinal analog of partial-correlation magnitude — robust to scale
    of the outcome.

``mixedlm_marginal_r2``
    Greenery's contribution to the Nakagawa marginal R²: the share of total
    variance (fixed + random + residual) attributable to the greenery fixed
    effect alone. Computed from one fit as
    ``(var(X·β) − var(X·β with greenery zeroed)) / total_var``.

``mixedlm_lr``
    Likelihood-ratio statistic vs the same model refit without the greenery
    fixed effect: ``2·(ll_full − ll_null)``. Two fits per call.

``mixedlm_coef``
    Signed greenery fixed-effect coefficient. Sign matters — useful when
    the user wants to see direction of the relationship (e.g. higher
    greenery → lower blood pressure → negative coefficient).

All four scores are higher-is-better for Optuna (``mixedlm_coef`` is the
exception in spirit, but the engine optimises ``|coef|``-style magnitude
by configuration; see :data:`HIGHER_IS_BETTER`). A MixedLM convergence
failure, singular design, or input with no variance returns the
degenerate value ``0.0`` so a bad trial fails soft instead of crashing.
"""

from __future__ import annotations

import warnings

import numpy as np

MIXEDLM_METRICS: frozenset[str] = frozenset(
    {"mixedlm_tstat", "mixedlm_marginal_r2", "mixedlm_lr", "mixedlm_coef"}
)
DEFAULT_MIXEDLM_METRIC: str = "mixedlm_tstat"

# All four mixed-effects metrics are higher-is-better for Optuna. The
# signed ``mixedlm_coef`` is the practical exception — see the docstring
# above; the engine still maximises it by treating "more positive
# greenery effect" as the optimisation direction.
HIGHER_IS_BETTER: frozenset[str] = MIXEDLM_METRICS

# Metrics that produce a p-value (the greenery fixed-effect Wald test).
HAS_PVALUE: frozenset[str] = frozenset({"mixedlm_tstat", "mixedlm_coef"})

_DEGENERATE: dict[str, float] = {
    "mixedlm_tstat": 0.0,
    "mixedlm_marginal_r2": 0.0,
    "mixedlm_lr": 0.0,
    "mixedlm_coef": 0.0,
}

# Below this many post-NaN rows a MixedLM fit is unreliable; return degenerate.
_MIN_ROWS = 6


def _coerce_2d(x: np.ndarray | None) -> np.ndarray | None:
    if x is None:
        return None
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return None
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a


def _entity_missing_mask(arr: np.ndarray) -> np.ndarray:
    """True where the entity-id entry is None / NaN. Object-array safe."""
    out = np.zeros(len(arr), dtype=bool)
    for i, v in enumerate(arr):
        if v is None:
            out[i] = True
            continue
        try:
            out[i] = bool(np.isnan(v))
        except (TypeError, ValueError):
            out[i] = False
    return out


def _drop_nan(
    outcome: np.ndarray,
    greenery: np.ndarray,
    entity_id: np.ndarray,
    years_since_baseline: np.ndarray,
    covariates: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    eids = np.asarray(entity_id)
    mask = (
        np.isfinite(outcome)
        & np.isfinite(greenery)
        & np.isfinite(years_since_baseline)
        & ~_entity_missing_mask(eids)
    )
    if covariates is not None:
        mask &= np.isfinite(covariates).all(axis=1)
    if mask.all():
        return outcome, greenery, eids, years_since_baseline, covariates
    return (
        outcome[mask],
        greenery[mask],
        eids[mask],
        years_since_baseline[mask],
        None if covariates is None else covariates[mask],
    )


def _build_fixed_design(
    greenery: np.ndarray,
    covariates: np.ndarray | None,
    years_since_baseline: np.ndarray,
    *,
    include_time_fixed: bool,
    include_greenery: bool,
) -> tuple[np.ndarray, int]:
    """Build the fixed-effects design X. Returns (X, greenery_col_idx).

    Column order is fixed at: [intercept, (greenery), (covariates...), (time)].
    ``greenery_col_idx`` is ``-1`` when greenery is excluded from the design.
    """
    n = len(greenery)
    parts: list[np.ndarray] = [np.ones((n, 1))]
    greenery_col = -1
    if include_greenery:
        parts.append(greenery.reshape(-1, 1))
        greenery_col = 1
    if covariates is not None and covariates.shape[1] > 0:
        parts.append(covariates)
    if include_time_fixed:
        parts.append(years_since_baseline.reshape(-1, 1))
    X = np.hstack(parts)
    return X, greenery_col


def _build_re_design(
    years_since_baseline: np.ndarray, *, random_slope: bool
) -> np.ndarray:
    """Random-effects design: intercept + (optional) slope on time."""
    parts: list[np.ndarray] = [np.ones((len(years_since_baseline), 1))]
    if random_slope:
        parts.append(years_since_baseline.reshape(-1, 1))
    return np.hstack(parts)


def _fit_mixedlm(
    outcome: np.ndarray,
    X: np.ndarray,
    groups: np.ndarray,
    exog_re: np.ndarray,
):
    """Fit a MixedLM with REML+lbfgs, swallowing fit failures.

    Returns the fitted result or ``None`` if anything went wrong (singular
    design, non-convergence, numerical failure). Callers map ``None`` to
    the metric's degenerate score so a bad trial doesn't kill the study.
    """
    from statsmodels.regression.mixed_linear_model import MixedLM

    try:
        model = MixedLM(endog=outcome, exog=X, groups=groups, exog_re=exog_re)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(method="lbfgs", reml=True)
        if result is None:
            return None
        # Some statsmodels versions surface convergence via ``.converged``.
        converged = getattr(result, "converged", True)
        if not converged:
            return None
        return result
    except Exception:
        # MixedLM raises a wide variety of exceptions for ill-conditioned
        # inputs (LinAlgError, ValueError, OverflowError, ...). Catch them
        # all so per-trial bad data fails soft.
        return None


def _as_array(x) -> np.ndarray:
    """Coerce a statsmodels params/bse/pvalues object to a plain ndarray."""
    return x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x)


def _compute_all_metrics(
    result_full,
    X_full: np.ndarray,
    greenery_col: int,
    exog_re: np.ndarray,
    result_null=None,
) -> dict[str, float]:
    """Extract all four metric values from the (already fitted) full result."""
    fe = _as_array(result_full.fe_params)
    bse = _as_array(result_full.bse_fe)
    coef = float(fe[greenery_col])
    se = float(bse[greenery_col])
    tstat = abs(coef / se) if se > 0 else 0.0

    # Marginal R² (Nakagawa-style) — greenery's variance contribution as a
    # share of total variance. Total variance = fixed-effects variance +
    # random-effects variance contribution + residual variance.
    yhat_full = X_full @ fe
    var_fe_full = float(np.var(yhat_full, ddof=0))
    fe_zero_g = fe.copy()
    fe_zero_g[greenery_col] = 0.0
    yhat_no_g = X_full @ fe_zero_g
    var_fe_no_g = float(np.var(yhat_no_g, ddof=0))

    sigma2_resid = float(result_full.scale)
    re_var = 0.0
    cov_re = getattr(result_full, "cov_re", None)
    if cov_re is not None:
        cov_re_mat = np.asarray(cov_re)
        # Approximate the random-effects variance contribution as
        # sum_i cov_re[i,i] · E[Z_i²]. This is the standard Nakagawa
        # approximation for a (1 + time | entity) structure and is exact
        # for random-intercept-only models.
        for i in range(cov_re_mat.shape[0]):
            re_var += float(cov_re_mat[i, i]) * float(np.mean(exog_re[:, i] ** 2))

    total_var = var_fe_full + re_var + sigma2_resid
    marginal_r2 = (var_fe_full - var_fe_no_g) / total_var if total_var > 0 else 0.0

    lr_stat = 0.0
    if result_null is not None:
        try:
            lr_stat = 2.0 * (float(result_full.llf) - float(result_null.llf))
            if not np.isfinite(lr_stat) or lr_stat < 0.0:
                lr_stat = 0.0
        except Exception:
            lr_stat = 0.0

    return {
        "mixedlm_tstat": float(tstat),
        "mixedlm_marginal_r2": float(marginal_r2),
        "mixedlm_lr": float(lr_stat),
        "mixedlm_coef": coef,
    }


def score_mixedlm(
    metric: str,
    outcome: np.ndarray,
    greenery: np.ndarray,
    entity_id: np.ndarray,
    years_since_baseline: np.ndarray,
    covariates: np.ndarray | None = None,
    *,
    include_time_fixed: bool = True,
    random_slope: bool = True,
    return_pvalue: bool = False,
    return_all: bool = False,
) -> float | tuple[float, float] | dict[str, float]:
    """Score the greenery fixed effect in a mixed-effects linear model.

    Parameters
    ----------
    metric
        One of :data:`MIXEDLM_METRICS`.
    outcome, greenery, years_since_baseline
        1-D arrays of equal length.
    entity_id
        1-D array (object dtype OK) — the random-effects grouping variable.
    covariates
        Optional ``(n, n_cov)`` matrix of additional fixed effects.
    include_time_fixed
        Add ``years_since_baseline`` as a fixed-effect predictor.
    random_slope
        Random slope on time in addition to the random intercept.
    return_pvalue
        For tstat / coef metrics, also return the greenery fixed effect's
        Wald p-value. Other metrics return ``1.0`` as a sentinel.
    return_all
        Return a ``dict`` of all four metric values from the same fit (uses
        one extra fit for the LR test). Overrides ``return_pvalue``.
    """
    if metric not in MIXEDLM_METRICS:
        raise ValueError(
            f"Unknown metric {metric!r}; expected one of {sorted(MIXEDLM_METRICS)}."
        )

    y = np.asarray(outcome, dtype=np.float64)
    g = np.asarray(greenery, dtype=np.float64)
    t = np.asarray(years_since_baseline, dtype=np.float64)
    cov = _coerce_2d(covariates)

    y, g, eids, t, cov = _drop_nan(y, g, entity_id, t, cov)

    n_groups = len(np.unique(eids)) if len(eids) else 0
    if (
        len(y) < _MIN_ROWS
        or n_groups < 2
        or float(np.var(y)) == 0
        or float(np.var(g)) == 0
    ):
        if return_all:
            return {m: _DEGENERATE[m] for m in MIXEDLM_METRICS}
        s = _DEGENERATE[metric]
        return (s, 1.0) if return_pvalue else s

    X_full, g_col = _build_fixed_design(
        g,
        cov,
        t,
        include_time_fixed=include_time_fixed,
        include_greenery=True,
    )
    exog_re = _build_re_design(t, random_slope=random_slope)
    result_full = _fit_mixedlm(y, X_full, eids, exog_re)
    if result_full is None:
        if return_all:
            return {m: _DEGENERATE[m] for m in MIXEDLM_METRICS}
        s = _DEGENERATE[metric]
        return (s, 1.0) if return_pvalue else s

    # Optional null refit for the LR statistic.
    result_null = None
    if metric == "mixedlm_lr" or return_all:
        X_null, _ = _build_fixed_design(
            g,
            cov,
            t,
            include_time_fixed=include_time_fixed,
            include_greenery=False,
        )
        result_null = _fit_mixedlm(y, X_null, eids, exog_re)

    all_metrics = _compute_all_metrics(
        result_full, X_full, g_col, exog_re, result_null=result_null
    )
    if return_all:
        return all_metrics

    pval = 1.0
    if metric in HAS_PVALUE:
        try:
            pvals = _as_array(result_full.pvalues)
            pval = float(pvals[g_col])
            if not np.isfinite(pval):
                pval = 1.0
        except Exception:
            pval = 1.0

    score = all_metrics[metric]
    return (score, pval) if return_pvalue else score
