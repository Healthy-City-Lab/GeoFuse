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
exception in spirit, but the engine optimizes ``|coef|``-style magnitude
by configuration; see :data:`HIGHER_IS_BETTER`). A MixedLM convergence
failure, singular design, or input with no variance returns the
degenerate value ``0.0`` so a bad trial fails soft instead of crashing.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

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

# Metrics the fast fixed-V GLS search scorer produces directly from one GLS
# solve (the greenery coefficient and its Wald t). The variance-decomposition
# metrics (marginal_r2, lr) need the full likelihood machinery, so a search on
# those falls back to the exact per-trial fit.
FAST_SEARCH_METRICS: frozenset[str] = frozenset({"mixedlm_tstat", "mixedlm_coef"})

# Which model term the metric is computed on (what the search optimises the CGI
# for). ``level`` is the greenery main effect (association with the outcome
# level); the ``decline_*`` targets score a greenery × time slope — overall,
# between-person (person-mean × time), or within-person (deviation × time).
ASSOCIATION_TARGETS: frozenset[str] = frozenset(
    {"level", "decline_overall", "decline_average", "decline_change"}
)
DEFAULT_ASSOCIATION_TARGET: str = "level"

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
    """True where the entity-id entry is None / NaN / NaT. Object-array safe."""
    return np.asarray(pd.isna(np.asarray(arr)), dtype=bool)


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


def _within_between(
    g: np.ndarray, eids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Person-mean (between) and deviation-from-mean (within) of ``g``.

    Returns ``(mean, deviation, within_varies)``. ``within_varies`` is False when
    the exposure is constant within every entity (no over-time variation).
    """
    order = np.argsort(eids, kind="stable")
    _uniq, first_idx, counts = np.unique(
        eids[order], return_index=True, return_counts=True
    )
    means = np.add.reduceat(g[order], first_idx) / counts
    gmean = np.empty_like(g)
    gmean[order] = np.repeat(means, counts)
    gdev = g - gmean
    return gmean, gdev, bool(float(np.var(gdev)) > 0)


def _build_target_design(
    g: np.ndarray,
    cov: np.ndarray | None,
    t: np.ndarray,
    eids: np.ndarray,
    *,
    target: str,
    include_time_fixed: bool,
    include_target: bool,
):
    """Fixed design + scored-column index for the requested association target.

    ``level`` reproduces the greenery main-effect design exactly. The
    ``decline_*`` targets add a greenery × time slope (overall, or the
    between/within decomposition) and score its coefficient; time is always a
    fixed effect there. ``include_target=False`` drops the scored term (the LR
    null). Returns ``(X, target_col)`` or ``None`` when the target is not
    estimable (a within-person slope with no over-time exposure variation).
    """
    if target == "level":
        return _build_fixed_design(
            g,
            cov,
            t,
            include_time_fixed=include_time_fixed,
            include_greenery=include_target,
        )

    n = len(g)
    named: list[tuple[str, np.ndarray]] = [("intercept", np.ones(n))]

    def _add_cov() -> None:
        if cov is not None and cov.shape[1] > 0:
            for j in range(cov.shape[1]):
                named.append((f"cov{j}", cov[:, j]))

    if target == "decline_overall":
        named.append(("greenery", g))
        _add_cov()
        named.append(("time", t))
        if include_target:
            named.append(("__target__", g * t))
    else:
        gmean, gdev, within_ok = _within_between(g, eids)
        if target == "decline_change" and not within_ok:
            return None
        named.append(("g_between", gmean))
        if within_ok:
            named.append(("g_within", gdev))
        _add_cov()
        named.append(("time", t))
        if target == "decline_average":
            if include_target:
                named.append(("__target__", gmean * t))
            if within_ok:
                named.append(("g_within_x_time", gdev * t))
        else:  # decline_change (within_ok is True here)
            named.append(("g_between_x_time", gmean * t))
            if include_target:
                named.append(("__target__", gdev * t))

    X = np.column_stack([c[1] for c in named])
    target_col = next(
        (i for i, (nm, _) in enumerate(named) if nm == "__target__"), -1
    )
    return X, target_col


def _fit_mixedlm(
    outcome: np.ndarray,
    X: np.ndarray,
    groups: np.ndarray,
    exog_re: np.ndarray,
    *,
    reml: bool = True,
):
    """Fit a MixedLM with lbfgs, swallowing fit failures.

    Returns the fitted result or ``None`` if anything went wrong (singular
    design, non-convergence, numerical failure). Callers map ``None`` to
    the metric's degenerate score so a bad trial doesn't kill the study.

    ``reml=True`` (default) is the right estimator for variance components and
    for the scoring metrics. AIC/BIC comparisons across models with **different
    fixed effects** must pass ``reml=False`` — REML likelihoods aren't
    comparable when the fixed-effects design changes.
    """
    from statsmodels.regression.mixed_linear_model import MixedLM

    try:
        model = MixedLM(endog=outcome, exog=X, groups=groups, exog_re=exog_re)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(method="lbfgs", reml=reml)
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
    # A proportion of variance explained can't be negative; tiny negatives are
    # floating-point noise around zero, but a large one signals a real problem.
    if marginal_r2 < 0.0:
        if marginal_r2 < -1e-6:
            logger.debug("marginal_r2 came back at %.3e; clamping to 0", marginal_r2)
        marginal_r2 = 0.0

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


def _residualize_on(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """OLS residuals of ``y ~ X`` (intercept added)."""
    Xc = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    return y - Xc @ beta


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
    spatial_basis: np.ndarray | None = None,
    spatial_method: str = "none",
    nan_on_fail: bool = False,
    target: str = DEFAULT_ASSOCIATION_TARGET,
) -> float | tuple[float, float] | dict[str, float]:
    """Score a greenery model term in a mixed-effects linear model.

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
    spatial_basis, spatial_method
        Optional df-selected coordinate smooth and the adjustment method. With
        ``ks_aic`` the smooth enters the model as extra fixed effects; with
        ``spatial_plus`` the greenery is additionally residualized on the smooth
        before fitting. ``none`` (or no basis) leaves the model unchanged.
    nan_on_fail
        Return ``NaN`` instead of the degenerate ``0.0`` when the model can't be
        fit or the input is degenerate. Off by default (a bad trial fails soft to
        ``0.0`` so Optuna avoids it); the cluster bootstrap turns it on so a
        non-converged replicate is dropped rather than counted as a zero effect.
    target
        Which greenery term the metric scores — one of :data:`ASSOCIATION_TARGETS`.
        ``level`` (default) is the main-effect association; the ``decline_*``
        targets score a greenery × time slope (overall / between / within).
    """
    if metric not in MIXEDLM_METRICS:
        raise ValueError(
            f"Unknown metric {metric!r}; expected one of {sorted(MIXEDLM_METRICS)}."
        )
    if target not in ASSOCIATION_TARGETS:
        raise ValueError(
            f"Unknown target {target!r}; expected one of {sorted(ASSOCIATION_TARGETS)}."
        )

    def _degenerate():
        if return_all:
            return {
                m: (float("nan") if nan_on_fail else _DEGENERATE[m])
                for m in MIXEDLM_METRICS
            }
        s = float("nan") if nan_on_fail else _DEGENERATE[metric]
        return (s, float("nan") if nan_on_fail else 1.0) if return_pvalue else s

    y = np.asarray(outcome, dtype=np.float64)
    g = np.asarray(greenery, dtype=np.float64)
    t = np.asarray(years_since_baseline, dtype=np.float64)
    cov = _coerce_2d(covariates)
    sb = _coerce_2d(spatial_basis) if spatial_method != "none" else None

    # Combine covariates and the smooth for joint NaN masking, then split back.
    n_cov = 0 if cov is None else cov.shape[1]
    combined = cov if sb is None else (sb if cov is None else np.hstack([cov, sb]))
    y, g, eids, t, combined = _drop_nan(y, g, entity_id, t, combined)
    if sb is None:
        cov = combined
    elif combined is None:
        cov, sb = None, None
    else:
        cov = combined[:, :n_cov] if n_cov > 0 else None
        sb = combined[:, n_cov:]

    # Fold the smooth in: as fixed effects (both methods) and, for spatial_plus,
    # by residualizing the greenery exposure on it first.
    if sb is not None and sb.shape[1] > 0 and len(g) > sb.shape[1] + 1:
        if spatial_method == "spatial_plus" and float(np.var(g)) > 0:
            g = _residualize_on(g, sb)
        cov = sb if cov is None else np.hstack([cov, sb])

    n_groups = len(np.unique(eids)) if len(eids) else 0
    if (
        len(y) < _MIN_ROWS
        or n_groups < 2
        or float(np.var(y)) == 0
        or float(np.var(g)) == 0
    ):
        return _degenerate()

    built = _build_target_design(
        g, cov, t, eids, target=target, include_time_fixed=include_time_fixed,
        include_target=True,
    )
    if built is None:  # e.g. a within-person slope with no over-time variation
        return _degenerate()
    X_full, g_col = built
    exog_re = _build_re_design(t, random_slope=random_slope)
    result_full = _fit_mixedlm(y, X_full, eids, exog_re)
    if result_full is None:
        return _degenerate()

    # Optional null refit for the LR statistic (drops the scored term only).
    result_null = None
    if metric == "mixedlm_lr" or return_all:
        null_built = _build_target_design(
            g, cov, t, eids, target=target,
            include_time_fixed=include_time_fixed, include_target=False,
        )
        X_null = null_built[0] if null_built is not None else None
        result_null = (
            _fit_mixedlm(y, X_null, eids, exog_re) if X_null is not None else None
        )
        # A failed null refit would make the LR statistic collapse to 0.0,
        # indistinguishable from a genuinely tiny LR. For a single-metric LR
        # request treat that as a fit failure so the bootstrap can drop it.
        if metric == "mixedlm_lr" and not return_all and result_null is None:
            return _degenerate()

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


# ---------------------------------------------------------------------------
# Fast fixed-V GLS search scorer (ranking only; reports use the exact fit)
# ---------------------------------------------------------------------------
#
# The exact per-trial MixedLM fit re-estimates the whole variance-component
# model on every trial when only the greenery column changes. Because the
# random-effects covariance ``D`` and residual variance ``σ²`` are driven by
# the outcome's within-entity structure — not by which greenery column is in
# the design — they are estimated **once per fold** and reused, reducing each
# trial to a GLS solve. This is a score-test / one-step estimator used only to
# **rank** trials; every reported number is an exact MixedLM refit.
#
# Marginal covariance is block-diagonal by entity, ``V_i = Z_i D Z_i' + σ² I``
# (blocks ``k×k`` with ``k`` the entity's observation count, typically 2-3).
# Entities are bucketed by ``k``; each bucket's inverses are one batched
# ``np.linalg.inv``; ``X'V^-1X`` / ``X'V^-1y`` accumulate with ``np.einsum``.
#
# Base-component estimators (``search_scoring_method``):
#   ``mom``      — closed-form method of moments (fastest).
#   ``mom_em1``  — MoM + 1 EM refinement step.
#   ``mom_em3``  — MoM + 3 EM refinement steps (near-REML fidelity; default).
#   ``reml``     — exact statsmodels REML fit of the base model.
# (The engine's ``"exact"`` scoring option bypasses this module entirely and
# fits a full MixedLM per trial via :func:`score_mixedlm`.)

# Fast-scorer component methods that use the fixed-V GLS path (i.e. not the
# per-trial exact fit).
FAST_COMPONENT_METHODS: frozenset[str] = frozenset(
    {"mom", "mom_em1", "mom_em3", "reml"}
)
_EM_STEPS: dict[str, int] = {"mom": 0, "mom_em1": 1, "mom_em3": 3}


def _entity_blocks(eids: np.ndarray) -> dict[int, np.ndarray]:
    """Row indices grouped by entity, bucketed by observation count ``k``.

    Returns ``{k: idx}`` where ``idx`` is ``(m_k, k)`` — one row per entity,
    its ``k`` observations. Same-``k`` entities share a stacked shape so their
    covariance blocks invert in one batched call.
    """
    order = np.argsort(eids, kind="stable")
    _u, first, counts = np.unique(eids[order], return_index=True, return_counts=True)
    by_k: dict[int, list[np.ndarray]] = {}
    for f, c in zip(first, counts):
        by_k.setdefault(int(c), []).append(order[f : f + c])
    return {k: np.asarray(v, dtype=np.int64) for k, v in by_k.items()}


def _mom_components(
    y: np.ndarray, Xb: np.ndarray, Z: np.ndarray, blocks: dict[int, np.ndarray]
) -> tuple[np.ndarray, float]:
    """Method-of-moments ``(D, σ²)`` from OLS residual products.

    ``E[r_ij r_ik] = z_ij' D z_ik + σ² 1{j=k}``. Regressing the within-entity
    residual products on that design gives ``D``'s unique entries and ``σ²`` in
    one least-squares solve; ``D`` is projected to the nearest PSD matrix.
    """
    r_dim = Z.shape[1]
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    resid = y - Xb @ beta
    pairs = [(a, b) for a in range(r_dim) for b in range(a, r_dim)]
    A_rows: list[np.ndarray] = []
    b_rows: list[np.ndarray] = []
    for k, gi in blocks.items():
        rr = resid[gi]
        Zi = Z[gi]
        prod = np.einsum("ej,el->ejl", rr, rr).reshape(-1)
        feats = []
        for a, b in pairs:
            za_zb = Zi[:, :, a][:, :, None] * Zi[:, :, b][:, None, :]
            if a != b:
                za_zb = za_zb + Zi[:, :, b][:, :, None] * Zi[:, :, a][:, None, :]
            feats.append(za_zb.reshape(-1))
        feats.append(np.broadcast_to(np.eye(k), (len(gi), k, k)).reshape(-1))
        A_rows.append(np.column_stack(feats))
        b_rows.append(prod)
    coef, *_ = np.linalg.lstsq(np.vstack(A_rows), np.concatenate(b_rows), rcond=None)
    D = np.zeros((r_dim, r_dim))
    for (a, b), c in zip(pairs, coef[:-1]):
        D[a, b] = D[b, a] = c
    w, V = np.linalg.eigh(D)
    D = V @ np.diag(np.clip(w, 1e-9, None)) @ V.T
    return D, max(float(coef[-1]), 1e-9)


def _block_vinv(
    Z: np.ndarray, blocks: dict[int, np.ndarray], D: np.ndarray, s2: float
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per-bucket ``(row_index, Z_bucket, V^-1)`` for the given components."""
    out = {}
    for k, gi in blocks.items():
        Zi = Z[gi]
        Vi = np.einsum("mka,ab,mlb->mkl", Zi, D, Zi) + s2 * np.eye(k)
        out[k] = (gi, Zi, np.linalg.inv(Vi))
    return out


def _gls_fixed_effects(
    y: np.ndarray,
    X: np.ndarray,
    vinv: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """GLS fixed-effect estimates ``beta`` and their standard errors."""
    p = X.shape[1]
    A = np.zeros((p, p))
    b = np.zeros(p)
    for _k, (gi, _Zi, Vv) in vinv.items():
        Xi = X[gi]
        yi = y[gi]
        A += np.einsum("mka,mkl,mlb->ab", Xi, Vv, Xi)
        b += np.einsum("mka,mkl,ml->a", Xi, Vv, yi)
    A_inv = np.linalg.inv(A)
    return A_inv @ b, np.sqrt(np.clip(np.diag(A_inv), 0.0, None))


def _em_step(
    y: np.ndarray,
    Xb: np.ndarray,
    Z: np.ndarray,
    blocks: dict[int, np.ndarray],
    D: np.ndarray,
    s2: float,
) -> tuple[np.ndarray, float]:
    """One EM update of ``(D, σ²)`` (Laird-Ware), given the base design."""
    vinv = _block_vinv(Z, blocks, D, s2)
    beta, _ = _gls_fixed_effects(y, Xb, vinv)
    rfull = y - Xb @ beta
    r_dim = Z.shape[1]
    Dacc = np.zeros((r_dim, r_dim))
    ete = 0.0
    tr = 0.0
    m_tot = 0
    n_tot = 0
    for k, (gi, Zi, Vv) in vinv.items():
        m = len(gi)
        m_tot += m
        n_tot += m * k
        ri = rfull[gi]
        ztv = np.einsum("mka,mkl->mal", Zi, Vv)
        bpost = np.einsum("ab,mbl,ml->ma", D, ztv, ri)
        M = np.einsum("mka,mkl,mlb->mab", Zi, Vv, Zi)
        Vb = D[None] - np.einsum("ac,mcd,db->mab", D, M, D)
        Dacc += np.einsum("ma,mb->ab", bpost, bpost) + Vb.sum(0)
        e = ri - np.einsum("mka,ma->mk", Zi, bpost)
        ete += float(np.sum(e**2))
        tr += float(np.einsum("mka,mab,mkb->m", Zi, Vb, Zi).sum())
    return Dacc / m_tot, max((ete + tr) / n_tot, 1e-9)


def _reml_base_components(
    y: np.ndarray, Xb: np.ndarray, eids: np.ndarray, Z: np.ndarray
) -> tuple[np.ndarray, float]:
    from statsmodels.regression.mixed_linear_model import MixedLM

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MixedLM(endog=y, exog=Xb, groups=eids, exog_re=Z).fit(
            method="lbfgs", reml=True
        )
    if not getattr(res, "converged", True):
        raise RuntimeError("REML base fit did not converge")
    return np.atleast_2d(np.asarray(res.cov_re)), float(res.scale)


def _estimate_components(
    y: np.ndarray,
    Xb: np.ndarray,
    eids: np.ndarray,
    Z: np.ndarray,
    blocks: dict[int, np.ndarray],
    method: str,
) -> tuple[np.ndarray, float]:
    """Estimate ``(D, σ²)`` for one fold by the requested method."""
    if method == "reml":
        return _reml_base_components(y, Xb, eids, Z)
    if method not in _EM_STEPS:
        raise ValueError(f"Unknown component method {method!r}.")
    D, s2 = _mom_components(y, Xb, Z, blocks)
    for _ in range(_EM_STEPS[method]):
        D, s2 = _em_step(y, Xb, Z, blocks, D, s2)
    return D, s2


def _fast_greenery_tstat(
    y: np.ndarray,
    X_full: np.ndarray,
    Z: np.ndarray,
    eids: np.ndarray,
    D: np.ndarray,
    s2: float,
    target_col: int,
) -> tuple[float, float]:
    """Greenery ``(|t|, coef)`` via fixed-``V`` GLS on this trial's rows.

    The block structure and inverses are rebuilt from this trial's rows, so
    NaN-dropped trials are handled naturally.
    """
    blocks = _entity_blocks(eids)
    vinv = _block_vinv(Z, blocks, D, s2)
    beta, se = _gls_fixed_effects(y, X_full, vinv)
    coef = float(beta[target_col])
    s = float(se[target_col])
    return (abs(coef / s) if s > 0 else 0.0), coef


def estimate_fold_components(
    outcome: np.ndarray,
    covariates: np.ndarray | None,
    years_since_baseline: np.ndarray,
    entity_id: np.ndarray,
    *,
    method: str,
    include_time_fixed: bool,
    random_slope: bool,
    target: str = DEFAULT_ASSOCIATION_TARGET,
) -> tuple[np.ndarray, float] | None:
    """Per-fold random-effects covariance ``D`` and residual variance ``σ²``.

    Estimated once per fold from the greenery-free base model and reused by
    every trial's :func:`score_mixedlm_fast`. Returns ``None`` when the fold is
    degenerate or estimation fails (caller falls back to the exact per-trial
    fit). ``method`` is one of :data:`FAST_COMPONENT_METHODS`.
    """
    y = np.asarray(outcome, dtype=np.float64)
    t = np.asarray(years_since_baseline, dtype=np.float64)
    cov = _coerce_2d(covariates)
    # NaN-mask on the base inputs only (greenery excluded here).
    g0 = np.zeros_like(y)
    y, _g, eids, t, cov = _drop_nan(y, g0, entity_id, t, cov)
    if len(y) < _MIN_ROWS or len(np.unique(eids)) < 2 or float(np.var(y)) == 0:
        return None
    base = _build_target_design(
        _g, cov, t, eids, target=target,
        include_time_fixed=include_time_fixed, include_target=False,
    )
    if base is None:
        return None
    X_base = base[0]
    Z = _build_re_design(t, random_slope=random_slope)
    try:
        blocks = _entity_blocks(eids)
        return _estimate_components(y, X_base, eids, Z, blocks, method)
    except Exception:
        return None


def score_mixedlm_fast(
    metric: str,
    outcome: np.ndarray,
    greenery: np.ndarray,
    entity_id: np.ndarray,
    years_since_baseline: np.ndarray,
    covariates: np.ndarray | None,
    *,
    components: tuple[np.ndarray, float],
    include_time_fixed: bool = True,
    random_slope: bool = True,
    return_pvalue: bool = False,
    target: str = DEFAULT_ASSOCIATION_TARGET,
    nan_on_fail: bool = False,
):
    """Fast greenery score for one trial via fixed-``V`` GLS.

    Mirrors :func:`score_mixedlm`'s NaN masking and design, but reuses the
    per-fold ``components`` (from :func:`estimate_fold_components`) and solves a
    single GLS instead of fitting a ``MixedLM``. Supports the
    :data:`FAST_SEARCH_METRICS` (``mixedlm_tstat`` / ``mixedlm_coef``); other
    metrics must use the exact scorer. Spatial-confounding smooths are not
    handled here — the caller uses the exact path when a smooth is active.
    """

    def _degenerate():
        s = float("nan") if nan_on_fail else _DEGENERATE[metric]
        return (s, float("nan") if nan_on_fail else 1.0) if return_pvalue else s

    if metric not in FAST_SEARCH_METRICS:
        return _degenerate()
    D, s2 = components
    y = np.asarray(outcome, dtype=np.float64)
    g = np.asarray(greenery, dtype=np.float64)
    t = np.asarray(years_since_baseline, dtype=np.float64)
    cov = _coerce_2d(covariates)
    y, g, eids, t, cov = _drop_nan(y, g, entity_id, t, cov)
    if (
        len(y) < _MIN_ROWS
        or len(np.unique(eids)) < 2
        or float(np.var(y)) == 0
        or float(np.var(g)) == 0
    ):
        return _degenerate()
    built = _build_target_design(
        g, cov, t, eids, target=target,
        include_time_fixed=include_time_fixed, include_target=True,
    )
    if built is None:
        return _degenerate()
    X_full, target_col = built
    if target_col < 0:
        return _degenerate()
    Z = _build_re_design(t, random_slope=random_slope)
    try:
        tstat, coef = _fast_greenery_tstat(y, X_full, Z, eids, D, s2, target_col)
    except Exception:
        return _degenerate()
    if not np.isfinite(tstat):
        return _degenerate()
    score = coef if metric == "mixedlm_coef" else tstat
    if return_pvalue:
        from scipy import stats as _sstats

        pval = float(2.0 * _sstats.norm.sf(abs(tstat)))
        return score, (pval if np.isfinite(pval) else 1.0)
    return score


# ---------------------------------------------------------------------------
# Penalized model comparison (CGI vs best standalone channel) — longitudinal
# ---------------------------------------------------------------------------


def compare_models_aic_bic_mixedlm(
    outcome: np.ndarray,
    channel_matrix: np.ndarray,
    channel_names: list[str],
    best_channel_idx: int,
    entity_id: np.ndarray,
    years_since_baseline: np.ndarray,
    covariates: np.ndarray | None = None,
    *,
    include_time_fixed: bool = True,
    random_slope: bool = True,
) -> dict:
    """AIC/BIC comparison of a full multi-channel MixedLM vs the best single channel.

    The longitudinal analogue of
    :func:`geofuse.objective_scoring.compare_models_aic_bic`. Fits two mixed
    models with the same random-effects structure used for scoring:

    * **full** — ``outcome ~ all channels + covariates [+ time] + (RE | entity)``
    * **reduced** — ``outcome ~ best channel + covariates [+ time] + (RE | entity)``

    Returns the same dict shape, with ``delta_*`` = reduced − full (positive
    favours the full CGI model). On a fit failure or non-finite criterion the
    result is ``{"ok": False, ...}`` so the caller can fall back gracefully.
    """
    y = np.asarray(outcome, dtype=np.float64).ravel()
    X = np.asarray(channel_matrix, dtype=np.float64)
    t = np.asarray(years_since_baseline, dtype=np.float64).ravel()
    cov = _coerce_2d(covariates)

    if X.ndim != 2 or X.shape[1] != len(channel_names):
        raise ValueError("channel_matrix shape must match channel_names length.")
    if not (0 <= best_channel_idx < X.shape[1]):
        raise ValueError(f"best_channel_idx {best_channel_idx} out of range.")

    eids = np.asarray(entity_id)
    mask = (
        np.isfinite(y)
        & np.isfinite(X).all(axis=1)
        & np.isfinite(t)
        & ~_entity_missing_mask(eids)
    )
    if cov is not None:
        mask &= np.isfinite(cov).all(axis=1)
    y, X, t, eids = y[mask], X[mask], t[mask], eids[mask]
    cov = None if cov is None else cov[mask]

    fail = {
        "ok": False,
        "channel_names": list(channel_names),
        "best_channel": channel_names[best_channel_idx],
    }
    if len(y) < _MIN_ROWS or len(np.unique(eids)) < 2:
        return {**fail, "reason": "too few rows / entities for a MixedLM comparison"}

    def _fixed(cols: np.ndarray) -> np.ndarray:
        parts: list[np.ndarray] = [np.ones((len(y), 1)), cols]
        if cov is not None and cov.shape[1] > 0:
            parts.append(cov)
        if include_time_fixed:
            parts.append(t.reshape(-1, 1))
        return np.hstack(parts)

    # ML (not REML) so the two fixed-effects structures are comparable.
    exog_re = _build_re_design(t, random_slope=random_slope)
    full = _fit_mixedlm(y, _fixed(X), eids, exog_re, reml=False)
    reduced = _fit_mixedlm(
        y, _fixed(X[:, [best_channel_idx]]), eids, exog_re, reml=False
    )
    if full is None or reduced is None:
        return {**fail, "reason": "MixedLM fit did not converge"}

    try:
        aic_full, bic_full = float(full.aic), float(full.bic)
        aic_reduced, bic_reduced = float(reduced.aic), float(reduced.bic)
    except Exception:
        return {**fail, "reason": "AIC/BIC unavailable from the fitted model"}
    if not all(np.isfinite([aic_full, bic_full, aic_reduced, bic_reduced])):
        return {**fail, "reason": "non-finite AIC/BIC"}

    from .objective_scoring import _verdict_from_delta_bic

    delta_bic = bic_reduced - bic_full
    return {
        "ok": True,
        "channel_names": list(channel_names),
        "best_channel": channel_names[best_channel_idx],
        "aic_full": aic_full,
        "bic_full": bic_full,
        "aic_reduced": aic_reduced,
        "bic_reduced": bic_reduced,
        "delta_aic": float(aic_reduced - aic_full),
        "delta_bic": float(delta_bic),
        "n": int(len(y)),
        "verdict": _verdict_from_delta_bic(delta_bic),
    }


# ---------------------------------------------------------------------------
# Covariate impact — mixed-effects analogue of the OLS covariate table
# ---------------------------------------------------------------------------


def _marginal_r2_from_fit(result, X: np.ndarray, exog_re: np.ndarray) -> float:
    """Nakagawa marginal R² of a fitted MixedLM: fixed-effects variance share.

    ``var(Xβ) / (var(Xβ) + random-effects variance + residual variance)``, using
    the same random-effects variance approximation as :func:`_compute_all_metrics`
    (exact for a random intercept, approximate under a random slope).
    """
    fe = _as_array(result.fe_params)
    var_fe = float(np.var(X @ fe, ddof=0))
    sigma2_resid = float(result.scale)
    re_var = 0.0
    cov_re = getattr(result, "cov_re", None)
    if cov_re is not None:
        cov_re_mat = np.asarray(cov_re)
        for i in range(cov_re_mat.shape[0]):
            re_var += float(cov_re_mat[i, i]) * float(np.mean(exog_re[:, i] ** 2))
    total = var_fe + re_var + sigma2_resid
    return var_fe / total if total > 0 else 0.0


def covariate_impact_mixedlm(
    outcome: np.ndarray,
    greenery: np.ndarray,
    entity_id: np.ndarray,
    years_since_baseline: np.ndarray,
    covariates: np.ndarray,
    covariate_names: list[str],
    *,
    include_time_fixed: bool = True,
    random_slope: bool = True,
) -> dict | None:
    """Mixed-effects analogue of :meth:`MetricFusionEngine.compute_covariate_impact`.

    Fits ``outcome ~ greenery + covariates [+ time] + (RE | entity)`` and reports
    each covariate's **fixed-effect** coefficient, standard error, Wald t / p, and
    marginal-R² contribution (drop-one refit), plus the greenery coefficient and
    the full / greenery-only marginal R². Because the standard errors come from
    the mixed model, they account for the within-entity correlation that makes a
    plain OLS covariate table anticonservative on panel data.

    Returns the same dict shape the OLS path returns (with an extra
    ``"model": "mixedlm"`` tag), or ``None`` on a fit failure / too-few-rows so
    the caller can fall back gracefully.
    """
    cov = _coerce_2d(covariates)
    if cov is None or cov.shape[1] == 0 or not covariate_names:
        return None

    y = np.asarray(outcome, dtype=np.float64).ravel()
    g = np.asarray(greenery, dtype=np.float64).ravel()
    t = np.asarray(years_since_baseline, dtype=np.float64).ravel()
    y, g, eids, t, cov = _drop_nan(y, g, entity_id, t, cov)

    n_groups = len(np.unique(eids)) if len(eids) else 0
    if (
        len(y) < max(_MIN_ROWS, len(covariate_names) + 3)
        or n_groups < 2
        or float(np.var(y)) == 0
        or float(np.var(g)) == 0
    ):
        return None

    exog_re = _build_re_design(t, random_slope=random_slope)

    # Full model: greenery + covariates [+ time]. Column order from
    # _build_fixed_design → [intercept, greenery, covariates..., time].
    X_full, g_col = _build_fixed_design(
        g, cov, t, include_time_fixed=include_time_fixed, include_greenery=True
    )
    full = _fit_mixedlm(y, X_full, eids, exog_re)
    if full is None:
        return None

    # Greenery-only model (no covariates) for the R² lift.
    X_cgi, _ = _build_fixed_design(
        g, None, t, include_time_fixed=include_time_fixed, include_greenery=True
    )
    cgi_only = _fit_mixedlm(y, X_cgi, eids, exog_re)

    fe = _as_array(full.fe_params)
    bse = _as_array(full.bse_fe)
    try:
        pvals = _as_array(full.pvalues)
    except Exception:
        pvals = np.full(len(fe), np.nan)

    r2_full = _marginal_r2_from_fit(full, X_full, exog_re)
    r2_cgi_only = (
        _marginal_r2_from_fit(cgi_only, X_cgi, exog_re) if cgi_only is not None else 0.0
    )

    n_cov = cov.shape[1]
    per_cov: list[dict] = []
    for i, name in enumerate(covariate_names[:n_cov]):
        slot = 2 + i  # [intercept(0), greenery(1), covariate_i(2+i), ...]
        coef = float(fe[slot])
        se = float(bse[slot]) if slot < len(bse) else float("nan")
        pval = float(pvals[slot]) if slot < len(pvals) else float("nan")
        t_stat = coef / se if np.isfinite(se) and se > 0 else float("nan")

        # Drop this covariate and refit for its marginal-R² contribution.
        cov_minus = np.delete(cov, i, axis=1)
        cov_minus = cov_minus if cov_minus.shape[1] > 0 else None
        X_minus, _ = _build_fixed_design(
            g, cov_minus, t, include_time_fixed=include_time_fixed, include_greenery=True
        )
        refit = _fit_mixedlm(y, X_minus, eids, exog_re)
        r2_minus = (
            _marginal_r2_from_fit(refit, X_minus, exog_re) if refit is not None else r2_full
        )
        partial_r2 = max(0.0, r2_full - r2_minus)

        direction = "positive" if coef > 0 else "negative" if coef < 0 else "—"
        per_cov.append(
            {
                "covariate": name,
                "coef": coef,
                "std_err": se,
                "t_stat": float(t_stat),
                "pvalue": pval,
                "direction": direction,
                "partial_r2": float(partial_r2),
            }
        )

    return {
        "model": "mixedlm",
        "n": int(len(y)),
        "r2_full": float(r2_full),
        "r2_cgi_only": float(r2_cgi_only),
        "r2_lift_from_covariates": float(r2_full - r2_cgi_only),
        "per_covariate": per_cov,
        "cgi_coef": float(fe[g_col]),
        "cgi_std_err": float(bse[g_col]) if g_col < len(bse) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Cluster (entity) bootstrap CI for a mixedlm_* metric
# ---------------------------------------------------------------------------


def cluster_bootstrap_metric_ci(
    metric: str,
    outcome: np.ndarray,
    greenery: np.ndarray,
    entity_id: np.ndarray,
    years_since_baseline: np.ndarray,
    covariates: np.ndarray | None = None,
    *,
    include_time_fixed: bool = True,
    random_slope: bool = True,
    spatial_basis: np.ndarray | None = None,
    spatial_method: str = "none",
    n_bootstrap: int = 300,
    ci_level: float = 0.95,
    seed: int = 42,
    target: str = DEFAULT_ASSOCIATION_TARGET,
) -> dict:
    """Cluster (entity) bootstrap percentile CI for a ``mixedlm_*`` metric.

    Resamples whole entities with replacement and **relabels each drawn entity
    with a fresh group id** — a doubled entity becomes two independent groups,
    which is required for a valid multilevel cluster bootstrap — then refits the
    mixed model per replicate via :func:`score_mixedlm` and takes the percentile
    CI on the metric. Resampling ``(entity, wave)`` rows independently instead
    would destroy the within-entity correlation and understate the interval.

    The point estimate (``observed``) and, for the ``tstat`` / ``coef`` metrics,
    the greenery fixed-effect Wald ``pvalue`` come from the single full-data fit.
    Returns a dict shaped like :func:`statistical_testing.bootstrap_score_ci`
    (``observed``, ``mean``, ``lower``, ``upper``, ``ci_level``, ``method``,
    ``n``) plus ``n_boot`` and (when available) ``pvalue``.

    ``n_bootstrap`` is a per-replicate model refit, so it is deliberately modest
    (a few hundred) rather than the thousands the O(1) OLS metrics use. A refit
    that fails to converge returns the metric's degenerate value (``0.0``) and so
    contributes to the distribution; with a well-powered test panel this is rare.
    """
    if metric not in MIXEDLM_METRICS:
        raise ValueError(
            f"cluster_bootstrap_metric_ci expects a mixedlm_* metric; got {metric!r}."
        )

    y = np.asarray(outcome, dtype=np.float64).ravel()
    g = np.asarray(greenery, dtype=np.float64).ravel()
    t = np.asarray(years_since_baseline, dtype=np.float64).ravel()
    eid = np.asarray(entity_id)
    cov = _coerce_2d(covariates)
    sb = _coerce_2d(spatial_basis) if spatial_method != "none" else None

    # Drop non-finite rows up front so the resampled group-row indices are valid.
    mask = (
        np.isfinite(y)
        & np.isfinite(g)
        & np.isfinite(t)
        & ~_entity_missing_mask(eid)
    )
    if cov is not None:
        mask &= np.isfinite(cov).all(axis=1)
    if sb is not None:
        mask &= np.isfinite(sb).all(axis=1)
    y, g, t, eid = y[mask], g[mask], t[mask], eid[mask]
    cov = None if cov is None else cov[mask]
    sb = None if sb is None else sb[mask]

    def _score(yy, gg, ee, tt, cc, ss, *, return_pvalue=False, nan_on_fail=False):
        return score_mixedlm(
            metric,
            yy,
            gg,
            entity_id=ee,
            years_since_baseline=tt,
            covariates=cc,
            include_time_fixed=include_time_fixed,
            random_slope=random_slope,
            return_pvalue=return_pvalue,
            spatial_basis=ss,
            spatial_method=spatial_method,
            nan_on_fail=nan_on_fail,
            target=target,
        )

    result: dict = {
        "observed": float("nan"),
        "mean": float("nan"),
        "lower": float("nan"),
        "upper": float("nan"),
        "ci_level": float(ci_level),
        "method": "cluster_bootstrap",
        "n": int(len(y)),
        "status": "ok",
    }

    # The observed point estimate uses ``nan_on_fail=True`` so a non-converged
    # or degenerate full-data fit surfaces as NaN, not a spurious ``0.0`` — the
    # bootstrap replicates already do this, so both halves now agree.
    if metric in HAS_PVALUE:
        obs, pval = _score(  # type: ignore[misc]
            y, g, eid, t, cov, sb, return_pvalue=True, nan_on_fail=True
        )
        result["observed"] = float(obs)
        result["pvalue"] = float(pval)
    else:
        result["observed"] = float(_score(y, g, eid, t, cov, sb, nan_on_fail=True))

    # A failed observed fit has no point estimate, so emit no CI — reporting a
    # bootstrap interval around a missing estimate is what produced CIs that
    # excluded their own ``0.0`` point.
    if not np.isfinite(result["observed"]):
        result["status"] = "fit_failed"
        return result

    uniq, inv = np.unique(eid, return_inverse=True)
    n_groups = len(uniq)
    if n_groups < 2 or len(y) < _MIN_ROWS:
        return result
    group_rows = [np.where(inv == k)[0] for k in range(n_groups)]

    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(n_bootstrap), dtype=np.float64)
    for i in range(int(n_bootstrap)):
        draw = rng.integers(0, n_groups, size=n_groups)
        idx = np.concatenate([group_rows[k] for k in draw])
        # Fresh per-slot group id so a repeated entity forms independent groups.
        new_eid = np.concatenate(
            [
                np.full(len(group_rows[k]), slot, dtype=np.int64)
                for slot, k in enumerate(draw)
            ]
        )
        try:
            boot[i] = float(
                _score(
                    y[idx],
                    g[idx],
                    new_eid,
                    t[idx],
                    None if cov is None else cov[idx],
                    None if sb is None else sb[idx],
                    nan_on_fail=True,
                )
            )
        except Exception:
            boot[i] = np.nan

    valid = boot[np.isfinite(boot)]
    if len(valid) == 0:
        return result
    alpha = (1.0 - float(ci_level)) / 2.0
    result["mean"] = float(np.mean(valid))
    result["lower"] = float(np.quantile(valid, alpha))
    result["upper"] = float(np.quantile(valid, 1.0 - alpha))
    result["n_boot"] = int(len(valid))
    return result


# ---------------------------------------------------------------------------
# Exposure-decline terms (greenery × time)
# ---------------------------------------------------------------------------


def _fit_terms(y, cols, eids, exog_re):
    """Fit ``y ~ intercept + cols`` (MixedLM) → {name: (coef, se, pvalue)} or None."""
    X = np.column_stack([np.ones(len(y))] + [c[1] for c in cols])
    res = _fit_mixedlm(y, X, eids, exog_re)
    if res is None:
        return None
    fe, bse = _as_array(res.fe_params), _as_array(res.bse_fe)
    try:
        pv = _as_array(res.pvalues)
    except Exception:
        pv = np.full(len(fe), np.nan)
    names = ["intercept"] + [c[0] for c in cols]
    return {
        nm: (
            float(fe[i]),
            float(bse[i]) if i < len(bse) else float("nan"),
            float(pv[i]) if i < len(pv) else float("nan"),
        )
        for i, nm in enumerate(names)
    }


def decline_terms_mixedlm(
    outcome: np.ndarray,
    greenery: np.ndarray,
    entity_id: np.ndarray,
    years_since_baseline: np.ndarray,
    covariates: np.ndarray | None = None,
    *,
    random_slope: bool = True,
    want_between: bool = False,
    want_within: bool = False,
) -> dict | None:
    """Greenery × time terms testing whether exposure is linked to the outcome's
    rate of change.

    Always reports the **overall** greenery × time slope. With ``want_between`` /
    ``want_within`` it additionally fits a within-between decomposition and
    reports the **average-exposure** (person-mean × time) and **exposure-change**
    (within-person deviation × time) slopes, each controlling for the other.
    Returns ``{"n", "within_estimable", "terms": [{key, coef, std_err, t_stat,
    pvalue, direction}]}`` or ``None`` on a fit failure / too little data.
    """
    y = np.asarray(outcome, dtype=np.float64).ravel()
    g = np.asarray(greenery, dtype=np.float64).ravel()
    t = np.asarray(years_since_baseline, dtype=np.float64).ravel()
    cov = _coerce_2d(covariates)
    y, g, eids, t, cov = _drop_nan(y, g, entity_id, t, cov)
    if len(y) < _MIN_ROWS or len(np.unique(eids)) < 2 or float(np.var(y)) == 0:
        return None

    cov_cols = [] if cov is None else [(f"cov{j}", cov[:, j]) for j in range(cov.shape[1])]
    exog_re = _build_re_design(t, random_slope=random_slope)

    # Person-mean (between) and deviation (within) exposure.
    gmean, gdev, within_estimable = _within_between(g, eids)

    def _row(key, entry):
        coef, se, pval = entry
        tstat = coef / se if np.isfinite(se) and se > 0 else float("nan")
        direction = "positive" if coef > 0 else "negative" if coef < 0 else "—"
        return {
            "key": key,
            "coef": coef,
            "std_err": se,
            "t_stat": float(tstat),
            "pvalue": pval,
            "direction": direction,
        }

    terms: list[dict] = []
    # Overall greenery × time (always).
    pooled = _fit_terms(
        y,
        [("greenery", g)] + cov_cols + [("time", t), ("greenery_x_time", g * t)],
        eids,
        exog_re,
    )
    if pooled is None:
        return None
    terms.append(_row("overall", pooled["greenery_x_time"]))

    # Within-between decomposition for the average / change slopes.
    if want_between or want_within:
        cols = [("g_between", gmean)]
        if within_estimable:
            cols.append(("g_within", gdev))
        cols += cov_cols + [("time", t)]
        if want_between:
            cols.append(("between_x_time", gmean * t))
        if want_within and within_estimable:
            cols.append(("within_x_time", gdev * t))
        decomposed = _fit_terms(y, cols, eids, exog_re)
        if decomposed is not None:
            if want_between and "between_x_time" in decomposed:
                terms.append(_row("between", decomposed["between_x_time"]))
            if want_within:
                if within_estimable and "within_x_time" in decomposed:
                    terms.append(_row("within", decomposed["within_x_time"]))
                else:
                    terms.append(
                        {
                            "key": "within",
                            "coef": float("nan"),
                            "std_err": float("nan"),
                            "t_stat": float("nan"),
                            "pvalue": float("nan"),
                            "direction": "—",
                        }
                    )

    return {"n": int(len(y)), "within_estimable": bool(within_estimable), "terms": terms}
