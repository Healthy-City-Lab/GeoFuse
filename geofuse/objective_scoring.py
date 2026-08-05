"""Covariate-aware objective scoring for the fusion engine.

When the user supplies covariates (additional outcome predictors that should
be controlled for), the optimizer's score becomes the **greenery term's
partial contribution** to predicting the outcome — not the full-model fit.
A dominant covariate must not be allowed to let the optimizer ignore the CGI
parameters; the run is meant to maximize how much the CGI itself explains
*over and above* the covariates.

Per-metric semantics
--------------------

``partial_distance_corr`` (default)
    Partial distance correlation (Székely–Rizzo) between target and CGI given
    the covariates (and spatial smooth). Detects linear *and* nonlinear
    association and conditions on the controls nonlinearly, so a covariate with
    a curved effect is removed rather than partly leaking into the score. Falls
    back to plain distance correlation when there is nothing to condition on.
    The ``residualize`` option does not apply (conditioning is intrinsic).
``distance_corr``
    Distance correlation between target and CGI. With covariates both sides are
    residualized first (linear or spline, see ``residualize``), then
    distance-correlated — the fast O(n log n) estimator, so it stays cheap even
    with covariates. Unsigned; direction is reported via
    :func:`relationship_sign`.
``spearman``
    Partial rank correlation magnitude: residualize the *ranks* of target and
    CGI on the covariates, then take ``|Pearson|`` of the residuals.
``r2``
    Incremental (partial) R²: ``R²(target ~ covariates + cgi) − R²(target ~
    covariates)``. Negative values are possible (CGI hurts the fit) and are
    returned as-is so Optuna can rank them.
``nrmse``
    Min-max normalized RMSE: target and CGI are each scaled to ``[0, 1]``
    (residualized first when covariates are present), then the RMSE of the
    pattern alignment is taken. Dimensionless and scale-free. Lower is better.
``mutual_info``
    Plain MI between target and CGI; **covariates are ignored** for this
    metric.

Residualization
---------------

For the residualizing metrics (``distance_corr``, ``spearman``, ``r2``,
``nrmse``) the covariates are partialled out by OLS. ``residualize_method``
picks the covariate basis: ``"linear"`` (default) enters each covariate as-is;
``"spline"`` expands continuous covariates into a natural-cubic-spline basis
first, so nonlinear covariate effects are removed. ``partial_distance_corr``
and ``mutual_info`` ignore this setting.

When no covariates are supplied, every metric collapses to its no-covariate
baseline.
"""

from __future__ import annotations

import hashlib
import logging
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, pearsonr, rankdata
from sklearn.metrics import mutual_info_score, r2_score

from . import pdcor

logger = logging.getLogger(__name__)

# Fallback events already warned about — each degraded code path logs once per
# process instead of flooding a 1000-trial study log.
_warned_fallbacks: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned_fallbacks:
        _warned_fallbacks.add(key)
        logger.warning(message)

# Public set of metric names this module knows how to score; engine-level
# validation should compare against it before calling :func:`score`.
SUPPORTED_METRICS: frozenset[str] = frozenset(
    {
        "partial_distance_corr",
        "distance_corr",
        "spearman",
        "r2",
        "nrmse",
        "mutual_info",
        # Binary outcomes: logistic regression of a 0/1 target on the composite
        # plus covariates. A Gaussian fit to a dichotomous column is a linear
        # probability model whose standard errors are wrong in a known
        # direction, so the greenspace literature's dichotomised outcomes get
        # their own scorer. See :mod:`geofuse.binary_longitudinal`.
        "logit_tstat",
        "logit_coef",
    }
)

# Metrics that require the target to hold exactly two distinct values. Selected
# on a continuous outcome they return the degenerate score, so the UI and the
# runner check this set before offering them.
BINARY_ONLY_METRICS: frozenset[str] = frozenset({"logit_tstat", "logit_coef"})

# The default cross-sectional objective: partial distance correlation captures
# linear and nonlinear association while conditioning on covariates nonlinearly.
DEFAULT_METRIC: str = "partial_distance_corr"

# Metrics whose return value is "higher is better" — useful when the caller
# needs a uniform direction for ranking. ``nrmse`` is the only one where lower
# is better.
HIGHER_IS_BETTER: frozenset[str] = frozenset(
    {
        "partial_distance_corr",
        "distance_corr",
        "spearman",
        "r2",
        "mutual_info",
        "logit_tstat",
    }
)

# Metrics for which the covariate-adjusted score *ignores* covariates. The UI
# uses this list to render a tooltip on the covariate multi-select.
COVARIATE_IGNORED: frozenset[str] = frozenset({"mutual_info"})

# Covariate residualization bases. ``linear`` enters covariates as-is; ``spline``
# expands continuous covariates into a natural-cubic-spline basis before OLS
# residualization. Ignored by ``partial_distance_corr`` and ``mutual_info``.
RESIDUALIZE_METHODS: frozenset[str] = frozenset({"linear", "spline"})

# Metrics that condition on covariates intrinsically (so ``residualize_method``
# has no effect). The UI greys the residualization control for these.
RESIDUALIZE_IGNORED: frozenset[str] = frozenset(
    {"partial_distance_corr", "mutual_info"}
)

# Spatial-confounding adjustment methods. ``none`` reproduces the plain
# covariate-residualized score. ``ks_aic`` (Keller & Szpiro) and ``spatial_plus``
# (df-Spatial+) fold a coordinate smooth into the residualization — see
# :func:`score` and :mod:`geofuse.spatial_basis`.
SPATIAL_METHODS: frozenset[str] = frozenset({"none", "ks_aic", "spatial_plus"})

# Worst score returned when the inputs are degenerate (all-NaN, constant, < 3
# rows).
_DEGENERATE_SCORE: dict[str, float] = {
    "partial_distance_corr": 0.0,
    "distance_corr": 0.0,
    "spearman": 0.0,
    "r2": 0.0,
    "nrmse": float("inf"),
    "mutual_info": 0.0,
    "logit_tstat": 0.0,
    "logit_coef": 0.0,
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


def _residualize(y: np.ndarray, X: np.ndarray | None) -> np.ndarray:
    """OLS residuals of ``y ~ X`` (intercept added). ``X=None`` returns ``y``."""
    if X is None or X.shape[1] == 0:
        return y
    Xc = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    return y - Xc @ beta


# Columns with at most this many distinct values are treated as discrete
# (dummies / ordinal codes) and left un-splined.
_SPLINE_MIN_UNIQUE: int = 7
_SPLINE_DF: int = 4


def _expand_covariate_basis(
    X: np.ndarray | None, method: str, cache: dict | None = None
) -> np.ndarray | None:
    """Optionally expand continuous covariate columns into a spline basis.

    ``method="linear"`` (or a degenerate input) returns ``X`` unchanged. With
    ``method="spline"`` each near-continuous column is replaced by a
    natural-cubic-spline basis so the later OLS residualization removes
    nonlinear covariate effects; discrete columns (dummies, coarse codes) pass
    through as-is. Falls back to the raw column on any expansion error.

    ``cache`` (optional dict) memoizes the expanded basis on a content
    fingerprint of ``X`` — per Optuna trial the covariate matrix is fixed per
    scored subset, so the patsy expansion only runs once per subset.
    """
    if X is None or method != "spline" or X.shape[1] == 0:
        return X
    cache_key: bytes | None = None
    if cache is not None:
        # Byte digest, not moment sums: a resampled covariate matrix holding the
        # same values in a different order must not hit another subset's basis.
        _x = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        _h = hashlib.blake2b(digest_size=16)
        _h.update(repr(_x.shape).encode())
        _h.update(_x.tobytes())
        cache_key = _h.digest()
        hit = cache.get(cache_key)
        if hit is not None:
            return hit
    try:
        from patsy import dmatrix
    except Exception:
        return X
    cols: list[np.ndarray] = []
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.unique(col[np.isfinite(col)]).size < _SPLINE_MIN_UNIQUE:
            cols.append(col.reshape(-1, 1))
            continue
        try:
            basis = np.asarray(
                dmatrix(
                    "cr(x, df=df) - 1",
                    {"x": col, "df": _SPLINE_DF},
                    return_type="dataframe",
                ),
                dtype=np.float64,
            )
            cols.append(basis if basis.shape[1] else col.reshape(-1, 1))
        except Exception:
            cols.append(col.reshape(-1, 1))
    expanded = np.column_stack(cols)
    if cache is not None and cache_key is not None:
        cache[cache_key] = expanded
        while len(cache) > 8:
            cache.pop(next(iter(cache)))
    return expanded


def _stack(*mats: np.ndarray | None) -> np.ndarray | None:
    """Column-stack the non-empty design matrices, or ``None`` if all are empty."""
    parts = [m for m in mats if m is not None and m.shape[1] > 0]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else np.column_stack(parts)


def _spatial_designs(
    cov: np.ndarray | None, spatial_basis: np.ndarray | None, method: str
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Control designs to residualize the (target, cgi) sides on, per method.

    * ``none`` / no basis → both sides get the covariates.
    * ``ks_aic`` → both sides get ``[covariates, spatial_basis]`` (symmetric
      partial association of cgi in ``target ~ cgi + covariates + smooth``).
    * ``spatial_plus`` → target gets ``[covariates, spatial_basis]``; cgi gets the
      spatial smooth only (the exposure is residualized on the smooth).
    """
    if spatial_basis is None or method == "none":
        return cov, cov
    both = _stack(cov, spatial_basis)
    if method == "spatial_plus":
        return both, spatial_basis
    return both, both


def _finite_mask(
    target: np.ndarray,
    cgi: np.ndarray,
    cov: np.ndarray | None,
    spatial_basis: np.ndarray | None,
) -> np.ndarray:
    """Row mask keeping observations finite across target / cgi / cov / basis."""
    mask = np.isfinite(target) & np.isfinite(cgi)
    if cov is not None:
        mask &= np.isfinite(cov).all(axis=1)
    if spatial_basis is not None:
        mask &= np.isfinite(spatial_basis).all(axis=1)
    return mask


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


# ────────────────────────────────────────────────────────────────────
# Distance correlation
# ────────────────────────────────────────────────────────────────────


def distance_correlation(x: np.ndarray, y: np.ndarray, *, seed: int = 0) -> float:
    """Distance correlation between two 1-D arrays (``[0, 1]``).

    Uses :mod:`dcor`'s fast O(n log n) mergesort estimator for the univariate
    case, so the reporting bootstrap / permutation passes stay cheap at large
    n. Returns ``0.0`` when either input is constant. ``seed`` is retained for
    signature stability.
    """
    import dcor

    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    if xa.shape != ya.shape:
        raise ValueError(
            f"x and y must have the same shape; got {xa.shape} vs {ya.shape}."
        )
    if len(xa) < 3:
        return 0.0
    if float(np.var(xa)) == 0 or float(np.var(ya)) == 0:
        return 0.0
    try:
        s = float(
            dcor.distance_correlation(
                xa, ya, method=dcor.DistanceCovarianceMethod.MERGESORT
            )
        )
    except Exception as exc:
        _warn_once(
            "dcor-mergesort",
            f"Fast mergesort distance correlation failed ({exc!r}); falling "
            "back to the O(n²) estimator for this process.",
        )
        s = float(dcor.distance_correlation(xa, ya))
    return s if np.isfinite(s) else 0.0


def partial_distance_correlation(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | None,
    *,
    cache: "dict | None" = None,
) -> float:
    """Partial distance correlation of ``x`` and ``y`` given ``z`` (Székely–Rizzo).

    Conditions on ``z`` in the U-centred distance space, so both the retained
    ``x``↔``y`` association and the removed ``z`` effect can be nonlinear. With
    ``z=None`` reduces to plain distance correlation. Can be slightly negative
    (``x`` adds nothing beyond ``z``); returned as-is so the optimizer ranks it.
    The estimate is O(n²); the runner warns at large n.

    ``cache`` (an :class:`collections.OrderedDict` owned by the engine) enables
    the fast path that reuses the U-centered matrices of the ``(x, z)`` sides
    across calls — pass the vector that stays constant per scored subset (the
    target) as ``x``. Oversized inputs fall back to the stock estimator.
    """
    import dcor

    xa = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    ya = np.asarray(y, dtype=np.float64).reshape(len(y), -1)
    if z is None or np.asarray(z).size == 0:
        return distance_correlation(xa.ravel(), ya.ravel())
    za = np.asarray(z, dtype=np.float64).reshape(len(xa), -1)
    if float(np.var(xa)) == 0 or float(np.var(ya)) == 0:
        return 0.0
    n = xa.shape[0]
    if cache is not None and 4 <= n <= pdcor.MAX_CACHE_N:
        try:
            s = pdcor.partial_distance_correlation_cached(xa, ya, za, cache)
            return s if np.isfinite(s) else 0.0
        except Exception as exc:
            _warn_once(
                "pdcor-cached",
                f"Cached partial distance correlation failed ({exc!r}); "
                "falling back to the stock dcor estimator for this process.",
            )
    try:
        s = float(dcor.partial_distance_correlation(xa, ya, za))
    except Exception as exc:
        _warn_once(
            "pdcor-unconditioned",
            "dcor.partial_distance_correlation failed "
            f"({exc!r}); returning the UNCONDITIONED distance correlation — "
            "covariates/smooth are NOT partialled out of this score.",
        )
        return distance_correlation(xa.ravel(), ya.ravel())
    return s if np.isfinite(s) else 0.0


# ────────────────────────────────────────────────────────────────────
# Direction reporting
# ────────────────────────────────────────────────────────────────────


def relationship_sign(
    target: np.ndarray,
    cgi: np.ndarray,
    covariates: np.ndarray | None = None,
    *,
    spatial_basis: np.ndarray | None = None,
    spatial_method: str = "none",
    residualize_method: str = "linear",
) -> int:
    """Sign (``+1`` / ``-1``) of the greenery↔outcome relationship.

    Distance correlation is unsigned, so the report needs a separate direction
    indicator. Uses the sign of the partial Spearman correlation (rank residuals
    on the covariates — spline-expanded when ``residualize_method="spline"`` — plus
    the spatial smooth), so the reported direction uses the same control basis as
    the score. Returns ``+1`` when higher greenery tracks higher outcome, ``-1``
    otherwise, and ``+1`` as a neutral default for degenerate inputs.
    """
    t = np.asarray(target, dtype=np.float64).ravel()
    c = np.asarray(cgi, dtype=np.float64).ravel()
    cov = _coerce_covariates(covariates)
    sb = _coerce_covariates(spatial_basis) if spatial_method != "none" else None
    mask = _finite_mask(t, c, cov, sb)
    if not mask.all():
        t, c = t[mask], c[mask]
        cov = None if cov is None else cov[mask]
        sb = None if sb is None else sb[mask]
    if len(t) < 3 or float(np.var(t)) == 0 or float(np.var(c)) == 0:
        return 1
    tt = rankdata(t)
    cc = rankdata(c)
    if cov is not None:
        cov_use = _expand_covariate_basis(
            np.column_stack([rankdata(cov[:, j]) for j in range(cov.shape[1])]),
            residualize_method,
        )
    else:
        cov_use = None
    x_t, x_c = _spatial_designs(cov_use, sb, spatial_method)
    tr = _residualize(tt, x_t)
    cr = _residualize(cc, x_c)
    if float(np.var(tr)) == 0 or float(np.var(cr)) == 0:
        return 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = float(np.corrcoef(tr, cr)[0, 1])
    if not np.isfinite(corr) or corr == 0:
        return 1
    return 1 if corr > 0 else -1


# ────────────────────────────────────────────────────────────────────
# Scoring
# ────────────────────────────────────────────────────────────────────


def score(
    metric: str,
    target: np.ndarray,
    cgi: np.ndarray,
    covariates: np.ndarray | None = None,
    *,
    return_pvalue: bool = False,
    spatial_basis: np.ndarray | None = None,
    spatial_method: str = "none",
    residualize_method: str = "linear",
    pdcor_cache: "dict | None" = None,
    spline_cache: "dict | None" = None,
) -> float | tuple[float, float]:
    """Score CGI's predictive power for ``target``, controlling for ``covariates``.

    ``target`` and ``cgi`` are 1-D arrays of equal length. ``covariates`` is
    either ``None`` (no control variables) or a 2-D ``(n_samples, n_covs)``
    array. NaN rows are dropped jointly. Returns a single float by default; if
    ``return_pvalue=True`` returns ``(score, 1.0)`` — the ``1.0`` is a retained
    sentinel (no metric feeds a p-value gate anymore; robustness comes from
    bootstrap stability selection).

    ``spatial_basis`` is an optional already-df-selected coordinate smooth (the
    output of :func:`geofuse.spatial_basis.select_df_aic`). With ``spatial_method``
    ``ks_aic`` it joins the covariates on both sides; with ``spatial_plus`` the
    exposure is residualized on the smooth only. ``spatial_method="none"`` (or a
    missing basis) reproduces the plain covariate-residualized score. The smooth
    is ignored by ``mutual_info`` (like covariates).

    ``residualize_method`` (``"linear"`` / ``"spline"``) sets the covariate basis
    for the residualizing metrics. ``partial_distance_corr`` conditions on the
    controls intrinsically and ``mutual_info`` ignores them, so both ignore it.

    ``pdcor_cache`` / ``spline_cache`` are optional engine-owned caches for the
    inputs that stay constant across many calls on one scored subset: the
    U-centered target/conditioning matrices of ``partial_distance_corr`` and
    the spline-expanded covariate basis. Both are pure accelerations — scores
    are identical (to float32 noise for the pdcor path) with or without them.
    """
    _validate_metric(metric)
    if spatial_method not in SPATIAL_METHODS:
        raise ValueError(
            f"Unknown spatial_method '{spatial_method}'. "
            f"Expected one of {sorted(SPATIAL_METHODS)}."
        )
    if residualize_method not in RESIDUALIZE_METHODS:
        raise ValueError(
            f"Unknown residualize_method '{residualize_method}'. "
            f"Expected one of {sorted(RESIDUALIZE_METHODS)}."
        )
    t = np.asarray(target, dtype=np.float64)
    c = np.asarray(cgi, dtype=np.float64)
    if t.shape != c.shape:
        raise ValueError(
            f"target and cgi must have the same shape, got {t.shape} vs {c.shape}."
        )

    cov = _coerce_covariates(covariates)
    sb = _coerce_covariates(spatial_basis)
    if metric in COVARIATE_IGNORED or spatial_method == "none":
        sb = None
    if metric in COVARIATE_IGNORED:
        # MI ignores covariates by design (documented).
        cov = None

    mask = _finite_mask(t, c, cov, sb)
    if not mask.all():
        t, c = t[mask], c[mask]
        cov = None if cov is None else cov[mask]
        sb = None if sb is None else sb[mask]

    if len(t) < 3 or float(np.var(t)) == 0 or float(np.var(c)) == 0:
        s = _DEGENERATE_SCORE[metric]
        return (s, 1.0) if return_pvalue else s

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=ConstantInputWarning)

        if metric == "partial_distance_corr":
            # Condition on [covariates, smooth] in the distance space (both are
            # partialled out nonlinearly; no linear residualization).
            z = _stack(cov, sb)
            s = partial_distance_correlation(t, c, z, cache=pdcor_cache)
            return (s, 1.0) if return_pvalue else s

        if metric in BINARY_ONLY_METRICS:
            # Logistic regression enters covariates and the spatial smooth as
            # design columns rather than residualizing on them: for a non-linear
            # link, adjusting the outcome and the exposure separately is not the
            # same model as conditioning inside one fit.
            from . import binary_longitudinal

            return binary_longitudinal.score_logit(
                metric, t, c, _stack(cov, sb), return_pvalue=return_pvalue
            )

        # Covariate basis for the residualizing metrics (spline-expanded when
        # asked). Deliberately below the pdcor branch — that metric (and
        # mutual_info, whose cov is already None here) never consumes it.
        cov_res = _expand_covariate_basis(cov, residualize_method, spline_cache)

        if metric == "distance_corr":
            x_t, x_c = _spatial_designs(cov_res, sb, spatial_method)
            tr = _residualize(t, x_t)
            cr = _residualize(c, x_c)
            if float(np.var(tr)) == 0 or float(np.var(cr)) == 0:
                return (0.0, 1.0) if return_pvalue else 0.0
            s = distance_correlation(tr, cr)
            return (s, 1.0) if return_pvalue else s

        if metric == "spearman":
            tt = rankdata(t)
            cc = rankdata(c)
            if cov is not None:
                cov_use = _expand_covariate_basis(
                    np.column_stack([rankdata(cov[:, j]) for j in range(cov.shape[1])]),
                    residualize_method,
                    spline_cache,
                )
            else:
                cov_use = None
            x_t, x_c = _spatial_designs(cov_use, sb, spatial_method)
            tr = _residualize(tt, x_t)
            cr = _residualize(cc, x_c)
            if float(np.var(tr)) == 0 or float(np.var(cr)) == 0:
                return (0.0, 1.0) if return_pvalue else 0.0
            corr, _pval = pearsonr(tr, cr)
            if np.isnan(corr):
                return (0.0, 1.0) if return_pvalue else 0.0
            s = float(abs(corr))
            return (s, 1.0) if return_pvalue else s

        if metric == "r2":
            # Incremental R² with the spatial smooth folded into the control
            # design (both methods condition the outcome on cov + smooth).
            ctrl = _stack(cov_res, sb)
            if ctrl is None:
                # Legacy path: r2 of cgi treated as a direct prediction of
                # target — preserves the previous engine behaviour exactly.
                s = float(r2_score(t, c))
            else:
                X_red = ctrl
                X_full = np.column_stack([ctrl, c])
                s = _ols_r2(t, X_full) - _ols_r2(t, X_red)
            if np.isnan(s):
                s = 0.0
            return (s, 1.0) if return_pvalue else s

        if metric == "nrmse":
            # Min-max normalized RMSE: scale-free pattern error. Residualize on
            # covariates (+ spatial smooth) first, matching distance_corr, so the
            # score reflects the greenery term's partial contribution.
            x_t, x_c = _spatial_designs(cov_res, sb, spatial_method)
            tr = _residualize(t, x_t)
            cr = _residualize(c, x_c)
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


# ────────────────────────────────────────────────────────────────────
# Penalized model comparison (CGI vs best standalone channel)
# ────────────────────────────────────────────────────────────────────


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
    spatial_basis: np.ndarray | None = None,
) -> dict:
    """AIC/BIC comparison of a full multi-channel model vs the best single channel.

    Fits two OLS models on ``target``:

    * **full** — ``target ~ all channels + covariates (+ spatial smooth)``
    * **reduced** — ``target ~ best single channel + covariates (+ spatial smooth)``

    and returns their AIC/BIC plus ``delta_aic`` / ``delta_bic`` (reduced minus
    full, so a *positive* delta means the extra channels improve the penalized
    fit enough to justify their complexity) and a ``verdict`` keyed off ΔBIC.

    The full model is naturally penalized for its extra predictors, so this
    answers "do the other channels earn their keep over the best one alone?".
    When ``spatial_basis`` is supplied it conditions both models on the same
    coordinate smooth, matching the spatially-adjusted objective.
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
    sb = _coerce_covariates(spatial_basis)
    mask = np.isfinite(t) & np.isfinite(X).all(axis=1)
    if cov is not None:
        mask &= np.isfinite(cov).all(axis=1)
    if sb is not None:
        mask &= np.isfinite(sb).all(axis=1)
    t = t[mask]
    X = X[mask]
    cov = None if cov is None else cov[mask]
    sb = None if sb is None else sb[mask]

    n_ctrl = (0 if cov is None else cov.shape[1]) + (0 if sb is None else sb.shape[1])
    if len(t) < (X.shape[1] + n_ctrl + 3):
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
        if sb is not None:
            parts.append(sb)
        return sm.add_constant(np.column_stack(parts), has_constant="add")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = sm.OLS(t, _design(X)).fit()
        reduced = sm.OLS(t, _design(X[:, [best_channel_idx]])).fit()

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
