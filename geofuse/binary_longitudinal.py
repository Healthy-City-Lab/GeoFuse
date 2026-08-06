"""Logistic association scoring for binary outcomes, cross-sectional and panel.

A dichotomised outcome — CES-D-10 screened at 10 or more, a physician
diagnosis, a loneliness item collapsed to yes/no — is what most of the
greenspace literature actually models, and a Gaussian fit to a 0/1 column is a
linear probability model whose standard errors are wrong in a known direction.
This module supplies the logistic alternative on both paths.

Cross-sectional
---------------

:func:`score_logit` fits a plain logistic regression and returns the greenery
term's Wald statistic or its coefficient (a log-odds ratio).

Longitudinal
------------

A participant contributes up to three person-waves, so the rows are not
independent. ``statsmodels.MixedLM`` is Gaussian-only and cannot take a
binomial family, so the panel path uses **GEE** with an exchangeable working
correlation clustered on the participant: a population-averaged log-odds ratio
with cluster-robust (sandwich) standard errors, which is the standard choice
for a binary repeated-measures exposure–outcome association and the one whose
estimand matches the marginal odds ratios the literature reports.

Search cost, and what makes it affordable
-----------------------------------------

An exact GEE fit iterates IRLS to convergence and re-estimates the working
correlation on every pass. Inside a stability-selection search that is hundreds
of fits per fold, and it dominates the runtime.

The same observation that makes the Gaussian fast scorer work applies here.
Across trials only the greenery column changes; the outcome, the covariates and
the cluster structure do not. So the working weights ``mu (1 - mu)``, the
working correlation, and the cluster sandwich pieces are estimated **once per
fold** from a covariates-only fit (:func:`estimate_fold_baseline`), and each
trial is then a single Newton step from that null fit
(:func:`score_gee_logit_fast`) rather than a fresh IRLS.

That one-step statistic is the **generalized score test** for the greenery
term. It is not an approximation invented here: evaluating the score and its
cluster-robust variance at the null fit is the standard efficient-score
construction, and its one-step coefficient is the first Newton–Raphson iterate
of the full estimator. Measured against exact refits over a trial pool it holds
a rank correlation of 0.9997 on |z|, a maximum |Δz| of 0.06, identical power,
and the same top-ranked trial — see
``tests/test_binary_and_exposure_response.py``.

Separately, and independently of the one-step idea, the exchangeable working
correlation is solved in **closed form**: see the derivation above
:class:`_ClusterLayout`. Every cluster's contribution to the estimating
equations collapses to four segment sums, so no per-cluster matrix is ever
formed and there is no Python-level loop over participants. That speedup
applies to the exact fit too — it changes the arithmetic, not the answer.

``search_scoring_method="exact"`` forces the full refit for anyone who wants it.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

#: Metric names for a binary outcome on the panel path. ``gee_logit_tstat`` is
#: the |z| of the greenery term; ``gee_logit_coef`` its log-odds ratio.
GEE_LOGIT_METRICS: tuple[str, ...] = ("gee_logit_tstat", "gee_logit_coef")

#: Metric names for a binary outcome on the single-wave path.
LOGIT_METRICS: tuple[str, ...] = ("logit_tstat", "logit_coef")

#: Score returned for a degenerate input, so a bad trial fails soft rather than
#: aborting the study — the same convention the Gaussian scorer uses.
_DEGENERATE: dict[str, float] = {
    "gee_logit_tstat": 0.0,
    "gee_logit_coef": 0.0,
    "logit_tstat": 0.0,
    "logit_coef": 0.0,
}

#: IRLS controls for the exact fits.
_MAX_IRLS: int = 50
_IRLS_TOL: float = 1e-8

#: Probabilities are held away from 0 and 1 so the working weight mu(1-mu)
#: cannot underflow to zero and take the whole design's conditioning with it.
_EPS: float = 1e-10


# ──────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────


def is_binary(y: np.ndarray) -> bool:
    """True when *y* holds exactly two distinct finite values."""
    v = np.asarray(y, dtype=np.float64)
    v = v[np.isfinite(v)]
    return v.size > 0 and np.unique(v).size == 2


def binary_levels(y: np.ndarray) -> tuple[float, float]:
    """``(reference level, modelled level)`` for a two-valued column."""
    v = np.asarray(y, dtype=np.float64)
    levels = np.unique(v[np.isfinite(v)])
    if levels.size != 2:
        raise ValueError(f"expected two distinct values, found {levels.size}")
    return float(levels[0]), float(levels[1])


def describe_coding(y: np.ndarray) -> dict:
    """What the logistic models are treating as the event, and how risky that is.

    The convention is **the larger of the two values is the event**. That is
    right for a 0/1 indicator and wrong for a survey column coded 1 = Yes,
    2 = No — where it silently estimates the odds of the *absence* and flips
    the sign of every odds ratio. Nothing in the data distinguishes the two
    cases, so the levels are reported with every result and ``suspicious`` is
    set whenever they are not ``{0, 1}``, which is exactly when a reader needs
    to go and check the codebook.
    """
    reference, modelled = binary_levels(y)
    return {
        "reference_level": reference,
        "modelled_level": modelled,
        "suspicious": {reference, modelled} != {0.0, 1.0},
    }


def to_binary(y: np.ndarray) -> np.ndarray:
    """Map a two-valued column to 0/1, keeping the larger value as the event.

    See :func:`describe_coding` for why the caller must confirm that the larger
    value really is the event.
    """
    v = np.asarray(y, dtype=np.float64)
    _reference, modelled = binary_levels(v)
    return (v == modelled).astype(np.float64)


def _clip_mu(mu: np.ndarray) -> np.ndarray:
    return np.clip(mu, _EPS, 1.0 - _EPS)


def _design(greenery: np.ndarray | None, covariates: np.ndarray | None, n: int):
    """``(X, greenery column index)`` with an intercept first.

    ``greenery=None`` builds the null design, whose greenery index is -1.
    """
    cols = [np.ones(n)]
    idx = -1
    if greenery is not None:
        cols.append(np.asarray(greenery, dtype=np.float64))
        idx = 1
    if covariates is not None and np.size(covariates):
        cov = np.asarray(covariates, dtype=np.float64).reshape(n, -1)
        for j in range(cov.shape[1]):
            cols.append(cov[:, j])
    return np.column_stack(cols), idx


def _column_scales(X: np.ndarray) -> np.ndarray:
    """Per-column scale factors that put a design on comparable magnitudes.

    The estimating equations are solved through normal equations, whose
    condition number goes as the square of the design's. A covariate measured
    in units a billion times another's therefore loses roughly twice as many
    digits as it should, and the solve returns confident nonsense rather than
    failing. Scaling here — and unscaling the coefficients afterwards — makes
    the result invariant to the units a covariate happens to arrive in, which
    is what a user is entitled to assume.

    Constant columns (the intercept) and all-zero columns keep a scale of 1.
    """
    scales = np.max(np.abs(X), axis=0)
    scales[~np.isfinite(scales) | (scales <= 0)] = 1.0
    return scales


def _irls(y: np.ndarray, X: np.ndarray, beta0: np.ndarray | None = None):
    """Plain logistic IRLS. Returns ``(beta, mu)`` or ``None`` if it diverges."""
    n, p = X.shape
    beta = np.zeros(p) if beta0 is None else np.array(beta0, dtype=np.float64)
    for _ in range(_MAX_IRLS):
        eta = X @ beta
        mu = _clip_mu(1.0 / (1.0 + np.exp(-eta)))
        w = mu * (1.0 - mu)
        # Working response of the Fisher-scoring step.
        z = eta + (y - mu) / w
        XtW = X.T * w
        try:
            new = np.linalg.solve(XtW @ X, XtW @ z)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(new)):
            return None
        if np.max(np.abs(new - beta)) < _IRLS_TOL:
            beta = new
            break
        beta = new
    eta = X @ beta
    return beta, _clip_mu(1.0 / (1.0 + np.exp(-eta)))


# ──────────────────────────────────────────────────────────────────────
# Vectorised exchangeable-GEE accumulation
# ──────────────────────────────────────────────────────────────────────
# The estimating equations for an exchangeable working correlation have a
# closed form that never needs a per-cluster matrix. Writing
# ``V^-1 = (1/(1-a)) [diag(1/s^2) - c (1/s)(1/s)'] `` with ``s = sqrt(mu(1-mu))``
# and ``c = a / (1 + (m-1) a)``, and using ``D = diag(var) X``, every cluster's
# contribution collapses to four segment sums:
#
#     A_c = sum_i var_i x_i x_i'      b_c = sum_i s_i x_i
#     g_c = sum_i r_i x_i             h_c = sum_i r_i / s_i
#
#     bread_c = (A_c - c b_c b_c') / (1-a)
#     score_c = (g_c - c h_c b_c)   / (1-a)
#
# ``A_c`` and ``g_c`` are only ever needed summed over clusters, so they are one
# gemm each on the whole design; ``b_c`` and ``g_c`` per cluster are (n_clusters,
# p) arrays, which are small. Nothing scales with the square of a cluster size,
# and there is no Python-level loop over participants.


class _ClusterLayout:
    """Row ordering and segment boundaries for one panel's clusters."""

    __slots__ = ("order", "starts", "sizes", "n_clusters", "n_rows")

    def __init__(self, entity_id: np.ndarray) -> None:
        _, inverse = np.unique(np.asarray(entity_id), return_inverse=True)
        self.order = np.argsort(inverse, kind="stable")
        sorted_ids = inverse[self.order]
        self.starts = np.concatenate(
            [[0], np.flatnonzero(np.diff(sorted_ids)) + 1]
        ).astype(np.int64)
        self.sizes = np.diff(
            np.concatenate([self.starts, [len(sorted_ids)]])
        ).astype(np.int64)
        self.n_clusters = len(self.starts)
        self.n_rows = len(sorted_ids)

    def segment_sum(self, values: np.ndarray) -> np.ndarray:
        """Per-cluster sums of *values*, which must already be in sorted order."""
        if self.n_rows == 0:
            return np.zeros((0,) + values.shape[1:])
        return np.add.reduceat(values, self.starts, axis=0)

    def exchangeable_c(self, alpha: float) -> np.ndarray:
        """``a / (1 + (m-1) a)`` per cluster — the rank-one correction weight."""
        if alpha == 0.0:
            return np.zeros(self.n_clusters)
        return alpha / (1.0 + (self.sizes - 1) * alpha)


def _accumulate(
    X: np.ndarray,
    var: np.ndarray,
    resid: np.ndarray,
    layout: _ClusterLayout,
    alpha: float,
):
    """``(bread, score, meat)`` for the exchangeable GEE, without a cluster loop.

    Inputs are already in ``layout.order``. See the derivation above the class:
    every quantity is a segment sum or a whole-design product.
    """
    s = np.sqrt(var)
    inv_1ma = 1.0 / (1.0 - alpha) if alpha != 1.0 else 0.0
    c = layout.exchangeable_c(alpha)

    a_total = X.T @ (X * var[:, None])            # sum_c A_c
    g_total = X.T @ resid                          # sum_c g_c
    b = layout.segment_sum(X * s[:, None])         # (n_clusters, p)
    g = layout.segment_sum(X * resid[:, None])     # (n_clusters, p)
    h = layout.segment_sum(resid / s)              # (n_clusters,)

    bread = (a_total - b.T @ (b * c[:, None])) * inv_1ma
    score = (g_total - b.T @ (c * h)) * inv_1ma
    per_cluster_score = (g - (c * h)[:, None] * b) * inv_1ma
    meat = per_cluster_score.T @ per_cluster_score
    return bread, score, meat


def _alpha_from_residuals(
    y: np.ndarray, mu: np.ndarray, layout: _ClusterLayout, p: int
) -> float:
    """Liang–Zeger moment estimate of the exchangeable correlation, vectorised.

    Each cluster's off-diagonal Pearson cross-products are
    ``(sum r)^2 - sum r^2`` over two, so no pairwise products are formed.
    """
    r = (y - mu) / np.sqrt(mu * (1.0 - mu))
    sums = layout.segment_sum(r)
    squares = layout.segment_sum(r * r)
    multi = layout.sizes >= 2
    if not np.any(multi):
        return 0.0
    total = float(np.sum((sums[multi] ** 2 - squares[multi]) / 2.0))
    pairs = float(np.sum(layout.sizes[multi] * (layout.sizes[multi] - 1) // 2))
    phi = float(r @ r) / max(layout.n_rows - p, 1)
    if pairs <= 0 or phi <= 0:
        return 0.0
    alpha = total / (pairs * phi)
    largest = int(layout.sizes.max())
    lower = -1.0 / (largest - 1) + 1e-6 if largest > 1 else -0.999
    return float(np.clip(alpha, lower, 0.99))


# ──────────────────────────────────────────────────────────────────────
# Cross-sectional logistic
# ──────────────────────────────────────────────────────────────────────


def score_logit(
    metric: str,
    outcome: np.ndarray,
    greenery: np.ndarray,
    covariates: np.ndarray | None = None,
    *,
    return_all: bool = False,
    return_pvalue: bool = False,
):
    """Logistic regression of a binary *outcome* on greenery plus covariates.

    Returns the greenery term's ``|z|`` (``logit_tstat``) or its log-odds ratio
    (``logit_coef``). With ``return_all`` both come back in a dict; with
    ``return_pvalue`` the return is ``(value, p)``.
    """
    from scipy import stats

    y = np.asarray(outcome, dtype=np.float64)
    g = np.asarray(greenery, dtype=np.float64)
    finite = np.isfinite(y) & np.isfinite(g)
    if covariates is not None and np.size(covariates):
        cov = np.asarray(covariates, dtype=np.float64).reshape(len(y), -1)
        finite &= np.all(np.isfinite(cov), axis=1)
    else:
        cov = None
    y, g = y[finite], g[finite]
    cov = cov[finite] if cov is not None else None

    if y.size < 10 or not is_binary(y) or np.ptp(g) <= 0:
        return _degenerate(metric, return_all, return_pvalue)
    y = to_binary(y)

    X, gi = _design(g, cov, len(y))
    fit = _irls(y, X)
    if fit is None:
        return _degenerate(metric, return_all, return_pvalue)
    beta, mu = fit

    w = mu * (1.0 - mu)
    try:
        cov_beta = np.linalg.pinv((X.T * w) @ X)
    except np.linalg.LinAlgError:
        return _degenerate(metric, return_all, return_pvalue)
    se = float(np.sqrt(max(cov_beta[gi, gi], 0.0)))
    coef = float(beta[gi])
    z = abs(coef / se) if se > 0 else 0.0
    p = float(2.0 * stats.norm.sf(z))

    values = {"logit_tstat": z, "logit_coef": coef}
    return _package(metric, values, p, return_all, return_pvalue)


# ──────────────────────────────────────────────────────────────────────
# Exact GEE (exchangeable working correlation, cluster-robust variance)
# ──────────────────────────────────────────────────────────────────────


def _fit_gee_logit(
    y: np.ndarray,
    X: np.ndarray,
    layout: _ClusterLayout,
    *,
    max_iter: int = 25,
    tol: float = 1e-7,
):
    """Fit a GEE logistic model. Returns ``(beta, mu, alpha, cov)`` or ``None``.

    ``y`` and ``X`` must already be in ``layout.order``. Fisher scoring on the
    estimating equations, alternating with a moment update of the working
    correlation — the standard Liang–Zeger loop, with every step vectorised
    through :func:`_accumulate`.

    The design is solved column-scaled and reported back in the caller's units,
    so a covariate's measurement unit cannot change the answer.
    """
    scales = _column_scales(X)
    Xs = X / scales

    start = _irls(y, Xs)
    if start is None:
        return None
    beta = start[0]
    alpha = 0.0
    for _ in range(max_iter):
        mu = _clip_mu(1.0 / (1.0 + np.exp(-(Xs @ beta))))
        alpha = _alpha_from_residuals(y, mu, layout, Xs.shape[1])
        bread, score, _ = _accumulate(Xs, mu * (1.0 - mu), y - mu, layout, alpha)
        try:
            step = np.linalg.solve(bread, score)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(step)):
            return None
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    mu = _clip_mu(1.0 / (1.0 + np.exp(-(Xs @ beta))))
    alpha = _alpha_from_residuals(y, mu, layout, Xs.shape[1])
    bread, _score, meat = _accumulate(Xs, mu * (1.0 - mu), y - mu, layout, alpha)
    try:
        bread_inv = np.linalg.pinv(bread)
    except np.linalg.LinAlgError:
        return None
    cov_scaled = bread_inv @ meat @ bread_inv
    # Back to the caller's units: a coefficient divides by its column's scale,
    # and a covariance entry by the product of the two columns' scales.
    return beta / scales, mu, alpha, cov_scaled / np.outer(scales, scales)


# ──────────────────────────────────────────────────────────────────────
# Fold baseline for the fast path
# ──────────────────────────────────────────────────────────────────────


class GEEFoldBaseline:
    """Everything a fold's trials reuse: the covariates-only GEE fit.

    Built once per fold. Across trials only the greenery column changes, so the
    fitted probabilities, the working weights, the exchangeable correlation and
    the cluster layout are all properties of the null model and are shared.

    ``order`` is the row permutation that groups the panel by participant; the
    stored arrays are already permuted, and a trial's greenery vector is
    permuted on arrival.
    """

    __slots__ = ("y", "layout", "mu", "var", "resid", "alpha", "beta_null", "X_null")

    def __init__(self, y, layout, mu, alpha, beta_null, X_null):
        self.y = y
        self.layout = layout
        self.mu = mu
        self.var = mu * (1.0 - mu)
        self.resid = y - mu
        self.alpha = alpha
        self.beta_null = beta_null
        self.X_null = X_null


def estimate_fold_baseline(
    outcome: np.ndarray,
    entity_id: np.ndarray,
    covariates: np.ndarray | None = None,
) -> GEEFoldBaseline | None:
    """Fit the covariates-only GEE once, for :func:`score_gee_logit_fast`.

    Returns ``None`` when the outcome is not binary or the null model will not
    fit; the caller falls back to the exact path.
    """
    y = np.asarray(outcome, dtype=np.float64)
    if y.size < 10 or not is_binary(y):
        return None
    y = to_binary(y)
    cov = (
        np.asarray(covariates, dtype=np.float64).reshape(len(y), -1)
        if covariates is not None and np.size(covariates)
        else None
    )
    layout = _ClusterLayout(entity_id)
    X_null, _ = _design(None, cov, len(y))
    y_s = y[layout.order]
    X_s = X_null[layout.order]
    fit = _fit_gee_logit(y_s, X_s, layout)
    if fit is None:
        return None
    beta, mu, alpha, _ = fit
    return GEEFoldBaseline(y_s, layout, mu, alpha, beta, X_s)


def score_gee_logit_fast(
    metric: str,
    greenery: np.ndarray,
    baseline: GEEFoldBaseline,
    *,
    return_all: bool = False,
    return_pvalue: bool = False,
):
    """One Newton step from the null fit — the generalized score test.

    The greenery column joins the null design and a single Fisher scoring step
    is taken from the null fit's coefficients, holding the working weights and
    the exchangeable correlation at their null values. The step's greenery
    component is the one-step coefficient; its cluster-robust standard error is
    the sandwich evaluated at the same point.

    Cost is one :func:`_accumulate` pass plus a ``(p+1)`` solve, against a full
    Fisher-scoring-plus-correlation loop for the exact fit.
    """
    from scipy import stats

    if baseline is None:
        return _degenerate(metric, return_all, return_pvalue)
    g = np.asarray(greenery, dtype=np.float64)
    if g.size != baseline.y.size or not np.all(np.isfinite(g)) or np.ptp(g) <= 0:
        return _degenerate(metric, return_all, return_pvalue)
    g = g[baseline.layout.order]

    X = np.column_stack([baseline.X_null[:, [0]], g, baseline.X_null[:, 1:]])
    gi = 1
    # The null coefficients padded with a zero for the term being introduced,
    # which is exactly the hypothesis the score is evaluated under.
    beta0 = np.insert(baseline.beta_null, gi, 0.0)

    # Solved column-scaled, like the exact fit, so a covariate's units cannot
    # change the score. ``beta0`` is in the caller's units, so it scales up on
    # the way in and the step scales back down on the way out.
    scales = _column_scales(X)
    bread, score, meat = _accumulate(
        X / scales, baseline.var, baseline.resid, baseline.layout, baseline.alpha
    )
    try:
        bread_inv = np.linalg.pinv(bread)
    except np.linalg.LinAlgError:
        return _degenerate(metric, return_all, return_pvalue)
    coef = float(beta0[gi] + (bread_inv @ score)[gi] / scales[gi])
    cov_beta = (bread_inv @ meat @ bread_inv) / np.outer(scales, scales)
    se = float(np.sqrt(max(cov_beta[gi, gi], 0.0)))
    z = abs(coef / se) if se > 0 else 0.0
    pval = float(2.0 * stats.norm.sf(z))

    values = {"gee_logit_tstat": z, "gee_logit_coef": coef}
    return _package(metric, values, pval, return_all, return_pvalue)


# ──────────────────────────────────────────────────────────────────────
# Exact panel scorer
# ──────────────────────────────────────────────────────────────────────


def score_gee_logit(
    metric: str,
    outcome: np.ndarray,
    greenery: np.ndarray,
    entity_id: np.ndarray,
    covariates: np.ndarray | None = None,
    *,
    return_all: bool = False,
    return_pvalue: bool = False,
):
    """Full GEE logistic fit with the greenery term in the design.

    The reference implementation the fast path is checked against, and what
    ``search_scoring_method="exact"`` selects.
    """
    from scipy import stats

    y = np.asarray(outcome, dtype=np.float64)
    g = np.asarray(greenery, dtype=np.float64)
    finite = np.isfinite(y) & np.isfinite(g)
    cov = None
    if covariates is not None and np.size(covariates):
        cov = np.asarray(covariates, dtype=np.float64).reshape(len(y), -1)
        finite &= np.all(np.isfinite(cov), axis=1)
    y, g = y[finite], g[finite]
    cov = cov[finite] if cov is not None else None
    eids = np.asarray(entity_id)[finite]

    if y.size < 10 or not is_binary(y) or np.ptp(g) <= 0:
        return _degenerate(metric, return_all, return_pvalue)
    y = to_binary(y)

    X, gi = _design(g, cov, len(y))
    layout = _ClusterLayout(eids)
    fit = _fit_gee_logit(y[layout.order], X[layout.order], layout)
    if fit is None:
        return _degenerate(metric, return_all, return_pvalue)
    beta, _mu, _alpha, cov_beta = fit

    coef = float(beta[gi])
    se = float(np.sqrt(max(cov_beta[gi, gi], 0.0)))
    z = abs(coef / se) if se > 0 else 0.0
    pval = float(2.0 * stats.norm.sf(z))

    values = {"gee_logit_tstat": z, "gee_logit_coef": coef}
    return _package(metric, values, pval, return_all, return_pvalue)


# ──────────────────────────────────────────────────────────────────────
# Reporting fitters
# ──────────────────────────────────────────────────────────────────────


def make_logit_fitter(outcome: np.ndarray):
    """A :data:`geofuse.exposure_response.Fitter` backed by logistic regression."""
    from scipy import stats

    y = to_binary(outcome) if is_binary(outcome) else None

    def _fit(design: np.ndarray, names):
        if y is None:
            return None
        X = np.asarray(design, dtype=np.float64)
        fit = _irls(y, X)
        if fit is None:
            return None
        beta, mu = fit
        w = mu * (1.0 - mu)
        try:
            cov_beta = np.linalg.pinv((X.T * w) @ X)
        except np.linalg.LinAlgError:
            return None
        return _shape_fit(beta, cov_beta, names)

    return _fit


def make_gee_logit_fitter(outcome: np.ndarray, entity_id: np.ndarray):
    """A :data:`geofuse.exposure_response.Fitter` backed by GEE logistic.

    Used for the quartile and spline reports on a binary panel outcome, so
    those tables carry the same cluster-robust standard errors as the headline
    estimate rather than pretending the person-waves are independent.
    """
    from scipy import stats

    y = to_binary(outcome) if is_binary(outcome) else None
    layout = _ClusterLayout(entity_id)
    y_s = y[layout.order] if y is not None else None

    def _fit(design: np.ndarray, names):
        if y_s is None:
            return None
        X = np.asarray(design, dtype=np.float64)[layout.order]
        fit = _fit_gee_logit(y_s, X, layout)
        if fit is None:
            return None
        beta, _mu, _alpha, cov_beta = fit
        return _shape_fit(beta, cov_beta, names)

    return _fit


def _shape_fit(beta: np.ndarray, cov_beta: np.ndarray, names) -> dict:
    """Fitter return shape: per-term triples plus the full covariance.

    The covariance travels under
    :data:`geofuse.exposure_response.COV_KEY` so the spline block can be tested
    jointly rather than term by term.
    """
    from scipy import stats

    from .exposure_response import COV_KEY

    se = np.sqrt(np.clip(np.diag(cov_beta), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, beta / se, 0.0)
    p = 2.0 * stats.norm.sf(np.abs(z))
    out: dict = {
        nm: (float(beta[i]), float(se[i]), float(p[i]))
        for i, nm in enumerate(names)
    }
    out[COV_KEY] = (list(names), np.asarray(cov_beta))
    return out


# ──────────────────────────────────────────────────────────────────────
# Return shaping
# ──────────────────────────────────────────────────────────────────────


def _degenerate(metric: str, return_all: bool, return_pvalue: bool):
    if return_all:
        return dict(_DEGENERATE)
    value = _DEGENERATE.get(metric, 0.0)
    return (value, float("nan")) if return_pvalue else value


def _package(metric: str, values: dict, pval: float, return_all: bool, return_pvalue: bool):
    if return_all:
        return values
    value = values.get(metric, 0.0)
    return (value, pval) if return_pvalue else value
