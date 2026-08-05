"""Exposure–response reporting: IQR scaling, quartile contrasts, spline tests.

The search reports a greenery coefficient in the composite's own raw units,
which is the right thing for optimisation and the wrong thing for a paper. The
environmental-epidemiology literature reports three other shapes, and this
module produces all three from an already-chosen composite:

* **Per-IQR effect** — the coefficient rescaled to one interquartile-range
  increase in the exposure. This is how every greenspace effect estimate is
  published (the CLSA papers use an NDVI IQR of 0.06). It is a pure change of
  units, so it does not require a refit.
* **Quartile contrasts** — the exposure cut at its own quartiles and entered
  as three indicators against Q1. This is the form used when the authors want
  to show the shape of the gradient without assuming one, and it is what the
  CLSA self-rated-social-standing paper models.
* **Non-linearity test** — a restricted cubic spline on the exposure against
  the linear term, so "is the exposure–response function actually a line" gets
  an answer instead of an assumption.

All three take a *fitter* callable rather than a model, so one implementation
serves the cross-sectional OLS path and the longitudinal mixed-effects and GEE
paths. A fitter maps ``(design, column_names) -> {name: (coef, se, pvalue)}``
and returns ``None`` when the model will not fit; :func:`ols_fitter` is the
cross-sectional default.

Nothing here participates in the search. These are reporting-stage functions
run once on the winning composite, so a refit or two costs nothing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: A fitter takes the design matrix and its column names and returns
#: ``{name: (coef, std_err, pvalue)}``, or ``None`` if the fit failed. The
#: design already carries its own intercept column. Fitters additionally place
#: the full coefficient covariance under :data:`COV_KEY` as
#: ``(ordered_names, matrix)``, which the multi-term Wald test in
#: :func:`spline_nonlinearity_test` needs — testing a block of coefficients
#: from their standard errors alone would assume they are uncorrelated, and
#: spline coefficients never are.
Fitter = Callable[[np.ndarray, Sequence[str]], "dict[str, tuple] | None"]

#: Key under which a fitter returns ``(names, covariance_matrix)``.
COV_KEY = "__cov__"

#: Interquartile range of NDVI in the CLSA cohort, the scaling constant the
#: published CLSA greenspace effect estimates are expressed in. Supplied as a
#: default only so a replication can reproduce their numbers exactly; a
#: composite index has its own IQR and should be scaled by that instead.
CLSA_NDVI_IQR: float = 0.06

#: Spline basis size for the non-linearity test. Four knots is the usual
#: choice for an exposure–response curve — enough to bend twice, not enough to
#: chase noise.
SPLINE_DF: int = 4


# ──────────────────────────────────────────────────────────────────────
# IQR scaling
# ──────────────────────────────────────────────────────────────────────


def exposure_iqr(exposure: np.ndarray) -> float:
    """Interquartile range of the finite values of *exposure*."""
    values = np.asarray(exposure, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 4:
        return float("nan")
    return float(np.percentile(values, 75) - np.percentile(values, 25))


def iqr_scaled_effect(
    coef: float,
    std_err: float,
    exposure: np.ndarray | None = None,
    *,
    iqr: float | None = None,
    logistic: bool = False,
) -> dict:
    """Rescale a per-unit effect to a per-IQR increase.

    Pass *exposure* to use its own interquartile range (the right choice for a
    composite index), or *iqr* to pin the divisor — ``CLSA_NDVI_IQR`` when the
    point is to reproduce a published NDVI estimate.

    With ``logistic=True`` the estimate and its interval are also exponentiated
    into an odds ratio, which is the form the CLSA depression and loneliness
    papers report.

    Scaling is linear in the coefficient and its standard error, so the
    z-statistic and p-value are unchanged by construction — only the units move.
    """
    scale = float(iqr) if iqr is not None else exposure_iqr(exposure)
    if not np.isfinite(scale) or scale <= 0:
        return {"iqr": float("nan"), "estimate": float("nan")}

    est = float(coef) * scale
    se = float(std_err) * scale
    lo, hi = est - 1.959964 * se, est + 1.959964 * se
    out = {
        "iqr": scale,
        "estimate": est,
        "std_error": se,
        "ci_low": lo,
        "ci_high": hi,
        "scale_basis": "supplied" if iqr is not None else "exposure",
    }
    if logistic:
        out.update(
            odds_ratio=float(np.exp(est)),
            or_ci_low=float(np.exp(lo)),
            or_ci_high=float(np.exp(hi)),
        )
    return out


# ──────────────────────────────────────────────────────────────────────
# Default cross-sectional fitter
# ──────────────────────────────────────────────────────────────────────


def make_ols_fitter(outcome: np.ndarray) -> Fitter:
    """An OLS :data:`Fitter` bound to *outcome*."""
    from scipy import stats

    y = np.asarray(outcome, dtype=np.float64)

    def _fit(design: np.ndarray, names: Sequence[str]) -> dict | None:
        X = np.asarray(design, dtype=np.float64)
        n, p = X.shape
        if n <= p:
            return None
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = n - p
        sigma2 = float(resid @ resid) / dof
        try:
            xtx_inv = np.linalg.pinv(X.T @ X)
        except np.linalg.LinAlgError:
            return None
        cov_beta = xtx_inv * sigma2
        se = np.sqrt(np.clip(np.diag(cov_beta), 0.0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            tvals = np.where(se > 0, beta / se, 0.0)
        pvals = 2.0 * stats.t.sf(np.abs(tvals), dof)
        out: dict = {
            nm: (float(beta[i]), float(se[i]), float(pvals[i]))
            for i, nm in enumerate(names)
        }
        out[COV_KEY] = (list(names), cov_beta)
        return out

    return _fit


# ──────────────────────────────────────────────────────────────────────
# Quartile contrasts
# ──────────────────────────────────────────────────────────────────────


def exposure_quartiles(exposure: np.ndarray, n_groups: int = 4):
    """``(group index per row, cut points)`` for *exposure*.

    Ties are common in a composite built from binned weights, so the cuts come
    from :func:`numpy.percentile` and group membership from ``searchsorted``;
    a quantile that collapses onto its neighbour yields a smaller number of
    non-empty groups rather than an error.
    """
    values = np.asarray(exposure, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size < n_groups * 2:
        return None, np.array([])
    qs = np.linspace(0, 100, n_groups + 1)[1:-1]
    cuts = np.unique(np.percentile(finite, qs))
    groups = np.searchsorted(cuts, values, side="right")
    groups[~np.isfinite(values)] = -1
    return groups, cuts


def quartile_terms(
    exposure: np.ndarray,
    fitter: Fitter,
    covariates: np.ndarray | None = None,
    *,
    n_groups: int = 4,
    logistic: bool = False,
) -> dict | None:
    """Exposure as quantile indicators against the lowest group.

    Returns one contrast per group above the reference, each with its
    coefficient, standard error, p-value, and (for a logistic fit) odds ratio.
    ``trend_p`` comes from re-entering the group index as a single ordered
    numeric term, which is the usual test-for-trend across the categories.

    This does not assume the gradient is linear, so a monotone set of contrasts
    is evidence about the shape rather than a restatement of the model.
    """
    groups, cuts = exposure_quartiles(exposure, n_groups)
    if groups is None:
        return None
    valid = groups >= 0
    present = sorted(int(g) for g in np.unique(groups[valid]))
    if len(present) < 2:
        return None
    reference = present[0]

    n = len(groups)
    columns: list[np.ndarray] = [np.ones(n)]
    names: list[str] = ["intercept"]
    for g in present[1:]:
        columns.append((groups == g).astype(np.float64))
        names.append(f"q{g + 1}")
    if covariates is not None and np.size(covariates):
        cov = np.asarray(covariates, dtype=np.float64)
        cov = cov.reshape(len(groups), -1)
        for j in range(cov.shape[1]):
            columns.append(cov[:, j])
            names.append(f"cov{j}")

    fitted = fitter(np.column_stack(columns), names)
    if fitted is None:
        return None

    contrasts = []
    for g in present[1:]:
        coef, se, pval = fitted[f"q{g + 1}"]
        row = {
            "group": g + 1,
            "n": int((groups == g).sum()),
            "coef": coef,
            "std_error": se,
            "p_value": pval,
        }
        if logistic:
            row["odds_ratio"] = float(np.exp(coef))
            row["or_ci_low"] = float(np.exp(coef - 1.959964 * se))
            row["or_ci_high"] = float(np.exp(coef + 1.959964 * se))
        contrasts.append(row)

    # Test for trend: the ordered group index as one numeric term.
    trend_cols = [np.ones(n), groups.astype(np.float64)]
    trend_names = ["intercept", "trend"]
    if covariates is not None and np.size(covariates):
        cov = np.asarray(covariates, dtype=np.float64).reshape(n, -1)
        for j in range(cov.shape[1]):
            trend_cols.append(cov[:, j])
            trend_names.append(f"cov{j}")
    trend_fit = fitter(np.column_stack(trend_cols), trend_names)
    trend_p = trend_fit["trend"][2] if trend_fit else float("nan")

    return {
        "n_groups": len(present),
        "reference_group": reference + 1,
        "cut_points": [float(c) for c in cuts],
        "reference_n": int((groups == reference).sum()),
        "contrasts": contrasts,
        "trend_p": float(trend_p),
    }


# ──────────────────────────────────────────────────────────────────────
# Non-linearity test
# ──────────────────────────────────────────────────────────────────────


def _natural_cubic_basis(x: np.ndarray, df: int = SPLINE_DF) -> np.ndarray | None:
    """Natural cubic spline basis for *x*, or ``None`` if it cannot be built."""
    try:
        from patsy import dmatrix
    except Exception:
        logger.warning("patsy is unavailable; the spline non-linearity test is skipped.")
        return None
    try:
        basis = np.asarray(
            dmatrix("cr(v, df=d) - 1", {"v": x, "d": df}, return_type="dataframe"),
            dtype=np.float64,
        )
    except Exception as exc:
        logger.warning(f"spline basis failed: {exc}")
        return None
    return basis if basis.ndim == 2 and basis.shape[1] >= 2 else None


# ──────────────────────────────────────────────────────────────────────
# Effect modification
# ──────────────────────────────────────────────────────────────────────


def _simple_slope(
    fitted: dict, names: Sequence[str], main: str, products: Sequence[tuple[str, float]]
):
    """Greenery slope at one moderator level, with its standard error.

    The slope is ``b_main + sum(w * b_product)``; its variance needs the full
    coefficient covariance, not just the standard errors, because the main and
    product terms are correlated by construction. Returns ``None`` when the
    fitter did not supply a covariance matrix.
    """
    cov_entry = fitted.get(COV_KEY)
    if cov_entry is None:
        return None
    order, cov_matrix = cov_entry
    cov_matrix = np.asarray(cov_matrix, dtype=np.float64)

    contrast = np.zeros(len(order))
    contrast[order.index(main)] = 1.0
    for term, weight in products:
        contrast[order.index(term)] += weight

    beta = np.array([fitted[nm][0] for nm in order], dtype=np.float64)
    estimate = float(contrast @ beta)
    variance = float(contrast @ cov_matrix @ contrast)
    return estimate, float(np.sqrt(max(variance, 0.0)))


def moderation_terms(
    exposure: np.ndarray,
    moderator: np.ndarray,
    fitter: Fitter,
    covariates: np.ndarray | None = None,
    *,
    categorical: bool = False,
    logistic: bool = False,
    moderator_name: str = "moderator",
) -> dict | None:
    """Test whether the greenery effect differs across levels of *moderator*.

    Fits ``outcome ~ greenery + moderator + greenery x moderator + covariates``
    and reports three things, which is what the greenspace papers report when
    they claim moderation:

    * the **interaction** coefficients and a joint Wald test over them — the
      formal "does the effect differ" question;
    * **simple slopes**, the greenery effect evaluated at each level of a
      categorical moderator, or at mean-SD / mean / mean+SD of a continuous
      one, each with a standard error that accounts for the correlation
      between the main and product terms;
    * the **main effect**, which after centring is the greenery slope at the
      average moderator value rather than at a meaningless zero.

    A continuous moderator is mean-centred so the greenery main effect stays
    interpretable. A categorical one is expanded drop-first, so the main effect
    is the slope in the reference level.

    This is a reporting-stage analysis run on the winning composite; the search
    still optimises the greenery association, exactly as the source papers fit
    one pre-specified model rather than tuning against the interaction.
    """
    from scipy import stats

    g = np.asarray(exposure, dtype=np.float64)
    m_raw = np.asarray(moderator, dtype=np.float64)
    n = len(g)
    finite = np.isfinite(g) & np.isfinite(m_raw)
    if finite.sum() < 20:
        return None

    columns: list[np.ndarray] = [np.ones(n), g]
    names: list[str] = ["intercept", "greenery"]
    products: list[tuple[str, float]] = []
    levels: list[float] = []

    if categorical:
        levels = sorted(np.unique(m_raw[np.isfinite(m_raw)]).tolist())
        if len(levels) < 2 or len(levels) > 12:
            return None
        reference = levels[0]
        for i, lv in enumerate(levels[1:], start=1):
            indicator = (m_raw == lv).astype(np.float64)
            columns.append(indicator)
            names.append(f"mod_{i}")
            columns.append(indicator * g)
            names.append(f"greenery_x_mod_{i}")
        centre = float(reference)
    else:
        centre = float(np.nanmean(m_raw[np.isfinite(m_raw)]))
        centred = m_raw - centre
        columns.append(centred)
        names.append("mod")
        columns.append(centred * g)
        names.append("greenery_x_mod")

    if covariates is not None and np.size(covariates):
        cov = np.asarray(covariates, dtype=np.float64).reshape(n, -1)
        for j in range(cov.shape[1]):
            columns.append(cov[:, j])
            names.append(f"cov{j}")

    design = np.column_stack(columns)
    mask = np.isfinite(design).all(axis=1)
    if mask.sum() < 20:
        return None
    fitted = fitter(design, names)
    if fitted is None:
        return None

    interaction_names = [nm for nm in names if nm.startswith("greenery_x_mod")]
    beta_int = np.array([fitted[nm][0] for nm in interaction_names])

    # Joint Wald test across every interaction term — the single "is there
    # moderation" number. For a two-level moderator this is the square of the
    # one interaction's z.
    stat, dof = float("nan"), len(interaction_names)
    cov_entry = fitted.get(COV_KEY)
    if cov_entry is not None:
        order, cov_matrix = cov_entry
        idx = [order.index(nm) for nm in interaction_names]
        sub = np.asarray(cov_matrix)[np.ix_(idx, idx)]
        try:
            stat = float(beta_int @ np.linalg.solve(sub, beta_int))
        except np.linalg.LinAlgError:
            stat = float(beta_int @ np.linalg.pinv(sub) @ beta_int)
    interaction_p = (
        float(stats.chi2.sf(stat, dof)) if np.isfinite(stat) else float("nan")
    )

    # ── Simple slopes ────────────────────────────────────────────
    slopes: list[dict] = []

    def _record(label: str, value, products_at: list[tuple[str, float]], count):
        got = _simple_slope(fitted, names, "greenery", products_at)
        if got is None:
            return
        est, se = got
        z = est / se if se > 0 else 0.0
        row = {
            "level": label,
            "moderator_value": value,
            "n": count,
            "slope": est,
            "std_error": se,
            "z": z,
            "p_value": float(2.0 * stats.norm.sf(abs(z))),
            "ci_low": est - 1.959964 * se,
            "ci_high": est + 1.959964 * se,
        }
        if logistic:
            row["odds_ratio"] = float(np.exp(est))
            row["or_ci_low"] = float(np.exp(row["ci_low"]))
            row["or_ci_high"] = float(np.exp(row["ci_high"]))
        slopes.append(row)

    if categorical:
        _record(
            f"{moderator_name}={levels[0]:g} (reference)",
            float(levels[0]),
            [],
            int((m_raw == levels[0]).sum()),
        )
        for i, lv in enumerate(levels[1:], start=1):
            _record(
                f"{moderator_name}={lv:g}",
                float(lv),
                [(f"greenery_x_mod_{i}", 1.0)],
                int((m_raw == lv).sum()),
            )
    else:
        sd = float(np.nanstd(m_raw[np.isfinite(m_raw)]))
        for label, offset in (("mean - 1 SD", -sd), ("mean", 0.0), ("mean + 1 SD", sd)):
            _record(
                f"{moderator_name} {label}",
                centre + offset,
                [("greenery_x_mod", offset)],
                int(finite.sum()),
            )

    return {
        "moderator": moderator_name,
        "categorical": bool(categorical),
        "centred_at": centre,
        "n": int(mask.sum()),
        "interaction_wald_chi2": stat,
        "interaction_df": dof,
        "interaction_p": interaction_p,
        "interaction_terms": [
            {
                "term": nm,
                "coef": fitted[nm][0],
                "std_error": fitted[nm][1],
                "p_value": fitted[nm][2],
            }
            for nm in interaction_names
        ],
        "simple_slopes": slopes,
    }


def _orthogonalise(block: np.ndarray, against: np.ndarray, tol: float = 1e-8):
    """Columns of *block* with the span of *against* projected out.

    Returns the residual columns that still carry independent variation, and
    the indices kept. Used to split a spline basis into "the straight line" and
    "everything the straight line cannot express".
    """
    q, _ = np.linalg.qr(against)
    residual = block - q @ (q.T @ block)
    scale = np.linalg.norm(residual, axis=0)
    reference = max(float(np.max(np.linalg.norm(block, axis=0))), 1e-30)
    keep = np.flatnonzero(scale > tol * reference)
    return residual[:, keep], keep


def spline_nonlinearity_test(
    exposure: np.ndarray,
    fitter: Fitter,
    covariates: np.ndarray | None = None,
    *,
    df: int = SPLINE_DF,
    n_curve_points: int = 50,
) -> dict | None:
    """Restricted cubic spline on the exposure, tested against the straight line.

    The exposure enters twice: as a plain linear term, and as the natural cubic
    spline basis with the span of ``[1, x]`` projected out. That split is the
    point — the linear term absorbs the straight-line effect, so the remaining
    block is exactly what a straight line cannot express, and a joint Wald test
    on it is the test for departure from linearity. (A raw spline basis will not
    do: none of its columns is the linear term, so no subset of them isolates
    the non-linear part.)

    A small ``nonlinearity_p`` means the linear effect the search optimised is
    the wrong functional form. Also returns the fitted curve over the exposure
    range, centred at the median, ready to plot.
    """
    from scipy import stats

    x = np.asarray(exposure, dtype=np.float64)
    basis = _natural_cubic_basis(x, df)
    if basis is None:
        return None

    n = len(x)
    linear_part = np.column_stack([np.ones(n), x])
    nonlinear_block, kept = _orthogonalise(basis, linear_part)
    if nonlinear_block.shape[1] == 0:
        return None

    columns: list[np.ndarray] = [np.ones(n), x]
    names: list[str] = ["intercept", "linear"]
    for j in range(nonlinear_block.shape[1]):
        columns.append(nonlinear_block[:, j])
        names.append(f"nonlinear{j}")
    if covariates is not None and np.size(covariates):
        cov = np.asarray(covariates, dtype=np.float64).reshape(n, -1)
        for j in range(cov.shape[1]):
            columns.append(cov[:, j])
            names.append(f"cov{j}")

    fitted = fitter(np.column_stack(columns), names)
    if fitted is None:
        return None

    nonlinear_names = [nm for nm in names if nm.startswith("nonlinear")]
    beta_nl = np.array([fitted[nm][0] for nm in nonlinear_names])

    # Joint Wald test on the non-linear block, using the full covariance
    # sub-matrix. Spline coefficients are correlated even after
    # orthogonalising the columns, so summing squared z-statistics would be the
    # wrong chi-square.
    stat, dof = float("nan"), len(nonlinear_names)
    cov_entry = fitted.get(COV_KEY)
    if cov_entry is not None:
        order, cov_matrix = cov_entry
        idx = [order.index(nm) for nm in nonlinear_names]
        sub = np.asarray(cov_matrix)[np.ix_(idx, idx)]
        try:
            stat = float(beta_nl @ np.linalg.solve(sub, beta_nl))
        except np.linalg.LinAlgError:
            stat = float(beta_nl @ np.linalg.pinv(sub) @ beta_nl)
    else:
        # No covariance available: fall back to the independence approximation
        # and say so, rather than presenting it as an exact test.
        zs = [
            fitted[nm][0] / fitted[nm][1] if fitted[nm][1] > 0 else 0.0
            for nm in nonlinear_names
        ]
        stat = float(np.sum(np.square(zs)))
    p_nonlinear = float(stats.chi2.sf(stat, dof)) if np.isfinite(stat) else float("nan")

    # Fitted curve: the linear term plus the non-linear block, evaluated on a
    # grid. The grid rows are appended to the data before building the basis so
    # the knot placement is identical, then orthogonalised the same way.
    curve = None
    grid = np.linspace(np.nanpercentile(x, 1), np.nanpercentile(x, 99), n_curve_points)
    joint = _natural_cubic_basis(np.concatenate([x, grid]), df)
    if joint is not None and joint.shape[1] == basis.shape[1]:
        joint_linear = np.column_stack(
            [np.ones(len(joint)), np.concatenate([x, grid])]
        )
        q, _ = np.linalg.qr(joint_linear)
        joint_nl = (joint - q @ (q.T @ joint))[:, kept]
        yhat = fitted["linear"][0] * grid + joint_nl[n:] @ beta_nl
        centre = float(np.interp(float(np.nanmedian(x)), grid, yhat))
        curve = {
            "exposure": [float(v) for v in grid],
            "effect": [float(v - centre) for v in yhat],
        }

    return {
        "df": df,
        "linear_coef": fitted["linear"][0],
        "linear_p": fitted["linear"][2],
        "wald_chi2": stat,
        "wald_df": dof,
        "nonlinearity_p": p_nonlinear,
        "exact_wald": cov_entry is not None,
        "per_term": [
            {"term": nm, "coef": fitted[nm][0], "std_error": fitted[nm][1],
             "p_value": fitted[nm][2]}
            for nm in nonlinear_names
        ],
        "curve": curve,
    }
