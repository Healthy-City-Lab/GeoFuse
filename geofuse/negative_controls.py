"""Negative-control outcomes: how much of a tuned association is specific.

A tuned exposure's association with its target is pathway signal, confounding
and chance together. Held-out scoring guards against chance only. A negative
control is an outcome expected to share the target's confounders but not to be
caused by greenery (grip strength for a cognitive target, for example); applying
the frozen exposure to it shows how much of the association is non-specific
(Lipsitch, Tchetgen Tchetgen & Cohen 2010, *Epidemiology* 21:383-388).

The control never enters tuning. Everything here takes an exposure that is
already fixed and scores it, with the same covariates, against the target and
against each control:

- :func:`partial_slope` — covariate-adjusted slope with exposure and outcome in
  SD units of their residuals (Frisch-Waugh), its exact OLS interval and t;
- :func:`paired_contrast` — ``Δ = |β_target| − |β_control|`` with a percentile
  bootstrap interval, both slopes re-estimated on every resample, whole
  clusters resampled when entities are nested in areas;
- :func:`transfer_test` — both, for each control, plus the ``nonspecific``
  flag: the control's interval excludes zero and Δ's interval includes it.

Pure numpy and scipy, so the same code serves the engine and a direct caller.
"""

from __future__ import annotations

import math

import numpy as np

from .bayesian_index import covariate_basis

# Resamples per bootstrap batch: bounds the (batch x cluster) weight matrix.
_BATCH = 100


def _residual_scale(a: np.ndarray, q: np.ndarray) -> float:
    r = a - q @ (q.T @ a)
    return float(r.std())


def partial_slope(exposure, outcome, covariates=None, *, level: float = 0.95) -> dict:
    """Slope of ``outcome`` on ``exposure`` given ``covariates``, in SD units.

    Both sides are residualised on the covariates plus an intercept and divided
    by their residual SD, so ``beta`` is the outcome change in SDs per SD of
    exposure, conditional on the covariates — the partial correlation. Its
    standard error and interval are the exact OLS ones with ``n − rank − 1``
    residual degrees of freedom. Rows with any non-finite value are dropped.
    """
    from scipy import stats

    x = np.asarray(exposure, dtype=np.float64)
    y = np.asarray(outcome, dtype=np.float64)
    cov = None if covariates is None else np.asarray(covariates, dtype=np.float64)
    if cov is not None:
        cov = cov.reshape(len(x), -1)
    keep = np.isfinite(x) & np.isfinite(y)
    if cov is not None and cov.size:
        keep &= np.isfinite(cov).all(1)
    x, y = x[keep], y[keep]
    cov = None if cov is None else cov[keep]
    n = int(len(x))
    q, _ = covariate_basis(cov, n)
    dof = n - q.shape[1] - 1
    empty = {"beta": float("nan"), "se": float("nan"), "ci_low": float("nan"),
             "ci_high": float("nan"), "t": float("nan"), "n": n, "df": dof}
    if dof < 2:
        return empty
    xr = x - q @ (q.T @ x)
    yr = y - q @ (q.T @ y)
    sx, sy = float(xr.std()), float(yr.std())
    if sx <= 1e-12 or sy <= 1e-12:
        return empty
    beta = float(np.mean((xr / sx) * (yr / sy)))
    beta = min(max(beta, -1.0), 1.0)
    se = math.sqrt(max(1.0 - beta * beta, 0.0) / dof)
    crit = float(stats.t.ppf(0.5 + level / 2.0, dof))
    return {"beta": beta, "se": se, "ci_low": beta - crit * se,
            "ci_high": beta + crit * se,
            "t": beta / se if se > 0 else float("inf"), "n": n, "df": dof}


def paired_contrast(exposure, target, control, covariates=None, *,
                    n_boot: int = 1000, clusters=None, seed: int = 0,
                    level: float = 0.95) -> dict:
    """``Δ = |β_target| − |β_control|``, both re-estimated on each resample.

    Computed on the rows where exposure, target, control and covariates are all
    finite, so the two slopes describe the same entities. Each slope comes from
    a weighted least-squares fit of the outcome on ``[exposure, covariates, 1]``
    with exposure and outcomes pre-scaled by their full-sample residual SDs;
    unit weights reproduce :func:`partial_slope`. A resample is a vector of
    multiplicities — over entities, or over whole clusters when ``clusters``
    labels each row with its area — applied to per-cluster Gram blocks, so
    every resample costs one small solve per outcome (``pinv``, rank-safe)
    rather than a pass over the rows.
    """
    x = np.asarray(exposure, dtype=np.float64)
    ya = np.asarray(target, dtype=np.float64)
    yb = np.asarray(control, dtype=np.float64)
    cov = None if covariates is None else np.asarray(covariates, dtype=np.float64)
    if cov is not None:
        cov = cov.reshape(len(x), -1)
    keep = np.isfinite(x) & np.isfinite(ya) & np.isfinite(yb)
    if cov is not None and cov.size:
        keep &= np.isfinite(cov).all(1)
    cl = None if clusters is None else np.asarray(clusters)[keep]
    x, ya, yb = x[keep], ya[keep], yb[keep]
    cov = None if cov is None else cov[keep]
    n = int(len(x))
    nan = float("nan")
    out = {"delta": nan, "ci_low": nan, "ci_high": nan, "ratio": nan,
           "ratio_ci_low": nan, "ratio_ci_high": nan, "n": n,
           "n_boot": int(n_boot), "clusters": None}

    q, _ = covariate_basis(cov, n)
    if n - q.shape[1] - 1 < 2:
        return out
    scales = [_residual_scale(v, q) for v in (x, ya, yb)]
    if min(scales) <= 1e-12:
        return out
    xs, ya_s, yb_s = x / scales[0], ya / scales[1], yb / scales[2]
    Z = np.column_stack([xs, q])

    if cl is None:
        codes = np.arange(n)
    else:
        _, codes = np.unique(cl, return_inverse=True)
    n_cl = int(codes.max()) + 1
    out["clusters"] = None if cl is None else n_cl
    p = Z.shape[1]
    G = np.zeros((n_cl, p, p))
    np.add.at(G, codes, Z[:, :, None] * Z[:, None, :])
    Ha = np.zeros((n_cl, p))
    Hb = np.zeros((n_cl, p))
    np.add.at(Ha, codes, Z * ya_s[:, None])
    np.add.at(Hb, codes, Z * yb_s[:, None])

    G_flat = G.reshape(n_cl, p * p)

    def slopes(m):
        gi = np.linalg.pinv((m @ G_flat).reshape(-1, p, p))
        ba = np.einsum("bpq,bq->bp", gi, m @ Ha)[:, 0]
        bb = np.einsum("bpq,bq->bp", gi, m @ Hb)[:, 0]
        return ba, bb

    ba0, bb0 = slopes(np.ones((1, n_cl)))
    out["delta"] = float(abs(ba0[0]) - abs(bb0[0]))
    out["ratio"] = float(bb0[0] / ba0[0]) if abs(ba0[0]) > 1e-12 else nan
    if n_boot <= 0:
        return out

    rng = np.random.default_rng(seed)
    deltas, ratios = [], []
    for start in range(0, int(n_boot), _BATCH):
        b = min(_BATCH, int(n_boot) - start)
        draws = rng.integers(0, n_cl, size=(b, n_cl))
        m = np.zeros((b, n_cl))
        np.add.at(m, (np.repeat(np.arange(b), n_cl), draws.ravel()), 1.0)
        ba, bb = slopes(m)
        deltas.append(np.abs(ba) - np.abs(bb))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios.append(np.where(np.abs(ba) > 1e-12, bb / ba, np.nan))
    deltas = np.concatenate(deltas)
    ratios = np.concatenate(ratios)
    tail = (1.0 - level) / 2.0 * 100.0
    out["ci_low"], out["ci_high"] = (float(v) for v in
                                     np.percentile(deltas, [tail, 100.0 - tail]))
    if np.isfinite(ratios).any():
        out["ratio_ci_low"], out["ratio_ci_high"] = (
            float(v) for v in np.nanpercentile(ratios, [tail, 100.0 - tail]))
    return out


def concordance(target: dict, control: dict) -> dict:
    """How closely a study re-tuned on a control reproduces the target's.

    Takes two posterior summaries (``IndexPosterior.summary()``) from the same
    protocol, one tuned on the target and one on the control. Per channel:
    the total-variation distance between the posterior-mean radius profiles
    and between the aggregator blends (0 identical, 1 disjoint), the R50 of
    each with their absolute difference, the channel weight of each with their
    absolute difference, and whether the projected (radius, statistic) picks
    are identical. Plus the control's own effect and both forms. Near-identical
    configurations for target and control are the symptom of a search that
    found the confounding rather than the pathway.
    """

    def tv(a, b) -> float:
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        return 0.5 * float(np.abs(a - b).sum()) if a.shape == b.shape else float("nan")

    def at(summary, key, i):
        vals = summary.get(key)
        return vals[i] if vals is not None and i < len(vals) else None

    out = {
        "form_target": target.get("form"),
        "form_control": control.get("form"),
        "control_beta": control.get("beta_mean"),
        "control_beta_ci": [control.get("beta_ci_low"), control.get("beta_ci_high")],
        "channels": {},
    }
    c_chans = list(control.get("channels") or [])
    for i, ch in enumerate(target.get("channels") or []):
        if ch not in c_chans:
            continue
        j = c_chans.index(ch)
        row: dict = {}
        rp_t, rp_c = at(target, "radius_profile", i), at(control, "radius_profile", j)
        if rp_t is not None and rp_c is not None:
            row["radius_profile_tv"] = tv(rp_t, rp_c)
        ag_t, ag_c = at(target, "aggregator_mean", i), at(control, "aggregator_mean", j)
        if ag_t is not None and ag_c is not None:
            row["aggregator_tv"] = tv(ag_t, ag_c)
        for key, name in (("r50_mean", "r50"), ("weight_mean", "weight")):
            a, b = at(target, key, i), at(control, key, j)
            if a is not None and b is not None:
                row[f"{name}_target"], row[f"{name}_control"] = float(a), float(b)
                row[f"abs_delta_{name}"] = abs(float(a) - float(b))
        pk_t, pk_c = at(target, "projected_pick", i), at(control, "projected_pick", j)
        if pk_t is not None and pk_c is not None:
            row["same_pick"] = list(pk_t) == list(pk_c)
        out["channels"][ch] = row
    return out


def transfer_test(exposure, target, controls: dict, covariates=None, *,
                  n_boot: int = 1000, clusters=None, seed: int = 0,
                  level: float = 0.95) -> dict:
    """The frozen exposure scored against the target and every control.

    Returns ``{"target": partial_slope, "controls": {name: {...}}}``. Each
    control entry holds its own slope (on the rows where it is observed — a
    missing control costs that control's statistics and nothing else), the
    paired contrast against the target, and ``nonspecific``: the control's
    interval excludes zero while Δ's interval includes it. The flag reports;
    it does not stop anything.
    """
    out = {"target": partial_slope(exposure, target, covariates, level=level),
           "controls": {}}
    for i, (name, values) in enumerate(controls.items()):
        own = partial_slope(exposure, values, covariates, level=level)
        contrast = paired_contrast(exposure, target, values, covariates,
                                   n_boot=n_boot, clusters=clusters,
                                   seed=seed + i, level=level)
        excludes_zero = (np.isfinite(own["ci_low"])
                         and (own["ci_low"] > 0 or own["ci_high"] < 0))
        delta_has_zero = (np.isfinite(contrast["ci_low"])
                          and contrast["ci_low"] <= 0 <= contrast["ci_high"])
        out["controls"][name] = {
            **{k: own[k] for k in ("beta", "se", "ci_low", "ci_high", "t", "n")},
            "delta": contrast["delta"],
            "delta_ci_low": contrast["ci_low"],
            "delta_ci_high": contrast["ci_high"],
            "ratio": contrast["ratio"],
            "ratio_ci_low": contrast["ratio_ci_low"],
            "ratio_ci_high": contrast["ratio_ci_high"],
            "n_paired": contrast["n"],
            "nonspecific": bool(excludes_zero and delta_has_zero),
        }
    return out
