"""Sampling weights in the reported estimates.

A cohort drawn with unequal selection probabilities, or thinned by
non-response, is asked for design-weighted estimates next to the plain ones.
This module gives the weighted versions of what the fusion reports on the
held-out rows; nothing here touches the search.

Every estimate is a weighted least-squares fit over ``[1, covariates,
smooth]`` with the weights normalised to mean 1 on the rows it uses. Rows
with a missing or non-positive weight are left out of the weighted estimate
only, and counted (``n_dropped``); ``n_eff`` is Kish's effective sample size,
``(Σw)² / Σw²``.

- :func:`slope`: the weighted partial correlation (the slope in SD units) and
  the raw-unit slope per ``scale``, with a heteroscedasticity-robust (HC1)
  standard error, the design-based variance of a weighted regression with
  independent units: a weight stands for the population members a row
  represents, so its variance carries ``w²`` (duplicating a row instead would
  count each copy as an independent observation).
- :class:`Replicates`: weighted fits of several slopes and scores on the same
  bootstrap resamples, each resample's multiplicities times the weights, from
  :class:`geofuse.resampling.BlockDesign` cross-products. :func:`r2` (the
  incremental R²), :func:`r2_difference` (two exposures' R² on the same
  resamples), :func:`d` (``|β target| − |β control|``) and
  :func:`delta_d` (``D_a − D_b``) build on it.

With constant weights every point estimate is the unweighted one.
"""

from __future__ import annotations

import math

import numpy as np

from .resampling import BlockDesign, incremental_r2, weighted_sst

#: Resamples per batch of the bootstrap.
_BATCH = 256


# ────────────────────────────────────────────────────────────────────
# Weights
# ────────────────────────────────────────────────────────────────────


def usable(weights: np.ndarray) -> np.ndarray:
    """Rows whose weight is finite and positive."""
    w = np.asarray(weights, dtype=np.float64)
    return np.isfinite(w) & (w > 0)


def normalised(weights: np.ndarray) -> np.ndarray:
    """``weights`` divided by their mean (all assumed usable)."""
    w = np.asarray(weights, dtype=np.float64)
    return w / w.mean()


def kish(weights: np.ndarray) -> float:
    """Kish's effective sample size ``(Σw)² / Σw²``."""
    w = np.asarray(weights, dtype=np.float64)
    s2 = float(w @ w)
    return float(w.sum() ** 2 / s2) if s2 > 0 else 0.0


def _rows(arrays, controls, weights):
    """The rows where every array, the controls and the weight are usable,
    and how many were left out for their weight alone."""
    finite = np.ones(len(weights), dtype=bool)
    for a in arrays:
        finite &= np.isfinite(np.asarray(a, dtype=np.float64))
    for c in controls:
        if c is not None and np.size(c):
            c = np.asarray(c, dtype=np.float64).reshape(len(weights), -1)
            finite &= np.isfinite(c).all(1)
    ok = usable(weights)
    return finite & ok, int((finite & ~ok).sum())


def _sub(c, rows):
    if c is None or not np.size(c):
        return None
    return np.asarray(c, dtype=np.float64).reshape(len(rows), -1)[rows]


# ────────────────────────────────────────────────────────────────────
# One weighted slope
# ────────────────────────────────────────────────────────────────────


def slope(
    exposure,
    outcome,
    controls=None,
    weights=None,
    *,
    scale: float = 1.0,
    level: float = 0.95,
) -> dict:
    """Weighted slope of ``outcome`` on ``exposure`` given ``controls``.

    ``beta`` is the weighted partial correlation (outcome SDs per exposure SD,
    both weighted and residualised on ``[1, controls]``); ``estimate`` the
    raw-unit slope per ``scale`` exposure units. Both share the HC1 standard
    error's t on ``n − rank − 1`` degrees of freedom; their intervals scale
    with them. Also ``n``, ``n_eff`` and ``n_dropped``.
    """
    from scipy import stats

    x = np.asarray(exposure, dtype=np.float64)
    y = np.asarray(outcome, dtype=np.float64)
    w_all = np.asarray(weights, dtype=np.float64)
    rows, dropped = _rows((x, y), (controls,), w_all)
    nan = float("nan")
    out = {
        "beta": nan,
        "se": nan,
        "ci_low": nan,
        "ci_high": nan,
        "estimate": nan,
        "estimate_se": nan,
        "estimate_ci_low": nan,
        "estimate_ci_high": nan,
        "t": nan,
        "n": int(rows.sum()),
        "n_eff": 0.0,
        "n_dropped": dropped,
        "df": 0,
        "scale": float(scale),
    }
    n = int(rows.sum())
    if n < 4:
        return out
    w = normalised(w_all[rows])
    out["n_eff"] = kish(w)
    design = BlockDesign(n, _sub(controls, rows))
    dof = n - design.rank - 1
    out["df"] = dof
    if dof < 2:
        return out
    r = design.residuals(np.column_stack([x[rows], y[rows]]), w)
    xr, yr = r[:, 0], r[:, 1]
    sxx, syy = float(w @ (xr * xr)), float(w @ (yr * yr))
    if sxx <= 1e-24 * n or syy <= 1e-24 * n:
        return out
    b = float(w @ (xr * yr)) / sxx
    e = yr - b * xr
    var = float(((w * xr * e) ** 2).sum()) / (sxx * sxx) * n / dof
    se = math.sqrt(max(var, 0.0))
    crit = float(stats.t.ppf(0.5 + level / 2.0, dof))
    to_sd = math.sqrt(sxx / syy)
    out.update(
        beta=b * to_sd,
        se=se * to_sd,
        ci_low=(b - crit * se) * to_sd,
        ci_high=(b + crit * se) * to_sd,
        t=b / se if se > 0 else float("inf"),
    )
    if np.isfinite(scale) and scale > 0:
        out.update(
            estimate=b * scale,
            estimate_se=se * scale,
            estimate_ci_low=(b - crit * se) * scale,
            estimate_ci_high=(b + crit * se) * scale,
        )
    return out


# ────────────────────────────────────────────────────────────────────
# Weighted fits on shared resamples
# ────────────────────────────────────────────────────────────────────


class Replicates:
    """Weighted fits on the point weights and on shared bootstrap resamples.

    ``rows`` are the rows every fit uses (all usable); ``weights`` are
    normalised over them. Each fit is ``(responses, controls, smooth)``: the
    residual cross-products of ``responses`` over ``[1, controls, smooth]``.
    :meth:`point` gives them at the weights, :meth:`resampled` at ``n_boot``
    resamples of the rows (multiplicities times the weights), the same
    resamples for every fit.
    """

    def __init__(self, weights: np.ndarray, rows: np.ndarray, *, seed: int = 0):
        self.rows = np.asarray(rows, dtype=bool)
        self.n = int(self.rows.sum())
        self.w = normalised(np.asarray(weights, dtype=np.float64)[self.rows])
        self.seed = int(seed)
        self._designs: dict = {}

    def _design(self, key, controls, smooth) -> BlockDesign:
        if key not in self._designs:
            self._designs[key] = BlockDesign(
                self.n, _sub(controls, self.rows), _sub(smooth, self.rows)
            )
        return self._designs[key]

    def cross(self, fits: list[tuple], W: np.ndarray) -> list[np.ndarray]:
        """``M`` per fit at the weight rows ``W`` (``(B, n)``)."""
        out = []
        for responses, controls, smooth in fits:
            design = self._design((id(controls), id(smooth)), controls, smooth)
            R = np.column_stack(
                [np.asarray(v, dtype=np.float64)[self.rows] for v in responses]
            )
            out.append(design.cross(R, W))
        return out

    def point(self) -> np.ndarray:
        return self.w[None, :]

    def resampled(self, n_boot: int):
        """Weight rows of the resamples, ``_BATCH`` at a time."""
        rng = np.random.default_rng(self.seed)
        for start in range(0, int(n_boot), _BATCH):
            b = min(_BATCH, int(n_boot) - start)
            draws = rng.integers(0, self.n, size=(b, self.n))
            m = np.zeros((b, self.n))
            np.add.at(m, (np.repeat(np.arange(b), self.n), draws.ravel()), 1.0)
            yield m * self.w[None, :]


def _correlation(M: np.ndarray, x: int, y: int) -> np.ndarray:
    """Partial correlation of responses ``x`` and ``y`` per replicate."""
    den = M[:, x, x] * M[:, y, y]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = M[:, x, y] / np.sqrt(np.where(den > 0, den, 1.0))
    return np.where(den > 0, r, np.nan)


def _interval(values: np.ndarray, level: float) -> tuple[float, float]:
    v = values[np.isfinite(values)]
    if not v.size:
        return float("nan"), float("nan")
    tail = (1.0 - level) / 2.0
    return float(np.quantile(v, tail)), float(np.quantile(v, 1.0 - tail))


def _summary(point: float, boot: list[np.ndarray], level: float, rep) -> dict:
    """``estimate`` with its percentile interval and the one-sided bootstrap
    p of ``estimate > 0`` (share of resamples at or below zero)."""
    out = {
        "estimate": point,
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "p_value": float("nan"),
        "n": rep.n,
        "n_eff": kish(rep.w),
        "n_boot": 0,
    }
    values = np.concatenate(boot) if boot else np.empty(0)
    values = values[np.isfinite(values)]
    if values.size:
        out["ci_low"], out["ci_high"] = _interval(values, level)
        out["p_value"] = float((np.sum(values <= 0.0) + 1) / (values.size + 1))
        out["n_boot"] = int(values.size)
    return out


def _r2_values(rep: Replicates, fits, W) -> np.ndarray:
    """Incremental R² of each fit's exposure (response 1) per weight row."""
    cols = []
    for M, (responses, _c, _s) in zip(rep.cross(fits, W), fits, strict=True):
        y, x = (np.asarray(v, dtype=np.float64)[rep.rows] for v in responses)
        cols.append(incremental_r2(M, weighted_sst(y, W), weighted_sst(x, W)))
    return np.column_stack(cols)


def r2(
    target,
    exposure,
    controls=None,
    smooth=None,
    weights=None,
    *,
    n_boot=0,
    seed=0,
    level=0.95,
) -> dict:
    """Weighted incremental R² of ``exposure`` over ``[1, controls, smooth]``,
    with its bootstrap interval (``p_value`` is not a test of the score)."""
    rows, dropped = _rows((target, exposure), (controls, smooth), weights)
    if rows.sum() < 4:
        return {"estimate": float("nan"), "n": int(rows.sum()), "n_dropped": dropped}
    rep = Replicates(weights, rows, seed=seed)
    fits = [((target, exposure), controls, smooth)]
    point = float(_r2_values(rep, fits, rep.point())[0, 0])
    boot = [_r2_values(rep, fits, W)[:, 0] for W in rep.resampled(n_boot)]
    out = _summary(point, boot, level, rep)
    out["p_value"] = float("nan")
    out["n_dropped"] = dropped
    return out


def r2_difference(
    target,
    exposure,
    other,
    controls=None,
    smooth=None,
    other_smooth=None,
    weights=None,
    *,
    n_boot=0,
    seed=0,
    level=0.95,
) -> dict:
    """Weighted incremental R² of ``exposure`` minus that of ``other``, each
    over its own smooth, on the same rows and resamples; ``p_value`` is the
    one-sided bootstrap p of a positive difference."""
    rows, dropped = _rows(
        (target, exposure, other), (controls, smooth, other_smooth), weights
    )
    if rows.sum() < 4:
        return {"estimate": float("nan"), "n": int(rows.sum()), "n_dropped": dropped}
    rep = Replicates(weights, rows, seed=seed)
    fits = [
        ((target, exposure), controls, smooth),
        ((target, other), controls, other_smooth),
    ]

    def diff(W):
        v = _r2_values(rep, fits, W)
        return v[:, 0] - v[:, 1]

    point = float(diff(rep.point())[0])
    out = _summary(point, [diff(W) for W in rep.resampled(n_boot)], level, rep)
    out["n_dropped"] = dropped
    return out


def _d_values(rep, fits, W) -> np.ndarray:
    """``|r_target| − |r_control|`` per pair of fits and weight row."""
    Ms = rep.cross(fits, W)
    r = [_correlation(M, 0, 1) for M in Ms]
    return np.column_stack(
        [np.abs(r[i]) - np.abs(r[i + 1]) for i in range(0, len(r), 2)]
    )


def d(
    exposure,
    target,
    control,
    controls=None,
    smooth=None,
    control_smooth=None,
    weights=None,
    *,
    n_boot=0,
    seed=0,
    level=0.95,
) -> dict:
    """``D = |β target| − |β control|`` of one exposure, both weighted partial
    correlations on the same rows (the control's own smooth when given), with
    its bootstrap interval and one-sided p of ``D > 0``."""
    control_smooth = smooth if control_smooth is None else control_smooth
    rows, dropped = _rows(
        (exposure, target, control), (controls, smooth, control_smooth), weights
    )
    if rows.sum() < 4:
        return {"estimate": float("nan"), "n": int(rows.sum()), "n_dropped": dropped}
    rep = Replicates(weights, rows, seed=seed)
    fits = [
        ((exposure, target), controls, smooth),
        ((exposure, control), controls, control_smooth),
    ]
    point = float(_d_values(rep, fits, rep.point())[0, 0])
    boot = [_d_values(rep, fits, W)[:, 0] for W in rep.resampled(n_boot)]
    out = _summary(point, boot, level, rep)
    out["n_dropped"] = dropped
    return out


def delta_d(
    exposure,
    other,
    target,
    control,
    controls=None,
    smooths: "dict | None" = None,
    weights=None,
    *,
    n_boot=0,
    seed=0,
    level=0.95,
) -> dict:
    """``ΔD = D(exposure) − D(other)`` on the same rows and resamples.

    ``smooths`` maps ``(exposure role, outcome role)`` (roles ``exposure`` /
    ``other`` and ``target`` / ``control``) to that slope's smooth columns;
    missing pairs use none. One-sided bootstrap p of ``ΔD > 0``.
    """
    smooths = smooths or {}
    rows, dropped = _rows(
        (exposure, other, target, control), (controls, *smooths.values()), weights
    )
    if rows.sum() < 4:
        return {"estimate": float("nan"), "n": int(rows.sum()), "n_dropped": dropped}
    rep = Replicates(weights, rows, seed=seed)
    fits = [
        ((x, y), controls, smooths.get((xr, yr)))
        for x, xr in ((exposure, "exposure"), (other, "other"))
        for y, yr in ((target, "target"), (control, "control"))
    ]

    def contrast(W):
        v = _d_values(rep, fits, W)
        return v[:, 0] - v[:, 1]

    point = float(contrast(rep.point())[0])
    out = _summary(point, [contrast(W) for W in rep.resampled(n_boot)], level, rep)
    out["n_dropped"] = dropped
    return out
