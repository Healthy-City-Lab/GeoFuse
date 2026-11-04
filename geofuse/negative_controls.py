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
  flag: the control's interval excludes zero and Δ's interval includes it;
- :func:`specificity_contrast` — whether one exposure is more specific than
  another on the same rows: ``ΔD = D_a − D_b`` with ``D = |β_target| −
  |β_control|``, all four slopes re-estimated on every resample.

The paired statistics take ``extra``: control columns that belong to one
variable and join the covariates of every slope involving it, such as a
spatial smooth selected on the exposure (Spatial+) or on each outcome
(KS-AIC). Their rows are resampled with the observations. With ``extra``
columns the resamples are refitted by :mod:`geofuse.resampling`, which keeps a
block-diagonal smooth's clusters apart instead of forming one dense Gram matrix
per row; without them, by per-cluster Gram blocks of the covariate basis.

Pure numpy and scipy, so the same code serves the engine and a direct caller.
"""

from __future__ import annotations

import math

import numpy as np

from .bayesian_index import covariate_basis

# Resamples per bootstrap batch: bounds the (batch x cluster) weight matrix.
_BATCH = 100

# Rank rule of the bases the slopes are fitted on (:func:`covariate_basis`).
_BASIS_RTOL = 1e-10


def _residual_scale(a: np.ndarray, q: np.ndarray) -> float:
    r = a - q @ (q.T @ a)
    return float(r.std())


def _extra_blocks(extra: dict | None, n: int, roles: tuple[str, ...]) -> dict:
    """The ``extra`` control columns as ``{role: (n, k) array}``.

    ``None`` entries and empty blocks are dropped; a key outside ``roles``
    raises, so a misspelt role cannot leave a slope unadjusted.
    """
    out = {}
    for role, cols in (extra or {}).items():
        if role not in roles:
            raise ValueError(f"extra: unknown role {role!r}; expected one of {roles}.")
        if cols is None:
            continue
        block = np.asarray(cols, dtype=np.float64).reshape(n, -1)
        if block.shape[1]:
            out[role] = block
    return out


def _picked(blocks: dict, *roles: str) -> list[np.ndarray]:
    return [blocks[r] for r in roles if r in blocks]


def _with_blocks(cov: np.ndarray | None, blocks: list[np.ndarray]):
    """``cov`` with ``blocks`` appended as columns (``cov`` itself without)."""
    if not blocks:
        return cov
    parts = ([] if cov is None or not cov.size else [cov]) + blocks
    return parts[0] if len(parts) == 1 else np.column_stack(parts)


def _resample_weights(rng, b: int, n_cl: int, codes: np.ndarray) -> np.ndarray:
    """``(b, n)`` row weights of ``b`` cluster resamples (multiplicities)."""
    draws = rng.integers(0, n_cl, size=(b, n_cl))
    m = np.zeros((b, n_cl))
    np.add.at(m, (np.repeat(np.arange(b), n_cl), draws.ravel()), 1.0)
    return m[:, codes]


def _engine_slopes(design, x, ys, sx: float, sys_: list, W: np.ndarray) -> np.ndarray:
    """``(B, len(ys))`` slopes of each scaled outcome on the scaled exposure.

    From the design's weighted residual cross-products (Frisch–Waugh): with
    exposure and outcomes pre-divided by their full-sample residual SDs the
    slope is ``M_xy / M_xx · s_x / s_y``. An exposure with no residual
    variance in a resample gets slope 0, as the pseudo-inverse gives it.
    """
    M = design.cross(np.column_stack([x, *ys]), W)
    mxx = M[:, 0, 0]
    ok = mxx > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        b = M[:, 0, 1:] / np.where(ok, mxx, 1.0)[:, None]
    b = np.where(ok[:, None], b, 0.0)
    return b * (sx / np.asarray(sys_, dtype=np.float64))[None, :]


def _design(n: int, cov, smooth):
    from .resampling import BlockDesign

    return BlockDesign(n, cov, smooth, rtol=_BASIS_RTOL)


def _residualised(exposure, outcome, covariates):
    """Exposure and outcome residualised on the covariates plus an intercept,
    over the rows where all are finite: ``(xr, yr, n, residual df)``."""
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
    if dof < 2:
        return x, y, n, dof
    return x - q @ (q.T @ x), y - q @ (q.T @ y), n, dof


def partial_slope(exposure, outcome, covariates=None, *, level: float = 0.95) -> dict:
    """Slope of ``outcome`` on ``exposure`` given ``covariates``, in SD units.

    Both sides are residualised on the covariates plus an intercept and divided
    by their residual SD, so ``beta`` is the outcome change in SDs per SD of
    exposure, conditional on the covariates — the partial correlation. Its
    standard error and interval are the exact OLS ones with ``n − rank − 1``
    residual degrees of freedom. Rows with any non-finite value are dropped.
    """
    from scipy import stats

    xr, yr, n, dof = _residualised(exposure, outcome, covariates)
    empty = {
        "beta": float("nan"),
        "se": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "t": float("nan"),
        "n": n,
        "df": dof,
    }
    if dof < 2:
        return empty
    sx, sy = float(xr.std()), float(yr.std())
    if sx <= 1e-12 or sy <= 1e-12:
        return empty
    beta = float(np.mean((xr / sx) * (yr / sy)))
    beta = min(max(beta, -1.0), 1.0)
    se = math.sqrt(max(1.0 - beta * beta, 0.0) / dof)
    crit = float(stats.t.ppf(0.5 + level / 2.0, dof))
    return {
        "beta": beta,
        "se": se,
        "ci_low": beta - crit * se,
        "ci_high": beta + crit * se,
        "t": beta / se if se > 0 else float("inf"),
        "n": n,
        "df": dof,
    }



def raw_slope(
    exposure, outcome, covariates=None, *, scale: float = 1.0, level: float = 0.95
) -> dict:
    """Slope of ``outcome`` on ``exposure`` given ``covariates``, in raw units.

    The OLS coefficient of the outcome on ``exposure / scale`` with the
    covariates and an intercept (Frisch–Waugh on the residualised pair), so
    ``estimate`` is the outcome change, in its own units, per ``scale`` units
    of exposure (``scale`` = an IQR gives the per-IQR contrast). Its standard
    error and interval are the exact OLS ones with ``n − rank − 1`` residual
    degrees of freedom; ``t`` is the same as :func:`partial_slope`'s. Rows with
    any non-finite value are dropped.
    """
    from scipy import stats

    xr, yr, n, dof = _residualised(exposure, outcome, covariates)
    nan = float("nan")
    out = {
        "estimate": nan,
        "se": nan,
        "ci_low": nan,
        "ci_high": nan,
        "t": nan,
        "n": n,
        "df": dof,
        "scale": float(scale),
    }
    if dof < 2 or not np.isfinite(scale) or scale <= 0:
        return out
    sxx = float(xr @ xr)
    if sxx <= 1e-24 * max(len(xr), 1):
        return out
    b = float(xr @ yr) / sxx
    resid = yr - b * xr
    se = math.sqrt(max(float(resid @ resid), 0.0) / dof / sxx)
    crit = float(stats.t.ppf(0.5 + level / 2.0, dof))
    out.update(
        estimate=b * scale,
        se=se * scale,
        ci_low=(b - crit * se) * scale,
        ci_high=(b + crit * se) * scale,
        t=b / se if se > 0 else float("inf"),
    )
    return out


def paired_contrast(
    exposure,
    target,
    control,
    covariates=None,
    *,
    extra: dict | None = None,
    n_boot: int = 1000,
    clusters=None,
    seed: int = 0,
    level: float = 0.95,
) -> dict:
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

    ``extra`` maps ``"exposure"``, ``"target"`` or ``"control"`` to more
    control columns: the exposure's join both slopes, an outcome's only the
    slope on that outcome. Rows where any of them is non-finite are dropped.
    """
    x = np.asarray(exposure, dtype=np.float64)
    ya = np.asarray(target, dtype=np.float64)
    yb = np.asarray(control, dtype=np.float64)
    cov = None if covariates is None else np.asarray(covariates, dtype=np.float64)
    if cov is not None:
        cov = cov.reshape(len(x), -1)
    blocks = _extra_blocks(extra, len(x), ("exposure", "target", "control"))
    keep = np.isfinite(x) & np.isfinite(ya) & np.isfinite(yb)
    if cov is not None and cov.size:
        keep &= np.isfinite(cov).all(1)
    for block in blocks.values():
        keep &= np.isfinite(block).all(1)
    cl = None if clusters is None else np.asarray(clusters)[keep]
    x, ya, yb = x[keep], ya[keep], yb[keep]
    cov = None if cov is None else cov[keep]
    blocks = {k: v[keep] for k, v in blocks.items()}
    n = int(len(x))
    nan = float("nan")
    out = {
        "delta": nan,
        "ci_low": nan,
        "ci_high": nan,
        "ratio": nan,
        "ratio_ci_low": nan,
        "ratio_ci_high": nan,
        "n": n,
        "n_boot": int(n_boot),
        "clusters": None,
    }

    if blocks:
        return _paired_contrast_blocks(
            x, ya, yb, cov, blocks, cl, out, n_boot=n_boot, seed=seed, level=level
        )
    shared = _picked(blocks, "exposure")
    q, _ = covariate_basis(_with_blocks(cov, shared + _picked(blocks, "target")), n)
    qb = q
    if "target" in blocks or "control" in blocks:
        qb, _ = covariate_basis(
            _with_blocks(cov, shared + _picked(blocks, "control")), n
        )
    if n - max(q.shape[1], qb.shape[1]) - 1 < 2:
        return out
    scales = [_residual_scale(v, b) for v, b in ((x, q), (ya, q), (yb, qb))]
    if qb is not q:
        scales.append(_residual_scale(x, qb))
    if min(scales) <= 1e-12:
        return out
    xs, ya_s, yb_s = x / scales[0], ya / scales[1], yb / scales[2]
    Z = np.column_stack([xs, q])
    Zb = Z if qb is q else np.column_stack([x / scales[3], qb])

    if cl is None:
        codes = np.arange(n)
    else:
        _, codes = np.unique(cl, return_inverse=True)
    n_cl = int(codes.max()) + 1
    out["clusters"] = None if cl is None else n_cl

    def gram(Z):
        p = Z.shape[1]
        G = np.zeros((n_cl, p, p))
        np.add.at(G, codes, Z[:, :, None] * Z[:, None, :])
        return G.reshape(n_cl, p * p), p

    G_flat, p = gram(Z)
    Gb_flat, pb = (G_flat, p) if Zb is Z else gram(Zb)
    Ha = np.zeros((n_cl, p))
    Hb = np.zeros((n_cl, pb))
    np.add.at(Ha, codes, Z * ya_s[:, None])
    np.add.at(Hb, codes, Zb * yb_s[:, None])

    def slopes(m):
        gi = np.linalg.pinv((m @ G_flat).reshape(-1, p, p))
        gib = gi if Zb is Z else np.linalg.pinv((m @ Gb_flat).reshape(-1, pb, pb))
        ba = np.einsum("bpq,bq->bp", gi, m @ Ha)[:, 0]
        bb = np.einsum("bpq,bq->bp", gib, m @ Hb)[:, 0]
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
    out["ci_low"], out["ci_high"] = (
        float(v) for v in np.percentile(deltas, [tail, 100.0 - tail])
    )
    if np.isfinite(ratios).any():
        out["ratio_ci_low"], out["ratio_ci_high"] = (
            float(v) for v in np.nanpercentile(ratios, [tail, 100.0 - tail])
        )
    return out


def _paired_contrast_blocks(
    x, ya, yb, cov, blocks, cl, out, *, n_boot, seed, level
) -> dict:
    """:func:`paired_contrast` with extra control columns, by the block engine.

    Same rows, scales, resamples and statistics; the slopes come from the
    weighted cross-products of :class:`resampling.BlockDesign` instead of a
    dense Gram block per cluster.
    """
    n = len(x)
    nan = float("nan")
    shared = _picked(blocks, "exposure")
    own = "target" in blocks or "control" in blocks
    da = _design(n, cov, _with_blocks(None, shared + _picked(blocks, "target")))
    db = (
        _design(n, cov, _with_blocks(None, shared + _picked(blocks, "control")))
        if own
        else da
    )
    if n - max(da.rank, db.rank) - 1 < 2:
        return out
    scales = [float(da.residuals(v).std()) for v in (x, ya)]
    scales.append(float(db.residuals(yb).std()))
    if own:
        scales.append(float(db.residuals(x).std()))
    if min(scales) <= 1e-12:
        return out
    sxb = scales[3] if own else scales[0]

    if cl is None:
        codes = np.arange(n)
    else:
        _, codes = np.unique(cl, return_inverse=True)
    n_cl = int(codes.max()) + 1
    out["clusters"] = None if cl is None else n_cl

    def slopes(W):
        if not own:
            b = _engine_slopes(da, x, [ya, yb], scales[0], scales[1:3], W)
            return b[:, 0], b[:, 1]
        ba = _engine_slopes(da, x, [ya], scales[0], [scales[1]], W)[:, 0]
        bb = _engine_slopes(db, x, [yb], sxb, [scales[2]], W)[:, 0]
        return ba, bb

    ba0, bb0 = slopes(np.ones((1, n)))
    out["delta"] = float(abs(ba0[0]) - abs(bb0[0]))
    out["ratio"] = float(bb0[0] / ba0[0]) if abs(ba0[0]) > 1e-12 else nan
    if n_boot <= 0:
        return out

    rng = np.random.default_rng(seed)
    deltas, ratios = [], []
    for start in range(0, int(n_boot), _BATCH):
        b = min(_BATCH, int(n_boot) - start)
        ba, bb = slopes(_resample_weights(rng, b, n_cl, codes))
        deltas.append(np.abs(ba) - np.abs(bb))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios.append(np.where(np.abs(ba) > 1e-12, bb / ba, np.nan))
    deltas = np.concatenate(deltas)
    ratios = np.concatenate(ratios)
    tail = (1.0 - level) / 2.0 * 100.0
    out["ci_low"], out["ci_high"] = (
        float(v) for v in np.percentile(deltas, [tail, 100.0 - tail])
    )
    if np.isfinite(ratios).any():
        out["ratio_ci_low"], out["ratio_ci_high"] = (
            float(v) for v in np.nanpercentile(ratios, [tail, 100.0 - tail])
        )
    return out


def specificity_contrast(
    exposure,
    other,
    target,
    control,
    covariates=None,
    *,
    extra: dict | None = None,
    n_boot: int = 1000,
    clusters=None,
    seed: int = 0,
    level: float = 0.95,
) -> dict:
    """``ΔD = D − D_other``: is ``exposure`` more specific than ``other``.

    ``D = |β_target| − |β_control|`` per exposure, as in
    :func:`paired_contrast`, on the rows where both exposures, the target, the
    control and the covariates are all finite. The two exposures' slopes on the
    same rows are strongly correlated, so their separate intervals overlap even
    when the difference is clear: every resample (multiplicities over entities,
    or whole clusters with ``clusters``) re-estimates all four slopes from
    per-cluster Gram blocks, one design per exposure. Returns the four signed
    slopes (``beta_target``, ``beta_control``, ``beta_target_other``,
    ``beta_control_other``), ``d``, ``d_other``, ``delta`` with its percentile
    interval, the one-sided bootstrap ``p_value`` for ``delta > 0`` (share of
    resamples ≤ 0), ``n``, ``n_boot`` and ``clusters``. Unit weights reproduce
    :func:`partial_slope`, and two identical exposures give ``delta = 0`` on
    every resample.

    ``extra`` maps ``"exposure"``, ``"other"``, ``"target"`` or ``"control"``
    to more control columns, which join the covariates of the two slopes that
    involve that variable (one design per exposure and outcome when an outcome
    has its own). Rows where any of them is non-finite are dropped.
    """
    xa = np.asarray(exposure, dtype=np.float64)
    xb = np.asarray(other, dtype=np.float64)
    ya = np.asarray(target, dtype=np.float64)
    yb = np.asarray(control, dtype=np.float64)
    cov = None if covariates is None else np.asarray(covariates, dtype=np.float64)
    if cov is not None:
        cov = cov.reshape(len(xa), -1)
    blocks = _extra_blocks(extra, len(xa), ("exposure", "other", "target", "control"))
    keep = np.isfinite(xa) & np.isfinite(xb) & np.isfinite(ya) & np.isfinite(yb)
    if cov is not None and cov.size:
        keep &= np.isfinite(cov).all(1)
    for block in blocks.values():
        keep &= np.isfinite(block).all(1)
    cl = None if clusters is None else np.asarray(clusters)[keep]
    xa, xb, ya, yb = xa[keep], xb[keep], ya[keep], yb[keep]
    cov = None if cov is None else cov[keep]
    blocks = {k: v[keep] for k, v in blocks.items()}
    n = int(len(xa))
    nan = float("nan")
    out = {
        "beta_target": nan,
        "beta_control": nan,
        "beta_target_other": nan,
        "beta_control_other": nan,
        "d": nan,
        "d_other": nan,
        "delta": nan,
        "ci_low": nan,
        "ci_high": nan,
        "p_value": nan,
        "n": n,
        "n_boot": int(n_boot),
        "clusters": None,
    }

    if blocks:
        return _specificity_blocks(
            xa, xb, ya, yb, cov, blocks, cl, out, n_boot=n_boot, seed=seed, level=level
        )

    # ── Designs: per exposure, or per exposure and outcome ──────
    # Each is ``(exposure, basis, outcomes)``, in the order of the four slopes.
    bases: dict = {}

    def basis(*roles):
        key = tuple(r for r in roles if r in blocks)
        if key not in bases:
            bases[key], _ = covariate_basis(_with_blocks(cov, _picked(blocks, *key)), n)
        return bases[key]

    own = "target" in blocks or "control" in blocks
    specs = []
    for x, role in ((xa, "exposure"), (xb, "other")):
        if own:
            specs += [
                (x, basis(role, "target"), [ya]),
                (x, basis(role, "control"), [yb]),
            ]
        else:
            specs.append((x, basis(role), [ya, yb]))
    if any(n - q.shape[1] - 1 < 2 for _, q, _ in specs):
        return out
    scales = [[_residual_scale(v, q) for v in (x, *ys)] for x, q, ys in specs]
    if min(min(s) for s in scales) <= 1e-12:
        return out

    if cl is None:
        codes = np.arange(n)
    else:
        _, codes = np.unique(cl, return_inverse=True)
    n_cl = int(codes.max()) + 1
    out["clusters"] = None if cl is None else n_cl

    # ── Per-cluster Gram blocks, one per design ─────────────────
    designs = []
    for (x, q, ys), s in zip(specs, scales, strict=True):
        Z = np.column_stack([x / s[0], q])
        Y = np.column_stack([y / sy for y, sy in zip(ys, s[1:], strict=True)])
        p, k = Z.shape[1], Y.shape[1]
        G = np.zeros((n_cl, p, p))
        np.add.at(G, codes, Z[:, :, None] * Z[:, None, :])
        H = np.zeros((n_cl, p, k))
        np.add.at(H, codes, Z[:, :, None] * Y[:, None, :])
        designs.append((G.reshape(n_cl, p * p), H.reshape(n_cl, p * k), p, k))

    def slopes(m):
        """``(b, 4)`` slopes: target and control on each exposure."""
        cols = []
        for G_flat, H_flat, p, k in designs:
            gi = np.linalg.pinv((m @ G_flat).reshape(-1, p, p))
            b = np.einsum("bpq,bqk->bpk", gi, (m @ H_flat).reshape(-1, p, k))
            cols.append(b[:, 0, :])
        return np.concatenate(cols, axis=1)

    def contrast(b):
        d = np.abs(b[:, 0]) - np.abs(b[:, 1])
        return d - (np.abs(b[:, 2]) - np.abs(b[:, 3]))

    b0 = slopes(np.ones((1, n_cl)))
    for key, v in zip(
        ("beta_target", "beta_control", "beta_target_other", "beta_control_other"),
        b0[0],
        strict=True,
    ):
        out[key] = float(v)
    out["d"] = float(abs(b0[0, 0]) - abs(b0[0, 1]))
    out["d_other"] = float(abs(b0[0, 2]) - abs(b0[0, 3]))
    out["delta"] = float(contrast(b0)[0])
    if n_boot <= 0:
        return out

    rng = np.random.default_rng(seed)
    deltas = []
    for start in range(0, int(n_boot), _BATCH):
        b = min(_BATCH, int(n_boot) - start)
        draws = rng.integers(0, n_cl, size=(b, n_cl))
        m = np.zeros((b, n_cl))
        np.add.at(m, (np.repeat(np.arange(b), n_cl), draws.ravel()), 1.0)
        deltas.append(contrast(slopes(m)))
    deltas = np.concatenate(deltas)
    tail = (1.0 - level) / 2.0 * 100.0
    out["ci_low"], out["ci_high"] = (
        float(v) for v in np.percentile(deltas, [tail, 100.0 - tail])
    )
    out["p_value"] = float((np.sum(deltas <= 0.0) + 1) / (len(deltas) + 1))
    return out


def _specificity_blocks(
    xa, xb, ya, yb, cov, blocks, cl, out, *, n_boot, seed, level
) -> dict:
    """:func:`specificity_contrast` with extra control columns, by the block
    engine: the same designs, scales and resamples as the dense path."""
    n = len(xa)
    designs: dict = {}

    def design(*roles):
        key = tuple(r for r in roles if r in blocks)
        if key not in designs:
            designs[key] = _design(n, cov, _with_blocks(None, _picked(blocks, *key)))
        return designs[key]

    own = "target" in blocks or "control" in blocks
    specs = []
    for x, role in ((xa, "exposure"), (xb, "other")):
        if own:
            specs += [
                (x, design(role, "target"), [ya]),
                (x, design(role, "control"), [yb]),
            ]
        else:
            specs.append((x, design(role), [ya, yb]))
    if any(n - d.rank - 1 < 2 for _, d, _ in specs):
        return out
    scales = [
        [float(d.residuals(v).std()) for v in (x, *ys)] for x, d, ys in specs
    ]
    if min(min(s) for s in scales) <= 1e-12:
        return out

    if cl is None:
        codes = np.arange(n)
    else:
        _, codes = np.unique(cl, return_inverse=True)
    n_cl = int(codes.max()) + 1
    out["clusters"] = None if cl is None else n_cl

    def slopes(W):
        """``(b, 4)`` slopes: target and control on each exposure."""
        return np.concatenate(
            [
                _engine_slopes(d, x, ys, sc[0], sc[1:], W)
                for (x, d, ys), sc in zip(specs, scales, strict=True)
            ],
            axis=1,
        )

    def contrast(b):
        d = np.abs(b[:, 0]) - np.abs(b[:, 1])
        return d - (np.abs(b[:, 2]) - np.abs(b[:, 3]))

    b0 = slopes(np.ones((1, n)))
    for key, v in zip(
        ("beta_target", "beta_control", "beta_target_other", "beta_control_other"),
        b0[0],
        strict=True,
    ):
        out[key] = float(v)
    out["d"] = float(abs(b0[0, 0]) - abs(b0[0, 1]))
    out["d_other"] = float(abs(b0[0, 2]) - abs(b0[0, 3]))
    out["delta"] = float(contrast(b0)[0])
    if n_boot <= 0:
        return out

    rng = np.random.default_rng(seed)
    deltas = []
    for start in range(0, int(n_boot), _BATCH):
        b = min(_BATCH, int(n_boot) - start)
        deltas.append(contrast(slopes(_resample_weights(rng, b, n_cl, codes))))
    deltas = np.concatenate(deltas)
    tail = (1.0 - level) / 2.0 * 100.0
    out["ci_low"], out["ci_high"] = (
        float(v) for v in np.percentile(deltas, [tail, 100.0 - tail])
    )
    out["p_value"] = float((np.sum(deltas <= 0.0) + 1) / (len(deltas) + 1))
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


def transfer_test(
    exposure,
    target,
    controls: dict,
    covariates=None,
    *,
    extra: dict | None = None,
    n_boot: int = 1000,
    clusters=None,
    seed: int = 0,
    level: float = 0.95,
) -> dict:
    """The frozen exposure scored against the target and every control.

    Returns ``{"target": partial_slope, "controls": {name: {...}}}``. Each
    control entry holds its own slope (on the rows where it is observed — a
    missing control costs that control's statistics and nothing else), the
    paired contrast against the target, and ``nonspecific``: the control's
    interval excludes zero while Δ's interval includes it. The flag reports;
    it does not stop anything.

    ``extra`` holds more control columns: ``"exposure"`` for every slope,
    ``"target"`` for the target's, ``"controls"`` ``{name: columns}`` for
    each control's; see :func:`paired_contrast`.
    """
    extra = dict(extra or {})
    own_extra = dict(extra.pop("controls", None) or {})
    unknown = set(extra) - {"exposure", "target"}
    if unknown:
        raise ValueError(f"extra: unknown role(s) {sorted(unknown)}.")

    n = len(np.asarray(exposure))

    def design(*blocks):
        """The covariates with the given blocks appended (as given without)."""
        parts = [
            np.asarray(b, dtype=np.float64).reshape(n, -1)
            for b in blocks
            if b is not None
        ]
        if not parts:
            return covariates
        cov = None if covariates is None else np.asarray(covariates, dtype=np.float64)
        return _with_blocks(None if cov is None else cov.reshape(n, -1), parts)

    out = {
        "target": partial_slope(
            exposure,
            target,
            design(extra.get("exposure"), extra.get("target")),
            level=level,
        ),
        "controls": {},
    }
    for i, (name, values) in enumerate(controls.items()):
        own = partial_slope(
            exposure,
            values,
            design(extra.get("exposure"), own_extra.get(name)),
            level=level,
        )
        contrast = paired_contrast(
            exposure,
            target,
            values,
            covariates,
            extra={
                "exposure": extra.get("exposure"),
                "target": extra.get("target"),
                "control": own_extra.get(name),
            },
            n_boot=n_boot,
            clusters=clusters,
            seed=seed + i,
            level=level,
        )
        excludes_zero = np.isfinite(own["ci_low"]) and (
            own["ci_low"] > 0 or own["ci_high"] < 0
        )
        delta_has_zero = (
            np.isfinite(contrast["ci_low"])
            and contrast["ci_low"] <= 0 <= contrast["ci_high"]
        )
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
