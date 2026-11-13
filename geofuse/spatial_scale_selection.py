"""Scale selection and effective bandwidth of a spatial adjustment.

Keller & Szpiro (2020, §4) choose the amount of spatial adjustment by fitting
outcome models that include the adjustment basis but not the exposure: the
basis whose model has the smallest AIC (AIC-NE) or BIC (BIC-NE) is chosen,
the observed confounders included. Here the candidates are the fixed-scale
thin-plate bases of :func:`geofuse.spatial_basis.build_block_basis` at a list
of pattern scales, all built on **one** set of clusters, so only the smooth
changes between candidates, plus no smooth at all.

The adjustment is quantified as a distance (their §3.1): for each of a sample
of observations, the weights the smoother ``S`` gives the other observations
are read against their distance. :func:`effective_bandwidth` returns two
summaries: the Keller & Szpiro rule (a loess of every sampled row's weights
on distance, the pointwise median of those curves, and its first zero
crossing) and the Rainey & Keller (2024) rule (per sampled row the first
distance with a negative weight, and the median of those). Block bases give
no weight across clusters, so distances are taken within each row's cluster.
"""

from __future__ import annotations

import numpy as np

from geofuse import spatial_basis as sb

#: Candidate pattern scales (m), besides no smooth.
DEFAULT_SCALES_M: tuple[int, ...] = (
    50000,
    30000,
    20000,
    15000,
    10000,
    7000,
    5000,
    4000,
    3000,
)

#: Cluster gap shared by every candidate when the job sets none.
DEFAULT_EPS_M = 10_000.0

#: Criteria of the selection.
CRITERIA: tuple[str, ...] = ("aic", "bic")

#: Rows sampled for the effective bandwidth, and the loess span.
BANDWIDTH_SAMPLE = 1000
LOESS_SPAN = 0.1

#: Distances on which the smoothed weight curves are read.
_GRID_POINTS = 200


# ────────────────────────────────────────────────────────────────────
# Information criteria of the outcome model without the exposure
# ────────────────────────────────────────────────────────────────────


def information_criteria(y: np.ndarray, X: np.ndarray | None) -> dict:
    """Gaussian AIC and BIC of ``OLS(y ~ 1 + X)``, as statsmodels reports them.

    ``k`` is the rank of ``[1, X]`` (statsmodels' ``df_model + 1``), the
    log-likelihood ``−n/2 (ln 2π + ln(RSS/n) + 1)``, so ``aic = −2ℓ + 2k`` and
    ``bic = −2ℓ + k ln n``.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    D = np.ones((n, 1))
    if X is not None and np.size(X):
        D = np.column_stack([D, np.asarray(X, dtype=np.float64).reshape(n, -1)])
    coef, _res, rank, _sv = np.linalg.lstsq(D, y, rcond=None)
    r = y - D @ coef
    rss = float(r @ r)
    llf = -0.5 * n * (np.log(2 * np.pi) + np.log(max(rss, 1e-300) / n) + 1.0)
    k = int(rank)
    return {
        "n": n,
        "columns": int(D.shape[1] - 1),
        "rank": k,
        "aic": float(-2 * llf + 2 * k),
        "bic": float(-2 * llf + k * np.log(n)),
    }


def candidate_bases(
    xy: np.ndarray,
    scales: "list[float] | tuple[float, ...]",
    *,
    eps_m: float | None = None,
    neighbours: int = sb.DEFAULT_NEIGHBOURS,
    min_component: int | None = sb.DEFAULT_MIN_COMPONENT,
) -> "dict[float, sb.SpatialBasis]":
    """The fixed-scale basis at every scale on one set of clusters (gap
    ``eps_m``, default :data:`DEFAULT_EPS_M`)."""
    eps = float(eps_m) if eps_m else DEFAULT_EPS_M
    return {
        float(s): sb.build_block_basis(
            xy,
            eps=eps,
            neighbours=neighbours,
            scale_m=float(s),
            min_component=min_component,
        )
        for s in scales
    }


def _columns(basis: "sb.SpatialBasis") -> np.ndarray:
    if not basis.has_spatial:
        return np.empty((basis.n, 0))
    return basis.design(max(1, basis.max_df))


def profile(
    xy: np.ndarray,
    outcomes: "dict[str, np.ndarray]",
    covariates: np.ndarray | None,
    scales: "list[float] | tuple[float, ...]",
    *,
    eps_m: float | None = None,
    neighbours: int = sb.DEFAULT_NEIGHBOURS,
    min_component: int | None = sb.DEFAULT_MIN_COMPONENT,
    bases: "dict | None" = None,
) -> list[dict]:
    """AIC-NE / BIC-NE of every outcome at every candidate.

    One row per outcome and candidate (``scale_m`` ``None`` = covariates
    only): ``n``, ``columns`` (smooth columns), ``aic``, ``bic``, their
    differences from the outcome's best, and whether each criterion chose it.
    Rows with a missing value in an outcome, the covariates or the locations
    are left out of that outcome's fits.
    """
    xy = np.asarray(xy, dtype=np.float64)
    cov = None if covariates is None or not np.size(covariates) else covariates
    finite_xy = np.isfinite(xy).all(axis=1)
    if cov is not None:
        finite_xy &= np.isfinite(cov).all(axis=1)
    rows: list[dict] = []
    for name, y in outcomes.items():
        y = np.asarray(y, dtype=np.float64)
        keep = finite_xy & np.isfinite(y)
        if keep.sum() < 10:
            continue
        local = bases if bases is not None and keep.all() else candidate_bases(
            xy[keep],
            scales,
            eps_m=eps_m,
            neighbours=neighbours,
            min_component=min_component,
        )
        c = None if cov is None else cov[keep]
        fits = [(None, information_criteria(y[keep], c), 0)]
        for s in scales:
            S = _columns(local[float(s)])
            design = S if c is None else np.column_stack([c, S])
            fits.append((float(s), information_criteria(y[keep], design), S.shape[1]))
        best = {k: min(f[1][k] for f in fits) for k in CRITERIA}
        winner = {
            k: min(fits, key=lambda f: (f[1][k], f[1]["rank"]))[0] for k in CRITERIA
        }
        for scale, ic, ncol in fits:
            rows.append(
                {
                    "outcome": name,
                    "scale_m": scale,
                    "n": ic["n"],
                    "columns": int(ncol),
                    "aic": ic["aic"],
                    "bic": ic["bic"],
                    "delta_aic": ic["aic"] - best["aic"],
                    "delta_bic": ic["bic"] - best["bic"],
                    "chosen_aic": scale == winner["aic"],
                    "chosen_bic": scale == winner["bic"],
                }
            )
    return rows


def chosen(rows: list[dict], outcome: str, criterion: str) -> "float | None | str":
    """The scale ``criterion`` chose for ``outcome`` (``None`` = no smooth);
    ``"missing"`` when the profile holds no row of that outcome."""
    for r in rows:
        if r["outcome"] == outcome and r[f"chosen_{criterion}"]:
            return r["scale_m"]
    return "missing"


def summary(rows: list[dict], criterion: str) -> list[dict]:
    """Per stage and outcome over the splits: the scale chosen on each split,
    how often each scale won by each criterion, and the median chosen scale
    (no smooth counts as 0 m)."""
    out = []
    keys = dict.fromkeys((r.get("stage"), r["outcome"]) for r in rows)
    for stage, outcome in keys:
        sub = [r for r in rows if r.get("stage") == stage and r["outcome"] == outcome]
        entry: dict = {"stage": stage, "outcome": outcome, "criterion": criterion}
        for crit in CRITERIA:
            picks = {
                r.get("split"): r["scale_m"] for r in sub if r[f"chosen_{crit}"]
            }
            wins: dict = {}
            for scale in picks.values():
                label = "none" if scale is None else f"{float(scale):g}"
                wins[label] = wins.get(label, 0) + 1
            values = [0.0 if s is None else float(s) for s in picks.values()]
            entry[f"{crit}_per_split"] = {
                str(k): (None if v is None else float(v)) for k, v in picks.items()
            }
            entry[f"{crit}_wins"] = wins
            entry[f"{crit}_median_m"] = float(np.median(values)) if values else None
        out.append(entry)
    return out


# ────────────────────────────────────────────────────────────────────
# Effective bandwidth
# ────────────────────────────────────────────────────────────────────


def _loess(x: np.ndarray, y: np.ndarray, at: np.ndarray, span: float) -> np.ndarray:
    """Local quadratic regression with tricube weights (R's ``loess`` with
    ``degree = 2``, evaluated directly), ``span`` of the points per fit; every
    evaluation point is solved at once."""
    n = len(x)
    out = np.full(len(at), np.nan)
    if n < 4 or not len(at):
        return out
    q = min(n, max(4, int(np.ceil(span * n))))
    dist = np.abs(x[None, :] - at[:, None])
    idx = np.argpartition(dist, q - 1, axis=1)[:, :q]
    d = np.take_along_axis(dist, idx, axis=1)
    h = d.max(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(h > 0, (1 - (d / np.where(h > 0, h, 1.0)) ** 3) ** 3, 1.0)
    t = x[idx] - at[:, None]
    A = np.stack([np.ones_like(t), t, t * t], axis=2)
    Aw = A * w[..., None]
    N = np.swapaxes(Aw, 1, 2) @ A
    r = np.swapaxes(Aw, 1, 2) @ y[idx][..., None]
    # Scale the polynomial terms so the normal equations stay conditioned.
    sc = np.maximum(h[:, 0], 1e-300)
    S = np.stack([np.ones_like(sc), 1 / sc, 1 / sc**2], axis=1)
    Ns = N * S[:, :, None] * S[:, None, :]
    rs = r[..., 0] * S
    try:
        coef = np.linalg.solve(Ns, rs[..., None])[..., 0]
    except np.linalg.LinAlgError:
        coef = (np.linalg.pinv(Ns) @ rs[..., None])[..., 0]
    out[:] = coef[:, 0]
    return out


def _first_crossing(grid: np.ndarray, curve: np.ndarray) -> float | None:
    """First distance at which ``curve`` reaches zero, linearly interpolated."""
    ok = np.isfinite(curve)
    g, c = grid[ok], curve[ok]
    below = np.flatnonzero(c <= 0)
    if not below.size:
        return None
    j = int(below[0])
    if j == 0:
        return float(g[0])
    x0, x1, y0, y1 = g[j - 1], g[j], c[j - 1], c[j]
    return float(x0 + (x1 - x0) * y0 / (y0 - y1)) if y0 != y1 else float(x1)


def effective_bandwidth(
    xy: np.ndarray,
    labels: np.ndarray,
    weights_of,
    *,
    nsamp: int = BANDWIDTH_SAMPLE,
    span: float = LOESS_SPAN,
    seed: int = 0,
) -> dict:
    """``k̂`` of a smoother by the Keller & Szpiro and the Rainey & Keller rules.

    ``weights_of(j)`` returns column ``j`` of the smoother (the weights it
    gives every row when smoothing row ``j``'s indicator; symmetric smoothers
    give the same as the row). ``min(nsamp, n)`` rows are sampled; each is
    read against the rows of its own cluster (``labels``). Returns ``{k_ks,
    k_rk, n_sampled, clusters}``; a ``k`` is ``None`` when no weight turns
    negative.
    """
    xy = np.asarray(xy, dtype=np.float64)
    labels = np.asarray(labels)
    n = len(xy)
    rng = np.random.default_rng(int(seed))
    pick = np.sort(rng.choice(n, size=min(int(nsamp), n), replace=False))
    dmax = 0.0
    per: list[tuple[np.ndarray, np.ndarray]] = []
    first_negative = []
    for j in pick:
        same = np.flatnonzero(labels == labels[j])
        w = np.asarray(weights_of(int(j)), dtype=np.float64)[same]
        d = np.sqrt(((xy[same] - xy[j]) ** 2).sum(axis=1))
        order = np.argsort(d, kind="stable")
        d, w = d[order], w[order]
        per.append((d, w))
        if d.size:
            dmax = max(dmax, float(d.max()))
        neg = np.flatnonzero(w < 0)
        first_negative.append(float(d[neg[0]]) if neg.size else np.nan)
    grid = np.linspace(0.0, dmax, _GRID_POINTS) if dmax > 0 else np.zeros(1)
    curves = np.full((len(per), len(grid)), np.nan)
    for i, (d, w) in enumerate(per):
        if len(d) < 4:
            continue
        inside = grid <= d.max()
        curves[i, inside] = _loess(d, w, grid[inside], span)
    with np.errstate(all="ignore"):
        median = (
            np.nanmedian(curves, axis=0) if len(per) else np.full(len(grid), np.nan)
        )
    rk = np.asarray(first_negative, dtype=float)
    return {
        "k_ks": _first_crossing(grid, median),
        "k_rk": float(np.nanmedian(rk)) if np.isfinite(rk).any() else None,
        "n_sampled": int(len(pick)),
        "clusters": int(len(np.unique(labels))),
    }


def projection_weights(columns: np.ndarray):
    """``weights_of`` for the projection onto ``[1, columns]`` (an unpenalized
    basis): column ``j`` of ``H = Q Qᵀ``."""
    from geofuse import resampling

    n = len(columns)
    Q = resampling.orthonormal(np.column_stack([np.ones(n), columns]))
    return lambda j: Q @ Q[j]


def basis_labels(basis: "sb.SpatialBasis", n: int) -> np.ndarray:
    """Cluster label of every row of a block basis: rows sharing a non-zero
    column belong together (rows in no block get their own label)."""
    from geofuse import resampling

    cols = np.column_stack([basis.linear, basis.radial]) if basis.has_spatial else None
    labels = np.arange(n) + n
    if cols is None or not np.size(cols):
        return labels
    for k, (rows, _c) in enumerate(resampling.smooth_blocks(cols)):
        labels[rows] = k
    return labels
