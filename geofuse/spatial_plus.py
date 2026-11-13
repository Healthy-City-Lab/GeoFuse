"""Spatial+ (Dupont, Wood & Augustin 2022) with one penalized smooth per site.

The model is ``y = β x + Zγ + f(t) + ε`` with ``f`` a partial thin-plate spline
fitted with a wiggliness penalty. Spatial+ fits the exposure on location
first, ``r_x = x − f̂_x`` (step 1), then the outcome on the residual,
``y = β r_x + Zγ + f⁺(t) + ε`` (step 2), so

    β̂⁺ = r_xᵀ (I − S) y / r_xᵀ (I − S) r_x,

``S`` the smoother of the step-2 model without the exposure, at the penalties
chosen with it. Unpenalized, both steps reduce to the plain basis adjustment
(Dupont et al. 2022, §4); the bias Spatial+ removes is the penalty's.

Each site (group) has its own intercept and linear trend, unpenalized, and its
own penalized thin-plate regression spline with its own smoothing parameter,
the analogue of mgcv's ``y ~ group + s(x, y, by = group, bs = "tp", k = K)``.
The penalties are chosen by GCV (as published) or REML.

Construction
------------

A group's spline is the knot-based thin-plate regression spline of Wood
(2003): ``K`` maximin knots among the group's rows, the ``K − 3`` radial
functions constrained to carry no linear trend
(:func:`geofuse.spatial_basis.radial_coefficients`), and the knot energy as
the penalty, which is diagonal on those functions (``w``). Dividing each
function by ``√w`` makes the penalty a ridge, and rotating the group's
functions, residualized on its trend, to the eigenvectors of their cross
product makes that cross product diagonal (``s``). The rotation leaves a ridge
a ridge, so for any smoothing parameters every quantity of the fit is a
diagonal solve plus a correction of the rank of the covariates (Woodbury):

    A(λ) = diag(s + λ) − V Vᵀ,   V = Cᵀ Q_Z,

``C`` the rotated functions and ``Q_Z`` an orthonormal basis of the covariates
after the group trends. A criterion and its gradient in ``log λ`` cost
``O(p · q²)`` for ``p`` functions and ``q`` covariates, so the penalties of
every group are chosen jointly by quasi-Newton, as mgcv does.

Resamples keep the penalties at their full-sample values: a replicate is a
weighted penalized least squares with fixed penalties, refitted block by
block (:func:`replicate_slopes`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from geofuse import resampling
from geofuse import spatial_basis as sb

#: Criteria that pick the smoothing parameters.
CRITERIA: tuple[str, ...] = ("gcv", "reml")

#: Knots per group (mgcv's ``k``; the paper's simulation size).
DEFAULT_K = 300

#: Cluster gap of the gap-rule groups, and the smallest group with a level of
#: its own (smaller ones join their nearest larger group).
DEFAULT_EPS_M = 10_000.0
DEFAULT_MIN_COMPONENT = sb.DEFAULT_MIN_COMPONENT

#: A group whose smooth uses more than this share of its penalized functions
#: logs a warning: the basis is too small for the data (mgcv's ``gam.check``).
EDF_WARN_SHARE = 0.9

#: Range of ``log λ`` searched, around the log of the largest eigenvalue.
_LOG_SPAN = 30.0

#: Eigenvalues below this share of the largest carry no data and are dropped.
_EIG_RTOL = 1e-11

#: Replicates refitted at once.
REPLICATE_CHUNK = 32


# ────────────────────────────────────────────────────────────────────
# Groups and their smooths
# ────────────────────────────────────────────────────────────────────


def group_labels(
    xy: np.ndarray,
    *,
    eps_m: float | None = None,
    min_component: int | None = None,
    values: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Dense group labels of the rows and the rule that made them.

    ``values`` (a site column) gives one group per distinct value; otherwise
    the rows split where they are more than ``eps_m`` apart (default
    :data:`DEFAULT_EPS_M`) and groups smaller than ``min_component`` (default
    :data:`DEFAULT_MIN_COMPONENT`) join their nearest larger one.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if values is not None:
        _, labels = np.unique(np.asarray(values).astype(str), return_inverse=True)
        n_groups = int(labels.max()) + 1
        return labels.astype(np.int64), {"rule": "column", "n_groups": n_groups}
    eps = float(eps_m) if eps_m else DEFAULT_EPS_M
    labels, _ = sb.connected_components(xy, eps=eps)
    size = DEFAULT_MIN_COMPONENT if min_component is None else int(min_component)
    n_merged = 0
    if size > 1:
        labels, _moved, n_merged = sb.merge_small_components(xy, labels, size)
    return labels, {
        "rule": "gap",
        "eps_m": eps,
        "min_component": size,
        "merged_components": int(n_merged),
        "n_groups": int(labels.max()) + 1 if len(labels) else 0,
    }


@dataclass
class Group:
    """One group's columns: ``Qt`` an orthonormal basis of its intercept and
    trend, ``C`` its penalized functions (residualized on ``Qt``, rotated so
    ``CᵀC = diag(s)``, ridge penalty), on ``rows`` of the row set."""

    rows: np.ndarray
    Qt: np.ndarray
    C: np.ndarray
    s: np.ndarray
    knots: int
    #: Knot coordinates in the group's scaled frame and the map from a point's
    #: radial kernel to the columns of ``C``, for the smoother's weights at
    #: other points (the effective bandwidth).
    center: np.ndarray = field(default_factory=lambda: np.zeros(2))
    span: float = 1.0
    knot_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    to_columns: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    @property
    def m(self) -> int:
        return int(self.C.shape[1])


@dataclass
class Smoother:
    """Every group's smooth over ``n`` rows."""

    n: int
    groups: list[Group]
    labels: np.ndarray
    k: int

    @property
    def p(self) -> int:
        """Penalized functions in all."""
        return int(sum(g.m for g in self.groups))

    @property
    def offsets(self) -> np.ndarray:
        return np.concatenate([[0], np.cumsum([g.m for g in self.groups])]).astype(int)

    def group_of_column(self) -> np.ndarray:
        """Group index of every penalized function."""
        return np.repeat(np.arange(len(self.groups)), [g.m for g in self.groups])

    def residualize_trends(self, u: np.ndarray) -> np.ndarray:
        """``u`` (``(n,)`` or ``(n, k)``) minus each group's trend fit."""
        out = np.array(u, dtype=np.float64, copy=True)
        for g in self.groups:
            sub = out[g.rows]
            out[g.rows] = sub - g.Qt @ (g.Qt.T @ sub)
        return out

    def cross(self, u: np.ndarray) -> np.ndarray:
        """``Cᵀ u`` stacked over groups (``u`` already trend-free)."""
        u = np.asarray(u, dtype=np.float64)
        return np.concatenate([g.C.T @ u[g.rows] for g in self.groups], axis=0)

    def times(self, c: np.ndarray) -> np.ndarray:
        """``C c`` in row space, for a stacked coefficient vector."""
        out = np.zeros(self.n if c.ndim == 1 else (self.n, c.shape[1]))
        off = self.offsets
        for j, g in enumerate(self.groups):
            out[g.rows] = g.C @ c[off[j] : off[j + 1]]
        return out

    def subset(self, j: int) -> tuple["Smoother", np.ndarray]:
        """Group ``j`` alone, over its own rows, and those rows."""
        g = self.groups[j]
        local = Group(
            rows=np.arange(len(g.rows)),
            Qt=g.Qt,
            C=g.C,
            s=g.s,
            knots=g.knots,
            center=g.center,
            span=g.span,
            knot_xy=g.knot_xy,
            to_columns=g.to_columns,
        )
        alone = Smoother(len(g.rows), [local], np.zeros(len(g.rows), int), self.k)
        return alone, g.rows


def _knot_penalty(knots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Constrained radial coefficients and their penalty (knot energy)
    eigenvalues, both coarsest first; positive eigenvalues only."""
    k = len(knots)
    T = np.column_stack([np.ones(k), knots])
    Q, _ = np.linalg.qr(T, mode="complete")
    Z = Q[:, 3:]
    E = sb._tps_eta(cdist(knots, knots))
    w, V = np.linalg.eigh(Z.T @ E @ Z)
    order = np.argsort(np.abs(w))[::-1]
    w, V = w[order], V[:, order]
    keep = w > _EIG_RTOL * max(float(np.abs(w).max(initial=0.0)), 1e-300)
    return Z @ V[:, keep], w[keep]


def build(xy: np.ndarray, labels: np.ndarray, *, k: int = DEFAULT_K) -> Smoother:
    """The per-group thin-plate smooths of rows ``xy`` with group ``labels``.

    Group ``g`` gets ``K = min(n_g − 4, k)`` maximin knots among its rows (no
    radial functions below 4 knots, no trend below 3 rows).
    """
    xy = np.asarray(xy, dtype=np.float64)
    labels = np.asarray(labels)
    n = len(xy)
    groups = []
    for lab in np.unique(labels):
        rows = np.flatnonzero(labels == lab)
        n_g = len(rows)
        center = xy[rows].mean(axis=0)
        z = xy[rows] - center
        span = float(np.sqrt((z**2).sum(axis=1)).std()) or 1.0
        z = z / span
        T = np.column_stack([np.ones(n_g), z]) if n_g >= 3 else np.ones((n_g, 1))
        Qt = resampling.orthonormal(T)
        kk = min(n_g - 4, int(k))
        C = np.zeros((n_g, 0))
        s = np.zeros(0)
        knot_xy = np.zeros((0, 2))
        to_columns = np.zeros((0, 0))
        if kk >= 4:
            knot_xy = sb._maximin_knots(z, kk)
            delta, w = _knot_penalty(knot_xy)
            scaled = delta / np.sqrt(w)
            phi = sb._tps_eta(cdist(z, knot_xy)) @ scaled
            phi = phi - Qt @ (Qt.T @ phi)
            s_all, U = np.linalg.eigh(phi.T @ phi)
            keep = s_all > _EIG_RTOL * max(float(s_all.max(initial=0.0)), 1e-300)
            s, U = s_all[keep], U[:, keep]
            C = phi @ U
            to_columns = scaled @ U
        groups.append(
            Group(
                rows=rows,
                Qt=Qt,
                C=C,
                s=s,
                knots=int(max(kk, 0)),
                center=center,
                span=span,
                knot_xy=knot_xy,
                to_columns=to_columns,
            )
        )
    return Smoother(n, groups, labels.astype(np.int64), int(k))


# ────────────────────────────────────────────────────────────────────
# Fits at given smoothing parameters
# ────────────────────────────────────────────────────────────────────


class Design:
    """The penalized model ``u ~ dense + group trends + Σ_g s_g(t)``.

    ``dense`` holds the unpenalized covariates (``None`` for none; the
    intercept is in the group trends). Every method takes the smoothing
    parameters as a vector over the groups.
    """

    def __init__(self, smoother: Smoother, dense: np.ndarray | None = None):
        self.sm = smoother
        n = smoother.n
        if dense is not None and np.size(dense):
            Z = smoother.residualize_trends(
                np.asarray(dense, dtype=np.float64).reshape(n, -1)
            )
            self.Qz = resampling.orthonormal(Z, rtol=1e-10)
        else:
            self.Qz = np.zeros((n, 0))
        self.V = (
            smoother.cross(self.Qz)
            if smoother.p
            else np.zeros((0, self.Qz.shape[1]))
        )
        self.s = (
            np.concatenate([g.s for g in smoother.groups])
            if smoother.p
            else np.zeros(0)
        )
        self.col_group = smoother.group_of_column()
        self.m = np.array([g.m for g in smoother.groups], dtype=float)
        self.r0 = int(sum(g.Qt.shape[1] for g in smoother.groups) + self.Qz.shape[1])

    # ── Projections ─────────────────────────────────────────────

    def residual0(self, u: np.ndarray) -> np.ndarray:
        """``u`` residualized on the unpenalized columns (trends and dense)."""
        t = self.sm.residualize_trends(u)
        return t - self.Qz @ (self.Qz.T @ t)

    def _b(self, u0: np.ndarray) -> np.ndarray:
        return self.sm.cross(u0) if self.sm.p else np.zeros((0,) + u0.shape[1:])

    def state(self, lam: np.ndarray) -> dict:
        """The factors of ``A(λ)`` shared by every quantity at ``lam``."""
        lam_col = np.asarray(lam, dtype=float)[self.col_group]
        d = self.s + lam_col
        U = self.V / d[:, None]
        K = np.eye(self.V.shape[1]) - self.V.T @ U
        Kinv = np.linalg.inv(K) if K.size else K
        return {"lam": lam_col, "d": d, "U": U, "Kinv": Kinv}

    @staticmethod
    def solve(st: dict, b: np.ndarray) -> np.ndarray:
        """``A(λ)⁻¹ b``."""
        b2 = b if b.ndim == 2 else b[:, None]
        out = b2 / st["d"][:, None]
        if st["Kinv"].size:
            out = out + st["U"] @ (st["Kinv"] @ (st["U"].T @ b2))
        return out if b.ndim == 2 else out[:, 0]

    def quad(self, u: np.ndarray, v: np.ndarray, lam: np.ndarray) -> float:
        """``uᵀ (I − H) v``, ``H`` the hat matrix at ``lam``."""
        u0, v0 = self.residual0(u), self.residual0(v)
        st = self.state(lam)
        bu, bv = self._b(u0), self._b(v0)
        return float(u0 @ v0 - bu @ self.solve(st, bv))

    def residual(self, u: np.ndarray, lam: np.ndarray) -> np.ndarray:
        """``(I − H) u`` in row space."""
        u0 = self.residual0(u)
        if not self.sm.p:
            return u0
        c = self.solve(self.state(lam), self._b(u0))
        fit = self.sm.times(c)
        return u0 - (fit - self.Qz @ (self.Qz.T @ fit))

    def edf(self, lam: np.ndarray) -> tuple[float, np.ndarray]:
        """Total edf ``tr(H)`` and the penalized part per group."""
        st = self.state(lam)
        diag = self._diag_inv(st)
        lam_col = st["lam"]
        per_col = 1.0 - lam_col * diag
        per_group = np.bincount(self.col_group, weights=per_col, minlength=len(self.m))
        return float(self.r0 + per_col.sum()), per_group

    def _diag_inv(self, st: dict) -> np.ndarray:
        """Diagonal of ``A(λ)⁻¹``."""
        out = 1.0 / st["d"]
        if st["Kinv"].size:
            out = out + np.einsum("ij,jk,ik->i", st["U"], st["Kinv"], st["U"])
        return out

    # ── Criteria ────────────────────────────────────────────────

    def _parts(self, y: np.ndarray, rho: np.ndarray) -> dict:
        lam = np.exp(rho)
        st = self.state(lam)
        y0 = self.residual0(y)
        b = self._b(y0)
        c = self.solve(st, b)
        lam_col = st["lam"]
        P = float(y0 @ y0 - b @ c)
        pen = float(lam_col @ (c * c))
        return {"st": st, "c": c, "b": b, "P": P, "rss": P - pen, "lam": lam}

    def criterion(
        self, y: np.ndarray, rho: np.ndarray, kind: str
    ) -> tuple[float, np.ndarray]:
        """``(value, gradient in log λ)`` of GCV or profiled REML for ``y``."""
        n = self.sm.n
        q = self._parts(y, rho)
        st, c, lam = q["st"], q["c"], q["lam"]
        lam_col, d = st["lam"], st["d"]
        G = len(self.m)
        cg = self.col_group
        dinv = self._diag_inv(st)
        if kind == "reml":
            P = max(q["P"], 1e-300)
            nu = n - self.r0
            logdet = float(np.sum(np.log(d)))
            if st["Kinv"].size:
                K = np.eye(self.V.shape[1]) - self.V.T @ st["U"]
                logdet += float(np.linalg.slogdet(K)[1])
            value = nu * np.log(P / nu) + logdet - float(self.m @ np.log(lam))
            grad = (
                nu * lam * np.bincount(cg, weights=c * c, minlength=G) / P
                + lam * np.bincount(cg, weights=dinv, minlength=G)
                - self.m
            )
            return float(value), grad
        if kind != "gcv":
            raise ValueError(f"criterion must be one of {CRITERIA}; got {kind!r}.")
        # tr(H) = r0 + p − tr(A⁻¹Λ); RSS from the penalized fit.
        U, Kinv = st["U"], st["Kinv"]
        diag_ali = lam_col * dinv
        tr_al = float(diag_ali.sum())
        trH = self.r0 + len(d) - tr_al
        denom = n - trH
        rss = q["rss"]
        value = n * rss / denom**2
        # ∂RSS/∂ρ_g = 2 λ_g (A⁻¹ Λ c)_gᵀ c_g
        alc = self.solve(st, lam_col * c)
        d_rss = 2.0 * lam * np.bincount(cg, weights=alc * c, minlength=G)
        # ∂tr(A⁻¹Λ)/∂ρ_g = λ_g [tr_g(A⁻¹) − tr_g(A⁻¹ΛA⁻¹)]
        diag_ala = lam_col / d**2
        if Kinv.size:
            uk = U @ Kinv
            ukuu = np.einsum("ij,ij->i", uk, U)
            M = U.T @ (U * lam_col[:, None])
            ukm = uk @ M
            diag_ala = (
                diag_ala
                + 2.0 * (lam_col / d) * ukuu
                + np.einsum("ij,ij->i", ukm @ Kinv.T, U)
            )
        d_tr_al = lam * (
            np.bincount(cg, weights=dinv, minlength=G)
            - np.bincount(cg, weights=diag_ala, minlength=G)
        )
        d_trH = -d_tr_al
        grad = n * d_rss / denom**2 + 2.0 * n * rss * d_trH / denom**3
        return float(value), grad

    def select(self, y: np.ndarray, kind: str, start: np.ndarray | None = None) -> dict:
        """Smoothing parameters minimizing ``kind`` for ``y``.

        A common ``log λ`` is scanned first, then every group's is refined
        jointly by L-BFGS-B with the analytic gradient. Returns ``{lam,
        value, edf, edf_groups}``.
        """
        G = len(self.m)
        if not self.sm.p:
            lam = np.ones(G)
            edf, per = self.edf(lam)
            return {"lam": lam, "value": None, "edf": edf, "edf_groups": per}
        top = float(np.log(max(self.s.max(), 1e-300)))
        lo, hi = top - _LOG_SPAN, top + 8.0
        if start is None:
            grid = np.linspace(lo, hi, 25)
            vals = [self.criterion(y, np.full(G, r), kind)[0] for r in grid]
            start = np.full(G, grid[int(np.nanargmin(vals))])
        res = minimize(
            lambda r: self.criterion(y, r, kind),
            np.clip(np.asarray(start, dtype=float), lo, hi),
            jac=True,
            method="L-BFGS-B",
            bounds=[(lo, hi)] * G,
            options={"maxiter": 500, "ftol": 1e-13, "gtol": 1e-9},
        )
        lam = np.exp(res.x)
        edf, per = self.edf(lam)
        return {"lam": lam, "value": float(res.fun), "edf": edf, "edf_groups": per}


def smoother_weights(smoother: Smoother, lam: np.ndarray):
    """``weights_of(j)``: column ``j`` of the smoother of ``group +
    Σ_g s_g(t)`` at ``lam`` (no covariates), the weights it gives every row
    when smoothing row ``j``. The smoother is block diagonal over the groups,
    so a column is zero outside its row's group."""
    where = np.empty(smoother.n, dtype=int)
    local = np.empty(smoother.n, dtype=int)
    for k, g in enumerate(smoother.groups):
        where[g.rows] = k
        local[g.rows] = np.arange(len(g.rows))

    def weights_of(j: int) -> np.ndarray:
        g = smoother.groups[where[j]]
        i = local[j]
        col = g.Qt @ g.Qt[i]
        if g.m:
            col = col + g.C @ (g.C[i] / (g.s + float(lam[where[j]])))
        out = np.zeros(smoother.n)
        out[g.rows] = col
        return out

    return weights_of


# ────────────────────────────────────────────────────────────────────
# The two steps
# ────────────────────────────────────────────────────────────────────


def exposure_residual(
    smoother: Smoother,
    x: np.ndarray,
    kind: str,
    covariates: np.ndarray | None = None,
) -> dict:
    """Step 1: ``x`` minus its fit on group + Σ_g s_g(t).

    Without ``covariates`` the groups share no parameter, so every group's
    smoothing parameter is chosen on its own rows; with them (the
    ``spatial_plus_step1_covariates`` option) jointly. Returns ``{r, lam,
    edf_groups, values}``.
    """
    x = np.asarray(x, dtype=np.float64)
    G = len(smoother.groups)
    if covariates is not None and np.size(covariates):
        design = Design(smoother, covariates)
        sel = design.select(x, kind)
        # The residual also leaves out the covariates' part, which step 2's
        # unpenalized covariates absorb anyway: β is the same either way.
        return {
            "r": design.residual(x, sel["lam"]),
            "lam": sel["lam"],
            "edf_groups": sel["edf_groups"],
            "values": [sel["value"]] * G,
        }
    r = np.empty_like(x)
    lam = np.ones(G)
    edf = np.zeros(G)
    values: list = []
    for j in range(G):
        sub, rows = smoother.subset(j)
        design = Design(sub, None)
        sel = design.select(x[rows], kind)
        lam[j] = sel["lam"][0]
        edf[j] = sel["edf_groups"][0]
        values.append(sel["value"])
        r[rows] = design.residual(x[rows], sel["lam"])
    return {"r": r, "lam": lam, "edf_groups": edf, "values": values}


def outcome_fit(
    smoother: Smoother,
    y: np.ndarray,
    r_x: np.ndarray,
    covariates: np.ndarray | None,
    kind: str,
) -> dict:
    """Step 2: ``y ~ covariates + β r_x + group + Σ_g s_g(t)``.

    The smoothing parameters are chosen with ``r_x`` in the model. Returns
    ``{beta, se_vp, se_ve, t, df, edf, edf_groups, lam, value, sigma2,
    delta_pen_rss}``: ``se_vp`` from the Bayesian covariance mgcv reports,
    ``se_ve`` from the frequentist one, ``df = n − edf``, and
    ``delta_pen_rss`` the drop in the penalized residual sum of squares when
    ``r_x`` joins the model at those penalties.
    """
    y = np.asarray(y, dtype=np.float64)
    r_x = np.asarray(r_x, dtype=np.float64)
    dense = r_x[:, None] if covariates is None else np.column_stack([covariates, r_x])
    full = Design(smoother, dense)
    sel = full.select(y, kind)
    lam = sel["lam"]
    reduced = Design(smoother, covariates)
    e_r = reduced.residual(r_x, lam)
    rr = reduced.quad(r_x, r_x, lam)
    ry = reduced.quad(r_x, y, lam)
    n = smoother.n
    if rr <= 1e-12 * max(float(r_x @ r_x), 1e-300):
        return {
            "beta": float("nan"),
            "se_vp": float("nan"),
            "se_ve": float("nan"),
            "t": float("nan"),
            "df": float("nan"),
            "edf": sel["edf"],
            "edf_groups": sel["edf_groups"],
            "lam": lam,
            "value": sel["value"],
            "sigma2": float("nan"),
            "delta_pen_rss": 0.0,
        }
    beta = ry / rr
    resid = full.residual(y, lam)
    rss = float(resid @ resid)
    edf = sel["edf"]
    df = n - edf
    sigma2 = rss / df if df > 0 else float("nan")
    se_vp = float(np.sqrt(sigma2 / rr))
    se_ve = float(np.sqrt(sigma2 * float(e_r @ e_r)) / rr)
    return {
        "beta": float(beta),
        "se_vp": se_vp,
        "se_ve": se_ve,
        "t": float(beta / se_vp) if se_vp > 0 else float("nan"),
        "df": float(df),
        "edf": float(edf),
        "edf_groups": sel["edf_groups"],
        "lam": lam,
        "value": sel["value"],
        "sigma2": float(sigma2),
        "delta_pen_rss": float(ry * ry / rr),
    }


# ────────────────────────────────────────────────────────────────────
# Resamples at fixed smoothing parameters
# ────────────────────────────────────────────────────────────────────


def _group_columns(g: Group) -> np.ndarray:
    return np.column_stack([g.Qt, g.C]) if g.m else g.Qt


def _penalty(g: Group, lam: float) -> np.ndarray:
    return np.concatenate([np.zeros(g.Qt.shape[1]), np.full(g.m, float(lam))])


def replicate_slopes(
    smoother: Smoother,
    W: np.ndarray,
    exposures: list[dict],
    outcomes: list[np.ndarray],
    pairs: list[dict],
    covariates: np.ndarray | None = None,
    step1_covariates: bool = False,
) -> np.ndarray:
    """Step-2 slopes of every pair on every replicate, penalties fixed.

    ``W`` is ``(B, n)`` row multiplicities. ``exposures[e]`` holds ``x`` and
    its step-1 ``lam``; ``pairs[j]`` names ``exposure`` (an index),
    ``outcome`` (an index into ``outcomes``) and the step-2 ``lam``. Every
    replicate refits step 1 (each exposure's weighted penalized fit) and
    step 2 by weighted penalized least squares with those penalties. Returns
    ``{slope, delta}``, each ``(B, len(pairs))``: the slope in the exposure's
    units and the drop in the penalized residual sum of squares when the
    exposure joins the model.
    """
    W = np.atleast_2d(np.asarray(W, dtype=np.float64))
    B, n = W.shape
    groups = smoother.groups
    X = [_group_columns(g) for g in groups]
    dense = None
    if covariates is not None and np.size(covariates):
        Z = smoother.residualize_trends(
            np.asarray(covariates, dtype=float).reshape(n, -1)
        )
        dense = resampling.orthonormal(Z, rtol=1e-10)
        if dense.shape[1] == 0:
            dense = None
    out = np.full((B, len(pairs)), np.nan)
    delta = np.full((B, len(pairs)), np.nan)
    ys = [np.asarray(y, dtype=float) for y in outcomes]
    for start in range(0, B, REPLICATE_CHUNK):
        Wc = W[start : start + REPLICATE_CHUNK]
        b = len(Wc)
        grams = []
        for g, Xg in zip(groups, X, strict=True):
            Xw = Xg.T[None, :, :] * Wc[:, None, g.rows]
            grams.append((Xw, Xw @ Xg))

        # ── Step 1 on every replicate ───────────────────────────
        r_reps = []
        for ex in exposures:
            x = np.asarray(ex["x"], dtype=float)
            if step1_covariates and dense is not None:
                r = _weighted_fit_residual(groups, X, grams, Wc, dense, x, ex["lam"])
            else:
                r = np.empty((b, n))
                for j, (g, Xg) in enumerate(zip(groups, X, strict=True)):
                    Xw, Gg = grams[j]
                    A = Gg + np.diag(_penalty(g, ex["lam"][j]))[None]
                    rhs = Xw @ x[g.rows]
                    theta = _solve(A, rhs[..., None])[..., 0]
                    r[:, g.rows] = x[g.rows][None, :] - theta @ Xg.T
            r_reps.append(r)

        # ── Step 2: slope of each pair ──────────────────────────
        for j, pair in enumerate(pairs):
            r = r_reps[pair["exposure"]]
            y = ys[pair["outcome"]]
            R = np.stack([r, np.broadcast_to(y, r.shape)], axis=2)
            M = _weighted_cross(groups, X, grams, Wc, dense, R, pair["lam"])
            with np.errstate(divide="ignore", invalid="ignore"):
                ok = M[:, 0, 0] > 0
                safe = np.where(ok, M[:, 0, 0], 1.0)
                out[start : start + b, j] = np.where(ok, M[:, 0, 1] / safe, np.nan)
                delta[start : start + b, j] = np.where(
                    ok, M[:, 0, 1] ** 2 / safe, np.nan
                )
    return {"slope": out, "delta": delta}


def _weighted_cross(groups, X, grams, Wc, dense, R, lam) -> np.ndarray:
    """Residual cross-products of per-replicate responses ``R`` (``(b, n,
    k)``) on the weighted penalized design (dense + every group's block)."""
    pd_ = 0 if dense is None else dense.shape[1]
    if dense is not None:
        Dw = dense.T[None] * Wc[:, None, :]
        Gdd = Dw @ dense
        Gdr = Dw @ R
    Grr = np.einsum("bik,bi,bil->bkl", R, Wc, R)
    for j, (g, Xg) in enumerate(zip(groups, X, strict=True)):
        Xw, Gg = grams[j]
        A = Gg + np.diag(_penalty(g, lam[j]))[None]
        H = Xw @ R[:, g.rows, :]
        if dense is not None:
            F = Xw @ dense[g.rows]
            S = _solve(A, np.concatenate([F, H], axis=2))
            Sf, Sh = S[..., :pd_], S[..., pd_:]
            Gdd = Gdd - np.swapaxes(F, 1, 2) @ Sf
            Gdr = Gdr - np.swapaxes(F, 1, 2) @ Sh
            Grr = Grr - np.swapaxes(H, 1, 2) @ Sh
        else:
            Sh = _solve(A, H)
            Grr = Grr - np.swapaxes(H, 1, 2) @ Sh
    if dense is not None:
        T = resampling._pinv_factor(Gdd) @ Gdr
        Grr = Grr - np.swapaxes(T, 1, 2) @ T
    return Grr


def _solve(A: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """``A⁻¹ rhs`` for a stack of symmetric matrices; a replicate that drew
    too few rows of a group to fix its trend gets the pseudo-inverse, whose
    undetermined directions touch none of its weighted rows."""
    try:
        return np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        F = resampling._pinv_factor(A)
        return np.swapaxes(F, 1, 2) @ (F @ rhs)


def _weighted_fit_residual(groups, X, grams, Wc, dense, x, lam) -> np.ndarray:
    """Per-replicate residual of ``x`` on the weighted penalized design with
    dense covariates."""
    b = len(Wc)
    pd_ = dense.shape[1]
    Dw = dense.T[None] * Wc[:, None, :]
    Gdd = Dw @ dense
    gdx = Dw @ x
    blocks = []
    for j, (g, Xg) in enumerate(zip(groups, X, strict=True)):
        Xw, Gg = grams[j]
        A = Gg + np.diag(_penalty(g, lam[j]))[None]
        F = Xw @ dense[g.rows]
        h = Xw @ x[g.rows]
        S = _solve(A, np.concatenate([F, h[..., None]], axis=2))
        Gdd = Gdd - np.swapaxes(F, 1, 2) @ S[..., :pd_]
        gdx = gdx - (np.swapaxes(F, 1, 2) @ S[..., pd_:])[..., 0]
        blocks.append(S)
    F_ = resampling._pinv_factor(Gdd)
    gamma = (np.swapaxes(F_, 1, 2) @ (F_ @ gdx[..., None]))[..., 0]
    r = np.repeat(x[None, :], b, axis=0) - gamma @ dense.T
    for j, (g, Xg) in enumerate(zip(groups, X, strict=True)):
        S = blocks[j]
        theta = S[..., pd_] - (S[..., :pd_] @ gamma[..., None])[..., 0]
        r[:, g.rows] -= theta @ Xg.T
    return r


# ────────────────────────────────────────────────────────────────────
# The report of one row set
# ────────────────────────────────────────────────────────────────────


def _t_interval(estimate: float, se: float, df: float) -> tuple[float, float, float]:
    """``(ci_low, ci_high, p)`` of a t statistic with ``df`` degrees of freedom."""
    from scipy import stats

    if not (np.isfinite(estimate) and np.isfinite(se) and se > 0 and df > 0):
        return float("nan"), float("nan"), float("nan")
    q = float(stats.t.ppf(0.975, df))
    p = float(2 * stats.t.sf(abs(estimate / se), df))
    return estimate - q * se, estimate + q * se, p


def _percentile(draws: np.ndarray) -> tuple[float | None, float | None]:
    d = draws[np.isfinite(draws)]
    if d.size < 2:
        return None, None
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def _greater_p(draws: np.ndarray) -> float | None:
    """One-sided bootstrap p of ``> 0``: the share of draws at or below 0."""
    d = draws[np.isfinite(draws)]
    return float(np.mean(d <= 0)) if d.size else None


def _scaled(estimate, lo, hi, p, factor) -> dict | None:
    """A slope and its interval times ``factor`` (``None`` without one)."""
    if factor is None or not np.isfinite(factor):
        return None
    f = float(factor)
    return {
        "estimate": estimate * f,
        "ci_low": lo * f if f >= 0 else hi * f,
        "ci_high": hi * f if f >= 0 else lo * f,
        "p_value": p,
    }


def report(
    xy: np.ndarray,
    exposures: "dict[str, tuple[str, np.ndarray]]",
    target: np.ndarray,
    controls: "dict[str, np.ndarray]",
    covariates: np.ndarray | None,
    labels: np.ndarray,
    *,
    criterion: str = "gcv",
    checks: "list[str] | tuple[str, ...]" = (),
    k: int = DEFAULT_K,
    step1_covariates: bool = False,
    scales: "dict[str, dict] | None" = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Spatial+ on one row set: every exposure on the target and each control.

    ``exposures`` maps a name to ``(kind, values)`` (the first is the CGI);
    ``labels`` are the rows' groups (:func:`group_labels`). For ``criterion``
    and each of ``checks``, every exposure gets its step-1 residual and every
    (exposure, outcome) pair its step-2 fit (:func:`outcome_fit`), reported as

    - ``slopes``: β̂⁺ per exposure unit with its Bayesian (``se_vp``, the
      interval and p) and frequentist (``se_ve``) standard errors; per SD0
      (the exposure's SD after the covariates over the outcome's, both without
      a smooth), per IQR and per p5–p95 range of ``scales`` (``{name: {iqr,
      span}}``); the edf, and ``delta_r2``: the drop in the penalized residual
      sum of squares when the exposure joins, over the outcome's total sum of
      squares;
    - ``d``: ``|β_target| − |β_control|`` per SD0 for every exposure and
      control, with an ``n_boot`` bootstrap interval and one-sided p (> 0);
    - ``delta_d``: ``D_cgi − D_e`` for every other exposure and control;
    - ``paired``: ``delta_r2`` of the CGI minus that of every other exposure,
      on the target;
    - ``groups``: one row per step, exposure, outcome and group (knots, λ,
      edf, criterion value).

    Rows of a check criterion carry ``check = True``. The bootstraps refit
    both steps on every replicate with every λ fixed at its full-sample value.
    """
    xy = np.asarray(xy, dtype=float)
    target = np.asarray(target, dtype=float)
    n = len(target)
    names = list(exposures)
    cgi = names[0]
    outcomes = {"target": target}
    outcomes.update({c: np.asarray(v, dtype=float) for c, v in controls.items()})
    out_names = list(outcomes)
    smoother = build(xy, labels, k=k)
    cov = None if covariates is None or not np.size(covariates) else covariates
    tss = {o: float(((v - v.mean()) ** 2).sum()) for o, v in outcomes.items()}

    # ── SD0: residual SDs after the covariates alone ────────────
    base = resampling.BlockDesign(n, cov)
    raw = np.column_stack(
        [exposures[e][1] for e in names] + [outcomes[o] for o in out_names]
    )
    resid0 = base.residuals(raw)
    sd_x = {e: float(resid0[:, i].std()) for i, e in enumerate(names)}
    sd_y = {o: float(resid0[:, len(names) + i].std()) for i, o in enumerate(out_names)}
    W = sd_rep = tss_rep = None
    if n_boot and int(n_boot) > 0:
        rng = np.random.default_rng(int(seed))
        W = resampling.row_weights(rng.integers(0, n, size=(int(n_boot), n)), n)
        M0 = base.cross(raw, W)
        sd_rep = np.sqrt(np.maximum(np.diagonal(M0, axis1=1, axis2=2), 0.0))
        tss_rep = resampling.weighted_sst(target, W)

    def unit(e, o):
        return sd_x[e] / sd_y[o] if sd_y[o] > 0 else float("nan")

    def unit_rep(e, o):
        xi, yi = names.index(e), len(names) + out_names.index(o)
        with np.errstate(divide="ignore", invalid="ignore"):
            return sd_rep[:, xi] / sd_rep[:, yi]

    warnings: list[str] = []
    result: dict = {
        "criterion": criterion,
        "checks": list(checks),
        "n": n,
        "k": int(k),
        "n_groups": int(len(smoother.groups)),
        "group_sizes": [int(len(g.rows)) for g in smoother.groups],
        "knots": [int(g.knots) for g in smoother.groups],
        "lambda_fixed_in_resamples": True,
        "n_boot": int(n_boot or 0),
        "slopes": [],
        "d": [],
        "delta_d": [],
        "paired": [],
        "groups": [],
        "warnings": warnings,
        "effective_bandwidth": [],
    }
    for crit in [criterion, *[c for c in checks if c != criterion]]:
        is_check = crit != criterion
        tag = {"criterion": crit, "check": is_check}

        # ── Step 1 per exposure ─────────────────────────────────
        step1 = {}
        for e in names:
            step1[e] = exposure_residual(
                smoother,
                np.asarray(exposures[e][1], dtype=float),
                crit,
                cov if step1_covariates else None,
            )
            for j, g in enumerate(smoother.groups):
                result["groups"].append(
                    {
                        **tag,
                        "step": "exposure",
                        "exposure": e,
                        "outcome": None,
                        "group": j,
                        "n": int(len(g.rows)),
                        "knots": int(g.knots),
                        "lambda": float(step1[e]["lam"][j]),
                        "edf": float(step1[e]["edf_groups"][j]) + g.Qt.shape[1] - 1,
                        "criterion_value": step1[e]["values"][j],
                    }
                )

        # ── Step 2 per (exposure, outcome) ──────────────────────
        fits: dict = {}
        for e in names:
            for o in out_names:
                fit = outcome_fit(smoother, outcomes[o], step1[e]["r"], cov, crit)
                fits[(e, o)] = fit
                for j, g in enumerate(smoother.groups):
                    edf_g = float(fit["edf_groups"][j])
                    if g.m and edf_g > EDF_WARN_SHARE * g.m:
                        warnings.append(
                            f"group {j} ({len(g.rows)} rows, {g.knots} knots) uses "
                            f"{edf_g:.0f} of its {g.m} penalized functions for "
                            f"'{o}' on '{e}' ({crit}): the basis may be too small"
                        )
                    result["groups"].append(
                        {
                            **tag,
                            "step": "outcome",
                            "exposure": e,
                            "outcome": o,
                            "group": j,
                            "n": int(len(g.rows)),
                            "knots": int(g.knots),
                            "lambda": float(fit["lam"][j]),
                            "edf": edf_g + g.Qt.shape[1] - 1,
                            "criterion_value": fit["value"],
                        }
                    )
        keys = list(fits)
        if not is_check:
            # The distance the step-2 smoother of the CGI's target model
            # removes variation over (Keller & Szpiro 2020, §3.1).
            from geofuse import spatial_scale_selection as _ss

            k_hat = _ss.effective_bandwidth(
                xy,
                smoother.labels,
                smoother_weights(smoother, fits[(cgi, "target")]["lam"]),
                seed=int(seed),
            )
            result["effective_bandwidth"].append(
                {
                    "row_set": "test",
                    "smooth": "spatial_plus",
                    "criterion": crit,
                    "scale_m": None,
                    "columns": int(smoother.p),
                    **k_hat,
                }
            )
        reps = None
        if W is not None:
            reps = replicate_slopes(
                smoother,
                W,
                [{"x": exposures[e][1], "lam": step1[e]["lam"]} for e in names],
                [outcomes[o] for o in out_names],
                [
                    {
                        "exposure": names.index(e),
                        "outcome": out_names.index(o),
                        "lam": fits[(e, o)]["lam"],
                    }
                    for e, o in keys
                ],
                covariates=cov,
                step1_covariates=step1_covariates,
            )

        # ── Slopes ──────────────────────────────────────────────
        per_sd0_draws = {}
        for j, (e, o) in enumerate(keys):
            fit = fits[(e, o)]
            b = fit["beta"]
            lo, hi, p = _t_interval(b, fit["se_vp"], fit["df"])
            sc = (scales or {}).get(e) or {}
            result["slopes"].append(
                {
                    **tag,
                    "exposure": e,
                    "kind": exposures[e][0],
                    "outcome": o,
                    "role": "target" if o == "target" else "control",
                    "n": n,
                    "estimate": b,
                    "se_vp": fit["se_vp"],
                    "se_ve": fit["se_ve"],
                    "t": fit["t"],
                    "df": fit["df"],
                    "p_value": p,
                    "ci_low": lo,
                    "ci_high": hi,
                    "sd0": sd_x[e],
                    "outcome_sd0": sd_y[o],
                    "edf": fit["edf"],
                    "delta_r2": fit["delta_pen_rss"] / tss[o] if tss[o] > 0 else None,
                    "per_sd0": _scaled(b, lo, hi, p, unit(e, o)),
                    "per_iqr": _scaled(b, lo, hi, p, sc.get("iqr")),
                    "per_p5_p95": _scaled(b, lo, hi, p, sc.get("span")),
                }
            )
            if reps is not None:
                per_sd0_draws[(e, o)] = reps["slope"][:, j] * unit_rep(e, o)

        # ── D, ΔD and the paired ΔR² ────────────────────────────
        def d_point(e, c):
            return abs(fits[(e, "target")]["beta"] * unit(e, "target")) - abs(
                fits[(e, c)]["beta"] * unit(e, c)
            )

        def d_draws(e, c):
            return np.abs(per_sd0_draws[(e, "target")]) - np.abs(per_sd0_draws[(e, c)])

        def interval(draws):
            if draws is None:
                return {"ci_low": None, "ci_high": None, "p_value": None}
            lo, hi = _percentile(draws)
            return {"ci_low": lo, "ci_high": hi, "p_value": _greater_p(draws)}

        for c in controls:
            for e in names:
                result["d"].append(
                    {
                        **tag,
                        "exposure": e,
                        "kind": exposures[e][0],
                        "control": c,
                        "estimate": d_point(e, c),
                        **interval(None if reps is None else d_draws(e, c)),
                    }
                )
            for e in names[1:]:
                draws = None
                if reps is not None:
                    draws = d_draws(cgi, c) - d_draws(e, c)
                result["delta_d"].append(
                    {
                        **tag,
                        "exposure": e,
                        "kind": exposures[e][0],
                        "control": c,
                        "estimate": d_point(cgi, c) - d_point(e, c),
                        **interval(draws),
                    }
                )
        for e in names[1:]:
            a, b_ = fits[(cgi, "target")], fits[(e, "target")]
            draws = None
            if reps is not None:
                ja, jb = keys.index((cgi, "target")), keys.index((e, "target"))
                with np.errstate(divide="ignore", invalid="ignore"):
                    draws = (reps["delta"][:, ja] - reps["delta"][:, jb]) / tss_rep
            result["paired"].append(
                {
                    **tag,
                    "exposure": e,
                    "kind": exposures[e][0],
                    "estimate": (a["delta_pen_rss"] - b_["delta_pen_rss"])
                    / tss["target"],
                    **interval(draws),
                }
            )
    return result


# ────────────────────────────────────────────────────────────────────
# Spatial+ on the search inputs
# ────────────────────────────────────────────────────────────────────


def residual_at(smoother: Smoother, x: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """``x`` minus its fit on group + Σ_g s_g(t) at fixed ``lam`` (no
    covariates; each group alone)."""
    out = np.empty_like(np.asarray(x, dtype=np.float64))
    for j, g in enumerate(smoother.groups):
        v = np.asarray(x, dtype=np.float64)[g.rows]
        v = v - g.Qt @ (g.Qt.T @ v)
        if g.m:
            v = v - g.C @ ((g.C.T @ v) / (g.s + float(lam[j])))
        out[g.rows] = v
    return out


def half_residual(design: Design, lam: np.ndarray):
    """``apply(V)``: ``(I − H)^{1/2} V`` for the hat matrix ``H`` of ``design``
    at ``lam``.

    ``I − H`` is zero on the unpenalized columns (trends and dense
    covariates) and the identity off every column's span, so its square root
    is computed on the span of the penalized functions and the covariates
    only: the functions scaled to unit length (``U_b``, orthonormal, one block
    per group) and the covariates' part orthogonal to them (``E``). Without
    covariates the operator is diagonal there (``λ / (s + λ)`` per function).
    """
    sm = design.sm
    s = design.s
    lam_col = np.asarray(lam, dtype=float)[design.col_group]
    Qz = design.Qz

    def ub_t(V):
        if not sm.p:
            return np.zeros((0, V.shape[1]))
        return sm.cross(V) / np.sqrt(s)[:, None]

    def ub(c):
        if not sm.p:
            return np.zeros((sm.n, c.shape[1]))
        return sm.times(c / np.sqrt(s)[:, None])

    def trends_t(V):
        return [g.Qt.T @ V[g.rows] for g in sm.groups]

    def trends(parts, n_cols):
        out = np.zeros((sm.n, n_cols))
        for g, p in zip(sm.groups, parts, strict=True):
            out[g.rows] = g.Qt @ p
        return out

    if Qz.shape[1] == 0:
        root = np.sqrt(lam_col / (s + lam_col))

        def apply(V):
            V = np.asarray(V, dtype=np.float64)
            one = V.ndim == 1
            V2 = V[:, None] if one else V
            out = V2 - trends(trends_t(V2), V2.shape[1])
            if sm.p:
                c = ub_t(V2)
                out = out - ub(c) + ub(root[:, None] * c)
            return out[:, 0] if one else out

        return apply

    # ── With covariates: the square root on span([U_b, E]) ──────
    from geofuse import resampling

    UbQz = ub_t(Qz)
    E = resampling.orthonormal(Qz - ub(UbQz), rtol=1e-10)
    pb, pe = len(s), E.shape[1]
    # F = [U_b, E]; Fᵀ Qz and Fᵀ B (B = U_b diag(√s)).
    FQz = np.vstack([UbQz, E.T @ Qz])
    FB = np.vstack([np.diag(np.sqrt(s)), np.zeros((pe, pb))]) if pb else np.zeros(
        (pe, 0)
    )
    # Fᵀ (I − H) F = Fᵀ M_0 F − (Fᵀ B̃) A⁻¹ (B̃ᵀ F), with B̃ = M_0 B.
    st = design.state(lam)
    G = FB - FQz @ design.V.T
    M = np.eye(pb + pe) - FQz @ FQz.T
    if pb:
        AG = Design.solve(st, G.T)
        M = M - G @ AG
    M = 0.5 * (M + M.T)
    w, Vm = np.linalg.eigh(M)
    # Eigenvalues at rounding level are the unpenalized directions, exactly
    # zero; their square roots would otherwise come back near 1e-8.
    w = np.where(w > _EIG_RTOL * max(float(w.max(initial=0.0)), 1e-300), w, 0.0)
    root = (Vm * np.sqrt(w)) @ Vm.T

    def apply(V):
        V = np.asarray(V, dtype=np.float64)
        one = V.ndim == 1
        V2 = V[:, None] if one else V
        out = V2 - trends(trends_t(V2), V2.shape[1])
        f = np.vstack([ub_t(V2), E.T @ V2])
        g = root @ f
        out = out - ub(f[:pb]) - E @ f[pb:] + ub(g[:pb]) + E @ g[pb:]
        return out[:, 0] if one else out

    return apply


def search_transform(
    xy: np.ndarray,
    labels: np.ndarray,
    columns: np.ndarray,
    y: np.ndarray,
    covariates: np.ndarray | None,
    *,
    kind: str = "gcv",
    k: int = DEFAULT_K,
    lam_x: np.ndarray | None = None,
    lam_y: np.ndarray | None = None,
) -> dict:
    """Spatial+ step 1 on every search input, and the step-2 transform.

    Every column of ``columns`` (``(n, p)``) is residualized on its own
    penalized smooth per group (:func:`exposure_residual`; ``lam_x``,
    ``(p, groups)``, fixes the penalties instead), the outcome smoother ``H``
    of ``y ~ covariates + group + Σ_g s_g(t)`` is fitted without any exposure
    (``lam_y`` fixes it), and the outcome and every residual are multiplied
    by ``(I − H)^{1/2}`` (:func:`half_residual`). An OLS slope of the
    transformed outcome on any combination of transformed residuals is then
    Spatial+'s step-2 slope of that combination with the outcome smoother's
    penalties fixed. Returns ``{y, X, lam_x, lam_y, edf_x, absorbed, edf_y,
    smoother, design}``.
    """
    columns = np.asarray(columns, dtype=np.float64)
    n, p = columns.shape
    sm = build(xy, labels, k=k)
    G = len(sm.groups)
    R = np.empty_like(columns)
    lams = np.empty((p, G))
    edf = np.empty((p, G))
    for j in range(p):
        if lam_x is not None:
            lams[j] = lam_x[j]
            R[:, j] = residual_at(sm, columns[:, j], lam_x[j])
            edf[j] = np.nan
        else:
            step = exposure_residual(sm, columns[:, j], kind)
            R[:, j], lams[j], edf[j] = step["r"], step["lam"], step["edf_groups"]
    centred = columns - columns.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        absorbed = 1.0 - (R * R).sum(axis=0) / (centred * centred).sum(axis=0)
    design = Design(sm, covariates)
    edf_y = None
    if lam_y is None:
        sel = design.select(np.asarray(y, dtype=np.float64), kind)
        lam_y, edf_y = sel["lam"], sel["edf"]
    else:
        edf_y = design.edf(lam_y)[0]
    W = half_residual(design, lam_y)
    return {
        "y": W(np.asarray(y, dtype=np.float64)),
        "X": W(R),
        "lam_x": lams,
        "lam_y": np.asarray(lam_y, dtype=float),
        "edf_x": edf,
        "absorbed": absorbed,
        "edf_y": float(edf_y),
        "smoother": sm,
        "design": design,
    }
