"""Low-rank spatial basis for spatial-confounding adjustment of the objective.

The fusion objective controls for measured covariates by OLS partialling. That
removes only the confounding carried by the covariates the user listed; any
*unmeasured*, spatially-structured driver that co-varies with greenery and the
outcome still leaks into the score. This module builds a flexible smooth of the
entity coordinates that absorbs that unmeasured spatial structure, so the partial
association reflects only the finer-scale greenery↔outcome co-variation a smooth
confounder cannot explain.

Two estimators consume the basis (see :mod:`geofuse.objective_scoring`):

* **KS-AIC** (Keller & Szpiro, 2020) — add the smooth to the outcome model and
  pick its degrees of freedom by AIC on the outcome model *without* the exposure.
* **df-Spatial+** (the unpenalized, AIC-df variant of Dupont–Wood–Augustin used by
  Rainey et al., 2025) — residualize the exposure on the smooth, df by AIC on the
  exposure model.

Construction
------------

The smooth is a low-rank **thin-plate regression spline** (Wood, 2003): radial
terms ``η(r) = r² log r`` against a space-filling subset of the *observed* points
plus the linear null space ``[x, y]``. Radial columns are ordered by resolution
(coarsest first) via the eigendecomposition of the knot energy matrix, so taking
the first ``df`` of them yields a df-``df`` smooth.

For study areas that are clusters of entities separated by large empty voids the
basis is assembled **block-diagonally per connected component** (DBSCAN). Each
disconnected cluster gets its own local trend; no basis function spans a void and
no knot is ever placed in empty space. The basis is only evaluated at entity
coordinates — the void interior is never predicted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as _csgraph_components
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Default neighbour count for the k-NN connectivity graph used to detect spatial
# components when no explicit ``eps`` is given. A dense cluster stays whole via
# neighbour chains; a void wider than the local k-NN reach severs the graph.
_DEFAULT_KNN: int = 10


def _tps_eta(r: np.ndarray) -> np.ndarray:
    """Thin-plate radial kernel ``r² log r`` in 2-D, with ``η(0) = 0``."""
    out = np.zeros_like(r, dtype=np.float64)
    nz = r > 0
    out[nz] = (r[nz] ** 2) * np.log(r[nz])
    return out


def connected_components(
    xy: np.ndarray, eps: float | None = None, *, k: int = _DEFAULT_KNN
) -> tuple[np.ndarray, float]:
    """Label points by connected spatial component.

    By default uses connectivity of the symmetric k-nearest-neighbour graph: a
    dense cluster is connected through neighbour chains, while a void wider than
    the local k-NN reach severs the graph into separate components. This is
    robust to internally-spread clusters (unlike a fixed radius) and never
    produces singletons for ``n > 1``.

    When ``eps`` (metres) is given, falls back to DBSCAN with that radius
    (``min_samples=1``) so a caller who knows the void scale can set it directly.

    Returns ``(labels, scale)`` with dense labels ``0 .. n_components-1``;
    ``scale`` is the DBSCAN ``eps`` used, or the median nearest-neighbour
    distance (a descriptive connection scale) in the k-NN path.
    """
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    if n == 0:
        return np.empty(0, dtype=np.int64), 0.0
    if n == 1:
        return np.zeros(1, dtype=np.int64), 0.0

    if eps is not None and float(eps) > 0:
        labels = DBSCAN(eps=float(eps), min_samples=1).fit_predict(xy)
        _uniq, dense = np.unique(labels, return_inverse=True)
        return dense.astype(np.int64), float(eps)

    kk = min(n - 1, max(1, int(k)))
    nn = NearestNeighbors(n_neighbors=kk + 1).fit(xy)
    dists, idx = nn.kneighbors(xy)
    rows = np.repeat(np.arange(n), kk)
    cols = idx[:, 1:].ravel()
    graph = csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(n, n))
    graph = graph.maximum(
        graph.T
    )  # symmetric: connect if either is the other's neighbour
    _ncomp, labels = _csgraph_components(graph, directed=False)
    scale = float(np.median(dists[:, 1])) if n > 1 else 0.0
    return labels.astype(np.int64), scale


def _maximin_knots(z: np.ndarray, k: int) -> np.ndarray:
    """Space-filling (farthest-point) subset of ``k`` rows of ``z``.

    Returns actual observed points, so knots never fall in empty space. Starts
    from the point nearest the centroid and greedily adds the point maximizing
    the minimum distance to those already chosen.
    """
    n = len(z)
    if k >= n:
        return z.copy()
    c0 = int(np.argmin(((z - z.mean(axis=0)) ** 2).sum(axis=1)))
    chosen = [c0]
    d = cdist(z, z[[c0]]).ravel()
    for _ in range(k - 1):
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, cdist(z, z[[i]]).ravel())
    return z[chosen]


def _component_columns(
    xy_c: np.ndarray, max_df: int, knot_cap: int
) -> tuple[np.ndarray, np.ndarray]:
    """Linear + resolution-ordered radial columns for one component.

    Returns ``(linear, radial)`` where ``linear`` is ``[n_c, 0|2]`` (the
    thin-plate null space ``x, y``) and ``radial`` is ``[n_c, R]`` with columns
    ordered coarsest-first. Degrees of freedom are capped to leave residual room
    (total spatial params ≤ ``n_c − 2``); tiny components contribute nothing.
    """
    n_c = len(xy_c)
    empty = np.empty((n_c, 0), dtype=np.float64)
    if n_c < 4:
        return empty, empty

    # Center and scale so the radial kernel is well-conditioned (thin-plate is
    # invariant to this up to the linear null space).
    center = xy_c.mean(axis=0)
    z = xy_c - center
    span = float(np.sqrt((z**2).sum(axis=1)).std())
    if span <= 0:
        # Coincident points carry no within-component spatial information.
        return empty, empty
    z = z / span

    linear = z.copy()  # [n_c, 2]

    r_max = min(int(max_df), n_c - 4)
    if r_max <= 0:
        return linear, empty

    k = min(n_c, max(2 * int(max_df) + 1, 35), int(knot_cap))
    knots = _maximin_knots(z, k)
    energy = _tps_eta(cdist(knots, knots))
    # Eigenvectors ordered by descending |eigenvalue| span coarse→fine spatial
    # patterns; project the data-to-knot kernel onto them.
    w, U = np.linalg.eigh(energy)
    order = np.argsort(np.abs(w))[::-1]
    U = U[:, order]
    phi = _tps_eta(cdist(z, knots)) @ U  # [n_c, k]
    phi = phi[:, : min(r_max, phi.shape[1])]

    # Standardize radial columns for downstream lstsq conditioning (column
    # scaling is absorbed by the coefficients, so the column space is unchanged).
    std = phi.std(axis=0)
    std[std == 0] = 1.0
    phi = (phi - phi.mean(axis=0)) / std
    return linear, phi


@dataclass
class SpatialBasis:
    """Block-diagonal candidate spatial basis with a resolution-indexed df ladder.

    ``linear`` (``[n, *]``) holds the structural columns always included when
    spatial adjustment is on: per-component intercepts (cluster fixed effects,
    reference component dropped) plus the per-component null-space ``x, y``.
    ``radial`` (``[n, *]``) holds the resolution-ordered radial columns;
    ``radial_rank[j]`` is the within-component rank of radial column ``j``
    (0 = coarsest). :meth:`design` selects the first ``df`` radial functions per
    component on top of the structural columns.
    """

    linear: np.ndarray
    radial: np.ndarray
    radial_rank: np.ndarray
    n: int
    summary: dict = field(default_factory=dict)

    @property
    def max_df(self) -> int:
        """Largest selectable radial-df level (radial functions per component)."""
        if self.radial.shape[1] == 0:
            return 0
        return int(self.radial_rank.max()) + 1

    @property
    def has_spatial(self) -> bool:
        """True when any spatial column (linear or radial) exists."""
        return self.linear.shape[1] > 0 or self.radial.shape[1] > 0

    def design(self, df: int) -> np.ndarray:
        """Spatial design at radial-df ``df`` (``0`` → linear-only). Shape ``[n, *]``."""
        parts = []
        if self.linear.shape[1] > 0:
            parts.append(self.linear)
        if df > 0 and self.radial.shape[1] > 0:
            parts.append(self.radial[:, self.radial_rank < df])
        if not parts:
            return np.empty((self.n, 0), dtype=np.float64)
        return np.column_stack(parts)


def build_block_basis(
    xy: np.ndarray,
    *,
    max_df: int = 10,
    eps: float | None = None,
    knn: int = _DEFAULT_KNN,
    knot_cap: int = 150,
    family: str = "tprs",
) -> SpatialBasis:
    """Assemble the per-component block-diagonal spatial basis for ``xy``.

    ``xy`` is an ``[n, 2]`` array of entity coordinates in a metre CRS. ``max_df``
    bounds the radial functions per component. ``family`` selects the basis
    family; only ``"tprs"`` (thin-plate regression spline) is implemented.

    Non-finite rows contribute no spatial columns (they fall back to covariate
    control). Returns a :class:`SpatialBasis`; ``has_spatial`` is ``False`` when
    the geometry is degenerate (all coincident / too few points).
    """
    if family != "tprs":
        raise ValueError(f"Unknown spatial basis family '{family}'.")
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must be [n, 2]; got shape {xy.shape}.")
    n = len(xy)

    finite = np.isfinite(xy).all(axis=1)
    empty = np.empty((n, 0), dtype=np.float64)
    base = SpatialBasis(
        linear=empty,
        radial=empty,
        radial_rank=np.empty(0, dtype=np.int64),
        n=n,
        summary={
            "n_components": 0,
            "eps_used": 0.0,
            "max_df_avail": 0,
            "knots_per_component": [],
        },
    )
    if n < 4 or finite.sum() < 4:
        return base

    labels = np.full(n, -1, dtype=np.int64)
    lab_f, eps_used = connected_components(xy[finite], eps=eps, k=knn)
    labels[finite] = lab_f

    n_comp = int(lab_f.max()) + 1 if len(lab_f) else 0

    # Per-component intercepts (cluster fixed effects) absorb between-component
    # level differences. Across a large void a smooth confounder is not separable
    # from the CGI effect, so between-cluster contrasts are removed and the CGI
    # effect is identified from within-cluster variation only — the coarsest
    # spatial term, always present. The first component is the reference (its
    # level rides on the global intercept) to stay full rank.
    intercept_cols: list[np.ndarray] = []
    linear_cols: list[np.ndarray] = []
    radial_cols: list[np.ndarray] = []
    radial_rank: list[int] = []
    knots_per_component: list[int] = []

    for c in range(n_comp):
        idx = np.where(labels == c)[0]
        if c > 0:
            ind = np.zeros(n, dtype=np.float64)
            ind[idx] = 1.0
            intercept_cols.append(ind)
        lin_c, phi_c = _component_columns(xy[idx], max_df=max_df, knot_cap=knot_cap)
        knots_per_component.append(int(phi_c.shape[1]))
        for j in range(lin_c.shape[1]):
            col = np.zeros(n, dtype=np.float64)
            col[idx] = lin_c[:, j]
            linear_cols.append(col)
        for r in range(phi_c.shape[1]):
            col = np.zeros(n, dtype=np.float64)
            col[idx] = phi_c[:, r]
            radial_cols.append(col)
            radial_rank.append(r)

    linear = (
        np.column_stack(intercept_cols + linear_cols)
        if (intercept_cols or linear_cols)
        else empty
    )
    radial = np.column_stack(radial_cols) if radial_cols else empty
    summary = {
        "n_components": n_comp,
        "eps_used": float(eps_used),
        "knots_per_component": knots_per_component,
        "max_df_avail": (int(max(radial_rank)) + 1) if radial_rank else 0,
        "n_singletons": int(np.sum(np.bincount(lab_f) == 1)) if len(lab_f) else 0,
    }
    return SpatialBasis(
        linear=linear,
        radial=radial,
        radial_rank=np.asarray(radial_rank, dtype=np.int64),
        n=n,
        summary=summary,
    )


def _ols_info_criterion(y: np.ndarray, X: np.ndarray | None, criterion: str) -> float:
    """Gaussian AIC/BIC of ``OLS(y ~ X)`` (intercept added).

    Uses the MLE log-likelihood ``ℓ = −n/2 (ln 2π + ln(RSS/n) + 1)`` and counts
    ``p`` slope/intercept params plus the variance. Lower is better.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(y)
    if X is None or (hasattr(X, "shape") and X.shape[1] == 0):
        Xc = np.ones((n, 1))
    else:
        Xc = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    resid = y - Xc @ beta
    rss = float(resid @ resid)
    if rss <= 0:
        rss = 1e-300  # perfect fit → finite, very negative criterion
    k = Xc.shape[1] + 1  # params + variance
    loglik = -0.5 * n * (np.log(2 * np.pi) + np.log(rss / n) + 1.0)
    if criterion == "bic":
        return float(np.log(n) * k - 2.0 * loglik)
    return float(2.0 * k - 2.0 * loglik)


def select_df_aic(
    y: np.ndarray,
    basis: SpatialBasis,
    covariates: np.ndarray | None = None,
    *,
    criterion: str = "aic",
) -> tuple[int | None, np.ndarray | None]:
    """Choose the spatial df by information criterion (Keller–Szpiro rule).

    Fits ``y ~ spatial(df) + covariates`` for each candidate df and the
    no-spatial baseline ``y ~ covariates``, returning the lowest-AIC choice as
    ``(df, columns)``. ``df is None`` (``columns is None``) means the criterion
    preferred no spatial adjustment. For KS-AIC pass the *outcome* as ``y``; for
    df-Spatial+ pass the *exposure*.

    Non-finite rows of ``y`` / ``covariates`` are dropped for the criterion fits
    only; the returned ``columns`` stay full length (and finite by construction),
    aligned with the original rows.
    """
    if not basis.has_spatial:
        return None, None

    y = np.asarray(y, dtype=np.float64).ravel()
    fin = np.isfinite(y)
    if covariates is not None:
        fin &= np.isfinite(covariates).all(axis=1)
    if fin.sum() < 4:
        return None, None
    yf = y[fin]
    covf = None if covariates is None else covariates[fin]

    best_df: int | None = None
    best_cols: np.ndarray | None = None
    best_ic = _ols_info_criterion(yf, covf, criterion)  # baseline: no spatial

    for df in range(0, basis.max_df + 1):
        spatial = basis.design(df)
        if spatial.shape[1] == 0:
            continue
        sf = spatial[fin]
        X = sf if covf is None else np.column_stack([covf, sf])
        ic = _ols_info_criterion(yf, X, criterion)
        if ic < best_ic:
            best_ic = ic
            best_df = df
            best_cols = spatial

    return best_df, best_cols
