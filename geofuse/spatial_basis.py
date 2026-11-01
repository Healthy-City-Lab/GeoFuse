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

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as _csgraph_components
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

#: Neighbour count of the k-NN connectivity graph that detects spatial
#: components when no explicit ``eps`` is given. A dense cluster stays whole via
#: neighbour chains; a void wider than the local k-NN reach severs the graph.
DEFAULT_NEIGHBOURS: int = 10

#: Knots per component: at least ``2·df + 1`` and :data:`DEFAULT_KNOTS_MIN`, at
#: most :data:`DEFAULT_KNOTS_MAX` (caps the O(k³) energy-matrix
#: eigendecomposition on dense areas) and the component's size.
DEFAULT_KNOTS_MIN: int = 35
DEFAULT_KNOTS_MAX: int = 150

#: Candidate radial df per component a spatially adjusted run compares, besides
#: "no smooth": roughly log-spaced up to 20.
DEFAULT_DF_CANDIDATES: tuple[int, ...] = (0, 1, 2, 3, 4, 6, 8, 11, 15, 20)

#: Information criteria the df is selected by.
CRITERIA: tuple[str, ...] = ("aic", "bic")

#: A scale-defined basis (``scale_m``) above this many radial columns in all, or
#: in one cluster, logs a warning naming the scale and the largest cluster.
SCALE_WARN_COLUMNS = 2000
SCALE_WARN_BLOCK = 400

#: Smallest cluster that keeps a level of its own when a scale defines the
#: smooth; a smaller one joins its nearest larger cluster's level.
DEFAULT_MIN_COMPONENT = 30

#: A cluster is reported as chained when its extent exceeds this multiple of the
#: entity-weighted median extent, or it holds more than this share of entities
#: while being wider than twice that median.
CHAINED_EXTENT = 5.0
CHAINED_SHARE = 0.25


def _tps_eta(r: np.ndarray) -> np.ndarray:
    """Thin-plate radial kernel ``r² log r`` in 2-D, with ``η(0) = 0``."""
    out = np.zeros_like(r, dtype=np.float64)
    nz = r > 0
    out[nz] = (r[nz] ** 2) * np.log(r[nz])
    return out


def connected_components(
    xy: np.ndarray, eps: float | None = None, *, k: int = DEFAULT_NEIGHBOURS
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


def covering_knots(z: np.ndarray, radius: float) -> np.ndarray:
    """Farthest-point sample of ``z`` until every point is within ``radius`` of a
    knot.

    Starts from the point nearest the centroid, as :func:`_maximin_knots` does,
    and adds the point farthest from the knots chosen so far, so a cluster gets
    as many knots as its occupied area needs at that radius, wherever it lies.
    """
    n = len(z)
    c0 = int(np.argmin(((z - z.mean(axis=0)) ** 2).sum(axis=1)))
    chosen = [c0]
    d = cdist(z, z[[c0]]).ravel()
    while len(chosen) < n and float(d.max()) > radius:
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, cdist(z, z[[i]]).ravel())
    return z[chosen]


def extent_m(xy: np.ndarray) -> float:
    """A cluster's extent: the square root of its convex hull's area (metres),
    or of its bounding box's when the hull is degenerate."""
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 3:
        return 0.0
    try:
        from scipy.spatial import ConvexHull

        return float(np.sqrt(ConvexHull(xy).volume))
    except Exception:
        span = xy.max(axis=0) - xy.min(axis=0)
        return float(np.sqrt(max(span[0] * span[1], 0.0)))


def merge_small_components(
    xy: np.ndarray, labels: np.ndarray, min_size: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Relabel each cluster smaller than ``min_size`` to its nearest larger one.

    The nearest larger cluster is the one holding the entity closest to any of
    the small cluster's entities. Returns ``(labels, merged, n_merged)``: the new
    dense labels, a mask of the entities that were moved, and how many clusters
    were merged. Nothing moves when no cluster reaches ``min_size``.
    """
    from sklearn.neighbors import NearestNeighbors

    labels = np.asarray(labels).copy()
    sizes = np.bincount(labels)
    large = np.flatnonzero(sizes >= int(min_size))
    small = np.flatnonzero(sizes < int(min_size))
    merged = np.zeros(len(labels), dtype=bool)
    if not len(large) or not len(small):
        _, dense = np.unique(labels, return_inverse=True)
        return dense.astype(np.int64), merged, 0
    in_large = np.isin(labels, large)
    nn = NearestNeighbors(n_neighbors=1).fit(xy[in_large])
    target = labels[in_large]
    for c in small:
        rows = np.flatnonzero(labels == c)
        dist, idx = nn.kneighbors(xy[rows])
        labels[rows] = target[int(idx[int(np.argmin(dist[:, 0])), 0])]
        merged[rows] = True
    _, dense = np.unique(labels, return_inverse=True)
    return dense.astype(np.int64), merged, int(len(small))


def _scale_columns(xy_c: np.ndarray, scale_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Linear + radial columns of one cluster at pattern scale ``scale_m``.

    Knots cover the cluster at radius ``scale_m / 2`` (:func:`covering_knots`)
    and every radial function is kept, up to ``n_c − 4`` so the cluster leaves
    room for residuals.
    """
    n_c = len(xy_c)
    empty = np.empty((n_c, 0), dtype=np.float64)
    if n_c < 4:
        return empty, empty
    center = xy_c.mean(axis=0)
    z = xy_c - center
    span = float(np.sqrt((z**2).sum(axis=1)).std())
    if span <= 0:
        return empty, empty
    z = z / span
    linear = z.copy()
    r_max = n_c - 4
    if r_max <= 0:
        return linear, empty
    knots = covering_knots(z, 0.5 * float(scale_m) / span)
    if len(knots) < 2:
        return linear, empty
    w, U = np.linalg.eigh(_tps_eta(cdist(knots, knots)))
    U = U[:, np.argsort(np.abs(w))[::-1]]
    phi = (_tps_eta(cdist(z, knots)) @ U)[:, :r_max]
    std = phi.std(axis=0)
    std[std == 0] = 1.0
    return linear, (phi - phi.mean(axis=0)) / std


def component_report(xy: np.ndarray, labels: np.ndarray, knots: list) -> dict:
    """Per-cluster entities, extent and knots, and whether one cluster chains.

    ``chained`` is a message when the widest cluster's extent exceeds
    :data:`CHAINED_EXTENT` times the entity-weighted median extent, or a
    cluster wider than twice that median holds more than
    :data:`CHAINED_SHARE` of the entities, else ``None``.
    """
    n_comp = int(labels.max()) + 1 if len(labels) else 0
    sizes = np.bincount(labels, minlength=n_comp)
    extents = np.array([extent_m(xy[labels == c]) for c in range(n_comp)])
    out = {
        "components": [
            {
                "n": int(sizes[c]),
                "extent_m": round(float(extents[c]), 1),
                "knots": int(knots[c]) if c < len(knots) else 0,
            }
            for c in range(n_comp)
        ],
        "chained": None,
    }
    if n_comp < 2:
        return out
    order = np.argsort(extents)
    cum = np.cumsum(sizes[order]) / sizes.sum()
    median = float(extents[order][np.searchsorted(cum, 0.5)])
    big = int(np.argmax(sizes))
    wide = int(np.argmax(extents))
    if median > 0 and extents[wide] > CHAINED_EXTENT * median:
        out["chained"] = (
            f"one spatial cluster spans {extents[wide] / 1000:.0f} km against a median "
            f"of {median / 1000:.1f} km"
        )
    elif sizes[big] > CHAINED_SHARE * sizes.sum() and extents[big] > 2 * median:
        out["chained"] = (
            f"one spatial cluster holds {sizes[big] / sizes.sum():.0%} of the entities"
        )
    return out


def _component_columns(
    xy_c: np.ndarray, max_df: int, knot_cap: int, knot_floor: int = DEFAULT_KNOTS_MIN
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

    k = min(n_c, max(2 * int(max_df) + 1, int(knot_floor)), int(knot_cap))
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
    neighbours: int = DEFAULT_NEIGHBOURS,
    knots_min: int = DEFAULT_KNOTS_MIN,
    knots_max: int = DEFAULT_KNOTS_MAX,
    scale_m: float | None = None,
    min_component: int | None = None,
) -> SpatialBasis:
    """Assemble the per-component block-diagonal spatial basis for ``xy``.

    ``xy`` is an ``[n, 2]`` array of entity coordinates in a metre CRS.
    ``max_df`` bounds the radial functions per component; ``eps`` and
    ``neighbours`` are forwarded to :func:`connected_components`. The basis is
    a thin-plate regression spline with ``min(n_c, max(2·max_df + 1,
    knots_min), knots_max)`` knots per component.

    ``scale_m`` defines the smooth by a distance instead: clusters come from a
    gap of ``eps`` (default twice the scale), each cluster's knots cover it at
    radius ``scale_m / 2`` (:func:`covering_knots`) and all its radial
    functions are used, so the smooth removes patterns coarser than about
    ``scale_m`` everywhere; ``max_df`` and the knot bounds do not apply.
    ``min_component`` joins every smaller cluster to its nearest larger one's
    level (:func:`merge_small_components`): its entities share that cluster's
    intercept and get no trend or radial terms of their own.

    Non-finite rows contribute no spatial columns (they fall back to covariate
    control). Returns a :class:`SpatialBasis`; ``has_spatial`` is ``False`` when
    the geometry is degenerate (all coincident / too few points).
    """
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
            "neighbours": int(neighbours),
            "knots_min": int(knots_min),
            "knots_max": int(knots_max),
        },
    )
    if n < 4 or finite.sum() < 4:
        return base

    labels = np.full(n, -1, dtype=np.int64)
    if scale_m is not None and eps is None:
        eps = 2.0 * float(scale_m)
    lab_f, eps_used = connected_components(xy[finite], eps=eps, k=neighbours)
    moved_f = np.zeros(len(lab_f), dtype=bool)
    n_merged = 0
    if min_component:
        lab_f, moved_f, n_merged = merge_small_components(
            xy[finite], lab_f, int(min_component)
        )
    labels[finite] = lab_f
    moved = np.zeros(n, dtype=bool)
    moved[finite] = moved_f

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
        members = np.where(labels == c)[0]
        idx = members[~moved[members]]
        if c > 0:
            ind = np.zeros(n, dtype=np.float64)
            ind[members] = 1.0
            intercept_cols.append(ind)
        if scale_m is not None:
            lin_c, phi_c = _scale_columns(xy[idx], float(scale_m))
        else:
            lin_c, phi_c = _component_columns(
                xy[idx], max_df=max_df, knot_cap=knots_max, knot_floor=knots_min
            )
        knots_per_component.append(int(phi_c.shape[1]))
        for j in range(lin_c.shape[1]):
            col = np.zeros(n, dtype=np.float64)
            col[idx] = lin_c[:, j]
            linear_cols.append(col)
        for r in range(phi_c.shape[1]):
            col = np.zeros(n, dtype=np.float64)
            col[idx] = phi_c[:, r]
            radial_cols.append(col)
            # Scale mode keeps every function, so all share the first level.
            radial_rank.append(0 if scale_m is not None else r)

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
        "neighbours": int(neighbours),
        "knots_min": int(knots_min),
        "knots_max": int(knots_max),
        **component_report(
            xy[finite][~moved_f], lab_f[~moved_f], knots_per_component
        ),
    }
    if min_component:
        summary.update(
            min_component=int(min_component),
            merged_components=int(n_merged),
            merged_entities=int(moved.sum()),
        )
    if scale_m is not None:
        summary.update(scale_m=float(scale_m), radial_columns=len(radial_cols))
        biggest = max(knots_per_component, default=0)
        if len(radial_cols) > SCALE_WARN_COLUMNS or biggest > SCALE_WARN_BLOCK:
            summary["size_warning"] = (
                f"the smooth at {float(scale_m) / 1000:g} km has {len(radial_cols)} "
                f"radial functions, {biggest} in its largest cluster; a larger scale "
                "or a smaller cluster gap (spatial_adjust_eps_m) keeps it smaller"
            )
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


def candidate_dfs(
    basis: SpatialBasis, candidates: Sequence[int] | None = None
) -> list[int]:
    """The df levels a selection compares on ``basis``, ascending.

    ``None`` means every level ``0 … basis.max_df``. A listed df above what the
    basis holds is capped to ``basis.max_df`` (a component smaller than a df is
    capped per component by :meth:`SpatialBasis.design` as well), and
    duplicates are dropped, so the levels compared are distinct designs.
    """
    top = basis.max_df
    if candidates is None:
        return list(range(0, top + 1))
    return sorted({min(int(d), top) for d in candidates if int(d) >= 0})


def _check_criterion(criterion: str) -> None:
    if criterion not in CRITERIA:
        raise ValueError(
            f"criterion must be one of {list(CRITERIA)}, not {criterion!r}."
        )


def select_df_aic(
    y: np.ndarray,
    basis: SpatialBasis,
    covariates: np.ndarray | None = None,
    *,
    criterion: str = "aic",
    candidates: Sequence[int] | None = None,
    return_profile: bool = False,
):
    """Choose the spatial df by information criterion (Keller–Szpiro rule).

    Fits ``y ~ spatial(df) + covariates`` for each candidate df
    (:func:`candidate_dfs`) and the no-spatial baseline ``y ~ covariates``,
    returning the lowest-criterion choice as ``(df, columns)``; on a tie the
    smaller design wins. ``df is None`` (``columns is None``) means the
    criterion preferred no spatial adjustment. For KS-AIC pass the *outcome* as
    ``y``; for df-Spatial+ pass the *exposure*.

    With ``return_profile`` a third value holds the criterion of every design
    compared, ``{"none": ic, "<df>": ic, …}`` in the order compared (empty when
    nothing was compared).

    Non-finite rows of ``y`` / ``covariates`` are dropped for the criterion fits
    only; the returned ``columns`` stay full length (and finite by construction),
    aligned with the original rows.
    """
    _check_criterion(criterion)
    profile: dict[str, float] = {}
    if not basis.has_spatial:
        return (None, None, profile) if return_profile else (None, None)

    y = np.asarray(y, dtype=np.float64).ravel()
    fin = np.isfinite(y)
    if covariates is not None:
        fin &= np.isfinite(covariates).all(axis=1)
    if fin.sum() < 4:
        return (None, None, profile) if return_profile else (None, None)
    yf = y[fin]
    covf = None if covariates is None else covariates[fin]

    best_df: int | None = None
    best_cols: np.ndarray | None = None
    best_ic = _ols_info_criterion(yf, covf, criterion)  # baseline: no spatial
    profile["none"] = best_ic

    for df in candidate_dfs(basis, candidates):
        spatial = basis.design(df)
        if spatial.shape[1] == 0:
            continue
        sf = spatial[fin]
        X = sf if covf is None else np.column_stack([covf, sf])
        ic = _ols_info_criterion(yf, X, criterion)
        profile[str(df)] = ic
        if ic < best_ic:
            best_ic = ic
            best_df = df
            best_cols = spatial

    if return_profile:
        return best_df, best_cols, profile
    return best_df, best_cols


@dataclass
class DfSelectionPrecompute:
    """Precomputed nested-QR structures for repeated df selection on one basis.

    The candidate designs are nested in df (linear columns first, then radial
    columns in resolution order), so one economy QR of the full-df design
    ``[1, linear, radial(rank-ordered)]`` yields every candidate's RSS as a
    prefix of ``‖Qᵀy‖²`` — the per-call df search becomes a single matvec
    instead of ``max_df + 1`` least-squares fits.

    ``k_of_df[df]`` is the column count in ``Q`` (intercept included) at that
    df level; ``designs[df]`` is the spatial design (no intercept) returned to
    the caller — same column *span* as ``SpatialBasis.design(df)``.
    """

    Q: np.ndarray
    k_of_df: np.ndarray
    designs: list


def precompute_df_selection(basis: SpatialBasis) -> DfSelectionPrecompute | None:
    """Build the nested-QR precompute for ``select_df_aic_fast``, or ``None``."""
    if not basis.has_spatial:
        return None
    order = np.argsort(basis.radial_rank, kind="stable")
    radial_sorted = basis.radial[:, order]
    sorted_ranks = basis.radial_rank[order]
    n_linear = basis.linear.shape[1]

    M = np.column_stack([np.ones(basis.n), basis.linear, radial_sorted])
    Q, _r = np.linalg.qr(M, mode="reduced")

    k_of_df = np.empty(basis.max_df + 1, dtype=np.int64)
    designs: list = []
    for df in range(basis.max_df + 1):
        n_radial = int(np.searchsorted(sorted_ranks, df, side="left"))
        k_of_df[df] = 1 + n_linear + n_radial
        parts = []
        if n_linear:
            parts.append(basis.linear)
        if n_radial:
            parts.append(radial_sorted[:, :n_radial])
        designs.append(np.column_stack(parts) if parts else None)
    return DfSelectionPrecompute(Q=Q, k_of_df=k_of_df, designs=designs)


def select_df_aic_fast(
    y: np.ndarray,
    basis: SpatialBasis,
    pre: DfSelectionPrecompute | None,
    *,
    criterion: str = "aic",
    candidates: Sequence[int] | None = None,
    return_profile: bool = False,
):
    """Covariate-free :func:`select_df_aic` via the nested-QR precompute.

    Matches ``select_df_aic(y, basis, None)`` (same RSS, hence the same
    information criterion and the same selected df) at one ``Qᵀy`` matvec per
    call. Returned columns span the same space as ``basis.design(df)`` — the
    downstream residualization is order-invariant. Falls back to the exact
    legacy search when the precompute is missing or ``y`` has non-finite rows
    (the legacy path drops those rows for the criterion fits).
    ``candidates`` and ``return_profile`` are as in :func:`select_df_aic`.
    """
    _check_criterion(criterion)
    y = np.asarray(y, dtype=np.float64).ravel()
    if pre is None or not np.isfinite(y).all():
        return select_df_aic(
            y,
            basis,
            None,
            criterion=criterion,
            candidates=candidates,
            return_profile=return_profile,
        )
    n = len(y)
    if n < 4:
        return (None, None, {}) if return_profile else (None, None)

    yy = float(y @ y)
    proj = pre.Q.T @ y
    cum = np.cumsum(proj * proj)

    def _ic(rss: float, k_cols: int) -> float:
        if rss <= 0:
            rss = 1e-300
        k = k_cols + 1  # params + variance, matching _ols_info_criterion
        loglik = -0.5 * n * (np.log(2 * np.pi) + np.log(rss / n) + 1.0)
        if criterion == "bic":
            return float(np.log(n) * k - 2.0 * loglik)
        return float(2.0 * k - 2.0 * loglik)

    best_df: int | None = None
    best_cols: np.ndarray | None = None
    best_ic = _ic(yy - float(cum[0]), 1)  # baseline: intercept only
    profile: dict[str, float] = {"none": best_ic}

    for df in candidate_dfs(basis, candidates):
        k_cols = int(pre.k_of_df[df])
        if k_cols <= 1:  # no spatial columns at this level
            continue
        ic = _ic(yy - float(cum[k_cols - 1]), k_cols)
        profile[str(df)] = ic
        if ic < best_ic:
            best_ic = ic
            best_df = df
            best_cols = pre.designs[df]

    if return_profile:
        return best_df, best_cols, profile
    return best_df, best_cols
