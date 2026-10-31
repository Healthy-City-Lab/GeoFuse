"""Exact resampled least squares over a block-diagonal spatial smooth.

Bootstrap intervals, paired differences, negative-control slopes and the
specificity contrast all refit one least-squares problem many times: responses
(the outcome, one or more exposures, controls) on ``[1, covariates, smooth]``,
over resampled rows. A resample that draws row ``i`` ``k`` times is weighted
least squares with weight ``k``, so everything those statistics read comes
from the weighted cross-products of the design, and a replicate never needs a
pass of ``lstsq`` over its rows.

Two facts make that cheap:

- **The smooth is block diagonal.** :func:`geofuse.spatial_basis.build_block_basis`
  gives every spatial cluster its own columns, zero outside its rows, so the
  smooth's weighted Gram is one small matrix per cluster. Each block is
  accumulated from its own rows and eliminated on its own (a Schur complement),
  and the dense ``p × p`` system of covariates plus smooth is never formed. The
  blocks are read off the smooth's non-zero pattern (:func:`smooth_blocks`).
- **One GEMM serves a chunk of replicates.** The per-row products of the
  columns are formed once per slab of rows; a matrix of replicate weights times
  those products gives every replicate's cross-products at once.

The covariates and each block are orthonormalised once on the full sample
(:func:`orthonormal`). A replicate selects rows of the same columns, so its
column space and every projection are unchanged, and its cross-product
matrices stay well conditioned. Directions a replicate does not support (a
cluster it drew no rows from) have eigenvalues at rounding level and are
dropped by the pseudo-inverse's relative floor :data:`PINV_RTOL`.

Measured against the per-replicate ``lstsq`` / ``pinv`` fits it replaces, the
largest difference in any score, slope or interval end is ~1e-11 (see
``docs/FEATURES.md``, "Numerical equivalence of the fast paths").
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

#: Relative eigenvalue floor of a weighted cross-product matrix: directions
#: below it carry no rows of the replicate and are dropped, as a
#: pseudo-inverse drops them.
PINV_RTOL = 1e-10

#: Replicates whose weights are multiplied in one GEMM.
CHUNK = 256

#: Rows whose per-row column products are formed at once.
ROW_SLAB = 4096

#: An exposure whose residual variance is below this share of its total is
#: inside the controls' span and scores zero.
_COLLINEAR = 1e-13


# ────────────────────────────────────────────────────────────────────
# Structure of the design
# ────────────────────────────────────────────────────────────────────


def smooth_blocks(S: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """The ``(rows, columns)`` blocks of a block-diagonal matrix.

    Two columns share a block when they are non-zero on a common row; a block's
    rows are those where any of its columns is non-zero. Rows no column touches
    belong to no block. Only the disjointness of the blocks' column supports
    matters for the arithmetic, so a row that happens to be zero in every
    column of its cluster may sit outside the block without changing a result.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    S = np.asarray(S)
    n, m = S.shape
    if m == 0:
        return []
    r, c = np.nonzero(S)
    graph = coo_matrix(
        (np.ones(len(r), dtype=np.int8), (r, n + c)), shape=(n + m, n + m)
    )
    _, labels = connected_components(graph, directed=False)
    row_lab, col_lab = labels[:n], labels[n:]
    out = []
    order = np.argsort(col_lab, kind="stable")
    bounds = np.flatnonzero(np.diff(col_lab[order])) + 1
    for cols in np.split(order, bounds):
        rows = np.flatnonzero(row_lab == col_lab[cols[0]])
        out.append((rows, np.sort(cols)))
    return out


def orthonormal(A: np.ndarray, rtol: float | None = None) -> np.ndarray:
    """Orthonormal basis of the column space of ``A`` (SVD).

    Keeps singular values above ``rtol`` times the largest; ``None`` uses the
    rule of :func:`numpy.linalg.lstsq` (``eps · max(shape)``), so a projection
    onto the basis is the one ``lstsq`` fits.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim == 1:
        A = A[:, None]
    if A.size == 0:
        return np.zeros((len(A), 0))
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    if not s.size or s[0] <= 0:
        return U[:, :0]
    tol = np.finfo(np.float64).eps * max(A.shape) if rtol is None else rtol
    return np.ascontiguousarray(U[:, s > tol * s[0]])


def _triu(k: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(k)


def _unpack(flat: np.ndarray, k: int, iu) -> np.ndarray:
    """``(B, k(k+1)/2)`` upper triangles as ``(B, k, k)`` symmetric matrices."""
    out = np.empty((flat.shape[0], k, k))
    out[:, iu[0], iu[1]] = flat
    out[:, iu[1], iu[0]] = flat
    return out


def _pinv_factor(G: np.ndarray) -> np.ndarray:
    """``F`` with ``Fᵀ F = pinv(G)`` for a stack of symmetric PSD ``G``.

    Eigenvalues under :data:`PINV_RTOL` times each matrix's largest get a zero
    row, so a direction the replicate does not support adds nothing.
    """
    lam, V = np.linalg.eigh(G)
    top = lam.max(axis=-1, keepdims=True)
    keep = lam > PINV_RTOL * np.maximum(top, np.finfo(np.float64).tiny)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(keep, 1.0 / np.sqrt(np.where(keep, lam, 1.0)), 0.0)
    return np.swapaxes(V, -1, -2) * scale[..., :, None]


# ────────────────────────────────────────────────────────────────────
# The prepared design
# ────────────────────────────────────────────────────────────────────


class BlockDesign:
    """``[1, dense, smooth]`` prepared once for weighted fits of many responses.

    ``dense`` is ``(n, p)`` or ``None``; ``smooth`` is ``(n, m)`` block diagonal
    or ``None``. The intercept is always in the span. ``rtol`` is the rank
    rule of :func:`orthonormal` for the dense part and for every block.
    """

    def __init__(
        self,
        n: int,
        dense: np.ndarray | None = None,
        smooth: np.ndarray | None = None,
        *,
        rtol: float | None = None,
    ):
        self.n = int(n)
        parts = [np.ones((self.n, 1))]
        if dense is not None and np.size(dense):
            parts.append(np.asarray(dense, dtype=np.float64).reshape(self.n, -1))
        self.Qd = orthonormal(np.column_stack(parts), rtol)

        order, self.blocks, pos = [], [], 0
        in_block = np.zeros(self.n, dtype=bool)
        if smooth is not None and np.size(smooth):
            S = np.asarray(smooth, dtype=np.float64).reshape(self.n, -1)
            for rows, cols in smooth_blocks(S):
                Qb = orthonormal(S[np.ix_(rows, cols)], rtol)
                if Qb.shape[1] == 0:
                    continue
                order.append(rows)
                in_block[rows] = True
                self.blocks.append((pos, pos + len(rows), Qb))
                pos += len(rows)
        order.append(np.flatnonzero(~in_block))
        self.order = np.concatenate(order)
        self.Qd_sorted = np.ascontiguousarray(self.Qd[self.order])

    @property
    def rank(self) -> int:
        """Columns of the prepared design (its rank on the full sample)."""
        return int(self.Qd.shape[1] + sum(Q.shape[1] for *_, Q in self.blocks))

    # ── Cross-products ──────────────────────────────────────────

    def cross(self, R: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Residual cross-products of the responses for each weight vector.

        ``R`` is ``(n, k)`` (or ``(n,)``); ``W`` is ``(B, n)`` non-negative row
        weights (bootstrap multiplicities). Returns ``(B, k, k)``: ``M[b] =
        Rᵀ W_b R − Rᵀ W_b Z (Zᵀ W_b Z)⁺ Zᵀ W_b R`` with ``Z`` the design — the
        weighted cross-products of the responses' residuals on the design.
        """
        R = np.asarray(R, dtype=np.float64)
        if R.ndim == 1:
            R = R[:, None]
        W = np.atleast_2d(np.asarray(W, dtype=np.float64))
        mean = R.mean(axis=0)
        sd = R.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        Rs = ((R - mean) / sd)[self.order]
        Wp = W[:, self.order]
        a = np.ascontiguousarray(np.column_stack([self.Qd_sorted, Rs]))
        K = a.shape[1]
        iu = _triu(K)
        flat = np.zeros((W.shape[0], len(iu[0])))
        for s in range(0, self.n, ROW_SLAB):
            e = min(s + ROW_SLAB, self.n)
            flat += Wp[:, s:e] @ (a[s:e, iu[0]] * a[s:e, iu[1]])
        G = _unpack(flat, K, iu)

        # ── Eliminate the smooth, block by block ────────────────
        for start, stop, Qb in self.blocks:
            r = Qb.shape[1]
            ib = _triu(r)
            gbb = np.zeros((W.shape[0], len(ib[0])))
            C = np.zeros((W.shape[0], r * K))
            for s in range(start, stop, ROW_SLAB):
                e = min(s + ROW_SLAB, stop)
                q = Qb[s - start : e - start]
                w = Wp[:, s:e]
                gbb += w @ (q[:, ib[0]] * q[:, ib[1]])
                C += w @ (q[:, :, None] * a[s:e, None, :]).reshape(e - s, r * K)
            T = _pinv_factor(_unpack(gbb, r, ib)) @ C.reshape(-1, r, K)
            G -= np.swapaxes(T, 1, 2) @ T

        # ── Eliminate the covariates and the intercept ──────────
        d = self.Qd.shape[1]
        T = _pinv_factor(G[:, :d, :d]) @ G[:, :d, d:]
        M = G[:, d:, d:] - np.swapaxes(T, 1, 2) @ T
        return M * (sd[:, None] * sd[None, :])

    def cross_chunks(
        self, R: np.ndarray, weights: Iterator[np.ndarray]
    ) -> np.ndarray:
        """:meth:`cross` over an iterator of ``(B, n)`` weight chunks, stacked."""
        out = [self.cross(R, W) for W in weights]
        k = 1 if np.ndim(R) == 1 else np.shape(R)[1]
        return np.concatenate(out) if out else np.zeros((0, k, k))

    # ── Fits on one weight vector ───────────────────────────────

    def residuals(self, R: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
        """Residuals of each response on the design, weighted by ``w``.

        ``(n, k)`` (or ``(n,)`` for one response). The fit is the weighted
        least-squares one, so the residuals of the drawn rows equal those of an
        unweighted fit on the rows a bootstrap replicate repeats. Rows with
        weight zero get the residual the fit implies for them.
        """
        R = np.asarray(R, dtype=np.float64)
        one = R.ndim == 1
        R2 = R[:, None] if one else R
        w = np.ones(self.n) if w is None else np.asarray(w, dtype=np.float64)
        mean = R2.mean(axis=0)
        sd = R2.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        Rs = ((R2 - mean) / sd)[self.order]
        wp = w[self.order]
        Qd = self.Qd_sorted
        d = Qd.shape[1]

        # Normal equations block by block: each block's coefficients are
        # expressed through the dense ones (Schur complement), solved, then
        # substituted back.
        Gdd = (Qd * wp[:, None]).T @ Qd
        Gda = (Qd * wp[:, None]).T @ Rs
        back = []
        for start, stop, Qb in self.blocks:
            sl = slice(start, stop)
            qw = Qb * wp[sl, None]
            F = _pinv_factor(qw.T @ Qb)
            Pb = F.T @ F
            Cbd = qw.T @ Qd[sl]
            Cba = qw.T @ Rs[sl]
            Gdd -= Cbd.T @ Pb @ Cbd
            Gda -= Cbd.T @ Pb @ Cba
            back.append((sl, Qb, Pb, Cbd, Cba))
        Fd = _pinv_factor(Gdd)
        beta_d = Fd.T @ (Fd @ Gda)
        res = Rs - Qd @ beta_d
        for sl, Qb, Pb, Cbd, Cba in back:
            beta_b = Pb @ (Cba - Cbd @ beta_d)
            res[sl] -= Qb @ beta_b
        out = np.empty_like(res)
        out[self.order] = res
        out *= sd
        return out[:, 0] if one else out


# ────────────────────────────────────────────────────────────────────
# Replicate weights
# ────────────────────────────────────────────────────────────────────


def row_weights(idx: np.ndarray, n: int) -> np.ndarray:
    """Multiplicity of every row in each replicate: ``(B, n)`` from ``(B, n_draw)``."""
    idx = np.atleast_2d(idx)
    W = np.empty((idx.shape[0], n))
    for b in range(idx.shape[0]):
        W[b] = np.bincount(idx[b], minlength=n)
    return W


def group_weights(draws: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Row weights of whole-group resamples.

    ``draws`` ``(B, n_draw)`` are group indices, ``codes`` the group of every
    row; a row's weight is how often its group was drawn.
    """
    draws = np.atleast_2d(draws)
    m = np.empty((draws.shape[0], n_groups))
    for b in range(draws.shape[0]):
        m[b] = np.bincount(draws[b], minlength=n_groups)
    return m[:, codes]


def chunked(n_total: int, make: Callable[[int], np.ndarray], size: int = CHUNK):
    """``make(b)`` for consecutive chunks of at most ``size`` of ``n_total``."""
    for start in range(0, int(n_total), size):
        yield make(min(size, int(n_total) - start))


# ────────────────────────────────────────────────────────────────────
# Statistics from the cross-products
# ────────────────────────────────────────────────────────────────────


def weighted_sst(v: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Weighted total sum of squares of ``v`` about its weighted mean, per row
    of ``W``."""
    v = np.asarray(v, dtype=np.float64)
    c = v - v.mean()
    s0 = W.sum(axis=1)
    s1 = W @ c
    s2 = W @ (c * c)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(s0 > 0, s2 - s1 * s1 / np.where(s0 > 0, s0, 1.0), 0.0)


def incremental_r2(M: np.ndarray, sst_y: np.ndarray, sst_x: np.ndarray) -> np.ndarray:
    """``ΔR²`` of exposure ``x`` over the design, from ``M`` of ``[y, x]``.

    By Frisch–Waugh the drop in residual sum of squares when ``x`` joins the
    design is ``(r_yᵀ W r_x)² / (r_xᵀ W r_x)``; divided by the outcome's total
    sum of squares it is the incremental R² of :func:`objective_scoring.score`.
    Zero where the outcome or the exposure is constant, or the exposure lies in
    the design's span.
    """
    mxy, mxx = M[:, 0, 1], M[:, 1, 1]
    ok = (sst_y > 0) & (sst_x > 0) & (mxx > _COLLINEAR * np.maximum(sst_x, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ok, mxy * mxy / np.where(ok, mxx * sst_y, 1.0), 0.0)


def slope(M: np.ndarray, x: int, y: int) -> np.ndarray:
    """Weighted least-squares slope of response ``y`` on response ``x`` given
    the design, from ``M``; ``nan`` where ``x`` has no residual variance."""
    mxx = M[:, x, x]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(mxx > 0, M[:, x, y] / np.where(mxx > 0, mxx, 1.0), np.nan)


# ────────────────────────────────────────────────────────────────────
# Scorers for the bootstrap and permutation loops
# ────────────────────────────────────────────────────────────────────


def _split(cov: np.ndarray | None, n_dense: int):
    """``[dense | smooth]`` control columns as ``(dense, smooth)``; ``None``
    when there is no smooth, so the caller keeps its direct fits."""
    if cov is None:
        return None
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim == 1:
        cov = cov[:, None]
    if cov.shape[1] <= n_dense:
        return None
    return (cov[:, :n_dense] if n_dense else None), cov[:, n_dense:]


def r2_bootstrap_scorer(n_dense: int):
    """``batch_scorer_factory`` of :func:`statistical_testing.bootstrap_score_ci`
    for the incremental R² of a prediction over ``[covariates | smooth]``
    controls (the first ``n_dense`` columns are the covariates)."""

    def factory(t, p, cov):
        parts = _split(cov, n_dense)
        if parts is None:
            return None
        design = BlockDesign(len(t), *parts)
        R = np.column_stack([t, p])

        def scorer(W):
            M = design.cross(R, W)
            return incremental_r2(M, weighted_sst(t, W), weighted_sst(p, W))

        return scorer

    return factory


def r2_permutation_scorer(n_dense: int):
    """``permutation_scorer_factory`` of
    :func:`statistical_testing.permutation_pvalue` for the incremental R².

    The controls and the prediction stay fixed under Freedman–Lane, so the
    prediction's residual ``r_x`` on the controls is formed once. A permuted
    outcome ``t* = t_fit + t_res[π]`` has residual cross-product ``r_xᵀ t*``
    with it (``r_x`` is orthogonal to the controls), and its total sum of
    squares follows from three fixed vectors, so each permutation costs two
    dot products over the rows.
    """

    def factory(t_fit, t_res, p, cov):
        parts = _split(cov, n_dense)
        if parts is None:
            return None
        design = BlockDesign(len(p), *parts)
        rx = design.residuals(p)
        mxx = float(rx @ rx)
        pc = p - p.mean()
        sst_x = float(pc @ pc)
        f = t_fit - (t_fit.mean() + t_res.mean())
        base = float(f @ f + t_res @ t_res)
        c0 = float(rx @ t_fit)
        if not (sst_x > 0 and mxx > _COLLINEAR * sst_x):
            return lambda perms: np.zeros(len(perms))

        def scorer(perms):
            T = t_res[perms]
            num = c0 + T @ rx
            sst = base + 2.0 * (T @ f)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(sst > 0, num * num / (mxx * sst), 0.0)

        return scorer

    return factory


#: Metrics whose replicates take residuals on fixed designs, scored on them.
RESIDUAL_METRICS: frozenset[str] = frozenset({"distance_corr", "nrmse"})


def _on_residuals(metric: str, rt: np.ndarray, rc: np.ndarray) -> float:
    """``metric`` from the outcome's and the exposure's residuals, as
    :func:`objective_scoring.score` finishes it."""
    from .objective_scoring import _minmax01, distance_correlation

    if metric == "distance_corr":
        if float(np.var(rt)) == 0 or float(np.var(rc)) == 0:
            return 0.0
        return float(distance_correlation(rt, rc))
    t01, c01 = _minmax01(rt), _minmax01(rc)
    return float(np.sqrt(np.mean((t01 - c01) ** 2)))


def _residual_designs(n: int, parts, spatial_method: str):
    """The outcome's and the exposure's designs: both ``[1, covariates,
    smooth]`` under KS-AIC; the exposure's ``[1, smooth]`` under Spatial+."""
    dense, smooth = parts
    d_t = BlockDesign(n, dense, smooth)
    d_c = d_t if spatial_method != "spatial_plus" else BlockDesign(n, None, smooth)
    return d_t, d_c


def residual_bootstrap_scorer(metric: str, n_dense: int, spatial_method: str):
    """``batch_scorer_factory`` for ``distance_corr`` / ``nrmse`` over
    ``[covariates | smooth]`` controls.

    Each replicate's residuals come from one weighted block fit per side
    (:meth:`BlockDesign.residuals`) instead of two ``lstsq`` passes over the
    replicate's rows; both metrics are invariant to the order of the paired
    rows, so the replicate is the drawn rows repeated by their weight.
    """

    def factory(t, p, cov):
        parts = _split(cov, n_dense)
        if parts is None or metric not in RESIDUAL_METRICS:
            return None
        n = len(t)
        d_t, d_c = _residual_designs(n, parts, spatial_method)
        degenerate = 0.0 if metric == "distance_corr" else float("inf")

        def scorer(W):
            out = np.empty(len(W))
            for b, w in enumerate(W):
                rows = np.repeat(np.arange(n), np.asarray(w, dtype=np.int64))
                if np.var(t[rows]) == 0 or np.var(p[rows]) == 0 or len(rows) < 3:
                    out[b] = degenerate
                    continue
                rt = d_t.residuals(t, w)[rows]
                rc = d_c.residuals(p, w)[rows]
                out[b] = _on_residuals(metric, rt, rc)
            return out

        return scorer

    return factory


def residual_permutation_scorer(metric: str, n_dense: int, spatial_method: str):
    """``permutation_scorer_factory`` for ``distance_corr`` / ``nrmse``.

    The exposure's residual is fixed under Freedman–Lane; a permuted outcome's
    residual on ``[1, covariates, smooth]`` is the projection of the permuted
    covariate residuals (the fitted part lies in that span), taken for a chunk
    of permutations at once against one orthonormal basis of the design.
    """

    def factory(t_fit, t_res, p, cov):
        parts = _split(cov, n_dense)
        if parts is None or metric not in RESIDUAL_METRICS:
            return None
        n = len(p)
        dense, smooth = parts
        cols = [np.ones((n, 1)), smooth] + ([] if dense is None else [dense])
        U = orthonormal(np.column_stack(cols))
        _, d_c = _residual_designs(n, parts, spatial_method)
        rc = d_c.residuals(p)
        degenerate = 0.0 if metric == "distance_corr" else float("inf")

        def scorer(perms):
            V = t_res[perms]
            R = V - (V @ U) @ U.T
            out = np.empty(len(perms))
            for b in range(len(perms)):
                if np.var(t_fit + V[b]) == 0 or np.var(p) == 0:
                    out[b] = degenerate
                    continue
                out[b] = _on_residuals(metric, R[b], rc)
            return out

        return scorer

    return factory
