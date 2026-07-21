"""Cached partial distance correlation (Székely–Rizzo) for the fusion scorers.

``dcor.partial_distance_correlation`` rebuilds, on every call, the U-centered
distance matrices of all three inputs plus six O(n²) inner products. In the
fusion engine two of the three sides are constant across many calls:

* per Optuna trial, the target and the conditioning matrix ``z = [covariates,
  spatial smooth]`` are fixed per scored subset — only the composite changes;
* per permutation replicate (Freedman–Lane), the prediction and ``z`` are
  fixed — only the surrogate outcome changes;
* per bootstrap replicate, the *raw* distance matrix of a resampled vector is
  a fancy-indexed submatrix of the full one — no distances need recomputing.

This module provides the shared math (float32 matrices, in-place U-centering,
BLAS flattened-dot inner products) plus a small side cache and the replicate /
surrogate scorers built on it. Results match the stock ``dcor`` estimator to
~1e-6 — far below trial-ranking noise — at a measured 12–17× speedup per call.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Callable

import numpy as np

# Entity counts above this skip the cached paths and fall back to the stock
# ``dcor`` estimator. One cached side entry holds two float32 n² matrices
# (137 MB each at n = 6,000) and the cache keeps several entries, so this cap
# bounds the estimator's resident footprint near 1 GB.
MAX_CACHE_N: int = 6_000

# Live side entries kept per cache dict — one per actively scored subset
# (in-bag + OOB during the bootstrap; test / all during reporting).
_MAX_SIDE_ENTRIES: int = 4


def distance_matrix(v: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Raw pairwise Euclidean distance matrix of a 1-D or ``(n, d)`` array."""
    a = np.asarray(v, dtype=np.float64)
    if a.ndim == 1 or a.shape[1] == 1:
        col = a.reshape(-1).astype(dtype)
        return np.abs(col[:, None] - col[None, :])
    from scipy.spatial.distance import cdist

    return cdist(a, a).astype(dtype, copy=False)


def u_center(D: np.ndarray) -> np.ndarray:
    """U-center a raw distance matrix in place (Székely–Rizzo) and return it.

    With row sums ``r_i = Σ_j D_ij`` and grand sum ``g = Σ_ij D_ij``::

        Ã_ij = D_ij − r_i/(n−2) − r_j/(n−2) + g/((n−1)(n−2)),   Ã_ii = 0
    """
    n = D.shape[0]
    col_sums = D.sum(axis=0, dtype=np.float64)
    r = col_sums / (n - 2)
    # The grand sum is the column sums' total, so it needs no second full pass.
    g = float(col_sums.sum()) / ((n - 1) * (n - 2))
    r = r.astype(D.dtype, copy=False)
    D -= r[None, :]
    D -= r[:, None]
    D += D.dtype.type(g)
    np.fill_diagonal(D, 0.0)
    return D


def u_centered_distance_matrix(v: np.ndarray, dtype=np.float32) -> np.ndarray:
    """U-centered distance matrix of ``v`` (fresh array, safe to mutate)."""
    return u_center(distance_matrix(v, dtype=dtype))


def u_dot(a: np.ndarray, b: np.ndarray) -> float:
    """Flattened BLAS dot of two matrices — no n² temporary.

    The ``1/(n(n−3))`` normalization of ``dcor``'s ``u_product`` cancels in
    every correlation ratio below, so raw dots are correct here.
    """
    return float(a.ravel() @ b.ravel())


def _pdcor_from_products(
    xx: float, yy: float, zz: float, xy: float, xz: float, yz: float
) -> float:
    """r*(x, y; z) from the six U-inner-products, with dcor's zero guards."""
    denom_sqr = xx * yy
    r_xy = xy / np.sqrt(denom_sqr) if denom_sqr > 0 else 0.0
    r_xy = float(np.clip(r_xy, -1.0, 1.0))

    denom_sqr = xx * zz
    r_xz = xz / np.sqrt(denom_sqr) if denom_sqr > 0 else 0.0
    r_xz = float(np.clip(r_xz, -1.0, 1.0))

    denom_sqr = yy * zz
    r_yz = yz / np.sqrt(denom_sqr) if denom_sqr > 0 else 0.0
    r_yz = float(np.clip(r_yz, -1.0, 1.0))

    denom = np.sqrt(1.0 - r_xz**2) * np.sqrt(1.0 - r_yz**2)
    if denom == 0.0:
        return 0.0
    return float((r_xy - r_xz * r_yz) / denom)


def _side_fingerprint(y: np.ndarray, z: np.ndarray | None) -> bytes:
    """Content key for one (fixed-vector, conditioning-matrix) pair.

    Digests the raw bytes rather than moment sums so a permutation of the same
    values — a different resample of one subset — cannot collide onto a cached
    side matrix.
    """
    h = hashlib.blake2b(digest_size=16)
    for arr in (y, z):
        if arr is None:
            h.update(b"\x00")
            continue
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
        h.update(repr(a.shape).encode())
        h.update(a.tobytes())
    return h.digest()


class PdcorSideCache:
    """U-centered matrices + self/cross products of one fixed ``(y, z)`` pair."""

    __slots__ = ("B", "C", "bb", "cc", "bc")

    def __init__(self, y: np.ndarray, z: np.ndarray | None) -> None:
        self.B = u_centered_distance_matrix(y)
        self.bb = u_dot(self.B, self.B)
        if z is None:
            self.C: np.ndarray | None = None
            self.cc = 0.0
            self.bc = 0.0
        else:
            self.C = u_centered_distance_matrix(z)
            self.cc = u_dot(self.C, self.C)
            self.bc = u_dot(self.B, self.C)


def _get_side(
    cache: "OrderedDict[tuple, PdcorSideCache]", y: np.ndarray, z: np.ndarray | None
) -> PdcorSideCache:
    key = _side_fingerprint(y, z)
    entry = cache.get(key)
    if entry is None:
        entry = PdcorSideCache(y, z)
        cache[key] = entry
        while len(cache) > _MAX_SIDE_ENTRIES:
            cache.popitem(last=False)
    else:
        cache.move_to_end(key)
    return entry


def partial_distance_correlation_cached(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cache: "OrderedDict[tuple, PdcorSideCache]",
) -> float:
    """pdcor(x, y; z) reusing cached U-centered matrices for the (x, z) sides.

    The estimator is symmetric in (x, y); callers pass the vector that stays
    constant across calls as ``x`` (the fusion scorer passes the target) so the
    side cache hits on every call after the first. ``y`` (the per-trial
    composite) is the only side computed fresh. Caller guarantees ``z`` is not
    ``None`` and ``4 <= n <= MAX_CACHE_N``.
    """
    side = _get_side(cache, x, z)
    A = u_centered_distance_matrix(y)
    aa = u_dot(A, A)
    ab = u_dot(A, side.B)
    ac = u_dot(A, side.C) if side.C is not None else 0.0
    # Product naming follows pdcor(x, y; z): xx ↔ the cached x side.
    return _pdcor_from_products(side.bb, aa, side.cc, ab, side.bc, ac)


# ---------------------------------------------------------------------------
# Replicate scorers for the reporting bootstrap / permutation passes
# ---------------------------------------------------------------------------


class _ReplicateScorer:
    """pdcor(t[idx], p[idx]; z[idx]) for arbitrary index vectors.

    Precomputes the raw (un-centered) float32 distance matrices once; each
    replicate is three fancy-indexed submatrices (a valid distance matrix of
    the resampled rows, repeats included), in-place U-centering, and six BLAS
    dots — no distance recomputation.
    """

    __slots__ = ("Dt", "Dp", "Dz")

    def __init__(self, t: np.ndarray, p: np.ndarray, z: np.ndarray) -> None:
        self.Dt = distance_matrix(t)
        self.Dp = distance_matrix(p)
        self.Dz = distance_matrix(z)

    def score(self, idx: np.ndarray) -> float:
        ix = np.ix_(idx, idx)
        Bt = u_center(self.Dt[ix])
        Ap = u_center(self.Dp[ix])
        Cz = u_center(self.Dz[ix])
        tt = u_dot(Bt, Bt)
        pp = u_dot(Ap, Ap)
        zz = u_dot(Cz, Cz)
        tp = u_dot(Bt, Ap)
        tz = u_dot(Bt, Cz)
        pz = u_dot(Ap, Cz)
        return _pdcor_from_products(tt, pp, zz, tp, tz, pz)


def pdcor_replicate_scorer_factory(
    t: np.ndarray, p: np.ndarray, z: np.ndarray | None
) -> Callable[[np.ndarray], float] | None:
    """Factory handed to ``statistical_testing.bootstrap_score_ci``.

    Returns an index-vector scorer for the ``partial_distance_corr`` metric, or
    ``None`` to keep the generic ``score_fn`` path (no conditioning matrix —
    the plain mergesort estimator is already cheap per replicate — or a row
    count outside the cached-path budget).
    """
    if z is None:
        return None
    n = len(np.asarray(t).ravel())
    if n < 4 or n > MAX_CACHE_N:
        return None
    return _ReplicateScorer(t, p, z).score


def pdcor_paired_replicate_scorers(
    t: np.ndarray, p1: np.ndarray, p2: np.ndarray, z: np.ndarray | None
) -> tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], float]] | None:
    """Two index-vector scorers sharing the target / conditioning matrices.

    Used by the paired CGI-vs-standalone bootstrap, which scores two composites
    on the same resample: the target and ``z`` distance matrices are built once
    and shared, halving the precompute versus two independent scorers.
    """
    if z is None:
        return None
    n = len(np.asarray(t).ravel())
    if n < 4 or n > MAX_CACHE_N:
        return None
    Dt = distance_matrix(t)
    Dz = distance_matrix(z)
    Dp1 = distance_matrix(p1)
    Dp2 = distance_matrix(p2)

    def _make(Dp: np.ndarray) -> Callable[[np.ndarray], float]:
        def _score(idx: np.ndarray) -> float:
            ix = np.ix_(idx, idx)
            Bt = u_center(Dt[ix])
            Ap = u_center(Dp[ix])
            Cz = u_center(Dz[ix])
            return _pdcor_from_products(
                u_dot(Bt, Bt),
                u_dot(Ap, Ap),
                u_dot(Cz, Cz),
                u_dot(Bt, Ap),
                u_dot(Bt, Cz),
                u_dot(Ap, Cz),
            )

        return _score

    return _make(Dp1), _make(Dp2)


class _SurrogateScorer:
    """pdcor(t*, p; z) where only the surrogate outcome ``t*`` changes.

    The prediction and conditioning sides are U-centered once; each call pays
    one 1-D abs-diff matrix + centering for the surrogate plus three dots —
    the Freedman–Lane permutation null keeps ``p`` and ``z`` fixed.
    """

    __slots__ = ("A", "C", "aa", "cc", "ac")

    def __init__(self, p: np.ndarray, z: np.ndarray) -> None:
        self.A = u_centered_distance_matrix(p)
        self.C = u_centered_distance_matrix(z)
        self.aa = u_dot(self.A, self.A)
        self.cc = u_dot(self.C, self.C)
        self.ac = u_dot(self.A, self.C)

    def score(self, t_star: np.ndarray) -> float:
        B = u_centered_distance_matrix(t_star)
        bb = u_dot(B, B)
        ab = u_dot(B, self.A)
        bc = u_dot(B, self.C)
        # pdcor(x = t*, y = p; z).
        return _pdcor_from_products(bb, self.aa, self.cc, ab, bc, self.ac)


def pdcor_surrogate_scorer_factory(
    p: np.ndarray, z: np.ndarray | None
) -> Callable[[np.ndarray], float] | None:
    """Factory handed to ``statistical_testing.permutation_pvalue``."""
    if z is None:
        return None
    n = len(np.asarray(p).ravel())
    if n < 4 or n > MAX_CACHE_N:
        return None
    return _SurrogateScorer(p, z).score
