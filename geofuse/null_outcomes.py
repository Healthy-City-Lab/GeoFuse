"""Null outcomes for calibrating a decision rule, and the rate it rejects at.

A decision rule that searches the exposure (the CGI's held-out slope with the
corrected resampled t over K splits) needs a measured false-positive rate:
the share of outcomes unrelated to greenery for which the whole rule, search
included, still declares an association. This module draws such outcomes from
the real one and summarises the rate; :mod:`geofuse.jobs.outer_null` runs the
rule on them.

Two kinds of null outcome:

- **free** (:func:`free_null`): the outcome permuted over entities, within
  strata (e.g. the year fixed-effect groups) when given. Removes every spatial
  structure.
- **spatial** (:class:`SpatialNull`): keeps the outcome's spatial structure at
  two scales with its alignment to the exposures randomised. Between groups,
  the observed group means are permuted across groups; within groups, the
  deviations from the group mean are replaced by a variogram-matched
  surrogate (Viladomat, Mazumder, McInturff, McCauley & Hastie 2014,
  Biometrics 70:409–418): permuted within group, smoothed by the mean of the
  k nearest neighbours in the same group and rescaled so the surrogate's
  empirical variogram matches the deviations' over distance bins
  (pairs within 20 km), ``z = √β · S_k(perm(e)) + √α · u``,
  ``u ~ N(0, 1)``, with ``(α, β)`` by non-negative least squares and ``k``
  chosen per surrogate by the smallest residual.

Neither kind reads the exposures. :func:`clopper_pearson` gives the exact
interval of the rate and :func:`stop_decision` the optional sequential rule.
"""

from __future__ import annotations

import numpy as np

#: Null kinds a run may ask for; ``both`` runs each.
NULL_KINDS = ("spatial", "free", "both")

#: Distance-bin edges of the variogram (metres): pairs within 20 km.
VARIOGRAM_BINS_M = (0, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000)

#: Neighbourhood sizes the surrogate's smoothing is chosen from.
K_GRID = (3, 5, 10, 20, 35, 50, 75, 100, 150, 200, 300, 400)

#: Anchor entities whose neighbours within the largest bin form the pairs,
#: and the largest number of pairs kept.
PAIR_ANCHORS = 4000
PAIR_CAP = 8_000_000

#: The level a null rejection is counted at.
NULL_ALPHA = 0.05


def kinds(kind: str) -> list[str]:
    """The null kinds ``kind`` runs: ``both`` is ``spatial`` then ``free``."""
    if kind not in NULL_KINDS:
        raise ValueError(f"null kind must be one of {NULL_KINDS}; got {kind!r}.")
    return ["spatial", "free"] if kind == "both" else [kind]


def null_seed(seed: int, kind: str, index: int) -> np.random.Generator:
    """The generator of null ``index`` of ``kind``: the same draw on a rerun,
    whichever nulls ran before it."""
    return np.random.default_rng([int(seed), NULL_KINDS.index(kind), int(index)])


# ────────────────────────────────────────────────────────────────────
# Free permutation
# ────────────────────────────────────────────────────────────────────


def free_null(
    y: np.ndarray, rng: np.random.Generator, strata: "np.ndarray | None" = None
) -> np.ndarray:
    """``y`` permuted over entities, within each stratum when given."""
    y = np.asarray(y, dtype=np.float64)
    if strata is None:
        return rng.permutation(y)
    out = y.copy()
    strata = np.asarray(strata)
    for level in _levels(strata):
        rows = np.flatnonzero(strata == level)
        out[rows] = y[rng.permutation(rows)]
    return out


def _levels(labels: np.ndarray) -> list:
    return list(dict.fromkeys(labels.tolist()))


# ────────────────────────────────────────────────────────────────────
# Variogram
# ────────────────────────────────────────────────────────────────────


def variogram_pairs(
    xy: np.ndarray,
    rng: np.random.Generator,
    *,
    bins_m: "tuple[float, ...]" = VARIOGRAM_BINS_M,
    anchors: int = PAIR_ANCHORS,
    cap: int = PAIR_CAP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Entity pairs ``(i, j, bin)`` within the largest bin edge.

    The pairs are every neighbour of up to ``anchors`` random entities within
    ``bins_m[-1]``, at most ``cap`` of them, each with its distance bin.
    """
    from scipy.spatial import cKDTree

    xy = np.asarray(xy, dtype=np.float64)
    edges = np.asarray(bins_m, dtype=np.float64)
    a = rng.choice(len(xy), size=min(int(anchors), len(xy)), replace=False)
    near = cKDTree(xy).query_ball_point(xy[a], r=float(edges[-1]))
    i = np.concatenate([np.full(len(n), k) for k, n in zip(a, near, strict=True)])
    j = np.concatenate([np.asarray(n, dtype=np.int64) for n in near])
    keep = i != j
    i, j = i[keep].astype(np.int64), j[keep]
    if len(i) > cap:
        s = rng.choice(len(i), int(cap), replace=False)
        i, j = i[s], j[s]
    d = np.hypot(*(xy[i] - xy[j]).T)
    b = np.digitize(d, edges) - 1
    ok = (b >= 0) & (b < len(edges) - 1)
    return i[ok], j[ok], b[ok]


def variogram(
    v: np.ndarray, i: np.ndarray, j: np.ndarray, b: np.ndarray, n_bins: int
) -> np.ndarray:
    """Empirical semivariogram of ``v`` per bin; NaN where a bin has no pair."""
    g = 0.5 * (v[i] - v[j]) ** 2
    total = np.bincount(b, weights=g, minlength=n_bins)
    count = np.bincount(b, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


# ────────────────────────────────────────────────────────────────────
# Spatial surrogates
# ────────────────────────────────────────────────────────────────────


class SpatialNull:
    """Spatially structured null outcomes of one outcome.

    ``y`` and ``xy`` (metres) are one entity per row, all finite; ``groups``
    labels the areas whose means are permuted (``None`` = one area). The
    pairs, the deviations' variogram and the neighbour lists are built once
    from ``seed``; each :meth:`draw` takes its own generator.
    """

    def __init__(
        self,
        y: np.ndarray,
        xy: np.ndarray,
        groups: "np.ndarray | None" = None,
        *,
        seed: int = 0,
        bins_m: "tuple[float, ...]" = VARIOGRAM_BINS_M,
        k_grid: "tuple[int, ...]" = K_GRID,
        anchors: int = PAIR_ANCHORS,
        cap: int = PAIR_CAP,
    ):
        from scipy.spatial import cKDTree

        y = np.asarray(y, dtype=np.float64)
        self.xy = np.asarray(xy, dtype=np.float64)
        n = len(y)
        if n < 4:
            raise ValueError("A spatial null needs at least four entities.")
        self.groups = (
            np.zeros(n, dtype=np.int64) if groups is None else np.asarray(groups)
        )
        self.levels = _levels(self.groups)
        codes = {g: c for c, g in enumerate(self.levels)}
        self.codes = np.array(
            [codes[g] for g in self.groups.tolist()], dtype=np.int64
        )
        self.means = np.bincount(self.codes, weights=y) / np.bincount(self.codes)
        self.deviations = y - self.means[self.codes]
        self.bins_m = tuple(float(e) for e in bins_m)
        self.n_bins = len(self.bins_m) - 1
        rng = np.random.default_rng(int(seed))
        self.i, self.j, self.b = variogram_pairs(
            self.xy, rng, bins_m=self.bins_m, anchors=anchors, cap=cap
        )
        self.n_pairs = np.bincount(self.b, minlength=self.n_bins)
        self.target = variogram(self.deviations, self.i, self.j, self.b, self.n_bins)
        self.k_grid = tuple(int(k) for k in k_grid if k <= n)
        # Each group's rows and their nearest neighbours within the group,
        # nearest first (the entity itself included).
        self._members: list[tuple[np.ndarray, np.ndarray]] = []
        for c in range(len(self.levels)):
            rows = np.flatnonzero(self.codes == c)
            k_c = min(max(self.k_grid), len(rows))
            local = cKDTree(self.xy[rows]).query(self.xy[rows], k=k_c)[1]
            self._members.append((rows, rows[np.asarray(local).reshape(len(rows), -1)]))
        self._sd = float(self.deviations.std())

    def _permute_within(self, rng: np.random.Generator) -> np.ndarray:
        out = self.deviations.copy()
        for c in range(len(self.levels)):
            rows = np.flatnonzero(self.codes == c)
            out[rows] = self.deviations[rng.permutation(rows)]
        return out

    def _centred(self, v: np.ndarray) -> np.ndarray:
        means = np.bincount(self.codes, weights=v) / np.bincount(self.codes)
        return v - means[self.codes]

    def within(self, rng: np.random.Generator) -> tuple[np.ndarray, int]:
        """A variogram-matched surrogate of the deviations and its ``k``.

        The smoothing averages each entity's ``k`` nearest neighbours in its
        own group (the whole group when it is smaller), and the surrogate is
        centred within groups, so it carries no between-group difference.
        """
        from scipy.optimize import nnls

        perm = self._permute_within(rng)
        cums = [(rows, np.cumsum(perm[nbr], axis=1)) for rows, nbr in self._members]
        ok = np.isfinite(self.target)
        best = None
        for k in self.k_grid:
            smooth = np.empty_like(perm)
            for rows, cum in cums:
                k_c = min(k, cum.shape[1])
                smooth[rows] = cum[:, k_c - 1] / k_c
            smooth = self._centred(smooth)
            g = variogram(smooth, self.i, self.j, self.b, self.n_bins)
            fit = ok & np.isfinite(g)
            if fit.sum() < 2:
                continue
            design = np.column_stack([np.ones(fit.sum()), g[fit]])
            coef, rss = nnls(design, self.target[fit])
            if best is None or rss < best[0]:
                best = (rss, k, smooth, coef)
        if best is None:
            return perm, 1
        _, k, smooth, (alpha, beta) = best
        noise = rng.standard_normal(len(smooth))
        z = self._centred(np.sqrt(beta) * smooth + np.sqrt(alpha) * noise)
        sd = float(z.std())
        return (z / sd * self._sd if sd > 0 else z), int(k)

    def draw(self, rng: np.random.Generator) -> dict:
        """One null outcome: ``{"y", "k", "variogram"}``, the last the
        within-group variogram of the surrogate's deviations."""
        z, k = self.within(rng)
        means = self.means[rng.permutation(len(self.means))]
        return {
            "y": z + means[self.codes],
            "k": k,
            "variogram": variogram(z, self.i, self.j, self.b, self.n_bins),
        }

    def match_table(self, surrogates: "list[np.ndarray]") -> list[dict]:
        """The deviations' variogram next to the surrogates' median and 5–95 %
        band, one row per distance bin."""
        g = np.asarray(surrogates, dtype=np.float64).reshape(-1, self.n_bins)
        rows = []
        for b in range(self.n_bins):
            col = g[:, b][np.isfinite(g[:, b])]
            rows.append(
                {
                    "bin_low_m": self.bins_m[b],
                    "bin_high_m": self.bins_m[b + 1],
                    "n_pairs": int(self.n_pairs[b]),
                    "outcome": _finite(self.target[b]),
                    "surrogate_median": _quantile(col, 0.5),
                    "surrogate_p05": _quantile(col, 0.05),
                    "surrogate_p95": _quantile(col, 0.95),
                    "n_surrogates": int(col.size),
                }
            )
        return rows


def _finite(value) -> float | None:
    v = float(value)
    return v if np.isfinite(v) else None


def _quantile(values: np.ndarray, q: float) -> float | None:
    return _finite(np.quantile(values, q)) if values.size else None


# ────────────────────────────────────────────────────────────────────
# Rejection rate and the sequential rule
# ────────────────────────────────────────────────────────────────────


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact binomial interval of ``k`` hits in ``n``."""
    from scipy import stats

    if n <= 0:
        return float("nan"), float("nan")
    a = 1.0 - level
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def stop_decision(
    hits: int, n: int, *, after: int | None, low: int | None, high: int | None
) -> str | None:
    """The sequential rule after ``n`` nulls with ``hits`` rejections.

    Evaluated once, at ``after`` nulls: ``"not inflated"`` when ``hits <=
    low``, ``"inflated"`` when ``hits >= high``, else ``None`` (run all).
    ``None`` too before ``after`` and without a rule.
    """
    if not after or n != int(after):
        return None
    if low is not None and hits <= int(low):
        return "not inflated"
    if high is not None and hits >= int(high):
        return "inflated"
    return None
