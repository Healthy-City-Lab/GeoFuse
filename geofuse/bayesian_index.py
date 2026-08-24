"""Data-driven CGI: sweep the grid for a shortlist, then fit it all jointly.

Two stages, in this order:

1. :func:`sweep` — exhaustive, cross-validated grid over ``(radius, statistic)``
   per channel and over the functional form. Channel weights are **not**
   searched: for the linear form the simplex-constrained optimum has a closed
   form, found by active-set enumeration. Once covariates are projected out,
   every candidate exposure column already sits in the pre-aggregation cache,
   so a candidate is scored from small slices of a precomputed Gram matrix and
   the n-row data is never touched again.

2. :func:`fit` — NUTS over the whole ``(channel, radius, statistic)`` grid at
   once. The spatial scale is a peaked kernel whose location and width are
   sampled, the aggregator is a Dirichlet blend over the statistics, and the
   channel weights sit on a simplex alongside them. Nothing about the scale or
   the aggregator is chosen before the sampler runs, so uncertainty about them
   widens the interval on the effect instead of vanishing into a pick.

Every association is scored and fitted on **magnitude**. The toolbox is used on
outcomes that rise with greenery and on outcomes that fall with it, so no stage
declares a direction; the sign lives in ``beta``, which the posterior estimates
freely, and the simplex weights stay non-negative so an index always reads as
"more exposure, larger index".

:func:`repeated_discovery` runs the sweep over independent train/test shuffles,
because one split cannot show that a discovery generalises.

The sweep still reads the outcome to rank a shortlist and to choose the form.
Three guards, all exercised here: the sweep sees the train pool only, the
headline effect comes from rows it never saw, and :func:`null_calibration`
permutes the outcome and refits the joint model, so the scale and the
aggregator are priced into the reported false-positive rate.
"""

from __future__ import annotations

import itertools
import math
import os
import tempfile
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from geofuse import JobCancelled

from . import parallel

FORMS: tuple[str, ...] = ("linear", "synergy")

# Powers on the main terms of the synergy form, as in the source paper's range.
_POWER_LO, _POWER_HI = 0.2, 1.0


# ────────────────────────────────────────────────────────────────────
# Data preparation
# ────────────────────────────────────────────────────────────────────


def prep(X: np.ndarray, y: np.ndarray, covariates: np.ndarray | None):
    """Frisch-Waugh: project the covariates out of **both** sides, then z-score.

    Residualising only the outcome leaves the exposure correlated with the
    covariates, and the index coefficient is then not the partial association.

    ``X`` is ``(n, channel, radius, stat)``; the return has the same shape.
    """
    y = np.asarray(y, dtype=np.float64)
    if covariates is None or covariates.size == 0:
        covariates = np.ones((len(y), 1))
    q, _ = np.linalg.qr(np.asarray(covariates, dtype=np.float64))

    def rz(a):
        return a - q @ (q.T @ a)

    yr = rz(y)
    yr = (yr - yr.mean()) / (yr.std() + 1e-12)
    flat = rz(np.asarray(X, dtype=np.float64).reshape(len(X), -1))
    flat = (flat - flat.mean(0)) / (flat.std(0) + 1e-12)
    return yr, flat.reshape(X.shape)


def _raise_if_cancelled(cancel_check) -> None:
    """Abandon the phase if the job's cancel flag is up.

    Every loop long enough to be worth parallelising is long enough to outlast
    a user's patience, so each one answers the flag itself rather than leaving
    it to the boundaries between phases.
    """
    if cancel_check is not None and cancel_check():
        raise JobCancelled("Index search cancelled by user.")


def _indexed(position: int, fn, task):
    """``fn(task)`` tagged with its position, so a pool may return out of order."""
    return position, fn(task)


def _map(fn, tasks, n_workers: int, cancel_check=None):
    """``fn`` over ``tasks``, in a process pool unless one worker was asked for.

    A single worker runs in-process. Spawning costs more than it saves at that
    width, and it keeps the parallel loops usable in an interpreter where spawn
    is unavailable, which is what a caller passing ``workers=1`` usually wants.

    Wider than that, the pool comes from :mod:`geofuse.parallel`, whose workers
    are created with the BLAS thread limits already in their environment. That
    holds each worker to ~0.11 GB of reserved commit instead of the ~3.0 GB
    numpy and OpenBLAS take when left to size themselves for the whole host, and
    it is the figure :func:`parallel.process_worker_count` budgets against: a
    pool built any other way overdraws the host by more than an order of
    magnitude, and small allocations then fail machine-wide.
    """
    tasks = list(tasks)
    n_workers = min(n_workers, len(tasks))
    if n_workers <= 1:
        out = []
        for t in tasks:
            _raise_if_cancelled(cancel_check)
            out.append(fn(t))
        return out
    out: list = [None] * len(tasks)

    def collect(result):
        position, value = result
        out[position] = value

    finished = parallel.map_batches(
        _indexed,
        [(i, fn, t) for i, t in enumerate(tasks)],
        workers=n_workers,
        on_result=collect,
        cancel_check=cancel_check,
    )
    if not finished:
        raise JobCancelled("Index search cancelled by user.")
    return out


def _pairs(k: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(k) for j in range(i + 1, k)]


def _tstat(e: np.ndarray, y: np.ndarray) -> float:
    e = (e - e.mean()) / (e.std() + 1e-12)
    r = float(np.clip(np.corrcoef(e, y)[0, 1], -0.999999, 0.999999))
    return abs(r) * math.sqrt(max(len(y) - 2, 1)) / math.sqrt(1.0 - r * r)


# ────────────────────────────────────────────────────────────────────
# Closed-form simplex weights
# ────────────────────────────────────────────────────────────────────


def simplex_fit(gram: np.ndarray, xty: np.ndarray):
    """Non-negative weights summing to one that maximise |association|, exactly.

    Enumerates the active sets — interior, every face, every vertex — and
    solves each with a Lagrange multiplier. With two or three channels that is
    at most seven tiny solves, so there is no iterative solver and no tolerance
    to tune. Returns ``(kept_indices, weights)`` or ``None``.

    The quantity maximised is the **magnitude** of the standardised association,
    so a protective exposure and a harmful one of equal strength score the same.
    A signed objective would instead walk away from a protective channel and
    hand the weight to whichever channel happened to correlate upward, which on
    greenery data is usually the one carrying no signal at all. Magnitude is a
    max over two signed problems — one in ``xty`` and one in ``-xty`` — because
    ``max |f| = max(max f, max -f)``, and each is the same Lagrange solve.

    The weights stay non-negative in both branches, so the index always reads as
    "more exposure is a larger index"; the direction of the *effect* lives in
    the sign of beta, which the posterior estimates freely.
    """
    k = len(xty)
    best, best_v = None, -np.inf
    for signed in (xty, -xty):
        for mask in range(1, 1 << k):
            idx = [i for i in range(k) if mask >> i & 1]
            g, c = gram[np.ix_(idx, idx)], signed[idx]
            try:
                gi = np.linalg.inv(g)
            except np.linalg.LinAlgError:
                continue
            one = np.ones(len(idx))
            denom = float(one @ gi @ one)
            if abs(denom) < 1e-15:
                continue
            w = gi @ (c - ((one @ gi @ c - 1.0) / denom) * one)
            if np.any(w < -1e-9):
                continue
            quad = float(w @ g @ w)
            if quad <= 1e-15:
                continue
            v = abs(float(w @ xty[idx])) / math.sqrt(quad)
            if v > best_v:
                best_v, best = v, (np.asarray(idx), np.clip(w, 0.0, None))
    return best


# ────────────────────────────────────────────────────────────────────
# Synergy form
# ────────────────────────────────────────────────────────────────────


@dataclass
class SynergyFit:
    """Weights, powers and the train-fold min/max needed to rebuild the index."""

    w: np.ndarray
    p: np.ndarray
    lo: np.ndarray
    hi: np.ndarray

    def apply(self, E: np.ndarray) -> np.ndarray:
        z = np.clip((E - self.lo) / (self.hi - self.lo + 1e-12), 0.0, 1.0)
        nc = z.shape[1]
        out = (z ** self.p) @ self.w[:nc]
        for k, (i, j) in enumerate(_pairs(nc)):
            out = out + self.w[nc + k] * z[:, i] * z[:, j]
        return out


def fit_synergy(E: np.ndarray, y: np.ndarray, restarts: int = 2) -> SynergyFit:
    """Simplex weights over mains + pairwise interactions, plus main powers.

    Powers need non-negative inputs, so the channel values are min-max scaled
    on the training fold before the exponent is applied.
    """
    nc = E.shape[1]
    nw = nc + len(_pairs(nc))
    lo, hi = E.min(0), E.max(0)

    def unpack(th):
        ex = np.exp(th[:nw] - th[:nw].max())
        w = ex / ex.sum()
        p = _POWER_LO + (_POWER_HI - _POWER_LO) / (1.0 + np.exp(-th[nw:nw + nc]))
        return w, p

    def neg(th):
        w, p = unpack(th)
        e = SynergyFit(w, p, lo, hi).apply(E)
        s = e.std()
        if s < 1e-12:
            return 1e9
        e = (e - e.mean()) / s
        b = float(e @ y / (e @ e))
        return float(np.sum((y - b * e) ** 2))

    best = None
    for seed in range(restarts):
        rng = np.random.default_rng(seed)
        th0 = np.concatenate([rng.normal(0.0, 0.3, nw), np.zeros(nc)])
        r = minimize(neg, th0, method="Powell",
                     options={"maxiter": 40000, "xtol": 1e-4, "ftol": 1e-9})
        if best is None or r.fun < best.fun:
            best = r
    w, p = unpack(best.x)
    return SynergyFit(w, p, lo, hi)


# ────────────────────────────────────────────────────────────────────
# Stage 1 — the sweep
# ────────────────────────────────────────────────────────────────────


def candidate_columns(channels, radius_idx, n_radii, n_stats, stat_idx=None):
    """Flat column index per channel for the (radius, stat) pairs in scope.

    ``radius_idx`` is per-channel, so NDVI and GVI can search different ladders.
    """
    stat_idx = range(n_stats) if stat_idx is None else stat_idx
    return [[(c * n_radii + r) * n_stats + s
             for r in radius_idx[ci] for s in stat_idx]
            for ci, c in enumerate(channels)]


def _sweep_split(task):
    """One inner split of the linear grid. Runs in its own process.

    Takes a memmap path rather than the array: shipping the exposure block
    through the pipe for every task exhausts Windows pipe handles once this is
    called a few dozen times.
    """
    path, shape, combos, y, seed, frac = task
    z = np.memmap(path, dtype=np.float64, mode="r", shape=shape)
    te = np.random.default_rng(seed).random(len(y)) < frac
    tr = ~te
    ztr, zte = np.asarray(z[tr]), np.asarray(z[te])
    g_tr, c_tr = ztr.T @ ztr, ztr.T @ y[tr]
    g_te, c_te = zte.T @ zte, zte.T @ y[te]
    yn = float(np.linalg.norm(y[te]))
    n_te = int(te.sum())
    out = np.zeros(len(combos))
    for k, combo in enumerate(combos):
        i = np.asarray(combo)
        fit = simplex_fit(g_tr[np.ix_(i, i)], c_tr[i])
        if fit is None:
            continue
        sub, w = fit
        j = i[sub]
        den = math.sqrt(max(float(w @ g_te[np.ix_(j, j)] @ w), 1e-15))
        r = float(np.clip(float(w @ c_te[j]) / den / max(yn, 1e-12), -0.999999, 0.999999))
        out[k] = abs(r) * math.sqrt(max(n_te - 2, 1)) / math.sqrt(1.0 - r * r)
    return out


@dataclass
class SweepResult:
    channels: tuple[str, ...]
    columns: tuple[int, ...]          # chosen flat column per channel
    picked: tuple[tuple[int, str], ...]   # (radius_m, stat) per channel
    form: str
    score: float                      # mean held-out |t| of the pick
    surface: np.ndarray = field(repr=False)   # (splits, combos)
    combos: list = field(repr=False)
    one_se_columns: tuple[int, ...] = ()
    winner_counts: dict = field(default_factory=dict, repr=False)
    boundary_hit: tuple[str, ...] = ()
    form_scores: dict = field(default_factory=dict)


def _decode(columns, n_radii, n_stats, radii, stats):
    out = []
    for col in columns:
        r, s = divmod(col % (n_radii * n_stats), n_stats)
        out.append((int(radii[r]), stats[s]))
    return tuple(out)


def sweep(X, radii, stats, y, *, channels, channel_index, radius_idx=None,
          forms=FORMS, splits=40, frac=0.25, seed=0, workers=None,
          objective=None, rescore_top=50, cancel_check=None):
    """Exhaustive grid over (radius, stat) per channel, then over the form.

    The linear grid is solved in closed form and swept exhaustively. The
    synergy form is then fitted at the linear pick's columns and kept only if
    it scores better held-out; a full synergy grid would need the Gram matrix
    rebuilt per power combination for no measured benefit.

    ``objective`` makes the job's own metric decide the winner. The Gram sweep
    ranks every candidate by a correlation t, which is what makes an exhaustive
    grid affordable, but that is not every objective the toolbox offers. When an
    objective is given, the top ``rescore_top`` candidates are re-scored with it
    on the same splits and the best of those wins: the cheap ranking stays a
    filter, and the requested metric makes the decision.
    """
    n_radii, n_stats = X.shape[2], X.shape[3]
    if radius_idx is None:
        radius_idx = [list(range(n_radii))] * len(channels)
    cols = candidate_columns(channel_index, radius_idx, n_radii, n_stats)
    combos = list(itertools.product(*cols))
    flat = np.ascontiguousarray(X.reshape(len(X), -1))

    fd, path = tempfile.mkstemp(suffix=".geofuse-sweep")
    os.close(fd)
    try:
        mm = np.memmap(path, dtype=np.float64, mode="w+", shape=flat.shape)
        mm[:] = flat
        mm.flush()
        del mm
        tasks = [(path, flat.shape, combos, y, seed + s, frac)
                 for s in range(splits)]
        n_workers = workers or parallel.process_worker_count(len(tasks))
        surface = np.array(_map(_sweep_split, tasks, n_workers, cancel_check))
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    mean, se = surface.mean(0), surface.std(0) / math.sqrt(max(splits, 1))
    objective_surface, shortlist = None, None
    if objective is not None and len(combos) > 1:
        shortlist = np.argsort(-mean)[: max(1, int(rescore_top))]
        objective_surface = _rescore(flat, y, combos, shortlist, objective,
                                     splits=splits, frac=frac, seed=seed,
                                     cancel_check=cancel_check)
        # Candidates outside the shortlist keep -inf so they cannot win, and the
        # one-SE rule below reads the same surface the winner came from.
        obj_mean = np.full(len(combos), -np.inf)
        obj_mean[shortlist] = objective_surface.mean(0)
        obj_se = np.zeros(len(combos))
        obj_se[shortlist] = objective_surface.std(0) / math.sqrt(max(splits, 1))
        mean, se = obj_mean, obj_se
    best = int(np.argmax(mean))
    picked = _decode(combos[best], n_radii, n_stats, radii, stats)

    # One-standard-error rule, ranked by total radius for parsimony -- but only
    # among combos that keep the same active channels. Ranking across channel
    # sets let the surviving channel flip and reversed the effect's sign.
    eligible = np.flatnonzero(mean >= mean[best] - se[best])
    totals = np.array([sum(r for r, _ in _decode(combos[k], n_radii, n_stats,
                                                 radii, stats))
                       for k in eligible])
    one_se = tuple(combos[int(eligible[int(np.argmin(totals))])])

    counts: dict = {}
    winner_surface = surface if objective_surface is None else objective_surface
    winner_cols = (list(range(len(combos))) if shortlist is None
                   else [int(c) for c in shortlist])
    for k in winner_surface.argmax(1):
        combo = tuple(combos[winner_cols[int(k)]])
        counts[combo] = counts.get(combo, 0) + 1

    edge = {int(radii[0]), int(radii[-1])}
    boundary = tuple(c for (r, _), c in zip(picked, channels) if r in edge)

    # Form selection at the picked columns, scored on the same splits.
    form_scores = {"linear": float(mean[best])}
    chosen_form, chosen_score = "linear", float(mean[best])
    if "synergy" in forms and len(channels) > 1:
        syn = _score_form_at(flat[:, list(combos[best])], y, "synergy",
                             splits=splits, frac=frac, seed=seed,
                             workers=workers, objective=objective,
                             cancel_check=cancel_check)
        form_scores["synergy"] = syn
        if syn > chosen_score:
            chosen_form, chosen_score = "synergy", syn

    return SweepResult(
        channels=tuple(channels), columns=tuple(combos[best]), picked=picked,
        form=chosen_form, score=chosen_score, surface=surface, combos=combos,
        one_se_columns=one_se, winner_counts=counts, boundary_hit=boundary,
        form_scores=form_scores,
    )


def _rescore(flat, y, combos, shortlist, objective, *, splits, frac, seed,
             cancel_check=None):
    """Score a shortlist of candidates with the job's objective, same splits.

    Weights still come from the closed-form simplex fit on the training rows;
    only the ranking changes. Runs in-process because a shortlist is small and
    the objective is a caller-owned closure, which does not pickle reliably.
    """
    out = np.zeros((splits, len(shortlist)))
    for si in range(splits):
        _raise_if_cancelled(cancel_check)
        te = np.random.default_rng(seed + si).random(len(y)) < frac
        tr = ~te
        ztr = flat[tr]
        g_tr, c_tr = ztr.T @ ztr, ztr.T @ y[tr]
        for k, ci in enumerate(shortlist):
            i = np.asarray(combos[int(ci)])
            fit = simplex_fit(g_tr[np.ix_(i, i)], c_tr[i])
            if fit is None:
                continue
            sub, w = fit
            e = flat[:, i[sub]] @ w
            out[si, k] = objective(e[te], y[te])
    return out


def _form_split_objective(E, y, form, seed, frac, objective):
    te = np.random.default_rng(seed).random(len(y)) < frac
    tr = ~te
    if form == "linear":
        fit = simplex_fit(E[tr].T @ E[tr], E[tr].T @ y[tr])
        if fit is None:
            return 0.0
        sub, w = fit
        return float(objective(E[te][:, sub] @ w, y[te]))
    return float(objective(fit_synergy(E[tr], y[tr]).apply(E[te]), y[te]))


def _form_split(task):
    E, y, form, seed, frac = task
    te = np.random.default_rng(seed).random(len(y)) < frac
    tr = ~te
    if form == "linear":
        fit = simplex_fit(E[tr].T @ E[tr], E[tr].T @ y[tr])
        if fit is None:
            return 0.0
        sub, w = fit
        return _tstat(E[te][:, sub] @ w, y[te])
    return _tstat(fit_synergy(E[tr], y[tr]).apply(E[te]), y[te])


def _score_form_at(E, y, form, *, splits, frac, seed, workers=None,
                   objective=None, cancel_check=None):
    if objective is not None:
        scores = []
        for s in range(splits):
            _raise_if_cancelled(cancel_check)
            scores.append(
                _form_split_objective(E, y, form, seed + s, frac, objective))
        return float(np.mean(scores))
    tasks = [(E, y, form, seed + s, frac) for s in range(splits)]
    n_workers = workers or parallel.process_worker_count(len(tasks))
    return float(np.mean(_map(_form_split, tasks, n_workers, cancel_check)))


def build_index(E_train, y_train, form):
    """Fit the chosen form on training rows; return ``(apply_fn, params)``.

    ``weights`` carries one entry per channel, zero where the simplex fit's
    active set dropped one, so position *i* always means channel *i*. A vector
    holding only the surviving channels would change length with the fit, and
    weights from different fits could then neither be compared nor averaged.
    """
    if form == "linear":
        k = E_train.shape[1]
        fit = simplex_fit(E_train.T @ E_train, E_train.T @ y_train)
        if fit is None:
            sub, w = np.arange(k), np.full(k, 1.0 / k)
        else:
            sub, w = fit
        dense = np.zeros(k)
        dense[sub] = w
        return (lambda M: M @ dense), {"weights": dense, "kept": sub}
    sf = fit_synergy(E_train, y_train)
    return sf.apply, {"weights": sf.w, "powers": sf.p}


# ────────────────────────────────────────────────────────────────────
# Stage 2 — the joint posterior
# ────────────────────────────────────────────────────────────────────


def _dirichlet_prior_ci(k: int, q=(0.025, 0.975)) -> tuple[float, float]:
    """Central interval of one component of a symmetric ``Dirichlet(1, …, 1)``.

    Each marginal is ``Beta(1, k-1)``, whose quantile function is closed form,
    so the prior interval costs nothing to state alongside the posterior one.
    """
    if k <= 1:
        return (1.0, 1.0)
    return tuple(float(1.0 - (1.0 - p) ** (1.0 / (k - 1))) for p in q)


def _radius_prior(radii) -> tuple[float, float]:
    """Centre and spread, in log-metres, of the kernel's location prior.

    Taken from the ladder itself, so a study that searched 50-500 m and one
    that searched 250-5000 m each get a prior covering their own range rather
    than a constant carried over from whichever was written down first.
    """
    lr = np.log(np.asarray(radii, dtype=np.float64))
    return float(lr.mean()), float(max(lr.std(), 1e-3))


def _width_ratio(post_lo, post_hi, prior_lo, prior_hi) -> float:
    """Posterior interval width as a share of the prior's.

    Near 1.0 the data moved nothing: the parameter is unidentified and the
    "estimate" is the prior speaking back. This is the check that separates a
    reportable scale from a number the model was always going to return.
    """
    prior_w = float(prior_hi) - float(prior_lo)
    if not np.isfinite(prior_w) or prior_w <= 0:
        return float("nan")
    return float((float(post_hi) - float(post_lo)) / prior_w)


@dataclass
class IndexPosterior:
    weights: np.ndarray               # (draws, nC [+ pairs])
    beta: np.ndarray                  # (draws,)
    powers: np.ndarray | None
    channels: tuple[str, ...]
    picked: tuple
    form: str
    rhat_max: float
    ess_min: float
    divergences: int
    sigma: np.ndarray | None = None                # (draws,)
    radius_weights: np.ndarray | None = None       # (draws, nC, nR)
    peak_radius: np.ndarray | None = None          # (draws, nC), metres
    kernel_width: np.ndarray | None = None         # (draws, nC), log-metres
    aggregator_weights: np.ndarray | None = None   # (draws, nC, nStats)
    radii: tuple = ()
    stats: tuple = ()
    radius_kernel: str = "fixed"

    def weight_labels(self) -> list[str]:
        """One label per weight, so the pair terms are not read as channels.

        The synergy form carries a weight per channel *and* per channel pair;
        a reader given only the channel names would attribute a pair's weight
        to whichever channel happened to sit at that index.
        """
        labels = list(self.channels)
        if self.weights.shape[1] > len(self.channels):
            labels += [
                f"{self.channels[i]} x {self.channels[j]}"
                for i, j in _pairs(len(self.channels))
            ]
        return labels

    def partial_r2(self) -> np.ndarray:
        """Share of the residualised outcome variance the index explains.

        The outcome is standardised in :func:`prep`, so ``beta`` and ``sigma``
        are on the same scale and the ratio is the partial R² directly. In this
        literature values near 0.001 are the published effect size rather than
        a defect of the fit.
        """
        if self.sigma is None:
            return np.full(len(self.beta), np.nan)
        b2 = self.beta ** 2
        return b2 / (b2 + self.sigma ** 2)

    def informative_aggregators(self) -> list[list[str]]:
        """Per channel, the statistics the data actually had an opinion about.

        A component is informative when its credible interval excludes the
        prior mean ``1/K`` — in either direction, since "certainly not p90" is
        as much of a finding as "mostly p10".

        Interval *width* is the wrong test here, unlike for the radius. The
        prior marginal is ``Beta(1, K-1)``, which piles its mass near zero, so
        a posterior that correctly concentrates on one statistic near 0.5 comes
        out wider than the prior and reads as unidentified exactly when it is
        most informative.
        """
        if self.aggregator_weights is None:
            return []
        uniform = 1.0 / self.aggregator_weights.shape[2]
        lo, hi = np.percentile(self.aggregator_weights, [2.5, 97.5], axis=0)
        return [
            [str(self.stats[i]) for i in range(len(self.stats))
             if lo[c, i] > uniform or hi[c, i] < uniform]
            for c in range(len(self.channels))
        ]

    def projected_pick(self) -> tuple:
        """The single ``(radius, stat)`` per channel closest to the posterior.

        The blend is the estimate; this is its projection onto the one cell per
        channel the composite/apply path can carry, since that path takes one
        radius and one statistic rather than a distribution over them. It is
        named a projection so it is not read back as the model.

        Where the blend is flat the argmax is a coin flip between statistics the
        data could not tell apart — on collinear aggregators it lands on ``p10``
        or ``p90`` depending only on the seed. That is a pick manufactured out
        of noise, and it would be shipped downstream as though it were a
        finding, so a channel with no informative component falls back to the
        mean: the conventional default, and a stable one.
        """
        if self.radius_weights is None or self.aggregator_weights is None:
            return tuple(self.picked)
        rw = self.radius_weights.mean(0)
        aw = self.aggregator_weights.mean(0)
        informative = self.informative_aggregators()
        default = "mean" if "mean" in self.stats else None
        out = []
        for c in range(len(self.channels)):
            if informative[c] or default is None:
                stat = str(self.stats[int(np.argmax(aw[c]))])
            else:
                stat = default
            out.append((int(self.radii[int(np.argmax(rw[c]))]), stat))
        return tuple(out)

    def summary(self) -> dict:
        lo, hi = np.percentile(self.beta, [2.5, 97.5])
        w_lo, w_hi = np.percentile(self.weights, [2.5, 97.5], axis=0)
        w_prior = _dirichlet_prior_ci(self.weights.shape[1])
        pr2 = self.partial_r2()
        out = {
            "channels": list(self.channels),
            "weight_labels": self.weight_labels(),
            "picked": [list(p) for p in self.picked],
            "form": self.form,
            "weight_mean": self.weights.mean(0).tolist(),
            "weight_ci_low": np.atleast_1d(w_lo).tolist(),
            "weight_ci_high": np.atleast_1d(w_hi).tolist(),
            "weight_prior_ci": list(w_prior),
            "weight_width_ratio": [
                _width_ratio(a, b, *w_prior)
                for a, b in zip(np.atleast_1d(w_lo), np.atleast_1d(w_hi))
            ],
            "beta_mean": float(self.beta.mean()),
            "beta_ci_low": float(lo),
            "beta_ci_high": float(hi),
            "p_direction": float(max((self.beta > 0).mean(), (self.beta < 0).mean())),
            "partial_r2_mean": float(np.nanmean(pr2)),
            "partial_r2_ci_low": float(np.nanpercentile(pr2, 2.5)),
            "partial_r2_ci_high": float(np.nanpercentile(pr2, 97.5)),
            "rhat_max": self.rhat_max,
            "ess_min": self.ess_min,
            "divergences": self.divergences,
            "powers": None if self.powers is None else self.powers.mean(0).tolist(),
            "radius_kernel": self.radius_kernel,
        }

        if self.radius_weights is not None:
            out["radii"] = [int(r) for r in self.radii]
            out["radius_profile"] = self.radius_weights.mean(0).tolist()
            out["projected_pick"] = [list(p) for p in self.projected_pick()]
        if self.peak_radius is not None:
            center, spread = _radius_prior(self.radii)
            prior = (float(np.exp(center - 1.96 * spread)),
                     float(np.exp(center + 1.96 * spread)))
            p_lo, p_hi = np.percentile(self.peak_radius, [2.5, 97.5], axis=0)
            out["peak_radius_mean"] = self.peak_radius.mean(0).tolist()
            out["peak_radius_ci_low"] = np.atleast_1d(p_lo).tolist()
            out["peak_radius_ci_high"] = np.atleast_1d(p_hi).tolist()
            out["peak_radius_prior_ci"] = list(prior)
            out["peak_radius_width_ratio"] = [
                _width_ratio(a, b, *prior)
                for a, b in zip(np.atleast_1d(p_lo), np.atleast_1d(p_hi))
            ]
        if self.kernel_width is not None:
            out["kernel_width_mean"] = self.kernel_width.mean(0).tolist()
        if self.aggregator_weights is not None:
            a_lo, a_hi = np.percentile(self.aggregator_weights, [2.5, 97.5], axis=0)
            a_prior = _dirichlet_prior_ci(self.aggregator_weights.shape[2])
            out["stats"] = list(self.stats)
            out["aggregator_mean"] = self.aggregator_weights.mean(0).tolist()
            out["aggregator_ci_low"] = a_lo.tolist()
            out["aggregator_ci_high"] = a_hi.tolist()
            out["aggregator_prior_ci"] = list(a_prior)
            out["aggregator_uniform"] = 1.0 / self.aggregator_weights.shape[2]
            out["aggregator_informative"] = self.informative_aggregators()
            out["aggregator_width_ratio"] = [
                [_width_ratio(a, b, *a_prior) for a, b in zip(lo_c, hi_c)]
                for lo_c, hi_c in zip(a_lo, a_hi)
            ]
        return out


def fit(E, y, *, form="linear", radii=None, stats=None, radius_mask=None,
        radius_kernel="lognormal", aggregator="dirichlet",
        draws=800, warmup=800, chains=4, seed=42):
    """NUTS over every unknown the index has: scale, aggregation, weights, form.

    ``E`` is either ``(n, channel)`` — one already-chosen column per channel,
    which fixes the scale and the aggregator outside the model — or
    ``(n, channel, radius, stat)``, the whole grid, in which case the radius
    profile and the aggregator blend are sampled alongside the weights and
    their uncertainty flows into the interval on the effect.

    **The radius kernel is peaked, not decaying.** ``lognormal`` places the
    kernel in *log* radius with a sampled location and width, so the profile
    may rise and then fall — the shape held-out radius sweeps on this data
    actually show. That does not give up monotone decay: decay is the special
    case where the location sits at or below the smallest rung, so the peaked
    family contains the decaying one rather than replacing it. ``dirichlet``
    assumes no shape at all and pays for it in identification; ``fixed``
    spreads mass evenly over whatever rungs are unmasked.

    Each channel's contracted exposure is standardised before the weighted sum.
    Without that step a channel whose rungs disagree gets a lower-variance
    blend and therefore less influence at the same weight, which both breaks
    the reading of a weight as a share and ties the kernel width to the weight.
    """
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    numpyro.set_host_device_count(chains)
    E = np.asarray(E, dtype=np.float64)
    joint = E.ndim == 4
    nc = E.shape[1]
    npair = len(_pairs(nc)) if form == "synergy" else 0

    if joint:
        nr, ns = E.shape[2], E.shape[3]
        if radii is None or len(radii) != nr:
            raise ValueError("A (n, channel, radius, stat) grid needs `radii`.")
        mask = (np.ones((nc, nr), dtype=bool) if radius_mask is None
                else np.asarray(radius_mask, dtype=bool))
        # Off-ladder cells are NaN in the tensor. Zeroing them is only safe
        # because the mask below also zeroes their kernel weight, so they enter
        # neither the contraction nor its normaliser.
        E = np.where(mask[None, :, :, None], np.nan_to_num(E), 0.0)
        if nr == 1:
            radius_kernel = "fixed"
        center, spread = _radius_prior(radii)
        log_r = jnp.asarray(np.log(np.asarray(radii, dtype=np.float64)))
        maskj = jnp.asarray(mask.astype(np.float64))

    def contract(Ej):
        # The kernel is built in log space and normalised by subtracting its own
        # maximum. A location far from the ladder — which the sampler visits
        # freely whenever a channel carries no signal — otherwise drives every
        # rung to underflow, and the row of weights becomes all zeros instead of
        # a distribution, silently dropping the channel from those draws.
        if radius_kernel == "lognormal":
            mu = numpyro.sample(
                "mu", dist.Normal(center, spread).expand([nc]).to_event(1))
            sd = numpyro.sample(
                "kw",
                dist.LogNormal(math.log(spread), 0.75).expand([nc]).to_event(1))
            logk = -0.5 * ((log_r[None, :] - mu[:, None]) / sd[:, None]) ** 2
        elif radius_kernel == "dirichlet":
            kr = numpyro.sample(
                "kr", dist.Dirichlet(jnp.ones(nr)).expand([nc]).to_event(1))
            logk = jnp.log(kr + 1e-30)
        else:
            logk = jnp.zeros((nc, nr))
        logk = jnp.where(maskj > 0, logk, -1e30)
        logk = logk - jnp.max(logk, axis=-1, keepdims=True)
        k = jnp.exp(logk)
        k = numpyro.deterministic("radius_w", k / jnp.sum(k, axis=-1, keepdims=True))
        if aggregator == "dirichlet":
            a = numpyro.sample(
                "a", dist.Dirichlet(jnp.ones(ns)).expand([nc]).to_event(1))
        else:
            a = jnp.ones((nc, ns)) / ns
        v = jnp.einsum("ncrs,cr,cs->nc", Ej, k, a)
        return (v - v.mean(0)) / jnp.maximum(v.std(0), 1e-9)

    def model(Ej, yj):
        v = contract(Ej) if joint else Ej
        w = numpyro.sample("w", dist.Dirichlet(jnp.ones(nc + npair)))
        if form == "linear":
            e = v @ w
        else:
            p = numpyro.sample(
                "p", dist.Uniform(_POWER_LO, _POWER_HI).expand([nc]).to_event(1))
            lo, hi = v.min(0), v.max(0)
            z = jnp.clip((v - lo) / (hi - lo + 1e-12), 0.0, 1.0)
            e = (z ** p) @ w[:nc]
            for k, (i, j) in enumerate(_pairs(nc)):
                e = e + w[nc + k] * z[:, i] * z[:, j]
        e = (e - e.mean()) / jnp.maximum(e.std(), 1e-9)
        beta = numpyro.sample("beta", dist.Normal(0.0, 1.0))
        sigma = numpyro.sample("sigma", dist.HalfNormal(2.0))
        numpyro.sample("obs", dist.Normal(beta * e, sigma), obs=yj)

    # A channel with no signal leaves its kernel location flat under the prior,
    # and the sampler has to traverse that ridge without stepping off it, so the
    # joint model runs at a shorter step than the fixed-column one.
    mcmc = MCMC(NUTS(model, target_accept_prob=0.95 if joint else 0.9),
                num_warmup=warmup, num_samples=draws, num_chains=chains,
                progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), jnp.asarray(E), jnp.asarray(y),
             extra_fields=("diverging",))
    return mcmc


def posterior_from(mcmc, *, channels, picked, form, radii=(), stats=(),
                   radius_kernel="fixed") -> IndexPosterior:
    import arviz as az

    idata = az.from_numpyro(mcmc)
    s = mcmc.get_samples()
    # A parameter with no posterior variance has no convergence to diagnose:
    # a one-channel study's weight vector is the constant ``[1.0]``, so its
    # R-hat is 0/0. Left in, that one NaN propagates through the max and the
    # sampler-health gate stops firing without ever saying so.
    names = [v for v in ("w", "beta", "mu", "kw", "kr", "a")
             if v in idata.posterior
             and float(np.nanvar(np.asarray(idata.posterior[v]))) > 1e-24]

    def _finite(values) -> np.ndarray:
        a = np.asarray(values, dtype=np.float64).ravel()
        return a[np.isfinite(a)]

    with np.errstate(invalid="ignore", divide="ignore"):
        rhats = np.concatenate(
            [_finite(az.rhat(idata, var_names=[n])[n]) for n in names]
            or [np.array([])])
        esss = np.concatenate(
            [_finite(az.ess(idata, var_names=[n])[n]) for n in names]
            or [np.array([])])
    rhat = float(rhats.max()) if rhats.size else float("nan")
    ess = float(esss.min()) if esss.size else float("nan")
    post = IndexPosterior(
        weights=np.asarray(s["w"]), beta=np.asarray(s["beta"]),
        powers=np.asarray(s["p"]) if "p" in s else None,
        channels=tuple(channels), picked=tuple(picked), form=form,
        rhat_max=rhat, ess_min=ess,
        divergences=int(np.sum(mcmc.get_extra_fields()["diverging"])),
        sigma=np.asarray(s["sigma"]) if "sigma" in s else None,
        radius_weights=np.asarray(s["radius_w"]) if "radius_w" in s else None,
        peak_radius=np.exp(np.asarray(s["mu"])) if "mu" in s else None,
        kernel_width=np.asarray(s["kw"]) if "kw" in s else None,
        aggregator_weights=np.asarray(s["a"]) if "a" in s else None,
        radii=tuple(int(r) for r in radii), stats=tuple(stats),
        radius_kernel=radius_kernel,
    )
    # The grid is in the model, so the cell the composite is built from is the
    # posterior's own projection rather than a pick made before it ran.
    if post.radius_weights is not None and post.aggregator_weights is not None:
        post.picked = post.projected_pick()
    return post


# ────────────────────────────────────────────────────────────────────
# Reproducibility and honesty loops
# ────────────────────────────────────────────────────────────────────


def _discovery_shuffle(task):
    """Sweep on train, fit the winning form, score on the held-out rows."""
    Zpath, shape, y, cols_per_channel, forms, seed, frac = task
    z = np.memmap(Zpath, dtype=np.float64, mode="r", shape=shape)
    rng = np.random.default_rng(seed)
    te = rng.random(len(y)) < frac
    tr = ~te
    combos = list(itertools.product(*cols_per_channel))
    ztr = np.asarray(z[tr])
    g_tr, c_tr = ztr.T @ ztr, ztr.T @ y[tr]

    best, best_v = None, -np.inf
    for combo in combos:
        i = np.asarray(combo)
        fit_ = simplex_fit(g_tr[np.ix_(i, i)], c_tr[i])
        if fit_ is None:
            continue
        sub, w = fit_
        v = _tstat(ztr[:, i[sub]] @ w, y[tr])
        if v > best_v:
            best_v, best = v, combo
    if best is None:
        return None

    E = np.asarray(z[:, list(best)])
    out = {"columns": tuple(best)}
    for form in forms:
        apply_fn, params = build_index(E[tr], y[tr], form)
        out[form] = {
            "train_t": _tstat(apply_fn(E[tr]), y[tr]),
            "test_t": _tstat(apply_fn(E[te]), y[te]),
            "weights": np.asarray(params["weights"]).tolist(),
        }
    out["form"] = max(forms, key=lambda f: out[f]["test_t"])
    return out


def repeated_discovery(X, radii, stats, y, *, channels, channel_index,
                       radius_idx=None, forms=FORMS, reps=5, shuffles=12,
                       frac=0.25, seed=0, workers=None, cancel_check=None):
    """Independent repeats of sweep -> fit -> score.

    Each replicate gets its own seed stream, so agreement across replicates is
    reproducibility rather than one long correlated run.
    """
    n_radii, n_stats = X.shape[2], X.shape[3]
    if radius_idx is None:
        radius_idx = [list(range(n_radii))] * len(channels)
    cols = candidate_columns(channel_index, radius_idx, n_radii, n_stats)
    flat = np.ascontiguousarray(X.reshape(len(X), -1))

    fd, path = tempfile.mkstemp(suffix=".geofuse-disc")
    os.close(fd)
    try:
        mm = np.memmap(path, dtype=np.float64, mode="w+", shape=flat.shape)
        mm[:] = flat
        mm.flush()
        del mm
        tasks = [(path, flat.shape, y, cols, tuple(forms),
                  10_000 * (rep + 1) + b, frac)
                 for rep in range(reps) for b in range(shuffles)]
        n_workers = workers or parallel.process_worker_count(len(tasks))
        res = [r for r in _map(_discovery_shuffle, tasks, n_workers, cancel_check)
               if r is not None]
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    picks: dict = {}
    for r in res:
        key = (_decode(r["columns"], n_radii, n_stats, radii, stats), r["form"])
        picks[key] = picks.get(key, 0) + 1
    per_form = {
        f: {
            "train_t": float(np.mean([r[f]["train_t"] for r in res])),
            "test_t": float(np.mean([r[f]["test_t"] for r in res])),
            "weights_mean": np.mean([r[f]["weights"] for r in res], axis=0).tolist(),
            "weights_sd": np.std([r[f]["weights"] for r in res], axis=0).tolist(),
        }
        for f in forms
    }
    for f in per_form:
        per_form[f]["shrinkage"] = per_form[f]["train_t"] - per_form[f]["test_t"]
    return {
        "n_results": len(res), "reps": reps, "shuffles": shuffles,
        "pick_counts": {str(k): v for k, v in
                        sorted(picks.items(), key=lambda kv: -kv[1])},
        "distinct_picks": len(picks),
        "per_form": per_form,
    }


def _gain_split(task):
    """One split: every standalone and the composite, same held-out rows."""
    path, shape, y, cols, per_channel_cols, forms, seed, frac = task
    z = np.memmap(path, dtype=np.float64, mode="r", shape=shape)
    rng = np.random.default_rng(seed)
    te = rng.random(len(y)) < frac
    tr = ~te
    out = {}

    # One gather of the training rows, reused by every candidate below. The
    # memmap is disk-backed, so re-slicing it per candidate would re-read the
    # whole training block hundreds of thousands of times.
    ztr = np.asarray(z[tr])

    train_t = {}
    for ci, single in enumerate(per_channel_cols):
        best, best_v = None, -np.inf
        for col in single:
            v = _tstat(ztr[:, col], y[tr])
            if v > best_v:
                best_v, best = v, col
        e = np.asarray(z[:, best])
        out[f"ch{ci}"] = _tstat(e[te], y[te])
        train_t[f"ch{ci}"] = best_v

    combos = list(itertools.product(*cols))
    g_tr, c_tr = ztr.T @ ztr, ztr.T @ y[tr]
    best, best_v = None, -np.inf
    for combo in combos:
        i = np.asarray(combo)
        f_ = simplex_fit(g_tr[np.ix_(i, i)], c_tr[i])
        if f_ is None:
            continue
        sub, w = f_
        v = _tstat(ztr[:, i[sub]] @ w, y[tr])
        if v > best_v:
            best_v, best = v, combo
    E = np.asarray(z[:, list(best)])
    scores = {}
    for form in forms:
        apply_fn, _ = build_index(E[tr], y[tr], form)
        scores[form] = _tstat(apply_fn(E[te]), y[te])
    out["cgi"] = max(scores.values())

    # Best standalone chosen on TRAIN. Choosing it on the held-out scores
    # selects on the test rows and inflates the comparator.
    pick = max(train_t, key=lambda k: train_t[k])
    out["best_single"] = out[pick]
    out["picked_channel"] = pick
    out["gain"] = out["cgi"] - out["best_single"]
    out["gain_first"] = out["cgi"] - out["ch0"]
    return out


def holdout_gain(X, y, *, channels, channel_index, radius_idx=None,
                 forms=FORMS, splits=20, perm=100, frac=0.25, seed=0,
                 workers=None, cancel_check=None):
    """Composite vs each standalone, with a permutation null on the *gain*."""
    n_radii, n_stats = X.shape[2], X.shape[3]
    if radius_idx is None:
        radius_idx = [list(range(n_radii))] * len(channels)
    cols = candidate_columns(channel_index, radius_idx, n_radii, n_stats)
    flat = np.ascontiguousarray(X.reshape(len(X), -1))

    fd, path = tempfile.mkstemp(suffix=".geofuse-gain")
    os.close(fd)
    try:
        mm = np.memmap(path, dtype=np.float64, mode="w+", shape=flat.shape)
        mm[:] = flat
        mm.flush()
        del mm
        base = [(path, flat.shape, y, cols, cols, tuple(forms), seed + s, frac)
                for s in range(splits)]
        nulls = [(path, flat.shape,
                  np.random.default_rng(90_000 + i).permutation(y),
                  cols, cols, tuple(forms), 5_000 + i, frac)
                 for i in range(perm)]
        n_workers = workers or parallel.process_worker_count(len(base) + len(nulls))
        obs = _map(_gain_split, base, n_workers, cancel_check)
        null = _map(_gain_split, nulls, n_workers, cancel_check)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    g = np.array([r["gain"] for r in obs])
    gn = np.array([r["gain"] for r in null])
    out = {
        "cgi": float(np.mean([r["cgi"] for r in obs])),
        "best_single": float(np.mean([r["best_single"] for r in obs])),
        "gain": float(g.mean()),
        "gain_null_mean": float(gn.mean()),
        "gain_p": float((1 + int((gn >= g.mean()).sum())) / (1 + len(gn))),
        "gain_won": int((g > 0).sum()),
        "splits": splits,
    }
    for ci, name in enumerate(channels):
        out[f"standalone_{name}"] = float(np.mean([r[f"ch{ci}"] for r in obs]))
    return out


def _null_fit(task):
    """One permuted refit. Reads the exposure from a memmap, not from the pipe.

    The joint model is handed the whole grid, which is orders of magnitude
    larger than a column per channel; sending a copy to each of n workers is
    what exhausts the pipe rather than the fitting.
    """
    path, shape, y, form, seed, kwargs = task
    E = np.asarray(np.memmap(path, dtype=np.float64, mode="r", shape=shape))
    yp = np.random.default_rng(seed).permutation(y)
    m = fit(E, yp, form=form, draws=300, warmup=300, chains=2, seed=seed,
            **kwargs)
    b = np.asarray(m.get_samples()["beta"])
    lo, hi = np.percentile(b, [2.5, 97.5])
    return int(lo > 0 or hi < 0)


def null_calibration(E, y, *, form="linear", n=16, workers=None,
                     cancel_check=None, **fit_kwargs):
    """How often the beta interval excludes zero on a permuted outcome (~5 %).

    ``E`` and ``fit_kwargs`` are handed to :func:`fit` unchanged, so passing the
    full grid re-estimates the radius profile and the aggregator blend on every
    permuted refit. What that prices in is exactly what the model contains: with
    the grid inside, the scale and the aggregator are integrated over rather
    than selected, and this rate covers them. What it cannot cover is anything
    still decided outside — the functional form, the channel list, the covariate
    set — so a clean rate here is not a licence to read the interval as though
    those had been pre-specified.

    One fit per process: numpyro recompiles per call, so a serial loop pays the
    JAX compile n times over.
    """
    E = np.ascontiguousarray(np.asarray(E, dtype=np.float64))
    fd, path = tempfile.mkstemp(suffix=".geofuse-null")
    os.close(fd)
    try:
        mm = np.memmap(path, dtype=np.float64, mode="w+", shape=E.shape)
        mm[:] = E
        mm.flush()
        del mm
        tasks = [(path, E.shape, y, form, 3_000 + i, dict(fit_kwargs))
                 for i in range(n)]
        n_workers = workers or min(n, parallel.process_worker_count(n))
        hits = _map(_null_fit, tasks, n_workers, cancel_check)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return {"runs": n, "excluded_zero": int(sum(hits)),
            "rate": float(sum(hits)) / max(n, 1)}


def _selfcheck():
    """Planted signal on synthetic columns: the sweep must find the right one."""
    rng = np.random.default_rng(0)
    n, n_radii, n_stats = 1500, 4, 3
    X = rng.normal(size=(n, 2, n_radii, n_stats))
    truth = X[:, 1, 2, 1]
    y = 0.6 * truth + rng.normal(size=n)
    yr, Xr = prep(X, y, None)
    res = sweep(Xr, np.array([200.0, 400.0, 600.0, 800.0]),
                ["mean", "p50", "p90"], yr, channels=("a", "b"),
                channel_index=(0, 1), splits=6, workers=2)
    assert res.picked[1] == (600, "p50"), res.picked

    Z = Xr.reshape(n, -1)[:, list(res.columns)]
    kept, w = simplex_fit(Z.T @ Z, Z.T @ yr)
    assert abs(w.sum() - 1.0) < 1e-9, w
    assert np.all(w >= 0.0), w
    # The planted channel must carry the weight, not the noise channel.
    assert 1 in kept, kept
    print("selfcheck ok:", res.picked, res.form, round(res.score, 2),
          "weights", np.round(w, 3))


if __name__ == "__main__":
    _selfcheck()
