"""Data-driven CGI: sweep the spatial/aggregator/form grid, then fit a posterior.

Two stages, in this order:

1. :func:`sweep` — exhaustive, cross-validated grid over ``(radius, statistic)``
   per channel and over the functional form. Channel weights are **not**
   searched: for the linear form the simplex-constrained optimum has a closed
   form, found by active-set enumeration. Once covariates are projected out,
   every candidate exposure column already sits in the pre-aggregation cache,
   so a candidate is scored from small slices of a precomputed Gram matrix and
   the n-row data is never touched again.

2. :func:`fit` — NUTS at the picked columns, estimating only what needs an
   interval: the channel weights and the effect.

:func:`repeated_discovery` runs both stages over independent train/test
shuffles, because one split cannot show that a discovery generalises.

Selection reads the outcome, so a stage-2 interval on the same rows is
post-selection. Three guards, all exercised here: the sweep sees the train pool
only, the headline effect comes from rows it never saw, and
:func:`null_calibration` permutes the outcome and re-runs *both* stages so the
reported false-positive rate prices the sweep in.
"""

from __future__ import annotations

import itertools
import math
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

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


def _map(fn, tasks, n_workers: int):
    """``fn`` over ``tasks``, in a process pool unless one worker was asked for.

    A single worker runs in-process. Spawning costs more than it saves at that
    width, and it keeps the parallel loops usable in an interpreter where spawn
    is unavailable, which is what a caller passing ``workers=1`` usually wants.
    """
    if n_workers <= 1:
        return [fn(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        return list(ex.map(fn, tasks, chunksize=1))


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
    """Non-negative weights summing to one that maximise fit, exactly.

    Enumerates the active sets — interior, every face, every vertex — and
    solves each with a Lagrange multiplier. With two or three channels that is
    at most seven tiny solves, so there is no iterative solver and no tolerance
    to tune. Returns ``(kept_indices, weights)`` or ``None``.
    """
    k = len(xty)
    best, best_v = None, -np.inf
    for mask in range(1, 1 << k):
        idx = [i for i in range(k) if mask >> i & 1]
        g, c = gram[np.ix_(idx, idx)], xty[idx]
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
        v = float(w @ c) / math.sqrt(quad)
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
          objective=None, rescore_top=50):
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
        surface = np.array(_map(_sweep_split, tasks, n_workers))
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
                                     splits=splits, frac=frac, seed=seed)
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
                             workers=workers, objective=objective)
        form_scores["synergy"] = syn
        if syn > chosen_score:
            chosen_form, chosen_score = "synergy", syn

    return SweepResult(
        channels=tuple(channels), columns=tuple(combos[best]), picked=picked,
        form=chosen_form, score=chosen_score, surface=surface, combos=combos,
        one_se_columns=one_se, winner_counts=counts, boundary_hit=boundary,
        form_scores=form_scores,
    )


def _rescore(flat, y, combos, shortlist, objective, *, splits, frac, seed):
    """Score a shortlist of candidates with the job's objective, same splits.

    Weights still come from the closed-form simplex fit on the training rows;
    only the ranking changes. Runs in-process because a shortlist is small and
    the objective is a caller-owned closure, which does not pickle reliably.
    """
    out = np.zeros((splits, len(shortlist)))
    for si in range(splits):
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
                   objective=None):
    if objective is not None:
        return float(np.mean([
            _form_split_objective(E, y, form, seed + s, frac, objective)
            for s in range(splits)
        ]))
    tasks = [(E, y, form, seed + s, frac) for s in range(splits)]
    n_workers = workers or parallel.process_worker_count(len(tasks))
    return float(np.mean(_map(_form_split, tasks, n_workers)))


def build_index(E_train, y_train, form):
    """Fit the chosen form on training rows; return ``(apply_fn, params)``."""
    if form == "linear":
        fit = simplex_fit(E_train.T @ E_train, E_train.T @ y_train)
        if fit is None:
            k = E_train.shape[1]
            sub, w = np.arange(k), np.full(k, 1.0 / k)
        else:
            sub, w = fit
        return (lambda M: M[:, sub] @ w), {"weights": w, "kept": sub}
    sf = fit_synergy(E_train, y_train)
    return sf.apply, {"weights": sf.w, "powers": sf.p}


# ────────────────────────────────────────────────────────────────────
# Stage 2 — posterior at the picked columns
# ────────────────────────────────────────────────────────────────────


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

    def summary(self) -> dict:
        lo, hi = np.percentile(self.beta, [2.5, 97.5])
        w_lo, w_hi = np.percentile(self.weights, [2.5, 97.5], axis=0)
        return {
            "channels": list(self.channels),
            "weight_labels": self.weight_labels(),
            "picked": [list(p) for p in self.picked],
            "form": self.form,
            "weight_mean": self.weights.mean(0).tolist(),
            "weight_ci_low": np.atleast_1d(w_lo).tolist(),
            "weight_ci_high": np.atleast_1d(w_hi).tolist(),
            "beta_mean": float(self.beta.mean()),
            "beta_ci_low": float(lo),
            "beta_ci_high": float(hi),
            "p_direction": float(max((self.beta > 0).mean(), (self.beta < 0).mean())),
            "rhat_max": self.rhat_max,
            "ess_min": self.ess_min,
            "divergences": self.divergences,
            "powers": None if self.powers is None else self.powers.mean(0).tolist(),
        }


def fit(E, y, *, form="linear", draws=800, warmup=800, chains=4, seed=42):
    """NUTS over the channel weights (and synergy powers) at fixed columns."""
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    numpyro.set_host_device_count(chains)
    nc = E.shape[1]
    npair = len(_pairs(nc)) if form == "synergy" else 0
    lo, hi = E.min(0), E.max(0)

    def model(Ej, yj):
        w = numpyro.sample("w", dist.Dirichlet(jnp.ones(nc + npair)))
        if form == "linear":
            e = Ej @ w
        else:
            p = numpyro.sample(
                "p", dist.Uniform(_POWER_LO, _POWER_HI).expand([nc]).to_event(1))
            z = jnp.clip((Ej - lo) / (hi - lo + 1e-12), 0.0, 1.0)
            e = (z ** p) @ w[:nc]
            for k, (i, j) in enumerate(_pairs(nc)):
                e = e + w[nc + k] * z[:, i] * z[:, j]
        e = (e - e.mean()) / jnp.maximum(e.std(), 1e-9)
        beta = numpyro.sample("beta", dist.Normal(0.0, 1.0))
        sigma = numpyro.sample("sigma", dist.HalfNormal(2.0))
        numpyro.sample("obs", dist.Normal(beta * e, sigma), obs=yj)

    mcmc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=warmup,
                num_samples=draws, num_chains=chains, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), jnp.asarray(E), jnp.asarray(y),
             extra_fields=("diverging",))
    return mcmc


def posterior_from(mcmc, *, channels, picked, form) -> IndexPosterior:
    import arviz as az

    idata = az.from_numpyro(mcmc)
    names = [v for v in ("w", "beta") if v in idata.posterior]
    rhat = max(float(np.nanmax(np.asarray(az.rhat(idata, var_names=[n])[n])))
               for n in names)
    ess = min(float(np.nanmin(np.asarray(az.ess(idata, var_names=[n])[n])))
              for n in names)
    s = mcmc.get_samples()
    return IndexPosterior(
        weights=np.asarray(s["w"]), beta=np.asarray(s["beta"]),
        powers=np.asarray(s["p"]) if "p" in s else None,
        channels=tuple(channels), picked=tuple(picked), form=form,
        rhat_max=rhat, ess_min=ess,
        divergences=int(np.sum(mcmc.get_extra_fields()["diverging"])),
    )


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
                       frac=0.25, seed=0, workers=None):
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
        res = [r for r in _map(_discovery_shuffle, tasks, n_workers)
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
                 workers=None):
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
        obs = _map(_gain_split, base, n_workers)
        null = _map(_gain_split, nulls, n_workers)
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
    E, y, form, seed = task
    yp = np.random.default_rng(seed).permutation(y)
    m = fit(E, yp, form=form, draws=300, warmup=300, chains=2, seed=seed)
    b = np.asarray(m.get_samples()["beta"])
    lo, hi = np.percentile(b, [2.5, 97.5])
    return int(lo > 0 or hi < 0)


def null_calibration(E, y, *, form="linear", n=16, workers=None):
    """How often the beta interval excludes zero on a permuted outcome (~5 %).

    One fit per process: numpyro recompiles per call, so a serial loop pays the
    JAX compile n times over.
    """
    tasks = [(E, y, form, 3_000 + i) for i in range(n)]
    n_workers = workers or min(n, parallel.process_worker_count(n))
    hits = _map(_null_fit, tasks, n_workers)
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
