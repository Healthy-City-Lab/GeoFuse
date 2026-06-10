"""Diagnostics + interval estimation for the fusion engine.

Two concerns live here, both used by the stability-selection pipeline:

* **Multicollinearity diagnostics** — :func:`pairwise_pearson_matrix`,
  :func:`compute_vif`, and :func:`iterative_vif_reduction` back the optional
  channel-collinearity check that drops redundant channels (high VIF) before a
  run, pinning their weights to zero.
* **Bootstrap CI** — :func:`bootstrap_score_ci` resamples paired
  ``(target, prediction)`` rows with replacement and returns a percentile (or
  BCa) confidence interval on any score function, so the report can quote an
  effect size as ``r = 0.31 [0.22, 0.40]`` rather than a bare point estimate.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Multicollinearity diagnostics
# ---------------------------------------------------------------------------


def pairwise_pearson_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise Pearson correlation matrix of the columns of ``X``.

    NaN-safe (drops rows with any non-finite value before correlating).
    Returns a ``(p, p)`` symmetric matrix with ``1.0`` on the diagonal.
    A constant-variance column gets ``nan`` correlations with everything.
    """
    Xa = np.asarray(X, dtype=np.float64)
    if Xa.ndim != 2:
        raise ValueError(f"X must be 2-D; got shape {Xa.shape}.")
    mask = np.isfinite(Xa).all(axis=1)
    Xc = Xa[mask]
    if Xc.shape[0] < 3:
        return np.full((Xa.shape[1], Xa.shape[1]), np.nan)
    return np.corrcoef(Xc, rowvar=False)


def compute_vif(X: np.ndarray) -> np.ndarray:
    """Variance-Inflation Factor for every column of ``X``.

    ``VIF_j = 1 / (1 - R²_j)`` where ``R²_j`` is the coefficient of
    determination of OLS regressing column ``j`` on the rest of the
    columns (intercept included). Rule of thumb: ``VIF > 10`` is severe
    multicollinearity, ``VIF > 5`` is moderate. A column that's perfectly
    explained by the others returns ``+inf``; a column with zero variance
    returns ``nan``.

    Implementation note: uses ``np.linalg.lstsq`` rather than the
    statsmodels VIF helper so this module stays dependency-light.
    NaN rows are dropped jointly before regressing.
    """
    Xa = np.asarray(X, dtype=np.float64)
    if Xa.ndim != 2:
        raise ValueError(f"X must be 2-D; got shape {Xa.shape}.")
    n, p = Xa.shape
    if p < 2:
        return np.full(p, np.nan)
    mask = np.isfinite(Xa).all(axis=1)
    Xc = Xa[mask]
    if Xc.shape[0] < p + 1:
        return np.full(p, np.nan)

    out = np.empty(p, dtype=np.float64)
    for j in range(p):
        y = Xc[:, j]
        others = np.delete(Xc, j, axis=1)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        if ss_tot <= 0:
            out[j] = float("nan")
            continue
        design = np.column_stack([np.ones(len(y)), others])
        try:
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            yhat = design @ beta
            ss_res = float(((y - yhat) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot
        except np.linalg.LinAlgError:
            r2 = 0.0
        # Numerical guard so ``1 - r2`` stays positive even when ``r2``
        # spikes microscopically above 1 due to floating point.
        denom = max(1.0 - r2, 1e-12)
        out[j] = 1.0 / denom
    return out


def iterative_vif_reduction(
    X: np.ndarray,
    names: list[str],
    *,
    vif_threshold: float = 10.0,
) -> dict:
    """Drop one column at a time (highest VIF first) until all remaining
    VIFs fall at or below ``vif_threshold``, or only one column is left.

    Why iterative: dropping a single highly-redundant column often pulls
    the rest of the VIFs down dramatically, because that column was
    inflating its neighbours' regressions. Computing VIFs once and
    dropping every column above threshold simultaneously over-prunes.

    Returns a dict with:

    * ``kept`` — list of column names that survived.
    * ``dropped`` — list of column names removed, in the order they were
      dropped.
    * ``initial_vifs`` — VIFs computed on the full input.
    * ``final_vifs`` — VIFs after the last drop (one per kept column).
    * ``pearson_matrix`` — pairwise Pearson r on the input columns (for
      reporting; not used in the decision rule).
    * ``iterations`` — list of per-iteration records, each
      ``{"removed", "vifs_before", "vifs_after"}``.

    No-op behaviour: if every initial VIF is at or below threshold (or
    there's only one column), ``dropped`` is empty and ``kept == names``.
    """
    if len(names) != X.shape[1]:
        raise ValueError(
            f"len(names)={len(names)} must equal X.shape[1]={X.shape[1]}."
        )
    initial_vifs = compute_vif(X)
    pearson = pairwise_pearson_matrix(X)
    if X.shape[1] < 2:
        return {
            "kept": list(names),
            "dropped": [],
            "initial_vifs": initial_vifs.tolist(),
            "final_vifs": initial_vifs.tolist(),
            "pearson_matrix": pearson.tolist(),
            "iterations": [],
        }

    active_idx = list(range(X.shape[1]))
    iterations: list[dict] = []

    while len(active_idx) > 1:
        sub = X[:, active_idx]
        vifs = compute_vif(sub)
        # Treat NaN as "below threshold" so a constant-variance column
        # doesn't trigger an infinite loop, but it's still reported.
        worst_vif = float(np.nanmax(vifs))
        if not np.isfinite(worst_vif) or worst_vif > vif_threshold:
            # Drop the offender. ``argmax`` on the raw (possibly-inf)
            # array picks the largest; ties resolve to the leftmost.
            worst_pos = int(np.nanargmax(vifs))
            removed_global = active_idx[worst_pos]
            iterations.append(
                {
                    "removed": names[removed_global],
                    "vifs_before": vifs.tolist(),
                }
            )
            active_idx.pop(worst_pos)
            # Tag the post-drop VIFs so the caller can see the effect.
            after_vifs = compute_vif(X[:, active_idx]) if len(active_idx) > 1 else (
                np.array([1.0])
            )
            iterations[-1]["vifs_after"] = after_vifs.tolist()
            if not (np.isfinite(np.nanmax(after_vifs)) and
                    float(np.nanmax(after_vifs)) > vif_threshold):
                break
        else:
            break

    kept_names = [names[i] for i in active_idx]
    dropped_names = [it["removed"] for it in iterations]
    final_vifs = (
        compute_vif(X[:, active_idx]) if len(active_idx) > 1
        else np.array([1.0] * len(active_idx))
    )
    return {
        "kept": kept_names,
        "dropped": dropped_names,
        "initial_vifs": initial_vifs.tolist(),
        "final_vifs": final_vifs.tolist(),
        "pearson_matrix": pearson.tolist(),
        "iterations": iterations,
    }


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def bootstrap_score_ci(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    score_fn,
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    method: str = "BCa",
    seed: int = 42,
    covariates: np.ndarray | None = None,
) -> dict:
    """Bootstrap CI on ``score_fn(target, prediction)``.

    Resamples paired (target, prediction) rows with replacement, recomputes
    the score per resample, and returns a CI. ``method`` chooses how the
    interval is constructed:

    * ``"percentile"`` — the empirical 2.5 / 97.5 quantiles of the
      bootstrap distribution.
    * ``"BCa"`` (default) — bias-corrected and accelerated bootstrap. Uses
      the bias correction ``z₀`` derived from the share of bootstrap
      replicates below the observed score, and the acceleration ``a`` from
      a jackknife estimate of the score's third-moment skewness. Adjusts
      the percentile cutoffs so the CI corrects for both bias and
      asymmetry in the bootstrap distribution. This matches the reference
      notebook's CI rule.

    Args:
        target: 1-D float array of outcomes.
        prediction: 1-D float array of model output.
        score_fn: ``score_fn(target, prediction)`` when ``covariates`` is ``None``,
            else ``score_fn(target, prediction, covariates)``.
        n_bootstrap: number of resamples; 1000-10000 is typical (BCa
            tolerates the lower end because it corrects for tail bias).
        ci_level: e.g. ``0.95`` for a 95 % CI.
        method: ``"BCa"`` (default) or ``"percentile"``.
        seed: RNG seed.
        covariates: optional ``(n, n_cov)`` matrix paired row-wise with
            ``target`` / ``prediction``. Resampled with the **same** indices on
            every bootstrap iteration so the partial-correlation interpretation
            is preserved (covariates are a property of the observation, not the
            score). ``score_fn`` must accept the 3-arg form when this is set.

    Returns:
        Dict with ``observed``, ``mean``, ``lower``, ``upper``, ``ci_level``,
        ``method``.
    """
    from scipy import stats as _scistats

    t = np.asarray(target, dtype=np.float64).ravel()
    p = np.asarray(prediction, dtype=np.float64).ravel()

    cov_arr: np.ndarray | None = None
    if covariates is not None:
        cov_arr = np.asarray(covariates, dtype=np.float64)
        if cov_arr.ndim == 1:
            cov_arr = cov_arr.reshape(-1, 1)
        if cov_arr.shape[0] != t.shape[0]:
            raise ValueError(
                "covariates row count must match target/prediction length; "
                f"got {cov_arr.shape[0]} vs {t.shape[0]}."
            )

    mask = ~(np.isnan(t) | np.isnan(p))
    if cov_arr is not None:
        mask &= np.isfinite(cov_arr).all(axis=1)
    t = t[mask]
    p = p[mask]
    if cov_arr is not None:
        cov_arr = cov_arr[mask]
    if len(t) < 3:
        return {
            "observed": float("nan"),
            "mean": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
            "ci_level": float(ci_level),
            "method": method,
        }

    def _score(target_arr: np.ndarray, pred_arr: np.ndarray, sel: np.ndarray | None) -> float:
        if cov_arr is None:
            return float(score_fn(target_arr, pred_arr))
        sub = cov_arr if sel is None else cov_arr[sel]
        return float(score_fn(target_arr, pred_arr, sub))

    observed = _score(t, p, None)
    rng = np.random.default_rng(seed)
    n = len(t)
    scores = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            scores[i] = _score(t[idx], p[idx], idx)
        except Exception:
            scores[i] = np.nan

    valid = scores[~np.isnan(scores)]
    if len(valid) == 0:
        return {
            "observed": observed,
            "mean": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
            "ci_level": float(ci_level),
            "method": method,
        }

    alpha = (1.0 - ci_level) / 2.0
    method_norm = (method or "BCa").lower()

    if method_norm == "bca":
        # Bias correction: z0 maps the fraction of bootstrap replicates
        # below the observed score back to a standard normal. A bootstrap
        # distribution that's symmetric around the observed value yields
        # z0 ≈ 0 and BCa reduces toward the percentile CI.
        n_below = int(np.sum(valid < observed))
        prop_below = n_below / len(valid)
        prop_below = min(max(prop_below, 1e-6), 1.0 - 1e-6)
        z0 = float(_scistats.norm.ppf(prop_below))

        # Acceleration: jackknife estimate of the score's skewness.
        # Captures how the variance of the score depends on the data —
        # without it, BCa collapses to a bias-corrected interval that
        # under-covers when the score is skewed (correlation near ±1, etc.).
        jack_scores = np.empty(n, dtype=np.float64)
        all_idx = np.arange(n)
        for i in range(n):
            keep = np.delete(all_idx, i)
            try:
                jack_scores[i] = _score(t[keep], p[keep], keep)
            except Exception:
                jack_scores[i] = np.nan
        jack_valid = jack_scores[~np.isnan(jack_scores)]
        if len(jack_valid) >= 2:
            jack_mean = float(np.mean(jack_valid))
            num = float(np.sum((jack_mean - jack_valid) ** 3))
            denom = 6.0 * (float(np.sum((jack_mean - jack_valid) ** 2)) ** 1.5)
            accel = num / denom if denom != 0.0 else 0.0
        else:
            accel = 0.0

        z_lo = float(_scistats.norm.ppf(alpha))
        z_hi = float(_scistats.norm.ppf(1.0 - alpha))
        denom_lo = 1.0 - accel * (z0 + z_lo)
        denom_hi = 1.0 - accel * (z0 + z_hi)
        if denom_lo == 0.0:
            denom_lo = 1e-9
        if denom_hi == 0.0:
            denom_hi = 1e-9
        p_lo = float(_scistats.norm.cdf(z0 + (z0 + z_lo) / denom_lo))
        p_hi = float(_scistats.norm.cdf(z0 + (z0 + z_hi) / denom_hi))
        # Numerical guards so a degenerate jackknife (constant score)
        # doesn't push the percentiles outside [0, 1].
        p_lo = min(max(p_lo, 1e-6), 1.0 - 1e-6)
        p_hi = min(max(p_hi, 1e-6), 1.0 - 1e-6)
        lower = float(np.quantile(valid, p_lo))
        upper = float(np.quantile(valid, p_hi))
    else:
        lower = float(np.quantile(valid, alpha))
        upper = float(np.quantile(valid, 1.0 - alpha))

    return {
        "observed": observed,
        "mean": float(np.mean(valid)),
        "lower": lower,
        "upper": upper,
        "ci_level": float(ci_level),
        "method": "BCa" if method_norm == "bca" else "percentile",
    }


# ---------------------------------------------------------------------------
# Permutation significance
# ---------------------------------------------------------------------------


def permutation_pvalue(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    score_fn,
    higher_is_better: bool = True,
    n_perm: int = 2000,
    seed: int = 42,
    covariates: np.ndarray | None = None,
) -> dict:
    """One-sided permutation p-value for ``score_fn(target, prediction)``.

    Shuffles ``prediction`` against the fixed ``target`` (and covariates, which
    stay row-aligned with ``target``) ``n_perm`` times to build the null
    distribution of the score under no association, then reports the add-one
    smoothed fraction of null scores at least as extreme as the observed one.
    ``higher_is_better`` sets the tail: the upper tail for correlation-style
    metrics, the lower tail for error-style metrics. When ``covariates`` is
    given, ``score_fn`` is called in its 3-argument form and the p-value
    reflects the covariate-adjusted score the scorer computes.

    Returns a dict with ``observed``, ``p_value``, ``n_perm`` (effective),
    ``null_mean``, and ``higher_is_better``.
    """
    t = np.asarray(target, dtype=np.float64).ravel()
    p = np.asarray(prediction, dtype=np.float64).ravel()

    cov_arr: np.ndarray | None = None
    if covariates is not None:
        cov_arr = np.asarray(covariates, dtype=np.float64)
        if cov_arr.ndim == 1:
            cov_arr = cov_arr.reshape(-1, 1)

    mask = ~(np.isnan(t) | np.isnan(p))
    if cov_arr is not None:
        mask &= np.isfinite(cov_arr).all(axis=1)
    t = t[mask]
    p = p[mask]
    if cov_arr is not None:
        cov_arr = cov_arr[mask]

    nan_result = {
        "observed": float("nan"),
        "p_value": float("nan"),
        "n_perm": 0,
        "null_mean": float("nan"),
        "higher_is_better": bool(higher_is_better),
    }
    if len(t) < 3:
        return nan_result

    def _score(pred_arr: np.ndarray) -> float:
        if cov_arr is None:
            return float(score_fn(t, pred_arr))
        return float(score_fn(t, pred_arr, cov_arr))

    observed = _score(p)
    if not np.isfinite(observed):
        return nan_result

    rng = np.random.default_rng(seed)
    n = len(t)
    null = np.empty(int(n_perm), dtype=np.float64)
    for i in range(int(n_perm)):
        idx = rng.permutation(n)
        try:
            null[i] = _score(p[idx])
        except Exception:
            null[i] = np.nan

    valid = null[np.isfinite(null)]
    if len(valid) == 0:
        return {**nan_result, "observed": observed}

    if higher_is_better:
        count = int(np.sum(valid >= observed))
    else:
        count = int(np.sum(valid <= observed))
    p_value = (count + 1) / (len(valid) + 1)
    return {
        "observed": float(observed),
        "p_value": float(p_value),
        "n_perm": int(len(valid)),
        "null_mean": float(np.mean(valid)),
        "higher_is_better": bool(higher_is_better),
    }


# ---------------------------------------------------------------------------
# Stability-selection threshold calibration (Bodinier et al., 2023)
# ---------------------------------------------------------------------------


def calibrate_stability_selection(
    per_resample_rankings: list,
    n_candidates: int,
    *,
    pi_min: float = 0.5,
) -> dict | None:
    """Automated calibration of the stability-selection threshold.

    Adapts the calibration of Bodinier et al. (2023) — chosen instead of a
    hand-set selection threshold. Each entry of ``per_resample_rankings`` is a
    sequence of candidate ids ordered best→worst for one resample. For a grid
    of selection sizes ``K`` (the top-K candidates counted as "selected" per
    resample — the sparsity analogue of the regularisation λ) and thresholds
    ``π``, the stability score ``S = −log L`` is computed, where ``L`` is the
    likelihood of the observed stably-selected / unstable / stably-excluded
    split under the null that every candidate is selected with the same
    probability ``γ = K / N`` each resample (``N = n_candidates``). Each
    candidate's selection count over ``B`` resamples is ``Binomial(B, γ)`` under
    the null; a candidate is **stably selected** when its count ``≥ ⌈Bπ⌉``,
    **stably excluded** when ``≤ ⌊B(1−π)⌋``, and unstable in between. The
    ``(K, π)`` maximising ``S`` is the calibrated configuration.

    Returns a dict with ``K``, ``pi``, ``score``, ``gamma``, ``n_candidates``,
    ``n_resamples``, ``n_stably_selected``, ``selection_counts`` (id → count at
    ``K``), and ``pfer`` — the Meinshausen–Bühlmann per-family error-rate upper
    bound ``E[V] ≤ K² / ((2π−1)·N)`` (rigorous under ⌊n/2⌋ subsampling; an
    approximate guide under bootstrap resampling). Returns ``None`` when there
    are too few candidates or resamples to calibrate.
    """
    import math

    from scipy.stats import binom

    rankings = [list(r) for r in per_resample_rankings if len(r) > 0]
    B = len(rankings)
    N = int(n_candidates)
    if B < 3 or N < 2:
        return None

    k_max = min(max(len(r) for r in rankings), N)
    best: dict | None = None
    for K in range(1, k_max + 1):
        counts: dict = {}
        for r in rankings:
            for cid in r[:K]:
                counts[cid] = counts.get(cid, 0) + 1
        # Selection counts over all N candidates (those never in any top-K
        # contribute a 0, i.e. stably excluded under the null).
        h_vals = np.array(list(counts.values()), dtype=np.int64)
        gamma = min(1.0, K / N)
        realized = sorted({h / B for h in h_vals if h / B > pi_min})
        for pi in realized:
            hi = math.ceil(pi * B)
            lo = math.floor((1.0 - pi) * B)
            n_ss = int(np.sum(h_vals >= hi))
            if n_ss == 0:
                continue
            n_se = int(N - np.sum(h_vals > lo))  # everything not above lo
            n_se = max(0, n_se)
            n_us = max(0, N - n_ss - n_se)
            p_high = max(float(binom.sf(hi - 1, B, gamma)), 1e-300)
            p_low = max(float(binom.cdf(lo, B, gamma)), 1e-300)
            p_mid = max(1.0 - p_high - p_low, 1e-300)
            score = -(
                n_ss * math.log(p_high)
                + n_se * math.log(p_low)
                + n_us * math.log(p_mid)
            )
            if best is None or score > best["score"]:
                pfer = (
                    (K * K) / ((2.0 * pi - 1.0) * N)
                    if pi > 0.5
                    else float("inf")
                )
                best = {
                    "K": int(K),
                    "pi": float(pi),
                    "score": float(score),
                    "gamma": float(gamma),
                    "n_candidates": int(N),
                    "n_resamples": int(B),
                    "n_stably_selected": int(n_ss),
                    "selection_counts": dict(counts),
                    "pfer": float(pfer),
                }
    return best
