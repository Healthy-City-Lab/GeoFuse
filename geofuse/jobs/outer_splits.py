"""Repeated held-out evaluation: the whole tuning redone on other test splits.

One held-out split can decide a weak association: equally valid test sets give
held-out scores several-fold apart. With ``outer_splits = K`` the fusion runner
keeps its first split (seed ``split_seed``) and repeats split → tuning →
held-out scoring for seeds ``split_seed + 1 … split_seed + K − 1``. Each split
gets its own sweep and posterior (no reproducibility, gain or null loops: those
stay on the first split), the comparison channel chosen on that split's
train+val rows, the paired differences and the negative-control slopes.

:func:`split_rows` turns one split's results into rows of
``(quantity, exposure)``; :func:`summarise` reads the K rows of each quantity
as a spread (median, 2.5 / 97.5 %, min / max), the share of splits where the
per-split result holds, and a Nadeau–Bengio corrected resampled t
(:func:`geofuse.statistical_testing.corrected_resampled_t`), since the splits
share data and their p-values cannot be combined naively. The held-out score
is unsigned for several objectives (``r2`` is an incremental R²), so it is
summarised but not tested; the signed slope ``held_out_beta`` carries the
across-split test of the association.

Each split's CGI posterior is reported too (:func:`posterior_rows`): R50 and
R90 per channel, the weight of every term, β and the chosen form. They are
fitted on that split's train+val rows, so they are summarised as a spread,
with the share of splits in which each term carries the largest weight, and
not tested.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Quantities, in report order, and the alternative each is tested against.
QUANTITIES: dict[str, str | None] = {
    "held_out_score": None,
    "held_out_beta": "two-sided",
    "cgi_minus_channel": "greater",
    "cgi_minus_comparator": "greater",
    "negative_control_beta": "two-sided",
    "posterior_beta": None,
    "posterior_weight": None,
    "posterior_r50": None,
    "posterior_r90": None,
}

# Posterior quantities: summarised only, with no share holding.
POSTERIOR_QUANTITIES = frozenset(q for q in QUANTITIES if q.startswith("posterior_"))

OUTER_COMPARATORS = ("primary", "all")

_ALPHA = 0.05


def _f(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _slope_p(slope: dict) -> float | None:
    """Two-sided p-value of a :func:`negative_controls.partial_slope` row."""
    from scipy import stats

    t, df = _f(slope.get("t")), _f(slope.get("df"))
    if t is None or df is None or df <= 0:
        return None
    return float(2 * stats.t.sf(abs(t), df))


def _row(quantity: str, exposure: str, **values) -> dict:
    return {
        "quantity": quantity,
        "exposure": exposure,
        "estimate": _f(values.get("estimate")),
        "ci_low": _f(values.get("ci_low")),
        "ci_high": _f(values.get("ci_high")),
        "p_value": _f(values.get("p_value")),
    }


def split_info(engine: Any) -> dict:
    """Test rows, blocks and share of the engine's current split."""
    key = "polygon_id"
    test, pool = engine.test_data, engine.train_val_data
    n_test = int(test[key].nunique()) if key in test.columns else int(len(test))
    n_pool = int(pool[key].nunique()) if key in pool.columns else int(len(pool))
    report = getattr(engine, "_spatial_split_report", None) or {}
    return {
        "n_test": n_test,
        "n_train_val": n_pool,
        "test_share": n_test / max(n_test + n_pool, 1),
        "test_blocks": report.get("test_blocks"),
    }


def split_rows(
    *,
    effects_test: dict | None,
    slope_test: dict | None,
    channel_paired: dict | None,
    comparator_paired: list[dict],
    negative_controls: dict | None,
    posterior: dict | None = None,
) -> list[dict]:
    """One split's quantities as ``(quantity, exposure)`` rows.

    ``effects_test`` is ``evaluate_effects(...)["test"]``, ``slope_test``
    ``compute_partial_slopes(...)["test"]``, ``channel_paired`` the paired row
    against the channel chosen on the split's train+val rows,
    ``comparator_paired`` the paired rows of the comparators scored on every
    split (:func:`scored_comparators`), ``negative_controls`` the
    :meth:`compute_negative_controls` report, ``posterior`` the CGI's
    ``_posterior_summary`` (:func:`posterior_rows`).
    """
    rows: list[dict] = []
    if effects_test:
        rows.append(
            _row(
                "held_out_score",
                "cgi",
                estimate=effects_test.get("score"),
                ci_low=effects_test.get("lower"),
                ci_high=effects_test.get("upper"),
                p_value=effects_test.get("p_value"),
            )
        )
    if slope_test:
        rows.append(
            _row(
                "held_out_beta",
                "cgi",
                estimate=slope_test.get("beta"),
                ci_low=slope_test.get("ci_low"),
                ci_high=slope_test.get("ci_high"),
                p_value=_slope_p(slope_test),
            )
        )
    if channel_paired:
        rows.append(
            _row(
                "cgi_minus_channel",
                str(channel_paired.get("standalone_channel")),
                estimate=channel_paired.get("observed_diff"),
                ci_low=channel_paired.get("lower"),
                ci_high=channel_paired.get("upper"),
                p_value=channel_paired.get("p_value"),
            )
        )
    for r in comparator_paired:
        rows.append(
            _row(
                "cgi_minus_comparator",
                str(r.get("comparator")),
                estimate=r.get("observed_diff"),
                ci_low=r.get("lower"),
                ci_high=r.get("upper"),
                p_value=r.get("p_value"),
            )
        )
    test = ((negative_controls or {}).get("splits") or {}).get("test") or {}
    for name, block in (test.get("controls") or {}).items():
        rows.append(
            _row(
                "negative_control_beta",
                str(name),
                estimate=block.get("beta"),
                ci_low=block.get("ci_low"),
                ci_high=block.get("ci_high"),
                p_value=_slope_p(block),
            )
        )
    rows += posterior_rows(posterior)
    return rows


def posterior_rows(summary: dict | None) -> list[dict]:
    """One split's CGI posterior as ``posterior_*`` rows.

    ``summary`` is ``_posterior_summary(params)``. ``posterior_beta`` for the
    composite, ``posterior_weight`` for every term (the synergy form's pair
    terms included), ``posterior_r50`` / ``posterior_r90`` for every channel
    with a radius grid: the posterior mean and credible interval under the
    names ``discovery.csv`` uses, each row carrying the split's ``form``.
    """
    if not summary:
        return []

    def at(key: str, i: int):
        values = summary.get(key) or []
        return values[i] if i < len(values) else None

    names = summary.get("weight_labels") or summary.get("channels") or []
    labels = [str(x) for x in names]
    rows: list[dict] = []
    if _f(summary.get("beta_mean")) is not None:
        rows.append(
            _row(
                "posterior_beta",
                "cgi",
                estimate=summary.get("beta_mean"),
                ci_low=summary.get("beta_ci_low"),
                ci_high=summary.get("beta_ci_high"),
            )
        )
    for i, name in enumerate(labels):
        rows.append(
            _row(
                "posterior_weight",
                name,
                estimate=at("weight_mean", i),
                ci_low=at("weight_ci_low", i),
                ci_high=at("weight_ci_high", i),
            )
        )
    n_grid = len(summary.get("radius_profile") or [])
    for quantity, key in (("posterior_r50", "r50"), ("posterior_r90", "r90")):
        for i, name in enumerate(labels[:n_grid]):
            if _f(at(f"{key}_mean", i)) is None:
                continue
            rows.append(
                _row(
                    quantity,
                    name,
                    estimate=at(f"{key}_mean", i),
                    ci_low=at(f"{key}_ci_low", i),
                    ci_high=at(f"{key}_ci_high", i),
                )
            )
    for r in rows:
        r["form"] = summary.get("form")
    return rows


def _holds(row: dict) -> bool | None:
    """Whether one split's result holds: its p-value below 5 %, else an
    interval that excludes zero. ``None`` when it has neither."""
    if row.get("p_value") is not None:
        return row["p_value"] < _ALPHA
    lo, hi = row.get("ci_low"), row.get("ci_high")
    if lo is None or hi is None:
        return None
    return lo > 0 or hi < 0


def summarise(rows: list[dict]) -> list[dict]:
    """Per ``(quantity, exposure)``: spread, share holding, corrected t.

    ``rows`` carry ``split`` and ``test_share`` next to the :func:`split_rows`
    fields. The corrected t uses the mean realised test share. The primary
    comparators form one Holm family on the corrected p-value, as on a single
    split; the comparison channel is one test. Posterior quantities get the
    spread only; each term's weight entry adds ``share_largest`` (the share of
    splits in which that term has the largest weight) and the β entry
    ``forms`` (how often each form was chosen).
    """
    from geofuse.statistical_testing import corrected_resampled_t, holm_bonferroni

    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["quantity"], r["exposure"]), []).append(r)
    order = list(QUANTITIES)
    out: list[dict] = []
    for (quantity, exposure), group in sorted(
        groups.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1])
    ):
        est = np.array(
            [r["estimate"] for r in group if r["estimate"] is not None], dtype=float
        )
        held = [
            h
            for h in (_holds(r) for r in group)
            if h is not None and quantity not in POSTERIOR_QUANTITIES
        ]
        shares = [r["test_share"] for r in group if r.get("test_share") is not None]
        entry = {
            "quantity": quantity,
            "exposure": exposure,
            "n_splits": len(group),
            "median": float(np.median(est)) if est.size else None,
            "q025": float(np.quantile(est, 0.025)) if est.size else None,
            "q975": float(np.quantile(est, 0.975)) if est.size else None,
            "min": float(est.min()) if est.size else None,
            "max": float(est.max()) if est.size else None,
            "share_holds": float(np.mean(held)) if held else None,
            "corrected_t": None,
            "corrected_p": None,
            "corrected_p_holm": None,
            "alternative": QUANTITIES.get(quantity),
            "primary": group[0].get("primary"),
        }
        if entry["alternative"] and shares:
            test = corrected_resampled_t(
                est, float(np.mean(shares)), alternative=entry["alternative"]
            )
            entry["corrected_t"], entry["corrected_p"] = test["t"], test["p_value"]
        out.append(entry)
    family = [
        e
        for e in out
        if e["quantity"] == "cgi_minus_comparator" and e.get("primary") is not False
    ]
    for e, hp in zip(
        family, holm_bonferroni([e["corrected_p"] for e in family]), strict=True
    ):
        e["corrected_p_holm"] = None if hp != hp else float(hp)
    _posterior_shares(rows, out)
    return out


def _posterior_shares(rows: list[dict], summary: list[dict]) -> None:
    """``share_largest`` on each weight entry and ``forms`` on the β entry."""
    weights: dict = {}
    forms: dict = {}
    for r in rows:
        if r["quantity"] not in POSTERIOR_QUANTITIES:
            continue
        forms[r.get("split")] = r.get("form")
        if r["quantity"] == "posterior_weight" and r["estimate"] is not None:
            weights.setdefault(r.get("split"), []).append(r)
    top = [max(g, key=lambda r: r["estimate"])["exposure"] for g in weights.values()]
    counts: dict = {}
    for form in forms.values():
        counts[form] = counts.get(form, 0) + 1
    for e in summary:
        if e["quantity"] == "posterior_weight":
            e["share_largest"] = top.count(e["exposure"]) / len(top) if top else None
        elif e["quantity"] == "posterior_beta" and forms:
            e["forms"] = ", ".join(
                f"{form} {n}/{len(forms)}"
                for form, n in sorted(counts.items(), key=lambda kv: -kv[1])
            )


def describe_posterior(summary: list[dict]) -> str:
    """The posterior entries of a :func:`summarise` result as one line:
    β with the forms chosen, each term's weight with its share largest, and
    R50 / R90 per channel, each as median [min, max]."""

    def spread(e: dict, spec: str, unit: str = "") -> str:
        return (
            f"{format(e['median'], spec)}{unit} "
            f"[{format(e['min'], spec)}, {format(e['max'], spec)}]"
        )

    parts: list[str] = []
    for e in summary:
        if e["quantity"] not in POSTERIOR_QUANTITIES or e.get("median") is None:
            continue
        if e["quantity"] == "posterior_beta":
            text = f"β {spread(e, '+.4f')}"
            parts.append(text + (f" (forms {e['forms']})" if e.get("forms") else ""))
        elif e["quantity"] == "posterior_weight":
            share = e.get("share_largest")
            parts.append(
                f"weight {e['exposure']} {spread(e, '.3f')}"
                + (f", largest in {share:.0%}" if share is not None else "")
            )
        else:
            name = "R50" if e["quantity"] == "posterior_r50" else "R90"
            parts.append(f"{name} {e['exposure']} {spread(e, '.0f', ' m')}")
    return "; ".join(parts)


def scored_comparators(columns: list[str], primary: list[str], mode: str) -> list[str]:
    """The comparators scored on every split: all of them with ``mode ==
    "all"``, else the primary ones (all of them when none is named primary).
    The first split keeps the full set in its own report."""
    if mode == "all" or not primary:
        return list(columns)
    return [c for c in columns if c in set(primary)]


def repeat(
    engine: Any,
    *,
    k_splits: int,
    split_seed: int,
    split_kwargs: dict,
    fit_kwargs: dict,
    metric: str,
    standalones: list[str],
    comparators: list[str],
    comparators_primary: list[str],
    outer_comparators: str,
    n_effects_boot: int,
    n_perm: int,
    n_paired: int,
    log: Any,
    progress: Any = None,
    cancelled: Any = None,
) -> list[dict]:
    """Splits ``1 … k_splits − 1`` (seeds ``split_seed + k``) as split rows.

    Each split re-splits the engine with ``split_kwargs``, fits the CGI and
    every standalone with ``fit_kwargs`` (the reproducibility, gain and null
    loops off), picks the comparison channel on the split's train+val rows and
    scores the paired differences, the comparators of
    :func:`scored_comparators` and the negative controls, and reports the
    split's CGI posterior (:func:`posterior_rows`). The caller restores the
    first split. ``progress(k)`` is called before each split;
    ``cancelled()`` stops between steps.
    """
    from ..fusion import comparator_mode
    from .fusion_outputs import _posterior_summary, comparison_channel

    loops = {"reps": 0, "shuffles": 0, "gain_splits": 0, "null_runs": 0}
    rows: list[dict] = []
    for k in range(1, int(k_splits)):
        if cancelled is not None and cancelled():
            break
        if progress is not None:
            progress(k)
        seed = int(split_seed) + k
        engine._active_greenery_channel = "cgi"
        engine.split_data(random_state=seed, **split_kwargs)
        info = split_info(engine)
        params = engine.fit_bayesian_index(metric=metric, **fit_kwargs, **loops)
        effects = engine.evaluate_effects(
            params, metric, n_bootstrap=int(n_effects_boot), n_perm=int(n_perm)
        )
        slopes = engine.compute_partial_slopes(params)

        channel_paired = None
        if standalones:
            fits, pool = {}, {}
            for ch in standalones:
                engine._active_greenery_channel = ch
                fits[ch] = engine.fit_bayesian_index(
                    metric=metric, **fit_kwargs, **loops
                )
                score = engine._score_data_subset(
                    data=engine.train_val_data, params=fits[ch], metric=metric
                )
                pool[ch] = {"subset_scores": {"train": score}}
            engine._active_greenery_channel = "cgi"
            best, _ = comparison_channel(pool, list(standalones), metric)
            if best is not None:
                channel_paired = engine.paired_objective_difference(
                    params, fits[best], best, metric, n_bootstrap=int(n_paired)
                )

        comparator_paired: list[dict] = []
        if not engine.is_panel:
            names = scored_comparators(
                comparators, comparators_primary, outer_comparators
            )
            for name in names:
                r = engine.paired_objective_difference(
                    params,
                    {"comparator": name},
                    comparator_mode(name),
                    metric,
                    n_bootstrap=int(n_paired),
                )
                if r:
                    r["comparator"] = name
                    comparator_paired.append(r)
        engine._active_greenery_channel = "cgi"
        controls = engine.compute_negative_controls(params, n_boot=int(n_paired))

        primary = set(comparators_primary or comparators)
        for r in split_rows(
            effects_test=(effects or {}).get("test"),
            slope_test=(slopes or {}).get("test"),
            channel_paired=channel_paired,
            comparator_paired=comparator_paired,
            negative_controls=controls,
            posterior=_posterior_summary(params),
        ):
            r.update(split=k, seed=seed, **info)
            if r["quantity"] == "cgi_minus_comparator":
                r["primary"] = r["exposure"] in primary
            rows.append(r)
        log(
            "OK",
            f"Held-out split {k + 1}/{int(k_splits)} (seed {seed}): "
            f"{info['n_test']} test entities ({info['test_share']:.1%}"
            + (
                f", {info['test_blocks']} blocks"
                if info.get("test_blocks") is not None
                else ""
            )
            + ")"
            + (
                f", CGI − {channel_paired.get('standalone_channel')} "
                f"Δ={channel_paired.get('observed_diff')}"
                if channel_paired
                else ""
            )
            + ".",
        )
    return rows
