"""Writing a fusion run's results, and the ledger that tracks its stages.

Everything ``run_fusion`` does *after* the search: turning study bundles into
CSVs and JSON on disk, summarising the discovery, deciding the reported
direction of an association, comparing the composite against each standalone
channel, and building / timing the staged-resume ledger the job monitor renders.

Split out from ``runners`` because none of it participates in the run loop —
these are pure functions over the results, which is what makes them readable
apart from the 1,400-line orchestration that calls them.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

from ..logger import get_logger
from .stage_ledger import StageLedger

_log_fusion = get_logger("FUSION")

_FUSION_STAGE_STEPS: tuple[tuple[str, str], ...] = (
    ("load_target", "Load target"),
    ("load_metrics", "Load metric maps"),
    ("preaggregate", "Spatial pre-processing"),
    ("split", "Split train / test folds"),
    ("optimize", "Stability selection (bootstrap search)"),
    ("evaluate", "Score held-out test set"),
    ("report_stats", "Bootstrap CIs, effects & permutation tests"),
    ("apply", "Apply fusion weights"),
    ("reports", "Generate reports and composite map"),
)

_FUSION_STAGE_WEIGHTS: dict[str, float] = {
    "load_target": 1.0,
    "load_metrics": 2.0,
    "prepare_longitudinal": 1.0,
    "preaggregate": 3.0,
    "split": 1.0,
    "optimize": 20.0,
    "evaluate": 1.0,
    "report_stats": 8.0,
    "apply": 1.0,
    "mixedlm_postscore": 2.0,
    "reports": 1.0,
}

_FUSION_LONGITUDINAL_STAGE: tuple[str, str] = (
    "prepare_longitudinal",
    "Load longitudinal data",
)

_FUSION_MIXEDLM_POSTSCORE_STAGE: tuple[str, str] = (
    "mixedlm_postscore",
    "Score all MixedLM metrics on robust + top trials",
)

_STANDALONE_CHANNEL_LABELS: dict[str, str] = {
    "veg": "Vegetation",
    "terrain": "Terrain",
    "ndvi": "NDVI",
    "gvi": "Green View (veg+terrain)",
}

_FUSION_STANDALONE_SEARCH_WEIGHT = 20.0


_FUSION_STANDALONE_REPORT_WEIGHT = 8.0


def _fusion_stage_key(label: str, step: str, *, multi: bool) -> str:
    """Stage-ledger key for ``step`` under outcome ``label`` (label-scoped if multi)."""
    return f"{label}::{step}" if multi else step


def _fusion_stage_weight(stage_key: str) -> float:
    """Relative wall-time weight of a ledger stage key (see _FUSION_STAGE_WEIGHTS).

    Strips the ``"<label>::"`` multi-outcome prefix, then maps standalone
    search / report keys onto their dedicated weights and everything else onto
    the per-step table. Unknown steps default to unit weight.
    """
    step = stage_key.split("::", 1)[1] if "::" in stage_key else stage_key
    if step.startswith("standalone_"):
        return (
            _FUSION_STANDALONE_REPORT_WEIGHT
            if step.endswith("_report")
            else _FUSION_STANDALONE_SEARCH_WEIGHT
        )
    return _FUSION_STAGE_WEIGHTS.get(step, 1.0)


def _clean_params(params: dict | None) -> dict:
    """Drop the ``__*__`` stability-selection bookkeeping keys from a params dict."""
    if not params:
        return {}
    return {k: v for k, v in params.items() if not str(k).startswith("__")}


def _study_bundles(
    cgi_bundle: dict, standalones_bundle: dict
) -> list[tuple[str, str, dict]]:
    """``(study_key, display, bundle)`` for the CGI study then each standalone."""
    out: list[tuple[str, str, dict]] = [("cgi", "CGI (combined)", cgi_bundle)]
    for ch in ("veg", "terrain", "ndvi"):
        b = standalones_bundle.get(ch)
        if b:
            out.append((ch, _STANDALONE_CHANNEL_LABELS.get(ch, ch), b))
    return out


def _write_fusion_outputs(
    *,
    output_dir: str,
    label: str,
    multi_outcome: bool,
    objective_metric: str,
    formula_name: str,
    cgi_bundle: dict,
    standalones_bundle: dict,
    aic_bic: dict | None,
    covariate_impact: dict | None,
    collinearity_report: dict | None,
    run_config_record: dict | None,
    log,
) -> list[str]:
    """Persist every test result a fusion job produces to disk.

    Writes, under ``output_dir`` (``study_results/``), a machine-readable
    manifest plus tidy CSVs so each result is recorded both for replay and
    for spreadsheet analysis. ``__<label>`` is appended to every basename in
    multi-outcome runs. Returns the list of files written (best-effort: a
    failed individual write is logged and skipped, never fatal).

    Files (per outcome):

    - ``run_config.json`` — every setting the job ran with (fidelity record).
    - ``results_summary.json`` — nested manifest: each study's params, test
      score + CI, direction, subset scores, and stability stats, plus the
      CGI-vs-standalone AIC/BIC verdict, covariate-impact summary, and the
      collinearity report.
    - ``test_scores.csv`` — one headline row per study (test score, CI,
      direction).
    - ``scores.csv`` — long form: study × subset (train/val/test/all) ×
      score/score_raw/n.
    - ``parameters.csv`` — long form: study × param → value.
    - ``stability_cells.csv`` — the ranked weight cells per study.
    - ``stability_bootstraps.csv`` — the per-bootstrap leaderboard per study.
    - ``covariate_impact.csv`` — per-covariate effects (when covariates set).
    - ``decline_terms.csv`` — greenery × time slopes (longitudinal runs).
    - ``exposure_response.csv`` — the per-IQR effect, the quantile gradient
      against the lowest group with its test for trend, and the spline test of
      departure from linearity, in the form the greenspace literature reports.
    - ``exposure_response_curve.csv`` — the fitted spline curve, ready to plot.
    """

    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    sfx = f"__{label}" if multi_outcome else ""
    written: list[str] = []
    studies = _study_bundles(cgi_bundle, standalones_bundle)

    def _path(name: str) -> str:
        return os.path.join(output_dir, name)

    def _f(v: Any) -> float | None:
        try:
            fv = float(v)
            return fv if np.isfinite(fv) else None
        except (TypeError, ValueError):
            return None

    def _ci(bundle: dict) -> dict:
        return (bundle.get("test_results") or {}).get("test_ci") or {}

    def _emit_json(name: str, payload: Any) -> None:
        try:
            p = _path(name)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            written.append(p)
        except Exception as exc:  # pragma: no cover - disk-IO guard
            log("WARN", f"[{label}] Could not write {name}: {exc}")

    def _emit_csv(name: str, rows: list[dict]) -> None:
        if not rows:
            return
        try:
            p = _path(name)
            pd.DataFrame(rows).to_csv(p, index=False)
            written.append(p)
        except Exception as exc:  # pragma: no cover - disk-IO guard
            log("WARN", f"[{label}] Could not write {name}: {exc}")

    # ── run_config.json (fidelity record) ───────────────────────
    if run_config_record is not None:
        _emit_json(f"run_config{sfx}.json", run_config_record)

    # ── test_scores.csv ─────────────────────────────────────────
    test_rows: list[dict] = []
    for key, _disp, b in studies:
        tr = b.get("test_results") or {}
        ci = _ci(b)
        test_rows.append(
            {
                "study": key,
                "metric": objective_metric,
                "test_score": _f(tr.get("test_score")),
                "ci_lower": _f(ci.get("lower")),
                "ci_upper": _f(ci.get("upper")),
                "direction": (
                    int(b["direction_sign"])
                    if b.get("direction_sign") is not None
                    else None
                ),
            }
        )
    _emit_csv(f"test_scores{sfx}.csv", test_rows)

    # ── scores.csv (every subset) ───────────────────────────────
    score_rows: list[dict] = []
    for key, _disp, b in studies:
        subsets = b.get("subset_scores") or {}
        for subset in ("train", "val", "test", "all"):
            block = subsets.get(subset) or {}
            if not block:
                continue
            score_rows.append(
                {
                    "study": key,
                    "subset": subset,
                    "metric": objective_metric,
                    "score": _f(block.get("score")),
                    "score_raw": _f(block.get("score_raw")),
                    "n": block.get("n"),
                }
            )
    _emit_csv(f"scores{sfx}.csv", score_rows)

    # ── parameters.csv (long form) ──────────────────────────────
    param_rows: list[dict] = []
    for key, _disp, b in studies:
        params = _clean_params(b.get("averaged_params") or b.get("best_params"))
        for pname, pval in params.items():
            param_rows.append({"study": key, "param": pname, "value": pval})
    _emit_csv(f"parameters{sfx}.csv", param_rows)

    # ── stability_cells.csv + stability_bootstraps.csv ──────────
    cell_rows: list[dict] = []
    bs_rows: list[dict] = []
    for key, _disp, b in studies:
        summ = b.get("stability_summary") or {}
        for rank, c in enumerate(summ.get("cell_stats") or [], start=1):
            row: dict = {"study": key, "rank": rank}
            for wk, wv in (c.get("weights") or {}).items():
                row[wk] = wv
            row["count"] = c.get("count")
            row["q_worst"] = _f(c.get("q_worst"))
            row["median"] = _f(c.get("median"))
            row["selection_probability"] = _f(c.get("selection_probability"))
            cell_rows.append(row)
        for entry in summ.get("per_bootstrap_summary") or []:
            row = {
                "study": key,
                "bootstrap": entry.get("bootstrap"),
                "n_trials": entry.get("n_trials"),
                "top_oob": _f(entry.get("top_oob")),
                "median_oob": _f(entry.get("median_oob")),
                "min_oob": _f(entry.get("min_oob")),
                "max_oob": _f(entry.get("max_oob")),
            }
            for pk, pv in (entry.get("top_params") or {}).items():
                row[pk] = pv
            bs_rows.append(row)
    _emit_csv(f"stability_cells{sfx}.csv", cell_rows)
    _emit_csv(f"stability_bootstraps{sfx}.csv", bs_rows)

    # ── covariate_impact.csv ────────────────────────────────────
    if covariate_impact and covariate_impact.get("per_covariate"):
        _emit_csv(
            f"covariate_impact{sfx}.csv",
            [dict(r) for r in covariate_impact["per_covariate"]],
        )

    # ── decline_terms.csv (longitudinal exposure × time) ────────
    decline_terms = (cgi_bundle or {}).get("decline_terms") or None
    if decline_terms and decline_terms.get("terms"):
        _emit_csv(f"decline_terms{sfx}.csv", [dict(r) for r in decline_terms["terms"]])

    # ── exposure_response.csv (per-IQR, quartiles, non-linearity) ─
    # One tidy table rather than three files: the rows are all statements
    # about the same fitted exposure–response, and a reader comparing them
    # against a published table wants them side by side.
    er_rows: list[dict] = []
    er_curve_rows: list[dict] = []
    for _study_key, _study_disp, _bundle in studies:
        er_report = (_bundle or {}).get("exposure_response") or None
        if not er_report:
            continue
        design = er_report.get("design")
        _row_start = len(er_rows)
        coding = er_report.get("coding") or {}
        if coding:
            # Which level the logistic models called the event. Carried in the
            # table because a 1=Yes/2=No survey column inverts every odds ratio
            # below without changing anything else about how they look.
            er_rows.append(
                {
                    "term": "modelled_event_level",
                    "design": design,
                    "detail": (
                        f"event={_f(coding.get('modelled_level'))} vs "
                        f"reference={_f(coding.get('reference_level'))}"
                        + ("  CHECK CODEBOOK" if coding.get("suspicious") else "")
                    ),
                    "estimate": None,
                    "std_error": None,
                    "ci_low": None,
                    "ci_high": None,
                    "odds_ratio": None,
                    "or_ci_low": None,
                    "or_ci_high": None,
                    "p_value": None,
                }
            )
        per_iqr = er_report.get("per_iqr") or {}
        if per_iqr.get("estimate") is not None:
            er_rows.append(
                {
                    "term": "per_iqr",
                    "design": design,
                    "detail": f"IQR={_f(per_iqr.get('iqr'))}",
                    "estimate": _f(per_iqr.get("estimate")),
                    "std_error": _f(per_iqr.get("std_error")),
                    "ci_low": _f(per_iqr.get("ci_low")),
                    "ci_high": _f(per_iqr.get("ci_high")),
                    "odds_ratio": _f(per_iqr.get("odds_ratio")),
                    "or_ci_low": _f(per_iqr.get("or_ci_low")),
                    "or_ci_high": _f(per_iqr.get("or_ci_high")),
                    "p_value": None,
                }
            )
        quartiles = er_report.get("quartiles") or {}
        for row in quartiles.get("contrasts") or []:
            er_rows.append(
                {
                    "term": f"quantile_{row.get('group')}",
                    "design": design,
                    "detail": (
                        f"vs group {quartiles.get('reference_group')}, "
                        f"n={row.get('n')}"
                    ),
                    "estimate": _f(row.get("coef")),
                    "std_error": _f(row.get("std_error")),
                    "ci_low": None,
                    "ci_high": None,
                    "odds_ratio": _f(row.get("odds_ratio")),
                    "or_ci_low": _f(row.get("or_ci_low")),
                    "or_ci_high": _f(row.get("or_ci_high")),
                    "p_value": _f(row.get("p_value")),
                }
            )
        if quartiles.get("trend_p") is not None:
            er_rows.append(
                {
                    "term": "quantile_trend",
                    "design": design,
                    "detail": f"{quartiles.get('n_groups')} groups",
                    "estimate": None,
                    "std_error": None,
                    "ci_low": None,
                    "ci_high": None,
                    "odds_ratio": None,
                    "or_ci_low": None,
                    "or_ci_high": None,
                    "p_value": _f(quartiles.get("trend_p")),
                }
            )
        nonlinear = er_report.get("nonlinearity") or {}
        if nonlinear.get("nonlinearity_p") is not None:
            er_rows.append(
                {
                    "term": "nonlinearity_wald",
                    "design": design,
                    "detail": (
                        f"spline df={nonlinear.get('df')}, "
                        f"chi2={_f(nonlinear.get('wald_chi2'))} on "
                        f"{nonlinear.get('wald_df')} df"
                    ),
                    "estimate": _f(nonlinear.get("linear_coef")),
                    "std_error": None,
                    "ci_low": None,
                    "ci_high": None,
                    "odds_ratio": None,
                    "or_ci_low": None,
                    "or_ci_high": None,
                    "p_value": _f(nonlinear.get("nonlinearity_p")),
                }
            )
        # Stamp this study onto the rows it just produced, so the composite
        # and each single channel sit in one table — which is the comparison
        # a published single-channel result is read against.
        for _r in er_rows[_row_start:]:
            _r["study"] = _study_key
        curve = (nonlinear or {}).get("curve")
        if curve:
            er_curve_rows.extend(
                {"study": _study_key, "exposure": x, "effect": y}
                for x, y in zip(curve["exposure"], curve["effect"])
            )
    if er_rows:
        _emit_csv(
            f"exposure_response{sfx}.csv",
            [{"study": r.pop("study"), **r} for r in er_rows],
        )
    if er_curve_rows:
        _emit_csv(f"exposure_response_curve{sfx}.csv", er_curve_rows)

    # ── moderation.csv (effect modification) ────────────────────
    # One row per simple slope plus one per interaction term, so a reader can
    # see both "does it differ" and "what is the effect in each group" without
    # opening two files.
    moderation = (cgi_bundle or {}).get("moderation") or []
    if moderation:
        mod_rows: list[dict] = []
        for block in moderation:
            name = block.get("moderator")
            for row in block.get("simple_slopes") or []:
                mod_rows.append(
                    {
                        "moderator": name,
                        "row_type": "simple_slope",
                        "level": row.get("level"),
                        "moderator_value": _f(row.get("moderator_value")),
                        "n": row.get("n"),
                        "estimate": _f(row.get("slope")),
                        "std_error": _f(row.get("std_error")),
                        "ci_low": _f(row.get("ci_low")),
                        "ci_high": _f(row.get("ci_high")),
                        "odds_ratio": _f(row.get("odds_ratio")),
                        "p_value": _f(row.get("p_value")),
                    }
                )
            for row in block.get("interaction_terms") or []:
                mod_rows.append(
                    {
                        "moderator": name,
                        "row_type": "interaction",
                        "level": row.get("term"),
                        "moderator_value": None,
                        "n": block.get("n"),
                        "estimate": _f(row.get("coef")),
                        "std_error": _f(row.get("std_error")),
                        "ci_low": None,
                        "ci_high": None,
                        "odds_ratio": None,
                        "p_value": _f(row.get("p_value")),
                    }
                )
            mod_rows.append(
                {
                    "moderator": name,
                    "row_type": "interaction_joint_test",
                    "level": (
                        f"chi2={_f(block.get('interaction_wald_chi2'))} on "
                        f"{block.get('interaction_df')} df"
                    ),
                    "moderator_value": _f(block.get("centred_at")),
                    "n": block.get("n"),
                    "estimate": None,
                    "std_error": None,
                    "ci_low": None,
                    "ci_high": None,
                    "odds_ratio": None,
                    "p_value": _f(block.get("interaction_p")),
                }
            )
        _emit_csv(f"moderation{sfx}.csv", mod_rows)

    # ── results_summary.json (master manifest) ──────────────────
    studies_manifest: dict[str, dict] = {}
    for key, disp, b in studies:
        tr = b.get("test_results") or {}
        summ = b.get("stability_summary") or {}
        studies_manifest[key] = {
            "display": disp,
            "channel": b.get("channel", "cgi"),
            "params": _clean_params(b.get("averaged_params") or b.get("best_params")),
            # A held-out mixed model that didn't converge has no honest score:
            # report ``None``, not the degenerate 0.0 the scorer returns.
            "test_score": (
                None
                if _ci(b).get("status") == "fit_failed"
                else _f(tr.get("test_score"))
            ),
            "test_ci": {
                "lower": _f(_ci(b).get("lower")),
                "upper": _f(_ci(b).get("upper")),
                "method": _ci(b).get("method", "percentile"),
                # "fit_failed" when the held-out mixed model did not converge —
                # distinguishes a genuine null from a non-fit, so a 0.0 with a
                # CI that excludes it is never reported as a real result.
                "status": _ci(b).get("status", "ok"),
            },
            "direction": (
                int(b["direction_sign"])
                if b.get("direction_sign") is not None
                else None
            ),
            "subset_scores": b.get("subset_scores") or {},
            "stability": {
                k: summ.get(k)
                for k in (
                    "q_worst",
                    "median",
                    "count",
                    "selection_probability",
                    "worst_quantile",
                    "n_bootstraps",
                    "n_trials_per_bootstrap",
                    "n_total_trials",
                    "higher_is_better",
                )
            },
        }

    cov_summary = None
    if covariate_impact:
        cov_summary = {
            k: covariate_impact.get(k)
            for k in (
                "r2_full",
                "r2_cgi_only",
                "r2_lift_from_covariates",
                "cgi_coef",
                "cgi_std_err",
                "n",
            )
        }
        cov_summary["per_covariate"] = covariate_impact.get("per_covariate") or []

    manifest = {
        "outcome": label,
        "objective_metric": objective_metric,
        "cgi_formula": formula_name,
        "studies": studies_manifest,
        "cgi_vs_standalone_aic_bic": aic_bic,
        "covariate_impact": cov_summary,
        "decline_terms": (cgi_bundle or {}).get("decline_terms"),
        # Carried in the manifest, not only in their CSVs: the results panel
        # reads this file when the live engine is gone, so anything absent here
        # silently disappears from a reloaded job.
        "exposure_response": (cgi_bundle or {}).get("exposure_response"),
        "moderation": (cgi_bundle or {}).get("moderation") or [],
        "collinearity": collinearity_report,
    }
    _emit_json(f"results_summary{sfx}.json", manifest)

    # ── aic_bic.json + collinearity.json (standalone copies) ────
    if aic_bic is not None:
        _emit_json(f"aic_bic{sfx}.json", aic_bic)
    if collinearity_report:
        _emit_json(f"collinearity{sfx}.json", collinearity_report)

    log(
        "OK",
        f"[{label}] Wrote {len(written)} result file(s) to {output_dir}.",
    )
    return written


def _log_stage_timing(ledger: StageLedger, log: Any) -> None:
    """Log where the run spent its wall clock, longest stage first.

    Written at the end of a run so tuning decisions rest on this machine's own
    numbers: stage costs shift with core count, disk speed, and how much of the
    greenery cache was reusable, so a breakdown measured elsewhere does not
    transfer.
    """
    rows = ledger.timing_report()
    if not rows:
        return
    total = sum(r[2] for r in rows)
    log("INFO", "====== STAGE WALL-CLOCK BREAKDOWN ======")
    log(
        "INFO",
        f"  accounted {total / 60:.1f} min across {len(rows)} timed stage(s) "
        f"on {os.cpu_count() or '?'} logical core(s).",
    )
    for key, label, secs, share in rows:
        if share < 0.005 and secs < 30:
            continue
        log(
            "INFO",
            f"  {share * 100:5.1f}%  {secs / 60:8.1f} min  {label} ({key})",
        )


def _build_fusion_ledger(
    labels: list[str],
    *,
    multi: bool,
    standalone_channels: list[str] | None = None,
    longitudinal: bool = False,
    mixedlm_postscore: bool = False,
) -> StageLedger:
    """Fresh ledger covering every (outcome, step) pair in run order.

    For each outcome the CGI pipeline (`_FUSION_STAGE_STEPS`) lands first, then
    two stages per enabled standalone metric — the sweep + posterior and the
    test scoring / reporting that follows it. The standalone list comes from the
    active formula's channels, so a two-channel study shows ``ndvi`` and ``gvi``
    rather than the legacy three. Standalones reuse the already-
    built split + pre-aggregation cache. When ``longitudinal`` is true an extra
    ``prepare_longitudinal`` stage is inserted between ``load_metrics`` and
    ``preaggregate`` to cover per-wave file loading. The MixedLM
    post-score stage is only added when ``mixedlm_postscore`` is true
    (a longitudinal study whose scoring metric is actually a
    ``mixedlm_*`` one — year-aware cross-sectional studies sit on a
    spec too but score with OLS so they skip the post-score step).
    """
    standalones = list(standalone_channels or [])
    steps: list[tuple[str, str]] = []
    for label in labels:
        for step_key, step_label in _FUSION_STAGE_STEPS:
            key = _fusion_stage_key(label, step_key, multi=multi)
            disp = f"[{label}] {step_label}" if multi else step_label
            steps.append((key, disp))
            # Slot the longitudinal-prep stage in right after load_metrics so
            # the monitor reads top-to-bottom in actual execution order.
            if longitudinal and step_key == "load_metrics":
                lon_key_raw, lon_label = _FUSION_LONGITUDINAL_STAGE
                lon_key = _fusion_stage_key(label, lon_key_raw, multi=multi)
                lon_disp = f"[{label}] {lon_label}" if multi else lon_label
                steps.append((lon_key, lon_disp))
            # And the post-score stage right after ``apply`` so the
            # multi-metric CSV is written before any standalone studies
            # take over the engine state.
            if mixedlm_postscore and step_key == "apply":
                ps_key_raw, ps_label = _FUSION_MIXEDLM_POSTSCORE_STAGE
                ps_key = _fusion_stage_key(label, ps_key_raw, multi=multi)
                ps_disp = f"[{label}] {ps_label}" if multi else ps_label
                steps.append((ps_key, ps_disp))
        for ch in standalones:
            ch_lbl = _STANDALONE_CHANNEL_LABELS.get(ch, ch)
            search_key = _fusion_stage_key(label, f"standalone_{ch}", multi=multi)
            report_key = _fusion_stage_key(
                label, f"standalone_{ch}_report", multi=multi
            )
            search_step = f"Standalone {ch_lbl} sweep + posterior"
            report_step = f"Standalone {ch_lbl} test scoring & reports"
            steps.append(
                (search_key, f"[{label}] {search_step}" if multi else search_step)
            )
            steps.append(
                (report_key, f"[{label}] {report_step}" if multi else report_step)
            )
    return StageLedger.from_steps(steps)


def _posterior_summary(params: dict) -> dict:
    """Lift the discovery diagnostics out of a winning-params dict.

    ``fit_bayesian_index`` stashes its bookkeeping under ``__``-prefixed keys
    (so the "Final params" panel strips them). This surfaces the ones the
    results UI shows as a plain summary dict.
    """

    def g(key: str, default: Any = None) -> Any:
        return params.get(key, default)

    sweep = g("__sweep__", {}) or {}
    post = g("__posterior__", {}) or {}
    disc = g("__discovery__", {}) or {}
    gain = g("__holdout_gain__", {}) or {}
    null = g("__null_calibration__", {}) or {}
    return {
        "selection_method": g("__selection_method__", "bayesian_index"),
        # What the sweep chose, and how confident that choice is.
        "picked": sweep.get("picked"),
        "form": sweep.get("form"),
        "form_scores": sweep.get("form_scores", {}),
        "sweep_score": sweep.get("score"),
        "one_se_picked": sweep.get("one_se_picked"),
        "boundary_hit": sweep.get("boundary_hit", []),
        "n_candidates": sweep.get("n_candidates"),
        "distinct_split_winners": sweep.get("distinct_split_winners"),
        "sweep_splits": sweep.get("splits"),
        # Weights and effect, with intervals.
        "channels": post.get("channels"),
        "weight_mean": post.get("weight_mean"),
        "weight_ci_low": post.get("weight_ci_low"),
        "weight_ci_high": post.get("weight_ci_high"),
        "powers": post.get("powers"),
        "beta_mean": post.get("beta_mean"),
        "beta_ci_low": post.get("beta_ci_low"),
        "beta_ci_high": post.get("beta_ci_high"),
        "p_direction": post.get("p_direction"),
        "rhat_max": post.get("rhat_max"),
        "ess_min": post.get("ess_min"),
        "divergences": post.get("divergences"),
        # Does the discovery reproduce, and does it beat a single channel?
        "discovery": disc,
        "holdout_gain": gain,
        "null_calibration": null,
        "elapsed_s": g("__elapsed_s__"),
    }


def _direction_sign(engine: Any, params: dict, metric: str) -> int:
    """``+1`` / ``-1`` sign of the greenery↔outcome relationship on the test set.

    Distance correlation is unsigned, so the report needs a separate direction
    indicator. Returns ``+1`` (neutral) on any failure.
    """
    from .. import mixed_effects_scoring as _me
    from .. import objective_scoring as _scoring

    try:
        res = engine.evaluate_on_test(
            params=params, metric=metric, return_predictions=True
        )
        target = np.asarray(res.get("targets"), dtype=np.float64)
        pred = np.asarray(res.get("predictions"), dtype=np.float64)
        cov = res.get("covariates")
        # A MixedLM objective's direction is the sign of its own greenery
        # coefficient; a pooled rank correlation would ignore the panel and can
        # disagree with the model being reported.
        if (
            getattr(engine, "is_longitudinal", False)
            and metric in _me.MIXEDLM_METRICS
            and res.get("entity_id") is not None
        ):
            spec = engine.longitudinal_spec
            coef = _me.score_mixedlm(
                "mixedlm_coef",
                target,
                pred,
                res.get("entity_id"),
                res.get("years_since_baseline"),
                covariates=cov,
                include_time_fixed=engine._effective_time_fixed(spec),
                random_slope=spec.random_slope_time,
                spatial_basis=res.get("spatial_basis"),
                spatial_method=engine.spatial_adjust_method,
                target=spec.association_target,
            )
            return 1 if float(coef) >= 0 else -1
        return int(
            _scoring.relationship_sign(
                target,
                pred,
                cov,
                residualize_method=getattr(engine, "residualize_method", "linear"),
            )
        )
    except Exception:
        return 1


def _compare_cgi_vs_standalone(
    engine: Any,
    standalones_bundle: dict,
    cgi_params: dict,
    metric: str,
    *,
    longitudinal: bool,
    log,
) -> dict | None:
    """AIC/BIC verdict: is CGI justified over the best single standalone channel?

    The best standalone is the channel with the strongest whole-data (``all``)
    score (direction-aware). The full (3-channel) and reduced (best-channel)
    models are fit on the whole dataset's per-entity channel design built at the
    CGI winning aggregation params, so the verdict is on the same ``all`` slice
    as the paired objective comparison. Returns ``None`` when no standalone
    qualifies or the design / fit fails.
    """
    from .. import mixed_effects_scoring as _me
    from .. import objective_scoring as _scoring

    chans = ["veg", "terrain", "ndvi"]
    higher_is_better = (
        metric in _scoring.HIGHER_IS_BETTER or metric in _me.HIGHER_IS_BETTER
    )
    scored: list[tuple[str, float]] = []
    for ch in chans:
        b = standalones_bundle.get(ch)
        if not b:
            continue
        ss = (b.get("subset_scores") or {}).get("all") or {}
        ts = ss.get("score")
        if ts is None or not np.isfinite(float(ts)):
            continue
        scored.append((ch, float(ts)))
    if not scored:
        return None
    best_ch = max(scored, key=lambda kv: kv[1] if higher_is_better else -kv[1])[0]
    best_idx = chans.index(best_ch)

    try:
        design = engine.build_channel_design(cgi_params, subset="all")
    except Exception as exc:
        log("WARN", f"AIC/BIC channel design failed: {exc}")
        return None

    X = design["channels"]
    target = design["target"]
    cov = design["covariates"]
    try:
        # Route on the *metric*, not merely on a spec being present — the same
        # rule ``_score_greenery`` uses. A year-aware cross-sectional run
        # carries a longitudinal spec (to assign a greenery file per
        # measurement year) but scores with OLS, and its entity ids are
        # synthesised one-per-row. Handing that to a per-entity random
        # intercept asks it to separate between- from within-entity variance
        # from singleton groups, which is unidentified and simply fails to
        # converge — taking the CGI-vs-standalone verdict down with it.
        panel_metric = metric in _me.MIXEDLM_METRICS
        if longitudinal and panel_metric and design.get("entity_id") is not None:
            return _me.compare_models_aic_bic_mixedlm(
                target,
                X,
                chans,
                best_idx,
                design["entity_id"],
                design["years_since_baseline"],
                covariates=cov,
            )
        return _scoring.compare_models_aic_bic(
            target, X, chans, best_idx, covariates=cov
        )
    except Exception as exc:
        log("WARN", f"AIC/BIC comparison failed: {exc}")
        return None


def _jsonsafe_results(obj, _depth: int = 0):
    """Recursively convert a fusion results payload to JSON-serializable types.

    Heavy or non-serializable values (DataFrames, ndarrays, per-trial pools)
    are dropped — they're reproducible from the on-disk artifacts — so the
    compact bundle written beside the job can rehydrate the results view after
    a Streamlit restart without pinning engines in memory.
    """
    if _depth > 12:
        return None
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return None
    if isinstance(obj, dict):
        out: dict = {}
        for k, v in obj.items():
            if k in (
                "composite_df",
                "robust_trials",
                "per_trial_test",
                "all_completed_trials",
            ):
                continue
            out[str(k)] = _jsonsafe_results(v, _depth + 1)
        return out
    if isinstance(obj, (list, tuple, set)):
        return [_jsonsafe_results(v, _depth + 1) for v in obj]
    try:
        import pandas as _pd

        if isinstance(obj, (_pd.DataFrame, _pd.Series)):
            return None
    except Exception:
        pass
    try:
        s = str(obj)
        return s if len(s) <= 2000 else None
    except Exception:
        return None
