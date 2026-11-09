"""A fusion run's results from the command line.

::

    geofuse results JOB | FOLDER [--outcome LABEL] [--json]
    geofuse export JOB | FOLDER [-o FILE.gpkg | .csv | .geojson] [--outcome LABEL]

``JOB`` is a command-line job id (a prefix works); ``FOLDER`` is any fusion
run's artifacts folder (``<output>/fusion/<stamp>__<id>/``), so runs started
from the GUI read the same way. Both read the run's ``results_bundle.json``;
nothing is recomputed.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from geofuse.cli.commands import EXIT_OK, CliError, _state
from geofuse.jobs.outer_splits import spatial_label

_CHANNELS = {"veg": "vegetation", "terrain": "terrain", "ndvi": "NDVI", "gvi": "GVI"}


# ────────────────────────────────────────────────────────────────────
# Finding and reading a run
# ────────────────────────────────────────────────────────────────────


def bundle_path(args) -> Path:
    """The ``results_bundle.json`` that ``args.run`` names."""
    ref = Path(args.run)
    if ref.is_file() and ref.name == "results_bundle.json":
        return ref
    if ref.is_dir():
        if (ref / "results_bundle.json").is_file():
            return ref / "results_bundle.json"
        raise CliError(f"{ref} has no results_bundle.json: the run did not finish.")
    state = _state(args)
    hits = state.find(args.run)
    if not hits:
        raise CliError(f"{args.run!r} is neither a job nor a fusion results folder.")
    if len(hits) > 1:
        raise CliError(f"{args.run!r} matches several jobs: {', '.join(hits)}.")
    job = state.job(hits[0])
    if job.spec.get("kind") != "fusion":
        raise CliError(f"Job {job.id} is a {job.spec.get('kind')} job, not fusion.")
    for out in job.status.get("outputs") or []:
        if str(out).endswith("results_bundle.json"):
            return Path(out)
    raise CliError(f"Job {job.id} has no results yet ({job.state}).")


def outcomes(payload: dict, wanted: str | None = None) -> list[tuple[str, dict]]:
    """``(outcome, bundle)`` for each outcome of a run, or the one ``wanted``."""
    labels = payload.get("ordered_labels") or [payload.get("outcome_label") or "?"]
    by_target = payload.get("by_target") or {}
    found = [(label, by_target.get(label) or payload) for label in labels]
    if wanted:
        found = [(label, b) for label, b in found if label == wanted]
        if not found:
            known = ", ".join(labels)
            raise CliError(f"No outcome {wanted!r} (this run has: {known}).")
    return found


# ────────────────────────────────────────────────────────────────────
# The summary
# ────────────────────────────────────────────────────────────────────


def _num(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _interval(block: dict, lo: str = "lower", hi: str = "upper") -> str:
    low, high = _num(block.get(lo)), _num(block.get(hi))
    return f"[{low:.4f}, {high:.4f}]" if low is not None and high is not None else ""


def _difference(row: dict) -> str:
    """``cgi − channel = diff`` when both held-out scores are there, else the
    difference alone. A lower-is-better objective's difference is signed so
    a positive value favours the CGI, so its scores are printed the other
    way round."""
    diff = _num(row.get("observed_diff"))
    cgi, other = _num(row.get("cgi_score")), _num(row.get("standalone_score"))
    if cgi is None or other is None:
        return f"{diff:+.4f}"
    if abs((other - cgi) - diff) < abs((cgi - other) - diff):
        return f"{other:.4f} − {cgi:.4f} = {diff:+.4f} (lower is better)"
    return f"{cgi:.4f} − {other:.4f} = {diff:+.4f}"


def _outer_name(entry: dict) -> str:
    quantity, exposure = entry.get("quantity"), entry.get("exposure")
    name = _CHANNELS.get(exposure, exposure)
    control = entry.get("control")
    return {
        "held_out_score": "Held-out score",
        "held_out_beta": "β per SD",
        "negative_control_beta": f"β {name}",
        "exposure_beta": f"β per SD, {name}",
        "exposure_negative_control_beta": f"β {name} on {control}",
        "specificity_difference": f"ΔD vs {name}, {control}",
        "negative_control_d": f"D {name}",
        "held_out_per_iqr": "per IQR",
        "held_out_per_p5_p95": "per p5–p95",
        "exposure_per_iqr": f"per IQR, {name}",
        "exposure_per_p5_p95": f"per p5–p95, {name}",
        "exposure_negative_control_per_iqr": f"per IQR {name} on {control}",
        "exposure_negative_control_per_p5_p95": f"per p5–p95 {name} on {control}",
        "held_out_per_sd0": "per SD0",
        "exposure_per_sd0": f"per SD0, {name}",
        "exposure_negative_control_per_sd0": f"per SD0 {name} on {control}",
    }.get(quantity, f"CGI − {name}")


def _outer_lines(outer: dict) -> list[str]:
    """The across-split summary: median [range], share holding, corrected p."""
    summary = outer.get("summary") or []
    if not summary:
        return []
    seeds = outer.get("seeds") or []
    methods = [e["spatial_method"] for e in summary if e.get("spatial_method")]
    # With several scales, each slope says which one it was adjusted at.
    scales = {e.get("spatial_scale_m") for e in summary} - {None}

    def ladder(e: dict) -> bool:
        return len(scales) > 1 and e.get("spatial_scale_m") is not None

    lines = [
        f"  Across {outer.get('k')} held-out splits"
        + (f" (seeds {seeds[0]}–{seeds[-1]})" if seeds else "")
        + ": median [min, max], share of splits holding, corrected p"
        + (f"; slopes adjusted for a {methods[0]} smooth" if methods else "")
    ]
    for e in summary:
        median, lo, hi = (_num(e.get(k)) for k in ("median", "min", "max"))
        if median is None or str(e.get("quantity")).startswith("posterior_"):
            continue
        share = _num(e.get("share_holds"))
        p = _num(e.get("corrected_p_holm"))
        p_text = f"Holm p={p:.3g}" if p is not None else ""
        if p is None and _num(e.get("corrected_p")) is not None:
            p_text = f"p={_num(e.get('corrected_p')):.3g}"
        name = _outer_name(e) + (" (w)" if e.get("weighting") else "")
        lines.append(
            f"    {name:<22}{median:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            + (f"  holds {share:.0%}" if share is not None else "")
            + (f"  {p_text}" if p_text else "")
            + (f"  {_scale_name(e['spatial_scale_m'])}" if ladder(e) else "")
        )
    return lines + _posterior_lines(summary)


def _raw_lines(raw: dict) -> list[str]:
    """The first split's held-out change in outcome units per IQR and per
    p5–p95 range of each exposure, and the composite's layers at its pick."""
    exposures = (raw or {}).get("exposures") or {}
    if not exposures:
        return []
    lines = [
        "  Raw units        held-out change in the outcome per IQR | per p5–p95 "
        "of each exposure (ranges on train+val)"
    ]
    for name, block in exposures.items():
        target = block.get("target") or {}
        parts = []
        for key in ("per_iqr", "per_p5_p95"):
            r = target.get(key) or {}
            est, lo, hi = (_num(r.get(k)) for k in ("estimate", "ci_low", "ci_high"))
            scale = _num(r.get("scale"))
            if est is None or scale is None:
                parts.append("—")
                continue
            parts.append(f"{est:+.4f} [{lo:+.4f}, {hi:+.4f}] per {scale:.4g}")
        fixed = " (fixed IQR)" if block.get("iqr_source") == "fixed" else ""
        label = "CGI" if name == "cgi" else _CHANNELS.get(name, name)
        lines.append(f"    {label:<15}{parts[0]}{fixed} | {parts[1]}")
    layers = (raw or {}).get("layers") or {}
    if layers:
        lines.append(
            "    CGI layers     IQR at the pick: "
            + ", ".join(
                f"{_CHANNELS.get(ch, ch)} {_num(d.get('iqr')):.4g}"
                for ch, d in layers.items()
                if _num(d.get("iqr")) is not None
            )
        )
    return lines


def _weighted_lines(weighted: dict) -> list[str]:
    """The first split's weighted estimates beside the unweighted ones: the
    weights' Kish effective size and each weighted result."""
    if not weighted:
        return []
    if weighted.get("skipped"):
        return [f"  Weighted         skipped ({weighted['skipped']})"]
    from geofuse.jobs.outer_splits import weighted_rows

    lines = [
        f"  Weighted         by '{weighted.get('weight_column')}' on the held-out "
        f"rows: Kish n_eff {_num(weighted.get('n_eff')) or 0:,.0f} of "
        f"{weighted.get('n_test')}, {weighted.get('n_dropped')} without a "
        "usable weight"
    ]
    for r in weighted_rows(weighted):
        est = _num(r.get("estimate"))
        if est is None:
            continue
        lo, hi, p = (_num(r.get(k)) for k in ("ci_low", "ci_high", "p_value"))
        lines.append(
            f"    {_outer_name(r):<30}{est:+.4f}"
            + (f" [{lo:+.4f}, {hi:+.4f}]" if lo is not None and hi is not None else "")
            + (f"  p={p:.3g}" if p is not None else "")
        )
    return lines


def _null_lines(null: dict) -> list[str]:
    """The decision's false-positive rate per null kind: hits / nulls with
    the exact interval, the real outcome's calibrated p and the stopping
    rule's decision; every other tested quantity on its own line."""
    summary = null.get("summary") or []
    if not summary:
        return []
    groups = (null.get("groups") or {}).get("source")
    lines = [
        "  Null calibration of the held-out decision"
        + (f" (areas: {groups})" if groups else "")
        + ": rejections at corrected p < 0.05, 95 % CI"
    ]
    for e in summary:
        n, hits = e.get("finished") or 0, e.get("hits") or 0
        lo, hi = _num(e.get("ci_low")), _num(e.get("ci_high"))
        calibrated = _num(e.get("calibrated_p"))
        lines.append(
            f"    {e.get('null_kind'):<8}{_outer_name(e):<22}{hits}/{n}"
            + (f" = {hits / n:.1%}" if n else "")
            + (f" [{lo:.1%}, {hi:.1%}]" if lo is not None and hi is not None else "")
            + (f"  calibrated p={calibrated:.3g}" if calibrated is not None else "")
            + (f"  {e['decision']}" if e.get("decision") else "")
        )
    if null.get("complete") is False:
        lines.append("    (not every null outcome finished)")
    return lines


def _scale_name(text: str) -> str:
    """A ``spatial_scale_m`` cell as words: ``"at 5 km"`` or ``"unadjusted"``."""
    if text == "none":
        return "unadjusted"
    return f"at {float(text) / 1000:g} km"


def _specificity_lines(report: dict) -> list[str]:
    """One "Specificity vs" block per exposure: per control, each exposure's
    D = |β target| − |β control| and ΔD = D_cgi − D_e with its interval."""
    rows = report.get("rows") or []
    if not rows:
        return []
    lines: list[str] = []
    for exposure in dict.fromkeys(r.get("exposure") for r in rows):
        name = _CHANNELS.get(exposure, exposure)
        kind = next(r.get("kind") for r in rows if r.get("exposure") == exposure)
        lines.append(
            f"  Specificity vs {name}"
            + (" (comparison channel)" if kind == "channel" else "")
            + ": held-out D = |β target| − |β control|, ΔD = D_cgi − D_e"
        )
        for r in rows:
            if r.get("exposure") != exposure:
                continue
            values = [_num(r.get(k)) for k in ("d_cgi", "d_exposure", "delta_d")]
            if None in values:
                continue
            d_cgi, d_e, delta = values
            p = _num(r.get("p_value_holm"))
            p_text = f"Holm p={p:.3g}" if p is not None else ""
            if p is None and _num(r.get("p_value")) is not None:
                p_text = f"p={_num(r.get('p_value')):.3g}"
            spatial = spatial_label(r.get("spatial_adjustment"))
            lines.append(
                f"    {str(r.get('control')):<15}D_cgi {d_cgi:+.4f}  D_e {d_e:+.4f}  "
                f"ΔD {delta:+.4f}  {_interval(r, 'ci_low', 'ci_high')}"
                + (f"  {p_text}" if p_text else "")
                + (f"  n={r['n']}" if r.get("n") else "")
                + (f"  {spatial}" if spatial else "")
            )
    return lines


def _spatial_lines(bundle: dict) -> list[str]:
    """The run's spatial adjustment: the method, and whether the slopes carry
    it (runs from before it reached them adjusted the scores only)."""
    adjustment = (bundle.get("test_results") or {}).get("spatial_adjustment") or {}
    if not adjustment.get("method"):
        return []
    splits = (bundle.get("negative_controls") or {}).get("splits") or {}
    target = (splits.get("test") or {}).get("target") or {}
    reference = (bundle.get("comparator_reference") or {}).get("test") or {}
    record = target.get("spatial_adjustment") or reference.get("spatial_adjustment")
    outer = (bundle.get("outer_splits") or {}).get("splits") or []
    line = f"  Spatial adjust   {adjustment['method']} smooth in the scores"
    if record is not None or any("spatial_method" in r for r in outer):
        line += ", slopes, negative controls and specificity"
    if record is not None and record.get("scale_m") is not None:
        line += f" (held-out β: {record.get('df')} smooth functions)"
    elif record is not None:
        df = record.get("df")
        line += f" (held-out β: df {'none' if df is None else df})"
    lines = [line]
    smoothing = bundle.get("spatial_smoothing") or {}
    if smoothing.get("scales_m"):
        scales = [f"{float(v) / 1000:g} km" for v in smoothing["scales_m"]]
        text = f"  Spatial scale    {scales[0]}"
        if len(scales) > 1:
            text += f" (slopes also at {', '.join(scales[1:])} and unadjusted)"
        text += (
            f"; {smoothing.get('clusters')} clusters split at "
            f"{float(smoothing.get('eps_m') or 0) / 1000:g} km"
        )
        if smoothing.get("merged_components"):
            text += (
                f", {smoothing['merged_components']} smaller ones "
                f"({smoothing.get('merged_entities')} entities) merged"
            )
        lines.append(text + "; paired tests share the smooth")
    tuning = bundle.get("spatial_tuning") or {}
    if tuning.get("scale_m"):
        absorbed = ", ".join(
            f"{ch} {share:.0%}"
            for ch, share in (tuning.get("absorbed") or {}).items()
            if share is not None
        )
        lines.append(
            f"  Tuned within     {float(tuning['scale_m']) / 1000:g} km areas; "
            f"{tuning.get('clusters')} clusters, {tuning.get('columns')} smooth "
            "columns" + (f"; variance absorbed {absorbed}" if absorbed else "")
        )
    if smoothing.get("candidates"):
        text = (
            f"  Spatial df       {smoothing.get('criterion', 'aic').upper()} over "
            f"{', '.join(str(d) for d in smoothing['candidates'])}"
        )
        if smoothing.get("selections"):
            text += (
                f"; the largest won {smoothing.get('selections_at_cap', 0)} of "
                f"{smoothing['selections']} selections"
            )
        if smoothing.get("paired_rule"):
            text += f"; paired tests: {smoothing['paired_rule']}"
        lines.append(text)
    return lines


def _posterior_lines(summary: list[dict]) -> list[str]:
    """Each split's CGI posterior, across the splits: median [min, max]."""
    rows = [
        e
        for e in summary
        if str(e.get("quantity")).startswith("posterior_")
        and _num(e.get("median")) is not None
    ]
    if not rows:
        return []
    lines = ["  Posterior on each split's train+val: median [min, max]"]
    for e in rows:
        median, lo, hi = (_num(e.get(k)) for k in ("median", "min", "max"))
        name = _CHANNELS.get(e.get("exposure"), e.get("exposure"))
        quantity = e["quantity"]
        if quantity == "posterior_beta":
            text = f"{'β':<22}{median:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            if e.get("forms"):
                text += f"  forms {e['forms']}"
        elif quantity == "posterior_weight":
            text = f"{'Weight ' + name:<22}{median:.3f} [{lo:.3f}, {hi:.3f}]"
            share = _num(e.get("share_largest"))
            if share is not None:
                text += f"  largest {share:.0%}"
        else:
            label = "R50 " if quantity == "posterior_r50" else "R90 "
            text = f"{label + name:<22}{median:.0f} m [{lo:.0f}, {hi:.0f}]"
        lines.append("    " + text)
    return lines


def resolve(value: str, artifacts: Path) -> Path:
    """A path the run recorded, found again: GUI runs record paths relative
    to the folder the GUI ran in, so those are rebased on the run's own
    artifacts folder."""
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    if artifacts.name in path.parts:
        tail = path.parts[path.parts.index(artifacts.name) + 1 :]
        return artifacts.joinpath(*tail)
    return path


def _params(bundle: dict) -> dict:
    params = bundle.get("averaged_params") or bundle.get("best_params") or {}
    return {k: v for k, v in params.items() if not str(k).startswith("__")}


def summary(label: str, bundle: dict, artifacts: Path) -> list[str]:
    """The headline of one outcome, as the results panel puts it."""
    metric = bundle.get("objective_metric") or "score"
    lines = [
        f"Outcome {label} · objective {metric} · formula "
        f"{bundle.get('cgi_formula') or '?'}"
    ]
    effects = bundle.get("cgi_effects") or {}
    test = effects.get("test") or {}
    test_ci = (bundle.get("test_results") or {}).get("test_ci") or {}
    if test_ci.get("status") == "fit_failed":
        lines.append("  Held-out test    fit failed (the held-out model diverged)")
    else:
        score = _num(test.get("score"))
        if score is None:
            score = _num(test_ci.get("observed"))
        ci = _interval(test_ci) or _interval(test)
        p = _num(test.get("p_value"))
        n = test_ci.get("n") or test.get("n")
        parts = [f"{score:.4f}" if score is not None else "—", ci]
        if n:
            parts.append(f"n={n}")
        if p is not None:
            parts.append(f"p={p:.3g}")
        lines.append("  Held-out test    " + "  ".join(x for x in parts if x))
    whole = effects.get("all") or {}
    if _num(whole.get("score")) is not None:
        lines.append(
            f"  Whole data       {_num(whole['score']):.4f}  {_interval(whole)}  "
            "(in-sample, optimistic)"
        )
    direction = bundle.get("direction_sign")
    if direction is not None:
        lines.append(
            "  Direction        "
            + (
                "positive: more greenery, higher outcome"
                if int(direction) > 0
                else "negative: more greenery, lower outcome"
            )
        )
    lines += _spatial_lines(bundle)

    standalones = bundle.get("standalones") or {}
    for channel, sb in standalones.items():
        tr = (sb or {}).get("test_results") or {}
        score = _num(tr.get("test_score"))
        if score is not None:
            name = _CHANNELS.get(channel, channel)
            ci = _interval(tr.get("test_ci") or {})
            lines.append(f"  {name + ' alone':<17}{score:.4f}  {ci}".rstrip())
    family = bundle.get("cgi_vs_standalone_paired_family") or []
    paired = family or [bundle.get("cgi_vs_standalone_paired") or {}]
    if any(_num(d.get("observed_diff")) is not None for d in paired):
        lines.append("  Held-out paired  CGI score − channel score = difference")
    for d in paired:
        diff = _num(d.get("observed_diff"))
        if diff is None:
            continue
        name = _CHANNELS.get(d.get("standalone_channel"), d.get("standalone_channel"))
        p = _num(d.get("p_value_holm"))
        p_text = f"Holm p={p:.3g}" if p is not None else ""
        if p is None and _num(d.get("p_value")) is not None:
            p_text = f"p={_num(d.get('p_value')):.3g}"
        lines.append(
            f"  CGI − {name:<10} {_difference(d)}  {_interval(d)}  {p_text}".rstrip()
        )
    head = bundle.get("cgi_vs_standalone_paired") or {}
    if head.get("verdict"):
        name = _CHANNELS.get(
            head.get("standalone_channel"), head.get("standalone_channel")
        )
        lines.append(
            f"  Verdict          CGI vs {name} alone (chosen on train+val): "
            f"{head['verdict']}"
        )
    comparators = bundle.get("comparators") or {}
    if comparators:
        lines.append("  Comparators      held-out score, then CGI − comparator")
    for row in bundle.get("cgi_vs_comparator_paired") or []:
        name = row.get("comparator")
        test = (comparators.get(name) or {}).get("test") or {}
        score, diff = _num(test.get("score")), _num(row.get("observed_diff"))
        p = _num(row.get("p_value_holm"))
        if row.get("descriptive"):
            p_text = "descriptive"
        elif p is not None:
            p_text = f"Holm p={p:.3g}"
        else:
            p_text = ""
        lines.append(
            f"    {name:<15}"
            + (f"{score:.4f}" if score is not None else "—")
            + (f"  {diff:+.4f}  {_interval(row)}" if diff is not None else "")
            + (f"  {p_text}" if p_text else "")
        )
    aic = bundle.get("cgi_vs_standalone_aic_bic") or {}
    if aic.get("ok"):
        best = _CHANNELS.get(aic.get("best_channel"), aic.get("best_channel"))
        lines.append(
            f"  AIC / BIC        in-sample, all channels vs {best} alone: "
            f"{aic.get('verdict')} (ΔAIC {_num(aic.get('delta_aic')):+.1f}, "
            f"ΔBIC {_num(aic.get('delta_bic')):+.1f})"
        )
    lines += _specificity_lines(bundle.get("specificity_contrast") or {})
    lines += _raw_lines(bundle.get("raw_effects") or {})
    lines += _weighted_lines(bundle.get("weighted") or {})
    lines += _outer_lines(bundle.get("outer_splits") or {})
    lines += _null_lines(bundle.get("outer_null") or {})

    params = _params(bundle)
    weights = {
        k[: -len("_weight")]: v for k, v in params.items() if k.endswith("_weight")
    }
    if weights:
        lines.append(
            "  Weights          "
            + ", ".join(f"{_CHANNELS.get(c, c)} {w}" for c, w in weights.items())
            + " (of 100)"
        )
    radii = {
        k[: -len("_radius")]: v for k, v in params.items() if k.endswith("_radius")
    }
    if radii:
        lines.append(
            "  Radii            "
            + ", ".join(f"{_CHANNELS.get(c, c)} {r} m" for c, r in radii.items())
        )
    disc = bundle.get("discovery_summary") or {}
    for i, channel in enumerate(disc.get("channels") or []):
        r50 = (disc.get("r50_mean") or [None] * (i + 1))[i]
        r90 = (disc.get("r90_mean") or [None] * (i + 1))[i]
        if _num(r50) is not None and _num(r90) is not None:
            name = _CHANNELS.get(channel, channel)
            lines.append(
                f"  Influence        {name}: half within {_num(r50):.0f} m, "
                f"90 % within {_num(r90):.0f} m"
            )

    for label_, key in (
        ("Reports", "report_dir"),
        ("Composite map", "composite_path"),
        ("Entity table", "entity_table_path"),
    ):
        if bundle.get(key):
            lines.append(f"  {label_:<17}{resolve(bundle[key], artifacts)}")
    return lines


# ────────────────────────────────────────────────────────────────────
# Commands
# ────────────────────────────────────────────────────────────────────


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CliError(f"Could not read {path}: {exc}") from exc


def cmd_results(args) -> int:
    path = bundle_path(args)
    payload = _read(path)
    picked = outcomes(payload, args.outcome)
    if args.json:
        report = {}
        for label, bundle in picked:
            folder = resolve(bundle.get("report_dir") or "study_results", path.parent)
            sfx = f"__{label}" if len(payload.get("ordered_labels") or []) > 1 else ""
            manifest = folder / f"results_summary{sfx}.json"
            report[label] = _read(manifest) if manifest.is_file() else bundle
        print(json.dumps(report, indent=1, default=str))
        return EXIT_OK
    print(f"Results: {path.parent}")
    for label, bundle in picked:
        print()
        print("\n".join(summary(label, bundle, path.parent)))
    return EXIT_OK


def cmd_export(args) -> int:
    path = bundle_path(args)
    payload = _read(path)
    picked = outcomes(payload, args.outcome)
    if len(picked) > 1:
        names = ", ".join(label for label, _ in picked)
        raise CliError(f"Several outcomes here; pick one with --outcome ({names}).")
    label, bundle = picked[0]
    table = bundle.get("entity_table_path")
    table = str(resolve(table, path.parent)) if table else None
    if not table or not Path(table).is_file():
        raise CliError(
            f"Run {path.parent.name} has no per-entity table (outcome {label}); "
            "runs write composite_entities.gpkg since this version."
        )
    if not args.output:
        print(table)
        return EXIT_OK
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    if suffix == Path(table).suffix.lower():
        shutil.copyfile(table, out)
    else:
        import geopandas as gpd
        import pandas as pd

        frame = (
            gpd.read_file(table, layer="cgi")
            if table.endswith(".gpkg")
            else pd.read_csv(table)
        )
        if suffix == ".csv":
            frame.drop(columns="geometry", errors="ignore").to_csv(out, index=False)
        elif suffix == ".geojson" and isinstance(frame, gpd.GeoDataFrame):
            frame.to_crs("EPSG:4326").to_file(out, driver="GeoJSON")
        elif suffix == ".gpkg" and isinstance(frame, gpd.GeoDataFrame):
            frame.to_file(out, driver="GPKG", layer="cgi")
        else:
            raise CliError(f"Cannot write {out.name}: use .gpkg, .csv or .geojson.")
    print(f"Wrote {out} ({label}).")
    return EXIT_OK


def cmd_cgi_map(args) -> int:
    from geofuse import cgi_generator

    if not cgi_generator.AVAILABLE:
        raise CliError("cgi-map is not available yet.")
    return EXIT_OK


def add_parsers(sub, common) -> None:
    """This module's commands on ``sub`` (which adds ``common`` itself)."""
    p = sub.add_parser("results", help="a fusion run's headline results")
    p.add_argument("run", metavar="JOB|FOLDER", help="a job id or a results folder")
    p.add_argument("--outcome", help="one outcome of a multi-outcome run")
    p.add_argument("--json", action="store_true", help="the full results manifest")
    p.set_defaults(func=cmd_results)

    p = sub.add_parser("export", help="a fusion run's per-entity CGI table")
    p.add_argument("run", metavar="JOB|FOLDER", help="a job id or a results folder")
    p.add_argument("-o", "--output", help="write it here (.gpkg, .csv or .geojson)")
    p.add_argument("--outcome", help="one outcome of a multi-outcome run")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser(
        "cgi-map", help="a fusion run's CGI over a whole area (not available yet)"
    )
    p.add_argument("run", metavar="JOB|FOLDER", help="a job id or a results folder")
    p.add_argument("area", help="the area to map (a vector file)")
    p.add_argument("-o", "--output", help="write it here (.tif)")
    p.set_defaults(func=cmd_cgi_map)
