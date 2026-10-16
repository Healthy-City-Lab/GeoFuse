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
            f"  CGI − {name:<10} {diff:+.4f}  {_interval(d)}  {p_text}".rstrip()
        )
    aic = bundle.get("cgi_vs_standalone_aic_bic") or {}
    if aic.get("ok"):
        best = _CHANNELS.get(aic.get("best_channel"), aic.get("best_channel"))
        lines.append(
            f"  AIC / BIC        CGI vs {best} alone: {aic.get('verdict')} "
            f"(ΔAIC {_num(aic.get('delta_aic')):+.1f}, "
            f"ΔBIC {_num(aic.get('delta_bic')):+.1f})"
        )

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
