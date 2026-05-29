"""Longitudinal data intake for the fusion engine's mixed-effects mode.

This module owns the structural
transformation from a user-supplied longitudinal target (one file with a wave
column, or N per-wave files joined on a shared entity-id column) into the
internal long-format GeoDataFrame the mixed-effects scorer consumes — one
row per ``(entity_id, wave)`` with a continuous ``years_since_baseline``
column.

File I/O is intentionally out of scope: callers (runner / UI) load each
input with ``gpd.read_file`` first and hand the frames to
:func:`build_long_format`. This keeps the module trivially unit-testable and
keeps file-resolution (loaded / uploaded / auto-downloaded) in the same place
as the cross-sectional path.

Time variable
-------------

``years_since_baseline`` is computed per ``(entity_id, wave)`` as the number
of days between the row's measurement date and the entity's earliest
measurement date, divided by 365.25 (so leap years average out). Each entity
gets its own baseline date — i.e. for entity X with waves
``[2010-01-15, 2013-02-20, 2017-08-01]`` the time column is
``[0.0, 3.10..., 7.55...]``. This is deliberately not an ordinal wave number
nor a calendar year: real cohorts have irregular inter-wave intervals per
participant, and the random-slope-on-time model needs the intervals to be on
the right scale.

Greenery file assignment
------------------------

The spec carries one filename per channel per wave (veg, terrain, ndvi). A
single file may be reused across every wave; :func:`describe_file_reuse`
flags channels that have no longitudinal variation so the UI can surface a
non-blocking warning (the model still fits — the channel just contributes
between-entity variation only, with no within-entity time effect for that
channel).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd

# Mixed-effects scoring metric names. Kept here (not in the scorer module) so
# this module can be imported without dragging statsmodels in.
MIXEDLM_METRICS: tuple[str, ...] = (
    "mixedlm_tstat",
    "mixedlm_marginal_r2",
    "mixedlm_lr",
    "mixedlm_coef",
)
DEFAULT_MIXEDLM_METRIC = "mixedlm_tstat"

# Channels that participate in the per-wave greenery file assignment. Terrain
# is GVI Cityscapes class 9 (horizontal flat greenery), not DEM; it varies
# year-to-year only if the user re-runs GVI with new street-view imagery.
GREENERY_CHANNELS: tuple[str, ...] = ("veg", "terrain", "ndvi")

IntakeMode = Literal["long", "wide"]


@dataclass(frozen=True)
class LongitudinalSpec:
    """Configuration for a longitudinal / mixed-effects fusion run.

    The spec is built by the UI (or by a CLI / programmatic caller) and
    threaded through :class:`geofuse.fusion.MetricFusionEngine` and the
    fusion runner. It is intentionally a plain dataclass so it round-trips
    cleanly through the job-record JSON used by the restart panel.

    Parameters
    ----------
    intake_mode
        ``"long"`` — one target file with one row per ``(entity, wave)`` and
        an explicit wave column. ``"wide"`` — N target files, one per wave,
        joined on a shared entity-id column.
    entity_id_col
        Name of the stable per-entity identifier column. Present in the long
        target, or in every wide per-wave file.
    wave_labels
        Ordered list of wave labels (strings). In long mode, every value in
        ``wave_col`` must appear in this list. In wide mode, the per-wave
        files are keyed by these labels.
    wave_col
        Long-mode only: the column name carrying the wave label per row.
        Ignored in wide mode (each file is one wave).
    date_col
        Name of the measurement-date column. In long mode this is a column
        in the single target. In wide mode every per-wave file must have a
        column with this name. Used to compute ``years_since_baseline``;
        parsed with :func:`pandas.to_datetime`.
    greenery_files
        Mapping ``channel -> {wave_label: file_path}`` for each of the three
        greenery channels. A channel may map every wave to the same path
        (warning surfaced by :func:`describe_file_reuse`). The runner is
        responsible for resolving each path against its source (loaded
        results / uploads / auto-download cache) before handing the spec to
        the engine.
    include_time_fixed_effect
        If true, the MixedLM gets ``+ years_since_baseline`` as a fixed
        effect (default: true). Turn off only for niche cases — a global
        temporal trend otherwise leaks into the greenery coefficient.
    random_slope_time
        If true, the random-effects structure is
        ``(1 + years_since_baseline | entity_id)`` (random intercept + random
        slope on time). If false, ``(1 | entity_id)`` (random intercept
        only). Default: true (per user 2026-05-28).
    scoring_metric
        One of :data:`MIXEDLM_METRICS`; the metric Optuna optimizes per
        trial. The other three are computed post-hoc on robust + top-20% +
        final trials. Default: ``"mixedlm_tstat"``.
    """

    intake_mode: IntakeMode
    entity_id_col: str
    wave_labels: tuple[str, ...]
    wave_col: str | None = None
    date_col: str = "measurement_date"
    greenery_files: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    include_time_fixed_effect: bool = True
    random_slope_time: bool = True
    scoring_metric: str = DEFAULT_MIXEDLM_METRIC

    def to_payload(self) -> dict:
        """Serialise to a plain-dict payload (for ``rec.params`` / restart)."""
        return {
            "intake_mode": self.intake_mode,
            "entity_id_col": self.entity_id_col,
            "wave_labels": list(self.wave_labels),
            "wave_col": self.wave_col,
            "date_col": self.date_col,
            "greenery_files": {
                ch: dict(per_wave) for ch, per_wave in self.greenery_files.items()
            },
            "include_time_fixed_effect": self.include_time_fixed_effect,
            "random_slope_time": self.random_slope_time,
            "scoring_metric": self.scoring_metric,
        }

    @classmethod
    def from_payload(cls, payload: Mapping) -> LongitudinalSpec:
        """Reverse of :meth:`to_payload`."""
        return cls(
            intake_mode=payload["intake_mode"],
            entity_id_col=payload["entity_id_col"],
            wave_labels=tuple(payload["wave_labels"]),
            wave_col=payload.get("wave_col"),
            date_col=payload.get("date_col", "measurement_date"),
            greenery_files={
                ch: dict(per_wave)
                for ch, per_wave in (payload.get("greenery_files") or {}).items()
            },
            include_time_fixed_effect=bool(
                payload.get("include_time_fixed_effect", True)
            ),
            random_slope_time=bool(payload.get("random_slope_time", True)),
            scoring_metric=payload.get("scoring_metric", DEFAULT_MIXEDLM_METRIC),
        )


def validate_spec(spec: LongitudinalSpec) -> list[str]:
    """Return a list of human-readable error messages for a malformed spec.

    Empty list = spec is internally consistent. Does not check the contents
    of any GeoDataFrame (that's :func:`build_long_format`'s job).
    """
    errs: list[str] = []
    if spec.intake_mode not in ("long", "wide"):
        errs.append(f"intake_mode must be 'long' or 'wide', got {spec.intake_mode!r}.")
    if not spec.entity_id_col:
        errs.append("entity_id_col is required.")
    if not spec.wave_labels:
        errs.append("wave_labels must contain at least one wave.")
    if len(set(spec.wave_labels)) != len(spec.wave_labels):
        errs.append("wave_labels contains duplicates.")
    if spec.intake_mode == "long" and not spec.wave_col:
        errs.append("wave_col is required when intake_mode == 'long'.")
    if spec.scoring_metric not in MIXEDLM_METRICS:
        errs.append(
            f"scoring_metric must be one of {MIXEDLM_METRICS}, "
            f"got {spec.scoring_metric!r}."
        )
    for ch in GREENERY_CHANNELS:
        per_wave = spec.greenery_files.get(ch, {})
        missing = [w for w in spec.wave_labels if w not in per_wave]
        if missing:
            errs.append(
                f"greenery_files[{ch!r}] is missing entries for waves: {missing}."
            )
    extra_channels = set(spec.greenery_files) - set(GREENERY_CHANNELS)
    if extra_channels:
        errs.append(
            f"greenery_files contains unknown channels: {sorted(extra_channels)}. "
            f"Expected one or more of {list(GREENERY_CHANNELS)}."
        )
    return errs


def describe_file_reuse(spec: LongitudinalSpec) -> dict[str, bool]:
    """Map ``channel -> True`` when every wave shares one file for that channel.

    The UI uses this to surface a non-blocking warning ("vegetation has no
    longitudinal variation — only between-entity contrast will inform that
    channel"). Returns ``True`` for channels with exactly one distinct file
    across all waves (including the degenerate one-wave case).
    """
    out: dict[str, bool] = {}
    for ch in GREENERY_CHANNELS:
        per_wave = spec.greenery_files.get(ch, {})
        distinct = {per_wave[w] for w in spec.wave_labels if w in per_wave}
        out[ch] = len(distinct) <= 1
    return out


def build_long_format(
    spec: LongitudinalSpec,
    target_input: gpd.GeoDataFrame | Sequence[tuple[str, gpd.GeoDataFrame]],
    outcome_col: str,
    covariate_cols: Iterable[str] = (),
) -> gpd.GeoDataFrame:
    """Normalise the user's longitudinal target into the internal long format.

    Long mode (``target_input`` is a single :class:`gpd.GeoDataFrame`) — the
    frame already has one row per ``(entity, wave)``; this function validates
    columns and derives ``years_since_baseline`` per entity.

    Wide mode (``target_input`` is a sequence of ``(wave_label, frame)``
    pairs) — each frame holds one wave's worth of rows keyed by
    ``entity_id_col``. The function concatenates them in wave-label order
    after tagging each row with its wave, then derives
    ``years_since_baseline`` per entity.

    Output columns
    --------------

    The returned GeoDataFrame carries (in this order):

    - ``entity_id`` — the value of ``spec.entity_id_col`` (renamed for the
      engine's convenience).
    - ``wave`` — the wave label (string).
    - ``years_since_baseline`` — float, per entity (baseline wave = ``0.0``).
    - ``outcome_col`` — passed through unchanged.
    - every column in ``covariate_cols`` — passed through unchanged.
    - ``geometry`` — passed through unchanged from the input row.

    Rows missing the outcome, the date column, or the entity id are dropped
    (with a count returned in ``GeoDataFrame.attrs["dropped_rows"]`` so the
    runner can log a summary). Rows with valid data but a wave label not in
    ``spec.wave_labels`` are also dropped (with the count surfaced the same
    way).
    """
    errs = validate_spec(spec)
    if errs:
        raise ValueError("Invalid LongitudinalSpec: " + "; ".join(errs))

    cov_cols = list(covariate_cols)
    if spec.intake_mode == "long":
        if not isinstance(target_input, gpd.GeoDataFrame):
            raise TypeError(
                "intake_mode 'long' requires target_input to be a GeoDataFrame."
            )
        frame = _prepare_long_input(target_input, spec, outcome_col, cov_cols)
    else:
        if isinstance(target_input, gpd.GeoDataFrame):
            raise TypeError(
                "intake_mode 'wide' requires target_input to be a sequence of "
                "(wave_label, GeoDataFrame) pairs."
            )
        frame = _prepare_wide_input(list(target_input), spec, outcome_col, cov_cols)

    n_before = len(frame)
    frame = _drop_rows_missing_required(frame, outcome_col, spec.entity_id_col)
    dropped_missing = n_before - len(frame)

    # Wave filter — drop any rows whose wave label isn't in the configured set.
    in_set = frame["wave"].astype(str).isin(spec.wave_labels)
    dropped_unknown_wave = int((~in_set).sum())
    frame = frame.loc[in_set].copy()

    # Per-(entity, wave) duplicate guard — only valid when the same entity
    # has multiple rows for distinct waves; same entity + same wave is an
    # input error.
    dup_mask = frame.duplicated([spec.entity_id_col, "wave"], keep=False)
    if dup_mask.any():
        sample = (
            frame.loc[dup_mask, [spec.entity_id_col, "wave"]].head(5).to_dict("records")
        )
        raise ValueError(
            f"Duplicate (entity_id, wave) rows detected — first few: {sample}. "
            "Each (entity, wave) must appear exactly once across the longitudinal "
            "input."
        )

    frame = _compute_years_since_baseline(frame, spec)
    out = _project_columns(frame, spec, outcome_col, cov_cols)

    out.attrs["dropped_rows"] = {
        "missing_required": int(dropped_missing),
        "unknown_wave": int(dropped_unknown_wave),
    }
    out.attrs["wave_labels"] = list(spec.wave_labels)
    return out


def parse_date_column(s: pd.Series) -> pd.Series:
    """Parse a measurement-date column with flexible input formats.

    Accepts:
    - Full ISO dates (``"2010-01-15"``).
    - Year + month (``"2010-01"``, ``"2010/01"``).
    - Year-only strings (``"2010"``).
    - Numeric years (``2010``, ``2010.0`` as int / float columns).
    - Anything else :func:`pandas.to_datetime` can parse natively.

    Year-only values normalise to ``YYYY-01-01``; year+month values to
    ``YYYY-MM-01``. Unparseable values become ``NaT`` and the
    corresponding rows are later dropped by :func:`build_long_format`.
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):
        non_null = s.dropna()
        # Treat as year-only when every non-null value lies in a plausible
        # calendar-year range; otherwise fall through to the string path
        # (which will produce NaT and surface the column as unparseable).
        if len(non_null) and ((non_null >= 1500) & (non_null <= 2200)).all():
            parsed = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
            parsed.loc[non_null.index] = pd.to_datetime(
                non_null.astype(int).astype(str), errors="coerce"
            )
            return parsed
    return pd.to_datetime(s.astype(str), errors="coerce")


def _prepare_long_input(
    gdf: gpd.GeoDataFrame,
    spec: LongitudinalSpec,
    outcome_col: str,
    cov_cols: list[str],
) -> gpd.GeoDataFrame:
    required = {spec.entity_id_col, spec.wave_col, spec.date_col, outcome_col}
    _assert_columns_present(gdf, required, "long-format target")
    _assert_columns_present(gdf, set(cov_cols), "long-format target (covariates)")
    out = gdf.copy()
    out["wave"] = out[spec.wave_col].astype(str)
    out[spec.date_col] = parse_date_column(out[spec.date_col])
    return out


def _prepare_wide_input(
    pairs: list[tuple[str, gpd.GeoDataFrame]],
    spec: LongitudinalSpec,
    outcome_col: str,
    cov_cols: list[str],
) -> gpd.GeoDataFrame:
    if not pairs:
        raise ValueError("intake_mode 'wide' requires at least one (wave, frame) pair.")
    seen_labels: set[str] = set()
    tagged: list[gpd.GeoDataFrame] = []
    for wave_label, frame in pairs:
        if wave_label in seen_labels:
            raise ValueError(f"Wave label {wave_label!r} supplied more than once.")
        seen_labels.add(wave_label)
        required = {spec.entity_id_col, spec.date_col, outcome_col}
        _assert_columns_present(frame, required, f"wave {wave_label!r} target")
        _assert_columns_present(
            frame, set(cov_cols), f"wave {wave_label!r} target (covariates)"
        )
        copy = frame.copy()
        copy["wave"] = wave_label
        copy[spec.date_col] = parse_date_column(copy[spec.date_col])
        tagged.append(copy)

    # Concatenate; pandas preserves the union of columns. Use the first
    # frame's CRS as authoritative; warn on mismatch by reprojecting silently
    # would mask a user error, so require explicit match here.
    base_crs = tagged[0].crs
    for i, frame in enumerate(tagged[1:], start=1):
        if frame.crs != base_crs:
            raise ValueError(
                f"Wide-mode wave {pairs[i][0]!r} has CRS {frame.crs} but the first "
                f"wave is {base_crs}. Reproject inputs to a common CRS before submit."
            )
    combined = gpd.GeoDataFrame(
        pd.concat(tagged, ignore_index=True), geometry="geometry", crs=base_crs
    )
    return combined


def _assert_columns_present(
    frame: gpd.GeoDataFrame, cols: set[str], context: str
) -> None:
    if not cols:
        return
    missing = sorted(c for c in cols if c not in frame.columns)
    if missing:
        raise ValueError(
            f"{context} is missing required columns: {missing}. "
            f"Available: {sorted(frame.columns)}."
        )


def _drop_rows_missing_required(
    frame: gpd.GeoDataFrame, outcome_col: str, entity_id_col: str
) -> gpd.GeoDataFrame:
    mask = frame[entity_id_col].notna() & frame[outcome_col].notna()
    return frame.loc[mask].copy()


def _compute_years_since_baseline(
    frame: gpd.GeoDataFrame, spec: LongitudinalSpec
) -> gpd.GeoDataFrame:
    # Baseline date is per-entity: each entity's own earliest measurement
    # date. Handles irregular inter-wave intervals that differ per participant
    # (the standard structure of cohort studies).
    baseline = frame.groupby(spec.entity_id_col)[spec.date_col].transform("min")
    delta_days = (frame[spec.date_col] - baseline).dt.total_seconds() / 86400.0
    out = frame.copy()
    out["years_since_baseline"] = delta_days / 365.25
    # Drop rows with unparseable dates (NaT propagates to NaN here).
    keep = out["years_since_baseline"].notna()
    return out.loc[keep].copy()


def _project_columns(
    frame: gpd.GeoDataFrame,
    spec: LongitudinalSpec,
    outcome_col: str,
    cov_cols: list[str],
) -> gpd.GeoDataFrame:
    cols = [
        spec.entity_id_col,
        "wave",
        "years_since_baseline",
        outcome_col,
        *cov_cols,
        "geometry",
    ]
    # Deduplicate while preserving order (entity_id_col may appear if a
    # covariate accidentally points at it).
    seen: set[str] = set()
    ordered: list[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    out = frame[ordered].copy()
    out = out.rename(columns={spec.entity_id_col: "entity_id"})
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=frame.crs)
    return out


def channel_files_per_entity(
    spec: LongitudinalSpec, long_frame: gpd.GeoDataFrame, channel: str
) -> np.ndarray:
    """Return an array of length ``len(long_frame)`` giving each row's file path.

    Used by the pre-aggregation cache's wave-aware fingerprinting: the file
    assigned to ``(channel, row.wave)`` is the source for that row's
    aggregation. The runner uses this to resolve which channel-fingerprint
    each cache row belongs to.
    """
    if channel not in GREENERY_CHANNELS:
        raise ValueError(
            f"Unknown channel {channel!r}; expected one of {GREENERY_CHANNELS}."
        )
    per_wave = spec.greenery_files.get(channel, {})
    waves = long_frame["wave"].astype(str).to_numpy()
    return np.array([per_wave.get(w, "") for w in waves], dtype=object)
