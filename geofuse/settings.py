"""Every user-facing NDVI, GVI and fusion setting, with its default.

One schema per engine records each setting's type, allowed values, bounds and
default — the values the tabs start from. Presets, the tabs and the command
line all read it, so a default cannot differ between them. A run's inputs
(files, dates, outcome and covariate columns) are not settings and are not
listed here.

The module imports no engine (NDVI aggregation chains are checked by the
pure :mod:`geofuse.ndvi_chains`). Its choice lists repeat the engines' own
constants, and a test holds the two equal.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .ndvi_chains import ChainError, check_chains


class SettingError(ValueError):
    """A value a setting does not accept, with the setting named."""


@dataclass(frozen=True)
class Setting:
    """One setting: its type, default, allowed values and a one-line help.

    ``kind`` is ``bool``, ``int``, ``float``, ``choice``, ``multichoice``,
    ``text`` or ``chains`` (NDVI aggregation chains: ``{name, steps}`` tables
    or ``"name: op, op"`` lines, stored as tables). A ``None`` default marks
    an optional setting whose absence has a meaning of its own (given in
    ``help``). ``auto_zero`` marks a number whose 0 means "work it out" and is
    stored as ``None``, the way the tabs read a 0 in those fields. ``step``
    restricts an int to ``lo + k * step``.
    """

    kind: str
    default: Any = None
    choices: tuple = ()
    lo: float | None = None
    hi: float | None = None
    step: int | None = None
    auto_zero: bool = False
    help: str = ""

    def check(self, key: str, value: Any) -> Any:
        """The value as stored, or :class:`SettingError` naming ``key``."""
        if value is None:
            if self.default is None or self.auto_zero:
                return None
            raise SettingError(f"'{key}' needs a value.")
        if self.kind == "bool":
            if not isinstance(value, bool):
                raise SettingError(f"'{key}' must be true or false, not {value!r}.")
            return value
        if self.kind in ("int", "float"):
            return self._number(key, value)
        if self.kind == "choice":
            if value not in self.choices:
                raise SettingError(
                    f"'{key}' must be one of {list(self.choices)}, not {value!r}."
                )
            return value
        if self.kind == "multichoice":
            if not isinstance(value, list) or not value:
                raise SettingError(f"'{key}' must be a non-empty list.")
            bad = [v for v in value if v not in self.choices]
            if bad:
                raise SettingError(
                    f"'{key}' may only contain {list(self.choices)}, not {bad}."
                )
            return [c for c in self.choices if c in value]
        if self.kind == "text":
            if not isinstance(value, str):
                raise SettingError(f"'{key}' must be text, not {value!r}.")
            return value
        if self.kind == "chains":
            try:
                return check_chains(value)
            except ChainError as exc:
                raise SettingError(f"'{key}': {exc}") from None
        raise AssertionError(self.kind)

    def _number(self, key: str, value: Any) -> int | float | None:
        whole = self.kind == "int"
        if isinstance(value, bool) or not isinstance(value, int | float):
            noun = "a whole number" if whole else "a number"
            raise SettingError(f"'{key}' must be {noun}, not {value!r}.")
        if whole and not isinstance(value, int):
            raise SettingError(f"'{key}' must be a whole number, not {value!r}.")
        if self.auto_zero and value == 0:
            return None
        if (self.lo is not None and value < self.lo) or (
            self.hi is not None and value > self.hi
        ):
            hi = "" if self.hi is None else f" and {self.hi}"
            span = f"between {self.lo}{hi}" if hi else f"at least {self.lo}"
            raise SettingError(f"'{key}' must be {span}, not {value}.")
        if self.step and (value - (self.lo or 0)) % self.step:
            raise SettingError(
                f"'{key}' must be a multiple of {self.step} from {self.lo}, "
                f"not {value}."
            )
        return int(value) if whole else float(value)


def _bool(default: bool, help: str = "") -> Setting:
    return Setting("bool", default, help=help)


def _int(default, lo, hi, help: str = "", **kw) -> Setting:
    return Setting("int", default, lo=lo, hi=hi, help=help, **kw)


def _choice(default, choices, help: str = "") -> Setting:
    return Setting("choice", default, tuple(choices), help=help)


# ────────────────────────────────────────────────────────────────────
# Choice lists (each equals an engine constant; see the module docstring)
# ────────────────────────────────────────────────────────────────────

#: ``geofuse.ndvi.TEMPORAL_REDUCERS``.
TEMPORAL_REDUCERS = ("median", "mean", "max")
#: ``geofuse.ndvi.SATELLITES``.
SATELLITES = ("auto", "sentinel2", "landsat", "landsat_no_slc_off")
#: ``geofuse.ndvi.REFLECTANCES``.
REFLECTANCES = ("surface", "toa")
#: ``geofuse.ndvi.CLOUD_HANDLINGS``.
CLOUD_HANDLINGS = ("scene_filter", "pixel_only", "scene_only")
#: ``geofuse.metric_sampling.LADDER_SPACINGS``.
LADDER_SPACINGS = ("linear", "log")
#: ``geofuse.bayesian_index.RADIUS_PRIORS``.
RADIUS_PRIORS = ("rungs", "range")
#: ``geofuse.bayesian_index.FORMS``, plus ``sweep`` (fit both, keep the better).
INDEX_FORMS = ("sweep", "linear", "synergy")
#: Channel sets a fusion job can pick between (``runners._CHANNEL_SETS``).
CHANNEL_SETS: dict[str, tuple[str, ...]] = {
    "ndvi + gvi": ("ndvi", "gvi"),
    "ndvi + veg + terrain": ("ndvi", "veg", "terrain"),
}
#: Objective metrics of a cross-sectional fusion run, first = default.
CROSS_SECTIONAL_METRICS = (
    "partial_distance_corr",
    "distance_corr",
    "spearman",
    "r2",
    "nrmse",
    "mutual_info",
    "quartile_contrast",
    "logit_tstat",
    "logit_coef",
)
#: Objective metrics of a longitudinal (mixed-effects) run, first = default.
LONGITUDINAL_METRICS = (
    "mixedlm_tstat",
    "mixedlm_marginal_r2",
    "mixedlm_lr",
    "mixedlm_coef",
    "gee_logit_tstat",
    "gee_logit_coef",
)
#: Metrics that need a two-valued outcome.
BINARY_ONLY_METRICS = frozenset(
    {"logit_tstat", "logit_coef", "gee_logit_tstat", "gee_logit_coef"}
)
#: What a longitudinal run tunes the composite to.
ASSOCIATION_TARGETS = ("level", "decline_overall", "decline_average", "decline_change")
#: How each trial's mixed-model variance components are estimated.
SEARCH_SCORING_METHODS = ("mom_em3", "mom_em1", "mom", "exact")
#: Spatial-confounding adjustments of the fusion objective.
SPATIAL_ADJUST_METHODS = ("none", "ks_aic", "spatial_plus")

# ────────────────────────────────────────────────────────────────────
# NDVI
# ────────────────────────────────────────────────────────────────────

_STATS = Setting("multichoice", ["median"], TEMPORAL_REDUCERS)
_CHAINS = Setting(
    "chains", [], help="Aggregation chains in place of the statistics; [] = none."
)

NDVI: dict[str, Setting] = {
    "satellite": _choice(
        "auto", SATELLITES, "Sensor; auto = Sentinel-2 from 2017-03-28, Landsat before."
    ),
    "reflectance": _choice("surface", REFLECTANCES, "Surface or top-of-atmosphere."),
    "cloud_handling": _choice(
        "scene_filter", CLOUD_HANDLINGS, "Drop cloudy scenes and/or mask cloudy pixels."
    ),
    "cloud_max": _int(10, 0, 100, "Scenes above this cloud cover (%) are dropped."),
    "resolution_m": _int(10, 10, 1000, "Pixel size of the export (m)."),
    "buffer_m": _int(0, 0, 2000, "Grow the study area by this distance (m)."),
    "coverage_rescue": _bool(True, "Widen a window by ±50 % once if under 3 images."),
    "water_mask": _bool(False, "Mask pixels open water covers at least half the time."),
    "substitute_2012": _bool(False, "Fill a window inside 2012 with 2013."),
    "save_geotiff": _bool(True, "Write the GeoTIFF."),
    "save_gpkg": _bool(False, "Write per-pixel points as a GeoPackage."),
    "save_geojson": _bool(False, "Write per-pixel points as GeoJSON."),
    "save_cluster_tiles": _bool(False, "One GeoTIFF per connected component."),
    "statistics_ranges": _STATS,
    "statistics_specific": _STATS,
    "statistics_column": _STATS,
    "chains_ranges": _CHAINS,
    "chains_specific": _CHAINS,
    "chains_column": _CHAINS,
    "season_start_month": _int(6, 1, 12, "First month of the per-year season."),
    "season_end_month": _int(9, 1, 12, "Last month of the per-year season."),
    "specific_window_days": _int(30, 7, 180, "Days either side of a specific date."),
}

# ────────────────────────────────────────────────────────────────────
# GVI
# ────────────────────────────────────────────────────────────────────

GVI: dict[str, Setting] = {
    "download_mode": _choice(
        "package", ("package", "api"), "Scraper, or the Street View API (needs a key)."
    ),
    "grid_resolution_m": _int(50, 20, 500, "Sampling grid spacing (m)."),
    "buffer_m": _int(0, 0, 2000, "Grow the study area by this distance (m)."),
    "save_raw": _bool(False, "Keep downloaded panoramas and segmentation masks."),
    "show_sampling_grid": _bool(False, "Draw the sampling grid on the tab's map."),
    "save_gpkg": _bool(True, "Write the GeoPackage."),
    "save_geotiff": _bool(False, "Write per-cluster GeoTIFF tiles."),
    "save_geojson": _bool(False, "Write GeoJSON."),
    "capture_mode": _choice(
        "recent", ("recent", "year", "column"), "Most recent, one year, or per year."
    ),
    "target_year": _int(None, 2007, 2100, "Year to match; none = the current year."),
    "limit_year_difference": _bool(False, "Only accept captures near the year."),
    "max_year_difference": _int(2, 0, 20, "Largest accepted year gap."),
    "limit_months": _bool(False, "Only use captures taken in a month window."),
    "month_start": _int(5, 1, 12, "First month of the capture window."),
    "month_end": _int(10, 1, 12, "Last month of the capture window."),
    "month_policy": _choice(
        "exclude", ("exclude", "prefer"), "Outside the window: drop, or last resort."
    ),
    "download_concurrency": _int(
        16, 1, 64, "Panoramas downloaded at once (Google's throttle still applies)."
    ),
    "batch_size": _int(
        1, 1, 32, "Images per GPU forward; 1 gives image-by-image results exactly."
    ),
    "batch_wait_ms": _int(
        20, 0, 2000, "Longest wait for a GPU batch to fill, in milliseconds."
    ),
    "precision": _choice(
        "fp16", ("fp16", "fp32"), "GPU arithmetic; fp32 repeats earlier runs' masks."
    ),
    "log_points": _bool(False, "Log a line for every point and panorama (debugging)."),
}

#: GVI settings that size the download pool and the GPU batches.
GVI_THROUGHPUT = ("download_concurrency", "batch_size", "batch_wait_ms")
#: GVI settings a job records and presets leave out: how a run computes and
#: what it logs.
GVI_RUN_OPTIONS = (*GVI_THROUGHPUT, "precision", "log_points")

# ────────────────────────────────────────────────────────────────────
# Fusion
# ────────────────────────────────────────────────────────────────────


def _ladder(channel: str) -> dict[str, Setting]:
    up = channel.upper()
    return {
        f"{channel}_buffer_min_m": _int(100, 50, 4900, f"Smallest {up} radius (m)."),
        f"{channel}_buffer_max_m": _int(1000, 100, 5000, f"Largest {up} radius (m)."),
        f"{channel}_buffer_step_m": _int(
            100, 10, 500, f"{up} radius step (m) of a linear ladder."
        ),
        f"{channel}_ladder_spacing": _choice(
            "linear", LADDER_SPACINGS, f"{up} ladder: fixed steps or log-spaced."
        ),
        f"{channel}_ladder_rungs": _int(15, 2, 60, f"{up} radii of a log ladder."),
    }


FUSION: dict[str, Setting] = {
    **_ladder("gvi"),
    **_ladder("ndvi"),
    # ── Objective and split ─────────────────────────────────────
    "objective_metric": _choice(
        CROSS_SECTIONAL_METRICS[0],
        CROSS_SECTIONAL_METRICS + LONGITUDINAL_METRICS,
        "Score every candidate is ranked on (longitudinal runs: mixedlm_tstat).",
    ),
    "test_size": Setting("float", 0.25, lo=0.1, hi=0.5, help="Held-out test fraction."),
    "n_bins": _int(5, 3, 10, "Quantile bins of the stratified split."),
    "residualize_method": _choice(
        "linear", ("linear", "spline"), "How covariates are partialled out."
    ),
    "search_scoring_method": _choice(
        "mom_em3", SEARCH_SCORING_METHODS, "Mixed-model fit per trial (longitudinal)."
    ),
    "exposure_iqr": Setting(
        "float",
        None,
        lo=0.0,
        hi=10.0,
        auto_zero=True,
        help="Greenery increment of the per-IQR effect; 0 = from the composite.",
    ),
    "multi_objective_requested": _bool(False, "Several outcomes as one joint run."),
    # ── Discovery ───────────────────────────────────────────────
    "channel_set": _choice(
        "ndvi + gvi", tuple(CHANNEL_SETS), "Greenery channels the composite mixes."
    ),
    "index_form": _choice("sweep", INDEX_FORMS, "Functional form; sweep fits both."),
    "sweep_splits": _int(40, 5, 200, "Held-out splits each candidate is scored on."),
    "radius_prior": _choice("rungs", RADIUS_PRIORS, "Distance-of-influence prior."),
    "run_standalones": _bool(False, "Also run each channel on its own."),
    "check_collinearity": _bool(True, "Drop channels above the VIF threshold."),
    "vif_threshold": Setting(
        "float", 10.0, lo=2.0, hi=100.0, help="Largest variance inflation kept."
    ),
    # ── Validation and reporting ────────────────────────────────
    "discovery_reps": _int(5, 0, 50, "Independent repeats of the discovery."),
    "discovery_shuffles": _int(12, 1, 100, "Shuffles per discovery repeat."),
    "gain_splits": _int(20, 0, 100, "Splits comparing composite and standalones."),
    "null_calibration_runs": _int(16, 0, 1000, "Permuted-outcome refits."),
    "posterior_draws": _int(800, 100, 4000, "Posterior draws per chain."),
    "posterior_warmup": _int(None, 100, 4000, "Warm-up draws; none = as many."),
    "posterior_chains": _int(4, 1, 8, "Posterior chains."),
    "negative_control_retune": _bool(
        False, "Also re-tune on each negative control and compare."
    ),
    "report_ci_bootstrap": _int(
        None, 100, 100000, "Test-CI bootstrap; none = 2,000 or 10,000 by metric."
    ),
    "report_effects_bootstrap": _int(2000, 100, 100000, "Effects bootstrap."),
    "report_effects_permutations": _int(1000, 0, 100000, "Effects permutations."),
    "report_paired_bootstrap": _int(2000, 100, 100000, "Paired-difference bootstrap."),
    # ── Run control ─────────────────────────────────────────────
    "resume_existing_study": _bool(True, "Reuse a finished study's caches."),
    "clear_preaggregation_cache": _bool(False, "Rebuild the target's cache first."),
    # ── Per-pixel scoring (vector targets) ──────────────────────
    "cgi_grid_spacing_m": _int(50, 10, 500, "CGI grid pixel size (m).", step=10),
    "whole_grid_scaling": _bool(True, "Scale the composite map to [0, 1]."),
    "normalize_channels": _bool(True, "Scale channels to [0, 1] before mixing."),
    "area_balanced_split": _bool(True, "Balance polygon area across the split."),
    "spatial_split": _bool(True, "Hold out whole spatial blocks."),
    "spatial_block_size_m": Setting(
        "float", None, lo=0.0, auto_zero=True, help="Block edge (m); 0 = automatic."
    ),
    "spatial_adjust_method": _choice(
        "none", SPATIAL_ADJUST_METHODS, "Spatial-confounding adjustment."
    ),
    "spatial_adjust_max_df": _int(10, 1, 50, "Largest spatial df per cluster."),
    "spatial_adjust_eps_m": Setting(
        "float", None, lo=0.0, auto_zero=True, help="Cluster gap (m); 0 = automatic."
    ),
    # ── Longitudinal (mixed-effects) ────────────────────────────
    "association_target": _choice(
        "level", ASSOCIATION_TARGETS, "Level, or one of the decline terms."
    ),
    "random_slope_time": _bool(True, "Random slope on time per entity."),
    "include_time_fixed_effect": _bool(True, "Time since baseline as a fixed effect."),
    "include_wave_fixed_effects": _bool(True, "One indicator per wave."),
    "decline_average_exposure": _bool(False, "Report the average-exposure term."),
    "decline_exposure_change": _bool(False, "Report the exposure-change term."),
}

#: Objective metric of a longitudinal run unless one is chosen.
LONGITUDINAL_DEFAULT_METRIC = LONGITUDINAL_METRICS[0]

SCHEMAS: dict[str, dict[str, Setting]] = {"ndvi": NDVI, "gvi": GVI, "fusion": FUSION}

#: One key that sets several settings at once (run files, presets, ``--set``):
#: NDVI ``chains`` gives every date mode the same aggregation chains.
SHORTHANDS: dict[str, dict[str, tuple[str, ...]]] = {
    "ndvi": {"chains": ("chains_ranges", "chains_specific", "chains_column")},
}


def expand_shorthands(engine: str, table: Mapping[str, Any]) -> dict[str, Any]:
    """``table`` with each shorthand of ``engine`` spread over the settings it
    stands for; a setting given in the same table keeps its own value."""
    out = dict(table)
    for short, keys in SHORTHANDS.get(engine, {}).items():
        if short in out:
            value = out.pop(short)
            for key in keys:
                out.setdefault(key, value)
    return out


def defaults(engine: str) -> dict[str, Any]:
    """Every setting of ``engine`` at its default (lists copied)."""
    return {
        key: list(s.default) if isinstance(s.default, list) else s.default
        for key, s in SCHEMAS[engine].items()
    }


def check(engine: str, values: Mapping[str, Any]) -> dict[str, Any]:
    """``values`` checked against ``engine``'s schema, as stored.

    Shorthands are spread first (:func:`expand_shorthands`). Raises
    :class:`SettingError` naming the first unknown setting or bad value, and
    when a month window or season runs backwards where the engine needs it
    forwards.
    """
    schema = SCHEMAS[engine]
    out: dict[str, Any] = {}
    for key, value in expand_shorthands(engine, values).items():
        if key not in schema:
            raise SettingError(f"Unknown {engine.upper()} setting '{key}'.")
        out[key] = schema[key].check(key, value)
    lo, hi = "season_start_month", "season_end_month"
    if engine == "ndvi" and lo in out and hi in out and out[lo] > out[hi]:
        raise SettingError(f"'{lo}' must be on or before '{hi}'.")
    return out
