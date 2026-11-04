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
    ``intlist`` (whole numbers within ``lo`` .. ``hi``, given as a list or as
    comma-separated text, stored sorted without repeats, or in the order given
    when ``ordered``), ``text``, ``floatmap`` (positive numbers by name, given
    as a table or as ``"name=value, name=value"`` text, stored as a table) or
    ``chains`` (NDVI aggregation chains: ``{name, steps}`` tables or
    ``"name: op, op"`` lines, stored as tables). A ``None`` default marks
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
    ordered: bool = False
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
        if self.kind == "intlist":
            return self._int_list(key, value)
        if self.kind == "text":
            if not isinstance(value, str):
                raise SettingError(f"'{key}' must be text, not {value!r}.")
            return value
        if self.kind == "floatmap":
            return self._float_map(key, value)
        if self.kind == "chains":
            try:
                return check_chains(value)
            except ChainError as exc:
                raise SettingError(f"'{key}': {exc}") from None
        raise AssertionError(self.kind)

    def _int_list(self, key: str, value: Any) -> list[int]:
        if isinstance(value, str):
            parts = [p.strip() for p in value.replace(";", ",").split(",")]
            try:
                value = [int(p) for p in parts if p]
            except ValueError:
                raise SettingError(
                    f"'{key}' must be whole numbers separated by commas, "
                    f"not {value!r}."
                ) from None
        if not isinstance(value, list | tuple) or not value:
            raise SettingError(f"'{key}' must be a non-empty list of whole numbers.")
        whole = Setting("int", 0, lo=self.lo, hi=self.hi)
        numbers = [whole._number(key, v) for v in value]
        if self.ordered:
            return list(dict.fromkeys(numbers))
        return sorted(set(numbers))

    def _float_map(self, key: str, value: Any) -> dict[str, float]:
        if isinstance(value, str):
            pairs = {}
            for part in value.replace(";", ",").split(","):
                if not part.strip():
                    continue
                name, sep, number = part.partition("=")
                try:
                    pairs[name.strip()] = float(number)
                except ValueError:
                    sep = ""
                if not sep or not name.strip():
                    raise SettingError(
                        f"'{key}' must read name=value, name=value, not {value!r}."
                    ) from None
            value = pairs
        if not isinstance(value, Mapping) or not value:
            raise SettingError(f"'{key}' must be a table of names and numbers.")
        positive = Setting("float", 0.0, lo=0.0, hi=self.hi)
        out = {}
        for name, number in value.items():
            v = positive._number(f"{key}.{name}", number)
            if v <= 0:
                raise SettingError(f"'{key}.{name}' must be above 0, not {v}.")
            out[str(name)] = v
        return out

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
# Choice lists (each equals an engine constant; see the docstring)
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
#: ``geofuse.exposure.KERNELS``.
EXPOSURE_KERNELS = ("none", "uniform", "linear", "gaussian")
#: ``geofuse.bayesian_index.FORMS``, plus ``sweep`` (fit both, keep the better).
INDEX_FORMS = ("sweep", "linear", "synergy")
#: Channel sets a fusion job can pick between (``runners._CHANNEL_SETS``).
CHANNEL_SETS: dict[str, tuple[str, ...]] = {
    "ndvi + gvi": ("ndvi", "gvi"),
    "ndvi + veg + terrain": ("ndvi", "veg", "terrain"),
}
#: Objective metrics of a cross-sectional fusion run, first = default: ``r2``
#: for a linear association, ``spearman`` for a monotone one, the distance
#: correlations for general dependence.
CROSS_SECTIONAL_METRICS = (
    "r2",
    "spearman",
    "partial_distance_corr",
    "distance_corr",
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
#: ``geofuse.spatial_basis.DEFAULT_DF_CANDIDATES``.
SPATIAL_DF_CANDIDATES = (0, 1, 2, 3, 4, 6, 8, 11, 15, 20)
#: ``geofuse.spatial_basis.CRITERIA``.
SPATIAL_CRITERIA = ("aic", "bic")
#: ``geofuse.fusion.SPATIAL_PAIRED_SMOOTHS``.
SPATIAL_PAIRED_SMOOTHS = ("own", "cgi", "outcome", "max")
#: ``geofuse.pdcor.MAX_CACHE_N``: above it ``partial_distance_corr`` leaves its
#: shared-matrix fast path.
PDCOR_CACHE_N = 6_000
#: Peak bytes of one ``partial_distance_corr`` call per n²: 66 n × n float64
#: matrices (measured 2.8 GB at n = 2,305, 13 GB at 5,000, ~53 GB at 10,000).
PDCOR_BYTES_PER_N2 = 66 * 8
#: Objective metrics whose memory grows with n².
QUADRATIC_METRICS = frozenset({"partial_distance_corr"})
#: ``geofuse.fusion.CATEGORICAL_MISSING``.
CATEGORICAL_MISSING = ("drop", "indicator")
#: ``geofuse.fusion.SPATIAL_SPLIT_MODES``.
SPATIAL_SPLIT_MODES = ("stripe", "random")
#: ``geofuse.jobs.outer_splits.OUTER_COMPARATORS``.
OUTER_COMPARATORS = ("primary", "all")
#: ``geofuse.jobs.outer_splits.OUTER_REPORTS``.
OUTER_REPORTS = ("full", "decision")
#: ``geofuse.null_outcomes.NULL_KINDS``.
NULL_KINDS = ("spatial", "free", "both")

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
        "Score candidates are ranked on, by the association expected: r2 linear, "
        "spearman monotone, partial_distance_corr any (small samples only). "
        "Longitudinal runs: mixedlm_tstat.",
    ),
    "allow_large_objective": _bool(
        False, "Run partial_distance_corr even where it may not fit in memory."
    ),
    "test_size": Setting("float", 0.25, lo=0.1, hi=0.5, help="Held-out test fraction."),
    "split_seed": _int(42, 0, 2**31 - 1, "Seed of the held-out test split."),
    "outer_splits": _int(
        1, 1, 100, "Held-out splits, each tuned anew; seeds split_seed + k."
    ),
    "outer_comparators": _choice(
        "primary", OUTER_COMPARATORS, "Comparators scored on every held-out split."
    ),
    "outer_report": _choice(
        "full",
        OUTER_REPORTS,
        "What the later held-out splits report: every interval and p-value, or "
        "only the estimates the across-split decisions read.",
    ),
    "outer_null_runs": _int(
        0,
        0,
        10_000,
        "Null outcomes run through the whole held-out decision rule; 0 = none.",
    ),
    "outer_null_kind": _choice(
        "spatial",
        NULL_KINDS,
        "Null outcomes: spatially structured, freely permuted, or each.",
    ),
    "outer_null_group": Setting(
        "text",
        None,
        help="Column whose area means the spatial null permutes; unset = the "
        "smooth's spatial clusters.",
    ),
    "outer_null_stop_after": _int(
        None,
        1,
        10_000,
        "Sequential rule: decide after this many nulls; 0 = run all.",
        auto_zero=True,
    ),
    "outer_null_stop_low": _int(
        None, 0, 10_000, "Stop as not inflated at or below this many rejections."
    ),
    "outer_null_stop_high": _int(
        None, 1, 10_000, "Stop as inflated at or above this many rejections."
    ),
    "outer_null_workers": _int(
        None,
        1,
        256,
        "Null outcomes run at once; 0 = sized on cores and memory.",
        auto_zero=True,
    ),
    "n_bins": _int(5, 3, 10, "Quantile bins of the stratified split."),
    "residualize_method": _choice(
        "linear", ("linear", "spline"), "How covariates are partialled out."
    ),
    "residualize_spline_df": _int(
        4, 3, 20, "Spline df of each continuous covariate (spline residualizing)."
    ),
    "residualize_spline_min_unique": _int(
        7, 2, 1000, "Covariates with fewer distinct values enter linearly."
    ),
    "categorical_missing": _choice(
        "drop",
        CATEGORICAL_MISSING,
        "A missing categorical covariate: drop the row, or a level of its own.",
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
    "weight_column": Setting(
        "text",
        None,
        help="Target column of sampling weights; the reports add weighted "
        "estimates next to the unweighted ones.",
    ),
    "weight_search": _bool(
        False,
        "Also weight the search itself (a weighted likelihood); linear form only.",
    ),
    "fixed_iqr": Setting(
        "floatmap",
        None,
        help="Per-IQR increment by exposure (cgi, a channel, a comparator), "
        "e.g. ndvi=0.1; unset = each exposure's train+val IQR.",
    ),
    "exposure_response_spline_df": _int(
        4, 3, 20, "Spline df of the exposure-response curve and linearity test."
    ),
    "multi_objective_requested": _bool(False, "Several outcomes as one joint run."),
    # ── Discovery ───────────────────────────────────────────────
    "channel_set": _choice(
        "ndvi + gvi", tuple(CHANNEL_SETS), "Greenery channels the composite mixes."
    ),
    "index_form": _choice("sweep", INDEX_FORMS, "Functional form; sweep fits both."),
    "sweep_splits": _int(40, 5, 200, "Held-out splits each candidate is scored on."),
    "radius_prior": _choice("rungs", RADIUS_PRIORS, "Distance-of-influence prior."),
    "gvi_min_coverage": Setting(
        "float",
        0.95,
        lo=0.05,
        hi=1.0,
        help="Share of entities the smallest searched GVI rung must cover.",
    ),
    "run_standalones": _bool(False, "Also run each channel on its own."),
    "check_collinearity": _bool(True, "Report channels above the VIF threshold."),
    "drop_collinear_channels": _bool(
        False, "Leave channels above the VIF threshold out of the search."
    ),
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
    "synergy_warm_start": _bool(
        True, "Start a synergy posterior from a short linear fit."
    ),
    "outer_warm_start": _bool(
        True, "Start each later held-out split's posteriors from the previous fit."
    ),
    "outer_posterior_draws": _int(
        None, 100, 4000, "Posterior draws of the later held-out splits; none = all."
    ),
    "outer_posterior_warmup": _int(
        None, 100, 4000, "Their warm-up draws; none = as many as their draws."
    ),
    "outer_posterior_chains": _int(
        None, 1, 8, "Their posterior chains; none = as the first split."
    ),
    "negative_control_retune": _bool(
        False, "Also re-tune on each negative control and compare."
    ),
    "specificity_contrast": _bool(
        False, "Paired test: is the CGI more specific than each other exposure."
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
    "exposure_kernel": _choice(
        "none", EXPOSURE_KERNELS, "Exposure buffer; none = touched cells only."
    ),
    "exposure_radius_m": Setting(
        "float", 100.0, lo=1.0, hi=5000.0, help="Exposure buffer radius (m)."
    ),
    "exposure_plateau_m": Setting(
        "float", 0.0, lo=0.0, hi=5000.0, help="Linear buffer: full weight up to (m)."
    ),
    "exposure_sigma_m": Setting(
        "float", 35.0, lo=1.0, hi=5000.0, help="Gaussian buffer SD (m)."
    ),
    "whole_grid_scaling": _bool(True, "Scale the composite map to [0, 1]."),
    "area_balanced_split": _bool(True, "Balance polygon area across the split."),
    "spatial_split": _bool(True, "Hold out whole spatial blocks."),
    "spatial_block_size_m": Setting(
        "float", None, lo=0.0, auto_zero=True, help="Block edge (m); 0 = automatic."
    ),
    "spatial_split_mode": _choice(
        "stripe", SPATIAL_SPLIT_MODES, "Test blocks: evenly striped, or at random."
    ),
    "spatial_adjust_method": _choice(
        "none", SPATIAL_ADJUST_METHODS, "Spatial-confounding adjustment."
    ),
    "spatial_adjust_df": Setting(
        "intlist",
        list(SPATIAL_DF_CANDIDATES),
        lo=0,
        hi=50,
        help="Spatial df per cluster the criterion compares (0 = linear trends).",
    ),
    "spatial_adjust_max_df": _int(
        None,
        1,
        50,
        "Older form of spatial_adjust_df: compare every df 0..N; 0 = unset.",
        auto_zero=True,
    ),
    "spatial_adjust_criterion": _choice(
        "aic", SPATIAL_CRITERIA, "Criterion that picks the spatial df."
    ),
    "spatial_adjust_eps_m": Setting(
        "float", None, lo=0.0, auto_zero=True, help="Cluster gap (m); 0 = automatic."
    ),
    "spatial_adjust_neighbours": _int(
        10, 1, 100, "Neighbours that link points into one cluster (gap automatic)."
    ),
    "spatial_adjust_knots_min": _int(35, 3, 2000, "Fewest knots per cluster."),
    "spatial_adjust_knots_max": _int(150, 3, 2000, "Most knots per cluster."),
    "spatial_paired_smooth": _choice(
        "own",
        SPATIAL_PAIRED_SMOOTHS,
        "Spatial+ paired tests: each exposure's own smooth, the CGI's, the "
        "outcome's, or the larger df.",
    ),
    "spatial_adjust_scale_m": Setting(
        "intlist",
        None,
        lo=100,
        hi=1_000_000,
        ordered=True,
        help="Smooth by pattern scale (m) instead of df; the first is primary, "
        "the others are reported for the slopes. Unset = df.",
    ),
    "spatial_adjust_min_component": _int(
        None,
        1,
        10_000,
        "Smaller clusters join their nearest larger one; 0 = 30 with a scale, "
        "none with df.",
        auto_zero=True,
    ),
    "spatial_tune_scale_m": _int(
        None,
        100,
        1_000_000,
        "Tune within areas of this size (m): the smooth at this scale joins "
        "the search's covariates. Unset = tune on all variation.",
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


def objective_size_check(
    metric: str, n: int, memory_bytes: int
) -> tuple[str, str] | None:
    """Whether a run scoring ``metric`` on ``n`` rows can finish here.

    ``None`` when the metric's memory does not grow with n² or ``n`` is small;
    otherwise ``("warn", message)`` above :data:`PDCOR_CACHE_N`, and
    ``("refuse", message)`` when one call's estimated peak exceeds
    ``memory_bytes`` (the machine's physical memory; ``0`` = unknown). A run
    scores its objective thousands of times, so a call that pages never ends.
    """
    if metric not in QUADRATIC_METRICS or n <= PDCOR_CACHE_N:
        return None
    need = PDCOR_BYTES_PER_N2 * n * n
    text = (
        f"{metric} at n = {n:,} needs ~{need / 1e9:.0f} GB per call and is "
        "scored thousands of times in a run"
    )
    alternatives = "r2 (linear) or spearman (monotone) run at any n"
    if memory_bytes and need > memory_bytes:
        return (
            "refuse",
            f"{text}; this machine has {memory_bytes / 1e9:.0f} GB, so it would "
            f"page without finishing. Use {alternatives}, or set "
            "allow_large_objective to run it anyway.",
        )
    return (
        "warn",
        f"{text}, without the fast path above n = {PDCOR_CACHE_N:,}; "
        f"{alternatives}.",
    )


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
    if engine == "fusion":
        _check_spatial(out)
        _check_outer_null(out)
        _check_weights(out)
    return out


def _check_weights(out: Mapping[str, Any]) -> None:
    """Refuse a weighted search without weights or with the synergy form."""
    if not out.get("weight_search"):
        return
    if "weight_column" in out and not out["weight_column"]:
        raise SettingError("'weight_search' needs a 'weight_column'.")
    if out.get("index_form", "linear") != "linear":
        raise SettingError(
            "'weight_search' fits the linear form only; set 'index_form' to "
            "'linear'."
        )


def _check_outer_null(out: Mapping[str, Any]) -> None:
    """Refuse null outcomes without repeated splits, and a sequential rule
    that is incomplete or cannot decide."""
    if out.get("outer_null_runs") and out.get("outer_splits", 2) < 2:
        raise SettingError(
            "'outer_null_runs' calibrates the decision across held-out splits; "
            "set 'outer_splits' to 2 or more."
        )
    after = out.get("outer_null_stop_after")
    low, high = out.get("outer_null_stop_low"), out.get("outer_null_stop_high")
    if (low is not None or high is not None) and not after:
        raise SettingError(
            "'outer_null_stop_low' / '_high' need 'outer_null_stop_after'."
        )
    if after and low is None and high is None:
        raise SettingError(
            "'outer_null_stop_after' needs 'outer_null_stop_low' or '_high'."
        )
    if low is not None and high is not None and low >= high:
        raise SettingError("'outer_null_stop_low' must be below '_high'.")
    if after and high is not None and high > after:
        raise SettingError("'outer_null_stop_high' must not exceed '_after'.")


def _check_spatial(out: Mapping[str, Any]) -> None:
    """Refuse a df list given twice, knots that run backwards, and a df the
    knots cannot hold."""
    dfs = spatial_df_candidates(out)
    lo, hi = "spatial_adjust_knots_min", "spatial_adjust_knots_max"
    if lo in out and hi in out and out[lo] > out[hi]:
        raise SettingError(f"'{lo}' must not exceed '{hi}'.")
    if hi in out and max(dfs) > out[hi]:
        raise SettingError(
            f"The largest spatial df ({max(dfs)}) exceeds '{hi}' ({out[hi]}); a "
            "cluster has at most that many radial functions."
        )


def spatial_df_candidates(values: Mapping[str, Any]) -> list[int]:
    """The df list a fusion job compares: ``spatial_adjust_max_df`` N as
    ``0 … N`` when given (the older form), else ``spatial_adjust_df``.

    Raises :class:`SettingError` when both are set and the list is not its
    default (a merged settings table cannot tell a default list from a given
    one).
    """
    legacy = values.get("spatial_adjust_max_df")
    listed = values.get("spatial_adjust_df")
    if legacy is None:
        return list(listed if listed is not None else SPATIAL_DF_CANDIDATES)
    if listed is not None and list(listed) != list(SPATIAL_DF_CANDIDATES):
        raise SettingError(
            "Give 'spatial_adjust_df' or the older 'spatial_adjust_max_df', not both."
        )
    return list(range(0, int(legacy) + 1))
