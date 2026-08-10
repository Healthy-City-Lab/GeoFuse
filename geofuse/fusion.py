"""
Fusion Module: Optimized metric fusion for geospatial composite indices.

This module implements the optimization logic from CGI.ipynb for tuning
weighted combinations of NDVI and GVI metrics against target outcomes.
"""

import gc
import hashlib
import logging
import os
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import optuna
import pandas as pd
import rasterio
from rasterio.transform import from_origin, xy
from shapely.geometry import box
from sklearn.model_selection import train_test_split

from . import (
    JobCancelled,
    bayesian_index,
    binary_longitudinal,
    cgi_formulas,
    exposure_response,
)
from . import fusion_helpers as _helpers
from . import (
    longitudinal,
    metric_intake,
    metric_sampling,
    mixed_effects_scoring,
    objective_scoring,
    parallel,
    pdcor,
    preaggregation,
    spatial_basis,
)
from .cgi_formulas import WEIGHTED_AVERAGE, compute_cgi
from .crs_utils import (
    assign_spatial_blocks,
    build_internal_overviews,
    crs_uses_metre_axes,
    default_geotiff_creation_options,
    estimate_metre_projected_crs_for_gdf,
    metres_per_degree_at_lat,
    normalize_geographic_gdf_to_wgs84,
    reproject_geodataframe_to_wgs84,
    select_grid_crs_with_warning,
)
from .logger import attach_external_logger, get_logger
from .raster_sampling import LazyRasterArray
from .vector_io import (
    match_column_alias,
    read_vector_subset,
    target_path_is_raster,
)

logger = logging.getLogger(__name__)
_log = get_logger("FUSION")

# Files whose buffered-extent shortfall has already been reported this process,
# keyed by (abspath, rounded bounds). A cached NDVI file's sub-pixel shortfall
# is permanent, so without this the multi-outcome loop (a fresh engine per
# outcome) and every repeated run would re-emit the same warning.
_WARNED_METRIC_BOUNDS: set[tuple] = set()


def _finite_or_none(value) -> float | None:
    """``float(value)`` when it is finite, else ``None``.

    Reporting payloads travel to JSON and to the results UI, where a ``NaN``
    renders as the literal text "nan"; ``None`` is the only value both layers
    read as "not available".
    """
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return None
    return fv if np.isfinite(fv) else None


def _log_parallel_efficiency(
    log,
    phase: str,
    *,
    wall_s: float,
    busy_s: float,
    workers: int,
) -> None:
    """Report a phase's achieved concurrency against what the machine offers.

    ``busy_s`` is summed worker time, so ``busy_s / wall_s`` is the average
    number of workers actually running. Comparing that to ``workers`` says
    whether the pool is saturated, and comparing ``workers`` to the core count
    says whether the pool itself is the ceiling — the two questions that decide
    whether raising the worker count would buy anything on *this* host.
    """
    if wall_s <= 0 or busy_s <= 0:
        return
    cores = os.cpu_count() or 0
    achieved = busy_s / wall_s
    saturation = achieved / workers if workers else 0.0
    parts = [
        f"{phase}: {achieved:.1f} of {workers} worker(s) busy on average "
        f"({saturation * 100:.0f}% pool saturation)"
    ]
    if cores:
        parts.append(f"{achieved / cores * 100:.0f}% of {cores} logical core(s)")
        if saturation >= 0.85 and workers < cores:
            parts.append(
                f"pool-bound — the cap, not the work, is the limit "
                f"({cores - workers} core(s) idle)"
            )
        elif saturation < 0.6:
            parts.append("work-bound — raising the cap would not help")
    log("INFO", "  " + " · ".join(parts) + ".")


class MetricFusionEngine:
    """
    Engine for fusing vegetation, terrain, and NDVI metrics using optimization.

    Based on the methodology in CGI.ipynb, this class:
    1. Loads vegetation (GVI veg/terrain) and NDVI metrics
    2. Aligns them spatially with a target outcome (GeoJSON points or GeoTIFF)
    3. Splits data: holdout test set + k-fold CV on training data
    4. Optimizes 9 parameters matching CGI.ipynb:
       - Weights: veg_weight, terrain_weight, ndvi_weight (0-100, sum=100)
       - Radii: veg_radius, terrain_radius, ndvi_radius (GVI / NDVI ladder grids)
       - Streetview agg: streetview_stat, streetview_percentile (shared for veg+terrain)
       - NDVI agg: ndvi_stat, ndvi_percentile (separate for NDVI)
    5. Validates performance using cross-validation and held-out test set
    """

    def __init__(
        self,
        target_file: str,
        target_feature: str | None = None,
        target_band: int = 1,
        target_layer: str | int | None = None,
        buffer_meters: float = 1500.0,
        gvi_buffer_min_m: float | None = None,
        gvi_buffer_max_m: float | None = None,
        gvi_buffer_step_m: float | None = None,
        ndvi_buffer_min_m: float | None = None,
        ndvi_buffer_max_m: float | None = None,
        ndvi_buffer_step_m: float | None = None,
        n_bins: int = 5,
        cache_dir: str = "output_results/fusion_cache",
        cgi_formula: str = WEIGHTED_AVERAGE,
        covariate_columns: list[str] | None = None,
        covariate_types: dict[str, str] | None = None,
        longitudinal_spec: longitudinal.LongitudinalSpec | None = None,
        cgi_grid_spacing_m: float | None = None,
        whole_grid_scaling: bool = False,
        area_balanced_split: bool = False,
        normalize_channels: bool = False,
        spatial_adjust_method: str = "none",
        spatial_adjust_max_df: int = 10,
        spatial_adjust_eps_m: float | None = None,
        residualize_method: str = "linear",
        search_scoring_method: str = "mom_em3",
    ):
        """
        Initialize the fusion engine.

        Args:
            target_file: Path to vector (GeoJSON, GPKG, Shapefile, zip) or GeoTIFF target
            target_feature: For vector targets, the column name to optimize towards
            target_band: For GeoTIFF, the band number to optimize towards
            target_layer: Fiona layer name or index for multi-layer files (e.g. GPKG)
            buffer_meters: Maximum buffer (m) around target for extent padding and downloads;
                typically max(GVI max, NDVI max). If modality maxima are omitted, they default here.
            gvi_buffer_min_m / gvi_buffer_max_m / gvi_buffer_step_m: GVI (veg/terrain) radius search grid (m).
            ndvi_buffer_min_m / ndvi_buffer_max_m / ndvi_buffer_step_m: NDVI radius search grid (m).
            n_bins: Number of bins for stratified splitting
            cache_dir: Directory to cache downloaded metrics
            cgi_formula: Which composite formula to optimize against. One of
                :data:`geofuse.cgi_formulas.available_formulas` — defaults to
                ``"weighted_average"`` so existing studies replay unchanged.
            covariate_columns: Numeric column names on the (vector) target to
                carry through ``prepare_fusion_data`` and use as additional
                predictors in the objective. The score becomes the greenery
                term's *partial* contribution (partial correlation /
                incremental R² / full-model RMSE) — see
                :mod:`geofuse.objective_scoring` for the per-metric semantics.
                ``mutual_info`` ignores covariates by design. Not supported
                for raster targets (no attribute table); a non-empty list
                with a raster target raises ``ValueError`` from
                :meth:`prepare_fusion_data`. Defaults to ``None`` → score
                reduces exactly to the legacy ``_calculate_metric``.
            longitudinal_spec: When set, switches the engine into
                mixed-effects / longitudinal mode. The spec controls intake
                shape (long-format target or N per-wave files joined on
                ``entity_id``) and carries the per-wave greenery file
                assignment. :meth:`prepare_fusion_data` then routes through
                :func:`geofuse.longitudinal.build_long_format` and emits a
                fusion DataFrame keyed by ``(entity_id, wave)`` plus the
                derived continuous ``years_since_baseline`` predictor.
                :meth:`split_data` keeps every row of an entity together
                (mirrors polygon mode) so the mixed-effects scorer sees a
                clean within-entity panel. Defaults to ``None`` → engine
                stays in cross-sectional mode.
        """
        # Route Optuna's chatter ("Trial X finished with value Y …") into
        # the per-job log instead of the Streamlit host terminal. Fusion
        # auto-downloads NDVI for the buffered extent, so attach ``ee`` too —
        # those records would otherwise propagate to root from this process.
        import logging as _logging

        attach_external_logger("optuna", _logging.INFO)
        attach_external_logger("ee", _logging.INFO)
        # This module's own stdlib ``logger`` (``geofuse.fusion``) carries
        # engine diagnostics — bounds checks, metric load notes, cache messages.
        # Attaching it routes those into the per-job log (and stops them
        # propagating to the host terminal) alongside the engine's ``_log``
        # output, instead of leaking to the backend console.
        attach_external_logger("geofuse.fusion", _logging.INFO)

        self.target_file = target_file
        self.target_feature = target_feature
        self.target_band = target_band
        self.target_layer = target_layer
        self.buffer_meters = float(buffer_meters)
        self.gvi_buffer_max_m = (
            float(gvi_buffer_max_m)
            if gvi_buffer_max_m is not None
            else self.buffer_meters
        )
        self.gvi_buffer_min_m = (
            float(gvi_buffer_min_m)
            if gvi_buffer_min_m is not None
            else min(100.0, self.gvi_buffer_max_m)
        )
        self.gvi_buffer_step_m = (
            float(gvi_buffer_step_m) if gvi_buffer_step_m is not None else 50.0
        )
        self.ndvi_buffer_max_m = (
            float(ndvi_buffer_max_m)
            if ndvi_buffer_max_m is not None
            else self.buffer_meters
        )
        self.ndvi_buffer_min_m = (
            float(ndvi_buffer_min_m)
            if ndvi_buffer_min_m is not None
            else min(100.0, self.ndvi_buffer_max_m)
        )
        self.ndvi_buffer_step_m = (
            float(ndvi_buffer_step_m) if ndvi_buffer_step_m is not None else 50.0
        )
        if self.gvi_buffer_min_m > self.gvi_buffer_max_m:
            self.gvi_buffer_min_m, self.gvi_buffer_max_m = (
                self.gvi_buffer_max_m,
                self.gvi_buffer_min_m,
            )
        if self.ndvi_buffer_min_m > self.ndvi_buffer_max_m:
            self.ndvi_buffer_min_m, self.ndvi_buffer_max_m = (
                self.ndvi_buffer_max_m,
                self.ndvi_buffer_min_m,
            )

        self.n_bins = n_bins
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Validate up-front so a typo in the runner / UI doesn't blow up mid-
        # optimization. ``get_formula`` raises ValueError with the supported list.
        self.cgi_formula = cgi_formula
        cgi_formulas.get_formula(cgi_formula)

        # Greenery channel for the active search. ``cgi`` runs the combined
        # formula (the standard fusion behaviour); ``veg`` / ``terrain`` /
        # ``ndvi`` run a standalone single-metric search that uses that
        # channel's normalised value directly as the greenery value and
        # searches only its radius + aggregation. Set per call by the runner
        # before each stability search; ``evaluate_on_test`` reads it so the
        # held-out test score stays aligned with what the search scored.
        self._active_greenery_channel: str = "cgi"

        # Covariate columns are stashed on self; per-row values are carried into
        # the prepared fusion DataFrame, then split + scored alongside target
        # and CGI. Empty list = legacy single-variable scoring (no change).
        # De-duplicated to avoid a singular design matrix in the partial-
        # correlation / incremental-R² OLS.
        cov_in = list(covariate_columns) if covariate_columns else []
        self.covariate_columns: list[str] = list(dict.fromkeys(cov_in))
        # Per-covariate kind: ``"numeric"`` (default) or ``"categorical"``.
        # Categorical columns are one-hot expanded (drop-first) onto the target
        # frame at prepare time and their dummy columns replace the original name
        # in ``covariate_columns``. ``_covariate_dummy_map`` records original →
        # dummy names for reporting; ``_covariate_columns_user`` keeps the
        # user-facing names. Defaults make every covariate numeric (legacy).
        self.covariate_types: dict[str, str] = {
            k: str(v).lower() for k, v in (covariate_types or {}).items()
        }
        self._covariate_columns_user: list[str] = list(self.covariate_columns)
        self._covariate_dummy_map: dict[str, list[str]] = {}
        # Expanded user covariates, and the wave / area indicators that follow
        # them. Reporting names only the former; both are fitted.
        self._user_covariate_columns: list[str] = list(self.covariate_columns)
        self._period_control_columns: list[str] = []
        self._wave_control_columns: list[str] = []
        self._time_fixed_effect_dropped: bool = False
        self._covariates_expanded: bool = False

        # Spatial-confounding adjustment. ``none`` keeps the plain covariate-
        # residualized objective. ``ks_aic`` / ``spatial_plus`` fold a low-rank
        # coordinate smooth (built per spatial cluster, df chosen by AIC) into
        # the residualization so the score reflects greenery↔outcome co-variation
        # beyond an unmeasured smooth spatial confounder. Entity coordinates ride
        # on the prepared frame as ``_cx`` / ``_cy``; the per-split candidate
        # basis is cached on ``_spatial_basis_cache`` keyed by split id.
        method = str(spatial_adjust_method or "none").lower()
        if method not in objective_scoring.SPATIAL_METHODS:
            raise ValueError(
                f"spatial_adjust_method must be one of "
                f"{sorted(objective_scoring.SPATIAL_METHODS)}; got {method!r}."
            )
        self.spatial_adjust_method: str = method
        self.spatial_adjust_max_df: int = max(1, int(spatial_adjust_max_df))
        self.spatial_adjust_eps_m: float | None = (
            float(spatial_adjust_eps_m)
            if spatial_adjust_eps_m is not None and float(spatial_adjust_eps_m) > 0
            else None
        )
        self._spatial_basis_cache: dict = {}
        self._spatial_adjust_summary: dict | None = None

        # Covariate residualization basis for the residualizing metrics
        # (distance_corr / spearman / r2 / nrmse). ``spline`` removes nonlinear
        # covariate effects; ``partial_distance_corr`` / ``mutual_info`` ignore it.
        res_method = str(residualize_method or "linear").lower()
        if res_method not in objective_scoring.RESIDUALIZE_METHODS:
            raise ValueError(
                f"residualize_method must be one of "
                f"{sorted(objective_scoring.RESIDUALIZE_METHODS)}; got {res_method!r}."
            )
        self.residualize_method: str = res_method

        # Longitudinal search scorer. ``mom_em3`` (default) / ``mom_em1`` /
        # ``mom`` / ``reml`` estimate the per-fold variance components for the
        # fast fixed-V GLS trial scorer (see ``mixed_effects_scoring``);
        # ``exact`` bypasses it and fits a full MixedLM per trial (the original
        # behaviour). Reports always use the exact MixedLM refit regardless.
        self.search_scoring_method: str = str(search_scoring_method)

        # Longitudinal / mixed-effects spec. ``None`` keeps the engine in
        # cross-sectional mode (no behaviour change). When set, every code
        # path that today threads a single per-entity row through Optuna
        # forks to use (entity_id, wave) rows + a continuous
        # ``years_since_baseline`` time predictor instead.
        self.longitudinal_spec: longitudinal.LongitudinalSpec | None = longitudinal_spec
        if self.longitudinal_spec is not None:
            spec_errs = longitudinal.validate_spec(self.longitudinal_spec)
            if spec_errs:
                raise ValueError("Invalid longitudinal_spec: " + "; ".join(spec_errs))

        self._ndvi_export_resolution_m = 10.0
        self._gvi_grid_spacing_m = 75.0

        # Polygon-target CGI grid: pixel centroids generated inside the union
        # of the target polygons, scored as per-pixel CGI and averaged to
        # per-polygon. ``None`` selects the default 25 m (high fidelity).
        self.cgi_grid_spacing_m: float = (
            float(cgi_grid_spacing_m) if cgi_grid_spacing_m is not None else 25.0
        )
        self.whole_grid_scaling: bool = bool(whole_grid_scaling)
        # When True, each channel is min-max scaled to [0, 1] (per-channel
        # bounds fixed at prepare time from the predictor distribution only —
        # no outcome leakage) before the composite is formed, so weights are
        # interpretable and the synergy powers see their assumed [0, 1] domain.
        # Default False keeps legacy raw-channel composites bit-for-bit.
        self.normalize_channels: bool = bool(normalize_channels)
        # Per-channel (lo, hi) min-max bounds, filled by prepare_fusion_data.
        self._channel_minmax: dict[str, tuple[float, float]] | None = None
        # When True, the area-balanced train-val split tries to keep the total
        # area balanced across train / val / test sets.
        self.area_balanced_split: bool = bool(area_balanced_split)
        # Per-polygon-id area (km²) populated by ``_prepare_polygon_fusion``
        # so the area-balanced split has stable polygon → area lookups even
        # after the NaN-outcome filter renumbers polygons.
        self._polygon_id_to_area_km2: dict[int, float] | None = None
        self._polygon_grid_crs: Any = None
        self._polygon_grid_transform: Any = None
        self._polygon_grid_shape: tuple[int, int] | None = None
        # Per-pixel CGI grid metadata for point / line targets (the composite
        # raster reuses these so it matches the scored CGI field).
        self._entity_grid_crs: Any = None
        self._entity_grid_transform: Any = None
        self._entity_grid_shape: tuple[int, int] | None = None

        # Data containers
        self.target_gdf = None
        self.target_polygons_gdf = (
            None  # Set when target is polygon-shaped (areal mode)
        )
        self.is_polygon_target = False
        self.target_raster = None
        self.buffered_extent = None
        self.veg_data = None  # Vegetation component (GVI vegetation)
        self.terrain_data = None  # Terrain component (GVI terrain)
        self.ndvi_data = None  # NDVI satellite data
        self.train_val_data = None  # Train+val pool the bootstrap resamples
        self.test_data = None  # Held-out test set
        self.cv_folds = None  # In-bag/OOB fold spliced in per bootstrap resample
        self.study = None  # Per-resample Optuna study (set inside the bootstrap)
        self.best_params = None
        self.scaler = None

        # Per-trial objective durations, appended by every search thread and
        # summed into the phase's achieved-concurrency figure.
        self._objective_secs: list[float] = []

        # Donut / ring cache: per-point annulus samples keyed by fold subset + indices
        self._ring_raster_cache: dict = {}
        self._ring_vector_cache: dict = {}
        self._max_points_ring_cache = 8000
        # Payload ceiling for one ring-cache entry. The point cap above bounds
        # rows only; a fine raster under a wide radius stores tens of thousands
        # of values per row, so the estimated byte size gates the build too.
        self._max_ring_cache_bytes = 1_000_000_000

        # Reporting-phase caches (all invalidated by ``split_data``): the
        # train+val+test union frame is rebuilt by several report helpers,
        # ``apply_fusion`` is called repeatedly with the same winning params,
        # and ``evaluate_on_test`` is re-invoked with identical params by the
        # runner headline, the effects block, and the subset-score table.
        self._full_data_cache: pd.DataFrame | None = None
        self._apply_fusion_cache: tuple[tuple, pd.DataFrame] | None = None
        self._evaluate_test_cache: dict[tuple, dict] = {}

        # Scoring-side caches, cleared with the spatial-basis cache (their
        # contents are constant per scored subset, not per trial): the fixed
        # U-centered sides of the partial-distance-correlation estimator and
        # the spline-expanded covariate bases.
        self._pdcor_cache: OrderedDict = OrderedDict()
        self._spline_basis_cache: dict = {}
        # Trial-invariant per-fold data (collapse codes, collapsed target /
        # covariates / coords, points slices); cleared with the ring caches.
        self._fold_static_cache: dict = {}
        # Monotonic counter bumped whenever the live fold row sets are replaced
        # (a fresh split, or each bootstrap resample). Fold-static entries key
        # on it so an entry can never be served for a different row set.
        self._split_generation: int = 0

        # Trial threads for the stability search. Only the longitudinal
        # fast-GLS path uses them: the other paths keep LRU scoring caches that
        # are not thread-safe (see ``_search_n_jobs``).
        self._search_workers: int = parallel.worker_count()

        # Spatial pre-aggregation (mandatory; built by precompute_aggregations()).
        # Backed by an on-disk SQLite cache (geofuse/preaggregation.py) so the
        # per-(entity, radius) stat table survives cancels/crashes and is reused
        # across runs.
        self._preaggregation_done: bool = False
        self._preaggr_cache: preaggregation.GreeneryCache | None = None
        # RAM-resident mirror of the complete cache (kept for compatibility;
        # the GreeneryCache is itself resident, so this stays ``None``).
        self._preaggr_mem = None
        # Maps frame-local ``_preaggr_id`` → stable global pixel id, so scoring
        # rows resolve to the greenery cache's keys at lookup time.
        self._preaggr_pid_translate: np.ndarray | None = None
        # Unique-pixel source for the pre-aggregation cache. Point/line targets
        # duplicate each catchment pixel per overlapping entity for scoring, but
        # a pixel's per-radius stats are identical across those duplicates, so
        # the cache is built over these unique pixels (keyed by ``_preaggr_id``)
        # and duplicated scoring rows translate to a pixel id at lookup time.
        # ``None`` for polygon / raster targets, where one row == one entity.
        self._preaggr_entity_gdf: gpd.GeoDataFrame | None = None
        # Spatial-block labels per split group, set by ``split_data`` when
        # spatial blocking is enabled and reused by the bootstrap resampler.
        self._spatial_block_by_group: dict | None = None
        self._spatial_block_group_col: str | None = None
        self._cancel_callback: Callable[[], bool] | None = None

        # Channels turned off by the optional collinearity check
        # (``check_channel_collinearity``). Disabled channels are pinned to
        # weight 0 in every trial and skipped in the per-channel
        # aggregation path. Default: all three channels active.
        self._disabled_channels: set[str] = set()
        # Collinearity report from the most recent check (or ``None`` if
        # the check wasn't requested). Stored so the runner can hand it to
        # the results UI without re-running the analysis.
        self._collinearity_report: dict | None = None
        # Outcome / covariate dispersion sanity report from the most recent
        # pre-flight check (or ``None`` if it hasn't run yet).
        self._dispersion_report: dict | None = None

        # Vector vs raster (``is_points`` kept for backward compatibility = vector target)
        self.is_raster = target_path_is_raster(target_file)
        self.is_points = not self.is_raster

        if not (self.is_points or self.is_raster):
            raise ValueError("Target file must be a supported vector format or GeoTIFF")

        # Wide-mode longitudinal frames are populated by the runner before
        # ``prepare_fusion_data`` via :meth:`set_longitudinal_target_frames`.
        # Long-mode skips this — the engine reads the single target file via
        # ``load_target`` exactly as in cross-sectional mode and converts it
        # to long format inside ``prepare_fusion_data``.
        self._longitudinal_wave_frames: list[tuple[str, gpd.GeoDataFrame]] | None = None
        # Per-temporal-key metric sources for the mixed-effects mode. Populated
        # by ``set_longitudinal_metric_sources`` (paths, loaded on demand)
        # before ``precompute_aggregations``. Cross-sectional runs leave this
        # ``None``.
        self._metric_sources: metric_intake.LongitudinalMetricSources | None = None

        if self.is_longitudinal and self.is_raster:
            raise ValueError(
                "longitudinal_spec is only supported for vector targets; "
                "raster targets have no entity-id attribute column."
            )

    # ────────────────────────────────────────────────────────────
    # Longitudinal mode — entry points + helpers
    # ────────────────────────────────────────────────────────────

    @property
    def is_longitudinal(self) -> bool:
        """True when a :class:`LongitudinalSpec` was provided at construction."""
        return self.longitudinal_spec is not None

    def _temporal_key_noun(self) -> str:
        """What one temporal key represents, for logs and messages.

        The spec's ``derive_wave_from_date`` decides this: with it on, a row's
        key is the calendar year it was measured in (so the same year can be a
        baseline for one participant and a follow-up for another); with it off,
        the key is the wave label the file was supplied under.
        """
        spec = self.longitudinal_spec
        if spec is not None and spec.derive_wave_from_date:
            return "year"
        return "wave"

    def _validate_temporal_coverage(self, per_key: Mapping[str, Any]) -> None:
        """Every temporal key in the spec must have an entry."""
        spec = self.longitudinal_spec
        assert spec is not None
        missing = [w for w in spec.wave_labels if w not in per_key]
        if missing:
            raise ValueError(
                f"metric sources are missing entries for waves: {missing}."
            )

    def set_longitudinal_metric_sources(
        self,
        paths: Mapping[str, Mapping[str, str]],
        loader: Callable[[str, str], Any],
    ) -> None:
        """Register per-temporal-key metric *file paths*, loaded on demand.

        ``paths`` maps ``channel -> temporal key -> file path``, where the
        temporal key is the measurement year when the spec derives waves from
        dates, and the wave label otherwise. Nothing is read here: the engine
        loads one key's files when it needs them and releases them before
        moving to the next, so a study with many waves never holds more than
        one key's metrics in memory.
        """
        if not self.is_longitudinal:
            raise RuntimeError(
                "set_longitudinal_metric_sources() requires longitudinal_spec "
                "to be set."
            )
        for channel in longitudinal.GREENERY_CHANNELS:
            if channel not in paths:
                raise ValueError(f"metric source paths are missing {channel!r}.")
            self._validate_temporal_coverage(paths[channel])
        self._metric_sources = metric_intake.LongitudinalMetricSources(paths, loader)

    def set_longitudinal_wave_frames(
        self, frames: list[tuple[str, gpd.GeoDataFrame]]
    ) -> None:
        """Inject pre-loaded per-wave target frames for wide-mode runs.

        The runner loads each per-wave target file separately (under each
        wave's date column and shared ``entity_id_col`` from the spec) and
        hands the list of ``(wave_label, GeoDataFrame)`` pairs to the engine
        via this method before :meth:`prepare_fusion_data`. Long-mode runs
        do not call this — the standard :meth:`load_target` flow reads the
        single long-format file into ``self.target_gdf``.
        """
        if not self.is_longitudinal:
            raise RuntimeError(
                "set_longitudinal_wave_frames() requires longitudinal_spec to be set."
            )
        if self.longitudinal_spec.intake_mode != "wide":
            raise RuntimeError(
                "set_longitudinal_wave_frames() is only valid for intake_mode='wide'; "
                f"current spec uses intake_mode={self.longitudinal_spec.intake_mode!r}."
            )
        self._longitudinal_wave_frames = list(frames)

    _LONGITUDINAL_EXTRA_COLS: tuple[str, ...] = (
        "entity_id",
        "wave",
        "years_since_baseline",
    )

    def _longitudinal_intake_columns(self) -> list[str] | None:
        """Columns worth reading from the target file, or ``None`` to read all.

        Longitudinal intake replaces ``target_gdf`` with the projected long
        frame, so any column outside this set is read and then discarded.
        Cross-sectional mode returns ``None``: its target frame is consumed
        directly and carries columns this engine does not enumerate.
        """
        if not self.is_longitudinal:
            return None
        return longitudinal.target_intake_columns(
            self.longitudinal_spec,
            self.target_feature,
            self.covariate_columns or (),
        )

    def _longitudinal_extra_cols(self) -> tuple[str, ...]:
        """Extra DataFrame columns carried through every prep path in long mode."""
        return self._LONGITUDINAL_EXTRA_COLS if self.is_longitudinal else ()

    def _materialize_longitudinal_target(self) -> None:
        """Replace ``self.target_gdf`` with the normalised long-format frame.

        Run as the first step of :meth:`prepare_fusion_data` whenever
        ``longitudinal_spec`` is set. After this returns, every row of
        ``self.target_gdf`` represents one ``(entity_id, wave)`` observation
        carrying ``years_since_baseline`` and the original geometry — so the
        existing point / polygon prep paths sample metrics at the right
        location for each observation without further changes.
        """
        spec = self.longitudinal_spec
        assert spec is not None  # guaranteed by caller

        outcome_col = self.target_feature
        if not outcome_col:
            raise ValueError(
                "longitudinal mode requires target_feature (outcome column) to be set."
            )

        if spec.intake_mode == "long":
            if self.target_gdf is None:
                raise RuntimeError(
                    "longitudinal long-mode requires load_target() to have run first."
                )
            target_input: gpd.GeoDataFrame | list[tuple[str, gpd.GeoDataFrame]] = (
                self.target_gdf
            )
        else:
            if not self._longitudinal_wave_frames:
                raise RuntimeError(
                    "longitudinal wide-mode requires set_longitudinal_wave_frames() "
                    "to have been called before prepare_fusion_data()."
                )
            target_input = self._longitudinal_wave_frames

        long_gdf = longitudinal.build_long_format(
            spec,
            target_input,
            outcome_col=outcome_col,
            covariate_cols=self.covariate_columns,
        )
        # The per-file frames have been concatenated into ``long_gdf``; drop
        # the engine's reference so their memory is reclaimed rather than held
        # alongside the combined frame for the rest of the run.
        target_input = None
        self._longitudinal_wave_frames = None

        dropped = long_gdf.attrs.get("dropped_rows", {})
        if dropped:
            _log(
                "INFO",
                "longitudinal intake — dropped rows: "
                + ", ".join(f"{k}={v}" for k, v in dropped.items()),
            )

        # Re-detect polygon mode based on the long-format frame's geometry
        # (per-entity geometries may differ from what load_target inferred
        # when wide-mode supplies one frame per wave).
        first_geom_type = long_gdf.geometry.iloc[0].geom_type
        self.is_polygon_target = first_geom_type in ("Polygon", "MultiPolygon")
        if self.is_polygon_target:
            self.target_polygons_gdf = long_gdf.copy()
        self.target_gdf = long_gdf
        _log(
            "INFO",
            f"longitudinal intake: {len(long_gdf)} rows across "
            f"{long_gdf['entity_id'].nunique()} entities × "
            f"{long_gdf['wave'].nunique()} waves "
            f"(geometry: {first_geom_type}).",
        )

    def load_target(self) -> gpd.GeoDataFrame:
        """Load and prepare target data, return buffered extent."""
        logger.info(f"Loading target file: {self.target_file}")

        if self.is_points:
            layer = None
            if self.target_layer is not None and Path(
                self.target_file
            ).suffix.lower() in (
                ".gpkg",
                ".zip",
            ):
                layer = self.target_layer
            keep_cols = self._longitudinal_intake_columns()
            if keep_cols is None:
                raw = gpd.read_file(
                    self.target_file, **({} if layer is None else {"layer": layer})
                )
            else:
                # Longitudinal intake projects the frame down to these columns
                # anyway; skipping the rest keeps a wide cohort file's unused
                # survey columns out of memory instead of loading then dropping.
                raw = read_vector_subset(self.target_file, keep_cols, layer=layer)
            self.target_gdf = reproject_geodataframe_to_wgs84(raw)

            # Validate feature exists
            if (
                self.target_feature
                and self.target_feature not in self.target_gdf.columns
            ):
                raise ValueError(
                    f"Target feature '{self.target_feature}' not found in columns: {list(self.target_gdf.columns)}"
                )

            # Detect polygon target — switches fusion into areal-aggregation mode
            # (per-polygon mean of pixel/point CGIs vs polygon outcome).
            first_geom_type = self.target_gdf.geometry.iloc[0].geom_type
            self.is_polygon_target = first_geom_type in ("Polygon", "MultiPolygon")
            if self.is_polygon_target:
                self.target_polygons_gdf = self.target_gdf.copy()
                _log(
                    "INFO",
                    f"Polygon target detected ({len(self.target_polygons_gdf)} "
                    "features). Fusion will aggregate per-polygon mean CGI vs outcome.",
                )

            # Create buffered extent for metric download (metre-accurate buffer in local UTM)
            gdf_wgs84 = self.target_gdf
            utm_crs = gdf_wgs84.estimate_utm_crs()
            gdf_utm = gdf_wgs84.to_crs(utm_crs)
            bounds = gdf_utm.total_bounds
            buffered_box = box(
                bounds[0] - self.buffer_meters,
                bounds[1] - self.buffer_meters,
                bounds[2] + self.buffer_meters,
                bounds[3] + self.buffer_meters,
            )
            self.buffered_extent = gpd.GeoDataFrame(
                {"geometry": [buffered_box]}, crs=utm_crs
            ).to_crs("EPSG:4326")

        else:  # Raster
            with rasterio.open(self.target_file) as src:
                self.target_raster = {
                    "bounds": src.bounds,
                    "crs": src.crs,
                    "transform": src.transform,
                    "width": src.width,
                    "height": src.height,
                    "data": src.read(self.target_band, masked=True),
                }

            # Create buffered bounds
            minx, miny, maxx, maxy = self.target_raster["bounds"]
            buffer_deg = self.buffer_meters / 111000  # Rough conversion
            buffered_box = box(
                minx - buffer_deg,
                miny - buffer_deg,
                maxx + buffer_deg,
                maxy + buffer_deg,
            )
            self.buffered_extent = gpd.GeoDataFrame(
                {"geometry": [buffered_box]}, crs=self.target_raster["crs"]
            )

        return self.buffered_extent

    def _crop_metric_to_target_extent(
        self,
        metric_data,
        channel_label: str,
        extra_margin_m: float = 100.0,
    ):
        """Trim a *vector* metric source to the target's buffered extent.

        For a small study area, most of a national-scale source is dead
        weight: it's never queried because no entity's buffer reaches it.
        Cropping a vector source at load time shrinks RAM and sindex build
        time without changing any aggregation values.

        Raster metrics are **not** cropped: they're sampled through small
        per-entity windowed reads (``LazyRasterArray`` / windowed slices), so
        a cropped in-memory copy buys nothing and, for a national-scale grid,
        can be hundreds of GB. The raster dict is returned unchanged after a
        cheap, read-free overlap check that warns on a CRS / extent mismatch.

        The vector crop box = the target's pre-computed ``buffered_extent``
        (already buffered by ``buffer_meters`` = max(GVI, NDVI) at job submit)
        plus a small ``extra_margin_m`` slack. Returns the input unchanged
        when ``self.buffered_extent`` isn't yet available.
        """
        if self.buffered_extent is None or metric_data is None:
            return metric_data

        if isinstance(metric_data, dict):  # raster — sampled lazily, not cropped
            # No materialization: downstream reads small per-entity windows
            # from disk (LazyRasterArray). Run only a cheap, read-free sanity
            # check on the reprojected extent so a CRS / extent mismatch is
            # flagged loudly instead of silently sampling the wrong pixels.
            try:
                raster_crs = metric_data["crs"]
                bounds = metric_data.get("bounds")
                minx, miny, maxx, maxy = (
                    float(v)
                    for v in self.buffered_extent.to_crs(raster_crs).total_bounds
                )
                if bounds is not None and np.all(np.isfinite([minx, miny, maxx, maxy])):
                    rl, rb, rr, rt = (
                        float(bounds.left),
                        float(bounds.bottom),
                        float(bounds.right),
                        float(bounds.top),
                    )
                    if maxx < rl or minx > rr or maxy < rb or miny > rt:
                        _log(
                            "WARN",
                            f"{channel_label}: target extent does not overlap the "
                            "metric raster — check the target / raster CRS.",
                        )
                    else:
                        ix = max(0.0, min(maxx, rr) - max(minx, rl))
                        iy = max(0.0, min(maxy, rt) - max(miny, rb))
                        fx, fy = (rr - rl), (rt - rb)
                        frac = (ix / fx) * (iy / fy) if fx > 0 and fy > 0 else 0.0
                        if frac > 0.5:
                            _log(
                                "WARN",
                                f"{channel_label}: the buffered target extent covers "
                                f"~{frac * 100:.0f}% of the metric raster — a study "
                                "area should be a small slice, so this likely means a "
                                "target/raster CRS mismatch (sampled values would be "
                                "wrong). Sampling proceeds via windowed reads.",
                            )
            except Exception:
                pass
            return metric_data

        # Vector path
        try:
            extent_in_metric_crs = self.buffered_extent.to_crs(metric_data.crs)
            if getattr(extent_in_metric_crs.crs, "is_geographic", False):
                metric_crs_buf = estimate_metre_projected_crs_for_gdf(
                    extent_in_metric_crs
                )
                crop_geom = (
                    extent_in_metric_crs.to_crs(metric_crs_buf)
                    .buffer(extra_margin_m)
                    .to_crs(metric_data.crs)
                    .union_all()
                )
            else:
                crop_geom = extent_in_metric_crs.buffer(extra_margin_m).union_all()
        except Exception:
            crop_geom = self.buffered_extent.to_crs(metric_data.crs).union_all()
        n_before = len(metric_data)
        sindex = metric_data.sindex
        hits = list(sindex.query(crop_geom, predicate="intersects"))
        if not hits:
            _log(
                "WARN",
                f"{channel_label}: vector metric has no features inside the "
                "target extent — leaving the full source in place.",
            )
            return metric_data
        cropped = metric_data.iloc[hits].reset_index(drop=True)
        # Preserve attrs (metric_column hint set by _load_metric_file).
        cropped.attrs.update(metric_data.attrs)
        if len(cropped) == n_before:
            return metric_data
        saved_pct = 100.0 * (1.0 - len(cropped) / max(n_before, 1))
        _log(
            "INFO",
            f"{channel_label}: vector cropped to target extent "
            f"({n_before:,} → {len(cropped):,} features, ~{saved_pct:.1f}% saved).",
        )
        return cropped

    def adopt_metric_data(
        self,
        veg_data,
        terrain_data,
        ndvi_data,
        *,
        ndvi_resolution_m: float | None = None,
        gvi_grid_spacing_m: float | None = None,
    ) -> None:
        """Reuse already-loaded, already-cropped metric frames from another engine.

        Multi-outcome fusion builds a fresh engine per outcome, but every outcome
        shares the same target file, buffer, and metric paths, so the cropped
        veg / terrain / NDVI sources are identical. Adopting the first engine's
        frames skips N-fold file reads, reprojection, cropping, and spatial-index
        rebuilds. The frames are treated as read-only downstream (spatial-index
        caching onto the shared GeoDataFrame only makes later engines faster), so
        sharing references is safe. The two sampling-grid steps are carried over
        because they drive aggregation, not just download, and are otherwise only
        set inside :meth:`load_metrics`.
        """
        self.veg_data = veg_data
        self.terrain_data = terrain_data
        self.ndvi_data = ndvi_data
        if ndvi_resolution_m is not None:
            self._ndvi_export_resolution_m = float(ndvi_resolution_m)
        if gvi_grid_spacing_m is not None:
            self._gvi_grid_spacing_m = float(gvi_grid_spacing_m)
        self._clear_ring_caches()

    def load_metrics(
        self,
        veg_file: str | None = None,
        terrain_file: str | None = None,
        ndvi_file: str | None = None,
        cache_metrics: bool = True,
        gvi_api_key: str | None = None,
        ndvi_start_date: str = "2023-01-01",
        ndvi_end_date: str = "2023-12-31",
        ndvi_project_id: str | None = None,
        progress_callback: Callable[..., Any] | None = None,
        cancel_callback: Callable[..., Any] | None = None,
        force_download: bool = False,
        ndvi_resolution_m: float | None = None,
        gvi_grid_spacing_m: float | None = None,
    ) -> None:
        """
        Load vegetation, terrain, and NDVI metrics from files or auto-download.

        If metric files are not provided, automatically computes them within
        the buffered extent using GVI and NDVI engines.

        Args:
            veg_file: Path to pre-computed vegetation (GVI veg) GeoJSON/Raster
            terrain_file: Path to pre-computed terrain (GVI terrain) GeoJSON/Raster
            ndvi_file: Path to pre-computed NDVI GeoJSON/Raster
            cache_metrics: Whether to cache downloaded metrics for future use
            gvi_api_key: Google Street View API key for GVI download
            ndvi_start_date: Start date for NDVI composite (YYYY-MM-DD)
            ndvi_end_date: End date for NDVI composite (YYYY-MM-DD)
            ndvi_project_id: Google Earth Engine project ID for NDVI download
            force_download: If True, bypass cache and force fresh download of all metrics
            ndvi_resolution_m: GEE export resolution (m) when fetching NDVI; default 10
            gvi_grid_spacing_m: Street-view sampling grid spacing (m) when fetching GVI; default 75
        """
        if self.buffered_extent is None:
            self.load_target()

        self._ndvi_export_resolution_m = (
            float(ndvi_resolution_m) if ndvi_resolution_m is not None else 10.0
        )
        self._gvi_grid_spacing_m = (
            float(gvi_grid_spacing_m) if gvi_grid_spacing_m is not None else 75.0
        )
        self._clear_ring_caches()
        # Check for multi-band GVI cache (Band 1=Veg, Band 2=Terrain)
        gvi_multiband_cache = self._get_cache_filename("gvi_combined", ".tif")

        # Helper function to load multi-band GVI
        def load_multiband_gvi(filepath: str) -> bool:
            """Try to load multi-band GVI from filepath. Returns True if successful."""
            try:
                logger.info(f"Checking if {filepath} is a multi-band GVI raster...")
                with rasterio.open(filepath) as src:
                    if src.count >= 2:
                        logger.info(
                            f"Found {src.count}-band raster, loading as multi-band GVI"
                        )
                        # Read Band 1 (Vegetation) and Band 2 (Terrain)
                        veg_band = src.read(1)
                        terrain_band = src.read(2)
                        transform = src.transform
                        crs = src.crs
                        height, width = veg_band.shape

                        # Pixel centres straight from the affine, and only for
                        # the pixels that survive the NaN filter — building a
                        # geometry per grid cell first would cost tens of GB on
                        # a large raster.
                        def _band_to_points(values: np.ndarray, col: str):
                            flat = values.ravel()
                            keep = np.flatnonzero(~np.isnan(flat))
                            rows_i = (keep // width).astype(np.int64)
                            cols_i = (keep % width).astype(np.int64)
                            xs_k, ys_k = xy(transform, rows_i, cols_i, offset="center")
                            gdf = gpd.GeoDataFrame(
                                {col: flat[keep]},
                                geometry=gpd.points_from_xy(
                                    np.asarray(xs_k), np.asarray(ys_k)
                                ),
                                crs=crs,
                            )
                            gdf.attrs["metric_column"] = col
                            return gdf

                        veg_gdf = _band_to_points(veg_band, "veg")
                        self.veg_data = veg_gdf
                        terrain_gdf = _band_to_points(terrain_band, "terrain")
                        self.terrain_data = terrain_gdf

                        logger.info(
                            f"✓ Loaded {len(veg_gdf)} veg points and {len(terrain_gdf)} terrain points from multi-band raster"
                        )
                        return True
                    else:
                        logger.info(
                            f"Raster has only {src.count} band(s), not a multi-band GVI cache"
                        )
                        return False
            except Exception as e:
                logger.debug(f"Failed to load as multi-band GVI: {e}")
                return False

        def load_combined_gvi_vector(filepath: str) -> bool:
            """Split a single GVI vector file carrying both veg + terrain columns.

            The GVI engine writes its GeoPackage with both ``gvi_veg`` and
            ``gvi_ter`` columns in one file (see :func:`geofuse.gvi._make_result`);
            uploading just that file as ``veg_file`` should populate both
            ``self.veg_data`` and ``self.terrain_data``. Returns True if a
            split was performed, False otherwise (caller falls back to the
            single-column path).
            """
            try:
                gdf = gpd.read_file(filepath)
            except Exception as exc:
                logger.debug(f"Failed to read combined GVI vector: {exc}")
                return False
            veg_col = match_column_alias(gdf.columns, metric_intake.VEG_COLUMNS)
            ter_col = match_column_alias(gdf.columns, metric_intake.TERRAIN_COLUMNS)
            if not (veg_col and ter_col):
                return False
            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)
            gdf = normalize_geographic_gdf_to_wgs84(gdf)
            veg_gdf = gdf[["geometry", veg_col]].rename(columns={veg_col: "veg"}).copy()
            veg_gdf = veg_gdf.dropna(subset=["veg"])
            veg_gdf.attrs["metric_column"] = "veg"
            ter_gdf = (
                gdf[["geometry", ter_col]].rename(columns={ter_col: "terrain"}).copy()
            )
            ter_gdf = ter_gdf.dropna(subset=["terrain"])
            ter_gdf.attrs["metric_column"] = "terrain"
            self.veg_data = veg_gdf
            self.terrain_data = ter_gdf
            logger.info(
                f"✓ Split combined GVI vector ({filepath}): "
                f"{len(veg_gdf):,} veg samples (column {veg_col!r}), "
                f"{len(ter_gdf):,} terrain samples (column {ter_col!r})."
            )
            return True

        # Load or Auto-download Vegetation and Terrain

        # Check if uploaded veg_file is a multi-band raster
        if (
            veg_file
            and os.path.exists(veg_file)
            and veg_file.endswith((".tif", ".tiff"))
        ):
            if load_multiband_gvi(veg_file):
                pass
            else:
                self.veg_data = _helpers.load_metric_file(veg_file, "veg")
                if terrain_file and os.path.exists(terrain_file):
                    self.terrain_data = _helpers.load_metric_file(
                        terrain_file, "terrain"
                    )
                elif not terrain_file or not os.path.exists(terrain_file):
                    logger.warning(
                        "Veg file provided but terrain file missing. Auto-downloading terrain..."
                    )
                    _, terrain_file = self._auto_download_gvi_both(
                        api_key=gvi_api_key,
                        cache=cache_metrics,
                        progress_callback=progress_callback,
                        cancel_callback=cancel_callback,
                    )
                    self.terrain_data = _helpers.load_metric_file(
                        terrain_file, "terrain"
                    )
        # Check if uploaded veg_file is a combined GVI vector
        # (single GeoPackage / GeoJSON with both ``gvi_veg`` and ``gvi_ter``
        # columns — the default shape produced by ``geofuse.gvi``).
        elif (
            veg_file
            and os.path.exists(veg_file)
            and not terrain_file
            and load_combined_gvi_vector(veg_file)
        ):
            pass
        # Check cached multi-band file
        elif os.path.exists(gvi_multiband_cache):
            if load_multiband_gvi(gvi_multiband_cache):
                pass
            else:
                logger.warning("Cached multi-band file corrupted. Re-downloading...")
                veg_file, terrain_file = self._auto_download_gvi_both(
                    api_key=gvi_api_key,
                    cache=cache_metrics,
                    progress_callback=progress_callback,
                    cancel_callback=cancel_callback,
                )
                self.veg_data = _helpers.load_metric_file(veg_file, "veg")
                self.terrain_data = _helpers.load_metric_file(terrain_file, "terrain")
        # Load separate veg/terrain files if provided
        elif (
            veg_file
            and os.path.exists(veg_file)
            and terrain_file
            and os.path.exists(terrain_file)
        ):
            logger.info("Loading separate veg and terrain files...")
            self.veg_data = _helpers.load_metric_file(veg_file, "veg")
            self.terrain_data = _helpers.load_metric_file(terrain_file, "terrain")
        # Auto-download if nothing provided
        elif (not veg_file or not os.path.exists(veg_file)) and (
            not terrain_file or not os.path.exists(terrain_file)
        ):
            logger.info("No GVI files provided. Auto-downloading...")
            veg_file, terrain_file = self._auto_download_gvi_both(
                api_key=gvi_api_key,
                cache=cache_metrics,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
            )
            self.veg_data = _helpers.load_metric_file(veg_file, "veg")
            self.terrain_data = _helpers.load_metric_file(terrain_file, "terrain")
        else:
            # Partial files provided - try to load what we have
            if veg_file and os.path.exists(veg_file):
                logger.info(f"Loading vegetation from: {veg_file}")
                self.veg_data = _helpers.load_metric_file(veg_file, "veg")
            if terrain_file and os.path.exists(terrain_file):
                logger.info(f"Loading terrain from: {terrain_file}")
                self.terrain_data = _helpers.load_metric_file(terrain_file, "terrain")

        # Validate that we have both veg and terrain data
        if self.veg_data is None or self.terrain_data is None:
            missing = []
            if self.veg_data is None:
                missing.append("vegetation")
            if self.terrain_data is None:
                missing.append("terrain")
            raise ValueError(
                f"Failed to load {' and '.join(missing)} data. "
                f"Please provide valid GVI files or enable auto-download."
            )

        if self.veg_data is not None and self.terrain_data is not None:
            logger.info("✓ GVI data loaded successfully")

        # Load or Auto-download NDVI
        if ndvi_file and os.path.exists(ndvi_file) and not force_download:
            logger.info(f"Loading NDVI from: {ndvi_file}")
            self._validate_metric_bounds(ndvi_file)
            self.ndvi_data = _helpers.load_metric_file(ndvi_file, "ndvi")
        else:
            if force_download:
                logger.info("Force download enabled. Skipping cache check for NDVI...")
            else:
                logger.info("Auto-downloading NDVI metrics (Sentinel-2)...")
            ndvi_file = self._auto_download_ndvi(
                start_date=ndvi_start_date,
                end_date=ndvi_end_date,
                project_id=ndvi_project_id,
                cache=cache_metrics,
                force_download=force_download,
                cancel_callback=cancel_callback,
            )
            if cancel_callback and cancel_callback():
                return
            self.ndvi_data = _helpers.load_metric_file(ndvi_file, "ndvi")

        # Pre-crop every metric source to the target's buffered extent
        # (target.bounds + buffer_meters, which is max(GVI, NDVI) buffer).
        # Cropping once at load time shrinks RAM, sindex build time, and
        # per-window read latency without changing any aggregation values:
        # entities never query metric pixels outside their max buffer.
        self.veg_data = self._crop_metric_to_target_extent(
            self.veg_data, channel_label="GVI vegetation"
        )
        self.terrain_data = self._crop_metric_to_target_extent(
            self.terrain_data, channel_label="GVI terrain"
        )
        self.ndvi_data = self._crop_metric_to_target_extent(
            self.ndvi_data, channel_label="NDVI"
        )

    def _validate_metric_bounds(self, metric_file: str) -> bool:
        """Warn only when a metric misses the buffered extent beyond real slack.

        Earth Engine exports snap to their pixel grid, and reprojecting the
        extent transforms only the box's corners (the true region bulges past
        the straight-edged polygon), so a file downloaded for this very extent
        routinely falls a fraction of a pixel short on an edge. Strict
        containment flags that harmless shortfall; instead the gap is measured
        in metres and only a shortfall beyond ``max(2·pixel, 20 m)`` (rasters)
        or ``20 m`` (vectors) is reported — once per file per process, with the
        size of the gap and the share of the extent left uncovered.

        Returns ``True`` when coverage is within tolerance, ``False`` otherwise.
        """
        is_raster = metric_file.endswith((".tif", ".tiff"))
        px = py = 0.0
        if is_raster:
            with rasterio.open(metric_file) as src:
                metric_box = box(*src.bounds)
                metric_crs = src.crs
                px, py = abs(src.transform.a), abs(src.transform.e)
        else:
            metric_gdf = gpd.read_file(metric_file)
            metric_box = box(*metric_gdf.total_bounds)
            metric_crs = metric_gdf.crs

        # Measure the shortfall in metres: reproject both boxes into a metre CRS
        # (the metric CRS itself if already projected, otherwise a local UTM).
        extent_in_metric = self.buffered_extent.to_crs(metric_crs)
        if getattr(metric_crs, "is_geographic", True):
            measure_crs = estimate_metre_projected_crs_for_gdf(extent_in_metric)
        else:
            measure_crs = metric_crs
        ext_m = extent_in_metric.to_crs(measure_crs).geometry.iloc[0]
        met_m = gpd.GeoSeries([metric_box], crs=metric_crs).to_crs(measure_crs).iloc[0]
        exb, meb = ext_m.bounds, met_m.bounds  # (minx, miny, maxx, maxy), metres
        shortfalls = {
            "west": max(0.0, meb[0] - exb[0]),
            "south": max(0.0, meb[1] - exb[1]),
            "east": max(0.0, exb[2] - meb[2]),
            "north": max(0.0, exb[3] - meb[3]),
        }
        worst_edge = max(shortfalls, key=shortfalls.__getitem__)
        max_short = shortfalls[worst_edge]

        if is_raster:
            # Convert a geographic pixel to metres via the box's own deg→m scale.
            if getattr(metric_crs, "is_geographic", True):
                deg_w = metric_box.bounds[2] - metric_box.bounds[0]
                scale = ((meb[2] - meb[0]) / deg_w) if deg_w > 0 else 111_320.0
                pixel_m = max(px, py) * scale
            else:
                pixel_m = max(px, py)
            tolerance_m = max(2.0 * pixel_m, 20.0)
        else:
            tolerance_m = 20.0

        if max_short <= tolerance_m:
            return True

        try:
            uncovered_pct = (
                100.0 * (1.0 - ext_m.intersection(met_m).area / ext_m.area)
                if ext_m.area > 0
                else float("nan")
            )
        except Exception:
            uncovered_pct = float("nan")

        key = (
            os.path.abspath(metric_file),
            tuple(round(b, 3) for b in metric_box.bounds),
        )
        if key not in _WARNED_METRIC_BOUNDS:
            _WARNED_METRIC_BOUNDS.add(key)
            logger.warning(
                "%s metric falls short of the buffered extent by up to %.0f m on "
                "the %s edge (~%.1f%% of the extent uncovered); rings near that "
                "edge will be partially empty. File: %s",
                "Raster" if is_raster else "Vector",
                max_short,
                worst_edge,
                uncovered_pct,
                metric_file,
            )
        return False

    def _get_cache_filename(self, metric_type: str, extension: str = ".geojson") -> str:
        """Generate deterministic cache filename based on target file and boundary."""

        # Use target filename as base
        filename = os.path.basename(self.target_file)
        name, _ = os.path.splitext(filename)

        # Add boundary hash to make it unique per area
        if self.buffered_extent is not None:
            # Create hash from boundary coordinates
            bounds = self.buffered_extent.total_bounds
            bounds_str = (
                f"{bounds[0]:.6f}_{bounds[1]:.6f}_{bounds[2]:.6f}_{bounds[3]:.6f}"
            )
            bounds_hash = hashlib.md5(bounds_str.encode()).hexdigest()[:8]
            cache_name = f"cache-{metric_type}-{name}-{bounds_hash}{extension}"
        else:
            # Fallback to simple naming if no boundary yet
            cache_name = f"cache-{metric_type}-{name}{extension}"

        return os.path.join(self.cache_dir, cache_name)

    def _save_points_as_multiband_raster(
        self, points_gdf: gpd.GeoDataFrame, value_cols: list[str], output_path: str
    ) -> None:
        """
        Convert point GeoDataFrame to multi-band raster matching the target raster grid.

        Args:
            points_gdf: GeoDataFrame with point geometries and values
            value_cols: List of column names for each band
            output_path: Path to save the multi-band GeoTIFF
        """
        if not hasattr(self, "target_raster") or self.target_raster is None:
            raise ValueError("No target raster available for grid alignment")

        from scipy.interpolate import griddata

        # Get target raster properties
        transform = self.target_raster["transform"]
        crs = self.target_raster["crs"]
        height, width = self.target_raster["data"].shape

        # Reproject points to match target raster CRS
        points_in_raster_crs = points_gdf.to_crs(crs)

        # Extract point coordinates
        coords = np.array([(p.x, p.y) for p in points_in_raster_crs.geometry])

        # Generate target grid coordinates
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        xs, ys = xy(transform, rows.flatten(), cols.flatten(), offset="center")
        grid_coords = np.column_stack([xs, ys])

        # Interpolate each band
        bands = []
        for value_col in value_cols:
            values = points_in_raster_crs[value_col].values

            # Filter out NaN values
            valid_mask = ~np.isnan(values)
            valid_coords = coords[valid_mask]
            valid_values = values[valid_mask]

            if len(valid_values) == 0:
                logger.warning(f"No valid values in {value_col} column, using zeros")
                grid_values = np.zeros((height, width), dtype=np.float32)
            else:
                # Interpolate using nearest neighbor
                grid_values = griddata(
                    valid_coords, valid_values, grid_coords, method="nearest"
                )
                grid_values = grid_values.reshape(height, width).astype(np.float32)

            bands.append(grid_values)

        # Write multi-band GeoTIFF — shared compression / tiling defaults.
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=len(bands),
            dtype=rasterio.float32,
            crs=crs,
            transform=transform,
            **default_geotiff_creation_options(rasterio.float32),
        ) as dst:
            for i, band_data in enumerate(bands, 1):
                dst.write(band_data, i)
                dst.set_band_description(i, value_cols[i - 1])

    def _auto_download_gvi_both(
        self,
        api_key: str | None = None,
        cache: bool = True,
        progress_callback: Callable[..., Any] | None = None,
        cancel_callback: Callable[..., Any] | None = None,
    ) -> tuple[str, str]:
        """
        Auto-download GVI metrics (both veg and terrain) in a single analysis.

        Args:
            api_key: Google Street View API key (optional)
            cache: Whether to save to cache directory
            progress_callback: Callback for progress updates
            cancel_callback: Callback to check if analysis should be cancelled

        Returns:
            Tuple of (veg_file_path, terrain_file_path)
        """
        from .gvi import GVIEngine

        # Check cache first using deterministic filenames
        veg_cache_path = self._get_cache_filename("veg", ".geojson")
        terrain_cache_path = self._get_cache_filename("terrain", ".geojson")

        if os.path.exists(veg_cache_path) and os.path.exists(terrain_cache_path):
            logger.info(f"Found cached veg at: {veg_cache_path}")
            logger.info(f"Found cached terrain at: {terrain_cache_path}")
            return veg_cache_path, terrain_cache_path

        # Initialize GVI engine with model path
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "geofuse",
            "model",
            "best_model.pth",
        )
        gvi_engine = GVIEngine(model_path=model_path, api_key=api_key)

        # Run GVI analysis on buffered extent
        logger.info(
            "Computing GVI metrics for buffered extent (this may take a while)..."
        )

        gvi_step = max(1, int(round(self._gvi_grid_spacing_m)))

        # Convert buffered extent to target points or grid
        if self.is_points:
            # Use target points directly
            analysis_gdf = self.target_gdf.copy()
            logger.info(f"Using {len(analysis_gdf)} target points for GVI analysis")
        else:
            # Use buffered extent in EPSG:4326 - GVI engine will handle grid generation
            analysis_gdf = self.buffered_extent.copy()
            if analysis_gdf.crs.to_epsg() != 4326:
                analysis_gdf = analysis_gdf.to_crs("EPSG:4326")

            bounds = analysis_gdf.total_bounds
            area_km2 = ((bounds[2] - bounds[0]) * 111) * ((bounds[3] - bounds[1]) * 111)
            logger.info(f"Buffered extent area: ~{area_km2:.2f} km²")
            logger.info(f"Bounds (EPSG:4326): {bounds}")
            logger.info(f"Geometry type: {analysis_gdf.geometry.iloc[0].geom_type}")
            logger.info(
                f"GVI will generate grid at ~{gvi_step} m spacing within this polygon"
            )

            # Verify the polygon is valid
            if not analysis_gdf.geometry.iloc[0].is_valid:
                logger.warning("Invalid geometry detected, attempting to fix...")
                analysis_gdf.geometry = analysis_gdf.geometry.buffer(0)

            logger.info(
                f"Polygon area: {analysis_gdf.geometry.iloc[0].area:.6f} square degrees"
            )

        # Initialize result collector
        accumulated_results = []

        def collect_result(res):
            accumulated_results.append(res)
            if len(accumulated_results) % 10 == 0:
                logger.info(f"GVI: Collected {len(accumulated_results)} results so far")

        def on_progress(curr, total):
            if progress_callback:
                # Use "gvi" as component name for combined download
                progress_callback("gvi", curr, total)

        logger.info("Starting GVI analysis...")
        logger.info(
            f"Input GDF: {len(analysis_gdf)} features, CRS: {analysis_gdf.crs}, Geometry type: {analysis_gdf.geometry.iloc[0].geom_type}"
        )

        gvi_engine.run_analysis(
            analysis_gdf,
            folder=self.cache_dir,
            step=gvi_step,
            save_panos=False,
            save_masks=False,
            result_callback=collect_result,
            progress_callback=on_progress,
            cancel_callback=cancel_callback,
        )

        # Convert accumulated results to GeoDataFrame
        if not accumulated_results:
            # Check if analysis was cancelled
            if cancel_callback and cancel_callback():
                logger.info("GVI analysis cancelled by user")
                raise InterruptedError("GVI analysis cancelled by user")
            raise ValueError(
                "No GVI data collected. Check if Street View is available in this area."
            )

        result_gdf = gpd.GeoDataFrame(accumulated_results, crs=analysis_gdf.crs)

        # Save both veg and terrain components
        if cache:
            # Save as GeoJSON (always)
            if "gvi_veg" in result_gdf.columns:
                veg_gdf = result_gdf[["geometry", "gvi_veg"]].copy()
                veg_gdf.rename(columns={"gvi_veg": "veg"}, inplace=True)
                veg_gdf.attrs["metric_column"] = "veg"
                veg_gdf.to_file(veg_cache_path, driver="GeoJSON")
                logger.info(f"Cached veg to: {veg_cache_path}")

            if "gvi_ter" in result_gdf.columns:
                terrain_gdf = result_gdf[["geometry", "gvi_ter"]].copy()
                terrain_gdf.rename(columns={"gvi_ter": "terrain"}, inplace=True)
                terrain_gdf.attrs["metric_column"] = "terrain"
                terrain_gdf.to_file(terrain_cache_path, driver="GeoJSON")
                logger.info(f"Cached terrain to: {terrain_cache_path}")

            # Also save as multi-band GeoTIFF for easier upload/reuse
            if (
                not self.is_points
                and "gvi_veg" in result_gdf.columns
                and "gvi_ter" in result_gdf.columns
            ):
                gvi_multiband_path = self._get_cache_filename("gvi_combined", ".tif")

                try:
                    # Convert point data to raster with 2 bands (veg + terrain)
                    self._save_points_as_multiband_raster(
                        result_gdf, ["gvi_veg", "gvi_ter"], gvi_multiband_path
                    )
                    logger.info(
                        f"Cached multi-band GVI GeoTIFF to: {gvi_multiband_path}"
                    )
                    logger.info(
                        "  Band 1: Vegetation (gvi_veg), Band 2: Terrain (gvi_ter)"
                    )
                except Exception as e:
                    logger.warning(f"Could not create multi-band GeoTIFF cache: {e}")

            return veg_cache_path, terrain_cache_path
        else:
            # Use deterministic filenames even when not caching to temp directory
            import tempfile

            temp_dir = tempfile.mkdtemp(dir=self.cache_dir)
            veg_temp_path = os.path.join(temp_dir, "temp_veg.geojson")
            terrain_temp_path = os.path.join(temp_dir, "temp_terrain.geojson")

            veg_gdf = result_gdf[["geometry", "gvi_veg"]].copy()
            veg_gdf.rename(columns={"gvi_veg": "veg"}, inplace=True)
            veg_gdf.to_file(veg_temp_path, driver="GeoJSON")

            terrain_gdf = result_gdf[["geometry", "gvi_ter"]].copy()
            terrain_gdf.rename(columns={"gvi_ter": "terrain"}, inplace=True)
            terrain_gdf.to_file(terrain_temp_path, driver="GeoJSON")

            return veg_temp_path, terrain_temp_path

    def _auto_download_ndvi(
        self,
        start_date: str,
        end_date: str,
        project_id: str | None = None,
        cache: bool = True,
        force_download: bool = False,
        cancel_callback: Callable[..., Any] | None = None,
    ) -> str:
        """
        Auto-download NDVI metrics within buffered extent.

        Args:
            start_date: Start date for temporal composite (YYYY-MM-DD)
            end_date: End date for temporal composite (YYYY-MM-DD)
            project_id: Google Earth Engine project ID
            cache: Whether to save to cache directory
            force_download: If True, bypass cache and force fresh download
            cancel_callback: Optional ``() -> bool`` predicate. The NDVI engine
                polls this between tile downloads; when it returns True the
                run aborts with status ``"cancelled"``.

        Returns:
            Path to generated GeoTIFF file
        """
        from .ndvi import NDVIEngine

        # Check cache first using deterministic filename (unless force_download)
        cache_path = self._get_cache_filename("ndvi", ".tif")

        if os.path.exists(cache_path) and not force_download:
            logger.info(f"Found cached NDVI at: {cache_path}")
            return cache_path
        elif force_download and os.path.exists(cache_path):
            logger.info(f"Force download enabled. Removing old cache: {cache_path}")
            os.remove(cache_path)
            # Also remove corresponding geojson if it exists
            cache_geojson = cache_path.replace(".tif", ".geojson")
            if os.path.exists(cache_geojson):
                os.remove(cache_geojson)

        # Initialize NDVI engine
        ndvi_engine = NDVIEngine(project_id=project_id)

        # Download NDVI for buffered extent
        bounds = self.buffered_extent.total_bounds
        ndvi_res = int(max(5, round(self._ndvi_export_resolution_m)))
        logger.info(f"Downloading NDVI from Sentinel-2 ({start_date} to {end_date})...")
        logger.info(f"Area extent: {bounds}")
        logger.info(f"NDVI export resolution: {ndvi_res} m")
        logger.info("This may take several minutes depending on area size...")

        # Extract base name for output
        filename = os.path.basename(self.target_file)
        name, _ = os.path.splitext(filename)

        if cancel_callback and cancel_callback():
            raise InterruptedError("NDVI download cancelled by user")

        result = ndvi_engine.download_and_process(
            geometry=self.buffered_extent,
            start_date=start_date,
            end_date=end_date,
            output_name=f"{name if cache else 'temp'}",
            folder=self.cache_dir,
            resolution=ndvi_res,
            cancel_callback=cancel_callback,
        )

        if result.get("status") == "cancelled":
            raise InterruptedError("NDVI download cancelled by user")
        if result["status"] != "success":
            raise RuntimeError(f"NDVI download failed: {result['message']}")

        # If caching, move to deterministic filename
        if cache and result["tif"] != cache_path:
            import shutil

            shutil.move(result["tif"], cache_path)
            logger.info(f"NDVI cached to: {cache_path}")
            return cache_path

        logger.info(f"NDVI downloaded: {result['tif']}")
        return result["tif"]

    def _ring_cache_bytes_estimate(
        self,
        points_gdf: gpd.GeoDataFrame,
        metric_data: gpd.GeoDataFrame | dict,
        max_radius_m: float,
    ) -> float:
        """Rough resident size of the ring cache this call would build.

        Rasters: points × disc area ÷ pixel area. Vectors: points × the mean
        neighbour count of a small sample. Both times 8 bytes per value.
        """
        n_pts = len(points_gdf)
        if n_pts == 0 or max_radius_m <= 0:
            return 0.0
        disc_area = np.pi * float(max_radius_m) ** 2
        try:
            if isinstance(metric_data, dict):
                transform = metric_data["transform"]
                px = abs(float(transform.a))
                py = abs(float(transform.e))
                if getattr(metric_data["crs"], "is_geographic", False):
                    # Degree pixels: convert with the local metre-per-degree
                    # scale so the estimate is not off by ~10^5.
                    lat = float(
                        np.mean(
                            self.buffered_extent.to_crs("EPSG:4326")
                            .geometry.iloc[0]
                            .bounds[1::2]
                        )
                        if self.buffered_extent is not None
                        else 0.0
                    )
                    m_lon, m_lat = metres_per_degree_at_lat(lat)
                    px *= m_lon
                    py *= m_lat
                pixel_area = max(px * py, 1e-9)
                per_point = disc_area / pixel_area
            else:
                if len(metric_data) == 0:
                    return 0.0
                # Feature density from the extent's bounding box — cheap and
                # good enough to separate "fits easily" from "many GB".
                minx, miny, maxx, maxy = (float(v) for v in metric_data.total_bounds)
                span_x = max(maxx - minx, 1e-9)
                span_y = max(maxy - miny, 1e-9)
                if getattr(metric_data.crs, "is_geographic", False):
                    m_lon, m_lat = metres_per_degree_at_lat((miny + maxy) / 2.0)
                    span_x *= m_lon
                    span_y *= m_lat
                density = len(metric_data) / max(span_x * span_y, 1.0)
                per_point = disc_area * density
        except Exception:
            return 0.0
        return float(n_pts) * float(per_point) * 8.0

    def _search_n_jobs(self, metric: str) -> int:
        """Thread count for ``study.optimize``.

        Every metric runs its trials in parallel. All trials of a bootstrap
        share one fold, so the per-fold state they touch — fold statics, the
        pre-aggregation lookup plan, ring caches, fast components — takes
        idempotent writes: two threads may build the same entry, but neither
        can observe a wrong one. The two scoring caches that were *not* safe
        (:mod:`geofuse.pdcor`'s side cache and the spline-basis cache) evicted
        by walking their own iterator while another thread inserted; both now
        guard that bookkeeping with a lock and keep the expensive build outside
        it, which is what lifted the restriction to the fast-GLS metrics.

        An exact ``statsmodels`` MixedLM refit stays the exception: it is
        dominated by Python-level optimiser work holding the GIL, and threading
        it measured 0.62x of serial, so it is left alone.
        """
        exact_refit = (
            self.is_longitudinal
            and metric not in mixed_effects_scoring.FAST_SEARCH_METRICS
            and metric in mixed_effects_scoring.MIXEDLM_METRICS
        )
        if exact_refit:
            return 1
        return max(1, int(self._search_workers))

    def _clear_ring_caches(self) -> None:
        self._ring_raster_cache.clear()
        self._ring_vector_cache.clear()
        # Fold-static data (collapse codes, collapsed target/covariates,
        # points slices) shares the ring caches' lifecycle: both key off the
        # live fold row sets, which change with every bootstrap resample.
        self._fold_static_cache.clear()

    def _clear_scoring_caches(self) -> None:
        """Drop the per-subset scoring caches (pdcor sides, spline bases).

        Shares the spatial-basis cache's lifecycle: their entries fingerprint
        the scored subset's rows, which change on every split / resample.
        """
        self._pdcor_cache.clear()
        self._spline_basis_cache.clear()

    def _outer_radii_metres(self, *, gvi: bool) -> np.ndarray:
        if gvi:
            lo, hi, st = metric_sampling.radius_int_bounds(
                self.gvi_buffer_min_m,
                self.gvi_buffer_max_m,
                self.gvi_buffer_step_m,
            )
        else:
            lo, hi, st = metric_sampling.radius_int_bounds(
                self.ndvi_buffer_min_m,
                self.ndvi_buffer_max_m,
                self.ndvi_buffer_step_m,
            )
        return np.arange(lo, hi + 1, st, dtype=np.int64)

    def _aggregate_with_ring_cache(
        self,
        points_gdf: gpd.GeoDataFrame,
        metric_data: gpd.GeoDataFrame | dict,
        radius_m: float,
        stat: str,
        percentile: int,
        *,
        channel: str,
        fold_idx: int | None,
        subset: str | None,
    ) -> np.ndarray:
        """
        Circular neighbourhood aggregation using precomputed annuli when possible.

        When ``precompute_aggregations()`` has populated the pre-aggregation
        table and the trial's (radius, stat, percentile) maps to a cell, the
        call becomes a vectorised ``numpy.take`` and returns immediately.

        Otherwise falls back to the ring cache (or the direct circular buffer
        path) — same behaviour as before.
        """
        # ── Fast path: pre-aggregation lookup table ─────────
        if getattr(self, "_preaggregation_done", False):
            attrs = getattr(points_gdf, "attrs", None) or {}
            wave_indices = attrs.get("_gf_wave_idx")
            if (
                wave_indices is None
                and self.is_longitudinal
                and "wave" in points_gdf.columns
            ):
                spec = self.longitudinal_spec
                assert spec is not None
                wave_index_of = {w: i for i, w in enumerate(spec.wave_labels)}
                wave_indices = np.asarray(
                    [wave_index_of[str(w)] for w in points_gdf["wave"]],
                    dtype=np.int64,
                )
            # Point/line targets cache stats per unique pixel: resolve each
            # duplicated catchment row to its pixel id. Other targets key the
            # cache by row index directly. Fold slices carry a pre-built id
            # vector whose identity the cache's row indexer keys on.
            lookup_ids = attrs.get("_gf_lookup_ids")
            if lookup_ids is None:
                if "_preaggr_id" in points_gdf.columns:
                    lookup_ids = points_gdf["_preaggr_id"].to_numpy()
                else:
                    lookup_ids = points_gdf.index.values
            looked_up = self._lookup_preaggregation(
                lookup_ids,
                channel,
                radius_m,
                stat,
                percentile,
                wave_indices=wave_indices,
            )
            if looked_up is not None:
                return looked_up

        # Fell through the fast path (or pre-aggregation off): the ring-cache and
        # circular-buffer routines need geometry, so re-materialize it if a
        # geometry-free slice was passed in (see the fast path in ``_objective``).
        if not isinstance(points_gdf, gpd.GeoDataFrame):
            points_gdf = self.target_gdf.loc[points_gdf.index]

        if (
            fold_idx is None
            or subset is None
            or len(points_gdf) > self._max_points_ring_cache
        ):
            return _helpers.apply_circular_buffer_aggregation(
                points_gdf, metric_data, radius_m, stat, percentile
            )

        gvi = channel in ("veg", "terrain")
        radii = self._outer_radii_metres(gvi=gvi)
        if radii.size == 0:
            return _helpers.apply_circular_buffer_aggregation(
                points_gdf, metric_data, radius_m, stat, percentile
            )

        # A ring cache holds every raw metric value inside the largest radius
        # for every point, so the row count alone does not bound it: a fine
        # raster under a wide radius is tens of thousands of values per point.
        # Estimate the payload and take the direct path when it would not fit.
        if self._ring_cache_bytes_estimate(
            points_gdf, metric_data, float(radii[-1])
        ) > (self._max_ring_cache_bytes):
            return _helpers.apply_circular_buffer_aggregation(
                points_gdf, metric_data, radius_m, stat, percentile
            )

        cache_store = (
            self._ring_raster_cache
            if isinstance(metric_data, dict)
            else self._ring_vector_cache
        )
        key = metric_sampling.ring_cache_key(
            channel, fold_idx, subset, points_gdf, radii
        )

        if key not in cache_store:
            try:
                if isinstance(metric_data, dict):
                    rows = metric_sampling.precompute_raster_ring_values(
                        metric_data, points_gdf, radii
                    )
                else:
                    col = metric_sampling.vector_metric_column(metric_data, channel)
                    rows = metric_sampling.precompute_vector_ring_values(
                        metric_data, points_gdf, radii, col
                    )
                prefix = metric_sampling.ring_prefix_stats(rows)
                cache_store[key] = (radii, rows, prefix, {})
                logger.info(
                    f"Ring cache built: channel={channel} subset={subset} fold={fold_idx} "
                    f"points={len(points_gdf)} rings={len(radii)}"
                )
            except Exception as e:
                logger.warning(f"Ring cache build failed ({channel}): {e}")
                return _helpers.apply_circular_buffer_aggregation(
                    points_gdf, metric_data, radius_m, stat, percentile
                )

        _, rows, prefix, memo = cache_store[key]
        # Result memo: trials repeat the same (radius, stat, percentile) cells
        # constantly, and the grid is small (n_radii × mean/median/deciles) —
        # so the per-point percentile reduce runs once per distinct cell.
        end_idx = metric_sampling.ring_end_index(radii, radius_m)
        memo_key = (end_idx, stat, int(percentile) if stat == "percentile" else 0)
        hit = memo.get(memo_key)
        if hit is not None:
            return hit
        out = metric_sampling.aggregate_from_ring_cache(
            radii, rows, radius_m, stat, percentile, prefix=prefix
        )
        memo[memo_key] = out
        return out

    def _aggregate_channel_for_fold(
        self,
        static: dict,
        metric_data: gpd.GeoDataFrame | dict,
        radius_m: float,
        stat: str,
        percentile: int,
        *,
        channel: str,
        fold_idx: int | None,
        subset: str | None,
    ) -> tuple[np.ndarray, bool]:
        """Channel values for one scored fold, plus whether they are deduplicated.

        Returns ``(values, on_unique_axis)``. When the fold has a dedup axis and
        the cache can serve the cell, the values are one per distinct
        ``(pixel, wave)`` — the caller forms the composite on that shorter axis
        and gathers to rows with ``uniq_inverse`` before the entity collapse.
        Otherwise the values are per row, exactly as before.
        """
        uniq_ids = static.get("uniq_lookup_ids")
        if uniq_ids is not None and getattr(self, "_preaggregation_done", False):
            # The fold's row → cache-row resolution is the same for every trial,
            # so it is built once and held on the fold's static dict; a trial
            # then only reads its own (radius, stat) cell out of it.
            source = self._ensure_memory_loaded() or self._preaggr_cache
            column = preaggregation.stat_to_column(stat, percentile)
            if source is not None and column is not None:
                plan = self._preaggr_plan(
                    source, uniq_ids, static.get("uniq_wave_idx"), static
                )
                if plan is not None:
                    looked_up = source.gather(
                        plan, channel, int(round(radius_m)), column
                    )
                    if looked_up is not None:
                        return looked_up, True
        return (
            self._aggregate_with_ring_cache(
                static["points"],
                metric_data,
                radius_m,
                stat,
                percentile,
                channel=channel,
                fold_idx=fold_idx,
                subset=subset,
            ),
            False,
        )

    def prepare_fusion_data(self) -> pd.DataFrame:
        """
        Align vegetation, terrain, NDVI, and target data into a single DataFrame.

        Follows CGI.ipynb logic:
        - For point targets: sample metrics at each point location
        - For raster targets: align all data to target grid

        Returns:
            DataFrame with columns: [target, veg, terrain, ndvi] plus any
            configured covariate columns (broadcast per-row from the target
            attributes, or per-polygon for the polygon path).
        """
        _log("INFO", "====== PREPARE FUSION DATA ======")

        # Longitudinal mode replaces the as-loaded target with a long-format
        # frame keyed by (entity_id, wave) BEFORE the point/polygon prep
        # path runs — so the existing samplers see one row per (entity, wave)
        # observation and need no other changes.
        if self.is_longitudinal:
            self._materialize_longitudinal_target()

        # Covariates need attribute columns — raster targets don't have them
        # and the polygon/point paths need to validate the requested names
        # before any heavy work happens.
        if self.covariate_columns:
            if self.is_raster and not self.is_polygon_target:
                raise ValueError(
                    "covariate_columns are only supported for vector (point / "
                    "polygon) targets; this target is a raster. Got "
                    f"covariate_columns={self.covariate_columns}."
                )
            if self.target_gdf is not None:
                self._expand_categorical_covariates()

        # Only the point/line per-pixel path dedups the cache to unique pixels;
        # reset here so a re-prepared study can't inherit a stale source. The
        # greenery cache is dropped too; the coverage gate re-attaches a
        # reusable one (or the build makes a fresh one) for this run.
        self._preaggr_entity_gdf = None
        self._preaggregation_done = False
        self._preaggr_cache = None
        self._preaggr_pid_translate = None
        if self._preaggr_mem is not None:
            self._preaggr_mem.close()
            self._preaggr_mem = None
        if self.is_polygon_target:
            _log("INFO", "Target type: POLYGON (per-pixel CGI)")
            df = self._prepare_polygon_fusion()
        else:
            _log(
                "INFO",
                f"Target type: {'POINT/LINE (per-pixel CGI)' if self.is_points else 'RASTER'}",
            )
            df = (
                self._prepare_entity_fusion()
                if self.is_points
                else self._prepare_raster_fusion()
            )
        self._compute_channel_scale(df)
        # RAM management: narrow the resident per-pixel frames' dtypes (wave
        # string -> category, 64-bit -> 32-bit) before the search + the RAM
        # cache load. On a 36M-row per-pixel panel this frees several GB.
        df = _helpers.downcast_fusion_dtypes(df)
        self._downcast_target_gdf_dtypes()
        import gc as _gc

        _gc.collect()
        return df

    def _downcast_target_gdf_dtypes(self) -> None:
        """Apply the same narrowing to the resident ``target_gdf`` (and the
        unique-pixel source), which the search keeps in memory alongside the
        RAM cache."""
        if self.target_gdf is not None:
            _helpers.downcast_fusion_dtypes(self.target_gdf)
        if self._preaggr_entity_gdf is not None:
            _helpers.downcast_fusion_dtypes(self._preaggr_entity_gdf)

    def _compute_channel_scale(self, fusion_df: pd.DataFrame) -> None:
        """Fix per-channel [0, 1] min-max bounds from the predictor distribution.

        Robust 2nd/98th-percentile bounds per channel, computed once on the
        prepared probe values (no outcome involved, so no CV leakage) and
        applied identically to every fold / split / output. No-op unless
        ``normalize_channels`` is on.
        """
        if not self.normalize_channels:
            self._channel_minmax = None
            return
        bounds: dict[str, tuple[float, float]] = {}
        for ch in ("veg", "terrain", "ndvi"):
            if ch not in fusion_df.columns:
                bounds[ch] = (0.0, 1.0)
                continue
            vals = pd.to_numeric(fusion_df[ch], errors="coerce").to_numpy(np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                bounds[ch] = (0.0, 1.0)
                continue
            lo = float(np.percentile(vals, 2.0))
            hi = float(np.percentile(vals, 98.0))
            if hi <= lo:
                lo, hi = float(vals.min()), float(vals.max())
            bounds[ch] = (lo, hi)
        self._channel_minmax = bounds
        _log("INFO", f"Channel normalization bounds (2–98 pct): {bounds}")

    def _normalize_channel_arrays(
        self, veg: np.ndarray, terrain: np.ndarray, ndvi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Min-max scale each channel to [0, 1] using the fixed bounds, or pass through."""
        if not self.normalize_channels or not self._channel_minmax:
            return veg, terrain, ndvi

        def _scale(a: np.ndarray, ch: str) -> np.ndarray:
            lo, hi = self._channel_minmax.get(ch, (0.0, 1.0))
            arr = np.asarray(a)
            if not np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float64)
            if hi <= lo:
                return arr
            # Scalar ops keep the input float width (float32 on the per-pixel
            # trial path, float64 elsewhere).
            dt = arr.dtype.type
            return np.clip((arr - dt(lo)) / dt(hi - lo), dt(0.0), dt(1.0))

        return _scale(veg, "veg"), _scale(terrain, "terrain"), _scale(ndvi, "ndvi")

    # ────────────────────────────────────────────────────────────
    # ── Pre-aggregation grid ────────────────────────────────────
    # Trial percentiles are snapped to this grid so every trial maps onto a
    # stored column of the per-(entity, radius) cache.
    _PREAGGR_PERCENTILES = preaggregation.PERCENTILES

    def _preaggr_radii(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """(gvi_radii, ndvi_radii) snapped to int + step from the buffer ladders."""
        gvi_lo, gvi_hi, gvi_st = metric_sampling.radius_int_bounds(
            self.gvi_buffer_min_m, self.gvi_buffer_max_m, self.gvi_buffer_step_m
        )
        ndvi_lo, ndvi_hi, ndvi_st = metric_sampling.radius_int_bounds(
            self.ndvi_buffer_min_m, self.ndvi_buffer_max_m, self.ndvi_buffer_step_m
        )
        return (
            tuple(range(gvi_lo, gvi_hi + 1, gvi_st)),
            tuple(range(ndvi_lo, ndvi_hi + 1, ndvi_st)),
        )

    def _metric_fingerprint(self, metric, channel: str) -> str:
        """Cheap, deterministic identity for a loaded metric (vector or raster)."""
        if isinstance(metric, dict):  # raster
            arr = metric["data"]
            if isinstance(arr, LazyRasterArray):
                # Lazy raster: identify by file stat + grid (no full pixel read).
                stt = os.stat(arr.path)
                return (
                    f"ras:{os.path.basename(arr.path)}:{stt.st_size}:"
                    f"{int(stt.st_mtime)}:{arr.shape}:{arr.dtype}:"
                    f"{tuple(metric['bounds'])}"
                )
            data = np.ma.getdata(arr)
            sample = np.ascontiguousarray(data[::17, ::17]).tobytes()
            digest = hashlib.sha256(sample).hexdigest()[:16]
            return f"ras:{data.shape}:{tuple(metric['bounds'])}:{data.dtype}:{digest}"
        col = _helpers.metric_value_column(metric, channel)
        vals = np.ascontiguousarray(metric[col].to_numpy(dtype=np.float64))
        digest = hashlib.sha256(vals.tobytes()).hexdigest()[:16]
        return f"vec:{len(metric)}:{tuple(metric.total_bounds)}:{col}:{digest}"

    def check_channel_collinearity(
        self,
        *,
        data: "pd.DataFrame | None" = None,
        vif_threshold: float = 10.0,
        sample_size: int = 20_000,
        seed: int = 42,
    ) -> dict:
        """Run iterative-VIF channel reduction on the CGI grid pixel values.

        Reads per-pixel ``veg`` / ``terrain`` / ``ndvi`` columns from
        ``data`` (typically the ``fusion_df`` produced by
        :meth:`prepare_fusion_data` — every CGI grid pixel before any
        train/val/test split), optionally subsamples to ``sample_size``
        rows for speed, then delegates the math to
        :func:`statistical_testing.iterative_vif_reduction`.

        When ``data`` is not supplied, falls back to the post-split pool
        (``train_val_data`` + ``test_data``) — but the typical runner
        path calls this **before** ``split_data`` runs, so passing
        ``fusion_df`` explicitly is the standard usage.

        Channels dropped by the procedure are recorded in
        ``self._disabled_channels`` and pinned to weight 0 in every trial.
        The full report (pairwise Pearson, initial / final VIFs, the
        per-iteration drop history) is stored on
        ``self._collinearity_report`` so the runner can hand it to the UI.

        The check uses the raw pixel values rather than the
        pre-aggregated cache because the aggregation step smooths spatial
        detail and inflates correlations between channels (1000m buffers
        make everything look similar). The raw test is the strictest.

        Returns the same dict written to ``self._collinearity_report``.
        """
        from . import statistical_testing as _stats_mod

        if data is not None:
            df = data
        else:
            frames: list[pd.DataFrame] = []
            if self.train_val_data is not None:
                frames.append(self.train_val_data)
            if self.test_data is not None:
                frames.append(self.test_data)
            if not frames:
                raise ValueError(
                    "check_channel_collinearity needs a pixel-level DataFrame: "
                    "either pass ``data=fusion_df`` (the output of "
                    "prepare_fusion_data) or call split_data first so "
                    "``train_val_data`` / ``test_data`` are populated."
                )
            df = pd.concat(frames, ignore_index=False) if len(frames) > 1 else frames[0]

        all_channels = ["veg", "terrain", "ndvi"]
        missing = [c for c in all_channels if c not in df.columns]
        if missing:
            raise ValueError(
                f"Channel columns missing from CGI pool: {missing}. "
                "Re-run prepare_fusion_data() with all channels loaded."
            )

        X = df[all_channels].to_numpy(dtype=np.float64)
        mask = np.isfinite(X).all(axis=1)
        X = X[mask]
        if len(X) > int(sample_size):
            rng = np.random.default_rng(int(seed))
            idx = rng.choice(len(X), size=int(sample_size), replace=False)
            X = X[idx]
        logger.info(
            f"Collinearity check: {len(X):,} pixel rows over channels "
            f"{all_channels}, VIF threshold = {vif_threshold}."
        )

        report = _stats_mod.iterative_vif_reduction(
            X, names=list(all_channels), vif_threshold=float(vif_threshold)
        )

        kept = set(report["kept"])
        dropped = set(report["dropped"])
        self._disabled_channels = {c for c in all_channels if c not in kept}
        self._collinearity_report = dict(report)
        self._collinearity_report["vif_threshold"] = float(vif_threshold)
        self._collinearity_report["sample_size"] = int(len(X))
        self._collinearity_report["channels_in"] = list(all_channels)

        if dropped:
            logger.warning(
                f"Collinearity reduction dropped: {sorted(dropped)}. "
                f"Kept: {sorted(kept)}. Initial VIFs: "
                f"{dict(zip(all_channels, report['initial_vifs']))}."
            )
        else:
            logger.info(
                f"Collinearity check passed: all VIFs ≤ {vif_threshold}. "
                f"VIFs: {dict(zip(all_channels, report['initial_vifs']))}."
            )
        return self._collinearity_report

    def check_covariate_dispersion(
        self,
        *,
        data: "pd.DataFrame | None" = None,
        spread_ratio: float = 20.0,
    ) -> dict:
        """Flag outcome / covariate columns whose spread looks corrupted.

        For the outcome and each covariate column, compares the standard
        deviation against the robust inter-percentile spread ``p90 - p10``.
        A column where ``std > spread_ratio × (p90 - p10)`` has a bulk that
        sits in a narrow band while a handful of extreme values (uncleaned
        sentinels, a units mix-up, a decimal-point error) blow up the SD — the
        signature of an input that will produce meaningless standard errors.

        Returns a report dict with a per-column table and a ``flagged`` list.
        Does not raise or drop anything: the caller decides how loudly to warn.
        """
        if data is not None:
            df = data
        elif self.target_gdf is not None:
            df = self.target_gdf
        else:
            frames = [f for f in (self.train_val_data, self.test_data) if f is not None]
            df = pd.concat(frames, ignore_index=False) if frames else None
        if df is None:
            return {"columns": [], "flagged": [], "spread_ratio": float(spread_ratio)}

        cols = [self.target_feature, *self.covariate_columns]
        seen: set[str] = set()
        table: list[dict] = []
        flagged: list[str] = []
        for col in cols:
            if not col or col in seen or col not in df.columns:
                continue
            seen.add(col)
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size < 2:
                continue
            std = float(np.std(vals, ddof=0))
            p10, p90 = (float(x) for x in np.percentile(vals, [10, 90]))
            spread = p90 - p10
            is_outcome = col == self.target_feature
            bad = std > spread_ratio * spread
            entry = {
                "column": col,
                "role": "outcome" if is_outcome else "covariate",
                "std": std,
                "p10": p10,
                "p90": p90,
                "spread": spread,
                "flagged": bool(bad),
            }
            table.append(entry)
            if bad:
                flagged.append(col)

        report = {
            "columns": table,
            "flagged": flagged,
            "spread_ratio": float(spread_ratio),
        }
        self._dispersion_report = report
        return report

    def precompute_aggregations(
        self,
        progress_callback: Callable[[int, int], None] | None = None,
        cancel_callback: Callable[[], bool] | None = None,
        max_workers: int | None = None,
    ) -> bool:
        """Build (or reuse/resume) the on-disk per-(entity, radius) stat cache.

        For every sample entity and every buffer radius in the ladder, stores
        (mean, p10..p90) per channel in a per-job SQLite database
        (``geofuse/preaggregation.py``). Handles both vector (point) and raster
        metrics. The cache is fingerprinted on the sample geometry, metric
        sources, radius ladders, and stat set, so an identical re-run reuses it;
        an interrupted build resumes from its per-entity watermark.

        Returns ``True`` on completion, ``False`` if cancelled.
        """
        # The coverage gate may already have attached a complete reusable cache
        # (loading it once for both coverage and scoring); nothing left to build.
        if self._preaggregation_done and isinstance(
            self._preaggr_cache, preaggregation.GreeneryCache
        ):
            if progress_callback is not None:
                progress_callback(1, 1)
            return True
        if self.is_longitudinal:
            return self._precompute_aggregations_longitudinal(
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
                max_workers=max_workers,
            )

        for label, data in (
            ("veg", self.veg_data),
            ("terrain", self.terrain_data),
            ("ndvi", self.ndvi_data),
        ):
            if data is None:
                raise ValueError(
                    f"Pre-aggregation requires '{label}' metric data; got None."
                )
        if self.target_gdf is None or len(self.target_gdf) == 0:
            raise ValueError(
                "Pre-aggregation requires sample points; call prepare_fusion_data() "
                "first."
            )

        gvi_radii, ndvi_radii = self._preaggr_radii()
        # Cross-sectional shares the greenery store and compute path with the
        # longitudinal build; it is simply a single implicit wave. Only the
        # scoring/objective differs by analysis type, not the data prep or cache.
        preaggr_gdf = (
            self._preaggr_entity_gdf
            if self._preaggr_entity_gdf is not None
            else self.target_gdf
        )
        n_points = len(preaggr_gdf)

        utm_crs = _helpers.grid_metric_crs(preaggr_gdf, _log)
        crs_key = str(utm_crs.to_epsg() or utm_crs.to_wkt())
        cache = preaggregation.GreeneryCache(
            self.cache_dir,
            spacing_m=float(self.cgi_grid_spacing_m or 0.0),
            crs_key=crs_key,
            stats=preaggregation.STAT_COLUMNS,
        )
        cfg_key = cache.unit_key(
            self._metric_fingerprint(self.veg_data, "veg"),
            self._metric_fingerprint(self.terrain_data, "terrain"),
            self._metric_fingerprint(self.ndvi_data, "ndvi"),
        )
        cache.bind_wave(preaggregation.DEFAULT_WAVE_INDEX, cfg_key)

        entities_utm = preaggr_gdf.to_crs(utm_crs)
        all_points = bool((entities_utm.geometry.geom_type == "Point").all())
        entity_geoms_utm = list(entities_utm.geometry)
        if all_points:
            point_xy_utm = np.column_stack(
                [
                    entities_utm.geometry.x.to_numpy(),
                    entities_utm.geometry.y.to_numpy(),
                ]
            ).astype(np.float64)
        else:
            point_xy_utm = np.empty((0, 2), dtype=np.float64)
        if "_global_pid" in preaggr_gdf.columns:
            global_pids = np.asarray(
                preaggr_gdf["_global_pid"].to_numpy(), dtype=np.int64
            )
        else:
            global_pids = np.asarray(preaggr_gdf.index, dtype=np.int64)
        pid_to_pos = {int(p): i for i, p in enumerate(global_pids)}

        eff_gvi, eff_ndvi, missing, absent_ch = cache.open_unit(
            cfg_key,
            gvi_radii=gvi_radii,
            ndvi_radii=ndvi_radii,
            required_ids=global_pids,
            channels=self._cache_channels(),
        )
        if absent_ch:
            # A channel the stored unit never held has to be built for every id
            # it already covers, so the whole unit is recomputed once.
            _log("INFO", f"Cache extension: computing {list(absent_ch)} for all pixels.")
            missing = global_pids
        _log(
            "INFO",
            f"Pre-aggregation: {n_points:,} pixels · GVI radii {gvi_radii} m · "
            f"NDVI radii {ndvi_radii} m · stats {preaggregation.STAT_COLUMNS}. "
            f"Cache: {cache.dir} · {len(missing):,} to compute"
            + (" (fully reused)" if len(missing) == 0 else "")
            + ".",
        )
        if progress_callback is not None:
            progress_callback(0, max(len(missing), 1))

        if len(missing):
            if cancel_callback is not None and cancel_callback():
                self._preaggregation_done = False
                return False
            pos = np.fromiter(
                (pid_to_pos[int(p)] for p in missing),
                dtype=np.int64,
                count=len(missing),
            )
            stage_secs = {"vector_agg": 0.0, "raster_agg": 0.0}
            build_t0 = time.perf_counter()
            total = max(len(missing), 1)
            result = self._aggregate_pixel_channels(
                pos,
                veg_src=self.veg_data,
                terrain_src=self.terrain_data,
                ndvi_src=self.ndvi_data,
                shared_gvi=True,
                gvi_radii=eff_gvi,
                ndvi_radii=eff_ndvi,
                utm_crs=utm_crs,
                all_points=all_points,
                point_xy_utm=point_xy_utm,
                entity_geoms_utm=entity_geoms_utm,
                max_workers=max_workers,
                cancel_callback=cancel_callback,
                stage_secs=stage_secs,
                progress_callback=(
                    None
                    if progress_callback is None
                    else lambda done: progress_callback(done, total)
                ),
            )
            if result is None:
                _log("WARN", "Pre-aggregation cancelled (resumable).")
                self._preaggregation_done = False
                return False
            cache.commit_unit(cfg_key, missing, self._blocks(result))
            if progress_callback is not None:
                progress_callback(len(missing), total)
            build_secs = time.perf_counter() - build_t0
            _log(
                "OK",
                f"Pre-aggregation: built {len(missing):,} pixel(s) in "
                f"{build_secs:.0f}s — vector "
                f"{stage_secs['vector_agg']:.0f}s, raster "
                f"{stage_secs['raster_agg']:.0f}s.",
            )
            _log_parallel_efficiency(
                _log,
                "pre-aggregation",
                wall_s=build_secs,
                busy_s=stage_secs["vector_agg"] + stage_secs["raster_agg"],
                workers=int(stage_secs.get("workers", 1)),
            )

        self._preaggr_cache = cache
        self._preaggr_mem = None
        self._preaggregation_done = True
        self._build_pid_translation()
        _log(
            "OK",
            f"Pre-aggregation complete for {n_points:,} pixels "
            f"(resident {cache.n_bytes / 2**30:.2f} GB).",
        )
        return True

    def _prep_channel_source(
        self, src: Any, channel: str, utm_crs: Any, all_points: bool
    ) -> tuple:
        """Normalise one channel's metric into the form both run paths consume.

        Returns one of:

        * ``("raster", array_or_lazy, transform, raster_crs)``
        * ``("point", xy, values)`` — coordinates and values in ``utm_crs``
        * ``("geom", gdf, value_column, cell_buffer_m)``

        The point form stops short of the ``BallTree``: the in-process path
        builds one here, while the pool publishes the coordinates and lets each
        worker build its own, so no tree ever crosses the pickle channel.
        """
        if isinstance(src, dict):  # raster source
            return ("raster", src["data"], src["transform"], src["crs"])
        col = _helpers.metric_value_column(src, channel)
        metric = src.to_crs(utm_crs)
        metric = metric[metric[col].notna()]
        if not all_points:
            metric = metric.reset_index(drop=True)
            return (
                "geom",
                metric,
                col,
                preaggregation.metric_cell_buffer_m(metric),
            )
        if len(metric) == 0:
            raise ValueError(f"pre-aggregation: metric column '{col}' is all-NaN")
        xy = np.column_stack(
            [metric.geometry.x.to_numpy(), metric.geometry.y.to_numpy()]
        ).astype(np.float64)
        return ("point", xy, metric[col].to_numpy(dtype=np.float32))

    @staticmethod
    def _shared_gvi_geometry(veg_prep: tuple, ter_prep: tuple) -> np.ndarray | None:
        """The coordinates veg and terrain share, or ``None`` if they differ.

        ``veg`` and ``terrain`` are usually two attribute columns of the same
        GVI point layer, so their geometry — and the radius query over it — is
        identical and one query can serve both.
        """
        if veg_prep[0] != "point" or ter_prep[0] != "point":
            return None
        veg_xy, ter_xy = veg_prep[1], ter_prep[1]
        if len(veg_xy) == 0 or not np.array_equal(veg_xy, ter_xy):
            return None
        return veg_xy

    def _aggregate_pixel_channels(
        self,
        positions: np.ndarray,
        *,
        veg_src: Any,
        terrain_src: Any,
        ndvi_src: Any,
        shared_gvi: bool,
        gvi_radii: tuple[int, ...],
        ndvi_radii: tuple[int, ...],
        utm_crs: Any,
        all_points: bool,
        point_xy_utm: np.ndarray,
        entity_geoms_utm: list,
        max_workers: int | None,
        cancel_callback: Callable[[], bool] | None,
        stage_secs: dict[str, float],
        progress_callback: Callable[[int], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Compute veg / terrain / ndvi stats for the pixels at ``positions``.

        ``positions`` index into ``point_xy_utm`` / ``entity_geoms_utm``. Each
        channel samples through a raster disc or its own point/geometry tree,
        whichever its source is. When ``shared_gvi`` and veg + terrain are two
        point layers with identical geometry, one ``BallTree`` and one radius
        query serve both. Returns the three ``[n_positions, n_radii, n_stats]``
        arrays, or ``None`` if cancelled.

        The batches run in worker **processes** whenever the entities are points
        — which is every gridded target, so every per-pixel CGI run. The work is
        a long chain of small numpy calls per entity and therefore GIL-bound, so
        threads leave the machine idle where processes do not (measurements in
        :mod:`geofuse.parallel`). Entity geometry that is not points keeps the
        in-process path, since its channels are backed by spatial indexes that
        would have to be pickled rather than rebuilt.

        ``progress_callback`` receives the running count of finished entities.
        The pool size actually used is recorded as ``stage_secs["workers"]`` for
        the caller's parallel-efficiency log.
        """
        n = len(positions)
        nstats = len(preaggregation.STAT_COLUMNS)
        veg_out = np.full((n, len(gvi_radii), nstats), np.nan, dtype=np.float32)
        ter_out = np.full((n, len(gvi_radii), nstats), np.nan, dtype=np.float32)
        gvi_out = np.full((n, len(gvi_radii), nstats), np.nan, dtype=np.float32)
        ndvi_out = np.full((n, len(ndvi_radii), nstats), np.nan, dtype=np.float32)
        if n == 0:
            return veg_out, ter_out, ndvi_out, gvi_out

        preps = {
            "veg": self._prep_channel_source(veg_src, "veg", utm_crs, all_points),
            "terrain": self._prep_channel_source(
                terrain_src, "terrain", utm_crs, all_points
            ),
            "ndvi": self._prep_channel_source(ndvi_src, "ndvi", utm_crs, all_points),
        }
        shared_xy = (
            self._shared_gvi_geometry(preps["veg"], preps["terrain"])
            if shared_gvi and all_points
            else None
        )

        # Raster and geometry channels poll per entity and are the slow path, so
        # keep batches small unless every channel is a point layer.
        pure_points = shared_xy is not None and preps["ndvi"][0] == "point"
        batch_size = 1024 if pure_points else 128
        bounds = list(range(0, n, batch_size)) + [n]
        batches = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

        done_entities = 0

        def _store(res) -> None:
            nonlocal done_entities
            lo, hi, veg_b, ter_b, ndvi_b, gvi_b, vsec, rsec = res
            veg_out[lo:hi] = veg_b
            ter_out[lo:hi] = ter_b
            ndvi_out[lo:hi] = ndvi_b
            # ``None`` when the components came from separate source layers,
            # where an exact merged channel cannot be formed.
            gvi_out[lo:hi] = (veg_b + ter_b) if gvi_b is None else gvi_b
            stage_secs["vector_agg"] += vsec
            stage_secs["raster_agg"] += rsec
            done_entities += hi - lo
            if progress_callback is not None:
                progress_callback(done_entities)

        if all_points:
            workers = parallel.process_worker_count(len(batches), cap=max_workers)
            stage_secs["workers"] = float(workers)
            finished = self._aggregate_in_processes(
                positions,
                preps=preps,
                shared_xy=shared_xy,
                gvi_radii=gvi_radii,
                ndvi_radii=ndvi_radii,
                utm_crs=utm_crs,
                point_xy_utm=point_xy_utm,
                batches=batches,
                workers=workers,
                cancel_callback=cancel_callback,
                on_result=_store,
            )
        else:
            workers = parallel.workers_for(len(batches), cap=max_workers)
            stage_secs["workers"] = float(workers)
            finished = self._aggregate_in_threads(
                positions,
                preps=preps,
                shared_xy=shared_xy,
                gvi_radii=gvi_radii,
                ndvi_radii=ndvi_radii,
                utm_crs=utm_crs,
                point_xy_utm=point_xy_utm,
                entity_geoms_utm=entity_geoms_utm,
                batches=batches,
                workers=workers,
                cancel_callback=cancel_callback,
                on_result=_store,
            )
        if not finished:
            return None
        return veg_out, ter_out, ndvi_out, gvi_out

    def _aggregate_in_threads(
        self,
        positions: np.ndarray,
        *,
        preps: dict,
        shared_xy: np.ndarray | None,
        gvi_radii: tuple[int, ...],
        ndvi_radii: tuple[int, ...],
        utm_crs: Any,
        point_xy_utm: np.ndarray,
        entity_geoms_utm: list,
        batches: list,
        workers: int,
        cancel_callback: Callable[[], bool] | None,
        on_result: Callable[[Any], None],
    ) -> bool:
        """Run the batches in this process. ``False`` if cancelled."""
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

        state = _helpers.resolved_state(
            preps,
            shared_xy,
            gvi_radii=gvi_radii,
            ndvi_radii=ndvi_radii,
            utm_crs=utm_crs,
        )

        def _compute(lo: int, hi: int):
            bpos = positions[lo:hi]
            xy = point_xy_utm[bpos] if len(point_xy_utm) else None
            geoms = [entity_geoms_utm[i] for i in bpos]
            veg_b, ter_b, ndvi_b, gvi_b, vsec, rsec = (
                preaggregation.aggregate_entity_batch(
                    state, point_xy=xy, geoms=geoms
                )
            )
            return lo, hi, veg_b, ter_b, ndvi_b, gvi_b, vsec, rsec

        if workers <= 1:
            for lo, hi in batches:
                if cancel_callback is not None and cancel_callback():
                    return False
                on_result(_compute(lo, hi))
            return True

        with ThreadPoolExecutor(max_workers=workers) as ex:
            it = iter(batches)
            in_flight = {
                ex.submit(_compute, *b)
                for b in (next(it, None) for _ in range(min(2 * workers, len(batches))))
                if b is not None
            }
            while in_flight:
                if cancel_callback is not None and cancel_callback():
                    for fut in in_flight:
                        fut.cancel()
                    return False
                done, in_flight = wait(
                    in_flight, timeout=0.5, return_when=FIRST_COMPLETED
                )
                for fut in done:
                    on_result(fut.result())
                    nb = next(it, None)
                    if nb is not None:
                        in_flight.add(ex.submit(_compute, *nb))
        return True

    def _aggregate_in_processes(
        self,
        positions: np.ndarray,
        *,
        preps: dict,
        shared_xy: np.ndarray | None,
        gvi_radii: tuple[int, ...],
        ndvi_radii: tuple[int, ...],
        utm_crs: Any,
        point_xy_utm: np.ndarray,
        batches: list,
        workers: int,
        cancel_callback: Callable[[], bool] | None,
        on_result: Callable[[Any], None],
    ) -> bool:
        """Run the batches across worker processes. ``False`` if cancelled.

        Publishes every bulk array once into a scratch directory that the
        workers memory-map, so widening the pool costs no extra transfer and no
        extra copy of the metric in RAM. Falls back to the in-process path if a
        channel cannot be described to a worker.
        """
        import shutil
        import tempfile

        share_dir = tempfile.mkdtemp(prefix="preaggr-", dir=self.cache_dir)
        try:
            spec = _helpers.pool_spec(
                positions,
                preps=preps,
                shared_xy=shared_xy,
                gvi_radii=gvi_radii,
                ndvi_radii=ndvi_radii,
                utm_crs=utm_crs,
                point_xy_utm=point_xy_utm,
                share_dir=share_dir,
            )
            # Say which limit set the width, so a pool that is narrower than the
            # machine reads as a deliberate choice rather than a mystery.
            by_memory = parallel.memory_worker_cap()
            headroom = parallel.available_memory_bytes()
            limit = "cores"
            if by_memory is not None and by_memory <= workers:
                limit = f"free memory ({headroom / 2**30:.1f} GB)"
            elif workers >= len(batches):
                limit = "batches available"
            _log(
                "INFO",
                f"    aggregating {len(positions):,} pixel(s) across "
                f"{workers} worker process(es) — limited by {limit}.",
            )
            return parallel.map_batches(
                preaggregation.worker_aggregate,
                batches,
                workers=workers,
                initializer=preaggregation.worker_init,
                initargs=(spec,),
                on_result=on_result,
                cancel_check=cancel_callback,
            )
        finally:
            # A single-worker pool runs the initializer here rather than in a
            # child, so the scratch files stay mapped into *this* process until
            # that state is dropped — and Windows will not delete a mapped file.
            preaggregation.worker_release()
            shutil.rmtree(share_dir, ignore_errors=True)

    def _precompute_aggregations_longitudinal(
        self,
        progress_callback: Callable[[int, int], None] | None = None,
        cancel_callback: Callable[[], bool] | None = None,
        max_workers: int | None = None,
    ) -> bool:
        """Wave-aware variant of :meth:`precompute_aggregations`.

        The runner registers one metric file per channel per temporal key
        (measurement year, or wave label when waves are not date-derived).
        This method:

        - Groups temporal keys by which file they resolve to, so keys sharing
          a file share a single compute pass (a static channel that reuses one
          file across every key is computed once).
        - Registers key aliases on the cache so every aliased key's lookup
          resolves to the same representative-key storage. Stored rows stay
          keyed by ``(entity, temporal key)``; aliasing only decides which
          key's computation is reused, never which key's values are returned.
        - For each ``(channel, representative key)`` group, computes stats for
          the entities whose long-format rows fall in any key of that group,
          writes them to the cache, then releases that group's file before
          opening the next.
        """
        spec = self.longitudinal_spec
        assert spec is not None
        if self.target_gdf is None or len(self.target_gdf) == 0:
            raise ValueError(
                "Pre-aggregation requires sample points; call prepare_fusion_data() "
                "first."
            )
        provider = self._metric_sources
        if provider is None or not all(
            provider.has_channel(ch) for ch in longitudinal.GREENERY_CHANNELS
        ):
            raise ValueError(
                "Longitudinal pre-aggregation requires per-wave metric sources "
                "for every channel; call set_longitudinal_metric_sources() (or "
                "set_longitudinal_metric_sources()) first."
            )

        gvi_radii, ndvi_radii = self._preaggr_radii()
        # Unique-pixel source for point/line/polygon targets; entity rows
        # otherwise. Greenery is stored per pixel, keyed by a stable global id.
        preaggr_gdf = (
            self._preaggr_entity_gdf
            if self._preaggr_entity_gdf is not None
            else self.target_gdf
        )
        n_points = len(preaggr_gdf)

        # Reuse the grid's metric CRS (the pixels are already in it) so the
        # cache key is stable between the coverage probe and this build.
        utm_crs = _helpers.grid_metric_crs(preaggr_gdf, _log)
        crs_key = str(utm_crs.to_epsg() or utm_crs.to_wkt())
        cache = preaggregation.GreeneryCache(
            self.cache_dir,
            spacing_m=float(self.cgi_grid_spacing_m or 0.0),
            crs_key=crs_key,
            stats=preaggregation.STAT_COLUMNS,
        )

        # Entity geometry in the metric CRS; ``pid_to_pos`` maps a global pixel
        # id to its row so a unit's missing ids resolve to coordinates.
        entities_utm = preaggr_gdf.to_crs(utm_crs)
        all_points = bool((entities_utm.geometry.geom_type == "Point").all())
        entity_geoms_utm: list = list(entities_utm.geometry)
        if all_points:
            point_xy_utm = np.column_stack(
                [
                    entities_utm.geometry.x.to_numpy(),
                    entities_utm.geometry.y.to_numpy(),
                ]
            ).astype(np.float64)
        else:
            point_xy_utm = np.empty((0, 2), dtype=np.float64)
        if "_global_pid" in preaggr_gdf.columns:
            global_pids = np.asarray(
                preaggr_gdf["_global_pid"].to_numpy(), dtype=np.int64
            )
        else:
            global_pids = np.asarray(preaggr_gdf.index, dtype=np.int64)
        pid_to_pos = {int(p): i for i, p in enumerate(global_pids)}

        wave_index_of = {w: i for i, w in enumerate(spec.wave_labels)}

        # Global pixel ids referenced by each wave (via the duplicated catchment
        # rows for a gridded target; the entity's own id otherwise).
        tg = self.target_gdf
        tg_waves = tg["wave"].astype(str).to_numpy()
        if "_global_pid" in tg.columns:
            tg_pids = np.asarray(tg["_global_pid"].to_numpy(), dtype=np.int64)
        else:
            tg_pids = np.asarray(tg.index, dtype=np.int64)
        wave_to_pids = {
            w: np.unique(tg_pids[tg_waves == w]) for w in np.unique(tg_waves)
        }

        # Group waves by their greenery-file triple → one cache unit. Waves
        # resolving to the same files share a unit automatically (static-channel
        # dedup). Bind each wave to its unit for lookups.
        units: dict[str, dict] = {}
        for w in spec.wave_labels:
            cfg = cache.unit_key(
                provider.identity("veg", w),
                provider.identity("terrain", w),
                provider.identity("ndvi", w),
            )
            cache.bind_wave(wave_index_of[w], cfg)
            u = units.setdefault(cfg, {"waves": [], "rep": w})
            u["waves"].append(w)
        _log(
            "INFO",
            f"Longitudinal pre-aggregation: {n_points:,} pixels · "
            f"{len(spec.wave_labels)} waves → {len(units)} greenery unit(s) · "
            f"GVI radii {gvi_radii} m · NDVI radii {ndvi_radii} m · "
            f"stats {preaggregation.STAT_COLUMNS}. Cache: {cache.dir}",
        )

        # Open every unit: reusable files load into RAM, missing pixels counted.
        for cfg, u in units.items():
            parts = [wave_to_pids[w] for w in u["waves"] if w in wave_to_pids]
            required = (
                np.unique(np.concatenate(parts)) if parts else np.empty(0, np.int64)
            )
            eff_gvi, eff_ndvi, missing, absent_ch = cache.open_unit(
                cfg,
                gvi_radii=gvi_radii,
                ndvi_radii=ndvi_radii,
                required_ids=required,
                channels=self._cache_channels(),
            )
            if absent_ch:
                _log(
                    "INFO",
                    f"Cache extension for {cfg}: computing {list(absent_ch)}.",
                )
                missing = required
            u["eff_gvi"] = eff_gvi
            u["eff_ndvi"] = eff_ndvi
            u["missing"] = missing
        total_missing = int(sum(len(u["missing"]) for u in units.values()))
        n_reused = sum(1 for u in units.values() if len(u["missing"]) == 0)
        if progress_callback is not None:
            progress_callback(0, max(total_missing, 1))

        stage_secs = {"vector_agg": 0.0, "raster_agg": 0.0}
        build_t0 = time.perf_counter()
        processed = 0
        cancelled = False
        for cfg, u in units.items():
            provider.release()
            missing = u["missing"]
            if len(missing) == 0:
                continue  # fully served by a reusable cache file
            if cancel_callback is not None and cancel_callback():
                cancelled = True
                break
            rep = u["rep"]
            veg_src = provider.get("veg", rep)
            terrain_src = provider.get("terrain", rep)
            ndvi_src = provider.get("ndvi", rep)
            shared_gvi = provider.path("veg", rep) is not None and provider.path(
                "veg", rep
            ) == provider.path("terrain", rep)
            _log(
                "INFO",
                f"  unit {cfg[:8]} ({'/'.join(u['waves'])}): {len(missing):,} "
                f"pixel(s) to compute"
                + (" · veg+terrain share one read" if shared_gvi else "")
                + ".",
            )
            pos = np.fromiter(
                (pid_to_pos[int(p)] for p in missing),
                dtype=np.int64,
                count=len(missing),
            )
            unit_base = processed
            result = self._aggregate_pixel_channels(
                pos,
                veg_src=veg_src,
                terrain_src=terrain_src,
                ndvi_src=ndvi_src,
                shared_gvi=shared_gvi,
                gvi_radii=u["eff_gvi"],
                ndvi_radii=u["eff_ndvi"],
                utm_crs=utm_crs,
                all_points=all_points,
                point_xy_utm=point_xy_utm,
                entity_geoms_utm=entity_geoms_utm,
                max_workers=max_workers,
                cancel_callback=cancel_callback,
                stage_secs=stage_secs,
                progress_callback=(
                    None
                    if progress_callback is None
                    else lambda done: progress_callback(
                        min(unit_base + done, total_missing), max(total_missing, 1)
                    )
                ),
            )
            if result is None:
                cancelled = True
                break
            cache.commit_unit(cfg, missing, self._blocks(result))
            veg_arr, ter_arr, ndvi_arr, gvi_arr = result
            processed += len(missing)
            if progress_callback is not None:
                progress_callback(min(processed, total_missing), max(total_missing, 1))
            del veg_arr, ter_arr, ndvi_arr, gvi_arr, result
            provider.release()
            gc.collect()

        provider.release()
        if cancelled:
            _log("WARN", "Longitudinal pre-aggregation cancelled (resumable).")
            self._preaggregation_done = False
            return False

        self._preaggr_cache = cache
        self._preaggr_mem = None
        self._preaggregation_done = True
        self._build_pid_translation()
        total_build = time.perf_counter() - build_t0
        _log(
            "OK",
            f"Longitudinal pre-aggregation complete: {len(units)} greenery "
            f"unit(s) ({n_reused} reused, {len(units) - n_reused} built) in "
            f"{total_build:.0f}s — vector agg {stage_secs['vector_agg']:.0f}s, "
            f"raster agg {stage_secs['raster_agg']:.0f}s. "
            f"Resident {cache.n_bytes / 2**30:.2f} GB (float32).",
        )
        _log_parallel_efficiency(
            _log,
            "pre-aggregation",
            wall_s=total_build,
            busy_s=stage_secs["vector_agg"] + stage_secs["raster_agg"],
            workers=int(stage_secs.get("workers", 1)),
        )
        return True

    def _build_pid_translation(self) -> None:
        """Cache the ``_preaggr_id`` → ``_global_pid`` map for lookups.

        The greenery cache keys on the stable global pixel id, while scoring
        rows carry the frame-local ``_preaggr_id``; a small array indexed by the
        latter yields the former. ``None`` when the target has no grid ids (the
        cache then keys on the ids scoring already uses).
        """
        tg = self.target_gdf
        if (
            tg is None
            or "_global_pid" not in tg.columns
            or "_preaggr_id" not in tg.columns
        ):
            self._preaggr_pid_translate = None
            return
        pi = np.asarray(tg["_preaggr_id"].to_numpy(), dtype=np.int64)
        gp = np.asarray(tg["_global_pid"].to_numpy(), dtype=np.int64)
        if pi.size == 0:
            self._preaggr_pid_translate = None
            return
        m = np.full(int(pi.max()) + 1, -1, dtype=np.int64)
        m[pi] = gp
        self._preaggr_pid_translate = m

    def _ensure_memory_loaded(self):
        """Return the RAM-resident greenery store.

        The :class:`GreeneryCache` loads each unit into RAM as it is opened or
        built, so it is itself the resident structure — every trial's lookups
        are pure array gathers, shared read-only across worker threads. Returns
        ``None`` when no cache has been built yet.
        """
        if not self._preaggregation_done:
            return None
        return self._preaggr_cache

    def _lookup_preaggregation(
        self,
        point_indices: np.ndarray,
        channel: str,
        radius_m: float,
        stat: str,
        percentile: int | None,
        *,
        wave_indices: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Cache-backed lookup. Returns float32 array (or None to fall back).

        ``wave_indices`` (longitudinal mode only) is a per-row wave index of
        the same shape as ``point_indices``; rows are grouped by wave and
        looked up per-wave, then re-assembled in the input order. When
        ``None`` (cross-sectional mode) the cache returns the single
        implicit wave 0 for all entities.
        """
        if not self._preaggregation_done:
            return None
        # Prefer the RAM-resident structure (loaded at engine start); every
        # per-trial gather then avoids SQLite entirely. Falls back to the disk
        # cache only if the RAM load hasn't happened.
        source = self._ensure_memory_loaded() or self._preaggr_cache
        if source is None:
            return None
        column = preaggregation.stat_to_column(stat, percentile)
        if column is None:
            return None
        plan = self._preaggr_plan(source, point_indices, wave_indices)
        if plan is None:
            return None
        # ``None`` here means an off-grid (channel, wave, radius, column) cell —
        # the caller falls back to the ring/buffer aggregation path.
        return source.gather(plan, channel, int(round(radius_m)), column)

    def fit_bayesian_index(
        self,
        metric: str,
        *,
        forms: tuple[str, ...] = bayesian_index.FORMS,
        sweep_splits: int = 40,
        reps: int = 5,
        shuffles: int = 12,
        gain_splits: int = 20,
        gain_perm: int = 100,
        null_runs: int = 16,
        draws: int = 800,
        warmup: int = 800,
        chains: int = 4,
        seed: int = 42,
        cancel_callback: Callable[..., bool] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict:
        """Select the CGI configuration from data, then quantify it.

        Replaces the stability search. Stage one sweeps ``(radius, statistic)``
        per channel and the functional form, scoring every candidate on held-out
        rows of the train+val pool. Stage two fits a posterior over the channel
        weights at the winning columns. Neither stage touches the test split, so
        the headline effect stays out of sample.

        Returns a params dict the composite/apply path consumes, plus ``__``-
        prefixed diagnostics for the results bundle.
        """
        t0 = time.perf_counter()
        steps, done = 5, 0

        def tick():
            nonlocal done
            done += 1
            if progress_callback is not None:
                try:
                    progress_callback(done, steps)
                except Exception:
                    pass

        def cancelled() -> bool:
            return cancel_callback is not None and cancel_callback()

        X, radii, stats, tensor_channels, static = self.build_index_tensor("train_val")
        y = np.asarray(static["target"], dtype=np.float64)
        yr, Xr = bayesian_index.prep(X, y, static.get("cov"))
        tick()

        index_channels = list(cgi_formulas.formula_channels(self.cgi_formula))
        if self._active_greenery_channel != "cgi":
            index_channels = [self._active_greenery_channel]
        channel_index = [tensor_channels.index(c) for c in index_channels]

        # A channel is only offered the radii its own ladder was computed at.
        gvi_radii, ndvi_radii = self._preaggr_radii()
        radius_idx = [
            [i for i, r in enumerate(radii)
             if int(r) in (ndvi_radii if c == "ndvi" else gvi_radii)]
            for c in index_channels
        ]

        workers = parallel.process_worker_count()
        if cancelled():
            raise RuntimeError("Cancelled before the sweep.")
        res = bayesian_index.sweep(
            Xr, radii, stats, yr, channels=index_channels,
            channel_index=channel_index, radius_idx=radius_idx,
            forms=forms, splits=sweep_splits, seed=seed, workers=workers,
        )
        tick()
        _log(
            "INFO",
            f"Sweep: {res.picked} form={res.form} held-out |t|={res.score:.3f}"
            + (f"  BOUNDARY at {list(res.boundary_hit)}" if res.boundary_hit else ""),
        )

        disc = (
            bayesian_index.repeated_discovery(
                Xr, radii, stats, yr, channels=index_channels,
                channel_index=channel_index, radius_idx=radius_idx,
                forms=forms, reps=reps, shuffles=shuffles, seed=seed,
                workers=workers,
            )
            if reps and shuffles
            else {}
        )
        tick()

        gain = (
            bayesian_index.holdout_gain(
                Xr, yr, channels=index_channels, channel_index=channel_index,
                radius_idx=radius_idx, forms=forms, splits=gain_splits,
                perm=gain_perm, seed=seed, workers=workers,
            )
            if gain_splits and len(index_channels) > 1
            else {}
        )
        tick()

        E = Xr.reshape(len(Xr), -1)[:, list(res.columns)]
        mcmc = bayesian_index.fit(
            E, yr, form=res.form, draws=draws, warmup=warmup,
            chains=chains, seed=seed,
        )
        post = bayesian_index.posterior_from(
            mcmc, channels=index_channels, picked=res.picked, form=res.form
        )
        null = (
            bayesian_index.null_calibration(
                E, yr, form=res.form, n=null_runs, workers=workers
            )
            if null_runs
            else {}
        )
        tick()

        params = self._params_from_sweep(res, post, index_channels)
        params["__selection_method__"] = "bayesian_index"
        params["__sweep__"] = {
            "picked": [list(p) for p in res.picked],
            "form": res.form,
            "score": res.score,
            "form_scores": res.form_scores,
            "one_se_picked": [
                list(p) for p in bayesian_index._decode(
                    res.one_se_columns, len(radii), len(stats), radii, stats
                )
            ],
            "boundary_hit": list(res.boundary_hit),
            "n_candidates": len(res.combos),
            "distinct_split_winners": len(res.winner_counts),
            "splits": sweep_splits,
        }
        params["__posterior__"] = post.summary()
        params["__discovery__"] = disc
        params["__holdout_gain__"] = gain
        params["__null_calibration__"] = null
        params["__elapsed_s__"] = float(time.perf_counter() - t0)
        _log(
            "OK",
            f"Bayesian index fitted in {params['__elapsed_s__']:.0f}s "
            f"(R-hat {post.rhat_max:.3f}, ESS {post.ess_min:.0f}, "
            f"{post.divergences} divergences).",
        )
        return params

    def _params_from_sweep(self, res, post, index_channels) -> dict:
        """Sweep pick + posterior weights -> the params dict the engine uses."""
        formula = cgi_formulas.get_formula(self.cgi_formula)
        params: dict[str, Any] = {}

        stat_of = {"mean": ("mean", 50)}
        for (radius, column), ch in zip(res.picked, index_channels):
            stat, pct = stat_of.get(column, ("percentile", 0))
            if column != "mean":
                pct = int(column[1:])
            if ch == "ndvi":
                params["ndvi_radius"] = int(radius)
                params["ndvi_stat"] = stat
                params["ndvi_percentile"] = int(pct)
            else:
                params[f"{ch}_radius"] = int(radius)
                # veg and terrain share one street-view statistic in the engine.
                params["streetview_stat"] = stat
                params["streetview_percentile"] = int(pct)
        for key, default in (
            ("veg_radius", self.gvi_buffer_max_m),
            ("terrain_radius", self.gvi_buffer_max_m),
            ("ndvi_radius", self.ndvi_buffer_max_m),
        ):
            params.setdefault(key, int(round(float(default))))
        params.setdefault("streetview_stat", "mean")
        params.setdefault("streetview_percentile", 50)
        params.setdefault("ndvi_stat", "mean")
        params.setdefault("ndvi_percentile", 50)

        # Posterior-mean weights onto the formula's keys as integers summing to
        # 100, which is the scale every downstream consumer expects.
        w = np.asarray(post.weights).mean(0)
        keys = list(formula.weight_keys)
        vals = np.zeros(len(keys))
        for i, ch in enumerate(index_channels):
            key = cgi_formulas._CHANNEL_MAIN_KEY[self.cgi_formula].get(ch)
            if key in keys and i < len(w):
                vals[keys.index(key)] = float(w[i])
        total = vals.sum()
        if total <= 0:
            vals[:] = 1.0 / len(vals)
            total = 1.0
        scaled = vals * 100.0 / total
        floors = [int(v) for v in scaled]
        for i in sorted(range(len(scaled)), key=lambda i: scaled[i] - floors[i],
                        reverse=True)[: 100 - sum(floors)]:
            floors[i] += 1
        for key, v in zip(keys, floors):
            params[key] = int(v)
        if post.powers is not None:
            for ch, p in zip(index_channels, np.asarray(post.powers).mean(0)):
                params[f"{ch}_power"] = float(p)
        return params

    def build_index_tensor(self, subset: str = "train_val"):
        """Per-entity values for every (channel, radius, statistic) cell.

        The sweep scores candidates out of this one tensor, so the cache is read
        once per run instead of once per candidate. Shape is
        ``(n_entities, n_channels, n_radii, n_stats)`` with channels in
        :meth:`_cache_channels` order.

        The entity collapse depends on the catchment radius for point and line
        targets, so the mask is rebuilt per radius; polygon targets average
        every in-footprint pixel and the mask is constant.
        """
        if subset == "test":
            data = self.test_data
        elif subset == "train_val":
            data = self.train_val_data
        else:
            raise ValueError(f"subset must be 'train_val' or 'test'; got {subset!r}")
        if data is None or len(data) == 0:
            raise ValueError(f"No {subset} rows; call split_data() first.")

        channels = self._cache_channels()
        gvi_radii, ndvi_radii = self._preaggr_radii()
        # One ladder for the tensor: NDVI and the street-view family can be
        # configured separately, so the union is stored and each channel's
        # off-ladder cells stay NaN and are never selected.
        radii = tuple(sorted(set(gvi_radii) | set(ndvi_radii)))
        stats = tuple(preaggregation.STAT_COLUMNS)

        static = self._fold_entity_statics(data, f"tensor::{subset}", subset)
        n_ent = static["n_uniq"] if static["has_pid"] else len(static["points"])
        X = np.full((n_ent, len(channels), len(radii), len(stats)), np.nan)

        for ri, radius in enumerate(radii):
            mask = None
            if static["has_pid"]:
                mask = _helpers.entity_collapse_mask(data, float(radius))
            for ci, ch in enumerate(channels):
                ladder = ndvi_radii if ch == "ndvi" else gvi_radii
                if radius not in ladder:
                    continue
                for si, column in enumerate(stats):
                    stat, pct = (
                        ("mean", None) if column == "mean"
                        else ("percentile", int(column[1:]))
                    )
                    vals, on_uniq = self._channel_values_for_static(
                        static, ch, radius, stat, pct, subset
                    )
                    if vals is None:
                        continue
                    v = np.asarray(vals, dtype=np.float64)
                    if on_uniq and static.get("uniq_inverse") is not None:
                        v = v[static["uniq_inverse"]]
                    X[:, ci, ri, si] = (
                        self._collapse_mean_from_codes(
                            v, static["codes"], static["n_uniq"], mask
                        )
                        if static["has_pid"]
                        else v
                    )
        return X, np.asarray(radii, dtype=float), list(stats), list(channels), static

    def _channel_values_for_static(self, static, channel, radius, stat, pct, subset):
        """One (channel, radius, stat) column for a prepared fold, cache first.

        ``gvi`` has no source layer of its own — it is a stored cache channel —
        so a cache miss for it is fatal rather than falling back to on-the-fly
        aggregation over the wrong metric.
        """
        metric = {
            "veg": self.veg_data,
            "terrain": self.terrain_data,
            "ndvi": self.ndvi_data,
        }.get(channel)
        if metric is None:
            uniq = static.get("uniq_lookup_ids")
            source = self._ensure_memory_loaded() or self._preaggr_cache
            column = preaggregation.stat_to_column(stat, pct)
            if uniq is None or source is None or column is None:
                return None, False
            plan = self._preaggr_plan(
                source, uniq, static.get("uniq_wave_idx"), static
            )
            if plan is None:
                return None, False
            return source.gather(plan, channel, int(round(radius)), column), True
        return self._aggregate_channel_for_fold(
            static, metric, radius, stat, pct if pct is not None else 50,
            channel=channel, fold_idx=None, subset=subset,
        )

    def _cache_channels(self) -> tuple[str, ...]:
        """Greenery channels this job needs stored.

        A two-channel study reads ``gvi``, whose percentiles cannot be recovered
        by summing the components, so it must be stored in its own right. A
        three-channel study never reads it and does not pay for it. The cache
        keeps whatever any job has asked for, so switching a study from two
        channels to three appends rather than rebuilds.
        """
        wanted = set(cgi_formulas.formula_channels(self.cgi_formula))
        if self._active_greenery_channel != "cgi":
            wanted.add(self._active_greenery_channel)
        if "gvi" in wanted:
            # ``gvi``'s mean is the component sum, but its percentiles are not,
            # so the components stay alongside it for standalone reporting.
            wanted |= {"veg", "terrain"}
        return tuple(c for c in preaggregation.GreeneryCache.ALL_CHANNELS
                     if c in wanted)

    def _blocks(self, result: tuple) -> dict:
        """Aggregation output -> ``{channel: array}``, limited to this job's set.

        The aggregation always produces all four (one radius query serves the
        street-view family), but only the requested channels are stored.
        """
        veg, ter, ndvi, gvi = result
        allb = {"veg": veg, "terrain": ter, "ndvi": ndvi, "gvi": gvi}
        return {c: allb[c] for c in self._cache_channels()}

    def _preaggr_plan(
        self,
        source: "preaggregation.GreeneryCache",
        point_indices: np.ndarray,
        wave_indices: np.ndarray | None,
        store: dict | None = None,
    ) -> dict | None:
        """Where a fixed set of scoring rows lives in the greenery cache.

        Independent of anything a trial chooses, so ``store`` (a fold's static
        dict) holds it for the whole search: resolving it costs a binary search
        over every row, and repeating that per trial dominated the search.
        """
        if store is not None:
            cached = store.get("_preaggr_plan")
            if cached is not None:
                return cached
        # Scoring rows carry the frame-local ``_preaggr_id``; the greenery cache
        # keys on the stable global pixel id, so translate before resolving.
        ids = np.asarray(point_indices, dtype=np.int64)
        tr = self._preaggr_pid_translate
        if tr is not None:
            safe = (ids >= 0) & (ids < tr.shape[0])
            ids = np.where(safe, tr[np.clip(ids, 0, tr.shape[0] - 1)], ids)
        plan = source.build_plan(ids, wave_indices)
        if store is not None and plan is not None:
            store["_preaggr_plan"] = plan
        return plan

    # ────────────────────────────────────────────────────────────
    # Polygon-target areal aggregation
    # ────────────────────────────────────────────────────────────

    def _prepare_polygon_fusion(self) -> pd.DataFrame:
        """Build a per-pixel dataset for polygon targets.

        Generates a regular grid of pixel centroids at
        ``self.cgi_grid_spacing_m`` spacing covering the union of the
        target polygons, tags each pixel with the polygon it falls inside
        (first-match for overlapping polygons), and returns one row per
        pixel. Each pixel carries the parent polygon's outcome,
        covariates, and longitudinal extras. Downstream:

        - ``precompute_aggregations`` runs the point-fast path because
          every entity is a Point.
        - ``_objective`` / ``evaluate_on_test`` compute per-pixel CGI and
          collapse to per-polygon via ``groupby("polygon_id").mean()``.
        """

        _log("INFO", "====== POLYGON FUSION (per-pixel CGI) ======")
        if self.target_polygons_gdf is None:
            self.target_polygons_gdf = self.target_gdf.copy()
        polygons = self.target_polygons_gdf

        outcome_col = self.target_feature
        spacing_m = float(self.cgi_grid_spacing_m)
        _log(
            "INFO",
            f"Target polygons: {len(polygons)} · CGI grid pixel size: "
            f"{spacing_m:g} m",
        )

        # Drop NaN-outcome polygons up front so the grid never includes
        # pixels for entities that would be filtered later.
        valid_mask = polygons[outcome_col].notna()
        if self.covariate_columns:
            for col in self.covariate_columns:
                valid_mask &= polygons[col].notna()
        skipped_nan = int((~valid_mask).sum())
        if skipped_nan:
            _log(
                "WARN",
                f"Skipped {skipped_nan} polygon(s) with NaN outcome / covariate.",
            )
        polygons = polygons.loc[valid_mask].reset_index(drop=True)
        if len(polygons) < 2:
            raise ValueError(
                "Polygon fusion needs ≥2 polygons with a valid outcome "
                f"(and covariates if configured); got {len(polygons)}."
            )

        # Distance math (grid step) is metres, so move the polygons into a
        # projected CRS chosen for low distortion across the study extent.
        grid_crs, _grid_distortion, _grid_name = select_grid_crs_with_warning(
            polygons, _log, role="CGI grid CRS"
        )
        polygons_grid = polygons.to_crs(grid_crs)
        polygons_grid["__polygon_id"] = np.arange(len(polygons_grid), dtype=np.int64)
        # Per-polygon area lookup for the area-balanced split.
        self._polygon_id_to_area_km2 = {
            int(pid): float(area)
            for pid, area in zip(
                polygons_grid["__polygon_id"].to_numpy(),
                polygons_grid.geometry.area.to_numpy() / 1_000_000.0,
            )
        }
        union_geom = polygons_grid.geometry.union_all()
        minx, miny, maxx, maxy = union_geom.bounds

        # Sparse lattice: generate pixel centres only inside the polygons'
        # connected components, on one global lattice anchored at the grid
        # bounding box so ``(_grid_row, _grid_col)`` stay consistent across
        # components. The empty space between dispersed polygons is never
        # materialised as geometry.
        import shapely as _shapely
        from shapely.geometry import Polygon as _Polygon

        n_cols = int(np.floor((maxx - minx) / spacing_m))
        n_rows = int(np.floor((maxy - miny) / spacing_m))
        if n_cols <= 0 or n_rows <= 0:
            raise ValueError(
                f"CGI grid spacing {spacing_m:g} m is larger than the target "
                f"extent ({maxx - minx:.0f} m × {maxy - miny:.0f} m)."
            )
        cluster_polys = (
            [union_geom]
            if isinstance(union_geom, _Polygon)
            else [
                g for g in getattr(union_geom, "geoms", []) if isinstance(g, _Polygon)
            ]
        )
        row_parts: list[np.ndarray] = []
        col_parts: list[np.ndarray] = []
        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for poly in cluster_polys:
            if poly.is_empty:
                continue
            cminx, cminy, cmaxx, cmaxy = poly.bounds
            col_lo = max(0, int(np.floor((cminx - minx) / spacing_m)))
            col_hi = min(n_cols, int(np.ceil((cmaxx - minx) / spacing_m)))
            # Rows run top-down so the composite writer can index pixels back
            # to (row, col) without flipping.
            row_lo = max(0, int(np.floor((maxy - cmaxy) / spacing_m)))
            row_hi = min(n_rows, int(np.ceil((maxy - cminy) / spacing_m)))
            if col_hi <= col_lo or row_hi <= row_lo:
                continue
            cg, rg = np.meshgrid(
                np.arange(col_lo, col_hi, dtype=np.int64),
                np.arange(row_lo, row_hi, dtype=np.int64),
                indexing="xy",
            )
            cg = cg.ravel()
            rg = rg.ravel()
            cx = minx + (cg + 0.5) * spacing_m
            cy = maxy - (rg + 0.5) * spacing_m
            inside = _shapely.contains_xy(poly, cx, cy)
            if not inside.any():
                continue
            col_parts.append(cg[inside])
            row_parts.append(rg[inside])
            x_parts.append(cx[inside])
            y_parts.append(cy[inside])

        if not x_parts:
            raise ValueError(
                "CGI grid produced zero in-polygon pixels — spacing may exceed "
                "polygon size, or the union extent has no interior points."
            )
        g_col = np.concatenate(col_parts)
        g_row = np.concatenate(row_parts)
        g_x = np.concatenate(x_parts)
        g_y = np.concatenate(y_parts)
        pixels = gpd.GeoDataFrame(
            {
                "_grid_row": g_row.astype(np.int32),
                "_grid_col": g_col.astype(np.int32),
            },
            geometry=gpd.points_from_xy(g_x, g_y),
            crs=grid_crs,
        )
        _log(
            "INFO",
            f"CGI grid span {n_rows} × {n_cols} · {len(cluster_polys)} cluster(s) "
            f"· {len(pixels):,} in-union candidate pixels.",
        )

        # Pixel → polygon assignment via sjoin(within). For overlapping
        # polygons each pixel falls in multiple — keep the first match per
        # pixel (decided in the plan; census-tract use case has no overlap).
        joined = gpd.sjoin(
            pixels[["geometry", "_grid_row", "_grid_col"]],
            polygons_grid[["__polygon_id", "geometry"]],
            how="inner",
            predicate="within",
        )
        joined = joined[~joined.index.duplicated(keep="first")]
        if len(joined) == 0:
            raise ValueError(
                "CGI grid produced zero in-polygon pixels — spacing may exceed "
                "polygon size, or the union extent has no interior points."
            )

        # Carry outcome + covariates + longitudinal extras onto each pixel
        # by indexing back into the (NaN-filtered) polygon table.
        poly_attr_cols = [outcome_col, *self.covariate_columns]
        if self.is_longitudinal:
            poly_attr_cols.extend(self._longitudinal_extra_cols())
        # Drop duplicates conservatively so covariate / longitudinal aliases
        # to outcome don't collide.
        poly_attr_cols = list(dict.fromkeys(poly_attr_cols))
        attr_lookup = polygons_grid.set_index("__polygon_id")[poly_attr_cols]
        attrs = attr_lookup.loc[joined["__polygon_id"].values].reset_index(drop=True)

        polygon_ids = joined["__polygon_id"].values.astype(np.int64)
        if self.is_longitudinal:
            entity_ids = attrs["entity_id"].astype(str).to_numpy()
            wave_labels = attrs["wave"].astype(str).to_numpy()
            collapse_keys = np.array(
                [f"{e}|{w}" for e, w in zip(entity_ids, wave_labels)]
            )
        else:
            collapse_keys = polygon_ids

        entity_gdf = gpd.GeoDataFrame(
            {
                "polygon_id": collapse_keys,
                "target": attrs[outcome_col].to_numpy(),
                "_grid_row": joined["_grid_row"].to_numpy(),
                "_grid_col": joined["_grid_col"].to_numpy(),
            },
            geometry=joined.geometry.values,
            crs=grid_crs,
        ).reset_index(drop=True)
        for col in self.covariate_columns:
            entity_gdf[col] = attrs[col].to_numpy()
        for col in self._longitudinal_extra_cols():
            if col in attrs.columns:
                entity_gdf[col] = attrs[col].to_numpy()

        # Stash the grid metadata so the composite-map writer can build the
        # output raster directly from the pixel grid (no re-sampling step).
        self._polygon_grid_crs = grid_crs
        self._polygon_grid_transform = from_origin(minx, maxy, spacing_m, spacing_m)
        self._polygon_grid_shape = (n_rows, n_cols)

        _log(
            "INFO",
            f"Per-pixel entities: {len(entity_gdf):,} "
            f"(unique polygon_id keys: {pd.Series(collapse_keys).nunique()}).",
        )

        # Re-bind target_gdf to the per-pixel view. Pixel geometries are
        # Points, so the pre-aggregation cache takes the point-fast path
        # (BallTree / circular raster mask) automatically.
        self.target_gdf = entity_gdf.copy()

        # Coverage probe: nearest-feature sample at each pixel using the
        # channel's max radius. The values feed only the NaN-filtering dropna
        # gate; per-trial channel values come from the cache. Rows map 1:1 onto
        # the probe frame here (one row per polygon pixel already).
        probe_cols = self._probe_channel_coverage(
            entity_gdf[["geometry"]].copy(),
            np.arange(len(entity_gdf), dtype=np.int64),
            entity_gdf,
        )

        fusion_df = pd.DataFrame(
            {
                "polygon_id": entity_gdf["polygon_id"].values,
                "target": entity_gdf["target"].values,
                # float32 halves the pixel-frame cost.
                "veg": probe_cols["veg"],
                "terrain": probe_cols["terrain"],
                "ndvi": probe_cols["ndvi"],
            },
            index=entity_gdf.index,
        )
        for col in self.covariate_columns:
            fusion_df[col] = entity_gdf[col].values
        for col in self._longitudinal_extra_cols():
            if col in entity_gdf.columns:
                fusion_df[col] = entity_gdf[col].values
        fusion_df = self._attach_entity_coords(fusion_df, entity_gdf)

        _log("INFO", "====== DATA QUALITY SUMMARY (POLYGON / PER-PIXEL) ======")
        _log(
            "INFO",
            f"Pixel rows: {len(fusion_df):,} "
            f"({fusion_df['polygon_id'].nunique()} unique polygon_id keys)",
        )
        for col in ["target", "veg", "terrain", "ndvi", *self.covariate_columns]:
            nan_count = fusion_df[col].isna().sum()
            pct = nan_count / max(len(fusion_df), 1) * 100
            _log("INFO", f"{col} NaN: {nan_count} ({pct:.1f}%)")

        result = fusion_df.dropna(
            subset=["target", "veg", "terrain", "ndvi", *self.covariate_columns]
        )
        polygons_left = result["polygon_id"].nunique() if len(result) else 0
        _log(
            "OK" if polygons_left else "WARN",
            f"After dropna: {len(result):,} pixel rows "
            f"({polygons_left} unique polygon_id keys)",
        )
        if polygons_left < 2:
            raise ValueError(
                "Polygon fusion needs ≥2 polygons with valid coverage after NaN "
                f"filtering; got {polygons_left}. Check metric coverage or "
                "increase the buffer."
            )
        # Trim target_gdf to the surviving pixels so the cache + split
        # only see entities that passed the coverage probe.
        self.target_gdf = entity_gdf.loc[result.index].copy()
        return result

    def _prepare_entity_fusion(self) -> pd.DataFrame:
        """Per-pixel CGI dataset for point / line targets.

        Mirrors :meth:`_prepare_polygon_fusion` so every geometry type scores
        the same way: build a regular pixel grid from the pre-aggregation
        cache, evaluate per-pixel CGI, and aggregate the pixels in each
        entity's catchment to the entity value. Points and lines have no
        footprint, so each entity's catchment is the pixels within its
        **buffer**; the per-trial collapse
        (:meth:`_entity_collapse_mask`) later restricts that buffer to the
        trial's largest channel radius.

        Returns one row per ``(entity, catchment-pixel)``, keyed by
        ``polygon_id`` (= the entity/observation index) and carrying
        ``_catchment_dist`` (pixel→entity distance, metres) and
        ``_is_nearest`` (the entity's closest surviving pixel, always kept so
        no entity drops out at small radii). Shared pixels between nearby
        entities are duplicated per entity, which keeps fixed ``polygon_id``
        membership so :meth:`split_data` and the in-bag/OOB scaler work
        exactly as in the polygon path.
        """
        _log("INFO", "====== POINT/LINE FUSION (per-pixel CGI) ======")
        entities = self.target_gdf.copy()
        outcome_col = self.target_feature
        spacing_m = float(self.cgi_grid_spacing_m)

        # Drop NaN-outcome / NaN-covariate entities up front.
        valid_mask = entities[outcome_col].notna()
        if self.covariate_columns:
            for col in self.covariate_columns:
                valid_mask &= entities[col].notna()
        skipped_nan = int((~valid_mask).sum())
        if skipped_nan:
            _log(
                "WARN",
                f"Skipped {skipped_nan} entity(ies) with NaN outcome / covariate.",
            )
        entities = entities.loc[valid_mask].reset_index(drop=True)
        if len(entities) < 2:
            raise ValueError(
                "Point/line fusion needs ≥2 entities with a valid outcome "
                f"(and covariates if configured); got {len(entities)}."
            )

        # Project to a low-distortion CRS so the grid step + buffers are metres.
        grid_crs, _grid_distortion, _grid_name = select_grid_crs_with_warning(
            entities, _log, role="CGI grid CRS"
        )
        ent = entities.to_crs(grid_crs)
        ent["__entity_id"] = np.arange(len(ent), dtype=np.int64)

        # Catchment cap = the largest radius any trial can request. Each
        # trial's collapse then masks down to its own max channel radius.
        r_max = float(max(self.gvi_buffer_max_m, self.ndvi_buffer_max_m))
        buffers = ent.geometry.buffer(r_max)

        # Cluster-aware sparse grid: generate pixel centres only inside the
        # buffered catchments. The buffered union is split into connected
        # components and each component is gridded on a single global lattice
        # anchored at the CRS origin, so ``(_grid_row, _grid_col)`` stay
        # consistent across components and the empty space between dispersed
        # entities is never materialised.
        import shapely
        from shapely.geometry import Polygon

        union_geom = buffers.union_all()
        if union_geom.is_empty:
            raise ValueError(
                "Buffered target catchments are empty — check the target "
                "geometry and buffer settings."
            )
        cluster_polys = (
            [union_geom]
            if isinstance(union_geom, Polygon)
            else [g for g in getattr(union_geom, "geoms", []) if isinstance(g, Polygon)]
        )

        col_parts: list[np.ndarray] = []
        row_parts: list[np.ndarray] = []
        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for poly in cluster_polys:
            if poly.is_empty:
                continue
            cminx, cminy, cmaxx, cmaxy = poly.bounds
            # Global lattice indices (anchor = CRS origin); rows increase south.
            col_lo = int(np.floor(cminx / spacing_m))
            col_hi = int(np.ceil(cmaxx / spacing_m))
            row_lo = int(np.floor(-cmaxy / spacing_m))
            row_hi = int(np.ceil(-cminy / spacing_m))
            if col_hi <= col_lo or row_hi <= row_lo:
                continue
            cg, rg = np.meshgrid(
                np.arange(col_lo, col_hi, dtype=np.int64),
                np.arange(row_lo, row_hi, dtype=np.int64),
                indexing="xy",
            )
            cg = cg.ravel()
            rg = rg.ravel()
            cx = (cg + 0.5) * spacing_m
            cy = -(rg + 0.5) * spacing_m
            inside = shapely.contains_xy(poly, cx, cy)
            if not inside.any():
                continue
            col_parts.append(cg[inside])
            row_parts.append(rg[inside])
            x_parts.append(cx[inside])
            y_parts.append(cy[inside])

        if not x_parts:
            raise ValueError(
                f"CGI grid spacing {spacing_m:g} m produced no pixels inside the "
                f"buffered target extent — lower the grid spacing or raise the buffer."
            )

        g_col = np.concatenate(col_parts)
        g_row = np.concatenate(row_parts)
        g_x = np.concatenate(x_parts)
        g_y = np.concatenate(y_parts)

        # Stable global pixel id: the pixel's global lattice (row, col) packed
        # into one int64. Anchored at the CRS origin, so the same physical
        # pixel gets the same id in any job — this is what the greenery cache
        # keys on, making it reusable across targets/outcomes/covariates.
        g_pid = (g_row.astype(np.int64) << np.int64(32)) | (
            g_col.astype(np.int64) & np.int64(0xFFFFFFFF)
        )

        # Local raster indices span only the populated cells' global extent.
        col_origin = int(g_col.min())
        row_origin = int(g_row.min())
        n_cols = int(g_col.max()) - col_origin + 1
        n_rows = int(g_row.max()) - row_origin + 1
        minx = col_origin * spacing_m
        maxy = -row_origin * spacing_m
        pixels = gpd.GeoDataFrame(
            {
                "_grid_row": (g_row - row_origin).astype(np.int32),
                "_grid_col": (g_col - col_origin).astype(np.int32),
                "_global_pid": g_pid,
            },
            geometry=gpd.points_from_xy(g_x, g_y),
            crs=grid_crs,
        )
        _log(
            "INFO",
            f"Entities: {len(ent)} · catchment cap: {r_max:g} m · CGI grid "
            f"pixel size: {spacing_m:g} m · {len(cluster_polys)} cluster(s) · "
            f"{len(pixels):,} in-buffer pixels · grid span {n_rows}×{n_cols}.",
        )

        # Assign every pixel to each entity whose catchment contains it
        # (shared pixels are duplicated per entity → fixed polygon_id
        # membership). For point entities the catchment is a disc, so the
        # assignment is a radius neighbour query on the entity coordinates —
        # far cheaper than a polygon ``within`` join plus a separate distance
        # pass. Line entities keep the buffer-polygon join (their catchment
        # distance is perpendicular, not point-to-point).
        if bool((ent.geom_type == "Point").all()):
            from scipy.spatial import cKDTree

            ent_xy = np.column_stack(
                [ent.geometry.x.to_numpy(), ent.geometry.y.to_numpy()]
            )
            pix_xy = np.column_stack([g_x, g_y])
            sdm = cKDTree(pix_xy).sparse_distance_matrix(
                cKDTree(ent_xy), r_max, output_type="coo_matrix"
            )
            pix_idx = sdm.row.astype(np.int64)
            ent_idx = sdm.col.astype(np.int64)
            catchment_dist = sdm.data.astype(np.float32)
        else:
            ent_buffers = gpd.GeoDataFrame(
                {"__entity_id": ent["__entity_id"].to_numpy()},
                geometry=buffers.values,
                crs=grid_crs,
            )
            joined = gpd.sjoin(pixels, ent_buffers, how="inner", predicate="within")
            pix_idx = joined.index.to_numpy()
            ent_idx = joined["__entity_id"].to_numpy()
            ent_geom_by_id = ent.set_index("__entity_id").geometry
            aligned_geoms = gpd.GeoSeries(
                ent_geom_by_id.loc[ent_idx].to_numpy(), crs=grid_crs
            )
            pix_geoms = gpd.GeoSeries(joined.geometry.to_numpy(), crs=grid_crs)
            catchment_dist = pix_geoms.distance(aligned_geoms).to_numpy(
                dtype=np.float32
            )

        if pix_idx.size == 0:
            raise ValueError(
                "Entity catchment grid produced zero pixels — the grid spacing "
                "may exceed the buffer extent."
            )

        # Per-unique-pixel columns, indexed onto the duplicated catchment rows.
        pixel_geom = pixels.geometry.to_numpy()
        pixel_row = pixels["_grid_row"].to_numpy()
        pixel_col = pixels["_grid_col"].to_numpy()
        pixel_pid = pixels["_global_pid"].to_numpy()

        # ``polygon_id`` = the entity/observation index; each (entity, wave)
        # row in longitudinal mode is already its own observation, so the
        # row index doubles as the collapse key.
        target_vals = ent[outcome_col].to_numpy()
        entity_gdf = gpd.GeoDataFrame(
            {
                "polygon_id": ent_idx,
                "target": target_vals[ent_idx],
                "_grid_row": pixel_row[pix_idx],
                "_grid_col": pixel_col[pix_idx],
                "_catchment_dist": catchment_dist,
                # Unique-pixel id this duplicated catchment row maps to. The
                # pre-aggregation cache stores one row per pixel; scoring rows
                # resolve their pixel's stats through this id at lookup time.
                "_preaggr_id": pix_idx.astype(np.int64),
                # Stable global pixel id (grid row/col packed); the greenery
                # cache keys on this so it is reusable across jobs.
                "_global_pid": pixel_pid[pix_idx],
            },
            geometry=pixel_geom[pix_idx],
            crs=grid_crs,
        ).reset_index(drop=True)
        for col in self.covariate_columns:
            entity_gdf[col] = ent[col].to_numpy()[ent_idx]
        for col in self._longitudinal_extra_cols():
            if col in ent.columns:
                entity_gdf[col] = ent[col].to_numpy()[ent_idx]

        # Grid metadata for the composite-map writer (the CGI field raster).
        self._entity_grid_crs = grid_crs
        self._entity_grid_transform = from_origin(minx, maxy, spacing_m, spacing_m)
        self._entity_grid_shape = (n_rows, n_cols)

        self.target_gdf = entity_gdf.copy()

        # Coverage probe — used only for NaN-filtering and the channel-scale
        # bounds; per-trial channel values come from the cache.
        _probe_t0 = time.perf_counter()
        probe_cols = self._probe_channel_coverage(pixels, pix_idx, entity_gdf)
        _log(
            "INFO",
            f"Coverage probe: {len(entity_gdf):,} pixel rows in "
            f"{time.perf_counter() - _probe_t0:.0f}s.",
        )

        fusion_df = pd.DataFrame(
            {
                "polygon_id": entity_gdf["polygon_id"].values,
                "target": entity_gdf["target"].values,
                # float32 halves the pixel-frame cost.
                "veg": probe_cols["veg"],
                "terrain": probe_cols["terrain"],
                "ndvi": probe_cols["ndvi"],
                "_catchment_dist": entity_gdf["_catchment_dist"].values,
            },
            index=entity_gdf.index,
        )
        for col in self.covariate_columns:
            fusion_df[col] = entity_gdf[col].values
        for col in self._longitudinal_extra_cols():
            if col in entity_gdf.columns:
                fusion_df[col] = entity_gdf[col].values
        fusion_df = self._attach_entity_coords(fusion_df, entity_gdf)

        _log("INFO", "====== DATA QUALITY SUMMARY (POINT/LINE / PER-PIXEL) ======")
        for col in ["target", "veg", "terrain", "ndvi", *self.covariate_columns]:
            nan_count = int(fusion_df[col].isna().sum())
            pct = nan_count / max(len(fusion_df), 1) * 100
            _log("INFO", f"{col} NaN: {nan_count} ({pct:.1f}%)")

        result = fusion_df.dropna(
            subset=["target", "veg", "terrain", "ndvi", *self.covariate_columns]
        )
        entities_left = result["polygon_id"].nunique() if len(result) else 0
        _log(
            "OK" if entities_left else "WARN",
            f"After dropna: {len(result):,} pixel rows "
            f"({entities_left} unique entities)",
        )
        if entities_left < 2:
            raise ValueError(
                "Point/line fusion needs ≥2 entities with valid coverage after "
                f"NaN filtering; got {entities_left}. Check metric coverage or "
                "increase the buffer."
            )

        # Always keep each entity's nearest surviving pixel so no entity drops
        # out when a trial's catchment radius is smaller than its closest
        # pixel — computed on the surviving rows so the flag stays valid.
        nearest_idx = result.groupby("polygon_id")["_catchment_dist"].idxmin()
        result = result.copy()
        result["_is_nearest"] = False
        result.loc[nearest_idx, "_is_nearest"] = True

        # Trim target_gdf to surviving pixels and carry the nearest flag.
        self.target_gdf = entity_gdf.loc[result.index].copy()
        self.target_gdf["_is_nearest"] = result["_is_nearest"].to_numpy()

        # Pre-aggregation source: the unique pixels still referenced after the
        # coverage dropna, keyed by ``_preaggr_id``. The cache iterates these
        # (~1/16th of the duplicated catchment rows for dense point targets);
        # scoring rows resolve their stats through ``_preaggr_id`` →
        # ``_global_pid``.
        referenced = np.unique(self.target_gdf["_preaggr_id"].to_numpy())
        self._preaggr_entity_gdf = pixels.loc[referenced].copy()
        _log(
            "INFO",
            f"Pre-aggregation source: {len(self._preaggr_entity_gdf):,} unique "
            f"pixels (vs {len(self.target_gdf):,} catchment rows; "
            f"{len(self.target_gdf) / max(len(self._preaggr_entity_gdf), 1):.1f}× "
            "dedup).",
        )
        return result

    # ────────────────────────────────────────────────────────────
    # Per-pixel CGI → per-entity collapse (shared by every scoring path)
    # ────────────────────────────────────────────────────────────
    @staticmethod
    def _collapse_to_entities(
        values: np.ndarray,
        polygon_id: np.ndarray,
        mask: np.ndarray | None = None,
        how: str = "mean",
    ) -> np.ndarray:
        """Collapse per-pixel ``values`` to one value per ``polygon_id``.

        ``how="mean"`` for the composite, ``"first"`` for per-entity constants
        (target / covariates / entity_id / years_since_baseline). When
        ``mask`` is given only the masked rows contribute; the
        nearest-pixel guarantee keeps every entity present, so the grouped
        key order matches across every collapsed array (pandas sorts the
        group keys).
        """
        v = np.asarray(values)
        pid = np.asarray(polygon_id)
        if mask is not None:
            v = v[mask]
            pid = pid[mask]
        if how == "mean":
            # factorize(sort=True) reproduces the sorted-key order pandas
            # groupby uses, so every collapsed array stays row-aligned, while
            # bincount avoids building a Series + GroupBy on the hot path.
            codes, uniq = pd.factorize(pid, sort=True)
            vv = np.asarray(v, dtype=np.float64)
            valid = ~np.isnan(vv)
            n = len(uniq)
            counts = np.bincount(codes[valid], minlength=n)
            sums = np.bincount(codes[valid], weights=vv[valid], minlength=n)
            out = np.full(n, np.nan, dtype=np.float64)
            nz = counts > 0
            out[nz] = sums[nz] / counts[nz]
            return out
        return pd.Series(v).groupby(pid).first().to_numpy()

    @staticmethod
    def _collapse_mean_from_codes(
        values: np.ndarray,
        codes: np.ndarray,
        n_uniq: int,
        mask: np.ndarray | None,
    ) -> np.ndarray:
        """Per-entity mean of ``values`` from precomputed collapse codes.

        Equivalent to ``_collapse_to_entities(values, pid, mask, "mean")``
        without the per-call ``factorize``: ``codes`` / ``n_uniq`` come from
        one sorted-key factorization of the fold's ``polygon_id``. The
        ``minlength`` pin keeps the output aligned to the full entity set, so
        the row order matches every other collapsed array even under a
        catchment mask (the nearest-pixel guarantee keeps each entity
        represented, exactly as the legacy masked factorize relied on).
        """
        v = np.asarray(values, dtype=np.float64)
        cd = codes
        if mask is not None:
            v = v[mask]
            cd = cd[mask]
        valid = ~np.isnan(v)
        counts = np.bincount(cd[valid], minlength=n_uniq)
        sums = np.bincount(cd[valid], weights=v[valid], minlength=n_uniq)
        out = np.full(n_uniq, np.nan, dtype=np.float64)
        nz = counts > 0
        out[nz] = sums[nz] / counts[nz]
        return out

    def _fold_entity_statics(
        self, data: pd.DataFrame, fold_idx: Any, subset: str
    ) -> dict:
        """Trial-invariant arrays for one scored fold subset, computed once.

        Everything here depends only on the fold's row set — never on trial
        parameters — so every trial of a bootstrap study reuses one
        computation: the sorted-key collapse codes of ``polygon_id``, each
        entity's first-occurrence row (target / covariates / coords /
        longitudinal keys are per-entity constants broadcast to pixels, so a
        "first" collapse is a plain row-take), and the points slice the
        aggregation path consumes. Cached on ``_fold_static_cache`` and
        cleared with the ring caches — both key off the live fold row sets.
        """
        key = (self._split_generation, fold_idx, subset, len(data))
        st = self._fold_static_cache.get(key)
        if st is not None:
            return st

        st = {}
        cov_cols = self.covariate_columns
        has_pid = "polygon_id" in data.columns
        have_coords = "_cx" in data.columns and "_cy" in data.columns
        st["has_pid"] = has_pid
        if has_pid:
            pid = data["polygon_id"].values
            codes, uniq = pd.factorize(pid, sort=True)
            codes = np.asarray(codes, dtype=np.int64)
            n_uniq = len(uniq)
            order = np.argsort(codes, kind="stable")
            first_idx = order[np.searchsorted(codes[order], np.arange(n_uniq))]
            st["codes"] = codes
            st["n_uniq"] = n_uniq
            st["target"] = np.asarray(data["target"].values, dtype=np.float64)[
                first_idx
            ]
            st["cov"] = (
                data[cov_cols].to_numpy(dtype=np.float64)[first_idx]
                if cov_cols
                else None
            )
            st["coords"] = (
                np.column_stack(
                    [
                        data["_cx"].to_numpy(np.float64)[first_idx],
                        data["_cy"].to_numpy(np.float64)[first_idx],
                    ]
                )
                if have_coords
                else None
            )
            if self.is_longitudinal:
                st["entity_id"] = np.asarray(data["entity_id"].values)[first_idx]
                st["ysb"] = np.asarray(
                    data["years_since_baseline"].values, dtype=np.float64
                )[first_idx]
                if "wave" in data.columns:
                    st["wave"] = np.asarray(data["wave"].values)[first_idx]
        else:
            st["target"] = data["target"].values
            st["cov"] = data[cov_cols].to_numpy(dtype=np.float64) if cov_cols else None
            st["coords"] = (
                data[["_cx", "_cy"]].to_numpy(np.float64) if have_coords else None
            )
            if self.is_longitudinal:
                st["entity_id"] = data["entity_id"].values
                st["ysb"] = data["years_since_baseline"].values
                if "wave" in data.columns:
                    st["wave"] = np.asarray(data["wave"].values)

        # Aggregation input: geometry-free on the pre-aggregation fast path
        # (the cache needs only row ids + wave), full geometry rows otherwise.
        if getattr(self, "_preaggregation_done", False):
            light_cols = [
                c for c in ("wave", "_preaggr_id") if c in self.target_gdf.columns
            ]
            pts = self.target_gdf.loc[data.index, light_cols]
        else:
            pts = self.target_gdf.loc[data.index]
        st["points"] = pts

        # Cache-lookup ids for this fold, materialized once. The cache keys its
        # row indexer on this array's identity, so handing back the same object
        # every trial turns the lookup's position search into a dict hit.
        if "_preaggr_id" in pts.columns:
            lookup_ids = np.ascontiguousarray(
                pts["_preaggr_id"].to_numpy(), dtype=np.int64
            )
        else:
            lookup_ids = np.ascontiguousarray(pts.index.to_numpy(), dtype=np.int64)
        st["lookup_ids"] = lookup_ids
        pts.attrs["_gf_lookup_ids"] = lookup_ids

        # Per-row wave index (longitudinal only) — trial-invariant per fold.
        wave_idx = None
        if self.is_longitudinal and "wave" in pts.columns:
            spec = self.longitudinal_spec
            assert spec is not None
            wave_index_of = {w: i for i, w in enumerate(spec.wave_labels)}
            wave_idx = np.asarray(
                [wave_index_of[str(w)] for w in pts["wave"]], dtype=np.int64
            )
        st["wave_idx"] = wave_idx
        pts.attrs["_gf_wave_idx"] = wave_idx

        # Dedup axis for the per-trial channel arithmetic. Point / line targets
        # repeat a catchment pixel once per overlapping entity, and a pixel's
        # cached stats are identical across those repeats, so the lookup and the
        # composite run over the distinct (pixel, wave) pairs and are gathered
        # back onto rows just before the entity collapse.
        st["uniq_lookup_ids"] = None
        st["uniq_wave_idx"] = None
        st["uniq_inverse"] = None
        if "_preaggr_id" in pts.columns:
            if wave_idx is not None:
                span = int(wave_idx.max()) + 1 if wave_idx.size else 1
                keys = lookup_ids * span + wave_idx
            else:
                keys = lookup_ids
            uniq_keys, first_idx, inverse = np.unique(
                keys, return_index=True, return_inverse=True
            )
            if uniq_keys.size < keys.size:
                st["uniq_lookup_ids"] = np.ascontiguousarray(lookup_ids[first_idx])
                st["uniq_inverse"] = np.asarray(inverse, dtype=np.int64)
                if wave_idx is not None:
                    st["uniq_wave_idx"] = np.ascontiguousarray(wave_idx[first_idx])

        self._fold_static_cache[key] = st
        return st

    def _expand_categorical_covariates(self) -> None:
        """Validate covariates and one-hot expand the categorical ones in place.

        Runs once on the full ``self.target_gdf`` (before any train/val/test
        split) so the dummy columns are identical across splits and ride through
        the carry → split → collapse flow as plain numeric controls. Each
        categorical covariate is replaced in ``self.covariate_columns`` by its
        drop-first dummy column names; numeric covariates are kept as-is. After
        expansion, any covariate the user tagged numeric that isn't numeric is
        rejected.
        """
        gdf = self.target_gdf
        if gdf is None or not self._covariate_columns_user:
            return

        missing = [c for c in self._covariate_columns_user if c not in gdf.columns]
        if missing:
            raise ValueError(
                f"covariate_columns not found on target: {missing}. "
                f"Available attribute columns: {sorted(gdf.columns)}"
            )

        if not self._covariates_expanded:
            expanded: list[str] = []
            new_frames: list[pd.DataFrame] = []
            for c in self._covariate_columns_user:
                if self.covariate_types.get(c) == "categorical":
                    dummies = pd.get_dummies(
                        gdf[c],
                        prefix=c,
                        prefix_sep="=",
                        drop_first=True,
                        dummy_na=False,
                    ).astype(np.float64)
                    if dummies.shape[1] == 0:
                        _log(
                            "WARN",
                            f"Categorical covariate '{c}' has <2 levels after "
                            "drop-first; dropping it.",
                        )
                        continue
                    new_frames.append(dummies)
                    self._covariate_dummy_map[c] = list(dummies.columns)
                    expanded.extend(dummies.columns)
                else:
                    expanded.append(c)
            if new_frames:
                # One concat rather than a column-at-a-time insert, so the
                # frame is consolidated once.
                merged = pd.concat(new_frames, axis=1)
                for dcol in merged.columns:
                    gdf[dcol] = merged[dcol].to_numpy()
            self._user_covariate_columns = list(dict.fromkeys(expanded))
            period = self._expand_period_controls(gdf)
            self.covariate_columns = self._user_covariate_columns + period
            self._covariates_expanded = True

        non_numeric = [
            c
            for c in self.covariate_columns
            if not pd.api.types.is_numeric_dtype(gdf[c])
        ]
        if non_numeric:
            raise ValueError(
                "Numeric covariate columns must be numeric for the regression-"
                f"based scorers; got non-numeric: {non_numeric}. Tag them as "
                "categorical to one-hot encode instead."
            )

    def _expand_period_controls(self, gdf: "gpd.GeoDataFrame") -> list[str]:
        """One-hot the wave and area groupings into extra control columns.

        These ride at the end of ``covariate_columns`` so every scoring path
        picks them up as ordinary fixed effects, while the reporting layer —
        which names only ``_user_covariate_columns`` — leaves them out of the
        per-covariate table.

        Wave indicators absorb period effects: with per-wave greenery files the
        exposure carries the layer's vintage, which tracks calendar time and
        would otherwise land on the greenery × time terms. Area indicators
        absorb the neighbourhood level, which a person-only random effect does
        not represent — an exposure shared across neighbours makes the greenery
        term's standard error far too small without it.
        """
        self._period_control_columns = []
        self._wave_control_columns = []
        spec = self.longitudinal_spec
        if not self.is_longitudinal or spec is None:
            return []

        sources: list[str] = []
        if spec.include_wave_fixed_effects and "wave" in gdf.columns:
            sources.append("wave")
        if spec.area_id_col and spec.area_id_col in gdf.columns:
            sources.append(spec.area_id_col)

        out: list[str] = []
        for col in sources:
            dummies = pd.get_dummies(
                gdf[col], prefix=col, prefix_sep="=", drop_first=True, dummy_na=False
            ).astype(np.float64)
            if dummies.shape[1] == 0:
                _log("WARN", f"'{col}' has <2 levels; no fixed effects added.")
                continue
            for dcol in dummies.columns:
                gdf[dcol] = dummies[dcol].to_numpy()
            out.extend(dummies.columns)
            if col == "wave":
                self._wave_control_columns = list(dummies.columns)
            _log("INFO", f"Added {dummies.shape[1]} fixed effect(s) for '{col}'.")

        self._period_control_columns = out
        # Wave indicators can already span continuous time; keeping both would
        # be rank-deficient.
        if out and "wave" in sources and "years_since_baseline" in gdf.columns:
            t = gdf["years_since_baseline"].to_numpy(dtype=np.float64)
            wave_cols = [c for c in out if c.startswith("wave=")]
            if wave_cols and mixed_effects_scoring._time_spanned_by(
                t, gdf[wave_cols].to_numpy(dtype=np.float64)
            ):
                self._time_fixed_effect_dropped = True
                _log(
                    "INFO",
                    "Continuous time is spanned by the wave indicators; "
                    "dropping the time fixed effect.",
                )
        return out

    def _attach_entity_coords(
        self, fusion_df: pd.DataFrame, src_gdf: "gpd.GeoDataFrame"
    ) -> pd.DataFrame:
        """Attach representative-point metre coordinates as ``_cx`` / ``_cy``.

        Used only by the spatial-confounding adjustment: the coordinates ride
        through the train/val/test split and (in polygon mode) the entity
        collapse exactly like a covariate, so the per-split smooth can be built
        at scoring time. No-op when spatial adjustment is off, the geometry is
        missing, or the row counts do not line up; coordinates are not added to
        any NaN-drop subset so they never remove rows.
        """
        if self.spatial_adjust_method == "none":
            return fusion_df
        geom = getattr(src_gdf, "geometry", None)
        if geom is None or len(src_gdf) != len(fusion_df):
            return fusion_df
        try:
            g = src_gdf
            if g.crs is not None and not crs_uses_metre_axes(g.crs):
                g = g.to_crs(estimate_metre_projected_crs_for_gdf(g))
            gg = g.geometry
            if bool((gg.geom_type == "Point").all()):
                cx = gg.x.to_numpy(dtype=np.float64)
                cy = gg.y.to_numpy(dtype=np.float64)
            else:
                rep = gg.representative_point()
                cx = rep.x.to_numpy(dtype=np.float64)
                cy = rep.y.to_numpy(dtype=np.float64)
        except Exception as exc:  # geometry/CRS trouble → skip, fall back to covariates
            _log(
                "WARN",
                f"Could not attach entity coordinates for spatial adjustment: {exc}",
            )
            return fusion_df
        fusion_df = fusion_df.copy()
        fusion_df["_cx"] = cx
        fusion_df["_cy"] = cy
        return fusion_df

    def _spatial_basis_columns(
        self,
        coords_xy: np.ndarray | None,
        target: np.ndarray,
        cov: np.ndarray | None,
        cgi: np.ndarray,
    ) -> np.ndarray | None:
        """Df-selected spatial smooth columns for one scored split, or ``None``.

        Builds the per-cluster candidate basis once per split fingerprint (the
        coordinates and outcome are stable across trials) and memoizes it on
        ``_spatial_basis_cache``. For ``ks_aic`` the df is selected once on the
        outcome (cached); for ``spatial_plus`` the df is selected per call on the
        exposure ``cgi``. ``None`` means no spatial adjustment is applied (off,
        no coordinates, degenerate geometry, or AIC preferred no smooth).
        """
        if self.spatial_adjust_method == "none" or coords_xy is None:
            return None
        coords_xy = np.asarray(coords_xy, dtype=np.float64)
        cache_key = _helpers.spatial_cache_key(coords_xy, target, cov)
        entry = self._spatial_basis_cache.get(cache_key)
        if entry is None:
            basis = spatial_basis.build_block_basis(
                np.asarray(coords_xy, dtype=np.float64),
                max_df=self.spatial_adjust_max_df,
                eps=self.spatial_adjust_eps_m,
            )
            ks_df, ks_cols = (None, None)
            if basis.has_spatial:
                ks_df, ks_cols = spatial_basis.select_df_aic(
                    np.asarray(target, dtype=np.float64),
                    basis,
                    None if cov is None else np.asarray(cov, dtype=np.float64),
                )
            entry = {"basis": basis, "ks_cols": ks_cols, "ks_df": ks_df}
            self._spatial_basis_cache[cache_key] = entry
            if self._spatial_adjust_summary is None and basis.has_spatial:
                self._spatial_adjust_summary = {
                    "method": self.spatial_adjust_method,
                    "ks_df": ks_df,
                    **basis.summary,
                }
        basis = entry["basis"]
        if not basis.has_spatial:
            return None
        if self.spatial_adjust_method == "ks_aic":
            return entry["ks_cols"]
        # spatial_plus: the exposure is residualized on the smooth, so df is
        # selected on the exposure (df-Spatial+); the candidate basis is reused
        # and the per-call df search runs off the memoized nested-QR precompute
        # (one matvec instead of max_df+1 least-squares fits).
        pre = entry.get("plus_pre")
        if pre is None and "plus_pre" not in entry:
            pre = spatial_basis.precompute_df_selection(basis)
            entry["plus_pre"] = pre
        _df, sp_cols = spatial_basis.select_df_aic_fast(
            np.asarray(cgi, dtype=np.float64), basis, pre
        )
        return sp_cols

    def _finalize_composite(self, composite: np.ndarray) -> np.ndarray:
        """Apply the ``whole_grid_scaling`` toggle to a per-pixel composite.

        Per-channel scaling is never applied — the composite (CGI in combined
        mode, the single channel in standalone mode) is always a weighted
        combination of **raw** aggregated channel values. This toggle instead
        governs the composite itself: when on, the per-pixel composite is
        min-max normalized to ``[0, 1]`` over the supplied pixels before it is
        collapsed to entities / scored / written; when off, the raw composite
        is used everywhere. Applied identically in scoring and in the output
        raster (and mirrored for standalone single-channel maps) so the
        rendered map always matches the values that were scored.

        Note: the cross-sectional metrics are scale-invariant (distance
        correlation and r² are affine-invariant; Spearman and quantile-binned
        mutual information are rank-invariant; nRMSE min-max normalizes
        internally), so this changes the recorded composite values and the
        output raster — not the optimization objective.
        """
        if not self.whole_grid_scaling:
            return composite
        arr = np.asarray(composite)
        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float64)
        valid = ~np.isnan(arr)
        if np.any(valid):
            lo = float(arr[valid].min())
            hi = float(arr[valid].max())
            if hi > lo:
                arr = arr.copy()
                dt = arr.dtype.type
                arr[valid] = (arr[valid] - dt(lo)) / dt(hi - lo)
        return arr

    def _coverage_from_cache(
        self, entity_gdf: gpd.GeoDataFrame
    ) -> dict[str, np.ndarray] | None:
        """Per-row veg / terrain / NDVI coverage read from the attached cache.

        Returns the cached max-radius mean per ``(pixel, wave)`` row (NaN where
        no greenery is within range — the same coverage the sampler would find).
        ``None`` on any missing cell, so the caller falls back to sampling.
        """
        cache = self._preaggr_cache
        spec = self.longitudinal_spec
        if (
            not isinstance(cache, preaggregation.GreeneryCache)
            or spec is None
            or "_global_pid" not in entity_gdf.columns
        ):
            return None
        gvi_radii, ndvi_radii = self._preaggr_radii()
        wave_index_of = {w: i for i, w in enumerate(spec.wave_labels)}
        pids = np.asarray(entity_gdf["_global_pid"].to_numpy(), dtype=np.int64)
        waves = entity_gdf["wave"].astype(str).to_numpy()
        try:
            widx = np.array([wave_index_of[w] for w in waves], dtype=np.int64)
        except KeyError:
            return None
        out: dict[str, np.ndarray] = {}
        for ch in longitudinal.GREENERY_CHANNELS:
            radius = int((ndvi_radii if ch == "ndvi" else gvi_radii)[-1])
            col = np.full(len(pids), np.nan, dtype=np.float32)
            for w in np.unique(widx):
                m = widx == w
                vals = cache.lookup(pids[m], ch, radius, "mean", wave_index=int(w))
                if vals is None:
                    return None
                col[m] = vals
            out[ch] = col
        return out

    def _probe_channel_coverage(
        self,
        pixels: gpd.GeoDataFrame,
        pix_idx: np.ndarray,
        entity_gdf: gpd.GeoDataFrame,
    ) -> dict[str, np.ndarray]:
        """Per-catchment-row veg / terrain / NDVI values for the coverage gate.

        Cross-sectional runs probe every unique pixel once against the single
        metric source per channel.

        Longitudinal runs probe per wave: a row's coverage is decided by the
        metric files of the wave it was measured in. Probing every wave against
        one representative wave's source would gate the whole study on that
        wave's extent, discarding observations whose own wave covers them
        perfectly well.
        """
        n_rows = len(entity_gdf)
        provider = self._metric_sources
        waves = (
            entity_gdf["wave"].astype(str).to_numpy()
            if (provider is not None and "wave" in entity_gdf.columns)
            else None
        )

        # Coverage from a reusable cache (skips the per-wave re-sample) — only
        # when a signature-matched complete cache is available; falls back to
        # sampling on any miss.
        if waves is not None and _helpers.attach_cache_for_coverage(entity_gdf):
            cached = self._coverage_from_cache(entity_gdf)
            if cached is not None:
                _log(
                    "OK",
                    f"Coverage gate: derived veg / terrain / NDVI coverage for "
                    f"{n_rows:,} rows from the pre-aggregation cache.",
                )
                return cached

        if waves is None:
            _log(
                "INFO",
                f"Probing veg / terrain / NDVI coverage at {len(pixels):,} "
                "unique pixel centroids...",
            )
            probe = self._sample_metrics_at_points(pixels[["geometry"]].copy())
            return {
                ch: np.asarray(probe[ch].to_numpy(), dtype=np.float32)[pix_idx]
                for ch in ("veg", "terrain", "ndvi")
            }

        channels = longitudinal.GREENERY_CHANNELS
        out = {ch: np.full(n_rows, np.nan, dtype=np.float32) for ch in channels}

        # Temporal keys resolving to the same files across all three channels
        # are probed once together, so a channel that reuses one file across
        # keys is not re-read per key.
        present_keys = list(dict.fromkeys(waves.tolist()))
        groups: dict[tuple[str, ...], list[str]] = {}
        for key in present_keys:
            ident = tuple(provider.identity(ch, key) for ch in channels)
            groups.setdefault(ident, []).append(key)

        probed_total = 0
        _log(
            "INFO",
            f"Probing veg / terrain / NDVI coverage across {len(pixels):,} "
            f"unique pixel centroids — {len(groups)} distinct file set(s) "
            f"over {len(present_keys)} {self._temporal_key_noun()}(s).",
        )
        for group_keys in groups.values():
            # One file set per iteration, so cancel lands within a single
            # wave's probe rather than after the whole ladder.
            if self._cancel_callback is not None and self._cancel_callback():
                raise JobCancelled("Coverage probe cancelled by user.")
            row_mask = (
                waves == group_keys[0]
                if len(group_keys) == 1
                else np.isin(waves, group_keys)
            )
            group_pix = np.unique(pix_idx[row_mask])
            if group_pix.size == 0:
                continue
            try:
                sources = provider.get_all(group_keys[0], channels)
                probe = self._sample_metrics_at_points(
                    pixels.iloc[group_pix][["geometry"]].copy(), sources, quiet=True
                )
                counts = {}
                for ch in channels:
                    vals = np.asarray(probe[ch].to_numpy(), dtype=np.float32)
                    # Position of each row's pixel within ``group_pix``, mapping
                    # the probe result back onto this group's catchment rows.
                    out[ch][row_mask] = vals[
                        np.searchsorted(group_pix, pix_idx[row_mask])
                    ]
                    counts[ch] = int(np.isfinite(vals).sum())
            finally:
                # Free this key's files before the next one is opened.
                provider.release()
            probed_total += int(group_pix.size)
            label = (
                group_keys[0]
                if len(group_keys) == 1
                else f"{group_keys[0]}..{group_keys[-1]} ({len(group_keys)} keys)"
            )
            _log(
                "OK" if any(counts.values()) else "WARN",
                f"  {self._temporal_key_noun()} {label}: {group_pix.size:,} "
                f"pixel(s) probed — valid veg={counts['veg']:,} "
                f"terrain={counts['terrain']:,} ndvi={counts['ndvi']:,}",
            )
        _log(
            "INFO",
            f"Coverage probe sampled {probed_total:,} "
            f"(pixel, {self._temporal_key_noun()}) pairs.",
        )
        return out

    @staticmethod
    def _shared_gvi_layer(veg_src: Any, terrain_src: Any) -> tuple[Any, dict] | None:
        """One point layer carrying both GVI channels, when they share geometry.

        Returns ``(layer, {channel: column})``, or ``None`` when either side is
        a raster or their point geometry differs — then each channel is sampled
        on its own, as before. The two channels arrive as separate frames read
        from the same file, so the shared layer is assembled here by borrowing
        terrain's column onto veg's frame (positionally: identical coordinates
        in identical order is exactly what was just verified).
        """
        if veg_src is None or terrain_src is None:
            return None
        if isinstance(veg_src, dict) or isinstance(terrain_src, dict):
            return None
        try:
            if len(veg_src) != len(terrain_src) or len(veg_src) == 0:
                return None
            if veg_src.crs != terrain_src.crs:
                return None
            vx = veg_src.geometry.x.to_numpy()
            tx = terrain_src.geometry.x.to_numpy()
            if not np.array_equal(vx, tx):
                return None
            if not np.array_equal(
                veg_src.geometry.y.to_numpy(), terrain_src.geometry.y.to_numpy()
            ):
                return None
            veg_col = metric_sampling.vector_metric_column(veg_src, "veg")
            ter_col = metric_sampling.vector_metric_column(terrain_src, "terrain")
        except Exception:
            return None
        if veg_col == ter_col:
            return None
        layer = veg_src[["geometry", veg_col]].copy()
        layer[ter_col] = terrain_src[ter_col].to_numpy()
        return layer, {"veg": veg_col, "terrain": ter_col}

    def _sample_metrics_at_points(
        self,
        points_gdf: gpd.GeoDataFrame,
        sources: Mapping[str, Any] | None = None,
        *,
        quiet: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Sample vegetation, terrain, and NDVI metrics at point locations.

        This method is shared by both point and raster fusion workflows.
        For rasters, the pixel centers are converted to points first.

        Args:
            points_gdf: GeoDataFrame with point geometries to sample at
            sources: channel → metric source to sample against. Defaults to the
                engine's cross-sectional ``veg_data`` / ``terrain_data`` /
                ``ndvi_data``. Longitudinal callers pass one wave's sources so
                each observation is probed against the metrics of its own wave.
            quiet: suppress the per-channel log lines (used when sampling is
                repeated once per wave, which would otherwise flood the log)

        Returns:
            GeoDataFrame with added columns: veg, terrain, ndvi
        """
        src = {
            "veg": self.veg_data,
            "terrain": self.terrain_data,
            "ndvi": self.ndvi_data,
            **(dict(sources) if sources else {}),
        }
        radius = {
            "veg": self.gvi_buffer_max_m,
            "terrain": self.gvi_buffer_max_m,
            "ndvi": self.ndvi_buffer_max_m,
        }
        label = {"veg": "Veg", "terrain": "Terrain", "ndvi": "NDVI"}

        # ``veg`` and ``terrain`` are normally two attribute columns of the same
        # street-view points, and the nearest-feature join — the expensive half —
        # depends only on geometry. Joining once serves both. It runs before the
        # loop because the loop blanks each channel as it reaches it, which would
        # wipe a value written during another channel's turn.
        shared_gvi = self._shared_gvi_layer(src.get("veg"), src.get("terrain"))
        shared_values: dict[str, Any] = {}
        if shared_gvi is not None:
            layer, cols = shared_gvi
            joined = metric_sampling.nearest_metric_join_multi(
                points_gdf, layer, list(cols.values()), radius["veg"]
            )
            shared_values = {ch: joined[col] for ch, col in cols.items()}
            if not quiet:
                _log(
                    "INFO",
                    f"Veg + Terrain sampled from one nearest-feature join "
                    f"({len(layer):,} features).",
                )

        for channel in ("veg", "terrain", "ndvi"):
            points_gdf[channel] = np.nan
            data = src[channel]
            if data is None:
                continue
            if channel in shared_values:
                points_gdf[channel] = shared_values[channel]
                continue
            if isinstance(data, dict):  # Raster
                points_gdf[channel] = metric_sampling.sample_raster_values(
                    points_gdf.to_crs(data["crs"]), data
                )
                continue
            # Vector
            col = metric_sampling.vector_metric_column(data, channel)
            if not quiet:
                _log(
                    "INFO",
                    f"{label[channel]} data: {len(data)} features, "
                    f"columns: {data.columns.tolist()}",
                )
                _log("INFO", f"Using {channel} column: '{col}'")
                _log(
                    "INFO",
                    f"Nearest-feature max distance ({channel}): "
                    f"{radius[channel]} m",
                )
            points_gdf[channel] = metric_sampling.nearest_metric_join(
                points_gdf, data, col, radius[channel]
            )
            if not quiet:
                valid = int(points_gdf[channel].notna().sum())
                _log(
                    "OK" if valid else "WARN",
                    f"{label[channel]} sampling: {valid}/{len(points_gdf)} "
                    "points have valid values",
                )

        return points_gdf

    def _prepare_raster_fusion(self) -> pd.DataFrame:
        """
        Convert raster to point samples at pixel centers.

        Each pixel in the target raster becomes a sample point. During optimization,
        circular buffer aggregation will be applied around each point to sample
        greenery metrics with trial-specific radii.

        This matches CGI.ipynb methodology where each pixel is treated as a point
        sample with circular neighborhood aggregation.
        """
        logger.info("Preparing raster-based fusion data...")
        logger.info("Converting target raster pixels to point samples...")

        target_data = self.target_raster["data"]
        transform = self.target_raster["transform"]
        crs = self.target_raster["crs"]
        width = int(target_data.shape[1])

        # Keep only valid-target pixels before building any geometry: one
        # geometry per grid cell would cost tens of GB on a large raster.
        flat_target = np.ma.filled(target_data, np.nan).ravel()
        keep = np.flatnonzero(~np.isnan(flat_target))
        if keep.size == 0:
            raise ValueError(
                "No valid pixels found in target raster. "
                "Target raster may be empty or all NaN."
            )
        rows_i = (keep // width).astype(np.int64)
        cols_i = (keep % width).astype(np.int64)
        xs, ys = xy(transform, rows_i, cols_i, offset="center")

        pixel_points = gpd.GeoDataFrame(
            {
                "target": flat_target[keep],
                "row": rows_i,
                "col": cols_i,
            },
            geometry=gpd.points_from_xy(np.asarray(xs), np.asarray(ys)),
            crs=crs,
        )

        logger.info(f"Created {len(pixel_points):,} point samples from raster pixels")

        # Store as target_gdf for unified workflow
        self.target_gdf = pixel_points

        points_gdf = pixel_points.copy()

        # Sample each metric using the shared helper method
        points_gdf = self._sample_metrics_at_points(points_gdf)

        # Create final DataFrame
        fusion_df = pd.DataFrame(
            {
                "target": points_gdf["target"],
                "veg": points_gdf["veg"],
                "terrain": points_gdf["terrain"],
                "ndvi": points_gdf["ndvi"],
            }
        )
        fusion_df = self._attach_entity_coords(fusion_df, points_gdf)

        # Log data quality before dropping NaN
        _log("INFO", "====== DATA QUALITY SUMMARY (RASTER) ======")
        _log("INFO", f"Total rows: {len(fusion_df)}")
        for col in ("target", "veg", "terrain", "ndvi"):
            nan_count = fusion_df[col].isna().sum()
            pct = nan_count / max(len(fusion_df), 1) * 100
            _log("INFO", f"{col.capitalize()} NaN: {nan_count} ({pct:.1f}%)")

        result = fusion_df.dropna()
        _log(
            "OK" if len(result) else "WARN",
            f"After dropna: {len(result)} valid rows",
        )

        if len(result) == 0:
            raise ValueError(
                "No valid samples after spatial join. "
                "This usually means the target raster pixels and metric data don't overlap spatially. "
                "Check that all data covers the same geographic area."
            )

        return result

    def _polygon_areas_km2(self, polygon_ids: pd.Series) -> pd.Series:
        """Per-polygon area in km² keyed by polygon_id (NaN if unknown)."""
        lookup = self._polygon_id_to_area_km2 or {}
        return polygon_ids.map(
            lambda pid: (
                lookup.get(int(pid), np.nan)
                if isinstance(pid, (int, np.integer))
                else np.nan
            )
        )

    def _area_balanced_polygon_split(
        self,
        poly_df: pd.DataFrame,
        group_col: str,
        target_fractions: dict[str, float],
        random_state: int,
    ) -> dict[str, pd.DataFrame]:
        """Run the area-balanced per-bin allocation across all outcome bins.

        ``poly_df`` must have a ``target_bin`` column and ``group_col``
        identifying each polygon. Per-polygon areas are looked up via
        ``_polygon_areas_km2``. Returns a dict mapping split name to the
        sub-frame of ``poly_df``.
        """
        poly_df = poly_df.copy()
        poly_df["_area_km2"] = self._polygon_areas_km2(poly_df[group_col]).to_numpy()
        per_bin_assignments: dict[str, list] = {name: [] for name in target_fractions}
        for bin_value, bin_df in poly_df.groupby("target_bin", sort=False):
            alloc = _helpers.area_balanced_split_within_bin(
                bin_df,
                "_area_km2",
                target_fractions,
                random_state=int(random_state) + int(bin_value),
            )
            for name, idxs in alloc.items():
                per_bin_assignments[name].extend(idxs)
        return {
            name: poly_df.loc[idxs].drop(columns=["_area_km2"]).reset_index(drop=True)
            for name, idxs in per_bin_assignments.items()
        }

    def _resolve_group_block_ids(
        self,
        group_ids: np.ndarray,
        group_col: str,
        fusion_df: pd.DataFrame,
        *,
        block_size_m: float | None,
        n_blocks: int | None,
    ) -> tuple[dict, dict] | None:
        """Map each split group to a coarse spatial-block id.

        Resolves a representative point per group from ``target_gdf`` (point
        pixel centres, or polygon representative points), reducing to the
        split's group key — directly when ``group_col`` is a ``target_gdf``
        column, otherwise via the ``polygon_id → group_col`` map carried in
        ``fusion_df``. Returns ``({group_id: block_id}, info)`` or ``None`` when
        group geometry can't be resolved (caller then skips spatial blocking).

        The block edge is floored at twice the largest catchment radius so a
        block is wider than the greenery autocorrelation range, keeping a
        held-out block's catchments clear of the train blocks around it.
        """
        tg = self.target_gdf
        if tg is None or getattr(tg, "geometry", None) is None or len(tg) == 0:
            return None
        try:
            geom = tg.geometry
            is_point = bool((geom.geom_type == "Point").all())
            if is_point:
                rx = geom.x.to_numpy(dtype=np.float64)
                ry = geom.y.to_numpy(dtype=np.float64)
            else:
                rep = geom.representative_point()
                rx = rep.x.to_numpy(dtype=np.float64)
                ry = rep.y.to_numpy(dtype=np.float64)
        except Exception:
            return None

        if group_col in tg.columns:
            keys = tg[group_col].to_numpy()
        elif (
            "polygon_id" in tg.columns
            and "polygon_id" in fusion_df.columns
            and group_col in fusion_df.columns
        ):
            pid_to_group = dict(
                zip(
                    fusion_df["polygon_id"].to_numpy(),
                    fusion_df[group_col].to_numpy(),
                )
            )
            keys = np.array(
                [pid_to_group.get(p) for p in tg["polygon_id"].to_numpy()],
                dtype=object,
            )
        else:
            return None

        cdf = pd.DataFrame({"_g": keys, "x": rx, "y": ry}).dropna(subset=["_g"])
        if cdf.empty:
            return None
        agg = cdf.groupby("_g", sort=False)[["x", "y"]].mean()
        group_pts = gpd.GeoDataFrame(
            {"_g": agg.index.to_numpy()},
            geometry=gpd.points_from_xy(agg["x"].to_numpy(), agg["y"].to_numpy()),
            crs=tg.crs,
        )
        r_max = max(
            float(getattr(self, "gvi_buffer_max_m", 0.0) or 0.0),
            float(getattr(self, "ndvi_buffer_max_m", 0.0) or 0.0),
        )
        min_block = 2.0 * r_max if r_max > 0 else None
        try:
            block_ids, info = assign_spatial_blocks(
                group_pts,
                block_size_m=block_size_m,
                target_blocks=n_blocks,
                min_block_size_m=min_block,
            )
        except Exception:
            return None
        block_by_group = dict(zip(agg.index.to_numpy(), block_ids.tolist()))
        return block_by_group, info

    def split_data(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        fusion_df: pd.DataFrame | None = None,
        *,
        spatial_split: bool = False,
        spatial_block_size_m: float | None = None,
        n_spatial_blocks: int | None = None,
    ) -> None:
        """Carve a held-out test set and the train+val pool the bootstrap
        stability search resamples.

        Stability selection does its own complementary-half resampling of the
        train+val pool (see :meth:`bootstrap_stability_selection`), so this
        method only sets aside the untouched test split and the pool; it does
        not build CV folds. The split is leakage-safe (groups stay together
        when ``entity_id`` / ``polygon_id`` is present) and outcome-stratified,
        with optional spatial blocking so the held-out test tiles the extent.

        Args:
            test_size: Held-out test fraction.
            random_state: RNG seed for reproducibility.
            fusion_df: Prepared fusion frame; built via
                :meth:`prepare_fusion_data` when ``None``.
            spatial_split: Stripe whole spatial blocks into the test set so it
                tiles the full extent and no group straddles train and test.
            spatial_block_size_m: Block edge length (metres) when blocking.
            n_spatial_blocks: Target block count when blocking.
        """
        # A fresh split invalidates the reporting-phase caches (they key off the
        # train+val / test frames this method rebuilds).
        self._full_data_cache = None
        self._apply_fusion_cache = None
        self._evaluate_test_cache.clear()
        self._fold_static_cache.clear()
        self._split_generation += 1
        self._clear_scoring_caches()

        # Step 1: Sample all metrics at initial buffer distance
        logger.info("Step 1/4: Sampling metrics at point locations...")
        if fusion_df is None:
            fusion_df = self.prepare_fusion_data()

        logger.info(f"Initial samples before filtering: {len(fusion_df)}")

        # Step 2: Filter NaN values (already done in prepare_fusion_data with dropna)
        logger.info("Step 2/4: Filtering complete - removed samples with NaN values")
        logger.info(f"Valid samples after filtering: {len(fusion_df)}")

        # Determine the group column for leakage-safe splitting. entity_id
        # takes priority over polygon_id because in longitudinal+polygon mode
        # both are present, but we want every (entity, *) row to stay in the
        # same split. Per-entity outcome for stratification = mean across
        # waves; polygon-mode stratifies on each polygon's single outcome.
        group_col: str | None = None
        group_label: str = ""
        group_agg: str = "first"
        if "entity_id" in fusion_df.columns:
            group_col, group_label, group_agg = "entity_id", "entity", "mean"
        elif "polygon_id" in fusion_df.columns:
            group_col, group_label, group_agg = "polygon_id", "polygon", "first"

        if group_col is not None:
            # ── Group-level stratified split ────────────
            # Stratify on the per-group outcome so the distribution stays
            # balanced across train / val / test, and all rows of a group
            # stay together to avoid leakage.
            _log(
                "INFO",
                f"Splitting at {group_label} level (stratified on outcome, "
                f"agg={group_agg})...",
            )
            poly_df = (
                fusion_df.groupby(group_col, sort=False)["target"]
                .agg(group_agg)
                .reset_index()
            )

            n_polys = len(poly_df)
            n_test_target = max(1, int(round(test_size * n_polys)))

            # Bin count constrained by ``n_test_target`` so the test split can
            # include every class (at least 2 groups per bin).
            max_bins_for_test = max(2, n_test_target)
            requested_bins = max(2, self.n_bins)
            n_bins_eff = min(
                requested_bins,
                max_bins_for_test,
                max(2, poly_df["target"].nunique()),
            )

            try:
                poly_df["target_bin"] = pd.qcut(
                    poly_df["target"],
                    q=n_bins_eff,
                    labels=False,
                    duplicates="drop",
                )
            except ValueError:
                poly_df["target_bin"] = 0
            actual_bins = poly_df["target_bin"].nunique()
            stratifiable = actual_bins >= 2 and (
                poly_df["target_bin"].value_counts().min() >= 2
            )
            _log(
                "INFO",
                f"{n_polys} {group_label}s across {actual_bins} outcome bin(s) "
                f"(requested {requested_bins}; "
                f"{'stratified' if stratifiable else f'unstratified — too few {group_label}s per bin'}).",
            )

            # Optional spatial blocking: tag each group with a coarse grid
            # block so whole blocks — never split groups — can be striped
            # across the held-out test, spreading it over the full extent
            # while keeping a block's catchments clear of its neighbours.
            spatial_ok = False
            self._spatial_block_by_group = None
            self._spatial_block_group_col = None
            if spatial_split:
                resolved = self._resolve_group_block_ids(
                    poly_df[group_col].to_numpy(),
                    group_col,
                    fusion_df,
                    block_size_m=spatial_block_size_m,
                    n_blocks=n_spatial_blocks,
                )
                if resolved is not None:
                    block_by_group, binfo = resolved
                    poly_df["_block"] = poly_df[group_col].map(block_by_group)
                    n_mapped_blocks = int(poly_df["_block"].dropna().nunique())
                    if poly_df["_block"].notna().any() and n_mapped_blocks >= 2:
                        poly_df["_block"] = poly_df["_block"].fillna(-1).astype(int)
                        spatial_ok = True
                        self._spatial_block_by_group = block_by_group
                        self._spatial_block_group_col = group_col
                        _log(
                            "INFO",
                            f"Spatial blocking: {n_mapped_blocks} blocks "
                            f"(~{binfo.get('block_size_m', 0.0):.0f} m) over "
                            f"{n_polys} {group_label}s; held-out test and folds "
                            "striped across blocks.",
                        )
                    else:
                        _log(
                            "WARN",
                            "Spatial blocking produced <2 usable blocks; "
                            "falling back to outcome-stratified split.",
                        )
                else:
                    _log(
                        "WARN",
                        "Spatial blocking could not resolve group geometry; "
                        "falling back to outcome-stratified split.",
                    )

            if spatial_ok:
                # Spatial single-split: assign whole blocks to the held-out
                # test by even striping across the space-filling block order,
                # so the test tiles the full extent and no group straddles
                # train and test.
                rng_sp = np.random.default_rng(random_state)
                ordered_blocks = np.sort(poly_df["_block"].unique())
                test_blocks = _helpers.stripe_to_test_blocks(
                    ordered_blocks, test_size, rng_sp
                )
                test_mask = poly_df["_block"].isin(test_blocks)
                test_poly = poly_df[test_mask].copy()
                train_val_poly = poly_df[~test_mask].copy()
                _log(
                    "INFO",
                    f"Spatial block test split: {len(test_blocks)} of "
                    f"{len(ordered_blocks)} blocks held out — "
                    f"train+val={len(train_val_poly)} {group_label}s, "
                    f"test={len(test_poly)} {group_label}s.",
                )
            else:
                # Single-split mode: area-balanced or stratified random split
                # on ``test_size`` (original behaviour).
                use_area_balanced = (
                    stratifiable
                    and self.area_balanced_split
                    and group_label == "polygon"
                    and self._polygon_id_to_area_km2
                )
                if use_area_balanced:
                    target_fractions = {
                        "train_val": 1.0 - test_size,
                        "test": test_size,
                    }
                    allocations = self._area_balanced_polygon_split(
                        poly_df,
                        group_col=group_col,
                        target_fractions=target_fractions,
                        random_state=random_state,
                    )
                    train_val_poly = allocations["train_val"]
                    test_poly = allocations["test"]
                    _log(
                        "INFO",
                        "Area-balanced stratified test split: "
                        f"train+val={len(train_val_poly)} polygons, "
                        f"test={len(test_poly)} polygons.",
                    )
                elif stratifiable:
                    try:
                        train_val_poly, test_poly = train_test_split(
                            poly_df,
                            test_size=test_size,
                            stratify=poly_df["target_bin"],
                            random_state=random_state,
                        )
                    except ValueError as e:
                        _log(
                            "WARN",
                            f"Stratified split failed ({e}); using random split.",
                        )
                        train_val_poly, test_poly = train_test_split(
                            poly_df, test_size=test_size, random_state=random_state
                        )
                        stratifiable = False
                else:
                    train_val_poly, test_poly = train_test_split(
                        poly_df, test_size=test_size, random_state=random_state
                    )
            train_val_poly = train_val_poly.reset_index(drop=True)

            # Map group assignments back to row-level data.
            train_val_ids = set(train_val_poly[group_col])
            test_ids = set(test_poly[group_col])
            self.train_val_data = fusion_df[
                fusion_df[group_col].isin(train_val_ids)
            ].copy()
            self.test_data = fusion_df[fusion_df[group_col].isin(test_ids)].copy()

            _log(
                "INFO",
                f"Split: {len(train_val_poly)} train+val {group_label}s "
                f"({len(self.train_val_data)} rows), "
                f"{len(test_poly)} test {group_label}s ({len(self.test_data)} rows).",
            )

            # No CV folds — the bootstrap stability search resamples the pool
            # itself. Reset the caches the resampler rebuilds per run.
            self.cv_folds = []
            self._spatial_basis_cache = {}
            self._spatial_adjust_summary = None
            return

        # ── Row-level (point / raster) split ────────────────
        # Step 3: Create stratification bins based on target values
        logger.info("Step 3/4: Binning target values for stratified sampling...")
        fusion_df["target_bin"] = pd.qcut(
            fusion_df["target"], q=self.n_bins, labels=False, duplicates="drop"
        )
        logger.info(
            f"Created {fusion_df['target_bin'].nunique()} bins for stratification"
        )

        # Step 4: Stratified train/test split on ``test_size``.
        logger.info("Step 4/4: Performing stratified train/test split...")
        self.train_val_data, self.test_data = train_test_split(
            fusion_df,
            test_size=test_size,
            stratify=fusion_df["target_bin"],
            random_state=random_state,
        )

        logger.info(
            f"Split complete: {len(self.train_val_data)} train+val samples, "
            f"{len(self.test_data)} test samples (holdout)"
        )

        # No CV folds — the bootstrap stability search resamples the pool
        # itself. Reset the caches the resampler rebuilds per run.
        self.cv_folds = []
        self._spatial_basis_cache = {}
        self._spatial_adjust_summary = None

    def _suggest_gvi_radius(self, trial: optuna.Trial, name: str) -> int:
        lo, hi, step = metric_sampling.radius_int_bounds(
            self.gvi_buffer_min_m, self.gvi_buffer_max_m, self.gvi_buffer_step_m
        )
        if lo >= hi:
            return lo
        return trial.suggest_int(name, lo, hi, step=step)

    def _suggest_ndvi_radius(self, trial: optuna.Trial) -> int:
        lo, hi, step = metric_sampling.radius_int_bounds(
            self.ndvi_buffer_min_m, self.ndvi_buffer_max_m, self.ndvi_buffer_step_m
        )
        if lo >= hi:
            return lo
        return trial.suggest_int("ndvi_radius", lo, hi, step=step)

    def _effective_time_fixed(self, spec) -> bool:
        """The spec's time fixed effect, minus the case where wave indicators
        already span it (entering both would be rank-deficient)."""
        return (
            bool(spec.include_time_fixed_effect) and not self._time_fixed_effect_dropped
        )

    def _metric_has_pvalue(self, metric: str) -> bool:
        """True when scoring ``metric`` yields a usable parametric p-value.

        The longitudinal MixedLM ``tstat`` / ``coef`` metrics and both GEE
        logistic metrics carry a Wald p-value; the cross-sectional OLS metrics
        do not (their robustness comes from stability selection + bootstrap CIs,
        not a per-trial p-gate). The cross-sectional logistic metrics do produce
        one, and it is reported for consistency with the panel path.
        """
        if metric in objective_scoring.BINARY_ONLY_METRICS:
            return True
        if self.is_longitudinal and metric in longitudinal.GEE_LOGIT_METRICS:
            return True
        return self.is_longitudinal and metric in mixed_effects_scoring.HAS_PVALUE

    def _fold_fast_components(self, static: dict):
        """Per-fold ``(D, σ²)`` for the fast search scorer, cached on the static.

        Returns ``None`` (caller uses the exact per-trial fit) when the engine
        is not longitudinal, a spatial smooth is active, the fold lacks entity /
        time keys, or estimation fails. Computed once per fold subset and reused
        by every trial of that fold — the whole point of the fast path.
        """
        if "fast_components" in static:
            return static["fast_components"]
        comp = None
        spec = self.longitudinal_spec
        if (
            self.is_longitudinal
            and spec is not None
            and self.search_scoring_method
            in mixed_effects_scoring.FAST_COMPONENT_METHODS
            and self.spatial_adjust_method == "none"
            and static.get("entity_id") is not None
            and static.get("ysb") is not None
        ):
            comp = mixed_effects_scoring.estimate_fold_components(
                static["target"],
                static.get("cov"),
                static["ysb"],
                static["entity_id"],
                method=self.search_scoring_method,
                include_time_fixed=self._effective_time_fixed(spec),
                random_slope=spec.random_slope_time,
                target=spec.association_target,
            )
        static["fast_components"] = comp
        return comp

    def _fold_gee_baseline(self, static: dict):
        """Per-fold covariates-only GEE fit for the fast binary panel scorer.

        The binary analogue of :meth:`_fold_fast_components`: the null model's
        fitted probabilities, working weights and exchangeable correlation are
        properties of the outcome and covariates, so they are estimated once per
        fold and every trial takes one Newton step from them. ``None`` sends the
        caller to the exact per-trial GEE fit.
        """
        if "gee_baseline" in static:
            return static["gee_baseline"]
        baseline = None
        if (
            self.is_longitudinal
            and self.search_scoring_method != "exact"
            and self.spatial_adjust_method == "none"
            and static.get("entity_id") is not None
        ):
            baseline = binary_longitudinal.estimate_fold_baseline(
                static["target"], static["entity_id"], static.get("cov")
            )
        static["gee_baseline"] = baseline
        return baseline

    def _score_greenery(
        self,
        metric: str,
        target: "np.ndarray",
        composite: "np.ndarray",
        *,
        covariates: "np.ndarray | None" = None,
        spatial_basis: "np.ndarray | None" = None,
        entity_id: "np.ndarray | None" = None,
        years_since_baseline: "np.ndarray | None" = None,
        return_pvalue: bool = False,
        return_all: bool = False,
        nan_on_fail: bool = False,
        fast_components: "tuple[np.ndarray, float] | object | None" = None,
    ):
        """Single scoring seam: MixedLM (longitudinal) or OLS (cross-sectional).

        Routes to :func:`mixed_effects_scoring.score_mixedlm` when the engine is
        in longitudinal mode and ``metric`` is a ``mixedlm_*`` metric, otherwise
        to :func:`objective_scoring.score`. Every scoring call site — the
        objective, the held-out evaluation, and the reporting helpers — goes
        through here so the two modes cannot drift apart: a change to the
        cross-sectional scorer (residualization, spatial handling, caches)
        applies to the longitudinal OLS-metric path for free, and the MixedLM
        branch is the one isolated place the panel model lives.

        ``entity_id`` / ``years_since_baseline`` are consumed only by the MixedLM
        branch; the OLS branch ignores them (they are ``None`` in cross-sectional
        mode). ``return_all`` returns the four-metric MixedLM dict and is a no-op
        on the OLS path.

        ``nan_on_fail`` makes the MixedLM branch return ``NaN`` instead of the
        degenerate ``0.0`` when the model can't be fit, so reporting can tell a
        non-fit from a genuine null. The search leaves it off (a bad trial should
        rank last, not poison the pool) and the OLS path ignores it — the OLS
        scorers have no non-convergence mode.
        """
        if self.is_longitudinal and metric in longitudinal.GEE_LOGIT_METRICS:
            # Binary outcome on a panel: GEE with an exchangeable working
            # correlation clustered on the entity. ``fast_components`` carries a
            # GEEFoldBaseline here (the search path); its absence, or an exact
            # reporting call, takes the full refit.
            if (
                isinstance(fast_components, binary_longitudinal.GEEFoldBaseline)
                and not return_all
            ):
                return binary_longitudinal.score_gee_logit_fast(
                    metric,
                    composite,
                    fast_components,
                    return_all=return_all,
                    return_pvalue=return_pvalue,
                )
            return binary_longitudinal.score_gee_logit(
                metric,
                target,
                composite,
                entity_id,
                covariates,
                return_all=return_all,
                return_pvalue=return_pvalue,
            )

        if self.is_longitudinal and metric in mixed_effects_scoring.MIXEDLM_METRICS:
            spec = self.longitudinal_spec
            assert spec is not None  # guaranteed by is_longitudinal
            # Fast fixed-V GLS path for the trial search: reuse the per-fold
            # variance components, one GLS per trial instead of a MixedLM fit.
            # Only for the ranking metrics, with no spatial smooth, and never
            # for the exact reporting calls (return_all / no components).
            if (
                fast_components is not None
                and not return_all
                and metric in mixed_effects_scoring.FAST_SEARCH_METRICS
                and self.spatial_adjust_method == "none"
            ):
                return mixed_effects_scoring.score_mixedlm_fast(
                    metric,
                    target,
                    composite,
                    entity_id=entity_id,
                    years_since_baseline=years_since_baseline,
                    covariates=covariates,
                    components=fast_components,
                    include_time_fixed=self._effective_time_fixed(spec),
                    random_slope=spec.random_slope_time,
                    return_pvalue=return_pvalue,
                    nan_on_fail=nan_on_fail,
                    target=spec.association_target,
                )
            return mixed_effects_scoring.score_mixedlm(
                metric,
                target,
                composite,
                entity_id=entity_id,
                years_since_baseline=years_since_baseline,
                covariates=covariates,
                include_time_fixed=self._effective_time_fixed(spec),
                random_slope=spec.random_slope_time,
                return_pvalue=return_pvalue,
                return_all=return_all,
                nan_on_fail=nan_on_fail,
                spatial_basis=spatial_basis,
                spatial_method=self.spatial_adjust_method,
                target=spec.association_target,
            )
        return objective_scoring.score(
            metric,
            target,
            composite,
            covariates=covariates,
            return_pvalue=return_pvalue,
            spatial_basis=spatial_basis,
            spatial_method=self.spatial_adjust_method,
            residualize_method=self.residualize_method,
            pdcor_cache=self._pdcor_cache,
            spline_cache=self._spline_basis_cache,
        )

    def _objective(self, trial: optuna.Trial, metric: str) -> float:
        """Optuna objective with k-fold CV, timed for the concurrency report.

        Weight search matches CGI.ipynb (ndvi / veg / terrain summing to 100).
        Radii use separate GVI and NDVI buffer ladders (min / max / step metres)
        on the engine; extent padding remains ``buffer_meters``.
        Street-view aggregation parameters are shared for veg and terrain; NDVI
        uses separate stat / percentile choices.
        """
        # A cancelled job stops paying for trials it will never report. The
        # study-level callback stops the sampler from queueing more; this
        # prunes the ones already dispatched to the pool.
        if self._cancel_callback is not None and self._cancel_callback():
            raise optuna.TrialPruned("Cancelled by user")

        _t0 = time.perf_counter()
        try:
            return self._objective_inner(trial, metric)
        finally:
            # ``list.append`` is atomic under the GIL, so trials on the thread
            # pool accumulate without a lock; summing these gives worker-seconds.
            self._objective_secs.append(time.perf_counter() - _t0)

    def _objective_inner(self, trial: optuna.Trial, metric: str) -> float:
        gvi_cap = int(round(self.gvi_buffer_max_m))
        ndvi_cap = int(round(self.ndvi_buffer_max_m))

        if trial.number == 0:
            gvi_lo, gvi_hi, gvi_st = metric_sampling.radius_int_bounds(
                self.gvi_buffer_min_m,
                self.gvi_buffer_max_m,
                self.gvi_buffer_step_m,
            )
            ndvi_lo, ndvi_hi, ndvi_st = metric_sampling.radius_int_bounds(
                self.ndvi_buffer_min_m,
                self.ndvi_buffer_max_m,
                self.ndvi_buffer_step_m,
            )
            logger.info(
                f"Fusion extent buffer: {self.buffer_meters} m; "
                f"GVI radius search {gvi_lo}–{gvi_hi} m (step {gvi_st}); "
                f"NDVI radius search {ndvi_lo}–{ndvi_hi} m (step {ndvi_st})"
            )

        # ── Suggest formula parameters (weights + powers) ───
        # In ``cgi`` mode the formula's ``suggest_params`` bakes the weight-sum
        # / power constraints into the search space (sequential conditional
        # allocation on the simplex), and ``channel_active`` drives the
        # per-channel skip below. In a standalone single-metric study no
        # weights are recorded — only the active channel needs aggregation —
        # so ``formula_params`` is empty and the channel flags are hard-coded.
        channel_mode = self._active_greenery_channel
        if channel_mode == "cgi":
            formula = cgi_formulas.get_formula(self.cgi_formula)
            # Any channels turned off by check_channel_collinearity get
            # their weights pinned to 0 inside ``suggest_params`` so the
            # optimizer never wastes trials on disabled-channel
            # combinations and every trial's recorded params still satisfy
            # the formula's simplex constraint.
            formula_params = formula.suggest_params(
                trial, disabled_channels=set(self._disabled_channels)
            )
            channel_active = formula.channel_active(formula_params)
        else:
            formula = None
            formula_params = {}
            channel_active = {
                "veg": channel_mode == "veg",
                "terrain": channel_mode == "terrain",
                "ndvi": channel_mode == "ndvi",
            }

        # When pre-aggregation is active, percentile suggestions are restricted
        # to the 10 % grid {10,20,…,90} so every trial maps to a precomputed
        # cell. Otherwise the original 1-99 search space is used.
        preaggr_on = getattr(self, "_preaggregation_done", False)
        pct_grid = list(self._PREAGGR_PERCENTILES)

        # ── Streetview params (shared: veg + terrain) ───────
        # Only suggest if either veg or terrain channel contributes. The
        # gating mirrors the legacy weighted-average ``weight > 0`` skip so an
        # all-NDVI trial doesn't burn search dimensions on unused street-view
        # radii — under synergy a channel is virtually always active, so the
        # gating just doesn't fire there.
        if channel_active["veg"] or channel_active["terrain"]:
            streetview_stat = trial.suggest_categorical(
                "streetview_stat", ["mean", "median", "percentile"]
            )
            if streetview_stat == "percentile":
                if preaggr_on:
                    streetview_percentile = trial.suggest_categorical(
                        "streetview_percentile", pct_grid
                    )
                else:
                    streetview_percentile = trial.suggest_int(
                        "streetview_percentile", 1, 99
                    )
            else:
                streetview_percentile = 50

            veg_radius = (
                self._suggest_gvi_radius(trial, "veg_radius")
                if channel_active["veg"]
                else gvi_cap
            )
            terrain_radius = (
                self._suggest_gvi_radius(trial, "terrain_radius")
                if channel_active["terrain"]
                else gvi_cap
            )
        else:
            streetview_stat = "mean"
            streetview_percentile = 50
            veg_radius = gvi_cap
            terrain_radius = gvi_cap

        # ── Suggest NDVI Parameters (separate) ──────────────
        if channel_active["ndvi"]:
            ndvi_radius = self._suggest_ndvi_radius(trial)
            ndvi_stat = trial.suggest_categorical(
                "ndvi_stat", ["mean", "median", "percentile"]
            )
            if ndvi_stat == "percentile":
                if preaggr_on:
                    ndvi_percentile = trial.suggest_categorical(
                        "ndvi_percentile", pct_grid
                    )
                else:
                    ndvi_percentile = trial.suggest_int("ndvi_percentile", 1, 99)
            else:
                ndvi_percentile = 50
        else:
            ndvi_radius = ndvi_cap
            ndvi_stat = "mean"
            ndvi_percentile = 50

        # ── Evaluate Across All CV Folds ────────────────────
        fold_train_scores = []
        fold_val_scores = []
        fold_train_pvals = []
        fold_val_pvals = []

        for fold_idx, fold in enumerate(self.cv_folds):
            # Mid-trial cancellation: prune this trial immediately so the
            # study-level callback sees the next stop signal.
            cb = getattr(self, "_cancel_callback", None)
            if cb is not None and cb():
                raise optuna.TrialPruned("Cancelled by user")

            train_data = fold["train"]
            val_data = fold["val"]

            # All-zero trial guard. If every channel reports inactive (weighted-
            # average with all three weights == 0) we can't form a composite,
            # so return the metric's worst score so Optuna learns to avoid it.
            if not any(channel_active.values()):
                return -np.inf if metric != "nrmse" else np.inf

            # ── Apply Dynamic Radius and Aggregation ────
            # Both points and rasters use the same circular buffer aggregation
            # For rasters, _prepare_raster_fusion() converted pixels to points at centers.
            # Points slices, collapse codes, and every per-entity constant are
            # trial-invariant — served from the fold statics (computed once per
            # fold subset, reused by all of this bootstrap's trials).
            train_static = self._fold_entity_statics(train_data, fold_idx, "train")
            val_static = self._fold_entity_statics(val_data, fold_idx, "val")
            train_points = train_static["points"]
            val_points = val_static["points"]

            # Channel values are computed on the fold's dedup axis when it has
            # one (point / line catchments repeat a pixel per entity), then
            # gathered back onto rows just before the collapse below.
            def _channel(
                static, metric_data, radius, stat_name, pct, ch_name, sub
            ) -> tuple[np.ndarray, bool]:
                n = len(static["points"])
                if not channel_active[ch_name]:
                    uid = static.get("uniq_lookup_ids")
                    width = n if uid is None else len(uid)
                    return np.zeros(width, dtype=np.float32), uid is not None
                return self._aggregate_channel_for_fold(
                    static,
                    metric_data,
                    radius,
                    stat_name,
                    pct,
                    channel=ch_name,
                    fold_idx=fold_idx,
                    subset=sub,
                )

            train_parts = [
                _channel(
                    train_static,
                    self.veg_data,
                    veg_radius,
                    streetview_stat,
                    streetview_percentile,
                    "veg",
                    "train",
                ),
                _channel(
                    train_static,
                    self.terrain_data,
                    terrain_radius,
                    streetview_stat,
                    streetview_percentile,
                    "terrain",
                    "train",
                ),
                _channel(
                    train_static,
                    self.ndvi_data,
                    ndvi_radius,
                    ndvi_stat,
                    ndvi_percentile,
                    "ndvi",
                    "train",
                ),
            ]
            val_parts = [
                _channel(
                    val_static,
                    self.veg_data,
                    veg_radius,
                    streetview_stat,
                    streetview_percentile,
                    "veg",
                    "val",
                ),
                _channel(
                    val_static,
                    self.terrain_data,
                    terrain_radius,
                    streetview_stat,
                    streetview_percentile,
                    "terrain",
                    "val",
                ),
                _channel(
                    val_static,
                    self.ndvi_data,
                    ndvi_radius,
                    ndvi_stat,
                    ndvi_percentile,
                    "ndvi",
                    "val",
                ),
            ]
            train_on_uniq = all(u for _, u in train_parts)
            val_on_uniq = all(u for _, u in val_parts)
            train_veg, train_terrain, train_ndvi = _helpers.align_channel_axes(
                train_parts, train_static, len(train_points)
            )
            val_veg, val_terrain, val_ndvi = _helpers.align_channel_axes(
                val_parts, val_static, len(val_points)
            )

            # Per-channel scaling has been removed: the composite is always a
            # weighted combination of RAW aggregated channel values. The
            # ``whole_grid_scaling`` toggle instead normalizes the resulting
            # composite to [0, 1] (applied after compute_cgi, below).
            # Any row usable at all? Checked per channel so the full-width
            # stack (a copy of every channel) is never materialised here.
            if not (
                np.isfinite(train_veg)
                & np.isfinite(train_terrain)
                & np.isfinite(train_ndvi)
            ).any():
                continue  # Skip this fold if no valid data

            train_veg_norm = train_veg
            train_terrain_norm = train_terrain
            train_ndvi_norm = train_ndvi
            val_veg_norm = val_veg
            val_terrain_norm = val_terrain
            val_ndvi_norm = val_ndvi

            # Optional per-channel [0, 1] scaling (raw otherwise). Applied to
            # both the CGI and standalone paths so they share one footing.
            train_veg_norm, train_terrain_norm, train_ndvi_norm = (
                self._normalize_channel_arrays(
                    train_veg_norm, train_terrain_norm, train_ndvi_norm
                )
            )
            val_veg_norm, val_terrain_norm, val_ndvi_norm = (
                self._normalize_channel_arrays(
                    val_veg_norm, val_terrain_norm, val_ndvi_norm
                )
            )

            # Build the greenery value the objective scores against. In CGI
            # mode that's the selected formula's composite; in standalone mode
            # it's the active channel's normalised value used directly (the
            # role the CGI value plays in the combined run).
            if channel_mode == "cgi":
                train_components = {
                    "veg": train_veg_norm,
                    "terrain": train_terrain_norm,
                    "ndvi": train_ndvi_norm,
                }
                val_components = {
                    "veg": val_veg_norm,
                    "terrain": val_terrain_norm,
                    "ndvi": val_ndvi_norm,
                }
                train_composite = compute_cgi(
                    self.cgi_formula, formula_params, train_components
                )
                val_composite = compute_cgi(
                    self.cgi_formula, formula_params, val_components
                )
            else:
                single = {
                    "veg": (train_veg_norm, val_veg_norm),
                    "terrain": (train_terrain_norm, val_terrain_norm),
                    "ndvi": (train_ndvi_norm, val_ndvi_norm),
                }[channel_mode]
                train_composite, val_composite = single

            # Composite-level [0, 1] normalization when whole_grid_scaling is on
            # (raw otherwise) — per-pixel, before the entity collapse, and
            # applied to the standalone single-channel composite too.
            train_composite = self._finalize_composite(train_composite)
            val_composite = self._finalize_composite(val_composite)

            # Back onto the fold's rows when the composite was formed on the
            # dedup axis: the collapse below keys on per-row entity membership.
            if train_on_uniq and train_static.get("uniq_inverse") is not None:
                train_composite = np.asarray(train_composite)[
                    train_static["uniq_inverse"]
                ]
            if val_on_uniq and val_static.get("uniq_inverse") is not None:
                val_composite = np.asarray(val_composite)[val_static["uniq_inverse"]]

            # Per-fold covariate design matrix, per-entity target / coords /
            # longitudinal keys: all trial-invariant, served from the fold
            # statics (already entity-collapsed for polygon-keyed targets).
            train_cov = train_static["cov"]
            val_cov = val_static["cov"]
            train_targets_arr = train_static["target"]
            val_targets_arr = val_static["target"]
            train_coords = train_static["coords"]
            val_coords = val_static["coords"]
            train_entity_id = train_static.get("entity_id")
            val_entity_id = val_static.get("entity_id")
            train_ysb = train_static.get("ysb")
            val_ysb = val_static.get("ysb")

            # Polygon mode: per-row CGI → per-polygon mean CGI, then score
            # against the per-polygon outcome. In longitudinal+polygon mode
            # polygon_id was set to f"{entity}|{wave}" by
            # _prepare_polygon_fusion so the collapse produces one row per
            # (entity, wave) — the shape the MixedLM scorer wants. Polygons
            # average every in-footprint pixel; point/line entities average
            # only those inside this trial's catchment radius.
            if train_static["has_pid"]:
                catchment_r = _helpers.catchment_radius(
                    veg_radius,
                    terrain_radius,
                    ndvi_radius,
                    self._active_greenery_channel,
                )
                train_mask = _helpers.entity_collapse_mask(train_data, catchment_r)
                val_mask = _helpers.entity_collapse_mask(val_data, catchment_r)
                train_composite = self._collapse_mean_from_codes(
                    train_composite,
                    train_static["codes"],
                    train_static["n_uniq"],
                    train_mask,
                )
                val_composite = self._collapse_mean_from_codes(
                    val_composite,
                    val_static["codes"],
                    val_static["n_uniq"],
                    val_mask,
                )

            # Check for constant values (variance = 0) which cause NaN correlations
            train_valid_vals = train_composite[~np.isnan(train_composite)]
            val_valid_vals = val_composite[~np.isnan(val_composite)]

            if len(train_valid_vals) == 0 or np.var(train_valid_vals) == 0:
                # Prune trial early if train composite is constant
                raise optuna.TrialPruned(
                    "Train composite has no variance (constant values)"
                )

            if len(val_valid_vals) > 0 and np.var(val_valid_vals) == 0:
                # Prune trial early if validation composite is constant
                raise optuna.TrialPruned(
                    "Validation composite has no variance (constant values)"
                )

            # Spatial-confounding smooth (or None when off): a df-selected
            # coordinate basis folded into the scorer's residualization.
            train_sb = self._spatial_basis_columns(
                train_coords, train_targets_arr, train_cov, train_composite
            )
            val_sb = self._spatial_basis_columns(
                val_coords, val_targets_arr, val_cov, val_composite
            )

            # ── Score via the ``_score_greenery`` seam ──
            # Three modes collapse into it: cross-sectional OLS, mixed-effects
            # MixedLM (per-entity random effects), and year-aware cross-sectional
            # (spec present but an OLS scoring metric — the OLS scorer ignores
            # entity_id / years_since_baseline; the spec only routes the right
            # metric file per year through the pre-aggregation cache).
            wants_pval = self._metric_has_pvalue(metric)
            # Fast fixed-V GLS scoring for the search: one variance-component
            # estimate per fold subset, reused across trials. Only for the
            # ranking metrics with no spatial smooth; ``None`` elsewhere makes
            # ``_score_greenery`` take the exact per-trial fit.
            wants_fast = (
                self.is_longitudinal
                and metric in mixed_effects_scoring.FAST_SEARCH_METRICS
                and self.spatial_adjust_method == "none"
            )
            wants_gee_fast = (
                self.is_longitudinal and metric in longitudinal.GEE_LOGIT_METRICS
            )
            if wants_gee_fast:
                train_fc = self._fold_gee_baseline(train_static)
                val_fc = self._fold_gee_baseline(val_static)
            else:
                train_fc = (
                    self._fold_fast_components(train_static) if wants_fast else None
                )
                val_fc = self._fold_fast_components(val_static) if wants_fast else None
            train_out = self._score_greenery(
                metric,
                train_targets_arr,
                train_composite,
                covariates=train_cov,
                spatial_basis=train_sb,
                entity_id=train_entity_id,
                years_since_baseline=train_ysb,
                return_pvalue=wants_pval,
                fast_components=train_fc,
            )
            val_out = self._score_greenery(
                metric,
                val_targets_arr,
                val_composite,
                covariates=val_cov,
                spatial_basis=val_sb,
                entity_id=val_entity_id,
                years_since_baseline=val_ysb,
                return_pvalue=wants_pval,
                fast_components=val_fc,
            )

            if wants_pval:
                # return_pvalue=True returns (score, pvalue); type cast is for
                # the static checker.
                train_score, train_pval = train_out  # type: ignore[misc]
                val_score, val_pval = val_out  # type: ignore[misc]
                fold_train_pvals.append(train_pval)
                fold_val_pvals.append(val_pval)
            else:
                train_score = train_out
                val_score = val_out

            fold_train_scores.append(train_score)
            fold_val_scores.append(val_score)

        # Every fold was skipped (all-NaN composites) → no usable score; return
        # the metric's worst value so Optuna avoids this region (and no empty
        # np.mean warning fires).
        if not fold_val_scores:
            return -np.inf if metric != "nrmse" else np.inf

        # ── Store Aggregate Statistics ──────────────────────
        avg_train_score = np.mean(fold_train_scores)
        avg_val_score = np.mean(fold_val_scores)
        std_val_score = np.std(fold_val_scores)

        trial.set_user_attr("train_score_mean", avg_train_score)
        trial.set_user_attr("train_score_std", np.std(fold_train_scores))
        trial.set_user_attr("val_score_mean", avg_val_score)
        trial.set_user_attr("val_score_std", std_val_score)
        trial.set_user_attr("fold_train_scores", fold_train_scores)
        trial.set_user_attr("fold_val_scores", fold_val_scores)

        # p-value bookkeeping: only the longitudinal MixedLM tstat/coef metrics
        # carry a genuine Wald p-value. Recorded as user_attrs so the post-hoc
        # reporting can read them per trial.
        produced_pvals = self._metric_has_pvalue(metric)
        if produced_pvals:
            trial.set_user_attr("train_pvalue_mean", np.mean(fold_train_pvals))
            trial.set_user_attr("val_pvalue_mean", np.mean(fold_val_pvals))
            trial.set_user_attr("fold_train_pvals", fold_train_pvals)
            trial.set_user_attr("fold_val_pvals", fold_val_pvals)

        # Return average validation score across folds
        return avg_val_score

    def evaluate_on_test(
        self,
        params: dict | None = None,
        metric: str = "distance_corr",
        return_predictions: bool = False,
        *,
        return_all_mixedlm: bool = False,
    ) -> dict:
        """
        Evaluate best parameters on held-out test set.

        This is the final validation step to ensure the optimized parameters
        generalize to unseen data. Reports performance metrics and optionally
        returns predictions for further analysis.

        Args:
            params: Parameters to evaluate (uses best_params if None)
            metric: Evaluation metric ('distance_corr', 'spearman', 'r2', 'nrmse', 'mutual_info')
            return_predictions: Whether to include predictions in return dict
            return_all_mixedlm: Longitudinal mode only. When true, replaces
                ``test_score``/``test_pvalue`` in the returned dict with a
                ``mixedlm_metrics`` dict containing all four MixedLM scorer
                outputs (computed from a single fit + one null-model refit
                for the LR test). The post-hoc reporting layer uses this to
                surface every metric per trial without re-running the
                cross-sectional scoring path.

        Returns:
            Dictionary with:
                - test_score: Performance on test set
                - test_pvalue: P-value (for correlation metrics)
                - predictions: Test set predictions (if return_predictions=True)
                - targets: Test set targets (if return_predictions=True)
                - mixedlm_metrics: dict of all four scores (when
                  ``return_all_mixedlm=True`` in longitudinal mode)
        """
        if self.test_data is None:
            raise ValueError("No test data available. Run split_data() first.")

        if params is None:
            if self.best_params is None:
                raise ValueError(
                    "No parameters available. Run optimization or provide params."
                )
            params = self.best_params

        # Memoized on the composite-determining params (like apply_fusion):
        # the runner headline, the effects block, the subset-score table, and
        # the MixedLM post-score loop all re-evaluate identical configurations.
        # Invalidated by split_data. Shallow-copied on return so callers that
        # attach keys (e.g. ``test_ci``) don't mutate the cached entry.
        memo_key = (
            _helpers.fusion_param_key(self._active_greenery_channel, params),
            metric,
            bool(return_predictions),
            bool(return_all_mixedlm),
        )
        memo_hit = self._evaluate_test_cache.get(memo_key)
        if memo_hit is not None:
            return dict(memo_hit)

        logger.info("Evaluating on held-out test set...")

        # Use the engine's active mode to decide which channels contribute and
        # how the composite is built. CGI mode delegates to the formula's
        # ``channel_active`` + ``compute_cgi``; standalone mode uses the active
        # channel directly so the test score is on the same scale the study
        # optimized against.
        channel_mode = self._active_greenery_channel
        if channel_mode == "cgi":
            channel_active = cgi_formulas.get_formula(self.cgi_formula).channel_active(
                params
            )
        else:
            channel_active = {
                "veg": channel_mode == "veg",
                "terrain": channel_mode == "terrain",
                "ndvi": channel_mode == "ndvi",
            }

        # Extract aggregation parameters
        streetview_stat = params.get("streetview_stat", "mean")
        streetview_percentile = params.get("streetview_percentile", 50)
        veg_radius = params.get("veg_radius", int(round(self.gvi_buffer_max_m)))
        terrain_radius = params.get("terrain_radius", int(round(self.gvi_buffer_max_m)))
        ndvi_stat = params.get("ndvi_stat", "mean")
        ndvi_percentile = params.get("ndvi_percentile", 50)
        ndvi_radius = params.get("ndvi_radius", int(round(self.ndvi_buffer_max_m)))

        # Apply dynamic circular buffer aggregation
        # Both points and rasters use the same approach (rasters converted to points)
        # .loc with an index array materialises a new frame; read-only below.
        test_points = self.target_gdf.loc[self.test_data.index]

        # Sample vegetation with optimized radius/stat
        if channel_active["veg"]:
            test_veg = self._aggregate_with_ring_cache(
                test_points,
                self.veg_data,
                veg_radius,
                streetview_stat,
                streetview_percentile,
                channel="veg",
                fold_idx=-1,
                subset="test",
            )
        else:
            test_veg = np.zeros(len(test_points), dtype=np.float32)

        # Sample terrain with optimized radius/stat
        if channel_active["terrain"]:
            test_terrain = self._aggregate_with_ring_cache(
                test_points,
                self.terrain_data,
                terrain_radius,
                streetview_stat,
                streetview_percentile,
                channel="terrain",
                fold_idx=-1,
                subset="test",
            )
        else:
            test_terrain = np.zeros(len(test_points), dtype=np.float32)

        # Sample NDVI with optimized radius/stat
        if channel_active["ndvi"]:
            test_ndvi = self._aggregate_with_ring_cache(
                test_points,
                self.ndvi_data,
                ndvi_radius,
                ndvi_stat,
                ndvi_percentile,
                channel="ndvi",
                fold_idx=-1,
                subset="test",
            )
        else:
            test_ndvi = np.zeros(len(test_points), dtype=np.float32)

        # Per-channel scaling removed — the composite uses raw aggregated
        # channels. ``whole_grid_scaling`` normalizes the composite below.
        test_combined = np.column_stack([test_veg, test_terrain, test_ndvi])
        test_valid_mask = ~np.isnan(test_combined).any(axis=1)

        if test_valid_mask.sum() == 0:
            raise ValueError("No valid test data after aggregation")

        test_veg_norm = test_combined[:, 0]
        test_terrain_norm = test_combined[:, 1]
        test_ndvi_norm = test_combined[:, 2]
        test_veg_norm, test_terrain_norm, test_ndvi_norm = (
            self._normalize_channel_arrays(
                test_veg_norm, test_terrain_norm, test_ndvi_norm
            )
        )

        # Calculate composite via the active mode. Same fork as ``_objective``
        # so the held-out score is on the same scale the study optimized.
        if channel_mode == "cgi":
            test_composite = compute_cgi(
                self.cgi_formula,
                params,
                {
                    "veg": test_veg_norm,
                    "terrain": test_terrain_norm,
                    "ndvi": test_ndvi_norm,
                },
            )
        else:
            test_composite = {
                "veg": test_veg_norm,
                "terrain": test_terrain_norm,
                "ndvi": test_ndvi_norm,
            }[channel_mode]

        # Composite-level [0, 1] normalization (raw when the toggle is off),
        # mirrored for standalone single-channel composites.
        test_composite = self._finalize_composite(test_composite)

        # Per-entity constants (covariates, target, coords, longitudinal keys)
        # come from the fold statics — identical across every params
        # evaluation on this test split, so the MixedLM post-score loop and
        # the reporting passes stop re-collapsing the pixel frame per call.
        test_static = self._fold_entity_statics(self.test_data, -1, "test")
        test_cov = test_static["cov"]
        test_targets = test_static["target"]
        test_coords = test_static["coords"]
        test_entity_id = test_static.get("entity_id")
        test_ysb = test_static.get("ysb")

        # Polygon mode: aggregate per-row CGI by polygon before scoring against
        # the per-polygon outcome. Longitudinal+polygon mode keys the collapse
        # on f"{entity}|{wave}" so the result is one row per (entity, wave) —
        # the shape the MixedLM scorer expects.
        if test_static["has_pid"]:
            catchment_r = _helpers.catchment_radius(
                veg_radius,
                terrain_radius,
                ndvi_radius,
                self._active_greenery_channel,
            )
            test_mask = _helpers.entity_collapse_mask(self.test_data, catchment_r)
            test_composite = self._collapse_mean_from_codes(
                test_composite,
                test_static["codes"],
                test_static["n_uniq"],
                test_mask,
            )

        test_sb = self._spatial_basis_columns(
            test_coords, test_targets, test_cov, test_composite
        )

        # ── Score via the ``_score_greenery`` seam ──────────
        # Same routing as ``_objective``: MixedLM for a longitudinal mixedlm_*
        # metric, OLS otherwise (a year-aware cross-sectional study sits on a
        # spec whose scoring_metric is an OLS option and takes the OLS path).
        mixedlm_all: dict[str, float] | None = None
        wants_pval = self._metric_has_pvalue(metric)
        if (
            return_all_mixedlm
            and self.is_longitudinal
            and metric in mixed_effects_scoring.MIXEDLM_METRICS
        ):
            mixedlm_all = self._score_greenery(  # type: ignore[assignment]
                metric,
                test_targets,
                test_composite,
                covariates=test_cov,
                spatial_basis=test_sb,
                entity_id=test_entity_id,
                years_since_baseline=test_ysb,
                return_all=True,
            )
            # Surface the requested metric's value alongside the dict so
            # ``test_score`` still reflects the engine's active scoring metric.
            s = float(mixedlm_all.get(metric, 0.0))  # type: ignore[union-attr]
            score_out = (s, 1.0) if wants_pval else s
        else:
            score_out = self._score_greenery(
                metric,
                test_targets,
                test_composite,
                covariates=test_cov,
                spatial_basis=test_sb,
                entity_id=test_entity_id,
                years_since_baseline=test_ysb,
                return_pvalue=wants_pval,
            )
        if wants_pval:
            test_score, test_pval = score_out  # type: ignore[misc]
        else:
            test_score = float(score_out)
            test_pval = None

        result = {"test_score": test_score, "metric": metric}
        if self.spatial_adjust_method != "none":
            result["spatial_adjustment"] = self._spatial_adjust_summary or {
                "method": self.spatial_adjust_method,
                "applied": False,
            }
        if test_pval is not None:
            result["test_pvalue"] = test_pval
            logger.info(
                f"Test {metric}: {test_score:.4f} (p={test_pval:.4e}, n={len(test_targets)})"
            )
        else:
            logger.info(f"Test {metric}: {test_score:.4f} (n={len(test_targets)})")

        if mixedlm_all is not None:
            result["mixedlm_metrics"] = mixedlm_all

        # Include predictions if requested. Covariates ride alongside so the
        # downstream permutation / bootstrap helpers can score the same
        # partial-correlation the optimizer optimized.
        if return_predictions:
            result["predictions"] = test_composite
            result["targets"] = test_targets
            result["covariates"] = test_cov
            # The df-selected smooth so downstream bootstrap / permutation CIs can
            # condition on the same spatial adjustment the score used.
            result["spatial_basis"] = test_sb
            # Longitudinal keys so the held-out CI can cluster-resample entities
            # (and refit the MixedLM per replicate). ``None`` in cross-sectional
            # mode; the OLS reporting path ignores them.
            if self.is_longitudinal:
                result["entity_id"] = test_entity_id
                result["years_since_baseline"] = test_ysb
                result["wave"] = test_static.get("wave")

        self._evaluate_test_cache[memo_key] = result
        while len(self._evaluate_test_cache) > 256:
            self._evaluate_test_cache.pop(next(iter(self._evaluate_test_cache)))
        return dict(result)

    def build_channel_design(self, params: dict, subset: str = "train_val") -> dict:
        """Per-entity raw channel values for the CGI-vs-standalone AIC/BIC test.

        Aggregates **all three** channels (veg / terrain / ndvi) at ``params``'
        radii / stats / percentiles on the requested ``subset`` (``"train_val"``
        for the resampling pool, ``"test"`` for the held-out set, ``"all"`` for
        every entity), collapses to one row per entity (polygon mean when
        polygon-keyed), and returns the raw arrays plus the aligned target,
        covariates, and — in longitudinal mode — entity ids and
        ``years_since_baseline``.

        No scaling is applied: OLS / MixedLM AIC and BIC are invariant to an
        affine transform of an individual predictor, so raw channel values are
        what :func:`objective_scoring.compare_models_aic_bic` and its MixedLM
        analogue need.
        """
        if subset == "test":
            data = self.test_data
        elif subset == "train_val":
            data = self.train_val_data
        elif subset == "all":
            parts = [
                d
                for d in (self.train_val_data, self.test_data)
                if d is not None and len(d) > 0
            ]
            data = pd.concat(parts) if parts else None
        else:
            raise ValueError(
                f"subset must be 'train_val', 'test', or 'all'; got {subset!r}."
            )
        if data is None:
            raise ValueError(f"No {subset} data available. Run split_data() first.")

        points = self.target_gdf.loc[data.index].copy()
        streetview_stat = params.get("streetview_stat", "mean")
        streetview_percentile = params.get("streetview_percentile", 50)
        veg_radius = params.get("veg_radius", int(round(self.gvi_buffer_max_m)))
        terrain_radius = params.get("terrain_radius", int(round(self.gvi_buffer_max_m)))
        ndvi_stat = params.get("ndvi_stat", "mean")
        ndvi_percentile = params.get("ndvi_percentile", 50)
        ndvi_radius = params.get("ndvi_radius", int(round(self.ndvi_buffer_max_m)))

        veg = self._aggregate_with_ring_cache(
            points,
            self.veg_data,
            veg_radius,
            streetview_stat,
            streetview_percentile,
            channel="veg",
            fold_idx=None,
            subset=None,
        )
        terrain = self._aggregate_with_ring_cache(
            points,
            self.terrain_data,
            terrain_radius,
            streetview_stat,
            streetview_percentile,
            channel="terrain",
            fold_idx=None,
            subset=None,
        )
        ndvi = self._aggregate_with_ring_cache(
            points,
            self.ndvi_data,
            ndvi_radius,
            ndvi_stat,
            ndvi_percentile,
            channel="ndvi",
            fold_idx=None,
            subset=None,
        )

        cov_cols = self.covariate_columns
        cov = data[cov_cols].to_numpy(dtype=np.float64) if cov_cols else None

        entity_id = None
        ysb = None
        if "polygon_id" in data.columns:
            pid = data["polygon_id"].values
            # Full 3-channel design → catchment = max of the three radii for
            # point/line targets (mask None for polygons).
            catchment_r = _helpers.catchment_radius(
                veg_radius,
                terrain_radius,
                ndvi_radius,
                self._active_greenery_channel,
            )
            mask = _helpers.entity_collapse_mask(data, catchment_r)
            veg = self._collapse_to_entities(veg, pid, mask, "mean")
            terrain = self._collapse_to_entities(terrain, pid, mask, "mean")
            ndvi = self._collapse_to_entities(ndvi, pid, mask, "mean")
            target = self._collapse_to_entities(
                data["target"].values, pid, mask, "first"
            )
            if cov is not None:
                c = cov
                cov = np.column_stack(
                    [
                        self._collapse_to_entities(c[:, j], pid, mask, "first")
                        for j in range(c.shape[1])
                    ]
                )
            if self.is_longitudinal:
                entity_id = self._collapse_to_entities(
                    data["entity_id"].values, pid, mask, "first"
                )
                ysb = self._collapse_to_entities(
                    data["years_since_baseline"].values, pid, mask, "first"
                )
        else:
            target = data["target"].values
            if self.is_longitudinal:
                entity_id = data["entity_id"].values
                ysb = data["years_since_baseline"].values

        return {
            "channels": np.column_stack([veg, terrain, ndvi]),
            "channel_names": ["veg", "terrain", "ndvi"],
            "target": np.asarray(target, dtype=np.float64),
            "covariates": cov,
            "entity_id": entity_id,
            "years_since_baseline": ysb,
        }

    def bootstrap_test_score_ci(
        self,
        params: dict,
        metric: str,
        *,
        n_bootstrap: int = 10000,
        ci_level: float = 0.95,
        method: str = "percentile",
        seed: int = 42,
    ) -> dict:
        """Bootstrap CI for the test-set score.

        Builds the test composite from ``params`` via :meth:`evaluate_on_test`,
        then resamples paired ``(target, composite, covariates)`` rows to
        construct a percentile CI on the chosen metric. When the engine has
        ``covariate_columns`` configured, the test covariate matrix is resampled
        jointly with target/prediction so the CI is on the **partial**
        association (the same quantity the optimizer optimized). The score
        function is taken from :func:`objective_scoring.score`, which switches
        between full and partial residual scoring based on whether covariates
        are supplied.

        Longitudinal runs resample whole entities (cluster bootstrap) so the
        panel correlation isn't broken: an OLS-metric run passes ``groups`` to
        :func:`statistical_testing.bootstrap_score_ci`, while a ``mixedlm_*``
        metric routes to :func:`mixed_effects_scoring.cluster_bootstrap_metric_ci`,
        which refits the mixed model per (entity-relabelled) replicate at a
        capped replicate count.

        Returns a dict with ``observed``, ``mean``, ``lower``, ``upper``,
        ``ci_level``, ``method``, ``n``.
        """
        from . import statistical_testing as _stats_mod

        res = self.evaluate_on_test(
            params=params, metric=metric, return_predictions=True
        )
        target = np.asarray(res.get("targets"), dtype=np.float64)
        prediction = np.asarray(res.get("predictions"), dtype=np.float64)

        # Longitudinal MixedLM metric → cluster (entity) bootstrap that refits
        # the mixed model per replicate (resampling rows independently would
        # break the within-entity correlation).
        if self.is_longitudinal and metric in mixed_effects_scoring.MIXEDLM_METRICS:
            spec = self.longitudinal_spec
            assert spec is not None
            return mixed_effects_scoring.cluster_bootstrap_metric_ci(
                metric,
                target,
                prediction,
                res.get("entity_id"),
                res.get("years_since_baseline"),
                covariates=res.get("covariates"),
                include_time_fixed=self._effective_time_fixed(spec),
                random_slope=spec.random_slope_time,
                spatial_basis=res.get("spatial_basis"),
                spatial_method=self.spatial_adjust_method,
                n_bootstrap=min(int(n_bootstrap), self._MIXEDLM_TEST_CI_BOOTSTRAP_CAP),
                ci_level=float(ci_level),
                seed=int(seed),
                target=spec.association_target,
            )

        test_cov = res.get("covariates")
        cov_mat = (
            np.asarray(test_cov, dtype=np.float64) if test_cov is not None else None
        )
        # The smooth rides in the resampled control matrix so its rows are drawn
        # with the observation, then is split back out per replicate and handed
        # to the scorer through ``spatial_basis`` — the same seam the objective
        # uses, so the CI conditions exactly as the score being bounded did.
        n_cov_cols = 0 if cov_mat is None else int(cov_mat.shape[1])
        sb = res.get("spatial_basis")
        if sb is not None:
            sb = np.asarray(sb, dtype=np.float64)
            cov_mat = sb if cov_mat is None else np.column_stack([cov_mat, sb])
        mask = np.isfinite(target) & np.isfinite(prediction)
        if cov_mat is not None:
            mask &= np.isfinite(cov_mat).all(axis=1)
        n = int(mask.sum())

        if cov_mat is None:

            def _score_fn(t_arr: np.ndarray, p_arr: np.ndarray) -> float:
                score_out = objective_scoring.score(
                    metric,
                    t_arr,
                    p_arr,
                    covariates=None,
                    return_pvalue=False,
                    residualize_method=self.residualize_method,
                    pdcor_cache=self._pdcor_cache,
                    spline_cache=self._spline_basis_cache,
                )
                return float(score_out)  # type: ignore[arg-type]

        else:

            def _score_fn(
                t_arr: np.ndarray, p_arr: np.ndarray, c_arr: np.ndarray
            ) -> float:
                cov_part, sb_part = _helpers.split_control_matrix(c_arr, n_cov_cols)
                score_out = objective_scoring.score(
                    metric,
                    t_arr,
                    p_arr,
                    covariates=cov_part,
                    return_pvalue=False,
                    spatial_basis=sb_part,
                    spatial_method=self.spatial_adjust_method,
                    residualize_method=self.residualize_method,
                    pdcor_cache=self._pdcor_cache,
                    spline_cache=self._spline_basis_cache,
                )
                return float(score_out)  # type: ignore[arg-type]

        # OLS-metric longitudinal runs still have panel-correlated (entity, wave)
        # rows, so resample whole entities (cluster bootstrap) rather than rows;
        # cross-sectional runs pass ``groups=None`` and keep the row bootstrap.
        groups = None
        if self.is_longitudinal and res.get("entity_id") is not None:
            groups = np.asarray(res.get("entity_id"))

        ci = _stats_mod.bootstrap_score_ci(
            target,
            prediction,
            score_fn=_score_fn,
            n_bootstrap=int(n_bootstrap),
            ci_level=float(ci_level),
            method=method,
            seed=int(seed),
            covariates=cov_mat,
            replicate_scorer_factory=(
                pdcor.pdcor_replicate_scorer_factory
                if metric == "partial_distance_corr"
                else None
            ),
            groups=groups,
            cancel_callback=self._cancel_callback,
        )
        ci["n"] = n
        return ci

    def _full_data_frame(self) -> "pd.DataFrame | None":
        """train+val+test union frame, concatenated once and cached.

        Several report helpers need every entity's rows; rebuilding the concat
        each time is wasteful. Invalidated by :meth:`split_data`.
        """
        if self._full_data_cache is None:
            parts = [d for d in (self.train_val_data, self.test_data) if d is not None]
            self._full_data_cache = pd.concat(parts) if parts else None
        return self._full_data_cache

    def apply_fusion(self, weights: dict | None = None) -> pd.DataFrame:
        """
        Apply fusion weights to create composite index.

        Args:
            weights: Dictionary with 'veg_weight', 'terrain_weight', 'ndvi_weight' (0-100 scale)
                    and aggregation parameters (radii, stats, percentiles)
                    If None, uses best_params from optimization

        Returns:
            DataFrame with target, veg, terrain, ndvi, and composite columns

        The result is memoized on ``(active channel, composite params)`` so the
        reporting layer — which applies the same winning params on several
        subsets — recomputes it at most once. The cache is invalidated by
        :meth:`split_data`.
        """
        if weights is None:
            if self.best_params is None:
                raise ValueError(
                    "No weights available. Run optimization or provide weights."
                )
            weights = self.best_params

        cache_key = _helpers.fusion_param_key(self._active_greenery_channel, weights)
        cached = self._apply_fusion_cache
        if cached is not None and cached[0] == cache_key:
            return cached[1]

        # The active mode decides which channels contribute and how the
        # composite is built — apply_fusion routes through the same fork as
        # _objective / evaluate_on_test so all four agree. CGI mode delegates
        # to the formula; standalone mode isolates the active channel.
        channel_mode = self._active_greenery_channel
        if channel_mode == "cgi":
            channel_active = cgi_formulas.get_formula(self.cgi_formula).channel_active(
                weights
            )
        else:
            channel_active = {
                "veg": channel_mode == "veg",
                "terrain": channel_mode == "terrain",
                "ndvi": channel_mode == "ndvi",
            }

        # Extract aggregation parameters
        streetview_stat = weights.get("streetview_stat", "mean")
        streetview_percentile = weights.get("streetview_percentile", 50)
        veg_radius = weights.get("veg_radius", int(round(self.gvi_buffer_max_m)))
        terrain_radius = weights.get(
            "terrain_radius", int(round(self.gvi_buffer_max_m))
        )
        ndvi_stat = weights.get("ndvi_stat", "mean")
        ndvi_percentile = weights.get("ndvi_percentile", 50)
        ndvi_radius = weights.get("ndvi_radius", int(round(self.ndvi_buffer_max_m)))

        # Combine all data (train+val+test) — cached union frame.
        all_data = self._full_data_frame()
        if all_data is None:
            raise ValueError("No data available. Run split_data() first.")
        # .loc with an index array materialises a new frame; read-only below.
        all_points = self.target_gdf.loc[all_data.index]

        # Apply circular buffer aggregation with optimized parameters
        if channel_active["veg"]:
            all_veg = self._aggregate_with_ring_cache(
                all_points,
                self.veg_data,
                veg_radius,
                streetview_stat,
                streetview_percentile,
                channel="veg",
                fold_idx=-1,
                subset="all",
            )
        else:
            all_veg = np.zeros(len(all_points), dtype=np.float32)

        if channel_active["terrain"]:
            all_terrain = self._aggregate_with_ring_cache(
                all_points,
                self.terrain_data,
                terrain_radius,
                streetview_stat,
                streetview_percentile,
                channel="terrain",
                fold_idx=-1,
                subset="all",
            )
        else:
            all_terrain = np.zeros(len(all_points), dtype=np.float32)

        if channel_active["ndvi"]:
            all_ndvi = self._aggregate_with_ring_cache(
                all_points,
                self.ndvi_data,
                ndvi_radius,
                ndvi_stat,
                ndvi_percentile,
                channel="ndvi",
                fold_idx=-1,
                subset="all",
            )
        else:
            all_ndvi = np.zeros(len(all_points), dtype=np.float32)

        # Per-channel scaling removed — the composite uses raw aggregated
        # channel values.
        all_combined = np.column_stack([all_veg, all_terrain, all_ndvi])
        all_veg_norm = all_combined[:, 0]
        all_terrain_norm = all_combined[:, 1]
        all_ndvi_norm = all_combined[:, 2]
        all_veg_norm, all_terrain_norm, all_ndvi_norm = self._normalize_channel_arrays(
            all_veg_norm, all_terrain_norm, all_ndvi_norm
        )

        # Calculate composite via the active mode (same fork as _objective /
        # evaluate_on_test). CGI mode uses the formula; standalone mode takes
        # the single active channel directly.
        if channel_mode == "cgi":
            composite = compute_cgi(
                self.cgi_formula,
                weights,
                {
                    "veg": all_veg_norm,
                    "terrain": all_terrain_norm,
                    "ndvi": all_ndvi_norm,
                },
            )
        else:
            composite = {
                "veg": all_veg_norm,
                "terrain": all_terrain_norm,
                "ndvi": all_ndvi_norm,
            }[channel_mode]
        # Composite-level [0, 1] normalization over the whole grid when the
        # toggle is on (raw otherwise); mirrors the output raster + standalones.
        composite = self._finalize_composite(composite)

        result_df = all_data.copy()
        result_df["veg"] = all_veg
        result_df["terrain"] = all_terrain
        result_df["ndvi"] = all_ndvi
        result_df["composite"] = composite

        # Per-pixel mode: collapse per-pixel rows into one row per entity.
        # Each entity's composite is the mean of per-pixel CGIs in its
        # catchment (the whole footprint for polygons; the pixels within the
        # trial's catchment radius for point/line targets), matching what the
        # optimizer scored against the per-entity outcome.
        if "polygon_id" in result_df.columns:
            catchment_r = _helpers.catchment_radius(
                veg_radius,
                terrain_radius,
                ndvi_radius,
                self._active_greenery_channel,
            )
            mask = _helpers.entity_collapse_mask(result_df, catchment_r)
            masked_df = result_df if mask is None else result_df[mask]
            poly_df = (
                masked_df.groupby("polygon_id", sort=False)
                .agg(
                    target=("target", "first"),
                    veg=("veg", "mean"),
                    terrain=("terrain", "mean"),
                    ndvi=("ndvi", "mean"),
                    composite=("composite", "mean"),
                    n_samples=("composite", "count"),
                )
                .reset_index()
            )
            self._apply_fusion_cache = (cache_key, poly_df)
            return poly_df

        self._apply_fusion_cache = (cache_key, result_df)
        return result_df

    def _score_data_subset(
        self,
        data: "pd.DataFrame | None",
        params: dict,
        metric: str,
    ) -> dict[str, float | None]:
        """Score the winning params on an arbitrary data slice — both
        covariate-adjusted (partial) and unadjusted (raw).

        Used by ``compute_subset_scores`` for train+val / test / all blocks
        so the UI can surface BOTH the optimizer's actual objective
        (partial when covariates exist) AND the unadjusted correlation —
        they're complementary and answer different questions:

        * ``score`` (partial) — how much does CGI add **over and above**
          the covariates? Answers "is greenery doing real work?"
        * ``score_raw`` — what does CGI predict on its own? Answers "how
          predictive is the composite without controls?"

        Returns ``{"score", "score_raw", "pvalue", "pvalue_raw", "fit_failed"}``.
        ``score_raw`` collapses to ``score`` when no covariates are
        configured (they're the same number in that case). A longitudinal mixed
        model that doesn't converge yields ``None`` and ``fit_failed=True``
        rather than the scorer's degenerate ``0.0``, which reads as a genuine
        null effect everywhere downstream.
        """
        empty = {
            "score": None,
            "score_raw": None,
            "pvalue": None,
            "pvalue_raw": None,
            "fit_failed": False,
        }
        if data is None or len(data) == 0:
            return empty
        try:
            df_full = self.apply_fusion(weights=dict(params))
            if "polygon_id" in df_full.columns:
                target_polys = set(data["polygon_id"].unique().tolist())
                df = df_full[df_full["polygon_id"].isin(target_polys)].copy()
            else:
                # Row-keyed targets: ``apply_fusion`` returns rows in the
                # same order as ``self.train_val_data + self.test_data``.
                # Restrict to the subset's indices.
                df = df_full.loc[df_full.index.intersection(data.index)].copy()
            if len(df) == 0:
                return empty
            target = np.asarray(df["target"].values, dtype=np.float64)
            composite = np.asarray(df["composite"].values, dtype=np.float64)
            cov_mat: np.ndarray | None = None
            cov_cols = self.covariate_columns or []
            full = self._full_data_frame()
            if cov_cols and full is not None and "polygon_id" in df.columns:
                cov_per_poly = (
                    full.groupby("polygon_id", sort=False)[cov_cols]
                    .first()
                    .reindex(df["polygon_id"].values)
                )
                cov_mat = cov_per_poly.to_numpy(dtype=np.float64)
            # In-sample slices carry no p-value (the params were tuned on this
            # pool); the honest p comes from the held-out test elsewhere.
            wants_pval = False

            # Spatial smooth for the partial (adjusted) score; the raw score
            # stays fully unadjusted (no covariates, no smooth).
            sb = self._spatial_basis_columns(
                self._whole_data_coords(df), target, cov_mat, composite
            )
            # Longitudinal keys for the MixedLM branch of the scoring seam
            # (``None`` in cross-sectional mode, ignored by the OLS scorer).
            lon_eid, lon_ysb = self._whole_data_longitudinal_keys(df)

            def _do(
                cov: np.ndarray | None, spat: np.ndarray | None
            ) -> tuple[float | None, float | None]:
                out = self._score_greenery(
                    metric,
                    target,
                    composite,
                    covariates=cov,
                    spatial_basis=spat,
                    entity_id=lon_eid,
                    years_since_baseline=lon_ysb,
                    return_pvalue=wants_pval,
                    nan_on_fail=True,
                )
                if wants_pval:
                    s, p = out  # type: ignore[misc]
                    return _finite_or_none(s), _finite_or_none(p)
                return _finite_or_none(out), None  # type: ignore[arg-type]

            partial_s, partial_p = _do(cov_mat, sb)
            raw_s, raw_p = (
                _do(None, None)
                if (cov_mat is not None or sb is not None)
                else (partial_s, partial_p)
            )
            return {
                "score": partial_s,
                "score_raw": raw_s,
                "pvalue": partial_p,
                "pvalue_raw": raw_p,
                "fit_failed": partial_s is None,
            }
        except Exception as exc:
            logger.warning(f"_score_data_subset failed: {exc}")
            return empty

    def _whole_data_covariates(self, df: "pd.DataFrame") -> "np.ndarray | None":
        """Per-row covariate matrix aligned to a collapsed ``apply_fusion`` frame.

        Looks each row's covariates up by ``polygon_id`` from the union of
        train+val and test data (covariates are a property of the entity, not
        the split). Returns ``None`` when no covariates are configured or the
        frame isn't polygon-keyed.
        """
        cov_cols = self.covariate_columns or []
        if not cov_cols or "polygon_id" not in df.columns:
            return None
        full = self._full_data_frame()
        if full is None:
            return None
        cov_per_poly = (
            full.groupby("polygon_id", sort=False)[cov_cols]
            .first()
            .reindex(df["polygon_id"].values)
        )
        return cov_per_poly.to_numpy(dtype=np.float64)

    def _whole_data_coords(self, df: "pd.DataFrame") -> "np.ndarray | None":
        """Per-row entity coordinates aligned to a collapsed ``apply_fusion`` frame.

        Mirrors :meth:`_whole_data_covariates` for the spatial smooth: looks each
        row's ``_cx`` / ``_cy`` up by ``polygon_id`` (or by index for row-keyed
        targets) from the union of train+val and test data. Returns ``None`` when
        spatial adjustment is off or coordinates were not attached.
        """
        if self.spatial_adjust_method == "none":
            return None
        full = self._full_data_frame()
        if full is None:
            return None
        if "_cx" not in full.columns or "_cy" not in full.columns:
            return None
        if "polygon_id" in df.columns and "polygon_id" in full.columns:
            coords = (
                full.groupby("polygon_id", sort=False)[["_cx", "_cy"]]
                .first()
                .reindex(df["polygon_id"].values)
            )
        else:
            coords = full[["_cx", "_cy"]].reindex(df.index)
        return coords.to_numpy(dtype=np.float64)

    def _whole_data_longitudinal_keys(
        self, df: "pd.DataFrame"
    ) -> "tuple[np.ndarray | None, np.ndarray | None]":
        """Per-row ``(entity_id, years_since_baseline)`` for a collapsed frame.

        Mirrors :meth:`_whole_data_covariates` for the mixed-effects scorer: reads
        the longitudinal keys directly from a row-keyed frame, or looks them up by
        ``polygon_id`` from the train+val+test union for a polygon-collapsed
        ``apply_fusion`` frame. Returns ``(None, None)`` in cross-sectional mode or
        when the keys are absent, so the OLS scoring path is unaffected.
        """
        if not self.is_longitudinal:
            return None, None
        keys = ["entity_id", "years_since_baseline"]
        if set(keys) <= set(df.columns):
            return (
                df["entity_id"].to_numpy(),
                df["years_since_baseline"].to_numpy(dtype=np.float64),
            )
        full = self._full_data_frame()
        if (
            full is None
            or "polygon_id" not in df.columns
            or not set(keys) <= set(full.columns)
        ):
            return None, None
        lk = (
            full.groupby("polygon_id", sort=False)[keys]
            .first()
            .reindex(df["polygon_id"].values)
        )
        return (
            lk["entity_id"].to_numpy(),
            lk["years_since_baseline"].to_numpy(dtype=np.float64),
        )

    def _augment_cov_with_spatial(
        self,
        df: "pd.DataFrame",
        target: np.ndarray,
        composite: np.ndarray,
        cov: np.ndarray | None,
    ) -> tuple[np.ndarray | None, int]:
        """Stack the df-selected spatial smooth onto a control matrix.

        Returns ``(matrix, n_covariate_columns)`` so the caller can split the
        smooth back out per replicate (see :meth:`_split_control_matrix`) and
        hand it to the scorer through the same ``spatial_basis`` seam the
        objective uses. A no-op when spatial adjustment is off or the geometry
        is degenerate.
        """
        n_cov_cols = 0 if cov is None else int(np.asarray(cov).shape[1])
        if self.spatial_adjust_method == "none":
            return cov, n_cov_cols
        sb = self._spatial_basis_columns(
            self._whole_data_coords(df), target, cov, composite
        )
        if sb is None:
            return cov, n_cov_cols
        stacked = sb if cov is None else np.column_stack([cov, sb])
        return stacked, n_cov_cols

    def evaluate_effects(
        self,
        params: dict,
        metric: str,
        *,
        n_bootstrap: int = 2000,
        n_perm: int = 1000,
        ci_level: float = 0.95,
        seed: int = 42,
    ) -> dict[str, dict]:
        """Objective effect on the held-out test set (headline) + descriptive
        per-subset effects.

        The ``test`` slice is the headline: the stability-selected params never
        saw it, so it carries the permutation p-value — the honest
        generalizability check. The ``all`` (whole-data) and ``train_val``
        slices are descriptive (CI only, no p-value): the params were tuned on
        the train+val pool, so a p-value there would be optimistic
        (double-dipping). Each entry is ``{score, lower, upper, p_value, n}``.
        Defined for cross-sectional objective metrics; returns ``{}`` for metrics
        it doesn't support (e.g. longitudinal MixedLM metrics, scored elsewhere).
        """
        if metric not in objective_scoring.SUPPORTED_METRICS:
            if self.is_longitudinal and metric in mixed_effects_scoring.MIXEDLM_METRICS:
                return self._evaluate_effects_mixedlm(
                    params,
                    metric,
                    n_bootstrap=n_bootstrap,
                    ci_level=ci_level,
                    seed=seed,
                )
            return {}
        from . import statistical_testing as _stats_mod

        higher = metric in objective_scoring.HIGHER_IS_BETTER

        def _make_score_fn(n_cov_cols: int):
            """Scorer over a resampled ``[covariates | smooth]`` control matrix."""

            def _score_fn(t_arr, c_arr, cov=None):
                cov_part, sb_part = _helpers.split_control_matrix(cov, n_cov_cols)
                return objective_scoring.score(
                    metric,
                    t_arr,
                    c_arr,
                    cov_part,
                    spatial_basis=sb_part,
                    spatial_method=self.spatial_adjust_method,
                    residualize_method=self.residualize_method,
                    pdcor_cache=self._pdcor_cache,
                    spline_cache=self._spline_basis_cache,
                )

            return _score_fn

        # Replicate fast paths for the O(n²) metric: bootstrap replicates
        # fancy-index precomputed distance matrices; permutation replicates
        # reuse the fixed prediction/conditioning sides (Freedman–Lane only
        # varies the surrogate outcome).
        is_pdcor = metric == "partial_distance_corr"
        rep_factory = pdcor.pdcor_replicate_scorer_factory if is_pdcor else None
        sur_factory = pdcor.pdcor_surrogate_scorer_factory if is_pdcor else None

        def _block(t_arr, c_arr, cov, *, do_perm, sub_seed, n_cov_cols=0):
            out = {
                "score": None,
                "lower": None,
                "upper": None,
                "p_value": None,
                "p_kind": "permutation" if do_perm else None,
                "n": None,
            }
            if t_arr is None or c_arr is None or len(t_arr) < 3:
                return out
            _score_fn = _make_score_fn(n_cov_cols)
            try:
                ci = _stats_mod.bootstrap_score_ci(
                    t_arr,
                    c_arr,
                    score_fn=_score_fn,
                    n_bootstrap=int(n_bootstrap),
                    ci_level=ci_level,
                    method="percentile",
                    seed=int(sub_seed),
                    covariates=cov,
                    replicate_scorer_factory=rep_factory,
                    cancel_callback=self._cancel_callback,
                )
                out["score"] = ci.get("observed")
                out["lower"] = ci.get("lower")
                out["upper"] = ci.get("upper")
            except Exception as exc:
                logger.warning(f"evaluate_effects: CI failed: {exc}")
            if do_perm:
                try:
                    perm = _stats_mod.permutation_pvalue(
                        t_arr,
                        c_arr,
                        score_fn=_score_fn,
                        higher_is_better=higher,
                        n_perm=int(n_perm),
                        seed=int(sub_seed) + 7,
                        covariates=cov,
                        surrogate_scorer_factory=sur_factory,
                        cancel_callback=self._cancel_callback,
                    )
                    out["p_value"] = perm.get("p_value")
                    if out["score"] is None:
                        out["score"] = perm.get("observed")
                except Exception as exc:
                    logger.warning(f"evaluate_effects: permutation failed: {exc}")
            out["n"] = int(len(t_arr))
            return out

        results: dict[str, dict] = {}
        df_full = None
        try:
            df_full = self.apply_fusion(weights=dict(params))
            t_all = np.asarray(df_full["target"].values, dtype=np.float64)
            c_all = np.asarray(df_full["composite"].values, dtype=np.float64)
            cov_all, n_cov_all = self._augment_cov_with_spatial(
                df_full, t_all, c_all, self._whole_data_covariates(df_full)
            )
            # Whole-data effect is descriptive only (CI, no permutation p): the
            # params were tuned on most of these rows, so a p-value here would
            # double-dip. The held-out ``test`` block below carries the p-value.
            results["all"] = _block(
                t_all,
                c_all,
                cov_all,
                do_perm=False,
                sub_seed=seed,
                n_cov_cols=n_cov_all,
            )
        except Exception as exc:
            logger.warning(f"evaluate_effects: whole-data scoring failed: {exc}")

        try:
            if self.test_data is not None and len(self.test_data) > 0:
                tr = self.evaluate_on_test(
                    params=dict(params), metric=metric, return_predictions=True
                )
                tg = tr.get("targets")
                pr = tr.get("predictions")
                cv = tr.get("covariates")
                t_te = None if tg is None else np.asarray(tg, dtype=np.float64)
                c_te = None if pr is None else np.asarray(pr, dtype=np.float64)
                cov_te = None if cv is None else np.asarray(cv, dtype=np.float64)
                n_cov_te = 0 if cov_te is None else int(cov_te.shape[1])
                # The smooth rides along so its rows resample with the
                # observation; ``_block`` splits it back out per replicate.
                te_sb = tr.get("spatial_basis")
                if te_sb is not None:
                    te_sb = np.asarray(te_sb, dtype=np.float64)
                    cov_te = (
                        te_sb if cov_te is None else np.column_stack([cov_te, te_sb])
                    )
                results["test"] = _block(
                    t_te,
                    c_te,
                    cov_te,
                    do_perm=True,
                    sub_seed=seed + 101,
                    n_cov_cols=n_cov_te,
                )
        except Exception as exc:
            logger.warning(f"evaluate_effects: test scoring failed: {exc}")

        try:
            if (
                self.train_val_data is not None
                and df_full is not None
                and "polygon_id" in df_full.columns
            ):
                pids = set(self.train_val_data["polygon_id"].unique().tolist())
                sub = df_full[df_full["polygon_id"].isin(pids)]
                t_tv = np.asarray(sub["target"].values, dtype=np.float64)
                c_tv = np.asarray(sub["composite"].values, dtype=np.float64)
                cov_tv, n_cov_tv = self._augment_cov_with_spatial(
                    sub, t_tv, c_tv, self._whole_data_covariates(sub)
                )
                results["train_val"] = _block(
                    t_tv,
                    c_tv,
                    cov_tv,
                    do_perm=False,
                    sub_seed=seed + 202,
                    n_cov_cols=n_cov_tv,
                )
        except Exception as exc:
            logger.warning(f"evaluate_effects: train_val scoring failed: {exc}")

        return results

    # A MixedLM replicate is a full model refit, not the O(1) recomputation the
    # OLS metrics bootstrap, so the caller's thousands are clamped to these.
    # Reporting spends the bulk of a longitudinal run here: the effects cap
    # applies per metric per slice (|t| and the signed coefficient, over all /
    # test / train_val), and the test-CI cap once per study including each
    # standalone. Percentile bounds get lumpier as these fall — at 100
    # replicates the 2.5% bound sits between the 2nd and 3rd order statistic.
    _MIXEDLM_REPORT_BOOTSTRAP_CAP = 100

    # Replicate stride between cancel polls in the paired bootstrap. Each
    # replicate is a MixedLM refit, so a far shorter stride than the array
    # scorers in ``statistical_testing`` still costs nothing measurable.
    _PAIRED_CANCEL_POLL_EVERY = 16
    _MIXEDLM_TEST_CI_BOOTSTRAP_CAP = 150

    def _evaluate_effects_mixedlm(
        self,
        params: dict,
        metric: str,
        *,
        n_bootstrap: int,
        ci_level: float,
        seed: int,
    ) -> dict[str, dict]:
        """Per-subset effects for a longitudinal MixedLM objective.

        Each slice gets a cluster (entity) bootstrap percentile CI; the held-out
        ``test`` slice also carries the greenery fixed effect's Wald p. There is
        no permutation analogue here — a Freedman–Lane surrogate would have to be
        resampled and refit at the entity level, which the cluster bootstrap
        already covers — so every block is tagged ``p_kind="wald"``.
        """
        spec = self.longitudinal_spec
        assert spec is not None
        n_boot = max(50, min(int(n_bootstrap), self._MIXEDLM_REPORT_BOOTSTRAP_CAP))
        signed = "mixedlm_coef"

        def _block(y, g, eid, t, cov, sb, wave, sub_seed) -> dict:
            def _ci(m):
                return mixed_effects_scoring.cluster_bootstrap_metric_ci(
                    m,
                    y,
                    g,
                    eid,
                    t,
                    covariates=cov,
                    include_time_fixed=self._effective_time_fixed(spec),
                    random_slope=spec.random_slope_time,
                    spatial_basis=sb,
                    spatial_method=self.spatial_adjust_method,
                    n_bootstrap=n_boot,
                    ci_level=ci_level,
                    seed=int(sub_seed),
                    target=spec.association_target,
                    wave_index=wave,
                )

            r = _ci(metric)
            out = {
                "score": r.get("observed"),
                "lower": r.get("lower"),
                "upper": r.get("upper"),
                "p_value": r.get("pvalue"),
                "p_kind": "wald",
                "n": r.get("n"),
                # Replicates that actually converged. These bounds come from a
                # capped, modest count, so the reader needs to see it.
                "n_boot": r.get("n_boot"),
                "status": r.get("status"),
            }
            # |t| is folded, so its interval can never straddle zero. The signed
            # coefficient's interval is the one that can.
            if metric == "mixedlm_tstat":
                rc = _ci(signed)
                out["signed_score"] = rc.get("observed")
                out["signed_lower"] = rc.get("lower")
                out["signed_upper"] = rc.get("upper")
            return out

        results: dict[str, dict] = {}
        df_full = None
        try:
            df_full = self.apply_fusion(weights=dict(params))
            t_all = np.asarray(df_full["target"].values, dtype=np.float64)
            c_all = np.asarray(df_full["composite"].values, dtype=np.float64)
            eid, ysb = self._whole_data_longitudinal_keys(df_full)
            if eid is not None:
                cov_all, n_cov = self._augment_cov_with_spatial(
                    df_full, t_all, c_all, self._whole_data_covariates(df_full)
                )
                cov_part, sb_part = _helpers.split_control_matrix(cov_all, n_cov)
                results["all"] = _block(
                    t_all,
                    c_all,
                    eid,
                    ysb,
                    cov_part,
                    sb_part,
                    self._wave_labels_for_scoring(df_full),
                    seed,
                )
        except Exception as exc:
            logger.warning(f"evaluate_effects: whole-data scoring failed: {exc}")

        try:
            if self.test_data is not None and len(self.test_data) > 0:
                tr = self.evaluate_on_test(
                    params=dict(params), metric=metric, return_predictions=True
                )
                if tr.get("entity_id") is not None:
                    results["test"] = _block(
                        np.asarray(tr.get("targets"), dtype=np.float64),
                        np.asarray(tr.get("predictions"), dtype=np.float64),
                        tr.get("entity_id"),
                        tr.get("years_since_baseline"),
                        tr.get("covariates"),
                        tr.get("spatial_basis"),
                        self._wave_labels_for_scoring_array(tr.get("wave")),
                        seed + 101,
                    )
        except Exception as exc:
            logger.warning(f"evaluate_effects: test scoring failed: {exc}")

        try:
            if (
                self.train_val_data is not None
                and df_full is not None
                and "polygon_id" in df_full.columns
            ):
                pids = set(self.train_val_data["polygon_id"].unique().tolist())
                sub = df_full[df_full["polygon_id"].isin(pids)]
                eid, ysb = self._whole_data_longitudinal_keys(sub)
                if eid is not None:
                    t_tv = np.asarray(sub["target"].values, dtype=np.float64)
                    c_tv = np.asarray(sub["composite"].values, dtype=np.float64)
                    cov_tv, n_cov = self._augment_cov_with_spatial(
                        sub, t_tv, c_tv, self._whole_data_covariates(sub)
                    )
                    cov_part, sb_part = _helpers.split_control_matrix(cov_tv, n_cov)
                    results["train_val"] = _block(
                        t_tv,
                        c_tv,
                        eid,
                        ysb,
                        cov_part,
                        sb_part,
                        self._wave_labels_for_scoring(sub),
                        seed + 202,
                    )
        except Exception as exc:
            logger.warning(f"evaluate_effects: train_val scoring failed: {exc}")

        return results

    def _paired_difference_mixedlm(
        self,
        cgi_params: dict,
        standalone_params: dict,
        standalone_channel: str,
        metric: str,
        *,
        n_bootstrap: int,
        ci_level: float,
        seed: int,
    ) -> dict | None:
        """Whole-data paired difference for a MixedLM objective.

        Resamples **whole entities** and refits both mixed models on each
        resample. A row bootstrap would break the within-entity correlation the
        panel model exists to represent.
        """
        spec = self.longitudinal_spec
        assert spec is not None
        prev_ch = self._active_greenery_channel
        try:
            self._active_greenery_channel = "cgi"
            df_cgi = self.apply_fusion(weights=dict(cgi_params))
            self._active_greenery_channel = standalone_channel
            df_std = self.apply_fusion(weights=dict(standalone_params))
        except Exception as exc:
            logger.warning(f"paired difference: composite build failed: {exc}")
            return None
        finally:
            self._active_greenery_channel = prev_ch

        if "polygon_id" not in df_cgi.columns or "polygon_id" not in df_std.columns:
            logger.warning(
                f"paired difference ({standalone_channel}): composites are not "
                "entity-keyed, so CGI and the standalone cannot be paired."
            )
            return None
        merged = df_cgi[["polygon_id", "target", "composite"]].merge(
            df_std[["polygon_id", "composite"]],
            on="polygon_id",
            suffixes=("_cgi", "_std"),
        )
        eid, ysb = self._whole_data_longitudinal_keys(merged)
        if eid is None:
            logger.warning(
                f"paired difference ({standalone_channel}): no longitudinal keys "
                "on the merged frame."
            )
            return None
        wave = self._wave_labels_for_scoring(merged)
        cov = self._whole_data_covariates(merged)
        y = merged["target"].to_numpy(dtype=np.float64)
        g_cgi = merged["composite_cgi"].to_numpy(dtype=np.float64)
        g_std = merged["composite_std"].to_numpy(dtype=np.float64)

        def _score(yy, gg, ee, tt, cc, wv) -> float:
            return float(
                mixed_effects_scoring.score_mixedlm(
                    metric,
                    yy,
                    gg,
                    ee,
                    tt,
                    covariates=cc,
                    include_time_fixed=self._effective_time_fixed(spec),
                    random_slope=spec.random_slope_time,
                    target=spec.association_target,
                    wave_index=wv,
                    nan_on_fail=True,
                )
            )

        obs_cgi = _score(y, g_cgi, eid, ysb, cov, wave)
        obs_std = _score(y, g_std, eid, ysb, cov, wave)
        if not (np.isfinite(obs_cgi) and np.isfinite(obs_std)):
            which = ", ".join(
                nm
                for nm, v in (("CGI", obs_cgi), (standalone_channel, obs_std))
                if not np.isfinite(v)
            )
            logger.warning(
                f"paired difference ({standalone_channel}): the whole-data mixed "
                f"model did not fit for {which}, so there is no difference to "
                "bootstrap."
            )
            return None
        obs_diff = obs_cgi - obs_std  # every mixedlm_* metric is higher-is-better

        uniq, inv = np.unique(eid, return_inverse=True)
        if len(uniq) < 2:
            logger.warning(
                f"paired difference ({standalone_channel}): fewer than two "
                "entities to resample."
            )
            return None
        group_rows = [np.where(inv == k)[0] for k in range(len(uniq))]
        rng = np.random.default_rng(int(seed))
        n_boot = max(50, min(int(n_bootstrap), self._MIXEDLM_REPORT_BOOTSTRAP_CAP))
        diffs = np.empty(n_boot, dtype=np.float64)
        for i in range(n_boot):
            if (
                self._cancel_callback is not None
                and i % self._PAIRED_CANCEL_POLL_EVERY == 0
                and self._cancel_callback()
            ):
                raise JobCancelled("Paired bootstrap cancelled by user.")
            draw = rng.integers(0, len(uniq), size=len(uniq))
            idx = np.concatenate([group_rows[k] for k in draw])
            # A repeated entity has to form independent groups.
            new_eid = np.concatenate(
                [
                    np.full(len(group_rows[k]), slot, dtype=np.int64)
                    for slot, k in enumerate(draw)
                ]
            )
            try:
                sc = _score(
                    y[idx],
                    g_cgi[idx],
                    new_eid,
                    ysb[idx],
                    None if cov is None else cov[idx],
                    None if wave is None else wave[idx],
                )
                ss = _score(
                    y[idx],
                    g_std[idx],
                    new_eid,
                    ysb[idx],
                    None if cov is None else cov[idx],
                    None if wave is None else wave[idx],
                )
                diffs[i] = sc - ss
            except Exception:
                diffs[i] = np.nan

        valid = diffs[np.isfinite(diffs)]
        if len(valid) == 0:
            logger.warning(
                f"paired difference ({standalone_channel}): no bootstrap replicate "
                f"fit ({n_boot} attempted), so the interval has no support."
            )
            return None
        alpha = (1.0 - ci_level) / 2.0
        return {
            "observed_diff": float(obs_diff),
            "lower": float(np.quantile(valid, alpha)),
            "upper": float(np.quantile(valid, 1.0 - alpha)),
            "p_value": float((np.sum(valid <= 0.0) + 1) / (len(valid) + 1)),
            "cgi_score": float(obs_cgi),
            "standalone_score": float(obs_std),
            "standalone_channel": standalone_channel,
            "n": int(len(y)),
            "n_boot": int(len(valid)),
            "favors_cgi": bool(obs_diff > 0),
        }

    def paired_objective_difference(
        self,
        cgi_params: dict,
        standalone_params: dict,
        standalone_channel: str,
        metric: str,
        *,
        n_bootstrap: int = 2000,
        ci_level: float = 0.95,
        seed: int = 42,
    ) -> dict | None:
        """Whole-data objective difference between CGI and the best standalone.

        Builds both composites on the full dataset (CGI via the formula, the
        standalone via its single channel), then bootstraps entities once per
        replicate and scores both on the same resample so the paired
        difference is in the objective metric's own units. The difference is
        signed so a **positive** value favours CGI regardless of metric
        direction. Returns ``{observed_diff, lower, upper, p_value, cgi_score,
        standalone_score, standalone_channel, n, favors_cgi}`` or ``None`` when
        the metric isn't an objective-scoring metric or the composites can't be
        built. ``p_value`` is the one-sided bootstrap tail (share of replicates
        with difference ≤ 0).
        """
        if metric not in objective_scoring.SUPPORTED_METRICS:
            if self.is_longitudinal and metric in mixed_effects_scoring.MIXEDLM_METRICS:
                return self._paired_difference_mixedlm(
                    cgi_params,
                    standalone_params,
                    standalone_channel,
                    metric,
                    n_bootstrap=n_bootstrap,
                    ci_level=ci_level,
                    seed=seed,
                )
            return None
        prev_ch = self._active_greenery_channel
        try:
            self._active_greenery_channel = "cgi"
            df_cgi = self.apply_fusion(weights=dict(cgi_params))
            self._active_greenery_channel = standalone_channel
            df_std = self.apply_fusion(weights=dict(standalone_params))
        except Exception as exc:
            logger.warning(
                f"paired_objective_difference: composite build failed: {exc}"
            )
            return None
        finally:
            self._active_greenery_channel = prev_ch

        try:
            if "polygon_id" in df_cgi.columns and "polygon_id" in df_std.columns:
                merged = df_cgi[["polygon_id", "target", "composite"]].merge(
                    df_std[["polygon_id", "composite"]],
                    on="polygon_id",
                    suffixes=("_cgi", "_std"),
                )
                cov = self._whole_data_covariates(merged)
            else:
                merged = pd.DataFrame(
                    {
                        "target": np.asarray(df_cgi["target"].values),
                        "composite_cgi": np.asarray(df_cgi["composite"].values),
                        "composite_std": np.asarray(df_std["composite"].values),
                    }
                )
                cov = None
            target = merged["target"].to_numpy(dtype=np.float64)
            cgi_c = merged["composite_cgi"].to_numpy(dtype=np.float64)
            std_c = merged["composite_std"].to_numpy(dtype=np.float64)
            # Both composites share the same outcome-selected smooth; it rides in
            # the resampled controls and is split back out per replicate so the
            # paired difference is adjusted exactly as the objective was.
            cov, n_cov_cols = self._augment_cov_with_spatial(merged, target, cgi_c, cov)
        except Exception as exc:
            logger.warning(f"paired_objective_difference: alignment failed: {exc}")
            return None

        # Drop non-finite rows up front. ``score`` would mask them per call
        # anyway, and doing it here keeps the shared-distance-matrix replicate
        # path (which cannot mask per composite) available for the whole loop.
        finite = np.isfinite(target) & np.isfinite(cgi_c) & np.isfinite(std_c)
        if cov is not None:
            finite &= np.isfinite(cov).all(axis=1)
        if not finite.all():
            target = target[finite]
            cgi_c = cgi_c[finite]
            std_c = std_c[finite]
            if cov is not None:
                cov = cov[finite]

        if len(target) < 3:
            return None

        higher = metric in objective_scoring.HIGHER_IS_BETTER
        sign = 1.0 if higher else -1.0

        def _score(t_arr, c_arr, cov_arr):
            cov_part, sb_part = _helpers.split_control_matrix(cov_arr, n_cov_cols)
            return float(
                objective_scoring.score(
                    metric,
                    t_arr,
                    c_arr,
                    cov_part,
                    spatial_basis=sb_part,
                    spatial_method=self.spatial_adjust_method,
                    residualize_method=self.residualize_method,
                    pdcor_cache=self._pdcor_cache,
                    spline_cache=self._spline_basis_cache,
                )
            )

        obs_cgi = _score(target, cgi_c, cov)
        obs_std = _score(target, std_c, cov)
        obs_diff = sign * (obs_cgi - obs_std)

        # Replicate fast path for the O(n²) metric: both composites are scored
        # on the same resample, so the target / conditioning distance matrices
        # are precomputed once and shared. Requires fully finite inputs (the
        # generic path masks NaN rows inside score(), which an index-vector
        # scorer cannot reproduce per composite).
        fast_pair = None
        if metric == "partial_distance_corr" and cov is not None:
            fast_pair = pdcor.pdcor_paired_replicate_scorers(target, cgi_c, std_c, cov)

        rng = np.random.default_rng(int(seed))
        n = len(target)
        diffs = np.empty(int(n_bootstrap), dtype=np.float64)
        for i in range(int(n_bootstrap)):
            idx = rng.integers(0, n, size=n)
            try:
                if fast_pair is not None:
                    sc = fast_pair[0](idx)
                    ss = fast_pair[1](idx)
                else:
                    cov_i = None if cov is None else cov[idx]
                    sc = _score(target[idx], cgi_c[idx], cov_i)
                    ss = _score(target[idx], std_c[idx], cov_i)
                diffs[i] = sign * (sc - ss)
            except Exception:
                diffs[i] = np.nan

        valid = diffs[np.isfinite(diffs)]
        if len(valid) == 0:
            return None
        alpha = (1.0 - ci_level) / 2.0
        lower = float(np.quantile(valid, alpha))
        upper = float(np.quantile(valid, 1.0 - alpha))
        p_value = float((np.sum(valid <= 0.0) + 1) / (len(valid) + 1))
        return {
            "observed_diff": float(obs_diff),
            "lower": lower,
            "upper": upper,
            "p_value": p_value,
            "cgi_score": float(obs_cgi),
            "standalone_score": float(obs_std),
            "standalone_channel": standalone_channel,
            "n": int(n),
            "favors_cgi": bool(obs_diff > 0),
        }

    def compute_subset_scores(
        self,
        params: dict,
        metric: str,
    ) -> dict[str, dict[str, float | None]]:
        """Score ``params`` on every data slice the stability run produced.

        Returns a dict of ``{subset: {score, pvalue, n}}`` for the four
        canonical slices. There is no train→fit→validate step — stability
        selection resamples the train+val pool into complementary halves — so
        ``train`` / ``val`` are *labels for continuity*, relabelled in-pool /
        OOB in the results view:

        - ``train`` — the winning params scored on the full train+val pool
          (an in-pool fit; the pool is what the bootstrap resamples).
        - ``val`` — the winning cell's **median** out-of-bag score across the
          complementary-half resamples (``__cell_median__``), a direct measure
          of cross-resample predictive performance.
        - ``test`` — fresh test-set score by re-running
          :meth:`evaluate_on_test` with the supplied params.
        - ``all`` — composite applied to every entity (the full dataset)
          via :meth:`apply_fusion`, polygon-collapsed for polygon
          targets, scored against the per-entity outcome.

        The runner caches the returned dict on the bundle so the results UI
        doesn't re-score on every page rerun.
        """
        out: dict[str, dict[str, float | None]] = {}

        train_val_n = (
            int(self.train_val_data["polygon_id"].nunique())
            if self.train_val_data is not None
            and "polygon_id" in self.train_val_data.columns
            else (
                int(len(self.train_val_data))
                if self.train_val_data is not None
                else None
            )
        )

        # In-pool slice: the winning params scored on the full train+val pool
        # (the closest analogue to "train" — the pool is what the bootstrap
        # resamples).
        train_val_score = self._score_data_subset(
            data=self.train_val_data, params=params, metric=metric
        )
        out["train"] = {
            "score": train_val_score.get("score"),
            "score_raw": train_val_score.get("score_raw"),
            "pvalue": train_val_score.get("pvalue"),
            "pvalue_raw": train_val_score.get("pvalue_raw"),
            "n": train_val_n,
        }
        # OOB slice: the winning cell's median out-of-bag score across the
        # complementary-half resamples, recorded by
        # ``bootstrap_stability_selection`` under ``__cell_median__`` — a
        # cross-resample held-out measure. It's computed by ``_objective`` with
        # covariates configured on the engine (a partial-correlation analogue),
        # so there's no raw equivalent at this level.
        cell_median = (
            params.get("__cell_median__") if isinstance(params, dict) else None
        )
        out["val"] = {
            "score": (
                float(cell_median)
                if cell_median is not None and np.isfinite(float(cell_median))
                else None
            ),
            "score_raw": None,
            "pvalue": None,
            "pvalue_raw": None,
            "n": train_val_n,
        }

        # ── test: fresh evaluate_on_test with these params ──
        # ``evaluate_on_test`` returns the partial (covariate-adjusted)
        # score that the optimizer optimized; the raw equivalent is
        # produced by ``_score_data_subset`` on the test slice.
        try:
            test_result = self.evaluate_on_test(params=dict(params), metric=metric)
            test_n = (
                int(self.test_data["polygon_id"].nunique())
                if self.test_data is not None and "polygon_id" in self.test_data.columns
                else (int(len(self.test_data)) if self.test_data is not None else None)
            )
            test_both = self._score_data_subset(
                data=self.test_data, params=params, metric=metric
            )
            # ``evaluate_on_test`` fails soft to the metric's degenerate value,
            # which for a mixed model is a 0.0 indistinguishable from a real
            # null. The nan-on-fail rescore of the same slice says which it is.
            fit_failed = bool(test_both.get("fit_failed"))
            out["test"] = {
                "score": (
                    None
                    if fit_failed
                    else _finite_or_none(test_result.get("test_score"))
                ),
                "score_raw": test_both.get("score_raw"),
                "pvalue": (
                    None
                    if fit_failed
                    else _finite_or_none(test_result.get("test_pvalue"))
                ),
                "pvalue_raw": test_both.get("pvalue_raw"),
                "n": test_n,
            }
        except Exception as exc:
            logger.warning(f"compute_subset_scores: test scoring failed: {exc}")
            out["test"] = {
                "score": None,
                "score_raw": None,
                "pvalue": None,
                "pvalue_raw": None,
                "n": None,
            }

        # ── all: composite on every entity (full dataset) ───
        try:
            df = self.apply_fusion(weights=dict(params))
            target = np.asarray(df["target"].values, dtype=np.float64)
            composite = np.asarray(df["composite"].values, dtype=np.float64)
            cov: np.ndarray | None = None
            cov_cols = self.covariate_columns
            if cov_cols and "polygon_id" in df.columns:
                full = self._full_data_frame()
                cov_per_poly = (
                    full.groupby("polygon_id", sort=False)[cov_cols]
                    .first()
                    .reindex(df["polygon_id"].values)
                )
                cov = cov_per_poly.to_numpy(dtype=np.float64)
            # In-sample slice carries no p-value; the held-out test does.
            wants_pval = False

            sb = self._spatial_basis_columns(
                self._whole_data_coords(df), target, cov, composite
            )
            # Longitudinal keys for the MixedLM branch of the scoring seam
            # (``None`` in cross-sectional mode, ignored by the OLS scorer).
            lon_eid, lon_ysb = self._whole_data_longitudinal_keys(df)

            def _full_score(
                c: np.ndarray | None, spat: np.ndarray | None
            ) -> tuple[float | None, float | None]:
                s_out = self._score_greenery(
                    metric,
                    target,
                    composite,
                    covariates=c,
                    spatial_basis=spat,
                    entity_id=lon_eid,
                    years_since_baseline=lon_ysb,
                    return_pvalue=wants_pval,
                    nan_on_fail=True,
                )
                if wants_pval:
                    s, p = s_out  # type: ignore[misc]
                    return _finite_or_none(s), _finite_or_none(p)
                return _finite_or_none(s_out), None  # type: ignore[arg-type]

            partial_s, partial_p = _full_score(cov, sb)
            raw_s, raw_p = (
                _full_score(None, None)
                if (cov is not None or sb is not None)
                else (partial_s, partial_p)
            )
            out["all"] = {
                "score": partial_s,
                "score_raw": raw_s,
                "pvalue": partial_p,
                "pvalue_raw": raw_p,
                "n": int(len(target)),
            }
        except Exception as exc:
            logger.warning(f"compute_subset_scores: full-dataset scoring failed: {exc}")
            out["all"] = {
                "score": None,
                "score_raw": None,
                "pvalue": None,
                "pvalue_raw": None,
                "n": None,
            }

        return out

    def compute_covariate_impact(
        self,
        params: dict,
        metric: str,
    ) -> dict | None:
        """Quantify each covariate's effect on the OLS model of ``target``.

        Fits two OLS regressions on the full dataset (polygon-collapsed
        when applicable):

        - **Full**: ``target ~ CGI + covariate_1 + covariate_2 + …``
        - **Reduced**: ``target ~ CGI`` (CGI alone, no covariates)

        Returns a dict with per-covariate `coef`, `std_err`, `t_stat`,
        `pvalue`, `direction` (positive / negative effect), the full
        model's R², the reduced model's R², and the partial R² each
        covariate carries (the drop in residual variance when the
        covariate is removed from the full model). Returns ``None``
        when there are no covariates or when the regression fails.
        """
        cov_cols = list(self.covariate_columns or [])
        if not cov_cols:
            return None
        try:
            df = self.apply_fusion(weights=dict(params))
            target = np.asarray(df["target"].values, dtype=np.float64)
            composite = np.asarray(df["composite"].values, dtype=np.float64)
            full = self._full_data_frame()
            if full is None:
                return None
            if "polygon_id" in df.columns:
                cov_per_poly = (
                    full.groupby("polygon_id", sort=False)[cov_cols]
                    .first()
                    .reindex(df["polygon_id"].values)
                )
                cov_mat = cov_per_poly.to_numpy(dtype=np.float64)
            else:
                cov_mat = full[cov_cols].to_numpy(dtype=np.float64)

            # Longitudinal keys aligned to the collapsed frame, so the
            # mixed-effects branch below can fit a panel model instead of OLS.
            lon_entity_id = lon_ysb = None
            if self.is_longitudinal:
                if "polygon_id" in df.columns:
                    lon_keys = (
                        full.groupby("polygon_id", sort=False)[
                            ["entity_id", "years_since_baseline"]
                        ]
                        .first()
                        .reindex(df["polygon_id"].values)
                    )
                    lon_entity_id = lon_keys["entity_id"].to_numpy()
                    lon_ysb = lon_keys["years_since_baseline"].to_numpy(
                        dtype=np.float64
                    )
                elif {"entity_id", "years_since_baseline"} <= set(df.columns):
                    lon_entity_id = df["entity_id"].to_numpy()
                    lon_ysb = df["years_since_baseline"].to_numpy(dtype=np.float64)

            mask = ~(
                np.isnan(target) | np.isnan(composite) | np.isnan(cov_mat).any(axis=1)
            )
            target = target[mask]
            composite = composite[mask]
            cov_mat = cov_mat[mask]
            if lon_entity_id is not None:
                lon_entity_id = lon_entity_id[mask]
                lon_ysb = lon_ysb[mask]

            # Panel data: OLS standard errors are anticonservative because the
            # (entity, wave) rows are correlated within entity. Fit the same
            # mixed model the scorer uses so the covariate table's coefficients
            # and Wald p-values account for the random effects.
            if (
                self.is_longitudinal
                and metric in mixed_effects_scoring.MIXEDLM_METRICS
                and lon_entity_id is not None
            ):
                spec = self.longitudinal_spec
                assert spec is not None
                # Names cover the user's covariates only; the period / area
                # indicators trailing them stay unnamed controls.
                return mixed_effects_scoring.covariate_impact_mixedlm(
                    target,
                    composite,
                    lon_entity_id,
                    lon_ysb,
                    cov_mat,
                    [c for c in cov_cols if c in set(self._user_covariate_columns)],
                    include_time_fixed=self._effective_time_fixed(spec),
                    random_slope=spec.random_slope_time,
                )

            if target.size < len(cov_cols) + 3:
                return None

            def _ols(
                X: np.ndarray, y: np.ndarray
            ) -> tuple[np.ndarray, float, np.ndarray]:
                """Return (beta, r2, residuals) for OLS y = X·beta."""
                Xc = np.column_stack([np.ones(len(y)), X])
                beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
                yhat = Xc @ beta
                resid = y - yhat
                ss_res = float((resid**2).sum())
                ss_tot = float(((y - y.mean()) ** 2).sum())
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                return beta, r2, resid

            X_full = np.column_stack([composite, cov_mat])
            beta_full, r2_full, resid_full = _ols(X_full, target)
            _, r2_cgi_only, _ = _ols(composite.reshape(-1, 1), target)

            n = len(target)
            p_full = X_full.shape[1] + 1
            dof = max(1, n - p_full)
            sigma2 = float((resid_full**2).sum()) / dof
            Xc_full = np.column_stack([np.ones(n), X_full])
            try:
                cov_beta = sigma2 * np.linalg.pinv(Xc_full.T @ Xc_full)
                std_err_all = np.sqrt(np.maximum(np.diag(cov_beta), 0.0))
            except np.linalg.LinAlgError:
                std_err_all = np.full(p_full, np.nan)

            from scipy.stats import t as _t

            per_cov: list[dict] = []
            for i, name in enumerate(cov_cols):
                # Slot in beta_full / std_err_all is offset by 2:
                # intercept (0), CGI (1), covariate_i (2+i).
                slot = 2 + i
                coef = float(beta_full[slot])
                se = (
                    float(std_err_all[slot])
                    if std_err_all is not None
                    else float("nan")
                )
                if not np.isfinite(se) or se <= 0:
                    t_stat = float("nan")
                    pval = float("nan")
                else:
                    t_stat = coef / se
                    pval = float(2.0 * (1.0 - _t.cdf(abs(t_stat), df=dof)))

                # Drop this covariate from the full model and refit to get
                # its partial R² (the share of variance only it explains).
                X_minus = np.delete(X_full, slot - 1, axis=1)
                _, r2_minus, _ = _ols(X_minus, target)
                partial_r2 = max(0.0, r2_full - r2_minus)

                direction = "positive" if coef > 0 else "negative" if coef < 0 else "—"
                per_cov.append(
                    {
                        "covariate": name,
                        "coef": coef,
                        "std_err": se,
                        "t_stat": t_stat,
                        "pvalue": pval,
                        "direction": direction,
                        "partial_r2": partial_r2,
                    }
                )

            return {
                "n": int(n),
                "r2_full": float(r2_full),
                "r2_cgi_only": float(r2_cgi_only),
                "r2_lift_from_covariates": float(r2_full - r2_cgi_only),
                "per_covariate": per_cov,
                "cgi_coef": float(beta_full[1]),
                "cgi_std_err": (
                    float(std_err_all[1]) if std_err_all is not None else float("nan")
                ),
            }
        except Exception as exc:
            logger.warning(f"compute_covariate_impact failed: {exc}")
            return None

    def compute_decline_terms(self, params: dict) -> dict | None:
        """Greenery × time decline terms on the winning composite (longitudinal).

        Reports the overall greenery × time slope and, per the spec's decline
        knobs, the between-person (average-exposure) and within-person
        (exposure-change) slopes. ``None`` for cross-sectional runs, a non-MixedLM
        scoring metric, or when the mixed model cannot be fit.
        """
        if not self.is_longitudinal:
            return None
        spec = self.longitudinal_spec
        assert spec is not None
        if spec.scoring_metric not in mixed_effects_scoring.MIXEDLM_METRICS:
            return None
        try:
            df = self.apply_fusion(weights=dict(params))
            target = np.asarray(df["target"].values, dtype=np.float64)
            composite = np.asarray(df["composite"].values, dtype=np.float64)
            eid, ysb = self._whole_data_longitudinal_keys(df)
            if eid is None:
                return None
            cov_cols = list(self.covariate_columns or [])
            cov = None
            if cov_cols:
                full = self._full_data_frame()
                if full is not None and "polygon_id" in df.columns:
                    cov = (
                        full.groupby("polygon_id", sort=False)[cov_cols]
                        .first()
                        .reindex(df["polygon_id"].values)
                        .to_numpy(dtype=np.float64)
                    )
                elif set(cov_cols) <= set(df.columns):
                    cov = df[cov_cols].to_numpy(dtype=np.float64)
            out = mixed_effects_scoring.decline_terms_mixedlm(
                target,
                composite,
                eid,
                ysb,
                cov,
                random_slope=spec.random_slope_time,
                want_between=bool(spec.decline_average_exposure),
                want_within=bool(spec.decline_exposure_change),
            )
            if out is None:
                return None

            # ── Period-confounding checks ───────────────
            # A greenery × time slope is only about greenery if it survives an
            # exposure with no spatial content. The placebo holds each wave's
            # mean exposure, so it carries the period structure and nothing else.
            wave = self._whole_data_wave_labels(df)
            out["period_diagnostic"] = mixed_effects_scoring.period_confounding_report(
                composite, eid, ysb, wave_index=wave
            )
            if wave is not None:
                placebo = mixed_effects_scoring.decline_terms_mixedlm(
                    target,
                    mixed_effects_scoring.placebo_exposure(composite, wave),
                    eid,
                    ysb,
                    cov,
                    random_slope=spec.random_slope_time,
                    want_between=bool(spec.decline_average_exposure),
                    want_within=bool(spec.decline_exposure_change),
                )
                out["placebo_terms"] = (placebo or {}).get("terms") or []
            return out
        except Exception as exc:
            logger.warning(f"compute_decline_terms failed: {exc}")
            return None

    def compute_exposure_response(
        self, params: dict, *, iqr: float | None = None
    ) -> dict | None:
        """Per-IQR effect, quartile contrasts and a spline non-linearity test.

        Run once on the winning composite, this is the shape the greenspace
        literature reports its findings in: an effect per interquartile-range
        increase, a gradient across exposure quartiles against the lowest, and a
        test of whether the straight line the search optimised is the right
        functional form.

        The fitter matches the study design — logistic (or GEE logistic on a
        panel) for a binary outcome, OLS or MixedLM otherwise — so the standard
        errors in these tables are the same kind as the headline estimate's.
        Pass ``iqr`` to pin the scaling constant, e.g.
        ``exposure_response.CLSA_NDVI_IQR`` when the point is to land on a
        published NDVI number.
        """
        try:
            df = self.apply_fusion(weights=dict(params))
            target = np.asarray(df["target"].values, dtype=np.float64)
            composite = np.asarray(df["composite"].values, dtype=np.float64)
            cov = self._reporting_covariate_matrix(df)

            entity_id = None
            if self.is_longitudinal:
                entity_id, _ysb = self._whole_data_longitudinal_keys(df)

            keep = self._finite_rows(target, composite, cov)
            if keep is not None and not keep.all():
                if int(keep.sum()) < 10:
                    _log("WARN", "Exposure-response: too few complete rows to fit.")
                    return None
                _log(
                    "INFO",
                    f"Exposure-response: dropped {int((~keep).sum())} row(s) with a "
                    "missing outcome, composite or covariate.",
                )
                target, composite = target[keep], composite[keep]
                cov = None if cov is None else cov[keep]
                entity_id = None if entity_id is None else entity_id[keep]

            binary = binary_longitudinal.is_binary(target)
            coding = binary_longitudinal.describe_coding(target) if binary else None
            if coding and coding["suspicious"]:
                _log(
                    "WARN",
                    f"Binary outcome is coded "
                    f"{{{coding['reference_level']:g}, {coding['modelled_level']:g}}}, "
                    f"not {{0, 1}}. The logistic models treat "
                    f"{coding['modelled_level']:g} as the event — confirm that is "
                    "the affirmative level, or every odds ratio is inverted.",
                )
            if binary and entity_id is not None:
                fitter = binary_longitudinal.make_gee_logit_fitter(target, entity_id)
                design = "gee_logit"
            elif binary:
                fitter = binary_longitudinal.make_logit_fitter(target)
                design = "logit"
            else:
                fitter = exposure_response.make_ols_fitter(target)
                design = "ols"

            # Headline per-unit effect, from the same model family, so the
            # per-IQR rescale below is a change of units and nothing else.
            base_cols = [np.ones(len(target)), composite]
            base_names = ["intercept", "greenery"]
            if cov is not None and cov.size:
                for j in range(cov.shape[1]):
                    base_cols.append(cov[:, j])
                    base_names.append(f"cov{j}")
            fitted = fitter(np.column_stack(base_cols), base_names)
            per_iqr = None
            if fitted is not None:
                coef, se, _p = fitted["greenery"]
                per_iqr = exposure_response.iqr_scaled_effect(
                    coef, se, composite, iqr=iqr, logistic=binary
                )

            return {
                "design": design,
                "binary_outcome": bool(binary),
                "coding": coding,
                "per_iqr": per_iqr,
                "quartiles": exposure_response.quartile_terms(
                    composite, fitter, cov, logistic=binary
                ),
                "nonlinearity": exposure_response.spline_nonlinearity_test(
                    composite, fitter, cov
                ),
            }
        except Exception as exc:
            logger.warning(f"compute_exposure_response failed: {exc}")
            return None

    def compute_moderation(
        self, params: dict, moderator_columns: "list[str] | None"
    ) -> list[dict]:
        """Effect modification of the greenery association, one entry per moderator.

        Fits ``outcome ~ greenery + M + greenery x M + covariates`` on the
        winning composite and reports the interaction test plus the greenery
        slope at each level of ``M`` — the analysis the greenspace papers run
        when they claim an effect differs by sex, income or social standing.

        The moderator is removed from the covariate matrix first: the model
        enters its main effect explicitly, so leaving it in the controls too
        would make the design collinear. Categorical moderators are detected
        from ``covariate_types``, falling back to a distinct-value count.
        """
        moderators = [m for m in (moderator_columns or []) if m]
        if not moderators:
            return []
        out: list[dict] = []
        try:
            df = self.apply_fusion(weights=dict(params))
            target = np.asarray(df["target"].values, dtype=np.float64)
            composite = np.asarray(df["composite"].values, dtype=np.float64)
            entity_id = None
            if self.is_longitudinal:
                entity_id, _ysb = self._whole_data_longitudinal_keys(df)

            binary = binary_longitudinal.is_binary(target)

            def _fitter_for(y, ent):
                if binary and ent is not None:
                    return binary_longitudinal.make_gee_logit_fitter(y, ent)
                if binary:
                    return binary_longitudinal.make_logit_fitter(y)
                return exposure_response.make_ols_fitter(y)

            for name in moderators:
                values = self._reporting_raw_column(df, name)
                if values is None:
                    _log("WARN", f"Moderator '{name}' not found; skipping.")
                    continue
                cov = self._reporting_covariate_matrix(df, exclude=[name])
                # Per moderator, not once up front: a row missing this
                # moderator says nothing about the next one, and dropping it
                # from every analysis would throw away usable data.
                keep = self._finite_rows(target, composite, values, cov)
                y_m, comp_m, val_m, cov_m, ent_m = (
                    target,
                    composite,
                    values,
                    cov,
                    entity_id,
                )
                if keep is not None and not keep.all():
                    if int(keep.sum()) < 10:
                        _log(
                            "WARN",
                            f"Moderator '{name}': too few complete rows; skipping.",
                        )
                        continue
                    y_m, comp_m, val_m = target[keep], composite[keep], values[keep]
                    cov_m = None if cov is None else cov[keep]
                    ent_m = None if entity_id is None else entity_id[keep]
                declared = self.covariate_types.get(name)
                n_levels = int(np.unique(val_m).size)
                categorical = declared == "categorical" if declared else n_levels <= 12
                result = exposure_response.moderation_terms(
                    comp_m,
                    val_m,
                    _fitter_for(y_m, ent_m),
                    cov_m,
                    categorical=categorical,
                    logistic=binary,
                    moderator_name=name,
                )
                if result is None:
                    _log(
                        "WARN",
                        f"Moderation analysis for '{name}' could not be fit "
                        "(too few rows, or a moderator with one level).",
                    )
                    continue
                out.append(result)
        except Exception as exc:
            logger.warning(f"compute_moderation failed: {exc}")
        return out

    def _reporting_raw_column(self, df: "pd.DataFrame", name: str):
        """One un-expanded attribute column aligned to an ``apply_fusion`` frame."""
        full = self._full_data_frame()
        if full is not None and "polygon_id" in df.columns and name in full.columns:
            series = (
                full.groupby("polygon_id", sort=False)[name]
                .first()
                .reindex(df["polygon_id"].values)
            )
            return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64)
        return None

    @staticmethod
    def _finite_rows(*arrays) -> "np.ndarray | None":
        """Row mask where every supplied array is finite.

        The reporting matrices are aligned by reindexing on ``polygon_id``, so
        a key the fusion frame carries but the attribute table does not yields
        an all-NaN row. LAPACK takes the norm of the whole design before it
        factorises, so one such row makes that norm NaN and the least-squares
        driver rejects it outright — with a message printed from Fortran that
        never reaches the Python log. Dropping the rows here keeps the
        reporting fits complete-case, which is what the objective already does.
        """
        mask = None
        for arr in arrays:
            if arr is None:
                continue
            a = np.asarray(arr, dtype=np.float64)
            if a.size == 0:
                continue
            ok = np.isfinite(a) if a.ndim == 1 else np.isfinite(a).all(axis=1)
            mask = ok if mask is None else (mask & ok)
        return mask

    def _reporting_covariate_matrix(
        self, df: "pd.DataFrame", exclude: "list[str] | None" = None
    ):
        """Covariate matrix aligned to an ``apply_fusion`` frame, or ``None``.

        Shared by the decline-term, exposure-response and moderation reports so
        all three condition on exactly the same controls the objective did.

        ``exclude`` drops columns by their **user-facing** name, taking their
        dummy expansions with them — used by the moderation report, which
        enters the moderator's main effect itself and would otherwise hand the
        fitter the same information twice.
        """
        cov_cols = list(self.covariate_columns or [])
        for name in exclude or []:
            drop = set(self._covariate_dummy_map.get(name, [])) | {name}
            cov_cols = [c for c in cov_cols if c not in drop]
        if not cov_cols:
            return None
        full = self._full_data_frame()
        if full is not None and "polygon_id" in df.columns:
            return (
                full.groupby("polygon_id", sort=False)[cov_cols]
                .first()
                .reindex(df["polygon_id"].values)
                .to_numpy(dtype=np.float64)
            )
        if set(cov_cols) <= set(df.columns):
            return df[cov_cols].to_numpy(dtype=np.float64)
        return None

    def _wave_labels_for_scoring(self, df: "pd.DataFrame") -> "np.ndarray | None":
        """Wave labels to hand the mixed-effects scorer as period fixed effects.

        ``None`` once :meth:`_expand_period_controls` has folded the wave
        indicators into ``covariate_columns``: every control matrix built from
        those columns already carries them, and the scorer one-hots
        ``wave_index`` into the *same* drop-first indicators, so passing both
        makes the fixed-effects design exactly singular and the fit fails.
        """
        if self._wave_control_columns:
            return None
        return self._whole_data_wave_labels(df)

    def _wave_labels_for_scoring_array(
        self, wave: "np.ndarray | None"
    ) -> "np.ndarray | None":
        """:meth:`_wave_labels_for_scoring` for labels already in hand."""
        return None if self._wave_control_columns else wave

    def _whole_data_wave_labels(self, df: "pd.DataFrame") -> "np.ndarray | None":
        """Per-row wave label for a collapsed frame, or ``None`` when absent."""
        if not self.is_longitudinal:
            return None
        if "wave" in df.columns:
            return df["wave"].to_numpy()
        full = self._full_data_frame()
        if full is None or "polygon_id" not in df.columns or "wave" not in full.columns:
            return None
        return (
            full.groupby("polygon_id", sort=False)["wave"]
            .first()
            .reindex(df["polygon_id"].values)
            .to_numpy()
        )

    def bootstrap_stability_selection(
        self,
        metric: str,
        *,
        n_bootstraps: int = 20,
        n_trials_per_bootstrap: int = 50,
        weight_bin_pct: int = cgi_formulas.WEIGHT_BIN_PCT,
        weight_refine_bin_pct: int | None = None,
        top_percent_per_bootstrap: float = 0.2,
        min_cell_count: int = 3,
        worst_quantile: float = 0.10,
        max_pfer: float | None = 1.0,
        radius_bin_m: int | None = None,
        spatial_resample: bool = False,
        seed: int = 42,
        cancel_callback: Callable[..., bool] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Stability selection via complementary-pairs out-of-bag scoring.

        Adapts Meinshausen & Bühlmann's stability selection (2010) to
        hyperparameter search, using the complementary-pairs subsampling of
        Shah & Samworth (2013). The ⌊n/2⌋ subsampling is what the PFER bound
        assumes; because the candidate set here is the weight cells discovered
        by the QMC search (not a fixed selector applied identically each
        resample), the reported PFER is best read as a guide, not a certified
        bound. Enqueued seed trials are excluded from the selection counts so
        the calibration frequencies stay unbiased. The
        procedure draws ``n_bootstraps`` ⌊n/2⌋ subsamples (in complementary
        pairs), runs a short objective-blind QMC (Sobol) study on each, and
        rates each parameter region by how well it does **on the held-out OOB
        rows** of every subsample. The headline robustness metric is the
        worst-quantile OOB score across all trials that landed in each region
        — i.e. "how badly can this parameter region perform on a resample of
        my data?". This is a direct statement about predictive power across
        samples, not a selection-bias-prone p-value.

        Per subsample:

        1. Split entities (polygons when ``polygon_id`` is present, rows
           otherwise) into two complementary halves; the in-bag is one half
           and the OOB is the other (~50 % of entities, genuinely held out).
        2. Build a fresh in-memory Optuna study with a low-discrepancy QMC
           (Sobol) sampler (objective-aware samplers are intentionally avoided:
           stability wants even parameter-space coverage, not depth on this
           particular subsample). The engine's ``_objective`` already handles
           in-bag/OOB scaler fitting, channel aggregation, polygon collapse,
           covariate-aware partial-correlation scoring (cross-sectional), and
           MixedLM scoring (longitudinal) — so we splice the subsample split
           into ``self.cv_folds`` and reuse the existing pipeline.
        3. Record per-trial ``(snapped_params, oob_score)``.

        Across all subsamples:

        * Snap each trial's weights to ``weight_bin_pct``-wide buckets via
          :func:`cgi_formulas.weight_cell_key`.
        * Pool OOB scores per cell and rank cells within each resample.
        * Calibrate the selection size ``K`` and threshold ``π`` (Bodinier
          et al.) and take the winning cell as the one with the highest
          selection probability in the calibrated stable set, ties broken by
          the worst-quantile OOB score (``q_worst = quantile(scores,
          worst_quantile)`` for higher-is-better metrics; ``quantile(scores,
          1 - worst_quantile)`` otherwise) subject to ``count >=
          min_cell_count``.
        * Within the chosen cell, snap + renormalize the winners' weights to
          canonical integer steps so the output is drop-in compatible with the
          composite/apply path.

        Args:
            metric: Scoring metric (any value supported by ``_objective``).
            n_bootstraps: Target number of ⌊n/2⌋ subsamples (rounded to an even
                count of complementary pairs). 20–50 typical.
            n_trials_per_bootstrap: QMC trials inside each subsample.
                30–100 typical.
            weight_bin_pct: Cell width for the first stage's weight binning.
                20 % gives 21 cells for the weighted-average formula and 56 for
                synergy; 10 % gives 66 and 286, which is fine enough that
                selection frequency splits across near-identical neighbours and
                nothing is ever declared stable.
            weight_refine_bin_pct: Cell width for the third stage, which re-bins
                the winning radius sub-cell's trials to pick a narrower weight
                mix. ``None`` uses ``cgi_formulas.WEIGHT_REFINE_BIN_PCT``; a
                value at or above ``weight_bin_pct`` disables the stage and
                averages the coarse cell as before.
            top_percent_per_bootstrap: Fraction of each bootstrap's trials
                considered "selected" for the selection-probability sidecar.
            min_cell_count: A cell is eligible only when at least this many
                trials landed in it across all bootstraps. Prevents a
                single-trial outlier cell from claiming "best". Also gates the
                refinement stage.
            worst_quantile: 0.10 → 10th percentile worst-case score for
                higher-is-better metrics; 90th percentile for lower-is-
                better. Reported on every cell as a robustness read; the
                **median** is what ranks them, since the question is typical
                held-out performance rather than worst-case.
            max_pfer: Upper bound on the (approximate) per-family error rate
                the calibration is allowed to accept. Configs whose PFER bound
                ``K² / ((2π−1)·N)`` exceeds this are excluded, capping the
                selection size K and threshold π so the error control can't go
                vacuous. ``None`` (or a non-positive value, normalised by the
                caller) disables the cap.
            seed: RNG seed for reproducibility.
            cancel_callback: Bumped from the runner so a user-cancelled job
                aborts the bootstrap loop cleanly.
            progress_callback: ``(trials_done, trials_total)`` called after each
                trial, where the counts span every subsample of this study.

        Selection is two-stage: stage 1 picks the weight cell (channel mix)
        as above; stage 2 re-bins that cell's trials by a coarse radius key
        (active channels only, width ``radius_bin_m`` — auto-derived to ~3
        buckets across the ladder when ``None``) and picks the radius
        sub-cell with the best q_worst. The final params are averaged within
        that sub-cell, so the reported radii are a validated configuration
        rather than a mean across disagreeing trials.

        Returns: a final-params dict (snapped weights, radii, stats) plus
        bookkeeping keys
        ``__cell_q_worst__``, ``__cell_count__``, ``__cell_median__``,
        ``__cell_selection_probability__``, ``__n_bootstraps__``,
        ``__n_trials_per_bootstrap__``, ``__n_total_trials__``,
        ``__worst_quantile__``, stage-2 keys ``__radius_cell_q_worst__``,
        ``__radius_cell_median__``, ``__radius_cell_count__``,
        ``__radius_bin_m__``, ``__radius_cell_stats__``, and the per-trial
        ``__trial_history__`` table.
        """
        from statistics import mode

        if self.train_val_data is None:
            raise ValueError(
                "Call split_data() (or the runner's split stage) before "
                "bootstrap_stability_selection() — there's no train+val "
                "pool to resample."
            )

        # Direction-aware: higher-is-better → q_worst is the lower tail
        # (e.g. q10), and we pick the cell with the *highest* q_worst.
        # Lower-is-better (RMSE) inverts both.
        higher_is_better = metric in objective_scoring.HIGHER_IS_BETTER or (
            self.is_longitudinal and metric in mixed_effects_scoring.HIGHER_IS_BETTER
        )

        # Sampling unit. Polygon-mode targets must resample *polygons* —
        # row-level resampling would split a single polygon's pixels across
        # in-bag and OOB and produce a leaky OOB set. For row-keyed targets
        # we fall back to plain row resampling.
        train_val = self.train_val_data
        use_groups = "polygon_id" in train_val.columns
        # Resolve the formula descriptor up front — the per-bootstrap
        # summary block inside the loop needs ``weight_keys`` to record
        # the leader's weights, and the cell-stats block after the loop
        # uses the same descriptor for cell-key decoding.
        formula = cgi_formulas.get_formula(self.cgi_formula)
        if use_groups:
            group_col = "polygon_id"
            groups = pd.Series(train_val[group_col].unique())
        else:
            group_col = None
            groups = pd.Series(np.arange(len(train_val)))

        # Spatial block resampling: resample whole grid blocks instead of
        # individual groups so the out-of-bag set is genuinely out-of-region
        # and the worst-quantile OOB score stops rewarding spatial leakage.
        # Each block carries all of its groups; degenerate or too-coarse block
        # layouts fall back to plain group resampling.
        spatial_resample_ok = (
            bool(spatial_resample)
            and use_groups
            and self._spatial_block_by_group is not None
            and self._spatial_block_group_col == group_col
        )
        blocks = np.empty(0, dtype=np.int64)
        groups_in_block: dict = {}
        if spatial_resample_ok:
            block_of = self._spatial_block_by_group
            g_arr = groups.to_numpy()
            block_labels = np.empty(len(g_arr), dtype=np.int64)
            singleton = -1
            for i, g in enumerate(g_arr):
                bk = block_of.get(g)
                if bk is None:
                    block_labels[i] = singleton
                    singleton -= 1
                else:
                    block_labels[i] = int(bk)
            for g, bk in zip(g_arr, block_labels):
                groups_in_block.setdefault(int(bk), []).append(g)
            groups_in_block = {
                bk: np.asarray(gs, dtype=g_arr.dtype)
                for bk, gs in groups_in_block.items()
            }
            blocks = np.asarray(sorted(groups_in_block.keys()), dtype=np.int64)
            if len(blocks) < 4:
                spatial_resample_ok = False
                logger.warning(
                    "Spatial block resampling requested but only "
                    f"{len(blocks)} block(s) cover the train+val pool; "
                    "falling back to group resampling."
                )

        # Save engine state we're about to splice over. Restored in ``finally``
        # so a cancelled / failing bootstrap loop never leaves the engine in a
        # half-mutated state for downstream callers.
        prev_cv_folds = self.cv_folds
        prev_study = self.study
        prev_cancel_cb = getattr(self, "_cancel_callback", None)
        prev_optuna_verbosity = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # The runner passes its own cancellation hook — install it so
        # ``_objective`` can short-circuit when the user stops the job. We
        # also poll between bootstraps to abort the whole loop.
        if cancel_callback is not None:
            self._cancel_callback = cancel_callback

        # Ring caches are keyed by ``points_gdf.index`` so per-bootstrap
        # entries don't collide, but they accumulate at ~thousands of arrays
        # over the loop. Start clean and clear between bootstraps so memory
        # stays flat regardless of B / N_per_BS choice.
        self._clear_ring_caches()

        # Group → row positions once, so each bootstrap's in-bag materialisation
        # is a lookup + one positional take instead of an O(G·N) boolean scan
        # per group. Positions rather than row copies: the pool is stored once.
        group_rows: dict = {}
        if use_groups:
            _keys = train_val[group_col].to_numpy()
            _uniq, _inverse = np.unique(_keys, return_inverse=True)
            _order = np.argsort(_inverse, kind="stable")
            _bounds = np.searchsorted(_inverse[_order], np.arange(len(_uniq) + 1))
            for _i, _g in enumerate(_uniq):
                group_rows[_g] = _order[_bounds[_i] : _bounds[_i + 1]]

        records: list[dict] = []  # one entry per completed trial across all bootstraps
        per_bootstrap_top_cells: list[set] = []  # selection-probability sidecar
        per_bootstrap_summary: list[dict] = []  # one row per bootstrap for the UI table
        # Per-bootstrap diagnostics — used to build a useful error message
        # when zero trials end up usable so the caller can tell apart
        # "every resample was degenerate" from "objective returned NaN".
        diag = {
            "bootstraps_attempted": 0,
            "bootstraps_degenerate_split": 0,
            "bootstraps_zero_completed": 0,
            "bootstraps_all_nan_scores": 0,
            "bootstraps_with_records": 0,
            "trials_complete": 0,
            "trials_pruned": 0,
            "trials_failed": 0,
            "trials_nan_score": 0,
            "trials_seed_excluded": 0,
            "trials_kept": 0,
        }
        # Search cost, accumulated across the sequential bootstrap loop so the
        # phase can report its own concurrency at the end.
        search_secs = 0.0
        search_trials = 0
        search_jobs = 1

        try:
            rng = np.random.default_rng(int(seed))
            # Complementary-pairs subsampling (Shah & Samworth 2013): each
            # pair splits the units into halves scored on each other. This is
            # the ⌊n/2⌋ scheme the Meinshausen–Bühlmann PFER bound assumes.
            # Spatial resampling splits whole blocks, so each half is
            # out-of-region.
            n_pairs = max(1, int(round(int(n_bootstraps) / 2)))
            subsamples: list[tuple[np.ndarray, np.ndarray]] = []
            for _pair in range(n_pairs):
                if spatial_resample_ok:
                    perm_blocks = rng.permutation(blocks)
                    cut_b = len(perm_blocks) // 2

                    def _units(bk_arr: np.ndarray) -> np.ndarray:
                        if len(bk_arr) == 0:
                            return np.empty(0, dtype=groups.to_numpy().dtype)
                        return np.unique(
                            np.concatenate([groups_in_block[int(bk)] for bk in bk_arr])
                        )

                    half_a = _units(perm_blocks[:cut_b])
                    half_b = _units(perm_blocks[cut_b:])
                else:
                    perm_units = rng.permutation(groups.to_numpy())
                    cut_u = len(perm_units) // 2
                    half_a = np.unique(perm_units[:cut_u])
                    half_b = np.unique(perm_units[cut_u:])
                subsamples.append((half_a, half_b))
                subsamples.append((half_b, half_a))

            total_resamples = len(subsamples)
            total_trials = total_resamples * int(n_trials_per_bootstrap)
            for b, (in_bag_unique, oob_unique) in enumerate(subsamples):
                if cancel_callback is not None and cancel_callback():
                    break
                # Each subsample has its own index tuples, so cached annuli never
                # carry over — clear to keep memory flat.
                self._clear_ring_caches()
                diag["bootstraps_attempted"] += 1
                if len(oob_unique) < 3 or len(in_bag_unique) < 3:
                    # Degenerate half — too few units in-bag or OOB. Skip, but
                    # advance the trial bar past this subsample's allotment so
                    # the UI doesn't stall.
                    diag["bootstraps_degenerate_split"] += 1
                    if progress_callback is not None:
                        try:
                            progress_callback(
                                (b + 1) * int(n_trials_per_bootstrap), total_trials
                            )
                        except Exception:
                            pass
                    continue

                if use_groups:
                    # One positional take per half, preserving the original
                    # DataFrame index the pre-aggregation cache is keyed by.
                    # Both halves index the same pooled frame — no per-subsample
                    # isin scan over the pixel frame and no per-group copies.
                    in_bag_df = train_val.iloc[
                        np.concatenate([group_rows[g] for g in in_bag_unique])
                    ]
                    oob_df = train_val.iloc[
                        np.concatenate([group_rows[g] for g in oob_unique])
                    ]
                else:
                    # iloc with an index array materialises a new frame; the
                    # fold frames are read-only downstream.
                    in_bag_df = train_val.iloc[in_bag_unique]
                    oob_df = train_val.iloc[oob_unique]

                # Splice this subsample into ``cv_folds`` — the existing
                # ``_objective`` reads ``train`` / ``val`` from each fold and
                # handles aggregation, channel collapse, and OOB scoring.
                self.cv_folds = [{"train": in_bag_df, "val": oob_df}]
                self._split_generation += 1
                # Each resample is a different row set, so its spatial basis is
                # rebuilt; drop the previous resample's caches to bound memory
                # (the pdcor / spline caches fingerprint the same row sets).
                self._spatial_basis_cache = {}
                self._clear_scoring_caches()

                # Fresh in-memory study with a low-discrepancy QMC (Sobol)
                # sampler: objective-blind like RandomSampler (so the per-cell
                # selection frequencies the calibration relies on stay
                # unbiased), but it fills the high-dimensional CGI search space
                # far more evenly at the same trial budget. An objective-aware
                # sampler (TPE / CMA-ES) would concentrate on this resample's
                # local optimum and inflate the stability counts.
                import warnings as _warnings

                from optuna.exceptions import ExperimentalWarning as _ExpWarning

                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore", _ExpWarning)
                    study = optuna.create_study(
                        direction="maximize" if higher_is_better else "minimize",
                        sampler=optuna.samplers.QMCSampler(
                            qmc_type="sobol",
                            scramble=True,
                            seed=int(seed) + b,
                            warn_independent_sampling=False,
                        ),
                    )
                self.study = study
                # Seed each bootstrap with the single-channel vertices + the
                # centroid so the CGI search always evaluates the configurations
                # the standalone studies explore (CGI nests every standalone).
                if self._active_greenery_channel == "cgi":
                    for seed_params in cgi_formulas.seed_param_sets(
                        self.cgi_formula, self._disabled_channels
                    ):
                        try:
                            study.enqueue_trial(seed_params, skip_if_exists=True)
                        except Exception:
                            pass
                # Per-trial progress: report the global trial index across all
                # subsamples so the UI can show a live "k / N trials" bar for
                # this study.
                n_trials_b = int(n_trials_per_bootstrap)

                def _trial_progress(_study, _trial, _b=b):
                    if progress_callback is None:
                        return
                    try:
                        progress_callback(
                            _b * n_trials_b + _trial.number + 1, total_trials
                        )
                    except Exception:
                        pass

                def _stop_if_cancelled(_study, _trial):
                    """End this subsample's study as soon as cancel is seen.

                    ``optimize`` is otherwise uninterruptible for the whole
                    ``n_trials_per_bootstrap`` budget, which is the difference
                    between a cancel that lands in seconds and one that lands
                    a bootstrap later.
                    """
                    if cancel_callback is not None and cancel_callback():
                        _study.stop()

                _search_jobs = self._search_n_jobs(metric)
                _search_t0 = time.perf_counter()
                try:
                    study.optimize(
                        lambda t: self._objective(t, metric),
                        n_trials=n_trials_b,
                        n_jobs=_search_jobs,
                        show_progress_bar=False,
                        catch=(Exception,),
                        callbacks=[_trial_progress, _stop_if_cancelled],
                    )
                except Exception:
                    # Catastrophic study failure — skip this subsample and
                    # continue rather than aborting the whole selection.
                    continue
                finally:
                    search_secs += time.perf_counter() - _search_t0
                    search_trials += n_trials_b
                    search_jobs = _search_jobs

                # Extract per-trial (params, val_score). ``_objective`` puts
                # the mean val score into ``user_attrs["val_score_mean"]`` and
                # returns the same number as ``trial.value``.
                for _t in study.trials:
                    state = _t.state
                    if state == optuna.trial.TrialState.COMPLETE:
                        diag["trials_complete"] += 1
                    elif state == optuna.trial.TrialState.PRUNED:
                        diag["trials_pruned"] += 1
                    elif state == optuna.trial.TrialState.FAIL:
                        diag["trials_failed"] += 1
                completed = [
                    t
                    for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                ]
                if not completed:
                    diag["bootstraps_zero_completed"] += 1
                    continue
                bootstrap_records: list[dict] = []
                for t in completed:
                    # Enqueued seed trials (single-channel vertices + centroid)
                    # are non-random, so they'd give their cells a guaranteed
                    # selection in every resample and bias the calibration.
                    # Keep them for search coverage; drop them from the counts.
                    if "fixed_params" in t.system_attrs:
                        diag["trials_seed_excluded"] += 1
                        continue
                    score = t.user_attrs.get("val_score_mean", t.value)
                    if score is None or not np.isfinite(float(score)):
                        diag["trials_nan_score"] += 1
                        continue
                    diag["trials_kept"] += 1
                    bootstrap_records.append(
                        {
                            "bootstrap": b,
                            "params": dict(t.params),
                            "oob_score": float(score),
                            "cell": cgi_formulas.weight_cell_key(
                                self.cgi_formula, dict(t.params), weight_bin_pct
                            ),
                        }
                    )

                # Selection-probability sidecar: which cells landed in this
                # bootstrap's top-X %? Reported alongside q_worst so the user
                # can sanity-check ("this region won often AND won well").
                if bootstrap_records:
                    diag["bootstraps_with_records"] += 1
                    bootstrap_records.sort(
                        key=lambda r: r["oob_score"], reverse=higher_is_better
                    )
                    n_top = max(
                        1, int(len(bootstrap_records) * top_percent_per_bootstrap)
                    )
                    top_cells = {r["cell"] for r in bootstrap_records[:n_top]}
                    per_bootstrap_top_cells.append(top_cells)
                    records.extend(bootstrap_records)

                    # Per-bootstrap row for the UI table: the leader, its
                    # cell, and the spread of OOB scores within this
                    # resample. Lets the user see if all bootstraps agree
                    # on a region or scatter wildly.
                    top_record = bootstrap_records[0]
                    all_oob = [r["oob_score"] for r in bootstrap_records]
                    per_bootstrap_summary.append(
                        {
                            "bootstrap": int(b),
                            "n_trials": int(len(bootstrap_records)),
                            "top_oob": float(top_record["oob_score"]),
                            "top_cell": tuple(int(x) for x in top_record["cell"]),
                            "top_params": {
                                k: top_record["params"].get(k)
                                for k in formula.weight_keys
                                if k in top_record["params"]
                            },
                            "mean_oob": float(np.mean(all_oob)),
                            "median_oob": float(np.median(all_oob)),
                            "min_oob": float(np.min(all_oob)),
                            "max_oob": float(np.max(all_oob)),
                        }
                    )
                else:
                    diag["bootstraps_all_nan_scores"] += 1

        finally:
            self.cv_folds = prev_cv_folds
            self.study = prev_study
            self._cancel_callback = prev_cancel_cb
            optuna.logging.set_verbosity(prev_optuna_verbosity)
            # The last resample's cached matrices (up to hundreds of MB for
            # the pdcor sides) have no future hits — release them now.
            self._clear_scoring_caches()
            self._clear_ring_caches()
            if search_secs > 0 and search_trials:
                _log(
                    "INFO",
                    f"Stability search: {search_trials:,} trial(s) over "
                    f"{diag['bootstraps_attempted']} sequential bootstrap(s) in "
                    f"{search_secs:.0f}s "
                    f"({search_secs / search_trials * 1000:.1f} ms/trial, "
                    f"n_jobs={search_jobs}).",
                )
                # Summed in-objective time across threads: divided by the phase's
                # wall clock it gives the average number of workers actually
                # running, which is measured rather than assumed.
                _log_parallel_efficiency(
                    _log,
                    "stability search",
                    wall_s=search_secs,
                    busy_s=float(sum(self._objective_secs)),
                    workers=search_jobs,
                )
            self._objective_secs.clear()

        if not records:
            raise RuntimeError(
                "bootstrap_stability_selection produced zero usable trials. "
                "Diagnostics: "
                f"bootstraps_attempted={diag['bootstraps_attempted']}, "
                f"degenerate_split={diag['bootstraps_degenerate_split']}, "
                f"zero_completed={diag['bootstraps_zero_completed']}, "
                f"all_nan_scores={diag['bootstraps_all_nan_scores']}, "
                f"with_records={diag['bootstraps_with_records']}, "
                f"trials_complete={diag['trials_complete']}, "
                f"trials_pruned={diag['trials_pruned']}, "
                f"trials_failed={diag['trials_failed']}, "
                f"trials_nan_score={diag['trials_nan_score']}, "
                f"trials_kept={diag['trials_kept']}. "
                "If most trials were pruned → variance-zero composites "
                "(check that the pre-aggregation cache covers the bootstrap "
                "entity_ids). If most trials returned NaN → check "
                "covariate / target NaN coverage on OOB rows."
            )

        # ── Per-cell aggregation ────────────────────────────
        from . import statistical_testing as _stats_mod

        cells: dict[tuple, list[dict]] = {}
        for r in records:
            cells.setdefault(r["cell"], []).append(r)

        def _q_worst(scores: list[float]) -> float:
            # Note: a weight cell pools trials across resamples *and* across
            # radii / aggregation stats, so q_worst mixes resample-luck variance
            # with within-cell hyperparameter variance. It's a secondary
            # tie-breaker here; stage-2 re-selects the radii at a fixed cell.
            if higher_is_better:
                return float(np.quantile(scores, worst_quantile))
            return float(np.quantile(scores, 1.0 - worst_quantile))

        # ── Threshold calibration (stage 1: channel mix) ────
        # Rank cells within each bootstrap by their best out-of-bag score, then
        # calibrate the selection size K and threshold π by maximizing the
        # stability score (Bodinier et al.) — no hand-set threshold. The
        # channel-mix winner is the most consistently top-ranked cell in the
        # calibrated stable set (a reproducibility statement), with q_worst
        # kept as a secondary performance diagnostic.
        by_bootstrap: dict[int, dict[tuple, list[float]]] = {}
        for r in records:
            by_bootstrap.setdefault(r["bootstrap"], {}).setdefault(
                r["cell"], []
            ).append(r["oob_score"])
        rankings: list[list[tuple]] = []
        for _bs, cellmap in by_bootstrap.items():
            rep = {
                cell: (max(s) if higher_is_better else min(s))
                for cell, s in cellmap.items()
            }
            rankings.append(sorted(rep, key=lambda c: rep[c], reverse=higher_is_better))
        n_candidate_cells = len(cells)
        calib = _stats_mod.calibrate_stability_selection(
            rankings, n_candidate_cells, max_pfer=max_pfer
        )
        n_bs_used = max(1, len(rankings))

        if calib is not None:
            calib_counts = calib["selection_counts"]
            calib_b = calib["n_resamples"]
            sel_prob_of = {c: calib_counts.get(c, 0) / calib_b for c in cells}
        else:
            calib_counts = {}
            sel_prob_of = {
                c: sum(1 for s in per_bootstrap_top_cells if c in s) / n_bs_used
                for c in cells
            }

        cell_stats: list[dict] = []
        for cell_key, rs in cells.items():
            scores = [r["oob_score"] for r in rs]
            cell_stats.append(
                {
                    "cell": cell_key,
                    "count": len(rs),
                    "q_worst": _q_worst(scores),
                    "median": float(np.median(scores)),
                    "selection_probability": float(sel_prob_of.get(cell_key, 0.0)),
                    "records": rs,
                }
            )

        if calib is not None:
            hi_count = int(np.ceil(calib["pi"] * calib_b))
            stable_cells = {c for c, h in calib_counts.items() if h >= hi_count}
            candidates = [c for c in cell_stats if c["cell"] in stable_cells]
            if not candidates:
                candidates = cell_stats
            # Most consistently selected cell; ties broken by the cell's median
            # OOB score. Median, not the worst quantile: the question this
            # study asks is which composite predicts better on typical held-out
            # data, and the worst decile answers a maximin question instead.
            # It is also count-invariant, so a cell that happened to draw more
            # trials gets no advantage. ``q_worst`` is still recorded on every
            # cell and reported as a robustness read.
            best = max(
                candidates,
                key=lambda c: (
                    c["selection_probability"],
                    c["median"] if higher_is_better else -c["median"],
                ),
            )
            _log(
                "INFO",
                f"Stability calibration: K={calib['K']}, π={calib['pi']:.2f}, "
                f"score={calib['score']:.1f}, {calib['n_stably_selected']} of "
                f"{n_candidate_cells} cells stable, PFER≤{calib['pfer']:.2f}; "
                f"winner selection prob={best['selection_probability']:.2f}.",
            )
        else:
            # Too few candidate cells / resamples to calibrate (e.g. a
            # standalone's single weight cell) — fall back to the best median
            # cell.
            best = max(
                cell_stats,
                key=lambda c: c["median"] if higher_is_better else -c["median"],
            )

        # ── Stage 2: spatial tuning in the winning cell ─────
        # The weight cell fixes the channel mix; now stability-select the
        # radii the same way — re-bin the cell's trials by a coarse radius
        # key (active channels only) and pick the radius sub-cell with the
        # best q_worst. This stops the final radii from being a mean of
        # disagreeing values that no trial actually validated.
        active_ch = getattr(self, "_active_greenery_channel", "cgi") or "cgi"
        radius_keys: tuple[str, ...] = {
            "veg": ("veg_radius",),
            "terrain": ("terrain_radius",),
            "ndvi": ("ndvi_radius",),
        }.get(active_ch, ("veg_radius", "terrain_radius", "ndvi_radius"))

        if radius_bin_m is None:
            # Coarsen to ~3 buckets across the active channels' ladder so the
            # sub-cells stay populated within one weight cell's trials.
            gvi_r = self._outer_radii_metres(gvi=True)
            ndvi_r = self._outer_radii_metres(gvi=False)
            spans: list[float] = []
            if active_ch in ("veg", "terrain", "cgi") and gvi_r.size:
                spans.append(float(gvi_r.max()) - float(gvi_r.min()))
            if active_ch in ("ndvi", "cgi") and ndvi_r.size:
                spans.append(float(ndvi_r.max()) - float(ndvi_r.min()))
            span = max(spans) if spans else 0.0
            radius_bin_eff = (
                max(1, int(round(span / 3.0)))
                if span > 0
                else int(cgi_formulas.RADIUS_BIN_M)
            )
        else:
            radius_bin_eff = max(1, int(round(float(radius_bin_m))))

        radius_cells: dict[tuple, list[dict]] = {}
        for r in best["records"]:
            rk = cgi_formulas.radius_cell_key(
                r["params"], radius_keys, (), radius_bin_eff
            )
            radius_cells.setdefault(rk, []).append(r)

        def _radius_stat(rk: tuple, rs: list[dict]) -> dict:
            s = [r["oob_score"] for r in rs]
            return {
                "cell": rk,
                "count": len(rs),
                "q_worst": _q_worst(s),
                "median": float(np.median(s)),
                "records": rs,
            }

        radius_stats = [
            _radius_stat(rk, rs)
            for rk, rs in radius_cells.items()
            if len(rs) >= int(min_cell_count)
        ]
        if not radius_stats:
            # Same fallback as stage 1: no radius sub-cell met the count
            # threshold, so rank every sub-cell regardless of count.
            logger.warning(
                "No radius sub-cell reached min_cell_count="
                f"{min_cell_count} within the winning weight cell; ranking "
                "all radius sub-cells regardless of count. Consider raising "
                "n_trials_per_bootstrap."
            )
            radius_stats = [_radius_stat(rk, rs) for rk, rs in radius_cells.items()]
        radius_best = max(
            radius_stats,
            key=lambda c: c["median"] if higher_is_better else -c["median"],
        )

        # ── Average params in the winning radius sub-cell ───
        # ── Stage 3: weight refinement at the chosen scale ──
        # The coarse cell fixed the channel mix and the radius sub-cell fixed
        # the spatial scale. Both were decided at 20 % weight resolution, so
        # the surviving trials still disagree about the mix by up to a whole
        # bucket. Re-bin them at ``weight_refine_bin_pct`` and take the best
        # fine sub-cell by median, instead of averaging across the coarse cell
        # and landing on a mix no trial actually validated.
        #
        # Ordering matters and is deliberate: mix, then scale, then mix again.
        # Refining the weights before the radius is settled would tune them
        # against a spatial scale that is about to change.
        #
        # Note the fine keys do not nest inside the coarse ones —
        # ``_snap_weight_buckets`` rounds to nearest, so the two grids have
        # unaligned boundaries. That is fine here because the re-binning is
        # applied to trials that are *already* members of the winning coarse
        # cell, so the refined set is a subset of them and the averaged weights
        # stay inside their convex hull. Do not reuse these keys as a
        # hierarchy elsewhere.
        refine_bin = int(
            weight_refine_bin_pct
            if weight_refine_bin_pct is not None
            else cgi_formulas.WEIGHT_REFINE_BIN_PCT
        )
        winners = radius_best["records"]
        refine_stats: list[dict] = []
        if 0 < refine_bin < int(weight_bin_pct):
            fine_cells: dict[tuple, list[dict]] = {}
            for r in winners:
                fk = cgi_formulas.weight_cell_key(
                    self.cgi_formula, r["params"], refine_bin
                )
                fine_cells.setdefault(fk, []).append(r)
            refine_stats = [
                {
                    "cell": fk,
                    "count": len(rs),
                    "median": float(np.median([r["oob_score"] for r in rs])),
                    "q_worst": _q_worst([r["oob_score"] for r in rs]),
                    "records": rs,
                }
                for fk, rs in fine_cells.items()
            ]
            # Only accept the refinement when a sub-cell has enough trials to
            # rank on. Below that its median is a draw-noise artefact and the
            # coarse cell's average is the safer answer.
            eligible = [c for c in refine_stats if c["count"] >= int(min_cell_count)]
            if len(fine_cells) > 1 and eligible:
                refine_best = max(
                    eligible,
                    key=lambda c: c["median"] if higher_is_better else -c["median"],
                )
                _log(
                    "INFO",
                    f"Weight refinement: {len(fine_cells)} sub-cells at "
                    f"{refine_bin}% inside the winning {weight_bin_pct}% cell; "
                    f"kept {refine_best['count']} of {len(winners)} trials "
                    f"(median {refine_best['median']:.4f} vs "
                    f"{radius_best['median']:.4f} across the whole cell).",
                )
                winners = refine_best["records"]
            elif len(fine_cells) > 1:
                _log(
                    "INFO",
                    f"Weight refinement skipped: no {refine_bin}% sub-cell "
                    f"reached min_cell_count={min_cell_count}; averaging the "
                    f"{len(winners)} trials of the coarse cell instead.",
                )

        # Snap + renormalize the winners' weights to canonical integer steps so
        # downstream code (composite generation, apply path) consumes them
        # directly. ``formula`` was resolved up front (see above).

        def _mode_str(name: str, default: str) -> str:
            vals = [r["params"].get(name, default) for r in winners]
            try:
                return mode(vals)
            except Exception:
                return vals[0] if vals else default

        gvi_radii, ndvi_radii = self._preaggr_radii()
        gvi_choices = np.asarray(gvi_radii) if gvi_radii else None
        ndvi_choices = np.asarray(ndvi_radii) if ndvi_radii else None

        def _snap(value: float, choices: np.ndarray | None) -> int:
            if choices is None or len(choices) == 0:
                return int(round(value))
            idx = int(np.argmin(np.abs(choices - value)))
            return int(choices[idx])

        pct_grid = np.asarray(preaggregation.PERCENTILES)

        def _snap_pct(value: float) -> int:
            idx = int(np.argmin(np.abs(pct_grid - value)))
            return int(pct_grid[idx])

        final_params: dict[str, Any] = {
            "veg_radius": _snap(
                float(
                    np.mean(
                        [
                            r["params"].get("veg_radius", self.gvi_buffer_max_m)
                            for r in winners
                        ]
                    )
                ),
                gvi_choices,
            ),
            "terrain_radius": _snap(
                float(
                    np.mean(
                        [
                            r["params"].get("terrain_radius", self.gvi_buffer_max_m)
                            for r in winners
                        ]
                    )
                ),
                gvi_choices,
            ),
            "ndvi_radius": _snap(
                float(
                    np.mean(
                        [
                            r["params"].get("ndvi_radius", self.ndvi_buffer_max_m)
                            for r in winners
                        ]
                    )
                ),
                ndvi_choices,
            ),
            "streetview_stat": _mode_str("streetview_stat", "mean"),
            "ndvi_stat": _mode_str("ndvi_stat", "mean"),
            "streetview_percentile": _snap_pct(
                float(
                    np.mean(
                        [r["params"].get("streetview_percentile", 50) for r in winners]
                    )
                )
            ),
            "ndvi_percentile": _snap_pct(
                float(
                    np.mean([r["params"].get("ndvi_percentile", 50) for r in winners])
                )
            ),
        }
        for power_key in formula.power_keys:
            final_params[power_key] = float(
                np.mean([r["params"].get(power_key, 1.0) for r in winners])
            )
        avg_weights = {
            k: float(np.mean([r["params"].get(k, 0.0) for r in winners]))
            for k in formula.weight_keys
        }
        weight_sum = sum(avg_weights.values())
        if weight_sum > 0:
            # Renormalize to sum=100 on the 5% recorded grid. Snap to 5% steps
            # via largest-remainder so the downstream apply path sees
            # canonical integer weights compatible with the picker's output.
            step = cgi_formulas.WEIGHT_STEP_PCT
            n_slots = 100 // step
            scaled = [avg_weights[k] / weight_sum * (100.0 / step) for k in avg_weights]
            floors = [int(np.floor(s)) for s in scaled]
            remainder = n_slots - sum(floors)
            if remainder > 0:
                order = sorted(
                    range(len(scaled)),
                    key=lambda i: scaled[i] - floors[i],
                    reverse=True,
                )
                for i in order[:remainder]:
                    floors[i] += 1
            for k, slots in zip(list(avg_weights), floors):
                final_params[k] = int(slots) * step
        else:
            for k in avg_weights:
                final_params[k] = 0
            # Standalone fallback — keep the active channel at 100 % so the
            # composite generator can still build a meaningful raster.
            ch = getattr(self, "_active_greenery_channel", "cgi") or "cgi"
            channel_to_keys = {
                "veg": ("veg_weight", "w_veg"),
                "terrain": ("terrain_weight", "w_ter"),
                "ndvi": ("ndvi_weight", "w_ndvi"),
            }
            for candidate in channel_to_keys.get(ch, ()):
                if candidate in avg_weights:
                    final_params[candidate] = 100
                    break

        final_params["__cell_q_worst__"] = float(best["q_worst"])
        final_params["__cell_count__"] = int(best["count"])
        final_params["__cell_median__"] = float(best["median"])
        final_params["__weight_bin_pct__"] = int(weight_bin_pct)
        final_params["__weight_refine_bin_pct__"] = int(refine_bin)
        final_params["__refine_cell_count__"] = int(len(winners))
        final_params["__refine_applied__"] = bool(
            refine_stats and len(winners) < len(radius_best["records"])
        )
        final_params["__cell_selection_probability__"] = float(
            best["selection_probability"]
        )
        final_params["__n_bootstraps__"] = int(n_bs_used)
        final_params["__n_trials_per_bootstrap__"] = int(n_trials_per_bootstrap)
        final_params["__n_total_trials__"] = int(len(records))
        final_params["__worst_quantile__"] = float(worst_quantile)
        # Automated threshold calibration (Bodinier) diagnostics: the
        # calibrated selection size K, threshold π, stability score, stable-set
        # size, candidate count, and the PFER upper bound (rigorous under the
        # ⌊n/2⌋ complementary-pairs subsampling). Absent when calibration was
        # skipped.
        if calib is not None:
            final_params["__stability_score__"] = float(calib["score"])
            final_params["__selection_threshold__"] = float(calib["pi"])
            final_params["__selection_size_k__"] = int(calib["K"])
            final_params["__n_candidate_cells__"] = int(calib["n_candidates"])
            final_params["__n_stably_selected__"] = int(calib["n_stably_selected"])
            final_params["__pfer__"] = float(calib["pfer"])
            final_params["__pfer_controlled__"] = bool(
                calib.get("pfer_controlled", True)
            )
        # Stage-2 (radius sub-cell) diagnostics: the spatial-tuning winner
        # within the chosen weight cell. ``__cell_*__`` above describe the
        # channel-mix decision; these describe the radii the final params
        # were actually averaged from.
        final_params["__radius_cell_count__"] = int(radius_best["count"])
        final_params["__radius_cell_q_worst__"] = float(radius_best["q_worst"])
        final_params["__radius_cell_median__"] = float(radius_best["median"])
        final_params["__radius_bin_m__"] = int(radius_bin_eff)
        ranked_radius = sorted(
            radius_stats,
            key=lambda c: c["q_worst"] if higher_is_better else -c["q_worst"],
            reverse=True,
        )
        final_params["__radius_cell_stats__"] = [
            {
                "radii": {
                    radius_keys[i]: int(c["cell"][i] * radius_bin_eff)
                    for i in range(len(radius_keys))
                },
                "count": int(c["count"]),
                "q_worst": float(c["q_worst"]),
                "median": float(c["median"]),
            }
            for c in ranked_radius[:10]
        ]
        # Diagnostics for the results UI, under ``__`` keys so the "Final
        # params" panel strips them out. The cell keys on the main-component
        # weights only, so map the tuple back through those keys.
        cell_weight_keys = formula.main_weight_keys
        # Rank the diagnostics table by the decision criterion — calibrated
        # selection probability, with worst-quantile OOB breaking ties — so the
        # top row is the chosen winner; median is shown alongside as a
        # diagnostic.
        ranked = sorted(
            cell_stats,
            key=lambda c: (
                c["selection_probability"],
                c["q_worst"] if higher_is_better else -c["q_worst"],
            ),
            reverse=True,
        )
        final_params["__cell_stats__"] = [
            {
                "weights": {
                    k: int(c["cell"][i] * weight_bin_pct)
                    for i, k in enumerate(cell_weight_keys)
                },
                "count": int(c["count"]),
                "q_worst": float(c["q_worst"]),
                "median": float(c["median"]),
                "selection_probability": float(c["selection_probability"]),
            }
            for c in ranked[:10]
        ]
        final_params["__winning_cell_oob_scores__"] = [
            float(r["oob_score"]) for r in best["records"]
        ]
        final_params["__per_bootstrap_summary__"] = per_bootstrap_summary
        final_params["__higher_is_better__"] = bool(higher_is_better)

        # Per-trial history: every completed trial across all subsamples, with
        # its resample id, OOB score, snapped weight cell, flattened weights /
        # radii, and whether the trial's cell was in that resample's calibrated
        # top-K. This is the stability-selection analogue of an Optuna trial
        # log — a tidy table the results UI renders (OOB spread per resample,
        # cell-selection frequency, weight×score) and the MixedLM post-scoring
        # re-scores from. Cells are stored as lists so the bundle stays
        # JSON-serializable.
        k_for_top = int(calib["K"]) if calib is not None else None
        top_cells_by_bs: dict[int, set] = {}
        for bs, cellmap in by_bootstrap.items():
            rep = {
                cell: (max(s) if higher_is_better else min(s))
                for cell, s in cellmap.items()
            }
            order = sorted(rep, key=lambda c: rep[c], reverse=higher_is_better)
            k = (
                k_for_top
                if k_for_top is not None
                else max(1, int(len(order) * top_percent_per_bootstrap))
            )
            top_cells_by_bs[bs] = set(order[:k])
        trial_history: list[dict] = []
        for r in records:
            row: dict[str, Any] = {
                "bootstrap": int(r["bootstrap"]),
                "oob_score": float(r["oob_score"]),
                "cell": [int(x) for x in r["cell"]],
                "in_top_k": bool(
                    r["cell"] in top_cells_by_bs.get(r["bootstrap"], set())
                ),
            }
            # Full per-trial params (weights, radii, stats, percentiles,
            # powers) so the history both drives the UI panels and can be
            # re-scored by the MixedLM post-scoring without re-running the
            # search.
            for pk, pv in r["params"].items():
                row[pk] = pv
            trial_history.append(row)
        final_params["__trial_history__"] = trial_history
        final_params["__winning_cell__"] = [int(x) for x in best["cell"]]
        return final_params

    def generate_composite_greenery_map(
        self,
        output_path: str = "output_results/composite_greenery.tif",
        progress_callback: Callable[..., Any] | None = None,
    ) -> str:
        """Render the composite greenery map from the stability-selected params.

        Builds a composite greenery raster matching the target grid (if raster
        input) or the CGI grid (if vector input) using the winning weight cell's
        averaged parameters that ``bootstrap_stability_selection`` pinned on
        ``self.best_params``.

        Args:
            output_path: Path to save the composite greenery GeoTIFF
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            Path to the saved composite greenery map
        """
        logger.info(
            "Generating composite greenery map from stability-selected params..."
        )

        if progress_callback:
            progress_callback(0, 100)

        if self.best_params is None:
            raise ValueError(
                "Composite generation needs ``self.best_params`` set by a prior "
                "stability-selection step."
            )
        final_params = dict(self.best_params)
        logger.info(f"Composite TIFF stability-selected params: {final_params}")
        formula = cgi_formulas.get_formula(self.cgi_formula)

        if progress_callback:
            progress_callback(30, 100)

        # 4. Determine grid (raster, polygon target, or point/vector)
        if self.is_raster:
            # Use exact target raster grid
            transform = self.target_raster["transform"]
            crs = self.target_raster["crs"]
            height, width = self.target_raster["data"].shape

            # Create point grid at pixel centers
            rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
            xs, ys = xy(transform, rows.flatten(), cols.flatten(), offset="center")

            points_gdf = gpd.GeoDataFrame(
                {"row": rows.flatten(), "col": cols.flatten()},
                geometry=gpd.points_from_xy(xs, ys),
                crs=crs,
            )

            logger.info(
                f"Generating composite for {len(points_gdf)} pixels (target raster grid)"
            )

        elif (
            self.is_polygon_target
            and self._polygon_grid_shape is not None
            and self._polygon_grid_transform is not None
        ):
            # Polygon target: reuse the per-pixel CGI grid the engine
            # already constructed in ``_prepare_polygon_fusion``. Pixels
            # outside the polygon union are written as NaN in the output.
            # Index alignment with the entity cache requires keeping the
            # engine's row index on the new frame so cache lookups hit.
            transform = self._polygon_grid_transform
            crs = self._polygon_grid_crs
            height, width = self._polygon_grid_shape
            grid_rows = self.target_gdf["_grid_row"].to_numpy(dtype=np.int64)
            grid_cols = self.target_gdf["_grid_col"].to_numpy(dtype=np.int64)
            points_gdf = gpd.GeoDataFrame(
                {"row": grid_rows, "col": grid_cols},
                geometry=self.target_gdf.geometry.values,
                crs=crs,
                index=self.target_gdf.index,
            )
            logger.info(
                f"Generating composite for {len(points_gdf)} pixels "
                f"({self.cgi_grid_spacing_m:g} m grid inside polygon union)"
            )

        elif (
            self.is_points
            and self._entity_grid_shape is not None
            and self._entity_grid_transform is not None
        ):
            # Point/line target: render the CGI field over the same per-pixel
            # grid the scorer used (``_prepare_entity_fusion``), so the
            # composite matches the scored values. The catchment pixels were
            # duplicated per entity, so collapse to the unique grid cells.
            transform = self._entity_grid_transform
            crs = self._entity_grid_crs
            height, width = self._entity_grid_shape
            uniq = self.target_gdf.drop_duplicates(subset=["_grid_row", "_grid_col"])
            grid_rows = uniq["_grid_row"].to_numpy(dtype=np.int64)
            grid_cols = uniq["_grid_col"].to_numpy(dtype=np.int64)
            pts_cols = {"row": grid_rows, "col": grid_cols}
            # Carry the unique-pixel id so the cache resolves per-pixel stats
            # (the cache is keyed by pixel, not by the collapsed row index).
            if "_preaggr_id" in uniq.columns:
                pts_cols["_preaggr_id"] = uniq["_preaggr_id"].to_numpy()
            points_gdf = gpd.GeoDataFrame(
                pts_cols,
                geometry=uniq.geometry.values,
                crs=crs,
                index=uniq.index,
            )
            logger.info(
                f"Generating composite for {len(points_gdf)} pixels "
                f"({self.cgi_grid_spacing_m:g} m grid over point/line catchments)"
            )

        else:
            # Create 50m grid within target bounding box
            target_bbox = self.target_gdf.total_bounds
            crs = self.target_gdf.crs

            # If geographic CRS, convert to UTM for accurate 50m spacing
            if crs.is_geographic:
                # Determine UTM zone from geographic coordinates
                centroid = self.target_gdf.union_all().centroid
                lon, lat = centroid.x, centroid.y

                # Validate geographic coordinates
                if not (-180 <= lon <= 180):
                    raise ValueError(
                        f"Invalid longitude {lon}. CRS is marked as geographic "
                        f"but coordinates appear to be projected. CRS: {crs}"
                    )

                utm_zone = int((lon + 180) / 6) + 1
                utm_crs = (
                    f"EPSG:326{utm_zone:02d}" if lat >= 0 else f"EPSG:327{utm_zone:02d}"
                )
                target_utm = self.target_gdf.to_crs(utm_crs)
                minx, miny, maxx, maxy = target_utm.total_bounds
            else:
                # CRS is already projected (e.g., UTM), use it directly
                utm_crs = crs
                minx, miny, maxx, maxy = target_bbox

            # Create 50m grid
            resolution = 50  # meters
            width = int(np.ceil((maxx - minx) / resolution))
            height = int(np.ceil((maxy - miny) / resolution))

            transform = from_origin(minx, maxy, resolution, resolution)
            rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
            xs, ys = xy(transform, rows.flatten(), cols.flatten(), offset="center")

            points_gdf = gpd.GeoDataFrame(
                {"row": rows.flatten(), "col": cols.flatten()},
                geometry=gpd.points_from_xy(xs, ys),
                crs=utm_crs,
            )

            # Clip to target extent
            if crs.is_geographic:
                target_for_clip = self.target_gdf.to_crs(utm_crs)
            else:
                target_for_clip = self.target_gdf

            points_gdf = gpd.clip(points_gdf, target_for_clip)

            logger.info(
                f"Generating composite for {len(points_gdf)} points (50m grid within target extent)"
            )

            # Update grid dimensions after clipping
            crs = utm_crs

        if progress_callback:
            progress_callback(40, 100)

        # 5. Sample metrics at grid points using final parameters
        logger.info("Sampling vegetation at grid points...")
        veg_values = self._aggregate_with_ring_cache(
            points_gdf,
            self.veg_data,
            final_params["veg_radius"],
            final_params["streetview_stat"],
            final_params["streetview_percentile"],
            channel="veg",
            fold_idx=-1,
            subset="robust_map",
        )

        if progress_callback:
            progress_callback(55, 100)

        logger.info("Sampling terrain at grid points...")
        terrain_values = self._aggregate_with_ring_cache(
            points_gdf,
            self.terrain_data,
            final_params["terrain_radius"],
            final_params["streetview_stat"],
            final_params["streetview_percentile"],
            channel="terrain",
            fold_idx=-1,
            subset="robust_map",
        )

        if progress_callback:
            progress_callback(70, 100)

        logger.info("Sampling NDVI at grid points...")
        ndvi_values = self._aggregate_with_ring_cache(
            points_gdf,
            self.ndvi_data,
            final_params["ndvi_radius"],
            final_params["ndvi_stat"],
            final_params["ndvi_percentile"],
            channel="ndvi",
            fold_idx=-1,
            subset="robust_map",
        )

        if progress_callback:
            progress_callback(85, 100)

        # Compute the composite from RAW channels (no per-channel scaling),
        # then apply the ``whole_grid_scaling`` toggle: min-max normalise to
        # [0, 1] over the whole grid when on, raw otherwise — the same
        # treatment the scorer applied, mirrored for standalone single-channel
        # maps.
        logger.info(f"Calculating composite via formula '{formula.name}'...")
        veg_values, terrain_values, ndvi_values = self._normalize_channel_arrays(
            veg_values, terrain_values, ndvi_values
        )
        composite = compute_cgi(
            self.cgi_formula,
            final_params,
            {
                "veg": veg_values,
                "terrain": terrain_values,
                "ndvi": ndvi_values,
            },
        )
        composite = self._finalize_composite(np.asarray(composite, dtype=np.float64))

        # 7. Create raster
        logger.info(f"Saving composite greenery map to {output_path}")
        from rasterio.windows import Window

        rows_arr = np.asarray(points_gdf["row"].values, dtype=np.int64)
        cols_arr = np.asarray(points_gdf["col"].values, dtype=np.int64)
        vals_arr = np.asarray(composite, dtype=np.float32)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # A point/line catchment grid can span a country-wide extent that is
        # mostly empty (entities clustered in a few cities). Scatter-then-write
        # a dense (height × width) array only when it fits a modest budget;
        # above it, write in row bands into a SPARSE_OK GeoTIFF so neither
        # memory nor disk scale with the empty inter-cluster void.
        dense_cell_limit = 120_000_000  # ~0.45 GB as float32
        sparse_grid = height * width > dense_cell_limit
        creation_opts = default_geotiff_creation_options(
            rasterio.float32, sparse=sparse_grid
        )

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=rasterio.float32,
            crs=crs,
            transform=transform,
            nodata=float("nan") if sparse_grid else None,
            **creation_opts,
        ) as dst:
            if not sparse_grid:
                composite_grid = np.full((height, width), np.nan, dtype=np.float32)
                composite_grid[rows_arr, cols_arr] = vals_arr
                dst.write(composite_grid, 1)
            else:
                band_rows = max(1, min(height, 4_000_000 // max(1, width)))
                order = np.argsort(rows_arr, kind="stable")
                rs, cs, vs = rows_arr[order], cols_arr[order], vals_arr[order]
                for band_start in range(0, height, band_rows):
                    band_stop = min(band_start + band_rows, height)
                    lo = int(np.searchsorted(rs, band_start, "left"))
                    hi = int(np.searchsorted(rs, band_stop, "left"))
                    if hi == lo:
                        continue
                    block = np.full(
                        (band_stop - band_start, width), np.nan, dtype=np.float32
                    )
                    block[rs[lo:hi] - band_start, cs[lo:hi]] = vs[lo:hi]
                    dst.write(
                        block,
                        1,
                        window=Window(0, band_start, width, band_stop - band_start),
                    )

        # Composite is the user-visible fusion output — build overviews so
        # the result viewer (Folium) renders the full raster instantly.
        try:
            build_internal_overviews(output_path)
        except Exception:
            pass

        if progress_callback:
            progress_callback(100, 100)

        logger.info(f"✓ Composite greenery map saved: {output_path}")

        # Save parameters used. The double-underscore book-keeping keys carry
        # the selection diagnostics; everything else is the user-facing final
        # params the composite was built from.
        params_path = output_path.replace(".tif", "_params.json")
        import json

        clean_params = {k: v for k, v in final_params.items() if not k.startswith("__")}
        provenance = {
            "selection_method": "bootstrap_stability_selection",
            # Grid geometry, so a viewer can size its read before opening the
            # file. A catchment grid over distant clusters is mostly nodata.
            "raster_width": int(width),
            "raster_height": int(height),
            "raster_valid_cells": int(len(vals_arr)),
            "cell_q_worst": final_params.get("__cell_q_worst__"),
            "cell_median": final_params.get("__cell_median__"),
            "cell_count": final_params.get("__cell_count__"),
            "cell_selection_probability": final_params.get(
                "__cell_selection_probability__"
            ),
            "worst_quantile": final_params.get("__worst_quantile__"),
            "n_bootstraps": final_params.get("__n_bootstraps__"),
            "n_trials_per_bootstrap": final_params.get("__n_trials_per_bootstrap__"),
            "n_total_trials": final_params.get("__n_total_trials__"),
        }
        with open(params_path, "w") as f:
            json.dump(
                {"final_parameters": clean_params, **provenance},
                f,
                indent=2,
                default=str,
            )

        logger.info(f"✓ Parameters saved: {params_path}")

        return output_path
