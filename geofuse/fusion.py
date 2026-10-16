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
    exposure,
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
    buffer_gdf_union_metres,
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


# Objective metrics whose held-out ranking is not the sweep's correlation
# t, so the sweep re-scores its shortlist with them. Everything else is a
# monotone function of the same association and needs no second pass.
_SWEEP_OBJECTIVES: frozenset[str] = frozenset({"quartile_contrast"})


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
       - Aggregation per channel: <channel>_stat, <channel>_percentile
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
        spatial_adjust_method: str = "none",
        spatial_adjust_max_df: int = 10,
        spatial_adjust_eps_m: float | None = None,
        residualize_method: str = "linear",
        search_scoring_method: str = "mom_em3",
        negative_control_columns: list[str] | None = None,
        gvi_ladder_spacing: str = "linear",
        ndvi_ladder_spacing: str = "linear",
        gvi_ladder_rungs: int | None = None,
        ndvi_ladder_rungs: int | None = None,
        exposure_kernel: str = "none",
        exposure_radius_m: float = 100.0,
        exposure_plateau_m: float = 0.0,
        exposure_sigma_m: float = 35.0,
        gvi_min_coverage: float = 0.95,
        drop_collinear_channels: bool = False,
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
            negative_control_columns: Attribute columns on the (vector)
                target holding negative-control outcomes. Carried onto the
                frame for :meth:`compute_negative_controls` only — never
                tuned on, never a reason to drop an entity. Cross-sectional
                targets only.
            gvi_ladder_spacing / ndvi_ladder_spacing: ``"linear"`` (steps of
                ``*_buffer_step_m``) or ``"log"`` (``*_ladder_rungs`` rungs
                spaced geometrically between min and max); see
                :func:`metric_sampling.ladder_radii`.
            exposure_kernel / exposure_radius_m / exposure_plateau_m /
                exposure_sigma_m: which CGI cells an entity is exposed to and
                with what weight; see :func:`exposure.cell_weights`.
            gvi_min_coverage: share of entities that must keep a covered cell
                at a street-view rung for the search to offer it; see
                :meth:`_apply_coverage_gate`.
            drop_collinear_channels: leave the channels the collinearity check
                puts above its VIF threshold out of the index search (weight 0)
                instead of only reporting them.
        """
        # Route Optuna's chatter ("Trial X finished with value Y …") into
        # the per-job log instead of the Streamlit host terminal. Fusion
        # auto-downloads NDVI for the buffered extent, so attach ``ee`` too —
        # those records would otherwise propagate to root from this process.
        import logging as _logging

        attach_external_logger("ee", _logging.INFO)
        # The stdlib loggers of this module and of ``bayesian_index`` carry
        # engine diagnostics — bounds checks, metric load notes, cache and
        # rank warnings. Attaching them routes those into the per-job log
        # alongside the engine's ``_log`` output, not the host terminal.
        attach_external_logger("geofuse.fusion", _logging.INFO)
        attach_external_logger("geofuse.bayesian_index", _logging.INFO)

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
        for spacing in (gvi_ladder_spacing, ndvi_ladder_spacing):
            if spacing not in metric_sampling.LADDER_SPACINGS:
                raise ValueError(
                    f"ladder spacing must be one of {metric_sampling.LADDER_SPACINGS}."
                )
        self.gvi_ladder_spacing = gvi_ladder_spacing
        self.ndvi_ladder_spacing = ndvi_ladder_spacing
        self.gvi_ladder_rungs = int(gvi_ladder_rungs) if gvi_ladder_rungs else 15
        self.ndvi_ladder_rungs = int(ndvi_ladder_rungs) if ndvi_ladder_rungs else 15

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
        # before each discovery; ``evaluate_on_test`` reads it so the
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

        # Negative-control outcomes: attribute columns carried onto the frame
        # for the transfer test only. They never enter tuning, and a missing
        # value never drops an entity from it.
        self.negative_control_columns: list[str] = list(
            dict.fromkeys(c for c in (negative_control_columns or []) if c)
        )

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
        # Which CGI cells an entity is exposed to: the cells its geometry
        # touches, widened by an optional buffer (``geofuse.exposure``).
        exposure.check_kernel(
            exposure_kernel, exposure_radius_m, exposure_plateau_m, exposure_sigma_m
        )
        self.exposure_kernel = str(exposure_kernel)
        self.exposure_radius_m = float(exposure_radius_m)
        self.exposure_plateau_m = float(exposure_plateau_m)
        self.exposure_sigma_m = float(exposure_sigma_m)
        if not 0.0 < float(gvi_min_coverage) <= 1.0:
            raise ValueError(
                f"gvi_min_coverage must lie in (0, 1]; got {gvi_min_coverage}."
            )
        self.gvi_min_coverage = float(gvi_min_coverage)
        self.drop_collinear_channels = bool(drop_collinear_channels)
        # Set by the coverage gate after pre-aggregation: the smallest offered
        # street-view rung, the exposure rows kept, and what was measured.
        self._gvi_floor_m: int | None = None
        self._coverage_keep_index: pd.Index | None = None
        self._coverage_report: dict | None = None
        self._entity_coverage_share: dict | None = None
        self.whole_grid_scaling: bool = bool(whole_grid_scaling)
        # When True, the area-balanced train-val split tries to keep the total
        # area balanced across train / val / test sets.
        self.area_balanced_split: bool = bool(area_balanced_split)
        # Per-polygon-id area (km²) populated by ``_prepare_vector_fusion``
        # so the area-balanced split has stable polygon → area lookups even
        # after the NaN-outcome filter renumbers polygons.
        self._polygon_id_to_area_km2: dict[int, float] | None = None
        # CGI grid metadata for vector targets (the composite raster reuses
        # these so it matches the scored CGI field).
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
        self.train_val_data = None  # Train+val pool the sweep splits
        self.test_data = None  # Held-out test set
        # Distinct configurations scored on the test slice. A held-out set is
        # a budget, not a metric you can query freely.
        self._test_reads = 0
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

        # Trial threads for the search. Only the longitudinal
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
        # Unique-pixel source for the pre-aggregation cache. Vector targets
        # repeat a cell once per entity exposed to it for scoring, but
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

        # Channels above the VIF threshold in the optional collinearity check
        # (``check_channel_collinearity``).
        self._high_vif_channels: set[str] = set()
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

            # Buffered bounds in metres whatever the raster's CRS, then WGS84 as
            # for vector targets.
            extent = gpd.GeoDataFrame(
                {"geometry": [box(*self.target_raster["bounds"])]},
                crs=self.target_raster["crs"],
            )
            self.buffered_extent = buffer_gdf_union_metres(
                extent, float(self.buffer_meters)
            ).to_crs("EPSG:4326")

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
            ndvi_end_date: Last day of the NDVI composite (YYYY-MM-DD), included
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
            end_date: Last day of the temporal composite (YYYY-MM-DD), included
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
            end_inclusive=True,
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

    def _ladder(self, channel: str) -> tuple[int, ...]:
        """One channel family's buffer ladder: ``"gvi"`` or ``"ndvi"``."""
        return metric_sampling.ladder_radii(
            getattr(self, f"{channel}_buffer_min_m"),
            getattr(self, f"{channel}_buffer_max_m"),
            getattr(self, f"{channel}_buffer_step_m"),
            spacing=getattr(self, f"{channel}_ladder_spacing"),
            rungs=getattr(self, f"{channel}_ladder_rungs"),
        )

    def _ladder_description(self, channel: str) -> str:
        """How a channel family's ladder was spaced, for the results bundle."""
        if getattr(self, f"{channel}_ladder_spacing") == "log":
            return f"log ({len(self._ladder(channel))} rungs)"
        return f"linear (step {getattr(self, f'{channel}_buffer_step_m'):g} m)"

    def _outer_radii_metres(self, *, gvi: bool) -> np.ndarray:
        return np.asarray(self._ladder("gvi" if gvi else "ndvi"), dtype=np.int64)

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
            # Vector targets cache stats per unique cell: resolve each
            # exposure row to its cell id. Other targets key the
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

        # A channel with no source layer is served only from the cache, so a
        # miss cannot be recovered by aggregating — there is nothing to
        # aggregate. Falling through would silently reduce the wrong metric.
        if metric_data is None:
            raise RuntimeError(
                f"Channel {channel!r} is stored in the pre-aggregation cache "
                f"and has no source layer, but radius {radius_m} with "
                f"stat={stat!r} percentile={percentile!r} is not in the cache."
            )

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
        self._gvi_floor_m = None
        self._coverage_keep_index = None
        self._coverage_report = None
        self._entity_coverage_share = None
        if self._preaggr_mem is not None:
            self._preaggr_mem.close()
            self._preaggr_mem = None
        if self.is_points:
            _log("INFO", "Target type: VECTOR (per-cell CGI)")
            df = self._prepare_vector_fusion()
        else:
            _log("INFO", "Target type: RASTER")
            df = self._prepare_raster_fusion()
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

    # ────────────────────────────────────────────────────────────
    # ── Channel vocabulary ──────────────────────────────────────
    # A job's channels are whatever its formula consumes: ``(ndvi, gvi)`` for a
    # two-channel study, ``(ndvi, veg, terrain)`` for a three-channel one. The
    # composite path reads that set rather than assuming the three-channel
    # names, so both configurations reach the scoring, apply and map stages.

    def _composite_channels(self) -> tuple[str, ...]:
        """Channels this job's composite is built from, in formula order."""
        mode = self._active_greenery_channel
        if mode != "cgi":
            return (mode,)
        return tuple(cgi_formulas.formula_channels(self.cgi_formula))

    def _channel_active(self, params: dict) -> dict[str, bool]:
        """Which of this job's channels contribute at ``params``."""
        mode = self._active_greenery_channel
        if mode != "cgi":
            return {mode: True}
        return cgi_formulas.get_formula(self.cgi_formula).channel_active(params)

    def _channel_spec(self, params: dict, channel: str) -> tuple[float, str, int]:
        """``(radius, stat, percentile)`` for one channel out of a params dict.

        Every channel carries its own statistic. Street-view channels of params
        recorded with one shared ``streetview_stat`` / ``streetview_percentile``
        fall back to it.
        """
        if channel == "ndvi":
            return (
                params.get("ndvi_radius", int(round(self.ndvi_buffer_max_m))),
                params.get("ndvi_stat", "mean"),
                params.get("ndvi_percentile", 50),
            )
        return (
            params.get(f"{channel}_radius", int(round(self.gvi_buffer_max_m))),
            params.get(f"{channel}_stat", params.get("streetview_stat", "mean")),
            params.get(
                f"{channel}_percentile", params.get("streetview_percentile", 50)
            ),
        )

    def _channel_source(self, channel: str):
        """The metric layer a channel aggregates from, or ``None`` if it has none.

        ``gvi`` is a stored cache channel rather than a layer: its percentiles
        are not recoverable from the components, so it is written during
        pre-aggregation and only ever read back.
        """
        return {
            "veg": self.veg_data,
            "terrain": self.terrain_data,
            "ndvi": self.ndvi_data,
        }.get(channel)

    def _aggregate_channels(
        self,
        points_gdf,
        params: dict,
        *,
        subset: str,
        fold_idx: int = -1,
        channels: tuple[str, ...] | None = None,
    ) -> dict[str, np.ndarray]:
        """``{channel: aggregated values}`` at ``params`` for the rows given.

        A channel the formula gives no weight to is returned as zeros rather
        than aggregated, which is what makes a sparse weight vector cheap.
        """
        chans = self._composite_channels() if channels is None else channels
        active = self._channel_active(params)
        out: dict[str, np.ndarray] = {}
        for ch in chans:
            if not active.get(ch, False):
                out[ch] = np.zeros(len(points_gdf), dtype=np.float32)
                continue
            radius, stat, pct = self._channel_spec(params, ch)
            out[ch] = self._aggregate_with_ring_cache(
                points_gdf,
                self._channel_source(ch),
                radius,
                stat,
                pct,
                channel=ch,
                fold_idx=fold_idx,
                subset=subset,
            )
        return out

    def _composite_from(
        self, params: dict, blocks: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Build the composite from the channel blocks via the active mode.

        CGI mode delegates to the formula; standalone mode takes its single
        channel directly, so the reported score is on the scale the search
        optimized against.
        """
        mode = self._active_greenery_channel
        if mode == "cgi":
            return compute_cgi(self.cgi_formula, params, blocks)
        return blocks[mode]

    # ────────────────────────────────────────────────────────────
    # ── Pre-aggregation grid ────────────────────────────────────
    # Trial percentiles are snapped to this grid so every trial maps onto a
    # stored column of the per-(entity, radius) cache.
    _PREAGGR_PERCENTILES = preaggregation.PERCENTILES

    def _preaggr_radii(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """(gvi_radii, ndvi_radii): the integer radii of the two buffer ladders."""
        return self._ladder("gvi"), self._ladder("ndvi")

    def _search_radii(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The ladders the index searches: the street-view one from its floor."""
        gvi_radii, ndvi_radii = self._preaggr_radii()
        floor = getattr(self, "_gvi_floor_m", None)
        if floor is not None:
            gvi_radii = tuple(r for r in gvi_radii if r >= floor)
        return gvi_radii, ndvi_radii

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

        Reads the study's own channels per pixel from ``data`` (typically the
        ``fusion_df`` produced by :meth:`prepare_fusion_data` — every CGI grid
        pixel before any train/val/test split; ``gvi`` is the sum of ``veg``
        and ``terrain``), optionally subsamples to ``sample_size`` rows for
        speed, then delegates the math to
        :func:`statistical_testing.iterative_vif_reduction`.

        When ``data`` is not supplied, falls back to the post-split pool
        (``train_val_data`` + ``test_data``) — but the typical runner
        path calls this **before** ``split_data`` runs, so passing
        ``fusion_df`` explicitly is the standard usage.

        Channels the procedure would drop are reported in
        ``self._high_vif_channels``; they stay in the index search unless
        ``drop_collinear_channels`` is on (:meth:`_index_layout`). The full
        report (pairwise Pearson, initial / final VIFs, the per-iteration drop
        history) is stored on ``self._collinearity_report`` so the runner can
        hand it to the UI. A study with one channel has nothing to check.

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

        all_channels = list(self._composite_channels())
        sources = {"gvi": ["veg", "terrain"]}
        missing = [
            c
            for ch in all_channels
            for c in sources.get(ch, [ch])
            if c not in df.columns
        ]
        if missing:
            raise ValueError(
                f"Channel columns missing from CGI pool: {missing}. "
                "Re-run prepare_fusion_data() with all channels loaded."
            )
        self._high_vif_channels = set()
        if len(all_channels) < 2:
            self._collinearity_report = None
            return None

        X = np.column_stack(
            [
                df[sources.get(ch, [ch])].to_numpy(dtype=np.float64).sum(axis=1)
                for ch in all_channels
            ]
        )
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
        self._high_vif_channels = {c for c in all_channels if c not in kept}
        self._collinearity_report = dict(report)
        self._collinearity_report["vif_threshold"] = float(vif_threshold)
        self._collinearity_report["sample_size"] = int(len(X))
        self._collinearity_report["channels_in"] = list(all_channels)

        if dropped:
            logger.warning(
                f"Collinearity check: {sorted(dropped)} above the VIF threshold "
                f"(kept in the search). Initial VIFs: "
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
            # A 0/1 column (a dummy, a binary outcome) has no inter-percentile
            # spread whenever one level is under 10 %, so the test cannot apply.
            if vals.size < 2 or binary_longitudinal.is_binary(vals):
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

        On completion the coverage gate settles the study's cell set
        (:meth:`_apply_coverage_gate`). Returns ``True`` on completion,
        ``False`` if cancelled.
        """
        done = self._build_aggregations(
            progress_callback, cancel_callback, max_workers
        )
        if done:
            self._apply_coverage_gate()
        return done

    def _build_aggregations(
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
        # An earlier call already built the complete cache; nothing left to build.
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
            _log(
                "INFO", f"Cache extension: computing {list(absent_ch)} for all pixels."
            )
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
                preaggregation.aggregate_entity_batch(state, point_xy=xy, geoms=geoms)
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

    def _apply_coverage_gate(self) -> None:
        """Settle the one cell set every candidate of the study is scored on.

        The street-view ladder is searched from the first rung at which at
        least ``gvi_min_coverage`` of the entities keep a cell with street-view
        data (the largest rung when none does). A cell is kept when each index
        channel has data at its smallest searched rung; a larger disc holds the
        smaller one, so a kept cell is covered at every searched rung.
        """
        tg = self.target_gdf
        source = self._ensure_memory_loaded() or self._preaggr_cache
        if source is None or tg is None or "_preaggr_id" not in tg.columns:
            return
        plan = self._preaggr_plan(
            source, tg["_preaggr_id"].to_numpy(), self._wave_index(tg)
        )
        if plan is None:
            return
        column = preaggregation.stat_to_column("mean", None)
        gvi_radii, ndvi_radii = self._preaggr_radii()

        def covered(channel, radius):
            v = source.gather(plan, channel, int(radius), column)
            return np.ones(len(tg), dtype=bool) if v is None else np.isfinite(v)

        index_channels = (
            set(cgi_formulas.formula_channels(self.cgi_formula))
            if self._active_greenery_channel == "cgi"
            else {self._active_greenery_channel}
        )
        keep = np.ones(len(tg), dtype=bool)
        if "ndvi" in index_channels:
            keep &= covered("ndvi", ndvi_radii[0])
        if "polygon_id" in tg.columns:
            codes, uniq = pd.factorize(tg["polygon_id"], sort=True)
        else:
            codes, uniq = np.arange(len(tg)), tg.index.to_numpy()
        n_ent = len(uniq)

        street = [
            c for c in self._cache_channels() if c in index_channels and c != "ndvi"
        ]
        shares: dict[int, float] = {}
        if street:
            floor = None
            for r in gvi_radii:
                rung = keep & covered(street[0], r)
                hit = np.bincount(codes[rung], minlength=n_ent) > 0
                share = float(hit.mean())
                shares[int(r)] = share
                if share >= self.gvi_min_coverage:
                    floor = int(r)
                    break
            if floor is None:
                floor = int(gvi_radii[-1])
                _log(
                    "WARN",
                    f"No street-view rung reaches {self.gvi_min_coverage:.0%} entity "
                    f"coverage; searching {floor} m only "
                    f"({shares[floor]:.1%} covered).",
                )
            keep = rung
            self._gvi_floor_m = floor

        w = (
            tg["_weight"].to_numpy(dtype=np.float64)
            if "_weight" in tg.columns
            else np.ones(len(tg))
        )
        total = np.bincount(codes, weights=w, minlength=n_ent)
        kept = np.bincount(codes[keep], weights=w[keep], minlength=n_ent)
        self._entity_coverage_share = dict(
            zip(uniq.tolist(), (kept / np.where(total > 0, total, 1.0)).tolist())
        )
        self._coverage_keep_index = tg.index[keep]
        self._coverage_report = {
            "gvi_floor_m": self._gvi_floor_m,
            "smallest_offered_m": int(gvi_radii[0]) if street else None,
            "min_coverage": self.gvi_min_coverage,
            "gvi_rung_entity_share": shares,
            "rows_dropped": int((~keep).sum()),
            "entities_dropped": int(n_ent - np.unique(codes[keep]).size),
        }
        if street and self._gvi_floor_m != int(gvi_radii[0]):
            _log(
                "WARN",
                f"Street-view coverage: rungs below {self._gvi_floor_m} m cover fewer "
                f"than {self.gvi_min_coverage:.0%} of entities and are not searched.",
            )
        _log(
            "INFO",
            f"Coverage gate: {self._coverage_report['rows_dropped']:,} of {len(tg):,} "
            f"exposure rows and {self._coverage_report['entities_dropped']:,} of "
            f"{n_ent:,} entities lack data at the smallest searched rung.",
        )

    def _wave_index(self, frame: pd.DataFrame) -> np.ndarray | None:
        """Per-row wave index of ``frame`` (longitudinal only), else ``None``."""
        if not self.is_longitudinal or "wave" not in frame.columns:
            return None
        spec = self.longitudinal_spec
        assert spec is not None
        wave_index_of = {w: i for i, w in enumerate(spec.wave_labels)}
        return np.asarray(
            [wave_index_of[str(w)] for w in frame["wave"]], dtype=np.int64
        )

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

        # Global pixel ids referenced by each wave (via the exposure rows for a
        # gridded target; the entity's own id otherwise).
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
        null_runs: int = 16,
        null_reselect_form: bool = False,
        draws: int = 800,
        warmup: int = 800,
        chains: int = 4,
        radius_kernel: str = "lognormal",
        aggregator: str = "dirichlet",
        radius_prior: str = "rungs",
        seed: int = 42,
        cancel_callback: Callable[..., bool] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        outcome: str | None = None,
    ) -> dict:
        """Select the CGI configuration from data, then quantify it.

        Stage one sweeps ``(radius, statistic)``
        per channel and the functional form, scoring every candidate on held-out
        rows of the train+val pool. Stage two fits a posterior over the channel
        weights at the winning columns. Neither stage touches the test split, so
        the headline effect stays out of sample.

        Returns a params dict the composite/apply path consumes, plus ``__``-
        prefixed diagnostics for the results bundle.

        ``null_runs`` below ``bayesian_index.NULL_RUNS_FOR_CLAIM`` is a quick
        check; use ``bayesian_index.NULL_RUNS_PUBLICATION`` for a final run.
        ``null_reselect_form`` re-chooses the form on every permuted outcome,
        so the null rate covers the form choice too. ``outcome`` names a
        negative-control column to tune on instead of the target, which is
        what :meth:`retune_concordance` uses; entities missing it are dropped.
        ``radius_prior`` (``bayesian_index.RADIUS_PRIORS``) ties the radius
        kernel's prior to the listed rungs or to the range they span.
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
        if outcome is None:
            y = np.asarray(static["target"], dtype=np.float64)
        elif outcome in (static.get("controls") or {}):
            y = np.asarray(static["controls"][outcome], dtype=np.float64)
        else:
            raise ValueError(f"{outcome!r} is not a negative-control column.")

        index_channels, channel_index, radius_idx = self._index_layout(
            tensor_channels, radii
        )

        # The channels in play are settled before residualising, so a coverage
        # gap in a channel this study never reads cannot drop rows from it.
        yr, Xr, prep_info = bayesian_index.prep(
            X,
            y,
            static.get("cov"),
            channel_index=channel_index,
            return_info=True,
        )
        dropped = int(len(X) - len(Xr))
        if dropped:
            _log(
                "WARN",
                f"{dropped} of {len(X)} entities have no greenery coverage at "
                f"some searched radius and were dropped from the index search "
                f"({100.0 * dropped / max(len(X), 1):.1f}%). Every candidate is "
                f"scored on the remaining {len(Xr)}.",
            )
        groups = self._inner_split_groups(static)
        if groups is not None:
            groups = groups[prep_info["kept"]]
        # On a panel the posterior carries a per-entity intercept and its null
        # permutes whole entity histories, not rows.
        entities = waves = None
        eid = static.get("entity_id")
        if getattr(self, "is_longitudinal", False) and eid is not None:
            entities = np.asarray(eid)[prep_info["kept"]]
            if static.get("wave") is not None:
                waves = np.asarray(static["wave"])[prep_info["kept"]]
        # The synergy form's interactions are formed per cell from the raw
        # channel values, as the composite is.
        design = None
        if bayesian_index.eligible_forms(forms, len(index_channels)) != ("linear",):
            design = self._synergy_design(
                static, prep_info["kept"], channel_index, index_channels, radii, stats
            )
        tick()

        workers = parallel.process_worker_count()
        # Every stage below takes minutes and runs a process pool, so the flag
        # travels into them rather than being read only between them.
        res = bayesian_index.sweep(
            Xr,
            radii,
            stats,
            yr,
            channels=index_channels,
            channel_index=channel_index,
            radius_idx=radius_idx,
            forms=forms,
            splits=sweep_splits,
            seed=seed,
            workers=workers,
            objective=self._sweep_objective(metric),
            cancel_check=cancelled,
            groups=groups,
            design=design,
        )
        tick()
        _log(
            "INFO",
            f"Sweep: {res.picked} form={res.form} held-out |t|={res.score:.3f}"
            + (f"  BOUNDARY at {list(res.boundary_hit)}" if res.boundary_hit else ""),
        )

        disc = (
            bayesian_index.repeated_discovery(
                Xr,
                radii,
                stats,
                yr,
                channels=index_channels,
                channel_index=channel_index,
                radius_idx=radius_idx,
                forms=forms,
                reps=reps,
                shuffles=shuffles,
                seed=seed,
                workers=workers,
                cancel_check=cancelled,
                groups=groups,
                design=design,
            )
            if reps and shuffles
            else {}
        )
        tick()

        gain = (
            bayesian_index.holdout_gain(
                Xr,
                yr,
                channels=index_channels,
                channel_index=channel_index,
                radius_idx=radius_idx,
                forms=forms,
                splits=gain_splits,
                seed=seed,
                workers=workers,
                cancel_check=cancelled,
                groups=groups,
                design=design,
            )
            if gain_splits and len(index_channels) > 1
            else {}
        )
        tick()

        # The posterior gets the whole grid, not the sweep's pick: the radius
        # profile and the aggregator blend are sampled with the weights, so the
        # interval on the effect carries their uncertainty instead of being
        # conditioned on a choice made before the sampler ran. The sweep's pick
        # stays in the bundle as a diagnostic and as the shortlist that decided
        # the functional form.
        if cancelled():
            raise JobCancelled("Index posterior cancelled by user.")
        post, grid, grid_kwargs = bayesian_index.grid_posterior(
            Xr,
            yr,
            channels=index_channels,
            channel_index=channel_index,
            radii=radii,
            stats=stats,
            radius_idx=radius_idx,
            form=res.form,
            picked=res.picked,
            radius_kernel=radius_kernel,
            aggregator=aggregator,
            draws=draws,
            warmup=warmup,
            chains=chains,
            seed=seed,
            column_sd=prep_info["column_sd"],
            radius_prior=radius_prior,
            entities=entities,
            design=design,
        )
        null = (
            bayesian_index.null_calibration(
                grid,
                yr,
                form=res.form,
                n=null_runs,
                workers=workers,
                cancel_check=cancelled,
                reselect_form=null_reselect_form,
                form_E=bayesian_index.form_columns(
                    bayesian_index.eligible_forms(
                        forms, len(index_channels), design is not None
                    ),
                    Xr.reshape(len(Xr), -1),
                    res.columns,
                    design,
                ),
                groups=groups,
                waves=waves,
                **grid_kwargs,
            )
            if null_runs
            else {}
        )
        tick()

        # The form is a swept axis, so the formula the composite is built from
        # is only known once the sweep has run. Everything downstream reads
        # ``self.cgi_formula``, including the params dict built next.
        if self._active_greenery_channel == "cgi":
            self.cgi_formula = cgi_formulas.formula_for(
                tuple(cgi_formulas.formula_channels(self.cgi_formula)), res.form
            )

        # The composite path rebuilds the index from raw channel values at the
        # projected cells, so each weight is carried onto the raw scale of its
        # term.
        term_sd = None
        if self._active_greenery_channel == "cgi":
            rungs = [int(round(float(r))) for r in radii]
            cols = [
                (ci * len(rungs) + rungs.index(int(r))) * len(stats)
                + list(stats).index(st)
                for ci, (r, st) in zip(channel_index, post.picked)
            ]
            term_sd = (
                design.columns(cols)[1]
                if res.form == "synergy"
                else prep_info["column_sd"].reshape(-1)[cols]
            )
        params = self._params_from_sweep(res, post, index_channels, term_sd=term_sd)
        params["__selection_method__"] = "bayesian_index"
        params["__sweep__"] = {
            "picked": [list(p) for p in res.picked],
            "form": res.form,
            "score": res.score,
            "selection_score": res.selection_score,
            "form_scores": res.form_scores,
            "one_se_picked": [
                list(p)
                for p in bayesian_index._decode(
                    res.one_se_columns, len(radii), len(stats), radii, stats
                )
            ],
            "boundary_hit": list(res.boundary_hit),
            "n_candidates": len(res.combos),
            "distinct_split_winners": len(res.winner_counts),
            "splits": sweep_splits,
        }
        params["__posterior__"] = post.summary()
        params["__posterior__"]["ladder_spacing"] = {
            ch: self._ladder_description("ndvi" if ch == "ndvi" else "gvi")
            for ch in index_channels
        }
        params["__discovery__"] = disc
        params["__holdout_gain__"] = gain
        params["__null_calibration__"] = null
        params["__coverage__"] = getattr(self, "_coverage_report", None)
        params["__elapsed_s__"] = float(time.perf_counter() - t0)
        _log(
            "OK",
            f"Bayesian index fitted in {params['__elapsed_s__']:.0f}s "
            f"(R-hat {post.rhat_max:.3f}, ESS {post.ess_min:.0f}, "
            f"{post.divergences} divergences). Posterior grid: {post.picked}.",
        )
        return params

    def _inner_split_groups(self, static: dict) -> "np.ndarray | None":
        """One label per tensor row for the index search's inner splits.

        The spatial block when the held-out test is spatially blocked, else the
        entity on a panel, so no inner split puts one block or one entity on
        both sides. ``None`` (row-level splits) otherwise.
        """
        blocks = getattr(self, "_spatial_block_by_group", None)
        col = getattr(self, "_spatial_block_group_col", None)
        if blocks is not None and col is not None:
            keys = static.get("entity_id") if col == "entity_id" else static.get("pid")
            if keys is not None:
                return pd.Series(np.asarray(keys)).map(blocks).fillna(-1).to_numpy()
        eid = static.get("entity_id")
        if getattr(self, "is_longitudinal", False) and eid is not None:
            return np.asarray(eid)
        return None

    def _index_layout(self, tensor_channels, radii):
        """The study's index channels, their tensor positions and their ladders.

        A channel is only offered the radii its own ladder was computed at, so
        NDVI and the street-view family can search different rungs. With
        ``drop_collinear_channels`` the collinearity check's high-VIF channels
        are left out, so the composite gives them weight 0.
        """
        index_channels = list(cgi_formulas.formula_channels(self.cgi_formula))
        if self._active_greenery_channel != "cgi":
            index_channels = [self._active_greenery_channel]
        elif getattr(self, "drop_collinear_channels", False):
            high = getattr(self, "_high_vif_channels", set())
            index_channels = [c for c in index_channels if c not in high]
        channel_index = [list(tensor_channels).index(c) for c in index_channels]
        gvi_radii, ndvi_radii = self._search_radii()
        radius_idx = [
            [
                i
                for i, r in enumerate(radii)
                if int(r) in (ndvi_radii if c == "ndvi" else gvi_radii)
            ]
            for c in index_channels
        ]
        return index_channels, channel_index, radius_idx

    def plasmode_recovery(self, *, out_dir: str | None = None, **kwargs) -> dict:
        """Plasmode recovery on this study's own exposure tensor and covariates.

        Builds the train+val tensor exactly as :meth:`fit_bayesian_index` does,
        keeps the entities with complete coverage, and hands it to
        :func:`geofuse.validation.plasmode_recovery` with the study's channels
        and ladders; ``kwargs`` pass through (``reps``, ``partial_r2``,
        ``scenarios``, ``pipeline_kwargs``, ``seed``, ...). With ``out_dir`` the
        tidy table and the JSON bundle are written there too.
        """
        from . import validation

        X, radii, stats, tensor_channels, static = self.build_index_tensor("train_val")
        channels, channel_index, radius_idx = self._index_layout(tensor_channels, radii)
        cov = static.get("cov")
        scan = X[:, channel_index].reshape(len(X), -1)
        measured = ~np.isnan(scan).all(0)
        keep = np.isfinite(scan[:, measured]).all(1)
        if cov is not None and np.size(cov):
            keep &= np.isfinite(np.asarray(cov, dtype=np.float64)).all(1)
        result = validation.plasmode_recovery(
            X[keep],
            None if cov is None or not np.size(cov) else np.asarray(cov)[keep],
            channels=channels,
            radii=radii,
            stats=stats,
            channel_index=channel_index,
            radius_idx=radius_idx,
            **kwargs,
        )
        if out_dir:
            validation.write_plasmode_report(result, out_dir)
        return result

    def _sweep_objective(self, metric: str):
        """Held-out scorer the sweep ranks candidates by, or ``None``.

        ``None`` means the sweep's own correlation t, which is what makes the
        exhaustive grid affordable and is the right ranking for the correlation
        family. A metric that asks a different question of the same data gets
        its own scorer, so the setting decides the winner rather than only the
        number reported afterwards.

        The sweep works on Frisch-Waugh residualised columns, so covariates are
        already partialled out and the scorer is called without them.
        """
        from . import objective_scoring as _scoring

        if metric not in _SWEEP_OBJECTIVES:
            return None

        def objective(exposure, target):
            try:
                return float(_scoring.score(metric, target, exposure, None))
            except Exception:
                return 0.0

        return objective

    def _params_from_sweep(self, res, post, index_channels, term_sd=None) -> dict:
        """Posterior -> the params dict the composite/apply path consumes.

        That path carries one radius and one statistic per channel, so the
        posterior's blend over the grid is projected onto its modal cell here.
        The projection is what the composite is built from; the blend and its
        credible intervals stay in the results bundle, and the effect that gets
        reported is the one the blend produced, not a refit at the modal cell.

        The posterior's weights are influence shares on standardised terms.
        With ``term_sd`` (each term's covariate-adjusted SD at the projected
        cells, in weight order) every weight is divided by it, so the composite
        of the raw channel values is the fitted index up to a shift and a
        positive scale.
        """
        formula = cgi_formulas.get_formula(self.cgi_formula)
        params: dict[str, Any] = {}

        stat_of = {"mean": ("mean", 50)}
        for (radius, column), ch in zip(post.picked, index_channels):
            stat, pct = stat_of.get(column, ("percentile", 0))
            if column != "mean":
                pct = int(column[1:])
            params[f"{ch}_radius"] = int(radius)
            params[f"{ch}_stat"] = stat
            params[f"{ch}_percentile"] = int(pct)
        # Defaults only for the channels this formula consumes, so no key reads
        # as a fitted radius for a channel the run never fitted. The apply path
        # defaults any channel it needs that is absent.
        formula_channels = set(cgi_formulas.formula_channels(self.cgi_formula))
        formula_channels.update(index_channels)
        for ch in formula_channels:
            default = self.ndvi_buffer_max_m if ch == "ndvi" else self.gvi_buffer_max_m
            params.setdefault(f"{ch}_radius", int(round(float(default))))
            params.setdefault(f"{ch}_stat", "mean")
            params.setdefault(f"{ch}_percentile", 50)

        # Posterior-mean weights onto the formula's keys as shares of 100, the
        # scale every downstream consumer expects: one per channel, then the
        # synergy form's interactions in the formula's interaction-key order.
        w = np.asarray(post.weights).mean(0)
        if term_sd is not None:
            w = w / np.maximum(np.asarray(term_sd, dtype=np.float64)[: len(w)], 1e-12)
        keys = list(formula.weight_keys)
        main_key = cgi_formulas._CHANNEL_MAIN_KEY[self.cgi_formula]
        vals = np.zeros(len(keys))
        for i, ch in enumerate(index_channels):
            key = main_key.get(ch)
            if key in keys and i < len(w):
                vals[keys.index(key)] = float(w[i])
        # Interactions are matched on their channel set, so a study searching
        # fewer channels than its formula holds lands on the right keys.
        fchans = list(cgi_formulas.formula_channels(self.cgi_formula))
        fcombos = cgi_formulas.synergy_combos(len(fchans))
        n_main = len(index_channels)
        for k, combo in enumerate(cgi_formulas.synergy_combos(n_main)):
            if n_main + k >= len(w):
                break
            at = tuple(sorted(fchans.index(index_channels[i]) for i in combo))
            key = formula.interaction_weight_keys[fcombos.index(at)]
            vals[keys.index(key)] = float(w[n_main + k])
        total = vals.sum()
        if total <= 0:
            vals[:] = 1.0 / len(vals)
            total = 1.0
        for key, v in zip(keys, vals * 100.0 / total):
            params[key] = round(float(v), 4)
        return params

    def build_index_tensor(self, subset: str = "train_val"):
        """Per-entity values for every (channel, radius, statistic) cell.

        The sweep scores candidates out of this one tensor, so the cache is read
        once per run instead of once per candidate. Shape is
        ``(n_entities, n_channels, n_radii, n_stats)`` with channels in
        :meth:`_cache_channels` order.

        Every entity's value is the exposure-weighted mean over its cells,
        which do not depend on the radius.
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
        gvi_radii, ndvi_radii = self._search_radii()
        # One ladder for the tensor: NDVI and the street-view family can be
        # configured separately, so the union is stored and each channel's
        # off-ladder cells stay NaN and are never selected.
        radii = tuple(sorted(set(gvi_radii) | set(ndvi_radii)))
        stats = tuple(preaggregation.STAT_COLUMNS)

        static = self._fold_entity_statics(data, f"tensor::{subset}", subset)
        n_ent = static["n_uniq"] if static["has_pid"] else len(static["points"])
        X = np.full((n_ent, len(channels), len(radii), len(stats)), np.nan)

        for ri, radius in enumerate(radii):
            for ci, ch in enumerate(channels):
                ladder = ndvi_radii if ch == "ndvi" else gvi_radii
                if radius not in ladder:
                    continue
                for si, column in enumerate(stats):
                    stat, pct = (
                        ("mean", None)
                        if column == "mean"
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
                            v,
                            static["codes"],
                            static["n_uniq"],
                            None,
                            static.get("weight"),
                        )
                        if static["has_pid"]
                        else v
                    )
        return X, np.asarray(radii, dtype=float), list(stats), list(channels), static

    def _synergy_design(
        self, static, kept, channel_index, index_channels, radii, stats
    ):
        """The synergy form's raw values per CGI cell, over the kept tensor rows.

        Holds each distinct cell of the train+val fold once, for the index
        channels only, with the exposure rows that average it into an entity.
        """
        gvi_radii, ndvi_radii = self._search_radii()
        inverse = static.get("uniq_inverse")
        n_rows = len(static["lookup_ids"])
        if inverse is None:
            inverse, first = np.arange(n_rows), None
        else:
            first = np.unique(inverse, return_index=True)[1]
        n_cells = int(inverse.max()) + 1
        cells = np.full(
            (n_cells, len(index_channels), len(radii), len(stats)),
            np.nan,
            dtype=np.float32,
        )
        _log(
            "INFO",
            f"Synergy terms: {n_cells:,} cells x {len(index_channels)} channels x "
            f"{len(radii)} radii x {len(stats)} stats = {cells.nbytes / 2**20:,.0f} "
            "MB, held once on disk and once in the posterior.",
        )
        for ci, ch in enumerate(index_channels):
            ladder = ndvi_radii if ch == "ndvi" else gvi_radii
            for ri, radius in enumerate(radii):
                if int(radius) not in ladder:
                    continue
                for si, column in enumerate(stats):
                    stat, pct = (
                        ("mean", None)
                        if column == "mean"
                        else ("percentile", int(column[1:]))
                    )
                    vals, on_uniq = self._channel_values_for_static(
                        static, ch, radius, stat, pct, "train_val"
                    )
                    if vals is None:
                        continue
                    v = np.asarray(vals)
                    cells[:, ci, ri, si] = v if on_uniq or first is None else v[first]

        codes = static["codes"] if static["has_pid"] else np.arange(n_rows)
        rows = np.flatnonzero(np.asarray(kept)[codes])
        weight = static.get("weight")
        cov = static.get("cov")
        return bayesian_index.SynergyDesign.build(
            cells,
            None if cov is None else np.asarray(cov)[kept],
            rows=(
                inverse[rows],
                (np.cumsum(kept) - 1)[codes[rows]],
                np.ones(len(rows)) if weight is None else np.asarray(weight)[rows],
            ),
            channels=channel_index,
        )

    def _channel_values_for_static(self, static, channel, radius, stat, pct, subset):
        """One (channel, radius, stat) column for a prepared fold, cache first.

        ``gvi`` has no source layer of its own — it is a stored cache channel —
        so a cache miss for it is fatal rather than falling back to on-the-fly
        aggregation over the wrong metric. It is read on the fold's distinct
        cells when rows share cells, else row by row.
        """
        metric = {
            "veg": self.veg_data,
            "terrain": self.terrain_data,
            "ndvi": self.ndvi_data,
        }.get(channel)
        if metric is None:
            on_uniq = static.get("uniq_lookup_ids") is not None
            ids = static["uniq_lookup_ids" if on_uniq else "lookup_ids"]
            waves = static.get("uniq_wave_idx" if on_uniq else "wave_idx")
            source = self._ensure_memory_loaded() or self._preaggr_cache
            column = preaggregation.stat_to_column(stat, pct)
            if source is None or column is None:
                return None, False
            plan = self._preaggr_plan(source, ids, waves, static)
            if plan is None:
                return None, False
            return source.gather(plan, channel, int(round(radius)), column), on_uniq
        return self._aggregate_channel_for_fold(
            static,
            metric,
            radius,
            stat,
            pct if pct is not None else 50,
            channel=channel,
            fold_idx=None,
            subset=subset,
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
        return tuple(
            c for c in preaggregation.GreeneryCache.ALL_CHANNELS if c in wanted
        )

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

    def _prepare_vector_fusion(self, outcome_col: str | None = None) -> pd.DataFrame:
        """Per-cell dataset for vector targets of any geometry type.

        Lays the CGI grid over the targets in a low-distortion metre CRS and
        exposes every entity to the cells its geometry touches, widened by the
        optional exposure buffer (:func:`exposure.cell_weights`). Returns one
        row per ``(entity, cell)`` exposure, keyed by ``polygon_id`` (the entity
        or observation index) and carrying the exposure ``_weight``; an entity's
        exposure is the weighted mean of its cells' values. A cell shared by
        several entities appears once per entity, so group membership stays
        fixed for :meth:`split_data`, while the pre-aggregation cache holds each
        cell once (``_preaggr_id`` → ``_global_pid``).
        """
        _log("INFO", "====== VECTOR FUSION (per-cell CGI) ======")
        entities = self.target_gdf.copy()
        outcome_col = outcome_col or self.target_feature
        spacing_m = float(self.cgi_grid_spacing_m)

        # Drop NaN-outcome / NaN-covariate entities up front.
        valid_mask = entities[outcome_col].notna()
        for col in self.covariate_columns or []:
            valid_mask &= entities[col].notna()
        skipped_nan = int((~valid_mask).sum())
        if skipped_nan:
            _log(
                "WARN",
                f"Skipped {skipped_nan} entity(ies) with NaN outcome / covariate.",
            )
        # Entity keys stay the target's own row positions, so outputs join back
        # to the target's geometries.
        keep_pos = np.flatnonzero(valid_mask.to_numpy())
        entities = entities.loc[valid_mask].reset_index(drop=True)
        if len(entities) < 2:
            raise ValueError(
                "Fusion needs ≥2 entities with a valid outcome (and covariates "
                f"if configured); got {len(entities)}."
            )

        # Project to a low-distortion CRS so the grid step and buffers are metres.
        grid_crs, _grid_distortion, _grid_name = select_grid_crs_with_warning(
            entities, _log, role="CGI grid CRS"
        )
        ent = entities.to_crs(grid_crs)
        if self.is_polygon_target:
            # Per-entity area for the area-balanced split.
            self._polygon_id_to_area_km2 = dict(
                zip(
                    keep_pos.tolist(),
                    (ent.geometry.area.to_numpy() / 1_000_000.0).tolist(),
                )
            )

        ent_idx, g_row, g_col, weight = exposure.cell_weights(
            ent.geometry.values,
            spacing_m,
            kernel=self.exposure_kernel,
            radius_m=self.exposure_radius_m,
            plateau_m=self.exposure_plateau_m,
            sigma_m=self.exposure_sigma_m,
        )
        if ent_idx.size == 0:
            raise ValueError(
                "No target geometry touches a CGI cell — check the target geometry."
            )
        untouched = len(ent) - int(np.unique(ent_idx).size)
        if untouched:
            _log(
                "WARN",
                f"{untouched} entity(ies) have an empty geometry and no exposure; "
                "skipped.",
            )

        # One pixel per distinct cell. The global id packs the cell's lattice
        # (row, col), anchored at the CRS origin, so the same cell has the same
        # id in any job and the greenery cache is reusable across targets.
        g_pid = (g_row << np.int64(32)) | (g_col & np.int64(0xFFFFFFFF))
        uniq_pid, first, cell_of_row = np.unique(
            g_pid, return_index=True, return_inverse=True
        )
        u_row, u_col = g_row[first], g_col[first]
        col_origin = int(u_col.min())
        row_origin = int(u_row.min())
        n_cols = int(u_col.max()) - col_origin + 1
        n_rows = int(u_row.max()) - row_origin + 1
        minx = col_origin * spacing_m
        maxy = -row_origin * spacing_m
        pixels = gpd.GeoDataFrame(
            {
                "_grid_row": (u_row - row_origin).astype(np.int32),
                "_grid_col": (u_col - col_origin).astype(np.int32),
                "_global_pid": uniq_pid,
            },
            geometry=gpd.points_from_xy(
                (u_col + 0.5) * spacing_m, -(u_row + 0.5) * spacing_m
            ),
            crs=grid_crs,
        )
        _log(
            "INFO",
            f"Entities: {len(ent)} · exposure buffer: {self.exposure_kernel} · CGI "
            f"grid pixel size: {spacing_m:g} m · {len(pixels):,} cells · "
            f"{len(ent_idx):,} entity–cell exposures · grid span {n_rows}×{n_cols}.",
        )

        # ``polygon_id`` = the entity's row in the target; each (entity, wave)
        # row in longitudinal mode is already its own observation.
        entity_gdf = gpd.GeoDataFrame(
            {
                "polygon_id": keep_pos[ent_idx],
                "target": ent[outcome_col].to_numpy()[ent_idx],
                "_grid_row": pixels["_grid_row"].to_numpy()[cell_of_row],
                "_grid_col": pixels["_grid_col"].to_numpy()[cell_of_row],
                "_weight": weight,
                # Unique-cell id this exposure row maps to; the pre-aggregation
                # cache stores one row per cell and resolves it at lookup time.
                "_preaggr_id": cell_of_row.astype(np.int64),
                "_global_pid": uniq_pid[cell_of_row],
            },
            geometry=pixels.geometry.values[cell_of_row],
            crs=grid_crs,
        )
        extra = [
            *(self.covariate_columns or []),
            *self._longitudinal_extra_cols(),
            *self.negative_control_columns,
        ]
        for col in dict.fromkeys(extra):
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
        probe_cols = self._probe_channel_coverage(pixels, cell_of_row, entity_gdf)
        _log(
            "INFO",
            f"Coverage probe: {len(entity_gdf):,} exposure rows in "
            f"{time.perf_counter() - _probe_t0:.0f}s.",
        )

        fusion_df = pd.DataFrame(
            {
                "polygon_id": entity_gdf["polygon_id"].values,
                "target": entity_gdf["target"].values,
                # float32 halves the per-cell frame cost.
                "veg": probe_cols["veg"],
                "terrain": probe_cols["terrain"],
                "ndvi": probe_cols["ndvi"],
                "_weight": weight,
            },
            index=entity_gdf.index,
        )
        for col in self.covariate_columns or []:
            fusion_df[col] = entity_gdf[col].values
        for col in self._longitudinal_extra_cols():
            if col in entity_gdf.columns:
                fusion_df[col] = entity_gdf[col].values
        self._carry_negative_controls(fusion_df, entity_gdf)
        # Coordinates of the entity, not of the cell, so the spatial smooth
        # sees one location per entity.
        fusion_df = self._attach_entity_coords(
            fusion_df,
            gpd.GeoDataFrame(geometry=ent.geometry.values[ent_idx], crs=grid_crs),
        )

        _log("INFO", "====== DATA QUALITY SUMMARY (PER-CELL) ======")
        required = ["target", "veg", "terrain", "ndvi", *(self.covariate_columns or [])]
        for col in required:
            nan_count = int(fusion_df[col].isna().sum())
            pct = nan_count / max(len(fusion_df), 1) * 100
            _log("INFO", f"{col} NaN: {nan_count} ({pct:.1f}%)")

        result = fusion_df.dropna(subset=required)
        entities_left = result["polygon_id"].nunique() if len(result) else 0
        _log(
            "OK" if entities_left else "WARN",
            f"After dropna: {len(result):,} exposure rows "
            f"({entities_left} unique entities)",
        )
        if entities_left < 2:
            raise ValueError(
                "Fusion needs ≥2 entities with valid coverage after NaN filtering; "
                f"got {entities_left}. Check metric coverage or the exposure buffer."
            )

        # Trim target_gdf to the surviving rows so the cache and the split only
        # see entities that passed the coverage probe.
        self.target_gdf = entity_gdf.loc[result.index].copy()

        # Pre-aggregation source: the unique cells still referenced after the
        # coverage dropna, keyed by ``_preaggr_id``.
        referenced = np.unique(self.target_gdf["_preaggr_id"].to_numpy())
        self._preaggr_entity_gdf = pixels.loc[referenced].copy()
        _log(
            "INFO",
            f"Pre-aggregation source: {len(self._preaggr_entity_gdf):,} unique "
            f"cells (vs {len(self.target_gdf):,} exposure rows).",
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
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Collapse per-cell ``values`` to one value per ``polygon_id``.

        ``how="mean"`` for the composite (weighted by ``weights``, the exposure
        weight of each row, when given), ``"first"`` for per-entity constants
        (target / covariates / entity_id / years_since_baseline). When
        ``mask`` is given only the masked rows contribute. The grouped key
        order matches across every collapsed array (pandas sorts the group
        keys).
        """
        v = np.asarray(values)
        pid = np.asarray(polygon_id)
        w = None if weights is None else np.asarray(weights, dtype=np.float64)
        if mask is not None:
            v = v[mask]
            pid = pid[mask]
            w = None if w is None else w[mask]
        if how == "mean":
            # factorize(sort=True) reproduces the sorted-key order pandas
            # groupby uses, so every collapsed array stays row-aligned, while
            # bincount avoids building a Series + GroupBy on the hot path.
            codes, uniq = pd.factorize(pid, sort=True)
            vv = np.asarray(v, dtype=np.float64)
            valid = ~np.isnan(vv)
            n = len(uniq)
            ww = np.ones(int(valid.sum())) if w is None else w[valid]
            counts = np.bincount(codes[valid], weights=ww, minlength=n)
            sums = np.bincount(codes[valid], weights=vv[valid] * ww, minlength=n)
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
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Per-entity mean of ``values`` from precomputed collapse codes.

        Weighted by ``weights`` (each row's exposure weight) when given.

        Equivalent to ``_collapse_to_entities(values, pid, mask, "mean")``
        without the per-call ``factorize``: ``codes`` / ``n_uniq`` come from
        one sorted-key factorization of the fold's ``polygon_id``. The
        ``minlength`` pin keeps the output aligned to the full entity set, so
        the row order matches every other collapsed array.
        """
        v = np.asarray(values, dtype=np.float64)
        cd = codes
        w = None if weights is None else np.asarray(weights, dtype=np.float64)
        if mask is not None:
            v = v[mask]
            cd = cd[mask]
            w = None if w is None else w[mask]
        valid = ~np.isnan(v)
        ww = np.ones(int(valid.sum())) if w is None else w[valid]
        counts = np.bincount(cd[valid], weights=ww, minlength=n_uniq)
        sums = np.bincount(cd[valid], weights=v[valid] * ww, minlength=n_uniq)
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
            st["pid"] = np.asarray(uniq)
            # Exposure weight of each (entity, cell) row for the entity mean.
            st["weight"] = (
                data["_weight"].to_numpy(dtype=np.float64)
                if "_weight" in data.columns
                else None
            )
            st["target"] = np.asarray(data["target"].values, dtype=np.float64)[
                first_idx
            ]
            st["cov"] = (
                data[cov_cols].to_numpy(dtype=np.float64)[first_idx]
                if cov_cols
                else None
            )
            st["controls"] = {
                c: data[c].to_numpy(dtype=np.float64)[first_idx]
                for c in self.negative_control_columns
                if c in data.columns
            }
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
            st["controls"] = {
                c: data[c].to_numpy(dtype=np.float64)
                for c in self.negative_control_columns
                if c in data.columns
            }
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
        wave_idx = self._wave_index(pts)
        st["wave_idx"] = wave_idx
        pts.attrs["_gf_wave_idx"] = wave_idx

        # Dedup axis for the per-trial channel arithmetic. Vector targets
        # repeat a cell once per entity exposed to it, and a cell's
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
        """Apply the ``whole_grid_scaling`` toggle to the written composite map.

        When on, the per-pixel map is min-max normalized to ``[0, 1]`` over the
        whole grid; when off it is written raw. Scores, the entity table and
        every reported estimate use the raw composite, whose units do not
        depend on which pixels a subset happens to hold.
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

    def _probe_channel_coverage(
        self,
        pixels: gpd.GeoDataFrame,
        pix_idx: np.ndarray,
        entity_gdf: gpd.GeoDataFrame,
    ) -> dict[str, np.ndarray]:
        """Per-exposure-row veg / terrain / NDVI values for the coverage gate.

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
                    # the probe result back onto this group's exposure rows.
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
        """Raster targets: every valid pixel is an entity at its pixel centre.

        The target's own grid only locates the entities. Their exposure comes
        from the CGI grid the study's settings define, as for any vector target
        of points (:meth:`_prepare_vector_fusion`).
        """
        logger.info("Preparing raster-based fusion data...")
        target_data = self.target_raster["data"]
        transform = self.target_raster["transform"]
        width = int(target_data.shape[1])

        # Keep only valid-target pixels before building any geometry: one
        # geometry per grid cell would cost tens of GB on a large raster.
        flat_target = np.ma.filled(target_data.astype(np.float64), np.nan).ravel()
        keep = np.flatnonzero(~np.isnan(flat_target))
        if keep.size == 0:
            raise ValueError(
                "No valid pixels found in target raster. "
                "Target raster may be empty or all NaN."
            )
        xs, ys = xy(transform, keep // width, keep % width, offset="center")
        self.target_gdf = gpd.GeoDataFrame(
            {"target": flat_target[keep]},
            geometry=gpd.points_from_xy(np.asarray(xs), np.asarray(ys)),
            crs=self.target_raster["crs"],
        )
        logger.info(
            f"Created {len(self.target_gdf):,} pixel-centre entities from the "
            "target raster"
        )
        return self._prepare_vector_fusion(outcome_col="target")

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

        The block edge is floored at twice the largest channel radius so a
        block is wider than the greenery autocorrelation range, keeping a
        held-out block's buffers clear of the train blocks around it.
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
        the sweep splits.

        The sweep does its own repeated splitting of the
        train+val pool (see :meth:`fit_bayesian_index`), so this
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
        if self._coverage_keep_index is not None:
            fusion_df = fusion_df.loc[fusion_df.index.isin(self._coverage_keep_index)]

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
            # while keeping a block's buffers clear of its neighbours.
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

            # No CV folds — the sweep splits the pool
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

        # No CV folds — the sweep splits the pool
        # itself. Reset the caches the resampler rebuilds per run.
        self.cv_folds = []
        self._spatial_basis_cache = {}
        self._spatial_adjust_summary = None

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
        do not (their robustness comes from the held-out sweep + posterior CIs,
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

        # Every distinct configuration scored on the test slice spends part of
        # the held-out guarantee. Cache hits are free (same answer), so only
        # misses count. Surfaced in the bundle so a burned test set is visible
        # rather than inferred.
        self._test_reads += 1

        logger.info("Evaluating on held-out test set...")

        # Apply dynamic circular buffer aggregation over this job's own channel
        # set. Both points and rasters use the same approach (rasters converted
        # to points).
        # .loc with an index array materialises a new frame; read-only below.
        test_points = self.target_gdf.loc[self.test_data.index]
        test_blocks = self._aggregate_channels(test_points, params, subset="test")

        test_combined = np.column_stack(list(test_blocks.values()))
        test_valid_mask = ~np.isnan(test_combined).any(axis=1)

        if test_valid_mask.sum() == 0:
            raise ValueError("No valid test data after aggregation")

        # Calculate composite via the active mode. Same fork as ``apply_fusion``
        # so the held-out score is on the same scale the study optimized.
        test_composite = self._composite_from(params, test_blocks)

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

        # Per-cell mode: the exposure-weighted mean of each entity's per-cell
        # composites, one row per entity (per (entity, wave) on a panel, the
        # shape the MixedLM scorer expects).
        if test_static["has_pid"]:
            test_composite = self._collapse_mean_from_codes(
                test_composite,
                test_static["codes"],
                test_static["n_uniq"],
                None,
                test_static.get("weight"),
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

        Aggregates **every channel this job uses** at ``params``' radii / stats
        / percentiles on the requested ``subset`` (``"train_val"`` for the
        resampling pool, ``"test"`` for the held-out set, ``"all"`` for every
        entity), collapses to one row per entity (polygon mean when
        polygon-keyed), and returns the raw arrays plus the aligned target,
        covariates, and — in longitudinal mode — entity ids and
        ``years_since_baseline``. ``channel_names`` says which column is which,
        because the set differs between a two- and three-channel study.

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
        # The formula's channels, not the engine's active mode: this design is
        # what the composite is compared *against*, so it has to hold every
        # channel even while the engine is scoring one standalone.
        channels = tuple(cgi_formulas.formula_channels(self.cgi_formula))
        # Every channel is aggregated here regardless of its weight: the AIC/BIC
        # comparison fits the standalones as well as the composite, so a channel
        # the formula happens to zero is still one of the models under test.
        blocks: dict[str, np.ndarray] = {}
        for ch in channels:
            radius, stat, pct = self._channel_spec(params, ch)
            blocks[ch] = self._aggregate_with_ring_cache(
                points,
                self._channel_source(ch),
                radius,
                stat,
                pct,
                channel=ch,
                fold_idx=None,
                subset=None,
            )

        cov_cols = self.covariate_columns
        cov = data[cov_cols].to_numpy(dtype=np.float64) if cov_cols else None

        entity_id = None
        ysb = None
        if "polygon_id" in data.columns:
            pid = data["polygon_id"].values
            row_w = data["_weight"].values if "_weight" in data.columns else None
            blocks = {
                ch: self._collapse_to_entities(a, pid, None, "mean", row_w)
                for ch, a in blocks.items()
            }
            target = self._collapse_to_entities(
                data["target"].values, pid, None, "first"
            )
            if cov is not None:
                c = cov
                cov = np.column_stack(
                    [
                        self._collapse_to_entities(c[:, j], pid, None, "first")
                        for j in range(c.shape[1])
                    ]
                )
            if self.is_longitudinal:
                entity_id = self._collapse_to_entities(
                    data["entity_id"].values, pid, None, "first"
                )
                ysb = self._collapse_to_entities(
                    data["years_since_baseline"].values, pid, None, "first"
                )
        else:
            target = data["target"].values
            if self.is_longitudinal:
                entity_id = data["entity_id"].values
                ysb = data["years_since_baseline"].values

        return {
            "channels": np.column_stack(list(blocks.values())),
            "channel_names": list(blocks),
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

        # Combine all data (train+val+test) — cached union frame.
        all_data = self._full_data_frame()
        if all_data is None:
            raise ValueError("No data available. Run split_data() first.")
        # .loc with an index array materialises a new frame; read-only below.
        all_points = self.target_gdf.loc[all_data.index]

        # Apply circular buffer aggregation with optimized parameters, over the
        # job's own channel set — the same fork evaluate_on_test uses, so the
        # applied composite and the held-out score agree.
        all_blocks = self._aggregate_channels(all_points, weights, subset="all")

        composite = self._composite_from(weights, all_blocks)

        result_df = all_data.copy()
        for ch, values in all_blocks.items():
            result_df[ch] = values
        result_df["composite"] = composite

        # Per-cell mode: one row per entity. The composite is formed per cell
        # above and averaged here with each cell's exposure weight, exactly
        # as the search scored it; the channel columns are averaged the same
        # way for reporting.
        if "polygon_id" in result_df.columns:
            codes, uniq = pd.factorize(result_df["polygon_id"].to_numpy())
            n_ent = len(uniq)
            row_w = (
                result_df["_weight"].to_numpy(dtype=np.float64)
                if "_weight" in result_df.columns
                else None
            )

            def _mean(col: str) -> np.ndarray:
                return self._collapse_mean_from_codes(
                    result_df[col].to_numpy(), codes, n_ent, None, row_w
                )

            _, first = np.unique(codes, return_index=True)
            comp = result_df["composite"].to_numpy(dtype=np.float64)
            poly_df = pd.DataFrame(
                {
                    "polygon_id": uniq,
                    "target": result_df["target"].to_numpy()[first],
                    "composite": _mean("composite"),
                    "n_samples": np.bincount(
                        codes[np.isfinite(comp)], minlength=n_ent
                    ),
                    **{ch: _mean(ch) for ch in all_blocks},
                }
            )
            if self._entity_coverage_share is not None:
                # Share of the entity's exposure weight the coverage gate kept.
                poly_df["coverage_share"] = poly_df["polygon_id"].map(
                    self._entity_coverage_share
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

        The ``test`` slice is the headline: the discovered params never
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
                eid, _ = self._whole_data_longitudinal_keys(merged)
            else:
                merged = pd.DataFrame(
                    {
                        "target": np.asarray(df_cgi["target"].values),
                        "composite_cgi": np.asarray(df_cgi["composite"].values),
                        "composite_std": np.asarray(df_std["composite"].values),
                    }
                )
                cov = None
                eid, _ = self._whole_data_longitudinal_keys(df_cgi)
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
            if eid is not None:
                eid = eid[finite]

        if len(target) < 3:
            return None

        # A panel's waves are resampled with their entity, as the test-score
        # bootstrap does, so correlated rows are not counted as independent.
        blocks = None
        if eid is not None:
            _, codes = np.unique(eid, return_inverse=True)
            order = np.argsort(codes, kind="stable")
            blocks = np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)

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
            if blocks is None:
                idx = rng.integers(0, n, size=n)
            else:
                picks = rng.integers(0, len(blocks), size=len(blocks))
                idx = np.concatenate([blocks[j] for j in picks])
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
        """Score ``params`` on every data slice the discovery produced.

        Returns a dict of ``{subset: {score, pvalue, n}}`` for the four
        canonical slices. Selection resamples the train+val pool, so ``train``
        / ``val`` are *labels for continuity*, relabelled in-pool / held-out in
        the results view:

        - ``train`` — the winning params scored on the full train+val pool
          (an in-pool fit; the pool is what the sweep resamples).
        - ``val`` — the sweep's mean held-out score for the winning
          configuration (``__sweep__.score``), a cross-resample measure.
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
        # (the closest analogue to "train" — the pool is what the sweep
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
        # Held-out slice: the sweep's mean score for the winning configuration
        # and form, on fresh splits of the pool. Scored on covariate-
        # residualised rows, so there is no raw equivalent at this level.
        sweep_info = params.get("__sweep__") if isinstance(params, dict) else None
        cell_median = (sweep_info or {}).get("score")
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
        panel) for a binary outcome, OLS otherwise, with entity-clustered
        standard errors on a panel — so no table counts one entity's waves as
        independent evidence.
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
                fitter = exposure_response.make_ols_fitter(target, entity_id)
                design = "ols" if entity_id is None else "ols_cluster"

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
                return exposure_response.make_ols_fitter(y, ent)

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

    def _carry_negative_controls(
        self, fusion_df: "pd.DataFrame", entity_gdf: "gpd.GeoDataFrame"
    ) -> None:
        """Copy the negative-control columns onto a freshly built fusion frame.

        They stay out of every NaN gate, so an entity missing a control is
        still tuned on; the transfer test drops it from that control alone.
        """
        for col in self.negative_control_columns:
            if col in fusion_df.columns:
                continue
            if col not in entity_gdf.columns:
                _log(
                    "WARN",
                    f"Negative control '{col}' is not a target column; " "skipping it.",
                )
                continue
            fusion_df[col] = pd.to_numeric(entity_gdf[col], errors="coerce").values

    def _test_entity_mask(self, df: "pd.DataFrame") -> np.ndarray:
        """Rows of an ``apply_fusion`` frame that belong to the held-out split."""
        test = self.test_data
        if test is None or len(test) == 0:
            return np.zeros(len(df), dtype=bool)
        if "polygon_id" in df.columns and "polygon_id" in test.columns:
            return df["polygon_id"].isin(test["polygon_id"].unique()).to_numpy()
        return df.index.isin(test.index)

    def compute_negative_controls(
        self,
        params: dict,
        *,
        metric: str | None = None,
        n_boot: int = 1000,
        seed: int = 42,
        cluster_column: str | None = None,
    ) -> dict | None:
        """Frozen-exposure transfer test against each negative-control outcome.

        The composite tuned on train+val is applied unchanged; the controls
        never entered tuning. On the held-out split (the headline) and on
        train+val (reference) it reports, for the target and every control,
        the covariate-adjusted slope in residual SD units with its interval
        and t, the paired contrast ``|β_target| − |β_control|`` with an
        ``n_boot`` bootstrap interval, and the ``nonspecific`` flag — see
        :mod:`geofuse.negative_controls`. With ``metric``, the job's objective
        is also scored on each outcome. ``cluster_column`` resamples whole
        areas instead of entities. ``None`` when no controls are configured.
        """
        from . import negative_controls as _nc

        if not self.negative_control_columns:
            return None
        if self.is_longitudinal:
            _log(
                "WARN",
                "Negative controls are reported for cross-sectional "
                "targets only; skipped for this longitudinal run.",
            )
            return {
                "skipped": "longitudinal",
                "controls": list(self.negative_control_columns),
            }
        df = self.apply_fusion(weights=dict(params))
        target = np.asarray(df["target"].values, dtype=np.float64)
        composite = np.asarray(df["composite"].values, dtype=np.float64)
        cov = self._reporting_covariate_matrix(df)
        controls = {}
        for name in self.negative_control_columns:
            values = self._reporting_raw_column(df, name)
            if values is None:
                _log("WARN", f"Negative control '{name}' not found; skipping.")
                continue
            controls[name] = values
        if not controls:
            return None
        clusters = None
        if cluster_column:
            full = self._full_data_frame()
            if full is not None and cluster_column in full.columns:
                key = "polygon_id" if "polygon_id" in df.columns else None
                clusters = (
                    full.groupby(key, sort=False)[cluster_column]
                    .first()
                    .reindex(df[key].values)
                    .to_numpy()
                    if key
                    else full[cluster_column].reindex(df.index).to_numpy()
                )

        def objective(y, x, c):
            if metric is None:
                return None
            keep = self._finite_rows(y, x, c)
            try:
                return float(
                    objective_scoring.score(
                        metric, y[keep], x[keep], None if c is None else c[keep]
                    )
                )
            except Exception:
                return None

        test = self._test_entity_mask(df)
        out: dict = {
            "controls": list(controls),
            "n_boot": int(n_boot),
            "cluster_column": cluster_column if clusters is not None else None,
            "metric": metric,
            "splits": {},
        }
        for split, mask in (("test", test), ("train_val", ~test)):
            if int(mask.sum()) < 10:
                continue
            c_m = None if cov is None else cov[mask]
            res = _nc.transfer_test(
                composite[mask],
                target[mask],
                {k: v[mask] for k, v in controls.items()},
                c_m,
                n_boot=int(n_boot),
                seed=int(seed),
                clusters=None if clusters is None else clusters[mask],
            )
            res["target"]["objective"] = objective(target[mask], composite[mask], c_m)
            for k, v in controls.items():
                res["controls"][k]["objective"] = objective(
                    v[mask], composite[mask], c_m
                )
            out["splits"][split] = res
        flagged = [
            k
            for k, v in (out["splits"].get("test") or {}).get("controls", {}).items()
            if v.get("nonspecific")
        ]
        out["nonspecific"] = flagged
        if flagged:
            _log(
                "WARN",
                "Negative control(s) "
                + ", ".join(f"'{k}'" for k in flagged)
                + " carry an association indistinguishable from the "
                "target's on the held-out split: part of the tuned "
                "association is non-specific.",
            )
        return out

    def retune_concordance(
        self, target_params: dict, metric: str, **fit_kwargs
    ) -> dict:
        """Re-tune on each negative control and compare with the target's tuning.

        Each control runs the same sweep and posterior as the target — same
        protocol, settings and seed, passed in ``fit_kwargs`` — with the
        reproducibility, gain and null loops skipped because the comparison
        reads only the posterior. Returns ``{control: concordance}`` (see
        :func:`geofuse.negative_controls.concordance`). The engine's formula is
        restored afterwards, so the target study's state is untouched.
        """
        from . import negative_controls as _nc

        target_post = (target_params or {}).get("__posterior__") or {}
        out: dict = {}
        if not self.negative_control_columns or self.is_longitudinal:
            return out
        loops = dict(reps=0, shuffles=0, gain_splits=0, null_runs=0)
        saved_formula = self.cgi_formula
        for name in self.negative_control_columns:
            try:
                params = self.fit_bayesian_index(
                    metric, outcome=name, **{**fit_kwargs, **loops}
                )
            except JobCancelled:
                raise
            except Exception as exc:
                _log("WARN", f"Re-tune on negative control '{name}' failed: {exc}")
                continue
            finally:
                self.cgi_formula = saved_formula
            out[name] = _nc.concordance(target_post, params.get("__posterior__") or {})
            same = [
                ch for ch, row in out[name]["channels"].items() if row.get("same_pick")
            ]
            if same:
                _log(
                    "WARN",
                    f"Re-tuning on '{name}' lands on the target's cell for "
                    f"{', '.join(same)}: the search may be finding the "
                    "shared confounding.",
                )
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

    def generate_composite_greenery_map(
        self,
        output_path: str = "output_results/composite_greenery.tif",
        progress_callback: Callable[..., Any] | None = None,
    ) -> str:
        """Render the composite greenery map from the discovered params.

        Builds a composite greenery raster on the study's CGI grid, for raster
        and vector targets alike, using the winning weight cell's averaged
        parameters that ``fit_bayesian_index`` pinned on ``self.best_params``.

        Args:
            output_path: Path to save the composite greenery GeoTIFF
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            Path to the saved composite greenery map
        """
        logger.info("Generating composite greenery map from discovered params...")

        if progress_callback:
            progress_callback(0, 100)

        if self.best_params is None:
            raise ValueError(
                "Composite generation needs ``self.best_params`` set by a prior "
                "discovery step."
            )
        final_params = dict(self.best_params)
        logger.info(f"Composite TIFF discovered params: {final_params}")
        formula = cgi_formulas.get_formula(self.cgi_formula)

        if progress_callback:
            progress_callback(30, 100)

        # 4. Determine grid: the CGI grid of the prepared study, or a fallback
        if (
            self._entity_grid_shape is not None
            and self._entity_grid_transform is not None
        ):
            # Render the CGI field over the cells the entities
            # are exposed to (``_prepare_vector_fusion``), so the composite
            # matches the scored values. Exposure rows repeat a cell once per
            # entity, so collapse to the unique cells.
            transform = self._entity_grid_transform
            crs = self._entity_grid_crs
            height, width = self._entity_grid_shape
            uniq = self.target_gdf.drop_duplicates(subset=["_grid_row", "_grid_col"])
            grid_rows = uniq["_grid_row"].to_numpy(dtype=np.int64)
            grid_cols = uniq["_grid_col"].to_numpy(dtype=np.int64)
            pts_cols = {"row": grid_rows, "col": grid_cols}
            # Carry the unique-cell id so the cache resolves per-cell stats
            # (the cache is keyed by cell, not by the exposure row index).
            if "_preaggr_id" in uniq.columns:
                pts_cols["_preaggr_id"] = uniq["_preaggr_id"].to_numpy()
            points_gdf = gpd.GeoDataFrame(
                pts_cols,
                geometry=uniq.geometry.values,
                crs=crs,
                index=uniq.index,
            )
            logger.info(
                f"Generating composite for {len(points_gdf)} cells "
                f"({self.cgi_grid_spacing_m:g} m grid over the exposed cells)"
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
        channels = self._composite_channels()
        blocks: dict[str, np.ndarray] = {}
        for n, ch in enumerate(channels):
            logger.info(f"Sampling {ch} at grid points...")
            radius, stat, pct = self._channel_spec(final_params, ch)
            blocks[ch] = self._aggregate_with_ring_cache(
                points_gdf,
                self._channel_source(ch),
                radius,
                stat,
                pct,
                channel=ch,
                fold_idx=-1,
                subset="robust_map",
            )
            if progress_callback:
                progress_callback(40 + int(45 * (n + 1) / len(channels)), 100)

        # The composite from the raw channels, then the ``whole_grid_scaling``
        # toggle: min-max normalise the map to [0, 1] over the whole grid when
        # on, raw otherwise; standalone single-channel maps alike.
        logger.info(f"Calculating composite via formula '{formula.name}'...")
        composite = self._composite_from(final_params, blocks)
        composite = self._finalize_composite(np.asarray(composite, dtype=np.float64))

        # 7. Create raster
        logger.info(f"Saving composite greenery map to {output_path}")
        from rasterio.windows import Window

        rows_arr = np.asarray(points_gdf["row"].values, dtype=np.int64)
        cols_arr = np.asarray(points_gdf["col"].values, dtype=np.int64)
        vals_arr = np.asarray(composite, dtype=np.float32)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # A CGI grid of exposed cells can span a country-wide extent that is
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
            "selection_method": "bayesian_index",
            # Grid geometry, so a viewer can size its read before opening the
            # file. A grid over distant clusters is mostly nodata.
            "raster_width": int(width),
            "raster_height": int(height),
            "raster_valid_cells": int(len(vals_arr)),
            "sweep": final_params.get("__sweep__"),
            "posterior": final_params.get("__posterior__"),
            "holdout_gain": final_params.get("__holdout_gain__"),
            "null_calibration": final_params.get("__null_calibration__"),
            "coverage": final_params.get("__coverage__"),
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
