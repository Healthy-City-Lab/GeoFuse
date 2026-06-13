"""
Fusion Module: Optimized metric fusion for geospatial composite indices.

This module implements the optimization logic from CGI.ipynb for tuning
weighted combinations of NDVI and GVI metrics against target outcomes.
"""

import hashlib
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import optuna
import pandas as pd
import rasterio
from optuna.pruners import HyperbandPruner, MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler
from rasterio.transform import from_origin, rowcol, xy
from scipy.stats import pearsonr, spearmanr
from shapely.geometry import box
from sklearn.metrics import mean_squared_error, mutual_info_score, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import (
    cgi_formulas,
    longitudinal,
    mixed_effects_scoring,
    objective_scoring,
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
    normalize_geographic_gdf_to_wgs84,
    reproject_geodataframe_to_wgs84,
    select_grid_crs_with_warning,
)
from .logger import attach_external_logger, get_logger
from .raster_sampling import LazyRasterArray
from .vector_io import geometry_sha256, target_path_is_raster

logger = logging.getLogger(__name__)
_log = get_logger("FUSION")

# Metric rasters larger than this (uncompressed band bytes) are read lazily in
# windows from disk instead of loaded whole — keeps national-scale rasters off
# the heap. Smaller rasters stay in RAM (faster per-point reads).
_LAZY_RASTER_THRESHOLD_BYTES = 256 * 1024 * 1024


def _nearest_metric_join(
    points_gdf: gpd.GeoDataFrame,
    metric_gdf: gpd.GeoDataFrame,
    value_col: str,
    max_distance_m: float,
) -> pd.Series:
    """Nearest-neighbour join in a metric CRS, returning one value per source row.

    ``gpd.sjoin_nearest`` has two pitfalls that produced
    ``ValueError: cannot reindex on an axis with duplicate labels`` in fusion:

    1. Run in EPSG:4326 (degrees), the ``max_distance`` parameter is interpreted
       in *degrees* rather than metres — effectively unbounded.
    2. When multiple right-hand features are exactly equidistant from a source
       point, the join returns *multiple rows for the same source index*. The
       subsequent column assignment then fails because pandas cannot reindex
       onto a duplicated axis.

    This helper reprojects both sides to a common UTM CRS so ``max_distance_m``
    is honoured, then drops duplicate left-index rows by keeping the first
    match. The returned series is reindexed onto ``points_gdf.index`` so a
    simple ``points_gdf[col] = result`` assignment is always safe.
    """
    metric_crs = points_gdf.estimate_utm_crs()
    pts_m = points_gdf[["geometry"]].to_crs(metric_crs)
    src_m = metric_gdf[["geometry", value_col]].to_crs(metric_crs)
    joined = gpd.sjoin_nearest(
        pts_m,
        src_m,
        how="left",
        max_distance=max_distance_m,
    )
    # Collapse ties: keep the first match per source row.
    joined = joined[~joined.index.duplicated(keep="first")]
    col = value_col if value_col in joined.columns else f"{value_col}_right"
    return joined[col].reindex(points_gdf.index)


def _sample_raster_values(
    points_in_raster_crs: gpd.GeoDataFrame, raster: dict
) -> np.ndarray:
    """Vectorised nearest-pixel sample of a raster dict at point geometries.

    ``points_in_raster_crs`` is already in ``raster["crs"]``. Returns a
    ``float32`` array aligned with the input rows; points whose pixel falls
    outside the raster (or on a nodata pixel) are ``NaN``. Handles both an
    in-memory ``ndarray`` / masked array and a window-backed
    :class:`~geofuse.raster_sampling.LazyRasterArray` (national-scale rasters
    kept on disk), replacing a per-row ``iterrows`` loop that does not scale to
    the millions of pixel-centroids a per-pixel CGI grid produces.
    """
    geoms = points_in_raster_crs.geometry
    if bool((geoms.geom_type == "Point").all()):
        xs = geoms.x.to_numpy()
        ys = geoms.y.to_numpy()
    else:
        centroids = geoms.centroid
        xs = centroids.x.to_numpy()
        ys = centroids.y.to_numpy()

    data = raster["data"]
    h, w = int(data.shape[0]), int(data.shape[1])
    rows, cols = rowcol(raster["transform"], xs, ys)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    out = np.full(len(geoms), np.nan, dtype=np.float32)

    valid = np.flatnonzero((rows >= 0) & (rows < h) & (cols >= 0) & (cols < w))
    if valid.size == 0:
        return out
    vr = rows[valid]
    vc = cols[valid]

    # In-memory array (incl. masked): one fancy-indexed read.
    if isinstance(data, np.ndarray):
        sampled = data[vr, vc]
        if np.ma.isMaskedArray(sampled):
            sampled = sampled.filled(np.nan)
        out[valid] = sampled
        return out

    # Window-backed (lazy) raster: read a single bounding window when it is
    # small enough to materialise, else sample scattered points from disk.
    rmin, rmax = int(vr.min()), int(vr.max()) + 1
    cmin, cmax = int(vc.min()), int(vc.max()) + 1
    if (rmax - rmin) * (cmax - cmin) <= 64_000_000:
        window = data[rmin:rmax, cmin:cmax]
        sampled = window[vr - rmin, vc - cmin]
        if np.ma.isMaskedArray(sampled):
            sampled = sampled.filled(np.nan)
        out[valid] = sampled
        return out

    band = getattr(data, "band", 1)
    nodata = getattr(data, "nodata", None)
    with rasterio.open(data.path) as src:
        sampled = np.fromiter(
            (
                rec[0]
                for rec in src.sample(
                    np.column_stack([xs[valid], ys[valid]]), indexes=band
                )
            ),
            dtype=np.float32,
            count=valid.size,
        )
    if nodata is not None and np.isfinite(nodata):
        sampled[sampled == np.float32(nodata)] = np.nan
    out[valid] = sampled
    return out


def _radius_int_bounds(
    r_min: float, r_max: float, r_step: float
) -> tuple[int, int, int]:
    """Align Optuna integer radius search to user min/max/step (metres)."""
    lo = max(1, int(round(r_min)))
    hi = int(round(r_max))
    step = max(1, int(round(r_step)))
    if lo > hi:
        lo, hi = hi, lo
    span = hi - lo
    hi_adj = lo + (span // step) * step
    if hi_adj < lo:
        hi_adj = lo
    return lo, hi_adj, step


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
        longitudinal_spec: longitudinal.LongitudinalSpec | None = None,
        cgi_grid_spacing_m: float | None = None,
        whole_grid_scaling: bool = False,
        area_balanced_split: bool = False,
        normalize_channels: bool = False,
        spatial_adjust_method: str = "none",
        spatial_adjust_max_df: int = 10,
        spatial_adjust_eps_m: float | None = None,
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

        # Greenery channel for the active study. ``cgi`` runs the combined
        # formula (the standard fusion behaviour); ``veg`` / ``terrain`` /
        # ``ndvi`` run a standalone single-metric study that uses that
        # channel's normalised value directly as the greenery value and
        # searches only its radius + aggregation. Set per call by
        # ``optimize_fusion``; ``evaluate_on_test`` reads it so the held-out
        # test score stays aligned with what the study actually scored.
        self._active_greenery_channel: str = "cgi"

        # Covariate columns are stashed on self; per-row values are carried into
        # the prepared fusion DataFrame, then split + scored alongside target
        # and CGI. Empty list = legacy single-variable scoring (no change).
        # De-duplicated to avoid a singular design matrix in the partial-
        # correlation / incremental-R² OLS.
        cov_in = list(covariate_columns) if covariate_columns else []
        self.covariate_columns: list[str] = list(dict.fromkeys(cov_in))

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
        self.train_val_data = None  # Data for k-fold CV
        self.test_data = None  # Held-out test set
        self.cv_folds = None  # K-fold splits
        self.study = None
        self.best_params = None
        self.scaler = None
        self.k_folds = 5  # Number of CV folds

        # Donut / ring cache: per-point annulus samples keyed by fold subset + indices
        self._ring_raster_cache: dict = {}
        self._ring_vector_cache: dict = {}
        self._max_points_ring_cache = 8000

        # Spatial pre-aggregation (mandatory; built by precompute_aggregations()).
        # Backed by an on-disk SQLite cache (geofuse/preaggregation.py) so the
        # per-(entity, radius) stat table survives cancels/crashes and is reused
        # across runs.
        self._preaggregation_done: bool = False
        self._preaggr_cache: preaggregation.PreAggregationCache | None = None
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
        # Per-wave metric sources for the mixed-effects mode. Populated by
        # ``set_longitudinal_metric_data`` before ``precompute_aggregations``.
        # Outer dict: channel → wave_label → metric source (GeoDataFrame for
        # vector channels, raster-dict for raster channels — same layout
        # ``load_metrics`` produces). Cross-sectional runs leave this ``None``.
        self._longitudinal_metric_data: dict[str, dict[str, Any]] | None = None

        if self.is_longitudinal and self.is_raster:
            raise ValueError(
                "longitudinal_spec is only supported for vector targets; "
                "raster targets have no entity-id attribute column."
            )

    # ------------------------------------------------------------------
    # Longitudinal mode — entry points + helpers
    # ------------------------------------------------------------------

    @property
    def is_longitudinal(self) -> bool:
        """True when a :class:`LongitudinalSpec` was provided at construction."""
        return self.longitudinal_spec is not None

    def set_longitudinal_metric_data(
        self,
        channel: str,
        per_wave_data: dict[str, Any],
    ) -> None:
        """Inject pre-loaded per-wave metric data for one greenery channel.

        Cross-sectional callers populate ``self.veg_data`` / ``self.terrain_data``
        / ``self.ndvi_data`` via :meth:`load_metrics`. Mixed-effects callers
        instead supply one metric source per wave for each channel (the runner
        resolves each wave's file path against loaded results / uploads /
        auto-download and hands the result here). ``per_wave_data`` maps every
        wave label in the spec to either a ``GeoDataFrame`` (vector metric) or
        the raster-dict layout ``load_metrics`` produces; the cache build then
        consults the right per-wave source for each ``(entity_id, wave)`` row.
        """
        if not self.is_longitudinal:
            raise RuntimeError(
                "set_longitudinal_metric_data() requires longitudinal_spec to be set."
            )
        if channel not in longitudinal.GREENERY_CHANNELS:
            raise ValueError(
                f"Unknown channel {channel!r}; expected one of "
                f"{longitudinal.GREENERY_CHANNELS}."
            )
        spec = self.longitudinal_spec
        assert spec is not None
        missing = [w for w in spec.wave_labels if w not in per_wave_data]
        if missing:
            raise ValueError(
                f"per_wave_data is missing entries for {channel!r} waves: {missing}."
            )
        if self._longitudinal_metric_data is None:
            self._longitudinal_metric_data = {}
        self._longitudinal_metric_data[channel] = dict(per_wave_data)
        # Mirror to the cross-sectional attribute so any code path that
        # still checks ``self.<channel>_data is None`` (e.g. the
        # pre-aggregation entry guard) passes. The actual per-wave source
        # is consulted via ``_longitudinal_metric_data``.
        rep_source = per_wave_data[spec.wave_labels[0]]
        if channel == "veg":
            self.veg_data = rep_source
        elif channel == "terrain":
            self.terrain_data = rep_source
        elif channel == "ndvi":
            self.ndvi_data = rep_source

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
            read_kwargs: dict = {}
            if self.target_layer is not None and Path(
                self.target_file
            ).suffix.lower() in (
                ".gpkg",
                ".zip",
            ):
                read_kwargs["layer"] = self.target_layer
            self.target_gdf = reproject_geodataframe_to_wgs84(
                gpd.read_file(self.target_file, **read_kwargs)
            )

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
        """Trim a metric source to the target's buffered extent + a margin.

        For a small study area, most of a national-scale NDVI / GVI source
        is dead weight: it's never queried because no entity's buffer
        reaches it. Cropping once at load time shrinks RAM, sindex build
        time (vector), and per-window read latency (raster) without
        changing any aggregation values.

        The crop box = the target's pre-computed ``buffered_extent``
        (already buffered by ``buffer_meters`` = max(GVI, NDVI) at job
        submit) plus a small ``extra_margin_m`` slack so floating-point
        corner cases don't accidentally drop pixels at the very edge.

        Vector metrics are spatially filtered to features intersecting
        the crop polygon. Raster metrics get a windowed read covering the
        crop bbox; the returned dict has the same keys as ``load_metrics``
        produces, with updated ``data`` / ``transform`` / ``width`` /
        ``height`` / ``bounds``. Returns the input unchanged when
        ``self.buffered_extent`` isn't yet available.
        """
        if self.buffered_extent is None or metric_data is None:
            return metric_data

        if isinstance(metric_data, dict):  # raster
            from rasterio.windows import Window as _Window
            from rasterio.windows import transform as _window_transform

            raster_crs = metric_data["crs"]
            try:
                extent_in_raster_crs = self.buffered_extent.to_crs(raster_crs)
                if getattr(extent_in_raster_crs.crs, "is_geographic", False):
                    metric_crs_buf = estimate_metre_projected_crs_for_gdf(
                        extent_in_raster_crs
                    )
                    crop_in_raster = (
                        extent_in_raster_crs.to_crs(metric_crs_buf)
                        .buffer(extra_margin_m)
                        .to_crs(raster_crs)
                    )
                else:
                    crop_in_raster = extent_in_raster_crs.buffer(extra_margin_m)
            except Exception:
                crop_in_raster = self.buffered_extent.to_crs(raster_crs)
            minx, miny, maxx, maxy = crop_in_raster.total_bounds
            transform = metric_data["transform"]
            h_full = metric_data.get("height") or (
                metric_data["data"].shape[0]
                if hasattr(metric_data["data"], "shape")
                else None
            )
            w_full = metric_data.get("width") or (
                metric_data["data"].shape[1]
                if hasattr(metric_data["data"], "shape")
                else None
            )
            if h_full is None or w_full is None:
                return metric_data
            from rasterio.transform import rowcol

            r1, c1 = rowcol(transform, minx, maxy)
            r2, c2 = rowcol(transform, maxx, miny)
            rmin = max(0, min(int(r1), int(r2)))
            rmax = min(int(h_full), max(int(r1), int(r2)) + 1)
            cmin = max(0, min(int(c1), int(c2)))
            cmax = min(int(w_full), max(int(c1), int(c2)) + 1)
            if rmin >= rmax or cmin >= cmax:
                _log(
                    "WARN",
                    f"{channel_label}: cropped raster window is empty — target "
                    "extent does not overlap the metric raster. Leaving full "
                    "raster in place.",
                )
                return metric_data
            new_h = rmax - rmin
            new_w = cmax - cmin
            if new_h == h_full and new_w == w_full:
                return metric_data  # already fits the target
            arr = metric_data["data"]
            cropped = arr[rmin:rmax, cmin:cmax]
            cropped = np.ma.MaskedArray(
                np.asarray(np.ma.getdata(cropped)),
                mask=np.ma.getmaskarray(cropped),
            )
            new_transform = _window_transform(
                _Window(cmin, rmin, new_w, new_h), transform
            )
            saved_pct = 100.0 * (1.0 - (new_h * new_w) / (h_full * w_full))
            _log(
                "INFO",
                f"{channel_label}: raster cropped to target extent "
                f"({h_full:,}×{w_full:,} → {new_h:,}×{new_w:,} px, "
                f"~{saved_pct:.1f}% memory saved).",
            )
            # Recompute bounds from the new window so downstream code that
            # reads them sees the cropped footprint, not the original.
            from rasterio.coords import BoundingBox as _BBox

            new_left = new_transform.c
            new_top = new_transform.f
            new_right = new_left + new_w * new_transform.a
            new_bottom = new_top + new_h * new_transform.e
            return {
                "data": cropped,
                "transform": new_transform,
                "crs": raster_crs,
                "bounds": _BBox(
                    min(new_left, new_right),
                    min(new_top, new_bottom),
                    max(new_left, new_right),
                    max(new_top, new_bottom),
                ),
                "width": new_w,
                "height": new_h,
            }

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

                        # Convert to point GeoDataFrame
                        rows, cols = np.meshgrid(
                            np.arange(height), np.arange(width), indexing="ij"
                        )
                        xs, ys = xy(
                            transform, rows.flatten(), cols.flatten(), offset="center"
                        )

                        # Create veg GeoDataFrame
                        veg_values = veg_band.flatten()
                        veg_mask = ~np.isnan(veg_values)
                        veg_gdf = gpd.GeoDataFrame(
                            {"veg": veg_values[veg_mask]},
                            geometry=gpd.points_from_xy(xs[veg_mask], ys[veg_mask]),
                            crs=crs,
                        )
                        veg_gdf.attrs["metric_column"] = "veg"
                        self.veg_data = veg_gdf

                        # Create terrain GeoDataFrame
                        terrain_values = terrain_band.flatten()
                        terrain_mask = ~np.isnan(terrain_values)
                        terrain_gdf = gpd.GeoDataFrame(
                            {"terrain": terrain_values[terrain_mask]},
                            geometry=gpd.points_from_xy(
                                xs[terrain_mask], ys[terrain_mask]
                            ),
                            crs=crs,
                        )
                        terrain_gdf.attrs["metric_column"] = "terrain"
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

        _VEG_COLUMN_ALIASES = ("gvi_veg", "gvi", "veg", "vegetation")
        _TER_COLUMN_ALIASES = ("gvi_ter", "gvi_terrain", "terrain")

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
            lowered = {c.lower(): c for c in gdf.columns}
            veg_col = next(
                (lowered[a] for a in _VEG_COLUMN_ALIASES if a in lowered), None
            )
            ter_col = next(
                (lowered[a] for a in _TER_COLUMN_ALIASES if a in lowered), None
            )
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
                self.veg_data = self._load_metric_file(veg_file)
                if terrain_file and os.path.exists(terrain_file):
                    self.terrain_data = self._load_metric_file(terrain_file)
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
                    self.terrain_data = self._load_metric_file(terrain_file)
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
                self.veg_data = self._load_metric_file(veg_file)
                self.terrain_data = self._load_metric_file(terrain_file)
        # Load separate veg/terrain files if provided
        elif (
            veg_file
            and os.path.exists(veg_file)
            and terrain_file
            and os.path.exists(terrain_file)
        ):
            logger.info("Loading separate veg and terrain files...")
            self.veg_data = self._load_metric_file(veg_file)
            self.terrain_data = self._load_metric_file(terrain_file)
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
            self.veg_data = self._load_metric_file(veg_file)
            self.terrain_data = self._load_metric_file(terrain_file)
        else:
            # Partial files provided - try to load what we have
            if veg_file and os.path.exists(veg_file):
                logger.info(f"Loading vegetation from: {veg_file}")
                self.veg_data = self._load_metric_file(veg_file)
            if terrain_file and os.path.exists(terrain_file):
                logger.info(f"Loading terrain from: {terrain_file}")
                self.terrain_data = self._load_metric_file(terrain_file)

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
            self.ndvi_data = self._load_metric_file(ndvi_file)
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
            self.ndvi_data = self._load_metric_file(ndvi_file)

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
        """Check if manually provided metric covers the buffered extent."""
        if metric_file.endswith((".tif", ".tiff")):
            with rasterio.open(metric_file) as src:
                metric_box = box(*src.bounds)
                metric_crs = src.crs
        else:
            metric_gdf = gpd.read_file(metric_file)
            metric_box = box(*metric_gdf.total_bounds)
            metric_crs = metric_gdf.crs

        # Reproject extent to metric CRS for comparison
        extent_reprojected = self.buffered_extent.to_crs(metric_crs)
        extent_box = extent_reprojected.geometry.iloc[0]

        if not metric_box.contains(extent_box):
            logger.warning(
                f"Metric file {metric_file} does not fully cover buffered extent. "
                "Results may be incomplete."
            )
            return False
        return True

    def _load_metric_file(self, filepath: str) -> gpd.GeoDataFrame | dict:
        """Load metric from GeoJSON or GeoTIFF."""
        if filepath.endswith((".tif", ".tiff")):
            with rasterio.open(filepath) as src:
                transform, crs, bounds = src.transform, src.crs, src.bounds
                itemsize = np.dtype(src.dtypes[0]).itemsize
                est_bytes = src.width * src.height * itemsize
                data = (
                    src.read(1, masked=True)
                    if est_bytes <= _LAZY_RASTER_THRESHOLD_BYTES
                    else None
                )
            if data is None:
                # Too large to hold in RAM (national-scale): read windows from
                # disk on demand instead. Each thread gets its own handle.
                data = LazyRasterArray(filepath, band=1)
                logger.info(
                    f"Metric raster ~{est_bytes / (1024**2):.0f} MB exceeds the "
                    f"in-memory threshold; reading windows lazily from {filepath}"
                )
            return {
                "data": data,
                "transform": transform,
                "crs": crs,
                "bounds": bounds,
            }
        else:
            gdf = gpd.read_file(filepath)
            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)
            gdf = normalize_geographic_gdf_to_wgs84(gdf)

            # Detect metric column - look for common naming patterns
            metric_col = None
            for col in gdf.columns:
                col_lower = col.lower()
                if col_lower in [
                    "gvi",
                    "gvi_veg",
                    "veg",
                    "vegetation",
                    "terrain",
                    "gvi_ter",
                    "ndvi",
                    "value",
                    "metric",
                ]:
                    metric_col = col
                    break

            if metric_col is None:
                # Try to find first numeric column (excluding index columns)
                numeric_cols = gdf.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [
                    c
                    for c in numeric_cols
                    if c not in ["index", "index_right", "index_left"]
                ]
                if numeric_cols:
                    metric_col = numeric_cols[0]
                    logger.info(
                        f"Using '{metric_col}' as metric column from {filepath}"
                    )

            # Store the metric column name for later use
            gdf.attrs["metric_column"] = metric_col
            return gdf

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

    def _save_points_as_raster(
        self, points_gdf: gpd.GeoDataFrame, value_col: str, output_path: str
    ) -> None:
        """
        Convert point GeoDataFrame to raster matching the target raster grid.

        Args:
            points_gdf: GeoDataFrame with point geometries and values
            value_col: Column name containing values to rasterize
            output_path: Path to save the GeoTIFF
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

        # Extract point coordinates and values
        coords = np.array([(p.x, p.y) for p in points_in_raster_crs.geometry])
        values = points_in_raster_crs[value_col].values

        # Filter out NaN values
        valid_mask = ~np.isnan(values)
        coords = coords[valid_mask]
        values = values[valid_mask]

        if len(values) == 0:
            raise ValueError(f"No valid values in {value_col} column")

        # Generate target grid coordinates
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        xs, ys = xy(transform, rows.flatten(), cols.flatten(), offset="center")
        grid_coords = np.column_stack([xs, ys])

        # Interpolate point values to grid using nearest neighbor
        # (faster than cubic, appropriate for categorical-like GVI data)
        grid_values = griddata(coords, values, grid_coords, method="nearest")
        grid_values = grid_values.reshape(height, width)

        # Write to GeoTIFF — shared compression / tiling / BIGTIFF defaults
        # so internal cache rasters benefit from the same disk savings as
        # the user-facing outputs.
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
            **default_geotiff_creation_options(rasterio.float32),
        ) as dst:
            dst.write(grid_values.astype(rasterio.float32), 1)

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

    def _cache_metric(self, source_file: str, metric_type: str) -> None:
        """Cache metric file with standardized naming."""
        import shutil

        ext = os.path.splitext(source_file)[1]
        cache_path = self._get_cache_filename(metric_type, ext)

        shutil.copy(source_file, cache_path)
        logger.info(f"Cached {metric_type} to: {cache_path}")

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

    def _auto_download_gvi(
        self,
        component: str,
        api_key: str | None = None,
        cache: bool = True,
        progress_callback: Callable[..., Any] | None = None,
        cancel_callback: Callable[..., Any] | None = None,
    ) -> str:
        """
        Auto-download GVI metrics within buffered extent.

        Args:
            component: 'veg' or 'terrain'
            api_key: Google Street View API key (optional)
            cache: Whether to save to cache directory

        Returns:
            Path to generated GeoJSON file
        """
        from .gvi import GVIEngine

        # Check cache first using deterministic filename
        cache_path = self._get_cache_filename(component, ".geojson")

        if os.path.exists(cache_path):
            logger.info(f"Found cached {component} at: {cache_path}")
            return cache_path

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
            f"Computing GVI {component} for buffered extent (this may take a while)..."
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
                logger.info(
                    f"GVI {component}: Collected {len(accumulated_results)} results so far"
                )

        def on_progress(curr, total):
            if progress_callback:
                progress_callback(component, curr, total)

        logger.info(f"Starting GVI {component} analysis...")
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
                logger.info(f"GVI {component} analysis cancelled by user")
                raise InterruptedError(f"GVI {component} analysis cancelled by user")
            raise ValueError(
                f"No {component} data collected. Check if Street View is available in this area."
            )

        result_gdf = gpd.GeoDataFrame(accumulated_results, crs=analysis_gdf.crs)

        # Extract and save the requested component
        if component == "veg":
            metric_col = "gvi_veg"
        else:
            metric_col = "gvi_ter"

        output_gdf = result_gdf[["geometry", metric_col]].copy()
        output_gdf.rename(columns={metric_col: component}, inplace=True)

        if cache:
            output_gdf.to_file(cache_path, driver="GeoJSON")
            logger.info(f"Cached {component} to: {cache_path}")
            return cache_path
        else:
            temp_path = os.path.join(self.cache_dir, f"temp_{component}.geojson")
            output_gdf.to_file(temp_path, driver="GeoJSON")
            return temp_path

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

    def _apply_circular_buffer_aggregation(
        self,
        points_gdf: gpd.GeoDataFrame,
        metric_data: gpd.GeoDataFrame | dict,
        radius_meters: float,
        stat: str,
        percentile: int = 50,
    ) -> np.ndarray:
        """
        Apply circular buffer aggregation to sample metric at point locations.

        This matches CGI.ipynb calculate_gm() logic:
        - Creates circular buffer around each point
        - Samples all metric values within buffer
        - Aggregates using specified statistic (mean/median/percentile)

        Args:
            points_gdf: GeoDataFrame with point geometries
            metric_data: Raster dict or vector GeoDataFrame to sample from
            radius_meters: Buffer radius in meters
            stat: Aggregation function ('mean', 'median', 'percentile')
            percentile: Percentile value if stat='percentile'

        Returns:
            Array of aggregated metric values for each point
        """
        result = np.full(len(points_gdf), np.nan)

        # Create mapping from original index to position for correct array indexing
        idx_to_pos = {idx: pos for pos, idx in enumerate(points_gdf.index)}

        if isinstance(metric_data, dict):  # Raster
            metric_array = metric_data["data"]
            transform = metric_data["transform"]
            metric_crs = metric_data["crs"]

            # Reproject points to metric CRS
            points_metric_crs = points_gdf.to_crs(metric_crs)

            # Calculate radius in pixels - handle geographic vs projected CRS
            pixel_size = abs(transform.a)  # Pixel width in CRS units

            # Check if CRS is geographic (degrees) - pixel size will be very small
            if metric_crs.is_geographic:
                # For geographic CRS, estimate pixel size in meters
                # Using approximate conversion at equator: 1 degree ≈ 111,320 meters
                pixel_size_meters = pixel_size * 111320
            else:
                # For projected CRS, pixel size is already in meters (or similar units)
                pixel_size_meters = pixel_size

            radius_pixels = int(radius_meters / pixel_size_meters)

            # Sanity check to prevent memory explosion
            if radius_pixels > 10000:
                logger.warning(
                    f"Radius in pixels ({radius_pixels}) is too large. "
                    f"This may indicate CRS mismatch. Skipping raster aggregation."
                )
                return result  # Return all NaN

            if radius_pixels < 1:
                radius_pixels = 1  # Minimum 1 pixel

            # Create circular mask
            y, x = np.ogrid[
                -radius_pixels : radius_pixels + 1,
                -radius_pixels : radius_pixels + 1,
            ]
            circle_mask = (x**2 + y**2) <= radius_pixels**2

            # Sample each point
            for idx, point in points_metric_crs.iterrows():
                row, col = rowcol(transform, point.geometry.x, point.geometry.y)

                # Extract window around point
                rmin = max(row - radius_pixels, 0)
                rmax = min(row + radius_pixels + 1, metric_array.shape[0])
                cmin = max(col - radius_pixels, 0)
                cmax = min(col + radius_pixels + 1, metric_array.shape[1])

                if rmin >= rmax or cmin >= cmax:
                    continue

                window = metric_array[rmin:rmax, cmin:cmax]

                # Crop circular mask to match window size
                mask_crop = circle_mask[
                    (radius_pixels - (row - rmin)) : (radius_pixels + (rmax - row)),
                    (radius_pixels - (col - cmin)) : (radius_pixels + (cmax - col)),
                ]

                # Extract values within circular buffer
                if hasattr(window, "compressed"):  # Masked array
                    valid_mask = ~window.mask & mask_crop
                    values = window.data[valid_mask]
                else:
                    values = window[mask_crop]

                # Remove NaN values
                values = values[~np.isnan(values)]

                # Apply aggregation statistic
                if len(values) > 0:
                    pos = idx_to_pos[idx]
                    if stat == "mean":
                        result[pos] = np.mean(values)
                    elif stat == "median":
                        result[pos] = np.median(values)
                    elif stat == "percentile":
                        result[pos] = np.percentile(values, percentile)

        else:  # Vector GeoDataFrame
            # For vector data, use spatial join with buffered points
            # Convert both to a common metric CRS for accurate buffering

            # Determine appropriate UTM zone from point data
            # Convert to WGS84 first to get proper geographic coordinates (lon/lat in degrees)
            points_wgs84 = points_gdf.to_crs("EPSG:4326")
            centroid = points_wgs84.geometry.union_all().centroid
            lon = centroid.x
            lat = centroid.y

            # Validate we actually have geographic coordinates (lon should be -180 to 180)
            if not (-180 <= lon <= 180):
                raise ValueError(
                    f"Invalid longitude {lon} after WGS84 conversion. "
                    f"Input CRS was {points_gdf.crs}, conversion may have failed."
                )

            # Calculate UTM zone from geographic coordinates
            utm_zone = int((lon + 180) / 6) + 1
            # Determine hemisphere
            if lat >= 0:
                utm_crs = f"EPSG:326{utm_zone:02d}"  # Northern hemisphere
            else:
                utm_crs = f"EPSG:327{utm_zone:02d}"  # Southern hemisphere

            # Convert to UTM for buffering
            points_utm = points_gdf.to_crs(utm_crs)
            metric_utm = metric_data.to_crs(utm_crs)

            # Create buffers in UTM (meters)
            points_buffered = points_utm.copy()
            points_buffered["geometry"] = points_utm.buffer(radius_meters)

            # Spatial join to find metrics within buffers
            joined = gpd.sjoin(
                points_buffered, metric_utm, how="left", predicate="intersects"
            )

            # Detect metric column
            metric_col = metric_data.attrs.get("metric_column")
            if metric_col is None or metric_col not in metric_data.columns:
                numeric_cols = metric_data.select_dtypes(
                    include=[np.number]
                ).columns.tolist()
                # Filter out geometry-related columns
                numeric_cols = [
                    c
                    for c in numeric_cols
                    if c not in ["index_right", "index_left", "index"]
                ]
                if numeric_cols:
                    metric_col = numeric_cols[0]
                else:
                    logger.warning(
                        f"No numeric columns found in metric data. Columns: {metric_data.columns.tolist()}"
                    )
                    return result  # Return all NaN

            logger.info(
                f"Buffer aggregation using column '{metric_col}' from {metric_data.columns.tolist()}, radius={radius_meters}m, stat={stat}"
            )

            # Check if any data was joined
            non_null_joins = (
                joined[metric_col].notna().sum() if metric_col in joined.columns else 0
            )
            logger.info(
                f"Spatial join: {len(joined)} total rows, {non_null_joins} with valid metric values"
            )

            logger.debug(
                f"Using metric column: {metric_col} from {metric_data.columns.tolist()}"
            )
            logger.debug(
                f"Joined shape: {joined.shape}, Points shape: {len(points_gdf)}"
            )

            # Aggregate by original point index
            for idx in points_gdf.index:
                subset = joined[joined.index == idx]
                if len(subset) > 0 and metric_col in subset.columns:
                    values = subset[metric_col].dropna().values
                    if len(values) > 0:
                        pos = idx_to_pos[idx]
                        if stat == "mean":
                            result[pos] = np.mean(values)
                        elif stat == "median":
                            result[pos] = np.median(values)
                        elif stat == "percentile":
                            result[pos] = np.percentile(values, percentile)

            # Log summary of results
            valid_count = np.sum(~np.isnan(result))
            logger.info(
                f"Buffer aggregation result: {valid_count}/{len(result)} points have valid values"
            )

        return result

    def _clear_ring_caches(self) -> None:
        self._ring_raster_cache.clear()
        self._ring_vector_cache.clear()

    @staticmethod
    def _polygonal_parts(geom):
        gt = geom.geom_type
        if gt == "Polygon":
            return [geom]
        if gt == "MultiPolygon":
            return list(geom.geoms)
        if gt == "GeometryCollection":
            parts = []
            for g in geom.geoms:
                parts.extend(MetricFusionEngine._polygonal_parts(g))
            return parts
        return []

    def _outer_radii_metres(self, *, gvi: bool) -> np.ndarray:
        if gvi:
            lo, hi, st = _radius_int_bounds(
                self.gvi_buffer_min_m,
                self.gvi_buffer_max_m,
                self.gvi_buffer_step_m,
            )
        else:
            lo, hi, st = _radius_int_bounds(
                self.ndvi_buffer_min_m,
                self.ndvi_buffer_max_m,
                self.ndvi_buffer_step_m,
            )
        return np.arange(lo, hi + 1, st, dtype=np.int64)

    @staticmethod
    def _ring_end_index(radii: np.ndarray, radius_m: float) -> int:
        r = int(round(float(radius_m)))
        hits = np.flatnonzero(radii == r)
        if hits.size:
            return int(hits[-1])
        return int(np.searchsorted(radii, r, side="right") - 1)

    @staticmethod
    def _aggregate_disk_from_rings(
        ring_arrays: list[np.ndarray],
        end_ring_idx: int,
        stat: str,
        percentile: int,
    ) -> float:
        if end_ring_idx < 0:
            return np.nan
        parts = ring_arrays[: end_ring_idx + 1]
        nonempty = [p for p in parts if p.size > 0]
        if not nonempty:
            return np.nan
        vals = np.concatenate(nonempty)
        if stat == "mean":
            return float(np.mean(vals))
        if stat == "median":
            return float(np.median(vals))
        if stat == "percentile":
            return float(np.percentile(vals, percentile))
        return np.nan

    def _vector_metric_column(self, metric_data: gpd.GeoDataFrame, channel: str) -> str:
        metric_col = metric_data.attrs.get("metric_column")
        if metric_col and metric_col in metric_data.columns:
            return metric_col
        if channel == "veg":
            for col in ["veg", "gvi_veg", "gvi", "GVI", "value"]:
                if col in metric_data.columns:
                    return col
        elif channel == "terrain":
            for col in ["terrain", "gvi_ter", "NDVI", "ndvi", "value"]:
                if col in metric_data.columns:
                    return col
        else:
            for col in ["NDVI", "ndvi", "value"]:
                if col in metric_data.columns:
                    return col
        numeric_cols = metric_data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [
            c for c in numeric_cols if c not in ["index_right", "index_left", "index"]
        ]
        if numeric_cols:
            return numeric_cols[0]
        raise ValueError(f"No numeric metric column for channel={channel}")

    def _precompute_raster_ring_values(
        self,
        metric_dict: dict,
        points_gdf: gpd.GeoDataFrame,
        radii_m: np.ndarray,
    ) -> list[list[np.ndarray]]:
        n_pts = len(points_gdf)
        n_rings = len(radii_m)
        ring_values: list[list[np.ndarray]] = [
            [np.array([], dtype=np.float64) for _ in range(n_rings)]
            for _ in range(n_pts)
        ]

        metric_array = metric_dict["data"]
        transform = metric_dict["transform"]
        metric_crs = metric_dict["crs"]
        points_metric_crs = points_gdf.to_crs(metric_crs)

        pixel_size = abs(transform.a)
        if metric_crs.is_geographic:
            pixel_size_meters = pixel_size * 111320
        else:
            pixel_size_meters = pixel_size

        max_r_m = float(radii_m[-1])
        max_r_px = int(max_r_m / pixel_size_meters)
        max_r_px = max(1, min(max_r_px, 10000))

        idx_to_pos = {idx: pos for pos, idx in enumerate(points_gdf.index)}

        for idx, point in points_metric_crs.iterrows():
            pos = idx_to_pos[idx]
            row, col = rowcol(transform, point.geometry.x, point.geometry.y)

            rmin = max(row - max_r_px, 0)
            rmax = min(row + max_r_px + 1, metric_array.shape[0])
            cmin = max(col - max_r_px, 0)
            cmax = min(col + max_r_px + 1, metric_array.shape[1])
            if rmin >= rmax or cmin >= cmax:
                continue

            window = metric_array[rmin:rmax, cmin:cmax]
            rr = np.arange(rmin, rmax, dtype=np.float64)[:, None]
            cc = np.arange(cmin, cmax, dtype=np.float64)[None, :]
            dr = rr - float(row)
            dc = cc - float(col)
            dist_m = np.sqrt(dr * dr + dc * dc) * pixel_size_meters

            if hasattr(window, "mask"):
                base_valid = ~window.mask
                data = window.data
            else:
                base_valid = np.ones(window.shape, dtype=bool)
                data = window

            for k in range(n_rings):
                inner_m = 0.0 if k == 0 else float(radii_m[k - 1])
                outer_m = float(radii_m[k])
                if inner_m <= 0:
                    ring_mask = dist_m <= outer_m
                else:
                    ring_mask = (dist_m <= outer_m) & (dist_m > inner_m)
                valid = base_valid & ring_mask
                vals = np.asarray(data[valid], dtype=np.float64).ravel()
                vals = vals[~np.isnan(vals)]
                ring_values[pos][k] = vals

        return ring_values

    def _precompute_vector_ring_values(
        self,
        metric_data: gpd.GeoDataFrame,
        points_gdf: gpd.GeoDataFrame,
        radii_m: np.ndarray,
        metric_col: str,
    ) -> list[list[np.ndarray]]:
        n_pts = len(points_gdf)
        n_rings = len(radii_m)
        ring_values: list[list[np.ndarray]] = [
            [np.array([], dtype=np.float64) for _ in range(n_rings)]
            for _ in range(n_pts)
        ]

        points_wgs84 = points_gdf.to_crs("EPSG:4326")
        centroid = points_wgs84.geometry.union_all().centroid
        lon, lat = centroid.x, centroid.y
        utm_zone = int((lon + 180) / 6) + 1
        utm_crs = f"EPSG:326{utm_zone:02d}" if lat >= 0 else f"EPSG:327{utm_zone:02d}"

        points_utm = points_gdf.to_crs(utm_crs)
        metric_utm = metric_data.to_crs(utm_crs)

        idx_to_pos = {idx: pos for pos, idx in enumerate(points_gdf.index)}

        for idx, prow in points_utm.iterrows():
            pos = idx_to_pos[idx]
            pt = prow.geometry
            for k in range(n_rings):
                inner_m = 0.0 if k == 0 else float(radii_m[k - 1])
                outer_m = float(radii_m[k])
                buf_o = pt.buffer(outer_m)
                if inner_m <= 0:
                    ring_poly = buf_o
                else:
                    ring_poly = buf_o.difference(pt.buffer(inner_m))
                vals_list: list[np.ndarray] = []
                for poly in MetricFusionEngine._polygonal_parts(ring_poly):
                    if poly.is_empty:
                        continue
                    tmp = gpd.GeoDataFrame(geometry=[poly], crs=points_utm.crs)
                    joined = gpd.sjoin(
                        metric_utm, tmp, how="inner", predicate="intersects"
                    )
                    if metric_col in joined.columns:
                        v = joined[metric_col].dropna().values.astype(np.float64)
                        if v.size:
                            vals_list.append(v)
                ring_values[pos][k] = (
                    np.concatenate(vals_list) if vals_list else np.array([])
                )

        return ring_values

    def _ring_cache_key(
        self,
        channel: str,
        fold_idx: int,
        subset: str,
        points_gdf: gpd.GeoDataFrame,
        radii: np.ndarray,
    ) -> tuple:
        return (
            channel,
            fold_idx,
            subset,
            tuple(points_gdf.index),
            radii.tobytes(),
        )

    def _aggregate_from_ring_cache(
        self,
        radii: np.ndarray,
        ring_rows: list[list[np.ndarray]],
        radius_m: float,
        stat: str,
        percentile: int,
    ) -> np.ndarray:
        n = len(ring_rows)
        out = np.full(n, np.nan, dtype=np.float64)
        end_idx = MetricFusionEngine._ring_end_index(radii, radius_m)
        if end_idx < 0:
            return out
        for pos in range(n):
            out[pos] = MetricFusionEngine._aggregate_disk_from_rings(
                ring_rows[pos], end_idx, stat, percentile
            )
        return out

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
        # ── Fast path: pre-aggregation lookup table ──────────────────────────
        if getattr(self, "_preaggregation_done", False):
            wave_indices = None
            if self.is_longitudinal and "wave" in points_gdf.columns:
                spec = self.longitudinal_spec
                assert spec is not None
                wave_index_of = {w: i for i, w in enumerate(spec.wave_labels)}
                wave_indices = np.asarray(
                    [wave_index_of[str(w)] for w in points_gdf["wave"]],
                    dtype=np.int64,
                )
            # Point/line targets cache stats per unique pixel: resolve each
            # duplicated catchment row to its pixel id. Other targets key the
            # cache by row index directly.
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

        if (
            fold_idx is None
            or subset is None
            or len(points_gdf) > self._max_points_ring_cache
        ):
            return self._apply_circular_buffer_aggregation(
                points_gdf, metric_data, radius_m, stat, percentile
            )

        gvi = channel in ("veg", "terrain")
        radii = self._outer_radii_metres(gvi=gvi)
        if radii.size == 0:
            return self._apply_circular_buffer_aggregation(
                points_gdf, metric_data, radius_m, stat, percentile
            )

        cache_store = (
            self._ring_raster_cache
            if isinstance(metric_data, dict)
            else self._ring_vector_cache
        )
        key = self._ring_cache_key(channel, fold_idx, subset, points_gdf, radii)

        if key not in cache_store:
            try:
                if isinstance(metric_data, dict):
                    rows = self._precompute_raster_ring_values(
                        metric_data, points_gdf, radii
                    )
                else:
                    col = self._vector_metric_column(metric_data, channel)
                    rows = self._precompute_vector_ring_values(
                        metric_data, points_gdf, radii, col
                    )
                cache_store[key] = (radii, rows)
                logger.info(
                    f"Ring cache built: channel={channel} subset={subset} fold={fold_idx} "
                    f"points={len(points_gdf)} rings={len(radii)}"
                )
            except Exception as e:
                logger.warning(f"Ring cache build failed ({channel}): {e}")
                return self._apply_circular_buffer_aggregation(
                    points_gdf, metric_data, radius_m, stat, percentile
                )

        _, rows = cache_store[key]
        return self._aggregate_from_ring_cache(radii, rows, radius_m, stat, percentile)

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
                missing = [
                    c
                    for c in self.covariate_columns
                    if c not in self.target_gdf.columns
                ]
                if missing:
                    raise ValueError(
                        f"covariate_columns not found on target: {missing}. "
                        f"Available numeric attribute columns: "
                        f"{sorted(self.target_gdf.columns)}"
                    )
                non_numeric = [
                    c
                    for c in self.covariate_columns
                    if not pd.api.types.is_numeric_dtype(self.target_gdf[c])
                ]
                if non_numeric:
                    raise ValueError(
                        "covariate_columns must be numeric for the regression-"
                        f"based scorers; got non-numeric: {non_numeric}."
                    )

        # Only the point/line per-pixel path dedups the cache to unique pixels;
        # reset here so a re-prepared study can't inherit a stale source.
        self._preaggr_entity_gdf = None
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
        return df

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
            if hi <= lo:
                return np.asarray(a, dtype=np.float64)
            return np.clip((np.asarray(a, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)

        return _scale(veg, "veg"), _scale(terrain, "terrain"), _scale(ndvi, "ndvi")

    # ------------------------------------------------------------------
    # Spatial pre-aggregation (mandatory; on-disk SQLite cache)
    # ------------------------------------------------------------------
    #
    # ``precompute_aggregations()`` builds a per-(entity, radius) table of
    # (mean, p10..p90) for every channel and persists it via
    # ``geofuse/preaggregation.py`` so the table survives cancels/crashes and is
    # reused across runs. ``_aggregate_with_ring_cache`` short-circuits to a
    # single column read when the trial's (radius, stat) maps onto the stored
    # grid; off-grid percentiles fall back to the lazy ring cache.
    # Trial-suggested percentiles are constrained to this 10 % grid so every
    # trial maps to a stored column.
    _PREAGGR_PERCENTILES = preaggregation.PERCENTILES

    def _preaggr_radii(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """(gvi_radii, ndvi_radii) snapped to int + step from the buffer ladders."""
        gvi_lo, gvi_hi, gvi_st = _radius_int_bounds(
            self.gvi_buffer_min_m, self.gvi_buffer_max_m, self.gvi_buffer_step_m
        )
        ndvi_lo, ndvi_hi, ndvi_st = _radius_int_bounds(
            self.ndvi_buffer_min_m, self.ndvi_buffer_max_m, self.ndvi_buffer_step_m
        )
        return (
            tuple(range(gvi_lo, gvi_hi + 1, gvi_st)),
            tuple(range(ndvi_lo, ndvi_hi + 1, ndvi_st)),
        )

    @staticmethod
    def _metric_value_column(metric_gdf: gpd.GeoDataFrame, default_col: str) -> str:
        col = metric_gdf.attrs.get("metric_column")
        if not col or col not in metric_gdf.columns:
            for c in (default_col, "value"):
                if c in metric_gdf.columns:
                    col = c
                    break
        if not col:
            raise ValueError(
                "Pre-aggregation: cannot find value column in metric "
                f"(columns: {list(metric_gdf.columns)})."
            )
        return col

    def _metric_fingerprint(self, metric, default_col: str) -> str:
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
        col = self._metric_value_column(metric, default_col)
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
        # Pre-aggregate over the unique-pixel source for point/line targets;
        # over the entity rows directly otherwise. Stats are keyed by this
        # frame's index, which scoring rows resolve via ``_preaggr_id``.
        preaggr_gdf = (
            self._preaggr_entity_gdf
            if self._preaggr_entity_gdf is not None
            else self.target_gdf
        )
        n_points = len(preaggr_gdf)
        defaults = {"veg": "veg", "terrain": "terrain", "ndvi": "NDVI"}

        # ---- Fingerprint + cache file (per data-config; reused across runs) ----
        fp_src = "|".join(
            [
                f"geom:{geometry_sha256(preaggr_gdf)}",
                self._metric_fingerprint(self.veg_data, defaults["veg"]),
                self._metric_fingerprint(self.terrain_data, defaults["terrain"]),
                self._metric_fingerprint(self.ndvi_data, defaults["ndvi"]),
                f"gvi:{gvi_radii}",
                f"ndvi:{ndvi_radii}",
                f"stats:{preaggregation.STAT_COLUMNS}",
                f"cgi_grid:{self.cgi_grid_spacing_m if (self.is_polygon_target or self.is_points) else 'na'}",
            ]
        )
        fingerprint = hashlib.sha256(fp_src.encode()).hexdigest()
        preaggr_dir = os.path.join(self.cache_dir, "preaggr")
        os.makedirs(preaggr_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(self.target_file))[0]
        db_path = os.path.join(preaggr_dir, f"preaggr-{base}-{fingerprint[:12]}.sqlite")

        cache = preaggregation.PreAggregationCache(
            db_path,
            gvi_radii=gvi_radii,
            ndvi_radii=ndvi_radii,
            fingerprint=fingerprint,
        )

        if cache.is_complete():
            _log(
                "OK",
                f"Reusing pre-aggregation cache ({n_points:,} entities): {db_path}",
            )
            self._preaggr_cache = cache
            self._preaggregation_done = True
            if progress_callback is not None:
                progress_callback(n_points, n_points)
            return True

        if not cache.matches_fingerprint():
            cache.reset()
        cache.write_header(n_points)

        _log(
            "INFO",
            f"Pre-aggregation: {n_points:,} entities · GVI radii {gvi_radii} m · "
            f"NDVI radii {ndvi_radii} m · stats {preaggregation.STAT_COLUMNS}. "
            f"Cache: {db_path}",
        )

        # ---- Prepare per-channel samplers (format- and geometry-aware) ----
        # Entities of any geometry type produce one cache row per (radius,
        # stat) — the buffered-aggregation routines accept polygons / lines
        # / multi-* / points uniformly via ``geometry.buffer(R)``. The
        # point-only fast path (BallTree / circular raster mask) stays
        # active when every entity is a Point, since it benefits from
        # vectorised distance queries.
        #
        # All distance math happens in a projected CRS picked to minimise
        # planar distortion across the entire study extent (UTM for small
        # extents, Lambert Conformal Conic for larger ones, polar
        # stereographic for high-latitude). Single-UTM-zone math would
        # warp badly for national-scale targets, so we never use
        # ``estimate_utm_crs`` directly here.
        utm_crs, _grid_distortion, _grid_name = select_grid_crs_with_warning(
            preaggr_gdf, _log, role="Pre-aggregation CRS"
        )
        entities_utm = preaggr_gdf.to_crs(utm_crs)
        all_points = bool((entities_utm.geometry.geom_type == "Point").all())
        if all_points:
            point_xy_utm = np.column_stack(
                [
                    entities_utm.geometry.x.to_numpy(),
                    entities_utm.geometry.y.to_numpy(),
                ]
            ).astype(np.float64)
            entity_geoms_utm: list = []
        else:
            point_xy_utm = np.empty((0, 2), dtype=np.float64)
            entity_geoms_utm = list(entities_utm.geometry)
            _log(
                "INFO",
                f"Pre-aggregation: {len(entity_geoms_utm)} non-point entities "
                "→ buffered-geometry aggregation per (radius, stat).",
            )

        # Build per-channel preparation. The point fast path (BallTree /
        # circular raster mask) uses the legacy per-channel dict; the
        # geometry path builds a single ``channels_meta_geom`` list that
        # feeds the unified ``batch_geometry_stats`` (which loops
        # entity-outermost so each buffered geometry is computed once
        # per (entity, effective_radius) and reused across channels).
        prep: dict = {}
        channels_meta_geom: list = []
        for ch, metric in (
            ("veg", self.veg_data),
            ("terrain", self.terrain_data),
            ("ndvi", self.ndvi_data),
        ):
            radii_ch = cache.radii_for(ch)
            if isinstance(metric, dict):  # raster
                if all_points:
                    px_m = preaggregation.raster_pixel_size_m(
                        metric["transform"], metric["crs"].is_geographic
                    )
                    pts_r = preaggr_gdf.to_crs(metric["crs"])
                    xy_r = np.column_stack(
                        [pts_r.geometry.x.to_numpy(), pts_r.geometry.y.to_numpy()]
                    ).astype(np.float64)
                    prep[ch] = (
                        "raster_point",
                        metric["data"],
                        metric["transform"],
                        px_m,
                        xy_r,
                    )
                else:
                    # Raster stays in native CRS; buffer is built in the
                    # grid CRS and reprojected per query for the mask op.
                    if str(metric["crs"]) != str(utm_crs):
                        from pyproj import Transformer as _PyProjTransformer

                        utm_to_raster = _PyProjTransformer.from_crs(
                            utm_crs, metric["crs"], always_xy=True
                        )
                    else:
                        utm_to_raster = None
                    channels_meta_geom.append(
                        {
                            "name": ch,
                            "kind": "raster",
                            "radii": radii_ch,
                            "array": metric["data"],
                            "raster_transform": metric["transform"],
                            "to_raster_crs": utm_to_raster,
                        }
                    )
            else:  # vector points / lines / polygons
                col = self._metric_value_column(metric, defaults[ch])
                if all_points:
                    # Point fast path: BallTree distance queries need a
                    # projected CRS; the metric is reprojected to the
                    # grid CRS only inside the index builder (no other
                    # uses), so the user-facing "keep metric in native
                    # CRS" still holds for the geometry path.
                    tree, vals = preaggregation.build_vector_index(metric, utm_crs, col)
                    prep[ch] = ("vector_point", tree, vals)
                else:
                    # Keep the vector metric in its native CRS. Cell-
                    # buffer detection still needs spacing in metres, so
                    # detect it on a small sample temporarily reprojected
                    # to the grid CRS; the bulk metric is never moved.
                    metric_native = metric[metric[col].notna()].reset_index(drop=True)
                    sample_n = min(len(metric_native), 5000)
                    sample_for_detection = (
                        metric_native.sample(sample_n, random_state=0).to_crs(utm_crs)
                        if sample_n
                        else metric_native
                    )
                    cell_buffer = preaggregation.metric_cell_buffer_m(
                        sample_for_detection
                    )
                    if str(metric_native.crs) != str(utm_crs):
                        from pyproj import Transformer as _PyProjTransformer

                        grid_to_vector = _PyProjTransformer.from_crs(
                            utm_crs, metric_native.crs, always_xy=True
                        )
                    else:
                        grid_to_vector = None
                    _log(
                        "INFO",
                        f"  {ch}: vector metric in native CRS "
                        f"({metric_native.crs}); cell-buffer = "
                        f"{cell_buffer:.1f} m (sqrt(2)/2 × detected grid spacing). "
                        "Buffer is computed in grid CRS and reprojected per query.",
                    )
                    channels_meta_geom.append(
                        {
                            "name": ch,
                            "kind": "vector",
                            "radii": radii_ch,
                            "cell_buffer_m": cell_buffer,
                            "gdf": metric_native,
                            "col": col,
                            "sindex": metric_native.sindex,
                            "values": metric_native[col].to_numpy(dtype=np.float32),
                            "to_metric_crs": grid_to_vector,
                        }
                    )

        # ---- Resume: only compute entities not already written ----
        entity_ids = [int(x) for x in preaggr_gdf.index]
        pos_of = {eid: i for i, eid in enumerate(entity_ids)}
        pending = cache.pending_entities(entity_ids)
        done0 = n_points - len(pending)
        if progress_callback is not None and done0:
            progress_callback(done0, n_points)

        # Geometry-path batches are kept small so the per-batch cancel
        # check fires often (with 290 entities at batch_size=1024 the
        # whole job is one batch and the user can't interrupt mid-run).
        # The point path stays at 1024 because its inner ops are
        # vectorised and the batch cost is sub-second already.
        batch_size = 1024 if all_points else 32
        batches = [
            pending[i : i + batch_size] for i in range(0, len(pending), batch_size)
        ]

        def _compute_batch(bids: list[int]):
            bpos = np.fromiter(
                (pos_of[e] for e in bids), dtype=np.int64, count=len(bids)
            )
            if all_points:
                # Per-channel point fast path (BallTree / circular raster
                # mask). Vectorised distance queries make per-channel
                # processing the natural shape here; no per-entity buffer
                # to share across channels.
                channel_stats: dict[str, np.ndarray] = {}
                for ch in preaggregation.CHANNELS:
                    kind = prep[ch][0]
                    radii = cache.radii_for(ch)
                    if kind == "vector_point":
                        _, tree, vals = prep[ch]
                        channel_stats[ch] = preaggregation.vector_batch_stats(
                            tree, vals, point_xy_utm[bpos], radii
                        )
                    else:  # raster_point
                        _, array, transform, px_m, xy_r = prep[ch]
                        channel_stats[ch] = preaggregation.raster_batch_stats(
                            array, transform, px_m, xy_r[bpos], radii
                        )
                return bids, channel_stats

            # Geometry path: one sweep across all channels with a per-
            # entity buffer cache so shapely.buffer() is called once per
            # (entity, effective_radius) and reused across channels. The
            # batch function checks ``cancel_callback`` per entity and
            # returns ``None`` to abandon the batch early.
            batch_geoms = [entity_geoms_utm[i] for i in bpos]
            channel_stats = preaggregation.batch_geometry_stats(
                batch_geoms,
                channels_meta_geom,
                cancel_check=cancel_callback,
            )
            return bids, channel_stats

        if max_workers is None:
            max_workers = max(1, min((os.cpu_count() or 2), 8))
        max_workers = max(1, min(max_workers, len(batches) or 1))
        _log("INFO", f"Pre-aggregation workers: {max_workers}")

        # Worker threads do the compute (BallTree queries / numpy stats release
        # the GIL and share the read-only metric indices in memory — no pickling,
        # which matters for national-scale rasters); the SQLite writes stay on this
        # thread so there is a single writer and the build remains resumable.
        processed = done0
        cancelled = False

        def _on_done(bids, channel_stats) -> None:
            nonlocal processed, cancelled
            if channel_stats is None:
                # Batch was abandoned mid-run by the cancel-check inside
                # ``batch_geometry_stats``. Don't write; the next resume
                # will pick these entities up because they remain in the
                # cache's pending list.
                cancelled = True
                return
            cache.write_batch(bids, channel_stats)
            processed += len(bids)
            if progress_callback is not None:
                progress_callback(processed, n_points)

        if max_workers == 1:
            for bids in batches:
                if cancel_callback is not None and cancel_callback():
                    cancelled = True
                    break
                _on_done(*_compute_batch(bids))
                if cancelled:
                    break
        else:
            from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                it = iter(batches)
                in_flight = {
                    ex.submit(_compute_batch, b)
                    for b in (
                        next(it, None)
                        for _ in range(min(2 * max_workers, len(batches)))
                    )
                    if b is not None
                }
                # 100 ms poll keeps cancel latency well under a second;
                # combined with the per-entity cancel check inside the
                # batch function the user-visible cancel is sub-second.
                while in_flight:
                    if cancel_callback is not None and cancel_callback():
                        cancelled = True
                        for fut in in_flight:
                            fut.cancel()
                        break
                    done, in_flight = wait(
                        in_flight, timeout=0.1, return_when=FIRST_COMPLETED
                    )
                    for fut in done:
                        _on_done(*fut.result())
                        if cancelled:
                            for other in in_flight:
                                other.cancel()
                            in_flight = set()
                            break
                        nb = next(it, None)
                        if nb is not None:
                            in_flight.add(ex.submit(_compute_batch, nb))

        if cancelled:
            _log("WARN", "Pre-aggregation cancelled (resumable).")
            self._preaggregation_done = False
            cache.close()
            return False

        cache.mark_complete()
        self._preaggr_cache = cache
        self._preaggregation_done = True
        _log("OK", f"Pre-aggregation complete for {n_points:,} entities.")
        return True

    def _precompute_aggregations_longitudinal(
        self,
        progress_callback: Callable[[int, int], None] | None = None,
        cancel_callback: Callable[[], bool] | None = None,
        max_workers: int | None = None,
    ) -> bool:
        """Wave-aware variant of :meth:`precompute_aggregations`.

        For each greenery channel the runner has already loaded a per-wave
        metric source into ``self._longitudinal_metric_data[channel]``. This
        method:

        - Groups waves by file fingerprint so identical-file waves share a
          single compute pass (a static channel that reuses one file across
          all waves only gets computed once).
        - Registers wave aliases on the cache so every aliased wave's lookup
          resolves to the same representative-wave storage.
        - For each ``(channel, representative_wave)`` group, computes stats
          for the entities whose row in the long-format target falls in any
          wave of that group, and writes them to the cache under the
          representative wave index.
        """
        spec = self.longitudinal_spec
        assert spec is not None
        if self.target_gdf is None or len(self.target_gdf) == 0:
            raise ValueError(
                "Pre-aggregation requires sample points; call prepare_fusion_data() "
                "first."
            )
        if self._longitudinal_metric_data is None or not all(
            ch in self._longitudinal_metric_data
            for ch in longitudinal.GREENERY_CHANNELS
        ):
            raise ValueError(
                "Longitudinal pre-aggregation requires per-wave metric data for "
                "every channel; call set_longitudinal_metric_data() first."
            )

        gvi_radii, ndvi_radii = self._preaggr_radii()
        # Unique-pixel source for point/line targets; entity rows otherwise.
        # Each wave's stats are stored per pixel, with wave membership derived
        # from the duplicated rows further below.
        preaggr_gdf = (
            self._preaggr_entity_gdf
            if self._preaggr_entity_gdf is not None
            else self.target_gdf
        )
        n_points = len(preaggr_gdf)
        defaults = {"veg": "veg", "terrain": "terrain", "ndvi": "NDVI"}

        # Per-(channel, wave) source identity contributes to the fingerprint
        # so changing any wave's file invalidates the cache.
        fp_parts: list[str] = [f"geom:{geometry_sha256(preaggr_gdf)}"]
        for ch in longitudinal.GREENERY_CHANNELS:
            for wave_label in spec.wave_labels:
                src = self._longitudinal_metric_data[ch][wave_label]
                fp_parts.append(
                    f"{ch}@{wave_label}:" + self._metric_fingerprint(src, defaults[ch])
                )
        fp_parts.append(f"gvi:{gvi_radii}")
        fp_parts.append(f"ndvi:{ndvi_radii}")
        fp_parts.append(f"stats:{preaggregation.STAT_COLUMNS}")
        fp_parts.append(f"waves:{list(spec.wave_labels)}")
        fp_parts.append(
            f"cgi_grid:{self.cgi_grid_spacing_m if (self.is_polygon_target or self.is_points) else 'na'}"
        )
        fingerprint = hashlib.sha256("|".join(fp_parts).encode()).hexdigest()

        preaggr_dir = os.path.join(self.cache_dir, "preaggr")
        os.makedirs(preaggr_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(self.target_file))[0]
        db_path = os.path.join(preaggr_dir, f"preaggr-{base}-{fingerprint[:12]}.sqlite")

        cache = preaggregation.PreAggregationCache(
            db_path,
            gvi_radii=gvi_radii,
            ndvi_radii=ndvi_radii,
            fingerprint=fingerprint,
            wave_labels=spec.wave_labels,
        )

        if cache.is_complete():
            _log(
                "OK",
                f"Reusing longitudinal pre-aggregation cache "
                f"({n_points:,} rows): {db_path}",
            )
            self._preaggr_cache = cache
            self._preaggregation_done = True
            if progress_callback is not None:
                progress_callback(n_points, n_points)
            return True

        if not cache.matches_fingerprint():
            cache.reset()
        cache.write_header(n_points)

        _log(
            "INFO",
            f"Longitudinal pre-aggregation: {n_points:,} entities · "
            f"{len(spec.wave_labels)} waves · GVI radii {gvi_radii} m · "
            f"NDVI radii {ndvi_radii} m · stats {preaggregation.STAT_COLUMNS}. "
            f"Cache: {db_path}",
        )

        # Group waves by file fingerprint for each channel. Identical-file
        # waves share one compute pass; the first wave in each group is the
        # representative storage slot.
        wave_index_of = {w: i for i, w in enumerate(spec.wave_labels)}

        utm_crs, _grid_distortion_lon, _grid_name_lon = select_grid_crs_with_warning(
            preaggr_gdf, _log, role="Pre-aggregation CRS (longitudinal)"
        )
        entities_utm = preaggr_gdf.to_crs(utm_crs)
        all_points = bool((entities_utm.geometry.geom_type == "Point").all())
        if all_points:
            point_xy_utm = np.column_stack(
                [
                    entities_utm.geometry.x.to_numpy(),
                    entities_utm.geometry.y.to_numpy(),
                ]
            ).astype(np.float64)
            entity_geoms_utm: list = []
        else:
            point_xy_utm = np.empty((0, 2), dtype=np.float64)
            entity_geoms_utm = list(entities_utm.geometry)
        entity_ids_all = np.asarray([int(x) for x in preaggr_gdf.index], dtype=np.int64)
        pos_of = {int(eid): i for i, eid in enumerate(entity_ids_all)}

        # Cache entity ids referenced by each wave. For point/line targets the
        # cache stores one row per unique pixel, so a wave's entities are the
        # unique pixel ids its catchment rows touch; otherwise every row is its
        # own entity and a wave maps directly to its own rows.
        if self._preaggr_entity_gdf is not None:
            tg_waves = self.target_gdf["wave"].astype(str).to_numpy()
            tg_pixels = self.target_gdf["_preaggr_id"].to_numpy()
            wave_to_eids = {
                w: np.unique(tg_pixels[tg_waves == w]) for w in np.unique(tg_waves)
            }
        else:
            waves_by_row = self.target_gdf["wave"].astype(str).to_numpy()
            wave_to_eids = {
                w: entity_ids_all[waves_by_row == w] for w in np.unique(waves_by_row)
            }

        def _eids_for_group(group_waves: list[str]) -> np.ndarray:
            parts = [wave_to_eids[w] for w in group_waves if w in wave_to_eids]
            if not parts:
                return np.empty(0, dtype=np.int64)
            return np.unique(np.concatenate(parts))

        # Walk each channel: build file groups, register aliases, prep + fill
        # one group at a time so memory only ever holds one channel's metric
        # data per group.
        processed = 0
        total_jobs = 0
        plan: list[tuple[str, int, list[str], Any]] = (
            []
        )  # (channel, rep_wave_index, wave_labels_in_group, source)
        for ch in longitudinal.GREENERY_CHANNELS:
            per_wave = self._longitudinal_metric_data[ch]
            groups: dict[str, list[str]] = {}
            for wave_label in spec.wave_labels:
                key = self._metric_fingerprint(per_wave[wave_label], defaults[ch])
                groups.setdefault(key, []).append(wave_label)

            alias_map: dict[int, int] = {}
            for group_waves in groups.values():
                rep_label = group_waves[0]
                rep_idx = wave_index_of[rep_label]
                for w in group_waves:
                    alias_map[wave_index_of[w]] = rep_idx
                plan.append((ch, rep_idx, group_waves, per_wave[rep_label]))
            cache.register_wave_aliases(ch, alias_map)
            _log(
                "INFO",
                f"  {ch}: {len(groups)} unique file(s) across "
                f"{len(spec.wave_labels)} wave(s) — "
                f"{'static' if len(groups) == 1 else 'time-varying'}.",
            )

        # Pre-count the total work for the progress bar.
        for ch, rep_idx, group_waves, _src in plan:
            pending = cache.pending_entities_for(
                ch, rep_idx, _eids_for_group(group_waves).tolist()
            )
            total_jobs += len(pending)
        if progress_callback is not None and total_jobs > 0:
            progress_callback(0, total_jobs)

        cancelled = False
        for ch, rep_idx, group_waves, src in plan:
            if cancel_callback is not None and cancel_callback():
                cancelled = True
                break
            group_eids = _eids_for_group(group_waves).tolist()
            pending = cache.pending_entities_for(ch, rep_idx, group_eids)
            if not pending:
                continue

            radii = cache.radii_for(ch)
            xy_r: np.ndarray = np.empty((0, 2), dtype=np.float64)
            raster_array = transform = None
            px_m = 0.0
            tree = vals = None
            metric_utm = None
            col = ""
            utm_to_raster_lon = None
            if isinstance(src, dict):  # raster source
                px_m = preaggregation.raster_pixel_size_m(
                    src["transform"], src["crs"].is_geographic
                )
                raster_array = src["data"]
                transform = src["transform"]
                if all_points:
                    pts_r = preaggr_gdf.to_crs(src["crs"])
                    xy_r = np.column_stack(
                        [pts_r.geometry.x.to_numpy(), pts_r.geometry.y.to_numpy()]
                    ).astype(np.float64)
                    kind = "raster_point"
                else:
                    if str(src["crs"]) != str(utm_crs):
                        from pyproj import Transformer as _PyProjTransformer

                        utm_to_raster_lon = _PyProjTransformer.from_crs(
                            utm_crs, src["crs"], always_xy=True
                        )
                    kind = "raster_geometry"
            else:
                col = self._metric_value_column(src, defaults[ch])
                if all_points:
                    tree, vals = preaggregation.build_vector_index(src, utm_crs, col)
                    kind = "vector_point"
                else:
                    metric_utm = src.to_crs(utm_crs)
                    metric_utm = metric_utm[metric_utm[col].notna()].reset_index(
                        drop=True
                    )
                    cell_buffer_lon = preaggregation.metric_cell_buffer_m(metric_utm)
                    _log(
                        "INFO",
                        f"  {ch}: vector metric detected — cell-buffer = "
                        f"{cell_buffer_lon:.1f} m (sqrt(2)/2 × detected grid spacing).",
                    )
                    kind = "vector_geometry"

            batch_size = 1024
            batches = [
                pending[i : i + batch_size] for i in range(0, len(pending), batch_size)
            ]

            def _compute_one(bids: list[int]):
                bpos = np.fromiter(
                    (pos_of[int(e)] for e in bids), dtype=np.int64, count=len(bids)
                )
                if kind == "vector_point":
                    stats = preaggregation.vector_batch_stats(
                        tree, vals, point_xy_utm[bpos], radii
                    )
                elif kind == "raster_point":
                    stats = preaggregation.raster_batch_stats(
                        raster_array, transform, px_m, xy_r[bpos], radii
                    )
                elif kind == "vector_geometry":
                    batch_geoms = [entity_geoms_utm[i] for i in bpos]
                    stats = preaggregation.vector_batch_geometry_stats(
                        metric_utm,
                        col,
                        batch_geoms,
                        radii,
                        cell_buffer_m=cell_buffer_lon,
                    )
                else:  # raster_geometry
                    batch_geoms = [entity_geoms_utm[i] for i in bpos]
                    stats = preaggregation.raster_batch_geometry_stats(
                        raster_array,
                        transform,
                        batch_geoms,
                        radii,
                        to_raster_crs=utm_to_raster_lon,
                    )
                return bids, stats

            workers = (
                max(1, min((os.cpu_count() or 2), 8))
                if max_workers is None
                else max_workers
            )
            workers = max(1, min(workers, len(batches) or 1))

            def _on_done(bids: list[int], stats: np.ndarray) -> None:
                nonlocal processed
                cache.write_channel_wave_batch(ch, rep_idx, bids, stats)
                processed += len(bids)
                if progress_callback is not None and total_jobs > 0:
                    progress_callback(processed, total_jobs)

            if workers == 1:
                for bids in batches:
                    if cancel_callback is not None and cancel_callback():
                        cancelled = True
                        break
                    _on_done(*_compute_one(bids))
            else:
                from concurrent.futures import (
                    FIRST_COMPLETED,
                    ThreadPoolExecutor,
                    wait,
                )

                with ThreadPoolExecutor(max_workers=workers) as ex:
                    it = iter(batches)
                    in_flight = {
                        ex.submit(_compute_one, b)
                        for b in (
                            next(it, None)
                            for _ in range(min(2 * workers, len(batches)))
                        )
                        if b is not None
                    }
                    while in_flight:
                        if cancel_callback is not None and cancel_callback():
                            cancelled = True
                            for fut in in_flight:
                                fut.cancel()
                            break
                        done, in_flight = wait(
                            in_flight, timeout=0.5, return_when=FIRST_COMPLETED
                        )
                        for fut in done:
                            _on_done(*fut.result())
                            nb = next(it, None)
                            if nb is not None:
                                in_flight.add(ex.submit(_compute_one, nb))
            if cancelled:
                break

        if cancelled:
            _log("WARN", "Longitudinal pre-aggregation cancelled (resumable).")
            self._preaggregation_done = False
            cache.close()
            return False

        cache.mark_complete()
        self._preaggr_cache = cache
        self._preaggregation_done = True
        _log(
            "OK",
            f"Longitudinal pre-aggregation complete for {n_points:,} entities.",
        )
        return True

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
        cache = self._preaggr_cache
        if cache is None or not self._preaggregation_done:
            return None
        column = preaggregation.stat_to_column(stat, percentile)
        if column is None:
            return None
        ids = np.asarray(point_indices)
        radius = int(round(radius_m))
        if wave_indices is None:
            return cache.lookup(ids, channel, radius, column)

        waves = np.asarray(wave_indices)
        out = np.full(ids.shape[0], np.nan, dtype=np.float32)
        # Per-wave grouped reads: each wave maps to one cache column, so
        # repeated entity reads within a wave benefit from the cache's
        # column LRU.
        for w in np.unique(waves):
            mask = waves == w
            sub = cache.lookup(ids[mask], channel, radius, column, wave_index=int(w))
            if sub is None:
                # An off-grid (channel, wave, radius, column) cell — caller
                # falls back to the ring/buffer aggregation path.
                return None
            out[mask] = sub
        return out

    # ------------------------------------------------------------------
    # Polygon-target areal aggregation
    # ------------------------------------------------------------------

    def _reference_points_in_polygon(
        self,
        polygon_geom,
        polygon_crs,
        metric_data,
    ) -> "gpd.GeoDataFrame":
        """Return the in-polygon sample-point locations contributed by one metric.

        For raster metrics: pixel centers (as Points) that fall inside the
        polygon. For vector metrics: each feature whose geometry is strictly
        inside the polygon, represented by its centroid (always a Point).
        The returned GeoDataFrame's CRS matches ``polygon_crs``.
        """
        from rasterio.transform import xy as _xy
        from shapely.geometry import Point

        if isinstance(metric_data, dict):  # raster
            raster_crs = metric_data["crs"]
            if polygon_crs is not None and str(polygon_crs) != str(raster_crs):
                poly_r = (
                    gpd.GeoSeries([polygon_geom], crs=polygon_crs)
                    .to_crs(raster_crs)
                    .iloc[0]
                )
            else:
                poly_r = polygon_geom

            transform = metric_data["transform"]
            height, width = metric_data["data"].shape
            minx, miny, maxx, maxy = poly_r.bounds
            from rasterio.transform import rowcol

            r1, c1 = rowcol(transform, minx, maxy)
            r2, c2 = rowcol(transform, maxx, miny)
            r_start = max(0, min(r1, r2))
            r_end = min(height, max(r1, r2) + 1)
            c_start = max(0, min(c1, c2))
            c_end = min(width, max(c1, c2) + 1)
            if r_end <= r_start or c_end <= c_start:
                return gpd.GeoDataFrame(geometry=[], crs=polygon_crs)

            rows, cols = np.meshgrid(
                np.arange(r_start, r_end),
                np.arange(c_start, c_end),
                indexing="ij",
            )
            xs, ys = _xy(transform, rows.flatten(), cols.flatten(), offset="center")
            cand = gpd.GeoDataFrame(
                geometry=[Point(x, y) for x, y in zip(xs, ys)],
                crs=raster_crs,
            )
            inside = cand[cand.geometry.within(poly_r)]
            if len(inside) and polygon_crs is not None:
                inside = inside.to_crs(polygon_crs)
            return inside.reset_index(drop=True)

        # Vector metric data
        m = metric_data
        if (
            m.crs is not None
            and polygon_crs is not None
            and str(m.crs) != str(polygon_crs)
        ):
            m = m.to_crs(polygon_crs)
        inside = m[m.geometry.within(polygon_geom)]
        if len(inside) == 0:
            return gpd.GeoDataFrame(geometry=[], crs=polygon_crs)
        # Reduce non-Point geometries to centroids
        geoms = inside.geometry.map(
            lambda g: g if g.geom_type == "Point" else g.centroid
        )
        return gpd.GeoDataFrame(geometry=list(geoms), crs=polygon_crs).reset_index(
            drop=True
        )

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

        # Generate a regular grid of pixel centroids covering the union
        # bounding box, then keep only pixels whose centroid lies inside
        # the union.
        xs = np.arange(minx + spacing_m / 2.0, maxx, spacing_m, dtype=np.float64)
        ys = np.arange(miny + spacing_m / 2.0, maxy, spacing_m, dtype=np.float64)
        if xs.size == 0 or ys.size == 0:
            raise ValueError(
                f"CGI grid spacing {spacing_m:g} m is larger than the target "
                f"extent ({maxx - minx:.0f} m × {maxy - miny:.0f} m)."
            )
        # ys are top-to-bottom in raster row order so the composite TIFF
        # writer can index pixels back to (row, col) without flipping.
        ys_topdown = ys[::-1]
        gx, gy = np.meshgrid(xs, ys_topdown, indexing="xy")
        pixel_geoms = gpd.points_from_xy(gx.ravel(), gy.ravel())
        pixels = gpd.GeoDataFrame(geometry=pixel_geoms, crs=grid_crs)
        # Stash the (row, col) of each candidate pixel for the composite-map
        # writer. After sjoin + dedup the surviving subset keeps these tags.
        n_cols = len(xs)
        n_rows = len(ys_topdown)
        pixels["_grid_row"] = np.repeat(np.arange(n_rows, dtype=np.int64), n_cols)
        pixels["_grid_col"] = np.tile(np.arange(n_cols, dtype=np.int64), n_rows)
        _log(
            "INFO",
            f"CGI grid bbox: {len(xs)} × {len(ys)} = {len(pixels):,} candidate "
            "pixels before union mask.",
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
        # gate; per-trial channel values come from the cache.
        _log(
            "INFO",
            "Probing initial veg / terrain / NDVI coverage at pixel centroids "
            "(used only for NaN-filtering and feature normalisation)...",
        )
        probe_gdf = entity_gdf[["geometry"]].copy()
        probe_gdf = self._sample_metrics_at_points(probe_gdf)

        fusion_df = pd.DataFrame(
            {
                "polygon_id": entity_gdf["polygon_id"].values,
                "target": entity_gdf["target"].values,
                "veg": probe_gdf["veg"].values,
                "terrain": probe_gdf["terrain"].values,
                "ndvi": probe_gdf["ndvi"].values,
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

        # Local raster indices span only the populated cells' global extent.
        col_origin = int(g_col.min())
        row_origin = int(g_row.min())
        n_cols = int(g_col.max()) - col_origin + 1
        n_rows = int(g_row.max()) - row_origin + 1
        minx = col_origin * spacing_m
        maxy = -row_origin * spacing_m
        pixels = gpd.GeoDataFrame(
            {
                "_grid_row": (g_row - row_origin).astype(np.int64),
                "_grid_col": (g_col - col_origin).astype(np.int64),
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
            catchment_dist = sdm.data.astype(np.float64)
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
            catchment_dist = pix_geoms.distance(aligned_geoms).to_numpy()

        if pix_idx.size == 0:
            raise ValueError(
                "Entity catchment grid produced zero pixels — the grid spacing "
                "may exceed the buffer extent."
            )

        # Per-unique-pixel columns, indexed onto the duplicated catchment rows.
        pixel_geom = pixels.geometry.to_numpy()
        pixel_row = pixels["_grid_row"].to_numpy()
        pixel_col = pixels["_grid_col"].to_numpy()

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

        # Coverage probe — sample each unique pixel once (used only for NaN-
        # filtering; per-trial channel values come from the cache), then map
        # the per-pixel values onto the duplicated catchment rows.
        _log(
            "INFO",
            f"Probing veg / terrain / NDVI coverage at {len(pixels):,} unique "
            "pixel centroids...",
        )
        probe_unique = self._sample_metrics_at_points(pixels[["geometry"]].copy())
        veg_u = probe_unique["veg"].to_numpy()
        terrain_u = probe_unique["terrain"].to_numpy()
        ndvi_u = probe_unique["ndvi"].to_numpy()

        fusion_df = pd.DataFrame(
            {
                "polygon_id": entity_gdf["polygon_id"].values,
                "target": entity_gdf["target"].values,
                "veg": veg_u[pix_idx],
                "terrain": terrain_u[pix_idx],
                "ndvi": ndvi_u[pix_idx],
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
        # (~1/16th of the duplicated catchment rows for dense point targets),
        # and scoring rows resolve their stats through ``_preaggr_id``.
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

    # ------------------------------------------------------------------
    # Per-pixel CGI → per-entity collapse (shared by every scoring path)
    # ------------------------------------------------------------------
    def _catchment_radius(
        self, veg_radius: float, terrain_radius: float, ndvi_radius: float
    ) -> float:
        """Per-trial catchment radius for the point/line collapse.

        The active channel's radius for a standalone study (only that channel
        contributes to the score), or the largest of the three for the
        combined CGI run (every channel feeds the per-pixel composite).
        """
        ch = getattr(self, "_active_greenery_channel", "cgi") or "cgi"
        if ch == "veg":
            return float(veg_radius)
        if ch == "terrain":
            return float(terrain_radius)
        if ch == "ndvi":
            return float(ndvi_radius)
        return float(max(veg_radius, terrain_radius, ndvi_radius))

    def _entity_collapse_mask(
        self, data: pd.DataFrame, catchment_radius: float | None
    ) -> np.ndarray | None:
        """Row mask for the per-trial catchment collapse, or ``None``.

        Returns ``None`` for polygon / raster targets (no radius mask — every
        in-footprint pixel contributes, the original behavior). For point /
        line targets, keeps pixels within ``catchment_radius`` of their
        entity plus each entity's nearest pixel, so no entity drops out at
        small radii.
        """
        if "_catchment_dist" not in data.columns or catchment_radius is None:
            return None
        dist = data["_catchment_dist"].to_numpy(dtype=np.float64)
        if "_is_nearest" in data.columns:
            near = data["_is_nearest"].to_numpy(dtype=bool)
        else:
            near = np.zeros(len(data), dtype=bool)
        return (dist <= float(catchment_radius)) | near

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
            _log("WARN", f"Could not attach entity coordinates for spatial adjustment: {exc}")
            return fusion_df
        fusion_df = fusion_df.copy()
        fusion_df["_cx"] = cx
        fusion_df["_cy"] = cy
        return fusion_df

    @staticmethod
    def _spatial_cache_key(
        coords_xy: np.ndarray, target: np.ndarray, cov: np.ndarray | None
    ) -> tuple:
        """Content fingerprint of a scored split for memoizing its spatial basis.

        Keyed on the coordinates and outcome (both stable across Optuna trials)
        so studies sharing a fold reuse the basis, while bootstrap resamples —
        which change row membership — get a fresh one without manual cache
        invalidation.
        """
        t = np.asarray(target, dtype=np.float64)
        cs = float(0.0 if cov is None else np.nansum(np.asarray(cov, dtype=np.float64)))
        return (
            int(coords_xy.shape[0]),
            float(np.nansum(coords_xy[:, 0])),
            float(np.nansum(coords_xy[:, 1])),
            float(np.nansum(coords_xy[:, 0] ** 2)),
            float(np.nansum(t)),
            float(np.nansum(t**2)),
            cs,
        )

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
        cache_key = self._spatial_cache_key(coords_xy, target, cov)
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
        # selected on the exposure (df-Spatial+); the candidate basis is reused.
        _df, sp_cols = spatial_basis.select_df_aic(
            np.asarray(cgi, dtype=np.float64), basis, None
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
        arr = np.asarray(composite, dtype=np.float64)
        valid = ~np.isnan(arr)
        if np.any(valid):
            lo = float(arr[valid].min())
            hi = float(arr[valid].max())
            if hi > lo:
                arr = arr.copy()
                arr[valid] = (arr[valid] - lo) / (hi - lo)
        return arr

    def _prepare_point_fusion(self) -> pd.DataFrame:
        """Sample metrics at point locations."""
        _log("INFO", "====== POINT FUSION ======")
        _log("INFO", "Preparing point-based fusion data...")

        # Only use target points as samples, not buffered area
        points_gdf = self.target_gdf.copy()

        if len(points_gdf) == 0:
            raise ValueError(
                f"No sample points found in target file. "
                f"The target GeoJSON must contain point geometries, not just boundaries. "
                f"Loaded {len(points_gdf)} points from {self.target_file}"
            )

        logger.info(f"Using {len(points_gdf)} points from target file as samples")

        # Sample metrics at these point locations
        points_gdf = self._sample_metrics_at_points(points_gdf)

        # Create final DataFrame
        fusion_df = pd.DataFrame(
            {
                "target": points_gdf[self.target_feature],
                "veg": points_gdf["veg"],
                "terrain": points_gdf["terrain"],
                "ndvi": points_gdf["ndvi"],
            }
        )
        # Carry covariate values onto each row alongside the target so the
        # split / objective / evaluate paths all see them as plain columns.
        for col in self.covariate_columns:
            fusion_df[col] = points_gdf[col].to_numpy()
        # Longitudinal mode adds the (entity_id, wave) keys and the time
        # predictor; downstream split + MixedLM scorer key on these.
        for col in self._longitudinal_extra_cols():
            fusion_df[col] = points_gdf[col].to_numpy()
        fusion_df = self._attach_entity_coords(fusion_df, points_gdf)

        # Log data quality before dropping NaN
        _log("INFO", "====== DATA QUALITY SUMMARY (POINT) ======")
        _log("INFO", f"Total rows: {len(fusion_df)}")
        for col in ["target", "veg", "terrain", "ndvi", *self.covariate_columns]:
            nan_count = fusion_df[col].isna().sum()
            pct = nan_count / max(len(fusion_df), 1) * 100
            _log("INFO", f"{col} NaN: {nan_count} ({pct:.1f}%)")

        result = fusion_df.dropna(
            subset=["target", "veg", "terrain", "ndvi", *self.covariate_columns]
        )
        _log(
            "OK" if len(result) else "WARN",
            f"After dropna: {len(result)} valid rows",
        )

        if len(result) == 0:
            raise ValueError(
                "No valid samples after spatial join. "
                "This usually means the target points and metric data don't overlap spatially. "
                "Check that all data covers the same geographic area."
            )

        return result

    def _sample_metrics_at_points(
        self, points_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Sample vegetation, terrain, and NDVI metrics at point locations.

        This method is shared by both point and raster fusion workflows.
        For rasters, the pixel centers are converted to points first.

        Args:
            points_gdf: GeoDataFrame with point geometries to sample at

        Returns:
            GeoDataFrame with added columns: veg, terrain, ndvi
        """
        points_gdf["veg"] = np.nan
        points_gdf["terrain"] = np.nan
        points_gdf["ndvi"] = np.nan

        # Sample Vegetation
        if isinstance(self.veg_data, dict):  # Raster
            points_in_veg_crs = points_gdf.to_crs(self.veg_data["crs"])
            points_gdf["veg"] = _sample_raster_values(points_in_veg_crs, self.veg_data)
        else:  # GeoDataFrame
            # Detect metric column
            veg_col = self.veg_data.attrs.get("metric_column", "gvi")
            if veg_col not in self.veg_data.columns:
                # Fallback to common patterns
                for col in ["gvi", "gvi_veg", "veg", "GVI", "value"]:
                    if col in self.veg_data.columns:
                        veg_col = col
                        break

            _log(
                "INFO",
                f"Veg data: {len(self.veg_data)} features, "
                f"columns: {self.veg_data.columns.tolist()}",
            )
            _log("INFO", f"Using veg column: '{veg_col}'")
            _log(
                "INFO",
                f"Nearest-feature max distance (veg): {self.gvi_buffer_max_m} m",
            )

            points_gdf["veg"] = _nearest_metric_join(
                points_gdf, self.veg_data, veg_col, self.gvi_buffer_max_m
            )

            veg_valid = points_gdf["veg"].notna().sum()
            _log(
                "OK" if veg_valid else "WARN",
                f"Veg sampling: {veg_valid}/{len(points_gdf)} points have valid values",
            )

        # Sample Terrain
        if isinstance(self.terrain_data, dict):  # Raster
            points_in_terrain_crs = points_gdf.to_crs(self.terrain_data["crs"])
            points_gdf["terrain"] = _sample_raster_values(
                points_in_terrain_crs, self.terrain_data
            )
        else:  # GeoDataFrame
            # Detect metric column
            terrain_col = self.terrain_data.attrs.get("metric_column", "terrain")
            if terrain_col not in self.terrain_data.columns:
                # Fallback to common patterns
                for col in ["terrain", "gvi_ter", "NDVI", "ndvi", "value"]:
                    if col in self.terrain_data.columns:
                        terrain_col = col
                        break

            _log(
                "INFO",
                f"Terrain data: {len(self.terrain_data)} features, "
                f"columns: {self.terrain_data.columns.tolist()}",
            )
            _log("INFO", f"Using terrain column: '{terrain_col}'")

            points_gdf["terrain"] = _nearest_metric_join(
                points_gdf, self.terrain_data, terrain_col, self.gvi_buffer_max_m
            )

            terrain_valid = points_gdf["terrain"].notna().sum()
            _log(
                "OK" if terrain_valid else "WARN",
                f"Terrain sampling: {terrain_valid}/{len(points_gdf)} "
                "points have valid values",
            )

        # Sample NDVI
        points_gdf["ndvi"] = np.nan
        if isinstance(self.ndvi_data, dict):  # Raster
            points_in_ndvi_crs = points_gdf.to_crs(self.ndvi_data["crs"])
            points_gdf["ndvi"] = _sample_raster_values(
                points_in_ndvi_crs, self.ndvi_data
            )
        else:  # GeoDataFrame
            # Detect metric column
            ndvi_col = self.ndvi_data.attrs.get("metric_column", "NDVI")
            if ndvi_col not in self.ndvi_data.columns:
                # Fallback to common patterns
                for col in ["NDVI", "ndvi", "value"]:
                    if col in self.ndvi_data.columns:
                        ndvi_col = col
                        break

            _log(
                "INFO",
                f"NDVI data: {len(self.ndvi_data)} features, "
                f"columns: {self.ndvi_data.columns.tolist()}",
            )
            _log("INFO", f"Using NDVI column: '{ndvi_col}'")

            points_gdf["ndvi"] = _nearest_metric_join(
                points_gdf, self.ndvi_data, ndvi_col, self.ndvi_buffer_max_m
            )

            ndvi_valid = points_gdf["ndvi"].notna().sum()
            _log(
                "OK" if ndvi_valid else "WARN",
                f"NDVI sampling: {ndvi_valid}/{len(points_gdf)} points have valid values",
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
        height, width = target_data.shape

        # Generate pixel centers as point geometries
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        xs, ys = xy(transform, rows.flatten(), cols.flatten(), offset="center")

        # Create GeoDataFrame of pixel centers
        pixel_points = gpd.GeoDataFrame(
            {
                "target": target_data.flatten(),
                "row": rows.flatten(),
                "col": cols.flatten(),
            },
            geometry=gpd.points_from_xy(xs, ys),
            crs=crs,
        )

        # Filter out NaN target values
        pixel_points = pixel_points[~pixel_points["target"].isna()].copy()

        if len(pixel_points) == 0:
            raise ValueError(
                "No valid pixels found in target raster. "
                "Target raster may be empty or all NaN."
            )

        logger.info(f"Created {len(pixel_points):,} point samples from raster pixels")
        print(
            f"[FUSION DEBUG] Created {len(pixel_points):,} point samples from raster pixels",
            flush=True,
        )

        # Store as target_gdf for unified workflow
        self.target_gdf = pixel_points

        # Now sample metrics at these point locations using spatial joins
        # This is the same logic as _prepare_point_fusion()
        print(
            "[FUSION DEBUG] Now sampling metrics at pixel center points...", flush=True
        )

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

    @staticmethod
    def _area_balanced_split_within_bin(
        bin_df: pd.DataFrame,
        area_col: str,
        target_fractions: dict[str, float],
        random_state: int,
    ) -> dict[str, list]:
        """Allocate one outcome bin's polygons to splits with greedy area balance.

        Shuffles the polygons inside the bin, then for each one picks the
        split with the largest remaining area deficit (in fraction-of-bin
        units). Returns a dict mapping split name to a list of polygon
        row indices.
        """
        if len(bin_df) == 0:
            return {name: [] for name in target_fractions}
        total_area = float(bin_df[area_col].sum())
        if total_area <= 0:
            # Fall back to count-balanced when areas are missing.
            rng = np.random.default_rng(random_state)
            shuffled = rng.permutation(bin_df.index.to_numpy())
            n = len(shuffled)
            order = list(target_fractions)
            counts = {name: int(round(target_fractions[name] * n)) for name in order}
            # Reconcile rounding to total n.
            diff = n - sum(counts.values())
            counts[order[0]] += diff
            out: dict[str, list] = {name: [] for name in order}
            i = 0
            for name in order:
                out[name] = list(shuffled[i : i + counts[name]])
                i += counts[name]
            return out

        rng = np.random.default_rng(random_state)
        shuffled = rng.permutation(bin_df.index.to_numpy())
        target_area = {
            name: target_fractions[name] * total_area for name in target_fractions
        }
        used_area = dict.fromkeys(target_fractions, 0.0)
        assignments: dict[str, list] = {name: [] for name in target_fractions}
        for idx in shuffled:
            area = float(bin_df.at[idx, area_col])
            deficits = {
                name: target_area[name] - used_area[name] for name in target_fractions
            }
            choice = max(deficits, key=deficits.get)
            assignments[choice].append(idx)
            used_area[choice] += area
        return assignments

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
            alloc = self._area_balanced_split_within_bin(
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
            geometry=gpd.points_from_xy(
                agg["x"].to_numpy(), agg["y"].to_numpy()
            ),
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

    @staticmethod
    def _stripe_to_test_blocks(
        ordered_blocks: np.ndarray, test_fraction: float, rng: np.random.Generator
    ) -> set:
        """Pick ~``test_fraction`` of blocks, spread evenly across their order.

        ``ordered_blocks`` is the space-filling block sequence; selecting evenly
        spaced positions (with a seed-jittered start) spreads the held-out
        blocks across the full extent rather than into one contiguous corner.
        """
        n = len(ordered_blocks)
        if n == 0:
            return set()
        n_pick = max(1, int(round(test_fraction * n)))
        n_pick = min(n_pick, n)
        step = n / n_pick
        start = int(rng.integers(0, max(1, int(np.floor(step)))))
        idxs = (np.floor(np.arange(n_pick) * step).astype(int) + start) % n
        return set(ordered_blocks[np.unique(idxs)].tolist())

    def split_data(
        self,
        test_size: float = 0.2,
        k_folds: int = 5,
        random_state: int = 42,
        fusion_df: pd.DataFrame | None = None,
        single_split_val_ratio: float = 0.2,
        *,
        outer_fold_idx: int | None = None,
        n_outer_folds: int | None = None,
        spatial_split: bool = False,
        spatial_block_size_m: float | None = None,
        n_spatial_blocks: int | None = None,
    ) -> None:
        """Carve a held-out test set, then build k inner CV folds (or one
        single split) over the remainder.

        Nested cross-validation: when ``outer_fold_idx`` and ``n_outer_folds``
        are both set, the held-out test set is the ``outer_fold_idx``-th
        partition of a stratified ``n_outer_folds``-fold split over the data
        (group-aware when ``entity_id`` / ``polygon_id`` is present). Every
        entity appears in test exactly once across the K calls, so the runner
        can build an outer-CV-averaged prediction column with no leakage. When
        the two args are ``None`` the original ``test_size``-based stratified
        split is used (back-compat path for single-test-split runs).

        Args:
            test_size: Held-out fraction when ``outer_fold_idx`` is unset.
                Ignored in nested-CV mode (the outer fold sets the test size
                to ``≈ 1/n_outer_folds``).
            k_folds: Inner CV folds within train+val. ``1`` (or ``0``) =
                single stratified train/val split (``self.cv_folds`` becomes
                length-1 and each trial fits one model).
            random_state: RNG seed for reproducibility. In nested-CV mode
                this must be held constant across all K outer-fold calls so
                the K test partitions form a proper non-overlapping K-fold
                partition over the data. Inner CV partitions still differ
                per outer fold because they operate on a different train+val
                subset each call.
            single_split_val_ratio: When ``k_folds <= 1``, fraction of the
                non-test subset used as validation in the single fit.
            outer_fold_idx: 0-based index of the outer fold whose held-out
                partition becomes the test set. Must be set together with
                ``n_outer_folds``.
            n_outer_folds: Total outer folds (typically 5-10). When set, the
                test set is derived from a stratified K-fold partition; when
                unset, the legacy ``test_size`` random split is used.
        """
        # Step 1: Sample all metrics at initial buffer distance
        logger.info("Step 1/4: Sampling metrics at point locations...")
        if fusion_df is None:
            fusion_df = self.prepare_fusion_data()
        use_cv = k_folds > 1
        self.k_folds = k_folds if use_cv else 1

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
            # ── Group-level stratified split ──────────────────────────────
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
            n_trainval_target = n_polys - n_test_target

            # Bin count constrained by: at least 2 groups per bin so each
            # fold gets ≥1 group per class, and ≤ n_test_target so the test
            # split can include every class.
            max_bins_for_test = max(2, n_test_target)
            max_bins_for_kfold = max(2, n_trainval_target // max(k_folds, 1))
            requested_bins = max(2, self.n_bins)
            n_bins_eff = min(
                requested_bins,
                max_bins_for_test,
                max_bins_for_kfold,
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

            # Nested-CV mode: the test set is one slice of a stratified
            # K-fold partition over groups. Every group appears in test
            # exactly once across the K outer-fold calls.
            outer_cv_mode = outer_fold_idx is not None and n_outer_folds is not None

            # Optional spatial blocking: tag each group with a coarse grid
            # block so whole blocks — never split groups — can be striped
            # across the held-out test and inner folds, spreading them over
            # the full extent while keeping a block's catchments clear of its
            # neighbours.
            spatial_ok = False
            self._spatial_block_by_group = None
            self._spatial_block_group_col = None
            if spatial_split and not outer_cv_mode:
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

            if outer_cv_mode:
                from sklearn.model_selection import KFold

                assert n_outer_folds is not None and outer_fold_idx is not None
                k_outer = max(2, int(n_outer_folds))
                fold_i = max(0, min(int(outer_fold_idx), k_outer - 1))
                if stratifiable and (
                    poly_df["target_bin"].value_counts().min() >= k_outer
                ):
                    outer_splitter = StratifiedKFold(
                        n_splits=k_outer, shuffle=True, random_state=random_state
                    )
                    outer_iter = list(
                        outer_splitter.split(poly_df, poly_df["target_bin"])
                    )
                else:
                    outer_splitter = KFold(
                        n_splits=k_outer, shuffle=True, random_state=random_state
                    )
                    outer_iter = list(outer_splitter.split(poly_df))
                tv_idx, te_idx = outer_iter[fold_i]
                train_val_poly = poly_df.iloc[tv_idx].copy()
                test_poly = poly_df.iloc[te_idx].copy()
                _log(
                    "INFO",
                    f"Outer fold {fold_i + 1}/{k_outer}: "
                    f"train+val={len(train_val_poly)} {group_label}s, "
                    f"test={len(test_poly)} {group_label}s.",
                )
            elif spatial_ok:
                # Spatial single-split: assign whole blocks to the held-out
                # test by even striping across the space-filling block order,
                # so the test tiles the full extent and no group straddles
                # train and test.
                rng_sp = np.random.default_rng(random_state)
                ordered_blocks = np.sort(poly_df["_block"].unique())
                test_blocks = self._stripe_to_test_blocks(
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

            self.cv_folds = []
            self._spatial_basis_cache = {}
            self._spatial_adjust_summary = None
            if use_cv and spatial_ok and "_block" in train_val_poly.columns:
                # Spatial inner folds: stripe blocks across folds so each
                # validation fold is spread over the extent and a block's
                # groups never split across train and val.
                ordered_tv_blocks = np.sort(train_val_poly["_block"].unique())
                k_eff = max(2, min(k_folds, len(ordered_tv_blocks)))
                self.k_folds = k_eff
                rank_of_block = {b: i for i, b in enumerate(ordered_tv_blocks)}
                fold_of_block = {b: rank_of_block[b] % k_eff for b in ordered_tv_blocks}
                for fold_idx in range(1, k_eff + 1):
                    vl_blocks = {b for b, f in fold_of_block.items() if f == fold_idx - 1}
                    vl_polys = set(
                        train_val_poly[train_val_poly["_block"].isin(vl_blocks)][
                            group_col
                        ]
                    )
                    tr_polys = set(train_val_poly[group_col]) - vl_polys
                    train_fold = self.train_val_data[
                        self.train_val_data[group_col].isin(tr_polys)
                    ].copy()
                    val_fold = self.train_val_data[
                        self.train_val_data[group_col].isin(vl_polys)
                    ].copy()
                    self.cv_folds.append({"train": train_fold, "val": val_fold})
                    logger.info(
                        f"  Spatial fold {fold_idx}: train={len(tr_polys)} "
                        f"{group_label}s ({len(train_fold)} rows), "
                        f"val={len(vl_polys)} {group_label}s ({len(val_fold)} rows)"
                    )
                return
            if use_cv:
                # K-fold within train_val. Use stratified k-fold only if every bin
                # has ≥k_folds groups; otherwise fall back to plain KFold.
                from sklearn.model_selection import KFold

                min_per_bin = (
                    train_val_poly["target_bin"].value_counts().min()
                    if stratifiable
                    else 0
                )
                k_eff = max(2, min(k_folds, len(train_val_poly)))
                if stratifiable and min_per_bin < k_folds:
                    k_eff = max(2, min(k_folds, min_per_bin))
                    _log(
                        "WARN",
                        f"Requested {k_folds}-fold CV but smallest outcome bin has "
                        f"{min_per_bin} {group_label}s; reducing to {k_eff}-fold.",
                    )
                self.k_folds = k_eff

                if stratifiable and min_per_bin >= k_eff:
                    splitter = StratifiedKFold(
                        n_splits=k_eff, shuffle=True, random_state=random_state
                    )
                    split_iter = splitter.split(
                        train_val_poly, train_val_poly["target_bin"]
                    )
                else:
                    splitter = KFold(
                        n_splits=k_eff, shuffle=True, random_state=random_state
                    )
                    split_iter = splitter.split(train_val_poly)

                for fold_idx, (tr_idx, vl_idx) in enumerate(split_iter, 1):
                    tr_polys = set(train_val_poly.iloc[tr_idx][group_col])
                    vl_polys = set(train_val_poly.iloc[vl_idx][group_col])
                    train_fold = self.train_val_data[
                        self.train_val_data[group_col].isin(tr_polys)
                    ].copy()
                    val_fold = self.train_val_data[
                        self.train_val_data[group_col].isin(vl_polys)
                    ].copy()

                    # Per-channel scaling removed — folds keep raw channels;
                    # the composite is normalized (when enabled) at scoring.
                    self.cv_folds.append({"train": train_fold, "val": val_fold})
                    logger.info(
                        f"  Fold {fold_idx}: train={len(tr_polys)} polys "
                        f"({len(train_fold)} rows), val={len(vl_polys)} polys "
                        f"({len(val_fold)} rows)"
                    )
                return

            # Single stratified train/val split at the group level. Same
            # stratification logic as the test split above so val mirrors the
            # outcome distribution; scaler fits on the single train slice.
            if spatial_ok and "_block" in train_val_poly.columns:
                rng_v = np.random.default_rng(random_state + 1)
                ordered_tv_blocks = np.sort(train_val_poly["_block"].unique())
                val_blocks = self._stripe_to_test_blocks(
                    ordered_tv_blocks, single_split_val_ratio, rng_v
                )
                vl_polys = set(
                    train_val_poly[train_val_poly["_block"].isin(val_blocks)][group_col]
                )
                tr_polys = set(train_val_poly[group_col]) - vl_polys
                train_fold = self.train_val_data[
                    self.train_val_data[group_col].isin(tr_polys)
                ].copy()
                val_fold = self.train_val_data[
                    self.train_val_data[group_col].isin(vl_polys)
                ].copy()
                self.cv_folds.append({"train": train_fold, "val": val_fold})
                logger.info(
                    f"Spatial single train/val split: train={len(tr_polys)} "
                    f"{group_label}s ({len(train_fold)} rows), val={len(vl_polys)} "
                    f"{group_label}s ({len(val_fold)} rows)"
                )
                return
            try:
                train_poly, val_poly = train_test_split(
                    train_val_poly,
                    test_size=single_split_val_ratio,
                    stratify=(train_val_poly["target_bin"] if stratifiable else None),
                    random_state=random_state,
                )
            except ValueError:
                train_poly, val_poly = train_test_split(
                    train_val_poly,
                    test_size=single_split_val_ratio,
                    random_state=random_state,
                )
            tr_polys = set(train_poly[group_col])
            vl_polys = set(val_poly[group_col])
            train_fold = self.train_val_data[
                self.train_val_data[group_col].isin(tr_polys)
            ].copy()
            val_fold = self.train_val_data[
                self.train_val_data[group_col].isin(vl_polys)
            ].copy()
            self.cv_folds.append({"train": train_fold, "val": val_fold})
            logger.info(
                f"Single train/val split (no CV): train={len(tr_polys)} polys "
                f"({len(train_fold)} rows), val={len(vl_polys)} polys "
                f"({len(val_fold)} rows)"
            )
            return

        # ── Row-level (point / raster) split ────────────────────────────────
        # Step 3: Create stratification bins based on target values
        logger.info("Step 3/4: Binning target values for stratified sampling...")
        fusion_df["target_bin"] = pd.qcut(
            fusion_df["target"], q=self.n_bins, labels=False, duplicates="drop"
        )
        logger.info(
            f"Created {fusion_df['target_bin'].nunique()} bins for stratification"
        )

        # Step 4: Stratified split into train/val and test sets — nested-CV
        # mode pulls test from one slice of a stratified K-fold partition;
        # legacy single-split mode uses the ``test_size`` random split.
        outer_cv_mode_rows = outer_fold_idx is not None and n_outer_folds is not None
        if outer_cv_mode_rows:
            from sklearn.model_selection import KFold

            assert n_outer_folds is not None and outer_fold_idx is not None
            k_outer = max(2, int(n_outer_folds))
            fold_i = max(0, min(int(outer_fold_idx), k_outer - 1))
            bins_series = fusion_df["target_bin"]
            try:
                outer_splitter = StratifiedKFold(
                    n_splits=k_outer, shuffle=True, random_state=random_state
                )
                outer_iter = list(outer_splitter.split(fusion_df, bins_series))
            except ValueError:
                outer_splitter = KFold(
                    n_splits=k_outer, shuffle=True, random_state=random_state
                )
                outer_iter = list(outer_splitter.split(fusion_df))
            tv_idx, te_idx = outer_iter[fold_i]
            self.train_val_data = fusion_df.iloc[tv_idx].copy()
            self.test_data = fusion_df.iloc[te_idx].copy()
            logger.info(
                f"Outer fold {fold_i + 1}/{k_outer}: "
                f"train+val={len(self.train_val_data)} rows, "
                f"test={len(self.test_data)} rows."
            )
        else:
            logger.info("Step 4/4: Performing stratified train/test split...")
            self.train_val_data, self.test_data = train_test_split(
                fusion_df,
                test_size=test_size,
                stratify=fusion_df["target_bin"],
                random_state=random_state,
            )

        logger.info(
            f"Split complete: {len(self.train_val_data)} train+val samples "
            f"({k_folds if use_cv else 1} fold(s)), "
            f"{len(self.test_data)} test samples (holdout)"
        )

        self.cv_folds = []
        self._spatial_basis_cache = {}
        self._spatial_adjust_summary = None

        if use_cv:
            logger.info(f"Creating {k_folds}-fold cross-validation splits...")
            skf = StratifiedKFold(
                n_splits=k_folds, shuffle=True, random_state=random_state
            )
            for fold_idx, (train_idx, val_idx) in enumerate(
                skf.split(self.train_val_data, self.train_val_data["target_bin"]), 1
            ):
                train_fold = self.train_val_data.iloc[train_idx].copy()
                val_fold = self.train_val_data.iloc[val_idx].copy()

                # Per-channel scaling removed — folds keep raw channels.
                self.cv_folds.append({"train": train_fold, "val": val_fold})

                logger.info(
                    f"  Fold {fold_idx}: {len(train_fold)} train, {len(val_fold)} val"
                )
            return

        # Single stratified train/val split at the row level.
        logger.info("Creating single train/val split (no CV)...")
        try:
            train_fold, val_fold = train_test_split(
                self.train_val_data,
                test_size=single_split_val_ratio,
                stratify=self.train_val_data["target_bin"],
                random_state=random_state,
            )
        except ValueError:
            train_fold, val_fold = train_test_split(
                self.train_val_data,
                test_size=single_split_val_ratio,
                random_state=random_state,
            )
        train_fold = train_fold.copy()
        val_fold = val_fold.copy()
        # Per-channel scaling removed — folds keep raw channels.
        self.cv_folds.append({"train": train_fold, "val": val_fold})
        logger.info(f"  Single split: {len(train_fold)} train, {len(val_fold)} val")

    def optimize_fusion(
        self,
        n_trials: int = 300,
        n_startup_trials: int = 150,
        objective_metric: str = "distance_corr",
        pruner_type: str = "median",
        sampler_type: str = "TPE",
        seed: int = 42,
        show_progress: bool = True,
        progress_callback: Callable[..., Any] | None = None,
        study_name: str | None = None,
        study_dir: str | None = None,
        cancel_callback: Callable[[], bool] | None = None,
        cgi_formula: str | None = None,
        greenery_channel: str = "cgi",
    ) -> dict:
        """
        Run Optuna optimization with k-fold cross-validation.

        Args:
            n_trials: Total optimization trials
            n_startup_trials: Random exploration trials before the main optimizer
            objective_metric: 'distance_corr', 'spearman', 'r2', 'nrmse', 'mutual_info'
            pruner_type: 'median', 'hyperband', 'successive_halving', or None
            sampler_type: 'TPE', 'CMA-ES', or 'Random'
            seed: Random seed for reproducibility
            show_progress: Whether to show progress bar
            study_name: If set together with ``study_dir``, the study is persisted
                to ``<study_dir>/<study_name>.db`` via Optuna's SQLite storage
                backend. Re-running with the same name reloads completed trials
                and runs only the remaining count.
            study_dir: Output directory for the per-study SQLite file.

        Returns:
            Best parameters dictionary
        """
        if self.cv_folds is None:
            raise ValueError("Call split_data() first")

        # Expose cancel callback so _objective can prune long trials mid-fold.
        self._cancel_callback = cancel_callback

        # Per-call formula override falls back to the engine-level choice from
        # ``__init__``. Validated here so a bad name fails before any Optuna
        # state is touched. The active formula is stored on self so _objective
        # / evaluate_on_test / apply_fusion all see the same value.
        if cgi_formula is not None:
            cgi_formulas.get_formula(cgi_formula)
            self.cgi_formula = cgi_formula

        # Greenery channel — ``cgi`` runs the formula; any other value (one of
        # ``veg`` / ``terrain`` / ``ndvi``) runs a standalone single-metric
        # study. Validated here; the value is stashed on self so _objective
        # and evaluate_on_test see the same mode after this call returns.
        if greenery_channel not in ("cgi", "veg", "terrain", "ndvi"):
            raise ValueError(
                f"greenery_channel must be one of 'cgi','veg','terrain','ndvi'; "
                f"got {greenery_channel!r}."
            )
        self._active_greenery_channel = greenery_channel

        # In longitudinal mode the scoring metric is authoritative on the
        # spec — either one of the four ``mixedlm_*`` options (MixedLM
        # scoring, default ``mixedlm_tstat``) or one of the cross-sectional
        # OLS options (``distance_corr``/``spearman``/``r2``/``nrmse``/``mutual_info``)
        # used when the spec exists only as a metric-file routing key (year-
        # aware cross-sectional). Override whatever the caller passed so the
        # scorer fork in ``_objective`` and ``evaluate_on_test`` sees the
        # same metric the spec advertised.
        if self.is_longitudinal:
            spec = self.longitudinal_spec
            assert spec is not None
            if objective_metric != spec.scoring_metric:
                logger.info(
                    f"Longitudinal mode: overriding objective_metric "
                    f"{objective_metric!r} with spec.scoring_metric "
                    f"{spec.scoring_metric!r}."
                )
                objective_metric = spec.scoring_metric

        self._clear_ring_caches()

        # Build sampler
        if sampler_type == "CMA-ES":
            try:
                import cmaes  # noqa: F401
            except ImportError:
                _log.warning(
                    "CMA-ES sampler requested but the `cmaes` package is not "
                    "installed; falling back to TPE. Install with "
                    "`conda install -c conda-forge cmaes`."
                )
                sampler_type = "TPE"
        if sampler_type == "CMA-ES":
            sampler = CmaEsSampler(
                n_startup_trials=n_startup_trials,
                seed=seed,
            )
        elif sampler_type == "Random":
            sampler = RandomSampler(seed=seed)
        else:  # default: TPE
            sampler = TPESampler(
                n_startup_trials=n_startup_trials,
                multivariate=False,
                warn_independent_sampling=False,
                seed=seed,
            )

        # Select pruner based on objective
        if pruner_type == "median":
            pruner = MedianPruner(n_startup_trials=n_startup_trials)
        elif pruner_type == "hyperband":
            pruner = HyperbandPruner()
        elif pruner_type == "successive_halving":
            pruner = SuccessiveHalvingPruner()
        else:
            pruner = None

        # Determine optimization direction (nrmse is the only lower-is-better
        # metric; correlation / R² / MI / mixedlm metrics all maximize).
        direction = "minimize" if objective_metric == "nrmse" else "maximize"

        # Create study — durable (SQLite RDB) when study_name + study_dir are
        # set, otherwise in-memory.
        if study_name and study_dir:
            os.makedirs(study_dir, exist_ok=True)
            storage_path = os.path.join(study_dir, f"{study_name}.db")
            storage_url = f"sqlite:///{storage_path}"
            self.study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                load_if_exists=True,
                direction=direction,
                sampler=sampler,
                pruner=pruner,
            )
            completed = len(
                [
                    t
                    for t in self.study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                ]
            )
            remaining = max(0, int(n_trials) - completed)
            logger.info(
                f"Optuna study '{study_name}' loaded "
                f"({completed} completed); running {remaining} more trial(s)."
            )
        else:
            self.study = optuna.create_study(
                direction=direction, sampler=sampler, pruner=pruner
            )
            remaining = int(n_trials)

        # Run optimization
        logger.info(
            f"Starting {self.k_folds}-fold CV optimization: {n_trials} trials, "
            f"{objective_metric} metric"
        )

        # Combined callback: progress + cancellation. Optuna invokes this after
        # every trial finishes; calling ``study.stop()`` here ends the run at
        # the next iteration boundary.
        def optuna_callback(study, trial):
            if cancel_callback is not None and cancel_callback():
                logger.info(
                    f"Cancellation requested — stopping study '{study.study_name}' "
                    f"after trial {trial.number}."
                )
                study.stop()
                return
            if progress_callback:
                progress_callback(trial.number + 1, n_trials)

        if remaining > 0:
            self.study.optimize(
                lambda trial: self._objective(trial, objective_metric),
                n_trials=remaining,
                show_progress_bar=show_progress,
                callbacks=[optuna_callback],
            )

        self.best_params = self.study.best_params
        logger.info(
            f"Optimization complete. Best {objective_metric} (CV avg): {self.study.best_value:.4f}"
        )

        return self.best_params

    def _suggest_gvi_radius(self, trial: optuna.Trial, name: str) -> int:
        lo, hi, step = _radius_int_bounds(
            self.gvi_buffer_min_m, self.gvi_buffer_max_m, self.gvi_buffer_step_m
        )
        if lo >= hi:
            return lo
        return trial.suggest_int(name, lo, hi, step=step)

    def _suggest_ndvi_radius(self, trial: optuna.Trial) -> int:
        lo, hi, step = _radius_int_bounds(
            self.ndvi_buffer_min_m, self.ndvi_buffer_max_m, self.ndvi_buffer_step_m
        )
        if lo >= hi:
            return lo
        return trial.suggest_int("ndvi_radius", lo, hi, step=step)

    def _objective(self, trial: optuna.Trial, metric: str) -> float:
        """
        Optuna objective function with k-fold CV.

        Weight search matches CGI.ipynb (ndvi / veg / terrain summing to 100).
        Radii use separate GVI and NDVI buffer ladders (min / max / step metres)
        on the engine; extent padding remains ``buffer_meters``.
        Street-view aggregation parameters are shared for veg and terrain; NDVI
        uses separate stat / percentile choices.
        """
        gvi_cap = int(round(self.gvi_buffer_max_m))
        ndvi_cap = int(round(self.ndvi_buffer_max_m))

        if trial.number == 0:
            gvi_lo, gvi_hi, gvi_st = _radius_int_bounds(
                self.gvi_buffer_min_m,
                self.gvi_buffer_max_m,
                self.gvi_buffer_step_m,
            )
            ndvi_lo, ndvi_hi, ndvi_st = _radius_int_bounds(
                self.ndvi_buffer_min_m,
                self.ndvi_buffer_max_m,
                self.ndvi_buffer_step_m,
            )
            logger.info(
                f"Fusion extent buffer: {self.buffer_meters} m; "
                f"GVI radius search {gvi_lo}–{gvi_hi} m (step {gvi_st}); "
                f"NDVI radius search {ndvi_lo}–{ndvi_hi} m (step {ndvi_st})"
            )

        # ─── Suggest formula parameters (weights + powers) ────────────────────
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

        # ─── Suggest Streetview Parameters (SHARED for veg + terrain) ─────────
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

        # ─── Suggest NDVI Parameters (separate) ────────────────────────────────
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

        # ─── Evaluate Across All CV Folds ─────────────────────────────────────
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

            # ─── Apply Dynamic Radius and Aggregation ─────────────────────────────
            # Both points and rasters use the same circular buffer aggregation
            # For rasters, _prepare_raster_fusion() converted pixels to points at centers
            train_points = self.target_gdf.loc[train_data.index].copy()
            val_points = self.target_gdf.loc[val_data.index].copy()

            # Apply circular buffer aggregation for vegetation (with SHARED streetview_stat)
            if channel_active["veg"]:
                train_veg = self._aggregate_with_ring_cache(
                    train_points,
                    self.veg_data,
                    veg_radius,
                    streetview_stat,
                    streetview_percentile,
                    channel="veg",
                    fold_idx=fold_idx,
                    subset="train",
                )
                val_veg = self._aggregate_with_ring_cache(
                    val_points,
                    self.veg_data,
                    veg_radius,
                    streetview_stat,
                    streetview_percentile,
                    channel="veg",
                    fold_idx=fold_idx,
                    subset="val",
                )
            else:
                train_veg = np.zeros(len(train_points))
                val_veg = np.zeros(len(val_points))

            # Apply circular buffer aggregation for terrain (with SHARED streetview_stat)
            if channel_active["terrain"]:
                train_terrain = self._aggregate_with_ring_cache(
                    train_points,
                    self.terrain_data,
                    terrain_radius,
                    streetview_stat,
                    streetview_percentile,
                    channel="terrain",
                    fold_idx=fold_idx,
                    subset="train",
                )
                val_terrain = self._aggregate_with_ring_cache(
                    val_points,
                    self.terrain_data,
                    terrain_radius,
                    streetview_stat,
                    streetview_percentile,
                    channel="terrain",
                    fold_idx=fold_idx,
                    subset="val",
                )
            else:
                train_terrain = np.zeros(len(train_points))
                val_terrain = np.zeros(len(val_points))

            # Apply circular buffer aggregation for NDVI (separate stat)
            if channel_active["ndvi"]:
                train_ndvi = self._aggregate_with_ring_cache(
                    train_points,
                    self.ndvi_data,
                    ndvi_radius,
                    ndvi_stat,
                    ndvi_percentile,
                    channel="ndvi",
                    fold_idx=fold_idx,
                    subset="train",
                )
                val_ndvi = self._aggregate_with_ring_cache(
                    val_points,
                    self.ndvi_data,
                    ndvi_radius,
                    ndvi_stat,
                    ndvi_percentile,
                    channel="ndvi",
                    fold_idx=fold_idx,
                    subset="val",
                )
            else:
                train_ndvi = np.zeros(len(train_points))
                val_ndvi = np.zeros(len(val_points))

            # Per-channel scaling has been removed: the composite is always a
            # weighted combination of RAW aggregated channel values. The
            # ``whole_grid_scaling`` toggle instead normalizes the resulting
            # composite to [0, 1] (applied after compute_cgi, below).
            train_combined = np.column_stack([train_veg, train_terrain, train_ndvi])
            val_combined = np.column_stack([val_veg, val_terrain, val_ndvi])

            train_valid_mask = ~np.isnan(train_combined).any(axis=1)
            if train_valid_mask.sum() == 0:
                continue  # Skip this fold if no valid data

            train_veg_norm = train_combined[:, 0]
            train_terrain_norm = train_combined[:, 1]
            train_ndvi_norm = train_combined[:, 2]
            val_veg_norm = val_combined[:, 0]
            val_terrain_norm = val_combined[:, 1]
            val_ndvi_norm = val_combined[:, 2]

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

            # Per-fold covariate design matrix (or None when no covariates
            # configured). Polygon mode collapses these alongside the target.
            cov_cols = self.covariate_columns
            if cov_cols:
                train_cov = train_data[cov_cols].to_numpy(dtype=np.float64)
                val_cov = val_data[cov_cols].to_numpy(dtype=np.float64)
            else:
                train_cov = None
                val_cov = None

            # Polygon mode: per-row CGI → per-polygon mean CGI, then score
            # against the per-polygon outcome. In longitudinal+polygon mode
            # polygon_id was set to f"{entity}|{wave}" by
            # _prepare_polygon_fusion so the collapse produces one row per
            # (entity, wave) — the exact shape the MixedLM scorer wants.
            train_entity_id = None
            val_entity_id = None
            train_ysb = None
            val_ysb = None
            # Per-entity coordinates for the spatial-confounding smooth. ``None``
            # whenever the adjustment is off or coordinates were not attached.
            train_coords = None
            val_coords = None
            have_coords = "_cx" in train_data.columns and "_cy" in train_data.columns
            if "polygon_id" in train_data.columns:
                train_pid = train_data["polygon_id"].values
                val_pid = val_data["polygon_id"].values
                # Catchment collapse: polygons average every in-footprint pixel
                # (mask None); point/line entities average only the pixels
                # within this trial's catchment radius (+ each entity's nearest
                # pixel). ``__entity_id``/wave keys were set per observation in
                # the prepare step, so the collapse yields one row per
                # observation — the shape the OLS / MixedLM scorers want.
                catchment_r = self._catchment_radius(
                    veg_radius, terrain_radius, ndvi_radius
                )
                train_mask = self._entity_collapse_mask(train_data, catchment_r)
                val_mask = self._entity_collapse_mask(val_data, catchment_r)
                train_composite = self._collapse_to_entities(
                    train_composite, train_pid, train_mask, "mean"
                )
                val_composite = self._collapse_to_entities(
                    val_composite, val_pid, val_mask, "mean"
                )
                train_targets_arr = self._collapse_to_entities(
                    train_data["target"].values, train_pid, train_mask, "first"
                )
                val_targets_arr = self._collapse_to_entities(
                    val_data["target"].values, val_pid, val_mask, "first"
                )
                if train_cov is not None and val_cov is not None:
                    tc = train_cov  # local binding for the type checker
                    vc = val_cov
                    train_cov = np.column_stack(
                        [
                            self._collapse_to_entities(
                                tc[:, j], train_pid, train_mask, "first"
                            )
                            for j in range(tc.shape[1])
                        ]
                    )
                    val_cov = np.column_stack(
                        [
                            self._collapse_to_entities(
                                vc[:, j], val_pid, val_mask, "first"
                            )
                            for j in range(vc.shape[1])
                        ]
                    )
                if have_coords:
                    # Collapse coordinates with an unmasked "first" so the
                    # per-entity representative point is independent of the
                    # trial's catchment radius — the spatial basis can then be
                    # cached once per fold. The collapse key order (sorted
                    # polygon_id) matches the target/covariate collapse above.
                    train_coords = np.column_stack(
                        [
                            self._collapse_to_entities(
                                train_data["_cx"].to_numpy(np.float64), train_pid, None, "first"
                            ),
                            self._collapse_to_entities(
                                train_data["_cy"].to_numpy(np.float64), train_pid, None, "first"
                            ),
                        ]
                    )
                    val_coords = np.column_stack(
                        [
                            self._collapse_to_entities(
                                val_data["_cx"].to_numpy(np.float64), val_pid, None, "first"
                            ),
                            self._collapse_to_entities(
                                val_data["_cy"].to_numpy(np.float64), val_pid, None, "first"
                            ),
                        ]
                    )
                if self.is_longitudinal:
                    train_entity_id = self._collapse_to_entities(
                        train_data["entity_id"].values, train_pid, train_mask, "first"
                    )
                    val_entity_id = self._collapse_to_entities(
                        val_data["entity_id"].values, val_pid, val_mask, "first"
                    )
                    train_ysb = self._collapse_to_entities(
                        train_data["years_since_baseline"].values,
                        train_pid,
                        train_mask,
                        "first",
                    )
                    val_ysb = self._collapse_to_entities(
                        val_data["years_since_baseline"].values,
                        val_pid,
                        val_mask,
                        "first",
                    )
            else:
                train_targets_arr = train_data["target"].values
                val_targets_arr = val_data["target"].values
                if have_coords:
                    train_coords = train_data[["_cx", "_cy"]].to_numpy(np.float64)
                    val_coords = val_data[["_cx", "_cy"]].to_numpy(np.float64)
                if self.is_longitudinal:
                    train_entity_id = train_data["entity_id"].values
                    val_entity_id = val_data["entity_id"].values
                    train_ysb = train_data["years_since_baseline"].values
                    val_ysb = val_data["years_since_baseline"].values

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

            # ─── Score: MixedLM (longitudinal) or OLS partial-corr ────────────
            # Three modes route through this fork:
            #   1. Cross-sectional (no longitudinal_spec) → OLS scorer.
            #   2. Mixed-effects (spec + ``mixedlm_*`` scoring_metric) →
            #      MixedLM scorer with per-entity random effects.
            #   3. Year-aware cross-sectional (spec + cross-sectional
            #      scoring_metric) → OLS scorer, ignoring entity_id /
            #      years_since_baseline. The spec exists purely so the
            #      pre-aggregation cache can pick the right metric file per
            #      year; the model has no temporal predictor.
            use_mixedlm = (
                self.is_longitudinal and metric in mixed_effects_scoring.MIXEDLM_METRICS
            )
            if use_mixedlm:
                spec = self.longitudinal_spec
                assert spec is not None  # guaranteed by is_longitudinal
                wants_pval = metric in mixed_effects_scoring.HAS_PVALUE
                train_out = mixed_effects_scoring.score_mixedlm(
                    metric,
                    train_targets_arr,
                    train_composite,
                    entity_id=train_entity_id,
                    years_since_baseline=train_ysb,
                    covariates=train_cov,
                    include_time_fixed=spec.include_time_fixed_effect,
                    random_slope=spec.random_slope_time,
                    return_pvalue=wants_pval,
                    spatial_basis=train_sb,
                    spatial_method=self.spatial_adjust_method,
                )
                val_out = mixed_effects_scoring.score_mixedlm(
                    metric,
                    val_targets_arr,
                    val_composite,
                    entity_id=val_entity_id,
                    years_since_baseline=val_ysb,
                    covariates=val_cov,
                    include_time_fixed=spec.include_time_fixed_effect,
                    random_slope=spec.random_slope_time,
                    return_pvalue=wants_pval,
                    spatial_basis=val_sb,
                    spatial_method=self.spatial_adjust_method,
                )
            else:
                # OLS path: covariate-aware distance-correlation / partial rank
                # correlation / incremental-R² / normalized-RMSE / MI scorer.
                # None of these expose a usable p-value (robustness comes from
                # stability selection, not a per-trial p-gate).
                wants_pval = False
                train_out = objective_scoring.score(
                    metric,
                    train_targets_arr,
                    train_composite,
                    covariates=train_cov,
                    return_pvalue=wants_pval,
                    spatial_basis=train_sb,
                    spatial_method=self.spatial_adjust_method,
                )
                val_out = objective_scoring.score(
                    metric,
                    val_targets_arr,
                    val_composite,
                    covariates=val_cov,
                    return_pvalue=wants_pval,
                    spatial_basis=val_sb,
                    spatial_method=self.spatial_adjust_method,
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

        # ─── Store Aggregate Statistics ────────────────────────────────────────
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
        produced_pvals = self.is_longitudinal and metric in (
            mixed_effects_scoring.HAS_PVALUE
        )
        if produced_pvals:
            trial.set_user_attr("train_pvalue_mean", np.mean(fold_train_pvals))
            trial.set_user_attr("val_pvalue_mean", np.mean(fold_val_pvals))
            trial.set_user_attr("fold_train_pvals", fold_train_pvals)
            trial.set_user_attr("fold_val_pvals", fold_val_pvals)

        # Return average validation score across folds
        return avg_val_score

    def get_robust_trials(
        self,
        *,
        val_p_threshold: float = 0.05,
        consistency_tolerance: float = 0.1,
        min_trials: int = 10,
    ) -> list[optuna.Trial]:
        """Pool of trials credible enough to enter top-X% selection.

        Two-step gate, applied in this order:

        1. **Train↔val consistency.** Drop trials whose ``|train_score −
           val_score|`` exceeds ``consistency_tolerance``. Catches trials
           that exploit val-set noise — these have high val |r| but the
           train fit is much weaker, signalling a val-specific fluke.

        2. **Validation significance.** When the metric produces a
           p-value (Pearson / Spearman cross-sectional, MixedLM tstat or
           coef longitudinal), drop trials with
           ``val_pvalue_mean >= val_p_threshold``. The val partition is
           independent of the optimizer's training target, so its p-value
           is the right credibility gate.

        Callers (``compute_averaged_top_params``, ``compute_subset_scores``)
        rank the surviving pool by val score and take the top X %.
        Ordering matters: filtering by significance first then ranking by
        magnitude surfaces the strongest effects **among credible trials**,
        rather than ranking everything by |r| (which is biased upward by
        the best-of-N selection) and only then checking significance.

        Args:
            val_p_threshold: Validation p-value cutoff (correlation metrics
                only). Trials at or above this are dropped.
            consistency_tolerance: Maximum allowed ``|train − val|`` score
                gap. Tighter values filter more aggressively.
            min_trials: When fewer than this many trials survive both
                gates, fall back to the top ``min_trials`` ranked by val
                score, with a logged warning so the caller knows the
                gates didn't bind.

        Returns:
            List of trials passing both gates (or the fallback top-N when
            too few survive).
        """
        if self.study is None:
            raise ValueError("Run optimization first")

        completed_trials = [
            t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed_trials:
            logger.warning("No completed trials found")
            return []

        higher_is_better = self.study.direction.name == "MAXIMIZE"

        # Step 1: consistency pre-filter applies to every metric.
        consistent: list[optuna.Trial] = []
        for trial in completed_trials:
            train_score = trial.user_attrs.get("train_score_mean")
            val_score = trial.user_attrs.get("val_score_mean")
            if train_score is None or val_score is None:
                continue
            if abs(float(train_score) - float(val_score)) > consistency_tolerance:
                continue
            consistent.append(trial)

        # Step 2: validation-p significance, only when the metric produces one.
        has_pvals = any("val_pvalue_mean" in t.user_attrs for t in completed_trials)
        if has_pvals:
            robust_trials = [
                t
                for t in consistent
                if t.user_attrs.get("val_pvalue_mean", 1.0) < val_p_threshold
            ]
            logger.info(
                f"Found {len(robust_trials)} robust trials "
                f"(consistency<={consistency_tolerance}, val p<{val_p_threshold})"
            )
        else:
            robust_trials = consistent
            logger.info(
                f"Found {len(robust_trials)} consistent trials "
                f"(tolerance={consistency_tolerance})"
            )

        if len(robust_trials) < min_trials:
            logger.warning(
                f"Only {len(robust_trials)} trials passed the credibility gates "
                f"(< {min_trials}). Falling back to top {min_trials} by val score "
                "— the headline numbers reflect this fallback, not a credible pool."
            )
            sorted_trials = sorted(
                completed_trials,
                key=lambda t: t.user_attrs.get(
                    "val_score_mean",
                    float("-inf") if higher_is_better else float("inf"),
                ),
                reverse=higher_is_better,
            )
            return sorted_trials[:min_trials]

        return robust_trials

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
        test_points = self.target_gdf.loc[self.test_data.index].copy()

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
            test_veg = np.zeros(len(test_points))

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
            test_terrain = np.zeros(len(test_points))

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
            test_ndvi = np.zeros(len(test_points))

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

        # Test-set covariate matrix (or None when no covariates configured).
        cov_cols = self.covariate_columns
        if cov_cols:
            test_cov = self.test_data[cov_cols].to_numpy(dtype=np.float64)
        else:
            test_cov = None

        # Polygon mode: aggregate per-row CGI by polygon before scoring against
        # the per-polygon outcome. Longitudinal+polygon mode keys the collapse
        # on f"{entity}|{wave}" so the result is one row per (entity, wave) —
        # the shape the MixedLM scorer expects.
        test_entity_id = None
        test_ysb = None
        test_coords = None
        have_coords = (
            "_cx" in self.test_data.columns and "_cy" in self.test_data.columns
        )
        if "polygon_id" in self.test_data.columns:
            test_pid = self.test_data["polygon_id"].values
            catchment_r = self._catchment_radius(
                veg_radius, terrain_radius, ndvi_radius
            )
            test_mask = self._entity_collapse_mask(self.test_data, catchment_r)
            test_composite = self._collapse_to_entities(
                test_composite, test_pid, test_mask, "mean"
            )
            test_targets = self._collapse_to_entities(
                self.test_data["target"].values, test_pid, test_mask, "first"
            )
            if test_cov is not None:
                tc = test_cov  # type-narrow for the comprehension
                test_cov = np.column_stack(
                    [
                        self._collapse_to_entities(
                            tc[:, j], test_pid, test_mask, "first"
                        )
                        for j in range(tc.shape[1])
                    ]
                )
            if have_coords:
                test_coords = np.column_stack(
                    [
                        self._collapse_to_entities(
                            self.test_data["_cx"].to_numpy(np.float64),
                            test_pid, None, "first",
                        ),
                        self._collapse_to_entities(
                            self.test_data["_cy"].to_numpy(np.float64),
                            test_pid, None, "first",
                        ),
                    ]
                )
            if self.is_longitudinal:
                test_entity_id = self._collapse_to_entities(
                    self.test_data["entity_id"].values, test_pid, test_mask, "first"
                )
                test_ysb = self._collapse_to_entities(
                    self.test_data["years_since_baseline"].values,
                    test_pid,
                    test_mask,
                    "first",
                )
        else:
            test_targets = self.test_data["target"].values
            if have_coords:
                test_coords = self.test_data[["_cx", "_cy"]].to_numpy(np.float64)
            if self.is_longitudinal:
                test_entity_id = self.test_data["entity_id"].values
                test_ysb = self.test_data["years_since_baseline"].values

        test_sb = self._spatial_basis_columns(
            test_coords, test_targets, test_cov, test_composite
        )

        # ─── Score: MixedLM (longitudinal) or OLS partial-corr ────────────
        # Same three-mode fork as ``_objective``: year-aware cross-sectional
        # studies sit on a ``LongitudinalSpec`` whose ``scoring_metric`` is
        # one of the OLS options, and route here through the ``else`` arm.
        mixedlm_all: dict[str, float] | None = None
        use_mixedlm = (
            self.is_longitudinal and metric in mixed_effects_scoring.MIXEDLM_METRICS
        )
        if use_mixedlm:
            spec = self.longitudinal_spec
            assert spec is not None
            wants_pval = metric in mixed_effects_scoring.HAS_PVALUE
            if return_all_mixedlm:
                mixedlm_all = mixed_effects_scoring.score_mixedlm(  # type: ignore[assignment]
                    metric,
                    test_targets,
                    test_composite,
                    entity_id=test_entity_id,
                    years_since_baseline=test_ysb,
                    covariates=test_cov,
                    include_time_fixed=spec.include_time_fixed_effect,
                    random_slope=spec.random_slope_time,
                    return_all=True,
                    spatial_basis=test_sb,
                    spatial_method=self.spatial_adjust_method,
                )
                # Surface the requested metric's value alongside the dict so
                # ``test_score`` still reflects the engine's active scoring
                # metric for downstream code that inspects it.
                s = float(mixedlm_all.get(metric, 0.0))  # type: ignore[union-attr]
                score_out = (s, 1.0) if wants_pval else s
            else:
                score_out = mixed_effects_scoring.score_mixedlm(
                    metric,
                    test_targets,
                    test_composite,
                    entity_id=test_entity_id,
                    years_since_baseline=test_ysb,
                    covariates=test_cov,
                    include_time_fixed=spec.include_time_fixed_effect,
                    random_slope=spec.random_slope_time,
                    return_pvalue=wants_pval,
                    spatial_basis=test_sb,
                    spatial_method=self.spatial_adjust_method,
                )
        else:
            # OLS scoring (distance correlation / partial rank corr / R² /
            # normalized RMSE / MI). No usable p-value — robustness is reported
            # via the stability-selection OOB distribution + test-set CI.
            wants_pval = False
            score_out = objective_scoring.score(
                metric,
                test_targets,
                test_composite,
                covariates=test_cov,
                return_pvalue=wants_pval,
                spatial_basis=test_sb,
                spatial_method=self.spatial_adjust_method,
            )
        if wants_pval:
            test_score, test_pval = score_out  # type: ignore[misc]
        else:
            test_score = float(score_out)
            test_pval = None

        result = {"test_score": test_score, "metric": metric}
        if self.spatial_adjust_method != "none":
            result["spatial_adjustment"] = (
                self._spatial_adjust_summary
                or {"method": self.spatial_adjust_method, "applied": False}
            )
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

        return result

    def build_channel_design(self, params: dict, subset: str = "train_val") -> dict:
        """Per-entity raw channel values for the CGI-vs-standalone AIC/BIC test.

        Aggregates **all three** channels (veg / terrain / ndvi) at ``params``'
        radii / stats / percentiles on the requested ``subset`` (``"train_val"``
        for the resampling pool, ``"test"`` for the held-out set), collapses to
        one row per entity (polygon mean when polygon-keyed), and returns the
        raw arrays plus the aligned target, covariates, and — in longitudinal
        mode — entity ids and ``years_since_baseline``.

        No scaling is applied: OLS / MixedLM AIC and BIC are invariant to an
        affine transform of an individual predictor, so raw channel values are
        what :func:`objective_scoring.compare_models_aic_bic` and its MixedLM
        analogue need.
        """
        if subset == "test":
            data = self.test_data
        elif subset == "train_val":
            data = self.train_val_data
        else:
            raise ValueError(f"subset must be 'train_val' or 'test'; got {subset!r}.")
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
            catchment_r = self._catchment_radius(
                veg_radius, terrain_radius, ndvi_radius
            )
            mask = self._entity_collapse_mask(data, catchment_r)
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

        Returns a dict with ``observed``, ``mean``, ``lower``, ``upper``,
        ``ci_level``, ``method``, ``n``.
        """
        from . import statistical_testing as _stats_mod

        res = self.evaluate_on_test(
            params=params, metric=metric, return_predictions=True
        )
        target = np.asarray(res.get("targets"), dtype=np.float64)
        prediction = np.asarray(res.get("predictions"), dtype=np.float64)
        test_cov = res.get("covariates")
        cov_mat = (
            np.asarray(test_cov, dtype=np.float64) if test_cov is not None else None
        )
        # Condition the CI on the spatial smooth by folding the df-selected basis
        # columns into the resampled control matrix (equivalent to KS-AIC for the
        # symmetric metrics: both sides residualize on [covariates, smooth]).
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
                    metric, t_arr, p_arr, covariates=None, return_pvalue=False
                )
                return float(score_out)  # type: ignore[arg-type]

        else:

            def _score_fn(
                t_arr: np.ndarray, p_arr: np.ndarray, c_arr: np.ndarray
            ) -> float:
                score_out = objective_scoring.score(
                    metric, t_arr, p_arr, covariates=c_arr, return_pvalue=False
                )
                return float(score_out)  # type: ignore[arg-type]

        ci = _stats_mod.bootstrap_score_ci(
            target,
            prediction,
            score_fn=_score_fn,
            n_bootstrap=int(n_bootstrap),
            ci_level=float(ci_level),
            method=method,
            seed=int(seed),
            covariates=cov_mat,
        )
        ci["n"] = n
        return ci

    def build_test_prediction_column(self, params: dict, metric: str) -> dict:
        """Convenience: return the per-test-entity (composite, target) arrays.

        The nested-CV runner aggregates these across outer folds to form the
        outer-CV-averaged prediction column used for the headline polygon-level
        bootstrap CI.
        """
        res = self.evaluate_on_test(
            params=params, metric=metric, return_predictions=True
        )
        return {
            "predictions": np.asarray(res.get("predictions"), dtype=np.float64),
            "targets": np.asarray(res.get("targets"), dtype=np.float64),
            "test_score": float(res.get("test_score", float("nan"))),
            "test_pvalue": (
                float(res["test_pvalue"])
                if res.get("test_pvalue") is not None
                else None
            ),
        }

    def apply_fusion(self, weights: dict | None = None) -> pd.DataFrame:
        """
        Apply fusion weights to create composite index.

        Args:
            weights: Dictionary with 'veg_weight', 'terrain_weight', 'ndvi_weight' (0-100 scale)
                    and aggregation parameters (radii, stats, percentiles)
                    If None, uses best_params from optimization

        Returns:
            DataFrame with target, veg, terrain, ndvi, and composite columns
        """
        if weights is None:
            if self.best_params is None:
                raise ValueError(
                    "No weights available. Run optimization or provide weights."
                )
            weights = self.best_params

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

        # Combine all data (train+val+test)
        all_data = pd.concat([self.train_val_data, self.test_data])
        all_points = self.target_gdf.loc[all_data.index].copy()

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
            all_veg = np.zeros(len(all_points))

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
            all_terrain = np.zeros(len(all_points))

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
            all_ndvi = np.zeros(len(all_points))

        # Per-channel scaling removed — the composite uses raw aggregated
        # channel values.
        all_combined = np.column_stack([all_veg, all_terrain, all_ndvi])
        all_veg_norm = all_combined[:, 0]
        all_terrain_norm = all_combined[:, 1]
        all_ndvi_norm = all_combined[:, 2]
        all_veg_norm, all_terrain_norm, all_ndvi_norm = (
            self._normalize_channel_arrays(all_veg_norm, all_terrain_norm, all_ndvi_norm)
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
            catchment_r = self._catchment_radius(
                veg_radius, terrain_radius, ndvi_radius
            )
            mask = self._entity_collapse_mask(result_df, catchment_r)
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
            return poly_df

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

        Returns ``{"score", "score_raw", "pvalue", "pvalue_raw"}``.
        ``score_raw`` collapses to ``score`` when no covariates are
        configured (they're the same number in that case).
        """
        empty = {"score": None, "score_raw": None, "pvalue": None, "pvalue_raw": None}
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
            if cov_cols and "polygon_id" in df.columns:
                full = pd.concat([self.train_val_data, self.test_data])
                cov_per_poly = (
                    full.groupby("polygon_id", sort=False)[cov_cols]
                    .first()
                    .reindex(df["polygon_id"].values)
                )
                cov_mat = cov_per_poly.to_numpy(dtype=np.float64)
            # Cross-sectional metrics no longer expose a usable p-value.
            wants_pval = False

            # Spatial smooth for the partial (adjusted) score; the raw score
            # stays fully unadjusted (no covariates, no smooth).
            sb = self._spatial_basis_columns(
                self._whole_data_coords(df), target, cov_mat, composite
            )

            def _do(
                cov: np.ndarray | None, spat: np.ndarray | None
            ) -> tuple[float | None, float | None]:
                out = objective_scoring.score(
                    metric,
                    target,
                    composite,
                    covariates=cov,
                    return_pvalue=wants_pval,
                    spatial_basis=spat,
                    spatial_method=self.spatial_adjust_method,
                )
                if wants_pval:
                    s, p = out  # type: ignore[misc]
                    return float(s), float(p)
                return float(out), None  # type: ignore[arg-type]

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
        parts = [d for d in (self.train_val_data, self.test_data) if d is not None]
        if not parts:
            return None
        full = pd.concat(parts)
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
        parts = [d for d in (self.train_val_data, self.test_data) if d is not None]
        if not parts:
            return None
        full = pd.concat(parts)
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

    def _augment_cov_with_spatial(
        self,
        df: "pd.DataFrame",
        target: np.ndarray,
        composite: np.ndarray,
        cov: np.ndarray | None,
    ) -> np.ndarray | None:
        """Fold the df-selected spatial smooth into a control matrix for the
        reporting CIs (bootstrap / permutation), which resample the controls
        jointly with the rows.

        Equivalent to KS-AIC for the symmetric metrics (both sides residualize on
        ``[covariates, smooth]``); a no-op when spatial adjustment is off or the
        geometry is degenerate.
        """
        if self.spatial_adjust_method == "none":
            return cov
        sb = self._spatial_basis_columns(
            self._whole_data_coords(df), target, cov, composite
        )
        if sb is None:
            return cov
        return sb if cov is None else np.column_stack([cov, sb])

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
        """Objective effect on the full dataset and per-subset, with CIs + a
        held-out permutation p-value.

        The ``all`` slice (whole dataset, the stability-selected params applied
        to every entity) is the headline greenery effect; ``test`` is the
        held-out generalizability check and carries the permutation p-value;
        ``train_val`` is the tuning pool (CI only). Each entry is
        ``{score, lower, upper, p_value, n}``. Defined for cross-sectional
        objective metrics; returns ``{}`` for metrics it doesn't support
        (e.g. longitudinal MixedLM metrics, scored elsewhere).
        """
        if metric not in objective_scoring.SUPPORTED_METRICS:
            return {}
        from . import statistical_testing as _stats_mod

        higher = metric in objective_scoring.HIGHER_IS_BETTER

        def _score_fn(t_arr, c_arr, cov=None):
            return objective_scoring.score(metric, t_arr, c_arr, cov)

        def _block(t_arr, c_arr, cov, *, do_perm, sub_seed):
            out = {
                "score": None,
                "lower": None,
                "upper": None,
                "p_value": None,
                "n": None,
            }
            if t_arr is None or c_arr is None or len(t_arr) < 3:
                return out
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
            cov_all = self._augment_cov_with_spatial(
                df_full, t_all, c_all, self._whole_data_covariates(df_full)
            )
            results["all"] = _block(t_all, c_all, cov_all, do_perm=True, sub_seed=seed)
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
                # Fold the test smooth into the resampled controls (KS-style).
                te_sb = tr.get("spatial_basis")
                if te_sb is not None:
                    te_sb = np.asarray(te_sb, dtype=np.float64)
                    cov_te = te_sb if cov_te is None else np.column_stack([cov_te, te_sb])
                results["test"] = _block(
                    t_te, c_te, cov_te, do_perm=True, sub_seed=seed + 101
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
                cov_tv = self._augment_cov_with_spatial(
                    sub, t_tv, c_tv, self._whole_data_covariates(sub)
                )
                results["train_val"] = _block(
                    t_tv, c_tv, cov_tv, do_perm=False, sub_seed=seed + 202
                )
        except Exception as exc:
            logger.warning(f"evaluate_effects: train_val scoring failed: {exc}")

        return results

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
            # Both composites share the same outcome-selected smooth; fold it into
            # the resampled controls so the paired difference is spatially adjusted.
            cov = self._augment_cov_with_spatial(merged, target, cgi_c, cov)
        except Exception as exc:
            logger.warning(f"paired_objective_difference: alignment failed: {exc}")
            return None

        if len(target) < 3:
            return None

        higher = metric in objective_scoring.HIGHER_IS_BETTER
        sign = 1.0 if higher else -1.0

        def _score(t_arr, c_arr, cov_arr):
            return float(objective_scoring.score(metric, t_arr, c_arr, cov_arr))

        obs_cgi = _score(target, cgi_c, cov)
        obs_std = _score(target, std_c, cov)
        obs_diff = sign * (obs_cgi - obs_std)

        rng = np.random.default_rng(int(seed))
        n = len(target)
        diffs = np.empty(int(n_bootstrap), dtype=np.float64)
        for i in range(int(n_bootstrap)):
            idx = rng.integers(0, n, size=n)
            cov_i = None if cov is None else cov[idx]
            try:
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
        top_percent: float = 0.2,
    ) -> dict[str, dict[str, float | None]]:
        """Score ``params`` on every data slice the optimizer saw.

        Returns a dict of ``{subset: {score, pvalue, n}}`` for the four
        canonical slices:

        - ``train`` / ``val`` — the per-fold means across the **top
          ``top_percent`` of robust trials**, the same pool the composite
          GeoTIFF is built from. Pulled from each trial's ``user_attrs``.
        - ``test`` — fresh test-set score by re-running
          :meth:`evaluate_on_test` with the supplied params.
        - ``all`` — composite applied to every entity (the full dataset)
          via :meth:`apply_fusion`, polygon-collapsed for polygon
          targets, scored against the per-entity outcome.

        The runner caches the returned dict on each study's bundle so the
        results UI doesn't re-score on every page rerun.
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

        if self.study is not None:
            # Standard mode: train/val are the per-fold means across the
            # top-K robust trials (the same pool the composite is built
            # from).
            try:
                robust = self.get_robust_trials(
                    val_p_threshold=0.05,
                    consistency_tolerance=0.1,
                    min_trials=10,
                )
            except Exception:
                robust = []
            if not robust:
                robust = [
                    t
                    for t in self.study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                ]
            n_top = max(1, int(len(robust) * top_percent))
            top_trials = sorted(
                robust,
                key=lambda t: t.value if t.value is not None else float("nan"),
                reverse=(self.study.direction.name == "MAXIMIZE"),
            )[:n_top]

            def _mean_attr(name: str) -> float | None:
                vals = [
                    float(t.user_attrs.get(name))
                    for t in top_trials
                    if t.user_attrs.get(name) is not None
                ]
                return float(np.mean(vals)) if vals else None

            # Standard mode: per-trial user_attrs only carry the partial
            # (covariate-adjusted) score the objective optimized. To also
            # surface a raw correlation, apply the averaged params to the
            # train+val pool once and score without covariates — that's
            # what ``score_raw`` represents below.
            tv_both = self._score_data_subset(
                data=self.train_val_data, params=params, metric=metric
            )
            out["train"] = {
                "score": _mean_attr("train_score_mean"),
                "score_raw": tv_both.get("score_raw"),
                "pvalue": _mean_attr("train_pvalue_mean"),
                "pvalue_raw": tv_both.get("pvalue_raw"),
                "n": train_val_n,
            }
            out["val"] = {
                "score": _mean_attr("val_score_mean"),
                "score_raw": tv_both.get("score_raw"),
                "pvalue": _mean_attr("val_pvalue_mean"),
                "pvalue_raw": tv_both.get("pvalue_raw"),
                "n": train_val_n,
            }
        else:
            # Stability-selection mode: there's no per-trial study to read
            # train/val means from. Surrogate mapping:
            # * ``train`` → score on the full train+val pool with the
            #   winning params (in-pool fit, the closest analogue to
            #   "train" the paradigm has — the pool is what the bootstrap
            #   draws from).
            # * ``val`` → the cell-aggregation **median** OOB score
            #   already recorded by ``bootstrap_stability_selection`` under
            #   ``__cell_median__``. This is the per-bootstrap held-out
            #   score averaged across all trials in the winning cell — a
            #   direct measure of cross-resample predictive performance.
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
            cell_median = (
                params.get("__cell_median__") if isinstance(params, dict) else None
            )
            out["val"] = {
                "score": (
                    float(cell_median)
                    if cell_median is not None and np.isfinite(float(cell_median))
                    else None
                ),
                # ``val`` is the cell-median OOB score, which is computed
                # by ``_objective`` with covariates configured on the
                # engine — i.e. it's a partial-correlation analogue. No
                # raw equivalent exists at this level because each
                # bootstrap's OOB rows score with covariates always
                # present.
                "score_raw": None,
                "pvalue": None,
                "pvalue_raw": None,
                "n": train_val_n,
            }

        # ── test: fresh evaluate_on_test with these params ────────────
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
            out["test"] = {
                "score": (
                    float(test_result["test_score"])
                    if test_result.get("test_score") is not None
                    else None
                ),
                "score_raw": test_both.get("score_raw"),
                "pvalue": (
                    float(test_result["test_pvalue"])
                    if test_result.get("test_pvalue") is not None
                    else None
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

        # ── all: composite on every entity (full dataset) ─────────────
        try:
            df = self.apply_fusion(weights=dict(params))
            target = np.asarray(df["target"].values, dtype=np.float64)
            composite = np.asarray(df["composite"].values, dtype=np.float64)
            cov: np.ndarray | None = None
            cov_cols = self.covariate_columns
            if cov_cols and "polygon_id" in df.columns:
                full = pd.concat([self.train_val_data, self.test_data])
                cov_per_poly = (
                    full.groupby("polygon_id", sort=False)[cov_cols]
                    .first()
                    .reindex(df["polygon_id"].values)
                )
                cov = cov_per_poly.to_numpy(dtype=np.float64)
            # Cross-sectional metrics no longer expose a usable p-value.
            wants_pval = False

            sb = self._spatial_basis_columns(
                self._whole_data_coords(df), target, cov, composite
            )

            def _full_score(
                c: np.ndarray | None, spat: np.ndarray | None
            ) -> tuple[float | None, float | None]:
                s_out = objective_scoring.score(
                    metric,
                    target,
                    composite,
                    covariates=c,
                    return_pvalue=wants_pval,
                    spatial_basis=spat,
                    spatial_method=self.spatial_adjust_method,
                )
                if wants_pval:
                    s, p = s_out  # type: ignore[misc]
                    return float(s), float(p)
                return float(s_out), None  # type: ignore[arg-type]

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
            if "polygon_id" in df.columns:
                full = pd.concat([self.train_val_data, self.test_data])
                cov_per_poly = (
                    full.groupby("polygon_id", sort=False)[cov_cols]
                    .first()
                    .reindex(df["polygon_id"].values)
                )
                cov_mat = cov_per_poly.to_numpy(dtype=np.float64)
            else:
                full = pd.concat([self.train_val_data, self.test_data])
                cov_mat = full[cov_cols].to_numpy(dtype=np.float64)

            mask = ~(
                np.isnan(target) | np.isnan(composite) | np.isnan(cov_mat).any(axis=1)
            )
            target = target[mask]
            composite = composite[mask]
            cov_mat = cov_mat[mask]
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

    def _generate_all_optuna_plots(
        self, study: optuna.Study, output_dir: str, study_name: str = "study"
    ) -> None:
        """Generate all available Optuna visualization plots for a study."""
        try:
            from optuna.visualization import (
                plot_contour,
                plot_edf,
                plot_optimization_history,
                plot_parallel_coordinate,
                plot_param_importances,
                plot_rank,
                plot_slice,
                plot_timeline,
            )

            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Generating Optuna plots for {study_name} in {output_dir}")

            # Get completed trials
            completed_trials = [
                t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            ]

            if len(completed_trials) == 0:
                logger.warning(f"No completed trials for {study_name}")
                return

            # 1. Optimization History
            try:
                fig = plot_optimization_history(study)
                fig.write_html(os.path.join(output_dir, "optimization_history.html"))
            except Exception as e:
                logger.warning(f"Could not generate optimization_history: {e}")

            # 2. Parameter Importances
            try:
                fig = plot_param_importances(study)
                fig.write_html(os.path.join(output_dir, "param_importances.html"))
            except Exception as e:
                logger.warning(f"Could not generate param_importances: {e}")

            # 3. Parallel Coordinates (weights)
            try:
                fig = plot_parallel_coordinate(
                    study, params=["veg_weight", "terrain_weight", "ndvi_weight"]
                )
                fig.update_layout(title_text=f"Parameter Weights - {study_name}")
                fig.write_html(
                    os.path.join(output_dir, "parallel_coordinates_weights.html")
                )
            except Exception as e:
                logger.warning(f"Could not generate parallel_coordinates: {e}")

            # 4. Parallel Coordinates (all params)
            try:
                fig = plot_parallel_coordinate(study)
                fig.update_layout(title_text=f"All Parameters - {study_name}")
                fig.write_html(
                    os.path.join(output_dir, "parallel_coordinates_all.html")
                )
            except Exception as e:
                logger.warning(f"Could not generate parallel_coordinates_all: {e}")

            # 5. Slice Plot (weights)
            try:
                fig = plot_slice(
                    study, params=["veg_weight", "terrain_weight", "ndvi_weight"]
                )
                fig.write_html(os.path.join(output_dir, "slice_plot_weights.html"))
            except Exception as e:
                logger.warning(f"Could not generate slice_plot: {e}")

            # 6. Contour Plot (weights)
            try:
                fig = plot_contour(
                    study, params=["veg_weight", "terrain_weight", "ndvi_weight"]
                )
                fig.write_html(os.path.join(output_dir, "contour_weights.html"))
            except Exception as e:
                logger.warning(f"Could not generate contour: {e}")

            # 7. EDF (Empirical Distribution Function)
            try:
                fig = plot_edf(study)
                fig.write_html(os.path.join(output_dir, "edf.html"))
            except Exception as e:
                logger.warning(f"Could not generate edf: {e}")

            # 8. Rank Plot
            try:
                fig = plot_rank(study)
                fig.write_html(os.path.join(output_dir, "rank.html"))
            except Exception as e:
                logger.warning(f"Could not generate rank: {e}")

            # 9. Timeline
            try:
                fig = plot_timeline(study)
                fig.write_html(os.path.join(output_dir, "timeline.html"))
            except Exception as e:
                logger.warning(f"Could not generate timeline: {e}")

            logger.info(f"✓ Generated plots for {study_name}")

        except ImportError as e:
            logger.warning(f"Optuna visualization not available: {e}")
            logger.warning("Install with: pip install optuna[visualization] plotly")

    def bootstrap_stability_selection(
        self,
        metric: str,
        *,
        n_bootstraps: int = 20,
        n_trials_per_bootstrap: int = 50,
        weight_bin_pct: int = 10,
        top_percent_per_bootstrap: float = 0.2,
        min_cell_count: int = 3,
        worst_quantile: float = 0.10,
        radius_bin_m: int | None = None,
        spatial_resample: bool = False,
        seed: int = 42,
        cancel_callback: Callable[..., bool] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Stability selection via complementary-pairs out-of-bag scoring.

        Adapts Meinshausen & Bühlmann's stability selection (2010) to
        hyperparameter search, using the complementary-pairs subsampling of
        Shah & Samworth (2013) so the reported PFER bound is rigorous. The
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
        * Pool OOB scores per cell.
        * Robust cell = the one with the best worst-quantile score
          (``q_worst = quantile(scores, worst_quantile)`` for higher-is-
          better metrics; ``quantile(scores, 1 - worst_quantile)`` for
          lower-is-better metrics like RMSE) subject to ``count >=
          min_cell_count``.
        * Within the chosen cell, average all parameters using the same
          renormalization logic as :meth:`compute_averaged_top_params` so the
          output is drop-in compatible with downstream code.

        Args:
            metric: Scoring metric (any value supported by ``_objective``).
            n_bootstraps: Target number of ⌊n/2⌋ subsamples (rounded to an even
                count of complementary pairs). 20–50 typical.
            n_trials_per_bootstrap: QMC trials inside each subsample.
                30–100 typical.
            weight_bin_pct: Cell width for weight binning. 10 % bins yield
                ~78 valid simplex cells for the weighted-average formula.
            top_percent_per_bootstrap: Fraction of each bootstrap's trials
                considered "selected" for the selection-probability sidecar.
            min_cell_count: A cell is eligible only when at least this many
                trials landed in it across all bootstraps. Prevents a
                single-trial outlier cell from claiming "best".
            worst_quantile: 0.10 → 10th percentile worst-case score for
                higher-is-better metrics; 90th percentile for lower-is-
                better. Tighter (e.g. 0.05) penalises rare-bad-luck cells
                harder; looser (e.g. 0.25) tolerates more bad-luck draws.
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

        Returns: averaged-params dict in the same shape as
        :meth:`compute_averaged_top_params`, plus bookkeeping keys
        ``__cell_q_worst__``, ``__cell_count__``, ``__cell_median__``,
        ``__cell_selection_probability__``, ``__n_bootstraps__``,
        ``__n_trials_per_bootstrap__``, ``__n_total_trials__``,
        ``__worst_quantile__``, and stage-2 keys ``__radius_cell_q_worst__``,
        ``__radius_cell_median__``, ``__radius_cell_count__``,
        ``__radius_bin_m__``, ``__radius_cell_stats__``.
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

        # Group → rows once, so each bootstrap's in-bag materialisation is a
        # dict lookup + concat instead of an O(G·N) boolean scan per group.
        group_rows: dict = {}
        if use_groups:
            for g, sub in train_val.groupby(group_col, sort=False):
                group_rows[g] = sub

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
            "trials_kept": 0,
        }

        try:
            rng = np.random.default_rng(int(seed))
            # Complementary-pairs subsampling (Shah & Samworth 2013): each pair
            # partitions the sampling units into halves A | B; each half is an
            # in-bag scored on its complement, giving two ~50%-OOB subsamples.
            # This is the ⌊n/2⌋ scheme the Meinshausen–Bühlmann PFER bound
            # assumes, so the reported bound is rigorous rather than the
            # approximation bootstrap resampling gives. Whole blocks are split
            # when spatial resampling is on so each OOB half is out-of-region.
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
                    # One row-slice per in-bag group (no multiplicity under
                    # subsampling), preserving the original DataFrame index the
                    # pre-aggregation cache is keyed by.
                    in_bag_df = pd.concat([group_rows[g] for g in in_bag_unique])
                    oob_df = train_val[train_val[group_col].isin(oob_unique)].copy()
                else:
                    in_bag_df = train_val.iloc[in_bag_unique].copy()
                    oob_df = train_val.iloc[oob_unique].copy()

                # Splice this subsample into ``cv_folds`` — the existing
                # ``_objective`` reads ``train`` / ``val`` from each fold and
                # handles aggregation, channel collapse, and OOB scoring.
                self.cv_folds = [{"train": in_bag_df, "val": oob_df}]
                # Each resample is a different row set, so its spatial basis is
                # rebuilt; drop the previous resample's cache to bound memory.
                self._spatial_basis_cache = {}

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

                try:
                    study.optimize(
                        lambda t: self._objective(t, metric),
                        n_trials=n_trials_b,
                        show_progress_bar=False,
                        catch=(Exception,),
                        callbacks=[_trial_progress],
                    )
                except Exception:
                    # Catastrophic study failure — skip this subsample and
                    # continue rather than aborting the whole selection.
                    continue

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

        # ── Per-cell aggregation ─────────────────────────────────────────
        from . import statistical_testing as _stats_mod

        cells: dict[tuple, list[dict]] = {}
        for r in records:
            cells.setdefault(r["cell"], []).append(r)

        def _q_worst(scores: list[float]) -> float:
            if higher_is_better:
                return float(np.quantile(scores, worst_quantile))
            return float(np.quantile(scores, 1.0 - worst_quantile))

        # ── Automated threshold calibration (stage 1: channel mix) ────────
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
            rankings.append(
                sorted(rep, key=lambda c: rep[c], reverse=higher_is_better)
            )
        n_candidate_cells = len(cells)
        calib = _stats_mod.calibrate_stability_selection(rankings, n_candidate_cells)
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
            # Most consistently selected cell; ties broken by median OOB score.
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
            # standalone's single weight cell) — fall back to the best
            # worst-quantile cell.
            best = max(
                cell_stats,
                key=lambda c: c["q_worst"] if higher_is_better else -c["q_worst"],
            )

        # ── Stage 2: spatial tuning within the winning weight cell ────────
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
            key=lambda c: c["q_worst"] if higher_is_better else -c["q_worst"],
        )

        # ── Average params within the winning radius sub-cell ─────────────
        # Re-uses the same renormalization rules as compute_averaged_top_params
        # so downstream code (composite generation, report) can consume the
        # output identically. ``formula`` was resolved up front (see above).
        winners = radius_best["records"]

        def _mean_int(name: str, default: int) -> int:
            vals = [int(r["params"].get(name, default)) for r in winners]
            return int(round(float(np.mean(vals)))) if vals else int(default)

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
        # Compatibility with downstream code that reads the field names
        # ``compute_averaged_top_params`` populates.
        final_params["__n_top_trials__"] = int(best["count"])
        final_params["__n_robust_trials__"] = int(sum(c["count"] for c in cell_stats))

        # Diagnostics for the results UI: top-10 ranked cells (so the user
        # can see whether the winner is alone or part of a tight cluster of
        # similar regions) and the winner's OOB-score distribution (for a
        # histogram showing q_worst → median → max). These live under ``__``
        # keys so the "Final params" panel still strips them out, but the
        # raw bundle in the runner preserves them.
        weight_keys = formula.weight_keys
        # Rank the diagnostics table by the decision criterion — calibrated
        # selection probability (median OOB breaks ties) — so the top row is
        # the chosen winner; q_worst is shown alongside as a diagnostic.
        ranked = sorted(
            cell_stats,
            key=lambda c: (
                c["selection_probability"],
                c["median"] if higher_is_better else -c["median"],
            ),
            reverse=True,
        )
        final_params["__cell_stats__"] = [
            {
                "weights": {
                    k: int(c["cell"][i] * weight_bin_pct)
                    for i, k in enumerate(weight_keys)
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
        return final_params

    def compute_averaged_top_params(
        self,
        top_percent: float = 0.2,
    ) -> dict[str, Any]:
        """Return the ensemble-averaged params from the top X% of robust trials.

        The composite GeoTIFF and the canonical "final parameters" reported
        back to the user are both built from this average rather than from
        the single best trial — averaging across the top of the robust
        pool smooths out one-trial noise on the Optuna response surface
        and keeps the composite stable across reruns.

        Weights are averaged and renormalised to sum to 100 on the int
        scale both formulas record on; powers stay as floats; per-channel
        radii / percentiles take the integer mean; stat categoricals take
        the mode.
        """
        from statistics import mode

        if self.study is None:
            raise ValueError(
                "No optimization study available. Run optimize_fusion() first."
            )

        robust_trials = self.get_robust_trials(
            val_p_threshold=0.05,
            consistency_tolerance=0.1,
            min_trials=10,
        )
        if not robust_trials:
            robust_trials = [
                t
                for t in self.study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]
        if not robust_trials:
            raise ValueError("No completed trials available for ensemble averaging.")
        n_top = max(1, int(len(robust_trials) * top_percent))
        top_trials = sorted(
            robust_trials,
            key=lambda t: t.value,
            reverse=(self.study.direction.name == "MAXIMIZE"),
        )[:n_top]

        formula = cgi_formulas.get_formula(self.cgi_formula)
        radii_veg = [
            t.params.get("veg_radius", int(round(self.gvi_buffer_max_m)))
            for t in top_trials
        ]
        radii_ter = [
            t.params.get("terrain_radius", int(round(self.gvi_buffer_max_m)))
            for t in top_trials
        ]
        radii_ndvi = [
            t.params.get("ndvi_radius", int(round(self.ndvi_buffer_max_m)))
            for t in top_trials
        ]
        streetview_stats = [t.params.get("streetview_stat", "mean") for t in top_trials]
        ndvi_stats = [t.params.get("ndvi_stat", "mean") for t in top_trials]
        streetview_percentiles = [
            t.params.get("streetview_percentile", 50) for t in top_trials
        ]
        ndvi_percentiles = [t.params.get("ndvi_percentile", 50) for t in top_trials]

        # Snap averaged radii + percentiles to the cache grid so the
        # composite-map lookup hits cached cells.
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
            "veg_radius": _snap(float(np.mean(radii_veg)), gvi_choices),
            "terrain_radius": _snap(float(np.mean(radii_ter)), gvi_choices),
            "ndvi_radius": _snap(float(np.mean(radii_ndvi)), ndvi_choices),
            "streetview_stat": mode(streetview_stats),
            "ndvi_stat": mode(ndvi_stats),
            "streetview_percentile": _snap_pct(float(np.mean(streetview_percentiles))),
            "ndvi_percentile": _snap_pct(float(np.mean(ndvi_percentiles))),
        }
        for power_key in formula.power_keys:
            final_params[power_key] = float(
                np.mean([t.params.get(power_key, 1.0) for t in top_trials])
            )
        avg_weights = {
            k: float(np.mean([t.params.get(k, 0.0) for t in top_trials]))
            for k in formula.weight_keys
        }
        weight_sum = sum(avg_weights.values())
        if weight_sum > 0:
            scale = 100.0 / weight_sum
            for k, v in avg_weights.items():
                final_params[k] = int(round(v * scale))
        else:
            for k in avg_weights:
                final_params[k] = 0
            # Standalone mode: force the active channel's weight to 100
            # so the composite uses that channel only.
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
        final_params["__n_top_trials__"] = int(n_top)
        final_params["__n_robust_trials__"] = int(len(robust_trials))
        return final_params

    def generate_composite_greenery_map(
        self,
        output_path: str = "output_results/composite_greenery.tif",
        top_percent: float = 0.2,
        progress_callback: Callable[..., Any] | None = None,
    ) -> str:
        """
        Generate final composite greenery map using averaged parameters from top robust trials.

        Takes top 20% of robust trials, averages their parameters, and generates a composite
        greenery raster matching the target grid (if raster input) or 50m grid (if geojson input).

        Args:
            output_path: Path to save the composite greenery GeoTIFF
            top_percent: Top percentage of robust trials to average (default: 0.2 = 20%)
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            Path to the saved composite greenery map
        """
        logger.info("Generating composite greenery map from top robust trials...")

        if progress_callback:
            progress_callback(0, 100)

        # Source the final params from whichever selection path the run used:
        # * normal mode → averaged top-X % of robust trials from ``self.study``
        # * stability-selection mode → the cell-winner already pinned on
        #   ``self.best_params`` (bootstrap_stability_selection has no master
        #   study so ``compute_averaged_top_params`` would raise).
        if self.study is None:
            if self.best_params is None:
                raise ValueError(
                    "Composite generation needs either a study to average from "
                    "or ``self.best_params`` set by a prior selection step."
                )
            final_params = dict(self.best_params)
            logger.info(f"Composite TIFF stability-selected params: {final_params}")
        else:
            final_params = self.compute_averaged_top_params(top_percent=top_percent)
            logger.info(
                f"Composite TIFF averaged params (top {top_percent*100:.0f}% of "
                f"{final_params['__n_robust_trials__']} robust trials, "
                f"n_top={final_params['__n_top_trials__']}): {final_params}"
            )
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
        # params the composite was built from. The recorded provenance adapts
        # to the selection path: stability selection reports the winning-cell
        # diagnostics, the legacy averaging path reports the trial counts.
        params_path = output_path.replace(".tif", "_params.json")
        import json

        clean_params = {k: v for k, v in final_params.items() if not k.startswith("__")}
        if self.study is None:
            provenance = {
                "selection_method": "bootstrap_stability_selection",
                "cell_q_worst": final_params.get("__cell_q_worst__"),
                "cell_median": final_params.get("__cell_median__"),
                "cell_count": final_params.get("__cell_count__"),
                "cell_selection_probability": final_params.get(
                    "__cell_selection_probability__"
                ),
                "worst_quantile": final_params.get("__worst_quantile__"),
                "n_bootstraps": final_params.get("__n_bootstraps__"),
                "n_trials_per_bootstrap": final_params.get(
                    "__n_trials_per_bootstrap__"
                ),
                "n_total_trials": final_params.get("__n_total_trials__"),
            }
        else:
            provenance = {
                "selection_method": "averaged_top_robust_trials",
                "n_trials_averaged": int(final_params.get("__n_top_trials__", 0)),
                "total_robust_trials": int(final_params.get("__n_robust_trials__", 0)),
                "top_percent": top_percent,
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

    def generate_results_report(
        self,
        output_dir: str = "output_results/fusion/study_results",
        include_plots: bool = True,
        progress_callback: Callable[..., Any] | None = None,
        composite_path: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Generate comprehensive optimization results report with visualizations.

        Creates reports similar to CGI.ipynb including:
        - Separate analysis for ROBUST trials (main focus) and ALL trials (debugging)
        - Trial history plots for both studies
        - Parameter importance analysis
        - Parallel coordinates visualization
        - Best trial summary
        - Test set evaluation
        - Composite greenery map generation

        Args:
            output_dir: Directory to save reports and plots
            include_plots: Whether to generate and save visualization plots
            progress_callback: Optional callback for progress updates

        Returns:
            The ensemble-averaged top-20% parameter dict that was used to
            build the composite GeoTIFF, or ``None`` when no completed
            trials exist (warning is logged).
        """
        import json
        from datetime import datetime

        if self.study is None:
            raise ValueError(
                "No optimization study available. Run optimize_fusion() first."
            )

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Generating results report in {output_dir}")

        if progress_callback:
            progress_callback(0, 100)

        # Get completed trials
        all_completed_trials = [
            t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]

        if len(all_completed_trials) == 0:
            logger.warning("No completed trials to report on")
            return

        if progress_callback:
            progress_callback(5, 100)

        # ═══ GET ROBUST TRIALS ═══
        logger.info("Extracting robust trials (val-p + train-val consistency)...")
        robust_trials = self.get_robust_trials(
            val_p_threshold=0.05,
            consistency_tolerance=0.1,
            min_trials=10,
        )

        logger.info(
            f"Found {len(robust_trials)} robust trials out of {len(all_completed_trials)} total"
        )

        if not robust_trials:
            logger.warning(
                "⚠ No robust trials found (all p-values > 0.05 after FDR correction)."
            )
            logger.warning(
                "  Reports will be based on all trials. Consider relaxing p_threshold or check data quality."
            )
            robust_trials = all_completed_trials
        else:
            logger.info(
                f"✓ Robust trials: {len(robust_trials)}/{len(all_completed_trials)} "
                f"({len(robust_trials)/len(all_completed_trials)*100:.1f}%)"
            )

        if progress_callback:
            progress_callback(10, 100)

        # ═══ CREATE ROBUST TRIALS STUDY ═══
        logger.info("=" * 80)
        logger.info(
            f"Creating SEPARATE Optuna study from {len(robust_trials)} robust trials..."
        )
        logger.info("This study will ONLY contain statistically significant trials")
        logger.info("=" * 80)

        # Create a new study containing only robust trials
        robust_study = optuna.create_study(
            direction=self.study.direction,
            sampler=self.study.sampler,
        )

        # Add robust trials to the new study
        trial_numbers_added = []
        for trial in robust_trials:
            robust_study.add_trial(trial)
            trial_numbers_added.append(trial.number)

        logger.info(f"✓ Robust study created with {len(robust_study.trials)} trials")
        logger.info(
            f"  Trial numbers: {sorted(trial_numbers_added)[:20]}..."
        )  # Show first 20

        if progress_callback:
            progress_callback(15, 100)

        # ═══════════════════════════════════════════════════════════════════
        # PRIMARY ANALYSIS: ROBUST TRIALS ONLY
        # ═══════════════════════════════════════════════════════════════════

        robust_dir = os.path.join(output_dir, "robust_trials")
        os.makedirs(robust_dir, exist_ok=True)

        logger.info("=" * 80)
        logger.info("GENERATING PRIMARY REPORT: ROBUST TRIALS ONLY")
        logger.info(f"Study contains: {len(robust_study.trials)} trials (FILTERED)")
        logger.info(f"Original study: {len(self.study.trials)} trials (FULL)")
        logger.info("=" * 80)

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FUSION OPTIMIZATION RESULTS - ROBUST TRIALS (PRIMARY)")
        report_lines.append(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append(
            "⚠ IMPORTANT: This report ONLY includes statistically significant trials"
        )
        report_lines.append("  (FDR-corrected p-value < 0.05)")
        report_lines.append("")
        report_lines.append(f"Original Study: {len(all_completed_trials)} trials")
        report_lines.append(
            f"Robust Trials (FILTERED): {len(robust_trials)} ({len(robust_trials)/len(all_completed_trials)*100:.1f}%)"
        )
        report_lines.append(
            f"Excluded Trials: {len(all_completed_trials) - len(robust_trials)} (not statistically significant)"
        )
        report_lines.append("")
        report_lines.append(
            "This report focuses on statistically significant trials (FDR-corrected p<0.05)"
        )
        report_lines.append("")
        report_lines.append(f"Original Study: {len(all_completed_trials)} trials")
        report_lines.append(
            f"Robust Trials: {len(robust_trials)} ({len(robust_trials)/len(all_completed_trials)*100:.1f}%)"
        )
        report_lines.append("")

        # Best trial from robust trials
        best_robust = robust_study.best_trial
        report_lines.append(f"BEST ROBUST TRIAL (#{best_robust.number})")
        report_lines.append("-" * 80)
        report_lines.append(f"Best Value: {best_robust.value:.6f}")
        report_lines.append("")
        report_lines.append("Parameters:")
        for key, val in best_robust.params.items():
            report_lines.append(f"  {key}: {val}")
        report_lines.append("")

        # CV scores
        report_lines.append("Cross-Validation Performance:")
        if "train_score_mean" in best_robust.user_attrs:
            report_lines.append(
                f"  Train Score (mean): {best_robust.user_attrs['train_score_mean']:.6f}"
            )
            report_lines.append(
                f"  Val Score (mean): {best_robust.user_attrs['val_score_mean']:.6f}"
            )
            report_lines.append(
                f"  Val Score (std): {best_robust.user_attrs['val_score_std']:.6f}"
            )

        if "train_pvalue_mean" in best_robust.user_attrs:
            report_lines.append(
                f"  Train P-value (mean): {best_robust.user_attrs['train_pvalue_mean']:.6e}"
            )
            report_lines.append(
                f"  Val P-value (mean): {best_robust.user_attrs['val_pvalue_mean']:.6e}"
            )
        report_lines.append("")

        # Test set evaluation
        if self.test_data is not None:
            report_lines.append("TEST SET EVALUATION")
            report_lines.append("-" * 80)
            try:
                test_results = self.evaluate_on_test(
                    params=best_robust.params, return_predictions=False
                )
                report_lines.append(f"Test Score: {test_results['test_score']:.6f}")
                if "test_pvalue" in test_results:
                    report_lines.append(
                        f"Test P-value: {test_results['test_pvalue']:.6e}"
                    )
                report_lines.append(f"Test Samples: {len(self.test_data)}")
            except Exception as e:
                report_lines.append(f"Test evaluation failed: {e}")
        report_lines.append("")

        # Top 10 robust trials
        report_lines.append("TOP 10 ROBUST TRIALS")
        report_lines.append("-" * 80)
        sorted_robust = sorted(
            robust_trials,
            key=lambda t: t.value,
            reverse=(robust_study.direction.name == "MAXIMIZE"),
        )[:10]
        report_lines.append(
            f"{'Rank':<6} {'Trial':<8} {'Value':<12} {'P-val':<12} {'Veg%':<6} {'Ter%':<6} {'NDVI%':<6}"
        )
        report_lines.append("-" * 80)
        for rank, trial in enumerate(sorted_robust, 1):
            veg_w = trial.params.get("veg_weight", 0)
            ter_w = trial.params.get("terrain_weight", 0)
            ndvi_w = trial.params.get("ndvi_weight", 0)
            pval = trial.user_attrs.get("train_pvalue_mean", np.nan)
            report_lines.append(
                f"{rank:<6} #{trial.number:<7} {trial.value:<12.6f} {pval:<12.4e} {veg_w:<6} {ter_w:<6} {ndvi_w:<6}"
            )
        report_lines.append("")

        # Save robust trials report
        robust_report_path = os.path.join(robust_dir, "optimization_report.txt")
        with open(robust_report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        logger.info(f"✓ Robust trials report saved: {robust_report_path}")

        # Save best params
        robust_params_path = os.path.join(robust_dir, "best_params.json")
        with open(robust_params_path, "w") as f:
            json.dump(best_robust.params, f, indent=2)

        if progress_callback:
            progress_callback(30, 100)
        logger.info("GENERATING COMPREHENSIVE PLOTS FOR ROBUST TRIALS ONLY...")
        logger.info(f"  (Based on {len(robust_study.trials)} filtered trials)")
        self._generate_all_optuna_plots(
            robust_study, robust_dir, study_name="Robust Trials"
        )

        if progress_callback:
            progress_callback(50, 100)

        # ═══════════════════════════════════════════════════════════════════
        # SECONDARY ANALYSIS: ALL TRIALS (Debug Only)
        # ═══════════════════════════════════════════════════════════════════

        all_trials_dir = os.path.join(output_dir, "all_trials")
        os.makedirs(all_trials_dir, exist_ok=True)

        logger.info("=" * 80)
        logger.info("GENERATING DEBUG REPORT: ALL TRIALS (INCLUDING NON-SIGNIFICANT)")
        logger.info(f"Study contains: {len(self.study.trials)} trials (UNFILTERED)")
        logger.info("=" * 80)

        logger.info("Generating debug report: ALL trials...")

        debug_lines = []
        debug_lines.append("=" * 80)
        debug_lines.append("FUSION OPTIMIZATION RESULTS - ALL TRIALS (DEBUG)")
        debug_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        debug_lines.append("=" * 80)
        debug_lines.append("")
        debug_lines.append("This report includes ALL trials for debugging purposes.")
        debug_lines.append("Use 'robust_trials' folder for primary analysis.")
        debug_lines.append("")
        debug_lines.append(f"Total Trials: {len(self.study.trials)}")
        debug_lines.append(f"Completed: {len(all_completed_trials)}")
        debug_lines.append(
            f"Pruned: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED])}"
        )
        debug_lines.append(
            f"Failed: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL])}"
        )
        debug_lines.append("")

        best_all = self.study.best_trial
        debug_lines.append(f"BEST TRIAL (#{best_all.number})")
        debug_lines.append("-" * 80)
        debug_lines.append(f"Best Value: {best_all.value:.6f}")
        debug_lines.append(f"Parameters: {best_all.params}")
        debug_lines.append("")

        # Top 10 all trials
        debug_lines.append("TOP 10 TRIALS")
        debug_lines.append("-" * 80)
        sorted_all = sorted(
            all_completed_trials,
            key=lambda t: t.value,
            reverse=(self.study.direction.name == "MAXIMIZE"),
        )[:10]
        debug_lines.append(
            f"{'Rank':<6} {'Trial':<8} {'Value':<12} {'Veg%':<6} {'Ter%':<6} {'NDVI%':<6}"
        )
        debug_lines.append("-" * 80)
        for rank, trial in enumerate(sorted_all, 1):
            veg_w = trial.params.get("veg_weight", 0)
            ter_w = trial.params.get("terrain_weight", 0)
            ndvi_w = trial.params.get("ndvi_weight", 0)
            debug_lines.append(
                f"{rank:<6} #{trial.number:<7} {trial.value:<12.6f} {veg_w:<6} {ter_w:<6} {ndvi_w:<6}"
            )
        debug_lines.append("")

        all_report_path = os.path.join(all_trials_dir, "optimization_report.txt")
        with open(all_report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(debug_lines))
        logger.info(f"✓ All trials debug report saved: {all_report_path}")

        if progress_callback:
            progress_callback(60, 100)

        # Generate plots for all trials
        if include_plots:
            logger.info("Generating debug plots for ALL TRIALS (unfiltered)...")
            logger.info(f"  (Based on {len(self.study.trials)} total trials)")
            self._generate_all_optuna_plots(
                self.study, all_trials_dir, study_name="All Trials (Debug)"
            )

        if progress_callback:
            progress_callback(75, 100)

        # ═══ PRINT SUMMARY TO CONSOLE ═══
        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPLETE - SUMMARY")
        print("=" * 80)
        print(f"Total Trials: {len(all_completed_trials)}")
        print(
            f"Robust Trials (FDR p<0.05): {len(robust_trials)} ({len(robust_trials)/len(all_completed_trials)*100:.1f}%)"
        )
        print("")
        print(f"BEST ROBUST TRIAL: #{best_robust.number} = {best_robust.value:.6f}")
        print(f"  Parameters: {best_robust.params}")
        print("")
        print(f"Reports saved to: {output_dir}")
        print(f"  ✓ PRIMARY (robust trials): {robust_dir}/")
        print(f"  ✓ DEBUG (all trials):  : {best_robust.params}")
        print("")
        print(f"Reports saved to: {output_dir}")
        print(f"  - PRIMARY: {robust_dir}/")
        print(f"  - DEBUG:   {all_trials_dir}/")
        print("=" * 80 + "\n")

        if progress_callback:
            progress_callback(85, 100)

        # ═══ GENERATE COMPOSITE GREENERY MAP ═══
        logger.info("Generating composite greenery map from top 20% robust trials...")
        averaged_params: dict[str, Any] | None = None
        try:
            # Include the active greenery channel in the filename so
            # standalone runs (veg / terrain / ndvi) don't overwrite the
            # combined CGI composite TIFF. When the caller doesn't pass
            # an explicit path, fall back to writing one directory above
            # ``output_dir`` (legacy layout).
            ch = getattr(self, "_active_greenery_channel", "cgi") or "cgi"
            suffix = "" if ch == "cgi" else f"_{ch}"
            if composite_path is None:
                composite_path = os.path.join(
                    output_dir, f"../composite_greenery{suffix}.tif"
                )
            self.generate_composite_greenery_map(
                output_path=composite_path,
                top_percent=0.2,
                progress_callback=None,  # Nested progress not supported yet
            )
            averaged_params = self.compute_averaged_top_params(top_percent=0.2)
        except Exception as e:
            logger.warning(f"Could not generate composite greenery map: {e}")

        if progress_callback:
            progress_callback(100, 100)

        logger.info("✓ Results report generation complete!")
        return averaged_params
