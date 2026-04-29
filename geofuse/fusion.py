"""
Fusion Module: Optimized metric fusion for geospatial composite indices.

This module implements the optimization logic from CGI.ipynb for tuning
weighted combinations of NDVI and GVI metrics against target outcomes.
"""

import hashlib
import logging
import os

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
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class MetricFusionEngine:
    """
    Engine for fusing vegetation, terrain, and NDVI metrics using optimization.

    Based on the methodology in CGI.ipynb, this class:
    1. Loads vegetation (GVI veg/terrain) and NDVI metrics
    2. Aligns them spatially with a target outcome (GeoJSON points or GeoTIFF)
    3. Splits data: holdout test set + k-fold CV on training data
    4. Optimizes 9 parameters matching CGI.ipynb:
       - Weights: veg_weight, terrain_weight, ndvi_weight (0-100, sum=100)
       - Radii: veg_radius, terrain_radius, ndvi_radius (50-1500m)
       - Streetview agg: streetview_stat, streetview_percentile (shared for veg+terrain)
       - NDVI agg: ndvi_stat, ndvi_percentile (separate for NDVI)
    5. Validates performance using cross-validation and held-out test set
    """

    def __init__(
        self,
        target_file: str,
        target_feature: str | None = None,
        target_band: int = 1,
        buffer_meters: float = 1500.0,
        n_bins: int = 5,
        cache_dir: str = "output_results/fusion_cache",
    ):
        """
        Initialize the fusion engine.

        Args:
            target_file: Path to GeoJSON (points) or GeoTIFF (raster) target file
            target_feature: For GeoJSON, the column name to optimize towards
            target_band: For GeoTIFF, the band number to optimize towards
            buffer_meters: Buffer distance around target geometry for metric sampling
            n_bins: Number of bins for stratified splitting
            cache_dir: Directory to cache downloaded metrics
        """
        self.target_file = target_file
        self.target_feature = target_feature
        self.target_band = target_band
        self.buffer_meters = buffer_meters
        self.n_bins = n_bins
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Data containers
        self.target_gdf = None
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

        # Determine input type
        self.is_points = target_file.lower().endswith((".geojson", ".shp"))
        self.is_raster = target_file.lower().endswith((".tif", ".tiff"))

        if not (self.is_points or self.is_raster):
            raise ValueError("Target file must be GeoJSON/Shapefile or GeoTIFF")

    def load_target(self) -> gpd.GeoDataFrame:
        """Load and prepare target data, return buffered extent."""
        logger.info(f"Loading target file: {self.target_file}")

        if self.is_points:
            self.target_gdf = gpd.read_file(self.target_file)
            if self.target_gdf.crs is None:
                self.target_gdf.set_crs("EPSG:4326", inplace=True)

            # Validate feature exists
            if (
                self.target_feature
                and self.target_feature not in self.target_gdf.columns
            ):
                raise ValueError(
                    f"Target feature '{self.target_feature}' not found in columns: {list(self.target_gdf.columns)}"
                )

            # Create buffered extent for metric download
            gdf_utm = self.target_gdf.to_crs("EPSG:32612")  # UTM for meter-based buffer
            bounds = gdf_utm.total_bounds
            buffered_box = box(
                bounds[0] - self.buffer_meters,
                bounds[1] - self.buffer_meters,
                bounds[2] + self.buffer_meters,
                bounds[3] + self.buffer_meters,
            )
            self.buffered_extent = gpd.GeoDataFrame(
                {"geometry": [buffered_box]}, crs="EPSG:32612"
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
        progress_callback: callable | None = None,
        cancel_callback: callable | None = None,
        force_download: bool = False,
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
        """
        if self.buffered_extent is None:
            self.load_target()

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

        # Load or Auto-download Vegetation and Terrain
        # Priority: 1) Uploaded multi-band file, 2) Cached multi-band, 3) Separate files, 4) Auto-download

        # Check if uploaded veg_file is a multi-band raster
        if (
            veg_file
            and os.path.exists(veg_file)
            and veg_file.endswith((".tif", ".tiff"))
        ):
            if load_multiband_gvi(veg_file):
                # Successfully loaded both veg and terrain from uploaded file
                pass  # veg_data and terrain_data already set
            else:
                # Not multi-band, load normally below
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
        # Check cached multi-band file
        elif os.path.exists(gvi_multiband_cache):
            if load_multiband_gvi(gvi_multiband_cache):
                # Successfully loaded from cache
                pass
            else:
                # Cache corrupted, re-download
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

        # Continue with NDVI loading (unchanged)
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
            )
            self.ndvi_data = self._load_metric_file(ndvi_file)

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
                return {
                    "data": src.read(1, masked=True),
                    "transform": src.transform,
                    "crs": src.crs,
                    "bounds": src.bounds,
                }
        else:
            gdf = gpd.read_file(filepath)
            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)

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

        # Write to GeoTIFF
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
            compress="lzw",
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

        # Write multi-band GeoTIFF
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
            compress="lzw",
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
        progress_callback: callable | None = None,
        cancel_callback: callable | None = None,
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
            logger.info("GVI will generate grid at 75m spacing within this polygon")

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
            step=75,  # 75m grid spacing (matches default)
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
        progress_callback: callable | None = None,
        cancel_callback: callable | None = None,
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
            logger.info("GVI will generate grid at 10m spacing within this polygon")

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
            step=75,  # 75m grid spacing (matches default)
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
    ) -> str:
        """
        Auto-download NDVI metrics within buffered extent.

        Args:
            start_date: Start date for temporal composite (YYYY-MM-DD)
            end_date: End date for temporal composite (YYYY-MM-DD)
            project_id: Google Earth Engine project ID
            cache: Whether to save to cache directory
            force_download: If True, bypass cache and force fresh download

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
        logger.info(f"Downloading NDVI from Sentinel-2 ({start_date} to {end_date})...")
        logger.info(f"Area extent: {bounds}")
        logger.info("This may take several minutes depending on area size...")

        # Extract base name for output
        filename = os.path.basename(self.target_file)
        name, _ = os.path.splitext(filename)

        result = ndvi_engine.download_and_process(
            geometry=self.buffered_extent,
            start_date=start_date,
            end_date=end_date,
            output_name=f"{name if cache else 'temp'}",
            folder=self.cache_dir,
            resolution=10,
        )

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

    def prepare_fusion_data(self) -> pd.DataFrame:
        """
        Align vegetation, terrain, NDVI, and target data into a single DataFrame.

        Follows CGI.ipynb logic:
        - For point targets: sample metrics at each point location
        - For raster targets: align all data to target grid

        Returns:
            DataFrame with columns: [target, veg, terrain, ndvi]
        """
        import sys

        print("\n[FUSION DEBUG] ====== PREPARE FUSION DATA ======", flush=True)
        print(f"[FUSION DEBUG] is_points = {self.is_points}", flush=True)
        print(
            f"[FUSION DEBUG] Target type: {'POINT' if self.is_points else 'RASTER'}",
            flush=True,
        )
        sys.stdout.flush()

        if self.is_points:
            return self._prepare_point_fusion()
        else:
            return self._prepare_raster_fusion()

    def _prepare_point_fusion(self) -> pd.DataFrame:
        """Sample metrics at point locations."""
        import sys

        print("\n[FUSION DEBUG] ====== POINT FUSION ======", flush=True)
        logger.info("Preparing point-based fusion data...")
        print("[FUSION DEBUG] Preparing point-based fusion data...", flush=True)
        sys.stdout.flush()

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

        # Log data quality before dropping NaN
        print("\n[FUSION DEBUG] ====== DATA QUALITY SUMMARY (POINT) ======", flush=True)
        print(f"[FUSION DEBUG] Total rows: {len(fusion_df)}", flush=True)
        print(
            f"[FUSION DEBUG] Target NaN: {fusion_df['target'].isna().sum()} ({fusion_df['target'].isna().sum()/len(fusion_df)*100:.1f}%)",
            flush=True,
        )
        print(
            f"[FUSION DEBUG] Veg NaN: {fusion_df['veg'].isna().sum()} ({fusion_df['veg'].isna().sum()/len(fusion_df)*100:.1f}%)",
            flush=True,
        )
        print(
            f"[FUSION DEBUG] Terrain NaN: {fusion_df['terrain'].isna().sum()} ({fusion_df['terrain'].isna().sum()/len(fusion_df)*100:.1f}%)",
            flush=True,
        )
        print(
            f"[FUSION DEBUG] NDVI NaN: {fusion_df['ndvi'].isna().sum()} ({fusion_df['ndvi'].isna().sum()/len(fusion_df)*100:.1f}%)",
            flush=True,
        )
        print("[FUSION DEBUG] ================================\n", flush=True)

        result = fusion_df.dropna()
        print(f"[FUSION DEBUG] After dropna: {len(result)} valid rows", flush=True)

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
            # Reproject points to vegetation CRS
            points_in_veg_crs = points_gdf.to_crs(self.veg_data["crs"])

            for idx, point in points_in_veg_crs.iterrows():
                row, col = rowcol(
                    self.veg_data["transform"], point.geometry.x, point.geometry.y
                )
                if (
                    0 <= row < self.veg_data["data"].shape[0]
                    and 0 <= col < self.veg_data["data"].shape[1]
                ):
                    points_gdf.loc[idx, "veg"] = self.veg_data["data"][row, col]
        else:  # GeoDataFrame
            # Detect metric column
            veg_col = self.veg_data.attrs.get("metric_column", "gvi")
            if veg_col not in self.veg_data.columns:
                # Fallback to common patterns
                for col in ["gvi", "gvi_veg", "veg", "GVI", "value"]:
                    if col in self.veg_data.columns:
                        veg_col = col
                        break

            print(
                f"[FUSION DEBUG] Veg data: {len(self.veg_data)} features, columns: {self.veg_data.columns.tolist()}",
                flush=True,
            )
            print(f"[FUSION DEBUG] Using veg column: '{veg_col}'", flush=True)
            print(
                f"[FUSION DEBUG] Veg CRS: {self.veg_data.crs}, Points CRS: {points_gdf.crs}",
                flush=True,
            )
            print(
                f"[FUSION DEBUG] Veg bounds: {self.veg_data.total_bounds}", flush=True
            )
            print(
                f"[FUSION DEBUG] Points bounds: {points_gdf.total_bounds}", flush=True
            )
            print(f"[FUSION DEBUG] Buffer distance: {self.buffer_meters}m", flush=True)

            # Ensure CRS match before spatial join
            veg_data_matched = self.veg_data.to_crs(points_gdf.crs)
            print(
                f"[FUSION DEBUG] Reprojected veg data to {points_gdf.crs}", flush=True
            )

            # Spatial join nearest (within buffer distance)
            points_with_veg = gpd.sjoin_nearest(
                points_gdf,
                veg_data_matched[["geometry", veg_col]],
                how="left",
                max_distance=self.buffer_meters,
            )

            # Extract the metric column (may have been renamed with suffix)
            if veg_col in points_with_veg.columns:
                points_gdf["veg"] = points_with_veg[veg_col]
            elif f"{veg_col}_right" in points_with_veg.columns:
                points_gdf["veg"] = points_with_veg[f"{veg_col}_right"]
            else:
                print(
                    f"[FUSION DEBUG] WARNING: Could not find veg column. Available: {points_with_veg.columns.tolist()}",
                    flush=True,
                )
                points_gdf["veg"] = np.nan

            veg_valid = points_gdf["veg"].notna().sum()
            print(
                f"[FUSION DEBUG] Veg sampling: {veg_valid}/{len(points_gdf)} points have valid values",
                flush=True,
            )

        # Sample Terrain
        if isinstance(self.terrain_data, dict):  # Raster
            points_in_terrain_crs = points_gdf.to_crs(self.terrain_data["crs"])

            for idx, point in points_in_terrain_crs.iterrows():
                row, col = rowcol(
                    self.terrain_data["transform"], point.geometry.x, point.geometry.y
                )
                if (
                    0 <= row < self.terrain_data["data"].shape[0]
                    and 0 <= col < self.terrain_data["data"].shape[1]
                ):
                    points_gdf.loc[idx, "terrain"] = self.terrain_data["data"][row, col]
        else:  # GeoDataFrame
            # Detect metric column
            terrain_col = self.terrain_data.attrs.get("metric_column", "terrain")
            if terrain_col not in self.terrain_data.columns:
                # Fallback to common patterns
                for col in ["terrain", "gvi_ter", "NDVI", "ndvi", "value"]:
                    if col in self.terrain_data.columns:
                        terrain_col = col
                        break

            print(
                f"[FUSION DEBUG] Terrain data: {len(self.terrain_data)} features, columns: {self.terrain_data.columns.tolist()}",
                flush=True,
            )
            print(f"[FUSION DEBUG] Using terrain column: '{terrain_col}'", flush=True)

            # Ensure CRS match before spatial join
            terrain_data_matched = self.terrain_data.to_crs(points_gdf.crs)

            # Spatial join nearest (within buffer distance)
            points_with_terrain = gpd.sjoin_nearest(
                points_gdf,
                terrain_data_matched[["geometry", terrain_col]],
                how="left",
                max_distance=self.buffer_meters,
            )

            # Extract the metric column (may have been renamed with suffix)
            if terrain_col in points_with_terrain.columns:
                points_gdf["terrain"] = points_with_terrain[terrain_col]
            elif f"{terrain_col}_right" in points_with_terrain.columns:
                points_gdf["terrain"] = points_with_terrain[f"{terrain_col}_right"]
            else:
                points_gdf["terrain"] = np.nan

            terrain_valid = points_gdf["terrain"].notna().sum()
            print(
                f"[FUSION DEBUG] Terrain sampling: {terrain_valid}/{len(points_gdf)} points have valid values",
                flush=True,
            )

        # Sample NDVI
        points_gdf["ndvi"] = np.nan
        if isinstance(self.ndvi_data, dict):  # Raster
            points_in_ndvi_crs = points_gdf.to_crs(self.ndvi_data["crs"])

            for idx, point in points_in_ndvi_crs.iterrows():
                row, col = rowcol(
                    self.ndvi_data["transform"], point.geometry.x, point.geometry.y
                )
                if (
                    0 <= row < self.ndvi_data["data"].shape[0]
                    and 0 <= col < self.ndvi_data["data"].shape[1]
                ):
                    points_gdf.loc[idx, "ndvi"] = self.ndvi_data["data"][row, col]
        else:  # GeoDataFrame
            # Detect metric column
            ndvi_col = self.ndvi_data.attrs.get("metric_column", "NDVI")
            if ndvi_col not in self.ndvi_data.columns:
                # Fallback to common patterns
                for col in ["NDVI", "ndvi", "value"]:
                    if col in self.ndvi_data.columns:
                        ndvi_col = col
                        break

            print(
                f"[FUSION DEBUG] NDVI data: {len(self.ndvi_data)} features, columns: {self.ndvi_data.columns.tolist()}",
                flush=True,
            )
            print(f"[FUSION DEBUG] Using NDVI column: '{ndvi_col}'", flush=True)

            # Ensure CRS match before spatial join
            ndvi_data_matched = self.ndvi_data.to_crs(points_gdf.crs)

            # Spatial join nearest (within buffer distance)
            points_with_ndvi = gpd.sjoin_nearest(
                points_gdf,
                ndvi_data_matched[["geometry", ndvi_col]],
                how="left",
                max_distance=self.buffer_meters,
            )

            # Extract the metric column (may have been renamed with suffix)
            if ndvi_col in points_with_ndvi.columns:
                points_gdf["ndvi"] = points_with_ndvi[ndvi_col]
            elif f"{ndvi_col}_right" in points_with_ndvi.columns:
                points_gdf["ndvi"] = points_with_ndvi[f"{ndvi_col}_right"]
            else:
                points_gdf["ndvi"] = np.nan

            ndvi_valid = points_gdf["ndvi"].notna().sum()
            print(
                f"[FUSION DEBUG] NDVI sampling: {ndvi_valid}/{len(points_gdf)} points have valid values",
                flush=True,
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

        # Log data quality before dropping NaN
        print(
            "\n[FUSION DEBUG] ====== DATA QUALITY SUMMARY (RASTER) ======", flush=True
        )
        print(f"[FUSION DEBUG] Total rows: {len(fusion_df)}", flush=True)
        print(
            f"[FUSION DEBUG] Target NaN: {fusion_df['target'].isna().sum()} ({fusion_df['target'].isna().sum()/len(fusion_df)*100:.1f}%)",
            flush=True,
        )
        print(
            f"[FUSION DEBUG] Veg NaN: {fusion_df['veg'].isna().sum()} ({fusion_df['veg'].isna().sum()/len(fusion_df)*100:.1f}%)",
            flush=True,
        )
        print(
            f"[FUSION DEBUG] Terrain NaN: {fusion_df['terrain'].isna().sum()} ({fusion_df['terrain'].isna().sum()/len(fusion_df)*100:.1f}%)",
            flush=True,
        )
        print(
            f"[FUSION DEBUG] NDVI NaN: {fusion_df['ndvi'].isna().sum()} ({fusion_df['ndvi'].isna().sum()/len(fusion_df)*100:.1f}%)",
            flush=True,
        )
        print("[FUSION DEBUG] ================================\n", flush=True)

        result = fusion_df.dropna()
        print(f"[FUSION DEBUG] After dropna: {len(result)} valid rows", flush=True)

        if len(result) == 0:
            raise ValueError(
                "No valid samples after spatial join. "
                "This usually means the target raster pixels and metric data don't overlap spatially. "
                "Check that all data covers the same geographic area."
            )

        return result

    def split_data(
        self, test_size: float = 0.2, k_folds: int = 5, random_state: int = 42
    ) -> None:
        """
        Split data into holdout test set and k-fold CV training/validation sets.

        Workflow:
        1. Sample all metrics at point locations (initial sampling with buffer_meters)
        2. Filter out rows with NaN values in any metric
        3. Bin target values for stratification
        4. Stratified split into train/val/test sets

        During optimization, metrics will be re-sampled with trial-specific radii
        using circular buffer aggregation.

        Args:
            test_size: Proportion for holdout test set (e.g., 0.2 = 20%)
            k_folds: Number of cross-validation folds (e.g., 5)
            random_state: Random seed for reproducibility
        """
        # Step 1: Sample all metrics at initial buffer distance
        logger.info("Step 1/4: Sampling metrics at point locations...")
        fusion_df = self.prepare_fusion_data()
        self.k_folds = k_folds

        logger.info(f"Initial samples before filtering: {len(fusion_df)}")

        # Step 2: Filter NaN values (already done in prepare_fusion_data with dropna)
        logger.info("Step 2/4: Filtering complete - removed samples with NaN values")
        logger.info(f"Valid samples after filtering: {len(fusion_df)}")

        # Step 3: Create stratification bins based on target values
        logger.info("Step 3/4: Binning target values for stratified sampling...")
        fusion_df["target_bin"] = pd.qcut(
            fusion_df["target"], q=self.n_bins, labels=False, duplicates="drop"
        )
        logger.info(
            f"Created {fusion_df['target_bin'].nunique()} bins for stratification"
        )

        # Step 4: Stratified split into train/val and test sets
        logger.info("Step 4/4: Performing stratified train/test split...")
        self.train_val_data, self.test_data = train_test_split(
            fusion_df,
            test_size=test_size,
            stratify=fusion_df["target_bin"],
            random_state=random_state,
        )

        logger.info(
            f"Split complete: {len(self.train_val_data)} train+val samples ({k_folds} folds), "
            f"{len(self.test_data)} test samples (holdout)"
        )

        # Create stratified k-fold splits on train_val_data
        logger.info(f"Creating {k_folds}-fold cross-validation splits...")
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
        self.cv_folds = []

        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(self.train_val_data, self.train_val_data["target_bin"]), 1
        ):
            train_fold = self.train_val_data.iloc[train_idx].copy()
            val_fold = self.train_val_data.iloc[val_idx].copy()

            # Normalize features (0-1) using training fold data
            scaler = MinMaxScaler()
            train_fold[["veg", "terrain", "ndvi"]] = scaler.fit_transform(
                train_fold[["veg", "terrain", "ndvi"]]
            )
            val_fold[["veg", "terrain", "ndvi"]] = scaler.transform(
                val_fold[["veg", "terrain", "ndvi"]]
            )

            self.cv_folds.append(
                {"train": train_fold, "val": val_fold, "scaler": scaler}
            )

            logger.info(
                f"  Fold {fold_idx}: {len(train_fold)} train, {len(val_fold)} val"
            )

    def optimize_fusion(
        self,
        n_trials: int = 300,
        n_startup_trials: int = 150,
        objective_metric: str = "pearson",
        pruner_type: str = "median",
        sampler_type: str = "TPE",
        seed: int = 42,
        show_progress: bool = True,
        progress_callback: callable | None = None,
    ) -> dict:
        """
        Run Optuna optimization with k-fold cross-validation.

        Args:
            n_trials: Total optimization trials
            n_startup_trials: Random exploration trials before the main optimizer
            objective_metric: 'pearson', 'spearman', 'r2', 'rmse', 'mutual_info'
            pruner_type: 'median', 'hyperband', 'successive_halving', or None
            sampler_type: 'TPE', 'CMA-ES', or 'Random'
            seed: Random seed for reproducibility
            show_progress: Whether to show progress bar

        Returns:
            Best parameters dictionary
        """
        if self.cv_folds is None:
            raise ValueError("Call split_data() first")

        # Build sampler
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

        # Determine optimization direction
        direction = "minimize" if objective_metric == "rmse" else "maximize"

        # Create study
        self.study = optuna.create_study(
            direction=direction, sampler=sampler, pruner=pruner
        )

        # Run optimization
        logger.info(
            f"Starting {self.k_folds}-fold CV optimization: {n_trials} trials, "
            f"{objective_metric} metric"
        )

        # Create callback for progress tracking
        if progress_callback:

            def optuna_callback(study, trial):
                # Update progress: trial number / total trials
                progress_callback(trial.number + 1, n_trials)

        else:
            optuna_callback = None

        self.study.optimize(
            lambda trial: self._objective(trial, objective_metric),
            n_trials=n_trials,
            show_progress_bar=show_progress,
            callbacks=[optuna_callback] if optuna_callback else None,
        )

        self.best_params = self.study.best_params
        logger.info(
            f"Optimization complete. Best {objective_metric} (CV avg): {self.study.best_value:.4f}"
        )

        # Auto-generate results report with progress updates
        logger.info("Generating optimization results report...")
        try:
            report_dir = os.path.join(
                os.getcwd(), "output_results", "fusion", "study_results"
            )
            self.generate_results_report(
                output_dir=report_dir,
                include_plots=True,
                progress_callback=progress_callback,
            )
            logger.info(f"Results report saved to: {report_dir}")
        except Exception as e:
            logger.warning(f"Could not generate results report: {e}")
            import traceback

            traceback.print_exc()

        return self.best_params

    def _objective(self, trial: optuna.Trial, metric: str) -> float:
        """
        Optuna objective function with k-fold CV.

        Matches CGI.ipynb parameter structure exactly:
        - ndvi_weight, veg_weight, terrain_weight (0-100, summing to 100)
        - veg_radius, terrain_radius, ndvi_radius (100 to buffer_meters, step=50)
        - streetview_stat, streetview_percentile (SHARED for veg + terrain)
        - ndvi_stat, ndvi_percentile (separate for NDVI)

        Evaluates across all CV folds and returns average validation score.
        """
        # Log buffer_meters to verify it's being used correctly (only for first trial)
        if trial.number == 0:
            logger.info(f"Buffer distance for optimization: {self.buffer_meters}m")
            logger.info(
                f"Radius range will be: 100 to {int(self.buffer_meters)}m (step=50)"
            )

        # ─── Suggest Weights (matching CGI.ipynb logic) ───────────────────────
        ndvi_weight = trial.suggest_int("ndvi_weight", 0, 100)
        veg_weight = (
            trial.suggest_int("veg_weight", 0, 100 - ndvi_weight)
            if ndvi_weight < 100
            else trial.suggest_int("veg_weight", 0, 0)
        )
        terrain_weight = (
            trial.suggest_int(
                "terrain_weight",
                100 - ndvi_weight - veg_weight,
                100 - ndvi_weight - veg_weight,
            )
            if (ndvi_weight + veg_weight) < 100
            else trial.suggest_int("terrain_weight", 0, 0)
        )

        # ─── Suggest Streetview Parameters (SHARED for veg + terrain) ─────────
        # Only suggest if either veg or terrain has weight > 0
        if veg_weight > 0 or terrain_weight > 0:
            streetview_stat = trial.suggest_categorical(
                "streetview_stat", ["mean", "median", "percentile"]
            )
            if streetview_stat == "percentile":
                streetview_percentile = trial.suggest_int(
                    "streetview_percentile", 1, 99
                )
            else:
                streetview_percentile = 50

            # Separate radii for veg and terrain (min 100m to avoid spatial join mismatches)
            # Max radius is user-specified buffer_meters
            if veg_weight > 0:
                veg_radius = trial.suggest_int(
                    "veg_radius", 100, int(self.buffer_meters), step=50
                )
            else:
                veg_radius = self.buffer_meters

            if terrain_weight > 0:
                terrain_radius = trial.suggest_int(
                    "terrain_radius", 100, int(self.buffer_meters), step=50
                )
            else:
                terrain_radius = self.buffer_meters
        else:
            streetview_stat = "mean"
            streetview_percentile = 50
            veg_radius = self.buffer_meters
            terrain_radius = self.buffer_meters

        # ─── Suggest NDVI Parameters (separate) ────────────────────────────────
        # Max radius is user-specified buffer_meters
        if ndvi_weight > 0:
            ndvi_radius = trial.suggest_int(
                "ndvi_radius", 100, int(self.buffer_meters), step=50
            )
            ndvi_stat = trial.suggest_categorical(
                "ndvi_stat", ["mean", "median", "percentile"]
            )
            if ndvi_stat == "percentile":
                ndvi_percentile = trial.suggest_int("ndvi_percentile", 1, 99)
            else:
                ndvi_percentile = 50
        else:
            ndvi_radius = self.buffer_meters
            ndvi_stat = "mean"
            ndvi_percentile = 50

        # ─── Evaluate Across All CV Folds ─────────────────────────────────────
        fold_train_scores = []
        fold_val_scores = []
        fold_train_pvals = []
        fold_val_pvals = []

        for fold_idx, fold in enumerate(self.cv_folds):
            train_data = fold["train"]
            val_data = fold["val"]

            # Normalize weights to 0-1 range
            total = veg_weight + terrain_weight + ndvi_weight
            if total == 0:
                return -np.inf if metric != "rmse" else np.inf

            veg_w = veg_weight / total
            terrain_w = terrain_weight / total
            ndvi_w = ndvi_weight / total

            # ─── Apply Dynamic Radius and Aggregation ─────────────────────────────
            # Both points and rasters use the same circular buffer aggregation
            # For rasters, _prepare_raster_fusion() converted pixels to points at centers
            train_points = self.target_gdf.loc[train_data.index].copy()
            val_points = self.target_gdf.loc[val_data.index].copy()

            # Apply circular buffer aggregation for vegetation (with SHARED streetview_stat)
            if veg_weight > 0:
                train_veg = self._apply_circular_buffer_aggregation(
                    train_points,
                    self.veg_data,
                    veg_radius,
                    streetview_stat,  # SHARED
                    streetview_percentile,  # SHARED
                )
                val_veg = self._apply_circular_buffer_aggregation(
                    val_points,
                    self.veg_data,
                    veg_radius,
                    streetview_stat,  # SHARED
                    streetview_percentile,  # SHARED
                )
            else:
                train_veg = np.zeros(len(train_points))
                val_veg = np.zeros(len(val_points))

            # Apply circular buffer aggregation for terrain (with SHARED streetview_stat)
            if terrain_weight > 0:
                train_terrain = self._apply_circular_buffer_aggregation(
                    train_points,
                    self.terrain_data,
                    terrain_radius,
                    streetview_stat,  # SHARED
                    streetview_percentile,  # SHARED
                )
                val_terrain = self._apply_circular_buffer_aggregation(
                    val_points,
                    self.terrain_data,
                    terrain_radius,
                    streetview_stat,  # SHARED
                    streetview_percentile,  # SHARED
                )
            else:
                train_terrain = np.zeros(len(train_points))
                val_terrain = np.zeros(len(val_points))

            # Apply circular buffer aggregation for NDVI (separate stat)
            if ndvi_weight > 0:
                train_ndvi = self._apply_circular_buffer_aggregation(
                    train_points,
                    self.ndvi_data,
                    ndvi_radius,
                    ndvi_stat,  # SEPARATE
                    ndvi_percentile,  # SEPARATE
                )
                val_ndvi = self._apply_circular_buffer_aggregation(
                    val_points,
                    self.ndvi_data,
                    ndvi_radius,
                    ndvi_stat,  # SEPARATE
                    ndvi_percentile,  # SEPARATE
                )
            else:
                train_ndvi = np.zeros(len(train_points))
                val_ndvi = np.zeros(len(val_points))

            # Normalize sampled values (0-1) using fold's scaler
            scaler = MinMaxScaler()
            train_combined = np.column_stack([train_veg, train_terrain, train_ndvi])
            val_combined = np.column_stack([val_veg, val_terrain, val_ndvi])

            # Remove NaN rows before fitting scaler
            train_valid_mask = ~np.isnan(train_combined).any(axis=1)
            val_valid_mask = ~np.isnan(val_combined).any(axis=1)

            if train_valid_mask.sum() == 0:
                continue  # Skip this fold if no valid data

            train_combined[train_valid_mask] = scaler.fit_transform(
                train_combined[train_valid_mask]
            )
            if val_valid_mask.sum() > 0:
                val_combined[val_valid_mask] = scaler.transform(
                    val_combined[val_valid_mask]
                )

            train_veg_norm = train_combined[:, 0]
            train_terrain_norm = train_combined[:, 1]
            train_ndvi_norm = train_combined[:, 2]
            val_veg_norm = val_combined[:, 0]
            val_terrain_norm = val_combined[:, 1]
            val_ndvi_norm = val_combined[:, 2]

            # Calculate composite index with normalized values
            train_composite = (
                train_veg_norm * veg_w
                + train_terrain_norm * terrain_w
                + train_ndvi_norm * ndvi_w
            )
            val_composite = (
                val_veg_norm * veg_w
                + val_terrain_norm * terrain_w
                + val_ndvi_norm * ndvi_w
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

            # Calculate metrics
            train_score = self._calculate_metric(
                train_data["target"].values, train_composite, metric
            )
            val_score = self._calculate_metric(
                val_data["target"].values, val_composite, metric
            )

            fold_train_scores.append(train_score)
            fold_val_scores.append(val_score)

            # Calculate p-values for correlation metrics
            if metric in ["pearson", "spearman"]:
                import warnings

                from scipy.stats import ConstantInputWarning

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    warnings.filterwarnings("ignore", category=ConstantInputWarning)
                    if metric == "pearson":
                        _, train_pval = pearsonr(
                            train_data["target"].values, train_composite
                        )
                        _, val_pval = pearsonr(val_data["target"].values, val_composite)
                    else:
                        _, train_pval = spearmanr(
                            train_data["target"].values, train_composite
                        )
                        _, val_pval = spearmanr(
                            val_data["target"].values, val_composite
                        )

                    fold_train_pvals.append(train_pval)
                    fold_val_pvals.append(val_pval)

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

        if metric in ["pearson", "spearman"]:
            trial.set_user_attr("train_pvalue_mean", np.mean(fold_train_pvals))
            trial.set_user_attr("val_pvalue_mean", np.mean(fold_val_pvals))
            trial.set_user_attr("fold_train_pvals", fold_train_pvals)
            trial.set_user_attr("fold_val_pvals", fold_val_pvals)

        # Return average validation score across folds
        return avg_val_score

    def _calculate_metric(
        self, y_true: np.ndarray, y_pred: np.ndarray, metric: str
    ) -> float:
        """Calculate specified metric between target and composite."""
        import warnings

        from scipy.stats import ConstantInputWarning

        # Check for constant inputs before computing correlations
        if np.var(y_true) == 0 or np.var(y_pred) == 0:
            return 0.0  # Return worst score for constant inputs

        if metric == "pearson":
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=ConstantInputWarning)
                corr, _ = pearsonr(y_true, y_pred)
                if np.isnan(corr):
                    return 0.0
                return abs(corr)  # Return absolute correlation
        elif metric == "spearman":
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=ConstantInputWarning)
                corr, _ = spearmanr(y_true, y_pred)
                if np.isnan(corr):
                    return 0.0
                return abs(corr)
        elif metric == "r2":
            score = r2_score(y_true, y_pred)
            if np.isnan(score):
                return 0.0
            return score
        elif metric == "rmse":
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric == "mutual_info":
            # Bin data for MI calculation
            y_true_binned = pd.qcut(y_true, 10, labels=False, duplicates="drop")
            y_pred_binned = pd.qcut(y_pred, 10, labels=False, duplicates="drop")
            return mutual_info_score(y_true_binned, y_pred_binned)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def get_robust_trials(
        self,
        method: str = "auto",
        p_threshold: float = 0.05,
        tolerance: float = 0.1,
        min_trials: int = 10,
    ) -> list[optuna.Trial]:
        """
        Filter trials for robustness based on the optimization metric.

        For correlation metrics (pearson, spearman):
            - Uses Benjamini-Hochberg FDR correction (CGI.ipynb methodology)
            - Filters by corrected train p-value < threshold
            - Validates with test p-value < threshold

        For other metrics (r2, rmse, mutual_info):
            - Selects trials with test performance within tolerance of train
            - Filters for minimal train-test gap to avoid overfitting

        Args:
            method: 'auto' (detect from study), 'pvalue', or 'consistency'
            p_threshold: Significance threshold for correlation metrics
            tolerance: Max allowable gap between train/test for non-correlation metrics
            min_trials: Minimum number of trials to return

        Returns:
            List of robust trials
        """
        if self.study is None:
            raise ValueError("Run optimization first")

        completed_trials = [
            t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]

        if not completed_trials:
            logger.warning("No completed trials found")
            return []

        # Auto-detect method based on available user attributes
        if method == "auto":
            if "train_pvalue_mean" in completed_trials[0].user_attrs:
                method = "pvalue"
            else:
                method = "consistency"

        # Method 1: P-value based (correlation metrics)
        if method == "pvalue":
            trials_with_pvals = [
                t for t in completed_trials if "train_pvalue_mean" in t.user_attrs
            ]

            if not trials_with_pvals:
                logger.warning(
                    "No trials with p-values. Using consistency method instead."
                )
                return self.get_robust_trials(
                    method="consistency", tolerance=tolerance, min_trials=min_trials
                )

            try:
                from statsmodels.stats.multitest import multipletests

                has_statsmodels = True
            except ImportError:
                # Silently fall back to consistency method
                return self.get_robust_trials(
                    method="consistency", tolerance=tolerance, min_trials=min_trials
                )

            # Extract average p-values across folds
            train_pvals = [t.user_attrs["train_pvalue_mean"] for t in trials_with_pvals]

            # Benjamini-Hochberg FDR correction (CGI.ipynb methodology)
            _, corrected_pvals, _, _ = multipletests(train_pvals, method="fdr_bh")

            # Filter by corrected train p-value and validation p-value
            robust_trials = []
            for trial, corrected_pval in zip(trials_with_pvals, corrected_pvals):
                if (
                    corrected_pval < p_threshold
                    and trial.user_attrs.get("val_pvalue_mean", 1.0) < p_threshold
                ):
                    robust_trials.append(trial)

            logger.info(f"Found {len(robust_trials)} robust trials (p < {p_threshold})")

        # Method 2: Consistency based (non-correlation metrics)
        else:
            robust_trials = []
            for trial in completed_trials:
                train_score = trial.user_attrs.get("train_score_mean")
                val_score = trial.user_attrs.get("val_score_mean")
                val_std = trial.user_attrs.get("val_score_std", 0)

                if train_score is None or val_score is None:
                    continue

                # For maximization metrics (r2, correlation, mutual_info)
                if self.study.direction.name == "MAXIMIZE":
                    # Validation score should be reasonably close to train score
                    score_gap = abs(train_score - val_score)
                    # Val should not be too much worse than train, and low variance across folds
                    if (
                        score_gap <= tolerance
                        and val_score >= train_score * (1 - tolerance)
                        and val_std <= tolerance
                    ):
                        robust_trials.append(trial)

                # For minimization metrics (rmse)
                else:
                    # Val error should not be much higher than train error
                    score_gap = abs(val_score - train_score)
                    if (
                        score_gap <= tolerance
                        and val_score <= train_score * (1 + tolerance)
                        and val_std <= tolerance
                    ):
                        robust_trials.append(trial)

            logger.info(
                f"Found {len(robust_trials)} consistent trials (tolerance={tolerance})"
            )

        # If too few robust trials, relax criteria and return best performers
        if len(robust_trials) < min_trials:
            logger.warning(
                f"Only {len(robust_trials)} robust trials found (< {min_trials}). "
                f"Returning top {min_trials} by train score instead."
            )

            # Sort by validation score (descending for maximize, ascending for minimize)
            sorted_trials = sorted(
                completed_trials,
                key=lambda t: t.user_attrs.get(
                    "val_score_mean",
                    (
                        float("-inf")
                        if self.study.direction.name == "MAXIMIZE"
                        else float("inf")
                    ),
                ),
                reverse=(self.study.direction.name == "MAXIMIZE"),
            )
            return sorted_trials[:min_trials]

        return robust_trials

    def evaluate_on_test(
        self,
        params: dict | None = None,
        metric: str = "pearson",
        return_predictions: bool = False,
    ) -> dict:
        """
        Evaluate best parameters on held-out test set.

        This is the final validation step to ensure the optimized parameters
        generalize to unseen data. Reports performance metrics and optionally
        returns predictions for further analysis.

        Args:
            params: Parameters to evaluate (uses best_params if None)
            metric: Evaluation metric ('pearson', 'spearman', 'r2', 'rmse', 'mutual_info')
            return_predictions: Whether to include predictions in return dict

        Returns:
            Dictionary with:
                - test_score: Performance on test set
                - test_pvalue: P-value (for correlation metrics)
                - predictions: Test set predictions (if return_predictions=True)
                - targets: Test set targets (if return_predictions=True)
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

        # Extract weights
        veg_weight = params["veg_weight"]
        terrain_weight = params["terrain_weight"]
        ndvi_weight = params["ndvi_weight"]
        total = veg_weight + terrain_weight + ndvi_weight
        veg_w = veg_weight / total
        terrain_w = terrain_weight / total
        ndvi_w = ndvi_weight / total

        # Extract aggregation parameters
        streetview_stat = params.get("streetview_stat", "mean")
        streetview_percentile = params.get("streetview_percentile", 50)
        veg_radius = params.get("veg_radius", self.buffer_meters)
        terrain_radius = params.get("terrain_radius", self.buffer_meters)
        ndvi_stat = params.get("ndvi_stat", "mean")
        ndvi_percentile = params.get("ndvi_percentile", 50)
        ndvi_radius = params.get("ndvi_radius", self.buffer_meters)

        # Apply dynamic circular buffer aggregation
        # Both points and rasters use the same approach (rasters converted to points)
        test_points = self.target_gdf.loc[self.test_data.index].copy()

        # Sample vegetation with optimized radius/stat
        if veg_weight > 0:
            test_veg = self._apply_circular_buffer_aggregation(
                test_points,
                self.veg_data,
                veg_radius,
                streetview_stat,
                streetview_percentile,
            )
        else:
            test_veg = np.zeros(len(test_points))

        # Sample terrain with optimized radius/stat
        if terrain_weight > 0:
            test_terrain = self._apply_circular_buffer_aggregation(
                test_points,
                self.terrain_data,
                terrain_radius,
                streetview_stat,
                streetview_percentile,
            )
        else:
            test_terrain = np.zeros(len(test_points))

        # Sample NDVI with optimized radius/stat
        if ndvi_weight > 0:
            test_ndvi = self._apply_circular_buffer_aggregation(
                test_points,
                self.ndvi_data,
                ndvi_radius,
                ndvi_stat,
                ndvi_percentile,
            )
        else:
            test_ndvi = np.zeros(len(test_points))

        # Normalize using scaler fit on train+val data
        scaler = MinMaxScaler()
        train_val_combined = np.column_stack(
            [
                self.train_val_data["veg"].values,
                self.train_val_data["terrain"].values,
                self.train_val_data["ndvi"].values,
            ]
        )
        scaler.fit(train_val_combined)

        test_combined = np.column_stack([test_veg, test_terrain, test_ndvi])
        test_valid_mask = ~np.isnan(test_combined).any(axis=1)

        if test_valid_mask.sum() == 0:
            raise ValueError("No valid test data after aggregation")

        test_combined[test_valid_mask] = scaler.transform(
            test_combined[test_valid_mask]
        )

        test_veg_norm = test_combined[:, 0]
        test_terrain_norm = test_combined[:, 1]
        test_ndvi_norm = test_combined[:, 2]

        # Calculate composite
        test_composite = (
            test_veg_norm * veg_w
            + test_terrain_norm * terrain_w
            + test_ndvi_norm * ndvi_w
        )

        # Calculate test score
        test_targets = self.test_data["target"].values
        test_score = self._calculate_metric(test_targets, test_composite, metric)

        # Calculate p-value for correlation metrics
        result = {"test_score": test_score, "metric": metric}

        if metric in ["pearson", "spearman"]:
            if metric == "pearson":
                _, test_pval = pearsonr(test_targets, test_composite)
            else:
                _, test_pval = spearmanr(test_targets, test_composite)
            result["test_pvalue"] = test_pval
            logger.info(
                f"Test {metric}: {test_score:.4f} (p={test_pval:.4e}, n={len(test_targets)})"
            )
        else:
            logger.info(f"Test {metric}: {test_score:.4f} (n={len(test_targets)})")

        # Include predictions if requested
        if return_predictions:
            result["predictions"] = test_composite
            result["targets"] = test_targets

        return result

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

        # Normalize weights
        total = (
            weights["veg_weight"] + weights["terrain_weight"] + weights["ndvi_weight"]
        )
        veg_w = weights["veg_weight"] / total
        terrain_w = weights["terrain_weight"] / total
        ndvi_w = weights["ndvi_weight"] / total

        # Extract aggregation parameters
        streetview_stat = weights.get("streetview_stat", "mean")
        streetview_percentile = weights.get("streetview_percentile", 50)
        veg_radius = weights.get("veg_radius", self.buffer_meters)
        terrain_radius = weights.get("terrain_radius", self.buffer_meters)
        ndvi_stat = weights.get("ndvi_stat", "mean")
        ndvi_percentile = weights.get("ndvi_percentile", 50)
        ndvi_radius = weights.get("ndvi_radius", self.buffer_meters)

        # Combine all data (train+val+test)
        all_data = pd.concat([self.train_val_data, self.test_data])
        all_points = self.target_gdf.loc[all_data.index].copy()

        # Apply circular buffer aggregation with optimized parameters
        if weights["veg_weight"] > 0:
            all_veg = self._apply_circular_buffer_aggregation(
                all_points,
                self.veg_data,
                veg_radius,
                streetview_stat,
                streetview_percentile,
            )
        else:
            all_veg = np.zeros(len(all_points))

        if weights["terrain_weight"] > 0:
            all_terrain = self._apply_circular_buffer_aggregation(
                all_points,
                self.terrain_data,
                terrain_radius,
                streetview_stat,
                streetview_percentile,
            )
        else:
            all_terrain = np.zeros(len(all_points))

        if weights["ndvi_weight"] > 0:
            all_ndvi = self._apply_circular_buffer_aggregation(
                all_points,
                self.ndvi_data,
                ndvi_radius,
                ndvi_stat,
                ndvi_percentile,
            )
        else:
            all_ndvi = np.zeros(len(all_points))

        # Normalize using scaler fit on all data
        scaler = MinMaxScaler()
        all_combined = np.column_stack([all_veg, all_terrain, all_ndvi])
        all_valid_mask = ~np.isnan(all_combined).any(axis=1)

        if all_valid_mask.sum() > 0:
            all_combined[all_valid_mask] = scaler.fit_transform(
                all_combined[all_valid_mask]
            )

        all_veg_norm = all_combined[:, 0]
        all_terrain_norm = all_combined[:, 1]
        all_ndvi_norm = all_combined[:, 2]

        # Calculate composite
        composite = (
            all_veg_norm * veg_w + all_terrain_norm * terrain_w + all_ndvi_norm * ndvi_w
        )

        result_df = all_data.copy()
        result_df["veg"] = all_veg
        result_df["terrain"] = all_terrain
        result_df["ndvi"] = all_ndvi
        result_df["composite"] = composite

        return result_df

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

    def generate_composite_greenery_map(
        self,
        output_path: str = "output_results/composite_greenery.tif",
        top_percent: float = 0.2,
        progress_callback: callable | None = None,
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

        # 1. Get robust trials
        robust_trials = self.get_robust_trials(
            method="auto", p_threshold=0.05, tolerance=0.1, min_trials=10
        )

        if not robust_trials:
            logger.warning(
                "No robust trials found. Using all completed trials instead."
            )
            robust_trials = [
                t
                for t in self.study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]

        if not robust_trials:
            raise ValueError("No completed trials to generate composite from")

        if progress_callback:
            progress_callback(10, 100)

        # 2. Take top X% of robust trials
        n_top = max(1, int(len(robust_trials) * top_percent))
        top_trials = sorted(
            robust_trials,
            key=lambda t: t.value,
            reverse=(self.study.direction.name == "MAXIMIZE"),
        )[:n_top]

        logger.info(
            f"Using top {n_top} trials ({top_percent*100:.0f}%) out of {len(robust_trials)} robust trials"
        )

        if progress_callback:
            progress_callback(20, 100)

        # 3. Average parameters
        from statistics import mode

        weights_veg = [t.params.get("veg_weight", 0) for t in top_trials]
        weights_ter = [t.params.get("terrain_weight", 0) for t in top_trials]
        weights_ndvi = [t.params.get("ndvi_weight", 0) for t in top_trials]

        radii_veg = [t.params.get("veg_radius", self.buffer_meters) for t in top_trials]
        radii_ter = [
            t.params.get("terrain_radius", self.buffer_meters) for t in top_trials
        ]
        radii_ndvi = [
            t.params.get("ndvi_radius", self.buffer_meters) for t in top_trials
        ]

        streetview_stats = [t.params.get("streetview_stat", "mean") for t in top_trials]
        ndvi_stats = [t.params.get("ndvi_stat", "mean") for t in top_trials]
        streetview_percentiles = [
            t.params.get("streetview_percentile", 50) for t in top_trials
        ]
        ndvi_percentiles = [t.params.get("ndvi_percentile", 50) for t in top_trials]

        # Calculate averaged/mode parameters
        final_params = {
            "veg_weight": int(np.mean(weights_veg)),
            "terrain_weight": int(np.mean(weights_ter)),
            "ndvi_weight": int(np.mean(weights_ndvi)),
            "veg_radius": int(np.mean(radii_veg)),
            "terrain_radius": int(np.mean(radii_ter)),
            "ndvi_radius": int(np.mean(radii_ndvi)),
            "streetview_stat": mode(streetview_stats),
            "ndvi_stat": mode(ndvi_stats),
            "streetview_percentile": int(np.mean(streetview_percentiles)),
            "ndvi_percentile": int(np.mean(ndvi_percentiles)),
        }

        logger.info(f"Final averaged parameters: {final_params}")

        if progress_callback:
            progress_callback(30, 100)

        # 4. Determine grid (raster or geojson)
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
        veg_values = self._apply_circular_buffer_aggregation(
            points_gdf,
            self.veg_data,
            final_params["veg_radius"],
            final_params["streetview_stat"],
            final_params["streetview_percentile"],
        )

        if progress_callback:
            progress_callback(55, 100)

        logger.info("Sampling terrain at grid points...")
        terrain_values = self._apply_circular_buffer_aggregation(
            points_gdf,
            self.terrain_data,
            final_params["terrain_radius"],
            final_params["streetview_stat"],
            final_params["streetview_percentile"],
        )

        if progress_callback:
            progress_callback(70, 100)

        logger.info("Sampling NDVI at grid points...")
        ndvi_values = self._apply_circular_buffer_aggregation(
            points_gdf,
            self.ndvi_data,
            final_params["ndvi_radius"],
            final_params["ndvi_stat"],
            final_params["ndvi_percentile"],
        )

        if progress_callback:
            progress_callback(85, 100)

        # 6. Calculate composite
        logger.info("Calculating weighted composite...")
        composite = (
            (final_params["veg_weight"] / 100.0) * veg_values
            + (final_params["terrain_weight"] / 100.0) * terrain_values
            + (final_params["ndvi_weight"] / 100.0) * ndvi_values
        )

        # 7. Create raster
        logger.info(f"Saving composite greenery map to {output_path}")

        # Reshape composite back to grid
        composite_grid = np.full((height, width), np.nan, dtype=np.float32)
        for i, (row_idx, col_idx, value) in enumerate(
            zip(points_gdf["row"], points_gdf["col"], composite)
        ):
            composite_grid[row_idx, col_idx] = value

        # Save as GeoTIFF
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

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
            compress="lzw",
        ) as dst:
            dst.write(composite_grid, 1)

        if progress_callback:
            progress_callback(100, 100)

        logger.info(f"✓ Composite greenery map saved: {output_path}")

        # Save parameters used
        params_path = output_path.replace(".tif", "_params.json")
        import json

        with open(params_path, "w") as f:
            json.dump(
                {
                    "final_parameters": final_params,
                    "n_trials_averaged": n_top,
                    "total_robust_trials": len(robust_trials),
                    "top_percent": top_percent,
                },
                f,
                indent=2,
            )

        logger.info(f"✓ Parameters saved: {params_path}")

        return output_path

    def generate_results_report(
        self,
        output_dir: str = "output_results/fusion/study_results",
        include_plots: bool = True,
        progress_callback: callable | None = None,
    ) -> None:
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
        logger.info("Extracting robust trials using FDR correction...")
        robust_trials = self.get_robust_trials(
            method="auto", p_threshold=0.05, tolerance=0.1, min_trials=10
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
        try:
            composite_path = os.path.join(
                output_dir, "../composite_greenery.tif"
            )  # output_results/fusion/
            self.generate_composite_greenery_map(
                output_path=composite_path,
                top_percent=0.2,
                progress_callback=None,  # Nested progress not supported yet
            )
        except Exception as e:
            logger.warning(f"Could not generate composite greenery map: {e}")

        if progress_callback:
            progress_callback(100, 100)

        logger.info("✓ Results report generation complete!")


# ═══════════════════════════════════════════════════════════════════════════════
# TODO LIST - FUSION MODULE
# ═══════════════════════════════════════════════════════════════════════════════

# ─── SCIENTIFICALLY SIGNIFICANT (High Priority) ────────────────────────────────
# TODO: FUSION_SPATIAL_WEIGHTS - **CRITICAL** Add spatial autocorrelation handling
#       - Implement Moran's I testing on residuals
#       - Add spatial lag/error regression models
#       - Account for spatial autocorrelation in p-values and parameter estimates
#       - Without this, all p-values may be inflated and weights biased

# TODO: FUSION_MULTIPLE_OUTCOMES - **MODERATE** Multi-objective optimization
#       - Support optimizing for multiple health outcomes simultaneously
#       - Find Pareto-optimal solutions that balance trade-offs
#       - Useful when outcomes should be equally weighted

# ─── QUALITY OF LIFE (Lower Priority) ──────────────────────────────────────────
# TODO: FUSION_EXPORT - Add methods to export optimized weights and composite indices
# TODO: FUSION_VISUALIZATION - Add plotting for parameter importance, CV scores
