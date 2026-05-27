# GeoFuse — Feature Reference

## 1. Street View Intelligence (GVI)

* **Automated Sourcing**: Scrapes or downloads Google Street View panoramas for any study area (GeoJSON or Shapefile). Operates _with_ or _without_ an API Key.
* **Deep Learning Segmentation**: Uses the **DeepLabV3+** model (PyTorch) trained on the **Cityscapes** dataset to identify Vegetation (class 8) and Terrain (class 9) greenery coverage.
* **Nation-Scale Clustered Sampling Grids**: For widely-scattered inputs (e.g. neighbourhoods across multiple cities), the engine automatically dissolves touching buffers, splits the study area into spatial clusters, and generates a separate sampling grid per cluster — all anchored to one common reference grid. This avoids the exponential bbox blow-up that would otherwise produce hundreds of millions of empty cells across a country.
* **Automatic Projected CRS Selection**: For each study area, the engine picks the most accurate planar CRS by extent — a local UTM zone for compact areas (≤ 6° lon / 8° lat), a two-parallel Lambert Conformal Conic for continent-scale extents, or Polar Stereographic above 75° latitude. A measured distortion estimate is logged; warnings appear if distortion exceeds 2 %.
* **Subprocess Execution**: GVI workers run in a separate Python process so they no longer share the Python GIL with the Streamlit UI. This eliminates the GPU-utilisation drop that happened when the browser tab was in the foreground (measured ~37 percentage-point recovery on a Cityscapes / DeepLabV3+ workload).
* **Concurrent Per-Point Pipeline**: Up to 4 panoramas are downloaded, pre-processed, and queued for the GPU concurrently. Network I/O and CPU pre-processing run lock-free; only the GPU forward pass is serialised so one point's inference overlaps the next point's download.
* **Batch Processing**: Upload multiple study areas to process distinct regions simultaneously.
* **Smart Caching**: Shared, cross-process panorama cache (`logs/caches/gvi_panos.db`) prevents redundant downloads for overlapping areas, reducing processing time and API costs. An in-memory overlay keeps repeated lookups microsecond-fast.
* **Robust Processing**:
  * Async/multi-threaded downloading.
  * Image pre-processing: ensures 360° coverage, corrects panorama artifacts, detects corrupt panoramas, standardizes resolution.
* **Restart System for Interrupted Jobs**: If Streamlit (or the machine) restarts mid-run, the affected jobs appear as "Interrupted" in the affected tab. Re-uploading the **same** study area file shows a restart panel with "Resume Job" / "Discard" buttons; resume continues from the exact point where the job stopped. The re-uploaded file is hash-verified to ensure it is the same study area.
* **Refresh-Safe Job Monitor**: Jobs survive browser refresh and additional tabs. Track progress in the sidebar with a live health badge (active / stuck / errors). Job state is persisted to `logs/jobs.db`; per-job text logs are written to `logs/jobs/<job_id>.log` and can be opened directly from the UI with the "📄 Open log file" button.
* **Live Per-Job Logs**: Engine logs (including Earth Engine and Optuna internals) appear inside the job's own expander rather than the host terminal. Job log files persist on disk indefinitely for later inspection.
* **Parallel Study Areas**: Submit multiple study areas with different resolution or buffer settings simultaneously — each unique parameter combination is treated as a separate job.
* **Outputs**: **GeoPackage** point layer (canonical, recommended), per-cluster **GeoTIFF tiles** (optional, dense — no inter-cluster gaps), **GeoJSON** (compatibility), optional raw panoramas and segmentation masks. All canonical outputs in EPSG:4326.

---

## 2. Satellite Intelligence (NDVI)

* **Google Earth Engine Integration**: Fetches cloud-free Sentinel-2 or Landsat imagery for any study area.
* **Flexible Date Modes** (any combination):
  * **Date Range(s)**: Produces one composite output file per range.
  * **Specific Date(s)**: Builds a composite from imagery within a ± window around each date.
  * **Attribute Column**: Matches each feature to its own date from an attribute column, producing a single temporally-aligned output file.
* **Dynamic Calculation**: Computes NDVI for the exact timeframe matching your street view data.
* **Latitude-aware planar CRS**: Earth Engine exports use an auto-selected planar CRS (UTM / two-parallel LCC / Polar Stereographic) so pixels are rasterised in true ground metres at any latitude. The final raster ships in that same CRS — no WGS84 reprojection step — so every output pixel is a true square on the ground.
* **Cluster-aware tiling**: For nationally-scattered inputs, the engine dissolves the buffered geometry into connected components and tiles each component's bbox in true metres, dropping tiles that fall over empty bbox regions (ocean, gaps between provinces) before they reach Earth Engine.
* **Pixel-aligned streaming mosaic**: Tiles are exported in a single planar CRS with a global snap grid and stream-mosaicked directly into the user-facing GeoTIFF in that same CRS. Adjacent tile boundaries align to the pixel, peak mosaic memory never exceeds one tile, and there is no second reprojection pass to drag out the wall clock or smear nodata into edge pixels.
* **Parallel tile downloads**: Up to 4 tiles download from Earth Engine concurrently. A single failed tile is logged as a warning and skipped; the rest of the batch keeps going. Progress emits are throttled to a few seconds based on tile count so the UI heartbeat stays responsive on large-scale runs.
* **Automatic retry on flaky networks**: Every Earth Engine export retries up to 3 times with jittered exponential backoff before giving up. Tiles that exhaust the retry budget are recorded in the sidecar JSON (cluster + tile ID + last error) so a swiss-cheese mosaic from a bad network minute is auditable instead of mysterious.
* **Persistent tile cache + resume from interruption**: Every tile lands in a per-key cache directory (`logs/caches/ndvi_tiles/`) keyed by geometry + date range + cloud max + resolution. Repeat runs over the same area + date range are near-instant and every previously-downloaded tile is reused without touching Earth Engine. A cancelled or crashed run also benefits: the next attempt with the same parameters picks up exactly where it left off. The cache is LRU-evicted with a default 5 GB cap so it can't grow unbounded; the audit trail in the sidecar JSON records the cache key and how many tiles were reused.
* **Subprocess execution**: NDVI runs in a separate Python process (same scaffold as GVI). Earth Engine HTTP, zip-extract, and rasterio decode no longer share the GIL with the Streamlit UI — the job-monitor fragment and result-inspector stay responsive while a large mosaic is downloading.
* **Per-cluster GeoTIFF tiles (optional)**: For scattered inputs (multiple city / provinces), enable **Per-cluster tiles** to get one GeoTIFF per connected component in `{name}_ndvi_tiles/` plus a `tiles_index.json`. Avoids the single mostly-NaN continent-spanning mosaic so downstream fusion / GIS tools can index by cluster.
* **Compressed tiled BIGTIFF outputs**: NDVI and GVI GeoTIFFs use DEFLATE compression with a float-aware predictor, internal 256×256 tiling, BIGTIFF support, and `SPARSE_OK=TRUE` so all-nodata blocks cost zero bytes on disk. Overviews are deliberately omitted from the NDVI mosaic so the user-facing raster is exactly what the engine produced — QGIS and ArcGIS build local pyramids on demand if you need them.
* **Streaming vector export**: When **Save GeoPackage** is enabled, NDVI walks the raster in its native 256×256 blocks and appends to the GeoPackage in bounded chunks. Peak memory stays low regardless of raster size, so national-scale `*_ndvi.gpkg` exports will not OOM at the vector step. GeoJSON output is materialised from the streamed GPKG at the end.
* **Single-band NDVI raster**: NDVI GeoTIFFs hold one float32 band — the median NDVI over the requested date range — with `-9999` as nodata. Cloud-masked and out-of-collection pixels are painted with the sentinel before export so the downstream warp respects coverage gaps cleanly.
* **Actionable Earth Engine diagnostics**: When a job returns no usable imagery, the engine distinguishes "no images in the date range" from "all images exceeded the cloud threshold" so you get a useful error message instead of a generic "no images found." Pre-2017 ranges suggest switching to Landsat / auto mode.
* **Automatic coverage rescue**: When the cloud-filtered collection has fewer than 3 images, the engine widens the date window by ±50 % once and re-queries. If the wider window helps, the run proceeds and the sidecar JSON records the wider range so you know the composite is broader than what you asked for.
* **Landsat 8/9 fallback for pre-2017 dates**: The new `Satellite` option defaults to `auto`; Sentinel-2 for ranges ending on/after 2017-03-28, Landsat 8 + 9 (Collection 2 Level-2) otherwise. The NDVI band name and downstream consumers are identical between collections; only the source and per-pixel scale differ. Sidecar records which satellite was used.
* **Sample at uploaded features (optional)**: Enable **Sample at uploaded features** to also write `{name}_ndvi_at_features.gpkg` — your original features with NDVI attached (exact-pixel read or zonal mean/median/min/max/std/count over a buffer or polygon). Useful for fusion downstream where you need per-feature greenery values instead of dense rasters.
* **GeoPackage Output Option**: GeoTIFF remains the default, but a `Save GeoPackage` checkbox writes `*_ndvi.gpkg` (layer `ndvi_samples`) alongside the raster for QGIS / GeoPandas consumption.
* **Restart System**: Like GVI, NDVI jobs interrupted by a Streamlit restart appear as "Interrupted" and can be resumed by re-uploading the original study area.

---

## 3. Metric Fusion & Optimization

* **Automated Metric Alignment**: Auto-downloads and spatially aligns GVI (vegetation/terrain) and NDVI within your study area when pre-computed files are not provided.
* **Dual Input Support**: Works with **point-based** targets (GeoJSON with health/environmental data) and **raster-based** targets (GeoTIFF continuous surfaces).
* **Bayesian Optimization**: Uses **Optuna** (TPE sampler) to optimize 9 parameters:
  * **Weights**: Vegetation, Terrain, NDVI contribution (0–100%, sum = 100%)
  * **Spatial Aggregation**: Circular buffer radii (100m to user-defined max, step = 50m)
  * **Statistical Functions**: Mean, median, or percentile-based aggregation
  * Separate radius and aggregation controls per metric component
* **Robust Cross-Validation**:
  * Stratified K-Fold CV ensures representative sampling across the target distribution.
  * 20% held-out test set for final validation.
  * Benjamini-Hochberg FDR correction for correlation-based metrics.
  * Pruning support: Median, Hyperband, or Successive Halving.
* **Multi-Metric Objectives**: Pearson, Spearman, R², RMSE, Mutual Information.
* **Comprehensive Reporting**: Optuna visualization suite (history, parameter importance, parallel coordinates, contour plots, EDF, slice plots, timeline) for both robust and all-trials analyses.
* **Composite Map**: Ensemble-averaged parameters from the top 20% of robust trials generate a grid-aligned GeoTIFF composite greenery raster.
* **Intelligent Caching**: Deterministic filenames allow reuse of downloaded metrics for identical study areas.

---

## 4. High-Performance Computing (HPC) Integration

* **MPI-Enabled CLI**: The `scripts/cli.py` interface supports parallel execution via `mpiexec`/`mpirun`.
* **Headless Batch Processing**: Core engines are fully decoupled from the UI; run them directly from Python scripts or the CLI without a browser.
* **GPU Acceleration**: Automatic device selection (CUDA → MPS → CPU) via `geofuse.vision.get_best_device`.
* **Memory-Efficient I/O**: Asynchronous downloads and windowed raster reads/writes for maximum throughput on shared compute nodes.
