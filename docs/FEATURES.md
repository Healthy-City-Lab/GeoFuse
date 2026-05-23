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
* **Latitude-aware export CRS**: Earth Engine exports use an auto-selected planar CRS (UTM / two-parallel LCC / Polar Stereographic) so pixels are rasterised in true ground metres at any latitude; reprojection to WGS84 for delivery uses bilinear resampling.
* **Cluster-aware tiling**: For nationally-scattered inputs, the engine dissolves the buffered geometry into connected components and tiles each component's bbox in true metres, dropping tiles that fall over empty bbox regions (ocean, gaps between provinces) before they reach Earth Engine.
* **Streaming mosaic**: Per-tile rasters land in the final GeoTIFF one at a time via windowed writes. National-scale outputs no longer have to fit in RAM during the mosaic step. The mosaic progress bracket mirrors the download bracket so the UI heartbeat keeps ticking through the final stage.
* **Parallel tile downloads**: Up to 4 tiles download from Earth Engine and reproject concurrently. A single failed tile is logged as a warning and skipped; the rest of the batch keeps going. Progress emits are throttled to a few seconds based on tile count so the UI heartbeat stays responsive on large-scale runs.
* **Automatic retry on flaky networks**: Every Earth Engine export retries up to 3 times with jittered exponential backoff before giving up. Tiles that exhaust the retry budget are recorded in the sidecar JSON (cluster + tile ID + last error) so a swiss-cheese mosaic from a bad network minute is auditable instead of mysterious.
* **Resume from interruption**: Every tile lands in a deterministic per-run workspace keyed by geometry + date range + cloud max + resolution. If a run is cancelled or crashes, the next run with the same parameters reuses every tile that finished cleanly and only the remainder gets downloaded. Successful runs clean up after themselves; the audit trail in the sidecar JSON records how many tiles were reused.
* **Subprocess execution**: NDVI runs in a separate Python process (same scaffold as GVI). Earth Engine HTTP, zip-extract, and rasterio decode/reproject no longer share the GIL with the Streamlit UI — the job-monitor fragment and result-inspector stay responsive while a large mosaic is downloading.
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
