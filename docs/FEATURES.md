# GeoFuse — Feature Reference

## 1. Street View Intelligence (GVI)

* **Automated Sourcing**: Scrapes or downloads Google Street View panoramas for any study area (GeoJSON or Shapefile). Operates _with_ or _without_ an API Key.
* **Deep Learning Segmentation**: Uses the **DeepLabV3+** model (PyTorch) trained on the **Cityscapes** dataset to identify Vegetation (class 8) and Terrain (class 9) greenery coverage.
* **Batch Processing**: Upload multiple study areas to process distinct regions simultaneously.
* **Smart Caching**: Shared panorama cache across all batch files prevents redundant downloads for overlapping areas, reducing processing time and API costs.
* **Robust Processing**:
  * Async/multi-threaded downloading.
  * Image pre-processing: ensures 360° coverage, corrects panorama artifacts, detects corrupt panoramas, standardizes resolution.
* **Crash Recovery**: Re-upload the same input file and click Run to automatically resume from the last processed point within the same session.
* **Refresh-Safe Job Monitor**: Jobs run in a process-level thread pool and survive browser refresh or opening additional tabs. Track progress in the sidebar with a live health badge (active / stuck / errors). Job state is persisted to `logs/jobs.db`; jobs interrupted by a Streamlit restart appear as "Interrupted" and can be resubmitted. Use "Scan Output Folder" to reload completed results from prior sessions.
* **Parallel Study Areas**: Submit multiple study areas with different resolution or buffer settings simultaneously — each unique parameter combination is treated as a separate job.
* **Outputs**: GeoJSON point vectors, multi-band GeoTIFF heatmaps, optional raw panoramas and segmentation masks.

---

## 2. Satellite Intelligence (NDVI)

* **Google Earth Engine Integration**: Fetches cloud-free Sentinel-2 or Landsat imagery for any study area.
* **Flexible Date Modes** (any combination):
  * **Date Range(s)**: Produces one composite output file per range.
  * **Specific Date(s)**: Builds a composite from imagery within a ± window around each date.
  * **Attribute Column**: Matches each feature to its own date from an attribute column, producing a single temporally-aligned output file.
* **Dynamic Calculation**: Computes NDVI for the exact timeframe matching your street view data.

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
