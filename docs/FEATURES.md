# GeoFuse — Feature Reference

GeoFuse measures two complementary views of urban greenery, then fuses them into a single composite index tuned against a health or environmental outcome. This page is a high-level tour of what each part does and the design choices that matter for trusting the results.

---

## 1. Eye-Level Greenery — GVI

The **Green View Index** estimates how much greenery a person sees at street level.

- **Automated panorama sourcing.** Pulls Google Street View imagery for any study area (GeoJSON, Shapefile, or GeoPackage), with or without an API key.
- **Deep-learning segmentation.** A **DeepLabV3+** model trained on the **Cityscapes** classes labels each panorama; GVI is the share of pixels classed as **vegetation** and **terrain**.
- **Scales from a block to a nation.** For scattered inputs (neighbourhoods across many cities), the engine dissolves overlapping buffers, splits the area into spatial clusters, and builds one sampling grid per cluster on a shared reference grid — avoiding the millions of empty cells a single continent-wide bounding box would create.
- **True-metre grids.** Each study area is projected to the most accurate planar CRS for its extent (local UTM, Lambert Conformal Conic for continental spans, or Polar Stereographic near the poles), so every grid cell is a true square in metres. Estimated map distortion is logged, with a warning past 2 %.
- **Responsive and fast.** GVI runs in its own process so the dashboard stays interactive, and downloads, pre-processing, and GPU inference overlap across points.
- **Shared panorama cache.** Downloaded panoramas are cached across runs and study areas, cutting redundant downloads and API cost.
- **Crash-safe.** Interrupted jobs reappear as *Interrupted*; re-uploading the same study area resumes exactly where it stopped (the file is hash-verified first).
- **Outputs.** **GeoPackage** point layer (canonical), optional per-cluster **GeoTIFF** tiles and **GeoJSON**, plus optional raw panoramas and segmentation masks. See [OUTPUTS.md](OUTPUTS.md).

---

## 2. Overhead Greenery — NDVI

The **Normalized Difference Vegetation Index** measures greenery from above, from satellite imagery.

- **Earth Engine integration.** Fetches cloud-masked **Sentinel-2** or **Landsat 8/9** imagery for any study area. `auto` mode picks Sentinel-2 from 2017 onward and Landsat for earlier dates; the choice is recorded.
- **Flexible date modes** (mix freely):
  - **Date range(s)** — one composite per range.
  - **Specific date(s)** — a composite from a ± window around each date.
  - **Attribute column** — each feature matched to its own date, producing one temporally-aligned output.
- **True-metre, pixel-aligned rasters.** Tiles are exported in an auto-selected planar CRS on a shared snap grid and stream-mosaicked directly into the final GeoTIFF — no second reprojection, so output pixels are exactly what Earth Engine produced.
- **Robust at scale.** Cluster-aware tiling skips empty regions (ocean, gaps between provinces); tiles download in parallel, retry on flaky networks, and resume from a persistent tile cache after an interruption.
- **Honest coverage.** Clear diagnostics distinguish "no images in range" from "all images too cloudy." When a window is too sparse it widens once automatically and records that it did. Tiles that never arrive are listed in the sidecar so gaps are auditable, not mysterious.
- **Sample at your features (optional).** Attach an NDVI value (exact-pixel or a zonal statistic over a buffer/polygon) to your own uploaded features — handy for downstream fusion.
- **Outputs.** Single-band **GeoTIFF** (default), optional **GeoPackage**, **GeoJSON**, per-cluster tiles, and a metadata sidecar JSON. See [OUTPUTS.md](OUTPUTS.md).

---

## 3. Fusion & Optimization

This is the analytical core: it learns how to combine GVI (vegetation + terrain) and NDVI into one **composite greenery index (CGI)** that best tracks an outcome you provide, and it reports that relationship with statistics you can defend.

### Inputs and scoring

- **You supply the metrics.** Upload the GVI and NDVI files produced by the tabs above (one per measurement year/wave where relevant). They are spatially aligned to your outcome automatically.
- **Consistent per-pixel scoring.** Every vector target — **points, lines, or polygons** — is scored the same way: a regular CGI grid is computed per trial, and each entity's value is the **mean per-pixel CGI inside its catchment** (a polygon's footprint, or a point/line's buffer up to that trial's largest radius). The composite map you see therefore matches the values that were scored. Raster targets keep their native grid.

### Choosing the formula and its parameters

- **Pluggable CGI formula.**
  - **Weighted average** (default) — three channel weights (vegetation / terrain / NDVI).
  - **Synergy** — a three-metric generalization of Wang et al. 2026 ([doi:10.3390/rs18010009](https://doi.org/10.3390/rs18010009)) with interaction terms and tunable powers on the main channels.
- **What gets tuned.** Channel weights/powers, each channel's **spatial scale** (circular buffer radius, on a ladder up to a limit you set), and the **aggregation** (mean / median / percentile).

### Why you can trust the result

- **Bootstrap stability selection** (Meinshausen & Bühlmann 2010, adapted to hyperparameter search). The engine draws many resamples of the training pool, explores each with a **space-filling quasi-random (Sobol) search**, and scores every candidate on that resample's held-out **out-of-bag** rows. The winning configuration is the one whose **worst-case out-of-bag score is best across resamples** — i.e. cross-validated predictive power, not a selection-inflated in-sample fit. (A uniform-coverage search is used deliberately; an objective-chasing sampler would concentrate on each resample's local optimum and inflate the stability counts.)
- **Held-out test set.** A fraction of the data (default 25 %) is set aside and never touched during tuning. The winning configuration is scored on it once, with a **percentile bootstrap confidence interval** — an independent effect size, reported without a permutation p-value to hunt.
- **Objective metrics.** **Partial distance correlation** (default) detects non-linear as well as linear associations *and* conditions on the covariates non-linearly, so a curved covariate effect is removed rather than partly credited as greenery signal. It is unsigned, so a separate **direction** indicator is reported alongside it. Also available: **distance correlation** (a faster variant with linear covariate adjustment), **Spearman**, **R²**, **normalized RMSE**, and **mutual information**.

### Controlling for confounders

- **Covariate-aware objective.** Select numeric attribute columns to control for, and the score becomes the greenery term's *partial* contribution, so a dominant covariate can't crowd out the CGI parameters. (Mutual information ignores covariates by design.)
- **Covariate residualization.** For the residualizing metrics (distance correlation, Spearman, R², normalized RMSE), choose how covariates are partialled out: **linear** (default) or **spline** (natural cubic), which removes non-linear covariate effects. Partial distance correlation conditions on covariates intrinsically, so it ignores this setting.
- **Spatial-confounding adjustment (optional).** Adds a flexible smooth of location so the reported association reflects greenery↔outcome co-variation *beyond* an unmeasured smooth spatial confounder. Two methods: **KS-AIC** (Keller & Szpiro 2020; recommended) and **Spatial+** (Dupont, Wood & Augustin 2022 / Rainey et al. 2025). The smooth is built per spatial cluster, so it never spans gaps between clusters. *Note: with this on, the optimizer chases the de-confounded association, so the winning parameters shift versus an unadjusted run.*

### Is combining channels worth it?

- **Standalone single-metric studies (optional).** Run the same selection on each channel alone (NDVI / vegetation / terrain). When enabled, the report adds an **AIC/BIC verdict** on whether the multi-channel CGI is justified over the best single channel — so you can see whether fusion earns its added complexity.

### Longitudinal and multi-year data

- **Mixed-effects (longitudinal) mode.** When entities are measured at several time points, each trial is scored with a linear mixed model (`statsmodels.MixedLM`) that accounts for within-entity correlation over time. Four scorers are available (t-statistic by default); all four are also reported post-hoc on the winning composite.
- **Year-aware cross-sectional mode.** For a cohort sampled across different years, route each entity to its year-matched greenery file. The cross-sectional scorer is unchanged — the year is only a file-routing key, never a regression input.

### Outputs and reproducibility

- **Composite map.** A grid-aligned GeoTIFF of the winning composite (plus one per standalone). A whole-grid **[0, 1] scaling** toggle controls normalization of the written/rendered map.
- **Everything on disk, per run.** Each run writes to its own timestamped folder so reruns never overwrite earlier results. Alongside the composite rasters, a `study_results/` folder holds a machine-readable manifest, tidy CSVs (test scores, subset scores, parameters, stability diagnostics, covariate impact), and a full settings snapshot. See [OUTPUTS.md](OUTPUTS.md).
- **Faithful restarts.** A run records its exact configuration and replays it verbatim on restart. Per-job caches (metric alignment, pre-aggregation, the search study) are keyed by a fingerprint of the settings, so an identical re-run resumes while any change starts fresh.
- **Load results any time.** Completed jobs can rehydrate the results panel from disk, independent of the configuration form.

---

## 4. Running at Scale — HPC & CLI

- **MPI-parallel CLI.** `scripts/cli.py` runs the same engines headlessly under `mpiexec` / `srun`; each rank processes a separate study area. See [USAGE.md](USAGE.md).
- **No browser required.** The core engines are fully decoupled from the dashboard — call them from the CLI or your own Python scripts.
- **GPU acceleration.** Automatic device selection (CUDA → MPS → CPU).

> [!WARNING]
> **HPC Monitoring tab — work in progress.** The dashboard's *HPC Monitoring* tab (last tab) is incomplete and under active development. It reads status files written by CLI runs on a cluster; the workflow is not yet finalized. For interactive runs, track progress in the **sidebar Job Monitor** instead.
