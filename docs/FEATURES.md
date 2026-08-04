# GeoFuse — Feature Reference

GeoFuse measures two complementary views of urban greenery, then fuses them into a single composite index tuned against a health or environmental outcome. This page is a high-level tour of what each part does and the design choices that matter for trusting the results.

---

## 1. Eye-Level Greenery — GVI

The **Green View Index** estimates how much greenery a person sees at street level.

- **Automated panorama sourcing.** Pulls Google Street View imagery for any study area (GeoJSON, Shapefile, or GeoPackage), with or without an API key.
- **Deep-learning segmentation.** A **DeepLabV3+** model trained on the **Cityscapes** classes labels each panorama; GVI is the share of pixels classed as **vegetation** and **terrain**.
- **Historical capture selection.** Street View re-photographs each location over the years. By default the most recent coverage is used, but you can target a specific capture year — the engine picks the historical panorama closest to it at every point, with an optional maximum-year-difference window that leaves points with no in-window capture empty. The capture date used is written to a `pano_date` column, making year-matched runs suitable for longitudinal analysis.
- **Per-year from a column.** Point GVI at a year/date column and each layer is split by year and processed separately (target year = that year) into a `{name}_temporal_gvi/` folder, one output per year. All years share one CRS chosen from the whole layer, so the per-year grids align — the eye-level counterpart to NDVI's per-year mode.
- **Merge files into groups (drag & drop).** Upload several files and drag them into groups; files in a group are concatenated (in a common CRS) and run as one job, each group with its own capture-date settings. Add, rename, and remove groups freely. Handy when one year is split across files — e.g. survey waves where 2020 appears in both — so every 2020 record lands in a single output. Files left in separate groups keep the default one-job-per-file behaviour. Merged jobs record every source file, so a restart re-verifies them all and **re-runs quietly** when they're unchanged.
- **Scales from a block to a nation.** For scattered inputs (neighbourhoods across many cities), the engine dissolves overlapping buffers, splits the area into spatial clusters, and builds one sampling grid per cluster on a shared reference grid — avoiding the millions of empty cells a single continent-wide bounding box would create.
- **True-metre grids.** Each study area is projected to the most accurate planar CRS for its extent (local UTM, Lambert Conformal Conic for continental spans, or Polar Stereographic near the poles), so every grid cell is a true square in metres. Estimated map distortion is logged, with a warning past 2 %.
- **Responsive and fast.** Every job type (GVI, NDVI, Fusion) runs in its own process so the dashboard stays interactive and a running job isn't throttled by the foreground browser tab, and downloads, pre-processing, and GPU inference overlap across points.
- **Shared panorama cache.** Downloaded panoramas are cached across runs and study areas, cutting redundant downloads and API cost.
- **Crash-safe.** Interrupted jobs reappear as *Interrupted*; re-uploading the same study area resumes exactly where it stopped (the file is hash-verified first).
- **Outputs.** **GeoPackage** point layer (canonical), optional per-cluster **GeoTIFF** tiles and **GeoJSON**, plus optional raw panoramas and segmentation masks. See [OUTPUTS.md](OUTPUTS.md).

---

## 2. Overhead Greenery — NDVI

The **Normalized Difference Vegetation Index** measures greenery from above, from satellite imagery.

- **Earth Engine integration.** Fetches cloud-masked **Sentinel-2** or **Landsat** imagery for any study area. `auto` mode picks Sentinel-2 from 2017 onward and Landsat for earlier dates; the Landsat path spans every era (5/7/8/9), with the pre-2013 TM/ETM+ sensors harmonized to the Landsat-8 scale so values stay comparable across years. The choice is recorded.
- **Flexible date modes** (mix freely):
  - **Date range(s)** — one composite per range.
  - **Specific date(s)** — a composite from a ± window around each date.
  - **Attribute column (per-year)** — split the layer by the year in a column and run each year as its own NDVI job over only that year's features, composited across the growing-season months you pick (e.g. June–September). Every year shares one CRS chosen from the whole layer, so the per-year rasters align — ready for longitudinal comparison. Outputs land in a `{name}_temporal_ndvi/` folder, one raster per year.
- **Merge files into groups (drag & drop).** As in the GVI tab, drag several files into a group to merge them into one job before dating — so a year split across files (e.g. survey waves where some are 2020 in wave 1 and some in wave 2) yields a single 2020 output. Add/rename/remove groups; separate groups keep the one-job-per-file default; merged jobs restart quietly when their source files are unchanged.
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
- **Held-out test set (the headline).** A fraction of the data (default 25 %) is set aside and never touched during tuning. The winning configuration is scored on it once — a **percentile bootstrap confidence interval** plus a **held-out permutation p-value** (Freedman–Lane when covariates are controlled). This is the honest generalizability check. The whole-data ("all") figure is also shown, but only as a *descriptive, in-sample* number: the parameters were tuned on most of those rows, so it is optimistic and carries no p-value.
- **Objective metrics.** **Partial distance correlation** (default) detects non-linear as well as linear associations *and* conditions on the covariates non-linearly, so a curved covariate effect is removed rather than partly credited as greenery signal. It is unsigned, so a separate **direction** indicator is reported alongside it. Also available: **distance correlation** (a faster variant with linear covariate adjustment), **Spearman**, **R²**, **normalized RMSE**, and **mutual information**.
- **Runtime cost levers.** Partial distance correlation is **O(n²) in entity count** (the engine caches the fixed target/covariate sides per resample, but each trial still pays one distance matrix on the composite); **distance correlation with spline residualization** is the O(n log n) approximation of the same idea when entity counts are large. The trial budget is `resamples × trials per resample`; the CGI grid spacing scales the pixel-side work quadratically (halving the spacing quadruples the pixel count); and the reporting replicate budgets (test-set CI, effects CI/permutations, paired comparison) are configurable per job — the test CI defaults to 2,000 replicates under partial distance correlation and 10,000 otherwise. **Mixed-effects objectives are the exception:** each replicate is a full model refit rather than an O(1) recomputation, so they are clamped to 150 for the test CI and 100 per slice for the effects CI. Those clamps dominate a longitudinal run's wall clock — the effects cap applies per metric per slice, and the test-CI cap once per study including each standalone.

### Controlling for confounders

- **Covariate-aware objective.** Select numeric attribute columns to control for, and the score becomes the greenery term's *partial* contribution, so a dominant covariate can't crowd out the CGI parameters. (Mutual information ignores covariates by design.)
- **Covariate residualization.** For the residualizing metrics (distance correlation, Spearman, R², normalized RMSE), choose how covariates are partialled out: **linear** (default) or **spline** (natural cubic), which removes non-linear covariate effects. Partial distance correlation conditions on covariates intrinsically, so it ignores this setting.
- **Spatial-confounding adjustment (optional).** Adds a flexible smooth of location so the reported association reflects greenery↔outcome co-variation *beyond* an unmeasured smooth spatial confounder. Two methods: **KS-AIC** (Keller & Szpiro 2020; recommended) and **Spatial+** (Dupont, Wood & Augustin 2022 / Rainey et al. 2025). The smooth is built per spatial cluster, so it never spans gaps between clusters. *Note: with this on, the optimizer chases the de-confounded association, so the winning parameters shift versus an unadjusted run.*

### Is combining channels worth it?

- **Standalone single-metric studies (optional).** Run the same selection on each channel alone (NDVI / vegetation / terrain). When enabled, the report adds an **AIC/BIC verdict** on whether the multi-channel CGI is justified over the best single channel, plus a **paired objective difference** of CGI against each standalone. Those paired p-values are **Holm-corrected** across the family of channels so comparing CGI against several of them doesn't inflate significance; when several outcomes are optimised, treat those as a further family.

### Longitudinal and multi-year data

- **Mixed-effects (longitudinal) mode.** When entities are measured at several time points, each trial is scored with a linear mixed model (`statsmodels.MixedLM`) that accounts for within-entity correlation over time. Four scorers are available (t-statistic by default); all four are also reported post-hoc on the winning composite.
- **Wave fixed effects (on by default).** Per-wave greenery layers differ for reasons unrelated to anyone's neighbourhood — a different satellite, a different compositing window — and that drift tracks calendar time, so without an indicator per wave it lands on the greenspace × time terms. Turn it off only when the per-wave layers are known to be harmonised.
- **Neighbourhood grouping.** Point a column at the area people share (FSA, census subdivision, site) and it enters as fixed effects. Greenspace is an area attribute, so neighbours have almost the same exposure; a person-level random effect alone leaves that level unmodelled and the greenspace term's standard error far too small.
- **Period-confounding check.** The report states how strongly within-person exposure change tracks *when* each person was measured, and repeats the greenspace × time terms with the exposure replaced by its per-wave mean. That placebo carries the period structure and no spatial information at all: a term it reproduces was measuring the wave, not the neighbourhood.
- **Year-aware cross-sectional mode.** For a cohort sampled across different years, route each entity to its year-matched greenery file. The cross-sectional scorer is unchanged — the year is only a file-routing key, never a regression input.

### Outputs and reproducibility

- **Composite map.** A grid-aligned GeoTIFF of the winning composite (plus one per standalone). A whole-grid **[0, 1] scaling** toggle controls normalization of the written/rendered map.
- **Everything on disk, per run.** Each run writes to its own timestamped folder so reruns never overwrite earlier results. Alongside the composite rasters, a `study_results/` folder holds a machine-readable manifest, tidy CSVs (test scores, subset scores, parameters, stability diagnostics, covariate impact), and a full settings snapshot. See [OUTPUTS.md](OUTPUTS.md).
- **Re-run from a filled form.** Re-running a job loads its recorded settings into the setup form rather than replaying them, so a minor tweak costs one edit instead of a rebuild from scratch. The re-run is confirmed on the job card, not in the tab. Per-job caches (metric alignment, pre-aggregation, the search study) are keyed by a fingerprint of the settings, so an unchanged config resumes while any change starts fresh.
- **The form keeps what you typed.** Picking files, dragging years, or an error elsewhere in the app never empties a half-filled setup form.
- **Sized to the machine it runs on.** Pre-aggregation and the stability search open thread pools derived from the host's core count, so the same code fits a laptop or a compute node without edits; `GEOFUSE_WORKERS` overrides the width. Every run logs a stage-by-stage wall-clock breakdown and, per parallel phase, how many workers were actually busy — enough to tell a pool that is too small from work that will not divide.
- **Load results any time.** Completed jobs can rehydrate the results panel from disk, independent of the configuration form.

---

## 4. Running at Scale — HPC & CLI

- **MPI-parallel CLI.** `scripts/cli.py` runs the same engines headlessly under `mpiexec` / `srun`; each rank processes a separate study area. See [USAGE.md](USAGE.md).
- **No browser required.** The core engines are fully decoupled from the dashboard — call them from the CLI or your own Python scripts.
- **GPU acceleration.** Automatic device selection (CUDA → MPS → CPU).

> [!WARNING]
> **HPC Monitoring tab — work in progress.** The dashboard's *HPC Monitoring* tab (last tab) is incomplete and under active development. It reads status files written by CLI runs on a cluster; the workflow is not yet finalized. For interactive runs, track progress in the **sidebar Job Monitor** instead.
