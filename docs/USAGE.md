# GeoFuse — Usage Guide

## 1. Streamlit Web UI

The web UI is ideal for interactive exploration, visualization, and processing smaller study areas.

### Launch

```bash
conda activate geofuse
streamlit run ui/app.py
```

### Tab Overview

| Tab                       | Purpose                                                                                                                                                                                                                                     |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **NDVI**                  | Upload study areas, configure date modes, run Earth Engine downloads, and inspect results on an interactive map.                                                                                                                            |
| **GVI**                   | Upload study areas, generate sampling grids, run Street View + segmentation batch jobs, and visualize vegetation/terrain heatmaps.                                                                                                          |
| **HPC Monitoring**        | Track HPC CLI jobs by Job ID; view stage, progress, and metrics in real time. UI job progress is tracked in the GVI tab sidebar.                                                                                                            |
| **Fusion & Optimization** | Upload a target file (GeoJSON or GeoTIFF), pick a run mode (cross-sectional or mixed-effects longitudinal), assign one or more GVI/NDVI files per measurement year/wave, configure study details, run Bayesian fusion, and inspect results. |

### Typical Workflow

1. **NDVI tab** → upload a study area file → set date range → choose output formats (**GeoTIFF** is default; **GeoPackage** and **GeoJSON** optional) → "Run NDVI Analysis"
2. **GVI tab** → upload the same study area → set grid resolution and buffer → choose output formats (**GeoPackage** is default; **GeoTIFF** for per-cluster tile rasters; **GeoJSON** for compatibility) → "Generate Sampling Grids" → "Run GVI Analysis"
3. **Fusion tab** → upload a target outcomes file (GeoJSON or GeoTIFF) → pick a **Run mode** (Cross-sectional or Mixed-effects longitudinal) → upload one GVI file + one NDVI file per measurement year/wave in the **Metric File Assignment** section → set CGI formula, covariates, objective metric, optional **spatial-confounding adjustment** (KS-AIC / Spatial+), test-set size, and the stability-selection knobs in **Study Details** → "🚀 Run Fusion Optimization"
4. Inspect the composite greenery weights and export results.

> [!NOTE]
> Long-running jobs survive browser refresh. **GVI runs in a separate Python process** so the UI stays responsive (and the GPU stays fed) even while the browser tab is in the foreground. **NDVI** and **Fusion** run inside the Streamlit process. Progress is tracked in the **sidebar Job Monitor** (rendered by `ui/job_panel.py`), which is visible from all tabs and shows every job type — GVI, NDVI, NDVI-from-column, and Fusion. Multiple study areas with different settings (resolution, buffer, dates) can run in parallel as separate jobs.

#### Resuming after a crash or restart

If Streamlit (or the machine) restarts mid-job, the affected jobs reload as **"Interrupted"**. To resume: switch to the same tab the job came from, re-upload the **same** study area file you originally used. A restart panel appears with **Resume Job** and **Discard** buttons. Resume continues from the exact point where the job stopped — already-processed points and cached panoramas are skipped.

#### Per-job log files

Each job writes a persistent log to `logs/jobs/<job_id>.log`. The job-monitor expander has a "📄 Open log file" button that opens the file in your OS default text editor — useful for jobs longer than the 100-line in-memory deque.

#### National-scale GVI study areas

For widely-scattered inputs (e.g. neighbourhoods across multiple cities), the engine automatically clusters the buffered features and generates one sampling grid per cluster, all anchored to a common reference. There is no special "national mode" — just upload the file. The chosen projected CRS (UTM / LCC / Polar Stereographic) and an estimated distortion are reported in the job log.

#### Fusion form layout

The Fusion tab reads top-to-bottom in the order you reason about a run:

1. **Target Configuration** — upload the target file, pick outcome column(s), and preview on the map.
2. **Optimization Setup** — pick the **Run mode** (Cross-sectional or Mixed-effects longitudinal); the rest of the section adapts:
   - **Cross-sectional** offers a **Date column available?** toggle. With the toggle off, no date column is needed and each metric channel takes one file. With it on, pick the date column; distinct measurement years are discovered and drive the per-year file assignment in the next section (years are only a metric-file routing key — they never enter the regression).
   - **Mixed-effects longitudinal** offers **long** intake (one target file with explicit wave column — pick the entity-ID, date, and wave columns from selectboxes) or **wide** intake (one target file per wave, with per-file column pickers + a free-text wave label). A date column is mandatory in this mode.
3. **Metric File Assignment** — per channel (NDVI then GVI), set the buffer ladder (min / max / step in metres) that applies to every uploaded file in that channel, then upload one or more files. When years/waves were discovered above, each uploaded file gets a multi-select listing which years/waves it applies to. Every year/wave must be covered by exactly one file per channel — coverage status is shown live and the run fails with an error on submit if anything is unassigned.
4. **Study Details** (form) — CGI formula (`weighted_average` or `synergy`), covariates (numeric attribute columns to control for; wide-mode longitudinal intersects across wave files), a single mode-aware **Objective metric** dropdown (OLS-based options for cross-sectional, MixedLM-based options for mixed-effects), a **Test set size** slider (fraction of the whole dataset; default 0.25), stratification bins, the **Stability selection** knobs (bootstraps, trials per bootstrap, minimum trials per cell, **worst quantile**), MixedLM toggles, resume previous study, the **Also optimize each metric on its own** standalones checkbox, and — for any vector target — a **Per-pixel CGI scoring** block with the **CGI grid pixel size (m)** slider (25–500 m, step 25, default 50) and the **Scale composite map to [0, 1]** checkbox, plus — for polygon targets only — the **Area-balanced stratified split** checkbox.
5. **🚀 Run Fusion Optimization**.

#### Optimization results panel

After a job completes (or when you click **Load results** on a completed job in the sidebar Job Monitor) the results section renders top-to-bottom in these blocks:

- **Headline** — the CGI bottom line: held-out test score + 95% percentile CI, the greenery↔outcome direction, and the CGI-vs-best-standalone AIC/BIC verdict (when standalones ran), with the formula and covariates noted underneath.
- **Study detail** — a **study selector** (CGI · Vegetation · Terrain · NDVI, when standalones ran) drives a self-contained detail block for the picked study: its test score + CI / direction / n tiles, the stability-selected weights · radii · aggregators (standalones show their single 100% channel), per-subset scores (train / val / test / all, partial + raw when covariates are set), the final parameters JSON, and that study's full stability diagnostics (top weight cells, winning-cell OOB-score distribution, per-bootstrap leaderboard).
- **CGI vs standalone single-metric studies** — a grouped score bar chart across studies and subsets plus the penalized model comparison (AIC / BIC of the 3-channel full model vs the best single channel, ΔAIC / ΔBIC with the strength band); rendered only when standalones ran.
- **Channel collinearity check** — the iterative-VIF report, when the collinearity check was requested.
- **Covariate impact** — when covariates are set, two OLS regressions (`target ~ CGI + covariates` vs `target ~ CGI`) on the full dataset; per-covariate coefficient, t-stat, p-value, direction, and partial R²; lift over CGI-only R² summarised at the top.
- **Composite map viewer** — multi-select composites + target outcome rendered as subplots in a near-square grid at 300 dpi; composite subplots share a `[0, 1]` colorbar.
- **Mixed-effects metric tabs** — when MixedLM scoring is on, one tab per `mixedlm_metrics*.csv` written under the per-job folder.
- **Output files on disk** — an index of every file the job wrote (path + size), so the saved manifest, CSVs, composites, and `run_config.json` are easy to locate.

#### Per-job output folder

Every fusion run writes its artifacts to `output_results/fusion/<YYYYMMDDTHHMMSS>__<short_job_id>/` so reruns no longer overwrite previous outputs. Layout:

```text
<job_root>/
├── composite_greenery.tif            ← CGI winning-params raster
├── composite_greenery_params.json    ← winning params + stability provenance
├── composite_greenery_<veg|terrain|ndvi>.tif         ← one per standalone
└── study_results/
    ├── run_config.json                ← every setting the job ran with
    ├── results_summary.json           ← master manifest (all studies)
    ├── test_scores.csv                ← held-out test score + CI per study
    ├── scores.csv                     ← every subset score per study
    ├── parameters.csv                 ← winning params per study (long form)
    ├── stability_cells.csv            ← ranked weight cells per study
    ├── stability_bootstraps.csv       ← per-bootstrap leaderboard per study
    ├── aic_bic.json                   ← CGI-vs-standalone verdict (if standalones)
    ├── covariate_impact.csv           ← per-covariate effects (if covariates)
    ├── collinearity.json              ← VIF report (if requested)
    ├── mixedlm_metrics*.csv           ← if longitudinal + MixedLM
    └── standalone_<ch>/               ← per-standalone composite outputs
```

Every test the job produces is recorded both as the machine-readable `results_summary.json` manifest and as the tidy CSVs above (multi-outcome runs suffix each basename with `__<outcome>`). Run settings are recorded on the job record **and** snapshotted to `run_config.json`, and replayed verbatim on restart (see **Full run-setting fidelity** in [FEATURES.md](FEATURES.md)). The per-job caches — metric downloads, the pre-aggregation SQLite cache, and artifacts — are content-addressed via a fingerprint of the search-space settings (buffer ladder, formula, scaling toggles, test size, …), so a config change starts fresh while an identical re-run resumes.

#### Per-pixel CGI scoring (all vector targets)

Every vector target — points, lines, or polygons — is scored against the **average of per-pixel CGI** inside its catchment. The engine builds a regular grid of pixel centroids at the configured spacing (the **CGI grid pixel size** slider, shown for any vector target), evaluates per-trial CGI at every pixel from the pre-aggregation cache, and collapses the pixels in each entity's catchment to one value before scoring. The catchment is the **polygon footprint** for polygons, and the **buffer up to the trial's largest channel radius** for points/lines (so the radius parameter genuinely tunes the spatial scale; each entity always keeps its nearest pixel so none drops out at small radii). Smaller spacing = higher fidelity and a bigger SQLite cache; larger spacing = smaller cache + faster build. Because point/line scoring now evaluates many pixels per entity, watch the pixel count logged at the start of a run and raise the spacing if a dense point set with large buffers runs slowly.

The CGI is always combined from **raw** channel values — there is no per-channel rescaling. A **Scale composite map to [0, 1]** checkbox (all vector targets) instead controls the **composite**: when on, the optimized greenery map (CGI, or each standalone channel) is min-max normalized over the whole grid before it is scored and written — at every step (each optimization run, the held-out test, and the final raster); when off, the raw composite is used everywhere. Because the cross-sectional metrics are scale-invariant, this mainly changes the output raster and the recorded composite values, not the optimization score. Polygon targets add one more knob: **Area-balanced stratified split** greedily allocates polygons inside each outcome quartile so total polygon area is balanced across train / val / test, rather than just polygon counts.

#### Mixed-effects fusion (longitudinal data)

Pick **Mixed-effects (longitudinal)** as the run mode. The four `mixedlm_*` scorers measure greenery's contribution net of within-entity temporal correlation, so a date column is required. Date columns accept full ISO dates (`2010-01-15`), year + month (`2010-01`), year-only strings (`2010`), or integer years — `years_since_baseline` is derived per entity from each entity's earliest measurement date. Pick the MixedLM scorer Optuna should optimize (default `mixedlm_tstat`); the other three metrics are computed post-hoc and saved to `output_results/fusion/study_results/mixedlm_metrics.csv` with mean + 95 % CI rows per pool. A channel can use the same file across every wave (typical for terrain or one-snapshot NDVI) — just assign that one upload to every discovered wave.

#### Year-aware cross-sectional

Cross-sectional runs typically use one GVI file and one NDVI file. When the cohort was sampled in different years and you have per-year greenery files, turn **Date column available?** on and pick the date column; the discovered years drive the same per-channel file-assignment grid the longitudinal mode uses. The OLS scorer (distance correlation / spearman / r² / nrmse / mutual_info) is unchanged — the spec sits underneath only as a per-year metric-file routing key.

---

## 2. Command-Line Interface (CLI) for Batch Processing

The CLI is designed for large-scale, automated, parallel processing on servers and HPC clusters.

### Activate the Environment

```bash
conda activate geofuse
```

### Run in Parallel with MPI

```bash
# Example: 4 parallel processes
mpiexec -n 4 python scripts/cli.py --config config.csv
```

| Argument   | Description                                            |
| ---------- | ------------------------------------------------------ |
| `--config` | Path to a CSV file defining input files and parameters |

> [!NOTE]
> A template `config.csv` is auto-created if none is found when you first run the CLI.

### Scaling to HPC Clusters

Replace `mpiexec` with your cluster's MPI launcher (e.g., `srun` on SLURM):

```bash
srun -n 32 python scripts/cli.py --config config.csv
```

Each MPI rank processes a separate study area in parallel. The core engines write to `output_results/` using file-safe naming to avoid collisions.
