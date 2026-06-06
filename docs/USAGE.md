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
3. **Fusion tab** → upload a target outcomes file (GeoJSON or GeoTIFF) → pick a **Run mode** (Cross-sectional or Mixed-effects longitudinal) → upload one GVI file + one NDVI file per measurement year/wave in the **Metric File Assignment** section → set CGI formula, covariates, objective metric, test-set size, and the stability-selection knobs in **Study Details** → "🚀 Run Fusion Optimization"
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
4. **Study Details** (form) — CGI formula (`weighted_average` or `synergy`), covariates (numeric attribute columns to control for; wide-mode longitudinal intersects across wave files), a single mode-aware **Objective metric** dropdown (OLS-based options for cross-sectional, MixedLM-based options for mixed-effects), a **Test set size** slider (fraction of the whole dataset; default 0.25), stratification bins, the **Stability selection** knobs (bootstraps, trials per bootstrap, minimum trials per cell), MixedLM toggles, resume previous study, the **Also optimize each metric on its own** standalones checkbox, and — for polygon targets only — a **Polygon scoring** block with the **CGI grid pixel size (m)** slider (25–500 m, step 25, default 50), the **Whole-grid scaling** checkbox, and the **Area-balanced stratified split** checkbox.
5. **🚀 Run Fusion Optimization**.

#### Optimization results panel

After a job completes (or when you click **Load results** on a completed job in the sidebar Job Monitor) the results section renders with these blocks, in order:

- **Summary tiles** — formula name, held-out test score + 95% percentile CI, the greenery↔outcome direction, the CGI-vs-best-standalone AIC/BIC verdict (when standalones ran), weights, radii, aggregators.
- **Final (stability-selection) Parameters** — JSON dump of the winning weight cell's averaged params; the composite GeoTIFF is built from these.
- **Stability diagnostics** — the winning cell's worst-quantile / median OOB score and selection probability, the top-ranked competing weight cells, the winning cell's OOB-score spread, and a per-bootstrap summary table.
- **CGI vs Standalone Single-Metric Studies** — per-channel held-out test scores plus the AIC/BIC verdict (ΔAIC / ΔBIC, with the strength band) on whether the multi-channel CGI is justified over the best single channel; rendered only when standalones ran.
- **Covariate impact** — when covariates are set, two OLS regressions (`target ~ CGI + covariates` vs `target ~ CGI`) on the full dataset; per-covariate coefficient, t-stat, p-value, direction, and partial R²; lift over CGI-only R² summarised at the top.
- **Composite map viewer** — multi-select composites + target outcome rendered as subplots in a near-square grid at 300 dpi; composite subplots share a `[0, 1]` colorbar.
- **Mixed-effects metric tabs** — when MixedLM scoring is on, one tab per `mixedlm_metrics*.csv` written under the per-job folder.

#### Per-job output folder

Every fusion run writes its artifacts to `output_results/fusion/<YYYYMMDDTHHMMSS>__<short_job_id>/` so reruns no longer overwrite previous outputs. Layout:

```text
<job_root>/
├── composite_greenery.tif            ← CGI winning-params raster
├── composite_greenery_params.json
├── composite_greenery_<veg|terrain|ndvi>.tif         ← one per standalone
└── study_results/
    ├── mixedlm_metrics*.csv           ← if longitudinal + MixedLM
    └── standalone_<ch>/               ← per-standalone outputs
```

Run settings are recorded on the job record and replayed verbatim on restart (see **Full run-setting fidelity** in [FEATURES.md](FEATURES.md)). The per-job caches — metric downloads, the pre-aggregation SQLite cache, and artifacts — are content-addressed via a fingerprint of the search-space settings (buffer ladder, formula, scaling toggles, test size, …), so a config change starts fresh while an identical re-run resumes.

#### Polygon scoring (per-pixel CGI)

When the target carries polygon (or multipolygon) geometries, each polygon is scored against the **average of per-pixel CGI** inside its footprint. The engine builds a regular grid of pixel centroids at the configured spacing inside the union of the target polygons, every pixel becomes a pre-aggregation cache entity, per-trial CGI is evaluated at every pixel, and a `groupby(polygon).mean()` collapses to per-polygon CGI before scoring. Smaller spacing = higher fidelity and bigger SQLite cache; larger spacing = smaller cache + faster build. **Whole-grid scaling** skips per-channel MinMax scaling so the composite is a weighted sum on raw channel values (only the composite raster is min-max normalised); useful when channels are already on comparable scales. **Area-balanced stratified split** greedily allocates polygons inside each outcome quartile so total polygon area is balanced across train / val / test, rather than just polygon counts.

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
