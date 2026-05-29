# GeoFuse — Usage Guide

## 1. Streamlit Web UI

The web UI is ideal for interactive exploration, visualization, and processing smaller study areas.

### Launch

```bash
conda activate geofuse
streamlit run ui/app.py
```

### Tab Overview

| Tab | Purpose |
| --- | --- |
| **NDVI** | Upload study areas, configure date modes, run Earth Engine downloads, and inspect results on an interactive map. |
| **GVI** | Upload study areas, generate sampling grids, run Street View + segmentation batch jobs, and visualize vegetation/terrain heatmaps. |
| **HPC Monitoring** | Track HPC CLI jobs by Job ID; view stage, progress, and metrics in real time. UI job progress is tracked in the GVI tab sidebar. |
| **Fusion & Optimization** | Upload a target file (GeoJSON or GeoTIFF), configure metric sources and optimization settings, run Bayesian fusion, and inspect results. |

### Typical Workflow

1. **NDVI tab** → upload a study area file → set date range → choose output formats (**GeoTIFF** is default; **GeoPackage** and **GeoJSON** optional) → "Run NDVI Analysis"
2. **GVI tab** → upload the same study area → set grid resolution and buffer → choose output formats (**GeoPackage** is default; **GeoTIFF** for per-cluster tile rasters; **GeoJSON** for compatibility) → "Generate Sampling Grids" → "Run GVI Analysis"
3. **Fusion tab** → upload a target outcomes file (GeoJSON or GeoTIFF) → select loaded GVI/NDVI results → "Run Fusion Optimization"
4. Inspect the composite greenery weights and export results.

> [!NOTE]
> Long-running jobs survive browser refresh. **GVI runs in a separate Python process** so the UI stays responsive (and the GPU stays fed) even while the browser tab is in the foreground. **NDVI** and **Fusion** run inside the Streamlit process. Progress is tracked in the **sidebar Job Monitor** (rendered by `ui/job_panel.py`), which is visible from all tabs and shows every job type — GVI, NDVI, NDVI-from-column, and Fusion. Multiple study areas with different settings (resolution, buffer, dates) can run in parallel as separate jobs.

#### Resuming after a crash or restart

If Streamlit (or the machine) restarts mid-job, the affected jobs reload as **"Interrupted"**. To resume: switch to the same tab the job came from, re-upload the **same** study area file you originally used. A restart panel appears with **Resume Job** and **Discard** buttons. Resume continues from the exact point where the job stopped — already-processed points and cached panoramas are skipped.

#### Per-job log files

Each job writes a persistent log to `logs/jobs/<job_id>.log`. The job-monitor expander has a "📄 Open log file" button that opens the file in your OS default text editor — useful for jobs longer than the 100-line in-memory deque.

#### National-scale GVI study areas

For widely-scattered inputs (e.g. neighbourhoods across multiple cities), the engine automatically clusters the buffered features and generates one sampling grid per cluster, all anchored to a common reference. There is no special "national mode" — just upload the file. The chosen projected CRS (UTM / LCC / Polar Stereographic) and an estimated distortion are reported in the job log.

#### Mixed-effects fusion (longitudinal data)

When entities are measured at multiple time points, open the **Mixed-effects / longitudinal mode** expander at the top of the fusion form. Set the wave labels (comma-separated, baseline first), pick the intake mode — **long** (one target file with a wave column) or **wide** (one target file per wave joined on a shared entity-id column) — and fill in the entity-id, date, and (long mode) wave-column names. Then, per greenery channel, paste the per-wave file paths comma-separated in the same order as the wave labels; tick **"Same file all waves"** for channels that don't change over time (typical for terrain and one-snapshot NDVI). Pick the MixedLM scorer Optuna should optimise (default `mixedlm_tstat`); the other three metrics are computed post-hoc and saved to `output_results/fusion/study_results/mixedlm_metrics.csv` with mean + 95 % CI rows per pool. Date columns accept full ISO dates (`2010-01-15`), year + month (`2010-01`), year-only strings (`2010`), or integer years — `years_since_baseline` is derived per entity from the earliest measurement date.

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

| Argument | Description |
|----------|-------------|
| `--config` | Path to a CSV file defining input files and parameters |

> [!NOTE]
> A template `config.csv` is auto-created if none is found when you first run the CLI.

### Scaling to HPC Clusters

Replace `mpiexec` with your cluster's MPI launcher (e.g., `srun` on SLURM):

```bash
srun -n 32 python scripts/cli.py --config config.csv
```

Each MPI rank processes a separate study area in parallel. The core engines write to `output_results/` using file-safe naming to avoid collisions.
