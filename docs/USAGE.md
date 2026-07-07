# GeoFuse — Usage Guide

## 1. Streamlit Web UI

Best for interactive exploration, visualization, and small-to-medium study areas.

```bash
conda activate geofuse
streamlit run ui/app.py
```

### Tabs

| Tab | Purpose |
| --- | --- |
| **NDVI** | Upload study areas, configure date modes, run Earth Engine downloads, inspect results on a map. |
| **GVI** | Upload study areas, generate sampling grids, run Street View + segmentation jobs, view vegetation/terrain heatmaps. |
| **Fusion & Optimization** | Upload an outcome target plus your GVI/NDVI files, choose a run mode and objective, tune the composite index, and inspect results. |
| **HPC Monitoring** 🚧 | *Work in progress (incomplete).* Track HPC CLI jobs by Job ID. Interactive jobs are tracked in the sidebar Job Monitor instead. |

> [!NOTE]
> Jobs survive a browser refresh. Progress is tracked in the **sidebar Job Monitor**, visible from every tab, for all job types (GVI, NDVI, Fusion). GVI runs in a separate process so the UI stays responsive; NDVI and Fusion run inside the Streamlit process. Multiple study areas with different settings can run in parallel.

### Typical workflow

1. **NDVI tab** — upload a study area, set a date range, pick output formats (GeoTIFF default), run.
2. **GVI tab** — upload the same study area, set grid resolution + buffer, pick output formats (GeoPackage default), generate grids, run.
3. **Fusion tab** — upload an outcome target — a vector (GeoJSON, Shapefile, or GeoPackage) or raster (GeoTIFF) layer — plus the GVI + NDVI files from steps 1–2, choose a run mode and objective, and run the optimization.
4. Inspect the composite weights, statistics, and maps; export results.

### Resuming after a crash or restart

If Streamlit (or the machine) restarts mid-job, the affected jobs reload as **Interrupted**. To resume, return to the originating tab and re-upload the **same** study area file. A restart panel appears with **Resume Job** / **Discard**; resume continues from where it stopped (processed points and cached data are skipped). The re-uploaded file is hash-verified, so a mismatched file is refused.

### Per-job logs

Each job writes a full log to `logs/jobs/<job_id>.log`. The job-monitor expander has an **Open log file** button; the in-UI view keeps only the last 100 lines, but the file on disk is complete.

### National-scale study areas (GVI)

For widely scattered inputs, the engine automatically clusters the features and builds one sampling grid per cluster on a shared reference — no special mode, just upload the file. The chosen projected CRS and estimated distortion appear in the job log.

---

### The Fusion tab, step by step

The form reads top-to-bottom in the order you reason about a run:

1. **Target Configuration** — upload the outcome file, pick the outcome column(s), preview on the map.
2. **Optimization Setup** — pick the **Run mode**:
   - **Cross-sectional** — optionally enable **Date column available?** to route each entity to a year-matched greenery file (the year is only a file key, never a regression input).
   - **Mixed-effects (longitudinal)** — for repeated measures; a date column is required.
3. **Metric File Assignment** — per channel (NDVI, then GVI), set the buffer ladder (min / max / step in metres) and upload one or more files. When years/waves are in play, tag each file with the year(s)/wave(s) it covers; coverage is validated on submit.
4. **Study Details** — CGI formula (`weighted_average` or `synergy`), covariates to control for, the **objective metric**, an optional **spatial-confounding adjustment** (KS-AIC / Spatial+), the **test-set size**, the **stability-selection** knobs, the **per-pixel CGI grid pixel size**, the **[0, 1] composite scaling** toggle, and the **standalone single-metric** option.
5. **🚀 Run Fusion Optimization.**

> [!NOTE]
> Metric files are **uploaded** in step 3 and aligned to your target automatically. (The earlier auto-download path has been removed — provide the GVI/NDVI files the tabs produce.)

### Reading the results panel

After a job completes — or when you click **Load results** on a completed job — the panel renders top-to-bottom:

- **Headline** — held-out test score + 95 % CI, the greenery↔outcome direction, and (when standalones ran) the CGI-vs-best-standalone AIC/BIC verdict.
- **Study detail** — pick a study (CGI or a standalone) to see its score, selected weights / radii / aggregators, per-subset scores, final parameters, and stability diagnostics.
- **CGI vs standalone** — score comparison plus the penalized model verdict (when standalones ran).
- **Channel collinearity** — VIF report, when requested.
- **Covariate impact** — per-covariate coefficient, t-stat, p-value, direction, and partial R² (when covariates are set).
- **Composite map viewer** — selected composites + the target outcome as side-by-side maps.
- **Mixed-effects metrics** — the four MixedLM metrics (longitudinal mode).
- **Output files on disk** — an index of every file the run wrote.

See [OUTPUTS.md](OUTPUTS.md) for the per-job folder layout.

---

## 2. Command-Line Interface (HPC)

The CLI runs the same engines headlessly for large-scale, automated, parallel processing.

```bash
conda activate geofuse

# Example: 4 parallel processes
mpiexec -n 4 python scripts/cli.py --config config.csv
```

| Argument | Description |
| --- | --- |
| `--config` | Path to a CSV defining input files and parameters |

> [!NOTE]
> A template `config.csv` is auto-created if none is found on first run.

To scale on a cluster, swap `mpiexec` for your scheduler's launcher (e.g. `srun` on SLURM):

```bash
srun -n 32 python scripts/cli.py --config config.csv
```

Each MPI rank processes a separate study area; the engines write to `output_results/` with collision-safe naming.
