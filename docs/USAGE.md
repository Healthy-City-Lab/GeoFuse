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
> Jobs survive a browser refresh. Progress is tracked in the **sidebar Job Monitor**, visible from every tab, for all job types (GVI, NDVI, Fusion). GVI, NDVI, and Fusion each run in a separate process so the UI stays responsive and a running job isn't slowed down by keeping the browser tab in the foreground. Multiple study areas with different settings can run in parallel.

### Combining multiple files (NDVI & GVI)

Both tabs run each uploaded file as its own job by default. To combine files, set **File handling** to **Merge into groups**, then **drag files between groups** (use **➕ Add group** to make more, rename a group in its box, 🗑 to remove one). Files in the same group are concatenated into one layer (reprojected to a common CRS) and run as a single job, with the group's own date settings. This is the way to make one output for a year that is split across files (e.g. survey waves where 2020 falls in wave 1 for some records and wave 2 for others). Merged outputs are named from the group; leave files in separate groups to keep one job per file.

Merged jobs record all of their source files. If such a job is interrupted, the restart panel checks every source file: when they're all still on disk unchanged it offers a **quiet Re-run** (no re-upload); otherwise it asks you to re-upload the group's files. (Drag-and-drop grouping uses the `streamlit-sortables` package — it's in the environment installer; if absent, the tab falls back to an editable group table.)

### Typical workflow

1. **NDVI tab** — upload a study area, set a date range, pick output formats (GeoTIFF default), run. For year-by-year data, use **Attribute Column (per-year)**: pick the year/date column and the growing-season months, and each year is processed as its own aligned NDVI raster into a `{name}_temporal_ndvi/` folder.
2. **GVI tab** — upload the same study area, set grid resolution + buffer, pick output formats (GeoPackage default), generate grids, run. To match a past year, enable **Target a specific capture year** and set the year; add **Limit maximum year difference** to skip points with no capture close enough. The chosen capture date appears in the `pano_date` output column and the sample-point tooltips. For year-by-year data, enable **Use a year/date column (per-year runs)** and pick the column — each year runs separately (target year = that year) into a `{name}_temporal_gvi/` folder, all sharing one CRS.
3. **Fusion tab** — upload an outcome target — a vector (GeoJSON, Shapefile, or GeoPackage) or raster (GeoTIFF) layer — plus the GVI + NDVI files from steps 1–2, choose a run mode and objective, and run the optimization.
4. Inspect the composite weights, statistics, and maps; export results.

### Re-running a finished job

Click **🔄** on a completed job and confirm on the job card itself. The setup form comes up filled in from that job — run mode, target and metric files with their year assignment, outcome, covariates, standalone studies, buffer ladders and every search setting. Nothing is submitted until you press the usual run button, so any setting can be changed on the way. A file that has since moved or been deleted is flagged in its own picker entry; a file that merely changed is used as it now stands. Tick **Clear this target's pre-aggregation cache first** when the metric files changed at the same path.

Jobs recorded before a given setting existed cannot seed it, and that setting keeps its default. This applies to the per-file year column, which older records did not store.

### Resuming after a crash or restart

If Streamlit (or the machine) restarts mid-job, the affected jobs reload as **Interrupted**. To resume, return to the originating tab and re-upload the **same** study area file. A restart panel appears with **Resume Job** / **Discard**; resume continues from where it stopped (processed points and cached data are skipped). The re-uploaded file is hash-verified, so a mismatched file is refused.

### Per-job logs

Each job writes a full log to `logs/jobs/<job_id>.log`. The job-monitor expander has an **Open log file** button; the in-UI view keeps only the last 100 lines, but the file on disk is complete.

### Where the time went, and how many cores were used

Every fusion run ends with a **stage wall-clock breakdown** in its log — each stage's minutes and share, longest first — and the job-monitor stage list shows the same durations live. Parallel phases additionally log how many workers were actually busy, whether the pool or the work was the limit, and how many cores sat idle.

Pool sizes come from the host's core count, and the share depends on what the phase is waiting for:

- **Thread pools** (the stability search, the scorers) take about a *third* of the cores. Each task is already multi-threaded inside numpy, so a wider pool oversubscribes the cores rather than going faster.
- **Process pools** (the greenery pre-aggregation build) take about *two thirds*. That work is a long chain of small numpy calls per pixel, which spend most of their time holding the interpreter lock, so threads leave the machine idle no matter how many you open — worker processes each bring their own interpreter, and every worker is pinned to a single numpy thread so the shares don't compound.

**Trials run in parallel for every objective metric.** The one exception is an exact `statsmodels` MixedLM refit (`mixedlm_lr`, `mixedlm_marginal_r2`), which is dominated by Python-level optimiser work holding the interpreter lock — threading it measured 0.62x of serial, so it is deliberately left sequential.

The process pool is bounded by **memory** as well as by cores, because every worker is a fresh interpreter. Each run logs which limit set the width:

```text
aggregating 994,539 pixel(s) across 12 worker process(es) — limited by free memory (9.4 GB).
```

`limited by cores` means the machine is fully used. `limited by free memory` means closing other applications (or enlarging the Windows page file) would widen the pool and speed the stage up. If the pool is at full width but the *busy* worker count is far lower, the pool is not the limit — check whether the metric files sit on a slow or networked drive.

Set `GEOFUSE_WORKERS` to override either width when the machine is shared or when you want to test a different one:

```bash
GEOFUSE_WORKERS=8 streamlit run ui/app.py     # cap every pool at 8 workers
```

### National-scale study areas (GVI)

For widely scattered inputs, the engine automatically clusters the features and builds one sampling grid per cluster on a shared reference — no special mode, just upload the file. The chosen projected CRS and estimated distortion appear in the job log.

---

### The Fusion tab, step by step

The form reads top-to-bottom in the order you reason about a run:

1. **Target Configuration** — upload the outcome file, pick the outcome column(s), preview on the map.
2. **Optimization Setup** — pick the **Run mode**:
   - **Cross-sectional** — optionally enable **Date column available?** to route each entity to a year-matched greenery file (the year is only a file key, never a regression input).
   - **Mixed-effects (longitudinal)** — for repeated measures; a date column is required.
3. **Metric File Assignment** — per channel (NDVI, then GVI), set the buffer ladder (min / max / step in metres) and upload one or more files. When years/waves are in play, pick the covering file for each year from its own compact picker; a year can point at only one file. Flip **Drag and drop** to move year chips onto files instead. Anything left unassigned blocks submission.
4. **Study Details** — CGI formula (`weighted_average` or `synergy`), covariates to control for, the **objective metric**, an optional **spatial-confounding adjustment** (KS-AIC / Spatial+), the **test-set size**, the **stability-selection** knobs, the **per-pixel CGI grid pixel size**, the **[0, 1] composite scaling** toggle, and the **standalone single-metric** option. Longitudinal runs add **wave fixed effects** (on by default) and an optional **neighbourhood / site column**.
   - Covariates listed under **Categorical** are one-hot encoded and counted as covariates automatically — there is no need to add them to both lists.
5. **🚀 Run Fusion Optimization.**

> [!NOTE]
> Metric files are **uploaded** in step 3 and aligned to your target automatically. (The earlier auto-download path has been removed — provide the GVI/NDVI files the tabs produce.)

### Reading the results panel

After a job completes — or when you click **Load results** on a completed job — the panel renders top-to-bottom:

- **Headline** — held-out test score + 95 % CI, the held-out significance (labelled with the test it actually is — permutation for cross-sectional objectives, Wald for mixed-effects), the greenery↔outcome direction, and (when standalones ran) the CGI-vs-best-standalone verdicts.
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
