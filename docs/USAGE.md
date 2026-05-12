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
| **Job Monitor** | Track HPC CLI jobs by Job ID; view stage, progress, and metrics in real time. UI job progress is tracked in the GVI tab sidebar. |
| **NDVI Sourcing** | Upload study areas, configure date modes, run Earth Engine downloads, and inspect results on an interactive map. |
| **GVI Sourcing** | Upload study areas, generate sampling grids, run Street View + segmentation batch jobs, and visualize vegetation/terrain heatmaps. |
| **Fusion & Optimization** | Upload a target file (GeoJSON or GeoTIFF), configure metric sources and optimization settings, run Bayesian fusion, and inspect results. |

### Typical Workflow

1. **NDVI tab** → upload a study area file → set date range → "Run NDVI Analysis"
2. **GVI tab** → upload the same study area → set grid resolution → "Start Batch Analysis"
3. **Fusion tab** → upload a target outcomes file (GeoJSON or GeoTIFF) → select loaded GVI/NDVI results → "Run Fusion Optimization"
4. Inspect the composite greenery weights and export results.

> [!NOTE]
> Long-running jobs (GVI, NDVI, Fusion) execute in a process-level thread pool and **survive browser refresh**. Progress is tracked in the **sidebar Job Monitor** on the GVI tab, which is visible from all tabs. Jobs interrupted by a Streamlit restart reload automatically and appear as "Interrupted" with a re-submit prompt. Multiple study areas with different settings (resolution, buffer, dates) can run in parallel as separate jobs.

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
