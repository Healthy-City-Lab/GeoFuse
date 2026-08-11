# GeoFuse usage guide

## 1. Web dashboard

```bash
conda activate geofuse
streamlit run ui/app.py
```

| Tab | Purpose |
| --- | --- |
| NDVI | Upload study areas, set dates, run Earth Engine downloads, inspect on a map |
| GVI | Upload study areas, generate sampling grids, run Street View segmentation |
| Fusion | Upload an outcome plus your GVI/NDVI files, discover the composite index, read the results |
| HPC monitoring | Work in progress. Track CLI jobs by ID |

Jobs survive a browser refresh and are tracked in the sidebar job monitor from
every tab. Each engine runs in its own process, so the interface stays
responsive and several study areas can run at once.

### Typical workflow

1. **NDVI tab.** Upload a study area, set a date range, run. For year-by-year
   data choose the per-year mode, pick the year column and the growing-season
   months, and each year becomes its own aligned raster.
2. **GVI tab.** Upload the same area, set grid resolution and buffer, run. To
   match a past year, target a capture year and optionally cap the year gap.
   The per-year mode works as it does for NDVI.
3. **Fusion tab.** Upload the outcome layer plus the files from steps 1 and 2,
   set the run mode and objective, and run.
4. Read the results panel and export.

### Combining files

Both metric tabs run one job per uploaded file by default. Set file handling to
merge into groups and drag files together to concatenate them into a single job.
This is how you get one output for a year that is split across files, such as a
survey wave boundary falling mid-year.

### Re-running and resuming

Click the re-run icon on a completed job and the setup form comes up filled in
from that job. Nothing is submitted until you press run, so any setting can be
changed on the way. Settings that did not exist when the job was recorded keep
their defaults.

If the app or machine restarts mid-job, affected jobs reload as interrupted.
Re-upload the same study area and choose resume; the file is hash-verified, so a
mismatched file is refused, and processed points are skipped.

### The Fusion form, in order

1. **Target configuration.** Upload the outcome, pick the outcome columns.
2. **Optimization setup.** Choose cross-sectional or mixed-effects
   (longitudinal). Longitudinal requires a date column. Cross-sectional can
   optionally use one to route each entity to a year-matched greenery file; the
   year is only a file key, never a regression input.
3. **Metric file assignment.** Per channel, set the buffer ladder (min, max,
   step in metres) and assign files. When years are in play each year points at
   exactly one file. Anything unassigned blocks submission.
4. **Study details.** Covariates, objective metric, optional spatial-confounding
   adjustment, test-set size, the discovery block (channel set, index form,
   sweep splits) with validation settings behind an expander, per-pixel grid
   size, composite scaling, and whether to run each channel on its own.
   Longitudinal runs add wave fixed effects, on by default, and an optional
   neighbourhood column. There is no formula selector: the channel set names the
   modalities and the sweep picks the functional form.
5. **Run.**

Covariates listed as categorical are one-hot encoded and counted as covariates
automatically; there is no need to list them twice.

### Reading the results panel

- **Headline.** Held-out test score with its 95 % interval, the significance
  test that actually applies (permutation for cross-sectional, Wald for
  mixed-effects), the direction of the relationship, and the verdict against the
  best standalone channel.
- **Study detail.** Per study: the discovered radii, aggregators and weights,
  per-subset scores, and the discovery diagnostics, which cover what the sweep
  picked and how contested it was, the posterior weights with credible
  intervals, whether the discovery reproduced across reshuffles, the composite's
  gain over each channel alone, and the false-positive rate on permuted
  outcomes.
- **Covariate impact**, **channel collinearity**, **composite maps**,
  **mixed-effects metrics**, and an index of every file written.

See [OUTPUTS.md](OUTPUTS.md) for the folder layout.

### Performance

Every run logs a stage-by-stage wall-clock breakdown, and parallel phases log
how many workers were actually busy and what limited the pool.

Thread pools take about a third of the cores, because each task is already
multi-threaded inside numpy. Process pools take 80 % by default and are also
bounded by free memory, since every worker is a fresh interpreter. A log line
saying `limited by free memory` means closing other applications would widen the
pool; `limited by cores` means the machine is fully used.

```bash
GEOFUSE_CPU_SHARE=0.5 streamlit run ui/app.py   # halve the process-pool share
GEOFUSE_WORKERS=8 streamlit run ui/app.py       # cap every pool at 8 workers
```

Per-job logs are written in full to `logs/jobs/<job_id>.log`; the in-app view
keeps only the last 100 lines.

---

## 2. Command line, for HPC

The same engines run headlessly under MPI.

```bash
conda activate geofuse
mpiexec -n 4 python scripts/cli.py --config config.csv
```

`--config` points at a CSV defining input files and parameters. A template is
created on first run if none exists. On a cluster, swap `mpiexec` for your
scheduler's launcher, for example `srun -n 32`. Each rank processes a separate
study area and writes to `output_results/` with collision-safe naming.
