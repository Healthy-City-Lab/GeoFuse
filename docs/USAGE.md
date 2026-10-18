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
| Jobs | Follow and control command-line jobs on this computer and on linked remote machines |

Jobs survive a browser refresh and are tracked in the sidebar job monitor from
every tab. Each engine runs in its own process, so the interface stays
responsive and several study areas can run at once.

### Typical workflow

1. **NDVI tab.** Upload a study area, set a date range, run. Both ends of a
   range are included, so a range can be a single day. For year-by-year
   data choose the per-year mode, pick the year column and the growing-season
   months, and each year becomes its own aligned raster covering those months
   in full (May to September runs to 30 September). Each date mode has a
   **Temporal statistic** picker: median (default), mean, maximum, or any
   combination, each written as its own file. **Aggregation chains**, under
   the picker, replace the statistics with chains, one `name: operation, ...`
   per line (see *NDVI aggregation chains* below). The download settings hold
   **Satellite**, **Reflectance**, **Cloud handling** (drop cloudy scenes,
   mask cloudy pixels, or both), **Coverage rescue**, **Mask water** and
   **Fill 2012 with 2013**. The defaults reproduce the plain composite: water
   kept, both cloud steps on, surface reflectance. Turn rescue off when
   composites must cover exactly the dates you asked for, such as year-matched
   series.
2. **GVI tab.** Upload the same area, set grid resolution and buffer, run. To
   match a past year, target a capture year and optionally cap the year gap.
   **Limit capture months** keeps panoramas to a season, for example May to
   October so trees are in leaf. The per-year mode works as it does for NDVI.
   **Download & GPU Throughput**, collapsed under the settings, sizes the
   download pool and the GPU batches (see *Performance* below).
3. **Fusion tab.** Upload the outcome layer plus the files from steps 1 and 2,
   set the run mode and objective, and run.
4. Read the results panel and export.

### Presets

The NDVI and GVI settings boxes open with a **Preset** row. Choosing a preset
applies it: the download settings, the output formats, and the method settings
of every dataset (temporal statistics, season months and window for NDVI;
capture mode, year limit and month window for GVI). Dates and files are never
part of a preset. The row shows **Custom** once a setting has been changed.
**Save** stores the current settings under a name in `presets/<tab>/` (and
offers the JSON for download); **Load** imports a preset file, checks that it
belongs to this tab and holds valid values, and adds it to your presets. API
keys are never written, and a file containing one is refused. Built-in presets
cover the defaults, a Landsat top-of-atmosphere set with water masking, a
Sentinel-2 10 m set, and historical Street View captures within two years.
A job records the preset it came from in its sidecar.

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

1. **Target configuration.** Upload the outcome, pick the outcome columns, and
   optionally name **negative-control outcomes**: columns expected to share the
   outcome's confounders but not to be caused by greenery, such as grip
   strength for a cognitive outcome. They are never tuned on (cross-sectional
   runs only).
2. **Optimization setup.** Choose cross-sectional or mixed-effects
   (longitudinal). Longitudinal requires a date column. Cross-sectional can
   optionally use one to route each entity to a year-matched greenery file; the
   year is only a file key, never a regression input.
3. **Metric file assignment.** Per channel, set the buffer ladder and assign
   files. A ladder is a min and max in metres plus either a step (linear) or a
   number of rungs (log-spaced); the radii it produces are listed under the
   inputs. When years are in play each year points at exactly one file.
   Anything unassigned blocks submission.
4. **Study details.** Covariates, objective metric, optional spatial-confounding
   adjustment, test-set size, the discovery block (channel set, index form,
   sweep splits, radius prior, minimum GVI coverage) with validation settings
   behind an expander, the CGI grid pixel size (for every target type, rasters
   included), the exposure buffer, composite scaling, and whether to run each
   channel on its own. Choose the range-based radius prior when the distance of
   influence should not depend on the ladder's spacing.
   Longitudinal runs add wave fixed effects, on by default, and an optional
   neighbourhood column. There is no formula selector: the channel set names the
   modalities and the sweep picks the functional form.
5. **Run.**

Covariates listed as categorical are one-hot encoded and counted as covariates
automatically; there is no need to list them twice. **Missing categorical
values**, beside that list, decides what an empty category does: **Drop the
row** (default, as for a missing numeric covariate) or **Keep as a level of its
own** (a `<column>=missing` dummy). In a run file it is
`categorical_missing = "drop"` or `"indicator"`.

### Reading the results panel

- **Headline.** Held-out test score with its 95 % interval, the significance
  test that actually applies (permutation for cross-sectional, Wald for
  mixed-effects), the direction of the relationship, and the verdict against the
  best standalone channel. Under it, the covariates some rows were missing, with
  how each was handled and how many rows with an outcome were dropped.
- **Study detail.** Per study: the discovered radii, aggregators and weights,
  per-subset scores, and the discovery diagnostics, which cover what the sweep
  picked and how contested it was, the posterior weights with credible
  intervals, whether the discovery reproduced across reshuffles, the composite's
  gain over each channel alone, and the false-positive rate on permuted
  outcomes with its exact 95 % interval. The default of 16 null refits is a
  quick check, and the panel says so; set **Null calibration fits** to at least
  100 before claiming calibration, and to 200 for a final, reported run.
  With negative controls, each study also shows the frozen composite's slope on
  the target and on every control, the paired difference, and a warning when a
  control's association cannot be told apart from the target's.
- **GVI coverage.** Which GVI radii were searched, each radius's entity
  coverage, and how many cells and entities the coverage gate left out.
- **Covariate impact**, **channel collinearity**, **composite maps**,
  **mixed-effects metrics**, and an index of every file written. **Generate
  CGI Map**, under the map viewer, is a placeholder for a planned full-area map
  built from the study's definition; it is not available yet.

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

GVI runs end with a `Throughput:` log line: points searched, panoramas
downloaded and segmented, each per second, how busy the GPU was, and throttle
events. These settings size the stages and set the model's precision, in the
GVI tab's **Download & GPU Throughput** box or as run-file keys and
command-line flags:

| Setting | Default | What it does |
| --- | --- | --- |
| `download_concurrency` (**Concurrent Downloads**) | 16 | Panoramas downloaded at once |
| `batch_size` (**GPU Batch Size**) | 1 | Images per GPU forward |
| `batch_wait_ms` (**Batch Wait (ms)**) | 20 | Longest wait for a batch to fill |
| `precision` (**GPU Precision**) | `fp16` | `fp32` repeats the masks of runs made before FP16 was the default |

If the GPU is busy well below 90 % and there are no throttle events, raise
`download_concurrency`; if throttle events climb, lower it, since Google limits
requests per address whatever the pool size. Keep `batch_size` at 1 unless a
larger batch is measurably faster on your GPU: batched forwards change a few
pixels' labels. On one RTX 4060 Ti against Google, FP16 segments about 50
panoramas per second and FP32 about 28 to 31; at those rates the GPU, not
Google, is the limit, so a pool above 16 gains nothing there.

Keep one precision within a study: values differ by about 0.00005 GVI between
the two, far below anything a model of the outcome can see, but a series is
cleaner made one way.

```bash
geofuse gvi city.gpkg --download-concurrency 24
geofuse gvi city.gpkg --precision fp32
```

While a GVI run works, its log gets a summary line at most every 30 seconds:
points done; how many were sampled, had no panorama, had no capture in the
year or month window, or failed; cache hits, downloads, the rate and throttle
events. Warnings and errors appear as they happen. `log_points`
(`--log-points`, or `log_points = true` in a run file) adds a line for every
point and panorama, for debugging; the log then grows with the number of
points.

Per-job logs are written in full to `logs/jobs/<job_id>.log`; the in-app view
keeps only the last 100 lines.

---

## 2. Command line

Every engine also runs from the `geofuse` command, installed with the
environment (`python -m geofuse` does the same). It uses the dashboard's
settings and defaults, so a run started here writes the files a run started in
a tab would. Jobs run in a background supervisor: one terminal starts them,
another follows them, and closing either leaves them running.

### First steps

```bash
conda activate geofuse
geofuse doctor                                   # is this machine ready?
geofuse auth earthengine                         # sign in once (NDVI)
geofuse config set earthengine.project my-cloud-project
geofuse credentials set streetview-key           # optional: GVI scrapes without one
```

**Street View API key (optional).** The scraper needs no key. To download
through Google's documented API instead, create a key in a Google Cloud
project with the **Map Tiles API** enabled and billing on, store it with
`geofuse credentials set streetview-key`, and set `download_mode = "api"` on
the GVI step (or choose **API (Street View)** in the GVI tab, which also takes
a key typed there). `geofuse doctor` then runs one search through the API and
shows Google's reason if it refuses the key. Google bills each tile and search
request; the API sees only the most recent capture at each place.

### One engine, no run file

```bash
geofuse ndvi city.gpkg --range 2019-05-01:2019-09-30 --cloud-max 20
geofuse gvi city.gpkg --capture-mode column --year-column year
geofuse fusion --target cohort.gpkg --outcome cesd --covariate age \
    --ndvi city_ndvi.tif --gvi city_gvi.gpkg
```

Each submits its jobs and returns. `--foreground` runs them in this terminal
instead, `--plan` only lists them, and `--save run.toml` writes the run file the
command stands for. `--ndvi` / `--gvi` take one file, several files matched to
years by name, or `YEAR=FILE` pairs.

### Run files

A run file (TOML) holds a whole analysis: the downloads and the fit that uses
them.

```toml
format = "geofuse-run"
version = 1
output_dir = "results"

[[ndvi]]
name = "city"
study_area = "data/city.gpkg"
year_column = "year"
cloud_max = 20

[[gvi]]
name = "city"
study_area = "data/city.gpkg"
capture_mode = "column"
year_column = "year"

[[fusion]]
name = "cesd"
target = "data/cohort.gpkg"
outcomes = ["cesd"]
covariates = ["age"]
mode = "year-aware"
date_column = "interview_date"
ndvi = "@ndvi.city"        # the files the ndvi step writes ("@ndvi.city[max]" for one statistic or chain)
gvi = "@gvi.city"
```

`geofuse config template pipeline -o run.toml` writes a commented example with
every setting at its default (`ndvi`, `gvi` and `fusion` templates too). A
step's settings resolve in this order, later winning: the default, the step's
`preset`, the file's `[defaults.<engine>]`, the step, the command line. Unknown
keys and out-of-range values are refused with the key named, and so is anything
that looks like a credential. Relative paths are read from the run file's
folder.

```bash
geofuse plan run.toml                       # the jobs, numbered, with settings that differ from the defaults
geofuse submit run.toml                     # start them in the background
geofuse run run.toml                        # or run them here until they end
geofuse submit run.toml --cloud-max 30      # a setting for every step that has it
geofuse submit run.toml --set ndvi.city.cloud_max=30   # one step
geofuse submit run.toml --only gvi.city     # one step; the steps it needs are taken as done
```

A step starts when the steps it references finish; independent steps run side
by side within the limits below.

### NDVI aggregation chains

By default each date mode writes one composite per temporal statistic. A
reference product may instead summarise each scene's neighbourhood before the
temporal statistic, or mask water on the finished composite only. An
aggregation chain states the order:

```toml
[[ndvi]]
name = "city"
study_area = "data/city.gpkg"
year_column = "year"
chains = [
  {name = "mean", steps = ["temporal:mean"]},
  {name = "f500", steps = ["focal:mean:500", "temporal:mean", "water_mask"]},
]
```

Each chain has exactly one temporal step. Steps before it act on every scene,
steps after it on the composite:

| Step | What it does |
| --- | --- |
| `focal:<stat>:<radius_m>` | The mean, max, min or median over a circle of that radius (above 0, at most 5,000 m) around each pixel, as Earth Engine's `reduceNeighborhood` computes it; a masked pixel stays masked |
| `temporal:<stat>` | The median, mean or max over the window's scenes |
| `water_mask` | Masks pixels open water covers at least half the time |

A chain named `f500` writes `city_2018_f500_ndvi.*` and records its steps in
the sidecar. Names use letters, digits and `_`, and must differ ignoring case.
`chains` replaces the statistics of every date mode the step uses;
`chains_ranges`, `chains_specific` and `chains_column` set one date mode. With
a neighbourhood step, scenes are fetched over the study area grown by the
largest radius, so pixels at its edge get whole neighbourhoods. **Mask water**
(`water_mask = true`) still masks every scene first, before each chain. A
fusion step names one chain's files with `@ndvi.city[f500]`. A chain that is
only a temporal step, with or without a leading `water_mask`, shares that
statistic's cached tiles. In the NDVI tab the same chains go into
**Aggregation chains** under each date mode's **Temporal statistic** picker;
a chain that does not check is named under the picker and blocks the run. On
the command line, chains are text:
`--chains-column "f500: focal:mean:500, temporal:mean, water_mask"`.

### Reusing NDVI tiles from a larger area

Every tiled NDVI download records its tile cache: the settings that shape the
values (dates, clouds, sensors, reflectance, resolution, statistic or chain)
and its study area. A later download with the same settings over a smaller
area, such as a narrower buffer or fewer features, takes each tile from that
cache wherever the earlier run covers every pixel the tile needs, and asks
Earth Engine only for the rest. The values are those a fresh download
computes; the sidecar counts the tiles taken (`tiles_salvaged`).

Caches written before these records existed are recorded from the jobs that
wrote them:

```bash
geofuse cache adopt JOB...    # finished, cancelled or interrupted NDVI jobs
```

Downloads small enough for a single tile keep no cache, so there is nothing to
reuse for them.

### Following and managing jobs

```bash
geofuse watch                       # full-screen monitor
geofuse status                      # one table (--json for scripts)
geofuse logs JOB -f                 # a job's log, followed
geofuse pause | resume | cancel JOB...   # or --all
geofuse wait JOB...                 # returns 0 when all completed
geofuse rerun JOB                   # the same job again; caches make it resume
geofuse rm --finished               # tidy the list
```

`geofuse watch` shows every job with its progress and time left, the selected
job's stages, settings and outputs, and its log as it grows. Keys: `p` pause,
`r` resume, `c` cancel (asks first), `a` all / unfinished, `/` filter, `l` wide
log, `s` next source (this computer, then each remote), `q` quit. Quitting
leaves the jobs running.

The dashboard's **Jobs** tab shows the same jobs: choose a **Source**, filter
the table, and pick a **Job** to see its stages, changed settings, outputs and
log, refreshed every two seconds. **Pause** / **Resume**, **Cancel** (confirmed),
**Re-run**, and for a finished fusion run **Load results**, which opens it in
the Fusion tab's results view.

Job ids can be shortened to any unique prefix. The supervisor starts with the
first `submit` and stops after ten idle minutes; `geofuse supervisor --stop`
stops it now. Exit codes: 0 done, 1 a job failed or was cancelled, 2 invalid
settings or inputs, 3 an environment problem. Should the supervisor itself stop
on an error, its jobs show as interrupted, never cancelled, and `geofuse rerun`
continues them from their caches.

### Limits shared with the dashboard

GPU, Street View, Earth Engine and fusion jobs each hold a slot while they run,
whether the dashboard or the command line started them:

```bash
geofuse config show
geofuse config set limits.gpu 2              # two GVI jobs at once
geofuse config set limits.earthengine 3
geofuse config set limits.max_parallel 6     # jobs the supervisor runs at once
```

### Results

```bash
geofuse results JOB                         # held-out score, direction, each channel alone, weights, radii
geofuse results output_results/fusion/<run>/    # any run, dashboard runs included
geofuse export JOB -o cgi.csv               # the per-entity CGI table (.gpkg, .csv or .geojson)
geofuse cgi-map JOB area.gpkg               # planned full-area CGI map; exits 2, not available yet
```

### Presets, caches and credentials

```bash
geofuse preset list
geofuse preset show ndvi "Sentinel-2 10 m"           # as run-file lines
geofuse preset save ndvi "Lenient" --cloud-max 60    # or --from run.toml --step city
geofuse preset import shared.json
geofuse cache list
geofuse cache purge ndvi                # asks first; pano | ndvi | fusion | all
geofuse credentials show                # what is stored, never the values
```

Credentials go to the operating system's keyring (Windows Credential Manager,
macOS Keychain, Secret Service), or, where there is no keyring, to a file in
your configuration folder that only you can read. They never go to run files,
job records, logs or environment variables.

**Earth Engine sign-in without a browser** (a cluster login node): connect with
`ssh -L 8085:localhost:8085 HOST`, run `geofuse auth earthengine` there and open
the address it prints on your own computer. `geofuse auth earthengine
--no-browser` shows a code to paste instead (Google expires those sign-ins after
about a week), and `geofuse credentials set earthengine-service-account
KEY.json` uses a service account, which does not expire.

### On a cluster

The same commands run on any SLURM cluster; nothing is specific to one site.

1. Install the environment and sign in (above). Put the panorama and NDVI tile
   caches on a disk local to the nodes, or on scratch, since they are SQLite
   files and unsafe on a shared home directory:
   `geofuse config set paths.caches /scratch/$USER/geofuse-caches`.
2. Split large GVI steps with `shards = N` in the run file: N jobs over spatial
   blocks of the grid, then a merge that writes the usual outputs.
3. Write the job scripts, submit them, and follow them from the login node:

```bash
geofuse hpc script run.toml --account my-account --gpu-partition gpu \
    --setup "module load cuda; conda activate geofuse"
bash hpc-run/submit.sh
geofuse watch
```

`hpc script` writes one job array per dependency level and resource group (GVI
on a GPU, fusion with `--fusion-cpus` / `--fusion-mem`, the rest small) and a
`submit.sh` that makes each level wait for the one before. Each array task runs
one job, `geofuse run run.toml --level K --group G --array-task N`, which can
also be run by hand; `geofuse plan` shows every job's number, level and group.

### Following a remote machine from this one

A remote link shows another machine's jobs here, in `geofuse watch` and the
Jobs tab, over one SSH login. It needs `ssh` on this computer and GeoFuse on
the other machine; nothing is specific to one cluster or scheduler.

```bash
geofuse remote add cluster login.cluster.example.org --user me \
    --command "conda run -n geofuse geofuse" \
    --remote-state-dir /scratch/me/geofuse-state      # optional
geofuse remote attach cluster       # keep this terminal open
geofuse watch --remote cluster      # or the Jobs tab, Source "cluster"
geofuse remote fetch cluster JOB -o results/   # copy a job's outputs
```

`attach` logs in once (answer the password or second-factor prompt in that
terminal), starts `geofuse agent --stdio` there, and copies the remote jobs and
their logs into this computer's state folder every two seconds. Pause, resume,
cancel and re-run from the monitor or the Jobs tab travel back over the same
login. If the connection drops it reconnects by itself, with growing pauses;
meanwhile both views say the link is down and when the jobs were last read,
and send nothing. `--ssh-option "-i ~/.ssh/key"` adds options for `ssh` and
`scp`; `geofuse remote list` shows each remote and its link.

**Fetch outputs** in the Jobs tab copies a finished remote job's files over the
link into the state folder's `remotes/<name>/outputs/<job>/`. `geofuse remote
fetch` does the same with its own login and sends files over 16 MB through
`scp`.
