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
   strength for a cognitive outcome. They are never tuned on, and are scored
   for cross-sectional runs, year-matched ones included, but not for panels.
   **Comparator exposures** are existing exposure columns to benchmark the
   composite against, such as a published buffer mean. They are never tuned:
   each is scored on the composite's own held-out entities, covariates and
   objective, with a paired composite-vs-comparator test. **Primary
   comparators** are the pre-specified ones the Holm correction runs over
   (empty = all). In a run file: `comparators = [...]`,
   `comparators_primary = [...]`; on the command line `--comparator` and
   `--comparator-primary`. With controls picked, **Specificity contrast**
   tests whether the composite is more specific than each primary comparator
   and the comparison channel (`specificity_contrast = true`,
   `--specificity-contrast`).
2. **Optimization setup.** Choose cross-sectional or mixed-effects
   (longitudinal). Longitudinal requires a date column. Cross-sectional can
   optionally use one to route each entity to a year-matched greenery file.
   The years then also enter as fixed effects, which absorb differences
   between the yearly files; with one row per entity the run is otherwise
   cross-sectional, with no per-entity terms.
3. **Metric file assignment.** Per channel, set the buffer ladder and assign
   files. A ladder is a min and max in metres plus either a step (linear) or a
   number of rungs (log-spaced); the radii it produces are listed under the
   inputs. When years are in play each year points at exactly one file.
   Anything unassigned blocks submission.
4. **Study details.** Covariates, objective metric (Incremental R² by default;
   choose it by the association you expect, see FEATURES), optional
   spatial-confounding adjustment (a coordinate smooth in the held-out scores,
   paired tests, slopes, negative controls and specificity contrast; the tuning
   does not use it; its settings are under **Spatial adjustment — advanced**,
   see below), test-set size and split seed, the discovery block (channel
   set, index form, sweep splits, radius prior, minimum GVI coverage) with
   validation settings behind an expander (including "Held-out splits", which
   repeats the whole tuning on that many test splits), the CGI grid pixel size (for every target type, rasters
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

### Spatial smooth and spline settings

The spatial smooth compares a list of candidate sizes and keeps the one its
criterion prefers, or none. A size is the number of radial functions per
spatial cluster, on top of each cluster's level and linear trend (`0`). When
the largest candidate wins, the log warns and the results record `at_cap`: a
larger smooth might fit better, so add larger values (and raise the knots) to
let the criterion decide. Every selection's criterion values are kept in
`results_summary.json` (see OUTPUTS). With the default list a spatially
adjusted run can pick a larger smooth than the runs made before the list
existed; `spatial_adjust_max_df = 10` repeats those runs exactly.

| Setting (form label) | Default | What it does |
| --- | --- | --- |
| `spatial_adjust_df` (**Spatial df candidates**) | `[0, 1, 2, 3, 4, 6, 8, 11, 15, 20]` | Sizes compared, besides no smooth; a list in a run file, `0, 2, 5` in the form, `--spatial-adjust-df 0 2 5` on the command line |
| `spatial_adjust_max_df` | unset | Older form of the list: every size from 0 to N. Give one or the other |
| `spatial_adjust_criterion` (**Spatial df criterion**) | `aic` | `bic` penalises size more, so it picks a smaller or equal smooth |
| `spatial_adjust_eps_m` (**Cluster gap eps**) | 0 (automatic) | Entities farther apart than this fall into different clusters |
| `spatial_adjust_neighbours` (**Cluster neighbours**) | 10 | With eps automatic, entities linked through this many nearest neighbours form one cluster |
| `spatial_adjust_knots_min` / `_max` (**Fewest / Most knots per cluster**) | 35 / 150 | Knots of each cluster's smooth; a cluster holds at most `knots_max` radial functions |
| `spatial_paired_smooth` (**Paired-test smooth**) | `own` | Spatial+ only: the smooth each exposure of a paired CGI-vs-exposure test is scored with, see below |
| `spatial_adjust_scale_m` (**Spatial pattern scales (m)**) | unset | Define the smooth by a distance instead of a size, see below; a list in a run file, `5000, 20000, 10000` in the form, `--spatial-adjust-scale-m 5000 20000` on the command line |
| `spatial_adjust_min_component` (**Smallest cluster**) | 0 (30 with scales, none without) | A smaller cluster shares its nearest larger cluster's level instead of having its own |
| `spatial_tune_scale_m` (**Tune within areas of (m)**) | unset (0) | Tune the composite on variation within areas of about this size, see below |
| `residualize_spline_df` (**Covariate spline df**) | 4 | Spline size of each continuous covariate when **Covariate residualization** is Spline |
| `residualize_spline_min_unique` (**Distinct values for a spline**) | 7 | A covariate with fewer distinct values (a code, a dummy) enters linearly |
| `exposure_response_spline_df` (**Exposure-response spline df**) | 4 | Spline size of the exposure-response curve and its linearity test |

Under Spatial+ the smooth is selected on the exposure, so the CGI and the
exposure it is compared with (a channel or a comparator) can each have their
own. `spatial_paired_smooth` decides what a paired test uses: `own` (each
exposure's own smooth, as when it is scored alone, so the paired difference is
the difference of the two printed held-out scores), `cgi` (the CGI's smooth for
both, the behaviour before this setting existed), `outcome` (one smooth
selected on the outcome given the covariates, the KS-AIC rule) or `max` (the
larger of the two exposures' sizes for both). Under KS-AIC the smooth is
selected on the outcome already, so every value behaves as `outcome`, and the
paired rows record it so.

**Pattern scales.** A size per cluster removes patterns of different lengths
in different places: 20 functions in a cluster 800 km across remove regional
gradients, in a cluster 10 km across neighbourhood ones. `spatial_adjust_scale_m`
defines the smooth by a distance instead, so it means the same everywhere:
exposure variation coarser than about that scale is treated as confounded and
removed. Each cluster gets as many knots as its occupied area needs at that
scale, every radial function is used and no criterion runs. The scale is an
assumption about where confounding acts, so choose it from the study's
hypothesis rather than from a criterion (a criterion on the exposure keeps
preferring the smallest scale offered), and keep it above the error in the
entities' locations. With a scale set:

- clusters split where entities are more than `spatial_adjust_eps_m` apart,
  by default twice the first scale, and a cluster smaller than
  `spatial_adjust_min_component` (default 30) joins the nearest larger one;
- the smooth depends on the locations only, so the CGI, every channel,
  comparator and negative control share it, KS-AIC and Spatial+ give the same
  numbers, and paired tests use it on both sides;
- with several scales, as in `[5000, 20000, 10000, 3000]`, the first is the
  primary one (scores, paired tests, verdicts) and every slope is also reported
  at the others and without a smooth, on every held-out split (one split when
  **Held-out splits** is 1), so one run gives the curve of the slope across
  scales. The extra scales cost one more set of slopes and bootstraps each;
- a cluster needing more than 400 radial functions, or a smooth above 2,000 in
  all, is logged as a warning: a larger scale or a smaller cluster gap keeps it
  smaller.

Without a scale, the run warns when one cluster is more than five times as
wide as the typical one, or holds over a quarter of the entities while twice
as wide: the neighbour rule has then chained distant places into one cluster,
and a cluster gap or a scale separates them.

**Tuning within areas.** The smooth normally enters the reporting only: the
composite is tuned on all variation the covariates leave, then scored within
areas. That answers whether a choice made on the whole map survives within
areas. `spatial_tune_scale_m` asks the other question, which layer, radius,
statistic and weights carry the association among entities in the same area:
the smooth at that scale, built as above on the rows each search uses (every
held-out split builds its own), joins the covariates of the whole search, so
the sweep, the posterior, the null calibration, the standalone studies and the
negative-control re-tune all work on variation within areas. With it:

- the scores and slopes use the same scale unless `spatial_adjust_scale_m` is
  set; a first scale different from the tuning scale is allowed but logged as
  a warning, since the primary results then answer a different question than
  the search did;
- the composite's weights on the raw scale refer to the spread within areas,
  and the record gives, per channel, the share of its variance the smooth
  absorbs: a channel that is mostly regional has little left to tune on;
- one tuning scale per run: compare scales with one run each;
- a smooth with more columns than a quarter of the rows, or a cluster with
  nearly one knot per entity, is logged as a warning, since the search could
  then fit noise.

### Posterior warm starts

A synergy posterior sums its likelihood over the rows, and most of a cold fit's
time goes into tuning the sampler during warm-up. Two warm starts shorten that
without changing what the posterior targets:

| Setting (form label) | Default | What it does |
| --- | --- | --- |
| `synergy_warm_start` (**Start synergy from a linear fit**) | on | A short linear fit of the same columns starts the synergy sampler: its step size, its scales for the shared parameters and its mean as the starting point, the interaction weights starting small. The synergy warm-up is then 250 iterations |
| `outer_warm_start` (**Warm-start later splits**) | on | Each held-out split after the first starts every posterior from the latest fit of the same channel and form: its starting point and its scales for every parameter, with a 150-iteration warm-up that tunes only the step size |
| `outer_posterior_draws` (**Draws on later splits**) | unset (as the first) | Draws per chain on the later splits only; their warm-up follows unless `outer_posterior_warmup` is set |
| `outer_posterior_warmup` / `outer_posterior_chains` | unset (as the first) | Warm-up and chains on the later splits only (run file and command line) |

A warm-started fit is checked against the fit it started from: when its
largest R-hat is above 1.01, or its effective sample size per draw falls below
80 % of the source's, it is refitted cold with a warning. Every fit records
where it started, its warm-up and any fallback (see OUTPUTS). Turning a warm
start off gives the earlier draws exactly.

### Reading the results panel

- **Headline.** Held-out test score with its 95 % interval, the significance
  test that actually applies (permutation for cross-sectional, Wald for
  mixed-effects), the direction of the relationship, and the verdict against the
  channel that scores best on train+val: "CGI vs standalone — objective
  (held-out)" is the paired held-out test, and "CGI vs standalone — AIC/BIC
  (in-sample)" beside it is a secondary statistic. Under it, the covariates
  some rows were missing, with how each was handled and how many rows with an
  outcome were dropped.
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
- **CGI vs comparator exposures.** Each comparator's held-out score, interval,
  p-value, n and β next to the composite's, with the paired difference and its
  Holm p-value (descriptive for non-primary comparators).
- **Specificity contrast** (when on). Per control and exposure: the four
  slopes, each exposure's |β target| − |β control|, and their difference with
  its paired interval, p and Holm p; positive means the composite is the more
  specific.
- **Across held-out splits** (with several splits). Each result's spread,
  share of splits holding and corrected p, the specificity contrast per
  control included; **Posterior across held-out splits** gives the range of
  β, each term's weight and R50 / R90, the share of splits in which a term
  has the largest weight, and the forms chosen.
- In a spatially adjusted run the negative-control, comparator and
  specificity tables add a **Spatial smooth** column with the df of the smooth
  each slope adjusts for, and the captions say what was adjusted.
- **Covariate impact**, **channel collinearity**, **composite maps**,
  **mixed-effects metrics**, and an index of every file written. **Generate
  CGI Map**, under the map viewer, is a placeholder for a planned full-area map
  built from the study's definition; it is not available yet.

See [OUTPUTS.md](OUTPUTS.md) for the folder layout.

### Performance

Every run logs a stage-by-stage wall-clock breakdown with the peak memory the
job's processes held during each stage (the commit charge on Windows), and
parallel phases log how many workers were actually busy and what limited the
pool. The same breakdown is in `results_summary.json` (`stage_timing`,
`peak_memory_gb`).

A fusion job runs its own numpy / BLAS on 4 threads (`limits.job_threads`,
`geofuse config set limits.job_threads 8`): the engine gains little beyond
four, and each thread's buffers count against the machine's memory. Its process
pools run one thread per worker and cap GDAL's raster cache at 128 MB per worker
(set `GDAL_CACHEMAX` to change it). The command's own processes (the
supervisor, `watch`, `status`, the remote agent) compute nothing heavy and run
on one thread, a few tens of MB each.

How many fusion studies to run at once (`limits.fusion`) depends on memory:
each holds about 7 GB at its peak while it pre-aggregates the greenery. On a
32 GB machine three at once is the measured limit; see
[PERFORMANCE.md](PERFORMANCE.md) for the figures.

Thread pools take about a third of the cores, because each task is already
multi-threaded inside numpy. Process pools take 80 % by default and are also
bounded by free memory, since every worker is a fresh interpreter. A log line
saying `limited by free memory` means closing other applications would widen the
pool; `limited by cores` means the machine is fully used.

The posterior logs its own time, and the null calibration logs a progress line
about every tenth of its refits with the time left.

The posterior's chains (**Chains**, 4 by default) run in parallel, one CPU core
each. A log line saying the chains `run one after another` means the process
had already started JAX with fewer CPU devices than chains; the result is valid,
only slower.

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
geofuse config set limits.job_threads 8      # BLAS threads of each fusion job
```

### Results

```bash
geofuse results JOB                         # held-out score, direction, spatial adjustment, each channel alone, specificity, splits, weights, radii
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
