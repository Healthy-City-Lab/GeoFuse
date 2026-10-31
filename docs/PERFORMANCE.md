# GeoFuse performance reference

Several of the fusion engine's slowest computations have faster equivalents.
This page lists each one: what it replaces, how much faster it is, and how
closely it reproduces the slower method. None of them changes an estimator, a
seed, a resample or a default, so results agree to floating-point rounding, far
below the precision any table prints.

**Abbreviations.** CGI = Composite Greenery Index. OLS = ordinary least
squares. R² = coefficient of determination. df = degrees of freedom of the
spatial smooth (radial functions per spatial cluster).

---

## 1. Summary

| Area | Faster by | Memory | Largest difference from the slower method |
| --- | --- | --- | --- |
| Bootstrap intervals with a spatial smooth | 80–110× | — | 4 × 10⁻¹⁶ on the interval ends |
| Permutation tests with a spatial smooth | 300–720× | — | 3 × 10⁻¹⁵; identical p-values |
| Negative-control and specificity tests with a spatial smooth | 9–14× | 2.3 GB → 0.2 GB | 2 × 10⁻¹³ |
| Discovery sweep, reproducibility and gain checks | 120–190× | — | identical picks, forms and weights |
| `geofuse` command-line processes | — | 3.1 GB → 0.02 GB each | — |

Section 5 gives the time and memory of a whole study, alone and several at
once.

The first three rows apply to spatially adjusted runs. Runs without spatial
adjustment compute their reports as before.

---

## 2. Reports of a spatially adjusted run

With spatial adjustment on, every bootstrap replicate, permutation and paired
resample refits a regression on the covariates plus the spatial smooth, which
has one block of columns per spatial cluster (150–320 columns in total). Each
refit used to be a full least-squares fit over every row.

The fast version uses three facts:

- **A bootstrap replicate is a weighted fit.** A row drawn twice is the same
  as a row with weight 2, so every replicate's fit follows from weighted
  sums of the design, and one matrix product computes them for hundreds of
  replicates at once.
- **The smooth is block diagonal.** Each cluster's columns are zero outside
  its own rows, so each cluster is handled on its own and the full matrix of
  covariates plus smooth is never formed. This is also what removes the
  memory cost of the negative-control tests.
- **A permutation test keeps the controls fixed.** The exposure's residual is
  computed once, and each permuted outcome then costs two sums over the rows.

The covariates and each cluster's columns are made orthogonal once on the full
sample, which keeps every replicate's fit numerically stable. A replicate that
draws no rows from a cluster simply drops that cluster's columns.

| Computation | n = 4,662 (df 10) | n = 20,000 (df 10) | n = 20,000 (df 20) | Largest difference |
| --- | --- | --- | --- | --- |
| Bootstrap replicate of the incremental R² | 86× | 96× | 106× | 3 × 10⁻¹¹ relative |
| Bootstrap interval (400–1,000 replicates) | 81× | 107× | 110× | 4 × 10⁻¹⁶ |
| Permutation of the incremental R² | 716× | 313× | 422× | 3 × 10⁻¹⁵ |
| One incremental R² score | 1.7× | 1.4× | 1.9× | 7 × 10⁻¹⁶ |
| Negative-control paired contrast (500 resamples) | 9× | — | — | 9 × 10⁻¹⁴ |
| Specificity contrast (500 resamples) | 14× | — | — | 2 × 10⁻¹³ |
| Distance correlation, bootstrap / permutation | 15× / 8× | — | — | 4 × 10⁻¹³ |
| Normalized RMSE, bootstrap / permutation | 12× / 20× | — | — | 8 × 10⁻¹⁴ |

A dash means the slower method was not measured at that size; the previous
negative-control code needs about 15 GB per test at n = 20,000.

**Where it applies.** The incremental R² (the default objective), distance
correlation and normalized RMSE, in spatially adjusted runs with linear
covariate residualization: the held-out interval, the effect bootstrap and
permutation test on every held-out split, the paired CGI-vs-channel and
CGI-vs-comparator differences (incremental R²), the negative-control tests and
the specificity contrast. These keep the slower fits:

- runs without spatial adjustment, where the fits take seconds;
- spline covariate residualization, whose spline knots move with each
  replicate's rows;
- partial rank correlation (`spearman`), whose covariates are re-ranked in
  each replicate;
- paired differences of distance correlation and normalized RMSE.

---

## 3. Discovery sweep

The sweep scores every combination of radius and statistic across the
channels, 216,000 combinations per split for three channels of 60 cells each.
Each combination's weights come from a small closed-form fit. These fits are
now solved for all combinations at once, in the same order and with the same
rule for ties, and each candidate's training score comes from sums over the
rows rather than from the candidate's own index. The values reported for the
chosen combination are computed exactly as before.

| Step (one processor core) | Before | Now | Faster by |
| --- | --- | --- | --- |
| Sweep, 4 splits | 269 s | 2.2 s | 122× |
| Reproducibility check, 2 shuffles | 231 s | 1.2 s | 187× |
| CGI-vs-channel gain, 2 splits | 240 s | 2.0 s | 118× |

The sweep's scores agree to 2 × 10⁻¹⁵; the picks, forms, weights and every
reported value are identical.

---

## 4. Memory of a job's processes

numpy commits about 3 GB of memory per process when it is free to size its
linear-algebra library for every core, whether or not it uses it. Three
settings bound this:

- **Command-line processes** (`geofuse status`, `watch`, `agent`, the
  supervisor) run on one thread: 0.02 GB each instead of 3.1 GB. A
  `geofuse watch` window left open no longer holds 3 GB.
- **Fusion jobs** run on 4 threads (`limits.job_threads`; `geofuse config set
  limits.job_threads 8` to change it). Each worker of their process pools runs
  on one.
- **Pool workers** cap GDAL's raster cache at 128 MB. Left alone the cache
  grows towards 5 % of the machine's memory in every worker that reads a large
  raster; a national NDVI mosaic reached 1.26 GB per worker. Set
  `GDAL_CACHEMAX` to change the cap.

With these, a pre-aggregation worker holds 0.28–0.38 GB and a posterior worker
0.2–0.4 GB. Every run's log and `results_bundle.json` record the wall clock and
the peak memory of each stage.

---

## 5. A whole study, and several at once

A synthetic study the size of a national cohort run (20,000 entities in 12
cities, an NDVI raster and 720,000 street-view points, 11 covariates, two
negative controls, one comparator, three channels on 10-rung ladders, Spatial+
adjustment, two held-out splits with the specificity contrast, standalones on):

- **At the default budgets** it took 24–30 minutes. The posterior of the
  synergy form takes most of that: a held-out split that chose synergy spent
  22 minutes in its posterior, against 30 seconds for one that chose the
  linear form, whose posterior has a shortcut that does not grow with the
  number of entities. Everything else in the study took 3–5 minutes.
- **With smaller budgets** (20 sweep splits, 400 posterior draws, 200–500
  bootstrap replicates, 100 permutations) it took 2.3 minutes, which is what
  the comparison below uses. Memory peaks in pre-aggregation and the
  posterior, not in the reports, so the budgets barely change it.

| Studies at once | Time per study | Studies per hour | Peak memory, all studies | Memory left free |
| --- | --- | --- | --- | --- |
| 1 | 2.3 min | 26 | 7.3 GB | 9.7 GB |
| 2 | 3.1 min | 38 | 14.5 GB | 6.5 GB |
| 3 | 4.0 min | 45 | 21.7 GB | 3.4 GB |
| 4 | 4.8 min | 50 | 26.8 GB | 1.2 GB |

Measured on a 32 GB machine with about 10 GB free beside the usual desktop
applications. Each study's peak (7.2–7.5 GB) is its pre-aggregation pool of 19
workers; a study whose greenery is already cached skips that stage and peaks
lower. Three studies at once (`geofuse config set limits.fusion 3`) is the
sensible limit here: four still finished, but left too little memory for
anything else.

---

## 6. How these figures were measured

Synthetic data only: 4,662 or 20,000 entities in 12 spatial clusters and three
very small ones, 29 covariate columns (2 numeric, 9 categorical), a smooth
spatial pattern shared by the outcome, the exposures and the negative
controls, and the Spatial+ smooth at 10 or 20 df per cluster (169–319 control
columns). Each fast computation ran on the same data, seeds and resamples as
the computation it replaces. The sweep used 15,000 entities and a grid of 3
channels × 10 radii × 6 statistics. Windows 11, 24 logical cores, 32 GB of
memory, numpy with OpenBLAS on 4 threads (the sweep on 1).
