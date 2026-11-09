# GeoFuse performance reference

Several of the fusion engine's slowest computations have faster equivalents.
This page lists each one: what it replaces, how much faster it is, and how
closely it reproduces the slower method. Most change no estimator, seed,
resample or default, so results agree to floating-point rounding, far below the
precision any table prints. The posterior warm starts (section 4) change the
sampler's draws but not the posterior it samples, so their results differ from
a cold fit by Monte Carlo error, as two seeds do.

**Abbreviations.** CGI = Composite Greenery Index. OLS = ordinary least
squares. R² = coefficient of determination. df = degrees of freedom of the
spatial smooth (radial functions per spatial cluster).

---

## 1. Summary

| Area | Faster by | Memory | Largest difference from the slower method |
| --- | --- | --- | --- |
| Bootstrap intervals with a spatial smooth | 80–110× | — | 4 × 10⁻¹⁶ on the interval ends (synthetic); 4.8 × 10⁻⁵ on one real study, see §2 |
| Permutation tests with a spatial smooth | 300–720× | — | 3 × 10⁻¹⁵; identical p-values |
| Negative-control and specificity tests with a spatial smooth | 9–14× | 2.3 GB → 0.2 GB | 2 × 10⁻¹³ |
| Discovery sweep, reproducibility and gain checks | 120–190× | — | identical picks, forms and weights |
| Synergy posteriors: first fit and later held-out splits | 2.3–3.2× | — | other draws of the same posterior: means within 1.1 Monte Carlo errors of cold fits |
| `geofuse` command-line processes | — | 3.1 GB → 0.02 GB each | — |
| Null calibration: in-process nulls, compiled programs, pool size | 1.14× in-process | peak 13.4 GB → 10.1 GB (2 workers, 16 nulls) | identical null rows |

Section 6 gives the time and memory of a whole study, alone and several at
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

**On real data.** One Spatial+ study (about 17,000 entities, about 40 spatial
clusters, 2,000–10,000 resamples) was rerun with these computations and
compared with its original run, same design and seeds. Point estimates and
slopes agreed to 10⁻¹², and an unadjusted rerun to 10⁻¹¹, but resampled
interval ends differed by up to 4.8 × 10⁻⁵ (at most 1.6 % of a specificity
interval end, 6.8 % of a score interval end close to zero) and p-values by up
to 0.0015, three resamples in 2,000. That is well inside the intervals' own
Monte Carlo error. Synthetic designs built to provoke such a difference all
agree to 10⁻¹⁰ or better: 40 clusters too small for their smooth that
resamples empty, categories of one to three members that resamples drop, and
the original code on 24 against 4 processor threads.

A likely cause was found later. A replicate's fit dropped every direction
whose eigenvalue in its weighted cross-products fell below 10⁻¹⁰ of the
largest, which is a singular value below 10⁻⁵ of the largest: a cluster
that a replicate draws only a few rows from can support a direction that
weakly and still carry information the per-replicate fit uses. Synthetic
designs of that kind (saturated clusters of 4–11 entities, unscaled
covariates, eight seeds) on the constrained thin-plate basis (see FEATURES,
spatial adjustment) differed from the per-replicate fits by up to 1.0 on a
contrast's interval end at that floor. The floor is now 10⁻¹², still four orders above the rounding
level of a direction a replicate does not support, and the same designs
agree to 10⁻⁶ relative or better (most to 10⁻⁷). With the earlier
unconstrained basis two of the eight seeds differed by 0.07 and 13 at either
floor: its small clusters' designs had condition numbers up to 5 × 10⁴
against at most 14 for the constrained basis. The real study has not been
rerun, so its difference is given at the size measured before the change.

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

## 4. Synergy posteriors

The linear form's posterior has a shortcut whose cost does not grow with the
number of entities. The synergy form has none: its interaction terms are
rebuilt from the sampled radius and statistic blend at every step. A cold
synergy fit spends most of its gradient evaluations in warm-up, while the
sampler tunes its step size and its scale for each parameter. Two warm starts
reuse that tuning:

- **The first synergy fit starts from a linear fit** of the same columns, which
  takes seconds. The parameters both forms share start at the linear fit's
  posterior mean with its scales, the interaction weights start small, and a
  250-iteration warm-up tunes everything again.
- **A later held-out split starts from the latest fit** of the same channel and
  form. It keeps that fit's scales and tunes only the step size, over 150
  iterations. Tuning the scales again over so short a warm-up estimates them
  from too few draws: the effective sample size then fell to 50–90 % of the
  source fit's.

| Fit | Warm-up gradients, cold → warm | All gradients, cold → warm | Faster by | Time, cold → warm |
| --- | --- | --- | --- | --- |
| First synergy fit, from a linear fit | 0.99–1.11 M → 0.24 M | 1.38–1.64 M → 0.61 M | 2.3–2.7× | 59–95 → 37–42 min |
| Later held-out split, from the previous split | 0.91–0.99 M → 0.14 M | 1.09–1.50 M → 0.47 M | 2.3–3.2× | 63–84 → 27 min |

The linear fit adds 0.15 M gradients of its own shortcut, 3.6 seconds in all.
Warm-up shrinks 4–7×, after which sampling takes most of the time (this
posterior needs about 450 gradient evaluations per draw), so whole fits are
2.3–3.2× faster.

**How close.** Against two cold fits with other seeds, the posterior mean of
the effect differed by at most 0.0006 (1.1 %) on the first fit and 0.00006
(0.1 %) on a later split, and every channel and interaction weight by at most
0.006. All of these are within 1.1 Monte Carlo standard errors, which is how
far two cold fits with different seeds differ. The largest R-hat stayed at or
below 1.009, and the effective sample size at or above the cold fits'. A warm
fit whose largest R-hat exceeds 1.01, or whose effective sample size per draw
falls below 80 % of its source's, is refitted cold with a warning. Turning the
warm starts off gives the earlier draws exactly.

**The gradient itself.** At 15,000 entities the interaction columns cost about
2.2 ms per gradient: blending the cells 0.27 ms, averaging cells per entity
0.17 ms, removing the covariates 0.45 ms, and the rest spread over many small
operations. Two exact rewrites (square and cube roots in place of a fractional
power; all interaction terms in one array) were not faster, so the gradient is
unchanged.

---

## 5. Memory of a job's processes

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

## 6. A whole study, and several at once

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

## 7. Null calibration

The null calibration of the held-out decision (`outer_null_runs`) runs the
whole search on every split of every null outcome. Measured on a synthetic
study of 3,000 entities, 3 held-out splits, the decision report, free nulls,
no standalones:

| Change | Before | After | Null rows |
| --- | --- | --- | --- |
| In-process nulls (`outer_null_workers = 1`) keep their sweeps in the job's process instead of starting a process pool per sweep | 398 s, 9.1 GB for 8 nulls | 350 s, 8.4 GB | identical |
| A null worker frees JAX's compiled programs once it has grown 1 GiB since it last did | 351 s, 13.4 GB for 16 nulls on 2 workers | 360 s, 10.1 GB | identical |
| An automatically sized pool (16 nulls or more) is sized on one null run in the job's process first | 3 × engine copy + 1.5 GB per worker | that, plus the measured growth of one null, plus 1 GiB | identical |

Every posterior fit embeds its data in its compiled programs, so a program is
never reused by a later null, and a worker that keeps them all grows by about
0.19 GB per null at this size (2 → 8 nulls per worker: 8.7 → 13.4 GB).
Freeing them after every null brought the same run to 7.9 GB but took 13 %
longer (398 s), because the shape-generic helpers were compiled again each
time; freeing at 1 GiB of growth costs 2.5 %. The pool's old memory estimate
does not see this growth: one real study of about 18,000 entities reached an
85 GB process-tree peak with 4 workers. The pilot null costs one null run
sequentially (8 nulls on 4 workers: 151 s → 187 s), which is why it starts
from 16 nulls.

Measured and left out:

- **XLA threads.** `--xla_cpu_multi_thread_eigen=false` in `XLA_FLAGS` changed
  neither a matrix product's time nor the stage (151 s → 153 s, 8 nulls on 4
  workers, identical rows) with JAX 0.11, and `intra_op_parallelism_threads=1`
  in the same variable made XLA ignore the device-count flag after it, which
  would run the chains one after another.
- **Compile reuse** (the data as runtime arguments of one program per shape)
  would change the draws, not only their speed, and was not attempted.

---

## 8. How these figures were measured

Synthetic data only: 4,662 or 20,000 entities in 12 spatial clusters and three
very small ones, 29 covariate columns (2 numeric, 9 categorical), a smooth
spatial pattern shared by the outcome, the exposures and the negative
controls, and the Spatial+ smooth at 10 or 20 df per cluster (169–319 control
columns). Each fast computation ran on the same data, seeds and resamples as
the computation it replaces. The sweep used 15,000 entities and a grid of 3
channels × 10 radii × 6 statistics; the posteriors the same grid with 30
covariates and the synergy form, 2 chains of 400 draws (cold warm-up 800), a
later split keeping 75 % of the entities, the warm later split timed beside two
other fits. Windows 11, 24 logical cores, 32 GB of
memory, numpy with OpenBLAS on 4 threads (the sweep on 1).
