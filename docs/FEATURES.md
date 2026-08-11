# GeoFuse feature reference

GeoFuse measures two complementary views of urban greenery and fuses them into
a single composite index tuned against a health or environmental outcome. This
page says what each part does and what it expects. For how anything works
internally, read the code.

**Abbreviations.** GVI = Green View Index (eye-level greenery). NDVI =
Normalized Difference Vegetation Index (overhead greenery). CGI = Composite
Greenery Index (the fused index this toolbox produces). CRS = coordinate
reference system. CrI = credible interval.

---

## 1. Eye-level greenery: GVI

The share of a street-level view that is vegetation.

**Input.** A study area as GeoJSON, Shapefile or GeoPackage.
**Output.** A GeoPackage point layer (canonical), optionally per-cluster GeoTIFF
tiles, GeoJSON, and the raw panoramas and segmentation masks.

- Sources Google Street View panoramas over the study area and labels each with
  a DeepLabV3+ model trained on the Cityscapes classes. GVI is the share of
  pixels classed vegetation plus terrain.
- Can target a historical capture year rather than the latest imagery, with an
  optional maximum year gap. The capture date lands in a `pano_date` column, so
  year-matched runs are usable for longitudinal analysis.
- Can split a layer by a year or date column and produce one output per year.
- Files can be merged into groups before running, so a year split across several
  files yields a single output.
- Projects each study area to the most accurate planar CRS for its extent, so
  grid cells are true squares in metres. Distortion above 2 % is warned about.
- Panoramas are cached across runs. Interrupted jobs resume after the study area
  is re-uploaded and hash-verified.

---

## 2. Overhead greenery: NDVI

Vegetation greenness from satellite imagery, via Google Earth Engine.

**Input.** A study area, plus dates.
**Output.** A single-band GeoTIFF (default), optionally GeoPackage, GeoJSON,
per-cluster tiles, and a metadata sidecar.

- Fetches cloud-masked Sentinel-2 or Landsat imagery. `auto` picks Sentinel-2
  from 2017 onward and Landsat before that, harmonising the older TM/ETM+
  sensors to the Landsat-8 scale so values stay comparable across years.
- Dates can be given as ranges, as specific dates with a window, or as a year
  column that splits the layer and composites each year over the growing-season
  months you choose.
- Tiles export on a shared snap grid and are mosaicked without a second
  reprojection, so output pixels are exactly what Earth Engine produced.
- Distinguishes "no images in range" from "all images too cloudy", widens a
  sparse window once and records that it did, and lists any missing tiles in the
  sidecar.
- Can optionally sample values onto your own features, exactly or as a zonal
  statistic over a buffer.

---

## 3. Fusion: discovering the composite index

The analytical core. It discovers how to combine the greenery channels into one
CGI that best tracks an outcome you supply, and reports that relationship with
held-out statistics.

**Input.** Your outcome layer (points, lines, polygons or raster), the GVI and
NDVI files from the tabs above, and the covariates to control for.
**Output.** Composite rasters plus a results folder of tidy CSVs and a JSON
manifest. See [OUTPUTS.md](OUTPUTS.md).

### What you choose

- **Channel set.** `ndvi + gvi` (default) merges the street-view components into
  one green-view channel. `ndvi + veg + terrain` keeps vegetation and terrain
  apart. The cache stores whichever a job needs and extends rather than rebuilds
  if you switch later.
- **Objective metric.** What the search maximises, and what decides which
  configuration wins. Partial distance correlation is the default: it detects
  nonlinear as well as linear association and conditions on covariates
  nonlinearly. Also available: distance correlation, Spearman, R-squared,
  normalized RMSE, mutual information, quartile contrast, logistic and
  GEE-logistic terms for binary outcomes, and four mixed-effects terms for
  longitudinal studies.
- **Quartile contrast** deserves a note. It cuts the composite at its own
  quartiles, enters them as indicators against the lowest quarter, and scores
  the magnitude of the top-versus-bottom difference adjusted for covariates.
  This is the shape the greenspace literature reports a greenness gradient in
  (Villeneuve et al. 2022; Irvin et al. 2024). Because it is a magnitude, a
  protective and a harmful gradient of the same size score alike, and the
  search maximises separation either way.
- **Covariates.** Numeric or categorical columns to control for. Categorical
  ones are one-hot encoded automatically.
- **Test-set size** and, optionally, spatial block validation.

### What the toolbox discovers from the data

Nothing below is pre-specified.

- **Spatial scale.** Each channel's buffer radius, chosen from the ladder you
  set.
- **Aggregation statistic.** Mean, median or a percentile of the values inside
  that buffer.
- **Functional form.** A weighted sum, or a synergy form with powers on the main
  terms and pairwise products, following Wang et al. 2026
  ([doi:10.3390/rs18010009](https://doi.org/10.3390/rs18010009)).
- **Channel weights**, reported with credible intervals.

### Why the result is defensible

- **Exhaustive held-out sweep.** Every radius-by-aggregator combination is
  enumerated and scored on repeated held-out splits of the training pool, not
  sampled. A one-standard-error rule prefers the smaller radius among
  statistically indistinguishable configurations. A Bayesian fit at the winning
  columns then puts credible intervals on the weights and the effect.
- **A held-out test set is the headline.** A fraction of the data, 25 % by
  default, is never touched during tuning. The winning configuration is scored
  on it once, with a percentile bootstrap interval and a permutation p-value
  (Freedman-Lane when covariates are controlled). Whole-data figures are shown
  only as descriptive and carry no p-value.
- **Reproducibility is measured, not assumed.** The whole discovery is repeated
  on independent reshuffles and the report says how often it landed on the same
  configuration. A low share means the surface is flat, not that the pick is
  wrong.
- **Fusion has to earn itself.** The composite is compared against each channel
  alone under the identical sweep, with a permutation null on the gain.
- **Calibration is checked.** The procedure is re-run on permuted outcomes; the
  reported false-positive rate should sit near 5 %, and the panel says so when
  it does not.
- **Provenance is recorded.** Every run stores a hash of its configuration and
  counts how many distinct configurations were scored on the held-out set, so a
  spent test set is visible rather than inferred.

### Controlling for confounding

- Covariate-adjusted objectives, so a dominant covariate cannot crowd out the
  greenery signal. Mutual information ignores covariates by design.
- Optional spatial-confounding adjustment (KS-AIC or Spatial+) and spatial block
  cross-validation, so geographic autocorrelation cannot inflate scores.
- Optional collinearity check that drops redundant channels before a run.
- Longitudinal runs add wave fixed effects, on by default, and an optional
  neighbourhood or site column.

### Reporting

- Effects per interquartile-range increase, with odds ratios for binary
  outcomes; quantile gradients against the lowest group with a test for trend;
  and a spline test for departure from linearity.
- Effect modification by any column you nominate, with simple slopes per group.
- For longitudinal studies, greenspace-by-time slopes, decomposed into
  between-person and within-person components on request.

---

## 4. Running it

- **Dashboard.** A Streamlit app with a tab per engine. Jobs run in their own
  process, so the interface stays responsive and a running job is not throttled
  by the browser tab.
- **CLI.** An MPI-parallel command-line entry point for HPC batch runs.
- **Sized to the machine.** Thread pools take about a third of the cores;
  process pools take 80 % by default. Override with `GEOFUSE_WORKERS` and
  `GEOFUSE_CPU_SHARE`. Every run logs a stage-by-stage wall-clock breakdown.
- **Everything on disk, per run.** Each run writes to its own timestamped folder
  so reruns never overwrite earlier results.
