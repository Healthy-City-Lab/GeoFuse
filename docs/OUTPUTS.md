# GeoFuse — Output Reference

All outputs are written under `output_results/` by default.

## Coordinate reference system

Every raster and vector output — GeoTIFFs, GeoPackages, per-cluster tiles, and the fusion composite — ships in the **engine-selected planar CRS** (UTM, Lambert Conformal Conic, or Polar Stereographic, chosen from the input geometry). Cells stay square in metres, index values never pass through a resampler, and downstream tools read the embedded CRS automatically. The exact CRS for a run is recorded in its `_gvi.json` / `_ndvi.json` sidecar.

> [!NOTE]
> **GeoJSON is the one exception.** `*_gvi.geojson` and `*_ndvi.geojson` (including the per-year files inside `*_temporal_gvi/` and `*_temporal_ndvi/` folders) are reprojected to **EPSG:4326** (lon/lat) at write time, because the format has no reliable CRS metadata. Use GeoPackage for analysis in the planar CRS.

---

## GVI Engine Outputs

You choose which formats are written via checkboxes in the GVI tab. Defaults: **GeoPackage on**, GeoTIFF off, GeoJSON off.

### `[name]_gvi.gpkg` — canonical (default)

GeoPackage (layer `gvi_samples`) of sampled points in the planar CRS:

| Field | Description |
| --- | --- |
| `gvi_veg` | Green View Index — **vegetation only** (Cityscapes class 8), as a fraction of panorama pixels |
| `gvi_ter` | Green View Index — **terrain only** (Cityscapes class 9), as a fraction of panorama pixels |
| `pano_id` | Source Street View panorama ID |
| `pano_date` | Capture date of the panorama used (`YYYY-MM`) |
| `lat`, `lon` | Geographic coordinates of the panorama |
| `row`, `col` | Position on the shared anchored grid |
| `cluster_id` | Spatial cluster the point belongs to |

`gvi_veg` and `gvi_ter` are **independent, non-overlapping** classes — add them if you want a combined vegetation+terrain index.

Recommended for large or scattered study areas: sparse on disk and free of the empty cells that dominate a single bbox-wide raster.

### `[name]_gvi.json` — sidecar

Records the grid CRS (WKT), step size, grid anchor, cluster count, and measured distortion — enough to rasterize the GeoPackage on the same grid later.

### `[name]_gvi_tiles/` — optional (GeoTIFF on)

One dense GeoTIFF per spatial cluster, plus a `tiles_index.json` locating each tile. Band 1 = vegetation GVI, Band 2 = terrain GVI. All-NaN areas cost zero bytes on disk.

### `[name]_gvi.geojson` — optional (GeoJSON on)

The same points reprojected to EPSG:4326 for web-map tools.

### `[name]_temporal_gvi/` — per-year mode (year/date column)

When you run GVI with a year/date column, the layer is split by year and each year is processed as its own GVI job into this folder: `{name}_{year}_gvi.gpkg` (+ optional `.geojson`, `_gvi_tiles/`, and `_gvi.json` sidecar) per year. Every year is measured at the Street View capture nearest that year and shares one planar CRS chosen from the whole layer, so the years align pixel-for-pixel. The capture date used sits in each file's `pano_date` column.

### Optional debug outputs

| Path | Contents |
| --- | --- |
| `output_results/images/{pano_id}.jpg` | Original Street View panoramas |
| `output_results/masks/{pano_id}.png` | Segmentation masks (Cityscapes palette) |
| `logs/jobs/<job_id>.log` | Full per-job text log, kept indefinitely |

---

## NDVI Engine Outputs

Defaults: **GeoTIFF on**, GeoPackage off, GeoJSON off.

### `[name]_ndvi.tif` — default

Single-band GeoTIFF in the planar CRS:

- **Band 1 (`NDVI`, float32)** — median NDVI over the date range, after cloud masking.
- NoData: `−9999`.

### `[name]_ndvi.gpkg` / `[name]_ndvi.geojson` — optional

Per-pixel point data. The GeoPackage is in the planar CRS and loads far faster at scale; the GeoJSON is in EPSG:4326 for compatibility.

| Field | Description |
| --- | --- |
| `NDVI` | Normalized Difference Vegetation Index (−1 to 1) |

### `[name]_ndvi_tiles/` — optional (Per-cluster tiles)

One planar-CRS GeoTIFF per connected component plus a `tiles_index.json`. Avoids the giant mostly-NaN mosaic that scattered national-scale inputs would otherwise produce.

### `[name]_ndvi.json` — sidecar (always written on success)

Lets fusion and custom tools introspect a raster without re-running Earth Engine. Key fields:

| Field | Description |
| --- | --- |
| `export_crs` / `export_crs_name` / `export_crs_wkt` | Planar CRS the pixels were rasterized in |
| `satellite` / `ee_collection` / `bands` | Source (`sentinel2` / `landsat`), the collection queried, and band names |
| `start_date` / `end_date` | Date range you requested |
| `used_start_date` / `used_end_date` / `coverage_widened` | Range actually queried (wider if coverage rescue fired) |
| `cloud_max` / `n_cloud_filtered_images` | Cloud threshold and how many images survived it |
| `resolution_m` | Export resolution |
| `n_clusters` / `tiles_total` / `tiles_succeeded` / `tiles_failed` / `tiles_resumed` | How the input decomposed and how tiling went |
| `failed_tile_refs` | Tiles that exhausted retries (appear as NaN gaps), for auditing |
| `distortion` | Max relative map distortion across the extent |

### `[name]_temporal_ndvi/` — per-year mode (year/date column)

When you run NDVI with a year/date column, the layer is split by year and each year is processed as its own NDVI job into this folder: `{name}_{year}_ndvi.tif` (+ optional GeoPackage/GeoJSON/tiles and `_ndvi.json` sidecar) per year. Each year is composited over the same growing-season months you choose (e.g. June–September) of that year, and every year shares one planar CRS chosen from the whole layer, so the years align pixel-for-pixel.

---

## Fusion Engine Outputs

Every run writes to its own timestamped folder, so reruns never overwrite earlier results:

```text
output_results/fusion/<YYYYMMDDTHHMMSS>__<short_job_id>/
├── composite_greenery.tif              ← winning-params composite raster (CGI)
├── composite_greenery_params.json      ← winning params + discovery provenance
├── composite_greenery_<ndvi|gvi|veg|terrain>.tif  ← one per standalone (if enabled)
├── results_bundle.json                 ← everything the results panel needs to rehydrate
└── study_results/
    ├── run_config.json                 ← every setting the run used (reproducibility)
    ├── results_summary.json            ← machine-readable manifest across all studies
    ├── test_scores.csv                 ← held-out test score + CI per study
    ├── scores.csv                      ← every subset score (train / val / test / all)
    ├── parameters.csv                  ← winning params per study (long form)
    ├── discovery.csv                   ← the pick, weights + CrIs, gain per study
    ├── covariate_impact.csv            ← per-covariate effects (if covariates set)
    ├── decline_terms.csv               ← greenspace × time terms (longitudinal mode)
    ├── exposure_response.csv           ← per-IQR effect, quantile gradient, non-linearity test
    ├── exposure_response_curve.csv     ← fitted spline curve, ready to plot
    ├── moderation.csv                  ← effect modification: interaction test + simple slopes
    ├── aic_bic.json                    ← CGI-vs-standalone verdict (if standalones ran)
    ├── collinearity.json               ← VIF report (if requested)
    ├── mixedlm_metrics*.csv            ← four MixedLM metrics (longitudinal mode)
    └── standalone_<ch>/                ← per-standalone artifacts
```

- Each result is recorded both as the machine-readable `results_summary.json` manifest **and** as the tidy CSVs above, so it is equally easy to replay or to open in a spreadsheet.
- Multi-outcome runs suffix each basename with `__<outcome>`.
- `run_config.json` captures the full configuration, which seeds the setup form when the job is re-run.

### How to read the scores

- **Only the held-out `test` split carries a p-value.** The `train`, `val`, and `all` (whole-data) p-values are deliberately left empty (`None`). The winning parameters were chosen by maximising the objective on those same slices, so an in-sample Wald p-value would be a selective-inference artefact — it would look significant because the params were tuned to make it so, not because the effect generalizes. The `val` figure is a bootstrap median with no single fit behind it, so it has no p-value either. Read the `test` split for the honest, generalizable result; read `all` only as an optimistic, descriptive summary.
- **A failed fit is reported as empty, never as `0.0`.** When the held-out mixed model does not converge, `results_summary.json` records `test_ci.status = "fit_failed"` (with no confidence bounds and a null `test_score`), and the UI shows a "fit failed" badge. The same holds per subset: a slice whose mixed model did not fit has an empty `score` in `scores.csv` and `subset_scores`, and no bar in the cross-study comparison chart, rather than a zero that reads as a measured null. This distinguishes a genuine null effect from a model that never fit — a `0.0` score paired with a confidence interval that excludes it would otherwise be ambiguous.
- **`discovery.csv`** has one row per weight: the picked radius and aggregator for each channel, the posterior weight with its 95 % credible interval, whether that radius sat at the edge of the searched ladder, and — where the comparison ran — the channel's standalone held-out score next to the composite's gain and its permutation p. The synergy form's pair terms appear as their own rows with the radius and aggregator columns blank, because a pair has no spatial scale of its own. `results_summary.json` carries the same numbers plus the sampler's R-hat, minimum ESS and divergence count, the reproducibility block, and the null-calibration rate.
- **`results_summary.json` records a `config_hash` and a `test_reads` count.** The hash identifies the exact configuration that produced the result, so a pre-registered analysis can be shown to be the one that ran. `test_reads` counts how many distinct configurations were scored on the held-out slice — each one spends part of the out-of-sample guarantee, and a large number means the headline p-value is optimistic by roughly that many comparisons.
- **`mixedlm_metrics*.csv`** carries one row for the final composite, scored on the held-out test set with all four `mixedlm_*` metrics regardless of which one the run optimized. `mixedlm_marginal_r2` is greenery's share of the Nakagawa marginal R², measured as the drop when the term leaves the design; `mixedlm_lr` compares the two fits under ML, because REML likelihoods are not comparable across different fixed-effects designs.
- **Mixed-effects objectives report a Wald p, not a permutation p.** The greenery fixed effect's Wald test comes from the fit itself; a permutation analogue would have to be resampled and refit per entity, which the cluster bootstrap already covers. The report labels which one it is rather than assuming.
- **`mixedlm_tstat` intervals are folded.** The metric is `|t|`, so its confidence interval sits above zero by construction and is not a significance statement. The signed greenery coefficient and its interval are reported alongside — that is the one that can straddle zero.
- **`exposure_response.csv`** restates the winning composite's effect in the shapes the greenspace literature publishes: the effect per interquartile-range increase (with an odds ratio when the outcome is binary), the gradient across exposure quantiles against the lowest group plus a test for trend, and a joint Wald test of departure from linearity from an orthogonalised spline. A `modelled_event_level` row records which of a binary outcome's two values was treated as the event — a survey column coded `1=Yes, 2=No` inverts every odds ratio in the table, and this is where that shows up. **`exposure_response_curve.csv`** is the fitted curve over the exposure range, centred at the median.
- **`moderation.csv`** answers "is the greenery effect different for different people". One `simple_slope` row per moderator level (the greenery effect *within* that group, with its own CI and — for a binary outcome — odds ratio), one `interaction` row per product term, and one `interaction_joint_test` row carrying the single p-value for "does the effect differ at all". Selecting a column as an effect modifier does **not** adjust for it; list it as a covariate too if you want both.
- **`decline_terms.csv`** holds the greenspace × time slopes: `overall`, plus the between-person (average exposure) and within-person (exposure change) decomposition when requested. A term that is not estimable — a within-person slope where the exposure never varies over time — is reported as `NaN`, never as a fitted value.

### Reusable caches (`output_results/fusion_cache/`)

Reused automatically across runs with identical inputs:

| Path | Contents |
| --- | --- |
| `cache-veg-*.geojson`, `cache-terrain-*.geojson` | Aligned GVI channels |
| `cache-ndvi-*.tif`, `cache-gvi_combined-*.tif` | Aligned NDVI raster and the combined GVI raster |
| `preaggr/preaggr-*.sqlite` | Pre-aggregation cache: per-(entity, radius) statistics for each channel. Resumable; safe to delete to force a rebuild. |
