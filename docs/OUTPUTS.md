# GeoFuse output reference

Everything is written under `output_results/` by default.

**Abbreviations.** GVI = Green View Index. NDVI = Normalized Difference
Vegetation Index. CGI = Composite Greenery Index. CRS = coordinate reference
system. CI = confidence interval. CrI = credible interval. IQR = interquartile
range. VIF = variance inflation factor. ESS = effective sample size.

## Coordinate reference system

Every raster and vector output ships in the engine-selected planar CRS (UTM,
Lambert Conformal Conic or Polar Stereographic, chosen from the input geometry),
so cells stay square in metres and values never pass through a resampler. The
exact CRS is recorded in each run's sidecar JSON.

> [!NOTE]
> GeoJSON is the exception. Those files are reprojected to EPSG:4326 at write
> time because the format has no reliable CRS metadata. Use GeoPackage for
> analysis in the planar CRS.

---

## GVI outputs

Formats are chosen by checkbox. Default: GeoPackage on, GeoTIFF off, GeoJSON
off.

| Path | Contents |
| --- | --- |
| `[name]_gvi.gpkg` | Canonical point layer (`gvi_samples`) in the planar CRS |
| `[name]_gvi.json` | Sidecar: grid CRS, step, anchor, cluster count, distortion |
| `[name]_gvi_tiles/` | One dense GeoTIFF per cluster, band 1 vegetation, band 2 terrain, plus an index |
| `[name]_gvi.geojson` | The same points in EPSG:4326 for web maps |
| `[name]_temporal_gvi/` | Per-year mode: one full output set per year |
| `output_results/images/`, `masks/` | Raw panoramas and segmentation masks, if enabled |

Fields in `gvi_samples`:

| Field | Meaning |
| --- | --- |
| `gvi_veg` | Vegetation share of panorama pixels (Cityscapes class 8) |
| `gvi_ter` | Terrain share of panorama pixels (Cityscapes class 9) |
| `pano_id`, `pano_date` | Source panorama and its capture date (`YYYY-MM`) |
| `lat`, `lon` | Panorama coordinates |
| `row`, `col` | Position on the shared anchored grid |
| `cluster_id` | Spatial cluster |

`gvi_veg` and `gvi_ter` are independent, non-overlapping classes. Add them for a
combined index.

---

## NDVI outputs

Default: GeoTIFF on, GeoPackage off, GeoJSON off.

| Path | Contents |
| --- | --- |
| `[name]_ndvi.tif` | Single band, float32, median NDVI over the range after cloud masking. NoData `-9999` |
| `[name]_ndvi.json` | Sidecar, always written on success (see below) |
| `[name]_ndvi.gpkg` / `.geojson` | Per-pixel points, planar CRS and EPSG:4326 respectively |
| `[name]_ndvi_tiles/` | One GeoTIFF per connected component, plus an index |
| `[name]_temporal_ndvi/` | Per-year mode: one full output set per year |

The sidecar lets fusion and other tools introspect a raster without re-running
Earth Engine. It records the export CRS, the satellite and collection used, the
dates requested and the dates actually queried (which differ if a sparse window
was widened), the cloud threshold and surviving image count, resolution, how the
input decomposed into clusters and tiles, any tiles that exhausted their retries
and so appear as gaps, and the maximum map distortion.

---

## Fusion outputs

Each run writes to its own timestamped folder, so reruns never overwrite earlier
results.

```text
output_results/fusion/<YYYYMMDDTHHMMSS>__<short_job_id>/
├── composite_greenery.tif              composite raster at the discovered params
├── composite_greenery_params.json      discovered params + provenance
├── composite_greenery_<channel>.tif    one per standalone channel, if enabled
├── results_bundle.json                 everything the results panel needs
└── study_results/
    ├── run_config.json                 every setting the run used
    ├── results_summary.json            machine-readable manifest
    ├── test_scores.csv                 held-out score + CI per study
    ├── scores.csv                      every subset score
    ├── parameters.csv                  discovered params per study
    ├── discovery.csv                   the pick, weights + CrIs, gain per study
    ├── covariate_impact.csv            per-covariate effects
    ├── exposure_response.csv           per-IQR effect, quantile gradient, linearity test
    ├── exposure_response_curve.csv     fitted curve, ready to plot
    ├── moderation.csv                  interaction test and simple slopes
    ├── decline_terms.csv               greenspace-by-time terms, longitudinal only
    ├── mixedlm_metrics*.csv            all four mixed-effects metrics, longitudinal only
    ├── aic_bic.json                    CGI vs standalone verdict
    ├── collinearity.json               VIF report, if requested
    └── standalone_<channel>/           per-standalone artifacts
```

Multi-outcome runs suffix each basename with `__<outcome>`. `run_config.json`
seeds the setup form when a job is re-run.

### Reading the results

**Only the held-out `test` split carries a p-value.** The other slices are left
empty on purpose. The parameters were chosen by maximising the objective on
those same rows, so an in-sample p-value would look significant because it was
tuned to, not because the effect generalises. Read `test` for the honest result
and `all` only as a descriptive summary.

**A failed fit is reported empty, never as `0.0`.** A mixed model that did not
converge records `test_ci.status = "fit_failed"` with a null score, so a genuine
null is never confused with a model that never fit.

**`discovery.csv`** has one row per weight: the picked radius and aggregator per
channel, the posterior weight with its credible interval, whether that radius
sat at the edge of the searched ladder, and the channel's standalone score
against the composite's gain and permutation p-value. Synergy pair terms appear
as their own rows with radius and aggregator blank, since a pair has no spatial
scale of its own.

**`results_summary.json` records a `config_hash` and a `test_reads` count.** The
hash identifies the configuration that produced the result, so a pre-registered
analysis can be shown to be the one that ran. `test_reads` counts distinct
configurations scored on the held-out slice; a large number means the headline
p-value is optimistic by roughly that many comparisons.

**`exposure_response.csv`** restates the effect in the shapes the greenspace
literature publishes: per-IQR increase (with an odds ratio for binary outcomes),
the gradient across exposure quantiles against the lowest group with a test for
trend, and a Wald test for departure from linearity. A `modelled_event_level`
row records which value of a binary outcome was treated as the event, which
matters because a column coded `1=Yes, 2=No` inverts every odds ratio.

**`moderation.csv`** answers whether the effect differs between groups: one
simple slope per moderator level, one row per interaction term, and a single
joint p-value. Nominating a moderator does not adjust for it; add it as a
covariate too if you want both.

**`decline_terms.csv`** holds greenspace-by-time slopes, optionally decomposed
into between-person and within-person components. A term that is not estimable
is `NaN`, never a fitted value.

**Mixed-effects notes.** These objectives report a Wald p-value from the fit
itself rather than a permutation p-value; the cluster bootstrap covers the
resampling question. `mixedlm_tstat` is an absolute value, so its interval sits
above zero by construction and is not a significance statement; the signed
coefficient is reported alongside and is the one that can straddle zero.

### Caches

`output_results/fusion_cache/` holds aligned channel rasters and the
pre-aggregation store of per-entity, per-radius statistics. Both are reused
automatically across runs with identical inputs and are safe to delete to force
a rebuild.
