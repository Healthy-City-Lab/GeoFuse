# GeoFuse — Output Reference

All outputs are written under `output_results/` by default.

## Coordinate reference system

Every raster and vector output — GeoTIFFs, GeoPackages, per-cluster tiles, and the fusion composite — ships in the **engine-selected planar CRS** (UTM, Lambert Conformal Conic, or Polar Stereographic, chosen from the input geometry). Cells stay square in metres, index values never pass through a resampler, and downstream tools read the embedded CRS automatically. The exact CRS for a run is recorded in its `_gvi.json` / `_ndvi.json` sidecar.

> [!NOTE]
> **GeoJSON is the one exception.** `*_gvi.geojson`, `*_ndvi.geojson`, and `*_temporal_ndvi.geojson` are reprojected to **EPSG:4326** (lon/lat) at write time, because the format has no reliable CRS metadata. Use GeoPackage for analysis in the planar CRS.

---

## GVI Engine Outputs

You choose which formats are written via checkboxes in the GVI tab. Defaults: **GeoPackage on**, GeoTIFF off, GeoJSON off.

### `[name]_gvi.gpkg` — canonical (default)

GeoPackage (layer `gvi_samples`) of sampled points in the planar CRS:

| Field | Description |
| --- | --- |
| `gvi_veg` | Green View Index — vegetation (%) |
| `gvi_ter` | Green View Index — terrain (%) |
| `pano_id` | Source Street View panorama ID |
| `lat`, `lon` | Geographic coordinates of the panorama |
| `row`, `col` | Position on the shared anchored grid |
| `cluster_id` | Spatial cluster the point belongs to |

Recommended for large or scattered study areas: sparse on disk and free of the empty cells that dominate a single bbox-wide raster.

### `[name]_gvi.json` — sidecar

Records the grid CRS (WKT), step size, grid anchor, cluster count, and measured distortion — enough to rasterize the GeoPackage on the same grid later.

### `[name]_gvi_tiles/` — optional (GeoTIFF on)

One dense GeoTIFF per spatial cluster, plus a `tiles_index.json` locating each tile. Band 1 = vegetation GVI, Band 2 = terrain GVI. All-NaN areas cost zero bytes on disk.

### `[name]_gvi.geojson` — optional (GeoJSON on)

The same points reprojected to EPSG:4326 for web-map tools.

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
| `ndvi_date` | Source date (attribute-column mode only) |

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

---

## Fusion Engine Outputs

Every run writes to its own timestamped folder, so reruns never overwrite earlier results:

```text
output_results/fusion/<YYYYMMDDTHHMMSS>__<short_job_id>/
├── composite_greenery.tif              ← winning-params composite raster (CGI)
├── composite_greenery_params.json      ← winning params + stability provenance
├── composite_greenery_<veg|terrain|ndvi>.tif   ← one per standalone (if enabled)
├── results_bundle.json                 ← everything the results panel needs to rehydrate
└── study_results/
    ├── run_config.json                 ← every setting the run used (reproducibility)
    ├── results_summary.json            ← machine-readable manifest across all studies
    ├── test_scores.csv                 ← held-out test score + CI per study
    ├── scores.csv                      ← every subset score (train / val / test / all)
    ├── parameters.csv                  ← winning params per study (long form)
    ├── stability_cells.csv             ← ranked candidate cells per study
    ├── stability_bootstraps.csv        ← per-bootstrap leaderboard per study
    ├── covariate_impact.csv            ← per-covariate effects (if covariates set)
    ├── aic_bic.json                    ← CGI-vs-standalone verdict (if standalones ran)
    ├── collinearity.json               ← VIF report (if requested)
    ├── mixedlm_metrics*.csv            ← four MixedLM metrics (longitudinal mode)
    └── standalone_<ch>/                ← per-standalone artifacts
```

- Each result is recorded both as the machine-readable `results_summary.json` manifest **and** as the tidy CSVs above, so it is equally easy to replay or to open in a spreadsheet.
- Multi-outcome runs suffix each basename with `__<outcome>`.
- `run_config.json` captures the full configuration, which is replayed verbatim on restart.

### Reusable caches (`output_results/fusion_cache/`)

Reused automatically across runs with identical inputs:

| Path | Contents |
| --- | --- |
| `cache-veg-*.geojson`, `cache-terrain-*.geojson` | Aligned GVI channels |
| `cache-ndvi-*.tif`, `cache-gvi_combined-*.tif` | Aligned NDVI raster and the combined GVI raster |
| `preaggr/preaggr-*.sqlite` | Pre-aggregation cache: per-(entity, radius) statistics for each channel. Resumable; safe to delete to force a rebuild. |
