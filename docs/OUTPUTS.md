# GeoFuse — Output Reference

All outputs are saved to `output_results/` by default.

---

## GVI Engine Outputs

For each input file processed through the GVI pipeline. You choose which of the three formats are written via checkboxes in the GVI tab (defaults: **GeoPackage on**, GeoTIFF off, GeoJSON off).

### `[Filename]_gvi.gpkg` (canonical, default)

GeoPackage with layer `gvi_samples` — EPSG:4326 point geometries:

| Field | Description |
|-------|-------------|
| `gvi_veg` | Green View Index — Vegetation (%) |
| `gvi_ter` | Green View Index — Terrain (%) |
| `pano_id` | Google Street View panorama identifier |
| `lat`, `lon` | Geographic coordinates |
| `row`, `col` | Position on the shared anchored grid (cells across clusters are on one global grid) |
| `cluster_id` | Spatial cluster the point belongs to (0-indexed) |

This is the recommended format for large or geographically scattered study areas. GeoPackage is sparse on disk, opens in QGIS, GeoPandas, and ogr2ogr, and avoids the mostly-empty cells that would dominate a single bbox-wide raster.

### `[Filename]_gvi.json` (sidecar)

Plain-JSON sidecar written next to the GeoPackage. Records `grid_crs_wkt`, `step_m`, `anchor_x`, `anchor_y`, `n_clusters`, and the measured planar distortion. Used by the result inspector and useful if you later want to rasterise the GeoPackage on the same grid as the source run.

### `[Filename]_gvi_tiles/` (optional, GeoTIFF on)

One dense GeoTIFF per spatial cluster — instead of one huge mostly-empty raster:

```text
[Filename]_gvi_tiles/
  cluster_0000.tif   # 2-band: Band 1 = Vegetation GVI, Band 2 = Terrain GVI
  cluster_0001.tif
  ...
  tiles_index.json   # bbox in grid CRS and WGS84 + grid CRS WKT for each tile
```

Tiles are written in the auto-selected projected CRS (UTM / LCC / Polar Stereographic) — not WGS84 — so cells stay square in metres and no resampling is involved. Open `tiles_index.json` to reproject or merge on demand.

### `[Filename]_gvi.geojson` (optional, GeoJSON on)

Same point data as the GeoPackage. Kept for compatibility with tools that don't read GeoPackage. A warning is logged if the file exceeds 100,000 points (GeoJSON read performance degrades quickly past that).

### Optional Debug Outputs

| Path | Contents |
|------|----------|
| `output_results/images/{pano_id}.jpg` | Original Street View panoramas |
| `output_results/masks/{pano_id}.png` | Semantic segmentation masks (Cityscapes palette) |

### Per-job log files

| Path | Contents |
|------|----------|
| `logs/jobs/<job_id>.log` | Full plain-text log of one job (no ANSI escapes); kept indefinitely so you can inspect long runs after the in-memory deque has rolled over |

---

## NDVI Engine Outputs

For each input file / date range processed through the NDVI pipeline. Defaults: **GeoTIFF on**, GeoPackage off, GeoJSON off.

### `[Filename]_ndvi.tif` (default)

Single-band GeoTIFF in the engine-selected **planar CRS** (UTM / LCC /
Polar Stereographic — same family used by the GVI tiles, chosen by
`select_grid_crs` to keep pixels square in metres):

* **Band 1 (`NDVI`, float32)**: median NDVI over the date range, after cloud masking.
* NoData value: `−9999`.

The raster has no WGS84 reprojection step — it ships in the same CRS the Earth
Engine tiles were exported in, so every pixel is a true 10 m × 10 m square on
the ground. The exact CRS for the run is recorded in the `_ndvi.json` sidecar
(`export_crs`, `export_crs_wkt`) and in each output's own embedded CRS tag.

### `[Filename]_ndvi.gpkg` (optional)

GeoPackage with layer `ndvi_samples` in the raster's planar CRS — same
per-point data as the GeoJSON form, but loads orders of magnitude faster for
large extents. Written via streaming block iteration so peak memory usage is
kept low.

### `[Filename]_ndvi.geojson` (optional)

Vector point data with:

| Field | Description |
|-------|-------------|
| `NDVI` | Normalized Difference Vegetation Index (−1 to 1) |
| `x`, `y` | Coordinates in the raster's planar CRS, metres |
| `ndvi_date` | Source date (attribute-column mode only) |

### `[Filename]_ndvi_at_features.gpkg` (optional)

GeoPackage with layer `ndvi_at_features`, written when **Sample at uploaded features** is enabled. Each row preserves the original feature's attributes, plus:

| Field | Description |
|-------|-------------|
| `NDVI` | Aggregated NDVI under the feature — exact-pixel read for unbuffered points, otherwise the chosen stat (mean / median / min / max / std / count) over the buffer or polygon |
| `ndvi_obs` | Count of valid raster pixels that contributed to the aggregate. Zero ⇒ outside raster footprint or all-nodata |

Pairs cleanly with GVI sample points for fusion downstream; Both formats deliver per-feature greenery values keyed on the original uploaded geometry.

### `[Filename]_ndvi_tiles/` (optional)

Per-cluster GeoTIFF tile directory, written when the **Per-cluster tiles**
output is enabled. Avoids the giant mostly-NaN single mosaic that scattered
national-scale inputs would otherwise produce. Contents:

* `cluster_NNNN.tif` — one planar-CRS GeoTIFF per connected component of the
  buffered input geometry, sized to the cluster's bbox. Same CRS as the main
  `_ndvi.tif`.
* `tiles_index.json` — `{crs, export_crs, export_crs_name, resume_key,
  n_clusters, clusters: [{cluster_id, path, bounds, bounds_4326, n_tiles}, …]}`.
  Each entry carries the cluster's native-CRS extent in `bounds` and a
  derived geographic locator in `bounds_4326`, so downstream tools can pick
  the right cluster file without reopening every raster.

### `[Filename]_ndvi.json` (always written on success)

Sidecar metadata mirroring GVI's `_gvi.json`. Lets fusion and custom tooling
introspect a raster after the fact without re-running Earth Engine:

| Field | Description |
|-------|-------------|
| `export_crs` / `export_crs_name` / `export_crs_wkt` | Planar CRS pixels were rasterised in (EPSG ID, human-readable name, full WKT) |
| `distortion` | Max relative deviation of a 1000 m planar step from true geodesic distance across the extent |
| `n_clusters` / `tiles_total` / `tiles_succeeded` / `tiles_failed` / `tiles_resumed` | How the input decomposed into connected components and tiles, how many landed on disk vs. exhausted retry, and how many were reused from a previous interrupted run |
| `resume_key` | Short hash over geometry + date range + cloud max + resolution + collection. Identifies the cache entry under `logs/caches/ndvi_tiles/<resume_key>/`. Lets any future run with the same key reuse already-downloaded tiles instantly (whether the previous run was interrupted or finished cleanly). LRU-evicted by the cache once total size exceeds the cap |
| `failed_tile_refs` | List of `{cluster_id, tile_idx, error}` for tiles that hit the retry ceiling — these areas appear as NaN gaps in the mosaic, this lets you audit which |
| `start_date` / `end_date` | Date range the *user* asked for (unmodified) |
| `used_start_date` / `used_end_date` / `coverage_widened` | Date range actually queried (may be wider than the user's request if coverage rescue fired). `coverage_widened: true` flags that the composite spans a bigger window than requested |
| `cloud_max` / `n_cloud_filtered_images` | Cloud-percentage threshold + how many images survived it |
| `resolution_m` / `max_tile_size_km` | Export resolution and tiling cap |
| `satellite` / `ee_collection` / `bands` | `'sentinel2'` or `'landsat'` (auto-picked by date range), the ImageCollection ID actually queried, and the list of bands in the GeoTIFF (`["NDVI"]`) |

---

## Fusion Engine Outputs

### Composite Greenery Map

| File | Description |
|------|-------------|
| `composite_greenery.tif` | Single-band GeoTIFF with the optimized composite greenery index |
| `composite_greenery_params.json` | Final averaged parameters (weights, radii, aggregation functions) |

### Optimization Study (`output_results/fusion/study_results/`)

Two sub-directories are produced: `robust_trials/` (statistically significant only) and `all_trials/` (all trials, for debugging).

Each contains:

| File | Description |
|------|-------------|
| `optimization_report.txt` | Summary with performance metrics |
| `best_params.json` | Best parameter set |
| `optimization_history.html` | Trial value progression |
| `param_importances.html` | Feature importance analysis |
| `parallel_coordinates_all.html` | Multi-dimensional parameter visualization |
| `parallel_coordinates_weights.html` | Weight-focused coordinates plot |
| `contour_weights.html` | 2D parameter interaction heatmaps |
| `slice_plot_weights.html` | 1D parameter impact plots |
| `edf.html` | Empirical Distribution Function |
| `rank.html` | Trial ranking |
| `timeline.html` | Trial execution timeline |

### Metric Cache (`output_results/fusion_cache/`)

Automatically reused for identical study areas to avoid redundant downloads:

| File | Contents |
|------|----------|
| `cache-veg-{hash}.geojson` | Cached GVI vegetation component |
| `cache-terrain-{hash}.geojson` | Cached GVI terrain component |
| `cache-ndvi-{hash}.tif` | Cached NDVI raster |
| `cache-gvi_combined-{hash}.tif` | Multi-band raster (Band 1: Vegetation, Band 2: Terrain) |
