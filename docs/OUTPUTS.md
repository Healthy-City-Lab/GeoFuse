# GeoFuse — Output Reference

All outputs are saved to `output_results/` by default.

---

## GVI Engine Outputs

For each input file processed through the GVI pipeline:

### `[Filename]_gvi.geojson`

Vector point data (EPSG:4326) with:

| Field | Description |
|-------|-------------|
| `gvi_veg` | Green View Index — Vegetation (%) |
| `gvi_ter` | Green View Index — Terrain (%) |
| `pano_id` | Google Street View panorama identifier |
| `lat`, `lon` | Geographic coordinates |
| `row`, `col` | Grid position (if raster-aligned) |

### `[Filename]_gvi.tif`

Multi-band GeoTIFF (EPSG:4326):

* **Band 1**: Vegetation GVI heatmap
* **Band 2**: Terrain GVI heatmap

### Optional Debug Outputs

| Path | Contents |
|------|----------|
| `output_results/images/{pano_id}.jpg` | Original Street View panoramas |
| `output_results/masks/{pano_id}.png` | Semantic segmentation masks (Cityscapes palette) |

---

## NDVI Engine Outputs

For each input file / date range processed through the NDVI pipeline:

### `[Filename]_ndvi.geojson`

Vector point data with:

| Field | Description |
|-------|-------------|
| `NDVI` | Normalized Difference Vegetation Index (−1 to 1) |
| `x`, `y` | Geographic coordinates (EPSG:4326) |
| `ndvi_date` | Source date (attribute-column mode only) |

### `[Filename]_ndvi.tif`

Single-band GeoTIFF (EPSG:4326):

* **Band 1**: NDVI values sampled from satellite imagery
* NoData value: `−9999`

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
