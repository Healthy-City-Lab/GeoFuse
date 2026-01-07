import os
import time
import asyncio
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.transform import from_origin
from PIL import Image
from tqdm import tqdm
import warnings

# --- CONFIGURATION ---
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

# Cityscapes Palette
CITYSCAPES_PALETTE = [
    128,
    64,
    128,
    244,
    35,
    232,
    70,
    70,
    70,
    102,
    102,
    156,
    190,
    153,
    153,
    153,
    153,
    153,
    250,
    170,
    30,
    220,
    220,
    0,
    107,
    142,
    35,
    152,
    251,
    152,
    70,
    130,
    180,
    220,
    20,
    60,
    255,
    0,
    0,
    0,
    0,
    142,
    0,
    0,
    70,
    0,
    60,
    100,
    0,
    80,
    100,
    0,
    0,
    230,
    119,
    11,
    32,
] + [0, 0, 0] * 237

# Imports from streetview package
from streetview import search_panoramas, get_streetview, get_panorama_async

from .vision import DeepLabSegmenter


class GVIEngine:
    def __init__(
        self, model_path=None, download_mode="package", device="cuda", api_key=None
    ):
        self.download_mode = download_mode
        self.device = device
        self.api_key = api_key

        print(f"[GVI] Initializing DeepLabV3+ Model on {device}...")
        self.segmenter = DeepLabSegmenter(ckpt_path=model_path, device=device)
        print("[GVI] Model Ready.")

    def _extract_panoid(self, pano_obj):
        if hasattr(pano_obj, "pano_id"):
            return pano_obj.pano_id
        if hasattr(pano_obj, "panoid"):
            return pano_obj.panoid
        try:
            return pano_obj.get("pano_id") or pano_obj.get("panoid")
        except (TypeError, AttributeError):
            return None

    def _preprocess_image(self, img, target_width=1920):
        if img is None:
            return None

        # Ensure input is PIL
        if isinstance(img, np.ndarray):
            try:
                img = Image.fromarray(img)
            except:
                return None

        arr = np.array(img)

        # Energy check (threshold 5)
        row_energy = np.mean(arr, axis=(1, 2))
        col_energy = np.mean(arr, axis=(0, 2))

        valid_rows = np.where(row_energy > 5)[0]
        valid_cols = np.where(col_energy > 5)[0]

        if len(valid_rows) == 0 or len(valid_cols) == 0:
            return None

        y_min, y_max = valid_rows[0], valid_rows[-1] + 1
        x_min, x_max = valid_cols[0], valid_cols[-1] + 1

        img_cropped = img.crop((x_min, y_min, x_max, y_max))
        target_height = int(target_width / 2)
        return img_cropped.resize((target_width, target_height), Image.BILINEAR)

    def _download_async_wrapper(self, panoid):
        """
        Robust Async Wrapper: Creates a fresh loop for every download to prevent
        'Event loop is closed' errors in Streamlit/Threaded environments.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Zoom=1 prevents stitching errors and 134MP bombs
            return loop.run_until_complete(get_panorama_async(pano_id=panoid, zoom=1))
        except Exception:
            return None
        finally:
            loop.close()

    def _get_pano_img(self, lat, lon):
        try:
            # 1. Search (Synchronous)
            panos = search_panoramas(lat=lat, lon=lon)
            if not panos:
                return None

            # 2. Iterate ALL candidates
            for meta in panos:
                panoid = self._extract_panoid(meta)
                if not panoid:
                    continue

                raw_image = None
                try:
                    if self.api_key:
                        # Method A: Official API
                        raw_image = get_streetview(pano_id=panoid, api_key=self.api_key)
                    else:
                        # Method B: Async Scraper
                        raw_image = self._download_async_wrapper(panoid)

                    if raw_image is not None:
                        processed = self._preprocess_image(raw_image)
                        if processed:
                            return processed

                except Exception:
                    continue
            return None
        except Exception:
            return None

    def _generate_pixel_aligned_grid(self, gdf, resolution):
        """
        Internal fallback grid generation for Polygon inputs.
        """
        minx, miny, maxx, maxy = gdf.total_bounds

        width = int(np.ceil((maxx - minx) / resolution))
        height = int(np.ceil((maxy - miny) / resolution))
        transform = from_origin(minx, maxy, resolution, resolution)

        cols = np.arange(width)
        rows = np.arange(height)

        pixel_centers = []
        for r in rows:
            y_center = maxy - (r + 0.5) * resolution
            for c in cols:
                x_center = minx + (c + 0.5) * resolution

                p = Point(x_center, y_center)
                if gdf.contains(p).any():
                    pixel_centers.append(
                        {
                            "geometry": p,
                            "row": r,
                            "col": c,
                            "lat": y_center,
                            "lon": x_center,
                        }
                    )

        return pixel_centers, height, width, transform

    def run_analysis(
        self,
        gdf,
        step: float | int = 50,
        folder="output",
        save_panos=False,
        save_masks=False,
    ):
        print("[GVI] Starting Analysis (ASYNC MODE)...")
        os.makedirs(folder, exist_ok=True)
        if save_panos:
            os.makedirs(os.path.join(folder, "images"), exist_ok=True)
        if save_masks:
            os.makedirs(os.path.join(folder, "masks"), exist_ok=True)

        # ---------------------------------------------------------
        # 1. DETERMINE INPUT TYPE & PREPARE POINTS
        # ---------------------------------------------------------
        first_geom = gdf.geometry.iloc[0]
        points = []
        rows = 0
        cols = 0
        transform = None
        write_tif = False

        if first_geom.geom_type in ["Polygon", "MultiPolygon"]:
            print(
                f"[INFO] Input is Polygon. Generating internal grid (Step: {step})..."
            )
            # For Polygons, we generate new points, so we use a default index (0..N)
            # The _generate function returns a list of dicts directly
            points_data, rows, cols, transform = self._generate_pixel_aligned_grid(
                gdf, step
            )
            # Add a dummy original index since these are new points
            for i, p in enumerate(points_data):
                p["orig_index"] = i
            points = points_data
            write_tif = True
        else:
            print(f"[INFO] Input is Points ({len(gdf)}). Using provided geometry.")
            # Map input GDF to dicts, PRESERVING THE INDEX
            has_indices = "row" in gdf.columns and "col" in gdf.columns

            for idx, row in gdf.iterrows():
                points.append(
                    {
                        "orig_index": idx,  # <--- CRITICAL: Save original DataFrame index
                        "geometry": row.geometry,
                        "row": int(row["row"]) if has_indices else 0,
                        "col": int(row["col"]) if has_indices else idx,
                        "lat": row.geometry.y,
                        "lon": row.geometry.x,
                    }
                )

            if has_indices:
                rows = gdf["row"].max() + 1
                cols = gdf["col"].max() + 1

        if not points:
            print("[FAIL] No points to process.")
            return gpd.GeoDataFrame()

        # ---------------------------------------------------------
        # 2. RUN ANALYSIS LOOP
        # ---------------------------------------------------------
        if rows > 0 and cols > 0:
            raster_veg = np.full((rows, cols), np.nan, dtype=np.float32)
            raster_ter = np.full((rows, cols), np.nan, dtype=np.float32)
        else:
            raster_veg = None
            raster_ter = None

        results = []
        success_count = 0

        print(f"[GVI] Processing {len(points)} points...")

        for pt in tqdm(points):
            r, c = pt["row"], pt["col"]
            lat, lon = pt["lat"], pt["lon"]

            # Use preserved index
            orig_idx = pt["orig_index"]

            # Ensure we search using WGS84 coordinates
            search_lat, search_lon = lat, lon
            if not gdf.crs.is_geographic:
                p_geo = (
                    gpd.GeoSeries([pt["geometry"]], crs=gdf.crs)
                    .to_crs(epsg=4326)
                    .iloc[0]
                )
                search_lat, search_lon = p_geo.y, p_geo.x

            img = self._get_pano_img(search_lat, search_lon)

            val_veg = np.nan
            val_ter = np.nan

            if img:
                mask = self.segmenter.predict(img)
                metrics = self.segmenter.calculate_gvi_from_mask(mask)

                val_veg = metrics.get("GVI_Total", 0.0)
                val_ter = metrics.get("GVI_Terrain", 0.0)
                success_count += 1

                if save_panos:
                    if isinstance(img, np.ndarray):
                        img = Image.fromarray(img)
                    img.save(os.path.join(folder, "images", f"pano_{r}_{c}.jpg"))

                if save_masks:
                    if isinstance(mask, np.ndarray):
                        mask = mask.astype(np.uint8)
                        mask_img = Image.fromarray(mask)
                        mask_img.putpalette(CITYSCAPES_PALETTE)
                        mask_img.save(
                            os.path.join(folder, "masks", f"mask_{r}_{c}.png")
                        )
                    elif isinstance(mask, Image.Image):
                        mask.putpalette(CITYSCAPES_PALETTE)
                        mask.save(os.path.join(folder, "masks", f"mask_{r}_{c}.png"))

            if raster_veg is not None and not np.isnan(val_veg):
                raster_veg[r, c] = val_veg
                raster_ter[r, c] = val_ter

            results.append(
                {
                    "orig_index": orig_idx,  # Pass index to result
                    "geometry": pt["geometry"],
                    "gvi_veg": float(val_veg) if not np.isnan(val_veg) else None,
                    "gvi_ter": float(val_ter) if not np.isnan(val_ter) else None,
                    "lat": search_lat,
                    "lon": search_lon,
                    "row": r,
                    "col": c,
                }
            )

        print(f"[GVI] Finished. Successful Images: {success_count}/{len(points)}")

        # ---------------------------------------------------------
        # 3. SAVE TIFF (Polygon Mode Only)
        # ---------------------------------------------------------
        if write_tif and transform is not None:
            # (Keep existing TIF save logic here...)
            pass

        # Create GDF and RESTORE INDEX
        gdf_res = gpd.GeoDataFrame(results, crs=gdf.crs)
        if "orig_index" in gdf_res.columns:
            gdf_res.set_index("orig_index", inplace=True)
            gdf_res.index.name = None  # Clean up name

        return gdf_res
