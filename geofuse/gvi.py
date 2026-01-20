import asyncio
import os
import time
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.transform import from_origin
from shapely.geometry import Point
from tqdm import tqdm

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

from streetview import get_panorama_async, get_streetview, search_panoramas

from .vision import DeepLabSegmenter, get_best_device


class GVIEngine:
    def __init__(
        self, model_path=None, download_mode="package", device="cuda", api_key=None
    ):
        self.download_mode = download_mode
        # Use smart device selection
        self.device = get_best_device(device)
        self.api_key = api_key

        print(f"[GVI] Initializing DeepLabV3+ Model on {self.device}...")
        self.segmenter = DeepLabSegmenter(ckpt_path=model_path, device=str(self.device))
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

        if isinstance(img, np.ndarray):
            try:
                img = Image.fromarray(img)
            except:
                return None

        arr = np.array(img)

        # 1. Detect and remove black edges
        row_energy = np.mean(arr, axis=(1, 2))
        col_energy = np.mean(arr, axis=(0, 2))

        valid_rows = np.where(row_energy > 5)[0]
        valid_cols = np.where(col_energy > 5)[0]

        if len(valid_rows) == 0 or len(valid_cols) == 0:
            return None

        y_min, y_max = valid_rows[0], valid_rows[-1] + 1
        x_min, x_max = valid_cols[0], valid_cols[-1] + 1

        # 2. Extract current image height
        height = y_max - y_min

        # 3. Enforce 2:1 Aspect Ratio (Width = Height * 2)
        expected_width = height * 2
        current_width = x_max - x_min

        # Center Crop if too wide
        if current_width > expected_width:
            center_x = (x_min + x_max) // 2
            half_width = expected_width // 2
            x_min = max(0, center_x - half_width)
            x_max = x_min + expected_width

        img_cropped = img.crop((x_min, y_min, x_max, y_max))

        # 4. Resize to target dimensions
        target_height = int(target_width / 2)
        return img_cropped.resize((target_width, target_height), Image.BILINEAR)

    def _download_async_wrapper(self, panoid, timeout=30):
        """Download panorama with timeout to prevent hanging."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Add timeout to prevent indefinite blocking
            return loop.run_until_complete(
                asyncio.wait_for(
                    get_panorama_async(pano_id=panoid, zoom=1), timeout=timeout
                )
            )
        except asyncio.TimeoutError:
            print(f"[GVI] Timeout downloading panorama {panoid}")
            return None
        except Exception:
            return None
        finally:
            loop.close()

    def _generate_pixel_aligned_grid(self, gdf, resolution):
        """Generate grid with proper CRS handling."""
        # If input is in geographic coordinates, convert to local UTM for accurate grid spacing
        if gdf.crs.is_geographic:
            # Get center point to determine UTM zone
            centroid = gdf.geometry.iloc[0].centroid
            lon, lat = centroid.x, centroid.y

            # Calculate UTM zone
            utm_zone = int((lon + 180) / 6) + 1
            utm_crs = f"EPSG:326{utm_zone}" if lat >= 0 else f"EPSG:327{utm_zone}"

            print(f"[GVI] Converting from {gdf.crs} to {utm_crs} for grid generation")
            gdf_metric = gdf.to_crs(utm_crs)
        else:
            gdf_metric = gdf

        minx, miny, maxx, maxy = gdf_metric.total_bounds
        width = int(np.ceil((maxx - minx) / resolution))
        height = int(np.ceil((maxy - miny) / resolution))

        print(f"[GVI] Grid dimensions: {width}x{height} = {width*height} points")
        print(
            f"[GVI] Area bounds (metric): ({minx:.2f}, {miny:.2f}) to ({maxx:.2f}, {maxy:.2f})"
        )

        transform = from_origin(minx, maxy, resolution, resolution)
        cols = np.arange(width)
        rows = np.arange(height)
        pixel_centers = []

        for r in rows:
            y_center = maxy - (r + 0.5) * resolution
            for c in cols:
                x_center = minx + (c + 0.5) * resolution
                p = Point(x_center, y_center)
                if gdf_metric.contains(p).any():
                    pixel_centers.append(
                        {
                            "geometry": p,
                            "row": r,
                            "col": c,
                            "lat": y_center,  # These will be in metric CRS
                            "lon": x_center,
                        }
                    )

        print(f"[GVI] Generated {len(pixel_centers)} points within polygon")

        # Convert points back to original CRS if we converted
        if gdf.crs.is_geographic:
            gdf_points = gpd.GeoDataFrame(
                pixel_centers,
                geometry=[p["geometry"] for p in pixel_centers],
                crs=utm_crs,
            )
            gdf_points = gdf_points.to_crs(gdf.crs)

            # Update coordinates in pixel_centers
            for i, (idx, row) in enumerate(gdf_points.iterrows()):
                pixel_centers[i]["geometry"] = row.geometry
                pixel_centers[i]["lon"] = row.geometry.x
                pixel_centers[i]["lat"] = row.geometry.y

        return pixel_centers, height, width, transform

    # TODO: ERROR_RECOVERY - Implement automatic retry logic for failed panorama downloads
    # TODO: PARALLEL_DOWNLOAD - Add concurrent panorama downloading for multiple points
    # TODO: CHECKPOINT_SAVE - Periodically save progress to disk for crash recovery

    def run_analysis(
        self,
        gdf,
        step: float | int = 75,
        folder="output",
        save_panos=False,
        save_masks=False,
        external_cache=None,
        progress_callback=None,
        result_callback=None,
        cancel_callback=None,
        start_index=0,
    ):
        print(f"[GVI] Starting Analysis (Resume Index: {start_index})...")
        os.makedirs(folder, exist_ok=True)
        if save_panos:
            os.makedirs(os.path.join(folder, "images"), exist_ok=True)
        if save_masks:
            os.makedirs(os.path.join(folder, "masks"), exist_ok=True)

        # 1. PREPARE INPUT POINTS (FULL LIST)
        first_geom = gdf.geometry.iloc[0]
        points = []
        rows = 0
        cols = 0
        transform = None
        write_tif = False

        if first_geom.geom_type in ["Polygon", "MultiPolygon"]:
            points_data, rows, cols, transform = self._generate_pixel_aligned_grid(
                gdf, step
            )
            for i, p in enumerate(points_data):
                p["orig_index"] = i
            points = points_data
            write_tif = True
        else:
            has_indices = "row" in gdf.columns and "col" in gdf.columns
            for idx, row in gdf.iterrows():
                points.append(
                    {
                        "orig_index": idx,
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

        # 2. SLICE FOR RESUME
        total_points = len(points)
        points_to_process = points[start_index:]

        if len(points_to_process) == 0:
            print("[GVI] All points already processed.")
            return gpd.GeoDataFrame()

        # 3. RUN ANALYSIS LOOP
        pano_cache = external_cache if external_cache is not None else {}

        print(f"[GVI] Processing {len(points_to_process)} remaining points...")

        for i, pt in enumerate(tqdm(points_to_process)):

            # --- CHECK CANCELLATION ---
            if cancel_callback and cancel_callback():
                print("[GVI] Analysis Aborted by User.")
                break

            current_global_idx = start_index + i

            if progress_callback:
                progress_callback(current_global_idx, total_points)

            r, c = pt["row"], pt["col"]
            lat, lon = pt["lat"], pt["lon"]
            orig_idx = pt["orig_index"]

            search_lat, search_lon = lat, lon
            if not gdf.crs.is_geographic:
                try:
                    p_geo = (
                        gpd.GeoSeries([pt["geometry"]], crs=gdf.crs)
                        .to_crs(epsg=4326)
                        .iloc[0]
                    )
                    search_lat, search_lon = p_geo.y, p_geo.x
                except Exception as e:
                    # If CRS transform fails, skip this point
                    print(f"[GVI] CRS transform failed: {e}")
                    continue

            # Search for panoramas with timeout protection
            try:
                candidates = search_panoramas(lat=search_lat, lon=search_lon)
            except Exception as e:
                # If search fails, skip this point
                candidates = None
            final_panoid = None
            val_veg = np.nan
            val_ter = np.nan

            if candidates:
                for meta in candidates:
                    # Check for cancellation inside candidate loop
                    if cancel_callback and cancel_callback():
                        print(
                            "[GVI] Analysis Aborted by User (during panorama processing)."
                        )
                        break

                    pid = self._extract_panoid(meta)
                    if not pid:
                        continue

                    if pid in pano_cache:
                        cached = pano_cache[pid]
                        val_veg = cached["veg"]
                        val_ter = cached["ter"]
                        final_panoid = pid
                        break

                    try:
                        raw_image = None
                        if self.api_key:
                            raw_image = get_streetview(
                                pano_id=pid, api_key=self.api_key
                            )
                        else:
                            raw_image = self._download_async_wrapper(pid)

                        if raw_image is not None:
                            # Check cancel before expensive processing
                            if cancel_callback and cancel_callback():
                                print(
                                    "[GVI] Analysis Aborted by User (before image processing)."
                                )
                                break

                            img = self._preprocess_image(raw_image)
                            if img:
                                # GPU inference - could hang on CUDA issues
                                try:
                                    mask = self.segmenter.predict(img)
                                except Exception as e:
                                    # Skip this image if GPU inference fails
                                    print(f"[GVI] GPU inference failed for {pid}: {e}")
                                    continue
                                metrics = self.segmenter.calculate_gvi_from_mask(mask)

                                val_veg = metrics.get("GVI_Total", 0.0)
                                val_ter = metrics.get("GVI_Terrain", 0.0)
                                final_panoid = pid

                                if save_panos:
                                    p_path = os.path.join(
                                        folder, "images", f"{pid}.jpg"
                                    )
                                    if not os.path.exists(p_path):
                                        if isinstance(img, np.ndarray):
                                            img = Image.fromarray(img)
                                        img.save(p_path)

                                if save_masks:
                                    m_path = os.path.join(folder, "masks", f"{pid}.png")
                                    if not os.path.exists(m_path):
                                        if isinstance(mask, np.ndarray):
                                            mask = mask.astype(np.uint8)
                                            mask_img = Image.fromarray(mask)
                                            mask_img.putpalette(CITYSCAPES_PALETTE)
                                            mask_img.save(m_path)
                                        elif isinstance(mask, Image.Image):
                                            mask.putpalette(CITYSCAPES_PALETTE)
                                            mask.save(m_path)

                                pano_cache[pid] = {"veg": val_veg, "ter": val_ter}
                                break
                    except KeyboardInterrupt:
                        # Re-raise keyboard interrupt to stop everything
                        raise
                    except Exception as e:
                        # Log and skip problematic panoramas
                        # Don't let one bad panorama block the entire process
                        continue

                # Check if we should abort after processing candidates
                if cancel_callback and cancel_callback():
                    break

            res_dict = {
                "orig_index": orig_idx,
                "geometry": pt["geometry"],
                "gvi_veg": float(val_veg) if not np.isnan(val_veg) else None,
                "gvi_ter": float(val_ter) if not np.isnan(val_ter) else None,
                "pano_id": final_panoid,
                "lat": search_lat,
                "lon": search_lon,
                "row": r,
                "col": c,
            }

            if result_callback:
                result_callback(res_dict)

        return gpd.GeoDataFrame()
