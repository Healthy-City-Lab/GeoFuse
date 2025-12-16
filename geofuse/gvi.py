import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.transform import from_origin
from PIL import Image
from tqdm import tqdm
import streetview

from .vision import DeepLabSegmenter


class GVIEngine:
    def __init__(self, model_path=None, download_mode="package", device="cuda"):
        """
        Args:
            model_path (str): Path to .pth file.
            download_mode (str): 'package' or 'api'.
            device (str): 'cuda' or 'cpu'.
        """
        self.download_mode = download_mode
        self.device = device

        print(f"[GVI] Initializing DeepLabV3+ Model on {device}...")
        self.segmenter = DeepLabSegmenter(ckpt_path=model_path, device=device)
        print("[GVI] Model Ready.")

    def _extract_panoid(self, pano_obj):
        """Helper to safely get panoid from either Object or Dict."""
        if hasattr(pano_obj, "pano_id"):
            return pano_obj.pano_id
        if hasattr(pano_obj, "panoid"):
            return pano_obj.panoid
        try:
            return pano_obj.get("pano_id") or pano_obj.get("panoid")
        except (TypeError, AttributeError):
            return None

    def _preprocess_image(self, img, target_width=1920):
        """
        Robust Cleaning Pipeline:
        1. Energy-Based Trimming: Shaves off black bars (bottom/right) even if noisy.
        2. Resize: standardizes to 1920x960 for VRAM efficiency.
        """
        if img is None:
            return None

        # Convert to NumPy
        arr = np.array(img)
        h, w, c = arr.shape

        # --- STEP 1: ENERGY-BASED TRIMMING ---
        # Instead of looking for ANY non-black pixel (sensitive to noise),
        # we sum the intensity of rows/cols to find where the "Meat" of the image is.

        # Threshold: A row must have an average brightness > 10 to be kept.
        # This handles compression artifacts in black areas.
        # axis=2 (RGB) -> axis=1 (Width) -> Result is (Height,)
        row_energy = np.mean(arr, axis=(1, 2))
        col_energy = np.mean(arr, axis=(0, 2))

        valid_rows = np.where(row_energy > 10)[0]
        valid_cols = np.where(col_energy > 10)[0]

        if len(valid_rows) == 0 or len(valid_cols) == 0:
            return None  # Image is effectively empty/black

        y_min, y_max = valid_rows[0], valid_rows[-1] + 1
        x_min, x_max = valid_cols[0], valid_cols[-1] + 1

        # QA: If the trimming removes > 90% of the image, it was likely bad
        if (y_max - y_min) < (h * 0.1) or (x_max - x_min) < (w * 0.1):
            return None

        # Crop to the "Energy Box"
        img_cropped = img.crop((x_min, y_min, x_max, y_max))

        # --- STEP 2: RESIZE ---
        # 1920x960 is the standard for efficient spherical processing (2:1 Ratio)
        target_height = int(target_width / 2)
        img_resized = img_cropped.resize((target_width, target_height), Image.BILINEAR)

        # --- STEP 3: FINAL FLATNESS CHECK ---
        # Re-check std dev on the final clean image to catch gray loading screens
        clean_arr = np.array(img_resized.convert("L"))
        if np.std(clean_arr) < 5:
            return None

        return img_resized

    def _get_pano_img(self, lat, lon):
        """
        Robust Scraper with Pre-processing
        """
        try:
            if self.download_mode == "package":
                panos = streetview.search_panoramas(lat, lon)
                if not panos:
                    return None

                # Check top 3 candidates
                for meta in panos[:3]:
                    try:
                        panoid = self._extract_panoid(meta)
                        if not panoid:
                            continue

                        # Download raw
                        raw_image = streetview.get_panorama(panoid)

                        # CLEAN & RESIZE
                        clean_image = self._preprocess_image(
                            raw_image, target_width=1920
                        )

                        if clean_image:
                            return clean_image

                    except Exception:
                        continue

                return None

            return None

        except Exception as e:
            return None

        return None

    def _generate_grid(self, gdf, resolution):
        """Creates a regular grid of points within the polygon geometry."""
        minx, miny, maxx, maxy = gdf.total_bounds
        x_coords = np.arange(minx, maxx, resolution)
        y_coords = np.arange(miny, maxy, resolution)

        grid_points = []
        rows = len(y_coords)
        cols = len(x_coords)
        transform = from_origin(minx, maxy, resolution, resolution)

        print(f"[GVI] Generating Grid: {rows}x{cols} ({rows*cols} potential points)")

        for r, y in enumerate(y_coords[::-1]):
            for c, x in enumerate(x_coords):
                p = Point(x, y)
                if gdf.contains(p).any():
                    grid_points.append(
                        {"geometry": p, "row": r, "col": c, "lat": y, "lon": x}
                    )

        return grid_points, transform, (rows, cols)

    def process_polygon(
        self, input_shp_path, output_tif_path, resolution=50, save_files=False
    ):
        """Main Pipeline"""
        print(f"[GVI] Loading Polygon: {input_shp_path}")
        gdf = gpd.read_file(input_shp_path)

        if gdf.crs.is_geographic:
            print(
                "[WARN] Input is Geographic (Lat/Lon). Resolution will be in Degrees!"
            )
            print("       Recommend reprojecting to UTM (meters) before running.")

        points, transform, shape = self._generate_grid(gdf, resolution)

        if not points:
            print("[FAIL] No grid points generated.")
            return

        gvi_raster = np.full(shape, -1.0, dtype=np.float32)
        print(f"[GVI] Processing {len(points)} points...")

        success_count = 0

        for pt in tqdm(points):
            img = self._get_pano_img(pt["lat"], pt["lon"])

            if img:
                mask = self.segmenter.predict(img)
                metrics = self.segmenter.calculate_gvi_from_mask(mask)

                gvi_raster[pt["row"], pt["col"]] = metrics["GVI_Total"]
                success_count += 1

                if save_files and success_count < 5:
                    os.makedirs("debug_output", exist_ok=True)
                    img.save(f"debug_output/pano_{pt['row']}_{pt['col']}.jpg")

        print(f"[GVI] Finished. Successful Points: {success_count}/{len(points)}")
        print(f"[GVI] Saving Raster to {output_tif_path}...")

        with rasterio.open(
            output_tif_path,
            "w",
            driver="GTiff",
            height=shape[0],
            width=shape[1],
            count=1,
            dtype=gvi_raster.dtype,
            crs=gdf.crs,
            transform=transform,
            nodata=-1.0,
        ) as dst:
            dst.write(gvi_raster, 1)

        print("[SUCCESS] Pipeline Complete.")
