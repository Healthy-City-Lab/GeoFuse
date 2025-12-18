import sys
import os
import torch
import numpy as np
from PIL import Image
import geopandas as gpd
from shapely.geometry import box

# Add parent path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import geofuse

from geofuse.gvi import GVIEngine
from geofuse.ndvi import NDVIEngine


def test_real_gvi(model_path, output_dir):
    print("\n[TEST] Starting Real GVI Test (Visual Save Enabled)...")

    # 1. Setup Engine
    try:
        engine = GVIEngine(
            model_path=model_path, download_mode="package", device="cuda"
        )
        print("   [INFO] Engine initialized.")
    except Exception as e:
        print(f"   [FAIL] Engine init failed: {e}")
        return

    # 2. Test Locations (University of Calgary)
    test_points = [
        (51.0782, -114.1360),
        (51.0745, -114.1206),
        (51.0776, -114.1337),
    ]

    success = False

    for lat, lon in test_points:
        print(f"   [INFO] Searching for pano at: {lat}, {lon}...")
        img = engine._get_pano_img(lat, lon)

        if img:
            print(f"   [PASS] Image found! Size: {img.size}")

            # --- SAVE ORIGINAL ---
            pano_path = os.path.join(output_dir, "test_pano_rgb.jpg")
            img.save(pano_path)
            print(f"   [SAVE] Saved panorama to: {pano_path}")

            # 3. Test Segmentation
            try:
                mask = engine.segmenter.predict(img)
                print(f"   [PASS] Segmentation complete. Mask Shape: {mask.shape}")

                metrics = engine.segmenter.calculate_gvi_from_mask(mask)
                print(f"   [PASS] Metrics: {metrics}")

                # --- SAVE COLORED MASK ---
                color_mask = engine.segmenter.decode_fn(mask)
                mask_img = Image.fromarray(color_mask)

                mask_path = os.path.join(output_dir, "test_pano_mask.png")
                mask_img.save(mask_path)
                print(f"   [SAVE] Saved colorized mask to: {mask_path}")

                success = True
                break
            except Exception as e:
                print(f"   [FAIL] Vision pipeline error: {e}")
        else:
            print("   [WARN] No panorama found here. Retrying next point...")

    if not success:
        print("   [FAIL] Could not find any panoramas in test set.")


def test_demanding_ndvi(output_dir):
    print("\n[TEST] Starting Demanding NDVI Test (1km x 1km Area)...")
    try:
        engine = NDVIEngine()

        # 1. Create a 1km Box around UofC
        minx, miny = -114.1337 - 0.0045, 51.0776 - 0.0045
        maxx, maxy = -114.1337 + 0.0045, 51.0776 + 0.0045

        large_bbox = box(minx, miny, maxx, maxy)
        gdf = gpd.GeoDataFrame({"geometry": [large_bbox]}, crs="epsg:4326")

        print(f"   [INFO] Requesting Area: {large_bbox.area:.6f} sq deg (~1 sq km)")

        # Save temp file inside the specific test output folder
        temp_file = os.path.join(output_dir, "large_area_test.geojson")
        gdf.to_file(temp_file, driver="GeoJSON")

        out_file = os.path.join(output_dir, "large_ndvi.tif")

        # 2. Export with higher resolution (10m)
        success = engine.export_geotiff(
            temp_file, "2023-07-15", out_file, resolution=10
        )

        if success:
            file_size = os.path.getsize(out_file) / 1024  # KB
            print(f"   [PASS] Large NDVI GeoTIFF exported ({file_size:.2f} KB)")
            print(
                f"   [INFO] You can open '{out_file}' in QGIS/ArcGIS to verify location."
            )

    except Exception as e:
        print(f"   [FAIL] GEE Error: {e}")


if __name__ == "__main__":
    # Path Logic
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Setup Output Directory
    OUTPUT_DIR = os.path.join(base_dir, "output", "test2_system")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    MODEL_PATH = os.path.abspath(
        os.path.join(base_dir, "..", "geofuse", "model", "best_model.pth")
    )

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
    else:
        test_real_gvi(MODEL_PATH, OUTPUT_DIR)
        test_demanding_ndvi(OUTPUT_DIR)
