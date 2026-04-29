import os
import sys

# Must come before any geofuse import when the package is not editable-installed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import geopandas as gpd
from PIL import Image
from shapely.geometry import box

# search_panoramas is re-exported from geofuse.gvi (imported there from streetview)
from geofuse.gvi import GVIEngine, search_panoramas
from geofuse.ndvi import NDVIEngine

# ---------------------------------------------------------------------------
# Manual integration tests — run directly with `python tests/test_real_execution.py`
# These require network access (Street View API, Google Earth Engine) and a GPU.
# Not included in the automated CI suite.
# TODO: END_TO_END_TEST   – Add complete pipeline test: GVI + NDVI + Fusion
# TODO: BENCHMARK_TEST    – Add performance benchmarking for different hardware configs
# TODO: CLI_TEST          – Test MPI parallel execution with config.csv
# ---------------------------------------------------------------------------


def test_real_gvi(model_path, output_dir):
    print("\n[TEST] Starting Real GVI Test (Visual Save Enabled)...")

    try:
        engine = GVIEngine(model_path=model_path, download_mode="package")
        print("   [INFO] Engine initialized.")
    except Exception as e:
        print(f"   [FAIL] Engine init failed: {e}")
        return

    # Test locations: University of Calgary campus
    test_points = [
        (51.0782, -114.1360),
        (51.0745, -114.1206),
        (51.0776, -114.1337),
    ]

    success = False

    for lat, lon in test_points:
        print(f"   [INFO] Searching for pano at: {lat}, {lon}...")

        pano_img = None
        candidates = search_panoramas(lat=lat, lon=lon)
        if candidates:
            for meta in candidates:
                pid = engine._extract_panoid(meta)
                if not pid:
                    continue
                try:
                    pano_img = engine._download_async_wrapper(pid)
                    if pano_img is not None:
                        break
                except Exception as e:
                    print(f"   [WARN] Failed to download pano {pid}: {e}")

        if pano_img:
            print(f"   [PASS] Image found! Size: {pano_img.size}")

            pano_path = os.path.join(output_dir, "test_pano_rgb.jpg")
            pano_img.save(pano_path)
            print(f"   [SAVE] Saved panorama to: {pano_path}")

            try:
                mask = engine.segmenter.predict(pano_img)
                print(f"   [PASS] Segmentation complete. Mask Shape: {mask.shape}")

                metrics = engine.segmenter.calculate_gvi_from_mask(mask)
                print(f"   [PASS] Metrics: {metrics}")

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

        minx, miny = -114.1337 - 0.0045, 51.0776 - 0.0045
        maxx, maxy = -114.1337 + 0.0045, 51.0776 + 0.0045

        large_bbox = box(minx, miny, maxx, maxy)
        gdf = gpd.GeoDataFrame({"geometry": [large_bbox]}, crs="epsg:4326")

        print(f"   [INFO] Requesting Area: {large_bbox.area:.6f} sq deg (~1 sq km)")

        temp_file = os.path.join(output_dir, "large_area_test.geojson")
        gdf.to_file(temp_file, driver="GeoJSON")

        result = engine.download_and_process(
            geometry=gdf,
            start_date="2023-07-15",
            end_date="2023-07-31",
            output_name="test",
            folder=output_dir,
            resolution=10,
        )

        if result.get("status") == "success":
            out_file = result.get("tif")
            file_size = os.path.getsize(out_file) / 1024
            print(f"   [PASS] Large NDVI GeoTIFF exported ({file_size:.2f} KB)")
            print(f"   [INFO] Open '{out_file}' in QGIS/ArcGIS to verify location.")
        else:
            print(f"   [FAIL] NDVI export failed: {result}")

    except Exception as e:
        print(f"   [FAIL] GEE Error: {e}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(base_dir, "output", "test2_system")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    MODEL_PATH = os.path.abspath(
        os.path.join(base_dir, "..", "geofuse", "model", "best_model.pth")
    )

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        sys.exit(1)

    test_real_gvi(MODEL_PATH, OUTPUT_DIR)
    test_demanding_ndvi(OUTPUT_DIR)
