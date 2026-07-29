import os
import sys

# Must come before any geofuse import when the package is not editable-installed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import geopandas as gpd
from shapely.geometry import box

from geofuse.gvi import GVIEngine
from geofuse.ndvi import NDVIEngine
from geofuse.vision import get_best_device

# ────────────────────────────────────────────────────────────────────
# Manual integration tests — run directly with `python tests/test_real_execution.py`
# These require network access (Google Street View tile servers + Google
# Earth Engine) and a GPU. They are not included in the automated CI suite.
#
# Street View access uses GeoFuse's in-house scraper
# Earth Engine needs a one-time ``earthengine authenticate`` on
# the host. The NDVI test skips cleanly when EE is not authenticated.
# ────────────────────────────────────────────────────────────────────


def _print_skip(label: str, reason: str) -> None:
    print(f"   [SKIP] {label}: {reason}")


def test_real_gvi(model_path, output_dir):
    """Drive ``GVIEngine.run_analysis`` against live Street View tile servers.

    Uses the same engine entry point as the production runner
    (``geofuse/jobs/runners.py:run_gvi``) — a small WGS84 polygon, points
    materialised by ``run_analysis``'s polygon-fallback grid, real network
    calls through the in-house scraper.
    """
    print("\n[TEST] Starting Real GVI Test (Visual Save Enabled)...")

    try:
        device = get_best_device()
        engine = GVIEngine(
            model_path=model_path,
            device=str(device),
        )
        print(f"   [INFO] Engine initialized on {device}.")
    except Exception as e:
        print(f"   [FAIL] Engine init failed: {e}")
        return

    # ~500 m × 500 m polygon over downtown Calgary (8th Ave SW corridor —
    # dense Street View coverage). At step=75 m this materialises ~30–40
    # sample points, enough that the scraper's 50 m search radius reliably
    # resolves several panoramas regardless of which grid cells land off
    # the street centerline.
    cx, cy = -114.0719, 51.0447
    half = 0.0023  # ≈250 m at 51° N
    aoi = box(cx - half, cy - half, cx + half, cy + half)
    gdf = gpd.GeoDataFrame({"geometry": [aoi]}, crs="EPSG:4326")

    results: list[dict] = []
    try:
        engine.run_analysis(
            gdf,
            step=75,
            folder=output_dir,
            save_panos=True,
            save_masks=True,
            result_callback=results.append,
        )
    except Exception as e:
        print(f"   [FAIL] run_analysis raised: {e}")
        return

    # ``_make_result`` stores per-point GVI under ``gvi_veg``/``gvi_ter`` and
    # the resolved panorama id under ``pano_id``. Empty results carry
    # ``pano_id=None``.
    n_hits = sum(1 for r in results if r.get("pano_id"))
    print(f"   [INFO] Processed {len(results)} points; {n_hits} returned a panorama.")

    if n_hits == 0:
        print(
            "   [FAIL] No panoramas resolved within 50 m of any grid point. "
            "Street View scraper may be rate-limited, or the AOI has no "
            "coverage. Try a different location."
        )
        return

    # Pull one summary GVI value to confirm segmentation actually ran.
    first_hit = next((r for r in results if r.get("pano_id")), None)
    if first_hit is not None:
        veg = first_hit.get("gvi_veg")
        ter = first_hit.get("gvi_ter")
        pid = str(first_hit.get("pano_id"))[:12]
        print(f"   [PASS] Sample GVI: veg={veg:.3f}, ter={ter:.3f} (pano={pid}…)")

    # ``save_panos=True`` writes PNGs into <output_dir>/images. Verify at
    # least one was written so the pano-write path is exercised.
    image_dir = os.path.join(output_dir, "images")
    n_images = sum(1 for _ in os.scandir(image_dir)) if os.path.isdir(image_dir) else 0
    print(f"   [INFO] Pano images saved: {n_images} (folder: {image_dir})")


def test_demanding_ndvi(output_dir):
    """Drive ``NDVIEngine.download_and_process`` against live Earth Engine.

    Mirrors the production NDVI runner's call shape
    (``geofuse/jobs/runners.py:run_ndvi``). Skips cleanly when Earth Engine
    is not authenticated in the current env.
    """
    print("\n[TEST] Starting Demanding NDVI Test (1km x 1km Area)...")

    # Probe EE auth — engine init triggers ``ee.Initialize`` which is the
    # right point to surface a missing-auth condition early.
    try:
        engine = NDVIEngine()
    except Exception as e:
        _print_skip(
            "Real NDVI",
            f"NDVIEngine init failed (Earth Engine not authenticated?): {e}",
        )
        return

    try:
        minx, miny = -114.1337 - 0.0045, 51.0776 - 0.0045
        maxx, maxy = -114.1337 + 0.0045, 51.0776 + 0.0045

        large_bbox = box(minx, miny, maxx, maxy)
        gdf = gpd.GeoDataFrame({"geometry": [large_bbox]}, crs="epsg:4326")

        print(f"   [INFO] Requesting Area: {large_bbox.area:.6f} sq deg (~1 sq km)")

        result = engine.download_and_process(
            geometry=gdf,
            start_date="2023-07-15",
            end_date="2023-07-31",
            output_name="test",
            folder=output_dir,
            resolution=10,
        )

        if result.get("status") == "success":
            out_file = result.get("tif") or os.path.join(output_dir, "test_ndvi.tif")
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
