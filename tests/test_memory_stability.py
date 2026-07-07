import argparse
import gc
import os
import sys
from unittest.mock import AsyncMock, patch

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from PIL import Image
from tqdm import tqdm

# Add parent path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import geopandas as gpd
from shapely.geometry import Point

from geofuse.gvi import GVIEngine
from geofuse.vision import get_best_device


def get_memory_usage():
    """Returns current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def _build_point_grid(n_points: int) -> gpd.GeoDataFrame:
    """A WGS84 GeoDataFrame of N points along a small line near Calgary.

    ``GVIEngine.run_analysis`` accepts a points GeoDataFrame; the actual
    coordinates are irrelevant because ``find_panorama_async`` and
    ``get_panorama_async`` are mocked out below.
    """
    # ~0.001° spacing keeps the sample area bounded
    pts = [Point(-114.13 + i * 0.001, 51.07) for i in range(n_points)]
    return gpd.GeoDataFrame({"geometry": pts}, crs="EPSG:4326")


def stress_test_pipeline(model_path, iterations, output_dir):
    """Drive ``GVIEngine.run_analysis`` for ``iterations`` mocked points.

    This mirrors the path the GVI runner takes in production (it also calls
    ``run_analysis``; see ``geofuse/jobs/runners.py``) instead of poking the
    engine's private ``_preprocess_image`` + ``segmenter`` directly. That way
    any allocation leak introduced *inside* ``run_analysis`` (point loop,
    pano cache, callback bookkeeping) is captured too — not just leaks inside
    the per-frame inference.
    """
    print(f"\n[TEST] Starting Memory Stress Test ({iterations} iterations)...")
    print(f"[INFO] Initial RAM: {get_memory_usage():.2f} MB")

    # GVIEngine init mirrors the runner: explicit device + package download mode
    device = get_best_device()
    engine = GVIEngine(
        model_path=model_path,
        download_mode="package",
        device=str(device),
    )

    if torch.cuda.is_available():
        print(f"[INFO] Initial VRAM: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

    # Synthetic panorama returned by the mocked Street View fetcher — same
    # size as a typical Street View tile (zoom=1 yields ~1664x832 panoramas;
    # 1024x512 is the cheapest size that still exercises preprocess + seg).
    fake_pano = Image.fromarray(
        np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
    )

    class _FakePano:
        id = "stress_test_pano"

    ram_log: list[float] = []
    vram_log: list[float] = []

    print("[INFO] Starting Processing Loop...")

    # ``run_analysis`` iterates through ``gdf`` rows, awaits a panorama for
    # each, runs segmentation, computes GVI, then invokes ``result_callback``.
    # By mocking the network fetchers we keep the loop deterministic and let
    # the test target the in-process memory profile.
    pbar = tqdm(total=iterations)

    def on_result(_res) -> None:
        ram_log.append(get_memory_usage())
        if torch.cuda.is_available():
            vram_log.append(torch.cuda.memory_allocated() / 1024**2)
        pbar.update(1)
        if len(ram_log) % 25 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    gdf = _build_point_grid(iterations)

    try:
        with (
            patch(
                "geofuse.streetview.find_panorama_async",
                new_callable=AsyncMock,
                return_value=_FakePano(),
            ),
            patch(
                "geofuse.streetview.get_panorama_async",
                new_callable=AsyncMock,
                return_value=fake_pano,
            ),
        ):
            engine.run_analysis(
                gdf,
                step=75,
                folder=output_dir,
                save_panos=False,
                save_masks=False,
                result_callback=on_result,
            )
    except KeyboardInterrupt:
        print("\n[WARN] Test interrupted by user. Generating report...")
    finally:
        pbar.close()

    if len(ram_log) > 0:
        print(f"\n[INFO] Final RAM: {ram_log[-1]:.2f} MB")
        growth = ram_log[-1] - ram_log[0]
        print(f"[RESULT] RAM Growth: {growth:.2f} MB")

        output_plot = os.path.join(output_dir, "memory_test.png")

        plt.figure(figsize=(10, 5))
        plt.plot(ram_log, label="RAM (MB)")
        if torch.cuda.is_available():
            plt.plot(vram_log, label="VRAM (MB)")
        plt.title(f"Memory Stability Test ({iterations} iters)")
        plt.xlabel("Iterations")
        plt.ylabel("Memory (MB)")
        plt.legend()
        plt.savefig(output_plot)
        print(f"[INFO] Memory plot saved to: {output_plot}")

        if growth > 1000:
            print("[FAIL] Significant memory leak detected (>1GB).")
            sys.exit(1)
        elif growth > 200:
            print("[WARN] Some memory growth detected (>200MB).")
        else:
            print("[PASS] Memory usage appears stable.")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(base_dir, "output", "test3_stability")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    MODEL_PATH = os.path.abspath(
        os.path.join(base_dir, "..", "geofuse", "model", "best_model.pth")
    )

    parser = argparse.ArgumentParser(description="Run Memory Stress Test")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if os.path.exists(MODEL_PATH):
        stress_test_pipeline(MODEL_PATH, args.iterations, OUTPUT_DIR)
    else:
        print(f"[ERROR] Model not found at {MODEL_PATH}")
