import argparse
import gc
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from PIL import Image
from tqdm import tqdm

# Add parent path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import geofuse
from geofuse.gvi import GVIEngine


def get_memory_usage():
    """Returns current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def stress_test_pipeline(model_path, iterations, output_dir):
    print(f"\n[TEST] Starting Memory Stress Test ({iterations} iterations)...")
    print(f"[INFO] Initial RAM: {get_memory_usage():.2f} MB")

    # Auto-selects best device: CUDA > MPS > CPU
    engine = GVIEngine(model_path=model_path, download_mode="package")

    if torch.cuda.is_available():
        print(f"[INFO] Initial VRAM: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

    ram_log = []
    vram_log = []

    print("[INFO] Starting Processing Loop...")

    try:
        for i in tqdm(range(iterations)):
            # Simulate a 4K image
            raw_data = np.random.randint(0, 255, (2048, 4096, 3), dtype=np.uint8)
            raw_data[-200:, :] = 0
            fake_img = Image.fromarray(raw_data)

            clean_img = engine._preprocess_image(fake_img, target_width=1920)

            mask = None
            if clean_img:
                mask = engine.segmenter.predict(clean_img)
                _ = engine.segmenter.calculate_gvi_from_mask(mask)

            del raw_data, fake_img, clean_img, mask
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            ram_log.append(get_memory_usage())
            if torch.cuda.is_available():
                vram_log.append(torch.cuda.memory_allocated() / 1024**2)

    except KeyboardInterrupt:
        print("\n[WARN] Test interrupted by user. Generating report...")

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
    parser.add_argument("--iterations", type=int, default=5000)
    args = parser.parse_args()

    if os.path.exists(MODEL_PATH):
        stress_test_pipeline(MODEL_PATH, args.iterations, OUTPUT_DIR)
    else:
        print(f"[ERROR] Model not found at {MODEL_PATH}")
