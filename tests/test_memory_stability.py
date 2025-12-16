import sys
import os
import psutil
import torch
import gc
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import geofuse
from geofuse.gvi import GVIEngine


def get_memory_usage():
    """Returns current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def stress_test_pipeline(model_path, iterations):
    print(f"\n[TEST] Starting Memory Stress Test ({iterations} iterations)...")
    print(f"[INFO] Initial RAM: {get_memory_usage():.2f} MB")

    # 1. Initialize Engine
    engine = GVIEngine(model_path=model_path, download_mode="package", device="cuda")

    # Measure VRAM if available
    if torch.cuda.is_available():
        print(f"[INFO] Initial VRAM: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

    ram_log = []
    vram_log = []

    print("[INFO] Starting Processing Loop...")

    try:
        for i in tqdm(range(iterations)):
            # 2. Simulate a Raw Download (Random Noise Image)
            # 4096x2048 is roughly the size of a standard Street View pano
            raw_data = np.random.randint(0, 255, (2048, 4096, 3), dtype=np.uint8)

            # Add a black border to force the Trimmer to work
            raw_data[-200:, :] = 0

            fake_img = Image.fromarray(raw_data)

            # 3. Run the EXACT Pipeline
            # Step A: Preprocess (Trim + Resize)
            clean_img = engine._preprocess_image(fake_img, target_width=1920)

            if clean_img:
                # Step B: Segment
                mask = engine.segmenter.predict(clean_img)

                # Step C: Calculate Metrics
                _ = engine.segmenter.calculate_gvi_from_mask(mask)

            # 4. Cleanup
            del raw_data
            del fake_img
            del clean_img
            del mask

            # Force Garbage Collection
            # In a loop this tight, explicit GC helps prevent false positives
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Log Memory
            ram_log.append(get_memory_usage())
            if torch.cuda.is_available():
                vram_log.append(torch.cuda.memory_allocated() / 1024**2)

    except KeyboardInterrupt:
        print(
            "\n[WARN] Test interrupted by user. Generating report for completed steps..."
        )

    # 5. Analysis
    if len(ram_log) > 0:
        print(f"\n[INFO] Final RAM: {ram_log[-1]:.2f} MB")

        growth = ram_log[-1] - ram_log[0]
        print(f"[RESULT] RAM Growth: {growth:.2f} MB")

        if torch.cuda.is_available():
            vram_growth = vram_log[-1] - vram_log[0]
            print(f"[RESULT] VRAM Growth: {vram_growth:.2f} MB")

        # Plotting
        output_plot = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "output", "memory_test.png")
        )
        os.makedirs(os.path.dirname(output_plot), exist_ok=True)

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

        # Strict Pass/Fail for 5000 iterations
        # We allow slightly more growth over a long run due to Python fragmentation, but it should plateau.
        if growth > 1000:  # 1GB leak is a fail
            print("[FAIL] Significant memory leak detected (>1GB).")
        elif growth > 200:
            print(
                "[WARN] Some memory growth detected (>200MB). Check plot for plateau."
            )
        else:
            print("[PASS] Memory usage appears stable.")


if __name__ == "__main__":
    # Path Logic
    base_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.abspath(
        os.path.join(base_dir, "..", "geofuse", "model", "best_model.pth")
    )

    # Argument Parser
    parser = argparse.ArgumentParser(description="Run Memory Stress Test")
    parser.add_argument(
        "--iterations",
        type=int,
        default=5000,
        help="Number of iterations to run (default: 5000)",
    )
    args = parser.parse_args()

    if os.path.exists(MODEL_PATH):
        stress_test_pipeline(MODEL_PATH, args.iterations)
    else:
        print(f"[ERROR] Model not found at {MODEL_PATH}")
