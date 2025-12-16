import sys
import platform
import subprocess
import os

# ==============================================================================
# USER CONFIGURATION
# ==============================================================================
# Windows specific command (CUDA 12.8)
WINDOWS_TORCH_COMMAND = (
    "pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
)
# ==============================================================================


def run_cmd(command):
    try:
        print(f"[EXEC] {command}")
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError:
        print(f"[ERROR] Command failed: {command}")
        sys.exit(1)


def main():
    system = platform.system()
    print(f"[INFO] Detected OS: {system}")

    # 1. Install Heavy Dependencies
    if system == "Linux":
        print("[INFO] Linux detected. Installing PyTorch (HPC/CUDA)...")
        run_cmd(
            "conda install -y -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=12.1 rasterio gdal shapely"
        )

    elif system == "Windows":
        print("[INFO] Windows detected.")
        # A. Conda for Geospatial
        print("[INFO] Installing GDAL/Rasterio via Conda...")
        run_cmd("conda install -y -c conda-forge rasterio gdal shapely")

        # B. Custom PyTorch
        print(f"[INFO] Installing Custom PyTorch: {WINDOWS_TORCH_COMMAND}")
        run_cmd(WINDOWS_TORCH_COMMAND)

    else:
        run_cmd(
            "conda install -y -c pytorch -c conda-forge pytorch torchvision rasterio gdal shapely"
        )

    # 2. Install Pip Requirements (excluding torch/geo which we just did)
    print("[INFO] Installing remaining requirements...")
    req_path = os.path.join(os.path.dirname(__file__), "..", "requirements.txt")

    excluded = [
        "numpy",
        "pandas",
        "geopandas",
        "rasterio",
        "shapely",
        "torch",
        "torchvision",
    ]
    temp_reqs = "temp_requirements.txt"

    with open(req_path, "r") as f_in, open(temp_reqs, "w") as f_out:
        for line in f_in:
            pkg = line.split("=")[0].split("<")[0].split(">")[0].strip()
            if pkg.lower() not in excluded:
                f_out.write(line)

    try:
        run_cmd(f'"{sys.executable}" -m pip install -r {temp_reqs}')
    finally:
        if os.path.exists(temp_reqs):
            os.remove(temp_reqs)

    # 3. Install Custom Packages
    print("[INFO] Installing Streetview & GeoFuse...")
    try:
        run_cmd(
            f'"{sys.executable}" -m pip install git+https://github.com/robolyst/streetview'
        )
    except Exception as e:
        print(f"[WARN] Failed to install streetview package: {e}")

    root_dir = os.path.join(os.path.dirname(__file__), "..")
    run_cmd(f'"{sys.executable}" -m pip install -e "{root_dir}"')

    print("\n[SUCCESS] Environment setup complete!")
    print("Verification:")

    # FIXED LINE: We pass the string literal to the subprocess, avoiding local evaluation
    verify_cmd = f"\"{sys.executable}\" -c \"import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())\""
    run_cmd(verify_cmd)


if __name__ == "__main__":
    main()
