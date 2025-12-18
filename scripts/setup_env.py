import sys
import platform
import subprocess
import shutil
import os  # <--- Added back (Required for path operations)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONDA_PACKAGES = (
    "gdal geopandas rasterio shapely fiona matplotlib scikit-learn scipy tqdm"
)

PYTORCH_VERSION = "pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1"

PIP_PACKAGES = [
    "streamlit",
    "streamlit-folium",
    "folium",
    "earthengine-api",
    "geemap",
    "optuna",
    "skrebate",
    "visdom",
    "dominate",
    "mpi4py",
    "Pillow",
]

GIT_PACKAGES = ["git+https://github.com/robolyst/streetview"]


def run_cmd(command):
    try:
        print(f"[EXEC] {command}")
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError:
        print(f"[ERROR] Command failed: {command}")
        sys.exit(1)


def get_solver():
    return "mamba" if shutil.which("mamba") else "conda"


def main():
    system = platform.system()
    solver = get_solver()

    print(f"\n[INFO] Starting Setup on {system} using {solver}...")

    # 1. Install Conda Dependencies
    print(f"\n[1/5] Installing Geospatial Core ({solver})...")
    run_cmd(f"{solver} install -y -c conda-forge {CONDA_PACKAGES}")

    # 2. Install PyTorch
    print(f"\n[2/5] Installing PyTorch Acceleration...")
    if system == "Windows":
        cmd = f"{solver} install -y {PYTORCH_VERSION} pytorch-cuda=12.1 -c pytorch -c nvidia"
    elif system == "Linux":
        cmd = f"{solver} install -y {PYTORCH_VERSION} pytorch-cuda=12.1 -c pytorch -c nvidia"
    else:
        cmd = f"{solver} install -y {PYTORCH_VERSION} -c pytorch"
    run_cmd(cmd)

    # 3. Install Pip Libraries
    print(f"\n[3/5] Installing Python Libraries...")
    pip_str = " ".join(PIP_PACKAGES)
    run_cmd(f'"{sys.executable}" -m pip install {pip_str}')

    # 4. Install Git Packages
    print(f"\n[4/5] Installing Custom Git Packages...")
    for git_url in GIT_PACKAGES:
        run_cmd(f'"{sys.executable}" -m pip install {git_url}')

    # 5. Editable Install
    print(f"\n[5/5] Performing Editable Install of GeoFuse...")
    # This command looks for setup.py in the parent directory ("..")
    # assuming this script is running from /scripts/
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    run_cmd(f'"{sys.executable}" -m pip install -e "{root_dir}"')

    # Finish
    print("\n[SUCCESS] Environment Configured.")
    verify_script = (
        "import geofuse; print(f'   [OK] GeoFuse Package: {geofuse.__file__}'); "
        "import torch; print(f'   [OK] PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
    )
    run_cmd(f'"{sys.executable}" -c "{verify_script}"')


if __name__ == "__main__":
    main()
