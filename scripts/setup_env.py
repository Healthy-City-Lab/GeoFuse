import sys
import platform
import subprocess
import shutil
import os

# ==============================================================================
# CONFIGURATION (Strict Versions from environment.yml)
# ==============================================================================

# 1. CONDA PACKAGES (System Binaries & Core Geospatial)
#    We pin 'gdal' to 3.12.0 to match your 'libgdal-core=3.12.0'
#    We pin 'geopandas' to 1.1.1 to match 'geopandas-base=1.1.1'
CONDA_PACKAGES = "gdal=3.12.0 " "geopandas=1.1.1"

# 2. PYTORCH (Exact Match)
PYTORCH_VERSION = "pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1"

# 3. PIP PACKAGES (Python Libraries)
#    These are all pinned to the versions found in your pip freeze list.
PIP_PACKAGES = [
    # Geospatial (Installed via Pip in your YAML)
    "rasterio==1.4.4",
    "shapely==2.1.2",
    "fiona",  # Let GeoPandas/Rasterio resolve this, or pin if needed
    # Data Science
    "matplotlib==3.10.8",
    "scikit-learn==1.6.0",  # Your YAML implies 1.6.0 based on surrounding packages
    "scipy==1.16.3",
    "tqdm==4.67.1",
    "pillow==12.0.0",
    "optuna==4.6.0",
    "skrebate==0.62",
    # UI / Web
    "streamlit==1.52.1",
    "streamlit-folium==0.25.3",
    "folium==0.20.0",
    "geemap==0.36.6",
    "earthengine-api==1.7.4",
    # Visualization
    "visdom==0.2.4",
    "dominate==2.9.1",
    "mpi4py==4.1.1",
]

GIT_PACKAGES = [
    "git+https://github.com/robolyst/streetview.git@b07d69445161bc193a1e1a6aa5e098b6fcd6eef8"
]


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
    # Note: We do NOT use quotes around CONDA_PACKAGES string in the f-string
    # to allow the solver to parse the spaces correctly.
    run_cmd(f"{solver} install -y -c conda-forge {CONDA_PACKAGES}")

    # 2. Install PyTorch
    print(f"\n[2/5] Installing PyTorch Acceleration...")
    # Matches your YAML 'pytorch-cuda=12.1'
    if system == "Windows" or system == "Linux":
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
