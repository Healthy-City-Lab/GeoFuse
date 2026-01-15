import sys
import platform
import subprocess
import shutil
import os

# =========================================================================
# CONFIGURATION
# =========================================================================

# 0. ENVIRONMENT NAME
ENV_NAME = "geofuse"

# 1. CONDA PACKAGES (System Binaries & Core Geospatial)
#    MPI packages are added dynamically below based on OS.
CONDA_PACKAGES = ["gdal=3.12.0", "geopandas=1.1.1"]

# 2. PYTORCH
PYTORCH_VERSION = "pytorch=2.4.1 torchvision=0.19.1 torchaudio=2.4.1"

# 3. PIP PACKAGES
PIP_PACKAGES = [
    # Geospatial
    "rasterio==1.4.4",
    "shapely==2.1.2",
    "fiona",
    # Data Science
    "matplotlib==3.10.8",
    "scikit-learn==1.6.0",
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
]

GIT_PACKAGES = [
    "git+https://github.com/robolyst/streetview.git@b07d69445161bc193a1e1a6aa5e098b6fcd6eef8"
]


def run_cmd(command):
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        while True:
            line = process.stdout.readline()
            if not line:
                break
            print(line, end="")

        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed with exit code {e.returncode}: {command}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        sys.exit(1)


def get_solver():
    return "mamba" if shutil.which("mamba") else "conda"


def get_mpi_packages(system):
    """
    Returns the correct MPI list for Conda based on OS.
    Reference: https://mpi4py.readthedocs.io/en/stable/install.html
    """
    if system == "Windows":
        # Windows requires Microsoft MPI (msmpi) binaries + the mpi4py wrapper
        return ["mpi4py=4.1.1", "msmpi"]

    elif system == "Darwin":
        # macOS (Intel/M1/M2/M3): OpenMPI is the recommended implementation on Conda-forge
        return ["mpi4py=4.1.1", "openmpi"]

    elif system == "Linux":
        # Linux: OpenMPI is the standard
        return ["mpi4py=4.1.1", "openmpi"]

    else:
        print(f"[WARN] Unknown OS: {system}. Defaulting to openmpi.")
        return ["mpi4py=4.1.1", "openmpi"]


def main(env_name):
    system = platform.system()
    solver = get_solver()

    print(f"\n[INFO] Starting Setup on {system} using {solver}...")

    # 1. Prepare Conda List (Core + OS-Specific MPI)
    mpi_pkgs = get_mpi_packages(system)
    full_conda_list = CONDA_PACKAGES + mpi_pkgs
    conda_str = " ".join(full_conda_list)

    print(f"\n[1/5] Installing Core & MPI Binaries ({solver})...")
    # Conda handles the binary linking for mpi4py automatically here
    run_cmd(f"{solver} install -n {env_name} -y -c conda-forge {conda_str}")

    # 2. Install PyTorch
    print(f"\n[2/5] Installing PyTorch Acceleration...")
    if system == "Windows" or system == "Linux":
        cmd = f"{solver} install -n {env_name} -y {PYTORCH_VERSION} pytorch-cuda=12.1 -c pytorch -c nvidia"
    else:
        # macOS uses CPU or MPS (Metal Performance Shaders), no CUDA
        # Use the standard command, ensuring the pytorch channel is primary.
        cmd = f"{solver} install -n {env_name} -y {PYTORCH_VERSION} -c pytorch"
    run_cmd(cmd)

    # 3. Install Pip Libraries
    print(f"\n[3/5] Installing Python Libraries...")
    pip_str = " ".join(PIP_PACKAGES)
    run_cmd(f'"{sys.executable}" -m pip install -q {pip_str}')

    # 4. Install Git Packages
    print(f"\n[4/5] Installing Custom Git Packages...")
    for git_url in GIT_PACKAGES:
        run_cmd(f'"{sys.executable}" -m pip install -q {git_url}')

    # 5. Editable Install
    print(f"\n[5/5] Performing Editable Install of GeoFuse...")
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    run_cmd(f'"{sys.executable}" -m pip install -q -e "{root_dir}"')

    # Finish & Verify
    print("\n[SUCCESS] Environment Configured.")

    # Construct a verification script that checks for the correct GPU backend based on OS
    verify_imports = (
        "import geofuse; import torch; import platform; from mpi4py import MPI; "
    )
    # Use single quotes for f-strings and no parentheses in the output text to be shell-safe
    verify_geofuse = "print(f'   [OK] GeoFuse Package: {geofuse.__file__}'); "
    verify_mpi = "print(f'   [OK] MPI Rank: {MPI.COMM_WORLD.Get_rank()} - Vendor: {MPI.get_vendor()}');"

    if system == "Darwin":
        verify_torch = "print(f'   [OK] PyTorch: {torch.__version__} - MPS Available: {torch.backends.mps.is_available()}'); "
    else:
        verify_torch = "print(f'   [OK] PyTorch: {torch.__version__} - CUDA Available: {torch.cuda.is_available()}'); "

    verify_script = verify_imports + verify_geofuse + verify_torch + verify_mpi

    run_cmd(f'"{sys.executable}" -c "{verify_script}"')


if __name__ == "__main__":
    main(ENV_NAME)
