import os
import platform
import shutil
import subprocess
import sys

# =========================================================================
# CONFIGURATION
# =========================================================================

# 0. ENVIRONMENT NAME
ENV_NAME = "geofuse"

# 1. CONDA PACKAGES (System Binaries & Core Geospatial)
#    MPI packages are added dynamically below based on OS.
CONDA_PACKAGES = [
    "libblas=*=*openblas",
    "libcblas=*=*openblas",
    "liblapack=*=*openblas",
    "numpy=2.4.6",
    "scipy=1.17.1",
    "gdal=3.12.0",
    "geopandas=1.1.1",
    "statsmodels=0.14.6",
    "cmaes=0.13.0",
]

# 2. PYTORCH
PYTORCH_VERSION = "torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0"

# 3. PIP PACKAGES
PIP_PACKAGES = [
    # Geospatial
    "rasterio==1.4.4",
    "shapely==2.1.2",
    "fiona==1.10.1",
    # Data Science
    "matplotlib==3.10.8",
    "scikit-learn==1.6.0",
    "tqdm==4.67.1",
    "pillow==12.0.0",
    "optuna==4.6.0",
    "skrebate==0.62",
    # Partial distance correlation (default objective) + spline residualization
    "dcor==0.7",
    "patsy==1.0.2",
    # UI / Web
    "streamlit==1.52.1",
    "streamlit-folium==0.25.3",
    "streamlit-sortables==0.3.1",
    "folium==0.20.0",
    # Earth Engine & Geospatial APIs
    "geemap==0.36.6",
    "earthengine-api==1.7.4",
    # Async HTTP — used by the in-house geofuse.streetview client (GVIEngine)
    "aiohttp==3.9.5",
    # Utility
    "isort==7.0.0",
    # Dev tooling — must match the version pinned in .github/workflows/code-quality.yml
    "black==25.1.0",
    "ruff==0.15.12",
]


def run_cmd(command, log_file=None, optional=False):
    """Run a shell command, streaming output to both console and (optionally) a log file.

    Always echoes to the console so CI logs capture errors inline.  On
    non-zero exit the last 40 lines of output are reprinted before aborting,
    ensuring the failure reason is visible even when the surrounding log is large.

    If ``optional=True`` the installer prints a warning on failure and
    continues instead of aborting.  Use this for packages that degrade
    gracefully when absent (e.g. training-only visualisation tools).
    """
    output_lines = []
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
            output_lines.append(line)
            print(line, end="", flush=True)
            if log_file:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(line)

        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

    except subprocess.CalledProcessError as e:
        tail = output_lines[-40:] if len(output_lines) > 40 else output_lines
        if optional:
            print(f"\n[WARN] Optional command failed (continuing): {command}")
            if tail:
                print("\n--- last output ---")
                for ln in tail:
                    print(ln, end="")
                print("--- end ---\n")
        else:
            print(f"\nError:  Command failed with exit code {e.returncode}: {command}")
            if tail:
                print("\n--- last output ---")
                for ln in tail:
                    print(ln, end="")
                print("--- end ---\n")
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


def get_env_python(env_name):
    """Get the Python executable path for a conda environment."""
    system = platform.system()

    # Get conda root
    conda_info = subprocess.run(
        "conda info --base", shell=True, capture_output=True, text=True, check=True
    )
    conda_root = conda_info.stdout.strip()

    if system == "Windows":
        return os.path.join(conda_root, "envs", env_name, "python.exe")
    else:
        return os.path.join(conda_root, "envs", env_name, "bin", "python")


def main(env_name, log_file=None):
    system = platform.system()
    solver = get_solver()

    try:
        _conda_base = subprocess.run(
            "conda info --base",
            shell=True,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        os.environ["MAMBA_ROOT_PREFIX"] = _conda_base
        print(f"[INFO] MAMBA_ROOT_PREFIX -> {_conda_base}")
    except Exception as exc:
        print(f"[WARN] Could not detect conda base, mamba may fail: {exc}")

    # Create log file automatically if not provided
    if not log_file:
        log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "install.log")

    # Clear previous log file
    if os.path.exists(log_file):
        os.remove(log_file)

    print(f"\n[INFO] Starting Setup on {system} using {solver}...")
    print(f"[INFO] Detailed logs: {log_file}\n")

    # 0. Create/Reset Environment (with console output for debugging)
    print(f"[0/6] Checking environment '{env_name}'...")
    # Check if environment exists
    env_check = subprocess.run(
        "conda env list", shell=True, capture_output=True, text=True
    )
    env_exists = env_name in env_check.stdout

    if env_exists:
        print(f"[WARN] Environment '{env_name}' already exists. Removing...")
        subprocess.run(f"conda env remove -n {env_name} -y", shell=True, check=True)

    print(f"[INFO] Creating environment '{env_name}' with Python 3.13...")
    subprocess.run(
        f"conda create -n {env_name} python=3.13 pip -y -c conda-forge",
        shell=True,
        check=True,
    )
    print("[SUCCESS] Environment created successfully\n")

    # 1. Prepare Conda List (Core + OS-Specific MPI)
    mpi_pkgs = get_mpi_packages(system)
    full_conda_list = CONDA_PACKAGES + mpi_pkgs
    conda_str = " ".join(full_conda_list)
    env_python = get_env_python(env_name)

    print(f"[1/5] Installing Core & MPI Binaries ({solver})...")
    # Conda handles the binary linking for mpi4py automatically here
    run_cmd(f"{solver} install -n {env_name} -y -c conda-forge {conda_str}", log_file)

    # 2. Install PyTorch
    print("[2/5] Installing PyTorch Acceleration...")
    if system == "Windows" or system == "Linux":
        cmd = f'"{env_python}" -m pip install {PYTORCH_VERSION} --index-url https://download.pytorch.org/whl/cu128 --no-warn-script-location'
    else:
        # macOS uses CPU or MPS (Metal Performance Shaders), no CUDA
        # Use the standard command, ensuring the pytorch channel is primary.
        cmd = (
            f'"{env_python}" -m pip install {PYTORCH_VERSION} --no-warn-script-location'
        )
    run_cmd(cmd, log_file)

    # 3. Install Pip Libraries
    print("[3/5] Installing Python Libraries...")
    # Upgrade pip/setuptools first so the build backend has pkg_resources available.
    run_cmd(
        f'"{env_python}" -m pip install --upgrade pip setuptools wheel --no-warn-script-location',
        log_file,
    )
    pip_str = " ".join(PIP_PACKAGES)
    run_cmd(
        f'"{env_python}" -m pip install {pip_str} --no-warn-script-location', log_file
    )

    # 4. Editable Install
    print("[4/5] Performing Editable Install of GeoFuse...")
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    run_cmd(
        f'"{env_python}" -m pip install -e "{root_dir}" --no-warn-script-location',
        log_file,
    )

    # 5. Verify Installation
    print("[5/5] Verifying Installation...")

    env_python = get_env_python(env_name)

    verify_imports = (
        "import geofuse; import torch; import platform; from mpi4py import MPI; "
    )
    verify_geofuse = "print(f'   [OK] GeoFuse Package: {geofuse.__file__}'); "
    verify_mpi = "print(f'   [OK] MPI Rank: {MPI.COMM_WORLD.Get_rank()} - Vendor: {MPI.get_vendor()}');"

    if system == "Darwin":
        verify_torch = "print(f'   [OK] PyTorch: {torch.__version__} - MPS Available: {torch.backends.mps.is_available()}'); "
    else:
        verify_torch = "print(f'   [OK] PyTorch: {torch.__version__} - CUDA Available: {torch.cuda.is_available()}'); "

    verify_script = verify_imports + verify_geofuse + verify_torch + verify_mpi

    run_cmd(f'"{env_python}" -c "{verify_script}"', log_file)


if __name__ == "__main__":
    log_file_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(ENV_NAME, log_file_arg)
