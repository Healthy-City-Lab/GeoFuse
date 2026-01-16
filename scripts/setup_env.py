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
    # Utility
    "isort==7.0.0",
]

GIT_PACKAGES = [
    "git+https://github.com/robolyst/streetview.git@b07d69445161bc193a1e1a6aa5e098b6fcd6eef8"
]


def run_cmd(command, log_file=None, echo_to_console=False):
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

        output_lines = []
        while True:
            line = process.stdout.readline()
            if not line:
                break
            output_lines.append(line)
            if log_file:
                # Write to log file
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(line)
                # Also print to console if echo_to_console is True
                if echo_to_console:
                    print(line, end="")
            else:
                # Print to console only if no log file
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
        f"conda env list", shell=True, capture_output=True, text=True
    )
    env_exists = env_name in env_check.stdout

    if env_exists:
        print(f"[WARN] Environment '{env_name}' already exists. Removing...")
        subprocess.run(f"conda env remove -n {env_name} -y", shell=True, check=True)

    print(f"[INFO] Creating environment '{env_name}' with Python 3.12...")
    subprocess.run(
        f"conda create -n {env_name} python=3.12 -y -c conda-forge",
        shell=True,
        check=True,
    )
    print(f"[SUCCESS] Environment created successfully\n")

    # 1. Prepare Conda List (Core + OS-Specific MPI)
    mpi_pkgs = get_mpi_packages(system)
    full_conda_list = CONDA_PACKAGES + mpi_pkgs
    conda_str = " ".join(full_conda_list)

    print(f"[1/6] Installing Core & MPI Binaries ({solver})...")
    # Conda handles the binary linking for mpi4py automatically here
    run_cmd(f"{solver} install -n {env_name} -y -c conda-forge {conda_str}", log_file)

    # 2. Install PyTorch
    print(f"[2/6] Installing PyTorch Acceleration...")
    if system == "Windows" or system == "Linux":
        cmd = f"{solver} install -n {env_name} -y {PYTORCH_VERSION} pytorch-cuda=12.1 -c pytorch -c nvidia"
    else:
        # macOS uses CPU or MPS (Metal Performance Shaders), no CUDA
        # Use the standard command, ensuring the pytorch channel is primary.
        cmd = f"{solver} install -n {env_name} -y {PYTORCH_VERSION} -c pytorch"
    run_cmd(cmd, log_file)

    # 3. Install Pip Libraries
    print(f"[3/6] Installing Python Libraries...")
    env_python = get_env_python(env_name)
    pip_str = " ".join(PIP_PACKAGES)
    run_cmd(f'"{env_python}" -m pip install {pip_str}', log_file)

    # 4. Install Git Packages
    print(f"[4/6] Installing Custom Git Packages...")
    for git_url in GIT_PACKAGES:
        run_cmd(f'"{env_python}" -m pip install {git_url}', log_file)

    # 5. Editable Install
    print(f"[5/6] Performing Editable Install of GeoFuse...")
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    run_cmd(f'"{env_python}" -m pip install -e "{root_dir}"', log_file)

    # 6. Verify Installation
    print(f"[6/6] Verifying Installation...")

    # Get the Python executable from the geofuse environment
    env_python = get_env_python(env_name)

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

    run_cmd(f'"{env_python}" -c "{verify_script}"', log_file, echo_to_console=True)


if __name__ == "__main__":
    log_file_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(ENV_NAME, log_file_arg)
