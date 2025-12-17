import sys
import platform
import subprocess
import os
import shutil


# ==============================================================================
# HELPERS
# ==============================================================================
def run_cmd(command):
    try:
        print(f"[EXEC] {command}")
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError:
        print(f"[ERROR] Command failed: {command}")
        sys.exit(1)


def get_solver_executable():
    """
    Checks if 'mamba' is available. Returns the executable name.
    """
    if shutil.which("mamba"):
        return "mamba"
    return "conda"


# ==============================================================================
# MAIN LOGIC
# ==============================================================================
def main():
    system = platform.system()
    solver = get_solver_executable()

    # CONSOLIDATED LOGGING: One line, no clutter.
    print(f"[INFO] Setup Context -> OS: {system} | Solver: {solver}")

    # ------------------------------------------------------------------
    # 1. Install HEAVY Dependencies via Conda/Mamba
    # ------------------------------------------------------------------

    # Packages to install via Conda (Shared list to avoid typos)
    conda_packages = "geopandas matplotlib scikit-learn tqdm"

    # [FIXED] Windows Command: Uses PyTorch 2.4.1 (Stable) + CUDA 12.1
    # This combination avoids the 'missing package' errors on the Windows Conda channel.
    windows_torch_cmd = f"{solver} install -y pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia"

    if system == "Linux":
        print(f"[INFO] Installing Linux Full Stack via {solver}...")
        run_cmd(
            f"{solver} install -y -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=12.1 {conda_packages}"
        )

    elif system == "Windows":
        # A. Install Geospatial & Data Science Core (Conda-Forge)
        print(f"[INFO] Installing Geospatial/Data Core via {solver}...")
        run_cmd(f"{solver} install -y -c conda-forge {conda_packages}")

        # B. Install PyTorch (Official Channel)
        print(f"[INFO] Installing Custom PyTorch via {solver}...")
        run_cmd(windows_torch_cmd)

    else:
        # Mac/Other fallback
        print(f"[INFO] Installing Standard Stack via {solver}...")
        run_cmd(
            f"{solver} install -y -c pytorch -c conda-forge pytorch torchvision {conda_packages}"
        )

    # ------------------------------------------------------------------
    # 2. Install Remaining Requirements via Pip
    # ------------------------------------------------------------------
    print("[INFO] Installing remaining pure-Python requirements via Pip...")
    req_path = os.path.join(os.path.dirname(__file__), "..", "requirements.txt")

    # CRITICAL: Exclude anything we just installed via Conda
    excluded = [
        "numpy",
        "pandas",
        "geopandas",
        "rasterio",
        "shapely",
        "fiona",
        "pyproj",
        "torch",
        "torchvision",
        "torchaudio",
        "matplotlib",
        "scikit-learn",
        "tqdm",
    ]

    temp_reqs = "temp_requirements.txt"

    if os.path.exists(req_path):
        with open(req_path, "r") as f_in, open(temp_reqs, "w") as f_out:
            for line in f_in:
                # Clean line to check package name
                pkg = line.split("=")[0].split("<")[0].split(">")[0].strip()
                if pkg.lower() not in excluded:
                    f_out.write(line)

        try:
            # Install the filtered list
            run_cmd(f'"{sys.executable}" -m pip install -r {temp_reqs}')
        finally:
            if os.path.exists(temp_reqs):
                os.remove(temp_reqs)
    else:
        print(f"[WARN] requirements.txt not found at {req_path}")

    # ------------------------------------------------------------------
    # 3. Install Custom Packages (Pip Only)
    # ------------------------------------------------------------------
    print("[INFO] Installing Streetview & GeoFuse...")
    try:
        run_cmd(
            f'"{sys.executable}" -m pip install git+https://github.com/robolyst/streetview'
        )
    except Exception as e:
        print(f"[WARN] Failed to install streetview package: {e}")

    # Install the local package in editable mode
    root_dir = os.path.join(os.path.dirname(__file__), "..")
    run_cmd(f'"{sys.executable}" -m pip install -e "{root_dir}"')

    print("\n[SUCCESS] Environment setup complete!")
    print("Verification:")

    # Verify Import and CUDA
    verify_cmd = (
        f'"{sys.executable}" -c "'
        "import torch; print(f'Torch: {torch.__version__}'); "
        "print(f'CUDA: {torch.cuda.is_available()}'); "
        "import geopandas; print(f'GeoPandas: {geopandas.__version__}')\""
    )
    run_cmd(verify_cmd)


if __name__ == "__main__":
    main()
