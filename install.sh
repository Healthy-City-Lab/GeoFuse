#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- CONFIGURATION ---
ENV_NAME="geofuse"
PYTHON_VER="3.12.12"

echo "========================================================"
echo "        GeoFuse Installer (MPI-Enabled)"
echo "========================================================"

# --- STEP 1: AUTO-DETECT CONDA ---
CONDA_EXEC=""
CONDA_ROOT=""
PATHS=(
    "$HOME/miniforge3"
    "$HOME/anaconda3"
    "$HOME/miniconda3"
    "$HOME/opt/miniforge3"
    "$HOME/opt/anaconda3"
    "$HOME/opt/miniconda3"
    "/opt/miniforge3"
    "/opt/anaconda3"
    "/opt/miniconda3"
    "/opt/conda"
    "/usr/local/miniforge3"
    "/usr/local/anaconda3"
    "/usr/local/miniconda3"
    "/opt/homebrew/Caskroom/miniforge/base"
    "/usr/local/Caskroom/miniforge/base"
    "/opt/homebrew/anaconda3"
    "/Applications/anaconda3"
    "/Applications/miniforge3"
)

for TEST_PATH in "${PATHS[@]}"; do
    if [ -f "$TEST_PATH/bin/conda" ]; then
        CONDA_ROOT="$TEST_PATH"
        echo "[INFO] Found Conda at: $CONDA_ROOT"
        break
    fi
done

if [ -z "$CONDA_ROOT" ]; then
    if command -v conda &> /dev/null; then
        echo "[INFO] Found Conda in system PATH."
        CONDA_EXEC="conda"
        # Try to find CONDA_ROOT from conda info
        CONDA_ROOT=$(conda info --base 2>/dev/null)
    else
        read -p "Conda installation not found. Please paste the path to your Conda folder: " CONDA_ROOT
        if [ ! -f "$CONDA_ROOT/bin/conda" ]; then
            echo "[ERROR] Conda executable not found at specified path. Exiting."
            exit 1
        fi
        CONDA_EXEC="$CONDA_ROOT/bin/conda"
    fi
else
    CONDA_EXEC="$CONDA_ROOT/bin/conda"
fi

# Activate base environment (suppress output)
source "$CONDA_ROOT/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate base >/dev/null 2>&1 || true

# --- STEP 2: CREATE/RESET ENVIRONMENT ---
mkdir -p logs

if "$CONDA_EXEC" env list | grep -q -w "^$ENV_NAME "; then
    echo "[WARN] Environment '$ENV_NAME' already exists."
    read -p "Delete and clean install? (y/N): " DELETE
    if [[ "$DELETE" == "y" || "$DELETE" == "Y" ]]; then
        echo "[INFO] Removing existing environment..."
        "$CONDA_EXEC" remove -n "$ENV_NAME" --all -y >/dev/null 2>&1
    else
        echo "[INFO] Updating existing environment."
        conda activate "$ENV_NAME" >/dev/null 2>&1
        "$CONDA_ROOT/envs/$ENV_NAME/bin/python" scripts/setup_env.py
        echo ""
        echo "========================================================"
        echo "[SUCCESS] Installation complete."
        echo "[READY] To launch the CLI (example with 4 MPI processes):"
        echo "   conda activate $ENV_NAME"
        echo "   mpiexec -n 4 python scripts/cli.py --config config.csv"
        echo "========================================================"
        read -p "Press Enter to exit..."
        exit 0
    fi
fi

echo "[INFO] Creating Base Environment (Python $PYTHON_VER)..."
"$CONDA_EXEC" create -n "$ENV_NAME" python="$PYTHON_VER" -y >/dev/null 2>&1

echo "[INFO] Activating Environment..."
conda activate "$ENV_NAME" >/dev/null 2>&1

# --- STEP 3: RUN PYTHON SETUP SCRIPT ---
if [ -f "scripts/setup_env.py" ]; then
    "$CONDA_ROOT/envs/$ENV_NAME/bin/python" scripts/setup_env.py
else
    echo "[ERROR] scripts/setup_env.py not found!"
    exit 1
fi


echo ""
echo "========================================================"
echo "[SUCCESS] Installation complete."
echo "[READY] To launch the CLI (example with 4 MPI processes):"
echo "   conda activate $ENV_NAME"
echo "   mpiexec -n 4 python scripts/cli.py --config config.csv"
echo "========================================================"

# Only prompt for input if running interactively
if [ -t 0 ]; then
    read -p "Press Enter to exit..."
fi

exit 0
