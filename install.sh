#!/bin/bash

# --- CONFIGURATION ---
ENV_NAME="geofuse"
PYTHON_VER="3.12.12"

echo "========================================================"
echo "        GeoFuse Installer (MPI-Enabled)"
echo "========================================================"

# --- STEP 1: AUTO-DETECT CONDA ---
CONDA_ROOT=""
PATHS=(
    "$HOME/miniforge3"
    "$HOME/anaconda3"
    "$HOME/miniconda3"
    "/opt/miniforge3"
    "/opt/anaconda3"
)

for TEST_PATH in "${PATHS[@]}"; do
    if [ -f "$TEST_PATH/bin/activate" ]; then
        CONDA_ROOT="$TEST_PATH"
        echo "[INFO] Found Conda at: $CONDA_ROOT"
        break
    fi
done

if [ -z "$CONDA_ROOT" ]; then
    read -p "Conda installation not found. Please paste the path to your Conda folder: " CONDA_ROOT
fi

# Initialize Conda for this script's session
# The `conda shell.bash hook` ensures `conda activate` works correctly within the script.
eval "$("$CONDA_ROOT/bin/conda" 'shell.bash' 'hook' 2> /dev/null)"

# --- STEP 2: CREATE/RESET ENVIRONMENT ---
# Check if the environment already exists
if conda env list | grep -q -w "^$ENV_NAME "; then
    echo "[WARN] Environment '$ENV_NAME' already exists."
    read -p "Delete and clean install? (y/N): " DELETE
    if [[ "$DELETE" == "y" || "$DELETE" == "Y" ]]; then
        echo "[INFO] Removing existing environment..."
        conda remove -n "$ENV_NAME" --all -y
    else
        # If user chooses not to delete, we will just activate and run the script
        # This allows for partial installations or updates.
        goto="ACTIVATE" 
    fi
fi

# This check is necessary in case the user decided to keep the existing environment.
# A more structured way to do this without goto.
if [ "$goto" != "ACTIVATE" ]; then
    echo "[INFO] Creating Base Environment (Python $PYTHON_VER)..."
    conda create -n "$ENV_NAME" python="$PYTHON_VER" -y
fi

# --- STEP 3: ACTIVATE ENVIRONMENT ---
:ACTIVATE # Label for the "goto" logic
echo "[INFO] Activating Environment..."
conda activate "$ENV_NAME"

# --- STEP 4: RUN PYTHON SETUP SCRIPT ---
if [ -f "scripts/setup_env.py" ]; then
    python scripts/setup_env.py
else
    echo "[ERROR] scripts/setup_env.py not found!"
    read -p "Press Enter to exit..."
    exit 1
fi

echo ""
echo "========================================================"
echo "[READY] To launch the CLI (example with 4 MPI processes):"
echo "   conda activate $ENV_NAME"
echo "   mpiexec -n 4 python scripts/cli.py --config config.csv"
echo "========================================================"
read -p "Installation complete. Press Enter to exit..."
