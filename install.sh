#!/bin/bash
# GeoFuse installer for Linux and macOS.
# Usage: ./install.sh [--yes] [--torch auto|cuda|cpu] [--env-name NAME]
# The options are passed to scripts/setup_env.py.

set -euo pipefail

ENV_NAME="geofuse"
for ((i = 1; i <= $#; i++)); do
    if [ "${!i}" = "--env-name" ]; then
        j=$((i + 1))
        ENV_NAME="${!j:-geofuse}"
    fi
done

cd "$(dirname "$0")"

echo "========================================================"
echo "        GeoFuse Installer"
echo "========================================================"

# --- STEP 1: FIND CONDA ---
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

if [ -n "${CONDA_EXE:-}" ] && [ -x "$CONDA_EXE" ]; then
    CONDA_ROOT="$(cd "$(dirname "$CONDA_EXE")/.." && pwd)"
    echo "[INFO] Found Conda from CONDA_EXE: $CONDA_ROOT"
elif command -v conda &> /dev/null; then
    CONDA_ROOT="$(conda info --base 2>/dev/null)"
    echo "[INFO] Found Conda on PATH: $CONDA_ROOT"
else
    for TEST_PATH in "${PATHS[@]}"; do
        if [ -f "$TEST_PATH/bin/conda" ]; then
            CONDA_ROOT="$TEST_PATH"
            echo "[INFO] Found Conda at: $CONDA_ROOT"
            break
        fi
    done
fi

if [ -z "$CONDA_ROOT" ]; then
    if [ ! -t 0 ]; then
        echo "[ERROR] Conda not found. Install Miniforge, or put conda on PATH."
        exit 1
    fi
    read -r -p "Conda installation not found. Paste the path to your Conda folder: " CONDA_ROOT
fi
if [ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    echo "[ERROR] No Conda installation at: $CONDA_ROOT"
    exit 1
fi

# Activate base so that conda, mamba and python resolve to this installation.
# Activation scripts may read unset variables, so -u is off around them.
set +u
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate base
set -u

# --- STEP 2: CREATE THE ENVIRONMENT AND INSTALL ---
mkdir -p logs
if ! python scripts/setup_env.py "$@"; then
    echo "[ERROR] Setup script failed. See logs/install.log."
    exit 1
fi

echo ""
echo "========================================================"
echo "[SUCCESS] Installation complete."
echo "[READY] In the environment, the command line is geofuse:"
echo "   conda activate $ENV_NAME"
echo "   geofuse doctor"
echo "   geofuse --help"
echo "========================================================"

# Only wait for a key when run from a terminal
if [ -t 0 ]; then
    read -r -p "Press Enter to exit..."
fi
