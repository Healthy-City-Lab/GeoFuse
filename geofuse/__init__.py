import os
import sys

# -----------------------------------------------------------------------------
# WINDOWS FIXES: DLL Conflicts & OpenMP
# -----------------------------------------------------------------------------
if os.name == "nt":
    # 1. Suppress MKL/OpenMP conflict
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # 2. FORCE TORCH LOAD BEFORE GEOPANDAS/GDAL
    # This prevents [WinError 127] shm.dll errors by ensuring Torch claims
    # the necessary C++ runtimes first.
    try:
        import torch
    except ImportError:
        pass


# Helper to check package readiness
def is_ready():
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
