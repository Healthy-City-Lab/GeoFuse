import os
import sys

# -----------------------------------------------------------------------------
# WINDOWS FIXES: DLL Conflicts & OpenMP
# -----------------------------------------------------------------------------
if os.name == "nt":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    try:
        import torch  # noqa: F401 -- load before GeoPandas/GDAL on Windows (WinError 127)
    except ImportError:
        pass


def is_ready():
    """Return True if CUDA is available (False if torch is missing)."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
