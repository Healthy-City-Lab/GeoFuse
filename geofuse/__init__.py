import os
import sys

# ────────────────────────────────────────────────────────────────────
# WINDOWS FIXES: DLL Conflicts & OpenMP
# ────────────────────────────────────────────────────────────────────
if os.name == "nt":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Torch is imported by the modules that use it (``vision``, ``gvi``), not here.
# Importing it package-wide made every process pay for the CUDA DLLs — a
# 24-worker pool exhausted the Windows paging file — and the GDAL load-order
# crash it once guarded against (WinError 127) no longer reproduces on either
# import order. ``GEOFUSE_TORCH_PRELOAD=1`` restores the old behaviour if a
# machine still needs it.
if os.name == "nt" and os.environ.get("GEOFUSE_TORCH_PRELOAD") == "1":
    try:
        import torch  # noqa: F401
    except ImportError:
        pass


def is_ready():
    """Return True if CUDA is available (False if torch is missing)."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
