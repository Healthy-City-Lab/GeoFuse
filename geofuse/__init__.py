import os
import sys

# ────────────────────────────────────────────────────────────────────
# WINDOWS FIXES: DLL Conflicts & OpenMP
# ────────────────────────────────────────────────────────────────────
if os.name == "nt":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ────────────────────────────────────────────────────────────────────
# BLAS threads of command-line processes
# ────────────────────────────────────────────────────────────────────
# The ``geofuse`` command's own processes (status, watch, the supervisor that
# starts jobs, the remote agent) compute nothing heavy, but numpy and OpenBLAS
# left to size themselves for the host commit ~3 GB each as they load. Pinning
# them to one thread here, before anything imports numpy, holds each to a few
# tens of MB. Variables already set are kept; ``_BLAS_PINNED`` records what was
# set here, so jobs started from these processes get the host's defaults back
# (or their own count, see ``geofuse.parallel.job_thread_env``).
_BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "GDAL_NUM_THREADS",
)
_BLAS_PINNED: tuple[str, ...] = ()
_entry = os.path.basename(sys.argv[0] if sys.argv else "").lower()
if _entry == "-m" or _entry.startswith("geofuse"):
    _BLAS_PINNED = tuple(v for v in _BLAS_THREAD_VARS if v not in os.environ)
    for _var in _BLAS_PINNED:
        os.environ[_var] = "1"

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


class JobCancelled(BaseException):
    """Raised by an engine loop that saw the job's cancel flag.

    Derived from ``BaseException``, not ``Exception``, for the same reason
    ``KeyboardInterrupt`` is: the engine and runner wrap most work in
    ``except Exception`` blocks that log a warning and carry on with a
    degraded result. A cancel must travel through those untouched and reach
    the runner's own handler, which ends the run.
    """
