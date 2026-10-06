"""Worker sizing and process-backed fan-out for the engine's parallel phases.

Pools size themselves from the host rather than from a constant, so the same
code fits a laptop and a compute node. ``GEOFUSE_WORKERS`` overrides everything
when a run has to share the machine.

Threads or processes
--------------------
A phase belongs on a **thread** pool when each task spends its time inside a
single long numpy / BLAS call, because those release the GIL. It belongs on a
**process** pool when each task is a long chain of *small* numpy calls, because
every one of those holds the GIL for its Python-level part and the workers end
up queued for the interpreter instead of computing. Measured on a 24-core host
over the same greenery pre-aggregation batches:

    threads   x8       1.8x     (7% of the machine; the rest is GIL wait)
    procs     x8       8.4x
    procs     x16     12.2x
    procs     x22     11.7x

Thread sizing (:func:`worker_count`) is deliberately *not* one per core: each
task is already multi-threaded inside BLAS, so stacking a wide pool on top
oversubscribes the cores. Measured on the same host, the fast-GLS search
scorer over 240 trials:

    threads   2      4      8     12     16     23
    speedup  1.74   2.94   3.86   3.43   3.33   3.21

Throughput peaks near a third of the cores and decays after, so
:func:`worker_count` returns that third. Process workers are each pinned to a
single BLAS thread instead, so :func:`process_worker_count` claims two thirds
and peaks there.

Note what does **not** belong on a thread pool: an exact ``statsmodels``
MixedLM refit is dominated by Python-level optimiser work holding the GIL, and
threading it measured 0.62x of serial. That path stays sequential.

What a process pool costs
-------------------------
Cores are not the only budget: every worker is a fresh interpreter, and on
Windows the limit that actually bites is the system **commit** charge (RAM plus
page file), which a process reserves whether or not it ever touches it.

The dominant term is not the payload — it is what numpy and OpenBLAS reserve as
they load. Free to size themselves for a 24-core host they commit **~3.0 GB per
process**; pinned to one thread, **~0.11 GB**. So an unpinned pool of 16 asks
the operating system for ~48 GB before computing anything, and a 30 GB host
starts failing *5 MB* allocations machine-wide.

Two things follow, and both are handled here:

- The thread limits must be in the environment the worker is **created** with
  (:func:`_pinned_child_env`). Setting them from inside the worker is too late:
  it unpickles its own initializer — importing numpy in the process — before any
  of our code runs.
- Pool width is bounded by memory as well as by cores
  (:func:`memory_worker_cap`), including when ``GEOFUSE_WORKERS`` asks for more.
  A pool wide enough to exhaust the host takes the run down with it.

Using the process pool
----------------------
Everything expensive has to stay *out* of the pickle channel between parent and
worker — a pool that ships its inputs once per worker spends its speedup on
transfer:

- Publish bulk arrays with :func:`publish_array` and re-attach them in the
  worker with :func:`attach_array`. The array is written once and every worker
  memory-maps the same pages, so fan-out costs nothing per worker.
- Rebuild derived state (spatial indexes, file handles) inside the pool's
  ``initializer`` and keep it there for the pool's lifetime, rather than
  pickling it across.
- Address work by index range, so both the task message and its result stay
  small.

:func:`map_batches` drives such a pool with cancellation and per-batch
progress; :func:`worker_setup` is the initializer wrapper that pins each
worker's own BLAS threads and binds its lifetime to the parent's.

This module keeps numpy and GDAL out of its own import path on purpose: a
worker imports it *before* :func:`worker_setup` can set the thread-limit
environment variables, and the numeric libraries only read those once, as they
load.
"""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Callable, Iterable, Iterator
from typing import Any

# Share of the host's cores a pool may claim. Each task already spreads across
# several BLAS threads, so a pool this size is what saturates the machine —
# see the measurements in the module docstring.
_CORE_SHARE = 3

# Share of the host's cores a *process* pool may claim. Each worker is pinned
# to one BLAS thread, so this share stands alone rather than compounding with
# an inner pool. Tunable via ``GEOFUSE_CPU_SHARE`` (0 < share <= 1); the default
# leaves a fifth of the machine responsive while a job runs.
_PROCESS_CORE_SHARE = 0.8

_ENV_OVERRIDE = "GEOFUSE_WORKERS"
_ENV_CPU_SHARE = "GEOFUSE_CPU_SHARE"


def cpu_share() -> float:
    """Fraction of cores a process pool may claim, clamped to (0, 1]."""
    raw = os.environ.get(_ENV_CPU_SHARE, "").strip()
    if raw:
        try:
            return min(1.0, max(0.01, float(raw)))
        except ValueError:
            pass
    return _PROCESS_CORE_SHARE


def cpu_budget() -> int:
    """Logical cores this process may plan around (never below 1)."""
    return max(1, os.cpu_count() or 1)


def worker_count(*, cap: int | None = None) -> int:
    """Threads to run a parallel phase with on this host.

    ``cap`` bounds a phase that cannot use more. ``GEOFUSE_WORKERS`` overrides
    the host-derived figure, but a cap still applies — a phase never opens more
    threads than it has independent work for.
    """
    override = os.environ.get(_ENV_OVERRIDE, "").strip()
    n = 0
    if override:
        try:
            n = int(override)
        except ValueError:
            n = 0
    if n <= 0:
        n = max(2, cpu_budget() // _CORE_SHARE) if cpu_budget() > 2 else 1
    return min(n, cap) if cap else n


def workers_for(n_units: int, *, cap: int | None = None) -> int:
    """Pool size for ``n_units`` independent work items.

    Opening more threads than there are units costs setup and returns nothing,
    so the pool never exceeds the work available.
    """
    if n_units <= 1:
        return 1
    return max(1, min(worker_count(cap=cap), int(n_units)))


# What to budget for one pinned worker. Measured on the pre-aggregation pool:
# 0.11 GB of commit for the interpreter and its numeric stack, 0.19 GB resident
# once a BallTree over 1.7M metric points is built. This allows ~2.5x that, to
# cover a denser metric layer and the phase's working arrays. Deliberately
# generous — overestimating costs a few workers, underestimating takes the whole
# machine down.
_WORKER_COMMIT_BYTES = 512 * 1024 * 1024

# Share of the free commit a pool may take. The rest stays for the parent, which
# is holding the job's data while the pool runs, and for everything else on the
# machine.
_COMMIT_SHARE = 0.6


def available_memory_bytes() -> int:
    """Memory a *new process* can actually reserve on this host.

    On Windows the binding limit is the system **commit** limit (RAM plus page
    file), not free RAM: an allocation fails once processes have collectively
    promised more than that, however little of it they have touched. Elsewhere,
    available RAM is the right figure. Returns ``0`` when it cannot be read, so
    callers fall back to sizing on cores alone.
    """
    if os.name == "nt":
        try:
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return 0
            return int(status.ullAvailPageFile)
        except Exception:
            return 0
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return 0


def memory_worker_cap(bytes_per_worker: int = _WORKER_COMMIT_BYTES) -> int | None:
    """How many workers this host currently has the memory to reserve.

    ``None`` when the figure cannot be read and the caller should size on cores
    alone.
    """
    free = available_memory_bytes()
    if free <= 0 or bytes_per_worker <= 0:
        return None
    return max(1, int(free * _COMMIT_SHARE) // int(bytes_per_worker))


def process_worker_count(
    n_units: int | None = None,
    *,
    cap: int | None = None,
    bytes_per_worker: int = _WORKER_COMMIT_BYTES,
) -> int:
    """Processes to run a GIL-bound phase with on this host.

    Wider than :func:`worker_count` because each worker is pinned to one BLAS
    thread (see :func:`_pinned_child_env`), so the shares don't compound.
    ``n_units`` caps the pool at the work available.

    The result is bounded by memory as well as by cores, because a process pool
    spends both: every worker reserves an interpreter and its own copy of any
    index the phase rebuilds. The memory bound applies to ``GEOFUSE_WORKERS``
    too — a pool wide enough to exhaust the host takes the run down with it,
    which is worse than honouring the override.
    """
    override = os.environ.get(_ENV_OVERRIDE, "").strip()
    n = 0
    if override:
        try:
            n = int(override)
        except ValueError:
            n = 0
    if n <= 0:
        n = max(2, int(cpu_budget() * cpu_share()))
    if cap:
        n = min(n, cap)
    by_memory = memory_worker_cap(bytes_per_worker)
    if by_memory is not None:
        n = min(n, by_memory)
    if n_units is not None:
        n = min(n, max(1, int(n_units)))
    return max(1, n)


# ────────────────────────────────────────────────────────────────────
# Bulk data hand-off
# ────────────────────────────────────────────────────────────────────


def publish_array(directory: str, name: str, array: Any) -> str:
    """Write ``array`` where every worker can memory-map it, and return the path.

    The counterpart of :func:`attach_array`. One write in the parent replaces
    one pickled copy per worker, and because the workers map the same file the
    operating system keeps a single set of pages behind all of them.
    """
    import numpy as np

    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.npy")
    np.save(path, np.ascontiguousarray(array))
    return path


def attach_array(path: str) -> Any:
    """Memory-map an array published by :func:`publish_array`, read-only."""
    import numpy as np

    return np.load(path, mmap_mode="r")


# ────────────────────────────────────────────────────────────────────
# Process pool
# ────────────────────────────────────────────────────────────────────

# Every worker must have these set before its first numeric import. One pool
# slot is already one unit of parallelism, so letting each worker's BLAS /
# OpenMP open a pool of its own multiplies threads by cores and oversubscribes
# the machine — but the reason this matters most is memory. Left free to size
# themselves for a 24-core host, numpy and OpenBLAS reserve **~3.0 GB of
# Windows commit charge per process**; pinned to one thread they reserve
# ~0.11 GB. The reservation counts against the system commit limit even though
# it is never touched, so an unpinned pool of 16 asks the OS for ~48 GB before
# it computes anything, and allocations start failing machine-wide.
_PINNED_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "GDAL_NUM_THREADS",
)


@contextlib.contextmanager
def _pinned_child_env() -> Iterator[None]:
    """Hold the thread limits in this process's environment while a pool runs.

    Spawned workers inherit the environment as they are created, and that is
    the only point early enough to matter: a worker unpickles its initializer —
    which imports numpy in the process — before any code of ours gets to run,
    so setting the variables from inside the worker is already too late to
    change what BLAS reserves. The parent's own numpy was initialised long ago
    and re-reads nothing, so setting them here does not affect it.
    """
    previous = {var: os.environ.get(var) for var in _PINNED_THREAD_VARS}
    for var in _PINNED_THREAD_VARS:
        os.environ.setdefault(var, "1")
    try:
        yield
    finally:
        for var, value in previous.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value


def worker_setup(
    initializer: Callable[..., None] | None = None, *initargs: Any
) -> None:
    """Pool initializer: pin this worker's threads, then run ``initializer``.

    The limits are normally inherited from the parent (see
    :func:`_pinned_child_env`, which is what actually bounds what BLAS
    reserves); setting them again here covers a worker started some other way.
    Also binds the worker's lifetime to its parent's, so a pool whose owner is
    force-killed doesn't leave a machine's worth of workers resident.
    """
    for var in _PINNED_THREAD_VARS:
        os.environ.setdefault(var, "1")
    try:
        from geofuse.jobs.subprocess_runner import exit_with_parent

        exit_with_parent()
    except Exception:
        pass
    if initializer is not None:
        initializer(*initargs)


@contextlib.contextmanager
def _spawnable_main() -> Iterator[None]:
    """Hide a ``__main__`` whose script no longer exists from the spawn bootstrap.

    Before a spawn worker runs anything of ours it re-executes the parent's main
    script, so a parent that has repointed ``__main__`` at a scratch file — as
    Streamlit's script runner does, and notebook shims too — kills every worker
    on import once that file is cleaned up. Dropping the attribute makes the
    child skip the step entirely, which is the right answer either way: a main
    script that is no longer on disk was never going to re-execute.
    """
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    if (
        main is None
        or path is None
        or getattr(main, "__spec__", None) is not None
        or os.path.exists(path)
    ):
        yield
        return
    del main.__file__
    try:
        yield
    finally:
        main.__file__ = path


def map_batches(
    fn: Callable[..., Any],
    tasks: Iterable[tuple],
    *,
    workers: int,
    initializer: Callable[..., None] | None = None,
    initargs: tuple = (),
    on_result: Callable[[Any], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    poll_s: float = 0.25,
) -> bool:
    """Run ``fn(*task)`` for every task across ``workers`` processes.

    Results are handed to ``on_result`` on the calling process in completion
    order. Returns ``True`` when every task finished and ``False`` if
    ``cancel_check`` went truthy first — in which case queued tasks are dropped
    and the pool is torn down, so a cancel lands within one batch rather than
    one phase.

    Only ``workers * 2`` tasks are ever in flight, which bounds how much
    unconsumed output the pool can accumulate ahead of ``on_result``.
    """
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    pending = iter(list(tasks))
    if workers <= 1:
        if initializer is not None:
            initializer(*initargs)
        for task in pending:
            if cancel_check is not None and cancel_check():
                return False
            result = fn(*task)
            if on_result is not None:
                on_result(result)
        return True

    import multiprocessing as mp

    cancelled = False
    # Workers are spawned lazily as tasks are submitted, so both guards have to
    # cover the whole run rather than just the executor's construction.
    with _spawnable_main(), _pinned_child_env():
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=worker_setup,
            initargs=(initializer, *initargs),
        )
        try:
            in_flight = set()
            for task in pending:
                in_flight.add(executor.submit(fn, *task))
                if len(in_flight) >= 2 * workers:
                    break
            while in_flight:
                if cancel_check is not None and cancel_check():
                    cancelled = True
                    for fut in in_flight:
                        fut.cancel()
                    break
                done, in_flight = wait(
                    in_flight, timeout=poll_s, return_when=FIRST_COMPLETED
                )
                for fut in done:
                    if on_result is not None:
                        on_result(fut.result())
                    task = next(pending, None)
                    if task is not None:
                        in_flight.add(executor.submit(fn, *task))
        finally:
            # ``wait=True`` lets each worker finish the batch it is inside — a
            # sub-second wait — which is what keeps a cancelled pool from
            # leaving workers behind.
            executor.shutdown(wait=True, cancel_futures=True)
    return not cancelled
