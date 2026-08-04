"""Worker-count sizing for the engine's parallel phases.

Pools size themselves from the host rather than from a constant, so the same
code fits a laptop and a compute node. ``GEOFUSE_WORKERS`` overrides everything
when a run has to share the machine.

Why the default is not "one thread per core". The phases that parallelise here
spend their time in numpy / BLAS, which is *already* multi-threaded inside each
call. Stacking a wide pool on top of that oversubscribes the cores and the
extra threads start costing more than they return. Measured on a 24-core host,
the fast-GLS search scorer over 240 trials:

    threads   2      4      8     12     16     23
    speedup  1.74   2.94   3.86   3.43   3.33   3.21

Throughput peaks near a third of the cores and decays after. ``worker_count``
therefore returns that third, which reproduces the previously hand-tuned value
of 8 on this host while still scaling with the hardware.

Note what does **not** belong on a thread pool: an exact ``statsmodels``
MixedLM refit is dominated by Python-level optimiser work holding the GIL, and
threading it measured 0.62x of serial. That path stays sequential.
"""

from __future__ import annotations

import os

# Share of the host's cores a pool may claim. Each task already spreads across
# several BLAS threads, so a pool this size is what saturates the machine —
# see the measurements in the module docstring.
_CORE_SHARE = 3

_ENV_OVERRIDE = "GEOFUSE_WORKERS"


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
