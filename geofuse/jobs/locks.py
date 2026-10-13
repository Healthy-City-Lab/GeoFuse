"""Resource slots shared by every GeoFuse process on one machine.

While a job runs it holds one slot of each resource class it uses — Earth
Engine for NDVI, the GPU and Street View for GVI, the fusion slot for fusion —
whichever process started it: the GUI's executor or the command line's
supervisor. Each slot is a lock file in a folder local to the machine (the
user's runtime folder on Windows, the temporary folder elsewhere, so the nodes
of a cluster never share one); the number of slots per class is the
``[limits]`` of the user configuration. The operating system releases a lock
when its holder dies, so a crashed job never blocks the next one.

Slots are taken in the job's own process, around the runner, and classes in a
fixed order so two jobs can never wait on each other.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path

#: Resource classes each job type holds while it runs.
RESOURCES: dict[str, tuple[str, ...]] = {
    "ndvi": ("earthengine",),
    "ndvi_column": ("earthengine",),
    "gvi": ("gpu", "streetview"),
    "gvi_column": ("gpu", "streetview"),
    "fusion": ("fusion",),
}

_WAITING = {
    "earthengine": "an Earth Engine slot",
    "gpu": "the GPU",
    "streetview": "the Street View slot",
    "fusion": "the fusion slot",
}


def lock_dir() -> Path:
    """This machine's lock folder for the current user."""
    if sys.platform == "win32":
        import platformdirs

        runtime = platformdirs.user_runtime_dir("geofuse", appauthor=False)
        return Path(runtime) / "locks"
    return Path(tempfile.gettempdir()) / f"geofuse-{os.getuid()}" / "locks"


def _limits() -> dict[str, int]:
    from geofuse import userconfig

    try:
        return userconfig.load().limits
    except userconfig.ConfigError:
        return dict(userconfig.LIMIT_DEFAULTS)


def _try_slot(folder: Path, cls: str, slots: int):
    from filelock import FileLock, Timeout

    for slot in range(max(1, slots)):
        lock = FileLock(str(folder / f"{cls}-{slot}.lock"))
        try:
            lock.acquire(timeout=0)
        except Timeout:
            continue
        return lock
    return None


@contextlib.contextmanager
def hold(
    job_type: str,
    *,
    cancelled: Callable[[], bool] = lambda: False,
    waiting: Callable[[str], None] = lambda text: None,
    tick: Callable[[], None] = lambda: None,
    poll_s: float = 1.0,
    folder: str | os.PathLike | None = None,
    limits: dict[str, int] | None = None,
) -> Iterator[bool]:
    """Hold ``job_type``'s resource slots for the ``with`` block.

    Waits for a free slot of each class, saying so once through ``waiting``
    and calling ``tick`` on every poll. Yields ``True`` once every slot is
    held, or ``False`` when ``cancelled`` turned true first (nothing is held
    then).
    """
    classes = sorted(RESOURCES.get(job_type, ()))
    folder = Path(folder) if folder is not None else lock_dir()
    limits = limits if limits is not None else _limits()
    if classes:
        folder.mkdir(parents=True, exist_ok=True)
    held = []
    try:
        for cls in classes:
            told = False
            while (lock := _try_slot(folder, cls, int(limits.get(cls, 1)))) is None:
                if cancelled():
                    yield False
                    return
                if not told:
                    waiting(
                        f"Waiting for {_WAITING.get(cls, cls)} "
                        "(another GeoFuse job is using it)..."
                    )
                    told = True
                tick()
                time.sleep(poll_s)
            held.append(lock)
        yield True
    finally:
        for lock in reversed(held):
            with contextlib.suppress(Exception):
                lock.release()
