"""Worker callables that run inside the JobExecutor thread pool.

Modules in this package must not import ``streamlit`` — they receive every
piece of state they need (datasets, caches, engine instances) as plain
arguments from the executor.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar("T")


# ────────────────────────────────────────────────────────────────────
# Per-engine progress cadence
# ────────────────────────────────────────────────────────────────────
#
# Each engine emits progress at its own rhythm so the JobStore lock and the
# parent event queue aren't slammed on million-item runs, while the UI still
# feels live. The backend cadence (:class:`ProgressThrottle`) and the UI
# fragment refresh (:func:`job_ui_refresh_s`) read the same constants so the
# two never drift apart.

# GVI: emit only once BOTH thresholds are met — the slower of "500 images" and
# "15 seconds" governs. Bursts of cheap misses can't flood the store, and a
# genuinely slow stretch still updates every 500 images.
GVI_PROGRESS_MIN_ITEMS = 500
GVI_PROGRESS_MIN_SECONDS = 15.0

# NDVI: emit every 2 downloaded tiles. Purely count-based — tiles are coarse
# and comparatively slow, so there's no need for a time floor.
NDVI_PROGRESS_MIN_TILES = 2

# UI fragment refresh (seconds) per job type. The monitor renders every job in
# one fragment, so the active jobs' fastest requirement sets the shared rate
# (see :func:`job_ui_refresh_s`). GVI matches its 15 s backend floor; NDVI and
# fusion keep the responsive default.
_DEFAULT_UI_REFRESH_S = 2.0
_UI_REFRESH_S: dict[str, float] = {
    "gvi": GVI_PROGRESS_MIN_SECONDS,
    "gvi_column": GVI_PROGRESS_MIN_SECONDS,
    "ndvi": _DEFAULT_UI_REFRESH_S,
    "ndvi_column": _DEFAULT_UI_REFRESH_S,
    "fusion": _DEFAULT_UI_REFRESH_S,
}


def job_ui_refresh_s(active_types: Iterable[str]) -> float:
    """Fragment refresh interval (s) for a set of in-flight job types.

    The job monitor renders all jobs in a single fragment, so it must refresh
    fast enough for the most demanding active engine: the result is the
    minimum per-type interval across ``active_types``. With no active jobs (or
    only unknown types) it falls back to the default so a freshly-submitted job
    still appears promptly.
    """
    intervals = [
        _UI_REFRESH_S.get(t, _DEFAULT_UI_REFRESH_S) for t in active_types
    ]
    return min(intervals) if intervals else _DEFAULT_UI_REFRESH_S


class ProgressThrottle:
    """Rate-limit progress emission by item count and/or elapsed time.

    ``should_emit(count)`` returns ``True`` only when **both** thresholds are
    satisfied since the last emit — i.e. the slower of the two governs. Set
    ``min_seconds=0`` for a purely count-based cadence, or ``min_items=1`` for
    a purely time-based one. A ``final=True`` call always emits (and resets),
    so callers can force the terminal 100 % tick.

    One instance is stateful and not thread-safe; create it per run and call it
    from a single progress thread.
    """

    def __init__(self, min_items: int = 1, min_seconds: float = 0.0) -> None:
        self._min_items = max(1, int(min_items))
        self._min_seconds = float(min_seconds)
        self._last_count = 0
        self._last_t = time.monotonic()

    def should_emit(self, count: int, *, final: bool = False) -> bool:
        now = time.monotonic()
        if final:
            self._last_count = count
            self._last_t = now
            return True
        items_ok = (count - self._last_count) >= self._min_items
        time_ok = (now - self._last_t) >= self._min_seconds
        if items_ok and time_ok:
            self._last_count = count
            self._last_t = now
            return True
        return False


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 2.0,
    factor: float = 1.5,
    jitter: float = 0.25,
    cancel_callback: Callable[[], bool] | None = None,
    log_fn: Callable[[str, str], None] | None = None,
    label: str = "operation",
    non_retryable: tuple[type[BaseException], ...] = (),
) -> T:
    """Invoke ``fn()`` with up to ``attempts`` tries and jittered backoff.

    Returns the function's result on success; re-raises the last exception
    after the final failed attempt. Sleeps between attempts with cancel-
    aware polling (~quarter-second granularity); a fired ``cancel_callback``
    re-raises the last exception immediately rather than waiting out the
    remaining backoff.

    ``log_fn(level, msg)`` is the engine logger callable; on a failed
    attempt other than the last, logs a WARN with the attempt count and the
    next delay. Final-attempt failure is left for the caller to message —
    this helper just re-raises so retry policy is decoupled from error
    surfacing.

    ``non_retryable`` names exception types that are deterministic rather
    than transient (e.g. an Earth Engine "memory limit exceeded" rejection
    that will fail identically every attempt): they re-raise immediately so
    the caller can take a different path instead of burning the retry budget.

    Used by NDVI per-tile downloads (flaky EE / network) and is a good fit
    for any other network-bound worker that wants the same "transient
    failure shouldn't sink the batch" semantic.
    """
    last_exc: BaseException | None = None
    delay = float(base_delay)
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if non_retryable and isinstance(exc, non_retryable):
                raise
            if attempt >= attempts:
                break
            # ±jitter so concurrent workers don't retry in lockstep and
            # restampede the upstream service.
            jittered = delay * (1.0 + random.uniform(-jitter, jitter))
            if log_fn is not None:
                log_fn(
                    "WARN",
                    f"{label} failed (attempt {attempt}/{attempts}): "
                    f"{type(exc).__name__}: {exc} — retrying in "
                    f"{jittered:.1f}s",
                )
            slept = 0.0
            while slept < jittered:
                if cancel_callback and cancel_callback():
                    break
                step = min(0.25, jittered - slept)
                time.sleep(step)
                slept += step
            if cancel_callback and cancel_callback():
                break
            delay *= factor
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{label} failed but no exception captured.")
