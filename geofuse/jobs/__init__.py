"""Worker callables that run inside the JobExecutor thread pool.

Modules in this package must not import ``streamlit`` — they receive every
piece of state they need (datasets, caches, engine instances) as plain
arguments from the executor.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def progress_interval_s(total: int) -> float:
    """Heartbeat cadence (seconds) scaled to job size.

    Long-running engines and runners use this to throttle progress callbacks
    so the ``JobStore`` lock stays cheap on big jobs (and the UI doesn't get
    flooded with status updates). Shape: 2 s up to ~200 items, then +1 s per
    100 items, capped at 10 s. ``total`` is the aggregate work count —
    panoramas for GVI, tiles for NDVI, anything iterable.

    Callers should always emit the final tick regardless of the interval so
    the bar reaches 100 %.
    """
    return float(min(10, max(2, total // 100)))


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
