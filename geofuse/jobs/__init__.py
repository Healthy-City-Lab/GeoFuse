"""Worker callables that run inside the JobExecutor thread pool.

Modules in this package must not import ``streamlit`` — they receive every
piece of state they need (datasets, caches, engine instances) as plain
arguments from the executor.
"""


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
