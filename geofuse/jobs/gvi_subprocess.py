"""Subprocess-based runner for GVI jobs (work in progress).

This is the **start of a major refactor**: GVI workers move out of the
Streamlit Python process so they no longer compete for the GIL with the
job-monitor fragment, the WebSocket I/O thread, and the result-inspector
renderer. Measurements showed a ~37-percentage-point GPU utilisation drop
when the Streamlit tab was foreground vs hidden — almost entirely due to
this in-process GIL contention. NDVI and Fusion runners remain in-thread
for now.

This module is intentionally Streamlit-free so it can be imported in a
``multiprocessing.spawn``-launched child without dragging in the UI stack.

Message protocol
----------------
The child pushes tuples onto a ``multiprocessing.Queue`` provided by the
parent. The first element is a string tag from the ``MSG_*`` constants:

    (MSG_PROGRESS,  value: float | None, status_text: str | None, extras: dict)
    (MSG_HEARTBEAT,)
    (MSG_SET_EXTRA, extras: dict)
    (MSG_LOG,       colored_line: str,  plain_line: str)
    (MSG_COMPLETE,  output_paths: list[str])
    (MSG_ERROR,     short_msg: str,     traceback_text: str)

Parent-side handling (the queue reader thread that materialises these
events back into JobStore + log buffers) lives in a follow-up commit.
"""

from __future__ import annotations

import threading
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # only for type hints — runtime import would tie this module
    import multiprocessing as _mp  # to multiprocessing's spawn path unnecessarily

# Message tags (kept as short strings for legibility in queue dumps / logs)
MSG_PROGRESS = "progress"
MSG_HEARTBEAT = "heartbeat"
MSG_SET_EXTRA = "set_extra"
MSG_LOG = "log"
MSG_COMPLETE = "complete"
MSG_ERROR = "error"


class SubprocJobContext:
    """Child-process stand-in for :class:`geofuse.persistence.job_executor.JobContext`.

    The runner code in :mod:`geofuse.jobs.runners` doesn't care whether it's
    talking to a real ``JobContext`` (parent process) or this one (child
    process); both expose ``progress`` / ``heartbeat`` / ``set_extra`` /
    ``is_cancelled``. The only difference is that this implementation pushes
    every observable event onto a ``multiprocessing.Queue`` so a reader
    thread on the parent side can apply them to the real ``JobStore``.
    """

    def __init__(self, job_id, event_queue, cancel_event) -> None:
        self.job_id = job_id
        self._queue = event_queue
        self._cancel_event = cancel_event

    def progress(
        self, value: float | None = None, status_text: str | None = None, **extras
    ) -> None:
        self._queue.put((MSG_PROGRESS, value, status_text, dict(extras)))

    def heartbeat(self) -> None:
        self._queue.put((MSG_HEARTBEAT,))

    def set_extra(self, **extras) -> None:
        self._queue.put((MSG_SET_EXTRA, dict(extras)))

    def is_cancelled(self) -> bool:
        return bool(self._cancel_event.is_set())


def _route_engine_logging_to_queue(event_queue) -> None:
    """Replace :func:`geofuse.logger.get_logger` so each ``_log`` call pushes
    formatted lines onto the parent-side queue.

    Engines call ``_log = get_logger("GVI")`` at module import time, so they
    capture the *original* closure. We monkey-patch on the geofuse.logger
    module object directly to redirect future calls; existing closures are
    rebound via the module symbol below if needed.
    """
    from geofuse import logger as _logger

    ANSI = _logger._ANSI
    ANSI_RE = _logger._ANSI_ESCAPE_RE

    def _patched(engine: str):
        tag = engine.upper()

        def log(level: str, msg: str) -> None:
            color = ANSI.get(level, "")
            colored = f"{color}{ANSI['BOLD']}[{tag} {level}]{ANSI['RESET']} {msg}"
            plain = ANSI_RE.sub("", f"[{tag} {level}] {msg}")
            try:
                event_queue.put((MSG_LOG, colored, plain))
            except Exception:
                # Best-effort: never crash the worker on a logging hiccup.
                pass

        return log

    _logger.get_logger = _patched


def run_gvi_child(
    job_id: str,
    fname: str,
    dataset_data: dict,
    init_args: dict,
    run_args: dict,
    output_dir: str,
    save_geotiff: bool,
    save_geojson: bool,
    save_gpkg: bool,
    pano_cache_db_path: str,
    event_queue,
    cancel_event,
) -> None:
    """Entry point invoked by ``multiprocessing.Process(target=...)``.

    Imports heavy dependencies (torch, gvi engine) lazily so module-level
    import of :mod:`geofuse.jobs.gvi_subprocess` stays cheap. Runs the
    standard :func:`geofuse.jobs.runners.run_gvi` against a
    :class:`SubprocJobContext` so all progress / log / completion events
    flow through ``event_queue``. Errors are caught and turned into
    ``MSG_ERROR`` messages so the parent can apply them to the JobStore.
    """
    try:
        # Route engine log lines into the parent queue *before* the runner
        # imports anything that might cache a logger closure.
        _route_engine_logging_to_queue(event_queue)

        # Lazy: keep these out of module import path. The child re-imports
        # the modules under spawn anyway; doing it here makes failures more
        # local and lets us send them up as MSG_ERROR.
        from geofuse.jobs.runners import run_gvi
        from geofuse.persistence.pano_cache import PanoCache

        # The parent's PanoCache instance can't cross process boundaries, but
        # the SQLite file behind it can. Open a fresh cache on the same path
        # — WAL mode handles cross-process concurrency safely.
        pano_cache = PanoCache(pano_cache_db_path)
        dataset_data["cache_ref"] = pano_cache

        ctx = SubprocJobContext(job_id, event_queue, cancel_event)
        gpu_lock = threading.Lock()  # per-process; the parent's lock doesn't apply here

        result = run_gvi(
            ctx,
            fname=fname,
            dataset_data=dataset_data,
            init_args=init_args,
            run_args=run_args,
            output_dir=output_dir,
            save_geotiff=save_geotiff,
            save_geojson=save_geojson,
            save_gpkg=save_gpkg,
            gpu_lock=gpu_lock,
        )

        event_queue.put((MSG_COMPLETE, list(result.get("output_paths") or [])))

    except BaseException as exc:  # noqa: BLE001 — surface ANY failure to parent
        try:
            event_queue.put(
                (MSG_ERROR, f"{type(exc).__name__}: {exc}", traceback.format_exc())
            )
        except Exception:
            pass
