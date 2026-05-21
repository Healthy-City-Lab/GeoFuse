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


def _route_engine_logging_to_queue(job_id: str, event_queue) -> None:
    """Forward every engine ``_log()`` call to the parent via ``event_queue``.

    Engine modules cached ``_log = get_logger("ENGINE")`` at import time, so
    we can't intercept by replacing ``get_logger``. Instead we replace the
    queue the closures push onto — Python resolves free variables against
    the module namespace at call time, so existing closures see the swap.

    The ``_current_job.job_id`` setup ensures the closures' ``if job_id is
    None: return`` guard doesn't drop everything before we get a chance.
    """
    from geofuse import logger as _logger

    _logger._current_job.job_id = job_id

    class _ForwardingQueue:
        """Drop-in replacement for ``geofuse.logger._log_queue`` that pushes
        each item out to the parent process via the multiprocessing queue.

        Only the methods the engine closures touch (``put_nowait`` and
        ``put``) are implemented. The in-process listener never runs in the
        child so its ``get(...)`` side is irrelevant.
        """

        def put_nowait(self, item):
            try:
                _job_id, colored, plain = item
                event_queue.put((MSG_LOG, colored, plain))
            except Exception:
                pass

        def put(self, item):
            self.put_nowait(item)

    _logger._log_queue = _ForwardingQueue()


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
        _route_engine_logging_to_queue(job_id, event_queue)

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


# ---------------------------------------------------------------------------
# Parent-side queue reader
# ---------------------------------------------------------------------------


def drain_events_until_done(
    job_id: str,
    store,
    event_queue,
    cancel_event,
    process_handle=None,
    poll_timeout: float = 0.5,
):
    """Loop on ``event_queue.get(...)`` until a terminal message arrives.

    Dispatches each message to its parent-side effect:
      * ``MSG_PROGRESS`` / ``MSG_SET_EXTRA`` → ``store.update_progress(...)``
      * ``MSG_HEARTBEAT``                    → ``store.heartbeat(...)``
      * ``MSG_LOG``                          → forward the pre-formatted line
        into :data:`geofuse.logger._log_queue` so the same listener thread
        that handles in-process ``_log()`` calls also writes child-side
        lines to the job's deque + ``logs/jobs/<job_id>.log``. This keeps
        a single writer for the on-disk file (no cross-process races).
      * ``MSG_COMPLETE`` / ``MSG_ERROR``     → return so the caller can
        transition the JobRecord to the terminal status.

    Returns one of:
      ``("completed", output_paths: list[str])``
      ``("error",     (short_msg: str, traceback_text: str))``
      ``("cancelled", None)``  — only if ``cancel_event`` was set *and* the
                                 child died without sending COMPLETE/ERROR.

    The caller is responsible for joining ``process_handle`` after this
    function returns.
    """
    import queue as _queue

    from geofuse.logger import _log_queue

    while True:
        try:
            item = event_queue.get(timeout=poll_timeout)
        except (_queue.Empty, EOFError):
            # Child may have died without sending a terminal message.
            if process_handle is not None and not process_handle.is_alive():
                # Drain any straggler messages before declaring crash.
                try:
                    item = event_queue.get_nowait()
                except (_queue.Empty, EOFError):
                    if cancel_event.is_set():
                        return ("cancelled", None)
                    return (
                        "error",
                        (
                            "Worker process exited without sending result.",
                            f"exit_code={process_handle.exitcode}",
                        ),
                    )
            else:
                continue
        if not item:
            continue
        tag = item[0]
        if tag == MSG_PROGRESS:
            _, value, status_text, extras = item
            store.update_progress(
                job_id, progress=value, status_text=status_text, **(extras or {})
            )
        elif tag == MSG_HEARTBEAT:
            store.heartbeat(job_id)
        elif tag == MSG_SET_EXTRA:
            _, extras = item
            if extras:
                store.update_progress(job_id, **extras)
        elif tag == MSG_LOG:
            _, colored, plain = item
            try:
                _log_queue.put_nowait((job_id, colored, plain))
            except Exception:
                pass
        elif tag == MSG_COMPLETE:
            return ("completed", list(item[1] or []))
        elif tag == MSG_ERROR:
            return ("error", (item[1], item[2]))
        # Unknown tags are silently ignored — forward-compatibility hook.
