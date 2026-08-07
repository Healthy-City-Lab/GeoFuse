"""Subprocess scaffold and the per-engine child entry points.

Every engine job runs in a fresh ``multiprocessing.spawn`` process. This module
holds both halves of that arrangement, because they are one protocol:

* the scaffold — the ``MSG_*`` message vocabulary, :class:`SubprocJobContext`
  (the child's stand-in for ``JobContext``), stdout/stderr and engine-log
  routing into the event queue, :func:`exit_with_parent`, and
  :func:`drain_events_until_done`, which the parent's watcher thread runs;
* the child entry points — ``run_gvi_child``, ``run_gvi_column_child``,
  ``run_ndvi_child``, ``run_ndvi_column_child``, ``run_fusion_child`` — each a
  thin adapter that rebuilds the context in the child and calls the matching
  ``geofuse.jobs.runners`` callable.

The entry points keep their engine imports *inside* the function body: this
module is imported in every pre-aggregation worker (for ``exit_with_parent``),
and those workers must not pay for torch or GDAL.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time
import traceback

MSG_PROGRESS = "progress"
MSG_HEARTBEAT = "heartbeat"
MSG_SET_EXTRA = "set_extra"
MSG_STAGE_LEDGER = "stage_ledger"
MSG_LOG = "log"
MSG_COMPLETE = "complete"
MSG_ERROR = "error"

# Name of the daemon thread installed by :func:`exit_with_parent`.
_PARENT_GUARD_THREAD = "parent-exit-guard"


def exit_with_parent() -> None:
    """Bind this child's lifetime to its parent's.

    A child outlives its parent whenever the parent goes away without running
    its own shutdown — a force-kill from Task Manager, a closed terminal, a
    crash. Nothing reaps it afterwards, so an engine child holding tens of
    gigabytes stays resident, and its commit charge stays against the page
    file, until the machine reboots.

    ``parent_process().join()`` waits on the sentinel handle the spawn child
    already owns and returns the moment the parent goes, however it goes. The
    wait runs on its own daemon thread, so it still fires while the main
    thread sits inside a long uninterruptible call (a BLAS solve, a
    ``statsmodels`` refit). ``os._exit`` skips interpreter teardown on
    purpose: the parent that would have received a clean shutdown is already
    gone, and the artifacts on disk are whatever the last completed stage
    wrote.

    Idempotent per process and a no-op when there is no parent (a child run
    directly, or in-process tests).
    """
    parent = mp.parent_process()
    if parent is None:
        return
    for t in threading.enumerate():
        if t.name == _PARENT_GUARD_THREAD:
            return

    def _wait() -> None:
        parent.join()
        os._exit(1)

    threading.Thread(target=_wait, daemon=True, name=_PARENT_GUARD_THREAD).start()


class SubprocJobContext:
    """Child-process stand-in for :class:`geofuse.persistence.job_executor.JobContext`.

    Runner code in :mod:`geofuse.jobs.runners` doesn't care whether it's
    talking to a real ``JobContext`` (parent process) or this one (child
    process); both expose ``progress`` / ``heartbeat`` / ``set_extra`` /
    ``is_cancelled``. The only difference is that this implementation pushes
    every observable event onto a ``multiprocessing.Queue`` so a reader
    thread on the parent side can apply them to the real ``JobStore``.
    """

    def __init__(self, job_id, event_queue, cancel_event, pause_event=None) -> None:
        self.job_id = job_id
        self._queue = event_queue
        self._cancel_event = cancel_event
        self._pause_event = pause_event

    def progress(
        self, value: float | None = None, status_text: str | None = None, **extras
    ) -> None:
        self._queue.put((MSG_PROGRESS, value, status_text, dict(extras)))

    def heartbeat(self) -> None:
        self._queue.put((MSG_HEARTBEAT,))

    def set_extra(self, **extras) -> None:
        self._queue.put((MSG_SET_EXTRA, dict(extras)))

    def update_stage_ledger(self, ledger: dict) -> None:
        """Forward the staged-resume ledger to the parent (see stage_ledger)."""
        self._queue.put((MSG_STAGE_LEDGER, ledger))

    def is_cancelled(self) -> bool:
        return bool(self._cancel_event.is_set())

    def is_paused(self) -> bool:
        return bool(self._pause_event is not None and self._pause_event.is_set())

    def wait_while_paused(self, poll_s: float = 0.25) -> None:
        """Block at a safe point while the job is paused.

        Returns once resumed or cancelled. Work already queued is untouched,
        so the caller resumes with the next item and nothing is skipped.
        """
        if self._pause_event is None:
            return
        while self._pause_event.is_set() and not self._cancel_event.is_set():
            self.heartbeat()
            time.sleep(poll_s)


class _QueueStreamWriter:
    """File-like wrapper that forwards each line written to it onto an
    ``mp.Queue`` as an ``MSG_LOG`` event.

    Installed on the subprocess child's ``sys.stdout`` / ``sys.stderr`` so
    that third-party libraries which write directly to those streams
    (``geemap.ee_export_image`` prints "Generating URL …" / "Downloading
    data from …", Earth Engine's auth flow, tqdm progress, etc.) land in
    the per-job log instead of leaking through to the parent process's
    terminal. Buffered to a newline so partial writes (e.g. ``tqdm`` ``\r``
    repaints) don't produce a flood of one-character messages.
    """

    def __init__(self, tag: str, level: str, event_queue) -> None:
        self._tag = tag.upper()
        self._level = level
        self._queue = event_queue
        self._buf = ""

    def write(self, s):  # noqa: D401 — file-like API
        if not s:
            return 0
        # Some libraries write ``bytes`` to the underlying stream; coerce.
        if isinstance(s, bytes):
            try:
                s = s.decode("utf-8", errors="replace")
            except Exception:
                s = repr(s)
        self._buf += s
        # Treat ``\r`` like ``\n`` so tqdm progress bars get flushed
        # instead of silently accumulating in the buffer.
        normalised = self._buf.replace("\r", "\n")
        if "\n" not in normalised:
            return len(s)
        parts = normalised.split("\n")
        # Last fragment is the still-incomplete tail.
        self._buf = parts[-1]
        for line in parts[:-1]:
            line = line.rstrip()
            if line:
                self._emit(line)
        return len(s)

    def flush(self) -> None:
        if self._buf.strip():
            self._emit(self._buf.strip())
        self._buf = ""

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def fileno(self) -> int:
        # Some libraries probe for the underlying OS file descriptor.
        # ``UnsupportedOperation`` is the standard signal that this stream
        # is text-only / synthetic — libraries that handle it gracefully
        # (tqdm, click, …) fall back to non-fd code paths.
        import io

        raise io.UnsupportedOperation("fileno")

    def _emit(self, line: str) -> None:
        # Lazy import keeps this file importable before geofuse.logger
        # finishes initialising in a freshly-spawned child.
        from geofuse.logger import _ANSI, _ANSI_ESCAPE_RE

        color = _ANSI.get(self._level, "")
        colored = (
            f"{color}{_ANSI['BOLD']}[{self._tag} {self._level}]"
            f"{_ANSI['RESET']} {line}"
        )
        plain = _ANSI_ESCAPE_RE.sub("", f"[{self._tag} {self._level}] {line}")
        try:
            self._queue.put((MSG_LOG, colored, plain))
        except Exception:
            pass


def route_engine_logging_to_queue(job_id: str, event_queue) -> None:
    """Forward every engine log line *and stdout/stderr write* in the child to the parent.

    Three things happen here:

    1. ``geofuse.logger._current_job.job_id`` is set so the captured-at-
       import-time engine ``_log()`` closures stop short-circuiting on the
       unbound-job guard.
    2. ``geofuse.logger._log_queue`` is swapped for a forwarding queue that
       pushes onto ``event_queue``. Python resolves free variables against
       the module namespace at call time, so existing engine closures pick
       up the swap automatically.
    3. ``sys.stdout`` / ``sys.stderr`` are replaced with line-buffered
       writers that push onto the same queue. Without this, libraries that
       use ``print()`` or ``sys.stderr.write()`` (notably ``geemap`` —
       which prints "Generating URL …" / "Downloading data from …" for
       every tile — and Earth Engine's auth flow) bypass stdlib logging
       entirely and their output leaks straight to the parent's terminal.
       Subprocess children inherit the parent's stdio handles by default,
       so this replacement is what stops the PowerShell window from
       filling up with EE chatter during a run.
    """
    import sys

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
    # The forwarding queue has no ``get``; keep the in-process listener thread
    # from starting in the child (``attach_external_logger`` would otherwise
    # spin it up when EE logging is wired in). Lines reach the parent's
    # listener via ``event_queue`` instead.
    _logger._listener_enabled = False

    # Replace child stdio so ``print()`` / ``sys.stderr.write`` from any
    # library running in this process land in the per-job log instead of
    # the parent's terminal. The child is dedicated to one job, so this
    # is process-wide by design.
    sys.stdout = _QueueStreamWriter("STDOUT", "INFO", event_queue)
    sys.stderr = _QueueStreamWriter("STDERR", "WARN", event_queue)


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
      * ``MSG_STAGE_LEDGER``                 → ``store.update_stage_ledger(...)``
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
        elif tag == MSG_STAGE_LEDGER:
            _, ledger = item
            store.update_stage_ledger(job_id, ledger)
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
    pause_event=None,
) -> None:
    """Entry point invoked by ``multiprocessing.Process(target=...)``.

    Imports heavy dependencies (torch, gvi engine) lazily so module-level
    import of this file stays cheap. Runs the standard
    :func:`geofuse.jobs.runners.run_gvi` against a :class:`SubprocJobContext`
    so all progress / log / completion events flow through ``event_queue``.
    Errors are caught and turned into ``MSG_ERROR`` messages so the parent
    can apply them to the JobStore.
    """
    try:
        exit_with_parent()

        # Route engine log lines into the parent queue *before* the runner
        # imports anything that might cache a logger closure.
        route_engine_logging_to_queue(job_id, event_queue)

        # Lazy: keep these out of module import path. The child re-imports
        # the modules under spawn anyway; doing it here makes failures more
        # local and lets us send them up as MSG_ERROR.
        from geofuse.jobs.runners import run_gvi
        from geofuse.persistence.caches import PanoCache

        # The parent's PanoCache instance can't cross process boundaries, but
        # the SQLite file behind it can. Open a fresh cache on the same path
        # — WAL mode handles cross-process concurrency safely.
        pano_cache = PanoCache(pano_cache_db_path)
        dataset_data["cache_ref"] = pano_cache

        ctx = SubprocJobContext(job_id, event_queue, cancel_event, pause_event)
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


def run_gvi_column_child(
    job_id: str,
    fname: str,
    dataset_data: dict,
    date_column: str,
    init_args: dict,
    run_args: dict,
    output_dir: str,
    save_geotiff: bool,
    save_geojson: bool,
    save_gpkg: bool,
    pano_cache_db_path: str,
    event_queue,
    cancel_event,
    pause_event=None,
) -> None:
    """Entry point for the per-year GVI column job. See :func:`run_gvi_child`."""
    try:
        exit_with_parent()
        route_engine_logging_to_queue(job_id, event_queue)

        from geofuse.jobs.runners import run_gvi_column
        from geofuse.persistence.caches import PanoCache

        pano_cache = PanoCache(pano_cache_db_path)
        dataset_data["cache_ref"] = pano_cache

        ctx = SubprocJobContext(job_id, event_queue, cancel_event, pause_event)
        gpu_lock = threading.Lock()

        result = run_gvi_column(
            ctx,
            fname=fname,
            dataset_data=dataset_data,
            date_column=date_column,
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


def run_ndvi_child(
    job_id: str,
    fname: str,
    dataset_data: dict,
    start_date: str,
    end_date: str,
    cloud_pct: int,
    resolution: int,
    buffer_m: int,
    output_name: str,
    output_dir: str,
    save_geotiff: bool,
    save_geojson: bool,
    save_gpkg: bool,
    save_cluster_tiles: bool,
    satellite: str,
    coverage_rescue: bool,
    event_queue,
    cancel_event,
    pause_event=None,
) -> None:
    """Subprocess entry point for ``run_ndvi`` (single date range).

    Imports the engine + runner lazily so the child's spawn path stays as
    light as possible. All progress / log / completion events flow through
    ``event_queue``; failures land as ``MSG_ERROR`` so the parent can apply
    them to the JobStore.
    """
    try:
        exit_with_parent()
        route_engine_logging_to_queue(job_id, event_queue)

        from geofuse.jobs.runners import run_ndvi

        ctx = SubprocJobContext(job_id, event_queue, cancel_event, pause_event)
        result = run_ndvi(
            ctx,
            fname=fname,
            dataset_data=dataset_data,
            start_date=start_date,
            end_date=end_date,
            cloud_pct=cloud_pct,
            resolution=resolution,
            buffer_m=buffer_m,
            output_name=output_name,
            output_dir=output_dir,
            save_geotiff=save_geotiff,
            save_geojson=save_geojson,
            save_gpkg=save_gpkg,
            save_cluster_tiles=save_cluster_tiles,
            satellite=satellite,
            coverage_rescue=coverage_rescue,
        )
        event_queue.put((MSG_COMPLETE, list(result.get("output_paths") or [])))

    except BaseException as exc:  # noqa: BLE001
        try:
            event_queue.put(
                (MSG_ERROR, f"{type(exc).__name__}: {exc}", traceback.format_exc())
            )
        except Exception:
            pass


def run_ndvi_column_child(
    job_id: str,
    fname: str,
    dataset_data: dict,
    date_column: str,
    season_start_month: int,
    season_end_month: int,
    cloud_pct: int,
    resolution: int,
    buffer_m: int,
    output_dir: str,
    save_geotiff: bool,
    save_geojson: bool,
    save_gpkg: bool,
    save_cluster_tiles: bool,
    satellite: str,
    coverage_rescue: bool,
    event_queue,
    cancel_event,
    pause_event=None,
) -> None:
    """Subprocess entry point for ``run_ndvi_column`` (one raster per year)."""
    try:
        exit_with_parent()
        route_engine_logging_to_queue(job_id, event_queue)

        from geofuse.jobs.runners import run_ndvi_column

        ctx = SubprocJobContext(job_id, event_queue, cancel_event, pause_event)
        result = run_ndvi_column(
            ctx,
            fname=fname,
            dataset_data=dataset_data,
            date_column=date_column,
            season_start_month=season_start_month,
            season_end_month=season_end_month,
            cloud_pct=cloud_pct,
            resolution=resolution,
            buffer_m=buffer_m,
            output_dir=output_dir,
            save_geotiff=save_geotiff,
            save_geojson=save_geojson,
            save_gpkg=save_gpkg,
            save_cluster_tiles=save_cluster_tiles,
            satellite=satellite,
            coverage_rescue=coverage_rescue,
        )
        event_queue.put((MSG_COMPLETE, list(result.get("output_paths") or [])))

    except BaseException as exc:  # noqa: BLE001
        try:
            event_queue.put(
                (MSG_ERROR, f"{type(exc).__name__}: {exc}", traceback.format_exc())
            )
        except Exception:
            pass


def run_fusion_child(
    job_id: str,
    run_kwargs: dict,
    event_queue,
    cancel_event,
    pause_event=None,
) -> None:
    """Entry point invoked by ``multiprocessing.Process(target=...)``.

    Reroutes engine logs + stdout/stderr into the per-job log, then runs the
    standard :func:`geofuse.jobs.runners.run_fusion` against a
    :class:`SubprocJobContext` so progress / stage-ledger / log / completion
    events flow through ``event_queue``. The fusion engine class is imported by
    the runner itself, so nothing unpicklable has to cross the spawn boundary —
    ``run_kwargs`` is the JSON-persisted run config. Any failure is caught and
    turned into an ``MSG_ERROR`` so the parent can apply it to the JobStore.
    """
    try:
        exit_with_parent()

        # Route engine log lines + stdout/stderr into the parent queue before
        # the runner imports anything that might cache a logger closure.
        route_engine_logging_to_queue(job_id, event_queue)

        from geofuse.jobs.runners import run_fusion

        ctx = SubprocJobContext(job_id, event_queue, cancel_event, pause_event)
        result = run_fusion(ctx, **run_kwargs)

        event_queue.put((MSG_COMPLETE, list((result or {}).get("output_paths") or [])))

    except BaseException as exc:  # noqa: BLE001 — surface ANY failure to parent
        try:
            event_queue.put(
                (MSG_ERROR, f"{type(exc).__name__}: {exc}", traceback.format_exc())
            )
        except Exception:
            pass
