"""Thread-pool executor that runs ``geofuse.jobs.runners`` callables.

The executor owns a single process-level GPU lock (replaces the
``@st.cache_resource`` lock that previously lived in ``ui/tabs/gvi.py``).
``submit_runner`` wraps a runner callable so the surrounding
``JobStore.transition(...)`` calls happen automatically — runners only call
``ctx.progress(...)``, ``ctx.heartbeat()``, ``ctx.is_cancelled()``.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from geofuse.logger import bind_job_log_buffer, unbind_job_log_buffer
from geofuse.persistence.job_store import JobRecord, JobStore

logger = logging.getLogger(__name__)


@dataclass
class JobContext:
    """Handle a runner uses to publish progress and check for cancellation.

    Runners never touch the ``JobStore`` directly — everything flows through
    this object so the executor can stay in control of state transitions.
    """

    job_id: str
    store: JobStore
    cancel_event: threading.Event

    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def progress(
        self,
        value: float | None = None,
        status_text: str | None = None,
        **extra,
    ) -> None:
        self.store.update_progress(
            self.job_id, progress=value, status_text=status_text, **extra
        )

    def heartbeat(self) -> None:
        self.store.heartbeat(self.job_id)

    def set_extra(self, **kwargs) -> None:
        """Stash live, non-persisted values (engine handles, GeoDataFrames) on the record."""
        self.store.update_progress(self.job_id, **kwargs)


class JobExecutor:
    """ThreadPoolExecutor that runs runners with a shared GPU lock."""

    def __init__(self, store: JobStore, max_workers: int = 4):
        self._store = store
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="geofuse-job"
        )
        self._gpu_lock = threading.Lock()
        self._lock = threading.Lock()
        self._futures: dict[str, Future] = {}

    @property
    def gpu_lock(self) -> threading.Lock:
        """Module-level GPU serialization. Workers must acquire before GPU work."""
        return self._gpu_lock

    @property
    def store(self) -> JobStore:
        return self._store

    def submit_runner(
        self,
        record: JobRecord,
        runner_fn: Callable[..., Any],
        *args,
        **kwargs,
    ) -> Future:
        """Submit ``runner_fn(ctx, *args, **kwargs)`` to the pool.

        The executor transitions the record to ``running`` on entry and to
        ``completed`` / ``error`` / ``cancelled`` on exit. Runners only need
        to use ``ctx`` for progress and cancellation.
        """
        ctx = JobContext(
            job_id=record.id,
            store=self._store,
            cancel_event=record.cancel_event,
        )

        def _wrapped():
            # Bind this worker thread's _log(...) calls into the job's ring
            # buffer so the sidebar monitor can display per-job logs.
            bind_job_log_buffer(record.id)
            self._store.transition(record.id, "running")
            try:
                result = runner_fn(ctx, *args, **kwargs)
            except InterruptedError:
                # Treated as a cancellation signal — workers raise this when
                # they're aborted between steps (e.g. NDVI tile loop seeing
                # cancel_callback() return True mid-download).
                logger.info("Job %s (%s) cancelled by user.", record.id, record.type)
                self._store.transition(record.id, "cancelled")
                return
            except Exception as exc:  # noqa: BLE001 — workers surface any error
                logger.exception("Job %s (%s) failed", record.id, record.type)
                self._store.transition(
                    record.id,
                    "error",
                    error=f"{type(exc).__name__}: {exc}",
                )
                rec = self._store.get(record.id)
                if rec is not None:
                    rec.extra["error_detail"] = traceback.format_exc()
                return
            finally:
                unbind_job_log_buffer()
                with self._lock:
                    self._futures.pop(record.id, None)

            # Runner returned normally — decide terminal status.
            if ctx.is_cancelled():
                self._store.transition(record.id, "cancelled")
                return

            outputs = None
            if isinstance(result, dict):
                outputs = result.get("output_paths")
            self._store.transition(
                record.id,
                "completed",
                output_paths=outputs,
            )

        future = self._pool.submit(_wrapped)
        with self._lock:
            self._futures[record.id] = future
        return future

    def submit_gvi_subprocess(
        self,
        record: JobRecord,
        *,
        fname: str,
        dataset_data: dict,
        init_args: dict,
        run_args: dict,
        output_dir: str,
        save_geotiff: bool,
        save_geojson: bool,
        save_gpkg: bool,
        pano_cache_db_path: str,
    ) -> Future:
        """Run a GVI job in a fresh subprocess (separate GIL) and watch it.

        The thread pool still hosts a small watcher task that:
          1. Spawns the child via ``multiprocessing`` (spawn context).
          2. Bridges the parent's ``threading.Event`` cancel into the child's
             ``multiprocessing.Event``.
          3. Drains the event queue, applying progress / log / heartbeat
             events to the JobStore, until COMPLETE / ERROR / cancelled.
          4. Joins the child and transitions the record to its terminal status.

        ``dataset_data["cache_ref"]`` (the parent's :class:`PanoCache`) is
        not picklable; the child reopens its own from ``pano_cache_db_path``.
        """
        import multiprocessing as mp

        from geofuse.jobs.gvi_subprocess import (
            drain_events_until_done,
            run_gvi_child,
        )

        # Strip the parent-only PanoCache before pickling.
        shippable = {k: v for k, v in dataset_data.items() if k != "cache_ref"}

        # Use spawn explicitly so behaviour is identical across OSes and so
        # the child doesn't inherit the parent's CUDA / Streamlit state.
        mp_ctx = mp.get_context("spawn")
        event_queue = mp_ctx.Queue()
        cancel_event = mp_ctx.Event()

        proc = mp_ctx.Process(
            target=run_gvi_child,
            kwargs=dict(
                job_id=record.id,
                fname=fname,
                dataset_data=shippable,
                init_args=init_args,
                run_args=run_args,
                output_dir=output_dir,
                save_geotiff=save_geotiff,
                save_geojson=save_geojson,
                save_gpkg=save_gpkg,
                pano_cache_db_path=pano_cache_db_path,
                event_queue=event_queue,
                cancel_event=cancel_event,
            ),
            name=f"gvi-child-{record.id}",
            daemon=False,
        )

        parent_cancel = record.cancel_event

        def _watcher() -> None:
            bind_job_log_buffer(record.id)
            self._store.transition(record.id, "running")

            # Bridge parent threading.Event ↔ child mp.Event so the existing
            # JobStore.request_cancel(...) path continues to work unchanged.
            bridge_stop = threading.Event()

            def _cancel_bridge() -> None:
                while not bridge_stop.is_set():
                    if parent_cancel.is_set():
                        cancel_event.set()
                        return
                    if not proc.is_alive():
                        return
                    time.sleep(0.25)

            bridge_thread = threading.Thread(
                target=_cancel_bridge, daemon=True, name=f"cancel-bridge-{record.id}"
            )

            try:
                try:
                    proc.start()
                except Exception as exc:  # noqa: BLE001
                    self._store.transition(
                        record.id,
                        "error",
                        error=f"Failed to spawn GVI subprocess: "
                        f"{type(exc).__name__}: {exc}",
                    )
                    rec = self._store.get(record.id)
                    if rec is not None:
                        rec.extra["error_detail"] = traceback.format_exc()
                    return

                bridge_thread.start()

                status, payload = drain_events_until_done(
                    record.id,
                    self._store,
                    event_queue,
                    cancel_event,
                    process_handle=proc,
                )

                bridge_stop.set()
                proc.join(timeout=30)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)

                if status == "completed":
                    self._store.transition(
                        record.id, "completed", output_paths=payload
                    )
                elif status == "error":
                    short_msg, tb = payload
                    self._store.transition(record.id, "error", error=short_msg)
                    rec = self._store.get(record.id)
                    if rec is not None:
                        rec.extra["error_detail"] = tb
                elif status == "cancelled":
                    self._store.transition(record.id, "cancelled")
            finally:
                bridge_stop.set()
                try:
                    event_queue.close()
                except Exception:
                    pass
                unbind_job_log_buffer()
                with self._lock:
                    self._futures.pop(record.id, None)

        future = self._pool.submit(_watcher)
        with self._lock:
            self._futures[record.id] = future
        return future

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=True)
