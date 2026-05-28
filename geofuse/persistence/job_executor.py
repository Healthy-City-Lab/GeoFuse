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


def _trim_ndvi_dataset_for_subprocess(dataset_data: dict) -> dict:
    """Keep only the engine-input keys; drop UI-only state for pickling.

    Session-state dataset dicts can accumulate ``results`` and ``meta`` from
    earlier runs / restored scans. The NDVI runner only needs ``raw`` (the
    study-area GeoDataFrame). Shipping the rest across the spawn boundary
    is wasteful and would orphan any writes the runner did to the dict
    (the child's mutations aren't visible to the parent).
    """
    out: dict = {}
    if "raw" in dataset_data:
        out["raw"] = dataset_data["raw"]
    if "type" in dataset_data:
        out["type"] = dataset_data["type"]
    return out


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

    def update_stage_ledger(self, ledger: dict) -> None:
        """Persist the job's staged-resume ledger (see geofuse.jobs.stage_ledger)."""
        self.store.update_stage_ledger(self.job_id, ledger)


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

    def submit_subprocess_job(
        self,
        record: JobRecord,
        child_fn: Callable[..., None],
        child_kwargs: dict,
        *,
        process_name: str | None = None,
    ) -> Future:
        """Run any engine job in a fresh ``multiprocessing.spawn`` child.

        Engine-agnostic watcher: GVI, NDVI, and (eventually) Fusion all share
        this same scaffold — only the ``child_fn`` and its keyword arguments
        differ. The watcher task in the thread pool:

          1. Spawns the child via ``multiprocessing.get_context("spawn")``.
          2. Bridges the parent's ``threading.Event`` cancel into the child's
             ``multiprocessing.Event`` so ``JobStore.request_cancel(...)``
             works unchanged.
          3. Drains the event queue (via
             :func:`geofuse.jobs.subprocess_runner.drain_events_until_done`),
             applying progress / log / heartbeat events to the JobStore until
             COMPLETE / ERROR / cancelled.
          4. Joins the child and transitions the record to its terminal status.

        Callers pass ``child_kwargs`` containing every picklable argument the
        child needs; the watcher injects ``event_queue`` and ``cancel_event``
        on top. Any unpicklable parent state (e.g. ``PanoCache``) must be
        stripped before calling — pass a path or DB filename instead and let
        the child reopen its own handle.
        """
        import multiprocessing as mp

        from geofuse.jobs.subprocess_runner import drain_events_until_done

        # Use spawn explicitly so behaviour is identical across OSes and so
        # the child doesn't inherit the parent's CUDA / Streamlit state.
        mp_ctx = mp.get_context("spawn")
        event_queue = mp_ctx.Queue()
        cancel_event = mp_ctx.Event()

        proc_kwargs = dict(child_kwargs)
        proc_kwargs["event_queue"] = event_queue
        proc_kwargs["cancel_event"] = cancel_event

        proc = mp_ctx.Process(
            target=child_fn,
            kwargs=proc_kwargs,
            name=process_name or f"{record.type}-child-{record.id}",
            daemon=False,
        )

        parent_cancel = record.cancel_event

        def _watcher() -> None:
            bind_job_log_buffer(record.id)
            self._store.transition(record.id, "running")

            # Bridge parent threading.Event ↔ child mp.Event.
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
                        error=f"Failed to spawn {record.type} subprocess: "
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
                    # The child returns ``output_paths=[]`` when it short-
                    # circuits on cancel, which would otherwise look like a
                    # successful empty completion. Treat any COMPLETE that
                    # arrives after cancel was requested as a real cancel.
                    if cancel_event.is_set() or parent_cancel.is_set():
                        self._store.transition(record.id, "cancelled")
                    else:
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

    # ------------------------------------------------------------------
    # Engine-specific convenience wrappers around :meth:`submit_subprocess_job`.
    # ------------------------------------------------------------------

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
        """Run a GVI job in a fresh subprocess. See :meth:`submit_subprocess_job`."""
        from geofuse.jobs.gvi_subprocess import run_gvi_child

        # Strip the parent-only PanoCache before pickling; the child reopens
        # its own from ``pano_cache_db_path``.
        shippable = {k: v for k, v in dataset_data.items() if k != "cache_ref"}

        return self.submit_subprocess_job(
            record,
            run_gvi_child,
            dict(
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
            ),
            process_name=f"gvi-child-{record.id}",
        )

    def submit_ndvi_subprocess(
        self,
        record: JobRecord,
        *,
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
        save_cluster_tiles: bool = False,
        save_features_samples: bool = False,
        sample_radius_m: float = 0.0,
        sample_stat: str = "mean",
        satellite: str = "auto",
        coverage_rescue: bool = True,
    ) -> Future:
        """Run an NDVI (single-range) job in a fresh subprocess.

        Isolates Earth Engine HTTP + zip decode + rasterio reproject from the
        Streamlit GIL. See :meth:`submit_subprocess_job` for the watcher
        contract.
        """
        from geofuse.jobs.ndvi_subprocess import run_ndvi_child

        # The child only needs the raw study-area GeoDataFrame. Stripping the
        # stale ``results`` / ``meta`` from a prior scan avoids pickling
        # potentially large GDFs into the spawn payload.
        shippable = _trim_ndvi_dataset_for_subprocess(dataset_data)
        return self.submit_subprocess_job(
            record,
            run_ndvi_child,
            dict(
                job_id=record.id,
                fname=fname,
                dataset_data=shippable,
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
                save_features_samples=save_features_samples,
                sample_radius_m=sample_radius_m,
                sample_stat=sample_stat,
                satellite=satellite,
                coverage_rescue=coverage_rescue,
            ),
            process_name=f"ndvi-child-{record.id}",
        )

    def submit_ndvi_column_subprocess(
        self,
        record: JobRecord,
        *,
        fname: str,
        dataset_data: dict,
        date_column: str,
        window_days: int,
        cloud_pct: int,
        resolution: int,
        buffer_m: int,
        output_dir: str,
        save_geotiff: bool,
        save_geojson: bool,
        save_gpkg: bool,
    ) -> Future:
        """Run an NDVI per-feature-date job in a fresh subprocess."""
        from geofuse.jobs.ndvi_subprocess import run_ndvi_column_child

        shippable = _trim_ndvi_dataset_for_subprocess(dataset_data)
        return self.submit_subprocess_job(
            record,
            run_ndvi_column_child,
            dict(
                job_id=record.id,
                fname=fname,
                dataset_data=shippable,
                date_column=date_column,
                window_days=window_days,
                cloud_pct=cloud_pct,
                resolution=resolution,
                buffer_m=buffer_m,
                output_dir=output_dir,
                save_geotiff=save_geotiff,
                save_geojson=save_geojson,
                save_gpkg=save_gpkg,
            ),
            process_name=f"ndvi-col-child-{record.id}",
        )

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=True)
