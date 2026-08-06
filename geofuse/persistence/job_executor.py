"""Executor that runs ``geofuse.jobs.runners`` callables in child processes.

Every engine submits through :meth:`submit_subprocess_job` — GVI, NDVI and
fusion each have a thin wrapper (``submit_gvi_subprocess`` and friends) that
names the child entry point. The executor owns a single process-level GPU lock
and drives the surrounding ``JobStore.transition(...)`` calls, so runners only
call ``ctx.progress(...)``, ``ctx.heartbeat()``, ``ctx.is_cancelled()``.
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
    pause_event: threading.Event | None = None

    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def is_paused(self) -> bool:
        return bool(self.pause_event is not None and self.pause_event.is_set())

    def wait_while_paused(self, poll_s: float = 0.25) -> None:
        """Block at a safe point while the job is paused.

        Returns as soon as the job is resumed or cancelled, so nothing already
        queued is dropped — the caller simply continues with the next item.
        """
        if self.pause_event is None:
            return
        while self.pause_event.is_set() and not self.cancel_event.is_set():
            self.heartbeat()
            time.sleep(poll_s)

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
    """Runs each job in a child process, watched from a thread pool.

    The pool holds one watcher thread per running job — it drains the child's
    message queue and drives the store transitions; the engine work itself
    happens in the child. A shared GPU lock serialises the jobs that need it.
    """

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
        pause_event = mp_ctx.Event()

        proc_kwargs = dict(child_kwargs)
        proc_kwargs["event_queue"] = event_queue
        proc_kwargs["cancel_event"] = cancel_event
        proc_kwargs["pause_event"] = pause_event

        proc = mp_ctx.Process(
            target=child_fn,
            kwargs=proc_kwargs,
            name=process_name or f"{record.type}-child-{record.id}",
            daemon=False,
        )

        parent_cancel = record.cancel_event
        parent_pause = record.pause_event

        def _watcher() -> None:
            bind_job_log_buffer(record.id)
            self._store.transition(record.id, "running")

            # Bridge parent threading.Event ↔ child mp.Event.
            bridge_stop = threading.Event()

            def _cancel_bridge() -> None:
                while not bridge_stop.is_set():
                    # Pause can toggle both ways, so mirror it continuously
                    # rather than latching like cancel.
                    if parent_pause.is_set():
                        pause_event.set()
                    else:
                        pause_event.clear()
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

    # ────────────────────────────────────────────────────────────
    # Engine-specific convenience wrappers around :meth:`submit_subprocess_job`.
    # ────────────────────────────────────────────────────────────

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
        from geofuse.jobs.subprocess_runner import run_gvi_child

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

    def submit_fusion_subprocess(
        self,
        record: JobRecord,
        run_kwargs: dict,
    ) -> Future:
        """Run a fusion job in a fresh subprocess. See :meth:`submit_subprocess_job`.

        Isolates the GIL-bound stability-selection search + reporting from the
        Streamlit render loop. All of ``run_fusion``'s arguments are
        JSON-persisted run config, so ``run_kwargs`` pickles cleanly across the
        spawn boundary now that the engine class is imported inside the runner
        rather than passed in. Stage-ledger updates ride the wire protocol's
        ``MSG_STAGE_LEDGER`` message; ``cancel_check`` works unchanged via the
        bridged ``mp.Event``.
        """
        from geofuse.jobs.subprocess_runner import run_fusion_child

        return self.submit_subprocess_job(
            record,
            run_fusion_child,
            dict(job_id=record.id, run_kwargs=run_kwargs),
            process_name=f"fusion-child-{record.id}",
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
        satellite: str = "auto",
        coverage_rescue: bool = True,
    ) -> Future:
        """Run an NDVI (single-range) job in a fresh subprocess.

        Isolates Earth Engine HTTP + zip decode + rasterio reproject from the
        Streamlit GIL. See :meth:`submit_subprocess_job` for the watcher
        contract.
        """
        from geofuse.jobs.subprocess_runner import run_ndvi_child

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
        season_start_month: int,
        season_end_month: int,
        cloud_pct: int,
        resolution: int,
        buffer_m: int,
        output_dir: str,
        save_geotiff: bool,
        save_geojson: bool,
        save_gpkg: bool,
        save_cluster_tiles: bool = False,
        satellite: str = "auto",
        coverage_rescue: bool = True,
    ) -> Future:
        """Run a per-year NDVI (date-column) job in a fresh subprocess."""
        from geofuse.jobs.subprocess_runner import run_ndvi_column_child

        shippable = _trim_ndvi_dataset_for_subprocess(dataset_data)
        return self.submit_subprocess_job(
            record,
            run_ndvi_column_child,
            dict(
                job_id=record.id,
                fname=fname,
                dataset_data=shippable,
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
            ),
            process_name=f"ndvi-col-child-{record.id}",
        )

    def submit_gvi_column_subprocess(
        self,
        record: JobRecord,
        *,
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
    ) -> Future:
        """Run a per-year GVI (date-column) job in a fresh subprocess."""
        from geofuse.jobs.subprocess_runner import run_gvi_column_child

        shippable = {k: v for k, v in dataset_data.items() if k != "cache_ref"}
        return self.submit_subprocess_job(
            record,
            run_gvi_column_child,
            dict(
                job_id=record.id,
                fname=fname,
                dataset_data=shippable,
                date_column=date_column,
                init_args=init_args,
                run_args=run_args,
                output_dir=output_dir,
                save_geotiff=save_geotiff,
                save_geojson=save_geojson,
                save_gpkg=save_gpkg,
                pano_cache_db_path=pano_cache_db_path,
            ),
            process_name=f"gvi-col-child-{record.id}",
        )

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=True)
