"""GVI-specific child entry point for the subprocess runner scaffold.

GVI workers run in a separate Python process so they no longer compete for
the GIL with the Streamlit job-monitor fragment, the WebSocket I/O thread,
and the result-inspector renderer. Measurements showed a ~37-percentage-
point GPU utilisation drop when the Streamlit tab was foreground vs hidden
— almost entirely due to in-process GIL contention.

This module is intentionally Streamlit-free so it can be imported in a
``multiprocessing.spawn``-launched child without dragging in the UI stack.
The shared message protocol, the child-side ``SubprocJobContext``, the
logger replumbing trick, and the parent-side queue reader all live in
:mod:`geofuse.jobs.subprocess_runner`.
"""

from __future__ import annotations

import threading
import traceback

from .subprocess_runner import (  # noqa: F401 — re-exported for back-compat
    MSG_COMPLETE,
    MSG_ERROR,
    SubprocJobContext,
    drain_events_until_done,
    route_engine_logging_to_queue,
)


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
    import of this file stays cheap. Runs the standard
    :func:`geofuse.jobs.runners.run_gvi` against a :class:`SubprocJobContext`
    so all progress / log / completion events flow through ``event_queue``.
    Errors are caught and turned into ``MSG_ERROR`` messages so the parent
    can apply them to the JobStore.
    """
    try:
        # Route engine log lines into the parent queue *before* the runner
        # imports anything that might cache a logger closure.
        route_engine_logging_to_queue(job_id, event_queue)

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
