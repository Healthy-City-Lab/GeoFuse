"""Fusion-specific child entry point for the subprocess runner scaffold.

Fusion jobs run in a separate ``multiprocessing.spawn`` child so the stability-
selection search loop, Optuna / pandas bookkeeping, and cache lookups (all
GIL-bound Python) stop competing with the Streamlit job-monitor fragment, the
WebSocket I/O thread, and the result-inspector renderer. GVI and NDVI already
isolate their work the same way; the shared wire protocol, the child-side
``SubprocJobContext`` (including the stage-ledger message), the logger
replumbing trick, and the parent-side queue reader all live in
:mod:`geofuse.jobs.subprocess_runner`.

This module is intentionally Streamlit-free so it can be imported in a spawned
child without dragging in the UI stack.
"""

from __future__ import annotations

import traceback

from .subprocess_runner import (
    MSG_COMPLETE,
    MSG_ERROR,
    SubprocJobContext,
    route_engine_logging_to_queue,
)


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
