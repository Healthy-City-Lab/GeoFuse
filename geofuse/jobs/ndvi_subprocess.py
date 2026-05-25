"""NDVI-specific child entry points for the subprocess runner scaffold.

NDVI runs spawn a fresh ``multiprocessing.spawn`` child so Earth Engine HTTP,
zip-extraction, and rasterio decode/reproject don't share the GIL with the
Streamlit UI's WebSocket thread, the job-monitor fragment, and the result-
inspector renderer (same rationale as GVI). The wire protocol, child-side
``SubprocJobContext``, logger replumbing trick, and parent-side queue reader
all live in :mod:`geofuse.jobs.subprocess_runner`.
"""

from __future__ import annotations

import traceback

from .subprocess_runner import (
    MSG_COMPLETE,
    MSG_ERROR,
    SubprocJobContext,
    route_engine_logging_to_queue,
)


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
    save_features_samples: bool,
    sample_radius_m: float,
    sample_stat: str,
    satellite: str,
    coverage_rescue: bool,
    event_queue,
    cancel_event,
) -> None:
    """Subprocess entry point for ``run_ndvi`` (single date range).

    Imports the engine + runner lazily so the child's spawn path stays as
    light as possible. All progress / log / completion events flow through
    ``event_queue``; failures land as ``MSG_ERROR`` so the parent can apply
    them to the JobStore.
    """
    try:
        route_engine_logging_to_queue(job_id, event_queue)

        from geofuse.jobs.runners import run_ndvi

        ctx = SubprocJobContext(job_id, event_queue, cancel_event)
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
            save_features_samples=save_features_samples,
            sample_radius_m=sample_radius_m,
            sample_stat=sample_stat,
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
    window_days: int,
    cloud_pct: int,
    resolution: int,
    buffer_m: int,
    output_dir: str,
    save_geotiff: bool,
    save_geojson: bool,
    save_gpkg: bool,
    event_queue,
    cancel_event,
) -> None:
    """Subprocess entry point for ``run_ndvi_column`` (per-feature dates)."""
    try:
        route_engine_logging_to_queue(job_id, event_queue)

        from geofuse.jobs.runners import run_ndvi_column

        ctx = SubprocJobContext(job_id, event_queue, cancel_event)
        result = run_ndvi_column(
            ctx,
            fname=fname,
            dataset_data=dataset_data,
            date_column=date_column,
            window_days=window_days,
            cloud_pct=cloud_pct,
            resolution=resolution,
            buffer_m=buffer_m,
            output_dir=output_dir,
            save_geotiff=save_geotiff,
            save_geojson=save_geojson,
            save_gpkg=save_gpkg,
        )
        event_queue.put((MSG_COMPLETE, list(result.get("output_paths") or [])))

    except BaseException as exc:  # noqa: BLE001
        try:
            event_queue.put(
                (MSG_ERROR, f"{type(exc).__name__}: {exc}", traceback.format_exc())
            )
        except Exception:
            pass
