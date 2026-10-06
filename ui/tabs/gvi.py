import gc
import glob
import os
from datetime import date

import folium
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import streamlit as st
from helpers import (
    RESTART_SESSION_KEY,
    file_size_mtime_fingerprint,
    generate_clustered_grid,
    load_vector_paths,
    load_vector_upload_sessions,
    merge_gdfs_wgs84,
    path_drift_status,
    render_file_grouping_controls,
    render_job_restart_panel,
    sanitize_name_for_file,
)
from map_preview import (
    add_black_point_layer,
    add_study_area_layers,
    add_uniform_point_layer,
    trim_point_gdf_for_display,
)
from raster_overlay import add_mercator_image_overlay
from rasterio.transform import array_bounds
from shapely.geometry import box as shapely_box
from streamlit_folium import st_folium

from geofuse.crs_utils import reproject_geodataframe_to_wgs84
from geofuse.vector_io import geometry_sha256


def _gvi_output_tif_path(output_dir: str, base_name: str) -> str | None:
    """Resolve ``{base_name}_gvi.tif`` or ``.tiff`` on disk."""
    for ext in (".tif", ".tiff"):
        p = os.path.join(output_dir, f"{base_name}_gvi{ext}")
        if os.path.isfile(p):
            return p
    return None


def _gvi_dataset_uses_raster_grid(dataset: dict, buffer_m: float) -> bool:
    """Regular grid (GeoTIFF-capable) vs raw point features only."""
    if dataset.get("type") == "poly":
        return True
    return dataset.get("type") == "point" and buffer_m > 0


def _gvi_upload_signature(uploaded_files) -> tuple[tuple[str, int], ...] | None:
    """Stable fingerprint for the file uploader selection (None = widget not committed)."""
    if uploaded_files is None:
        return None
    return tuple((str(f.name), int(getattr(f, "size", 0) or 0)) for f in uploaded_files)


def _gvi_path_signature(paths) -> tuple[tuple[str, int], ...] | None:
    """Path-input variant of :func:`_gvi_upload_signature` — basename + size."""
    if paths is None:
        return None
    out: list[tuple[str, int]] = []
    for p in paths:
        if not p:
            continue
        try:
            out.append((os.path.basename(str(p)), int(os.path.getsize(p))))
        except OSError:
            out.append((os.path.basename(str(p)), 0))
    return tuple(out)


def _gvi_discard_heavy_dataset_fields() -> None:
    """Drop grid / result GeoDataFrames from this tab's ``datasets`` only; then GC."""
    for d in st.session_state.datasets.values():
        if d.get("type") == "restored":
            continue
        d["processed"] = None
        d["meta"] = None
        d["accumulated"] = []
        d["results"] = None
    gc.collect()


def _gvi_scan_outputs(output_dir: str) -> dict[str, dict]:
    """Discover GVI result sets, including per-year files in temporal folders.

    Scans ``output_dir`` itself plus every ``*_temporal_gvi/`` job folder
    produced by the per-year (date-column) mode, so each year appears as its
    own selectable result. Pure function — no Streamlit calls.
    """
    found = _gvi_scan_dir(output_dir)
    for folder in glob.glob(os.path.join(output_dir, "*_temporal_gvi")):
        if os.path.isdir(folder):
            found.update(_gvi_scan_dir(folder))
    return found


def _gvi_scan_dir(output_dir: str) -> dict[str, dict]:
    """Discover GVI ``*_gvi.*`` result sets directly inside ``output_dir``."""
    import json as _json

    from rasterio.warp import transform_bounds

    base_names: set[str] = set()
    for pat in ("*_gvi.gpkg", "*_gvi.geojson", "*_gvi.tif", "*_gvi.tiff"):
        for p in glob.glob(os.path.join(output_dir, pat)):
            stem = os.path.basename(p).rsplit(".", 1)[0]
            base_names.add(stem.removesuffix("_gvi"))
    for tiles_dir in glob.glob(os.path.join(output_dir, "*_gvi_tiles")):
        base_names.add(os.path.basename(tiles_dir).removesuffix("_gvi_tiles"))

    found: dict[str, dict] = {}
    for base_name in sorted(base_names):
        gpkg_path = os.path.join(output_dir, f"{base_name}_gvi.gpkg")
        gj_path = os.path.join(output_dir, f"{base_name}_gvi.geojson")
        sidecar_path = os.path.join(output_dir, f"{base_name}_gvi.json")
        single_tif: str | None = None
        for ext in (".tif", ".tiff"):
            p = os.path.join(output_dir, f"{base_name}_gvi{ext}")
            if os.path.isfile(p):
                single_tif = p
                break
        tiles_dir = os.path.join(output_dir, f"{base_name}_gvi_tiles")
        has_tiles = os.path.isdir(tiles_dir)

        try:
            results = None
            if os.path.isfile(gpkg_path):
                results = reproject_geodataframe_to_wgs84(gpd.read_file(gpkg_path))
            elif os.path.isfile(gj_path):
                results = reproject_geodataframe_to_wgs84(gpd.read_file(gj_path))

            meta: dict | None = None
            if single_tif:
                with rasterio.open(single_tif) as src:
                    meta = {
                        "transform": src.transform,
                        "width": src.width,
                        "height": src.height,
                        "crs": src.crs,
                    }
            elif os.path.isfile(sidecar_path):
                with open(sidecar_path) as f:
                    sc = _json.load(f)
                meta = {
                    "grid_crs_wkt": sc.get("grid_crs_wkt"),
                    "step_m": sc.get("step_m"),
                    "anchor_x": sc.get("anchor_x"),
                    "anchor_y": sc.get("anchor_y"),
                    "tiles_dir": tiles_dir if has_tiles else None,
                    "n_clusters": sc.get("n_clusters"),
                }
            elif has_tiles:
                meta = {"tiles_dir": tiles_dir}

            if results is not None and not results.empty:
                raw_geom = results.geometry.union_all().envelope
                raw_gdf = gpd.GeoDataFrame({"geometry": [raw_geom]}, crs="EPSG:4326")
            elif single_tif:
                with rasterio.open(single_tif) as src:
                    b = src.bounds
                    crs = src.crs
                w, s, e, n = transform_bounds(crs, "EPSG:4326", *b)
                raw_gdf = gpd.GeoDataFrame(
                    {"geometry": [shapely_box(w, s, e, n)]},
                    crs="EPSG:4326",
                )
            else:
                continue

            found[base_name] = {
                "raw": raw_gdf,
                "processed": None,
                "accumulated": [],
                "results": results,
                "meta": meta,
                "type": "restored",
            }
        except Exception as e:
            print(f"Error scanning {base_name}: {e}")
    return found


def _gvi_size_hint(buffer_m: int, step_m: int) -> None:
    """Show an inline banner with rough sample-count and CRS-choice expectations.

    Driven by previously-committed widget values (form widgets don't commit
    until submit), so the message updates on each form interaction cycle.
    """
    if step_m <= 0:
        return
    bbox_minx = bbox_miny = float("inf")
    bbox_maxx = bbox_maxy = float("-inf")
    total_area_m2 = 0.0
    have_data = False
    for d in st.session_state.datasets.values():
        if d.get("type") == "restored":
            continue
        raw = d.get("raw")
        if raw is None or len(raw) == 0:
            continue
        minx, miny, maxx, maxy = raw.total_bounds
        if not np.isfinite([minx, miny, maxx, maxy]).all():
            continue
        have_data = True
        bbox_minx = min(bbox_minx, minx)
        bbox_miny = min(bbox_miny, miny)
        bbox_maxx = max(bbox_maxx, maxx)
        bbox_maxy = max(bbox_maxy, maxy)
        if d.get("type") == "point":
            n = len(raw)
            disk = np.pi * (max(buffer_m, 0)) ** 2
            total_area_m2 += n * disk
        else:
            try:
                # Reproject to a metric CRS before computing area to avoid
                # geopandas' geographic-CRS warning and get accurate metres².
                m_gdf = raw.to_crs(raw.estimate_utm_crs())
                total_area_m2 += float(m_gdf.geometry.area.sum())
            except Exception:
                pass
    if not have_data:
        return

    est_pts = int(total_area_m2 / (step_m * step_m))
    span_lon = bbox_maxx - bbox_minx
    span_lat = bbox_maxy - bbox_miny
    msgs: list[str] = []
    if est_pts > 0:
        msgs.append(f"Estimated sample points: ~{est_pts:,}")
    if est_pts > 100_000:
        msgs.append(
            "GeoJSON is slow to read at this scale — keep GeoPackage on as your primary output."
        )
    if span_lon > 6.0 or span_lat > 6.0:
        msgs.append(
            f"Study area spans {span_lon:.1f}° lon × {span_lat:.1f}° lat — grid math will use Lambert Conformal Conic; outputs stay in WGS84."
        )
    if msgs:
        st.info(" · ".join(msgs))


def _gvi_restart_summary_lines(p: dict) -> list[str]:
    max_diff = p.get("max_year_diff")
    window = f" (±{max_diff} yr max)" if max_diff is not None else ""
    if p.get("mode") == "column":
        year_line = f"**Per year** from column `{p.get('date_column', '?')}`{window}"
    else:
        target_year = p.get("target_year")
        if target_year is None:
            year_line = "**Target year:** most recent capture"
        else:
            year_line = f"**Target year:** {target_year}" + (
                window or " (closest available)"
            )
    return [
        f"**Original file:** `{p.get('fname', '?')}`",
        f"**Grid step:** {p.get('step', '?')} m · **Buffer:** {p.get('buffer', '?')} m",
        year_line,
        f"**Outputs:** GeoPackage={bool(p.get('save_gpkg', True))} · "
        f"GeoTIFF={bool(p.get('save_geotiff'))} · "
        f"GeoJSON={bool(p.get('save_geojson'))}",
        f"**Save panos / masks:** {bool(p.get('save_panos'))} / {bool(p.get('save_masks'))}",
    ]


def _gvi_resubmit_from_params(
    store, executor, pano_cache, output_dir, rec_id, p, raw, api_key
):
    """Resubmit a GVI job from stored params + a (re)merged layer.

    Shared by the merged-job restart flow: ``p`` carries the original grid
    step / buffer / date settings; ``raw`` is the freshly merged layer. Column
    jobs go back through the per-year runner; recent/specific-year jobs get a
    fresh sampling grid.
    """
    out_fname = p["fname"]
    model_path = p.get("model_path")
    step = int(p.get("step", 50))
    buffer_m = int(p.get("buffer", 0))
    max_year_diff = p.get("max_year_diff")
    save_gp = bool(p.get("save_gpkg", True))
    save_gt = bool(p.get("save_geotiff"))
    save_gj = bool(p.get("save_geojson"))
    save_panos = bool(p.get("save_panos"))
    save_masks = bool(p.get("save_masks"))

    gtype = (
        "poly"
        if raw.geometry.iloc[0].geom_type in ("Polygon", "MultiPolygon")
        else "point"
    )
    base_dataset = {
        "raw": raw,
        "type": gtype,
        "processed": None,
        "meta": None,
        "accumulated": [],
        "results": None,
        "cache_ref": pano_cache,
    }

    new_params = dict(p)
    new_params["geometry_sha256"] = geometry_sha256(raw)
    new_params["restart_of"] = rec_id
    new_params["has_api_key"] = api_key is not None
    new_params["source_fingerprints"] = [
        file_size_mtime_fingerprint(x) for x in (p.get("source_paths") or [])
    ]

    init_args = {"model_path": model_path, "api_key": api_key}
    if p.get("mode") == "column":
        record = store.submit(
            type="gvi_column", name=os.path.splitext(out_fname)[0], params=new_params
        )
        executor.submit_gvi_column_subprocess(
            record,
            fname=out_fname,
            dataset_data=base_dataset,
            date_column=p["date_column"],
            init_args=init_args,
            run_args={
                "step": step,
                "buffer": buffer_m,
                "save_panos": save_panos,
                "save_masks": save_masks,
                "max_year_diff": max_year_diff,
            },
            output_dir=output_dir,
            save_gpkg=save_gp,
            save_geotiff=save_gt,
            save_geojson=save_gj,
            pano_cache_db_path=pano_cache.db_path,
        )
    else:
        proc, gmeta = _gvi_prepare_processed(raw, gtype, buffer_m, step)
        base_dataset["processed"] = proc
        base_dataset["meta"] = gmeta
        record = store.submit(
            type="gvi", name=os.path.splitext(out_fname)[0], params=new_params
        )
        executor.submit_gvi_subprocess(
            record,
            fname=out_fname,
            dataset_data=base_dataset,
            init_args=init_args,
            run_args={
                "step": step,
                "save_panos": save_panos,
                "save_masks": save_masks,
                "target_year": p.get("target_year"),
                "max_year_diff": max_year_diff,
            },
            output_dir=output_dir,
            save_gpkg=save_gp,
            save_geotiff=save_gt,
            save_geojson=save_gj,
            pano_cache_db_path=pano_cache.db_path,
        )
    return record


_GVI_RESTART_ACCEPT = [
    "geojson",
    "json",
    "gpkg",
    "shp",
    "dbf",
    "shx",
    "prj",
    "cpg",
    "zip",
]


def _render_gvi_merged_restart(store, executor, pano_cache, output_dir, rec, p) -> None:
    """Multi-file restart for a merged GVI job — quiet re-run when every source
    file is still on disk unchanged, otherwise a group re-upload."""
    sources = p.get("source_files") or []
    paths = p.get("source_paths") or []
    fps = p.get("source_fingerprints") or []
    had_api_key = bool(p.get("has_api_key"))

    with st.expander(f"↻ Restart merged job: {rec.name or rec.id}", expanded=True):
        for line in _gvi_restart_summary_lines(p):
            st.write(line)

        statuses = [
            (fn, pth, path_drift_status(pth, fp))
            for fn, pth, fp in zip(sources, paths, fps)
        ]
        all_ok = bool(paths) and all(s == "ok" for _, _, s in statuses)

        api_key = None
        if had_api_key:
            st.caption("Original job used the Street View API — re-supply the key.")
            api_key = (
                st.text_input(
                    "Street View API Key",
                    type="password",
                    autocomplete="off",
                    key=f"gmr_apikey_{rec.id}",
                )
                or None
            )

        _labels = {
            "ok": "✓ on disk",
            "missing": "✗ missing",
            "modified": "⚠ changed",
            "no-path": "? no recorded path",
        }
        for fn, _pth, s in statuses:
            st.write(f"• `{fn}` — {_labels.get(s, s)}")

        if all_ok:
            st.success("✓ All source files verified — no re-upload needed.")
            c1, c2 = st.columns(2)
            if c1.button("Cancel restart", key=f"gmr_cancel_{rec.id}", width="stretch"):
                st.session_state[RESTART_SESSION_KEY] = None
                st.rerun()
            if c2.button(
                "Re-run", type="primary", key=f"gmr_run_{rec.id}", width="stretch"
            ):
                try:
                    gdfs = [g for _, g in load_vector_paths(paths)]
                    raw = merge_gdfs_wgs84(gdfs)
                    _gvi_resubmit_from_params(
                        store, executor, pano_cache, output_dir, rec.id, p, raw, api_key
                    )
                except Exception as e:
                    st.error(f"Re-submission failed: {e}")
                    return
                st.session_state[RESTART_SESSION_KEY] = None
                st.success("Restart submitted. Monitor progress in the sidebar.")
                st.rerun()
            return

        st.warning(
            "Some source files moved or changed. Re-upload the original files "
            "for this group to restart."
        )
        uploads = st.file_uploader(
            "Re-upload the group's files",
            accept_multiple_files=True,
            type=_GVI_RESTART_ACCEPT,
            key=f"gmr_up_{rec.id}",
        )
        if not uploads:
            st.info("Select the group's original files to continue.")
            return
        try:
            loaded = load_vector_upload_sessions(uploads)
            raw = merge_gdfs_wgs84([g for _, g in loaded])
        except Exception as e:
            st.error(f"Failed to read uploaded files: {e}")
            return
        if p.get("geometry_sha256") and geometry_sha256(raw) != p["geometry_sha256"]:
            st.warning(
                "The combined geometry differs from the original job (different "
                "files, contents, or order). Re-running will process this new "
                "merge."
            )
        if st.button(
            "Verify & re-run",
            type="primary",
            key=f"gmr_up_run_{rec.id}",
            width="stretch",
        ):
            try:
                _gvi_resubmit_from_params(
                    store, executor, pano_cache, output_dir, rec.id, p, raw, api_key
                )
            except Exception as e:
                st.error(f"Re-submission failed: {e}")
                return
            st.session_state[RESTART_SESSION_KEY] = None
            st.success("Restart submitted. Monitor progress in the sidebar.")
            st.rerun()


def _render_gvi_restart_panel(
    store, executor, pano_cache, output_dir, parent_dir
) -> None:
    """Show the restart workflow when the user clicked ↻ on a GVI job."""
    job_id = st.session_state.get(RESTART_SESSION_KEY)
    if not job_id:
        return
    rec = store.get(job_id)
    if rec is None:
        return
    if rec.type in ("ndvi", "ndvi_column"):
        st.info(
            "A restart is pending for an NDVI job. Switch to the **NDVI "
            "Sourcing** tab to complete it."
        )
        return
    if rec.type == "fusion":
        st.info(
            "A restart is pending for a fusion job. Switch to the **Metric "
            "Fusion** tab to complete it."
        )
        return
    if rec.type not in ("gvi", "gvi_column"):
        return

    p = rec.params or {}
    if p.get("merged"):
        _render_gvi_merged_restart(store, executor, pano_cache, output_dir, rec, p)
        return
    is_column = rec.type == "gvi_column"
    had_api_key = bool(p.get("has_api_key"))

    def _extra_inputs() -> dict:
        if not had_api_key:
            return {}
        st.caption(
            "Original job used the Street View API. Re-supply the API key — "
            "secrets aren't persisted between runs."
        )
        key = st.text_input(
            "Street View API Key",
            type="password",
            autocomplete="off",
            key=f"restart_apikey_{rec.id}",
        )
        return {"api_key": key or None}

    def _on_confirm(gdf, fname_new: str, extras: dict) -> None:
        # Stage the verified GDF in this tab's session datasets so
        # generate_clustered_grid + the engine see it the same way they do
        # for a fresh upload.
        fname = fname_new
        try:
            gtype = (
                "poly"
                if gdf.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]
                else "point"
            )
        except Exception:
            gtype = "point"
        st.session_state.datasets[fname] = {
            "raw": gdf,
            "processed": None,
            "accumulated": [],
            "results": None,
            "meta": None,
            "type": gtype,
        }

        step_m = int(p.get("step", 50))
        buffer_m = int(p.get("buffer", 0))
        model_path = p.get("model_path") or os.path.join(
            parent_dir, "geofuse", "model", "best_model.pth"
        )
        api_key = extras.get("api_key") if had_api_key else None

        new_params = dict(p)
        new_params["geometry_sha256"] = geometry_sha256(gdf)
        new_params["restart_of"] = rec.id
        new_params["has_api_key"] = api_key is not None
        new_params["fname"] = fname

        st.session_state.datasets[fname]["cache_ref"] = pano_cache

        # Per-year column mode builds its grids inside the runner, so restart
        # only needs to re-stage the raw layer and resubmit the column job.
        if is_column:
            record = store.submit(
                type="gvi_column",
                name=os.path.splitext(fname)[0],
                params=new_params,
            )
            executor.submit_gvi_column_subprocess(
                record,
                fname=fname,
                dataset_data=st.session_state.datasets[fname],
                date_column=str(p.get("date_column", "")),
                init_args={"model_path": model_path, "api_key": api_key},
                run_args={
                    "step": step_m,
                    "buffer": buffer_m,
                    "save_panos": bool(p.get("save_panos")),
                    "save_masks": bool(p.get("save_masks")),
                    "max_year_diff": p.get("max_year_diff"),
                },
                output_dir=output_dir,
                save_gpkg=bool(p.get("save_gpkg", True)),
                save_geotiff=bool(p.get("save_geotiff")),
                save_geojson=bool(p.get("save_geojson")),
                pano_cache_db_path=pano_cache.db_path,
            )
            return

        # Materialize the sampling grid using the *original* step/buffer
        # rather than the form's current values, so the restart is faithful.
        if _gvi_dataset_uses_raster_grid(st.session_state.datasets[fname], buffer_m):
            pts, meta = generate_clustered_grid(
                gdf, buffer_m=float(buffer_m), step_m=float(step_m)
            )
            st.session_state.datasets[fname]["processed"] = pts
            st.session_state.datasets[fname]["meta"] = meta
        else:
            st.session_state.datasets[fname]["processed"] = gdf.copy()

        record = store.submit(
            type="gvi",
            name=os.path.splitext(fname)[0],
            params=new_params,
        )
        executor.submit_gvi_subprocess(
            record,
            fname=fname,
            dataset_data=st.session_state.datasets[fname],
            init_args={"model_path": model_path, "api_key": api_key},
            run_args={
                "step": step_m,
                "save_panos": bool(p.get("save_panos")),
                "save_masks": bool(p.get("save_masks")),
                "target_year": p.get("target_year"),
                "max_year_diff": p.get("max_year_diff"),
            },
            output_dir=output_dir,
            save_gpkg=bool(p.get("save_gpkg", True)),
            save_geotiff=bool(p.get("save_geotiff")),
            save_geojson=bool(p.get("save_geojson")),
            pano_cache_db_path=pano_cache.db_path,
        )

    render_job_restart_panel(
        rec,
        accept_types=[
            "geojson",
            "json",
            "gpkg",
            "shp",
            "dbf",
            "shx",
            "prj",
            "cpg",
            "zip",
        ],
        summary_lines=_gvi_restart_summary_lines(p),
        extra_inputs_renderer=_extra_inputs if had_api_key else None,
        on_confirm=_on_confirm,
    )


# ────────────────────────────────────────────────────────────────────
# Capture-date configuration (per-file), mirroring the NDVI tab's layout
# ────────────────────────────────────────────────────────────────────

_GVI_MODE_RECENT = "Most recent capture"
_GVI_MODE_YEAR = "A specific year"
_GVI_MODE_COLUMN = "Per year from a column"
_GVI_MODES = [_GVI_MODE_RECENT, _GVI_MODE_YEAR, _GVI_MODE_COLUMN]


def _gvi_input_datasets() -> dict:
    return {
        k: v
        for k, v in st.session_state.datasets.items()
        if v.get("type") != "restored"
    }


def _gvi_units() -> list[dict]:
    """Job units to configure and run.

    Separate mode → one unit per uploaded file. Merge mode → one unit per
    group name, with that group's files concatenated (EPSG:4326) into a single
    merged layer. Each unit: ``{key, raw, type, merged, sources, dataset}``
    (``dataset`` is the live per-file registry entry for separate units, or
    ``None`` for merged ones).
    """
    input_ds = _gvi_input_datasets()
    if st.session_state.get("gvi_run_mode") != "merge" or len(input_ds) < 2:
        return [
            {
                "key": fn,
                "raw": d["raw"],
                "type": d.get("type", "point"),
                "merged": False,
                "sources": [fn],
                "dataset": d,
            }
            for fn, d in input_ds.items()
        ]

    groups: dict[str, list[str]] = {}
    for fn in input_ds:
        g = st.session_state.gvi_file_groups.get(fn) or "Group 1"
        groups.setdefault(g, []).append(fn)

    cache = st.session_state.setdefault("_gvi_merge_cache", {})
    units: list[dict] = []
    for g, fns in groups.items():
        ck = tuple(sorted(fns))
        merged = cache.get(ck)
        if merged is None:
            merged = merge_gdfs_wgs84([input_ds[f]["raw"] for f in fns])
            cache[ck] = merged
        gtype = (
            "poly" if any(input_ds[f].get("type") == "poly" for f in fns) else "point"
        )
        units.append(
            {
                "key": g,
                "raw": merged,
                "type": gtype,
                "merged": True,
                "sources": fns,
                "dataset": None,
            }
        )
    return units


def _gvi_prepare_processed(raw, gtype: str, buffer_m: int, res: int):
    """Return ``(processed_points, meta)`` for a raw layer in recent/year mode.

    Polygons (or points with a buffer) get a clustered sampling grid; bare
    points are used as-is. Mirrors the per-file grid materialization for merged
    units, whose grids aren't pre-generated by the Generate-grids button.
    """
    uses_grid = gtype == "poly" or (gtype == "point" and buffer_m > 0)
    if uses_grid:
        return generate_clustered_grid(raw, buffer_m=float(buffer_m), step_m=float(res))
    return raw.copy(), None


def _gvi_discover_years(gdf, col: str) -> list[str]:
    """Unique years in ``col`` as sorted string labels (fusion-style preview)."""
    try:
        from geofuse.longitudinal import parse_date_column

        years = parse_date_column(gdf[col]).dt.year.dropna().astype(int).unique()
        return [str(y) for y in sorted(years)]
    except Exception:
        return []


@st.fragment
def _render_gvi_date_config() -> None:
    """Per-file capture-date configuration in its own section, one expander per
    layer — mirrors the NDVI tab's Date Configuration. In a fragment so a mode
    switch or column pick reruns only this section, not the settings + map."""
    units = _gvi_units()
    if not units:
        return

    st.markdown("**Capture Date Configuration**")
    this_year = date.today().year
    for unit in units:
        key = unit["key"]
        raw = unit["raw"]
        title = (
            f"{key}  ·  {len(unit['sources'])} files merged" if unit["merged"] else key
        )
        cfg = st.session_state.gvi_date_configs.setdefault(
            key,
            {
                "mode": _GVI_MODE_RECENT,
                "target_year": this_year,
                "date_column": None,
                "use_max_diff": False,
                "max_diff": 2,
            },
        )
        with st.expander(title, expanded=True):
            cfg["mode"] = st.radio(
                "Which capture to segment",
                _GVI_MODES,
                index=_GVI_MODES.index(cfg.get("mode", _GVI_MODE_RECENT)),
                key=f"gvi_mode_{key}",
                horizontal=True,
                help=(
                    "Most recent → newest Street View coverage. A specific year → "
                    "the historical capture closest to a year. Per year from a "
                    "column → split the layer by a year column and run each year "
                    "separately into a per-job folder."
                ),
            )
            mode = cfg["mode"]

            if mode == _GVI_MODE_YEAR:
                cfg["target_year"] = int(
                    st.number_input(
                        "Target year",
                        min_value=2007,
                        max_value=this_year,
                        value=int(cfg.get("target_year", this_year)),
                        step=1,
                        key=f"gvi_year_{key}",
                    )
                )
            elif mode == _GVI_MODE_COLUMN:
                attr_cols = [c for c in raw.columns if c.lower() != "geometry"]
                if attr_cols:
                    stored = cfg.get("date_column")
                    cfg["date_column"] = st.selectbox(
                        "Year / date column",
                        attr_cols,
                        index=attr_cols.index(stored) if stored in attr_cols else 0,
                        key=f"gvi_col_{key}",
                        help="Accepts ISO dates, year+month, year-only, or numeric years.",
                    )
                    years = _gvi_discover_years(raw, cfg["date_column"])
                    if years:
                        st.caption("Years found: " + ", ".join(f"`{y}`" for y in years))
                    else:
                        st.warning("No parseable years in the selected column.")
                else:
                    cfg["date_column"] = None
                    st.warning("No attribute columns found in this layer.")

            if mode in (_GVI_MODE_YEAR, _GVI_MODE_COLUMN):
                cfg["use_max_diff"] = st.checkbox(
                    "Limit maximum year difference",
                    value=bool(cfg.get("use_max_diff", False)),
                    key=f"gvi_usemd_{key}",
                    help=(
                        "Only accept a capture within this many years of the "
                        "target; points with none in the window are left empty."
                    ),
                )
                if cfg["use_max_diff"]:
                    cfg["max_diff"] = int(
                        st.number_input(
                            "Max acceptable difference (years)",
                            min_value=0,
                            max_value=20,
                            value=int(cfg.get("max_diff", 2)),
                            step=1,
                            key=f"gvi_md_{key}",
                        )
                    )


@st.fragment
def _render_gvi_settings_map() -> None:
    """Download settings + study-area map. In a fragment so slider/toggle edits
    rerun only this section and the values stay live in session_state (no
    separate Apply step before Run) — mirrors the NDVI tab."""
    gvi_buf_preview = int(st.session_state.get("gvi_buffer", 0))
    fc_gvi_l, fc_gvi_r = st.columns(2)
    with fc_gvi_l:
        with st.container(border=True):
            st.radio(
                "Download Mode",
                ["Package (Scraper)", "API (Street View)"],
                horizontal=True,
                key="gvi_download_mode",
                help="Google Street View scraper (no key) or the official API.",
            )
            if st.session_state.get("gvi_download_mode") == "API (Street View)":
                st.text_input(
                    "Street View API Key",
                    type="password",
                    autocomplete="off",
                    help="Optional Google Street View key; falls back to built-in access if empty.",
                    key="gvi_google_api_key",
                )
            st.slider(
                "Grid Resolution (m)",
                min_value=20,
                max_value=500,
                value=50,
                step=5,
                key="gvi_res",
                help="Spacing for sampling points in the generated grid (metres).",
            )
            st.slider(
                "Download Buffer (m)",
                min_value=0,
                max_value=2000,
                value=0,
                step=50,
                key="gvi_buffer",
                help=(
                    "Expand the study area outward by this distance (metres) before "
                    "building the sampling grid."
                ),
            )
            st.checkbox(
                "Save Raw Images & Masks",
                value=False,
                key="gvi_save_debug",
                help="Keep downloaded panoramas and segmentation masks under the output folder.",
            )
            st.checkbox(
                "Show Sampling Grid on Map",
                value=False,
                key="gvi_preview_sampling_grid",
                help=(
                    "Draw generated sampling grid points on the preview map. Turn "
                    "off for large grids to keep the browser responsive."
                ),
            )
    with fc_gvi_r:
        st.subheader("Study Area Preview")
        show_sampling_grid = st.session_state.get("gvi_preview_sampling_grid", False)
        m_input = folium.Map(location=[51.0447, -114.0719], zoom_start=11)
        all_bounds = []
        for fname, d in st.session_state.datasets.items():
            if d.get("type") == "restored":
                continue
            if d.get("raw") is not None:
                buf_bounds = add_study_area_layers(
                    m_input,
                    d["raw"],
                    study_name=f"{fname} (study area)",
                    buffer_m=gvi_buf_preview,
                    buffer_name=f"{fname} (buffer)",
                )
                all_bounds.append(d["raw"].total_bounds)
                if buf_bounds is not None:
                    all_bounds.append(buf_bounds)
            if (
                show_sampling_grid
                and d.get("processed") is not None
                and not d["processed"].empty
                and d.get("meta") is not None
            ):
                add_uniform_point_layer(
                    m_input,
                    d["processed"],
                    tooltip_fields=[],
                    tooltip_aliases=[],
                    geojson_marker=folium.Circle(
                        radius=3,
                        color="#4a148c",
                        weight=1,
                        fill=True,
                        fill_opacity=0.85,
                    ),
                    cluster_threshold=0,
                    cluster_circle_radius=5,
                    cluster_color="#4a148c",
                    cluster_fill_color="#9c27b0",
                    cluster_fill_opacity=0.82,
                    layer_name=f"{fname} sampling grid",
                )
        if all_bounds:
            min_x = min([b[0] for b in all_bounds])
            min_y = min([b[1] for b in all_bounds])
            max_x = max([b[2] for b in all_bounds])
            max_y = max([b[3] for b in all_bounds])
            m_input.fit_bounds([[min_y, min_x], [max_y, max_x]])
        st_folium(
            m_input, width="100%", height=500, key="map_input", returned_objects=[]
        )
    _gvi_size_hint(
        int(st.session_state.get("gvi_buffer", 0)),
        int(st.session_state.get("gvi_res", 50)),
    )


# ────────────────────────────────────────────────────────────────────
# Live dataset registry (per-session, in addition to the process-level store)
# ────────────────────────────────────────────────────────────────────
#
# Job records persisted in SQLite carry only JSON-serializable parameters.
# The live GeoDataFrames they operate on stay in ``st.session_state.datasets``
# (per-session) and are looked up by ``fname`` from ``record.params``.


# ────────────────────────────────────────────────────────────────────
# Tab render entry point
# ────────────────────────────────────────────────────────────────────


def render(output_dir: str, parent_dir: str) -> None:
    st.header("GVI")

    # --- SESSION STATE ---
    if "datasets" not in st.session_state:
        st.session_state.datasets = {}
    if "gvi_inspector_select" not in st.session_state:
        st.session_state.gvi_inspector_select = None
    if "gvi_date_configs" not in st.session_state:
        st.session_state.gvi_date_configs = {}
    if "gvi_file_groups" not in st.session_state:
        st.session_state.gvi_file_groups = {}

    # JobStore + executor + PanoCache are process-level singletons (see ui/services.py).
    from job_panel import render_sidebar_job_monitor
    from services import get_job_executor, get_job_store, get_pano_cache

    store = get_job_store()
    executor = get_job_executor()
    pano_cache = get_pano_cache()

    render_sidebar_job_monitor()

    # --- RESTART PANEL (above the main form when a terminal job was clicked) ---
    _render_gvi_restart_panel(store, executor, pano_cache, output_dir, parent_dir)

    st.subheader("Input Configuration")

    from file_picker import FT_VECTOR, pick_multiple_paths

    picked_paths = pick_multiple_paths(
        "Pick Study Areas",
        key="gvi_picked_paths",
        file_types=FT_VECTOR,
        help_text=(
            "GeoJSON, GeoPackage, shapefile (.shp with sidecars in the same "
            "folder), or vector zip. Pick one or several — every selected "
            "file becomes a separate dataset. Bytes are not read until "
            "preview, grid generation, or processing actually needs them."
        ),
    )

    valid_paths = [p for p in picked_paths if p and os.path.isfile(p)]
    if valid_paths:
        sig_new = _gvi_path_signature(valid_paths)
        sig_prev = st.session_state.get("_gvi_prev_upload_sig")
        if sig_prev is not None and sig_new is not None and sig_prev != sig_new:
            _gvi_discard_heavy_dataset_fields()
            st.session_state.pop("_gvi_merge_cache", None)
        st.session_state._gvi_prev_upload_sig = sig_new

        current_names = [os.path.basename(p) for p in valid_paths]
        for k in list(st.session_state.datasets.keys()):
            ds = st.session_state.datasets[k]
            if ds.get("type") == "restored":
                continue
            if k not in current_names:
                del st.session_state.datasets[k]
        # Read only files that aren't already loaded, so the study-area vectors
        # are not re-read from disk on every rerun.
        new_paths = [
            p
            for p in valid_paths
            if os.path.basename(p) not in st.session_state.datasets
        ]
        if new_paths:
            for fname, raw in load_vector_paths(new_paths):
                try:
                    gtype = (
                        "poly"
                        if raw.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]
                        else "point"
                    )
                    st.session_state.datasets[fname] = {
                        "raw": raw,
                        "processed": None,
                        "accumulated": [],
                        "results": None,
                        "meta": None,
                        "type": gtype,
                    }
                except Exception as e:
                    st.error(f"Error loading {fname}: {e}")

        gc.collect()

    if not valid_paths:
        for k in list(st.session_state.datasets.keys()):
            if st.session_state.datasets[k].get("type") != "restored":
                del st.session_state.datasets[k]
        gc.collect()

    # File handling: run each file as its own job, or merge files that share a
    # group name into one job (so a year split across files yields one output).
    gvi_input_fnames = list(_gvi_input_datasets().keys())
    st.session_state.gvi_run_mode = render_file_grouping_controls(
        gvi_input_fnames,
        run_mode_key="gvi_run_mode_radio",
        groups_key="gvi_file_groups",
        help_text=(
            "Separate runs each uploaded study area as its own GVI job. Merge "
            "combines files that share a Group name into one job (concatenated "
            "in one CRS), so a year split across several files yields a single "
            "output. Configure each group's capture dates below."
        ),
    )

    # Settings + map and the per-file date config are each isolated in a
    # fragment, so a slider or mode edit reruns only its own section (matching
    # the NDVI tab). Settings stay live in session_state, so Run always uses the
    # current values.
    _render_gvi_settings_map()

    _render_gvi_date_config()

    oc_gvi_a, oc_gvi_b, oc_gvi_c = st.columns(3)
    with oc_gvi_a:
        st.checkbox(
            "Save GeoPackage",
            value=True,
            key="gvi_out_gpkg",
            help=(
                "Recommended. Single-file vector samples (EPSG:4326) "
                "readable by every modern GIS. Scales to country-scale "
                "runs and supports sparse cluster layouts without voids."
            ),
        )
    with oc_gvi_b:
        st.checkbox(
            "Save GeoTIFF (per-cluster tiles)",
            value=False,
            key="gvi_out_geotiff",
            help=(
                "Optional. Writes one GeoTIFF tile per buffered cluster "
                "into a *_gvi_tiles/ folder, in the auto-selected planar "
                "CRS (no resampling). Skipped if no clusters are defined."
            ),
        )
    with oc_gvi_c:
        st.checkbox(
            "Save GeoJSON",
            value=False,
            key="gvi_out_geojson",
            help=(
                "Compatibility option only. Slow to read past ~100k "
                "points; prefer GeoPackage for large national runs."
            ),
        )

    gen_row_l, gen_row_r = st.columns([11, 1])
    with gen_row_l:
        gen = st.button(
            "Generate Sampling Grids",
            width="stretch",
            key="gvi_gen_sampling_grids",
        )
    with gen_row_r:
        gen_action_spinner = st.empty()
    run_row_l, run_row_r = st.columns([11, 1])
    with run_row_l:
        run = st.button(
            "🚀 Run GVI Analysis",
            type="primary",
            width="stretch",
            key="gvi_run_analysis",
        )
    with run_row_r:
        run_action_spinner = st.empty()

    gvi_buffer_for_gen = int(st.session_state.get("gvi_buffer", 0))
    gvi_res_for_gen = int(st.session_state.get("gvi_res", 50))

    # Files configured for per-year column mode build their grids inside the
    # runner, so they are excluded from any whole-layer grid materialization.
    column_fnames = {
        fname
        for fname, cfg in st.session_state.gvi_date_configs.items()
        if cfg.get("mode") == _GVI_MODE_COLUMN
    }

    if gen:
        if not st.session_state.datasets:
            st.warning("Upload at least one study area first.")
        else:
            _gvi_discard_heavy_dataset_fields()
            distortion_msgs: list[str] = []
            with gen_action_spinner:
                with st.spinner("​"):
                    for fname_g, d in st.session_state.datasets.items():
                        if d.get("type") == "restored" or fname_g in column_fnames:
                            continue
                        if _gvi_dataset_uses_raster_grid(d, gvi_buffer_for_gen):
                            pts, meta = generate_clustered_grid(
                                d["raw"],
                                buffer_m=float(gvi_buffer_for_gen),
                                step_m=float(gvi_res_for_gen),
                            )
                            d["processed"] = pts
                            d["meta"] = meta
                            dist = float(meta.get("distortion", 0.0) or 0.0)
                            if dist > 0.02:
                                distortion_msgs.append(
                                    f"{fname_g}: planar CRS "
                                    f"{meta.get('choice_name', '?')} — "
                                    f"distortion ~{dist * 100:.1f}% across "
                                    f"the extent. Outputs stay in WGS84; "
                                    f"distances may drift across far-apart "
                                    f"clusters."
                                )
                        else:
                            d["processed"] = d["raw"].copy()
                            d["meta"] = None
                        d["accumulated"] = []
                        d["results"] = None
            for msg in distortion_msgs:
                st.warning(msg)
            st.success("Grids generated!")
            gc.collect()
            st.rerun()

    gvi_buffer = int(st.session_state.get("gvi_buffer", 0))
    gvi_res = int(st.session_state.get("gvi_res", 50))
    gvi_out_ok = (
        st.session_state.get("gvi_out_gpkg", True)
        or st.session_state.get("gvi_out_geotiff", False)
        or st.session_state.get("gvi_out_geojson", False)
    )

    if run:
        if not gvi_out_ok:
            st.error("Select at least one output format.")
        elif not st.session_state.datasets:
            st.warning("Upload at least one study area to get started.")
        else:
            save_gp = st.session_state.get("gvi_out_gpkg", True)
            save_gt = st.session_state.get("gvi_out_geotiff", False)
            save_gj = st.session_state.get("gvi_out_geojson", False)
            save_debug = st.session_state.get("gvi_save_debug", False)
            mode = st.session_state.get("gvi_download_mode", "Package (Scraper)")
            api_key = None
            if mode == "API (Street View)":
                k = st.session_state.get("gvi_google_api_key", "")
                api_key = k if k else None

            model_path = os.path.join(parent_dir, "geofuse", "model", "best_model.pth")
            started = False

            # Identity tuple for an in-flight job — only an *identical*
            # resubmission is blocked. Changing resolution, buffer, or any
            # output flag produces a new signature and a new job.
            def _gvi_signature(p: dict) -> tuple:
                return (
                    p.get("fname"),
                    p.get("step"),
                    p.get("buffer"),
                    p.get("save_panos"),
                    p.get("save_masks"),
                    p.get("save_gpkg"),
                    p.get("save_geotiff"),
                    p.get("save_geojson"),
                    p.get("target_year"),
                    p.get("max_year_diff"),
                    p.get("has_api_key"),
                )

            # basename -> absolute path map so each (separate) job records the
            # on-disk location of its study area for silent-restart.
            path_by_basename = {os.path.basename(p): p for p in valid_paths}

            for unit in _gvi_units():
                key = unit["key"]
                raw = unit["raw"]
                gtype = unit["type"]
                merged = unit["merged"]

                cfg = st.session_state.gvi_date_configs.get(key, {})
                mode_cfg = cfg.get("mode", _GVI_MODE_RECENT)
                date_col = (
                    cfg.get("date_column") if mode_cfg == _GVI_MODE_COLUMN else None
                )
                target_year = (
                    int(cfg["target_year"]) if mode_cfg == _GVI_MODE_YEAR else None
                )
                # The max-year-difference window only applies once a year is
                # targeted (specific-year or column mode).
                max_year_diff = (
                    int(cfg["max_diff"])
                    if cfg.get("use_max_diff") and mode_cfg != _GVI_MODE_RECENT
                    else None
                )

                # A merged group has no single source file: name outputs from
                # the group and ship the concatenated layer as a fresh dataset.
                # Record each source file's path + fingerprint so restart can
                # verify the whole group and re-run quietly.
                if merged:
                    out_fname = f"{sanitize_name_for_file(key)}.gpkg"
                    ds_path = None
                    base_dataset = {
                        "raw": raw,
                        "type": gtype,
                        "processed": None,
                        "meta": None,
                        "accumulated": [],
                        "results": None,
                    }
                    src_paths = [path_by_basename.get(fn) for fn in unit["sources"]]
                    src_fps = [file_size_mtime_fingerprint(p) for p in src_paths]
                else:
                    out_fname = key
                    ds_path = path_by_basename.get(key)
                    base_dataset = unit["dataset"]
                    src_paths, src_fps = [], []
                base_dataset["cache_ref"] = pano_cache

                common_params = {
                    "fname": out_fname,
                    "input_path": ds_path,
                    "input_fingerprint": file_size_mtime_fingerprint(ds_path),
                    "step": gvi_res,
                    "buffer": gvi_buffer,
                    "save_panos": save_debug,
                    "save_masks": save_debug,
                    "save_gpkg": save_gp,
                    "save_geotiff": save_gt,
                    "save_geojson": save_gj,
                    "max_year_diff": max_year_diff,
                    "model_path": model_path,
                    "has_api_key": api_key is not None,
                    "geometry_sha256": geometry_sha256(raw),
                    "merged": merged,
                    "group_name": key if merged else None,
                    "source_files": list(unit["sources"]),
                    "source_paths": src_paths,
                    "source_fingerprints": src_fps,
                }

                # --- Per-year date-column job ---
                if mode_cfg == _GVI_MODE_COLUMN and date_col:
                    col_params = dict(
                        common_params, mode="column", date_column=date_col
                    )
                    record = store.submit(
                        type="gvi_column",
                        name=os.path.splitext(out_fname)[0],
                        params=col_params,
                    )
                    executor.submit_gvi_column_subprocess(
                        record,
                        fname=out_fname,
                        dataset_data=base_dataset,
                        date_column=date_col,
                        init_args={"model_path": model_path, "api_key": api_key},
                        run_args={
                            "step": gvi_res,
                            "buffer": gvi_buffer,
                            "save_panos": save_debug,
                            "save_masks": save_debug,
                            "max_year_diff": max_year_diff,
                        },
                        output_dir=output_dir,
                        save_gpkg=save_gp,
                        save_geotiff=save_gt,
                        save_geojson=save_gj,
                        pano_cache_db_path=pano_cache.db_path,
                    )
                    started = True
                    continue
                elif mode_cfg == _GVI_MODE_COLUMN and not date_col:
                    st.warning(f"{key}: no year/date column selected — skipped.")
                    continue

                # --- Recent / specific-year job (needs a sampling grid) ---
                if base_dataset.get("processed") is None:
                    with run_action_spinner:
                        with st.spinner("​"):
                            proc, gmeta = _gvi_prepare_processed(
                                raw, gtype, gvi_buffer, gvi_res
                            )
                    base_dataset["processed"] = proc
                    base_dataset["meta"] = gmeta

                job_params = dict(common_params, target_year=target_year)
                sig = _gvi_signature(job_params)

                # Skip only if an identical submission is still active.
                duplicate = [
                    r
                    for r in store.list_active()
                    if r.type == "gvi" and _gvi_signature(r.params) == sig
                ]
                if duplicate:
                    continue

                record = store.submit(
                    type="gvi",
                    name=os.path.splitext(out_fname)[0],
                    params=job_params,
                )
                executor.submit_gvi_subprocess(
                    record,
                    fname=out_fname,
                    dataset_data=base_dataset,
                    init_args={"model_path": model_path, "api_key": api_key},
                    run_args={
                        "step": gvi_res,
                        "save_panos": save_debug,
                        "save_masks": save_debug,
                        "target_year": target_year,
                        "max_year_diff": max_year_diff,
                    },
                    output_dir=output_dir,
                    save_gpkg=save_gp,
                    save_geotiff=save_gt,
                    save_geojson=save_gj,
                    pano_cache_db_path=pano_cache.db_path,
                )
                started = True

            if started:
                st.success("Analysis started. Monitor progress in the sidebar.")
            else:
                st.info("All study areas are already running or completed.")

    st.divider()

    col_btm_left, col_btm_right = st.columns(2)

    with col_btm_left:
        st.subheader("Result Inspector")

        scan_row_l, scan_row_r = st.columns([11, 1])
        with scan_row_l:
            scan_clicked = st.button(
                "🔄 Scan Output Folder",
                key="gvi_scan_folder",
                width="stretch",
            )
        with scan_row_r:
            scan_spinner_slot = st.empty()

        if scan_clicked:
            import threading as _threading

            holder: dict = {}

            def _scan_worker(out_dir: str, target: dict) -> None:
                try:
                    target["result"] = _gvi_scan_outputs(out_dir)
                except Exception as exc:  # noqa: BLE001
                    target["error"] = exc

            with scan_spinner_slot:
                with st.spinner("​"):
                    t = _threading.Thread(
                        target=_scan_worker, args=(output_dir, holder), daemon=True
                    )
                    t.start()
                    t.join()

            err = holder.get("error")
            if err is not None:
                st.error(f"Scan failed: {err}")
            else:
                result = holder.get("result") or {}
                count = 0
                for base_name, dataset_dict in result.items():
                    if base_name not in st.session_state.datasets:
                        st.session_state.datasets[base_name] = dataset_dict
                        count += 1
                if count > 0:
                    st.success(f"Loaded {count} result(s) from the output folder.")
                else:
                    st.info(
                        "No new GVI results found "
                        "(looked for *_gvi.gpkg, *_gvi.geojson, *_gvi.tif, "
                        "*_gvi_tiles/)."
                    )

        completed_ds = [
            k
            for k, v in st.session_state.datasets.items()
            if v.get("type") == "restored"
        ]

        options = ["All Regions"] + completed_ds

        selected_option = st.selectbox(
            "Select Result",
            options,
            index=None,
            placeholder="Choose a result...",
            key="gvi_inspector_select",
        )

        raster_layer = st.radio(
            "Background Raster",
            ["Vegetation", "Terrain"],
            horizontal=True,
            key="gvi_raster_layer",
        )
        r_opacity = st.slider("Layer Opacity", 0.0, 1.0, 0.7, key="gvi_layer_opacity")

        def _gvi_inspector_has_geojson(sel: str | None) -> bool:
            if not completed_ds:
                return False
            if sel is None:
                return False
            if sel == "All Regions":
                return all(
                    st.session_state.datasets[k].get("results") is not None
                    for k in completed_ds
                )
            return st.session_state.datasets.get(sel, {}).get("results") is not None

        gvi_pts_ok = _gvi_inspector_has_geojson(selected_option)
        if not gvi_pts_ok and st.session_state.get("gvi_show_points"):
            st.session_state.gvi_show_points = False
        show_points = st.checkbox(
            "Show Sample Points",
            value=False,
            key="gvi_show_points",
            disabled=not gvi_pts_ok,
            help=(
                "Requires a GeoJSON next to this GeoTIFF (same base name, "
                "_gvi.geojson). Enable Save GeoJSON when running GVI or add the file."
                if not gvi_pts_ok
                else None
            ),
        )

    with col_btm_right:
        st.subheader("Results Preview")
        m_result = folium.Map(location=[51.0447, -114.0719], zoom_start=11)

        if selected_option:
            if selected_option == "All Regions":
                targets = completed_ds
            else:
                targets = [selected_option]

            res_bounds = []

            for ds_name in targets:
                if ds_name not in st.session_state.datasets:
                    continue
                ds = st.session_state.datasets[ds_name]

                meta = ds.get("meta") or {}
                has_single_raster = (
                    meta.get("transform") is not None
                    and meta.get("height") is not None
                    and meta.get("width") is not None
                )

                # Always show an extent rectangle for the dataset, derived from
                # the legacy raster meta when available, otherwise from the
                # raw envelope produced by Scan Output Folder.
                if has_single_raster:
                    from rasterio.warp import transform_bounds

                    left, bottom, right, top = array_bounds(
                        meta["height"], meta["width"], meta["transform"]
                    )
                    meta_crs = meta.get("crs", "EPSG:4326")
                    if meta_crs != "EPSG:4326":
                        left, bottom, right, top = transform_bounds(
                            meta_crs, "EPSG:4326", left, bottom, right, top
                        )
                elif ds.get("raw") is not None and not ds["raw"].empty:
                    left, bottom, right, top = ds["raw"].total_bounds
                else:
                    left = bottom = right = top = None

                if left is not None:
                    folium.Rectangle(
                        bounds=[[bottom, left], [top, right]],
                        color="grey",
                        weight=1,
                        fill=False,
                        popup=f"{ds_name} Extent",
                    ).add_to(m_result)
                    res_bounds.append([left, bottom, right, top])

                # Raster overlay rendering:
                #   * Legacy single-TIF outputs use the on-disk grid directly.
                #   * GeoPackage-only outputs synthesise a small grid from the
                #     points' bbox via :func:`rasterize_points_for_preview`.
                # In both cases the rest of the pipeline (colormap + PNG +
                # ImageOverlay) is identical.
                if has_single_raster:
                    from rasterio.transform import rowcol

                    arr = np.full((meta["height"], meta["width"]), np.nan)
                    col_name = "gvi_ter" if "Terrain" in raster_layer else "gvi_veg"
                    if ds.get("results") is not None:
                        res = ds["results"].dropna(subset=[col_name])
                        if not res.empty:
                            rows, cols = rowcol(
                                meta["transform"],
                                res.geometry.x.values,
                                res.geometry.y.values,
                            )
                            mask_idx = (
                                (rows >= 0)
                                & (rows < meta["height"])
                                & (cols >= 0)
                                & (cols < meta["width"])
                            )
                            arr[rows[mask_idx], cols[mask_idx]] = res[col_name].values[
                                mask_idx
                            ]

                    tif_path_ds = _gvi_output_tif_path(output_dir, ds_name)
                    if not np.any(np.isfinite(arr)) and tif_path_ds:
                        with rasterio.open(tif_path_ds) as src:
                            bi = 2 if "Terrain" in raster_layer else 1
                            r = src.read(bi).astype(np.float64)
                            nodata = src.nodata
                        if r.shape == (meta["height"], meta["width"]):
                            arr = r
                            if nodata is not None:
                                arr = np.where(arr == nodata, np.nan, arr)

                    if np.any(np.isfinite(arr)):
                        cmap_name = "OrRd" if "Terrain" in raster_layer else "Greens"
                        try:
                            cmap = matplotlib.colormaps[cmap_name]
                        except (AttributeError, KeyError):
                            cmap = plt.get_cmap(cmap_name)

                        # Mercator-warp + colorize via shared helper so the
                        # overlay aligns with the basemap (Leaflet otherwise
                        # linearly stretches an EPSG:4326 image in Mercator
                        # screen space, displacing rows N-S).
                        add_mercator_image_overlay(
                            m_result,
                            arr,
                            src_transform=meta["transform"],
                            src_crs=meta.get("crs") or "EPSG:4326",
                            cmap=cmap,
                            vmin=0.0,
                            vmax=0.6,
                            nodata_mask=np.isnan(arr),
                            opacity=r_opacity,
                            interactive=False,
                        )
                elif ds.get("results") is not None:
                    # GeoPackage-only: black points, no tooltips (too heavy
                    # for ~2 M-point runs over a WebSocket).
                    add_black_point_layer(
                        m_result,
                        ds["results"],
                        radius=6,
                        fill_opacity=r_opacity,
                        layer_name=f"{ds_name} samples",
                    )

                if show_points and ds.get("results") is not None:
                    try:
                        gdf_viz = ds["results"].copy()
                        gdf_viz, trim_msg = trim_point_gdf_for_display(gdf_viz)
                        if trim_msg:
                            st.warning(f"{ds_name}: {trim_msg}")
                        if "gvi_veg" in gdf_viz.columns:
                            gdf_viz["gvi_veg"] = gdf_viz["gvi_veg"].round(4)
                        if "gvi_ter" in gdf_viz.columns:
                            gdf_viz["gvi_ter"] = gdf_viz["gvi_ter"].round(4)

                        valid_pts = gdf_viz.dropna(subset=["gvi_veg"])
                        has_date = "pano_date" in valid_pts.columns
                        tooltip_fields = ["gvi_veg", "gvi_ter"]
                        tooltip_aliases = ["Veg Index:", "Ter Index:"]
                        if has_date:
                            tooltip_fields.append("pano_date")
                            tooltip_aliases.append("Capture:")
                        add_uniform_point_layer(
                            m_result,
                            valid_pts,
                            tooltip_fields=tooltip_fields,
                            tooltip_aliases=tooltip_aliases,
                            geojson_marker=folium.Circle(
                                radius=20,
                                fill_color="green",
                                fill_opacity=0.8,
                                color=None,
                            ),
                            cluster_circle_radius=6,
                            cluster_color="#1a7f37",
                            cluster_fill_color="green",
                            cluster_fill_opacity=0.8,
                            layer_name=f"{ds_name} GVI sample points",
                        )
                    except Exception as e:
                        st.error(f"Error rendering sample points: {e}")

            # Only fit bounds when the selection changes — otherwise the
            # user's manual zoom/pan would be reset on every script rerun.
            last_sel = st.session_state.get("_gvi_inspector_last_sel")
            if res_bounds and selected_option != last_sel:
                min_x = min([b[0] for b in res_bounds])
                min_y = min([b[1] for b in res_bounds])
                max_x = max([b[2] for b in res_bounds])
                max_y = max([b[3] for b in res_bounds])
                m_result.fit_bounds([[min_y, min_x], [max_y, max_x]])
            st.session_state["_gvi_inspector_last_sel"] = selected_option

        st_folium(
            m_result, width="100%", height=500, key="map_result", returned_objects=[]
        )
