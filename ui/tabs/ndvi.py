import gc
import glob
import os
from datetime import date, timedelta

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import streamlit as st
from helpers import (
    RESTART_SESSION_KEY,
    file_size_mtime_fingerprint,
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
from raster_overlay import add_mercator_image_overlay_from_file
from shapely.geometry import box as shapely_box
from streamlit_folium import st_folium

from geofuse.crs_utils import (
    metres_per_degree_at_lat,
    reproject_geodataframe_to_wgs84,
)
from geofuse.vector_io import geometry_sha256

_MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def _ndvi_input_datasets() -> dict:
    return {
        k: v
        for k, v in st.session_state.ndvi_datasets.items()
        if v.get("type") != "restored"
    }


def _ndvi_units() -> list[dict]:
    """Job units: one per file (separate) or one per group (merge).

    Merge mode concatenates each group's files into one EPSG:4326 layer, so a
    year split across several files produces a single output. Each unit:
    ``{key, raw, merged, sources}``.
    """
    input_ds = _ndvi_input_datasets()
    if st.session_state.get("ndvi_run_mode") != "merge" or len(input_ds) < 2:
        return [
            {"key": fn, "raw": d["raw"], "merged": False, "sources": [fn]}
            for fn, d in input_ds.items()
        ]

    groups: dict[str, list[str]] = {}
    for fn in input_ds:
        g = st.session_state.ndvi_file_groups.get(fn) or "Group 1"
        groups.setdefault(g, []).append(fn)

    cache = st.session_state.setdefault("_ndvi_merge_cache", {})
    units: list[dict] = []
    for g, fns in groups.items():
        ck = tuple(sorted(fns))
        merged = cache.get(ck)
        if merged is None:
            merged = merge_gdfs_wgs84([input_ds[f]["raw"] for f in fns])
            cache[ck] = merged
        units.append({"key": g, "raw": merged, "merged": True, "sources": fns})
    return units


# ────────────────────────────────────────────────────────────────────
# Tab render entry point
# ────────────────────────────────────────────────────────────────────


def _ndvi_scan_dir(directory: str) -> dict[str, dict]:
    """Discover NDVI ``*_ndvi.*`` result sets directly inside ``directory``."""
    from rasterio.warp import transform_bounds

    base_names: set[str] = set()
    for pat in ("*_ndvi.tif", "*_ndvi.tiff", "*_ndvi.geojson", "*_ndvi.gpkg"):
        for p in glob.glob(os.path.join(directory, pat)):
            stem = os.path.basename(p).rsplit(".", 1)[0]
            base_names.add(stem.removesuffix("_ndvi"))

    found: dict[str, dict] = {}
    for base_name in sorted(base_names):
        tif_path = next(
            (
                os.path.join(directory, f"{base_name}_ndvi{ext}")
                for ext in (".tif", ".tiff")
                if os.path.isfile(os.path.join(directory, f"{base_name}_ndvi{ext}"))
            ),
            None,
        )
        gpkg_path = os.path.join(directory, f"{base_name}_ndvi.gpkg")
        geojson_path = os.path.join(directory, f"{base_name}_ndvi.geojson")

        try:
            meta: dict | None = None
            b = crs = None
            if tif_path:
                with rasterio.open(tif_path) as src:
                    meta = {
                        "transform": src.transform,
                        "width": src.width,
                        "height": src.height,
                        "crs": src.crs,
                    }
                    b = src.bounds
                    crs = src.crs

            results_gdf = None
            if os.path.isfile(gpkg_path):
                results_gdf = reproject_geodataframe_to_wgs84(gpd.read_file(gpkg_path))
            elif os.path.isfile(geojson_path):
                results_gdf = reproject_geodataframe_to_wgs84(
                    gpd.read_file(geojson_path)
                )

            if results_gdf is not None and not results_gdf.empty:
                raw_geom = results_gdf.geometry.union_all().envelope
                raw_gdf = gpd.GeoDataFrame({"geometry": [raw_geom]}, crs="EPSG:4326")
            elif tif_path:
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
                "results": results_gdf,
                "meta": meta,
                "type": "restored",
                "tif_path": tif_path,
            }
        except Exception as e:
            print(f"Error loading {base_name}: {e}")
    return found


def _ndvi_scan_outputs(output_dir: str) -> dict[str, dict]:
    """Discover NDVI result sets, including per-year files in temporal folders.

    Scans ``output_dir`` itself plus every ``*_temporal_ndvi/`` job folder
    produced by the per-year (date-column) mode, so each year appears as its
    own selectable result. Pure function — no Streamlit calls.
    """
    found = _ndvi_scan_dir(output_dir)
    for folder in glob.glob(os.path.join(output_dir, "*_temporal_ndvi")):
        if os.path.isdir(folder):
            found.update(_ndvi_scan_dir(folder))
    return found


def _ndvi_restart_summary_lines(p: dict) -> list[str]:
    mode = p.get("mode", "?")
    lines = [
        f"**Original file:** `{p.get('fname', '?')}`",
        f"**Mode:** {mode}",
    ]
    if mode == "range":
        lines.append(
            f"**Date range:** {p.get('start_date', '?')} → " f"{p.get('end_date', '?')}"
        )
    elif mode == "specific":
        lines.append(
            f"**Target date:** {p.get('target_date', '?')} "
            f"(±{p.get('window_days', '?')} d)"
        )
    elif mode == "column":
        sm = p.get("season_start_month")
        em = p.get("season_end_month")
        season = f"{_MONTH_NAMES[sm - 1]}–{_MONTH_NAMES[em - 1]}" if sm and em else "?"
        lines.append(
            f"**Year column:** `{p.get('date_column', '?')}` " f"(per year, {season})"
        )
    lines.append(
        f"**Cloud max:** {p.get('cloud_pct', '?')}% · "
        f"**Resolution:** {p.get('resolution', '?')} m · "
        f"**Buffer:** {p.get('buffer_m', '?')} m"
    )
    lines.append(
        f"**Outputs:** GeoTIFF={bool(p.get('save_geotiff'))} · "
        f"GeoPackage={bool(p.get('save_gpkg'))} · "
        f"GeoJSON={bool(p.get('save_geojson'))} · "
        f"ClusterTiles={bool(p.get('save_cluster_tiles'))}"
    )
    return lines


def _ndvi_resubmit_from_params(store, executor, output_dir, rec_id, p, raw):
    """Resubmit an NDVI job from stored params + a (re)merged layer."""
    dataset_data = {"raw": raw}
    fname = p.get("fname")
    cloud_pct = int(p.get("cloud_pct", 10))
    resolution = int(p.get("resolution", 10))
    buffer_m = int(p.get("buffer_m", 0))
    save_gt = bool(p.get("save_geotiff"))
    save_gp = bool(p.get("save_gpkg"))
    save_gj = bool(p.get("save_geojson"))
    save_ct = bool(p.get("save_cluster_tiles"))

    new_params = dict(p)
    new_params["geometry_sha256"] = geometry_sha256(raw)
    new_params["restart_of"] = rec_id
    new_params["source_fingerprints"] = [
        file_size_mtime_fingerprint(x) for x in (p.get("source_paths") or [])
    ]
    rec_name = os.path.splitext(fname or "merged")[0]

    if p.get("mode") == "column":
        record = store.submit(type="ndvi_column", name=rec_name, params=new_params)
        executor.submit_ndvi_column_subprocess(
            record,
            fname=fname,
            dataset_data=dataset_data,
            date_column=p["date_column"],
            season_start_month=int(p.get("season_start_month", 6)),
            season_end_month=int(p.get("season_end_month", 9)),
            cloud_pct=cloud_pct,
            resolution=resolution,
            buffer_m=buffer_m,
            output_dir=output_dir,
            save_geotiff=save_gt,
            save_gpkg=save_gp,
            save_geojson=save_gj,
            save_cluster_tiles=save_ct,
        )
    else:
        if p.get("mode") == "specific":
            td = date.fromisoformat(p["target_date"])
            w = int(p.get("window_days", 30))
            start_d = (td - timedelta(days=w)).isoformat()
            end_d = min(td + timedelta(days=w), date.today()).isoformat()
        else:  # range
            start_d = p["start_date"]
            end_d = p["end_date"]
        record = store.submit(type="ndvi", name=rec_name, params=new_params)
        executor.submit_ndvi_subprocess(
            record,
            fname=fname,
            dataset_data=dataset_data,
            start_date=start_d,
            end_date=end_d,
            cloud_pct=cloud_pct,
            resolution=resolution,
            buffer_m=buffer_m,
            output_name=p.get("output_name", rec_name),
            output_dir=output_dir,
            save_geotiff=save_gt,
            save_gpkg=save_gp,
            save_geojson=save_gj,
            save_cluster_tiles=save_ct,
        )
    return record


_NDVI_RESTART_ACCEPT = [
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


def _render_ndvi_merged_restart(store, executor, output_dir, rec, p) -> None:
    """Multi-file restart for a merged NDVI job — quiet re-run when every source
    file is still on disk unchanged, otherwise a group re-upload."""
    sources = p.get("source_files") or []
    paths = p.get("source_paths") or []
    fps = p.get("source_fingerprints") or []

    with st.expander(f"↻ Restart merged job: {rec.name or rec.id}", expanded=True):
        for line in _ndvi_restart_summary_lines(p):
            st.write(line)

        statuses = [
            (fn, pth, path_drift_status(pth, fp))
            for fn, pth, fp in zip(sources, paths, fps)
        ]
        all_ok = bool(paths) and all(s == "ok" for _, _, s in statuses)
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
            if c1.button("Cancel restart", key=f"nmr_cancel_{rec.id}", width="stretch"):
                st.session_state[RESTART_SESSION_KEY] = None
                st.rerun()
            if c2.button(
                "Re-run", type="primary", key=f"nmr_run_{rec.id}", width="stretch"
            ):
                try:
                    raw = merge_gdfs_wgs84([g for _, g in load_vector_paths(paths)])
                    _ndvi_resubmit_from_params(
                        store, executor, output_dir, rec.id, p, raw
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
            type=_NDVI_RESTART_ACCEPT,
            key=f"nmr_up_{rec.id}",
        )
        if not uploads:
            st.info("Select the group's original files to continue.")
            return
        try:
            raw = merge_gdfs_wgs84([g for _, g in load_vector_upload_sessions(uploads)])
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
            key=f"nmr_up_run_{rec.id}",
            width="stretch",
        ):
            try:
                _ndvi_resubmit_from_params(store, executor, output_dir, rec.id, p, raw)
            except Exception as e:
                st.error(f"Re-submission failed: {e}")
                return
            st.session_state[RESTART_SESSION_KEY] = None
            st.success("Restart submitted. Monitor progress in the sidebar.")
            st.rerun()


def _render_ndvi_restart_panel(store, executor, output_dir) -> None:
    """Show the restart workflow when the user clicked ↻ on an NDVI job."""
    job_id = st.session_state.get(RESTART_SESSION_KEY)
    if not job_id:
        return
    rec = store.get(job_id)
    if rec is None:
        return
    if rec.type in ("gvi", "gvi_column"):
        st.info(
            "A restart is pending for a GVI job. Switch to the **GVI "
            "Sourcing** tab to complete it."
        )
        return
    if rec.type == "fusion":
        st.info(
            "A restart is pending for a fusion job. Switch to the **Metric "
            "Fusion** tab to complete it."
        )
        return
    if rec.type not in ("ndvi", "ndvi_column"):
        return

    p = rec.params or {}
    if p.get("merged"):
        _render_ndvi_merged_restart(store, executor, output_dir, rec, p)
        return

    def _on_confirm(gdf, fname_new: str, _extras: dict) -> None:
        fname = fname_new
        # Stage the verified GDF in the NDVI tab's dataset registry.
        try:
            gtype = (
                "poly"
                if gdf.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]
                else "point"
            )
        except Exception:
            gtype = "point"
        if "ndvi_datasets" not in st.session_state:
            st.session_state.ndvi_datasets = {}
        st.session_state.ndvi_datasets[fname] = {"raw": gdf, "type": gtype}

        new_params = dict(p)
        new_params["geometry_sha256"] = geometry_sha256(gdf)
        new_params["restart_of"] = rec.id
        new_params["fname"] = fname

        base_name = os.path.splitext(fname)[0]
        record = store.submit(type=rec.type, name=base_name, params=new_params)

        common = {
            "fname": fname,
            "dataset_data": st.session_state.ndvi_datasets[fname],
            "cloud_pct": int(p.get("cloud_pct", 10)),
            "resolution": int(p.get("resolution", 10)),
            "buffer_m": int(p.get("buffer_m", 0)),
            "output_dir": output_dir,
            "save_geotiff": bool(p.get("save_geotiff", True)),
            "save_geojson": bool(p.get("save_geojson", False)),
            "save_gpkg": bool(p.get("save_gpkg", False)),
        }

        if rec.type == "ndvi":
            executor.submit_ndvi_subprocess(
                record,
                start_date=str(p.get("start_date", "")),
                end_date=str(p.get("end_date", "")),
                output_name=str(p.get("output_name", base_name)),
                save_cluster_tiles=bool(p.get("save_cluster_tiles", False)),
                **common,
            )
        else:  # ndvi_column
            executor.submit_ndvi_column_subprocess(
                record,
                date_column=str(p.get("date_column", "")),
                season_start_month=int(p.get("season_start_month", 6)),
                season_end_month=int(p.get("season_end_month", 9)),
                save_cluster_tiles=bool(p.get("save_cluster_tiles", False)),
                **common,
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
        summary_lines=_ndvi_restart_summary_lines(p),
        extra_inputs_renderer=None,
        on_confirm=_on_confirm,
    )


def _ndvi_size_hint(buffer_m: int, resolution_m: int) -> None:
    """Inline banner: estimated pixel count and GeoTIFF tile expectations.

    Driven by the bbox of all uploaded NDVI datasets (already in EPSG:4326),
    buffered by ``buffer_m``. Cheap heuristic — uses planar approximation at
    each dataset's centroid latitude.
    """
    datasets = st.session_state.get("ndvi_datasets") or {}
    if resolution_m <= 0:
        return
    bbox_minx = bbox_miny = float("inf")
    bbox_maxx = bbox_maxy = float("-inf")
    total_area_m2 = 0.0
    have_data = False
    for d in datasets.values():
        raw = d.get("raw") if isinstance(d, dict) else None
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
        cent_lat = (miny + maxy) / 2.0
        m_per_deg_lon, m_per_deg_lat = metres_per_degree_at_lat(cent_lat)
        # bbox area in metres (NDVI rasterizes the bbox of the buffered geom)
        width_m = (maxx - minx) * m_per_deg_lon + 2 * max(buffer_m, 0)
        height_m = (maxy - miny) * m_per_deg_lat + 2 * max(buffer_m, 0)
        total_area_m2 += max(0.0, width_m) * max(0.0, height_m)
    if not have_data:
        return

    est_cells = int(total_area_m2 / (resolution_m * resolution_m))
    span_lon = bbox_maxx - bbox_minx
    span_lat = bbox_maxy - bbox_miny
    msgs: list[str] = []
    if est_cells > 0:
        msgs.append(f"Estimated NDVI pixels: ~{est_cells:,}")
    if est_cells > 1_000_000:
        msgs.append(
            "Vector output of every pixel would be very large — prefer GeoTIFF, "
            "or use GeoPackage instead of GeoJSON if you need vector samples."
        )
    if span_lon > 6.0 or span_lat > 6.0:
        msgs.append(
            f"Study area spans {span_lon:.1f}° lon × {span_lat:.1f}° lat — "
            "Earth Engine downloads will be tiled automatically."
        )
    if msgs:
        st.info(" · ".join(msgs))


@st.fragment
def _render_ndvi_settings_map() -> None:
    """Download settings + study-area map, in a fragment so slider edits rerun
    only this section and the values stay live in session_state (no separate
    Apply step before Run)."""
    fc_ndvi_l, fc_ndvi_r = st.columns(2)
    with fc_ndvi_l:
        with st.container(border=True):
            st.slider(
                "Maximum Cloud Coverage (%)",
                0,
                100,
                10,
                key="ndvi_cloud",
                help="Cloud mask threshold for Earth Engine, applied on Run.",
            )
            st.number_input(
                "Resolution (m)",
                value=10,
                min_value=10,
                key="ndvi_res",
                help="Target pixel size for the NDVI raster export.",
            )
            buffer_m = st.slider(
                "Download Buffer (m)",
                min_value=0,
                max_value=2000,
                value=0,
                step=50,
                key="ndvi_buffer",
                help="Expand the study area outward by this distance (metres) before download.",
            )
    with fc_ndvi_r:
        st.subheader("Study Area Preview")
        m_ndvi_input = folium.Map(location=[51.0447, -114.0719], zoom_start=10)
        all_bounds = []
        for fname, d in st.session_state.ndvi_datasets.items():
            if d.get("type") == "restored" or d.get("raw") is None:
                continue
            buf_bounds = add_study_area_layers(
                m_ndvi_input,
                d["raw"],
                study_name=fname,
                buffer_m=int(buffer_m),
                buffer_name=f"{fname} (buffer)",
            )
            all_bounds.append(d["raw"].total_bounds)
            if buf_bounds is not None:
                all_bounds.append(buf_bounds)
        if all_bounds:
            min_x = min(b[0] for b in all_bounds)
            min_y = min(b[1] for b in all_bounds)
            max_x = max(b[2] for b in all_bounds)
            max_y = max(b[3] for b in all_bounds)
            m_ndvi_input.fit_bounds([[min_y, min_x], [max_y, max_x]])
        st_folium(
            m_ndvi_input,
            width="100%",
            height=500,
            key="map_ndvi_input",
            returned_objects=[],
        )
    _ndvi_size_hint(int(buffer_m), int(st.session_state.get("ndvi_res", 10)))


def _ndvi_discover_years(gdf, col: str) -> list[str]:
    """Unique years in ``col`` as sorted string labels (fusion-style preview)."""
    try:
        from geofuse.longitudinal import parse_date_column

        years = parse_date_column(gdf[col]).dt.year.dropna().astype(int).unique()
        return [str(y) for y in sorted(years)]
    except Exception:
        return []


@st.fragment
def _render_ndvi_date_config() -> None:
    """Per-dataset date configuration. In a fragment so add/remove-date buttons
    rerun only this section, not the settings + map above."""
    units = _ndvi_units()
    if units:
        st.markdown("**Date Configuration**")
        for unit in units:
            fname = unit["key"]
            d = {"raw": unit["raw"]}
            expander_title = (
                f"{fname}  ·  {len(unit['sources'])} files merged"
                if unit["merged"]
                else fname
            )
            if fname not in st.session_state.ndvi_date_configs:
                st.session_state.ndvi_date_configs[fname] = {
                    "use_ranges": True,
                    "use_specific": False,
                    "use_column": False,
                    "ranges": [(date(2023, 6, 1), date(2023, 9, 30))],
                    "specific_dates": [date(2023, 7, 15)],
                    "window_days_specific": 30,
                    "season_start_month": 6,
                    "season_end_month": 9,
                }
            cfg = st.session_state.ndvi_date_configs[fname]
            # Migrate old single-mode format
            if "mode" in cfg and "use_ranges" not in cfg:
                old_mode = cfg.pop("mode")
                cfg["use_ranges"] = old_mode == "Date Range"
                cfg["use_specific"] = old_mode == "Specific Date"
                cfg["use_column"] = old_mode == "Use Attribute Column"
                cfg.setdefault("ranges", [(date(2023, 6, 1), date(2023, 9, 30))])
                cfg.setdefault("specific_dates", [date(2023, 7, 15)])
                cfg["window_days_specific"] = cfg.pop("window_days", 30)
                cfg.setdefault("season_start_month", 6)
                cfg.setdefault("season_end_month", 9)

            today = date.today()

            with st.expander(expander_title, expanded=True):
                chk_c1, chk_c2, chk_c3 = st.columns(3)
                use_ranges = chk_c1.checkbox(
                    "Date Range(s)",
                    value=cfg.get("use_ranges", True),
                    key=f"ndvi_use_ranges_{fname}",
                )
                use_specific = chk_c2.checkbox(
                    "Specific Date(s)",
                    value=cfg.get("use_specific", False),
                    key=f"ndvi_use_specific_{fname}",
                )
                use_column = chk_c3.checkbox(
                    "Attribute Column",
                    value=cfg.get("use_column", False),
                    key=f"ndvi_use_col_{fname}",
                )
                cfg["use_ranges"] = use_ranges
                cfg["use_specific"] = use_specific
                cfg["use_column"] = use_column

                if not any([use_ranges, use_specific, use_column]):
                    st.warning("Select at least one date mode.")

                # ── Date Range(s) ───────────────────
                if use_ranges:
                    st.markdown("**Date Range(s)**")
                    remove_idx = None
                    for i, (s, e) in enumerate(cfg["ranges"]):
                        stored_s = st.session_state.get(f"ndvi_rs_{fname}_{i}", s)
                        stored_e = st.session_state.get(f"ndvi_re_{fname}_{i}", e)
                        range_invalid = stored_s >= stored_e
                        s_help = (
                            "Start date is on or after the end date."
                            if range_invalid
                            else None
                        )
                        e_help = (
                            "End date is on or before the start date."
                            if range_invalid
                            else None
                        )
                        c1, c2, c3 = st.columns([4, 4, 1])
                        new_s = c1.date_input(
                            "Start",
                            value=s,
                            key=f"ndvi_rs_{fname}_{i}",
                            max_value=today,
                            label_visibility="collapsed",
                            help=s_help,
                        )
                        new_e = c2.date_input(
                            "End",
                            value=e,
                            key=f"ndvi_re_{fname}_{i}",
                            max_value=today,
                            label_visibility="collapsed",
                            help=e_help,
                        )
                        cfg["ranges"][i] = (new_s, new_e)
                        if new_s >= new_e:
                            st.markdown(
                                '<p style="color:#ff4b4b;font-size:0.78em;'
                                'margin:0 0 4px 0;">'
                                "⚠ End date must be after start date.</p>",
                                unsafe_allow_html=True,
                            )
                        if len(cfg["ranges"]) > 1:
                            if c3.button(
                                "✕",
                                key=f"ndvi_rrem_{fname}_{i}",
                                help="Remove this range",
                            ):
                                remove_idx = i
                    if remove_idx is not None:
                        old_n = len(cfg["ranges"])
                        cfg["ranges"].pop(remove_idx)
                        for j in range(remove_idx, old_n):
                            st.session_state.pop(f"ndvi_rs_{fname}_{j}", None)
                            st.session_state.pop(f"ndvi_re_{fname}_{j}", None)
                        st.rerun(scope="fragment")
                    if st.button(
                        "Add Date Range",
                        key=f"ndvi_radd_{fname}",
                        help="Each range row produces its own NDVI output file.",
                    ):
                        cfg["ranges"].append((date(today.year, 1, 1), today))
                        st.rerun(scope="fragment")

                # ── Specific Date(s) ────────────────
                if use_specific:
                    if use_ranges:
                        st.divider()
                    st.markdown("**Specific Date(s)**")
                    cfg["window_days_specific"] = st.number_input(
                        "Composite window (± days)",
                        min_value=7,
                        max_value=180,
                        value=cfg.get("window_days_specific", 30),
                        key=f"ndvi_win_s_{fname}",
                        help=(
                            "Composite uses imagery within ± this many days around "
                            "each date of interest. One output file per date row."
                        ),
                    )
                    remove_idx = None
                    for i, d_val in enumerate(cfg["specific_dates"]):
                        c1, c2 = st.columns([9, 1])
                        new_d = c1.date_input(
                            "Date of interest",
                            value=d_val,
                            key=f"ndvi_sd_{fname}_{i}",
                            max_value=today,
                            label_visibility="collapsed",
                        )
                        cfg["specific_dates"][i] = new_d
                        if len(cfg["specific_dates"]) > 1:
                            if c2.button(
                                "✕",
                                key=f"ndvi_srem_{fname}_{i}",
                                help="Remove this date",
                            ):
                                remove_idx = i
                    if remove_idx is not None:
                        old_n = len(cfg["specific_dates"])
                        cfg["specific_dates"].pop(remove_idx)
                        for j in range(remove_idx, old_n):
                            st.session_state.pop(f"ndvi_sd_{fname}_{j}", None)
                        st.rerun(scope="fragment")
                    if st.button("Add Date", key=f"ndvi_sadd_{fname}"):
                        cfg["specific_dates"].append(today)
                        st.rerun(scope="fragment")

                # ── Attribute Column ────────────────
                if use_column:
                    if use_ranges or use_specific:
                        st.divider()
                    st.markdown("**Attribute Column (per-year)**")
                    attr_cols = [c for c in d["raw"].columns if c.lower() != "geometry"]
                    if attr_cols:
                        selected_col = st.selectbox(
                            "Year / date attribute column",
                            attr_cols,
                            key=f"ndvi_col_{fname}",
                            help=(
                                "Features are split by the year in this column. Each "
                                "year is processed as its own NDVI job over only that "
                                "year's features, into a per-job folder. A shared CRS "
                                "(chosen from the whole layer) keeps every year aligned."
                            ),
                        )
                        years = _ndvi_discover_years(d["raw"], selected_col)
                        if years:
                            st.caption(
                                "Years found: " + ", ".join(f"`{y}`" for y in years)
                            )
                        else:
                            st.warning("No parseable years in the selected column.")
                        st.caption(
                            "Growing-season months composited for each year — pick "
                            "the same months every year for a consistent longitudinal "
                            "comparison."
                        )
                        mc1, mc2 = st.columns(2)
                        cfg["season_start_month"] = mc1.selectbox(
                            "Season start month",
                            list(range(1, 13)),
                            index=cfg.get("season_start_month", 6) - 1,
                            format_func=lambda m: _MONTH_NAMES[m - 1],
                            key=f"ndvi_seas_start_{fname}",
                        )
                        cfg["season_end_month"] = mc2.selectbox(
                            "Season end month",
                            list(range(1, 13)),
                            index=cfg.get("season_end_month", 9) - 1,
                            format_func=lambda m: _MONTH_NAMES[m - 1],
                            key=f"ndvi_seas_end_{fname}",
                        )
                        if cfg["season_start_month"] > cfg["season_end_month"]:
                            st.markdown(
                                '<p style="color:#ff4b4b;font-size:0.78em;'
                                'margin:0;">⚠ Start month must be on or before '
                                "the end month.</p>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.warning("No attribute columns found in this file.")


def render(output_dir: str) -> None:
    st.header("NDVI")

    if "ndvi_datasets" not in st.session_state:
        st.session_state.ndvi_datasets = {}
    if "ndvi_inspector_select" not in st.session_state:
        st.session_state.ndvi_inspector_select = None
    if "ndvi_date_configs" not in st.session_state:
        st.session_state.ndvi_date_configs = {}
    if "ndvi_file_groups" not in st.session_state:
        st.session_state.ndvi_file_groups = {}

    from services import get_job_executor, get_job_store

    _render_ndvi_restart_panel(get_job_store(), get_job_executor(), output_dir)

    st.subheader("Input Configuration")

    from file_picker import FT_VECTOR, pick_multiple_paths

    ndvi_picked_paths = pick_multiple_paths(
        "Pick Study Areas",
        key="ndvi_picked_paths",
        file_types=FT_VECTOR,
        help_text=(
            "GeoJSON, GeoPackage, shapefile (.shp with sidecars in the same "
            "folder), or vector zip. Pick one or several — every selected "
            "file becomes a separate dataset and gets its own NDVI run."
        ),
    )

    ndvi_valid_paths = [p for p in ndvi_picked_paths if p and os.path.isfile(p)]
    if ndvi_valid_paths:
        current_names = [os.path.basename(p) for p in ndvi_valid_paths]
        # Drop datasets no longer picked (restored results stay in place).
        for k in list(st.session_state.ndvi_datasets.keys()):
            ds = st.session_state.ndvi_datasets[k]
            if ds.get("type") == "restored":
                continue
            if k not in current_names:
                del st.session_state.ndvi_datasets[k]
                st.session_state.ndvi_date_configs.pop(k, None)
        # Read only files that aren't already loaded, so the study-area vectors
        # are not re-read from disk on every rerun.
        new_paths = [
            p
            for p in ndvi_valid_paths
            if os.path.basename(p) not in st.session_state.ndvi_datasets
        ]
        if new_paths:
            try:
                for fname, raw in load_vector_paths(new_paths):
                    st.session_state.ndvi_datasets[fname] = {
                        "raw": raw,
                        "processed": None,
                        "results": None,
                        "meta": None,
                        "type": "input",
                    }
            except Exception as e:
                st.error(f"Failed to load study area(s): {e}")
        gc.collect()
    if not ndvi_valid_paths:
        for k in list(st.session_state.ndvi_datasets.keys()):
            if st.session_state.ndvi_datasets[k].get("type") != "restored":
                del st.session_state.ndvi_datasets[k]
                st.session_state.ndvi_date_configs.pop(k, None)
        gc.collect()

    ndvi_input_datasets = {
        k: v
        for k, v in st.session_state.ndvi_datasets.items()
        if v.get("type") != "restored"
    }

    # Invalidate cached merges whenever the set of input files changes.
    _ndvi_input_sig = tuple(sorted(ndvi_input_datasets))
    if st.session_state.get("_ndvi_input_sig") != _ndvi_input_sig:
        st.session_state.pop("_ndvi_merge_cache", None)
        st.session_state["_ndvi_input_sig"] = _ndvi_input_sig

    # File handling: run each file as its own job, or merge files that share a
    # group name into one job (so a year split across files yields one output).
    st.session_state.ndvi_run_mode = render_file_grouping_controls(
        list(ndvi_input_datasets.keys()),
        run_mode_key="ndvi_run_mode_radio",
        groups_key="ndvi_file_groups",
        help_text=(
            "Separate runs each uploaded study area as its own NDVI job. Merge "
            "combines files that share a Group name into one job (concatenated "
            "in one CRS), so a year split across several files yields a single "
            "output. Configure each group's dates below."
        ),
    )

    # Settings + map and the date config are each isolated in a fragment, so a
    # slider or add-date edit reruns only its own section — the rest of the tab
    # (and the disk reads above) is untouched. Settings stay live in
    # session_state, so Run always uses the current values.
    _render_ndvi_settings_map()

    _render_ndvi_date_config()

    oc_ndvi_a, oc_ndvi_b, oc_ndvi_c, oc_ndvi_d = st.columns(4)
    with oc_ndvi_a:
        st.checkbox(
            "Save GeoTIFF",
            value=True,
            key="ndvi_out_geotiff",
            help="Raster NDVI surface (primary format for NDVI).",
        )
    with oc_ndvi_b:
        st.checkbox(
            "Save GeoPackage",
            value=False,
            key="ndvi_out_gpkg",
            help=(
                "Vector samples in EPSG:4326, single file, readable by every "
                "modern GIS. Recommended over GeoJSON for large outputs."
            ),
        )
    with oc_ndvi_c:
        st.checkbox(
            "Save GeoJSON",
            value=False,
            key="ndvi_out_geojson",
            help="Compatibility option only. Slow to read past ~100k points.",
        )
    with oc_ndvi_d:
        st.checkbox(
            "Per-cluster tiles",
            value=False,
            key="ndvi_out_cluster_tiles",
            help=(
                "For scattered inputs (one feature per city / province), write "
                "one GeoTIFF per connected component into "
                "`{name}_ndvi_tiles/` plus a `tiles_index.json`. Avoids the "
                "single mostly-NaN continent-spanning mosaic."
            ),
        )

    run = st.button(
        "🚀 Run NDVI Analysis",
        type="primary",
        width="stretch",
        key="ndvi_run_btn",
    )

    cloud_pct = int(st.session_state.get("ndvi_cloud", 10))
    resolution = int(st.session_state.get("ndvi_res", 10))
    buffer_m = int(st.session_state.get("ndvi_buffer", 0))
    ndvi_out_ok = (
        st.session_state.get("ndvi_out_geotiff", True)
        or st.session_state.get("ndvi_out_gpkg", False)
        or st.session_state.get("ndvi_out_geojson", False)
        or st.session_state.get("ndvi_out_cluster_tiles", False)
    )

    if run:
        if not ndvi_out_ok:
            st.error("Select at least one output format.")
        elif not ndvi_input_datasets:
            st.warning("Upload at least one study area to get started.")
        else:
            from services import get_job_executor, get_job_store

            store = get_job_store()
            executor = get_job_executor()
            save_gt = st.session_state.get("ndvi_out_geotiff", True)
            save_gj = st.session_state.get("ndvi_out_geojson", False)
            save_gp = st.session_state.get("ndvi_out_gpkg", False)
            save_ct = st.session_state.get("ndvi_out_cluster_tiles", False)
            jobs_started = 0
            validation_errors = []

            # basename -> absolute path map so each submitted job records its
            # study-area location for silent-restart.
            ndvi_path_by_basename = {os.path.basename(p): p for p in ndvi_valid_paths}

            for unit in _ndvi_units():
                fname = unit["key"]
                d = {"raw": unit["raw"]}
                merged = unit["merged"]
                cfg = st.session_state.ndvi_date_configs.get(fname, {})
                # A merged group names its outputs from the group; a single file
                # strips its extension so the monitor title is just the stem.
                if merged:
                    base_name = sanitize_name_for_file(fname)
                    ds_path = None
                    src_paths = [
                        ndvi_path_by_basename.get(fn) for fn in unit["sources"]
                    ]
                    src_fps = [file_size_mtime_fingerprint(pp) for pp in src_paths]
                else:
                    base_name = os.path.splitext(fname)[0]
                    ds_path = ndvi_path_by_basename.get(fname)
                    src_paths, src_fps = [], []
                ds_fp = file_size_mtime_fingerprint(ds_path)
                merge_params = {
                    "merged": merged,
                    "group_name": fname if merged else None,
                    "source_files": list(unit["sources"]),
                    "source_paths": src_paths,
                    "source_fingerprints": src_fps,
                }

                use_ranges = st.session_state.get(
                    f"ndvi_use_ranges_{fname}", cfg.get("use_ranges", True)
                )
                use_specific = st.session_state.get(
                    f"ndvi_use_specific_{fname}", cfg.get("use_specific", False)
                )
                use_column = st.session_state.get(
                    f"ndvi_use_col_{fname}", cfg.get("use_column", False)
                )

                if not any([use_ranges, use_specific, use_column]):
                    validation_errors.append(f"{fname}: No date mode is enabled.")
                    continue

                # --- Date Range jobs ---
                if use_ranges:
                    for i, (s_def, e_def) in enumerate(
                        cfg.get("ranges", [(date(2023, 6, 1), date(2023, 9, 30))])
                    ):
                        start_d = st.session_state.get(f"ndvi_rs_{fname}_{i}", s_def)
                        end_d = st.session_state.get(f"ndvi_re_{fname}_{i}", e_def)
                        if start_d >= end_d:
                            validation_errors.append(
                                f"{fname}: Date range {i + 1} — "
                                "start date must be before end date."
                            )
                            continue
                        output_name = (
                            f"{base_name}_"
                            f"{start_d.strftime('%Y%m%d')}_"
                            f"{end_d.strftime('%Y%m%d')}"
                        )
                        record = store.submit(
                            type="ndvi",
                            name=base_name,
                            params={
                                "fname": fname,
                                "input_path": ds_path,
                                "input_fingerprint": ds_fp,
                                "mode": "range",
                                "start_date": start_d.isoformat(),
                                "end_date": end_d.isoformat(),
                                "cloud_pct": cloud_pct,
                                "resolution": resolution,
                                "buffer_m": buffer_m,
                                "output_name": output_name,
                                "save_geotiff": save_gt,
                                "save_gpkg": save_gp,
                                "save_geojson": save_gj,
                                "save_cluster_tiles": save_ct,
                                "geometry_sha256": geometry_sha256(d["raw"]),
                                **merge_params,
                            },
                        )
                        executor.submit_ndvi_subprocess(
                            record,
                            fname=fname,
                            dataset_data=d,
                            start_date=start_d.isoformat(),
                            end_date=end_d.isoformat(),
                            cloud_pct=cloud_pct,
                            resolution=resolution,
                            buffer_m=buffer_m,
                            output_name=output_name,
                            output_dir=output_dir,
                            save_geotiff=save_gt,
                            save_gpkg=save_gp,
                            save_geojson=save_gj,
                            save_cluster_tiles=save_ct,
                        )
                        jobs_started += 1

                # --- Specific Date jobs ---
                if use_specific:
                    window_days = st.session_state.get(
                        f"ndvi_win_s_{fname}",
                        cfg.get("window_days_specific", 30),
                    )
                    for i, d_def in enumerate(
                        cfg.get("specific_dates", [date(2023, 7, 15)])
                    ):
                        target_date = st.session_state.get(
                            f"ndvi_sd_{fname}_{i}", d_def
                        )
                        start_d = target_date - timedelta(days=window_days)
                        end_d = min(
                            target_date + timedelta(days=window_days),
                            date.today(),
                        )
                        output_name = f"{base_name}_{target_date.strftime('%Y%m%d')}"
                        record = store.submit(
                            type="ndvi",
                            name=base_name,
                            params={
                                "fname": fname,
                                "input_path": ds_path,
                                "input_fingerprint": ds_fp,
                                "mode": "specific",
                                "target_date": target_date.isoformat(),
                                "window_days": window_days,
                                "cloud_pct": cloud_pct,
                                "resolution": resolution,
                                "buffer_m": buffer_m,
                                "output_name": output_name,
                                "save_geotiff": save_gt,
                                "save_gpkg": save_gp,
                                "save_geojson": save_gj,
                                "save_cluster_tiles": save_ct,
                                "geometry_sha256": geometry_sha256(d["raw"]),
                                **merge_params,
                            },
                        )
                        executor.submit_ndvi_subprocess(
                            record,
                            fname=fname,
                            dataset_data=d,
                            start_date=start_d.isoformat(),
                            end_date=end_d.isoformat(),
                            cloud_pct=cloud_pct,
                            resolution=resolution,
                            buffer_m=buffer_m,
                            output_name=output_name,
                            output_dir=output_dir,
                            save_geotiff=save_gt,
                            save_gpkg=save_gp,
                            save_geojson=save_gj,
                            save_cluster_tiles=save_ct,
                        )
                        jobs_started += 1

                # --- Attribute Column (per-year) job ---
                if use_column:
                    date_col = st.session_state.get(f"ndvi_col_{fname}")
                    season_start = int(
                        st.session_state.get(
                            f"ndvi_seas_start_{fname}",
                            cfg.get("season_start_month", 6),
                        )
                    )
                    season_end = int(
                        st.session_state.get(
                            f"ndvi_seas_end_{fname}", cfg.get("season_end_month", 9)
                        )
                    )
                    if not date_col:
                        validation_errors.append(f"{fname}: No date column selected.")
                    elif season_start > season_end:
                        validation_errors.append(
                            f"{fname}: Season start month must be on or before "
                            "the end month."
                        )
                    else:
                        record = store.submit(
                            type="ndvi_column",
                            name=base_name,
                            params={
                                "fname": fname,
                                "input_path": ds_path,
                                "input_fingerprint": ds_fp,
                                "mode": "column",
                                "date_column": date_col,
                                "season_start_month": season_start,
                                "season_end_month": season_end,
                                "cloud_pct": cloud_pct,
                                "resolution": resolution,
                                "buffer_m": buffer_m,
                                "save_geotiff": save_gt,
                                "save_gpkg": save_gp,
                                "save_geojson": save_gj,
                                "save_cluster_tiles": save_ct,
                                "geometry_sha256": geometry_sha256(d["raw"]),
                                **merge_params,
                            },
                        )
                        executor.submit_ndvi_column_subprocess(
                            record,
                            fname=fname,
                            dataset_data=d,
                            date_column=date_col,
                            season_start_month=season_start,
                            season_end_month=season_end,
                            cloud_pct=cloud_pct,
                            resolution=resolution,
                            buffer_m=buffer_m,
                            output_dir=output_dir,
                            save_geotiff=save_gt,
                            save_gpkg=save_gp,
                            save_geojson=save_gj,
                            save_cluster_tiles=save_ct,
                        )
                        jobs_started += 1

            for err in validation_errors:
                st.error(err)

            if jobs_started:
                st.success(
                    f"{jobs_started} job(s) started. "
                    "Monitor progress in the sidebar Job Monitor."
                )
            elif not validation_errors:
                st.info("No new jobs were submitted.")

    st.divider()

    col_ndvi_btm_left, col_ndvi_btm_right = st.columns(2)
    with col_ndvi_btm_left:
        st.subheader("Result Inspector")
        scan_row_l, scan_row_r = st.columns([11, 1])
        with scan_row_l:
            ndvi_scan_clicked = st.button(
                "🔄 Scan Output Folder",
                key="ndvi_scan_folder",
                width="stretch",
            )
        with scan_row_r:
            ndvi_scan_spinner_slot = st.empty()

        if ndvi_scan_clicked:
            import threading as _threading

            holder: dict = {}

            def _scan_worker(out_dir: str, target: dict) -> None:
                try:
                    target["result"] = _ndvi_scan_outputs(out_dir)
                except Exception as exc:  # noqa: BLE001
                    target["error"] = exc

            with ndvi_scan_spinner_slot:
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
                    if base_name not in st.session_state.ndvi_datasets:
                        st.session_state.ndvi_datasets[base_name] = dataset_dict
                        count += 1
                if count > 0:
                    st.success(f"Loaded {count} result(s) from the output folder.")
                else:
                    st.info(
                        "No new NDVI results found "
                        "(looked for *_ndvi.tif, *_ndvi.tiff, *_ndvi.gpkg, "
                        "*_ndvi.geojson)."
                    )

        completed_ds = [
            k
            for k, v in st.session_state.ndvi_datasets.items()
            if v.get("type") == "restored"
        ]
        options = ["All Regions"] + completed_ds
        selected_opt = st.selectbox(
            "Select Result",
            options,
            index=None,
            placeholder="Choose a result...",
            key="ndvi_inspector_select",
        )
        r_opacity = st.slider("Layer Opacity", 0.0, 1.0, 0.7, key="ndvi_op")

        def _ndvi_inspector_has_geojson(sel: str | None) -> bool:
            if not completed_ds:
                return False
            if sel is None:
                return False
            if sel == "All Regions":
                return all(
                    st.session_state.ndvi_datasets[k].get("results") is not None
                    for k in completed_ds
                )
            return (
                st.session_state.ndvi_datasets.get(sel, {}).get("results") is not None
            )

        ndvi_pts_ok = _ndvi_inspector_has_geojson(selected_opt)
        if not ndvi_pts_ok and st.session_state.get("ndvi_show_points"):
            st.session_state.ndvi_show_points = False
        show_points = st.checkbox(
            "Show Sample Points",
            value=False,
            key="ndvi_show_points",
            disabled=not ndvi_pts_ok,
            help=(
                "Requires a GeoJSON next to this GeoTIFF (same base name, "
                "_ndvi.geojson). Enable Save GeoJSON when running NDVI or add the file."
                if not ndvi_pts_ok
                else None
            ),
        )
        val_container = st.empty()

    with col_ndvi_btm_right:
        st.subheader("Results Preview")
        m_ndvi_result = folium.Map(location=[51.0447, -114.0719], zoom_start=10)

        if selected_opt:
            targets = completed_ds if selected_opt == "All Regions" else [selected_opt]
            res_bounds = []

            for ds_name in targets:
                if ds_name not in st.session_state.ndvi_datasets:
                    continue
                ds = st.session_state.ndvi_datasets[ds_name]

                tif_path = ds.get("tif_path") or os.path.join(
                    output_dir, f"{ds_name}_ndvi.tif"
                )
                rendered_from_tif = False
                if tif_path and os.path.exists(tif_path):
                    try:
                        # ``extra_nodata_values=(0,)`` because the NDVI
                        # pipeline writes 0 as an undeclared sentinel at the
                        # polygon edge; without it the polygon's outline
                        # renders as a red halo at the bottom of the ramp.
                        w_lon, s_lat, e_lon, n_lat = (
                            add_mercator_image_overlay_from_file(
                                m_ndvi_result,
                                tif_path,
                                cmap=plt.get_cmap("RdYlGn"),
                                vmin=-0.2,
                                vmax=1.0,
                                extra_nodata_values=(0,),
                                opacity=r_opacity,
                                interactive=True,
                            )
                        )
                        res_bounds.append([w_lon, s_lat, e_lon, n_lat])
                        folium.Rectangle(
                            bounds=[[s_lat, w_lon], [n_lat, e_lon]],
                            color="red",
                            weight=2,
                            fill=False,
                        ).add_to(m_ndvi_result)
                        rendered_from_tif = True
                    except Exception as e:
                        print(f"Viz Error {ds_name}: {e}")

                # GeoPackage / GeoJSON fallback: black points, no tooltips.
                if not rendered_from_tif and ds.get("results") is not None:
                    pts = ds["results"]
                    if pts is not None and not pts.empty:
                        l, btm, r, t = pts.total_bounds
                        res_bounds.append([l, btm, r, t])
                        add_black_point_layer(
                            m_ndvi_result,
                            pts,
                            radius=6,
                            fill_opacity=r_opacity,
                            layer_name=f"{ds_name} NDVI samples",
                        )

                if show_points and ds.get("results") is not None:
                    try:
                        gdf_pts, trim_msg = trim_point_gdf_for_display(ds["results"])
                        if trim_msg:
                            st.warning(f"{ds_name}: {trim_msg}")
                        tooltip_fields = (
                            ["NDVI", "ndvi_date"]
                            if "ndvi_date" in gdf_pts.columns
                            else ["NDVI"]
                        )
                        tooltip_aliases = (
                            ["NDVI:", "Date:"]
                            if "ndvi_date" in gdf_pts.columns
                            else ["NDVI:"]
                        )
                        add_uniform_point_layer(
                            m_ndvi_result,
                            gdf_pts,
                            tooltip_fields=tooltip_fields,
                            tooltip_aliases=tooltip_aliases,
                            geojson_marker=folium.Circle(
                                radius=1,
                                color="blue",
                                fill=True,
                                fill_opacity=1,
                            ),
                            cluster_circle_radius=4,
                            cluster_color="blue",
                            cluster_fill_color="blue",
                            cluster_fill_opacity=1.0,
                            layer_name="NDVI sample points",
                        )
                    except Exception as e:
                        st.error(f"Error rendering sample points: {e}")

            # Only fit bounds when the selection changes — otherwise the
            # user's pan/zoom would be reset on every script rerun.
            last_sel = st.session_state.get("_ndvi_inspector_last_sel")
            if res_bounds and selected_opt != last_sel:
                min_x = min([b[0] for b in res_bounds])
                min_y = min([b[1] for b in res_bounds])
                max_x = max([b[2] for b in res_bounds])
                max_y = max([b[3] for b in res_bounds])
                m_ndvi_result.fit_bounds([[min_y, min_x], [max_y, max_x]])
            st.session_state["_ndvi_inspector_last_sel"] = selected_opt

        map_data = st_folium(
            m_ndvi_result,
            width="100%",
            height=500,
            key="map_ndvi_result",
            returned_objects=["last_clicked"],
        )

        if map_data and map_data.get("last_clicked"):
            lat = map_data["last_clicked"]["lat"]
            lon = map_data["last_clicked"]["lng"]
            val_found = False
            search_targets = (
                completed_ds
                if (selected_opt == "All Regions" or selected_opt is None)
                else [selected_opt]
            )

            for ds_name in search_targets:
                if ds_name not in st.session_state.ndvi_datasets:
                    continue
                ds_entry = st.session_state.ndvi_datasets[ds_name]
                tif_path = ds_entry.get("tif_path") or os.path.join(
                    output_dir, f"{ds_name}_ndvi.tif"
                )
                if tif_path and os.path.exists(tif_path):
                    with rasterio.open(tif_path) as src:
                        try:
                            r, c = src.index(lon, lat)
                            window = rasterio.windows.Window(c, r, 1, 1)
                            val = src.read(1, window=window)
                            if val.size > 0:
                                pixel_val = val[0][0]
                                if (
                                    pixel_val != -9999
                                    and not np.isnan(pixel_val)
                                    and pixel_val != 0
                                ):
                                    val_container.info(
                                        f"**{ds_name}**: NDVI = {pixel_val:.4f} "
                                        f"at ({lat:.4f}, {lon:.4f})"
                                    )
                                    val_found = True
                                    break
                        except Exception:
                            pass
            if not val_found:
                val_container.info(f"No valid NDVI data at ({lat:.4f}, {lon:.4f}).")
