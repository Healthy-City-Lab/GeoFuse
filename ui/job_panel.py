"""Shared sidebar job-monitor panel.

Renders one bordered card per job (active or terminal) with progress,
status, sub-progress brackets, parameter details, and log access. Engine
tabs (``ui/tabs/gvi.py``, ``ui/tabs/ndvi.py``, ...) call
:func:`render_sidebar_job_monitor` to drop the panel into
``st.sidebar``. The panel itself is engine-agnostic and pulls all state
from the shared :class:`JobStore`.

This module is the home for cross-engine job UI helpers — anything that
displays job records, formats sub-progress, or renders logs belongs here
rather than inside a single engine's tab module.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import streamlit as st
from helpers import RESTART_SESSION_KEY
from services import (
    ansi_log_lines_to_html,
    get_job_store,
    open_path_in_default_editor,
)

from geofuse.logger import get_job_log_lines, get_job_log_path

_ACTIVE = {"queued", "running"}
_TERMINAL = {"completed", "error", "cancelled", "interrupted"}
_EMOJI = {
    "fusion": "🔀",
    "ndvi": "🛰️",
    "ndvi_column": "🛰️",
    "gvi": "🌳",
}
_TERMINAL_LABELS = {
    "completed": "✅ Completed",
    "cancelled": "🚫 Cancelled",
    "interrupted": "⏸️ Interrupted (process restarted)",
    "error": "❌ Error",
}

_RESTART_ELIGIBLE_TYPES = {"gvi", "ndvi", "ndvi_column", "fusion"}
_RESTART_ELIGIBLE_STATUSES = {"interrupted", "cancelled", "error"}


def _load_fusion_results_into_session(rec) -> bool:
    """Hydrate ``st.session_state`` from a completed fusion job's bundle.

    Returns ``True`` when the load succeeds and the caller should rerun,
    ``False`` when the record has no usable results payload.
    """
    if rec.type != "fusion" or rec.status != "completed":
        return False
    results = rec.extra.get("results") if rec.extra else None
    if results is None:
        return False
    st.session_state.fusion_engine = rec.extra.get("engine")
    st.session_state.fusion_engines_by_target = rec.extra.get("engines_by_target") or {}
    st.session_state.fusion_results = results
    return True


# Staged-resume ledger glyphs (see geofuse.jobs.stage_ledger).
_STAGE_ICONS = {
    "done": "✅",
    "skipped": "⏭️",
    "running": "▶️",
    "failed": "❌",
    "pending": "⬜",
}
_STAGE_FINISHED = {"done", "skipped"}


def _fmt_timestamp(iso: str) -> str:
    """Parse a UTC ISO timestamp string and return a local-time string."""
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            ts = datetime.strptime(iso, fmt).replace(tzinfo=UTC)
            return ts.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    return iso.replace("T", " ")


def _render_details(rec) -> None:
    """Key/value summary of job parameters inside the Details expander."""
    p = rec.params or {}
    if rec.type == "gvi":
        st.write(f"**Grid step:** {p.get('step', '?')} m")
        st.write(f"**Buffer:** {p.get('buffer', '?')} m")
        st.write(
            f"**Save panos / masks:** "
            f"{bool(p.get('save_panos'))} / {bool(p.get('save_masks'))}"
        )
        st.write(
            f"**Outputs:** GeoPackage={bool(p.get('save_gpkg', True))} · "
            f"GeoTIFF={bool(p.get('save_geotiff'))} · "
            f"GeoJSON={bool(p.get('save_geojson'))}"
        )
        st.write(f"**Street View API key:** {'yes' if p.get('has_api_key') else 'no'}")
    elif rec.type in ("ndvi", "ndvi_column"):
        mode = p.get("mode", "?")
        st.write(f"**Mode:** {mode}")
        if mode == "range":
            st.write(
                f"**Date range:** {p.get('start_date', '?')} → "
                f"{p.get('end_date', '?')}"
            )
        elif mode == "specific":
            st.write(
                f"**Target date:** {p.get('target_date', '?')} "
                f"(window ±{p.get('window_days', '?')} d)"
            )
        elif mode == "column":
            st.write(
                f"**Date column:** {p.get('date_column', '?')} "
                f"(window ±{p.get('window_days', '?')} d)"
            )
        st.write(f"**Cloud max:** {p.get('cloud_pct', '?')}%")
        st.write(f"**Resolution:** {p.get('resolution', '?')} m")
        st.write(f"**Buffer:** {p.get('buffer_m', '?')} m")
        st.write(
            f"**Outputs:** GeoTIFF={bool(p.get('save_geotiff'))} · "
            f"GeoPackage={bool(p.get('save_gpkg'))} · "
            f"GeoJSON={bool(p.get('save_geojson'))} · "
            f"ClusterTiles={bool(p.get('save_cluster_tiles'))}"
        )
    elif rec.type == "fusion":
        st.write(
            f"**Trials:** {p.get('n_trials', '?')} "
            f"(startup {p.get('n_startup_trials', '?')})"
        )
        st.write(f"**Objective:** {p.get('objective_metric', '?')}")
        st.write(f"**CGI formula:** `{p.get('cgi_formula') or 'weighted_average'}`")
        covs = p.get("covariate_columns") or []
        st.write(f"**Covariates:** {', '.join(covs) if covs else '—'}")
        standalones = p.get("standalone_channels") or []
        _ch_disp = {"veg": "Vegetation", "terrain": "Terrain", "ndvi": "NDVI"}
        st.write(
            f"**Standalone metrics:** "
            f"{', '.join(_ch_disp.get(s, s) for s in standalones) if standalones else '—'}"
        )
        st.write(
            f"**Sampler:** {p.get('sampler_type', '?')} · "
            f"**Pruner:** {p.get('pruner_type', '?')}"
        )
        outcomes = p.get("outcome_columns") or []
        st.write(
            f"**Outcomes:** {len(outcomes)}{' — ' + ', '.join(outcomes) if outcomes else ''}"
        )
        st.write(f"**Resume study:** {bool(p.get('resume_existing_study', True))}")
    if rec.output_paths:
        st.write("**Output files:**")
        for path in rec.output_paths:
            st.code(path, language=None)
    if rec.submitted_at:
        st.caption("Submitted at:")
        st.caption(_fmt_timestamp(rec.submitted_at))
    if rec.completed_at:
        st.caption("Completed at:")
        st.caption(_fmt_timestamp(rec.completed_at))


def _render_stage_ledger(rec) -> None:
    """Render the staged-resume checklist for any job that carries a ledger.

    Driven by the runner via ``ctx.update_stage_ledger`` (see
    ``geofuse.jobs.stage_ledger``) and persisted on the ``JobStore`` record, so
    the checklist survives process death — an interrupted job re-loads showing
    exactly where it stopped, which is the same signal the re-run uses to
    explain "continue from here" to the user.
    """
    ledger = rec.stage_ledger or {}
    stages = ledger.get("stages") or []
    if not stages:
        return
    done_count = sum(1 for s in stages if s.get("status") in _STAGE_FINISHED)
    total = len(stages)
    expanded = rec.status in _ACTIVE or rec.status in {
        "interrupted",
        "cancelled",
        "error",
    }
    with st.expander(f"Stages ({done_count}/{total})", expanded=expanded):
        for s in stages:
            icon = _STAGE_ICONS.get(s.get("status", "pending"), "⬜")
            label = s.get("label") or s.get("key", "")
            msg = s.get("message") or ""
            line = f"{icon} {label}"
            if msg:
                line += f" — _{msg}_"
            st.markdown(line)


def _render_logs(rec_id: str) -> None:
    """Render the active-job log tail from the in-memory per-job deque."""
    lines = get_job_log_lines(rec_id)
    if not lines:
        st.caption("(no log output captured yet)")
        return
    st.markdown(ansi_log_lines_to_html(lines), unsafe_allow_html=True)


def _render_job_card(rec, store) -> None:
    """One bordered card for a single job."""
    with st.container(border=True):
        emoji = _EMOJI.get(rec.type, "•")
        st.markdown(f"### {emoji} {rec.name}")

        st.progress(float(rec.progress))

        if rec.status in _TERMINAL:
            label = _TERMINAL_LABELS.get(rec.status, rec.status.capitalize())
        else:
            label = rec.status_text or rec.status.capitalize()
        st.caption(label)

        bracket = rec.extra.get("ndvi_tile_bracket")
        if rec.type in ("ndvi", "ndvi_column") and bracket:
            st.caption(f"Tiles flushed: {bracket}")

        gvi_progress = rec.extra.get("gvi_progress")
        if gvi_progress:
            st.progress(
                gvi_progress["percent"] / 100,
                text=(f"{gvi_progress['current']:,} / " f"{gvi_progress['total']:,}"),
            )

        _render_stage_ledger(rec)

        with st.expander("Details", expanded=False):
            _render_details(rec)

        if rec.status in _TERMINAL:
            log_path = get_job_log_path(rec.id)
            have_file = os.path.isfile(log_path)
            if st.button(
                "Open log file",
                key=f"openlog_{rec.id}",
                use_container_width=True,
                disabled=not have_file,
                help=(log_path if have_file else "Log file not found on disk."),
            ):
                try:
                    open_path_in_default_editor(log_path)
                except Exception as e:
                    st.error(f"Could not open log file: {e}")

            if (
                rec.type == "fusion"
                and rec.status == "completed"
                and rec.extra.get("results") is not None
            ):
                if st.button(
                    "Load results",
                    key=f"loadres_{rec.id}",
                    use_container_width=True,
                    help=(
                        "Replace the active result overview with this job's "
                        "bundle (composite map, robust trials, standalone "
                        "studies, etc.)."
                    ),
                ):
                    if _load_fusion_results_into_session(rec):
                        st.rerun()
        else:
            with st.expander("Logs", expanded=False):
                _render_logs(rec.id)

        error_detail = rec.extra.get("error_detail")
        if rec.error:
            with st.expander("Error trace"):
                st.code(error_detail or rec.error)

        if rec.status in _ACTIVE:
            st.button(
                "Cancel",
                key=f"cancel_{rec.id}",
                on_click=store.request_cancel,
                args=(rec.id,),
            )
        else:
            restart_eligible = (
                rec.type in _RESTART_ELIGIBLE_TYPES
                and rec.status in _RESTART_ELIGIBLE_STATUSES
            )
            btn_cols = st.columns(2, gap="medium")
            if restart_eligible:
                restart_col, dismiss_col = btn_cols[0], btn_cols[1]
            else:
                restart_col, dismiss_col = None, btn_cols[0]
            if restart_col is not None:
                with restart_col:
                    if st.button(
                        "🔄",
                        key=f"restart_{rec.id}",
                        use_container_width=True,
                        help="Restart — re-upload the original input geometry.",
                    ):
                        st.session_state[RESTART_SESSION_KEY] = rec.id
                        st.rerun()
            with dismiss_col:
                st.button(
                    "🗑️",
                    key=f"del_{rec.id}",
                    use_container_width=True,
                    on_click=store.purge,
                    args=(rec.id,),
                    help="Dismiss — remove this job from history.",
                )


def render_sidebar_job_monitor() -> None:
    """Render the all-jobs monitor inside ``st.sidebar``.

    Pulls every record from the shared :class:`JobStore` and renders one
    card per job. The fragment reruns once per second to pick up
    in-flight progress updates without forcing a full page rerun.
    """
    store = get_job_store()

    @st.fragment(run_every=1)
    def _fragment():
        st.header("Job Monitor")

        h = store.health()
        st.caption(
            f"Active: {h['active']} · Stuck: {h['stuck']} · "
            f"Errors (1h): {h['errored_last_hour']}"
        )

        all_recs = store.list_all()
        active = sorted(
            [r for r in all_recs if r.status in _ACTIVE],
            key=lambda r: r.submitted_at or r.id,
        )
        terminal = sorted(
            [r for r in all_recs if r.status in _TERMINAL],
            key=lambda r: r.completed_at or r.updated_at or "",
            reverse=True,
        )
        ordered = active + terminal

        if not ordered:
            st.info("No active jobs.")
            return

        for rec in ordered:
            _render_job_card(rec, store)

    with st.sidebar:
        _fragment()
