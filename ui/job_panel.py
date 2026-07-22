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

import json
import os
from datetime import UTC, datetime

import streamlit as st
from helpers import RESTART_SESSION_KEY
from services import (
    ansi_log_lines_to_html,
    get_job_store,
    open_path_in_default_editor,
)

from geofuse.jobs import job_ui_refresh_s
from geofuse.logger import get_job_log_lines, get_job_log_path

_ACTIVE = {"queued", "running"}
_TERMINAL = {"completed", "error", "cancelled", "interrupted"}
_EMOJI = {
    "fusion": "🔀",
    "ndvi": "🛰️",
    "ndvi_column": "🛰️",
    "gvi": "🌳",
    "gvi_column": "🌳",
}
_TERMINAL_LABELS = {
    "completed": "✅ Completed",
    "cancelled": "🚫 Cancelled",
    "interrupted": "⏸️ Interrupted (process restarted)",
    "error": "❌ Error",
}

_RESTART_ELIGIBLE_TYPES = {"gvi", "gvi_column", "ndvi", "ndvi_column", "fusion"}
_RESTART_ELIGIBLE_STATUSES = {"interrupted", "cancelled", "error", "completed"}


def _fusion_results_bundle_path(rec) -> str | None:
    """Path to the job's on-disk ``results_bundle.json`` if one exists."""
    for p in rec.output_paths or []:
        try:
            if os.path.basename(str(p)) == "results_bundle.json" and os.path.isfile(p):
                return str(p)
        except Exception:
            continue
    return None


def _fusion_results_loadable(rec) -> bool:
    """True when a completed fusion job can hydrate the results view.

    Either the live in-memory payload survives (same process) or the persisted
    ``results_bundle.json`` is still on disk (after a Streamlit restart, where
    ``rec.extra`` is gone). Gates the "Load results" button so it stays visible
    once the process recycles.
    """
    if rec.type != "fusion" or rec.status != "completed":
        return False
    if rec.extra and rec.extra.get("results") is not None:
        return True
    return _fusion_results_bundle_path(rec) is not None


def _load_fusion_results_into_session(rec) -> bool:
    """Hydrate ``st.session_state`` from a completed fusion job's bundle.

    Prefers the live in-memory payload (``rec.extra``); when that's gone — e.g.
    after a Streamlit restart, since ``extra`` isn't persisted — it falls back
    to the on-disk ``results_bundle.json`` recorded in ``rec.output_paths``,
    rehydrating the results view without the engine objects (the composite map
    viewer reads its GeoTIFFs from disk and skips the target overlay when no
    engine is present). Returns ``True`` when a load succeeds.
    """
    if rec.type != "fusion" or rec.status != "completed":
        return False
    results = rec.extra.get("results") if rec.extra else None
    if results is not None:
        st.session_state.fusion_engine = rec.extra.get("engine")
        st.session_state.fusion_engines_by_target = (
            rec.extra.get("engines_by_target") or {}
        )
        st.session_state.fusion_results = results
        return True

    # Disk fallback: find the persisted results bundle among the job's outputs.
    bundle_path = _fusion_results_bundle_path(rec)
    if bundle_path is None:
        return False
    try:
        with open(bundle_path, encoding="utf-8") as f:
            disk_results = json.load(f)
    except Exception:
        return False
    st.session_state.fusion_engine = None
    st.session_state.fusion_engines_by_target = {}
    st.session_state.fusion_results = disk_results
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
    if p.get("merged"):
        srcs = p.get("source_files") or []
        st.write(f"**Merged from {len(srcs)} files:** {', '.join(srcs)}")
    if rec.type in ("gvi", "gvi_column"):
        st.write(f"**Grid step:** {p.get('step', '?')} m")
        st.write(f"**Buffer:** {p.get('buffer', '?')} m")
        if rec.type == "gvi_column":
            max_diff = p.get("max_year_diff")
            st.write(
                f"**Per year** from column `{p.get('date_column', '?')}`"
                + (f" (±{max_diff} yr max)" if max_diff is not None else "")
            )
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
            sm = p.get("season_start_month")
            em = p.get("season_end_month")
            season = f"months {sm}–{em}" if sm and em else "?"
            st.write(
                f"**Year column:** {p.get('date_column', '?')} " f"(per year, {season})"
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
            f"**Stability selection:** {p.get('n_bootstraps', '?')} bootstraps × "
            f"{p.get('n_trials_per_bootstrap', '?')} trials/bootstrap "
            f"(min {p.get('min_cell_count', '?')}/cell)"
        )
        st.write(f"**Objective:** {p.get('objective_metric', '?')}")
        st.write(f"**CGI formula:** `{p.get('cgi_formula') or 'weighted_average'}`")
        covs = p.get("covariate_columns") or []
        cov_types = p.get("covariate_types") or {}

        def _cov_disp(c: str) -> str:
            return f"{c} ({'cat' if str(cov_types.get(c)).lower() == 'categorical' else 'num'})"

        st.write(
            f"**Covariates:** {', '.join(_cov_disp(c) for c in covs) if covs else '—'}"
        )
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
        # Group multi-outcome ledgers by their "<label>::" key prefix. Finished
        # or not-yet-started outcomes collapse to one summary line and only the
        # active outcome shows per-stage detail — a 10-outcome run renders a
        # handful of lines per tick instead of ~110, which is what makes the
        # foreground monitor cheap.
        groups: dict[str | None, list] = {}
        order: list[str | None] = []
        for s in stages:
            key = s.get("key", "")
            label = key.split("::", 1)[0] if "::" in key else None
            if label not in groups:
                groups[label] = []
                order.append(label)
            groups[label].append(s)

        multi = any(label is not None for label in order)
        for label in order:
            grp = groups[label]
            if not multi:
                for s in grp:
                    _render_stage_line(s)
                continue
            g_done = sum(1 for s in grp if s.get("status") in _STAGE_FINISHED)
            g_total = len(grp)
            g_running = any(s.get("status") == "running" for s in grp)
            g_active = g_running or (0 < g_done < g_total)
            if not g_active:
                icon = "✅" if g_done == g_total else "⬜"
                st.markdown(f"{icon} **{label}** — {g_done}/{g_total} stages")
            else:
                st.markdown(f"**{label}** — {g_done}/{g_total} stages")
                for s in grp:
                    _render_stage_line(s)


def _render_stage_line(s: dict) -> None:
    """Render one ledger stage as an icon + label (+ optional message)."""
    icon = _STAGE_ICONS.get(s.get("status", "pending"), "⬜")
    label = s.get("label") or s.get("key", "")
    msg = s.get("message") or ""
    line = f"{icon} {label}"
    if msg:
        line += f" — _{msg}_"
    st.markdown(line)


# Cache the ANSI→HTML conversion of a job's log tail so the per-second-ish
# fragment rerun doesn't re-run the regex over ~100 lines every tick. Keyed on
# (job id, line count, last line) — the tail only grows, so that triple changes
# exactly when the rendered HTML would. One entry per job (process-wide).
_LOG_HTML_CACHE: dict[str, tuple[tuple[int, str], str]] = {}


def _render_logs(rec_id: str) -> None:
    """Render the active-job log tail from the in-memory per-job deque."""
    lines = get_job_log_lines(rec_id)
    if not lines:
        st.caption("(no log output captured yet)")
        return
    cache_key = (len(lines), lines[-1])
    cached = _LOG_HTML_CACHE.get(rec_id)
    if cached is None or cached[0] != cache_key:
        cached = (cache_key, ansi_log_lines_to_html(lines))
        _LOG_HTML_CACHE[rec_id] = cached
    st.markdown(cached[1], unsafe_allow_html=True)


def _render_job_card(rec, store) -> None:
    """One bordered card for a single job."""
    with st.container(border=True):
        emoji = _EMOJI.get(rec.type, "•")
        st.markdown(f"### {emoji} {rec.name}")

        st.progress(float(rec.progress))

        if rec.status in _TERMINAL:
            label = _TERMINAL_LABELS.get(rec.status, rec.status.capitalize())
        elif rec.pause_event.is_set():
            label = f"⏸️ Paused — {rec.status_text or 'holding'}"
        else:
            label = rec.status_text or rec.status.capitalize()
        st.caption(label)

        # Workload scale: how many points in total, and how they split by year.
        total_points = rec.extra.get("total_points")
        if total_points:
            st.caption(f"**{int(total_points):,}** sampling points total")
            breakdown = rec.extra.get("point_breakdown") or []
            if len(breakdown) > 1:
                st.caption(
                    "Per year — "
                    + " · ".join(
                        f"**{b['label']}**: {int(b['points']):,}" for b in breakdown
                    )
                )

        bracket = rec.extra.get("ndvi_tile_bracket")
        if rec.type in ("ndvi", "ndvi_column") and bracket:
            st.caption(f"Tiles flushed: {bracket}")

        gvi_progress = rec.extra.get("gvi_progress")
        if gvi_progress:
            st.progress(
                gvi_progress["percent"] / 100,
                text=(f"{gvi_progress['current']:,} / " f"{gvi_progress['total']:,}"),
            )

        fusion_progress = rec.extra.get("fusion_study_progress")
        if rec.type == "fusion" and rec.status not in _TERMINAL and fusion_progress:
            st.progress(
                min(1.0, fusion_progress["percent"] / 100),
                text=(
                    f"{fusion_progress['study']} · "
                    f"{fusion_progress['current']:,} / "
                    f"{fusion_progress['total']:,} trials"
                ),
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
                width="stretch",
                disabled=not have_file,
                help=(log_path if have_file else "Log file not found on disk."),
            ):
                try:
                    open_path_in_default_editor(log_path)
                except Exception as e:
                    st.error(f"Could not open log file: {e}")

            if _fusion_results_loadable(rec):
                if st.button(
                    "Load results",
                    key=f"loadres_{rec.id}",
                    width="stretch",
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
            pause_col, cancel_col = st.columns(2, gap="small")
            paused = rec.pause_event.is_set()
            with pause_col:
                if paused:
                    st.button(
                        "Resume",
                        key=f"resume_{rec.id}",
                        on_click=store.request_resume,
                        args=(rec.id,),
                        width="stretch",
                        type="primary",
                        help="Continue from where it paused — nothing queued is lost.",
                    )
                else:
                    st.button(
                        "Pause",
                        key=f"pause_{rec.id}",
                        on_click=store.request_pause,
                        args=(rec.id,),
                        width="stretch",
                        help=(
                            "Hold at the next safe point. Work already queued is "
                            "kept and continues on resume."
                        ),
                    )
            with cancel_col:
                st.button(
                    "Cancel",
                    key=f"cancel_{rec.id}",
                    on_click=store.request_cancel,
                    args=(rec.id,),
                    width="stretch",
                )
        else:
            restart_eligible = (
                rec.type in _RESTART_ELIGIBLE_TYPES
                and rec.status in _RESTART_ELIGIBLE_STATUSES
            )
            btn_cols = st.columns(2, gap="small")
            if restart_eligible:
                restart_col, dismiss_col = btn_cols[0], btn_cols[1]
            else:
                restart_col, dismiss_col = None, btn_cols[0]
            if restart_col is not None:
                with restart_col:
                    if st.button(
                        "🔄",
                        key=f"restart_{rec.id}",
                        width="stretch",
                        help=(
                            "Re-run — reuse the finished result or recalculate "
                            "from scratch."
                            if rec.status == "completed"
                            else "Restart — resume from where this job stopped."
                        ),
                    ):
                        st.session_state[RESTART_SESSION_KEY] = rec.id
                        st.rerun()
            with dismiss_col:
                st.button(
                    "🗑️",
                    key=f"del_{rec.id}",
                    width="stretch",
                    on_click=store.purge,
                    args=(rec.id,),
                    help="Dismiss — remove this job from history.",
                )


def render_sidebar_job_monitor() -> None:
    """Render the all-jobs monitor inside ``st.sidebar``.

    Pulls every record from the shared :class:`JobStore` and renders one
    card per job. Each rerun re-renders every card (regex over each job's log
    tail included), so this foreground work competes with running jobs when the
    tab is visible. The refresh interval therefore matches the active engines'
    backend cadence via :func:`job_ui_refresh_s` — e.g. 15 s while only a GVI
    job runs, faster when a more frequent engine is active — so the monitor
    never refreshes faster than there is progress to show.
    """
    store = get_job_store()
    active_types = {r.type for r in store.list_all() if r.status in _ACTIVE}
    refresh_s = job_ui_refresh_s(active_types)

    @st.fragment(run_every=refresh_s)
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
