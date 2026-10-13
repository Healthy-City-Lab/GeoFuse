"""The Jobs tab: command-line jobs on this computer and on linked machines.

Reads a job source (``geofuse.jobs.sources``): this computer's state folder, or
a remote's mirror that ``geofuse remote attach`` keeps (``geofuse.jobs.remote``).
The table and the selected job refresh every two seconds in a fragment, so
nothing else on the page reruns. Jobs started in this app stay in the sidebar
Job Monitor.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from geofuse import settings as _settings
from geofuse.cli.commands import _note, _shown_state
from geofuse.jobs.sources import list_outputs
from geofuse.jobs.state import ACTIVE, TERMINAL, Job, StateDir, parse_iso

#: The source that reads this computer's state folder.
LOCAL = "This computer"
_REFRESH_S = 2.0
_LOG_LINES = 200
_ENGINE = {"ndvi_column": "ndvi", "gvi_column": "gvi", "gvi_merge": "gvi"}
_STAGE_ICONS = {"done": "✅", "skipped": "⏭️", "running": "▶️", "failed": "❌"}


# ────────────────────────────────────────────────────────────────────
# Sources
# ────────────────────────────────────────────────────────────────────


def _sources() -> dict:
    """Every source by its label: this computer, then each configured remote."""
    from geofuse import userconfig
    from geofuse.jobs.remote import MirrorSource, remotes
    from geofuse.jobs.sources import LocalSource

    try:
        cfg = userconfig.load()
    except userconfig.ConfigError as exc:
        st.error(f"The GeoFuse configuration could not be read: {exc}")
        cfg = userconfig.Config()
    state = StateDir(cfg.state_dir)
    found: dict = {LOCAL: LocalSource(state)}
    try:
        names = list(remotes(cfg))
    except ValueError as exc:
        st.error(str(exc))
        names = []
    for name in names:
        found[name] = MirrorSource(state, name)
    return found


def _clock(stamp: str | None) -> str | None:
    seconds = parse_iso(stamp)
    if seconds is None:
        return None
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(seconds))


def _render_link(name: str, source) -> None:
    link = source.link()
    if link.get("state") == "local":
        st.caption(f"State folder: `{source.label}`")
        return
    host = link.get("host") or name
    seen = _clock(link.get("seen"))
    if not link["stale"]:
        st.caption(f"Linked to **{host}** · jobs read {seen}")
        return
    when = f"; showing the jobs as of {seen}" if seen else "; no jobs read yet"
    st.warning(
        f"The link to **{name}** is {link['state']}{when}. Hold it open in a "
        f"terminal with `geofuse remote attach {name}`.",
        icon="⚠️",
    )


# ────────────────────────────────────────────────────────────────────
# One job
# ────────────────────────────────────────────────────────────────────


def _matches(job: Job, text: str) -> bool:
    if not text:
        return True
    fields = (job.id, job.name, job.spec.get("kind"), job.spec.get("step"))
    haystack = " ".join(map(str, (*fields, _shown_state(job)))).lower()
    return text.lower() in haystack


def _changed_settings(job: Job) -> dict:
    kind = str(job.spec.get("kind"))
    engine = _ENGINE.get(kind, kind)
    defaults = _settings.defaults(engine) if engine in _settings.SCHEMAS else {}
    settings = job.spec.get("settings") or {}
    return {k: v for k, v in settings.items() if defaults.get(k) != v}


def _bundle(paths) -> str | None:
    for path in paths:
        if Path(str(path)).name == "results_bundle.json":
            return str(path)
    return None


def _act(source, job: Job, **flags: bool) -> None:
    with st.spinner("Sending..."):
        try:
            result = source.control(job.id, **flags)
        except Exception as exc:
            result = f"failed: {exc}"
    st.toast(f"{job.name}: {result}")


def _rerun(source, job: Job) -> None:
    with st.spinner("Sending..."):
        try:
            result = source.rerun(job.id)
        except Exception as exc:
            result = f"failed: {exc}"
    st.toast(f"{job.name}: {result}")


def _render_buttons(source, job: Job) -> None:
    remote = getattr(source, "kind", "local") == "remote"
    cols = st.columns(3, gap="small")
    if job.state in ACTIVE:
        if job.status.get("paused"):
            if cols[0].button(
                "Resume",
                key="job_resume",
                type="primary",
                width="stretch",
                help="Continue from where it paused; nothing queued is lost.",
            ):
                _act(source, job, pause=False)
        elif cols[0].button(
            "Pause",
            key="job_pause",
            width="stretch",
            help="Hold at the next safe point; queued work continues on resume.",
        ):
            _act(source, job, pause=True)
        with cols[1].popover("Cancel", width="stretch"):
            st.markdown(f"Cancel **{job.name}** ({job.id})?")
            if st.button(
                "Cancel job", key="job_cancel_confirm", type="primary", width="stretch"
            ):
                _act(source, job, cancel=True)
        return

    if cols[0].button(
        "Re-run",
        key="job_rerun",
        width="stretch",
        help="Queue this job again from its settings.",
    ):
        _rerun(source, job)
    if remote and job.state == "completed":
        if cols[1].button(
            "Fetch outputs",
            key="job_fetch",
            width="stretch",
            help="Copy this job's output files to this computer over the link.",
        ):
            st.toast(f"{job.name}: {source.fetch(job.id)}")
    if job.spec.get("kind") == "fusion" and job.state == "completed":
        if remote:
            paths = [str(p) for p in source.fetched(job.id)]
        else:
            paths = [f["path"] for f in list_outputs(job.status.get("outputs") or [])]
        bundle = _bundle(paths)
        if cols[2].button(
            "Load results",
            key="job_load_results",
            width="stretch",
            disabled=bundle is None,
            help=(
                "Show this run in the Fusion & Optimization tab's results."
                if bundle
                else "Fetch the outputs first: the results bundle is not here yet."
            ),
        ):
            from job_panel import load_fusion_bundle

            if load_fusion_bundle(bundle):
                st.toast("Loaded: open the Fusion & Optimization tab to read it.")
            else:
                st.toast("The results bundle could not be read.")


def _render_outputs(source, job: Job) -> None:
    remote = getattr(source, "kind", "local") == "remote"
    if remote:
        fetched = source.fetched(job.id)
        listed = job.status.get("outputs") or []
        st.markdown(f"**Outputs** · {len(listed)} recorded on the remote machine")
        if fetched:
            st.caption(f"Fetched to `{fetched[0].parent}`")
            for path in fetched[:50]:
                st.text(str(path))
        return
    files = list_outputs(job.status.get("outputs") or [])
    st.markdown(f"**Outputs** · {len(files)} file(s)")
    if not files:
        planned = job.spec.get("output_dir")
        st.caption(f"None yet; written to `{planned}`." if planned else "None.")
        return
    frame = pd.DataFrame(
        [{"File": f["path"], "MB": round(f["size"] / 1e6, 2)} for f in files]
    )
    height = min(300, 40 + 35 * len(files))
    st.dataframe(frame, hide_index=True, width="stretch", height=height)


def _render_job(name: str, source, job: Job) -> None:
    spec, status = job.spec, job.status
    with st.container(border=True):
        st.markdown(f"**{job.name}** · `{job.id}`")
        st.caption(
            f"{spec.get('kind')} · step {spec.get('step')} · run {spec.get('run_id')}"
        )
        shown = _shown_state(job)
        progress = float(status.get("progress") or 0)
        if job.state in ACTIVE:
            text = status.get("text") or shown
            st.progress(min(max(progress, 0.0), 1.0), text=f"{shown} · {text}")
        else:
            st.markdown(f"Status: **{shown}**")
        times = [
            f"{label} {_clock(status.get(key) or spec.get(key))}"
            for label, key in (
                ("submitted", "submitted_at"),
                ("started", "started_at"),
                ("ended", "completed_at"),
            )
            if _clock(status.get(key) or spec.get(key))
        ]
        if times:
            st.caption(" · ".join(times))
        if status.get("error") and job.state in TERMINAL:
            st.error(str(status["error"]).splitlines()[0])
            if status.get("error_detail"):
                with st.expander("Error trace"):
                    st.code(status["error_detail"], language=None)

        stages = (status.get("stage_ledger") or {}).get("stages") or []
        if stages:
            st.markdown("**Stages**")
            for stage in stages:
                icon = _STAGE_ICONS.get(stage.get("status"), "⬜")
                label = stage.get("label") or stage.get("key")
                extra = ""
                if stage.get("status") == "running" and stage.get("progress"):
                    extra = f" · {100 * float(stage['progress']):.0f}%"
                if stage.get("message"):
                    extra += f" · {stage['message']}"
                st.markdown(f"{icon} {label}{extra}")

        changed = _changed_settings(job)
        st.markdown("**Settings** (other than the defaults)")
        if changed:
            st.code("\n".join(f"{k} = {v}" for k, v in changed.items()), language=None)
        else:
            st.caption("All defaults.")

        _render_outputs(source, job)

        text, _offset = source.read_log(job.id, None)
        lines = text.splitlines()[-_LOG_LINES:]
        st.markdown(f"**Log** · last {len(lines)} line(s)")
        st.code("\n".join(lines) or "(empty)", language=None, height=280)

        _render_buttons(source, job)


# ────────────────────────────────────────────────────────────────────
# The tab
# ────────────────────────────────────────────────────────────────────


@st.fragment(run_every=_REFRESH_S)
def _render_jobs() -> None:
    sources = _sources()
    name = st.session_state.get("job_source") or LOCAL
    source = sources.get(name) or sources[LOCAL]
    _render_link(name, source)
    try:
        jobs = list(reversed(source.jobs()))
    except Exception as exc:
        st.error(f"Could not read the jobs: {exc}")
        return
    finished = st.session_state.get("job_show_finished", True)
    text = (st.session_state.get("job_filter") or "").strip()
    shown = [
        j for j in jobs if (finished or j.state not in TERMINAL) and _matches(j, text)
    ]
    active = sum(j.state in ACTIVE for j in jobs)
    st.caption(f"{active} active of {len(jobs)} · showing {len(shown)}")
    if not shown:
        st.info(
            "No jobs to show. Start some with `geofuse submit RUNFILE` or "
            "`geofuse ndvi | gvi | fusion ...`."
        )
        return
    frame = pd.DataFrame(
        [
            {
                "ID": j.id,
                "Kind": str(j.spec.get("kind", "")),
                "Name": j.name,
                "Status": _shown_state(j),
                "Progress": float(j.status.get("progress") or 0),
                "Step": str(j.spec.get("step", "")),
                "Note": _note(j),
            }
            for j in shown
        ]
    )
    st.dataframe(
        frame,
        hide_index=True,
        width="stretch",
        height=min(420, 40 + 35 * len(shown)),
        column_config={
            "Progress": st.column_config.ProgressColumn(
                "Progress", min_value=0.0, max_value=1.0, format="percent"
            )
        },
    )
    by_id = {j.id: j for j in shown}
    if st.session_state.get("job_selected") not in by_id:
        st.session_state["job_selected"] = shown[0].id
    st.selectbox(
        "Job",
        list(by_id),
        key="job_selected",
        format_func=lambda i: f"{by_id[i].name} · {i} · {_shown_state(by_id[i])}",
    )
    _render_job(name, source, by_id[st.session_state["job_selected"]])


def render() -> None:
    st.header("Jobs")
    st.caption(
        "Jobs run with the `geofuse` command on this computer, and on other "
        "machines linked with `geofuse remote attach`. Jobs started in this app "
        "are in the sidebar Job Monitor."
    )
    sources = _sources()
    if st.session_state.get("job_source") not in sources:
        st.session_state["job_source"] = LOCAL
    st.session_state.setdefault("job_show_finished", True)
    with st.container(border=True):
        c_source, c_filter, c_finished = st.columns([2, 3, 1.4])
        c_source.selectbox("Source", list(sources), key="job_source")
        c_filter.text_input(
            "Filter",
            key="job_filter",
            placeholder="id, name, kind, step or status",
        )
        c_finished.checkbox("Show finished jobs", key="job_show_finished")
    _render_jobs()
