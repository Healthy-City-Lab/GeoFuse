"""``geofuse watch``: a full-screen monitor of the jobs in a state folder.

::

    GeoFuse jobs · <state folder> · supervisor running
    ID  KIND  NAME  STATUS  PROG  ETA  STEP  NOTE            the job table
    stages, settings, outputs  │  the selected job's log
    p pause · r resume · c cancel · a all/active · / filter · l log ·
    s source · q quit

The screen reads its jobs through a source (``geofuse.jobs.sources``: a state
folder on this machine, or a remote one mirrored here by ``geofuse remote
attach``) once a second; ``s`` moves to the next source. A remote's title
says when its link is down and how old its jobs are. Leaving the screen leaves
every job running.
"""

from __future__ import annotations

import time

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Footer, Header, Input, RichLog, Static

from geofuse import settings as _settings
from geofuse.cli.commands import _fmt_progress, _note, _shown_state
from geofuse.jobs.sources import JobSource, LocalSource  # noqa: F401
from geofuse.jobs.state import ACTIVE, TERMINAL, Job, parse_iso

_STATE_STYLES = {
    "running": "bold cyan",
    "completed": "green",
    "queued": "yellow",
    "waiting": "yellow",
    "paused": "magenta",
    "error": "bold red",
    "interrupted": "red",
    "cancelled": "dim",
    "skipped": "dim",
}
_STAGE_MARKS = {
    "done": ("✓", "green"),
    "running": ("▸", "bold cyan"),
    "failed": ("✗", "red"),
    "skipped": ("–", "dim"),
}
_ENGINE = {"ndvi_column": "ndvi", "gvi_column": "gvi", "gvi_merge": "gvi"}
_COLUMNS = ("ID", "KIND", "NAME", "STATUS", "PROG", "ETA", "STEP", "NOTE")


# ────────────────────────────────────────────────────────────────────
# Cells and details
# ────────────────────────────────────────────────────────────────────


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def eta(job: Job, now: float | None = None) -> str:
    """Time left, from the progress so far; blank until there is some."""
    progress = float(job.status.get("progress") or 0)
    started = parse_iso(job.status.get("started_at"))
    if job.state != "running" or started is None or not 0.02 <= progress < 1:
        return ""
    elapsed = (now or time.time()) - started
    return _duration(elapsed * (1 - progress) / progress)


def row_cells(job: Job) -> tuple:
    shown = _shown_state(job)
    return (
        job.id,
        str(job.spec.get("kind", "")),
        job.name[:28],
        Text(shown, style=_STATE_STYLES.get(shown, "")),
        _fmt_progress(job),
        eta(job),
        str(job.spec.get("step", "")),
        _note(job),
    )


def details(job: Job) -> Text:
    """The selected job: where it is, its stages, settings and outputs."""
    spec, st = job.spec, job.status
    out = Text()
    out.append(f"{job.name}\n", style="bold")
    out.append(
        f"{spec.get('kind')} · step {spec.get('step')} · run {spec.get('run_id')}\n",
        style="dim",
    )
    shown = _shown_state(job)
    out.append(shown, style=_STATE_STYLES.get(shown, ""))
    progress = _fmt_progress(job).strip()
    out.append(f" {progress}" if progress else "")
    if st.get("text") and job.state in ACTIVE:
        out.append(f" · {st['text']}")
    out.append("\n")
    times = []
    for label, key in (
        ("submitted", "submitted_at"),
        ("started", "started_at"),
        ("ended", "completed_at"),
    ):
        stamp = parse_iso(st.get(key) or spec.get(key))
        if stamp is not None:
            times.append(f"{label} {time.strftime('%H:%M:%S', time.localtime(stamp))}")
    if times:
        out.append(" · ".join(times) + "\n", style="dim")
    if st.get("error") and job.state in TERMINAL:
        out.append(f"\n{st['error']}\n", style="red")

    stages = (st.get("stage_ledger") or {}).get("stages") or []
    if stages:
        out.append("\nStages\n", style="bold")
        for stage in stages:
            mark, style = _STAGE_MARKS.get(stage.get("status"), ("·", "dim"))
            out.append(f" {mark} ", style=style)
            out.append(str(stage.get("label") or stage.get("key")))
            if stage.get("status") == "running" and stage.get("progress"):
                out.append(f"  {100 * float(stage['progress']):.0f}%")
            if stage.get("message"):
                out.append(f"  {stage['message']}", style="dim")
            out.append("\n")

    engine = _ENGINE.get(str(spec.get("kind")), str(spec.get("kind")))
    defaults = _settings.defaults(engine) if engine in _settings.SCHEMAS else {}
    changed = {
        k: v for k, v in (spec.get("settings") or {}).items() if defaults.get(k) != v
    }
    out.append("\nSettings", style="bold")
    out.append(" (other than the defaults)\n", style="dim")
    for key, value in changed.items():
        out.append(f" {key} = {value}\n")
    if not changed:
        out.append(" all defaults\n", style="dim")

    outputs = st.get("outputs") or []
    out.append("\nOutputs\n", style="bold")
    for path in outputs:
        out.append(f" {path}\n")
    if not outputs:
        planned = spec.get("output_dir")
        out.append(f" none yet; written to {planned}\n" if planned else " none\n")
    return out


# ────────────────────────────────────────────────────────────────────
# Screens
# ────────────────────────────────────────────────────────────────────


class ConfirmScreen(ModalScreen[bool]):
    """A yes / no question; ``y`` / ``n`` / Escape answer it."""

    BINDINGS = [
        Binding("y", "answer(True)", "Yes"),
        Binding("n", "answer(False)", "No"),
        Binding("escape", "answer(False)", "No"),
    ]

    def __init__(self, question: str, yes: str):
        super().__init__()
        self.question = question
        self.yes = yes

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm"):
            yield Static(self.question)
            with Horizontal(id="confirm-buttons"):
                yield Button(f"{self.yes} (y)", variant="error", id="yes")
                yield Button("Keep it (n)", id="no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    def action_answer(self, answer: bool) -> None:
        self.dismiss(answer)


class WatchApp(App):
    """The monitor (see the module docstring)."""

    CSS_PATH = "watch.tcss"
    TITLE = "GeoFuse jobs"
    BINDINGS = [
        Binding("p", "pause", "Pause"),
        Binding("r", "resume", "Resume"),
        Binding("c", "cancel", "Cancel"),
        Binding("a", "toggle_all", "All / active"),
        Binding("slash", "filter", "Filter"),
        Binding("l", "toggle_log", "Wide log"),
        Binding("s", "next_source", "Source"),
        Binding("escape", "close_filter", show=False),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        source: JobSource,
        *,
        sources: list[JobSource] | None = None,
        run: str | None = None,
        active_only: bool = False,
        refresh_s: float = 1.0,
    ):
        super().__init__()
        self.source = source
        self.sources = list(sources or [source])
        self.run_prefix = run
        self.active_only = active_only
        self.refresh_s = refresh_s
        self.filter_text = ""
        self.selected: str | None = None
        self._order: list[str] = []
        self._cells: dict[str, tuple] = {}
        self._jobs: dict[str, Job] = {}
        self._log_job: str | None = None
        self._log_offset: int | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        table = DataTable(id="jobs", cursor_type="row", zebra_stripes=True)
        for label in _COLUMNS:
            table.add_column(label, key=label)
        yield table
        with Horizontal(id="lower"):
            with VerticalScroll(id="details"):
                yield Static("Select a job.", id="details-body")
            yield RichLog(id="log", max_lines=5000, wrap=True)
        yield Input(placeholder="Filter: id, name, kind, step or status", id="filter")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#details").border_title = "Job"
        self.query_one("#log").border_title = "Log"
        self.refresh_jobs()
        self.set_interval(self.refresh_s, self.refresh_jobs)
        self.query_one(DataTable).focus()

    # ── Refresh ─────────────────────────────────────────────────────

    def _visible(self, job: Job) -> bool:
        if self.active_only and job.state not in ACTIVE:
            return False
        if self.run_prefix and not str(job.spec.get("run_id", "")).startswith(
            self.run_prefix
        ):
            return False
        if self.filter_text:
            fields = (job.id, job.name, job.spec.get("kind"), job.spec.get("step"))
            haystack = " ".join(map(str, (*fields, _shown_state(job)))).lower()
            return self.filter_text.lower() in haystack
        return True

    def refresh_jobs(self) -> None:
        try:
            jobs = self.source.jobs()
            daemon = self.source.supervisor()
        except Exception as exc:
            self.sub_title = f"{self.source.label} · cannot read: {exc}"
            return
        self._jobs = {j.id: j for j in jobs}
        shown = [j for j in reversed(jobs) if self._visible(j)]
        active = sum(j.state in ACTIVE for j in jobs)
        sup = f"supervisor pid {daemon.get('pid')}" if daemon else "no supervisor"
        link = self.source.link()
        if link.get("stale"):
            seen = parse_iso(link.get("seen"))
            when = time.strftime("%H:%M:%S", time.localtime(seen)) if seen else None
            sup += f" · jobs as of {when}" if when else " · no jobs read yet"
        view = "active" if self.active_only else "all"
        if self.filter_text:
            view += f" matching '{self.filter_text}'"
        self.sub_title = (
            f"{self.source.label} · {sup} · {active} active of {len(jobs)} · "
            f"showing {view}"
        )
        table = self.query_one(DataTable)
        order = [j.id for j in shown]
        if order != self._order:
            keep = self.selected
            table.clear()
            self._cells = {}
            for job in shown:
                cells = row_cells(job)
                table.add_row(*cells, key=job.id)
                self._cells[job.id] = cells
            self._order = order
            if keep in order:
                table.move_cursor(row=order.index(keep))
        else:
            for job in shown:
                cells = row_cells(job)
                old = self._cells.get(job.id) or ()
                for label, value, before in zip(_COLUMNS, cells, old, strict=False):
                    if str(value) != str(before):
                        table.update_cell(job.id, label, value)
                self._cells[job.id] = cells
        if not order:
            self.selected = None
            self._show(None)
        elif self.selected not in order:
            self.selected = order[min(table.cursor_row, len(order) - 1)]
            self._show(self.selected)
        else:
            self._show(self.selected, follow=True)

    def _show(self, job_id: str | None, *, follow: bool = False) -> None:
        body = self.query_one("#details-body", Static)
        log = self.query_one(RichLog)
        job = self._jobs.get(job_id) if job_id else None
        if job is None:
            body.update("No jobs to show." if not self._order else "Select a job.")
            log.clear()
            self._log_job = None
            return
        body.update(details(job))
        if not follow or self._log_job != job_id:
            log.clear()
            log.border_title = f"Log · {job.name}"
            self._log_job, self._log_offset = job_id, None
        text, self._log_offset = self.source.read_log(job_id, self._log_offset)
        for line in text.splitlines():
            log.write(Text.from_ansi(line))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        job_id = event.row_key.value if event.row_key else None
        if job_id and job_id != self.selected:
            self.selected = job_id
            self._show(job_id)

    # ── Actions ─────────────────────────────────────────────────────

    def _act(self, **flags: bool) -> None:
        if not self.selected:
            return
        try:
            result = self.source.control(self.selected, **flags)
        except Exception as exc:
            result = f"failed: {exc}"
        self.notify(f"{self.selected}: {result}")
        self.refresh_jobs()

    def action_pause(self) -> None:
        self._act(pause=True)

    def action_resume(self) -> None:
        self._act(pause=False)

    def action_cancel(self) -> None:
        job = self._jobs.get(self.selected or "")
        if job is None or job.state in TERMINAL:
            return

        def answered(yes: bool | None) -> None:
            if yes:
                self._act(cancel=True)

        self.push_screen(
            ConfirmScreen(f"Cancel {job.name} ({job.id})?", "Cancel job"), answered
        )

    def action_toggle_all(self) -> None:
        self.active_only = not self.active_only
        self.refresh_jobs()

    def action_next_source(self) -> None:
        if len(self.sources) < 2:
            self.notify("No other source; add a remote with geofuse remote add.")
            return
        at = self.sources.index(self.source)
        self.source = self.sources[(at + 1) % len(self.sources)]
        self.selected, self._order, self._log_job = None, [], None
        self.query_one(DataTable).clear()
        self.refresh_jobs()

    def action_toggle_log(self) -> None:
        self.query_one("#lower").toggle_class("log-only")

    def action_filter(self) -> None:
        box = self.query_one("#filter", Input)
        box.add_class("shown")
        box.focus()

    def action_close_filter(self) -> None:
        box = self.query_one("#filter", Input)
        if box.has_class("shown"):
            box.remove_class("shown")
            self.query_one(DataTable).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        self.filter_text = event.value.strip()
        self.refresh_jobs()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.action_close_filter()
