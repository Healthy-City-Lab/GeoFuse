"""Process-level singletons used by every tab.

These factories use ``@st.cache_resource`` so the underlying objects are shared
across all browser sessions and survive script reruns. Tabs must import from
*this* module, never from ``ui/app.py`` — importing the entry-point script
re-executes it and triggers duplicate-widget-key errors.

This module also exposes ``ansi_log_lines_to_html`` so the job monitor can
render captured engine log lines in their original ANSI colours inside a
markdown block.
"""

from __future__ import annotations

import html
import re

import streamlit as st

from geofuse.persistence.job_executor import JobExecutor
from geofuse.persistence.job_store import JobStore
from geofuse.persistence.pano_cache import PanoCache

# Colours mirror geofuse.logger._ANSI. We don't import them so that this
# converter has no dependency on the engine side beyond strings on the wire.
_ANSI_ESC = re.compile(r"\x1b\[([0-9;]+)m")
_ANSI_COLOUR_HEX = {
    "36": "#4dd0e1",  # cyan-ish (INFO)
    "32": "#66bb6a",  # green   (OK)
    "33": "#ffd54f",  # yellow  (WARN)
    "31": "#ef5350",  # red     (ERROR)
}


def _ansi_line_to_html(line: str) -> str:
    """Convert one log line with ANSI escapes to safe HTML."""
    out: list[str] = []
    pos = 0
    open_spans = 0
    for m in _ANSI_ESC.finditer(line):
        out.append(html.escape(line[pos : m.start()]))
        for code in m.group(1).split(";"):
            if code in ("0", ""):
                while open_spans:
                    out.append("</span>")
                    open_spans -= 1
            elif code == "1":
                out.append('<span style="font-weight:600;">')
                open_spans += 1
            elif code in _ANSI_COLOUR_HEX:
                out.append(f'<span style="color:{_ANSI_COLOUR_HEX[code]};">')
                open_spans += 1
            # Other codes are ignored.
        pos = m.end()
    out.append(html.escape(line[pos:]))
    while open_spans:
        out.append("</span>")
        open_spans -= 1
    return "".join(out)


def ansi_log_lines_to_html(lines: list[str]) -> str:
    """Render a list of captured log lines as a black-background HTML block.

    The horizontal negative margin pulls the box out past Streamlit's expander
    content padding so the box fills the full expander width without gaps on
    the sides.
    """
    rendered = "<br>".join(_ansi_line_to_html(line) for line in lines)
    return (
        '<div style="font-family:ui-monospace,Menlo,Consolas,monospace;'
        "font-size:0.68em;background:#0b0b0b;color:#e0e0e0;"
        "padding:0.55em 0.7em;"
        "margin:0 -1rem;"
        "max-height:280px;overflow-y:auto;"
        'white-space:pre-wrap;word-break:break-word;line-height:1.35;">'
        f"{rendered}"
        "</div>"
    )


@st.cache_resource
def get_job_store() -> JobStore:
    """Process-level JobStore singleton."""
    return JobStore("logs/jobs.db")


@st.cache_resource
def get_job_executor() -> JobExecutor:
    """Process-level worker pool. Holds the GPU lock and the ThreadPoolExecutor."""
    return JobExecutor(get_job_store(), max_workers=4)


@st.cache_resource
def get_pano_cache() -> PanoCache:
    """SQLite-backed GVI pano cache. Global across study areas and runs."""
    return PanoCache("logs/caches/gvi_panos.db")
