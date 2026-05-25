"""Page-level chrome injection for the Streamlit UI.

All custom CSS/JS that operates on the page chrome -- the global stylesheet,
the toolbar tab-list relocation observer, the GeoFuse brand label, the
sidebar collapse/expand toggle -- lives in :mod:`ui.assets` and is injected
once per script run by :func:`inject_chrome`.

**Where to put new CSS/JS**

Any new long-form CSS or JS that targets the Streamlit app's DOM
(stylesheet rules, MutationObserver edits, click handlers that bypass
Streamlit's rerun cycle) must live in ``ui/assets/<name>.css`` or
``ui/assets/<name>.js`` and be loaded through :func:`load_asset`.

Do **not** embed multi-line CSS/JS as triple-quoted Python strings in
``app.py``, ``resource_monitor.py``, or any tab module. Reasons:

* The editor's CSS/JS language servers can't lint, format, or autocomplete
  string-embedded code.
* Diffs become noisy when Python escaping changes.
* The assets become reusable from other modules without re-importing
  hundreds of lines of string.

The exception is a few tokens of inline HTML (a single ``<div>`` element,
a one-line ``<style>`` snippet) that genuinely belongs next to the
Streamlit call -- those stay inline.

**Browser support**

Assets target evergreen Chromium (Chrome, Edge, Brave, Opera), Firefox,
and Safari (macOS + iOS). When adding rules that involve vendor prefixes
(``mask-*``, ``backdrop-filter``, ``user-select``, ``appearance``,
``clip-path``), include both the ``-webkit-`` / ``-moz-`` prefix **and**
the unprefixed property. See the header comments in
``ui/assets/chrome.css`` for the current list.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

_ASSETS_DIR = Path(__file__).parent / "assets"


@lru_cache(maxsize=32)
def load_asset(name: str) -> str:
    """Read a UTF-8 text asset from ``ui/assets/``.

    Cached for the lifetime of the process so repeated calls during a
    script rerun don't re-touch the filesystem. Streamlit's file watcher
    triggers a full page reload when an asset changes, which restarts the
    process and clears this cache, so stale content isn't a concern in
    practice.
    """
    return (_ASSETS_DIR / name).read_text(encoding="utf-8")


def inject_chrome() -> None:
    """Inject the global page chrome: CSS, the floater button HTML, and JS.

    Call this once near the top of ``ui/app.py`` after
    ``st.set_page_config``. The order matters:

    1. ``<style>`` block goes in first so initial render is already styled.
    2. The floater ``<div role="button">`` is emitted so it's part of
       Streamlit's normal render output -- the JS observer relocates it
       into the toolbar, but having it in the DOM up-front means we never
       depend on the JS path alone.
    3. The ``components.html`` iframe runs the MutationObserver that
       relocates the tab list, injects the brand label, and binds the
       sidebar toggle click handler. ``height=0`` keeps it invisible.

    We use a ``<div role="button">`` instead of ``<button>`` for the
    floater because some Streamlit versions' markdown sanitiser strips
    interactive form tags even with ``unsafe_allow_html=True``; ``<div>``
    is always passed through.
    """
    st.markdown(
        f"<style>{load_asset('chrome.css')}</style>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="gf-sidebar-floater" role="button" tabindex="0" '
        'aria-label="Toggle sidebar" title="Toggle sidebar"></div>',
        unsafe_allow_html=True,
    )
    components.html(
        f"<script>{load_asset('chrome.js')}</script>",
        height=0,
        scrolling=False,
    )
