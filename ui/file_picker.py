"""Native file-picker integration for the Streamlit UI.

``st.file_uploader`` ships the file's *bytes* over HTTP to the Streamlit
server; the original filesystem path is dropped by the browser for
security and a temp copy is written so the engines can do GDAL I/O. That
forces the engines to read every byte twice (once into RAM, once onto
disk) and prevents stable cross-session references to the same input.

This module replaces that flow with a native OS file dialog (via
``tkinter.filedialog``) wherever the Streamlit server and the user's
browser are on the same machine — which is the case for the toolbox's
intended local-development and HPC node deployment. The user clicks a
``Browse...`` button, picks a file in the native dialog, and the toolbox
stores the **real absolute path** in ``st.session_state``. Engines then
read from that path lazily — only when preview, sampling, or compute
actually needs the bytes — and a job's recorded path + hash can be
re-verified on restart without ever re-uploading.

The dialog runs in a **child process** (``file_picker_child.py``), not in the
Streamlit worker. Tk wants to own the main thread of its interpreter, and
Streamlit runs the script body and every ``on_click`` callback on a
per-rerun ScriptRunner thread — an in-process dialog raises
``RuntimeError: main thread is not in main loop`` as soon as any Tk state
outlives the thread that created it. One subprocess per click costs a few
hundred milliseconds on a user-initiated action and removes the failure mode
entirely.

Caveat: the dialog opens on the machine running the Streamlit server. That is
right for the local UI and HPC-node workflows and wrong when the server is
remote with the browser elsewhere. In that case the picker reports itself
unavailable and the button grows a "paste a full path" box instead of raising;
:func:`fallback_uploader_to_path` remains for deployments that want the
browser-side upload flow.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

# Common file-type filter tuples for the OS dialog. Mirrors the file types
# the toolbox actually consumes so the picker pre-filters extensions on
# Windows/macOS/Linux. ``("All files", "*.*")`` is always appended so the
# user can override the filter when they need to.
FT_VECTOR = [
    ("Vector files", "*.geojson *.gpkg *.zip *.shp *.json"),
    ("GeoJSON", "*.geojson *.json"),
    ("GeoPackage", "*.gpkg"),
    ("Shapefile (zipped)", "*.zip"),
    ("Shapefile", "*.shp"),
    ("All files", "*.*"),
]
FT_RASTER = [
    ("GeoTIFF", "*.tif *.tiff"),
    ("All files", "*.*"),
]
FT_VECTOR_OR_RASTER = [
    ("Vector or raster", "*.geojson *.gpkg *.zip *.shp *.json *.tif *.tiff"),
    *FT_VECTOR,
    *FT_RASTER,
]


@dataclass(frozen=True)
class PickedDataset:
    """Minimal wrapper around a real filesystem path so callers can swap in
    where they used to receive a ``MaterializedDataset``.

    ``cleanup_dir`` / ``cleanup_file`` are always ``None`` — the file lives
    at its user-chosen location and the toolbox never copies or deletes it.
    """

    path: str
    is_vector: bool
    display_name: str
    cleanup_dir: None = None
    cleanup_file: None = None


class PickerUnavailable(RuntimeError):
    """The native dialog could not be opened on this machine."""


#: Where the manual-path fallback stores its "why did the dialog fail" note.
_PICKER_ERROR_KEY = "__file_picker_error__"

#: Ceiling on how long the dialog may stay open. Generous — the user is
#: browsing a filesystem — but bounded, so a child that somehow never draws a
#: window cannot wedge the Streamlit worker thread forever.
_PICKER_TIMEOUT_S = 600


def _open_tk_picker(
    title: str,
    *,
    file_types: Sequence[tuple[str, str]],
    initial_dir: str | None,
    multi: bool,
) -> tuple[str, ...]:
    """Open the native OS file dialog in a child process; return the paths.

    The dialog runs out-of-process deliberately. Tk wants to own the main
    thread of whatever interpreter it lives in, and Streamlit runs both the
    script body and every ``on_click`` callback on a per-rerun ScriptRunner
    thread — so an in-process dialog raises ``RuntimeError: main thread is not
    in main loop`` the moment any Tk state survives the thread that made it.
    A child process has its own main thread and is gone before the next rerun.

    Raises :class:`PickerUnavailable` when there is no usable dialog (headless
    server, no Tk build, user's session has no display); callers fall back to
    typing a path.
    """
    import json
    import subprocess
    import tempfile

    child = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "file_picker_child.py"
    )
    if not os.path.isfile(child):
        raise PickerUnavailable(f"picker helper is missing at {child}")

    request = {
        "title": title,
        "file_types": [list(ft) for ft in file_types],
        "initial_dir": initial_dir or os.path.expanduser("~"),
        "multi": bool(multi),
    }
    tmp = tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    )
    try:
        json.dump(request, tmp)
        tmp.close()
        # CREATE_NO_WINDOW keeps a console from flashing up behind the dialog
        # on Windows; the dialog itself is a GUI window and still appears.
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            proc = subprocess.run(
                [sys.executable, child, tmp.name],
                capture_output=True,
                text=True,
                timeout=_PICKER_TIMEOUT_S,
                creationflags=creationflags,
            )
        except subprocess.TimeoutExpired as exc:
            raise PickerUnavailable(
                f"the file dialog did not close within {_PICKER_TIMEOUT_S}s"
            ) from exc
        except OSError as exc:
            raise PickerUnavailable(f"could not start the file dialog: {exc}") from exc
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if not lines:
        detail = (proc.stderr or "").strip().splitlines()
        raise PickerUnavailable(
            detail[-1] if detail else "the file dialog produced no output"
        )
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise PickerUnavailable(
            f"unreadable response from the file dialog: {exc}"
        ) from exc

    if isinstance(payload, dict) and payload.get("error"):
        raise PickerUnavailable(str(payload["error"]))
    if not isinstance(payload, list):
        raise PickerUnavailable("unexpected response from the file dialog")
    return tuple(str(p) for p in payload if p)


# All picker buttons mutate state via ``on_click`` callbacks (which run
# before the script body) instead of ``st.rerun()`` mid-script — an aborted
# run makes Streamlit drop the session state of every widget further down
# the page that didn't get to render.


def _browse_cb(
    key: str,
    title: str,
    file_types: Sequence[tuple[str, str]],
    initial_dir: str | None,
    multi: bool,
) -> None:
    """Open the OS dialog and store the selection under ``key``.

    A dialog that cannot open is a normal condition on a remote or headless
    server, not a bug — it records the reason and lets the caller offer the
    manual path box, rather than tearing the page down with a traceback.
    """
    st.session_state.pop(_PICKER_ERROR_KEY, None)
    try:
        picked = _open_tk_picker(
            title, file_types=file_types, initial_dir=initial_dir, multi=multi
        )
    except PickerUnavailable as exc:
        st.session_state[_PICKER_ERROR_KEY] = str(exc)
        return
    if not picked:
        return
    if multi:
        existing = list(st.session_state.get(key) or [])
        for new_path in picked:
            if new_path and new_path not in existing:
                existing.append(new_path)
        st.session_state[key] = existing
    else:
        st.session_state[key] = picked[0]


def _clear_path_cb(key: str) -> None:
    st.session_state.pop(key, None)


def _manual_path_cb(key: str, widget_key: str, multi: bool) -> None:
    """Accept a typed path as if it had come from the dialog."""
    raw = (st.session_state.get(widget_key) or "").strip().strip('"')
    if not raw:
        return
    if multi:
        existing = list(st.session_state.get(key) or [])
        if raw not in existing:
            existing.append(raw)
        st.session_state[key] = existing
    else:
        st.session_state[key] = raw
    st.session_state[widget_key] = ""


def _render_manual_fallback(key: str, *, widget_key: str, multi: bool) -> None:
    """Path text box shown only after the native dialog has failed.

    Kept out of the way until it is needed: on a normal local run the dialog
    works and an always-visible path box is just clutter.
    """
    reason = st.session_state.get(_PICKER_ERROR_KEY)
    if not reason:
        return
    st.warning(
        f"The native file dialog could not open ({reason}). "
        "Paste a full path instead — this happens when the app is served "
        "from a machine other than the one you are browsing from."
    )
    col_in, col_add = st.columns([5, 1])
    with col_in:
        st.text_input(
            "Full path to the file",
            key=widget_key,
            label_visibility="collapsed",
            placeholder=r"C:\path\to\file.gpkg",
        )
    with col_add:
        st.button(
            "Add",
            key=f"{widget_key}__add",
            width="stretch",
            on_click=_manual_path_cb,
            args=(key, widget_key, multi),
        )


def _remove_path_cb(key: str, path: str) -> None:
    st.session_state[key] = [p for p in (st.session_state.get(key) or []) if p != path]


def _move_path_cb(key: str, path: str, delta: int) -> None:
    lst = list(st.session_state.get(key) or [])
    if path not in lst:
        return
    i = lst.index(path)
    j = i + delta
    if 0 <= j < len(lst):
        lst[i], lst[j] = lst[j], lst[i]
        st.session_state[key] = lst


def _render_path_chip(path: str, *, key: str, on_remove, args: tuple) -> None:
    """Render one bordered chip for ``path`` with a red ✕ remove button.

    The chip shows the basename in bold (with the full absolute path as a
    native hover tooltip via ``help=``) and reports missing files inline.
    ``on_remove`` runs as the ✕ button's callback.
    """
    with st.container(border=True):
        row_main, row_rm = st.columns([10, 1])
        with row_main:
            base = os.path.basename(path)
            if os.path.isfile(path):
                st.markdown(f"📄 **{base}**", help=path)
            else:
                st.warning(f"⚠️ **{base}** — file no longer exists at `{path}`")
        with row_rm:
            st.button(
                "❌",
                key=key,
                help="Remove this file",
                width="stretch",
                on_click=on_remove,
                args=args,
            )


def pick_file_path(
    label: str,
    *,
    key: str,
    file_types: Sequence[tuple[str, str]] = FT_VECTOR_OR_RASTER,
    initial_dir: str | None = None,
    help_text: str | None = None,
) -> str | None:
    """Render a "Browse..." button; return the currently-selected path.

    The button gets ``help_text`` as a native hover tooltip (no inline text
    block). When the user picks a file the OS dialog returns its absolute
    path; the chip with a ✕ remove button renders **below** the button row
    so the layout stays tight. Returns the current selection or ``None``.
    """
    st.button(
        f"📂 {label}",
        key=f"{key}__btn",
        help=help_text,
        on_click=_browse_cb,
        args=(key, label, file_types, initial_dir, False),
    )
    _render_manual_fallback(key, widget_key=f"{key}__manual", multi=False)

    current = st.session_state.get(key) or None
    if current:
        _render_path_chip(
            current, key=f"{key}__rm", on_remove=_clear_path_cb, args=(key,)
        )
    return current


def pick_multiple_paths(
    label: str,
    *,
    key: str,
    file_types: Sequence[tuple[str, str]] = FT_VECTOR_OR_RASTER,
    initial_dir: str | None = None,
    help_text: str | None = None,
) -> list[str]:
    """Multi-file variant: each click **appends** to the kept list.

    Selecting files in the OS dialog deduplicates against what's already in
    ``st.session_state[key]`` so re-picking an existing file is a no-op.
    Each kept path renders as its own bordered chip below the button with
    a red ✕ remove button beside it; clicking ✕ drops just that entry.
    """
    st.button(
        f"📂 {label}",
        key=f"{key}__btn",
        help=help_text,
        on_click=_browse_cb,
        args=(key, label, file_types, initial_dir, True),
    )
    _render_manual_fallback(key, widget_key=f"{key}__manual", multi=True)

    current: list[str] = list(st.session_state.get(key) or [])
    for idx, p in enumerate(current):
        _render_path_chip(
            p, key=f"{key}__rm_{idx}", on_remove=_remove_path_cb, args=(key, p)
        )
    return current


def path_to_widget_id(path: str) -> str:
    """Stable, short, alnum-safe widget-key fragment for ``path``.

    Reordering the kept list must not move the user's label or column
    selections with the *index* — bindings stick to the **file**.
    """
    import hashlib

    return hashlib.md5(path.encode("utf-8", errors="replace")).hexdigest()[:10]


def pick_ordered_files(
    label: str,
    *,
    key: str,
    file_types: Sequence[tuple[str, str]] = FT_VECTOR_OR_RASTER,
    initial_dir: str | None = None,
    help_text: str | None = None,
) -> list[dict]:
    """Browse-and-list multi-file picker with reorder + editable labels.

    Each kept file renders as one bordered chip showing its custom label,
    a ▲/▼ pair to move it up/down in the list, an editable label input,
    and a ✕ remove button. The basename's stem is the default label; the
    full absolute path is the chip's native hover tooltip. The returned
    order reflects the user's reordering — the first entry is treated as
    the baseline by callers.
    """
    st.button(
        f"📂 {label}",
        key=f"{key}__btn",
        help=help_text,
        on_click=_browse_cb,
        args=(key, label, file_types, initial_dir, True),
    )
    _render_manual_fallback(key, widget_key=f"{key}__manual", multi=True)

    current: list[str] = list(st.session_state.get(key) or [])
    if not current:
        return []

    for idx, path in enumerate(current):
        wid = path_to_widget_id(path)
        label_widget_key = f"{key}__label__{wid}"
        if label_widget_key not in st.session_state:
            st.session_state[label_widget_key] = Path(path).stem
        base = os.path.basename(path)
        missing = not os.path.isfile(path)
        with st.container(border=True):
            # Compact 4-equal-column control row keeps the buttons from
            # being squeezed in a half-width parent column. The filename
            # lives in the row-2 placeholder + tooltip, not row 1, so
            # long names can't wrap into the controls.
            c_idx, c_up, c_dn, c_rm = st.columns([1, 1, 1, 1])
            with c_idx:
                st.markdown(f"**#{idx + 1}**", help=f"{path}")
            with c_up:
                st.button(
                    "▲",
                    key=f"{key}__up__{wid}",
                    help="Move up",
                    width="stretch",
                    disabled=(idx == 0),
                    on_click=_move_path_cb,
                    args=(key, path, -1),
                )
            with c_dn:
                st.button(
                    "▼",
                    key=f"{key}__dn__{wid}",
                    help="Move down",
                    width="stretch",
                    disabled=(idx == len(current) - 1),
                    on_click=_move_path_cb,
                    args=(key, path, 1),
                )
            with c_rm:
                st.button(
                    "❌",
                    key=f"{key}__rm__{wid}",
                    help="Remove this file",
                    width="stretch",
                    on_click=_remove_path_cb,
                    args=(key, path),
                )
            if missing:
                st.warning(f"⚠️ File no longer exists at `{path}`")
            st.text_input(
                f"Label #{idx + 1}",
                key=label_widget_key,
                label_visibility="collapsed",
                help=f"{base}\n\n{path}",
                placeholder=base,
            )

    out: list[dict] = []
    for path in current:
        wid = path_to_widget_id(path)
        lbl = st.session_state.get(f"{key}__label__{wid}") or Path(path).stem
        out.append({"path": path, "label": str(lbl)})
    return out


def clear_path_state(key: str) -> None:
    """Drop the current selection from session state (used by the "reset"
    buttons sprinkled around the tabs)."""
    st.session_state.pop(key, None)


def path_to_dataset(path: str) -> PickedDataset:
    """Wrap a real filesystem path in a ``PickedDataset`` for engine handoff.

    Detects vector-vs-raster from the file extension. The display name is
    the basename. Caller is responsible for confirming the path exists.
    """
    from geofuse.vector_io import target_path_is_raster

    is_vec = not target_path_is_raster(path)
    return PickedDataset(
        path=path, is_vector=is_vec, display_name=os.path.basename(path)
    )


def fallback_uploader_to_path(
    label: str,
    *,
    key: str,
    accepted_extensions: Iterable[str] | None = None,
    persist_dir: str | None = None,
) -> str | None:
    """Streamlit ``file_uploader`` fallback for remote deployments.

    Writes the uploaded bytes to ``persist_dir`` (default
    ``output_results/uploads/<basename>``) so the resulting path is stable
    across the session and the engine can read from it like any other
    on-disk input. Returns the persisted path or ``None``.
    """
    persist_dir = persist_dir or os.path.join("output_results", "uploads")
    os.makedirs(persist_dir, exist_ok=True)
    types = list(accepted_extensions) if accepted_extensions else None
    up = st.file_uploader(label, key=f"{key}__upload", type=types)
    if up is None:
        return st.session_state.get(key)
    dest = os.path.join(persist_dir, Path(up.name).name)
    with open(dest, "wb") as f:
        f.write(up.getvalue())
    st.session_state[key] = dest
    return dest
