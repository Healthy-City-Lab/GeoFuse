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

Caveat: tkinter dialogs open on the machine running the Streamlit server.
This is fine for the local UI and HPC-node CLI workflows; it doesn't work
when the server is remote with the browser elsewhere. The Streamlit
``st.file_uploader`` is still available via :func:`fallback_uploader_to_path`
for that deployment shape.
"""

from __future__ import annotations

import os
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


def _open_tk_picker(
    title: str,
    *,
    file_types: Sequence[tuple[str, str]],
    initial_dir: str | None,
    multi: bool,
) -> tuple[str, ...]:
    """Open the native OS file dialog and return the chosen paths.

    Imports tkinter lazily so a headless / remote deployment that never
    triggers the picker doesn't pay tkinter's import cost. The dialog is
    pinned topmost so it doesn't disappear behind the browser window.
    """
    from tkinter import Tk, filedialog

    root = Tk()
    try:
        root.withdraw()
        root.wm_attributes("-topmost", True)
        if multi:
            picked = filedialog.askopenfilenames(
                title=title,
                initialdir=initial_dir or os.path.expanduser("~"),
                filetypes=list(file_types),
            )
            return tuple(picked) if picked else ()
        single = filedialog.askopenfilename(
            title=title,
            initialdir=initial_dir or os.path.expanduser("~"),
            filetypes=list(file_types),
        )
        return (single,) if single else ()
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def _render_path_chip(path: str, *, key: str) -> bool:
    """Render one bordered chip for ``path`` with a red ✕ remove button.

    Returns ``True`` when the user clicked the remove button so the caller
    can drop the entry from its session-state list and trigger a rerun.
    The chip shows the basename in bold (with the full absolute path as a
    native hover tooltip via ``help=``) and reports missing files inline.
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
            return bool(
                st.button(
                    "❌",
                    key=key,
                    help="Remove this file",
                    width="stretch",
                )
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
    if st.button(
        f"📂 {label}",
        key=f"{key}__btn",
        help=help_text,
    ):
        picked = _open_tk_picker(
            label, file_types=file_types, initial_dir=initial_dir, multi=False
        )
        if picked:
            st.session_state[key] = picked[0]
            st.rerun()

    current = st.session_state.get(key) or None
    if current:
        if _render_path_chip(current, key=f"{key}__rm"):
            st.session_state.pop(key, None)
            st.rerun()
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
    if st.button(
        f"📂 {label}",
        key=f"{key}__btn",
        help=help_text,
    ):
        picked = _open_tk_picker(
            label, file_types=file_types, initial_dir=initial_dir, multi=True
        )
        if picked:
            existing = list(st.session_state.get(key) or [])
            for new_path in picked:
                if new_path and new_path not in existing:
                    existing.append(new_path)
            st.session_state[key] = existing
            st.rerun()

    current: list[str] = list(st.session_state.get(key) or [])
    if not current:
        return current

    # Build a removal list rather than mutating during the render loop —
    # Streamlit reruns on each button press so we'd otherwise drop one item
    # per click instead of just the one the user actually clicked.
    to_remove: list[int] = []
    for idx, p in enumerate(current):
        if _render_path_chip(p, key=f"{key}__rm_{idx}"):
            to_remove.append(idx)
    if to_remove:
        st.session_state[key] = [
            p for i, p in enumerate(current) if i not in set(to_remove)
        ]
        st.rerun()
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
    if st.button(
        f"📂 {label}",
        key=f"{key}__btn",
        help=help_text,
    ):
        picked = _open_tk_picker(
            label, file_types=file_types, initial_dir=initial_dir, multi=True
        )
        if picked:
            existing = list(st.session_state.get(key) or [])
            for new_path in picked:
                if new_path and new_path not in existing:
                    existing.append(new_path)
            st.session_state[key] = existing
            st.rerun()

    current: list[str] = list(st.session_state.get(key) or [])
    if not current:
        return []

    move_up: int | None = None
    move_down: int | None = None
    to_remove: int | None = None
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
                if st.button(
                    "▲",
                    key=f"{key}__up__{wid}",
                    help="Move up",
                    width="stretch",
                    disabled=(idx == 0),
                ):
                    move_up = idx
            with c_dn:
                if st.button(
                    "▼",
                    key=f"{key}__dn__{wid}",
                    help="Move down",
                    width="stretch",
                    disabled=(idx == len(current) - 1),
                ):
                    move_down = idx
            with c_rm:
                if st.button(
                    "❌",
                    key=f"{key}__rm__{wid}",
                    help="Remove this file",
                    width="stretch",
                ):
                    to_remove = idx
            if missing:
                st.warning(f"⚠️ File no longer exists at `{path}`")
            st.text_input(
                f"Label #{idx + 1}",
                key=label_widget_key,
                label_visibility="collapsed",
                help=f"{base}\n\n{path}",
                placeholder=base,
            )
    if to_remove is not None:
        new_list = [p for i, p in enumerate(current) if i != to_remove]
        st.session_state[key] = new_list
        st.rerun()
    if move_up is not None and move_up > 0:
        new_list = list(current)
        new_list[move_up - 1], new_list[move_up] = (
            new_list[move_up],
            new_list[move_up - 1],
        )
        st.session_state[key] = new_list
        st.rerun()
    if move_down is not None and move_down < len(current) - 1:
        new_list = list(current)
        new_list[move_down + 1], new_list[move_down] = (
            new_list[move_down],
            new_list[move_down + 1],
        )
        st.session_state[key] = new_list
        st.rerun()

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
