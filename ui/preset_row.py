"""The Preset row at the top of the NDVI and GVI settings boxes.

A select box lists the built-in and saved presets; choosing one applies it.
**Load** imports a preset file (validated for this tab, then copied into the
preset library and applied); **Save** writes the current settings to the
library and offers the JSON for download. The select box shows the preset in
force, and "Custom" once any of its settings has been changed.

Each tab supplies ``current()`` — its settings as a preset dict — and
``apply(settings)``, which writes them into session state. Every state change
happens in widget callbacks, before the widgets render. The tab calls
:func:`finish` at the end of its settings fragment, which reruns the whole app
once so controls outside the fragment (output formats, date configurations)
show the new values — at the end, because a rerun mid-fragment would drop the
state of every widget the aborted run had not reached yet. A fragment holding
other preset settings calls :func:`refresh_if_stale` at its end, so editing
one of them updates the row.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import streamlit as st

from geofuse import presets

CUSTOM = "Custom"


def _k(engine: str, name: str) -> str:
    return f"{engine}_preset_{name}"


def _matches(current: dict, settings: dict) -> bool:
    return all(current.get(k) == v for k, v in settings.items())


def preset_in_force(engine: str, current: Callable[[], dict]) -> str | None:
    """The preset applied, loaded or saved in this session, while every
    setting it holds is unchanged; otherwise ``None``. Jobs record this name."""
    applied = st.session_state.get(_k(engine, "applied"))
    if applied and _matches(current(), applied["settings"]):
        return applied["name"]
    return None


def _shown(engine: str, current: Callable[[], dict], in_force: str | None) -> str:
    """What the select box shows: the preset in force; before any preset was
    used, "Default" while the settings are the defaults; otherwise Custom."""
    if in_force:
        return in_force
    if st.session_state.get(_k(engine, "applied")):
        return CUSTOM
    try:
        default = presets.load_settings(engine, "Default")
    except presets.PresetError:
        return CUSTOM
    return "Default" if _matches(current(), default) else CUSTOM


def _say(engine: str, level: str, text: str) -> None:
    st.session_state[_k(engine, "message")] = (level, text)


def _applied(engine: str, name: str, settings: dict) -> None:
    st.session_state[_k(engine, "applied")] = {"name": name, "settings": dict(settings)}
    st.session_state[_k(engine, "rerun")] = True


def _on_select(engine: str, apply: Callable[[dict], None]) -> None:
    name = st.session_state.get(_k(engine, "select"))
    if not name or name == CUSTOM:
        return
    try:
        settings = presets.load_settings(engine, name)
    except presets.PresetError as exc:
        _say(engine, "error", str(exc))
        return
    apply(settings)
    _applied(engine, name, settings)
    _say(engine, "success", f"Applied preset '{name}'.")


def _import(engine: str, path: str, apply: Callable[[dict], None]) -> None:
    """Validate a preset file, add it to the library, apply it."""
    try:
        doc = presets.read_preset(path)
        settings, warnings = presets.validate(doc, engine)
    except presets.PresetError as exc:
        _say(engine, "error", f"Not loaded: {exc}")
        return
    name = str(doc.get("name") or "").strip() or "Imported preset"
    doc["name"] = name
    note = ""
    try:
        presets.save_user_preset(doc)
    except presets.PresetExists:
        st.session_state[_k(engine, "pending")] = doc
        note = " A saved preset has the same name — choose below whether to replace it."
    except presets.PresetError:
        note = (
            " A built-in preset has the same name, so it was applied but not "
            "added to the library."
        )
    apply(settings)
    _applied(engine, name, settings)
    extra = (" " + " ".join(warnings)) if warnings else ""
    level = "warning" if warnings or note else "success"
    _say(engine, level, f"Loaded '{name}'.{extra}{note}")


def _on_load(engine: str, apply: Callable[[dict], None]) -> None:
    from file_picker import PickerUnavailable, _open_tk_picker

    try:
        picked = _open_tk_picker(
            "Load preset",
            file_types=[("GeoFuse preset", "*.json"), ("All files", "*.*")],
            initial_dir=str(presets.USER_DIR),
            multi=False,
        )
    except PickerUnavailable as exc:
        st.session_state[_k(engine, "manual")] = True
        _say(
            engine,
            "warning",
            f"The file dialog is unavailable ({exc}); paste the path below.",
        )
        return
    if picked:
        _import(engine, picked[0], apply)


def _on_manual(engine: str, apply: Callable[[dict], None]) -> None:
    path = (st.session_state.get(_k(engine, "manual_path")) or "").strip().strip('"')
    if path:
        _import(engine, path, apply)
        st.session_state[_k(engine, "manual")] = False


def _on_save(engine: str, current: Callable[[], dict]) -> None:
    name = (st.session_state.get(_k(engine, "name")) or "").strip()
    description = st.session_state.get(_k(engine, "description")) or ""
    try:
        settings = {k: v for k, v in current().items() if v is not None}
        doc = presets.make_preset(engine, name, description, settings)
        presets.save_user_preset(doc)
    except presets.PresetExists:
        st.session_state[_k(engine, "pending")] = doc
        _say(engine, "warning", f"A saved preset named '{name}' exists — replace it?")
        return
    except presets.PresetError as exc:
        _say(engine, "error", f"Not saved: {exc}")
        return
    _saved(engine, doc)


def _saved(engine: str, doc: dict) -> None:
    st.session_state[_k(engine, "download")] = doc
    _applied(engine, doc["name"], doc["settings"])
    _say(engine, "success", f"Saved preset '{doc['name']}'.")


def _on_overwrite(engine: str) -> None:
    doc = st.session_state.pop(_k(engine, "pending"), None)
    if doc is None:
        return
    try:
        presets.save_user_preset(doc, overwrite=True)
    except presets.PresetError as exc:
        _say(engine, "error", f"Not saved: {exc}")
        return
    _saved(engine, doc)


def _on_cancel(engine: str) -> None:
    st.session_state.pop(_k(engine, "pending"), None)
    _say(engine, "info", "Kept the existing preset.")


def render_preset_row(
    engine: str, *, current: Callable[[], dict], apply: Callable[[dict], None]
) -> str | None:
    """Draw the row; return the preset in force for job provenance (``None``
    when no preset was used, or its settings have since been changed)."""
    in_force = preset_in_force(engine, current)
    names = [e.name for e in presets.list_presets(engine)]
    sel_key = _k(engine, "select")
    shown = _shown(engine, current, in_force)
    if shown not in names:
        shown = CUSTOM
    st.session_state[_k(engine, "shown")] = shown
    if st.session_state.get(sel_key) != shown:
        st.session_state[sel_key] = shown

    c_sel, c_load, c_save = st.columns([4, 1, 1], vertical_alignment="bottom")
    c_sel.selectbox(
        "Preset",
        [CUSTOM, *names],
        key=sel_key,
        on_change=_on_select,
        args=(engine, apply),
        help=(
            "Named settings for this tab. Choosing one applies it; "
            "**Custom** means the settings no longer match a preset."
        ),
    )
    c_load.button(
        "Load",
        key=_k(engine, "load"),
        on_click=_on_load,
        args=(engine, apply),
        width="stretch",
        help="Import a preset file, add it to your presets and apply it.",
    )
    with c_save.popover("Save", width="stretch"):
        st.text_input("Preset name", key=_k(engine, "name"))
        st.text_area(
            "Description",
            key=_k(engine, "description"),
            height=80,
            help="What the settings do, for whoever loads the preset.",
        )
        st.button(
            "Save preset",
            key=_k(engine, "save"),
            on_click=_on_save,
            args=(engine, current),
            type="primary",
            width="stretch",
        )
        doc = st.session_state.get(_k(engine, "download"))
        if doc:
            st.download_button(
                "Download JSON",
                data=json.dumps(doc, indent=2, ensure_ascii=False),
                file_name=f"{presets._slug(doc['name'])}.json",
                mime="application/json",
                key=_k(engine, "download_btn"),
                width="stretch",
            )

    if st.session_state.get(_k(engine, "manual")):
        st.text_input(
            "Preset file path",
            key=_k(engine, "manual_path"),
            on_change=_on_manual,
            args=(engine, apply),
        )
    if st.session_state.get(_k(engine, "pending")):
        name = st.session_state[_k(engine, "pending")].get("name")
        c_yes, c_no = st.columns(2)
        c_yes.button(
            f"Replace '{name}'",
            key=_k(engine, "overwrite"),
            on_click=_on_overwrite,
            args=(engine,),
            width="stretch",
        )
        c_no.button(
            "Keep the saved one",
            key=_k(engine, "cancel"),
            on_click=_on_cancel,
            args=(engine,),
            width="stretch",
        )
    # With a rerun pending, the message waits in state for the fresh run.
    if not st.session_state.get(_k(engine, "rerun")):
        message = st.session_state.pop(_k(engine, "message"), None)
        if message:
            getattr(st, message[0])(message[1])
    return in_force


def finish(engine: str) -> None:
    """Rerun the app if a preset was just applied; call last in the fragment."""
    if st.session_state.pop(_k(engine, "rerun"), False):
        st.rerun()


def refresh_if_stale(engine: str, current: Callable[[], dict]) -> None:
    """Rerun the app when the row no longer shows the right preset.

    For a fragment other than the settings one: editing a preset setting there
    reruns only that fragment, and the row would keep showing the preset.
    """
    shown_key = _k(engine, "shown")
    if shown_key not in st.session_state:
        return
    names = [e.name for e in presets.list_presets(engine)]
    shown = _shown(engine, current, preset_in_force(engine, current))
    if shown not in names:
        shown = CUSTOM
    if shown != st.session_state[shown_key]:
        st.rerun()
