"""Sidebar cache management: inspect and purge the toolbox's persistent caches.

Rendered once per script run from ``ui/app.py`` (like the resource monitor) so
it is reachable from every tab, not just the one that happens to own the job
monitor.

Purging is destructive and rare, so each cache uses an explicit two-step flow:
  1. "Purge" arms the action and shows exactly what will be deleted.
  2. "Yes, purge" performs it and reports what was actually freed.

Nothing here touches job history or fusion artifacts on disk, so completed
runs stay loadable in the Result Inspector.
"""

from __future__ import annotations

import streamlit as st

from geofuse.persistence import caches

_ARM_KEY = "_cache_purge_armed"
_DONE_KEY = "_cache_purge_done"


def _open_caches(output_dir: str) -> dict:
    """The GUI's open caches, so a purge also clears their in-memory state."""
    from services import get_ndvi_tile_cache, get_pano_cache

    return {
        "output_dir": output_dir,
        "pano": get_pano_cache(),
        "ndvi": get_ndvi_tile_cache(),
    }


def _stats(key: str) -> str:
    try:
        return caches.describe_cache(key, **_open_caches(_output_dir()))
    except Exception as e:
        return f"unavailable ({type(e).__name__})"


def _purge(key: str) -> str:
    return caches.purge_cache(key, **_open_caches(_output_dir()))


def _output_dir() -> str:
    return st.session_state.get("_gf_output_dir", "output_results")


_CACHES = [
    {
        "key": "pano",
        "label": caches.CACHE_LABELS["pano"],
        "help": (
            "Segmented GVI values keyed by panorama id (`gvi_panos.db` in the "
            "cache folder, `logs/caches/` unless `[paths] caches` moves it). "
            "Purge this after any change to the segmentation metric — otherwise "
            "old runs keep returning the previously-computed numbers."
        ),
        "stats": lambda: _stats("pano"),
        "purge": lambda: _purge("pano"),
    },
    {
        "key": "ndvi",
        "label": caches.CACHE_LABELS["ndvi"],
        "help": (
            "Downloaded Earth Engine tile workspaces (`ndvi_tiles/` in the "
            "cache folder). "
            "Safe to purge; the next run re-downloads what it needs."
        ),
        "stats": lambda: _stats("ndvi"),
        "purge": lambda: _purge("ndvi"),
    },
    {
        "key": "fusion",
        "label": caches.CACHE_LABELS["fusion"],
        "help": (
            "Content-addressed fusion inputs (`output_results/fusion_cache/`). "
            "Purging only forces re-computation — finished job reports and "
            "artifacts are kept and stay loadable."
        ),
        "stats": lambda: _stats("fusion"),
        "purge": lambda: _purge("fusion"),
    },
]


def render_cache_manager(output_dir: str = "output_results") -> None:
    """Render the cache panel into the sidebar. Call once per script run."""
    st.session_state["_gf_output_dir"] = output_dir

    with st.sidebar:
        with st.expander("🧹 Cache Management", expanded=False):
            st.caption(
                "Purge stored results to force a clean recomputation. "
                "Job history and finished reports are never touched."
            )

            done = st.session_state.get(_DONE_KEY)
            if done:
                st.success(done)
                if st.button("Dismiss", key="cache_done_dismiss", width="stretch"):
                    st.session_state.pop(_DONE_KEY, None)
                    st.rerun()

            armed = st.session_state.get(_ARM_KEY)
            for i, spec in enumerate(_CACHES):
                k = spec["key"]
                st.markdown(f"**{spec['label']}**")
                st.caption(f"{spec['stats']()}")
                if armed == k:
                    st.warning(
                        f"Permanently delete the {spec['label'].lower()}? "
                        "This cannot be undone."
                    )
                    c1, c2 = st.columns(2)
                    if c1.button(
                        "Yes, purge",
                        key=f"cache_yes_{k}",
                        type="primary",
                        width="stretch",
                    ):
                        try:
                            msg = spec["purge"]()
                            st.session_state[_DONE_KEY] = f"✅ {spec['label']}: {msg}"
                        except Exception as e:
                            st.session_state[_DONE_KEY] = (
                                f"❌ {spec['label']}: purge failed — "
                                f"{type(e).__name__}: {e}"
                            )
                        st.session_state.pop(_ARM_KEY, None)
                        st.rerun()
                    if c2.button("Cancel", key=f"cache_no_{k}", width="stretch"):
                        st.session_state.pop(_ARM_KEY, None)
                        st.rerun()
                else:
                    if st.button(
                        "Purge",
                        key=f"cache_arm_{k}",
                        width="stretch",
                        help=spec["help"],
                    ):
                        st.session_state[_ARM_KEY] = k
                        st.rerun()
                if i < len(_CACHES) - 1:
                    st.divider()
