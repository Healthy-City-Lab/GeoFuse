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

import os
import shutil

import streamlit as st

_ARM_KEY = "_cache_purge_armed"
_DONE_KEY = "_cache_purge_done"


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024.0
    return f"{n:,.1f} PB"


def _dir_size(path: str) -> tuple[int, int]:
    """(bytes, file_count) for a directory tree; (0, 0) when absent."""
    if not os.path.isdir(path):
        return 0, 0
    total = files = 0
    for root, _dirs, names in os.walk(path):
        for nm in names:
            try:
                total += os.path.getsize(os.path.join(root, nm))
                files += 1
            except OSError:
                pass
    return total, files


# ────────────────────────────────────────────────────────────────────
# Cache descriptors — each knows how to measure itself and how to purge itself.
# ────────────────────────────────────────────────────────────────────


def _pano_stats() -> str:
    from services import get_pano_cache

    try:
        return f"{len(get_pano_cache()):,} panoramas"
    except Exception as e:
        return f"unavailable ({type(e).__name__})"


def _pano_purge() -> str:
    from services import get_pano_cache

    cache = get_pano_cache()
    n = len(cache)
    cache.clear()
    return f"Cleared {n:,} cached panorama GVI values."


def _ndvi_stats() -> str:
    from services import get_ndvi_tile_cache

    try:
        s = get_ndvi_tile_cache().stats()
        return f"{_human_bytes(s.get('total_bytes', 0))} · {s.get('entries', 0):,} workspaces"
    except Exception as e:
        return f"unavailable ({type(e).__name__})"


def _ndvi_purge() -> str:
    from services import get_ndvi_tile_cache

    n, freed = get_ndvi_tile_cache().evict_lru(0)
    return f"Removed {n:,} tile workspaces, freed {_human_bytes(freed)}."


def _fusion_dir() -> str:
    return os.path.join(
        st.session_state.get("_gf_output_dir", "output_results"), "fusion_cache"
    )


def _fusion_stats() -> str:
    b, f = _dir_size(_fusion_dir())
    return f"{_human_bytes(b)} · {f:,} files" if f else "empty"


def _fusion_purge() -> str:
    p = _fusion_dir()
    b, f = _dir_size(p)
    if not f:
        return "Fusion cache was already empty."
    shutil.rmtree(p, ignore_errors=True)
    return f"Removed {f:,} files, freed {_human_bytes(b)}."


_CACHES = [
    {
        "key": "pano",
        "label": "GVI panorama cache",
        "help": (
            "Segmented GVI values keyed by panorama id (`logs/caches/gvi_panos.db`). "
            "Purge this after any change to the segmentation metric — otherwise "
            "old runs keep returning the previously-computed numbers."
        ),
        "stats": _pano_stats,
        "purge": _pano_purge,
    },
    {
        "key": "ndvi",
        "label": "NDVI tile cache",
        "help": (
            "Downloaded Earth Engine tile workspaces (`logs/caches/ndvi_tiles/`). "
            "Safe to purge; the next run re-downloads what it needs."
        ),
        "stats": _ndvi_stats,
        "purge": _ndvi_purge,
    },
    {
        "key": "fusion",
        "label": "Fusion pre-aggregation cache",
        "help": (
            "Content-addressed fusion inputs (`output_results/fusion_cache/`). "
            "Purging only forces re-computation — finished job reports and "
            "artifacts are kept and stay loadable."
        ),
        "stats": _fusion_stats,
        "purge": _fusion_purge,
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
