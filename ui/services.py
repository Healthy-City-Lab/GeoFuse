"""Process-level singletons used by every tab.

These factories use ``@st.cache_resource`` so the underlying objects are shared
across all browser sessions and survive script reruns. Tabs must import from
*this* module, never from ``ui/app.py`` — importing the entry-point script
re-executes it and triggers duplicate-widget-key errors.
"""

from __future__ import annotations

import streamlit as st

from geofuse.persistence.job_executor import JobExecutor
from geofuse.persistence.job_store import JobStore
from geofuse.persistence.pano_cache import PanoCache


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
