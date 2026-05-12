import os
import sys

# --- 1. GLOBAL PATH SETUP (must happen before other imports) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# Ensure ui/ is on sys.path so tab modules can import helpers
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# --- 2. GDAL ENVIRONMENT FIX ---
if "GDAL_DATA" not in os.environ:
    conda_prefix = sys.prefix
    gdal_data_path = os.path.join(conda_prefix, "Library", "share", "gdal")
    if os.path.exists(gdal_data_path):
        os.environ["GDAL_DATA"] = gdal_data_path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 3. LOAD GEOFUSE (forces torch DLL load on Windows before GDAL stack) ---
from geofuse.gvi import GVIEngine  # noqa: F401, E402
from geofuse.ndvi import NDVIEngine  # noqa: F401, E402
from geofuse.vision import get_best_device  # noqa: F401, E402

try:
    from geofuse.fusion import MetricFusionEngine  # noqa: F401, E402
except ImportError:
    pass

# --- 4. UI IMPORTS (after geofuse to preserve DLL order on Windows) ---
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from services import get_job_executor, get_job_store  # noqa: E402, F401
from tabs import fusion, gvi, job_monitor, ndvi  # noqa: E402

# Warm the singletons once per process so the heartbeat thread starts even
# before any job is submitted.
get_job_store()
get_job_executor()

# --- 5. PAGE CONFIG ---
st.set_page_config(page_title="GeoFuse Toolbox", layout="wide")

st.markdown(
    """
<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    iframe { width: 100% !important; }
    /* Ensure the brand-injection iframe takes no space */
    iframe[height="0"] { display: block; height: 0 !important; min-height: 0 !important; }

    /* Sidebar (Job Monitor on GVI, etc.): compact width + smaller type for all content */
    section[data-testid="stSidebar"] {
        width: min(13.25rem, 28vw) !important;
        min-width: 9.25rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        font-size: 0.74rem !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding: 0.35rem 0.4rem 0.45rem 0.4rem !important;
        font-size: 0.74rem !important;
        max-width: 100% !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-size: 0.92rem !important;
        line-height: 1.25 !important;
        margin-top: 0.2rem !important;
        margin-bottom: 0.3rem !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: 0.72rem !important;
        line-height: 1.35 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stCaption"],
    section[data-testid="stSidebar"] .stCaption {
        font-size: 0.65rem !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        font-size: 0.7rem !important;
        padding: 0.12rem 0.32rem !important;
        min-height: 1.6rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="column"] {
        font-size: 0.72rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
        padding: 0.3rem !important;
    }
    section[data-testid="stSidebar"] .streamlit-expanderHeader {
        font-size: 0.7rem !important;
    }
    section[data-testid="stSidebar"] .stCodeBlock,
    section[data-testid="stSidebar"] pre {
        font-size: 0.65rem !important;
    }
    section[data-testid="stSidebar"] .stAlert {
        font-size: 0.68rem !important;
        padding: 0.3rem 0.4rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stProgress"] > div {
        font-size: 0.65rem !important;
    }

    /* Styling for the GeoFuse brand span injected by JS below */
    .gf-brand {
        font-size: 1.4rem;
        font-weight: 700;
        font-family: "Source Sans Pro", "Source Sans 3", sans-serif;
        letter-spacing: -0.01em;
        white-space: nowrap;
        flex-shrink: 0;
        align-self: center;
        padding-right: 0.9rem;
        margin-right: 0.4rem;
        border-right: 1px solid rgba(49, 51, 63, 0.18);
    }

    /* Muted last tab (Job Monitor): de-emphasized until HPC workflow is finalized */
    button.gf-tab-muted {
        opacity: 0.48 !important;
        color: #8c8c8c !important;
    }
    button.gf-tab-muted[aria-selected="true"] {
        opacity: 0.72 !important;
        color: #6d6d6d !important;
    }

    /* Give the tab bar enough vertical room for the taller brand text */
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        padding-top: 6px;
        padding-bottom: 4px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Brand label in the tab bar: zero-height iframe + MutationObserver (BaseWeb owns ::before).
components.html(
    """
<script>
(function () {
    var d = window.parent.document;

    function inject() {
        var tabList = d.querySelector("[data-baseweb='tab-list']");
        if (!tabList) { setTimeout(inject, 200); return; }
        if (tabList.querySelector(".gf-brand")) return;
        var span = d.createElement("span");
        span.className = "gf-brand";
        span.textContent = "GeoFuse";
        tabList.insertBefore(span, tabList.firstChild);
    }

    function markMutedTab() {
        var tabList = d.querySelector("[data-baseweb='tab-list']");
        if (!tabList) return;
        var tabs = tabList.querySelectorAll("button[role='tab']");
        for (var i = 0; i < tabs.length; i++) {
            tabs[i].classList.remove("gf-tab-muted");
        }
        if (tabs.length > 0) {
            tabs[tabs.length - 1].classList.add("gf-tab-muted");
        }
    }

    new MutationObserver(function () {
        if (!d.querySelector("[data-baseweb='tab-list'] .gf-brand")) inject();
        markMutedTab();
    }).observe(d.body || d.documentElement, { childList: true, subtree: true });

    inject();
    markMutedTab();
})();
</script>
""",
    height=0,
    scrolling=False,
)

tab_ndvi, tab_gvi, tab_fusion, tab_job = st.tabs(
    ["NDVI Sourcing", "GVI Sourcing", "Fusion & Optimization", "Job Monitor"]
)

output_dir = "output_results"
os.makedirs(output_dir, exist_ok=True)
os.makedirs("logs", exist_ok=True)

with tab_ndvi:
    ndvi.render(output_dir)

with tab_gvi:
    gvi.render(output_dir, parent_dir)

with tab_fusion:
    fusion.render(output_dir)

with tab_job:
    job_monitor.render(output_dir)
