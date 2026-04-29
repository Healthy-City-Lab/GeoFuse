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
from tabs import fusion, gvi, job_monitor, ndvi  # noqa: E402

# --- 5. PAGE CONFIG ---
st.set_page_config(page_title="GeoFuse Toolbox", layout="wide")

st.markdown(
    """
<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    iframe { width: 100% !important; }
    /* Ensure the brand-injection iframe takes no space */
    iframe[height="0"] { display: block; height: 0 !important; min-height: 0 !important; }

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

    /* Give the tab bar enough vertical room for the taller brand text */
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        padding-top: 6px;
        padding-bottom: 4px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Inject the "GeoFuse" brand as the first flex item in the tab bar.
# Placed BEFORE st.tabs() so the zero-height iframe sits above the tab bar
# in the DOM and cannot clip or squish the tab bar height.
# CSS ::before cannot be used because BaseWeb already claims it for the
# sliding active-tab underline indicator.
# A MutationObserver re-injects the span after every Streamlit re-render.
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

    new MutationObserver(function () {
        if (!d.querySelector("[data-baseweb='tab-list'] .gf-brand")) inject();
    }).observe(d.body || d.documentElement, { childList: true, subtree: true });

    inject();
})();
</script>
""",
    height=0,
    scrolling=False,
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Job Monitor", "NDVI Sourcing", "GVI Sourcing", "Fusion & Optimization"]
)

output_dir = "output_results"
os.makedirs(output_dir, exist_ok=True)
os.makedirs("logs", exist_ok=True)

with tab1:
    job_monitor.render(output_dir)

with tab2:
    ndvi.render(output_dir)

with tab3:
    gvi.render(output_dir, parent_dir)

with tab4:
    fusion.render(output_dir)
