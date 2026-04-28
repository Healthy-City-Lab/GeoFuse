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

from tabs import fusion, gvi, job_monitor, ndvi  # noqa: E402

# --- 5. PAGE CONFIG ---
st.set_page_config(page_title="GeoFuse Toolbox", layout="wide")

st.markdown(
    """
<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    iframe { width: 100% !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("GeoFuse: Multimodal Environmental Profiling")

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
