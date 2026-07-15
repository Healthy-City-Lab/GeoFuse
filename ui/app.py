import os
import sys

if __name__ == "__main__":
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
    from cache_manager import render_cache_manager  # noqa: E402
    from chrome import inject_chrome  # noqa: E402
    from resource_monitor import render_resource_monitor  # noqa: E402
    from services import get_job_executor, get_job_store  # noqa: E402, F401
    from tabs import fusion, gvi, job_monitor, ndvi  # noqa: E402

    # Warm the singletons once per process so the heartbeat thread starts even
    # before any job is submitted.
    get_job_store()
    get_job_executor()

    # --- 5. PAGE CONFIG ---
    st.set_page_config(page_title="GeoFuse Toolbox", layout="wide")

    # Global page chrome: CSS, floater toggle, and the MutationObserver
    # iframe that relocates tabs / brand / sidebar control. See ui/chrome.py.
    inject_chrome()

    tab_ndvi, tab_gvi, tab_fusion, tab_job = st.tabs(
        ["NDVI", "GVI", "Fusion & Optimization", "HPC Monitoring"]
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

    # Cache management lives in the sidebar and is rendered here (not inside a
    # tab) so it is reachable from every tab — the job monitor is only mounted
    # by the GVI tab.
    render_cache_manager(output_dir)

    # Floating bottom-right resource monitor (CPU/RAM/GPU/Network).
    # Rendered outside any tab so it stays visible everywhere; uses
    # position:fixed internally, so it doesn't affect page layout.
    render_resource_monitor()
