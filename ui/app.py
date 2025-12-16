import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import json
import os
import sys
import time
import matplotlib.pyplot as plt

# Add parent dir to path so we can import geofuse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geofuse.fusion import FusionOptimizer

st.set_page_config(page_title="GeoFuse Toolbox", layout="wide")
st.title("GeoFuse: Multimodal Environmental Profiling")

# Tabs for the different modules (CLEAN TEXT ONLY)
tab1, tab2, tab3, tab4 = st.tabs(
    ["Job Monitor", "NDVI Sourcing", "GVI Sourcing", "Fusion & Optimization"]
)

# -----------------------------------------------------------------------------
# TAB 1: HPC JOB MONITOR
# -----------------------------------------------------------------------------
with tab1:
    st.header("HPC Job Monitor")
    st.markdown("Track the progress of remote HPC jobs via their Job ID.")

    col1, col2 = st.columns([1, 3])
    with col1:
        job_id = st.text_input("Enter Job ID (e.g., 42591):")
        auto_refresh = st.checkbox("Auto-refresh (2s)", value=True)

    with col2:
        if job_id:
            status_file = os.path.join("logs", f"{job_id}_status.json")

            if os.path.exists(status_file):
                placeholder = st.empty()

                while True:
                    try:
                        with open(status_file, "r") as f:
                            data = json.load(f)
                    except json.JSONDecodeError:
                        time.sleep(0.5)
                        continue

                    with placeholder.container():
                        st.subheader(f"Stage: {data.get('stage', 'Unknown')}")
                        prog = data.get("progress", 0)
                        st.progress(prog)

                        metrics = data.get("metrics", {})
                        if metrics:
                            st.json(metrics)

                        log_file = os.path.join("logs", f"{job_id}.txt")
                        if os.path.exists(log_file):
                            with st.expander("Live Logs (Tail)"):
                                with open(log_file, "r") as lf:
                                    lines = lf.readlines()[-10:]
                                    st.code("".join(lines))

                    if not auto_refresh or prog >= 100:
                        break
                    time.sleep(2)
            else:
                st.info("Waiting for job to start or invalid Job ID...")

# -----------------------------------------------------------------------------
# TAB 2: NDVI CONFIGURATOR
# -----------------------------------------------------------------------------
with tab2:
    st.header("Satellite Data Configuration")
    col_a, col_b = st.columns(2)

    with col_a:
        st.date_input("Target Date")
        st.number_input("Cloud Tolerance (Days)", value=15, min_value=1)
        st.file_uploader("Upload Catchment Area (Shapefile/GeoJSON)", key="ndvi_up")

    with col_b:
        st.info("Generates a CLI command for HPC execution.")
        st.code(
            "python scripts/cli.py ndvi --input data/catchment.shp --date 2024-06-01 --output results/ndvi.tif"
        )

# -----------------------------------------------------------------------------
# TAB 3: GVI CONFIGURATOR
# -----------------------------------------------------------------------------
with tab3:
    st.header("Street View & Vision Pipeline")

    mode = st.radio("Download Mode", ["Package (Scraper)", "API (Google Key)"])
    if mode == "API (Google Key)":
        st.text_input("Google API Key", type="password")

    st.checkbox(
        "Save Raw Images & Masks?",
        value=False,
        help="If unchecked, runs in-memory to save storage.",
    )

    st.code(
        "python scripts/cli.py gvi --mode package --input data/routes.shp --output results/gvi.tif"
    )

# -----------------------------------------------------------------------------
# TAB 4: FUSION DEMO
# -----------------------------------------------------------------------------
with tab4:
    st.header("Composite Metric Fusion")
    st.markdown(
        "Interactive demonstration of how weights affect the outcome correlation."
    )

    uploaded_csv = st.file_uploader("Upload Merged Data (CSV)", type="csv")

    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.write("Data Preview:", df.head())

        target = st.selectbox("Target Outcome Variable", df.columns)
        features = st.multiselect(
            "Environmental Features",
            df.columns,
            default=[c for c in df.columns if "NDVI" in c or "GVI" in c],
        )

        if features and target:
            col_opt, col_man = st.columns(2)

            with col_man:
                st.subheader("Manual Weighting")
                weights = {}
                for f in features:
                    weights[f] = st.slider(f"Weight: {f}", 0.0, 1.0, 0.5, key=f)

                df["Manual_CGI"] = 0
                for f, w in weights.items():
                    df["Manual_CGI"] += df[f] * w

                corr = df["Manual_CGI"].corr(df[target])
                st.metric("Pearson Correlation", f"{corr:.3f}")

            with col_opt:
                st.subheader("AI Optimization (Optuna)")
                n_trials = st.number_input("Optimization Trials", 50, 500, 100)

                if st.button("Run Optimization"):
                    with st.spinner("Optimizing..."):
                        opt = FusionOptimizer(df, target, features)
                        best_params = opt.run_optimization(
                            total_trials=n_trials, random_trials=20
                        )

                        st.success("Optimization Complete!")
                        st.json(best_params)

                        df_opt = opt.apply_best_weights()
                        best_corr = df_opt["CGI"].corr(df_opt[target])
                        st.metric("Optimized Correlation", f"{best_corr:.3f}")

                        delta = best_corr - corr
                        st.markdown(f"**Improvement over manual:** {delta:+.3f}")
