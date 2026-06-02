import json
import os
import time

import streamlit as st


def render(output_dir: str) -> None:
    st.header("HPC Monitoring")

    with st.form("job_mon_form"):
        with st.container(border=True):
            st.text_input(
                "Enter Job ID (e.g., 42591):",
                key="job_mon_job_id",
                help=(
                    "ID from your HPC submission. Values apply when you press "
                    "Load status; editing alone does not refresh the status panel."
                ),
            )
            st.checkbox(
                "Auto-refresh (2s)",
                value=True,
                key="job_mon_auto",
                help="Reload status every 2 seconds until the job reaches 100%.",
            )
        load = st.form_submit_button(
            "Load status", width="stretch", key="job_load_status"
        )

    if load:
        job_id = (st.session_state.get("job_mon_job_id") or "").strip()
        auto_refresh = st.session_state.get("job_mon_auto", True)
        if job_id:
            status_file = os.path.join("logs", f"{job_id}_status.json")
            if os.path.exists(status_file):
                placeholder = st.empty()
                while True:
                    try:
                        with open(status_file) as f:
                            data = json.load(f)
                    except Exception:
                        time.sleep(0.5)
                        continue
                    with placeholder.container():
                        st.subheader(f"Stage: {data.get('stage', 'Unknown')}")
                        st.progress(data.get("progress", 0))
                        if data.get("metrics"):
                            st.json(data.get("metrics"))
                    if not auto_refresh or data.get("progress", 0) >= 100:
                        break
                    time.sleep(2)
            else:
                st.info("Waiting for job...")
        else:
            st.info("Enter a job ID and press Load status.")
