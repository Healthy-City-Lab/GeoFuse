import os
import tempfile

import geopandas as gpd
import pandas as pd
import streamlit as st

from geofuse.core import generate_raster_grid  # noqa: F401  (re-exported for UI modules)


@st.cache_data
def load_clean_gdf(file_obj):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".geojson") as tmp:
        tmp.write(file_obj.getvalue())
        tmp_path = tmp.name
    try:
        gdf = gpd.read_file(tmp_path)
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        else:
            gdf = gdf.to_crs("EPSG:4326")
        for col in gdf.columns:
            if (
                pd.api.types.is_datetime64_any_dtype(gdf[col])
                or gdf[col].dtype == "object"
            ):
                try:
                    gdf[col] = gdf[col].astype(str)
                except Exception:
                    gdf = gdf.drop(columns=[col])
        return gdf
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
