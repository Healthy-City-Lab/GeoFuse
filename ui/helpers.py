import os
import tempfile

import geopandas as gpd
import pandas as pd
import streamlit as st

from geofuse.core import generate_raster_grid  # noqa: F401  (re-exported for UI modules)


def apply_buffer_m(gdf: gpd.GeoDataFrame, buffer_m: float) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame whose geometry is the union of ``gdf`` buffered by
    ``buffer_m`` metres, re-projected back to the original CRS.

    If ``buffer_m`` is 0 the original GeoDataFrame is returned unchanged.
    """
    if buffer_m <= 0:
        return gdf
    utm_crs = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm_crs)
    buffered_geom = gdf_utm.geometry.union_all().buffer(buffer_m)
    result = gpd.GeoDataFrame({"geometry": [buffered_geom]}, crs=utm_crs)
    return result.to_crs(gdf.crs)


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
