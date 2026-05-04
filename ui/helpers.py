import datetime
import os
import tempfile

import geopandas as gpd
import pandas as pd
import streamlit as st

from geofuse.core import (  # noqa: F401  (re-exported for UI modules)
    generate_raster_grid,
)


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


def sanitize_gdf_attributes_for_json(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Cast non-geometry attributes to JSON-friendly values (Folium, ``json.dumps``).

    GeoJSON sources with date/time properties are often read as ``datetime64`` or
    ``Timestamp`` in object columns; Folium's ``GeoJson(gdf)`` then fails with
    "Object of type Timestamp is not JSON serializable".
    """
    geom_col = gdf.geometry.name
    out = gdf.copy()
    for col in out.columns:
        if col == geom_col:
            continue
        s = out[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            out[col] = s.astype(str)
        elif isinstance(s.dtype, pd.CategoricalDtype):
            out[col] = s.astype(str)
        elif s.dtype == object:

            def _cell(v):
                if isinstance(v, (pd.Timestamp, datetime.datetime, datetime.date)):
                    return v.isoformat()
                if isinstance(v, datetime.time):
                    return str(v)
                return v

            out[col] = s.map(_cell)
    return out


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
        return sanitize_gdf_attributes_for_json(gdf)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
