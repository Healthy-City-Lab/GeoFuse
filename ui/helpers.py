import os
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st


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


def generate_raster_grid(gdf_4326, spacing_meters):
    from rasterio.transform import from_bounds, xy

    minx, miny, maxx, maxy = gdf_4326.total_bounds
    center_lat = (miny + maxy) / 2.0
    lat_rad = np.radians(center_lat)
    m_per_deg_lat = 111132.92 - 559.82 * np.cos(2 * lat_rad)
    m_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)
    res_x = spacing_meters / m_per_deg_lon
    res_y = spacing_meters / m_per_deg_lat
    width = int(np.ceil((maxx - minx) / res_x))
    height = int(np.ceil((maxy - miny) / res_y))
    transform = from_bounds(
        minx, miny, minx + (width * res_x), miny + (height * res_y), width, height
    )
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    xs, ys = xy(transform, rows.flatten(), cols.flatten(), offset="center")
    df = pd.DataFrame({"row": rows.flatten(), "col": cols.flatten(), "x": xs, "y": ys})
    gdf_grid = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:4326"
    )
    gdf_clipped = gpd.sjoin(gdf_grid, gdf_4326, how="inner", predicate="intersects")
    return gdf_clipped, {
        "transform": transform,
        "width": width,
        "height": height,
        "crs": "EPSG:4326",
    }
