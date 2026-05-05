import json
import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd

from .crs_utils import reproject_geodataframe_to_wgs84


class JobTracker:
    """Handles logging for HPC and status updates for the Web UI."""

    def __init__(self, job_id, log_dir="logs"):
        self.job_id = job_id
        self.status_file = os.path.join(log_dir, f"{job_id}_status.json")
        self.log_file = os.path.join(log_dir, f"{job_id}.txt")

        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
        )

    def update(self, stage, percent, metrics=None):
        status = {
            "job_id": self.job_id,
            "stage": stage,
            "progress": percent,
            "metrics": metrics or {},
        }
        with open(self.status_file, "w") as f:
            json.dump(status, f)
        logging.info(f"{stage}: {percent}% - {metrics}")


def load_geometry(input_path):
    """Load Shapefile/GeoJSON and return features in EPSG:4326 (lon/lat as x, y)."""
    gdf = gpd.read_file(input_path)
    if gdf.crs is None:
        raise ValueError("Input geometry missing CRS.")
    return reproject_geodataframe_to_wgs84(gdf)


def generate_raster_grid(gdf_4326, spacing_meters):
    """Create a raster sampling grid clipped to *gdf_4326*.

    Returns a tuple of (GeoDataFrame of grid points, metadata dict).
    The metadata dict contains ``transform``, ``width``, ``height``,
    and ``crs`` entries compatible with rasterio.
    """
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
