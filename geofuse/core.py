import json
import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features
from rasterio.transform import from_bounds, xy

from .crs_utils import reproject_geodataframe_to_wgs84

# Inside-buffer vs outside-buffer in the raster mask (any value other than *fill* works).
_RASTER_INSIDE = 1
_RASTER_OUTSIDE = 0


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


def _geometry_union_all(geoms: gpd.GeoSeries):
    if hasattr(geoms, "union_all"):
        return geoms.union_all()
    return geoms.unary_union


def generate_raster_grid(gdf_4326, spacing_meters):
    """Build a regular lon/lat grid over the union of *gdf_4326*, then keep cell centres inside it.

    Steps: axis-aligned bbox of the union of geometries (callers typically pass buffered
    features), raster at *spacing_meters* with value ``_RASTER_INSIDE`` under the union and
    ``_RASTER_OUTSIDE`` elsewhere, then emit points at pixel centres for inside cells.

    Returns ``(GeoDataFrame with row, col, x, y, geometry``, metadata dict with
    ``transform``, ``width``, ``height``, ``crs``).
    """
    if gdf_4326 is None or gdf_4326.empty:
        empty = gpd.GeoDataFrame(
            columns=["row", "col", "x", "y"], geometry=[], crs="EPSG:4326"
        )
        ident = from_bounds(0, 0, 1, 1, 1, 1)
        return empty, {
            "transform": ident,
            "width": 1,
            "height": 1,
            "crs": "EPSG:4326",
        }

    geom_union = _geometry_union_all(gdf_4326.geometry)
    if geom_union is None or geom_union.is_empty:
        empty = gpd.GeoDataFrame(
            columns=["row", "col", "x", "y"], geometry=[], crs="EPSG:4326"
        )
        ident = from_bounds(0, 0, 1, 1, 1, 1)
        return empty, {
            "transform": ident,
            "width": 1,
            "height": 1,
            "crs": "EPSG:4326",
        }

    minx, miny, maxx, maxy = geom_union.bounds
    center_lat = (miny + maxy) / 2.0
    lat_rad = np.radians(center_lat)
    m_per_deg_lat = 111132.92 - 559.82 * np.cos(2 * lat_rad)
    m_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)
    res_x = spacing_meters / m_per_deg_lon
    res_y = spacing_meters / m_per_deg_lat
    width = max(1, int(np.ceil(float(maxx - minx) / float(res_x))))
    height = max(1, int(np.ceil(float(maxy - miny) / float(res_y))))

    transform = from_bounds(
        minx, miny, minx + (width * res_x), miny + (height * res_y), width, height
    )

    mask = features.rasterize(
        [(geom_union, _RASTER_INSIDE)],
        out_shape=(height, width),
        transform=transform,
        fill=_RASTER_OUTSIDE,
        dtype=np.uint8,
        all_touched=True,
    )

    rows, cols = np.nonzero(mask == _RASTER_INSIDE)
    if rows.size == 0:
        gdf_pts = gpd.GeoDataFrame(
            columns=["row", "col", "x", "y"], geometry=[], crs="EPSG:4326"
        )
    else:
        xs, ys = xy(transform, rows, cols, offset="center")
        x_arr = np.asarray(xs, dtype=np.float64).ravel()
        y_arr = np.asarray(ys, dtype=np.float64).ravel()
        df = pd.DataFrame(
            {
                "row": rows.astype(np.int64),
                "col": cols.astype(np.int64),
                "x": x_arr,
                "y": y_arr,
            }
        )
        gdf_pts = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:4326"
        )

    return gdf_pts, {
        "transform": transform,
        "width": width,
        "height": height,
        "crs": "EPSG:4326",
    }
