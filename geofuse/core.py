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

    The full bounding box can contain billions of cells at fine spacing; building
    one giant ``meshgrid`` would exhaust RAM.  This implementation tiles the
    raster into chunks, clips each chunk to the geometry, and concatenates.
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

    total_cells = width * height
    # Beyond this, even chunked processing is usually impractical for the UI.
    max_cells = int(os.environ.get("GEOFUSE_MAX_GRID_CELLS", "5000000000"))
    if total_cells > max_cells:
        approx_side_m = max(
            (maxx - minx) * m_per_deg_lon, (maxy - miny) * m_per_deg_lat
        )
        min_spacing = max(spacing_meters, int(np.ceil(approx_side_m / 2000)))
        raise ValueError(
            f"Sampling grid would have {width}×{height} = {total_cells:,} cells "
            f"(bounding box ~{approx_side_m / 1000:.1f} km at {spacing_meters} m spacing). "
            f"That exceeds the limit of {max_cells:,} cells. "
            f"Increase grid spacing (try at least ~{min_spacing} m), reduce the "
            f"study area or buffer, or set GEOFUSE_MAX_GRID_CELLS to raise the cap."
        )

    # Fast path: small enough for a single meshgrid allocation.
    max_chunk = int(os.environ.get("GEOFUSE_GRID_CHUNK_CELLS", "2000000"))
    if total_cells <= max_chunk:
        rows, cols = np.meshgrid(
            np.arange(height, dtype=np.int32),
            np.arange(width, dtype=np.int32),
            indexing="ij",
        )
        xs, ys = xy(transform, rows.ravel(), cols.ravel(), offset="center")
        df = pd.DataFrame(
            {"row": rows.ravel(), "col": cols.ravel(), "x": xs, "y": ys}
        )
        gdf_grid = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:4326"
        )
        gdf_clipped = gpd.sjoin(
            gdf_grid, gdf_4326, how="inner", predicate="intersects"
        )
    else:
        col_step = min(
            width,
            max(64, int(np.ceil(np.sqrt(max_chunk * width / max(height, 1))))),
        )
        row_step = min(height, max(1, max_chunk // max(col_step, 1)))
        while col_step * row_step > max_chunk and col_step > 1:
            col_step = max(1, col_step // 2)
            row_step = min(height, max(1, max_chunk // max(col_step, 1)))

        parts: list[gpd.GeoDataFrame] = []
        for r0 in range(0, height, row_step):
            r1 = min(height, r0 + row_step)
            for c0 in range(0, width, col_step):
                c1 = min(width, c0 + col_step)
                rr, cc = np.meshgrid(
                    np.arange(r0, r1, dtype=np.int32),
                    np.arange(c0, c1, dtype=np.int32),
                    indexing="ij",
                )
                xs, ys = xy(transform, rr.ravel(), cc.ravel(), offset="center")
                df = pd.DataFrame(
                    {"row": rr.ravel(), "col": cc.ravel(), "x": xs, "y": ys}
                )
                gdf_chunk = gpd.GeoDataFrame(
                    df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:4326"
                )
                hit = gpd.sjoin(
                    gdf_chunk, gdf_4326, how="inner", predicate="intersects"
                )
                if not hit.empty:
                    hit = hit.drop(columns=["index_right"], errors="ignore")
                    parts.append(hit)

        if not parts:
            gdf_clipped = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        else:
            gdf_clipped = gpd.GeoDataFrame(
                pd.concat(parts, ignore_index=True), crs="EPSG:4326"
            )

    gdf_clipped = gdf_clipped.drop(columns=["index_right"], errors="ignore")
    return gdf_clipped, {
        "transform": transform,
        "width": width,
        "height": height,
        "crs": "EPSG:4326",
    }
