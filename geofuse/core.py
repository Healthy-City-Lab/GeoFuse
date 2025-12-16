import logging
import json
import os
import geopandas as gpd


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
    """Loads a file (Shapefile/GeoJSON) and ensures it is a GeoDataFrame."""
    gdf = gpd.read_file(input_path)
    if gdf.crs is None:
        raise ValueError("Input geometry missing CRS.")
    return gdf
