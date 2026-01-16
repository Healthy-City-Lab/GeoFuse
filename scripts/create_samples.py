import os

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon


def create_sample_data():
    os.makedirs("data/samples", exist_ok=True)

    # 1. Create a Sample Polygon (University of Calgary Park)
    # Coordinates are approx WGS84
    lat_point_list = [51.077, 51.077, 51.079, 51.079]
    lon_point_list = [-114.135, -114.132, -114.132, -114.135]

    polygon_geom = Polygon(zip(lon_point_list, lat_point_list))
    gdf = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[polygon_geom])

    # Save as GeoJSON (for NDVI) and Shapefile (for GVI)
    gdf.to_file("data/samples/test_area.geojson", driver="GeoJSON")
    gdf.to_file("data/samples/test_area.shp")
    print("[OK] Created sample geometry: data/samples/test_area.geojson")

    # 2. Create Sample Fusion Data (Synthetic)
    # We create a relationship: Outcome = 0.5*NDVI + 0.3*Trees + Noise
    np.random.seed(42)
    n = 200
    df = pd.DataFrame(
        {
            "id": range(n),
            "NDVI": np.random.uniform(0.1, 0.8, n),
            "GVI_Tree": np.random.uniform(0.0, 0.5, n),
            "GVI_Grass": np.random.uniform(0.0, 0.3, n),
        }
    )
    # Synthetic Outcome
    df["CognitiveScore"] = (
        (0.5 * df["NDVI"]) + (0.3 * df["GVI_Tree"]) + np.random.normal(0, 0.05, n)
    )

    df.to_csv("data/samples/test_fusion.csv", index=False)
    print("[OK] Created sample data: data/samples/test_fusion.csv")


if __name__ == "__main__":
    create_sample_data()
