import argparse
import sys
import os
import pandas as pd

# Ensure we can import from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geofuse.ndvi import NDVIEngine
from geofuse.gvi import GVIEngine
from geofuse.fusion import FusionOptimizer
from geofuse.core import JobTracker


def main():
    parser = argparse.ArgumentParser(
        description="GeoFuse: Multimodal Environmental Toolbox"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available modules")

    # --- NDVI COMMAND ---
    p_ndvi = subparsers.add_parser("ndvi", help="Source Satellite NDVI")
    p_ndvi.add_argument(
        "--input", required=True, help="Path to Geometry file (Shp/GeoJSON)"
    )
    p_ndvi.add_argument("--date", required=True, help="Target Date (YYYY-MM-DD)")
    p_ndvi.add_argument("--output", required=True, help="Output TIF path")
    p_ndvi.add_argument(
        "--tolerance", type=int, default=15, help="Days tolerance for cloud search"
    )
    p_ndvi.add_argument("--job_id", default="local", help="HPC Job ID for logging")

    # --- GVI COMMAND ---
    p_gvi = subparsers.add_parser("gvi", help="Source Street View GVI")
    p_gvi.add_argument("--input", required=True, help="Path to Polygon file")
    p_gvi.add_argument("--output", required=True, help="Output TIF path")

    # Model is now optional (defaults to geofuse/model/best_model.pth)
    p_gvi.add_argument(
        "--model",
        default=None,
        help="Path to DeepLab Checkpoint (.pth). Defaults to internal model folder.",
    )

    p_gvi.add_argument(
        "--mode",
        choices=["api", "package"],
        default="package",
        help="Download mode: 'api' (requires key) or 'package' (no key)",
    )
    p_gvi.add_argument(
        "--key", default=None, help="Google API Key (required for api mode)"
    )
    p_gvi.add_argument(
        "--save_files",
        action="store_true",
        help="If set, saves raw images/masks to disk",
    )
    p_gvi.add_argument("--job_id", default="local")

    # --- FUSION COMMAND ---
    p_opt = subparsers.add_parser("optimize", help="Run Fusion Optimization")
    p_opt.add_argument(
        "--data", required=True, help="CSV with NDVI, GVI, and Outcome columns"
    )
    p_opt.add_argument("--target", required=True, help="Name of the Outcome column")
    p_opt.add_argument(
        "--features",
        nargs="+",
        default=["NDVI", "GVI_Tree", "GVI_Grass"],
        help="Columns to fuse",
    )
    p_opt.add_argument(
        "--total_trials", type=int, default=100, help="Total optimization trials"
    )
    p_opt.add_argument(
        "--random_trials",
        type=int,
        default=20,
        help="Initial random exploration trials",
    )
    p_opt.add_argument("--job_id", default="local")

    args = parser.parse_args()

    # Initialize Logger
    tracker = JobTracker(args.job_id)

    try:
        if args.command == "ndvi":
            tracker.update("Initializing NDVI Engine", 0)
            engine = NDVIEngine()
            success = engine.export_geotiff(
                args.input, args.date, args.output, tolerance=args.tolerance
            )
            if success:
                tracker.update("NDVI Complete", 100)
            else:
                tracker.update("NDVI Failed (No Images)", 100)

        elif args.command == "gvi":
            tracker.update("Initializing GVI Engine", 0)
            engine = GVIEngine(
                api_key=args.key, model_path=args.model, download_mode=args.mode
            )

            tracker.update("Processing Polygons", 10)
            engine.process_polygon(args.input, args.output, save_files=args.save_files)

            tracker.update("GVI Complete", 100)

        elif args.command == "optimize":
            tracker.update("Starting Optimization", 0)
            df = pd.read_csv(args.data)
            optimizer = FusionOptimizer(df, args.target, args.features)

            tracker.update("Running Optuna Trials", 20)
            best_params = optimizer.run_optimization(
                total_trials=args.total_trials, random_trials=args.random_trials
            )

            tracker.update("Finalizing Weights", 90)
            final_df = optimizer.apply_best_weights()

            out_path = args.data.replace(".csv", "_fused.csv")
            final_df.to_csv(out_path, index=False)
            tracker.update("Optimization Complete", 100, metrics=best_params)

        else:
            parser.print_help()

    except Exception as e:
        tracker.update(f"Error: {str(e)}", 100)
        raise e


if __name__ == "__main__":
    main()
