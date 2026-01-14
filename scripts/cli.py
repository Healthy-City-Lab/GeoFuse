#!/usr/bin/env python
# coding: utf-8

import sys
import os
import argparse
import time
import json
import gc
import signal
import warnings
import psutil
from collections import deque
from datetime import datetime

import pandas as pd
import geopandas as gpd
import numpy as np
from mpi4py import MPI

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from geofuse.ndvi import NDVIEngine
from geofuse.gvi import GVIEngine
from streetview import search_panoramas, get_streetview

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------


def resolve_path(base_path, target_path):
    """Resolves file paths relative to the CSV location."""
    if os.path.exists(target_path):
        return target_path
    base_dir = os.path.dirname(os.path.abspath(base_path))
    rel_path = os.path.join(base_dir, target_path)
    if os.path.exists(rel_path):
        return rel_path
    return target_path


def load_config(csv_path):
    """Loads and validates the configuration CSV."""
    df = pd.read_csv(csv_path)
    # Added 'metric_type' to requirements
    required_cols = ["name", "geojson", "metric_type", "start_date", "end_date"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must contain column: {col}")

    # Standardize metric type (uppercase)
    df["metric_type"] = df["metric_type"].str.upper().str.strip()

    # Fix paths
    df["geojson"] = df["geojson"].apply(lambda x: resolve_path(csv_path, x))

    # Format Dates for Filenames (Remove dashes/slashes)
    df["date_suffix"] = df.apply(
        lambda r: f"{str(r['start_date']).replace('-','').replace('/','')}-{str(r['end_date']).replace('-','').replace('/','')}",
        axis=1,
    )
    return df


def generate_gvi_points(config_df, resolution_m):
    """
    Generates a master list of sampling points ONLY for rows marked 'GVI'.
    """
    master_list = []

    # Filter for GVI tasks only
    gvi_tasks = config_df[config_df["metric_type"] == "GVI"]

    if gvi_tasks.empty:
        return gpd.GeoDataFrame()

    print(f"[Rank 0] Generating GVI sampling grid for {len(gvi_tasks)} tasks...")

    from rasterio.transform import from_bounds, xy

    for _, row in gvi_tasks.iterrows():
        name = row["name"]
        suffix = row["date_suffix"]
        # Construct unique name: Name + Date Range
        unique_name = f"{name}_{suffix}"

        geo_path = row["geojson"]

        if not os.path.exists(geo_path):
            print(f"[WARN] GeoJSON not found: {geo_path}. Skipping.")
            continue

        try:
            gdf = gpd.read_file(geo_path)
            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)
            elif gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs("EPSG:4326")

            minx, miny, maxx, maxy = gdf.total_bounds

            # Metric grid estimation
            center_lat = (miny + maxy) / 2.0
            lat_rad = np.radians(center_lat)
            m_per_deg_lat = 111132.92 - 559.82 * np.cos(2 * lat_rad)
            m_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)

            res_x = resolution_m / m_per_deg_lon
            res_y = resolution_m / m_per_deg_lat

            width = int(np.ceil((maxx - minx) / res_x))
            height = int(np.ceil((maxy - miny) / res_y))

            transform = from_bounds(
                minx,
                miny,
                minx + (width * res_x),
                miny + (height * res_y),
                width,
                height,
            )
            cols, rows = np.meshgrid(np.arange(width), np.arange(height))
            xs, ys = xy(transform, rows.flatten(), cols.flatten(), offset="center")

            df_pts = pd.DataFrame({"x": xs, "y": ys})
            gdf_pts = gpd.GeoDataFrame(
                df_pts, geometry=gpd.points_from_xy(df_pts.x, df_pts.y), crs="EPSG:4326"
            )

            # Clip
            gdf_clipped = gpd.sjoin(gdf_pts, gdf, how="inner", predicate="intersects")

            # Metadata
            gdf_clipped["task_id"] = unique_name  # Unique ID for saving later
            gdf_clipped["aoi_name"] = name
            gdf_clipped["date_range"] = suffix
            gdf_clipped["gvi_veg"] = np.nan
            gdf_clipped["gvi_ter"] = np.nan
            gdf_clipped["pano_id"] = None
            gdf_clipped["status"] = "pending"

            master_list.append(gdf_clipped)
            print(f"  > {unique_name}: {len(gdf_clipped)} points generated.")

        except Exception as e:
            print(f"[ERROR] Failed to process {name}: {e}")

    if not master_list:
        return gpd.GeoDataFrame()

    return pd.concat(master_list, ignore_index=True)


# ------------------------------------------------------------------------------
# MAIN CLI
# ------------------------------------------------------------------------------


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser(description="GeoFuse HPC CLI")

    # Inputs
    parser.add_argument("--config", type=str, required=True, help="Configuration CSV.")
    parser.add_argument(
        "--output_dir", type=str, default="output_cli", help="Output directory."
    )
    parser.add_argument(
        "--checkpoint", type=str, default="gvi_checkpoint.pkl", help="Checkpoint file."
    )

    # Settings
    parser.add_argument(
        "--ee_project", type=str, default=None, help="Google Cloud Project."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="geofuse/model/best_model.pth",
        help="DeepLabV3+ model path.",
    )
    parser.add_argument(
        "--gvi_res", type=int, default=50, help="GVI Grid resolution (m)."
    )
    parser.add_argument(
        "--api_key", type=str, default=None, help="Street View API Key."
    )

    # HPC Tuning
    parser.add_argument("--max_batch_size", type=int, default=20)
    parser.add_argument("--min_batch_size", type=int, default=2)
    parser.add_argument("--target_batch_seconds", type=int, default=120)
    parser.add_argument("--heartbeat_interval", type=int, default=30)
    parser.add_argument("--memory_soft_limit_mb", type=int, default=8000)
    parser.add_argument("--download_timeout", type=int, default=15)

    args = parser.parse_args()

    # MPI Tags
    TAG_REQ = 10
    TAG_ASSIGN = 11
    TAG_UPDATE = 88
    TAG_HEARTBEAT = 55
    TAG_REQUEUE = 66
    TAG_DONE = 99

    # --------------------------------------------------------------------------
    # COORDINATOR (RANK 0)
    # --------------------------------------------------------------------------
    if rank == 0:
        print(f"--- GeoFuse CLI Started (Coordinator: Rank 0 | Workers: {size-1}) ---")
        os.makedirs(args.output_dir, exist_ok=True)

        # 1. Load Config
        try:
            config = load_config(args.config)
            print(f"[Init] Loaded config with {len(config)} rows.")
        except Exception as e:
            print(f"[Fatal] Config error: {e}")
            for w in range(1, size):
                comm.send(([], 0), dest=w, tag=TAG_ASSIGN)
            sys.exit(1)

        # 2. NDVI PHASE (Sequential)
        # Filter for NDVI tasks only
        ndvi_tasks = config[config["metric_type"] == "NDVI"]

        if not ndvi_tasks.empty:
            print("\n=== PHASE 1: NDVI PROCESSING ===")
            try:
                ndvi_engine = NDVIEngine(project_id=args.ee_project)

                for _, row in ndvi_tasks.iterrows():
                    # Construct unique filename with date
                    out_name = f"{row['name']}_{row['date_suffix']}"
                    print(f"[NDVI] Processing {out_name}...")

                    try:
                        gdf_aoi = gpd.read_file(row["geojson"])
                        if gdf_aoi.crs.to_epsg() != 4326:
                            gdf_aoi = gdf_aoi.to_crs("EPSG:4326")

                        # Execute
                        res = ndvi_engine.download_and_process(
                            geometry=gdf_aoi,
                            start_date=row["start_date"],
                            end_date=row["end_date"],
                            output_name=out_name,  # Pass unique name
                            cloud_max=row.get("cloud_max", 10),
                            resolution=row.get("resolution", 10),
                            folder=args.output_dir,
                        )

                        if res.get("status") == "success":
                            print(f"  -> Saved: {os.path.basename(res['tif'])}")
                        else:
                            print(f"  -> Failed: {res['message']}")
                    except Exception as e:
                        print(f"  -> Error: {e}")
            except Exception as e:
                print(f"[NDVI] Engine Init Failed: {e}. Skipping.")
        else:
            print("\n[NDVI] No NDVI tasks found in config. Skipping Phase 1.")

        # 3. GVI PREP PHASE
        print("\n=== PHASE 2: GVI PREPARATION ===")
        ckpt_path = os.path.join(args.output_dir, args.checkpoint)

        if os.path.exists(ckpt_path):
            print(f"[Init] Resuming from checkpoint: {ckpt_path}")
            master_gdf = pd.read_pickle(ckpt_path)
        else:
            # Generate points ONLY for 'GVI' rows
            master_gdf = generate_gvi_points(config, args.gvi_res)
            if not master_gdf.empty:
                master_gdf.to_pickle(ckpt_path)

        # 4. DISTRIBUTED PROCESSING (If GVI tasks exist)
        if master_gdf.empty:
            print("[Info] No GVI tasks pending. Exiting.")
            for w in range(1, size):
                comm.send(([], 0), dest=w, tag=TAG_ASSIGN)
            return

        unprocessed_mask = master_gdf["gvi_veg"].isna()
        task_indices = master_gdf.index[unprocessed_mask].tolist()
        task_queue = deque(task_indices)

        print(f"[Init] GVI Tasks: {len(task_queue)} pending / {len(master_gdf)} total")
        print("\n=== PHASE 3: DISTRIBUTED GVI PROCESSING ===")

        # Tracking Variables
        active_batches = {}
        worker_rates = {}
        worker_status = {
            w: {"processed": 0, "last_hb": time.time(), "status": "Idle"}
            for w in range(1, size)
        }
        processed_session = 0
        requeued_count = 0
        last_report = time.time()
        start_time = time.time()

        # Coordinator Loop
        while True:
            if not task_queue and not active_batches:
                break

            now = time.time()

            # A. Requests
            while comm.iprobe(source=MPI.ANY_SOURCE, tag=TAG_REQ):
                req = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_REQ)
                w_rank, last_size, last_dur, mem_flag = req

                # Rate calc
                if last_size > 0 and last_dur > 0:
                    rate = last_size / last_dur
                    old = worker_rates.get(w_rank, rate)
                    worker_rates[w_rank] = 0.7 * old + 0.3 * rate

                worker_status[w_rank]["last_hb"] = now
                active_batches.pop(w_rank, None)

                if not task_queue:
                    comm.send(([], 0), dest=w_rank, tag=TAG_ASSIGN)
                    worker_status[w_rank]["status"] = "Done"
                else:
                    rate = worker_rates.get(w_rank, 0.5)
                    ideal = int(rate * args.target_batch_seconds)
                    b_size = max(args.min_batch_size, min(args.max_batch_size, ideal))
                    if mem_flag:
                        b_size = max(args.min_batch_size, b_size // 2)

                    batch_idx = []
                    batch_payload = []
                    for _ in range(min(b_size, len(task_queue))):
                        idx = task_queue.popleft()
                        pt = master_gdf.loc[idx].geometry
                        batch_idx.append(idx)
                        batch_payload.append((idx, pt.y, pt.x))

                    active_batches[w_rank] = {"indices": batch_idx, "time": now}
                    worker_status[w_rank]["status"] = f"Batch({len(batch_idx)})"
                    comm.send((batch_payload, b_size), dest=w_rank, tag=TAG_ASSIGN)

            # B. Updates
            while comm.iprobe(source=MPI.ANY_SOURCE, tag=TAG_UPDATE):
                msg = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_UPDATE)
                w_rank, idx, res_dict, _ = msg

                if res_dict:
                    master_gdf.at[idx, "gvi_veg"] = res_dict.get("gvi_veg")
                    master_gdf.at[idx, "gvi_ter"] = res_dict.get("gvi_ter")
                    master_gdf.at[idx, "pano_id"] = res_dict.get("pano_id")
                    master_gdf.at[idx, "status"] = "done"
                else:
                    master_gdf.at[idx, "status"] = "failed"

                processed_session += 1
                worker_status[w_rank]["processed"] += 1

                # Auto-save every 500
                if processed_session % 500 == 0:
                    master_gdf.to_pickle(ckpt_path)

            # C. Requeues
            while comm.iprobe(source=MPI.ANY_SOURCE, tag=TAG_REQUEUE):
                w_rank, idx_list = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_REQUEUE)
                requeued_count += len(idx_list)
                task_queue.extendleft(idx_list)
                active_batches.pop(w_rank, None)
                worker_status[w_rank]["status"] = "Requeuing"

            # D. Heartbeats
            while comm.iprobe(source=MPI.ANY_SOURCE, tag=TAG_HEARTBEAT):
                w_rank, _ = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_HEARTBEAT)
                worker_status[w_rank]["last_hb"] = now

            # E. Report
            if now - last_report > 60:  # Every minute
                elapsed = now - start_time
                done_total = len(master_gdf) - len(task_queue)
                pct = (done_total / len(master_gdf) * 100) if len(master_gdf) > 0 else 0
                rate_global = processed_session / max(1, elapsed)
                rem_tasks = len(task_queue) + sum(
                    len(b["indices"]) for b in active_batches.values()
                )
                eta_min = (rem_tasks / rate_global / 60) if rate_global > 0 else 0

                print(
                    f"\n[Status] {datetime.now().strftime('%H:%M')} | Done: {done_total}/{len(master_gdf)} ({pct:.1f}%) | Speed: {rate_global:.1f} it/s | ETA: {eta_min:.1f} min"
                )
                print("╔══════╤════════════╤══════════════╤══════════╗")
                print("║ Rank │ Status     │ Processed    │ Last HB  ║")
                print("╠══════╪════════════╪══════════════╪══════════╣")
                for w in range(1, size):
                    s = worker_status[w]
                    hb = now - s["last_hb"]
                    print(
                        f"║ {w:4d} │ {s['status']:10s} │ {s['processed']:12d} │ {hb:6.1f}s  ║"
                    )
                print("╚══════╧════════════╧══════════════╧══════════╝")
                last_report = now

            time.sleep(0.02)

        # Final Save & Split
        master_gdf.to_pickle(ckpt_path)
        print("\n[Finalizing] Splitting results by task ID...")

        # Save individual GeoJSONs for each task ID
        for task_id in master_gdf["task_id"].unique():
            subset = master_gdf[master_gdf["task_id"] == task_id].copy()
            # Clean up temp columns
            out_cols = [
                c
                for c in subset.columns
                if c not in ["task_id", "status", "date_range"]
            ]
            subset = subset[out_cols]

            out_name = f"{task_id}_gvi.geojson"
            out_path = os.path.join(args.output_dir, out_name)
            subset.to_file(out_path, driver="GeoJSON")
            print(f"  -> Saved {out_name}")

        print(f"[Done] All tasks completed.")
        for w in range(1, size):
            comm.send(([], 0), dest=w, tag=TAG_ASSIGN)

    # --------------------------------------------------------------------------
    # WORKER (RANK > 0)
    # --------------------------------------------------------------------------
    else:
        try:
            engine = GVIEngine(
                model_path=args.model_path, device="cuda", api_key=args.api_key
            )
        except Exception as e:
            # If init fails, sleep to avoid crashing everything immediately
            while True:
                time.sleep(10)
            sys.exit(1)

        class TimeoutError(Exception):
            pass

        def timeout_handler(signum, frame):
            raise TimeoutError()

        comm.send((rank, 0, 0, False), dest=0, tag=TAG_REQ)
        last_hb = time.time()

        while True:
            tasks, _ = comm.recv(source=0, tag=TAG_ASSIGN)
            if not tasks:
                break

            batch_start = time.time()
            requeue_indices = []
            mem_flag = False

            for idx, lat, lon in tasks:
                # Soft Limit Check
                if args.memory_soft_limit_mb > 0:
                    rss = psutil.Process().memory_info().rss / (1024 * 1024)
                    if rss > args.memory_soft_limit_mb:
                        mem_flag = True
                        requeue_indices.append(idx)
                        continue

                if requeue_indices:
                    requeue_indices.append(idx)
                    continue

                # Process
                t0 = time.time()
                res = None

                signal.signal(signal.SIGALRM, timeout_handler)
                signal.setitimer(signal.ITIMER_REAL, args.download_timeout)

                try:
                    candidates = search_panoramas(lat=lat, lon=lon)
                    if candidates:
                        pid = engine._extract_panoid(candidates[0])
                        if pid:
                            raw = (
                                get_streetview(pano_id=pid, api_key=engine.api_key)
                                if engine.api_key
                                else engine._download_async_wrapper(pid)
                            )
                            if raw:
                                img = engine._preprocess_image(raw)
                                if img:
                                    mask = engine.segmenter.predict(img)
                                    met = engine.segmenter.calculate_gvi_from_mask(mask)
                                    res = {
                                        "gvi_veg": met.get("GVI_Total"),
                                        "gvi_ter": met.get("GVI_Terrain"),
                                        "pano_id": pid,
                                    }
                except TimeoutError:
                    requeue_indices.append(idx)
                except Exception:
                    pass
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)

                if idx not in requeue_indices:
                    comm.isend(
                        (rank, idx, res, time.time() - t0), dest=0, tag=TAG_UPDATE
                    )

                if time.time() - last_hb > args.heartbeat_interval:
                    comm.isend((rank, 0), dest=0, tag=TAG_HEARTBEAT)
                    last_hb = time.time()

                gc.collect()

            if requeue_indices:
                comm.send((rank, requeue_indices), dest=0, tag=TAG_REQUEUE)
                gc.collect()

            dur = time.time() - batch_start
            comm.send((rank, len(tasks), dur, mem_flag), dest=0, tag=TAG_REQ)


if __name__ == "__main__":
    main()
