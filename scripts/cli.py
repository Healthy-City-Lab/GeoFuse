#!/usr/bin/env python

import argparse
import gc
import hashlib
import os
import signal
import sys
import time
import warnings
from collections import deque
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import msgpack
import numpy as np
import pandas as pd
import psutil
from filelock import FileLock
from mpi4py import MPI

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from geofuse import streetview as gsv
from geofuse.core import generate_raster_grid
from geofuse.gvi import GVIEngine
from geofuse.ndvi import NDVIEngine
from geofuse.vector_io import read_vector_path
from geofuse.vision import get_best_device

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

CHECKPOINT_INTERVAL_S = 600  # 10 minutes
GC_INTERVAL = 50  # GC every N items per worker

# MPI Tags
TAG_REQ = 10
TAG_ASSIGN = 11
TAG_UPDATE = 88
TAG_HEARTBEAT = 55
TAG_REQUEUE = 66
TAG_DONE = 99

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def resolve_path(base_path, target_path):
    if os.path.exists(target_path):
        return target_path
    base_dir = os.path.dirname(os.path.abspath(base_path))
    rel_path = os.path.join(base_dir, target_path)
    if os.path.exists(rel_path):
        return rel_path
    return target_path


def load_config(csv_path):
    df = pd.read_csv(csv_path)
    required_cols = ["name", "geojson", "metric_type", "start_date", "end_date"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must contain column: {col}")
    df["metric_type"] = df["metric_type"].str.upper().str.strip()
    df["geojson"] = df["geojson"].apply(lambda x: resolve_path(csv_path, x))
    df["date_suffix"] = df.apply(
        lambda r: f"{str(r['start_date']).replace('-','').replace('/','')}"
        f"-{str(r['end_date']).replace('-','').replace('/','')}",
        axis=1,
    )
    return df


def _grid_cache_path(geojson_path: str, resolution_m: int) -> Path:
    """Return sidecar cache path next to the input geometry file."""
    p = Path(geojson_path).resolve()
    return p.parent / f"{p.stem}_gvi_grid_{resolution_m}m.gpkg"


def _geojson_hash(geojson_path: str, resolution_m: int) -> str:
    """Hash of file path + mtime + resolution to detect input changes."""
    stat = os.stat(geojson_path)
    raw = f"{geojson_path}|{stat.st_mtime}|{stat.st_size}|{resolution_m}"
    return hashlib.md5(raw.encode()).hexdigest()


def _load_or_generate_grid(geojson_path: str, resolution_m: int) -> gpd.GeoDataFrame:
    """Load grid from sidecar cache if valid, otherwise generate and cache it."""
    cache_path = _grid_cache_path(geojson_path, resolution_m)
    hash_attr_layer = "grid_hash"
    current_hash = _geojson_hash(geojson_path, resolution_m)

    if cache_path.exists():
        try:
            import fiona

            layers = fiona.listlayers(str(cache_path))
            if hash_attr_layer in layers:
                meta_gdf = gpd.read_file(str(cache_path), layer=hash_attr_layer)
                stored_hash = meta_gdf.iloc[0]["hash"] if len(meta_gdf) > 0 else None
                if stored_hash == current_hash:
                    print(f"  [Cache] Loading grid from {cache_path.name}")
                    gdf = gpd.read_file(str(cache_path), layer="points")
                    return gdf
                else:
                    print("  [Cache] Input changed — regenerating grid.")
        except Exception:
            print("  [Cache] Cache read failed — regenerating grid.")

    print(
        f"  [Cache] Generating grid for {Path(geojson_path).name} @ {resolution_m}m..."
    )
    gdf_aoi = read_vector_path(geojson_path)
    gdf_pts, _ = generate_raster_grid(gdf_aoi, resolution_m)

    if gdf_pts.empty:
        return gdf_pts

    # Geohash-sort for locality (reduces redundant panorama searches at cluster boundaries)
    try:
        import pygeohash as pgh

        gdf_pts["_gh"] = gdf_pts.geometry.apply(
            lambda g: pgh.encode(g.y, g.x, precision=6)
        )
        gdf_pts = gdf_pts.sort_values("_gh").drop(columns="_gh").reset_index(drop=True)
    except ImportError:
        pass

    # Write cache
    try:
        gdf_pts.to_file(str(cache_path), layer="points", driver="GPKG")
        hash_df = gpd.GeoDataFrame(
            {"hash": [current_hash]},
            geometry=gpd.points_from_xy([0], [0]),
            crs="EPSG:4326",
        )
        hash_df.to_file(str(cache_path), layer=hash_attr_layer, driver="GPKG")
        print(f"  [Cache] Saved to {cache_path.name}")
    except Exception as e:
        print(f"  [Cache] Warning: could not save cache: {e}")

    return gdf_pts


def generate_gvi_points(config_df, resolution_m):
    master_list = []
    gvi_tasks = config_df[config_df["metric_type"] == "GVI"]

    if gvi_tasks.empty:
        return gpd.GeoDataFrame()

    print(f"[Rank 0] Generating GVI sampling grid for {len(gvi_tasks)} tasks...")

    for _, row in gvi_tasks.iterrows():
        name = row["name"]
        suffix = row["date_suffix"]
        unique_name = f"{name}_{suffix}"
        geo_path = row["geojson"]

        if not os.path.exists(geo_path):
            print(f"[WARN] GeoJSON not found: {geo_path}. Skipping.")
            continue

        try:
            gdf_clipped = _load_or_generate_grid(geo_path, resolution_m)

            if gdf_clipped.empty:
                print(f"  > {unique_name}: 0 points — skipping.")
                continue

            gdf_clipped["task_id"] = unique_name
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
# Pano cache (msgpack) — coordinator-side, persisted separately from GDF
# ------------------------------------------------------------------------------


def _panocache_path(ckpt_path: str) -> str:
    stem = Path(ckpt_path).stem
    return str(Path(ckpt_path).parent / f"{stem}_panocache.msgpack")


def load_pano_cache(ckpt_path: str) -> dict:
    p = _panocache_path(ckpt_path)
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                return msgpack.unpackb(f.read(), raw=False)
        except Exception:
            pass
    return {}


def save_pano_cache(ckpt_path: str, cache: dict) -> None:
    p = _panocache_path(ckpt_path)
    try:
        with open(p + ".tmp", "wb") as f:
            f.write(msgpack.packb(cache, use_bin_type=True))
        os.replace(p + ".tmp", p)
    except Exception as e:
        print(f"[WARN] Could not save pano cache: {e}")


# ------------------------------------------------------------------------------
# GPU inference file-lock (coordinates concurrent segmentation across MPI ranks)
# ------------------------------------------------------------------------------


def _gpu_lockfile_path(output_dir: str, gpu_idx: int) -> str:
    return os.path.join(output_dir, f".gpu_{gpu_idx}.lock")


# ------------------------------------------------------------------------------
# MAIN CLI
# ------------------------------------------------------------------------------


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser(description="GeoFuse HPC CLI")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output_cli")
    parser.add_argument("--checkpoint", type=str, default="gvi_checkpoint.pkl")
    parser.add_argument("--ee_project", type=str, default=None)
    parser.add_argument(
        "--model_path", type=str, default="geofuse/model/best_model.pth"
    )
    parser.add_argument("--gvi_res", type=int, default=50)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_batch_size", type=int, default=20)
    parser.add_argument("--min_batch_size", type=int, default=2)
    parser.add_argument("--target_batch_seconds", type=int, default=120)
    parser.add_argument("--heartbeat_interval", type=int, default=30)
    parser.add_argument("--memory_soft_limit_mb", type=int, default=8000)
    parser.add_argument("--download_timeout", type=int, default=15)
    parser.add_argument(
        "--batch_timeout_factor",
        type=float,
        default=3.0,
        help="Requeue batch if elapsed > factor * expected duration.",
    )
    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # COORDINATOR (RANK 0)
    # --------------------------------------------------------------------------
    if rank == 0:
        print(
            f"--- GeoFuse CLI Started (Coordinator: Rank 0 | Workers: {size - 1}) ---"
        )
        os.makedirs(args.output_dir, exist_ok=True)

        # SIGTERM handler — flush checkpoint immediately on walltime kill
        _sigterm_received = [False]

        def _sigterm_handler(signum, frame):
            _sigterm_received[0] = True

        signal.signal(signal.SIGTERM, _sigterm_handler)

        try:
            config = load_config(args.config)
            print(f"[Init] Loaded config with {len(config)} rows.")
        except Exception as e:
            print(f"[Fatal] Config error: {e}")
            for w in range(1, size):
                comm.send(([], 0), dest=w, tag=TAG_ASSIGN)
            sys.exit(1)

        # NDVI Phase (sequential, rank 0 only)
        ndvi_tasks = config[config["metric_type"] == "NDVI"]
        if not ndvi_tasks.empty:
            print("\n=== PHASE 1: NDVI PROCESSING ===")
            try:
                ndvi_engine = NDVIEngine(project_id=args.ee_project)
                for _, row in ndvi_tasks.iterrows():
                    out_name = f"{row['name']}_{row['date_suffix']}"
                    print(f"[NDVI] Processing {out_name}...")
                    try:
                        gdf_aoi = read_vector_path(row["geojson"])
                        res = ndvi_engine.download_and_process(
                            geometry=gdf_aoi,
                            start_date=row["start_date"],
                            end_date=row["end_date"],
                            output_name=out_name,
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

        # GVI Prep Phase
        print("\n=== PHASE 2: GVI PREPARATION ===")
        ckpt_path = os.path.join(args.output_dir, args.checkpoint)

        if os.path.exists(ckpt_path):
            print(f"[Init] Resuming from checkpoint: {ckpt_path}")
            master_gdf = pd.read_pickle(ckpt_path)
        else:
            master_gdf = generate_gvi_points(config, args.gvi_res)
            if not master_gdf.empty:
                master_gdf.to_pickle(ckpt_path)

        if master_gdf.empty:
            print("[Info] No GVI tasks pending. Exiting.")
            for w in range(1, size):
                comm.send(([], 0), dest=w, tag=TAG_ASSIGN)
            return

        # Load coordinator pano cache
        pano_cache = load_pano_cache(ckpt_path)
        print(f"[Init] Pano cache loaded: {len(pano_cache)} entries.")

        unprocessed_mask = master_gdf["gvi_veg"].isna()
        task_indices = master_gdf.index[unprocessed_mask].tolist()
        task_queue = deque(task_indices)
        applied_indices = set(master_gdf.index[~unprocessed_mask].tolist())

        print(f"[Init] GVI Tasks: {len(task_queue)} pending / {len(master_gdf)} total")
        print("\n=== PHASE 3: DISTRIBUTED GVI PROCESSING ===")

        active_batches = {}
        worker_rates = {}
        worker_status = {
            w: {"processed": 0, "last_hb": time.time(), "status": "Idle"}
            for w in range(1, size)
        }
        processed_session = 0
        requeued_count = 0
        last_report = time.time()
        last_ckpt_time = time.time()
        start_time = time.time()

        def _do_checkpoint():
            master_gdf.to_pickle(ckpt_path)
            save_pano_cache(ckpt_path, pano_cache)
            _write_partial_geojsons()

        def _write_partial_geojsons():
            for task_id in master_gdf["task_id"].unique():
                subset = master_gdf[master_gdf["task_id"] == task_id].copy()
                done_subset = subset[subset["status"] == "done"]
                if done_subset.empty:
                    continue
                out_cols = [
                    c
                    for c in done_subset.columns
                    if c not in ["task_id", "status", "date_range"]
                ]
                out_name = f"{task_id}_gvi_partial.geojson"
                out_path = os.path.join(args.output_dir, out_name)
                done_subset[out_cols].to_file(out_path, driver="GeoJSON")

        # Coordinator loop
        while True:
            if _sigterm_received[0]:
                print("\n[SIGTERM] Saving checkpoint before exit...")
                _do_checkpoint()
                for w in range(1, size):
                    comm.send(([], 0), dest=w, tag=TAG_ASSIGN)
                sys.exit(0)

            if not task_queue and not active_batches:
                break

            now = time.time()

            # A. Requests
            while comm.iprobe(source=MPI.ANY_SOURCE, tag=TAG_REQ):
                req = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_REQ)
                w_rank, last_size, last_dur, mem_flag = req

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

                    active_batches[w_rank] = {
                        "indices": batch_idx,
                        "time": now,
                        "expected_s": b_size / max(rate, 0.01),
                    }
                    worker_status[w_rank]["status"] = f"Batch({len(batch_idx)})"
                    comm.send((batch_payload, b_size), dest=w_rank, tag=TAG_ASSIGN)

            # B. Updates
            while comm.iprobe(source=MPI.ANY_SOURCE, tag=TAG_UPDATE):
                msg = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_UPDATE)
                w_rank, idx, res_dict, _ = msg

                if idx in applied_indices:
                    continue
                applied_indices.add(idx)

                if res_dict:
                    master_gdf.at[idx, "gvi_veg"] = res_dict.get("gvi_veg")
                    master_gdf.at[idx, "gvi_ter"] = res_dict.get("gvi_ter")
                    master_gdf.at[idx, "pano_id"] = res_dict.get("pano_id")
                    master_gdf.at[idx, "status"] = "done"
                    pid = res_dict.get("pano_id")
                    if pid and pid not in pano_cache:
                        pano_cache[pid] = {
                            "gvi_veg": res_dict.get("gvi_veg"),
                            "gvi_ter": res_dict.get("gvi_ter"),
                        }
                else:
                    master_gdf.at[idx, "status"] = "failed"

                processed_session += 1
                worker_status[w_rank]["processed"] += 1

                if processed_session % 500 == 0:
                    _do_checkpoint()
                    last_ckpt_time = now

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

            # E. Dead worker / batch timeout detection
            for w_rank, batch in list(active_batches.items()):
                hb_age = now - worker_status[w_rank]["last_hb"]
                elapsed = now - batch["time"]
                expected = batch.get("expected_s", args.target_batch_seconds)
                timed_out = elapsed > args.batch_timeout_factor * max(expected, 30)
                dead = hb_age > args.heartbeat_interval * 4

                if timed_out or dead:
                    print(
                        f"[WARN] Worker {w_rank} {'dead' if dead else 'timed out'} "
                        f"— requeuing {len(batch['indices'])} items."
                    )
                    task_queue.extendleft(batch["indices"])
                    active_batches.pop(w_rank)
                    worker_status[w_rank]["status"] = "Requeued(timeout)"
                    requeued_count += len(batch["indices"])

            # F. Time-based checkpoint
            if now - last_ckpt_time > CHECKPOINT_INTERVAL_S:
                _do_checkpoint()
                last_ckpt_time = now

            # G. Progress report
            if now - last_report > 60:
                elapsed = now - start_time
                done_total = int((~master_gdf["gvi_veg"].isna()).sum())
                pct = (done_total / len(master_gdf) * 100) if len(master_gdf) > 0 else 0
                rate_global = processed_session / max(1, elapsed)
                rem_tasks = len(task_queue) + sum(
                    len(b["indices"]) for b in active_batches.values()
                )
                eta_min = (rem_tasks / rate_global / 60) if rate_global > 0 else 0

                print(
                    f"\n[Status] {datetime.now().strftime('%H:%M')} | "
                    f"Done: {done_total}/{len(master_gdf)} ({pct:.1f}%) | "
                    f"Speed: {rate_global:.1f} it/s | ETA: {eta_min:.1f} min | "
                    f"Requeued: {requeued_count}"
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

        # Drain remaining in-flight updates before final save
        drain_deadline = time.time() + 5.0
        while time.time() < drain_deadline:
            drained = False
            while comm.iprobe(source=MPI.ANY_SOURCE, tag=TAG_UPDATE):
                msg = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_UPDATE)
                w_rank, idx, res_dict, _ = msg
                if idx in applied_indices:
                    continue
                applied_indices.add(idx)
                if res_dict:
                    master_gdf.at[idx, "gvi_veg"] = res_dict.get("gvi_veg")
                    master_gdf.at[idx, "gvi_ter"] = res_dict.get("gvi_ter")
                    master_gdf.at[idx, "pano_id"] = res_dict.get("pano_id")
                    master_gdf.at[idx, "status"] = "done"
                    pid = res_dict.get("pano_id")
                    if pid and pid not in pano_cache:
                        pano_cache[pid] = {
                            "gvi_veg": res_dict.get("gvi_veg"),
                            "gvi_ter": res_dict.get("gvi_ter"),
                        }
                else:
                    master_gdf.at[idx, "status"] = "failed"
                drained = True
            if not drained:
                break
            time.sleep(0.01)

        # Final save
        _do_checkpoint()
        print("\n[Finalizing] Splitting results by task ID...")
        for task_id in master_gdf["task_id"].unique():
            subset = master_gdf[master_gdf["task_id"] == task_id].copy()
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

        print("[Done] All tasks completed.")
        for w in range(1, size):
            comm.send(([], 0), dest=w, tag=TAG_ASSIGN)

    # --------------------------------------------------------------------------
    # WORKER (RANK > 0)
    # --------------------------------------------------------------------------
    else:
        try:
            import torch

            n_gpus = torch.cuda.device_count()
            gpu_idx = (rank - 1) % max(n_gpus, 1) if n_gpus > 0 else 0
            device_str = args.device if args.device else str(get_best_device())
            engine = GVIEngine(
                model_path=args.model_path, device=device_str, api_key=args.api_key
            )
        except Exception as e:
            print(f"[Worker {rank}] GVIEngine init failed: {e}. Sending done signal.")
            comm.send((rank, 0, 0, False), dest=0, tag=TAG_REQ)
            tasks, _ = comm.recv(source=0, tag=TAG_ASSIGN)
            # Return empty batch immediately so coordinator requeues
            if tasks:
                comm.send((rank, [t[0] for t in tasks]), dest=0, tag=TAG_REQUEUE)
            sys.exit(1)

        os.makedirs(args.output_dir, exist_ok=True)
        lock_path = _gpu_lockfile_path(args.output_dir, gpu_idx)
        gpu_lock = FileLock(lock_path)

        class _TimeoutError(Exception):
            pass

        def _timeout_handler(signum, frame):
            raise _TimeoutError()

        comm.send((rank, 0, 0, False), dest=0, tag=TAG_REQ)
        last_hb = time.time()
        gc_counter = 0

        while True:
            tasks, _ = comm.recv(source=0, tag=TAG_ASSIGN)
            if not tasks:
                break

            batch_start = time.time()
            requeue_indices = []
            mem_flag = False

            for idx, lat, lon in tasks:
                if args.memory_soft_limit_mb > 0:
                    rss = psutil.Process().memory_info().rss / (1024 * 1024)
                    if rss > args.memory_soft_limit_mb:
                        mem_flag = True
                        requeue_indices.append(idx)
                        continue

                if requeue_indices:
                    requeue_indices.append(idx)
                    continue

                t0 = time.time()
                res = None

                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.setitimer(signal.ITIMER_REAL, args.download_timeout)

                try:
                    pano = gsv.find_panorama(lat, lon, radius=50)
                    if pano is not None:
                        raw = gsv.get_panorama(pano, zoom=1)
                        if raw is not None:
                            img = engine._preprocess_image(raw)
                            if img is not None:
                                # File-lock limits concurrent GPU inference to n_gpus workers
                                with gpu_lock:
                                    mask = engine.segmenter.predict(img)

                                met = engine.segmenter.calculate_gvi_from_mask(mask)
                                res = {
                                    "gvi_veg": met.get("GVI_Total"),
                                    "gvi_ter": met.get("GVI_Terrain"),
                                    "pano_id": pano.id,
                                }

                except _TimeoutError:
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

                gc_counter += 1
                if gc_counter % GC_INTERVAL == 0:
                    gc.collect()

            if requeue_indices:
                comm.send((rank, requeue_indices), dest=0, tag=TAG_REQUEUE)
                gc.collect()

            dur = time.time() - batch_start
            comm.send((rank, len(tasks), dur, mem_flag), dest=0, tag=TAG_REQ)


if __name__ == "__main__":
    main()
