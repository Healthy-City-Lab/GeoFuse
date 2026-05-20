import asyncio
import os
import warnings
from collections.abc import Callable

import aiohttp
import geopandas as gpd
import numpy as np
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

# Cityscapes Palette
CITYSCAPES_PALETTE = [
    128,
    64,
    128,
    244,
    35,
    232,
    70,
    70,
    70,
    102,
    102,
    156,
    190,
    153,
    153,
    153,
    153,
    153,
    250,
    170,
    30,
    220,
    220,
    0,
    107,
    142,
    35,
    152,
    251,
    152,
    70,
    130,
    180,
    220,
    20,
    60,
    255,
    0,
    0,
    0,
    0,
    142,
    0,
    0,
    70,
    0,
    60,
    100,
    0,
    80,
    100,
    0,
    0,
    230,
    119,
    11,
    32,
] + [0, 0, 0] * 237

from . import streetview as gsv
from .logger import get_logger
from .vision import DeepLabSegmenter, get_best_device

_log = get_logger("GVI")

# Search radius (metres) for find_panorama_async around each grid point.
_SEARCH_RADIUS_M = 50.0
# Zoom level for tile downloads: 0 = lowest, 5 = highest. zoom=1 is a 2×1
# tile grid (~1664×832 px) — plenty of resolution for segmentation, fast to fetch.
_DOWNLOAD_ZOOM = 1
# Hard timeout per panorama download (covers all tile fetches together).
_DOWNLOAD_TIMEOUT_S = 20.0
# Sliding window of points that may have their **panorama-search HTTP call**
# in flight at the same time. Small enough not to rate-limit Google's
# SingleImageSearch endpoint or oversubscribe local sockets; large enough
# that one task's TCP/TLS reconnect cost is hidden by others' in-flight
# requests. The heavy download+GPU stretch still serializes — see the
# inner lock built in _run_analysis_async — so cache-miss throughput is
# unchanged while cache-hit / no-pano points fly through concurrently.
_MAX_CONCURRENT_POINTS = 4
# Periodic ``torch.cuda.empty_cache()`` cadence (per-worker completions).
# Defensive against PyTorch allocator fragmentation over million-point runs.
_EMPTY_CACHE_EVERY_N = 200


class GVIEngine:
    def __init__(
        self, model_path=None, download_mode="package", device="cuda", api_key=None
    ):
        self.download_mode = download_mode
        self.device = get_best_device(device)
        self.api_key = api_key

        _log("INFO", f"Initializing DeepLabV3+ Model on {self.device}...")
        self.segmenter = DeepLabSegmenter(ckpt_path=model_path, device=str(self.device))
        _log("OK", "Model Ready.")

    def _preprocess_image(self, img, target_width=1920):
        if img is None:
            return None

        if isinstance(img, np.ndarray):
            try:
                img = Image.fromarray(img)
            except Exception:
                return None

        arr = np.array(img)
        row_energy = np.mean(arr, axis=(1, 2))
        col_energy = np.mean(arr, axis=(0, 2))

        valid_rows = np.where(row_energy > 5)[0]
        valid_cols = np.where(col_energy > 5)[0]

        if len(valid_rows) == 0 or len(valid_cols) == 0:
            return None

        y_min, y_max = valid_rows[0], valid_rows[-1] + 1
        x_min, x_max = valid_cols[0], valid_cols[-1] + 1
        height = y_max - y_min
        expected_width = height * 2
        current_width = x_max - x_min

        if current_width > expected_width:
            center_x = (x_min + x_max) // 2
            half_width = expected_width // 2
            x_min = max(0, center_x - half_width)
            x_max = x_min + expected_width

        img_cropped = img.crop((x_min, y_min, x_max, y_max))
        target_height = int(target_width / 2)
        return img_cropped.resize((target_width, target_height), Image.BILINEAR)

    async def _process_one_point_async(
        self,
        pt: dict,
        session: aiohttp.ClientSession,
        gdf: gpd.GeoDataFrame,
        folder: str,
        save_panos: bool,
        save_masks: bool,
        pano_cache: dict,
        failed_panos: set,
        cancel_callback: Callable[..., bool] | None,
        gpu_lock: asyncio.Lock,
    ) -> dict:
        idx = pt["orig_index"]
        lat, lon = pt["lat"], pt["lon"]
        search_lat, search_lon = lat, lon

        if not gdf.crs.is_geographic:
            try:
                p_geo = (
                    gpd.GeoSeries([pt["geometry"]], crs=gdf.crs)
                    .to_crs(epsg=4326)
                    .iloc[0]
                )
                search_lat, search_lon = p_geo.y, p_geo.x
            except Exception as e:
                _log("ERROR", f"Point {idx}: CRS transform failed — {e}")
                return self._empty_result(pt, lat, lon)

        _log("INFO", f"Point {idx} @ ({search_lat:.5f}, {search_lon:.5f})")

        # 1. Search for the nearest panorama
        try:
            pano = await gsv.find_panorama_async(
                search_lat,
                search_lon,
                session,
                radius=_SEARCH_RADIUS_M,
            )
        except Exception as e:
            _log(
                "ERROR",
                f"  Point {idx}: find_panorama_async raised — "
                f"{type(e).__name__}: {e}",
            )
            return self._empty_result(pt, search_lat, search_lon)

        if pano is None:
            _log("WARN", f"  Point {idx}: no panoramas within {_SEARCH_RADIUS_M:.0f} m")
            return self._empty_result(pt, search_lat, search_lon)

        pid = pano.id
        short = pid[:12]

        # 2. Cache lookups (run-local fail set + persistent success cache)
        if pid in failed_panos:
            _log("WARN", f"  Skipping {short}… (failed earlier this run)")
            return self._empty_result(pt, search_lat, search_lon)

        if pid in pano_cache:
            cached = pano_cache[pid]
            _log(
                "OK",
                f"  Cache hit {short}… → veg={cached['veg']:.3f} "
                f"ter={cached['ter']:.3f}",
            )
            return self._make_result(
                pt, search_lat, search_lon, cached["veg"], cached["ter"], pid
            )

        # 3. Download tiles — no lock. All workers can fetch tiles in parallel
        # (separate aiohttp keep-alive connections share the session pool).
        _log("INFO", f"  Downloading {short}… (zoom={_DOWNLOAD_ZOOM})")
        try:
            raw_image = await asyncio.wait_for(
                gsv.get_panorama_async(pano, session, zoom=_DOWNLOAD_ZOOM),
                timeout=_DOWNLOAD_TIMEOUT_S,
            )
        except TimeoutError:
            _log(
                "ERROR",
                f"  Timeout (>{_DOWNLOAD_TIMEOUT_S:.0f}s) downloading "
                f"{short}… — marking failed",
            )
            failed_panos.add(pid)
            return self._empty_result(pt, search_lat, search_lon)
        except Exception as e:
            _log(
                "ERROR",
                f"  Download error for {short}…: " f"{type(e).__name__}: {e}",
            )
            failed_panos.add(pid)
            return self._empty_result(pt, search_lat, search_lon)

        if cancel_callback and cancel_callback():
            return self._empty_result(pt, search_lat, search_lon)

        _log(
            "INFO",
            f"  Downloaded {short}… ({raw_image.width}×{raw_image.height}) "
            f"— preprocessing",
        )

        # Preprocess on the worker thread — no lock (CPU only). Multiple
        # workers can preprocess concurrently while another worker holds the
        # GPU lock for its forward pass.
        img = self._preprocess_image(raw_image)
        try:
            raw_image.close()
        except Exception:
            pass
        del raw_image

        if not img:
            _log("WARN", f"  Preprocess failed for {short}…: blank or black image")
            failed_panos.add(pid)
            return self._empty_result(pt, search_lat, search_lon)

        # Build the input tensor (CPU + pinned memory) outside the GPU lock so
        # the next forward can overlap with H2D transfer.
        loop = asyncio.get_running_loop()
        try:
            tensor = await loop.run_in_executor(
                None, self.segmenter.preprocess_to_tensor, img
            )
        except Exception as e:
            _log(
                "ERROR",
                f"  Tensor build failed for {short}…: "
                f"{type(e).__name__}: {e}",
            )
            return self._empty_result(pt, search_lat, search_lon)

        # 4. GPU inference — serialised so one image is on the device at a
        # time. While this lock is held by one task, others can download
        # tiles or preprocess in parallel.
        async with gpu_lock:
            # Another worker may have written the cache for this pano while
            # we were queued at the GPU lock; skip the forward in that case.
            if pid in pano_cache:
                cached = pano_cache[pid]
                _log(
                    "OK",
                    f"  Cache hit (after queue) {short}… → "
                    f"veg={cached['veg']:.3f} ter={cached['ter']:.3f}",
                )
                return self._make_result(
                    pt, search_lat, search_lon, cached["veg"], cached["ter"], pid
                )

            _log("INFO", f"  Running segmentation for {short}…")
            try:
                mask = await loop.run_in_executor(
                    None, self.segmenter.predict_from_tensor, tensor
                )
            except Exception as e:
                _log(
                    "ERROR",
                    f"  GPU inference failed for {short}…: "
                    f"{type(e).__name__}: {e}",
                )
                return self._empty_result(pt, search_lat, search_lon)

            metrics = self.segmenter.calculate_gvi_from_mask(mask)
            val_veg = metrics.get("GVI_Total", 0.0)
            val_ter = metrics.get("GVI_Terrain", 0.0)
            pano_cache[pid] = {"veg": val_veg, "ter": val_ter}

            _log("OK", f"  GVI {short}… → veg={val_veg:.3f}  ter={val_ter:.3f}")

        # 5. Optional disk artefacts
        if save_panos:
            p_path = os.path.join(folder, "images", f"{pid}.jpg")
            if not os.path.exists(p_path):
                try:
                    img_to_save = (
                        Image.fromarray(img) if isinstance(img, np.ndarray) else img
                    )
                    img_to_save.save(p_path)
                    _log("OK", f"  Saved panorama → {p_path}")
                except Exception as e:
                    _log(
                        "ERROR",
                        f"  Failed to save panorama {short}…: "
                        f"{type(e).__name__}: {e}",
                    )

        if save_masks:
            m_path = os.path.join(folder, "masks", f"{pid}.png")
            if not os.path.exists(m_path):
                try:
                    m = mask
                    if hasattr(m, "cpu"):  # torch.Tensor
                        m = m.cpu().numpy()
                    if isinstance(m, np.ndarray):
                        mask_img = Image.fromarray(m.astype(np.uint8))
                        mask_img.putpalette(CITYSCAPES_PALETTE)
                        mask_img.save(m_path)
                        _log("OK", f"  Saved mask     → {m_path}")
                    elif isinstance(m, Image.Image):
                        m.putpalette(CITYSCAPES_PALETTE)
                        m.save(m_path)
                        _log("OK", f"  Saved mask     → {m_path}")
                    else:
                        _log(
                            "WARN",
                            f"  Mask type '{type(mask).__name__}' "
                            f"not handled — skipping save",
                        )
                except Exception as e:
                    _log(
                        "ERROR",
                        f"  Failed to save mask {short}…: " f"{type(e).__name__}: {e}",
                    )

        return self._make_result(pt, search_lat, search_lon, val_veg, val_ter, pid)

    @staticmethod
    def _empty_result(pt: dict, search_lat: float, search_lon: float) -> dict:
        return {
            "orig_index": pt["orig_index"],
            "geometry": pt["geometry"],
            "gvi_veg": None,
            "gvi_ter": None,
            "pano_id": None,
            "lat": search_lat,
            "lon": search_lon,
            "row": pt["row"],
            "col": pt["col"],
            "cluster_id": pt.get("cluster_id", 0),
        }

    @staticmethod
    def _make_result(
        pt: dict, search_lat: float, search_lon: float, val_veg, val_ter, pid: str
    ) -> dict:
        return {
            "orig_index": pt["orig_index"],
            "geometry": pt["geometry"],
            "gvi_veg": float(val_veg) if val_veg is not None else None,
            "gvi_ter": float(val_ter) if val_ter is not None else None,
            "pano_id": pid,
            "lat": search_lat,
            "lon": search_lon,
            "row": pt["row"],
            "col": pt["col"],
            "cluster_id": pt.get("cluster_id", 0),
        }

    async def _run_analysis_async(
        self,
        points_to_process: list,
        total_points: int,
        start_index: int,
        gdf: gpd.GeoDataFrame,
        folder: str,
        save_panos: bool,
        save_masks: bool,
        pano_cache: dict,
        progress_callback: Callable | None,
        result_callback: Callable | None,
        cancel_callback: Callable | None,
    ) -> None:
        # Run-local set: tracks panos that already failed in *this* run.
        # Never written to pano_cache (which is the persistent session cache).
        failed_panos: set[str] = set()
        completed = 0

        # Fixed worker pool: exactly _MAX_CONCURRENT_POINTS workers pull
        # points from a shared queue. ``gpu_lock`` serializes only the GPU
        # forward pass; downloads and CPU preprocessing run in parallel.
        gpu_lock = asyncio.Lock()
        queue: asyncio.Queue = asyncio.Queue()
        for pt in points_to_process:
            queue.put_nowait(pt)

        completed_lock = asyncio.Lock()

        async with aiohttp.ClientSession() as session:

            async def _worker() -> None:
                nonlocal completed
                while True:
                    if cancel_callback and cancel_callback():
                        return
                    try:
                        pt = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    try:
                        res = await self._process_one_point_async(
                            pt,
                            session,
                            gdf,
                            folder,
                            save_panos,
                            save_masks,
                            pano_cache,
                            failed_panos,
                            cancel_callback,
                            gpu_lock,
                        )
                    except asyncio.CancelledError:
                        return
                    if result_callback:
                        result_callback(res)
                    async with completed_lock:
                        completed += 1
                        curr = completed
                    if progress_callback:
                        progress_callback(start_index + curr, total_points)
                    # Defensive: release cached blocks back to the device
                    # periodically so long-running jobs don't accumulate
                    # allocator fragmentation.
                    if (
                        curr % _EMPTY_CACHE_EVERY_N == 0
                        and self.device.type == "cuda"
                    ):
                        try:
                            import torch

                            torch.cuda.empty_cache()
                        except Exception:
                            pass

            workers: list[asyncio.Task] = [
                asyncio.create_task(_worker()) for _ in range(_MAX_CONCURRENT_POINTS)
            ]

            # Watcher: cancels every worker on user cancel.
            async def _cancel_watcher() -> None:
                while True:
                    await asyncio.sleep(0.25)
                    if cancel_callback and cancel_callback():
                        for t in workers:
                            if not t.done():
                                t.cancel()
                        return

            watcher = asyncio.create_task(_cancel_watcher())

            try:
                await asyncio.gather(*workers, return_exceptions=True)
            finally:
                watcher.cancel()
                try:
                    await watcher
                except asyncio.CancelledError:
                    pass

            if cancel_callback and cancel_callback():
                _log("WARN", "Analysis Aborted by User.")

    def run_analysis(
        self,
        gdf,
        step: float | int = 75,
        folder="output",
        save_panos=False,
        save_masks=False,
        external_cache=None,
        progress_callback=None,
        result_callback=None,
        cancel_callback=None,
        start_index=0,
    ):
        _log("INFO", f"Starting Analysis (Resume Index: {start_index})...")
        os.makedirs(folder, exist_ok=True)
        if save_panos:
            os.makedirs(os.path.join(folder, "images"), exist_ok=True)
        if save_masks:
            os.makedirs(os.path.join(folder, "masks"), exist_ok=True)

        # Polygon fallback: caller passed raw polygons instead of pre-generating
        # the sampling grid. The UI always pre-generates via generate_clustered_grid,
        # so this branch is only hit by direct API callers.
        first_geom = gdf.geometry.iloc[0]
        if first_geom.geom_type in ["Polygon", "MultiPolygon"]:
            from .core import generate_clustered_grid

            gdf_4326 = (
                gdf
                if gdf.crs is not None and gdf.crs.is_geographic
                else gdf.to_crs("EPSG:4326")
            )
            pts_gdf, _meta = generate_clustered_grid(
                gdf_4326, buffer_m=0, step_m=float(step)
            )
            gdf = pts_gdf

        points = []
        has_indices = "row" in gdf.columns and "col" in gdf.columns
        has_cluster = "cluster_id" in gdf.columns
        for idx, row in gdf.iterrows():
            points.append(
                {
                    "orig_index": idx,
                    "geometry": row.geometry,
                    "row": int(row["row"]) if has_indices else 0,
                    "col": int(row["col"]) if has_indices else int(idx),  # type: ignore[arg-type]
                    "cluster_id": int(row["cluster_id"]) if has_cluster else 0,
                    "lat": row.geometry.y,
                    "lon": row.geometry.x,
                }
            )

        if not points:
            _log("ERROR", "No points to process.")
            return gpd.GeoDataFrame()

        total_points = len(points)
        points_to_process = points[start_index:]

        if len(points_to_process) == 0:
            _log("INFO", "All points already processed.")
            return gpd.GeoDataFrame()

        pano_cache = external_cache if external_cache is not None else {}
        # Bulk-load the persistent cache into its in-memory overlay so every
        # per-point lookup is a dict hit instead of a SQLite round-trip.
        if hasattr(pano_cache, "preload"):
            try:
                n_cached = pano_cache.preload()
                _log("INFO", f"Pano cache preloaded: {n_cached:,} entries in-memory.")
            except Exception as e:
                _log(
                    "WARN",
                    f"Pano cache preload failed: {type(e).__name__}: {e}",
                )

        _progress_cb = progress_callback
        _close_pbar = False
        if _progress_cb is None:
            _pbar = tqdm(total=len(points_to_process), desc="GVI Analysis")

            def _progress_cb(curr, total, _p=_pbar):
                _p.update(1)

            _close_pbar = True
        else:
            _progress_cb(start_index, total_points)

        _log("INFO", f"Processing {len(points_to_process)} points...")

        asyncio.run(
            self._run_analysis_async(
                points_to_process=points_to_process,
                total_points=total_points,
                start_index=start_index,
                gdf=gdf,
                folder=folder,
                save_panos=save_panos,
                save_masks=save_masks,
                pano_cache=pano_cache,
                progress_callback=_progress_cb,
                result_callback=result_callback,
                cancel_callback=cancel_callback,
            )
        )

        if _close_pbar:
            _pbar.close()

        return gpd.GeoDataFrame()
