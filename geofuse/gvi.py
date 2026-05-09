import asyncio
import os
import warnings
from collections.abc import Callable

import aiohttp
import geopandas as gpd
import numpy as np
from PIL import Image
from rasterio.transform import from_origin
from shapely.geometry import MultiPoint
from shapely.strtree import STRtree
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


class GVIEngine:
    def __init__(
        self, model_path=None, download_mode="package", device="cuda", api_key=None
    ):
        self.download_mode = download_mode
        self.device = get_best_device(device)
        self.api_key = api_key

        print(f"[GVI] Initializing DeepLabV3+ Model on {self.device}...")
        self.segmenter = DeepLabSegmenter(ckpt_path=model_path, device=str(self.device))
        print("[GVI] Model Ready.")

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

    def _generate_pixel_aligned_grid(self, gdf, resolution):
        """Generate sampling grid using vectorized STRtree point-in-polygon query."""
        if gdf.crs.is_geographic:
            centroid = gdf.geometry.iloc[0].centroid
            lon, lat = centroid.x, centroid.y
            utm_zone = int((lon + 180) / 6) + 1
            utm_crs = (
                f"EPSG:326{utm_zone:02d}" if lat >= 0 else f"EPSG:327{utm_zone:02d}"
            )
            print(f"[GVI] Converting from {gdf.crs} to {utm_crs} for grid generation")
            gdf_metric = gdf.to_crs(utm_crs)
        else:
            gdf_metric = gdf
            utm_crs = None

        minx, miny, maxx, maxy = gdf_metric.total_bounds
        width = int(np.ceil((maxx - minx) / resolution))
        height = int(np.ceil((maxy - miny) / resolution))

        print(f"[GVI] Grid dimensions: {width}x{height} = {width * height} points")

        max_points = 5_000_000
        if width * height > max_points:
            raise ValueError(
                f"Grid too large: {width}x{height} = {width * height:,} points "
                f"(max {max_points:,}). "
                f"For a ~{(maxx - minx) / 1000:.1f}x{(maxy - miny) / 1000:.1f} km area, "
                f"try step >= {int(np.sqrt((maxx - minx) * (maxy - miny) / max_points))} m."
            )

        cx = minx + (np.arange(width) + 0.5) * resolution
        cy = maxy - (np.arange(height) + 0.5) * resolution
        col_idx, row_idx = np.meshgrid(np.arange(width), np.arange(height))
        x_all = (minx + (col_idx + 0.5) * resolution).ravel()
        y_all = (maxy - (row_idx + 0.5) * resolution).ravel()
        rows_all = row_idx.ravel()
        cols_all = col_idx.ravel()

        union_geom = gdf_metric.geometry.union_all()
        candidate_points = MultiPoint(list(zip(x_all.tolist(), y_all.tolist())))
        tree = STRtree(list(candidate_points.geoms))
        inside_indices = tree.query(union_geom, predicate="within")

        if len(inside_indices) == 0:
            if gdf.crs.is_geographic:
                return (
                    [],
                    height,
                    width,
                    from_origin(minx, maxy, resolution, resolution),
                )
            return [], height, width, from_origin(minx, maxy, resolution, resolution)

        x_in = x_all[inside_indices]
        y_in = y_all[inside_indices]
        r_in = rows_all[inside_indices]
        c_in = cols_all[inside_indices]

        print(f"[GVI] Generated {len(x_in)} points within polygon")

        if gdf.crs.is_geographic:
            pts_gdf = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(x_in, y_in),
                crs=utm_crs,
            ).to_crs(gdf.crs)
            x_out = pts_gdf.geometry.x.to_numpy()
            y_out = pts_gdf.geometry.y.to_numpy()
        else:
            x_out, y_out = x_in, y_in

        transform = from_origin(minx, maxy, resolution, resolution)
        pixel_centers = [
            {
                "geometry": (
                    pts_gdf.geometry.iloc[i]
                    if gdf.crs.is_geographic
                    else gpd.points_from_xy([x_out[i]], [y_out[i]])[0]
                ),
                "row": int(r_in[i]),
                "col": int(c_in[i]),
                "lat": float(y_out[i]),
                "lon": float(x_out[i]),
            }
            for i in range(len(x_out))
        ]

        return pixel_centers, height, width, transform

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

        # 3. Download tiles (parallel via aiohttp inside get_panorama_async)
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
            _log("ERROR", f"  Download error for {short}…: " f"{type(e).__name__}: {e}")
            failed_panos.add(pid)
            return self._empty_result(pt, search_lat, search_lon)

        if cancel_callback and cancel_callback():
            return self._empty_result(pt, search_lat, search_lon)

        _log(
            "INFO",
            f"  Downloaded {short}… ({raw_image.width}×{raw_image.height}) "
            f"— preprocessing",
        )

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

        # 4. GPU inference (offloaded to executor so it doesn't block the loop)
        _log("INFO", f"  Running segmentation for {short}…")
        loop = asyncio.get_running_loop()
        try:
            mask = await loop.run_in_executor(None, self.segmenter.predict, img)
        except Exception as e:
            _log(
                "ERROR",
                f"  GPU inference failed for {short}…: " f"{type(e).__name__}: {e}",
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
        current_task: asyncio.Task | None = None

        async with aiohttp.ClientSession() as session:
            # Watcher: cancel the active point's task as soon as the user clicks Cancel.
            async def _cancel_watcher() -> None:
                while True:
                    await asyncio.sleep(0.25)
                    if cancel_callback and cancel_callback():
                        if current_task and not current_task.done():
                            current_task.cancel()
                        return

            watcher = asyncio.create_task(_cancel_watcher())

            # Serial loop — one point at a time, search → download → GPU → next.
            for pt in points_to_process:
                if cancel_callback and cancel_callback():
                    break

                current_task = asyncio.create_task(
                    self._process_one_point_async(
                        pt,
                        session,
                        gdf,
                        folder,
                        save_panos,
                        save_masks,
                        pano_cache,
                        failed_panos,
                        cancel_callback,
                    )
                )
                try:
                    res = await current_task
                except asyncio.CancelledError:
                    break

                if result_callback:
                    result_callback(res)
                completed += 1
                if progress_callback:
                    progress_callback(start_index + completed, total_points)

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
        print(f"[GVI] Starting Analysis (Resume Index: {start_index})...")
        os.makedirs(folder, exist_ok=True)
        if save_panos:
            os.makedirs(os.path.join(folder, "images"), exist_ok=True)
        if save_masks:
            os.makedirs(os.path.join(folder, "masks"), exist_ok=True)

        first_geom = gdf.geometry.iloc[0]
        points = []

        if first_geom.geom_type in ["Polygon", "MultiPolygon"]:
            points_data, _, _, _ = self._generate_pixel_aligned_grid(gdf, step)
            for i, p in enumerate(points_data):
                p["orig_index"] = i
            points = points_data
        else:
            has_indices = "row" in gdf.columns and "col" in gdf.columns
            for idx, row in gdf.iterrows():
                points.append(
                    {
                        "orig_index": idx,
                        "geometry": row.geometry,
                        "row": int(row["row"]) if has_indices else 0,
                        "col": int(row["col"]) if has_indices else idx,
                        "lat": row.geometry.y,
                        "lon": row.geometry.x,
                    }
                )
            if has_indices:
                rows = gdf["row"].max() + 1
                cols = gdf["col"].max() + 1

        if not points:
            print("[FAIL] No points to process.")
            return gpd.GeoDataFrame()

        total_points = len(points)
        points_to_process = points[start_index:]

        if len(points_to_process) == 0:
            print("[GVI] All points already processed.")
            return gpd.GeoDataFrame()

        pano_cache = external_cache if external_cache is not None else {}

        _progress_cb = progress_callback
        _close_pbar = False
        if _progress_cb is None:
            _pbar = tqdm(total=len(points_to_process), desc="GVI Analysis")

            def _progress_cb(curr, total, _p=_pbar):
                _p.update(1)

            _close_pbar = True
        else:
            _progress_cb(start_index, total_points)

        print(f"[GVI] Processing {len(points_to_process)} points...")

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
