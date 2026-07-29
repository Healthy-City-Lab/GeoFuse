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


def _format_pano_date(date: tuple[int, int] | None) -> str | None:
    """Render a ``(year, month)`` capture date as ``"YYYY-MM"``, or ``None``."""
    try:
        year, month = date
        return f"{int(year):04d}-{int(month):02d}"
    except (TypeError, ValueError):
        return None

# Search radius (metres) for find_panorama_async around each grid point.
_SEARCH_RADIUS_M = 50.0
# Zoom level for tile downloads: 0 = lowest, 5 = highest. zoom=1 is a 2×1
# tile grid — plenty of resolution for segmentation, fast to fetch.
_DOWNLOAD_ZOOM = 1
# Width the panorama is normalised to before segmentation. Kept equal to the
# native width of a zoom=1 download (1024×512) so the image is never upscaled:
# upscaling to 1920 cost ~3.5x the GPU compute (measured 8.1 -> 35.2 img/s)
# without adding any information the download didn't already contain. Height is
# always width/2 (equirectangular), so every model input has the same shape.
_SEGMENT_WIDTH = 1024
# Hard timeout per panorama download (covers all tile fetches together).
_DOWNLOAD_TIMEOUT_S = 20.0
# The analysis runs as two stages with independent concurrency (see
# ``_run_analysis_async``):
#   * search stage — resolves every point's nearest panorama. These calls are
#     light and highly parallel, so a wide window keeps misses and cache hits
#     (which never touch the GPU) flowing without waiting behind downloads.
#   * segment stage — downloads a panorama, preprocesses it and runs the GPU
#     forward. The GPU is serialised by a lock, so a small pool is enough;
#     more would only queue at the lock.
# Both stages share the adaptive throttle below, which scales the *effective*
# request rate down automatically if Google starts pushing back, so these are
# ceilings rather than fixed rates.
_SEARCH_CONCURRENCY = 48
_MAX_CONCURRENT_POINTS = 6
# Periodic ``torch.cuda.empty_cache()`` cadence (per-worker completions).
# Defensive against PyTorch allocator fragmentation over million-point runs.
_EMPTY_CACHE_EVERY_N = 200
# Subsample step for the black-border scan in _preprocess_image. Only used to
# find the padding edges, never to sample colour, so a coarse step is safe.
_SCAN_STEP = 4

# ── Adaptive rate-limit handling ────────────────────────────────────
# Google's endpoints are unofficial and throttle by IP with no published quota,
# so the safe design is to *detect* push-back and back off rather than guess a
# fixed rate. Backoff grows exponentially per consecutive throttle event and
# decays again after a clean streak.
_RL_BACKOFF_START_S = 1.0
_RL_BACKOFF_MAX_S = 60.0
_RL_DELAY_MAX_S = 5.0
_RL_RECOVER_AFTER_OK = 40
# How many times a single point retries through throttling before giving up.
# Without this a throttled point is silently dropped as "no panorama".
_RL_MAX_RETRIES = 4


class AdaptiveThrottle:
    """Detects Google rate-limiting and automatically slows the whole run down.

    Two mechanisms, both shared by every worker in a run:

    * **Gate** — on a throttle event every worker is held for an exponentially
      growing backoff, so we stop hammering an endpoint that is already
      pushing back.
    * **Spacing** — a per-request delay that ramps up with repeated throttling
      and decays after a clean streak. This lowers the sustained request rate
      without resizing the worker pool, so no queued point is lost.

    The engine treats a throttle as *retryable*, not as "no coverage" — which
    matters because the search endpoint answers a 429 with a body that would
    otherwise parse as an empty result and silently drop the point.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._delay = 0.0
        self._backoff = _RL_BACKOFF_START_S
        self._gate_until = 0.0
        self._ok_streak = 0
        self.trips = 0

    async def before_request(self) -> None:
        """Wait out any active backoff, then apply the current spacing."""
        while True:
            async with self._lock:
                wait = self._gate_until - asyncio.get_running_loop().time()
                delay = self._delay
            if wait > 0:
                await asyncio.sleep(min(wait, 1.0))
                continue
            if delay:
                await asyncio.sleep(delay)
            return

    async def record_throttle(self, status: int | None = None) -> None:
        """A request was rate-limited: close the gate and widen the spacing."""
        async with self._lock:
            self.trips += 1
            self._ok_streak = 0
            now = asyncio.get_running_loop().time()
            # Only escalate if we're not already inside a backoff window, so a
            # burst of concurrent 429s counts as one event rather than six.
            if now >= self._gate_until:
                self._gate_until = now + self._backoff
                self._delay = min(max(self._delay * 2.0, 0.25), _RL_DELAY_MAX_S)
                _log(
                    "WARN",
                    f"Rate limited (HTTP {status}) — backing off "
                    f"{self._backoff:.0f}s, spacing requests {self._delay:.2f}s "
                    f"(event #{self.trips})",
                )
                self._backoff = min(self._backoff * 2.0, _RL_BACKOFF_MAX_S)

    async def record_success(self) -> None:
        """A clean response: decay the spacing after a sustained good streak."""
        async with self._lock:
            self._ok_streak += 1
            if self._ok_streak >= _RL_RECOVER_AFTER_OK and self._delay > 0:
                self._ok_streak = 0
                self._delay = max(self._delay / 2.0, 0.0)
                if self._delay < 0.05:
                    self._delay = 0.0
                self._backoff = max(self._backoff / 2.0, _RL_BACKOFF_START_S)
                _log(
                    "INFO",
                    f"Rate limit easing — spacing now {self._delay:.2f}s",
                )


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

    def _preprocess_image(self, img, target_width=_SEGMENT_WIDTH):
        if img is None:
            return None

        if isinstance(img, np.ndarray):
            try:
                img = Image.fromarray(img)
            except Exception:
                return None

        # The scan only locates the black padding around the equirectangular
        # frame, so it runs on a 1/4 subsample in float32 rather than the full
        # image in float64. Reducing over axis (0, 2) of a C-contiguous
        # (H, W, C) array is stride-hostile and dominated this function
        # (3.75 ms of 6.28 ms); subsampling makes the scan ~6x cheaper and
        # leaves the crop — and therefore every GVI value — bit-identical,
        # because the padding is far wider than the sample step.
        arr = np.asarray(img)
        sub = arr[::_SCAN_STEP, ::_SCAN_STEP]
        row_energy = sub.mean(axis=(1, 2), dtype=np.float32)
        col_energy = sub.mean(axis=(0, 2), dtype=np.float32)

        valid_rows = np.where(row_energy > 5)[0]
        valid_cols = np.where(col_energy > 5)[0]

        if len(valid_rows) == 0 or len(valid_cols) == 0:
            return None

        # Map subsampled indices back to full-resolution pixel bounds.
        y_min = int(valid_rows[0]) * _SCAN_STEP
        y_max = min((int(valid_rows[-1]) + 1) * _SCAN_STEP, arr.shape[0])
        x_min = int(valid_cols[0]) * _SCAN_STEP
        x_max = min((int(valid_cols[-1]) + 1) * _SCAN_STEP, arr.shape[1])
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

    async def _search_and_resolve(
        self,
        pt: dict,
        session: aiohttp.ClientSession,
        gdf: gpd.GeoDataFrame,
        failed_panos: set,
        pano_cache: dict,
        cancel_callback: Callable[..., bool] | None,
        target_year: int | None,
        max_year_diff: int | None,
        throttle: "AdaptiveThrottle | None" = None,
    ) -> tuple:
        """Stage 1: resolve a point to a panorama (search + cache lookup).

        Returns one of:

        * ``("done", result_dict)`` -- the point is fully resolved here: a CRS
          failure, a miss (no coverage), a capture-year miss, a pano that
          already failed this run, or a cache hit.
        * ``("segment", download_pano, pid, pano_date_str, search_lat,
          search_lon)`` -- the panorama exists and is not cached, so it needs
          downloading and segmentation in stage 2.

        Every request goes through the shared :class:`AdaptiveThrottle`, and a
        rate-limited search is retried rather than mistaken for "no coverage".
        """
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
                return ("done", self._empty_result(pt, lat, lon))

        _log("INFO", f"Point {idx} @ ({search_lat:.5f}, {search_lon:.5f})")

        # Search for the nearest panorama. Throttling is retried (with the
        # shared backoff) rather than treated as "no coverage" -- otherwise
        # rate limiting silently turns into missing sample points.
        pano = _RL_SENTINEL = object()
        try:
            for attempt in range(_RL_MAX_RETRIES):
                if throttle is not None:
                    await throttle.before_request()
                try:
                    pano = await gsv.find_panorama_async(
                        search_lat,
                        search_lon,
                        session,
                        radius=_SEARCH_RADIUS_M,
                    )
                    if throttle is not None:
                        await throttle.record_success()
                    break
                except gsv.RateLimitedError as rl:
                    if throttle is not None:
                        await throttle.record_throttle(rl.status)
                    if cancel_callback and cancel_callback():
                        return ("done", self._empty_result(pt, search_lat, search_lon))
                    if attempt == _RL_MAX_RETRIES - 1:
                        _log(
                            "ERROR",
                            f"  Point {idx}: still rate limited after "
                            f"{_RL_MAX_RETRIES} attempts — leaving unsampled",
                        )
                        return (
                            "done",
                            self._empty_result(pt, search_lat, search_lon),
                        )
            if pano is _RL_SENTINEL:
                return ("done", self._empty_result(pt, search_lat, search_lon))
        except Exception as e:
            _log(
                "ERROR",
                f"  Point {idx}: find_panorama_async raised — "
                f"{type(e).__name__}: {e}",
            )
            return ("done", self._empty_result(pt, search_lat, search_lon))

        if pano is None:
            _log(
                "WARN",
                f"  Point {idx}: no panoramas within {_SEARCH_RADIUS_M:.0f} m",
            )
            return ("done", self._empty_result(pt, search_lat, search_lon))

        # Pick which dated capture to use. Without a target year we keep the
        # panorama the search returned (the most recent coverage).
        download_pano = pano
        pano_date = getattr(pano, "date", None)
        if target_year is not None:
            capture = pano.select_capture(target_year, max_year_diff)
            if capture is None:
                _log(
                    "WARN",
                    f"  Point {idx}: no capture within "
                    f"{max_year_diff} yr of {target_year} "
                    f"(available: {[c.year for c in pano.captures]})",
                )
                return ("done", self._empty_result(pt, search_lat, search_lon))
            if capture.id != pano.id:
                download_pano = pano.clone_for_capture(capture)
            pano_date = (
                (capture.year, capture.month)
                if capture.year is not None and capture.month is not None
                else None
            )

        pid = download_pano.id
        pano_date_str = _format_pano_date(pano_date)
        short = pid[:12]

        # Cache lookups (run-local fail set + persistent success cache)
        if pid in failed_panos:
            _log("WARN", f"  Skipping {short}… (failed earlier this run)")
            return ("done", self._empty_result(pt, search_lat, search_lon))

        if pid in pano_cache:
            cached = pano_cache[pid]
            _log(
                "OK",
                f"  Cache hit {short}… → veg={cached['veg']:.3f} "
                f"ter={cached['ter']:.3f}",
            )
            return (
                "done",
                self._make_result(
                    pt,
                    search_lat,
                    search_lon,
                    cached["veg"],
                    cached["ter"],
                    pid,
                    pano_date_str,
                ),
            )

        return ("segment", download_pano, pid, pano_date_str, search_lat, search_lon)

    async def _segment_resolved(
        self,
        pt: dict,
        download_pano,
        pid: str,
        pano_date_str: str | None,
        search_lat: float,
        search_lon: float,
        session: aiohttp.ClientSession,
        folder: str,
        save_panos: bool,
        save_masks: bool,
        pano_cache: dict,
        failed_panos: set,
        cancel_callback: Callable[..., bool] | None,
        gpu_lock: asyncio.Lock,
        throttle: "AdaptiveThrottle | None" = None,
    ) -> dict:
        """Stage 2: download the resolved panorama, segment it, cache the value.

        Only called for panoramas that stage 1 found present and uncached. The
        GPU forward is serialised by ``gpu_lock``; downloads and CPU
        preprocessing run in parallel across the segment-stage pool.
        """
        short = pid[:12]
        if cancel_callback and cancel_callback():
            return self._empty_result(pt, search_lat, search_lon)

        # Download tiles -- no lock. All workers can fetch tiles in parallel
        # (separate aiohttp keep-alive connections share the session pool).
        # A throttled tile fetch surfaces as ClientResponseError from
        # raise_for_status(); treat those statuses as retryable, never as a
        # permanently-failed panorama.
        _log("INFO", f"  Downloading {short}… (zoom={_DOWNLOAD_ZOOM})")
        raw_image = None
        for attempt in range(_RL_MAX_RETRIES):
            if throttle is not None:
                await throttle.before_request()
            try:
                raw_image = await asyncio.wait_for(
                    gsv.get_panorama_async(download_pano, session, zoom=_DOWNLOAD_ZOOM),
                    timeout=_DOWNLOAD_TIMEOUT_S,
                )
                if throttle is not None:
                    await throttle.record_success()
                break
            except TimeoutError:
                _log(
                    "ERROR",
                    f"  Timeout (>{_DOWNLOAD_TIMEOUT_S:.0f}s) downloading "
                    f"{short}… — marking failed",
                )
                failed_panos.add(pid)
                return self._empty_result(pt, search_lat, search_lon)
            except Exception as e:
                status = getattr(e, "status", None)
                throttled = isinstance(e, gsv.RateLimitedError) or (
                    status in gsv._RATE_LIMIT_STATUSES
                )
                if not throttled:
                    _log(
                        "ERROR",
                        f"  Download error for {short}…: "
                        f"{type(e).__name__}: {e}",
                    )
                    failed_panos.add(pid)
                    return self._empty_result(pt, search_lat, search_lon)
                if throttle is not None:
                    await throttle.record_throttle(status)
                if cancel_callback and cancel_callback():
                    return self._empty_result(pt, search_lat, search_lon)
                if attempt == _RL_MAX_RETRIES - 1:
                    _log(
                        "ERROR",
                        f"  {short}…: still rate limited after "
                        f"{_RL_MAX_RETRIES} attempts — leaving unsampled",
                    )
                    # Deliberately NOT added to failed_panos: throttling is
                    # transient, so a later run should retry this panorama.
                    return self._empty_result(pt, search_lat, search_lon)
        if raw_image is None:
            return self._empty_result(pt, search_lat, search_lon)

        if cancel_callback and cancel_callback():
            return self._empty_result(pt, search_lat, search_lon)

        _log(
            "INFO",
            f"  Downloaded {short}… ({raw_image.width}×{raw_image.height}) "
            f"— preprocessing",
        )

        # Preprocess in the executor, never inline. Every worker here is a
        # coroutine on the *same* event-loop thread, so calling this directly
        # would block all of them for the duration and add latency to every
        # in-flight network callback. numpy and PIL release the GIL, so a
        # thread genuinely parallelises this.
        loop = asyncio.get_running_loop()
        img = await loop.run_in_executor(None, self._preprocess_image, raw_image)
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
        try:
            tensor = await loop.run_in_executor(
                None, self.segmenter.preprocess_to_tensor, img
            )
        except Exception as e:
            _log(
                "ERROR",
                f"  Tensor build failed for {short}…: " f"{type(e).__name__}: {e}",
            )
            return self._empty_result(pt, search_lat, search_lon)

        # GPU inference -- serialised so one image is on the device at a
        # time. While this lock is held by one task, others can download tiles
        # or preprocess in parallel.
        async with gpu_lock:
            # Another worker may have written the cache for this pano while we
            # were queued at the GPU lock; skip the forward in that case.
            if pid in pano_cache:
                cached = pano_cache[pid]
                _log(
                    "OK",
                    f"  Cache hit (after queue) {short}… → "
                    f"veg={cached['veg']:.3f} ter={cached['ter']:.3f}",
                )
                return self._make_result(
                    pt,
                    search_lat,
                    search_lon,
                    cached["veg"],
                    cached["ter"],
                    pid,
                    pano_date_str,
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
            # gvi_veg is vegetation only (Cityscapes class 8). Terrain (class 9)
            # is reported separately as gvi_ter, so the two bands stay
            # independent -- sum them downstream if a combined index is wanted.
            val_veg = metrics.get("GVI_Vegetation", 0.0)
            val_ter = metrics.get("GVI_Terrain", 0.0)
            pano_cache[pid] = {"veg": val_veg, "ter": val_ter}

            _log("OK", f"  GVI {short}… → veg={val_veg:.3f}  ter={val_ter:.3f}")

        # Optional disk artefacts
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
                            f"not handled -- skipping save",
                        )
                except Exception as e:
                    _log(
                        "ERROR",
                        f"  Failed to save mask {short}…: "
                        f"{type(e).__name__}: {e}",
                    )

        return self._make_result(
            pt, search_lat, search_lon, val_veg, val_ter, pid, pano_date_str
        )

    @staticmethod
    def _empty_result(pt: dict, search_lat: float, search_lon: float) -> dict:
        return {
            "orig_index": pt["orig_index"],
            "geometry": pt["geometry"],
            "gvi_veg": None,
            "gvi_ter": None,
            "pano_id": None,
            "pano_date": None,
            "lat": search_lat,
            "lon": search_lon,
            "row": pt["row"],
            "col": pt["col"],
            "cluster_id": pt.get("cluster_id", 0),
        }

    @staticmethod
    def _make_result(
        pt: dict,
        search_lat: float,
        search_lon: float,
        val_veg,
        val_ter,
        pid: str,
        pano_date: str | None = None,
    ) -> dict:
        return {
            "orig_index": pt["orig_index"],
            "geometry": pt["geometry"],
            "gvi_veg": float(val_veg) if val_veg is not None else None,
            "gvi_ter": float(val_ter) if val_ter is not None else None,
            "pano_id": pid,
            "pano_date": pano_date,
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
        target_year: int | None,
        max_year_diff: int | None,
        pause_callback: Callable[[], bool] | None = None,
    ) -> None:
        # Run-local set: tracks panos that already failed in *this* run.
        # Never written to pano_cache (which is the persistent session cache).
        failed_panos: set[str] = set()
        completed = 0
        # One throttle shared by every worker in both stages: a 429 seen by any
        # request slows the whole run down, and recovery is likewise global.
        throttle = AdaptiveThrottle()
        gpu_lock = asyncio.Lock()

        # Two-stage pipeline. The search stage resolves every point (misses and
        # cache hits finish there); only panoramas that are present *and*
        # uncached flow to the segment stage, which downloads and runs the GPU.
        # Decoupling the two lets the light, highly-parallel search run at
        # ``_SEARCH_CONCURRENCY`` while the GPU-bound segment stage stays small.
        search_q: asyncio.Queue = asyncio.Queue()
        for pt in points_to_process:
            search_q.put_nowait(pt)
        segment_q: asyncio.Queue = asyncio.Queue()
        _SEG_SENTINEL = object()

        # In-flight dedup: many neighbouring grid points resolve to the *same*
        # panorama. The first to reach an uncached pano enqueues it; the rest
        # register as waiters and are answered from the cache the moment that
        # one segmentation completes -- so each panorama downloads and runs the
        # GPU exactly once, no matter how many points share it.
        pending: dict[str, list] = {}

        completed_lock = asyncio.Lock()
        segmented = 0
        segmented_lock = asyncio.Lock()

        async def _emit(result: dict) -> None:
            nonlocal completed
            if result_callback:
                result_callback(result)
            async with completed_lock:
                completed += 1
                curr = completed
            if progress_callback:
                progress_callback(start_index + curr, total_points)

        async def _wait_while_paused() -> None:
            # Hold at a safe point while paused. Nothing queued is dropped, so
            # a resume picks up exactly where we left off.
            while pause_callback and pause_callback():
                if cancel_callback and cancel_callback():
                    return
                await asyncio.sleep(0.25)

        async with aiohttp.ClientSession() as session:

            async def _search_worker() -> None:
                while True:
                    if cancel_callback and cancel_callback():
                        return
                    await _wait_while_paused()
                    if cancel_callback and cancel_callback():
                        return
                    try:
                        pt = search_q.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    try:
                        outcome = await self._search_and_resolve(
                            pt,
                            session,
                            gdf,
                            failed_panos,
                            pano_cache,
                            cancel_callback,
                            target_year,
                            max_year_diff,
                            throttle,
                        )
                    except asyncio.CancelledError:
                        return
                    if outcome[0] == "done":
                        await _emit(outcome[1])
                        continue
                    # ("segment", download_pano, pid, pano_date_str, slat, slon)
                    _, download_pano, pid, pano_date_str, slat, slon = outcome
                    # The membership tests and the mutation below have no await
                    # between them, so two workers can never both enqueue the
                    # same pid -- one downloads it, the rest wait on the result.
                    if pid in pano_cache:
                        cached = pano_cache[pid]
                        await _emit(
                            self._make_result(
                                pt,
                                slat,
                                slon,
                                cached["veg"],
                                cached["ter"],
                                pid,
                                pano_date_str,
                            )
                        )
                    elif pid in pending:
                        pending[pid].append((pt, slat, slon, pano_date_str))
                    else:
                        pending[pid] = []
                        await segment_q.put(
                            (pt, download_pano, pid, pano_date_str, slat, slon)
                        )

            async def _segment_worker() -> None:
                nonlocal segmented
                while True:
                    item = await segment_q.get()
                    if item is _SEG_SENTINEL:
                        return
                    pt, download_pano, pid, pano_date_str, slat, slon = item
                    if cancel_callback and cancel_callback():
                        pending.pop(pid, None)
                        return
                    await _wait_while_paused()
                    try:
                        result = await self._segment_resolved(
                            pt,
                            download_pano,
                            pid,
                            pano_date_str,
                            slat,
                            slon,
                            session,
                            folder,
                            save_panos,
                            save_masks,
                            pano_cache,
                            failed_panos,
                            cancel_callback,
                            gpu_lock,
                            throttle,
                        )
                    except asyncio.CancelledError:
                        return
                    await _emit(result)
                    # Answer every point that was waiting on this same panorama
                    # with the value this segmentation produced. The values come
                    # from the primary's own result, never re-read from the
                    # cache, so dedup stays correct even when the cache is a
                    # no-op store (and a failed download makes every waiter a
                    # miss too, matching the primary).
                    waiters = pending.pop(pid, [])
                    res_veg = result.get("gvi_veg")
                    res_ter = result.get("gvi_ter")
                    for wpt, wlat, wlon, wdate in waiters:
                        if res_veg is not None:
                            await _emit(
                                self._make_result(
                                    wpt, wlat, wlon, res_veg, res_ter, pid, wdate
                                )
                            )
                        else:
                            await _emit(self._empty_result(wpt, wlat, wlon))
                    # Defensive: release cached blocks back to the device
                    # periodically so long-running jobs don't accumulate
                    # allocator fragmentation.
                    async with segmented_lock:
                        segmented += 1
                        do_empty = segmented % _EMPTY_CACHE_EVERY_N == 0
                    if do_empty and self.device.type == "cuda":
                        try:
                            import torch

                            torch.cuda.empty_cache()
                        except Exception:
                            pass

            searchers = [
                asyncio.create_task(_search_worker())
                for _ in range(_SEARCH_CONCURRENCY)
            ]
            segmenters = [
                asyncio.create_task(_segment_worker())
                for _ in range(_MAX_CONCURRENT_POINTS)
            ]

            # Watcher: cancels every worker on user cancel.
            async def _cancel_watcher() -> None:
                while True:
                    await asyncio.sleep(0.25)
                    if cancel_callback and cancel_callback():
                        for t in (*searchers, *segmenters):
                            if not t.done():
                                t.cancel()
                        return

            watcher = asyncio.create_task(_cancel_watcher())

            try:
                # Drain the search stage, then signal the segment stage to stop
                # once its queue empties.
                await asyncio.gather(*searchers, return_exceptions=True)
                for _ in segmenters:
                    segment_q.put_nowait(_SEG_SENTINEL)
                await asyncio.gather(*segmenters, return_exceptions=True)
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
        target_year=None,
        max_year_diff=None,
        pause_callback=None,
    ):
        _log("INFO", f"Starting Analysis (Resume Index: {start_index})...")
        if target_year is not None:
            _log(
                "INFO",
                f"Target capture year: {target_year}"
                + (
                    f" (±{max_year_diff} yr max)"
                    if max_year_diff is not None
                    else " (closest available)"
                ),
            )
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
                target_year=target_year,
                max_year_diff=max_year_diff,
                pause_callback=pause_callback,
            )
        )

        if _close_pbar:
            _pbar.close()

        return gpd.GeoDataFrame()
