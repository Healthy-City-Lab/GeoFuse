import asyncio
import os
import time
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


# Result key marking a point left without a sample because no panorama lies
# within the search radius, or because its location has panoramas but none
# inside the requested year / month window. Counted into the run's stats and
# dropped before outputs are written, so output schemas are unchanged.
CAPTURE_STATUS_KEY = "capture_status"
NO_PANORAMA = "no_panorama"
NO_CAPTURE_IN_WINDOW = "no_capture_in_window"
# Result key naming the rule a window miss broke: ``"year"`` (no capture in
# the year range) or ``"month"`` (captures in the year range, none in the
# months).
WINDOW_MISS_KEY = "window_miss"
#: What became of each point, counted in ``GVIEngine.last_run_stats``.
POINT_OUTCOMES = ("sampled", NO_PANORAMA, NO_CAPTURE_IN_WINDOW, "failed")


def _outcome(result: dict) -> str:
    """Which of :data:`POINT_OUTCOMES` a point's result is."""
    if result.get("pano_id") is not None:
        return "sampled"
    status = result.get(CAPTURE_STATUS_KEY)
    return status if status in (NO_PANORAMA, NO_CAPTURE_IN_WINDOW) else "failed"


def _window_miss(pano, target_year: int | None, max_year_diff: int | None) -> str:
    """The rule that left ``pano`` without a capture: ``"year"`` when none is
    within ``max_year_diff`` of ``target_year``, otherwise ``"month"``."""
    if target_year is None or max_year_diff is None:
        return "month"
    near = [
        c
        for c in pano.captures
        if c.year is not None and abs(c.year - target_year) <= max_year_diff
    ]
    return "month" if near else "year"


def _progress_summary(stats: dict, done: int, elapsed: float, throttled: int) -> str:
    """One INFO line: the points done, what became of them, and the rates."""
    return (
        f"Points {done:,}/{stats['points']:,}: {stats['sampled']:,} sampled, "
        f"{stats[NO_PANORAMA]:,} no panorama, "
        f"{stats[NO_CAPTURE_IN_WINDOW]:,} no capture in the window "
        f"(year {stats['no_capture_in_year_window']:,}, "
        f"month {stats['no_capture_in_month_window']:,}), "
        f"{stats['failed']:,} failed; {stats['cache_hits']:,} from cache, "
        f"{stats['downloads']:,} downloaded; "
        f"{done / max(elapsed, 1e-9):.1f} points/s, {throttled} throttle event(s)"
    )


# Search radius (metres) for find_panorama_async around each grid point.
_SEARCH_RADIUS_M = 50.0
# Seconds between the INFO summaries of a run's progress. The lines about
# single points and panoramas are DEBUG, logged only with ``log_points``.
_SUMMARY_EVERY_S = 30.0
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
# The analysis runs as three stages with independent concurrency (see
# ``_run_analysis_async``):
#   * search — resolves every point's nearest panorama. These calls are light
#     and highly parallel, so misses and cache hits (which never touch the GPU)
#     flow without waiting behind downloads.
#   * download — fetches and preprocesses each panorama that needs segmenting.
#     It is latency-bound, so the pool is wide enough to keep the GPU fed.
#   * segment — one consumer runs the model on what the downloads queued, up to
#     ``batch_size`` images per forward.
# Search and download share the adaptive throttle below, which lowers the
# *effective* request rate when Google pushes back, so the pool sizes are
# ceilings rather than fixed rates. The download -> segment queue is bounded,
# which caps the tensors held in memory.
_SEARCH_CONCURRENCY = 48
#: Default download pool (``download_concurrency``).
_DOWNLOAD_CONCURRENCY = 16
#: Default images per forward (``batch_size``). One image at this size already
#: occupies a single GPU, and keeps every mask identical to an image-by-image
#: run; larger batches are for devices where they pay off.
_BATCH_SIZE = 1
#: Default wait for a batch to fill before a partial one runs (``batch_wait_ms``).
_BATCH_WAIT_MS = 20
# Threads that decode, crop and normalise downloaded panoramas. The GPU forward
# runs on a thread of its own, so it never queues behind this work, and the
# pool stays small so its GIL hand-offs do not slow the forward's launches.
_PREP_THREADS = 2
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
# How long Street View may answer pages instead of data, with no clean answer
# between, before the run stops as "service retired" (seconds). Until then a
# point retries as long as it takes.
_UNEXPECTED_GIVE_UP_S = 15 * 60


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
        self._unexpected_since: float | None = None
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

    async def record_unexpected(self, error: gsv.UnexpectedAnswer) -> None:
        """Street View answered a page instead of data: back off as for a
        throttle. Raises :class:`~geofuse.streetview.ServiceRetired` once such
        answers have lasted :data:`_UNEXPECTED_GIVE_UP_S` with no clean answer
        between."""
        async with self._lock:
            now = asyncio.get_running_loop().time()
            if self._unexpected_since is None:
                self._unexpected_since = now
            lasted = now - self._unexpected_since
        if lasted >= _UNEXPECTED_GIVE_UP_S:
            raise gsv.ServiceRetired(
                f"{error}, and nothing else for {lasted / 60:.0f} minutes"
            )
        await self.record_throttle(reason=f"Street View answered a page ({error})")

    async def record_throttle(
        self, status: int | None = None, reason: str | None = None
    ) -> None:
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
                    f"{reason or f'Rate limited (HTTP {status})'} — backing off "
                    f"{self._backoff:.0f}s, spacing requests {self._delay:.2f}s "
                    f"(event #{self.trips})",
                )
                self._backoff = min(self._backoff * 2.0, _RL_BACKOFF_MAX_S)

    async def record_success(self) -> None:
        """A clean response: decay the spacing after a sustained good streak."""
        async with self._lock:
            self._unexpected_since = None
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


#: Where panoramas come from: the keyless scraper, or Google's keyed tile API.
DOWNLOAD_MODES = ("package", "api")


class GVIEngine:
    download_mode: str = "package"
    api_key: str | None = None
    log_points: bool = False

    def __init__(
        self, model_path=None, download_mode="package", device="cuda", api_key=None
    ):
        """``api_key`` selects the keyed Street View tile API (``download_mode``
        becomes ``"api"``); ``download_mode="api"`` without a key is refused."""
        if download_mode not in DOWNLOAD_MODES:
            raise ValueError(f"download_mode must be one of {DOWNLOAD_MODES}")
        self.download_mode = "api" if api_key else download_mode
        if self.download_mode == "api" and not api_key:
            raise ValueError(
                "The Street View API download needs a key: type one in the GVI "
                "tab or store it with `geofuse credentials set streetview-key`."
            )
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

    def _point_log(self, msg: str) -> None:
        """A line about one point or panorama: DEBUG, only with ``log_points``."""
        if self.log_points:
            _log("DEBUG", msg)

    def _find(self, lat: float, lon: float, session: aiohttp.ClientSession):
        """The search request for this engine's download mode."""
        if self.download_mode == "api":
            return gsv.find_panorama_api_async(
                lat, lon, session, self.api_key, radius=_SEARCH_RADIUS_M
            )
        return gsv.find_panorama_async(lat, lon, session, radius=_SEARCH_RADIUS_M)

    def _fetch(self, pano, session: aiohttp.ClientSession):
        """The download request for this engine's download mode."""
        if self.download_mode == "api":
            return gsv.get_panorama_api_async(
                pano, session, self.api_key, zoom=_DOWNLOAD_ZOOM
            )
        return gsv.get_panorama_async(pano, session, zoom=_DOWNLOAD_ZOOM)

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
        month_window: tuple[int, int] | None = None,
        month_policy: str = "exclude",
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
        A search answered with a page instead of data retries until it gets an
        answer, or until the throttle decides the service changed.
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

        self._point_log(f"Point {idx} @ ({search_lat:.5f}, {search_lon:.5f})")

        # Search for the nearest panorama. Throttling is retried (with the
        # shared backoff) rather than treated as "no coverage" -- otherwise
        # rate limiting silently turns into missing sample points.
        pano = _RL_SENTINEL = object()
        try:
            attempt = 0
            while attempt < _RL_MAX_RETRIES:
                if throttle is not None:
                    await throttle.before_request()
                try:
                    pano = await self._find(search_lat, search_lon, session)
                    if throttle is not None:
                        await throttle.record_success()
                    break
                except gsv.UnexpectedAnswer as odd:
                    if throttle is None:
                        raise gsv.ServiceRetired(str(odd)) from None
                    await throttle.record_unexpected(odd)
                    if cancel_callback and cancel_callback():
                        return ("done", self._empty_result(pt, search_lat, search_lon))
                    continue
                except gsv.RateLimitedError as rl:
                    attempt += 1
                    if throttle is not None:
                        await throttle.record_throttle(rl.status)
                    if cancel_callback and cancel_callback():
                        return ("done", self._empty_result(pt, search_lat, search_lon))
                    if attempt == _RL_MAX_RETRIES:
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
        except (gsv.ServiceRetired, gsv.ApiRequestRefused):
            raise
        except Exception as e:
            _log(
                "ERROR",
                f"  Point {idx}: the panorama search raised — "
                f"{type(e).__name__}: {e}",
            )
            return ("done", self._empty_result(pt, search_lat, search_lon))

        if pano is None:
            self._point_log(
                f"  Point {idx}: no panoramas within {_SEARCH_RADIUS_M:.0f} m"
            )
            miss = self._empty_result(pt, search_lat, search_lon)
            miss[CAPTURE_STATUS_KEY] = NO_PANORAMA
            return ("done", miss)

        # Pick which dated capture to use. Without a target year or a month
        # window we keep the panorama the search returned (the most recent
        # coverage). A point whose location has panoramas but none inside the
        # year / month window is marked, so the sidecar can count it.
        download_pano = pano
        pano_date = getattr(pano, "date", None)
        if target_year is not None or month_window is not None:
            capture = pano.select_capture(
                target_year,
                max_year_diff,
                month_window=month_window,
                month_policy=month_policy,
            )
            if capture is None:
                window = []
                if target_year is not None:
                    window.append(f"{max_year_diff} yr of {target_year}")
                if month_window is not None:
                    window.append(f"months {month_window[0]}-{month_window[1]}")
                self._point_log(
                    f"  Point {idx}: no capture within {' and '.join(window)} "
                    f"(available: "
                    f"{[(c.year, c.month) for c in pano.captures]})"
                )
                miss = self._empty_result(pt, search_lat, search_lon)
                miss[CAPTURE_STATUS_KEY] = NO_CAPTURE_IN_WINDOW
                miss[WINDOW_MISS_KEY] = _window_miss(pano, target_year, max_year_diff)
                return ("done", miss)
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
            self._point_log(f"  Skipping {short}… (failed earlier this run)")
            return ("done", self._empty_result(pt, search_lat, search_lon))

        if pid in pano_cache:
            cached = pano_cache[pid]
            self._point_log(
                f"  Cache hit {short}… → veg={cached['veg']:.3f} "
                f"ter={cached['ter']:.3f}"
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

    async def _download_and_prepare(
        self,
        pt: dict,
        download_pano,
        pid: str,
        search_lat: float,
        search_lon: float,
        session: aiohttp.ClientSession,
        failed_panos: set,
        cancel_callback: Callable[..., bool] | None,
        throttle: "AdaptiveThrottle | None" = None,
    ) -> tuple:
        """Stage 2: download a resolved panorama and build its model input.

        Returns ``("ready", image, tensor)``, or ``("done", empty_result)``
        when the panorama cannot be used (timeout, download error, blank
        image, persistent throttling, cancel).
        """
        short = pid[:12]
        empty = ("done", self._empty_result(pt, search_lat, search_lon))
        if cancel_callback and cancel_callback():
            return empty

        # A throttled tile fetch surfaces as ClientResponseError from
        # raise_for_status(); those statuses are retried, never taken as a
        # permanently-failed panorama.
        self._point_log(f"  Downloading {short}… (zoom={_DOWNLOAD_ZOOM})")
        raw_image = None
        for attempt in range(_RL_MAX_RETRIES):
            if throttle is not None:
                await throttle.before_request()
            try:
                raw_image = await asyncio.wait_for(
                    self._fetch(download_pano, session),
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
                return empty
            except gsv.ApiRequestRefused:
                raise
            except Exception as e:
                status = getattr(e, "status", None)
                throttled = isinstance(e, gsv.RateLimitedError) or (
                    status in gsv._RATE_LIMIT_STATUSES
                )
                if not throttled:
                    _log(
                        "ERROR",
                        f"  Download error for {short}…: " f"{type(e).__name__}: {e}",
                    )
                    failed_panos.add(pid)
                    return empty
                if throttle is not None:
                    await throttle.record_throttle(status)
                if cancel_callback and cancel_callback():
                    return empty
                if attempt == _RL_MAX_RETRIES - 1:
                    _log(
                        "ERROR",
                        f"  {short}…: still rate limited after "
                        f"{_RL_MAX_RETRIES} attempts — leaving unsampled",
                    )
                    # Not added to failed_panos: throttling is transient, so a
                    # later run should retry this panorama.
                    return empty
        if raw_image is None or (cancel_callback and cancel_callback()):
            return empty

        self._point_log(
            f"  Downloaded {short}… ({raw_image.width}×{raw_image.height}) "
            f"— preprocessing"
        )
        # Preprocess in the executor, never inline: every worker is a coroutine
        # on one event-loop thread, and numpy / PIL release the GIL, so a
        # thread genuinely parallelises this.
        loop = asyncio.get_running_loop()
        prep_pool = getattr(self, "_prep_pool", None)
        img = await loop.run_in_executor(prep_pool, self._preprocess_image, raw_image)
        try:
            raw_image.close()
        except Exception:
            pass
        del raw_image
        if not img:
            _log("WARN", f"  Preprocess failed for {short}…: blank or black image")
            failed_panos.add(pid)
            return empty
        try:
            tensor = await loop.run_in_executor(
                prep_pool, self.segmenter.preprocess_to_tensor, img
            )
        except Exception as e:
            _log(
                "ERROR",
                f"  Tensor build failed for {short}…: " f"{type(e).__name__}: {e}",
            )
            return empty
        return ("ready", img, tensor)

    async def _segment_batch(
        self,
        batch: list,
        folder: str,
        save_panos: bool,
        save_masks: bool,
        pano_cache: dict,
        stats: dict,
    ) -> list[tuple[str, dict]]:
        """Stage 3: one forward over ``batch`` (``(item, image, tensor)``
        entries); returns ``(panorama id, result)`` for each entry.

        The segmenter counts the classes on the device; class masks come back
        only when they are saved. A panorama another entry cached while this
        one queued is answered from the cache.
        """
        loop = asyncio.get_running_loop()
        out: list[tuple[str, dict]] = []
        todo = []
        for item, img, tensor in batch:
            pt, _pano, pid, pano_date_str, slat, slon = item
            if pid in pano_cache:
                cached = pano_cache[pid]
                self._point_log(
                    f"  Cache hit (after queue) {pid[:12]}… → "
                    f"veg={cached['veg']:.3f} ter={cached['ter']:.3f}"
                )
                out.append(
                    (
                        pid,
                        self._make_result(
                            pt,
                            slat,
                            slon,
                            cached["veg"],
                            cached["ter"],
                            pid,
                            pano_date_str,
                        ),
                    )
                )
            else:
                todo.append((item, img, tensor))
        if not todo:
            return out

        gpu_pool = getattr(self, "_gpu_pool", None)
        started = time.perf_counter()
        try:
            if len(todo) == 1:
                stacked = todo[0][2]
            else:
                import torch

                stacked = torch.cat([tensor for _item, _img, tensor in todo])
            fractions, masks = await loop.run_in_executor(
                gpu_pool, self.segmenter.segment, stacked, save_masks
            )
        except Exception as e:
            _log(
                "ERROR",
                f"  GPU inference failed for {len(todo)} panorama(s): "
                f"{type(e).__name__}: {e}",
            )
            for (pt, _pano, pid, _date, slat, slon), _img, _tensor in todo:
                out.append((pid, self._empty_result(pt, slat, slon)))
            return out
        stats["gpu_seconds"] += time.perf_counter() - started
        stats["gpu_batches"] += 1
        stats["segmented"] += len(todo)

        if masks is None:
            masks = [None] * len(todo)
        for (item, img, _tensor), metrics, mask in zip(
            todo, fractions, masks, strict=True
        ):
            pt, _pano, pid, pano_date_str, slat, slon = item
            # gvi_veg is vegetation only (Cityscapes class 8). Terrain (class 9)
            # is reported separately as gvi_ter, so the two bands stay
            # independent -- sum them downstream if a combined index is wanted.
            val_veg = metrics.get("GVI_Vegetation", 0.0)
            val_ter = metrics.get("GVI_Terrain", 0.0)
            pano_cache[pid] = {"veg": val_veg, "ter": val_ter}
            self._point_log(f"  GVI {pid[:12]}… → veg={val_veg:.3f}  ter={val_ter:.3f}")
            if save_panos or save_masks:
                await loop.run_in_executor(
                    getattr(self, "_prep_pool", None),
                    self._save_artefacts,
                    folder,
                    pid,
                    img,
                    mask,
                    save_panos,
                    save_masks,
                )
            out.append(
                (
                    pid,
                    self._make_result(
                        pt, slat, slon, val_veg, val_ter, pid, pano_date_str
                    ),
                )
            )
        return out

    def _save_artefacts(self, folder, pid, img, mask, save_panos, save_masks) -> None:
        """Write the panorama and / or its class mask, unless already there."""
        short = pid[:12]
        if save_panos:
            p_path = os.path.join(folder, "images", f"{pid}.jpg")
            if not os.path.exists(p_path):
                try:
                    img_to_save = (
                        Image.fromarray(img) if isinstance(img, np.ndarray) else img
                    )
                    img_to_save.save(p_path)
                    self._point_log(f"  Saved panorama → {p_path}")
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
                        self._point_log(f"  Saved mask     → {m_path}")
                    elif isinstance(m, Image.Image):
                        m.putpalette(CITYSCAPES_PALETTE)
                        m.save(m_path)
                        self._point_log(f"  Saved mask     → {m_path}")
                    else:
                        _log(
                            "WARN",
                            f"  Mask type '{type(mask).__name__}' "
                            f"not handled -- skipping save",
                        )
                except Exception as e:
                    _log(
                        "ERROR",
                        f"  Failed to save mask {short}…: " f"{type(e).__name__}: {e}",
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
        month_window: tuple[int, int] | None = None,
        month_policy: str = "exclude",
        download_concurrency: int = _DOWNLOAD_CONCURRENCY,
        batch_size: int = _BATCH_SIZE,
        batch_wait_ms: float = _BATCH_WAIT_MS,
    ) -> None:
        # Run-local set: tracks panos that already failed in *this* run.
        # Never written to pano_cache (which is the persistent session cache).
        failed_panos: set[str] = set()
        completed = 0
        # One throttle shared by every worker: a 429 seen by any request slows
        # the whole run down, and recovery is likewise global.
        throttle = AdaptiveThrottle()
        batch_size = max(1, int(batch_size))
        stats = {
            "points": len(points_to_process),
            "searched": 0,
            "cache_hits": 0,
            "downloads": 0,
            "segmented": 0,
            "gpu_batches": 0,
            "gpu_seconds": 0.0,
            **dict.fromkeys(POINT_OUTCOMES, 0),
            "no_capture_in_year_window": 0,
            "no_capture_in_month_window": 0,
        }
        started = time.perf_counter()
        from concurrent.futures import ThreadPoolExecutor

        self._prep_pool = ThreadPoolExecutor(_PREP_THREADS, "gvi-prep")
        self._gpu_pool = ThreadPoolExecutor(1, "gvi-gpu")

        # Three stages (see the module constants): every point is searched;
        # misses and cache hits finish there. Panoramas present and uncached
        # go to the download pool, which queues their model inputs for the
        # one GPU consumer. The queue is bounded, so downloads wait when the
        # GPU falls behind instead of piling tensors up in memory.
        search_q: asyncio.Queue = asyncio.Queue()
        for pt in points_to_process:
            search_q.put_nowait(pt)
        download_q: asyncio.Queue = asyncio.Queue()
        gpu_q: asyncio.Queue = asyncio.Queue(maxsize=max(2, 2 * batch_size))
        _STOP = object()

        # In-flight dedup: many neighbouring grid points resolve to the *same*
        # panorama. The first to reach an uncached pano enqueues it; the rest
        # register as waiters and are answered the moment that one
        # segmentation completes -- so each panorama downloads and runs the
        # GPU exactly once, no matter how many points share it.
        pending: dict[str, list] = {}

        completed_lock = asyncio.Lock()
        segmented = 0

        async def _emit(result: dict) -> None:
            nonlocal completed
            outcome = _outcome(result)
            stats[outcome] += 1
            if outcome == NO_CAPTURE_IN_WINDOW:
                stats[f"no_capture_in_{result[WINDOW_MISS_KEY]}_window"] += 1
            # Skip points with no panorama (pano_id=None) so the accumulated
            # list — and every output format built from it — only ever holds
            # points with real data. Progress still counts every attempt.
            if result_callback and result.get("pano_id") is not None:
                result_callback(result)
            async with completed_lock:
                completed += 1
                curr = completed
            if progress_callback:
                progress_callback(start_index + curr, total_points)

        async def _finish(pid: str, result: dict) -> None:
            """Emit a segmented (or failed) panorama's point and its waiters."""
            nonlocal segmented
            await _emit(result)
            # The waiters get the primary's own values, never re-read from the
            # cache, so dedup stays correct when the cache is a no-op store
            # (and a failed download makes every waiter a miss too).
            res_veg = result.get("gvi_veg")
            res_ter = result.get("gvi_ter")
            for wpt, wlat, wlon, wdate in pending.pop(pid, []):
                if res_veg is not None:
                    await _emit(
                        self._make_result(wpt, wlat, wlon, res_veg, res_ter, pid, wdate)
                    )
                else:
                    await _emit(self._empty_result(wpt, wlat, wlon))
            # Release cached blocks back to the device periodically so long
            # runs don't accumulate allocator fragmentation.
            segmented += 1
            if segmented % _EMPTY_CACHE_EVERY_N == 0 and self.device.type == "cuda":
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass

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
                            month_window=month_window,
                            month_policy=month_policy,
                        )
                    except asyncio.CancelledError:
                        return
                    stats["searched"] += 1
                    if outcome[0] == "done":
                        if outcome[1].get("pano_id") is not None:
                            stats["cache_hits"] += 1
                        await _emit(outcome[1])
                        continue
                    # ("segment", download_pano, pid, pano_date_str, slat, slon)
                    _, download_pano, pid, pano_date_str, slat, slon = outcome
                    # The membership tests and the mutation below have no await
                    # between them, so two workers can never both enqueue the
                    # same pid -- one downloads it, the rest wait on the result.
                    if pid in pano_cache:
                        cached = pano_cache[pid]
                        stats["cache_hits"] += 1
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
                        await download_q.put(
                            (pt, download_pano, pid, pano_date_str, slat, slon)
                        )

            async def _download_worker() -> None:
                while True:
                    item = await download_q.get()
                    if item is _STOP:
                        return
                    pt, download_pano, pid, _date, slat, slon = item
                    if cancel_callback and cancel_callback():
                        pending.pop(pid, None)
                        return
                    await _wait_while_paused()
                    try:
                        prepared = await self._download_and_prepare(
                            pt,
                            download_pano,
                            pid,
                            slat,
                            slon,
                            session,
                            failed_panos,
                            cancel_callback,
                            throttle,
                        )
                    except asyncio.CancelledError:
                        return
                    if prepared[0] == "done":
                        await _finish(pid, prepared[1])
                        continue
                    stats["downloads"] += 1
                    await gpu_q.put((item, prepared[1], prepared[2]))

            async def _gpu_consumer() -> None:
                loop = asyncio.get_running_loop()
                while True:
                    first = await gpu_q.get()
                    if first is _STOP:
                        return
                    batch, stop = [first], False
                    deadline = loop.time() + batch_wait_ms / 1000.0
                    while len(batch) < batch_size:
                        wait = deadline - loop.time()
                        if wait <= 0:
                            break
                        try:
                            nxt = await asyncio.wait_for(gpu_q.get(), timeout=wait)
                        except TimeoutError:
                            break
                        if nxt is _STOP:
                            stop = True
                            break
                        batch.append(nxt)
                    if cancel_callback and cancel_callback():
                        return
                    await _wait_while_paused()
                    for pid, result in await self._segment_batch(
                        batch, folder, save_panos, save_masks, pano_cache, stats
                    ):
                        await _finish(pid, result)
                    if stop:
                        return

            searchers = [
                asyncio.create_task(_search_worker())
                for _ in range(_SEARCH_CONCURRENCY)
            ]
            downloaders = [
                asyncio.create_task(_download_worker())
                for _ in range(max(1, int(download_concurrency)))
            ]
            consumer = asyncio.create_task(_gpu_consumer())

            # Watcher: cancels every worker on user cancel.
            async def _cancel_watcher() -> None:
                while True:
                    await asyncio.sleep(0.25)
                    if cancel_callback and cancel_callback():
                        for t in (*searchers, *downloaders, consumer):
                            if not t.done():
                                t.cancel()
                        return

            watcher = asyncio.create_task(_cancel_watcher())

            # Reporter: a summary at most every _SUMMARY_EVERY_S, when points
            # were done since the last one.
            async def _reporter() -> None:
                reported = 0
                while True:
                    await asyncio.sleep(_SUMMARY_EVERY_S)
                    if completed == reported:
                        continue
                    reported = completed
                    elapsed = time.perf_counter() - started
                    line = _progress_summary(stats, reported, elapsed, throttle.trips)
                    _log("INFO", line)

            reporter = asyncio.create_task(_reporter())

            try:
                # Drain the stages in order: searching ends, then the
                # downloads, then the consumer once its queue is empty.
                searched = await asyncio.gather(*searchers, return_exceptions=True)
                for _ in downloaders:
                    download_q.put_nowait(_STOP)
                downloaded = await asyncio.gather(*downloaders, return_exceptions=True)
                if not consumer.done():
                    await gpu_q.put(_STOP)
                await asyncio.gather(consumer, return_exceptions=True)
            finally:
                for task in (watcher, reporter):
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            if cancel_callback and cancel_callback():
                _log("WARN", "Analysis Aborted by User.")

        for pool in (self._prep_pool, self._gpu_pool):
            pool.shutdown(wait=False, cancel_futures=True)
        # A retired search or a refused API key fails the run instead of
        # reading as no coverage.
        fatal = (gsv.ServiceRetired, gsv.ApiRequestRefused)
        stopped = [r for r in (*searched, *downloaded) if isinstance(r, fatal)]
        if stopped:
            raise stopped[0]
        elapsed = max(time.perf_counter() - started, 1e-9)
        stats.update(
            seconds=round(elapsed, 2),
            throttle_events=throttle.trips,
            download_concurrency=max(1, int(download_concurrency)),
            batch_size=batch_size,
            searches_per_s=round(stats["searched"] / elapsed, 2),
            downloads_per_s=round(stats["downloads"] / elapsed, 2),
            gpu_images_per_s=round(stats["segmented"] / elapsed, 2),
            gpu_busy_share=round(stats["gpu_seconds"] / elapsed, 3),
            precision=self.segmenter.effective_precision,
            cuda_graph=self.segmenter.graphs_in_use,
            download_mode=self.download_mode,
        )
        stats["gpu_seconds"] = round(stats["gpu_seconds"], 2)
        self.last_run_stats = stats
        _log("INFO", _progress_summary(stats, completed, elapsed, throttle.trips))
        _log(
            "INFO",
            f"Throughput: {stats['searched']:,} searched "
            f"({stats['searches_per_s']}/s, {stats['cache_hits']:,} from cache), "
            f"{stats['downloads']:,} downloaded ({stats['downloads_per_s']}/s), "
            f"{stats['segmented']:,} segmented ({stats['gpu_images_per_s']}/s, "
            f"{stats['gpu_batches']:,} forwards, {stats['precision']}, GPU busy "
            f"{100 * stats['gpu_busy_share']:.0f} %), "
            f"{stats['throttle_events']} throttle event(s) in {stats['seconds']} s",
        )

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
        month_window=None,
        month_policy="exclude",
        download_concurrency=None,
        batch_size=None,
        batch_wait_ms=None,
        precision=None,
        log_points=False,
    ):
        """Segment the panorama nearest each point and record its GVI.

        ``target_year`` / ``max_year_diff`` choose the historical capture
        (``None`` keeps the most recent coverage); ``month_window`` /
        ``month_policy`` restrict it to a season (see
        :meth:`streetview.StreetViewPanorama.select_capture`).
        ``download_concurrency``, ``batch_size`` and ``batch_wait_ms`` size the
        download pool and the GPU batches (``None``: the module defaults).
        ``precision`` (``vision.PRECISIONS``; ``None`` keeps the segmenter's)
        sets the forward's arithmetic. Afterwards ``last_run_stats`` holds each
        stage's counts and rates, what became of the points
        (:data:`POINT_OUTCOMES`, with the window misses also split into
        ``no_capture_in_year_window`` / ``no_capture_in_month_window``), and
        the precision the forwards ran at; ``None`` when no point was run.

        The log gets a summary of the progress at most every 30 s and the
        warnings and errors; ``log_points`` adds a DEBUG line for every point
        searched and every panorama downloaded, segmented or saved.
        """
        self.last_run_stats = None
        self.log_points = bool(log_points)
        _log("INFO", f"Starting Analysis (Resume Index: {start_index})...")
        if precision is not None:
            self.segmenter.precision = precision
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
        if month_window is not None:
            _log(
                "INFO",
                f"Capture months {month_window[0]}-{month_window[1]} "
                f"({month_policy} captures outside them)",
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
                month_window=month_window,
                month_policy=month_policy,
                download_concurrency=(
                    _DOWNLOAD_CONCURRENCY
                    if download_concurrency is None
                    else download_concurrency
                ),
                batch_size=_BATCH_SIZE if batch_size is None else batch_size,
                batch_wait_ms=(
                    _BATCH_WAIT_MS if batch_wait_ms is None else batch_wait_ms
                ),
            )
        )

        if _close_pbar:
            _pbar.close()

        return gpd.GeoDataFrame()
