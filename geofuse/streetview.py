"""
Minimal Google Street View client for GeoFuse.

Adapted from `streetlevel <https://github.com/sk-zk/streetlevel>`_ (MIT
License) — only the parts needed to find the panorama nearest a point and
download its equirectangular image, all without an API key.

Why a custom port rather than ``pip install streetlevel``: the upstream
package pulls in ``pyfrpc``, ``bd09convertor``, ``CoordinatesConverter``,
``pyequilib``, ``pyexiv2``, and ``pycryptodome`` — several of which require
a Visual C++ build chain on Windows.  None of those are needed for Google
Street View; they are used by the Mapy.cz, Baidu, Apple Look Around, and
EXIF-writing modules.

Public surface
--------------
``find_panorama`` / ``find_panorama_async``
    The nearest panorama within a radius (from Google's coverage tiles, then
    its metadata), as a :class:`StreetViewPanorama` or ``None``.

``get_panorama`` / ``get_panorama_async``
    Download tiles in parallel and return a stitched PIL image.

``find_panorama_api_async`` / ``get_panorama_api_async``
    The same two steps through Google's keyed, billed Street View tile API
    (Map Tiles API), for runs given an API key.

``StreetViewPanorama``
    Lightweight dataclass with the minimum fields needed to download tiles
    (``id``, ``lat``, ``lon``, ``tile_size``, ``image_sizes``).
"""

from __future__ import annotations

import asyncio
import io
import itertools
import json
import math
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

import aiohttp
import requests
from PIL import Image

# ── Dataclasses ─────────────────────────────────────────────────────


class RateLimitedError(RuntimeError):
    """Google pushed back on the request rate (HTTP 429 / 403).

    Raised instead of returning a parsed result so callers can back off and
    **retry**. Without this the search endpoint's throttle response would fall
    through the JSONP repair as an empty list and be indistinguishable from
    "no panorama here" — silently turning throttling into missing data.
    """

    def __init__(self, status: int, url: str = "") -> None:
        super().__init__(f"rate limited: HTTP {status}")
        self.status = status
        self.url = url


#: Statuses that mean "you are being throttled", not "no data".
_RATE_LIMIT_STATUSES = frozenset({403, 429, 503})


class UnexpectedAnswer(RuntimeError):
    """A Street View endpoint answered something other than its data, such as
    a web page: usually a temporary block or error page. Callers back off and
    retry; only a persistent run of these means the service changed."""

    def __init__(self, what: str, text: str):
        super().__init__(f"the {what} answered {text[:80]!r}")


class ServiceRetired(RuntimeError):
    """A Street View endpoint no longer answers in the form this client reads."""

    def __init__(self, message: str):
        super().__init__(
            "Google's Street View service no longer answers as the GVI "
            f"downloader expects ({message}); GVI cannot sample new points until "
            "the downloader is updated."
        )


#: A place with Street View coverage, for checking that the search answers.
PROBE_POINT = (40.7580, -73.9855)


@dataclass
class Size:
    """A 2-D size in pixels."""

    x: int
    y: int


@dataclass
class Tile:
    """One tile of a tiled equirectangular panorama."""

    x: int
    y: int
    url: str


@dataclass
class PanoCapture:
    """One capture (panorama id + capture date) available at a location.

    Google Street View re-photographs the same spot over the years; each
    pass is a distinct panorama with its own id and ``(year, month)``.
    ``year`` / ``month`` are ``None`` only when the date could not be parsed.
    """

    id: str
    year: int | None
    month: int | None


def in_month_window(month: int | None, window: tuple[int, int]) -> bool:
    """Whether ``month`` lies in the inclusive window ``(first, last)``.

    ``first > last`` wraps over the new year (``(11, 3)`` is November to
    March). An unknown month is never inside.
    """
    if month is None:
        return False
    first, last = int(window[0]), int(window[1])
    if first <= last:
        return first <= month <= last
    return month >= first or month <= last


def month_distance(month: int | None, window: tuple[int, int]) -> float:
    """Months between ``month`` and the window's middle, around the year."""
    if month is None:
        return float("inf")
    first, last = int(window[0]), int(window[1])
    middle = first + ((last - first) % 12) / 2.0
    gap = abs((month - middle) % 12)
    return min(gap, 12 - gap)


@dataclass
class StreetViewPanorama:
    """Minimal Street View panorama metadata: ID, location, image grid.

    ``date`` is this panorama's own ``(year, month)`` capture date, and
    ``captures`` lists every capture available at this location (this
    panorama plus its historical passes), newest first.
    """

    id: str
    lat: float
    lon: float
    tile_size: Size
    image_sizes: list[Size]
    date: tuple[int, int] | None = None
    captures: list[PanoCapture] = field(default_factory=list)

    @property
    def is_third_party(self) -> bool:
        return is_third_party_panoid(self.id)

    def select_capture(
        self,
        target_year: int | None,
        max_year_diff: int | None = None,
        *,
        month_window: tuple[int, int] | None = None,
        month_policy: str = "exclude",
    ) -> PanoCapture | None:
        """Pick the capture closest to ``target_year``, or the most recent one.

        When ``max_year_diff`` is given, captures further than that many years
        from the target are filtered out first; if none survive the filter,
        returns ``None``. Ties resolve to the most recent capture.

        ``month_window = (first, last)`` (inclusive months; ``first > last``
        wraps over the new year) restricts the season: ``"exclude"`` drops
        captures outside it, and captures with no month, before the year rule;
        ``"prefer"`` ranks captures inside it first and falls back to the rest
        only when none is inside. With a window, captures of the same year are
        ranked by their distance from the window's middle month instead of by
        recency. ``target_year=None`` means the most recent capture. Without a
        window the choice is exactly the one made before windows existed.
        """
        caps = self.captures
        if not caps:
            # No temporal metadata was parsed. A hard window can't be
            # verified against an unknown date, so decline it; otherwise fall
            # back to this panorama itself.
            if max_year_diff is not None and target_year is not None:
                return None
            if self.date is not None:
                own = PanoCapture(self.id, self.date[0], self.date[1])
            else:
                own = PanoCapture(self.id, None, None)
            if month_window is None or month_policy == "prefer":
                return own
            return own if in_month_window(own.month, month_window) else None

        candidates = caps
        if max_year_diff is not None and target_year is not None:
            candidates = [
                c
                for c in caps
                if c.year is not None and abs(c.year - target_year) <= max_year_diff
            ]
        if not candidates:
            return None
        if month_window is None:
            if target_year is None:
                return max(candidates, key=lambda c: (c.year or 0, c.month or 0))
            return min(
                candidates,
                key=lambda c: (abs(c.year - target_year), -c.year, -(c.month or 0)),
            )

        inside = [c for c in candidates if in_month_window(c.month, month_window)]
        if inside or month_policy == "exclude":
            candidates = inside
        if not candidates:
            return None

        def season(c: PanoCapture) -> float:
            return month_distance(c.month, month_window)

        if target_year is None:
            return min(candidates, key=lambda c: (-(c.year or 0), season(c)))
        return min(
            candidates,
            key=lambda c: (
                abs(c.year - target_year) if c.year is not None else float("inf"),
                -(c.year or 0),
                season(c),
            ),
        )

    def clone_for_capture(self, capture: PanoCapture) -> StreetViewPanorama:
        """A downloadable panorama for ``capture`` reusing this pano's tiling.

        Street View panoramas share one tiling scheme (512-px tiles, fixed
        power-of-two image sizes per zoom), so a historical capture downloads
        through the same tile endpoint with the current pano's geometry.
        """
        return StreetViewPanorama(
            id=capture.id,
            lat=self.lat,
            lon=self.lon,
            tile_size=self.tile_size,
            image_sizes=self.image_sizes,
            date=(
                (capture.year, capture.month)
                if capture.year is not None and capture.month is not None
                else None
            ),
        )


def is_third_party_panoid(panoid: str) -> bool:
    """Distinguish user-uploaded panos from official Google coverage."""
    return panoid.startswith("CIHM0og") or len(panoid) > 22


# ── URL-encoded protobuf encoder (Google Maps' RPC format) ──────────


class _PbType(Enum):
    MESSAGE = "m"
    BOOL = "b"
    DOUBLE = "d"
    ENUM = "e"
    INT = "i"
    STRING = "s"


class ProtobufEnum:
    """Marker so an int can be distinguished from an enum value at encode time."""

    def __init__(self, value: int) -> None:
        self.value = value


def _datatype(value) -> _PbType:
    if isinstance(value, str):
        return _PbType.STRING
    if isinstance(value, bool):
        return _PbType.BOOL
    if isinstance(value, ProtobufEnum):
        return _PbType.ENUM
    if isinstance(value, int):
        return _PbType.INT
    if isinstance(value, float):
        return _PbType.DOUBLE
    if isinstance(value, Decimal):
        return _PbType.DOUBLE
    if isinstance(value, dict):
        return _PbType.MESSAGE
    raise NotImplementedError(value)


def _field_to_str(tag, value) -> tuple[int, str]:
    if isinstance(value, list):
        n, s = 0, ""
        for entry in value:
            ni, si = _field_to_str(tag, entry)
            n += ni
            s += si
        return n, s
    dt = _datatype(value)
    if dt is _PbType.MESSAGE:
        n, sub = _to_pb_url(value)
        return n + 1, f"!{tag}m{n}" + sub
    if dt is _PbType.BOOL:
        value = 1 if value else 0
    elif dt is _PbType.ENUM:
        value = value.value
    return 1, f"!{tag}{dt.value}{value}"


def _to_pb_url(fields: dict) -> tuple[int, str]:
    serialized = ""
    child_count = 0
    for tag, value in fields.items():
        n, s = _field_to_str(tag, value)
        serialized += s
        child_count += n
    return child_count, serialized


def _to_protobuf_url(fields: dict) -> str:
    return _to_pb_url(fields)[1]


# ── API: find the nearest panorama ──────────────────────────────────
#
# Two keyless Google Maps endpoints, as maps.google.com uses them: a coverage
# tile lists every panorama in a zoom-17 map tile (id and position), and the
# metadata request for one id returns its dates, historical captures and image
# sizes. The nearest panorama is chosen from the tiles around the point, so
# neighbouring points share one tile request.

_COVERAGE_ZOOM = 17
_COVERAGE_URL = (
    "https://www.google.com/maps/photometa/ac/v1?pb=!1m1!1smaps_sv.tactile"
    "!6m3!1i{x}!2i{y}!3i17!8b1"
)
_METADATA_URL = (
    "https://www.google.com/maps/photometa/v1?authuser=0&hl={lang}&gl={country}&pb="
)
_PHOTOMETA_HEADERS = {
    "Accept": "*/*",
    "Referer": "https://www.google.com/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:151.0) "
        "Gecko/20100101 Firefox/151.0"
    ),
}
_EARTH_RADIUS_M = 6_371_000.0
#: Image types in a coverage tile: Google's own panoramas and user photospheres.
_OFFICIAL, _THIRD_PARTY = 2, 10

# Tiles fetched by this process: ``(x, y) -> [(type, id, lat, lon), ...]``.
_TILES: dict[tuple[int, int], list[tuple[int, str, float, float]]] = {}
_TILES_MAX = 50_000
_TILE_TASKS: dict[tuple[int, int], asyncio.Task] = {}


def _tile_xy(lat: float, lon: float) -> tuple[int, int]:
    n = 2**_COVERAGE_ZOOM
    lat = max(min(lat, 85.0511), -85.0511)
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def _tiles_around(lat: float, lon: float, radius: float) -> list[tuple[int, int]]:
    """The coverage tiles a circle of ``radius`` metres can touch."""
    dlat = math.degrees(radius / _EARTH_RADIUS_M)
    dlon = dlat / max(math.cos(math.radians(lat)), 1e-6)
    x0, y0 = _tile_xy(lat + dlat, lon - dlon)
    x1, y1 = _tile_xy(lat - dlat, lon + dlon)
    return [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    h = (
        math.sin((p2 - p1) / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(h))


def _json_body(text: str, what: str):
    """The JSON after Google's ``)]}'`` guard. An answer without the guard
    raises :class:`UnexpectedAnswer`; guarded but unreadable JSON raises
    :class:`ServiceRetired`."""
    if not text.startswith(")]}'"):
        raise UnexpectedAnswer(what, text)
    try:
        return json.loads(text[4:])
    except json.JSONDecodeError as exc:
        raise ServiceRetired(f"the {what} answered unreadable JSON ({exc})") from None


def _parse_coverage_tile(text: str) -> list[tuple[int, str, float, float]]:
    """``(image type, panorama id, lat, lon)`` of every panorama in a tile."""
    tile = _json_body(text, "coverage tile service")
    found = []
    try:
        entries = tile[1][1] if tile and tile[1] else []
    except (IndexError, TypeError):
        raise ServiceRetired("the coverage tile service changed its answer") from None
    for entry in entries or []:
        try:
            kind, panoid = entry[0][0][0], entry[0][0][1]
            position = entry[0][2][0]
            lat, lon = float(position[2]), float(position[3])
            found.append((int(kind), str(panoid), lat, lon))
        except (IndexError, TypeError, ValueError):
            continue
    return found


def _nearest(
    tiles: list[list[tuple[int, str, float, float]]],
    lat: float,
    lon: float,
    radius: float,
    search_third_party: bool,
) -> str | None:
    wanted = _THIRD_PARTY if search_third_party else _OFFICIAL
    best, best_d = None, radius
    for tile in tiles:
        for kind, panoid, plat, plon in tile:
            if kind != wanted:
                continue
            d = _distance_m(lat, lon, plat, plon)
            if d <= best_d:
                best, best_d = panoid, d
    return best


def _remember(xy: tuple[int, int], panos: list) -> None:
    if len(_TILES) >= _TILES_MAX:
        _TILES.clear()
    _TILES[xy] = panos


def _metadata_url(panoid: str, locale: str) -> str:
    parts = locale.split("-")
    lang, country = parts[0], parts[1] if len(parts) > 1 else parts[0]
    pano_type = _THIRD_PARTY if is_third_party_panoid(panoid) else _OFFICIAL
    msg = {
        1: {1: "maps_sv.tactile", 11: {2: {1: True}}},
        2: {1: lang, 2: country},
        3: {1: {1: ProtobufEnum(pano_type), 2: panoid}},
        4: {
            1: [ProtobufEnum(i) for i in (1, 2, 3, 4, 5, 6, 8, 12)],
            2: {1: ProtobufEnum(1)},
            4: {1: 48},
            5: [{}],
            6: [{}],
            11: {3: {4: True}},
        },
    }
    return _METADATA_URL.format(lang=lang, country=country) + _to_protobuf_url(msg)


def _parse_metadata(text: str) -> StreetViewPanorama | None:
    """The panorama a metadata answer describes; ``None`` when Google does not
    know the id (status 2)."""
    response = _json_body(text, "panorama metadata service")
    try:
        status = response[1][0][0][0]
    except (IndexError, TypeError):
        message = "the panorama metadata service changed its answer"
        raise ServiceRetired(message) from None
    if status not in (1, 3):
        return None
    return _parse_pano_message(response[1][0])


def _parse_date(raw) -> tuple[int, int] | None:
    """Parse a ``[year, month, …]`` date list into ``(year, month)``."""
    try:
        return int(raw[0]), int(raw[1])
    except (TypeError, ValueError, IndexError):
        return None


def _parse_captures(
    msg, current_id: str, current_date: tuple[int, int] | None
) -> list[PanoCapture]:
    """Every capture at this location: the current pano plus historical passes.

    The current pano's date lives at ``msg[6][7]``; the historical timeline at
    ``msg[5][0][8]`` is a list of ``[index_into_pano_array, [year, month], …]``
    where the index points into the panorama array at ``msg[5][0][3][0]``.
    Returns captures newest-first, de-duplicated by panorama id.
    """
    captures: list[PanoCapture] = []
    if current_date is not None:
        captures.append(PanoCapture(current_id, current_date[0], current_date[1]))

    try:
        panos = msg[5][0][3][0]
        timeline = msg[5][0][8]
    except (IndexError, TypeError):
        panos = timeline = None

    if isinstance(timeline, list) and isinstance(panos, list):
        for entry in timeline:
            try:
                idx = entry[0]
                date = _parse_date(entry[1])
                pid = panos[idx][0][1]
            except (IndexError, TypeError):
                continue
            if date is None:
                continue
            captures.append(PanoCapture(pid, date[0], date[1]))

    seen: set[str] = set()
    unique: list[PanoCapture] = []
    for c in sorted(captures, key=lambda c: (c.year or 0, c.month or 0), reverse=True):
        if c.id in seen:
            continue
        seen.add(c.id)
        unique.append(c)
    return unique


def _parse_pano_message(msg) -> StreetViewPanorama:
    """Pull the minimum fields needed for tile download from the protobuf-as-list."""
    panoid = msg[1][1]
    img_sizes_raw = msg[2][3][0]
    image_sizes = [Size(x[0][1], x[0][0]) for x in img_sizes_raw]
    tile_size = Size(msg[2][3][1][0], msg[2][3][1][1])
    lat = msg[5][0][1][0][2]
    lon = msg[5][0][1][0][3]
    try:
        date = _parse_date(msg[6][7])
    except (IndexError, TypeError):
        date = None
    return StreetViewPanorama(
        id=panoid,
        lat=lat,
        lon=lon,
        tile_size=tile_size,
        image_sizes=image_sizes,
        date=date,
        captures=_parse_captures(msg, panoid, date),
    )


def find_panorama(
    lat: float,
    lon: float,
    radius: float = 50.0,
    locale: str = "en",
    search_third_party: bool = False,
    session: requests.Session | None = None,
    timeout: float | None = None,
) -> StreetViewPanorama | None:
    """The nearest Street View panorama within ``radius`` metres, or ``None``.

    Raises :class:`RateLimitedError` when Google throttles, so a caller can
    back off instead of mistaking the throttle for "no coverage",
    :class:`UnexpectedAnswer` when an endpoint answers a page instead of data,
    and :class:`ServiceRetired` when its data is no longer in the expected
    form.
    """
    requester = session if session is not None else requests

    def get(url: str) -> str:
        resp = requester.get(url, headers=_PHOTOMETA_HEADERS, timeout=timeout)
        if resp.status_code in _RATE_LIMIT_STATUSES:
            raise RateLimitedError(resp.status_code, url)
        return resp.text

    tiles = []
    for xy in _tiles_around(lat, lon, radius):
        if xy not in _TILES:
            text = get(_COVERAGE_URL.format(x=xy[0], y=xy[1]))
            _remember(xy, _parse_coverage_tile(text))
        tiles.append(_TILES[xy])
    panoid = _nearest(tiles, lat, lon, radius, search_third_party)
    if panoid is None:
        return None
    return _parse_metadata(get(_metadata_url(panoid, locale)))


async def _get_text(session: aiohttp.ClientSession, url: str) -> str:
    async with session.get(url, headers=_PHOTOMETA_HEADERS) as resp:
        if resp.status in _RATE_LIMIT_STATUSES:
            raise RateLimitedError(resp.status, url)
        return await resp.text()


async def _coverage_tile_async(
    session: aiohttp.ClientSession, xy: tuple[int, int]
) -> list[tuple[int, str, float, float]]:
    """One coverage tile, fetched once however many searches ask at a time."""
    if xy in _TILES:
        return _TILES[xy]
    loop = asyncio.get_running_loop()
    task = _TILE_TASKS.get(xy)
    if task is None or task.get_loop() is not loop:
        url = _COVERAGE_URL.format(x=xy[0], y=xy[1])

        async def fetch() -> list:
            try:
                panos = _parse_coverage_tile(await _get_text(session, url))
                _remember(xy, panos)
                return panos
            finally:
                _TILE_TASKS.pop(xy, None)

        task = _TILE_TASKS[xy] = loop.create_task(fetch())
    return await asyncio.shield(task)


async def find_panorama_async(
    lat: float,
    lon: float,
    session: aiohttp.ClientSession,
    radius: float = 50.0,
    locale: str = "en",
    search_third_party: bool = False,
) -> StreetViewPanorama | None:
    """Async variant of :func:`find_panorama`; concurrent searches share each
    coverage tile request."""
    tiles = [
        await _coverage_tile_async(session, xy)
        for xy in _tiles_around(lat, lon, radius)
    ]
    panoid = _nearest(tiles, lat, lon, radius, search_third_party)
    if panoid is None:
        return None
    return _parse_metadata(await _get_text(session, _metadata_url(panoid, locale)))


# ── API: download panorama image ────────────────────────────────────


_TILE_URL = (
    "https://streetviewpixels-pa.googleapis.com/v1/tile"
    "?cb_client=maps_sv.tactile&panoid={panoid}&x={x}&y={y}&zoom={zoom}"
)
_THIRD_PARTY_URL = "https://lh3.ggpht.com/jsapi2/a/b/c/w{w}-h{h}/{panoid}"

# Google's streetviewpixels-pa.googleapis.com tile endpoint now returns 403
# Forbidden for requests with a non-browser User-Agent (Python's default UA
# is blocked). Sending a plain Chrome UA restores 200 OK responses without
# any other auth.
_TILE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.google.com/maps",
}


def _validate_zoom(pano: StreetViewPanorama, zoom: int) -> int:
    if not pano.image_sizes:
        raise ValueError("pano.image_sizes is None.")
    return max(0, min(zoom, len(pano.image_sizes) - 1))


def _generate_tile_list(pano: StreetViewPanorama, zoom: int) -> list[Tile]:
    img_size = pano.image_sizes[zoom]
    cols = math.ceil(img_size.x / pano.tile_size.x)
    rows = math.ceil(img_size.y / pano.tile_size.y)
    return [
        Tile(x, y, _TILE_URL.format(panoid=pano.id, x=x, y=y, zoom=zoom))
        for x, y in itertools.product(range(cols), range(rows))
    ]


def _stitch_tiles(
    tile_data: dict, width: int, height: int, tile_w: int, tile_h: int
) -> Image.Image:
    panorama = Image.new("RGB", (width, height))
    for (x, y), data in tile_data.items():
        tile = Image.open(io.BytesIO(data))
        panorama.paste(im=tile, box=(x * tile_w, y * tile_h))
        del tile
    return panorama


def get_panorama(
    pano: StreetViewPanorama,
    zoom: int = 5,
    session: requests.Session | None = None,
) -> Image.Image:
    """Synchronously download every tile and return a stitched PIL image."""
    requester = session if session is not None else requests

    if pano.is_third_party:
        size = pano.image_sizes[_validate_zoom(pano, zoom)]
        url = _THIRD_PARTY_URL.format(w=size.x, h=size.y, panoid=pano.id)
        resp = requester.get(url, headers=_TILE_HEADERS)
        if resp.status_code in _RATE_LIMIT_STATUSES:
            raise RateLimitedError(resp.status_code, url)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content))

    zoom = _validate_zoom(pano, zoom)
    tile_list = _generate_tile_list(pano, zoom)

    tile_data: dict = {}
    for t in tile_list:
        resp = requester.get(t.url, headers=_TILE_HEADERS)
        if resp.status_code in _RATE_LIMIT_STATUSES:
            raise RateLimitedError(resp.status_code, t.url)
        resp.raise_for_status()
        tile_data[(t.x, t.y)] = resp.content

    return _stitch_tiles(
        tile_data,
        pano.image_sizes[zoom].x,
        pano.image_sizes[zoom].y,
        pano.tile_size.x,
        pano.tile_size.y,
    )


async def _fetch_tiles(pano: StreetViewPanorama, zoom: int, fetch) -> Image.Image:
    """Every tile of ``pano`` at ``zoom`` through ``fetch(tile)``, stitched."""
    tile_list = _generate_tile_list(pano, zoom)
    results = await asyncio.gather(*(fetch(t) for t in tile_list))
    tile_data = {(t.x, t.y): data for t, data in zip(tile_list, results, strict=True)}
    return _stitch_tiles(
        tile_data,
        pano.image_sizes[zoom].x,
        pano.image_sizes[zoom].y,
        pano.tile_size.x,
        pano.tile_size.y,
    )


async def get_panorama_async(
    pano: StreetViewPanorama,
    session: aiohttp.ClientSession,
    zoom: int = 5,
) -> Image.Image:
    """Async variant of :func:`get_panorama` — tiles fetch in parallel."""
    if pano.is_third_party:
        size = pano.image_sizes[_validate_zoom(pano, zoom)]
        url = _THIRD_PARTY_URL.format(w=size.x, h=size.y, panoid=pano.id)
        async with session.get(url, headers=_TILE_HEADERS) as resp:
            if resp.status in _RATE_LIMIT_STATUSES:
                raise RateLimitedError(resp.status, url)
            resp.raise_for_status()
            return Image.open(io.BytesIO(await resp.read()))

    zoom = _validate_zoom(pano, zoom)
    tile_list = _generate_tile_list(pano, zoom)

    async def _fetch(t: Tile) -> tuple[int, int, bytes]:
        async with session.get(t.url, headers=_TILE_HEADERS) as resp:
            if resp.status in _RATE_LIMIT_STATUSES:
                raise RateLimitedError(resp.status, t.url)
            resp.raise_for_status()
            return t.x, t.y, await resp.read()

    results = await asyncio.gather(*(_fetch(t) for t in tile_list))
    tile_data = {(x, y): data for x, y, data in results}

    return _stitch_tiles(
        tile_data,
        pano.image_sizes[zoom].x,
        pano.image_sizes[zoom].y,
        pano.tile_size.x,
        pano.tile_size.y,
    )


# ── API: the keyed Street View tiles (Map Tiles API) ────────────────
#
# Google's billed, documented route to the same equirectangular tiles. Every
# request carries a session token (created once per key and process, valid for
# about two weeks) and the key. The key only ever travels in query strings
# built here, and no error raised from this section quotes a URL, so a key
# never reaches a log or an error message. A search is one metadata request for
# the nearest panorama within the radius; the API knows only the current
# capture at a place, so ``captures`` holds that one capture.

_API_ROOT = "https://tile.googleapis.com/v1"
_API_SESSION = {"mapType": "streetview", "language": "en-US", "region": "US"}
#: Zoom levels of the tile API; the full-size image is at the last one.
_API_MAX_ZOOM = 5
# A session token is renewed when it has less than this long left.
_API_RENEW_S = 3600.0
_API_TOKENS: dict[str, tuple[str, float]] = {}
_API_TOKEN_TASKS: dict[str, asyncio.Task] = {}


class ApiRequestRefused(RuntimeError):
    """Google refused a Street View API request with this key: a wrong key, an
    API not enabled for the key's project, billing off, or a restriction.
    Retrying does not help, so a run stops with the reason."""

    def __init__(self, status: int, reason: str):
        super().__init__(
            f"Google refused the Street View API request (HTTP {status}: {reason}). "
            "Check the key, that the Map Tiles API is enabled for its Google Cloud "
            "project, and that billing is on."
        )
        self.status = status


def _api_reason(body) -> str:
    """Google's error message from an error answer, shortened."""
    try:
        return str(body["error"]["message"])[:200]
    except (KeyError, TypeError):
        return "no reason given"


async def _api_request(
    session: aiohttp.ClientSession,
    method: str,
    path: str,
    key: str,
    *,
    params: dict | None = None,
    body: dict | None = None,
    binary: bool = False,
):
    """One API request: the JSON (or bytes) answer, or ``None`` for 404.

    429 / 503 raise :class:`RateLimitedError`; other refusals raise
    :class:`ApiRequestRefused`; connection errors are re-raised as
    ``ConnectionError`` naming only the path.
    """
    query = {**(params or {}), "key": key}
    try:
        async with session.request(
            method, _API_ROOT + path, params=query, json=body
        ) as resp:
            if resp.status in (429, 503):
                raise RateLimitedError(resp.status, path)
            if resp.status == 404:
                return None
            if resp.status >= 400:
                try:
                    answer = await resp.json(content_type=None)
                except (ValueError, aiohttp.ContentTypeError):
                    answer = None
                raise ApiRequestRefused(resp.status, _api_reason(answer))
            if binary:
                return await resp.read()
            return await resp.json(content_type=None)
    except aiohttp.ClientError as exc:
        raise ConnectionError(
            f"Street View API request {path} failed: {type(exc).__name__}"
        ) from None


async def _api_token(session: aiohttp.ClientSession, key: str) -> str:
    """This process's session token for ``key``, created once and renewed
    before it expires; concurrent callers share one request."""
    held = _API_TOKENS.get(key)
    if held is not None and held[1] - time.time() > _API_RENEW_S:
        return held[0]
    loop = asyncio.get_running_loop()
    task = _API_TOKEN_TASKS.get(key)
    if task is None or task.get_loop() is not loop:

        async def create() -> str:
            try:
                answer = await _api_request(
                    session, "POST", "/createSession", key, body=_API_SESSION
                )
                if not answer or "session" not in answer:
                    raise ApiRequestRefused(404, "no session token in the answer")
                expiry = float(answer.get("expiry") or time.time() + 86400)
                _API_TOKENS[key] = (str(answer["session"]), expiry)
                return str(answer["session"])
            finally:
                _API_TOKEN_TASKS.pop(key, None)

        task = _API_TOKEN_TASKS[key] = loop.create_task(create())
    return await asyncio.shield(task)


def _parse_api_date(raw) -> tuple[int, int] | None:
    """``"2023-05"`` -> ``(2023, 5)``; a year alone or nothing -> ``None``."""
    parts = str(raw or "").split("-")
    try:
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return None


def _parse_api_metadata(answer: dict) -> StreetViewPanorama | None:
    """The panorama an API metadata answer describes; ``None`` for indoor
    imagery and user photospheres, which the keyless search skips too."""
    panoid = str(answer.get("panoId") or "")
    if not panoid or is_third_party_panoid(panoid):
        return None
    if str(answer.get("imageryType", "outdoor")).lower() == "indoor":
        return None
    width, height = int(answer["imageWidth"]), int(answer["imageHeight"])
    shifts = range(_API_MAX_ZOOM, -1, -1)
    date = _parse_api_date(answer.get("date"))
    year = date[0] if date else None
    if date is None and str(answer.get("date") or "").isdigit():
        year = int(answer["date"])
    return StreetViewPanorama(
        id=panoid,
        lat=float(answer["lat"]),
        lon=float(answer["lng"]),
        tile_size=Size(int(answer["tileWidth"]), int(answer["tileHeight"])),
        image_sizes=[Size(max(width >> s, 1), max(height >> s, 1)) for s in shifts],
        date=date,
        captures=[PanoCapture(panoid, year, date[1] if date else None)],
    )


async def find_panorama_api_async(
    lat: float,
    lon: float,
    session: aiohttp.ClientSession,
    key: str,
    radius: float = 50.0,
) -> StreetViewPanorama | None:
    """The nearest panorama within ``radius`` metres through the keyed API, or
    ``None``. Raises :class:`RateLimitedError` when throttled and
    :class:`ApiRequestRefused` when Google refuses the key."""
    token = await _api_token(session, key)
    params = {"session": token, "lat": f"{lat:.7f}", "lng": f"{lon:.7f}"}
    params["radius"] = str(int(round(radius)))
    path = "/streetview/metadata"
    answer = await _api_request(session, "GET", path, key, params=params)
    return _parse_api_metadata(answer) if answer else None


async def get_panorama_api_async(
    pano: StreetViewPanorama,
    session: aiohttp.ClientSession,
    key: str,
    zoom: int = 1,
) -> Image.Image:
    """Download ``pano`` at ``zoom`` through the keyed tile API, stitched."""
    token = await _api_token(session, key)
    zoom = _validate_zoom(pano, zoom)

    async def fetch(t: Tile) -> bytes:
        path = f"/streetview/tiles/{zoom}/{t.x}/{t.y}"
        params = {"session": token, "panoId": pano.id}
        data = await _api_request(session, "GET", path, key, params=params, binary=True)
        if data is None:
            raise FileNotFoundError(f"tile {zoom}/{t.x}/{t.y} of {pano.id[:12]}")
        return data

    return await _fetch_tiles(pano, zoom, fetch)
