"""
Minimal Google Street View client for GeoFuse.

Adapted from `streetlevel <https://github.com/sk-zk/streetlevel>`_ (MIT
License) — only the parts needed for radius-based panorama search and
equirectangular image download.

Why a custom port rather than ``pip install streetlevel``: the upstream
package pulls in ``pyfrpc``, ``bd09convertor``, ``CoordinatesConverter``,
``pyequilib``, ``pyexiv2``, and ``pycryptodome`` — several of which require
a Visual C++ build chain on Windows.  None of those are needed for Google
Street View; they are used by the Mapy.cz, Baidu, Apple Look Around, and
EXIF-writing modules.

Public surface
--------------
``find_panorama`` / ``find_panorama_async``
    Search for the nearest panorama within a radius, returning a
    :class:`StreetViewPanorama` or ``None``.

``get_panorama`` / ``get_panorama_async``
    Download tiles in parallel and return a stitched PIL image.

``download_panorama`` / ``download_panorama_async``
    Same as ``get_panorama`` but writes directly to a JPEG file.

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
        self, target_year: int, max_year_diff: int | None = None
    ) -> PanoCapture | None:
        """Pick the capture closest to ``target_year``.

        When ``max_year_diff`` is given, captures further than that many years
        from the target are filtered out first; if none survive the filter,
        returns ``None``. Ties resolve to the most recent capture.
        """
        caps = self.captures
        if not caps:
            # No temporal metadata was parsed. A hard window can't be
            # verified against an unknown date, so decline it; otherwise fall
            # back to this panorama itself.
            if max_year_diff is not None:
                return None
            if self.date is not None:
                return PanoCapture(self.id, self.date[0], self.date[1])
            return PanoCapture(self.id, None, None)

        candidates = caps
        if max_year_diff is not None:
            candidates = [
                c
                for c in caps
                if c.year is not None and abs(c.year - target_year) <= max_year_diff
            ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda c: (abs(c.year - target_year), -c.year, -(c.month or 0)),
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


# ── API: find panorama by radius (SingleImageSearch) ────────────────


def _build_find_panorama_url(
    lat: float, lon: float, radius: float, locale: str, search_third_party: bool
) -> str:
    parts = locale.split("-")
    ietf_lang = parts[0]
    ietf_country = parts[1] if len(parts) > 1 else parts[0]
    image_type = 10 if search_third_party else 2

    toggles = [ProtobufEnum(i) for i in (1, 2, 3, 4, 6, 8, 12)]

    msg = {
        1: {1: "apiv3", 5: "US", 11: {1: {1: False}}},
        2: {1: {3: float(lat), 4: float(lon)}, 2: float(radius)},
        3: {
            2: {1: ietf_lang, 2: ietf_country},
            9: {1: ProtobufEnum(2)},
            11: {1: {1: ProtobufEnum(image_type), 2: True, 3: ProtobufEnum(2)}},
        },
        4: {1: toggles, 5: {}, 6: {}},
    }
    return (
        "https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch?pb="
        + _to_protobuf_url(msg)
        + "&callback=_xdc_._v2mub5"
    )


def _repair_jsonp(text: str) -> str:
    """The endpoint wraps its JSON in ``_xdc_._v2mub5(…)`` JSONP — strip it."""
    try:
        first = text.index("(")
        last = text.rindex(")")
        return "[" + text[first + 1 : last] + "]"
    except ValueError:
        return "[]"


def _parse_radius_response(response: list) -> StreetViewPanorama | None:
    """Return the parsed panorama, or ``None`` if the API reported no result."""
    try:
        if response[0][0][0] != 0:
            return None
        return _parse_pano_message(response[0][1])
    except (IndexError, TypeError, KeyError):
        return None


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
) -> StreetViewPanorama | None:
    """Search for the nearest Street View panorama within ``radius`` metres.

    Raises :class:`RateLimitedError` when the endpoint throttles, so a caller
    can back off instead of mistaking the throttle body for "no coverage".
    """
    url = _build_find_panorama_url(lat, lon, radius, locale, search_third_party)
    requester = session if session is not None else requests
    resp = requester.get(url, headers=_TILE_HEADERS)
    if resp.status_code in _RATE_LIMIT_STATUSES:
        raise RateLimitedError(resp.status_code, url)
    return _parse_radius_response(json.loads(_repair_jsonp(resp.text)))


async def find_panorama_async(
    lat: float,
    lon: float,
    session: aiohttp.ClientSession,
    radius: float = 50.0,
    locale: str = "en",
    search_third_party: bool = False,
) -> StreetViewPanorama | None:
    """Async variant of :func:`find_panorama`.

    Raises :class:`RateLimitedError` on throttle statuses — see
    :func:`find_panorama` for why that distinction matters.
    """
    url = _build_find_panorama_url(lat, lon, radius, locale, search_third_party)
    async with session.get(url, headers=_TILE_HEADERS) as resp:
        if resp.status in _RATE_LIMIT_STATUSES:
            raise RateLimitedError(resp.status, url)
        text = await resp.text()
    return _parse_radius_response(json.loads(_repair_jsonp(text)))


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
