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
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

import aiohttp
import requests
from PIL import Image

# ── Dataclasses ─────────────────────────────────────────────────────────────


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
class StreetViewPanorama:
    """Minimal Street View panorama metadata: ID, location, image grid."""

    id: str
    lat: float
    lon: float
    tile_size: Size
    image_sizes: list[Size]

    @property
    def is_third_party(self) -> bool:
        return is_third_party_panoid(self.id)


def is_third_party_panoid(panoid: str) -> bool:
    """Distinguish user-uploaded panos from official Google coverage."""
    return panoid.startswith("CIHM0og") or len(panoid) > 22


# ── URL-encoded protobuf encoder (Google Maps' RPC format) ──────────────────


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


# ── API: find panorama by radius (SingleImageSearch) ────────────────────────


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


def _parse_pano_message(msg) -> StreetViewPanorama:
    """Pull the minimum fields needed for tile download from the protobuf-as-list."""
    panoid = msg[1][1]
    img_sizes_raw = msg[2][3][0]
    image_sizes = [Size(x[0][1], x[0][0]) for x in img_sizes_raw]
    tile_size = Size(msg[2][3][1][0], msg[2][3][1][1])
    lat = msg[5][0][1][0][2]
    lon = msg[5][0][1][0][3]
    return StreetViewPanorama(
        id=panoid,
        lat=lat,
        lon=lon,
        tile_size=tile_size,
        image_sizes=image_sizes,
    )


def find_panorama(
    lat: float,
    lon: float,
    radius: float = 50.0,
    locale: str = "en",
    search_third_party: bool = False,
    session: requests.Session | None = None,
) -> StreetViewPanorama | None:
    """Search for the nearest Street View panorama within ``radius`` metres."""
    url = _build_find_panorama_url(lat, lon, radius, locale, search_third_party)
    requester = session if session is not None else requests
    resp = requester.get(url)
    return _parse_radius_response(json.loads(_repair_jsonp(resp.text)))


async def find_panorama_async(
    lat: float,
    lon: float,
    session: aiohttp.ClientSession,
    radius: float = 50.0,
    locale: str = "en",
    search_third_party: bool = False,
) -> StreetViewPanorama | None:
    """Async variant of :func:`find_panorama`."""
    url = _build_find_panorama_url(lat, lon, radius, locale, search_third_party)
    async with session.get(url) as resp:
        text = await resp.text()
    return _parse_radius_response(json.loads(_repair_jsonp(text)))


# ── API: download panorama image ────────────────────────────────────────────


_TILE_URL = (
    "https://streetviewpixels-pa.googleapis.com/v1/tile"
    "?cb_client=maps_sv.tactile&panoid={panoid}&x={x}&y={y}&zoom={zoom}"
)
_THIRD_PARTY_URL = "https://lh3.ggpht.com/jsapi2/a/b/c/w{w}-h{h}/{panoid}"


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
        resp = requester.get(url)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content))

    zoom = _validate_zoom(pano, zoom)
    tile_list = _generate_tile_list(pano, zoom)

    tile_data: dict = {}
    for t in tile_list:
        resp = requester.get(t.url)
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
        async with session.get(url) as resp:
            resp.raise_for_status()
            return Image.open(io.BytesIO(await resp.read()))

    zoom = _validate_zoom(pano, zoom)
    tile_list = _generate_tile_list(pano, zoom)

    async def _fetch(t: Tile) -> tuple[int, int, bytes]:
        async with session.get(t.url) as resp:
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


def download_panorama(
    pano: StreetViewPanorama,
    path: str,
    zoom: int = 5,
    pil_args: dict | None = None,
    session: requests.Session | None = None,
) -> None:
    """Synchronously download a panorama and save it to ``path``."""
    image = get_panorama(pano, zoom=zoom, session=session)
    image.save(path, **(pil_args or {}))


async def download_panorama_async(
    pano: StreetViewPanorama,
    path: str,
    session: aiohttp.ClientSession,
    zoom: int = 5,
    pil_args: dict | None = None,
) -> None:
    """Async variant of :func:`download_panorama`."""
    image = await get_panorama_async(pano, session, zoom=zoom)
    image.save(path, **(pil_args or {}))
