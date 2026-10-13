from __future__ import annotations

import hashlib
import json
import logging
import os
import traceback
import zipfile
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import ee
import geopandas as gpd
import numpy as np
import rasterio
import requests
from shapely.geometry import box

from .core import build_planar_tiles
from .crs_utils import (
    default_geotiff_creation_options,
    raster_geographic_bounds,
    reproject_geodataframe_to_wgs84,
    select_grid_crs_with_warning,
    stream_mosaic_to_geotiff,
)
from .ee_auth import initialize as initialize_earth_engine
from .ee_utils import (
    crs_to_ee_string,
    shapely_to_ee_geometry,
    shrink_gdf_for_ee,
)
from .jobs import NDVI_PROGRESS_MIN_TILES, ProgressThrottle, retry_with_backoff
from .logger import attach_external_logger, get_logger
from .ndvi_chains import (
    WATER_MASK,
    as_statistic,
    check_chains,
    focal_radius,
    output_stems,
    statistic_chain,
    temporal_statistic,
)
from .persistence.caches import DEFAULT_MAX_BYTES, NdviTileCache, ndvi_tiles
from .vector_io import geometry_sha256

_log = get_logger("NDVI")

# TODO: NDVI_TEMPORAL - Add time-series analysis for seasonal greenery changes

# Concurrent tile-download workers. Earth Engine allows several parallel
# ``ee_export_image`` calls per user well above this; the cap exists to avoid
# saturating local sockets and to keep the post-download rasterio decode +
# reproject step from contending too heavily for the GIL on a small machine.
# Mirrors the GVI engine's ``_MAX_CONCURRENT_POINTS = 4`` choice.
_MAX_CONCURRENT_TILES = 4

# How many times a single export tile may be halved (2×2) when Earth Engine
# rejects it as too large to compute in one request. Depth 3 turns one tile
# into at most 64 sub-tiles (each ~1/8 the linear size); past that the region
# is almost certainly failing for a non-size reason and we surface it.
_MAX_TILE_SUBDIVISION_DEPTH = 3

# Substrings that mark an Earth Engine rejection as "this request was too big
# to compute in one shot" — halving the export region fixes these, unlike a
# transient network failure which a plain retry handles. Matched case-folded.
_EE_TOO_LARGE_MARKERS = (
    "user memory limit exceeded",
    "output of image computation is too large",
    "total request size",
    "request payload size exceeds",
    "computed value is too large",
)


# ────────────────────────────────────────────────────────────────────
# Temporal statistics
# ────────────────────────────────────────────────────────────────────

#: Statistics a composite can take, per pixel, over the cloud-free images in
#: its date window. ``median`` is the default and keeps the historical file
#: names and resume keys; every other statistic adds its name to both.
TEMPORAL_REDUCERS = ("median", "mean", "max")


def normalize_temporal_reducers(
    value: str | Iterable[str] = "median",
) -> tuple[str, ...]:
    """Validated statistics in :data:`TEMPORAL_REDUCERS` order, deduplicated."""
    names = [value] if isinstance(value, str) else list(value)
    chosen = {str(v).strip().lower() for v in names}
    unknown = sorted(chosen - set(TEMPORAL_REDUCERS))
    if unknown:
        raise ValueError(
            f"Unknown temporal statistic(s) {unknown}; "
            f"choose from {list(TEMPORAL_REDUCERS)}."
        )
    if not chosen:
        raise ValueError("Choose at least one temporal statistic.")
    return tuple(r for r in TEMPORAL_REDUCERS if r in chosen)


def ndvi_output_name(output_name: str, reducer: str) -> str:
    """Base name of one statistic's files: ``median`` keeps ``output_name``,
    any other statistic appends ``_<reducer>`` (``{name}_max_ndvi.tif``)."""
    return output_stems(output_name, [reducer])[reducer]


# ────────────────────────────────────────────────────────────────────
# Acquisition options
# ────────────────────────────────────────────────────────────────────

#: ``auto`` picks Sentinel-2 or Landsat by date; ``landsat`` merges Landsat 5,
#: 7, 8 and 9; ``landsat_no_slc_off`` keeps Landsat 5, Landsat 7 before its
#: scan-line corrector failed, and Landsat 8.
SATELLITES = ("auto", "sentinel2", "landsat", "landsat_no_slc_off")

#: ``scene_filter`` (default) drops scenes above the cloud threshold and masks
#: cloudy pixels in the rest; ``pixel_only`` keeps every scene and masks cloudy
#: pixels; ``scene_only`` drops cloudy scenes and keeps every pixel of the rest.
CLOUD_HANDLINGS = ("scene_filter", "pixel_only", "scene_only")

#: ``surface`` (default) reads Level-2 surface reflectance; ``toa`` reads
#: top-of-atmosphere reflectance (Landsat C2 TOA, Sentinel-2 Level-1C).
REFLECTANCES = ("surface", "toa")

#: Water mask source and threshold: pixels covered by open water at least
#: this percent of the time are masked when the water mask is on.
WATER_DATASET = "JRC/GSW1_4/GlobalSurfaceWater"
WATER_OCCURRENCE_MAX = 50

#: How every prep computes NDVI from red and near-infrared reflectance.
NDVI_FORMULA = "normalized_difference (negative inputs masked)"


def _normalized_difference(nir, red):
    """``(nir − red) / (nir + red)`` as band ``NDVI``, masked where either band
    is negative: Earth Engine's ``normalizedDifference`` rule, which keeps
    every value in [−1, 1]."""
    valid = nir.gte(0).And(red.gte(0))
    return nir.subtract(red).divide(nir.add(red)).updateMask(valid).rename("NDVI")


def _shift_day(day: str, days: int) -> str:
    """The ISO date ``days`` days after ``day`` (before it when negative)."""
    from datetime import date as _date
    from datetime import timedelta as _timedelta

    return (_date.fromisoformat(str(day)[:10]) + _timedelta(days=days)).isoformat()


def _window_text(start_date: str, end_date: str) -> str:
    """``start → last day`` of the half-open window ``[start_date, end_date)``."""
    return f"{start_date} → {_shift_day(end_date, -1)}"


def substitute_year(
    start_date: str, end_date: str, *, year: int = 2012, to: int = 2013
) -> tuple[str, str, bool]:
    """Shift a window that lies wholly inside ``year`` to the same days in ``to``.

    The window is half-open, ``[start_date, end_date)``, as Earth Engine reads
    it, so an end on 1 January of the next year still lies inside ``year``.
    Returns ``(start, end, shifted)``. A window only partly inside ``year`` is
    returned unchanged with ``shifted=False``; the caller warns about it. A
    29 February becomes the 28th.
    """
    from datetime import date as _date

    s = _date.fromisoformat(str(start_date))
    last = _date.fromisoformat(_shift_day(end_date, -1))
    if s.year != year or last.year != year:
        return str(start_date), str(end_date), False

    def move(d):
        try:
            return d.replace(year=to)
        except ValueError:
            return d.replace(year=to, day=28)

    return move(s).isoformat(), _shift_day(move(last).isoformat(), 1), True


def _overlaps_year(start_date: str, end_date: str, year: int) -> bool:
    """Whether the half-open window ``[start_date, end_date)`` reaches ``year``."""
    return str(start_date)[:4] <= str(year) <= _shift_day(end_date, -1)[:4]


def _stack_composites(composites: Mapping[str, Any], reducers: list[str]):
    """One image carrying ``reducers`` as bands, in order, for a single export."""
    if len(reducers) == 1:
        return composites[reducers[0]]
    return ee.Image.cat([composites[r].rename(r) for r in reducers])


def _export_bands(
    composites: Mapping[str, Any],
    names: list[str],
    path: str,
    export: Callable[[Any, str, bool], None],
    *,
    label: str,
) -> tuple[list[tuple[list[str], str]], bool]:
    """Export ``names`` as the bands of one Earth Engine request into ``path``.

    When Earth Engine refuses the request as too large, the names are halved
    into two requests, each halved again while refused, down to one output
    per request; ``export(image, path, alone)`` gets ``alone=True`` for a
    single output and may then subdivide the region instead. Returns every
    file written with the names it holds, in order, and whether the request
    was split. A failure leaves no files behind.
    """
    try:
        export(_stack_composites(composites, names), path, len(names) == 1)
        return [(list(names), path)], False
    except _EEComputeTooLargeError as too_large:
        if len(names) == 1:
            raise
        _log(
            "INFO",
            f"{label}: {too_large} — splitting {len(names)} outputs across "
            "two requests.",
        )
    half = len(names) // 2
    stem = path[:-4] if path.endswith(".tif") else path
    files: list[tuple[list[str], str]] = []
    try:
        for i, part in enumerate((names[:half], names[half:])):
            got, _ = _export_bands(
                composites, part, f"{stem}.{i}.tif", export, label=label
            )
            files.extend(got)
    except Exception:
        for _names, written in files:
            try:
                os.remove(written)
            except OSError:
                pass
        raise
    return files, True


def _combine_reducer_results(results: Mapping[str, dict]) -> dict:
    """The first statistic's result, with every statistic's under ``reducers``.

    ``status`` stays ``"success"`` only when every statistic succeeded; a
    cancellation wins over a failure, and a failure's message names the
    statistics that failed.
    """
    reducers = list(results)
    combined = dict(results[reducers[0]])
    combined["reducers"] = {r: dict(results[r]) for r in reducers}
    bad = [r for r in reducers if results[r].get("status") != "success"]
    if not bad:
        return combined
    if any(results[r].get("status") == "cancelled" for r in bad):
        combined["status"] = "cancelled"
        combined["message"] = "Cancelled by user"
    elif len(reducers) > 1:
        combined["status"] = "error"
        combined["message"] = "; ".join(
            f"{r}: {results[r].get('message', 'failed')}" for r in bad
        )
    return combined


class _EEComputeTooLargeError(RuntimeError):
    """Earth Engine rejected an export for exceeding a per-request compute
    budget (memory or output size). Signals the caller to subdivide the
    export region rather than retry the identical request."""


def _classify_ee_error(message: str) -> RuntimeError:
    """Wrap an Earth Engine error string in the right exception type.

    Returns an :class:`_EEComputeTooLargeError` when the message names a
    size/memory limit (subdivision helps), otherwise a plain ``RuntimeError``
    (transient — the retry path handles it).
    """
    lowered = message.casefold()
    if any(marker in lowered for marker in _EE_TOO_LARGE_MARKERS):
        return _EEComputeTooLargeError(message)
    return RuntimeError(message)


def _describe_ee_download_error(resp: requests.Response) -> str:
    """Pull Earth Engine's error message out of a failed download response.

    EE returns its failures as a JSON body of the shape
    ``{"error": {"code": .., "message": ".."}}``. Fall back to the raw
    (truncated) response text when the body isn't the expected JSON.
    """
    try:
        body = resp.json()
    except ValueError:
        text = (resp.text or "").strip()
        return text[:300]
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
    return str(body)[:300]


def _export_ee_image_to_tif(
    ee_image,
    filename: str,
    *,
    crs: str,
    crs_transform: list[float],
    region,
    timeout: int = 300,
) -> None:
    """Download a single-band ``ee.Image`` to ``filename`` as a GeoTIFF.

    A diagnostics-preserving stand-in for ``geemap.ee_export_image``. geemap
    catches every failure behind a generic ``"An error occurred while
    downloading."`` print and returns without writing the file, so a caller
    can't tell a rate-limit (HTTP 429) from an over-large request, an Earth
    Engine memory-limit rejection, or a local disk error. This runs the same
    ``ZIPPED_GEO_TIFF`` fetch-and-unzip but **raises** a message carrying the
    HTTP status and Earth Engine's own error text, so the caller's retry /
    logging path records exactly what went wrong.
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    params = {
        "name": name,
        "filePerBand": False,
        "region": region,
        "crs": crs,
        "crs_transform": crs_transform,
    }
    try:
        url = ee_image.getDownloadURL(params)
    except Exception as e:  # noqa: BLE001 — surface the EE-side rejection verbatim
        raise _classify_ee_error(f"Earth Engine getDownloadURL failed: {e}") from e

    resp = requests.get(url, stream=True, timeout=timeout)
    if resp.status_code != 200:
        detail = _describe_ee_download_error(resp)
        raise _classify_ee_error(
            f"Earth Engine pixel download failed with HTTP {resp.status_code}"
            + (f": {detail}" if detail else "")
        )

    filename_zip = (
        filename[:-4] + ".zip" if filename.endswith(".tif") else filename + ".zip"
    )
    try:
        with open(filename_zip, "wb") as fd:
            for chunk in resp.iter_content(chunk_size=1024):
                fd.write(chunk)
        with zipfile.ZipFile(filename_zip) as z:
            tif_members = [m for m in z.namelist() if m.lower().endswith(".tif")]
            if not tif_members:
                raise RuntimeError(
                    "Earth Engine returned a zip with no GeoTIFF member "
                    f"(members: {z.namelist()})."
                )
            z.extract(tif_members[0], os.path.dirname(filename))
        extracted = os.path.join(os.path.dirname(filename), tif_members[0])
        if os.path.abspath(extracted) != os.path.abspath(filename):
            os.replace(extracted, filename)
    finally:
        try:
            os.remove(filename_zip)
        except OSError:
            pass

    if not (os.path.isfile(filename) and os.path.getsize(filename) > 0):
        raise RuntimeError(f"Download completed but {filename} is empty or missing.")


def _split_geom_quadrants(region_geom):
    """Split a region into up to four bbox quadrants clipped to its footprint.

    Halves the geometry's bounding box on both axes and intersects each of the
    four cells with ``region_geom`` so the union of the returned pieces covers
    exactly the original footprint — no extra area, no gaps. Empty cells (the
    original geometry didn't reach that corner) are dropped. Returns ``[]`` for
    a degenerate (zero-width or zero-height) region, signalling the caller that
    no further subdivision is possible.
    """
    minx, miny, maxx, maxy = region_geom.bounds
    if maxx <= minx or maxy <= miny:
        return []
    midx = (minx + maxx) / 2.0
    midy = (miny + maxy) / 2.0
    cells = (
        box(minx, miny, midx, midy),
        box(midx, miny, maxx, midy),
        box(minx, midy, midx, maxy),
        box(midx, midy, maxx, maxy),
    )
    pieces = []
    for cell in cells:
        piece = region_geom.intersection(cell)
        if not piece.is_empty and piece.area > 0:
            pieces.append(piece)
    return pieces


def _compute_resume_key(
    geom_wgs84_gdf: gpd.GeoDataFrame,
    start_date: str,
    end_date: str,
    cloud_max: float,
    resolution: int,
    ee_collection: str,
    temporal_reducer: str = "median",
    variant: tuple[str, ...] = (),
) -> str:
    """Stable 16-hex-char key identifying a unique NDVI run.

    Two runs that hash to the same key produce identical output, so they can
    share an on-disk tile workspace — that's what lets resume work. The key
    includes everything that affects the per-tile contents: NDVI formula,
    study area (geometry + CRS), date range, cloud threshold, export
    resolution, Earth Engine collection ID and temporal statistic. Changing
    any one of these spawns a fresh workspace and forces a full re-download
    (which is what you want — old tiles would be from a different question).
    ``median`` adds nothing to the hash. ``variant`` names acquisition
    options set away from their defaults (water mask, cloud handling); the
    defaults add nothing either.
    """
    h = hashlib.sha256()
    # Schema tag — bump when the on-disk tile format changes. ``planar_v1``
    # marks the post-Step-13.5 layout (planar-CRS tiles snapped to a global
    # grid). Older WGS84-tile cache entries hash to different keys and are
    # naturally orphaned (LRU-evicted as space pressure builds).
    h.update(b"planar_v1|")
    # Formula tag: tiles computed with another NDVI formula never share a key.
    h.update(b"nd_masked|")
    h.update(geometry_sha256(geom_wgs84_gdf).encode())
    h.update(f"|{start_date}|{end_date}|{cloud_max}|{resolution}|".encode())
    h.update(ee_collection.encode())
    if temporal_reducer != "median":
        h.update(f"|{temporal_reducer}".encode())
    for option in variant:
        h.update(f"|{option}".encode())
    return h.hexdigest()[:16]


def _chain_key(
    steps: list[str], cloud_variant: tuple[str, ...]
) -> tuple[str, tuple[str, ...]]:
    """``(temporal_reducer, variant)`` of an output's resume key.

    A statistic's own chain keys as that statistic always has, its per-scene
    water mask a variant, so its cached tiles stay valid; any other chain
    keys by its steps.
    """
    water = f"water{WATER_OCCURRENCE_MAX}"
    plain = as_statistic(steps)
    if plain is not None:
        statistic, masked = plain
        return statistic, ((water,) if masked else ()) + cloud_variant
    token = ",".join(water if step == WATER_MASK else step for step in steps)
    return f"chain:{token}", cloud_variant


def _extract_band(
    src_path: str, band: int, dst_path: str, *, dtype: str | None = None
) -> None:
    """Write band ``band`` of ``src_path`` to its own compressed GeoTIFF.

    ``dtype`` defaults to the source's. Cache tiles are stored as float32:
    every mosaic built from them is float32, so outputs match a mosaic of
    Earth Engine's own float64 files exactly, and with DEFLATE plus the
    floating-point predictor a tile takes under half the bytes. The file is
    written to ``.part`` and renamed, so a tile on disk is always whole.
    """
    with rasterio.open(src_path) as src:
        out_dtype = dtype or src.dtypes[band - 1]
        data = src.read(band, out_dtype=out_dtype)
        profile = {
            "driver": "GTiff",
            "height": src.height,
            "width": src.width,
            "count": 1,
            "dtype": out_dtype,
            "crs": src.crs,
            "transform": src.transform,
            "nodata": src.nodata,
        }
    profile.update(default_geotiff_creation_options(out_dtype))
    part = dst_path + ".part"
    try:
        with rasterio.open(part, "w", **profile) as dst:
            dst.write(data, 1)
        os.replace(part, dst_path)
    finally:
        if os.path.exists(part):
            os.remove(part)


def _discover_resumable_tiles(
    work_dir: str, tiles: list[dict]
) -> tuple[list[str], list[dict]]:
    """Split ``tiles`` into (already-on-disk, still-pending).

    A tile counts as "already on disk" iff ``tile_<idx>.tif`` exists in
    ``work_dir`` and opens cleanly. Half-written / corrupt files are
    discarded so the next run re-downloads them.
    """
    existing_files: list[str] = []
    pending: list[dict] = []
    for ts in tiles:
        idx = int(ts["tile_idx"])
        path = os.path.join(work_dir, f"tile_{idx}.tif")
        if not os.path.isfile(path):
            pending.append(ts)
            continue
        try:
            with rasterio.open(path) as _src:
                _ = _src.bounds  # touch metadata to detect truncation
            existing_files.append(path)
        except Exception:
            try:
                os.remove(path)
            except OSError:
                pass
            pending.append(ts)
    return existing_files, pending


def _emit_ndvi_progress(
    ndvi_progress_callback: Callable[[Mapping[str, Any]], None] | None,
    *,
    sub_progress: float,
    phase: str | None = None,
    tiles: tuple[int, int] | None = None,
    clear_bracket: bool = False,
) -> None:
    if not ndvi_progress_callback:
        return
    payload: dict[str, Any] = {"sub_progress": float(sub_progress)}
    if phase is not None:
        payload["phase"] = phase
    if tiles is not None:
        payload["tiles"] = tiles
    if clear_bracket:
        payload["clear_bracket"] = True
    ndvi_progress_callback(payload)


def _write_ndvi_sidecar(
    folder: str,
    output_name: str,
    *,
    grid_crs,
    export_crs: str,
    export_crs_name: str,
    export_distortion: float,
    n_clusters: int,
    tiles_total: int,
    tiles_succeeded: int,
    tiles_failed: int = 0,
    tiles_resumed: int = 0,
    tiles_split_across_requests: int = 0,
    failed_tile_refs: list[dict] | None = None,
    resume_key: str = "",
    start_date: str,
    end_date: str,
    end_date_exclusive: str | None = None,
    used_start_date: str | None = None,
    used_end_date: str | None = None,
    temporal_reducer: str = "median",
    chain: list[str] | None = None,
    chain_name: str | None = None,
    coverage_widened: bool = False,
    year_substituted: str | None = None,
    water_mask: bool = False,
    cloud_handling: str = "scene_filter",
    reflectance: str = "surface",
    preset: str | None = None,
    cloud_max: float,
    resolution_m: int,
    max_tile_size_km: float,
    ee_collection: str,
    satellite: str = "sentinel2",
    bands: list[str] | None = None,
    n_cloud_filtered_images: int | None = None,
    n_images: int | None = None,
    n_images_after_cloud_filter: int | None = None,
) -> str:
    """Write ``{output_name}_ndvi.json`` capturing the parameters and grid CRS.

    Mirrors GVI's ``_gvi.json`` so any downstream tool (fusion, custom
    notebooks) can introspect an NDVI raster after the fact: which planar CRS
    pixels were rasterised in, how much distortion that introduced, how many
    clusters and tiles the input decomposed into, the exact date range, the
    temporal statistic (median / mean / max over the window's cloud-free
    images), the aggregation chain that made the file (``chain``, with
    ``chain_name`` when it was one of the run's chains; a statistic's own
    chain otherwise), the NDVI formula (:data:`NDVI_FORMULA`), and which
    Earth Engine ImageCollection was queried.
    ``start_date`` / ``end_date`` are the dates as given and
    ``end_date_exclusive`` the exclusive end of that window; ``used_*`` is
    the half-open window queried after any 2012 fill or coverage rescue.
    ``n_images`` counts the scenes over the study area in that window, and
    ``n_images_after_cloud_filter`` those left by the scene-level cloud filter
    (``None`` when every scene is kept). ``tiles_split_across_requests``
    counts the tiles Earth Engine took in several requests because one was
    too large. ``failed_tile_refs`` lists the cluster/tile IDs that exhausted
    their retry budget — those areas appear as NaN gaps in the mosaic.
    """
    sidecar_path = os.path.join(folder, f"{output_name}_ndvi.json")
    payload = {
        "export_crs": export_crs,
        "export_crs_name": export_crs_name,
        "export_crs_wkt": grid_crs.to_wkt() if grid_crs is not None else None,
        "distortion": float(export_distortion),
        "n_clusters": int(n_clusters),
        "tiles_total": int(tiles_total),
        "tiles_succeeded": int(tiles_succeeded),
        "tiles_failed": int(tiles_failed),
        "tiles_resumed": int(tiles_resumed),
        "tiles_split_across_requests": int(tiles_split_across_requests),
        "resume_key": resume_key,
        "failed_tile_refs": list(failed_tile_refs or []),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "end_date_exclusive": str(end_date_exclusive or end_date),
        "used_start_date": str(used_start_date or start_date),
        "used_end_date": str(used_end_date or end_date),
        "temporal_reducer": str(temporal_reducer),
        "chain": list(chain or statistic_chain(temporal_reducer, water_mask)),
        "chain_name": chain_name,
        "ndvi_formula": NDVI_FORMULA,
        "coverage_widened": bool(coverage_widened),
        "year_substituted": year_substituted,
        "water_mask": bool(water_mask),
        "water_occurrence_max": WATER_OCCURRENCE_MAX if water_mask else None,
        "cloud_handling": str(cloud_handling),
        "reflectance": str(reflectance),
        "cloud_max": float(cloud_max),
        "resolution_m": int(resolution_m),
        "max_tile_size_km": float(max_tile_size_km),
        "ee_collection": ee_collection,
        "satellite": satellite,
        "bands": list(bands or ["NDVI"]),
        "n_cloud_filtered_images": (
            int(n_cloud_filtered_images)
            if n_cloud_filtered_images is not None
            else None
        ),
        "n_images": int(n_images) if n_images is not None else None,
        "n_images_after_cloud_filter": (
            int(n_images_after_cloud_filter)
            if n_images_after_cloud_filter is not None
            else None
        ),
    }
    if preset:
        payload["preset"] = str(preset)
    with open(sidecar_path, "w") as f:
        json.dump(payload, f, indent=2)
    return sidecar_path


def _select_export_crs(geom_wgs84_gdf: gpd.GeoDataFrame):
    """Pick an EE export CRS for true-ground-metre pixels at any latitude.

    Thin NDVI-side adapter on top of :func:`select_grid_crs_with_warning`
    that also formats the CRS for Earth Engine. Returns
    ``(ee_string, crs_obj, distortion, choice_name)``; ``crs_obj`` is the
    pyproj CRS so the cluster-aware tile builder can reproject geometries
    without re-running the selector.
    """
    crs, distortion, choice_name = select_grid_crs_with_warning(
        geom_wgs84_gdf, _log, role="Export CRS"
    )
    return crs_to_ee_string(crs), crs, distortion, choice_name


class NDVIEngine:
    #: Default Earth Engine ImageCollection. ``download_and_process``
    #: substitutes a Landsat collection when ``satellite="landsat"`` or
    #: ``satellite="auto"`` resolves to Landsat (pre-Sentinel-2 dates).
    EE_COLLECTION_ID = "COPERNICUS/S2_SR_HARMONIZED"

    #: First date covered by ``COPERNICUS/S2_SR_HARMONIZED``. Anything
    #: earlier comes back empty; ``auto`` satellite mode picks Landsat for
    #: ranges that end before this. Diagnostics also reference it.
    S2_COLLECTION_START = "2017-03-28"

    #: Landsat Collection 2 Tier 1 Level-2 sources for the pre-Sentinel-2
    #: fallback, spanning every Landsat era so any historical date range finds
    #: a sensor: LT05 (TM) 1984 → 2012-05, LE07 (ETM+) 1999 → present, LC08
    #: (OLI) 2013-04+, LC09 2021-10+. They are merged era-aware at query time
    #: (see :meth:`get_collection_landsat`) — Earth Engine keeps only the
    #: sensors whose acquisitions overlap the requested window.
    LANDSAT5_COLLECTION_ID = "LANDSAT/LT05/C02/T1_L2"
    LANDSAT7_COLLECTION_ID = "LANDSAT/LE07/C02/T1_L2"
    LANDSAT8_COLLECTION_ID = "LANDSAT/LC08/C02/T1_L2"
    LANDSAT9_COLLECTION_ID = "LANDSAT/LC09/C02/T1_L2"

    #: Stable label for the merged Landsat source, recorded in the sidecar
    #: metadata and mixed into the tile-cache resume key.
    LANDSAT_MERGED_LABEL = "LANDSAT/LT05+LE07+LC08+LC09/C02/T1_L2"

    #: Top-of-atmosphere counterparts, Landsat 5 / 7 / 8 / 9, Collection 2
    #: Tier 1, and Sentinel-2 Level-1C.
    LANDSAT_TOA_COLLECTION_IDS = (
        "LANDSAT/LT05/C02/T1_TOA",
        "LANDSAT/LE07/C02/T1_TOA",
        "LANDSAT/LC08/C02/T1_TOA",
        "LANDSAT/LC09/C02/T1_TOA",
    )
    S2_TOA_COLLECTION_ID = "COPERNICUS/S2_HARMONIZED"

    #: First day of Landsat 7's scan-line-corrector failure; the
    #: ``landsat_no_slc_off`` sensor set keeps Landsat 7 scenes before it.
    LANDSAT7_SLC_OFF_START = "2003-06-01"

    #: Roy et al. (2016) RMA coefficients that put ETM+ surface reflectance on
    #: the OLI (Landsat 8) scale as ``OLI ≈ slope · ETM+ + intercept``, so
    #: pre-2013 TM/ETM+ NDVI is comparable to the LC08/LC09 (and, loosely,
    #: Sentinel-2) years. RMA (not OLS) is the harmonization-appropriate fit —
    #: it is symmetric and preserves the direction (OLI NDVI reads slightly
    #: higher than ETM+ over vegetation). Only the red/NIR bands NDVI uses are
    #: kept; ``(slope, intercept)`` in reflectance units. Applied to Landsat 5
    #: TM as well — spectrally near-identical to ETM+, with no separate
    #: published TM→OLI set.
    _ETM_TO_OLI = {"red": (0.9825, -0.0022), "nir": (1.0073, -0.0021)}

    def __init__(
        self,
        project_id=None,
        *,
        tile_cache: NdviTileCache | None = None,
        tile_cache_dir: str | None = None,
        tile_cache_max_bytes: int = DEFAULT_MAX_BYTES,
    ):
        """Initialize Earth Engine and open the persistent tile cache.

        Earth Engine is initialised by :func:`geofuse.ee_auth.initialize`: the
        stored service-account key or sign-in when there is one, else Earth
        Engine's own default; ``project_id`` falls back to the user
        configuration's ``earthengine.project``.

        ``tile_cache`` lets the caller inject a shared :class:`NdviTileCache`
        (e.g. a singleton from ``ui/services.py``). When omitted, the engine
        opens its own at ``tile_cache_dir`` (default ``ndvi_tiles/`` in the
        cache folder, :func:`~geofuse.persistence.caches.cache_root`) so
        direct API callers — fusion
        auto-download, CLI, notebooks — benefit from the same cross-run
        cache the UI uses. The cache is keyed by ``resume_key`` so repeat
        runs over the same area + date range are near-instant.

        Earth Engine's stdlib ``logging`` output is captured into the bound
        per-job log here (with propagation to the root logger disabled),
        keeping the Streamlit host terminal free of EE chatter.
        """
        # Route the Earth Engine stdlib logger (and its descendants) into
        # the per-job pipeline before EE itself starts talking. Idempotent
        # across engine instances and processes.
        attach_external_logger("ee", logging.INFO)

        source = initialize_earth_engine(project_id)
        _log("INFO", f"Earth Engine credentials: {source}")

        if tile_cache is not None:
            self.tile_cache = tile_cache
        else:
            root = tile_cache_dir if tile_cache_dir is not None else ndvi_tiles()
            self.tile_cache = NdviTileCache(root, max_bytes=tile_cache_max_bytes)

    @staticmethod
    def compute_export_crs(gdf: gpd.GeoDataFrame):
        """Select the export CRS for a geometry once, for reuse across runs.

        Returns the same ``(ee_string, crs_obj, distortion, choice_name)``
        tuple :meth:`download_and_process` computes internally. Pass it back in
        via ``crs_override`` so several downloads (e.g. per-year slices) share
        one CRS and snap to the same global pixel grid.
        """
        geom_wgs84 = reproject_geodataframe_to_wgs84(gdf)
        return _select_export_crs(geom_wgs84)

    def prep_ndvi(self, img, pixel_mask: bool = True):
        """NDVI from Sentinel-2 Level-2A surface reflectance (``B4`` / ``B8``).

        ``pixel_mask`` removes pixels the scene classification (``SCL``) marks
        as cloud shadow, unclassified, cloud, cirrus or snow; ``False`` keeps
        every pixel of the scene.
        """
        scale = 0.0001
        red = img.select("B4").multiply(scale)
        nir = img.select("B8").multiply(scale)
        ndvi = _normalized_difference(nir, red)
        if not pixel_mask:
            return ndvi.copyProperties(img, img.propertyNames())

        # SCL masking (Standard in Harmonized collection)
        scl = img.select("SCL")
        mask_scl = (
            scl.neq(3)
            .And(scl.neq(7))
            .And(scl.neq(8))
            .And(scl.neq(9))
            .And(scl.neq(10))
            .And(scl.neq(11))
        )
        return ndvi.updateMask(mask_scl).copyProperties(img, img.propertyNames())

    def prep_ndvi_s2_toa(self, img, pixel_mask: bool = True):
        """NDVI from Sentinel-2 Level-1C top-of-atmosphere reflectance.

        Same bands and scale as :meth:`prep_ndvi`; Level-1C has no scene
        classification, so ``pixel_mask`` removes the opaque-cloud and cirrus
        pixels flagged in ``QA60`` (bits 10 and 11).
        """
        scale = 0.0001
        red = img.select("B4").multiply(scale)
        nir = img.select("B8").multiply(scale)
        ndvi = _normalized_difference(nir, red)
        if not pixel_mask:
            return ndvi.copyProperties(img, img.propertyNames())
        qa = img.select("QA60")
        cloudy = qa.bitwiseAnd(1 << 10).Or(qa.bitwiseAnd(1 << 11))
        return ndvi.updateMask(cloudy.eq(0)).copyProperties(img, img.propertyNames())

    def get_collection(
        self,
        aoi,
        start_date,
        end_date,
        cloud_max=10,
        *,
        reflectance: str = "surface",
        pixel_mask: bool = True,
    ):
        """Cloud-screened Sentinel-2 NDVI collection.

        ``cloud_max=None`` keeps every scene (no scene-level cloud filter);
        ``reflectance="toa"`` reads Level-1C instead of Level-2A.
        """
        if reflectance == "toa":
            cid, prep = self.S2_TOA_COLLECTION_ID, self.prep_ndvi_s2_toa
        else:
            cid, prep = self.EE_COLLECTION_ID, self.prep_ndvi
        col = ee.ImageCollection(cid).filterBounds(aoi).filterDate(start_date, end_date)
        if cloud_max is not None:
            col = col.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_max))
        return col.map(prep if pixel_mask else lambda img: prep(img, pixel_mask=False))

    @staticmethod
    def _qa_pixel_clear(img):
        """1 where ``QA_PIXEL`` flags no cloud (bit 3), shadow (4) or snow (5)."""
        qa = img.select("QA_PIXEL")
        cloudy = (
            qa.bitwiseAnd(1 << 3).Or(qa.bitwiseAnd(1 << 4)).Or(qa.bitwiseAnd(1 << 5))
        )
        return cloudy.eq(0)

    def prep_ndvi_landsat(self, img, pixel_mask: bool = True):
        """NDVI from Landsat 8/9 Collection 2 Level-2 surface reflectance.

        Bands: ``SR_B4`` (red) / ``SR_B5`` (NIR). Apply the C2 L2 scale +
        offset (``DN * 0.0000275 - 0.2``) before computing the index, and
        mask cloud / cloud-shadow via the ``QA_PIXEL`` band so cloudy
        observations don't pollute the composite. Same output band name
        (``NDVI``) as :meth:`prep_ndvi` so downstream code is collection-
        agnostic. ``pixel_mask=False`` keeps every pixel.
        """
        scale = 0.0000275
        offset = -0.2
        red = img.select("SR_B4").multiply(scale).add(offset)
        nir = img.select("SR_B5").multiply(scale).add(offset)
        ndvi = _normalized_difference(nir, red)
        if not pixel_mask:
            return ndvi.copyProperties(img, img.propertyNames())
        return ndvi.updateMask(self._qa_pixel_clear(img)).copyProperties(
            img, img.propertyNames()
        )

    def prep_ndvi_landsat_tm(self, img, pixel_mask: bool = True):
        """NDVI from Landsat 5 TM / 7 ETM+ Collection 2 Level-2, harmonized to
        the OLI scale.

        TM/ETM+ carry red/NIR as ``SR_B3`` / ``SR_B4`` (vs OLI's ``SR_B4`` /
        ``SR_B5``). Reflectance is C2 L2-scaled (``DN * 0.0000275 - 0.2``),
        then Roy et al. (2016) coefficients (:data:`_ETM_TO_OLI`) put it on the
        Landsat 8 scale so the NDVI lines up with the LC08/LC09 years. Cloud /
        shadow / snow are masked with the same ``QA_PIXEL`` bitmask and the
        output band name (``NDVI``) matches :meth:`prep_ndvi_landsat`, so
        downstream code stays collection-agnostic.
        """
        scale = 0.0000275
        offset = -0.2
        red = img.select("SR_B3").multiply(scale).add(offset)
        nir = img.select("SR_B4").multiply(scale).add(offset)
        rs, ri = self._ETM_TO_OLI["red"]
        ns, ni = self._ETM_TO_OLI["nir"]
        red = red.multiply(rs).add(ri)
        nir = nir.multiply(ns).add(ni)
        ndvi = _normalized_difference(nir, red)
        if not pixel_mask:
            return ndvi.copyProperties(img, img.propertyNames())
        return ndvi.updateMask(self._qa_pixel_clear(img)).copyProperties(
            img, img.propertyNames()
        )

    def prep_ndvi_landsat_toa(self, img, pixel_mask: bool = True):
        """NDVI from Landsat 8/9 Collection 2 top-of-atmosphere reflectance.

        TOA collections carry reflectance directly (no scale or offset): red
        ``B4``, NIR ``B5``. No harmonisation (the published coefficients are
        for surface reflectance). Same ``QA_PIXEL`` mask as the SR preps.
        """
        red = img.select("B4")
        nir = img.select("B5")
        ndvi = _normalized_difference(nir, red)
        if not pixel_mask:
            return ndvi.copyProperties(img, img.propertyNames())
        return ndvi.updateMask(self._qa_pixel_clear(img)).copyProperties(
            img, img.propertyNames()
        )

    def prep_ndvi_landsat_toa_tm(self, img, pixel_mask: bool = True):
        """NDVI from Landsat 5 TM / 7 ETM+ top-of-atmosphere reflectance.

        Red ``B3``, NIR ``B4``, not harmonised to OLI; same ``QA_PIXEL`` mask.
        """
        red = img.select("B3")
        nir = img.select("B4")
        ndvi = _normalized_difference(nir, red)
        if not pixel_mask:
            return ndvi.copyProperties(img, img.propertyNames())
        return ndvi.updateMask(self._qa_pixel_clear(img)).copyProperties(
            img, img.propertyNames()
        )

    def _landsat_sensors(self, sensor_set: str = "all", reflectance: str = "surface"):
        """``(collection_id, prep_fn, pre_slc_only)`` per Landsat sensor, old → new.

        ``all`` merges LT05, LE07, LC08 and LC09. ``no_slc_off`` keeps LT05,
        LE07 acquired before its scan-line corrector failed
        (:data:`LANDSAT7_SLC_OFF_START`) and LC08: no striped Landsat 7 scenes
        and no Landsat 9. Surface reflectance runs TM/ETM+ through the
        harmonised :meth:`prep_ndvi_landsat_tm`; top-of-atmosphere is not
        harmonised.
        """
        if reflectance == "toa":
            tm, oli = self.prep_ndvi_landsat_toa_tm, self.prep_ndvi_landsat_toa
            ids = self.LANDSAT_TOA_COLLECTION_IDS
        else:
            tm, oli = self.prep_ndvi_landsat_tm, self.prep_ndvi_landsat
            ids = (
                self.LANDSAT5_COLLECTION_ID,
                self.LANDSAT7_COLLECTION_ID,
                self.LANDSAT8_COLLECTION_ID,
                self.LANDSAT9_COLLECTION_ID,
            )
        pre_slc = sensor_set == "no_slc_off"
        sensors = [
            (ids[0], tm, False),
            (ids[1], tm, pre_slc),
            (ids[2], oli, False),
            (ids[3], oli, False),
        ]
        return sensors[:3] if pre_slc else sensors

    def _landsat_label(
        self, sensor_set: str = "all", reflectance: str = "surface"
    ) -> str:
        """Stable label of a merged Landsat source (sidecar, resume key)."""
        if sensor_set == "all" and reflectance == "surface":
            return self.LANDSAT_MERGED_LABEL
        sensors = (
            f"LT05+LE07(pre-{self.LANDSAT7_SLC_OFF_START[:7]})+LC08"
            if sensor_set == "no_slc_off"
            else "LT05+LE07+LC08+LC09"
        )
        product = "T1_TOA" if reflectance == "toa" else "T1_L2"
        return f"LANDSAT/{sensors}/C02/{product}"

    def _landsat_source(self, cid, aoi, start_date, end_date, pre_slc: bool):
        col = ee.ImageCollection(cid).filterBounds(aoi).filterDate(start_date, end_date)
        if pre_slc:
            col = col.filterDate("1999-01-01", self.LANDSAT7_SLC_OFF_START)
        return col

    def get_collection_landsat(
        self,
        aoi,
        start_date,
        end_date,
        cloud_max=10,
        *,
        sensor_set: str = "all",
        reflectance: str = "surface",
        pixel_mask: bool = True,
    ):
        """Era-aware merged Landsat NDVI collection.

        Each sensor is filtered, cloud-screened, and mapped to a common
        ``NDVI`` band *before* merging — TM/ETM+ and OLI have different band
        layouts, so they can only be combined once normalized. Earth Engine
        keeps only the sensors whose acquisitions fall in the date range, so a
        1990s query composes from TM, a 2012 query from ETM+, and a 2015 query
        from OLI, all through the same call. Same output shape as
        :meth:`get_collection` for Sentinel-2. ``cloud_max=None`` keeps every
        scene; ``sensor_set`` / ``reflectance`` follow :meth:`_landsat_sensors`.
        """
        merged = None
        for cid, prep, pre_slc in self._landsat_sensors(sensor_set, reflectance):
            col = self._landsat_source(cid, aoi, start_date, end_date, pre_slc)
            if cloud_max is not None:
                col = col.filter(ee.Filter.lt("CLOUD_COVER", cloud_max))
            col = col.map(
                prep if pixel_mask else lambda img, p=prep: p(img, pixel_mask=False)
            )
            merged = col if merged is None else merged.merge(col)
        return merged

    def _landsat_base(
        self,
        aoi,
        start_date,
        end_date,
        *,
        sensor_set: str = "all",
        reflectance: str = "surface",
    ):
        """Raw (un-prepped) merged Landsat collection over the date range.

        Used only to *count* acquisitions when distinguishing "no images in
        range" from "all images cloud-filtered", so heterogeneous bands across
        sensors are fine — nothing reads pixels here.
        """
        merged = None
        for cid, _prep, pre_slc in self._landsat_sensors(sensor_set, reflectance):
            col = self._landsat_source(cid, aoi, start_date, end_date, pre_slc)
            merged = col if merged is None else merged.merge(col)
        return merged

    def _mask_water(self, img):
        """Mask pixels open water covers at least :data:`WATER_OCCURRENCE_MAX`
        percent of the time (``occurrence`` of :data:`WATER_DATASET`); pixels
        with no water record count as dry and are kept."""
        occurrence = ee.Image(WATER_DATASET).select("occurrence").unmask(0)
        return img.updateMask(occurrence.lt(WATER_OCCURRENCE_MAX))

    def _apply_steps(self, img, steps: list[str]):
        """``img`` (band ``NDVI``) after the chain operations ``steps``: water
        masks and neighbourhood statistics, in order."""
        for step in steps:
            if step == WATER_MASK:
                img = self._mask_water(img)
                continue
            _, statistic, radius = step.split(":")
            img = img.reduceNeighborhood(
                reducer=getattr(ee.Reducer, statistic)(),
                kernel=ee.Kernel.circle(float(radius), "meters"),
            ).rename("NDVI")
        return img

    def _chain_composite(self, col, steps: list[str]):
        """The composite ``steps`` make of the prepped scenes in ``col``: the
        operations before the temporal step act on every scene, those after
        it on the composite. Neighbourhoods are computed on the export grid,
        the projection every download requests."""
        t = next(i for i, step in enumerate(steps) if step.startswith("temporal:"))
        before, after = steps[:t], steps[t + 1 :]
        if before:
            col = col.map(lambda img: self._apply_steps(img, before))
        composite = getattr(col.select("NDVI"), temporal_statistic(steps))()
        return self._apply_steps(composite.rename("NDVI"), after)

    def _pick_satellite(self, start_date: str, end_date: str, satellite: str) -> str:
        """Resolve ``satellite='auto'`` → ``'sentinel2'`` or ``'landsat'``.

        A window whose exclusive ``end_date`` is on or before the Sentinel-2
        collection start picks Landsat; everything else stays on Sentinel-2
        (higher resolution + more frequent revisit). Every explicit choice in
        :data:`SATELLITES` is passed through untouched.
        """
        if satellite != "auto":
            return satellite
        return "landsat" if str(end_date) <= self.S2_COLLECTION_START else "sentinel2"

    #: Block-iteration buffer cap. Points accumulate here across raster
    #: windows; once we cross this threshold we build a GeoDataFrame and
    #: append it to the GeoPackage, then reset. Keeps peak memory bounded
    #: while batching enough features that GPKG open/append overhead doesn't
    #: dominate
    _VECTOR_FLUSH_POINTS = 500_000

    def _raster_to_ndvi_points(
        self,
        final_tif: str,
        geometry,
        *,
        geojson_path: str | None,
        gpkg_path: str | None,
        ndvi_progress_callback: Callable[[Mapping[str, Any]], None] | None,
        cancel_callback: Callable[[], bool] | None,
        sub_start: float,
        sub_end: float,
        meta_extra: dict[str, Any] | None = None,
    ) -> dict:
        """Stream raster blocks → points in the raster's native CRS.

        Walks the raster's native block grid (typically 256×256 tiles for
        our outputs) instead of slurping the whole band into RAM. Each
        valid-pixel chunk gets buffered until the buffer hits
        ``_VECTOR_FLUSH_POINTS``, then a small GeoDataFrame is built,
        clipped to the study geometry, and appended to the GeoPackage.

        The output GeoPackage carries the raster's native CRS (planar
        UTM/LCC for our pipeline) so its X/Y columns are already in metres
        on the engine's pixel grid. GeoJSON output (when requested) is
        materialised at the end from the GeoPackage and reprojected to
        EPSG:4326 — the only output format that leaves the planar CRS.
        """
        if cancel_callback and cancel_callback():
            return {"status": "cancelled", "message": "Cancelled by user"}

        _emit_ndvi_progress(
            ndvi_progress_callback,
            sub_progress=sub_start,
            phase="Writing vector samples",
        )

        layer_name = "ndvi_samples"
        if gpkg_path:
            stream_target = gpkg_path
            stream_target_is_temp = False
        elif geojson_path:
            stream_target = geojson_path + ".tmp.gpkg"
            stream_target_is_temp = True
        else:
            try:
                with rasterio.open(final_tif) as src:
                    crs_str = str(src.crs) if src.crs is not None else None
            except Exception:
                crs_str = None
            meta = {"crs": crs_str}
            if meta_extra:
                meta.update(meta_extra)
            return {
                "status": "success",
                "tif": final_tif,
                "geojson": None,
                "gpkg": None,
                "meta": meta,
            }

        buf_x: list[float] = []
        buf_y: list[float] = []
        buf_v: list[float] = []
        n_written = 0
        first_chunk = True

        try:
            with rasterio.open(final_tif) as src:
                raster_crs = src.crs
                nodata = src.nodata if src.nodata is not None else -9999
                blocks = list(src.block_windows(1))
                n_blocks = max(1, len(blocks))

                # Pre-compute the clip predicate in the raster's CRS so every
                # chunk filter is a single shapely call rather than a per-
                # chunk reprojection.
                clip_geom = None
                if isinstance(geometry, gpd.GeoDataFrame) and not geometry.empty:
                    clip_gdf = (
                        geometry
                        if raster_crs is not None and geometry.crs == raster_crs
                        else geometry.to_crs(raster_crs)
                    )
                    try:
                        clip_geom = clip_gdf.geometry.union_all()
                        if clip_geom is not None and clip_geom.is_empty:
                            clip_geom = None
                    except Exception:
                        clip_geom = None

                def flush() -> None:
                    nonlocal first_chunk, n_written
                    if not buf_v:
                        return
                    gdf_chunk = gpd.GeoDataFrame(
                        {"NDVI": np.round(np.asarray(buf_v, dtype=np.float32), 4)},
                        geometry=gpd.points_from_xy(buf_x, buf_y),
                        crs=raster_crs,
                    )
                    if clip_geom is not None:
                        gdf_chunk = gdf_chunk[gdf_chunk.geometry.intersects(clip_geom)]
                    if not gdf_chunk.empty:
                        mode = "w" if first_chunk else "a"
                        gdf_chunk.to_file(
                            stream_target,
                            driver="GPKG",
                            layer=layer_name,
                            mode=mode,
                        )
                        first_chunk = False
                        n_written += len(gdf_chunk)
                    buf_x.clear()
                    buf_y.clear()
                    buf_v.clear()

                span = max(sub_end - sub_start, 0.0)
                for bi, (_block_idx, window) in enumerate(blocks):
                    if cancel_callback and cancel_callback():
                        return {"status": "cancelled", "message": "Cancelled by user"}

                    arr = src.read(1, window=window)
                    # Treat both NaN and the EE nodata sentinel as missing.
                    if np.issubdtype(arr.dtype, np.floating):
                        valid = (arr != nodata) & ~np.isnan(arr)
                    else:
                        valid = arr != nodata
                    if not valid.any():
                        _emit_ndvi_progress(
                            ndvi_progress_callback,
                            sub_progress=sub_start + span * ((bi + 1) / n_blocks),
                            phase="Writing vector samples",
                        )
                        continue

                    rows_local, cols_local = np.nonzero(valid)
                    rows_global = rows_local + int(window.row_off)
                    cols_global = cols_local + int(window.col_off)
                    xs, ys = rasterio.transform.xy(
                        src.transform,
                        rows_global.tolist(),
                        cols_global.tolist(),
                        offset="center",
                    )
                    buf_x.extend(np.atleast_1d(np.asarray(xs)).ravel().tolist())
                    buf_y.extend(np.atleast_1d(np.asarray(ys)).ravel().tolist())
                    buf_v.extend(arr[valid].tolist())

                    if len(buf_v) >= self._VECTOR_FLUSH_POINTS:
                        flush()

                    _emit_ndvi_progress(
                        ndvi_progress_callback,
                        sub_progress=sub_start + span * ((bi + 1) / n_blocks),
                        phase="Writing vector samples",
                    )

                flush()

            if n_written == 0:
                # Clean up the empty temp GPKG before reporting empty raster.
                if stream_target_is_temp and os.path.exists(stream_target):
                    try:
                        os.remove(stream_target)
                    except OSError:
                        pass
                return {"status": "error", "message": "Raster is empty."}

            # GeoJSON is the one compatibility format that must ship in
            # EPSG:4326 — stream into the planar GPKG first, then dump.
            if geojson_path:
                try:
                    gdf_full = gpd.read_file(stream_target, layer=layer_name)
                    reproject_geodataframe_to_wgs84(gdf_full).to_file(
                        geojson_path, driver="GeoJSON"
                    )
                except Exception as e:
                    _log("WARN", f"GeoJSON conversion failed: {e}")
            if stream_target_is_temp and os.path.exists(stream_target):
                try:
                    os.remove(stream_target)
                except OSError:
                    pass

            _emit_ndvi_progress(
                ndvi_progress_callback,
                sub_progress=sub_end,
                phase="Writing vector samples",
            )

            meta: dict[str, Any] = {
                "crs": str(raster_crs) if raster_crs is not None else None,
                "vector_points": n_written,
            }
            if meta_extra:
                meta.update(meta_extra)
            return {
                "status": "success",
                "tif": final_tif,
                "geojson": geojson_path,
                "gpkg": gpkg_path,
                "meta": meta,
            }

        except Exception as e:
            return {"status": "error", "message": f"Pixel Extraction Failed: {str(e)}"}

    def download_and_process(
        self,
        geometry,
        start_date,
        end_date,
        output_name,
        cloud_max=10,
        resolution=10,
        folder="output_results",
        max_tile_size_km=5,
        cancel_callback: Callable[[], bool] | None = None,
        ndvi_progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        write_geotiff: bool = True,
        write_geojson: bool = True,
        write_geopackage: bool = False,
        write_cluster_tiles: bool = False,
        satellite: str = "auto",
        coverage_rescue: bool = True,
        crs_override=None,
        temporal_reducers: str | Iterable[str] = "median",
        water_mask: bool = False,
        cloud_handling: str = "scene_filter",
        reflectance: str = "surface",
        substitute_2012: bool = False,
        preset: str | None = None,
        end_inclusive: bool = False,
        chains: Iterable[Mapping[str, Any]] | None = None,
    ):
        """
        Download and process NDVI data with automatic tiling for large areas.

        Args:
            geometry: GeoDataFrame or Shapely geometry
            start_date: Start date (YYYY-MM-DD), included.
            end_date: End date (YYYY-MM-DD). Exclusive by default, as Earth
                Engine's ``filterDate`` reads it: the window is
                ``[start_date, end_date)``. See ``end_inclusive``.
            output_name: Base name for output files
            cloud_max: Maximum cloud percentage
            resolution: Pixel resolution in meters
            folder: Output directory
            max_tile_size_km: Maximum tile dimension in kilometers (prevents GEE size limit)
            cancel_callback: If provided, return True to stop between tiles / major steps.
            ndvi_progress_callback: Receives a mapping with optional keys:
                ``sub_progress`` (0–1 for this download), ``phase`` (status line),
                ``tiles`` ``(k, n)`` while downloading tiles (shows ``[k/n]`` on the bar),
                ``clear_bracket`` (drop tile bracket after tiles are done).
            write_geotiff: Persist ``{output_name}_ndvi.tif`` when True.
            write_geojson: Persist ``{output_name}_ndvi.geojson`` when True (needs raster).
            write_geopackage: Persist ``{output_name}_ndvi.gpkg`` when True (needs raster).
            write_cluster_tiles: For multi-cluster inputs, mosaic each cluster
                into its own ``{output_name}_ndvi_tiles/cluster_NNNN.tif`` plus
                a ``tiles_index.json``. Mirrors GVI's per-cluster GeoTIFF tile
                output and avoids the giant mostly-NaN mosaic that scattered
                national-scale inputs would otherwise produce.
            satellite: one of :data:`SATELLITES`. ``'auto'`` (default)
                picks Landsat for date ranges that end before
                :attr:`S2_COLLECTION_START` and Sentinel-2 otherwise;
                ``'landsat'`` merges Landsat 5 / 7 / 8 / 9;
                ``'landsat_no_slc_off'`` keeps Landsat 5, Landsat 7 before
                :attr:`LANDSAT7_SLC_OFF_START` and Landsat 8.
            coverage_rescue: When the cloud-filtered collection has fewer
                than 3 images, widen the date range by ±50 % once and retry.
                Documented in the sidecar so the user knows the composite
                spans a wider window than they originally asked for.
            temporal_reducers: Per-pixel statistic(s) over the window's
                cloud-free images, any of :data:`TEMPORAL_REDUCERS` (default
                ``"median"``). Each statistic writes its own files under
                :func:`ndvi_output_name` and caches its tiles under its own
                resume key; one Earth Engine request per tile carries every
                statistic that tile is missing. Not read when ``chains`` are
                given.
            water_mask: Mask pixels open water covers at least
                :data:`WATER_OCCURRENCE_MAX` percent of the time, on every
                image before the temporal statistic (or before every chain).
                Off by default (water stays in the composite).
            cloud_handling: one of :data:`CLOUD_HANDLINGS`: drop scenes above
                ``cloud_max`` and mask cloudy pixels (default), keep every
                scene and mask cloudy pixels, or drop cloudy scenes and keep
                every pixel of the rest.
            reflectance: ``'surface'`` (default, Level-2) or ``'toa'``
                (top-of-atmosphere: Landsat C2 TOA, Sentinel-2 Level-1C; no
                TM / ETM+ harmonisation).
            substitute_2012: Fill a window that lies wholly inside 2012 with
                the same days of 2013 (recorded as ``year_substituted``).
            preset: Name of the settings preset the run came from, recorded
                in the sidecar for provenance (nothing recorded when ``None``).
            end_inclusive: Treat ``end_date`` as the last day to include; the
                day after it is sent to Earth Engine. The sidecar records both
                (``end_date``, ``end_date_exclusive``), and the 2012 fill and
                coverage rescue work on the corrected window.
            chains: Aggregation chains (:mod:`geofuse.ndvi_chains`), each a
                ``{"name", "steps"}`` table; they replace
                ``temporal_reducers``. Each chain writes its own files,
                ``{output_name}_{name}_ndvi.*``, and caches its tiles under
                its own resume key (a statistic's own chain shares that
                statistic's); one request per tile carries every chain.

        Returns:
            The first output's result (``status``, ``tif``, ``geojson``,
            ``gpkg``, ``meta``, ``sidecar``) with ``reducers`` mapping every
            output (statistic, or chain name) to its own result. ``status``
            is ``"success"`` only when every output succeeded.
        """
        reducers = normalize_temporal_reducers(temporal_reducers)
        user_chains = check_chains(chains)
        # Each output's steps: a chain as given, after the per-scene water
        # mask when that is on, or a statistic's own chain.
        if user_chains:
            chain_steps = {
                c["name"]: (
                    [WATER_MASK, *c["steps"]]
                    if water_mask and c["steps"][0] != WATER_MASK
                    else list(c["steps"])
                )
                for c in user_chains
            }
            output_names = output_stems(output_name, chains=user_chains)
        else:
            chain_steps = {r: statistic_chain(r, water_mask) for r in reducers}
            output_names = output_stems(output_name, reducers)
        for name, value, allowed in (
            ("satellite", satellite, SATELLITES),
            ("cloud_handling", cloud_handling, CLOUD_HANDLINGS),
            ("reflectance", reflectance, REFLECTANCES),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, not {value!r}.")
        os.makedirs(folder, exist_ok=True)

        if not (
            write_geotiff or write_geojson or write_geopackage or write_cluster_tiles
        ):
            return {
                "status": "error",
                "message": "Enable at least one output format (GeoTIFF, GeoPackage, GeoJSON, or cluster tiles).",
            }

        # 1. Convert Geometry — reproject_geodataframe_to_wgs84 raises on
        # malformed inputs (geographic CRS metadata + metre-valued coords) so
        # direct API callers get the same guard the UI runner gets.
        if isinstance(geometry, gpd.GeoDataFrame):
            geom_wgs84 = reproject_geodataframe_to_wgs84(geometry)
            # ``geom_for_ee`` is the (possibly shrunk) version that fits the
            # EE 10 MB request budget; ``geom_for_crs`` keeps full input
            # precision for client-side per-tile clipping downstream.
            geom_for_ee = shrink_gdf_for_ee(geom_wgs84)
            js = json.loads(geom_for_ee.to_json())
            js.pop("crs", None)
            aoi = ee.FeatureCollection(js["features"]).geometry()
            geom_for_crs = geom_wgs84
        else:
            geom_for_crs = gpd.GeoDataFrame({"geometry": [geometry]}, crs="EPSG:4326")
            geom_for_ee_gdf = shrink_gdf_for_ee(geom_for_crs)
            aoi = shapely_to_ee_geometry(geom_for_ee_gdf.geometry.iloc[0])

        # 2. Pick an EE export CRS so pixels are rasterised in true ground
        # metres regardless of latitude. A caller running several downloads
        # that must align (per-year slices of one dataset) passes a shared
        # ``crs_override`` so every export snaps to the same global grid.
        if crs_override is not None:
            export_crs, grid_crs, export_distortion, export_crs_name = crs_override
        else:
            export_crs, grid_crs, export_distortion, export_crs_name = (
                _select_export_crs(geom_for_crs)
            )

        # Global snap grid for every EE export in this run. Anchoring the
        # transform at (0, 0) in the planar CRS means adjacent tiles share
        # one pixel grid, so boundaries align to the pixel and the mosaic
        # step joins them with zero sub-pixel drift. ``crs_transform`` is
        # the standard GDAL affine ``[a, b, c, d, e, f]`` form:
        # ``[scale, 0, x0, 0, -scale, y0]``.
        crs_transform = [float(resolution), 0.0, 0.0, 0.0, -float(resolution), 0.0]

        # 3. The window as Earth Engine reads it, ``[start, end)``: an
        # inclusive end moves to the next day. Every later step works on
        # this half-open window.
        req_start, req_end = str(start_date), str(end_date)
        end_exclusive = _shift_day(req_end, 1) if end_inclusive else req_end

        # 3a. Optionally fill a window inside 2012 with the same days of 2013
        # (no Landsat sensor without scan-line gaps imaged 2012). Applied
        # before anything is queried, so every date mode gets it.
        year_substituted = None
        if substitute_2012:
            q_start, q_end, shifted = substitute_year(req_start, end_exclusive)
            if shifted:
                year_substituted = "2012 → 2013"
                _log(
                    "INFO",
                    f"2012 window {_window_text(req_start, end_exclusive)} "
                    f"filled with {_window_text(q_start, q_end)}.",
                )
            elif _overlaps_year(req_start, end_exclusive, 2012):
                _log(
                    "WARN",
                    f"Window {_window_text(req_start, end_exclusive)} lies only "
                    "partly in 2012; it is not shifted to 2013.",
                )
        else:
            q_start, q_end = req_start, end_exclusive

        # 3b. Scenes are fetched over the study area grown by the largest
        # neighbourhood radius, so neighbourhoods at its edge are whole.
        radius = focal_radius(chain_steps.values())
        fetch = aoi.buffer(radius + float(resolution)) if radius else aoi
        if radius:
            _log("INFO", f"Scenes fetched within {radius:g} m of the study area.")

        # 4. Pick the satellite collection. ``auto`` falls back to Landsat
        # for ranges that end before Sentinel-2 SR_HARMONIZED started.
        chosen_satellite = self._pick_satellite(q_start, q_end, satellite)
        # ``scene_cmax`` is the scene-level cloud filter (``None`` keeps every
        # scene); ``pixel_mask`` the per-pixel cloud / shadow / snow mask.
        scene_cmax = None if cloud_handling == "pixel_only" else cloud_max
        pixel_mask = cloud_handling != "scene_only"
        if chosen_satellite in ("landsat", "landsat_no_slc_off"):
            sensor_set = (
                "no_slc_off" if chosen_satellite == "landsat_no_slc_off" else "all"
            )
            ee_collection_id = self._landsat_label(sensor_set, reflectance)

            def _get_col(s, e, cmax):
                if (sensor_set, reflectance, pixel_mask) == ("all", "surface", True):
                    return self.get_collection_landsat(fetch, s, e, cmax)
                return self.get_collection_landsat(
                    fetch,
                    s,
                    e,
                    cmax,
                    sensor_set=sensor_set,
                    reflectance=reflectance,
                    pixel_mask=pixel_mask,
                )

            def _get_base(s, e):
                return self._landsat_base(
                    fetch, s, e, sensor_set=sensor_set, reflectance=reflectance
                )

            cloud_field = "CLOUD_COVER"

            # Landsat 5 TM imaging ended ~2012-05 and Landsat 8 began
            # ~2013-04, so a window in between is served by Landsat 7 ETM+
            # alone — whose SLC-off striping (since 2003) leaves ~22% gaps a
            # seasonal median can't fully fill.
            if sensor_set == "all" and q_start < "2013-04-11" and q_end > "2012-05-05":
                _log(
                    "WARN",
                    "Date range lies in the Landsat-7-only era "
                    "(2012-05 → 2013-04): expect SLC-off striping gaps in the "
                    "NDVI composite.",
                )
        else:
            s2_id = self.EE_COLLECTION_ID
            if reflectance == "toa":
                s2_id = self.S2_TOA_COLLECTION_ID
            ee_collection_id = s2_id

            def _get_col(s, e, cmax):
                if (reflectance, pixel_mask) == ("surface", True):
                    return self.get_collection(fetch, s, e, cmax)
                return self.get_collection(
                    fetch, s, e, cmax, reflectance=reflectance, pixel_mask=pixel_mask
                )

            def _get_base(s, e):
                return ee.ImageCollection(s2_id).filterBounds(fetch).filterDate(s, e)

            cloud_field = "CLOUDY_PIXEL_PERCENTAGE"

        _log(
            "INFO",
            f"Satellite: {chosen_satellite} (collection: {ee_collection_id})",
        )
        acquisition = (water_mask, cloud_handling, reflectance)
        if acquisition != (False, "scene_filter", "surface"):
            _log(
                "INFO",
                f"Acquisition: water mask {'on' if water_mask else 'off'}, "
                f"clouds {cloud_handling}, {reflectance} reflectance.",
            )

        def _count(s, e):
            base = _get_base(s, e)
            if scene_cmax is not None:
                base = base.filter(ee.Filter.lt(cloud_field, scene_cmax))
            return base.size().getInfo()

        # 4a. Distinguish "no images in range" vs "all images filtered by
        # cloud threshold" so the user gets an actionable error.
        used_start, used_end = q_start, q_end
        coverage_widened = False
        n_total = _get_base(used_start, used_end).size().getInfo()
        if n_total == 0:
            suggestion = ""
            if chosen_satellite == "sentinel2" and used_end <= self.S2_COLLECTION_START:
                suggestion = (
                    f" Sentinel-2 SR_HARMONIZED starts {self.S2_COLLECTION_START}"
                    " — set satellite='landsat' (or 'auto') for earlier dates."
                )
            elif (
                chosen_satellite == "landsat_no_slc_off"
                and _overlaps_year(used_start, used_end, 2012)
                and not substitute_2012
            ):
                suggestion = (
                    " Without striped Landsat 7 scenes nothing images 2012 — "
                    "fill 2012 with 2013, or use all Landsat sensors."
                )
            return {
                "status": "error",
                "message": (
                    f"No {chosen_satellite} images in collection for "
                    f"{_window_text(used_start, used_end)}.{suggestion}"
                ),
            }
        n_cloud_filtered = (
            _count(used_start, used_end) if scene_cmax is not None else n_total
        )
        if n_cloud_filtered == 0:
            return {
                "status": "error",
                "message": (
                    f"All {n_total} {chosen_satellite} image(s) for "
                    f"{_window_text(used_start, used_end)} exceeded the "
                    f"{cloud_max}% cloud threshold. Try raising the cloud max."
                ),
            }

        # 4b. Coverage rescue: if too few cloud-free images, widen the date
        # range by ±50 % once and re-query. Single attempt — anything more
        # aggressive belongs in user-controlled config.
        MIN_IMAGES_FOR_COMPOSITE = 3
        if (
            coverage_rescue
            and n_cloud_filtered < MIN_IMAGES_FOR_COMPOSITE
            and used_end >= used_start
        ):
            try:
                from datetime import date as _date
                from datetime import timedelta as _timedelta

                s_dt = _date.fromisoformat(used_start)
                e_dt = _date.fromisoformat(used_end)
                span = max((e_dt - s_dt).days, 1)
                pad = _timedelta(days=int(span * 0.5))
                w_start = (s_dt - pad).isoformat()
                w_end = (e_dt + pad).isoformat()
                n_widened = _count(w_start, w_end)
                if n_widened > n_cloud_filtered:
                    _log(
                        "WARN",
                        f"Coverage rescue: only {n_cloud_filtered} cloud-free "
                        f"image(s) in {_window_text(used_start, used_end)}, "
                        f"widening to {_window_text(w_start, w_end)} "
                        f"({n_widened} images).",
                    )
                    used_start, used_end = w_start, w_end
                    n_cloud_filtered = n_widened
                    coverage_widened = True
                    n_total = (
                        n_widened
                        if scene_cmax is None
                        else _get_base(used_start, used_end).size().getInfo()
                    )
            except Exception as e:
                _log("WARN", f"Coverage rescue skipped: {e}")

        col = _get_col(used_start, used_end, scene_cmax)

        if cancel_callback and cancel_callback():
            return {"status": "cancelled", "message": "Cancelled by user"}

        # One single-band composite per output, clipped after its whole chain
        # so neighbourhoods read unclipped scenes. ``unmask(-9999)`` paints
        # cloud-masked / out-of-collection pixels with the sentinel so the
        # downstream reproject can mask them out instead of bilinear-blending
        # them with valid neighbours.
        composites = {
            n: self._chain_composite(col, steps).unmask(-9999).clip(aoi)
            for n, steps in chain_steps.items()
        }
        if user_chains:
            for n, steps in chain_steps.items():
                _log("INFO", f"Aggregation chain {n}: {', '.join(steps)}")
        else:
            _log("INFO", f"Temporal statistic: {', '.join(reducers)}")

        # 5. Build cluster-aware tile list in true metres. Scattered national
        # inputs decompose into connected components, and tiles that fall over
        # empty bbox regions (ocean, gaps between provinces) are STRtree-culled
        # before they reach Earth Engine.
        tiles = build_planar_tiles(geom_for_crs, grid_crs, max_tile_size_km)
        n_tiles = len(tiles)
        if n_tiles == 0:
            return {
                "status": "error",
                "message": "No tiles cover the study area — check input geometry.",
            }

        n_clusters = len({t["cluster_id"] for t in tiles})

        # 6. Compute the resume key from the *effective* run parameters
        # (post coverage-rescue date range + post satellite-pick collection).
        # Anchoring the key on what was actually downloaded means a future
        # run that hits the same widened window / Landsat fallback reuses
        # the cache; runs that resolve differently get isolated workspaces.
        # Each output has its own key (:func:`_chain_key`), so its tiles are
        # its own entry; an acquisition option away from its default adds
        # itself to the key.
        cloud_variant = (cloud_handling,) if cloud_handling != "scene_filter" else ()
        resume_keys = {}
        for n, steps in chain_steps.items():
            token, variant = _chain_key(steps, cloud_variant)
            resume_keys[n] = _compute_resume_key(
                geom_for_crs,
                used_start,
                used_end,
                "none" if scene_cmax is None else cloud_max,
                resolution,
                ee_collection_id,
                temporal_reducer=token,
                variant=variant,
            )

        if n_tiles == 1:
            _log(
                "INFO",
                f"Single-tile download ({n_clusters} cluster(s), planar metres).",
            )
            results = self._download_single(
                composites,
                aoi,
                geometry,
                output_names,
                resolution,
                folder,
                export_crs=export_crs,
                export_distortion=export_distortion,
                export_crs_name=export_crs_name,
                crs_transform=crs_transform,
                cancel_callback=cancel_callback,
                ndvi_progress_callback=ndvi_progress_callback,
                write_geotiff=write_geotiff,
                write_geojson=write_geojson,
                write_geopackage=write_geopackage,
            )
        else:
            _log(
                "INFO",
                f"Tiled download: {n_tiles} tiles across {n_clusters} cluster(s) "
                f"(max {max_tile_size_km} km/tile).",
            )
            results = self._download_with_tiling(
                composites,
                aoi,
                tiles,
                {n: (output_names[n], resume_keys[n]) for n in chain_steps},
                folder,
                max_tile_size_km,
                export_crs=export_crs,
                export_distortion=export_distortion,
                export_crs_name=export_crs_name,
                crs_transform=crs_transform,
                n_clusters=n_clusters,
                cancel_callback=cancel_callback,
                ndvi_progress_callback=ndvi_progress_callback,
                write_geotiff=write_geotiff,
                write_geojson=write_geojson,
                write_geopackage=write_geopackage,
                write_cluster_tiles=write_cluster_tiles,
            )

        # Sidecar: write on success so direct API callers (CLI / fusion auto-
        # download / notebooks) get the same parameter+CRS audit trail the GVI
        # ``_gvi.json`` provides. Failures and cancellations skip this — the
        # sidecar should only describe outputs that actually landed on disk.
        for n, steps in chain_steps.items():
            result = results[n]
            if result.get("status") != "success":
                continue
            raw_meta = result.get("meta")
            result_meta: dict = raw_meta if isinstance(raw_meta, dict) else {}
            tiles_succeeded = int(result_meta.get("tiles", n_tiles))
            tiles_failed = int(result_meta.get("tiles_failed", 0))
            tiles_resumed = int(result_meta.get("tiles_resumed", 0))
            tiles_split = int(result_meta.get("tiles_split_across_requests", 0))
            failed_refs = result_meta.get("failed_tile_refs") or []
            try:
                sidecar_path = _write_ndvi_sidecar(
                    folder,
                    output_names[n],
                    grid_crs=grid_crs,
                    export_crs=export_crs,
                    export_crs_name=export_crs_name,
                    export_distortion=export_distortion,
                    n_clusters=n_clusters,
                    tiles_total=n_tiles,
                    tiles_succeeded=tiles_succeeded,
                    tiles_failed=tiles_failed,
                    tiles_resumed=tiles_resumed,
                    tiles_split_across_requests=tiles_split,
                    failed_tile_refs=failed_refs,
                    resume_key=resume_keys[n],
                    start_date=str(start_date),
                    end_date=str(end_date),
                    end_date_exclusive=end_exclusive,
                    used_start_date=used_start,
                    used_end_date=used_end,
                    temporal_reducer=temporal_statistic(steps),
                    chain=steps,
                    chain_name=n if user_chains else None,
                    coverage_widened=coverage_widened,
                    year_substituted=year_substituted,
                    water_mask=water_mask,
                    cloud_handling=cloud_handling,
                    reflectance=reflectance,
                    preset=preset,
                    cloud_max=cloud_max,
                    resolution_m=resolution,
                    max_tile_size_km=max_tile_size_km,
                    ee_collection=ee_collection_id,
                    satellite=chosen_satellite,
                    bands=["NDVI"],
                    n_cloud_filtered_images=n_cloud_filtered,
                    n_images=n_total,
                    n_images_after_cloud_filter=(
                        n_cloud_filtered if scene_cmax is not None else None
                    ),
                )
                result["sidecar"] = sidecar_path
            except Exception as e:
                _log("WARN", f"Sidecar write failed (non-fatal): {e}")

        return _combine_reducer_results(results)

    def _download_single(
        self,
        composites: Mapping[str, Any],
        aoi,
        geometry,
        output_names: Mapping[str, str],
        resolution,
        folder,
        *,
        export_crs: str = "EPSG:3857",
        export_distortion: float = 0.0,
        export_crs_name: str = "Web Mercator (legacy)",
        crs_transform: list[float] | None = None,
        cancel_callback: Callable[[], bool] | None = None,
        ndvi_progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        write_geotiff: bool = True,
        write_geojson: bool = True,
        write_geopackage: bool = False,
    ) -> dict[str, dict]:
        """Download NDVI as a single tile (for small areas).

        Several outputs come down as the bands of one export (or of several,
        when Earth Engine refuses one as too large: :func:`_export_bands`)
        and are split into their own GeoTIFFs, in the downloaded pixel type,
        so each file holds the values a run of that output alone writes.
        Returns one result per output.
        """
        reducers = list(composites)

        def _each(status: dict) -> dict[str, dict]:
            return {r: dict(status) for r in reducers}

        cancelled = {"status": "cancelled", "message": "Cancelled by user"}
        _emit_ndvi_progress(
            ndvi_progress_callback,
            sub_progress=0.05,
            phase="Downloading from Earth Engine",
            tiles=(0, 1),
        )
        if cancel_callback and cancel_callback():
            return _each(cancelled)
        final_tifs = {
            r: os.path.join(folder, f"{output_names[r]}_ndvi.tif") for r in reducers
        }
        tmp_tif = final_tifs[reducers[0]][:-4] + ".tmp.tif"

        def _export(image, path: str, _alone: bool) -> None:
            retry_with_backoff(
                lambda: _export_ee_image_to_tif(
                    image,
                    path,
                    crs=export_crs,
                    crs_transform=crs_transform,
                    region=aoi,
                ),
                attempts=3,
                base_delay=2.0,
                cancel_callback=cancel_callback,
                log_fn=_log,
                label="EE export",
                non_retryable=(_EEComputeTooLargeError,),
            )

        files: list[tuple[list[str], str]] = []
        split = False
        try:
            files, split = _export_bands(
                composites, reducers, tmp_tif, _export, label="EE export"
            )

            if cancel_callback and cancel_callback():
                return _each(cancelled)

            # Atomic publish: the user-visible final raster only appears
            # once the EE write finished cleanly.
            for names, path in files:
                if len(names) == 1:
                    os.replace(path, final_tifs[names[0]])
                    continue
                for band, r in enumerate(names, start=1):
                    _extract_band(path, band, final_tifs[r])

            _emit_ndvi_progress(
                ndvi_progress_callback,
                sub_progress=0.32,
                phase="Downloading from Earth Engine",
                tiles=(1, 1),
            )

        except Exception as e:
            return _each({"status": "error", "message": f"EE export failed: {str(e)}"})
        finally:
            for path in [tmp_tif, *(p for _, p in files)]:
                if os.path.exists(path):
                    os.remove(path)

        if cancel_callback and cancel_callback():
            return _each(cancelled)

        span = (0.99 - 0.52) / len(reducers)
        return {
            r: self._single_tile_output(
                final_tifs[r],
                geometry,
                output_names[r],
                folder,
                export_crs=export_crs,
                export_distortion=export_distortion,
                export_crs_name=export_crs_name,
                cancel_callback=cancel_callback,
                ndvi_progress_callback=ndvi_progress_callback,
                write_geotiff=write_geotiff,
                write_geojson=write_geojson,
                write_geopackage=write_geopackage,
                sub_start=0.52 + i * span,
                sub_end=0.52 + (i + 1) * span,
                split=split,
            )
            for i, r in enumerate(reducers)
        }

    def _single_tile_output(
        self,
        final_tif: str,
        geometry,
        output_name: str,
        folder,
        *,
        export_crs: str,
        export_distortion: float,
        export_crs_name: str,
        cancel_callback: Callable[[], bool] | None,
        ndvi_progress_callback: Callable[[Mapping[str, Any]], None] | None,
        write_geotiff: bool,
        write_geojson: bool,
        write_geopackage: bool,
        sub_start: float,
        sub_end: float,
        split: bool = False,
    ) -> dict:
        """One output's single-tile result: metadata and vector samples.
        ``split`` records that the outputs came down in several requests."""
        try:
            with rasterio.open(final_tif) as src:
                crs_meta = str(src.crs)
        except Exception:
            crs_meta = export_crs
        meta: dict[str, Any] = {
            "crs": crs_meta,
            "tiles": 1,
            "tiles_total": 1,
            "tiles_failed": 0,
            "tiles_resumed": 0,
            "tiles_split_across_requests": int(split),
            "failed_tile_refs": [],
            "n_clusters": 1,
            "cluster_tiles_dir": None,
            "cluster_tiles_written": 0,
            "export_crs": export_crs,
            "export_crs_name": export_crs_name,
            "export_distortion": export_distortion,
        }

        if not (write_geojson or write_geopackage):
            _emit_ndvi_progress(
                ndvi_progress_callback,
                sub_progress=sub_end,
                phase="Finalizing",
            )
            return {
                "status": "success",
                "tif": final_tif if write_geotiff else None,
                "geojson": None,
                "gpkg": None,
                "meta": meta,
            }

        geojson_path = (
            os.path.join(folder, f"{output_name}_ndvi.geojson")
            if write_geojson
            else None
        )
        gpkg_path = (
            os.path.join(folder, f"{output_name}_ndvi.gpkg")
            if write_geopackage
            else None
        )
        meta.pop("crs")
        out = self._raster_to_ndvi_points(
            final_tif,
            geometry,
            geojson_path=geojson_path,
            gpkg_path=gpkg_path,
            ndvi_progress_callback=ndvi_progress_callback,
            cancel_callback=cancel_callback,
            sub_start=sub_start,
            sub_end=sub_end,
            meta_extra=meta,
        )
        if out.get("status") != "success":
            return out
        if not write_geotiff and os.path.isfile(final_tif):
            os.remove(final_tif)
            out["tif"] = None
        elif write_geotiff:
            out["tif"] = final_tif
        return out

    def _write_per_cluster_outputs(
        self,
        work_dir: str,
        tiles: list[dict],
        failed_tile_refs: list[dict],
        output_root: str,
        export_crs: str,
        export_crs_name: str,
        resume_key: str,
    ) -> list[dict]:
        """Mosaic cached tiles per cluster and write a ``tiles_index.json``.

        Each cluster's planar tiles are stream-mosaicked directly into the
        final per-cluster GeoTIFF (no intermediate reprojection): every tile
        shares the engine's ``crs_transform`` snap grid, so boundaries land
        on pixel centers.

        Failed tiles in ``failed_tile_refs`` are skipped. Per-cluster
        failures log a WARN and continue — the rest of the clusters still
        land. Each entry records both the cluster's native-CRS ``bounds``
        and a derived ``bounds_4326`` so downstream consumers can locate
        the tile geographically without re-reading the raster.
        """
        from collections import defaultdict

        os.makedirs(output_root, exist_ok=True)

        cluster_to_paths: dict[int, list[str]] = defaultdict(list)
        failed_pairs = {
            (int(r["cluster_id"]), int(r["tile_idx"])) for r in failed_tile_refs
        }
        for ts in tiles:
            cid = int(ts["cluster_id"])
            tidx = int(ts["tile_idx"])
            if (cid, tidx) in failed_pairs:
                continue
            p = os.path.join(work_dir, f"tile_{tidx}.tif")
            if os.path.isfile(p):
                cluster_to_paths[cid].append(p)

        entries: list[dict] = []
        index_crs: str | None = None
        for cid in sorted(cluster_to_paths):
            paths = cluster_to_paths[cid]
            if not paths:
                continue
            cluster_path = os.path.join(output_root, f"cluster_{cid:04d}.tif")
            try:
                stream_mosaic_to_geotiff(
                    paths,
                    cluster_path,
                    nodata=-9999,
                    build_overviews=False,
                    compress=True,
                    dst_dtype="float32",
                )
            except Exception as e:
                _log("WARN", f"Cluster {cid} mosaic failed: {e}")
                continue

            try:
                with rasterio.open(cluster_path) as src:
                    b = src.bounds
                    native_bounds = [
                        float(b.left),
                        float(b.bottom),
                        float(b.right),
                        float(b.top),
                    ]
                    if index_crs is None and src.crs is not None:
                        index_crs = str(src.crs)
            except Exception:
                native_bounds = None
            bounds_4326 = raster_geographic_bounds(cluster_path)
            entries.append(
                {
                    "cluster_id": cid,
                    "path": os.path.basename(cluster_path),
                    "bounds": native_bounds,
                    "bounds_4326": (
                        list(bounds_4326) if bounds_4326 is not None else None
                    ),
                    "n_tiles": len(paths),
                }
            )

        index_path = os.path.join(output_root, "tiles_index.json")
        with open(index_path, "w") as f:
            json.dump(
                {
                    "crs": index_crs or export_crs,
                    "export_crs": export_crs,
                    "export_crs_name": export_crs_name,
                    "resume_key": resume_key,
                    "n_clusters": len(entries),
                    "clusters": entries,
                },
                f,
                indent=2,
            )
        return entries

    def _download_one_tile(
        self,
        tile_spec: dict,
        composites: Mapping[str, Any],
        reducers: list[str],
        work_dirs: Mapping[str, str],
        export_crs: str,
        crs_transform: list[float],
        cancel_callback: Callable[[], bool] | None,
    ) -> dict:
        """Worker: one tile's missing outputs, in one Earth Engine request.

        Runs on a ThreadPoolExecutor thread (see :meth:`_download_with_tiling`).
        ``reducers`` are the outputs this tile has no cache file for; they
        are stacked as the bands of one image, exported once (split across
        requests when Earth Engine refuses it as too large:
        :func:`_export_bands`), and each band is stored in its own output's
        workspace. Returns ``{idx, reducers, tile_files, split, error}`` with
        ``tile_files`` mapping each stored output to its tile and ``split``
        whether the request was split; failures surface as ``error`` rather
        than raising so a single bad tile can't sink the whole batch, and
        leave no files.

        Tiles stay in the engine's chosen planar CRS — *no* per-tile reproject
        to WGS84 — so all tiles share one snap grid (``crs_transform`` fixes
        the origin globally) and the mosaic step joins them with zero
        sub-pixel drift. A single final reproject converts the merged planar
        mosaic to WGS84 with explicit nodata.

        Each band is stored as a compact float32 cache tile by
        :func:`_extract_band`. The request itself keeps Earth Engine's float64
        encoding: a float32 request comes back with ``-inf`` at edge pixels
        outside the tile's clip, which would overwrite a neighbour's valid
        pixels where tiles overlap in the mosaic.

        Cancel checks short-circuit at safe boundaries — Earth Engine's HTTP
        call is one blocking step that can't be killed mid-flight, so a
        freshly-cancelled run may still have a few in-flight downloads finish
        and be discarded by the caller.
        """
        idx = int(tile_spec["tile_idx"])
        result: dict = {
            "idx": idx,
            "reducers": list(reducers),
            "tile_files": {},
            "split": False,
            "error": None,
        }

        if cancel_callback and cancel_callback():
            return result

        download = os.path.join(work_dirs[reducers[0]], f"tile_{idx}.download.tif")
        label = f"Tile {idx + 1}"
        written: dict[str, str] = {}
        files: list[tuple[list[str], str]] = []

        def _export(image, path: str, alone: bool) -> None:
            # Several outputs split across requests before the region does.
            self._export_region_adaptive(
                image,
                tile_spec["tile_geom_4326"],
                path,
                export_crs=export_crs,
                crs_transform=crs_transform,
                cancel_callback=cancel_callback,
                depth=0,
                max_depth=_MAX_TILE_SUBDIVISION_DEPTH if alone else 0,
                label=label,
            )

        try:
            files, result["split"] = _export_bands(
                composites, list(reducers), download, _export, label=label
            )

            if cancel_callback and cancel_callback():
                return result

            for names, path in files:
                for band, r in enumerate(names, start=1):
                    tile_final = os.path.join(work_dirs[r], f"tile_{idx}.tif")
                    _extract_band(path, band, tile_final, dtype="float32")
                    written[r] = tile_final
            result["tile_files"] = written
            return result
        except Exception as e:
            for path in written.values():
                try:
                    os.remove(path)
                except OSError:
                    pass
            result["error"] = f"{type(e).__name__}: {e}"
            return result
        finally:
            for path in [download, *(p for _, p in files)]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    def _export_region_adaptive(
        self,
        ndvi_median,
        region_geom_4326,
        out_path: str,
        *,
        export_crs: str,
        crs_transform: list[float],
        cancel_callback: Callable[[], bool] | None,
        depth: int,
        label: str,
        max_depth: int = _MAX_TILE_SUBDIVISION_DEPTH,
    ) -> None:
        """Export one region to ``out_path``, subdividing on EE size limits.

        The common case is a single EE export of ``region_geom_4326`` (with
        transient-failure retries). When Earth Engine rejects the request as
        too large to compute — ``_EEComputeTooLargeError`` — the region is
        halved into a 2×2 grid of sub-regions, each exported the same way
        (recursively, so a still-too-big quadrant halves again up to
        ``max_depth`` levels; 0 re-raises at once), and the pieces are stream-
        mosaicked back into ``out_path``. Every sub-region shares the global
        ``crs_transform``, so the merged result is pixel-identical to what a
        single successful export would have produced — the on-disk tile is
        indistinguishable from a tile that never needed splitting.
        """
        region_aoi = shapely_to_ee_geometry(region_geom_4326)

        def _do_export() -> None:
            _export_ee_image_to_tif(
                ndvi_median.clip(region_aoi),
                out_path,
                crs=export_crs,
                crs_transform=crs_transform,
                region=region_aoi,
            )

        try:
            # 3 attempts with 2 s base delay for transient (network) failures.
            # A size/memory rejection is deterministic, so it is non-retryable
            # here and re-raises immediately for the subdivision path below.
            retry_with_backoff(
                _do_export,
                attempts=3,
                base_delay=2.0,
                cancel_callback=cancel_callback,
                log_fn=_log,
                label=f"{label} EE export",
                non_retryable=(_EEComputeTooLargeError,),
            )
            return
        except _EEComputeTooLargeError as too_large:
            quadrants = _split_geom_quadrants(region_geom_4326)
            if depth >= max_depth or len(quadrants) < 2:
                raise
            _log(
                "INFO",
                f"{label}: {too_large} — subdividing into {len(quadrants)} "
                f"sub-tiles (level {depth + 1}).",
            )
            sub_paths: list[str] = []
            try:
                for qi, quad in enumerate(quadrants):
                    if cancel_callback and cancel_callback():
                        return
                    sub_out = f"{out_path}.sub{depth}_{qi}.tif"
                    self._export_region_adaptive(
                        ndvi_median,
                        quad,
                        sub_out,
                        export_crs=export_crs,
                        crs_transform=crs_transform,
                        cancel_callback=cancel_callback,
                        depth=depth + 1,
                        label=f"{label}.{qi + 1}",
                        max_depth=max_depth,
                    )
                    sub_paths.append(sub_out)
                stream_mosaic_to_geotiff(
                    sub_paths,
                    out_path,
                    nodata=-9999,
                    build_overviews=False,
                    compress=True,
                    dst_dtype="float32",
                )
            except Exception:
                if os.path.exists(out_path):
                    try:
                        os.remove(out_path)
                    except OSError:
                        pass
                raise
            finally:
                for p in sub_paths:
                    try:
                        os.remove(p)
                    except OSError:
                        pass

    def _download_with_tiling(
        self,
        composites: Mapping[str, Any],
        aoi,
        tiles: list[dict],
        outputs: Mapping[str, tuple[str, str]],
        folder,
        max_tile_size_km,
        *,
        export_crs: str = "EPSG:3857",
        export_distortion: float = 0.0,
        export_crs_name: str = "Web Mercator (legacy)",
        crs_transform: list[float] | None = None,
        n_clusters: int = 1,
        cancel_callback: Callable[[], bool] | None = None,
        ndvi_progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        write_geotiff: bool = True,
        write_geojson: bool = True,
        write_geopackage: bool = False,
        write_cluster_tiles: bool = False,
    ) -> dict[str, dict]:
        """Download NDVI for a precomputed cluster-aware tile list and mosaic.

        ``tiles`` is the output of :func:`geofuse.core.build_planar_tiles` — a list of
        ``{cluster_id, tile_idx, tile_geom_4326}`` dicts. Each tile is exported
        independently and mosaicked into one GeoTIFF; per-cluster GeoTIFF tile
        outputs are written separately by the cluster-tiles path.

        ``outputs`` maps each temporal statistic to ``(output_name,
        resume_key)``. Every statistic has its own cache workspace, so a later
        run asking for any subset reuses what is on disk; tiles already present
        from a previous run with the same key are skipped, and a tile missing
        several statistics fetches them in one request. Returns one result per
        statistic.
        """
        n_tiles = len(tiles)
        reducers = list(composites)
        _log(
            "INFO",
            f"Downloading {n_tiles} tile(s) across {n_clusters} cluster(s) "
            f"(max {max_tile_size_km} km/tile)...",
        )

        _emit_ndvi_progress(
            ndvi_progress_callback,
            sub_progress=0.02,
            phase="Downloading tiles",
            tiles=(0, n_tiles),
        )

        # The cache directory doubles as both the in-progress workspace
        # *and* the cross-run cache: successful runs leave their tiles in
        # place so a future run with the same ``resume_key`` (same area +
        # date range + cloud max + resolution + collection + statistic)
        # reuses every tile instantly. The cache's LRU + size cap reclaims old
        # entries when it grows past the configured ceiling. Tiles are float32
        # with DEFLATE and the floating-point predictor (``_extract_band``);
        # older float64 tiles stay readable and give identical mosaics.
        work_dirs = {
            r: self.tile_cache.workspace_dir(
                outputs[r][1] or ndvi_output_name("scratch", r)
            )
            for r in reducers
        }

        # Resume: any tile already present on disk (and openable as a
        # rasterio dataset) is reused as-is; only the statistics a tile is
        # missing get queued for download. Same lookup whether we're resuming
        # an interrupted run or hitting a previously-cached one — only the
        # user-facing log message differs.
        tile_files: dict[str, list[str]] = {r: [] for r in reducers}
        n_resumed: dict[str, int] = {}
        missing: dict[int, list[str]] = {}
        for r in reducers:
            existing, pending = _discover_resumable_tiles(work_dirs[r], tiles)
            tile_files[r].extend(existing)
            n_resumed[r] = len(existing)
            for ts in pending:
                missing.setdefault(int(ts["tile_idx"]), []).append(r)
            if existing:
                which = "" if len(reducers) == 1 else f" ({r})"
                _log(
                    "INFO",
                    f"Resumed {len(existing)}/{n_tiles} tile(s){which} from "
                    f"previous run ({work_dirs[r]}).",
                )
        pending_tiles = [ts for ts in tiles if int(ts["tile_idx"]) in missing]

        # Tiles that exhausted the retry budget, per statistic. Recorded with
        # their ``cluster_id`` / ``tile_idx`` / final error so the sidecar can
        # surface exactly which areas are NaN gaps in the mosaic.
        failed_tile_refs: dict[str, list[dict]] = {r: [] for r in reducers}
        # Tiles whose outputs Earth Engine took in several requests, per output.
        n_split: dict[str, int] = {r: 0 for r in reducers}

        # Parallel tile downloads. Each worker handles one tile end-to-end
        # (Earth Engine export → compact cache tiles). The ThreadPoolExecutor
        # caps concurrency at ``_MAX_CONCURRENT_TILES`` so we don't saturate
        # local sockets or oversubscribe EE per-user. Progress emits are
        # throttled to every :data:`NDVI_PROGRESS_MIN_TILES` tiles so the UI
        # bracket stays responsive without slamming the JobStore lock on
        # big runs.
        done_count = n_tiles - len(pending_tiles)
        _tile_throttle = ProgressThrottle(min_items=NDVI_PROGRESS_MIN_TILES)

        def _emit_progress(k: int) -> None:
            if not _tile_throttle.should_emit(k, final=k >= n_tiles):
                return
            _emit_ndvi_progress(
                ndvi_progress_callback,
                sub_progress=0.70 * k / n_tiles if n_tiles else 0.0,
                phase="Downloading tiles",
                tiles=(k, n_tiles),
            )

        # Show the resumed count immediately so the bracket jumps to the
        # right starting point instead of flashing 0/N.
        _emit_progress(done_count)

        cancelled = False
        with ThreadPoolExecutor(
            max_workers=_MAX_CONCURRENT_TILES,
            thread_name_prefix="ndvi-tile",
        ) as pool:
            futures = {
                pool.submit(
                    self._download_one_tile,
                    ts,
                    composites,
                    missing[int(ts["tile_idx"])],
                    work_dirs,
                    export_crs,
                    crs_transform,
                    cancel_callback,
                ): ts
                for ts in pending_tiles
            }
            for fut in as_completed(futures):
                res = fut.result()
                done_count += 1
                for r, path in res["tile_files"].items():
                    tile_files[r].append(path)
                    n_split[r] += int(res["split"])
                if res.get("error"):
                    tile_spec = futures[fut]
                    for r in res["reducers"]:
                        failed_tile_refs[r].append(
                            {
                                "cluster_id": int(tile_spec["cluster_id"]),
                                "tile_idx": int(res["idx"]),
                                "error": str(res["error"]),
                            }
                        )
                    _log(
                        "WARN",
                        f"Tile {res['idx'] + 1} failed after retry: " f"{res['error']}",
                    )
                _emit_progress(done_count)
                if cancel_callback and cancel_callback():
                    cancelled = True
                    # Cancel any not-yet-running futures; in-flight tiles
                    # finish on their own (EE export is one blocking call,
                    # we can't kill it mid-flight) and their results are
                    # discarded.
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    break

        if cancelled or (cancel_callback and cancel_callback()):
            for r in reducers:
                _log("INFO", f"Tile workspace preserved for resume: {work_dirs[r]}")
            return {
                r: {"status": "cancelled", "message": "Cancelled by user"}
                for r in reducers
            }

        # Each statistic is mosaicked in turn, in its own share of the
        # progress bar after the shared download.
        share = (0.99 - 0.70) / len(reducers)
        results: dict[str, dict] = {}
        for i, r in enumerate(reducers):

            def _at(x: float, lo: float = 0.70 + i * share) -> float:
                return lo + (x - 0.70) / (0.99 - 0.70) * share

            output_name, resume_key = outputs[r]
            results[r] = self._finish_tiled_output(
                tile_files[r],
                failed_tile_refs[r],
                n_resumed[r],
                tiles,
                work_dirs[r],
                resume_key,
                output_name,
                folder,
                n_split=n_split[r],
                label="" if len(reducers) == 1 else f" ({r})",
                progress_at=_at,
                n_clusters=n_clusters,
                export_crs=export_crs,
                export_distortion=export_distortion,
                export_crs_name=export_crs_name,
                cancel_callback=cancel_callback,
                ndvi_progress_callback=ndvi_progress_callback,
                write_geotiff=write_geotiff,
                write_geojson=write_geojson,
                write_geopackage=write_geopackage,
                write_cluster_tiles=write_cluster_tiles,
            )

        # Trim the cache once every statistic's workspace has been recorded,
        # so a sibling workspace this run still needs is never the oldest.
        if any(res.get("status") == "success" for res in results.values()):
            try:
                n_evicted, bytes_freed = self.tile_cache.evict_lru()
                if n_evicted:
                    _log(
                        "INFO",
                        f"Tile cache: evicted {n_evicted} LRU entry/entries "
                        f"({bytes_freed / 1024**2:.0f} MB) to stay under cap.",
                    )
            except Exception as e:
                _log("WARN", f"Tile cache bookkeeping failed: {e}")
        return results

    def _finish_tiled_output(
        self,
        tile_files: list[str],
        failed_tile_refs: list[dict],
        n_resumed: int,
        tiles: list[dict],
        work_dir: str,
        resume_key: str,
        output_name: str,
        folder,
        *,
        n_split: int = 0,
        label: str,
        progress_at: Callable[[float], float],
        n_clusters: int,
        export_crs: str,
        export_distortion: float,
        export_crs_name: str,
        cancel_callback: Callable[[], bool] | None,
        ndvi_progress_callback: Callable[[Mapping[str, Any]], None] | None,
        write_geotiff: bool,
        write_geojson: bool,
        write_geopackage: bool,
        write_cluster_tiles: bool,
    ) -> dict:
        """Mosaic one statistic's tiles into its outputs.

        Progress values are those of a single-statistic run, mapped into this
        statistic's share of the bar by ``progress_at``. A successful mosaic
        records the workspace in the cache index; the caller evicts.
        """
        n_tiles = len(tiles)
        final_tif = os.path.join(folder, f"{output_name}_ndvi.tif")

        if not tile_files:
            _log("INFO", f"Tile workspace preserved for resume: {work_dir}")
            return {"status": "error", "message": "All tiles failed to download"}

        if failed_tile_refs:
            _log(
                "WARN",
                f"{len(failed_tile_refs)}/{n_tiles} tile(s){label} failed after "
                "retry — mosaic will have NaN gaps in those areas. "
                "See sidecar JSON for the cluster/tile IDs.",
            )

        _log(
            "OK",
            f"Successfully downloaded {len(tile_files)}/{n_tiles} tiles{label}. "
            "Mosaicking...",
        )

        if cancel_callback and cancel_callback():
            _log("INFO", f"Tile workspace preserved for resume: {work_dir}")
            return {"status": "cancelled", "message": "Cancelled by user"}

        _emit_ndvi_progress(
            ndvi_progress_callback,
            sub_progress=progress_at(0.72),
            phase="Mosaicking rasters",
            tiles=(0, len(tile_files)),
            clear_bracket=True,
        )

        # Stream-mosaic the per-tile rasters in their shared planar CRS
        # straight into the user-facing GeoTIFF. Tile boundaries are
        # pixel-aligned because every EE export was anchored to the same
        # crs_transform; mosaic memory never holds more than one tile.
        try:

            def _mosaic_progress(k: int, n: int) -> None:
                span = 0.80 - 0.72
                _emit_ndvi_progress(
                    ndvi_progress_callback,
                    sub_progress=progress_at(0.72 + (span * k / n if n else 0.0)),
                    phase="Mosaicking rasters",
                    tiles=(k, n),
                )

            stream_mosaic_to_geotiff(
                tile_files,
                final_tif,
                nodata=-9999,
                progress_cb=_mosaic_progress,
                build_overviews=False,
                compress=True,
                dst_dtype="float32",
            )

            _emit_ndvi_progress(
                ndvi_progress_callback,
                sub_progress=progress_at(0.80),
                phase="Mosaicking rasters",
            )
            _log("OK", f"Mosaic complete: {final_tif}")

        except Exception as e:
            _log(
                "ERROR",
                f"Mosaic step failed with traceback:\n{traceback.format_exc()}",
            )
            _log("INFO", f"Tile workspace preserved for resume: {work_dir}")
            return {"status": "error", "message": f"Mosaic failed: {str(e)}"}

        # Per-cluster GeoTIFF tiles + tiles_index.json. Best-effort: main
        # mosaic already succeeded, so per-cluster failures log a WARN rather
        # than failing the whole run.
        cluster_tiles_dir: str | None = None
        cluster_tiles_written = 0
        if write_cluster_tiles:
            cluster_tiles_dir = os.path.join(folder, f"{output_name}_ndvi_tiles")
            _emit_ndvi_progress(
                ndvi_progress_callback,
                sub_progress=progress_at(0.82),
                phase="Writing per-cluster tiles",
            )
            try:
                entries = self._write_per_cluster_outputs(
                    work_dir,
                    tiles,
                    failed_tile_refs,
                    cluster_tiles_dir,
                    export_crs,
                    export_crs_name,
                    resume_key,
                )
                _log(
                    "OK",
                    f"Wrote {len(entries)} per-cluster tile(s) to "
                    f"{cluster_tiles_dir}",
                )
                cluster_tiles_written = len(entries)
            except Exception as e:
                _log("WARN", f"Per-cluster tile output failed: {e}")
                cluster_tiles_dir = None

        # Don't delete the workspace — it's the persistent cache now. Record
        # the on-disk footprint + access time so the LRU eviction has correct
        # numbers.
        if resume_key:
            _emit_ndvi_progress(
                ndvi_progress_callback,
                sub_progress=progress_at(0.86),
                phase="Updating tile cache",
            )
            try:
                self.tile_cache.touch(resume_key)
            except Exception as e:
                _log("WARN", f"Tile cache bookkeeping failed: {e}")

        if cancel_callback and cancel_callback():
            return {"status": "cancelled", "message": "Cancelled by user"}

        meta_base: dict[str, Any] = {
            "tiles": len(tile_files),
            "tiles_total": n_tiles,
            "tiles_failed": len(failed_tile_refs),
            "tiles_resumed": n_resumed,
            "tiles_split_across_requests": n_split,
            "failed_tile_refs": failed_tile_refs,
            "n_clusters": n_clusters,
            "resume_key": resume_key,
            "cluster_tiles_dir": cluster_tiles_dir,
            "cluster_tiles_written": cluster_tiles_written,
            "export_crs": export_crs,
            "export_crs_name": export_crs_name,
            "export_distortion": export_distortion,
        }

        if not (write_geojson or write_geopackage):
            try:
                with rasterio.open(final_tif) as src:
                    meta_base["crs"] = str(src.crs)
            except Exception:
                meta_base["crs"] = export_crs
            _emit_ndvi_progress(
                ndvi_progress_callback,
                sub_progress=progress_at(0.99),
                phase="Finalizing",
            )
            return {
                "status": "success",
                "tif": final_tif if write_geotiff else None,
                "geojson": None,
                "gpkg": None,
                "meta": meta_base,
            }

        geojson_path = (
            os.path.join(folder, f"{output_name}_ndvi.geojson")
            if write_geojson
            else None
        )
        gpkg_path = (
            os.path.join(folder, f"{output_name}_ndvi.gpkg")
            if write_geopackage
            else None
        )
        out = self._raster_to_ndvi_points(
            final_tif,
            None,
            geojson_path=geojson_path,
            gpkg_path=gpkg_path,
            ndvi_progress_callback=ndvi_progress_callback,
            cancel_callback=cancel_callback,
            sub_start=progress_at(0.88),
            sub_end=progress_at(0.99),
            meta_extra=meta_base,
        )
        if out.get("status") != "success":
            return out
        if not write_geotiff and os.path.isfile(final_tif):
            os.remove(final_tif)
            out["tif"] = None
        elif write_geotiff:
            out["tif"] = final_tif
        return out
