"""Canonical WGS84 (lon/lat) handling for GeoFuse pipelines and Folium previews.

GeoJSON may declare ``OGC:CRS84`` (lon, lat) or legacy ``EPSG:4326``. PyProj / GDAL can
interpret axis order differently across CRS objects. We normalize geographic inputs with
``always_xy=True`` so stored coordinates stay **x = longitude, y = latitude** in degrees,
which matches Leaflet/Folium and ``generate_raster_grid`` in ``geofuse.core``.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS as PyProjCRS
from pyproj import Geod, Transformer
from rasterio.transform import array_bounds, from_bounds
from rasterio.warp import (
    Resampling,
    calculate_default_transform,
    reproject,
    transform_bounds,
)
from rasterio.warp import transform as rio_warp_transform
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.windows import transform as window_transform
from shapely.ops import transform as shapely_xy_transform

# Single canonical CRS for web maps, Earth Engine clip geometries, and GVI/NDVI download.
WGS84_EPSG = "EPSG:4326"


def metres_per_degree_at_lat(lat_deg: float) -> tuple[float, float]:
    """Geodesic metres-per-degree at ``lat_deg`` for square-metre raster math.

    Returns ``(m_per_deg_lon, m_per_deg_lat)`` using the standard WGS84 series
    (third-order in latitude). Replaces several copies of this formula that
    had drifted in precision across the engines and UI hint helpers; one
    canonical value here keeps grid sizing, raster reprojection, and the
    size-estimate banners consistent.
    """
    lat_rad = np.radians(lat_deg)
    m_per_deg_lat = 111132.954 - 559.822 * np.cos(2 * lat_rad)
    m_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)
    return float(m_per_deg_lon), float(m_per_deg_lat)


def select_grid_crs_with_warning(
    gdf: gpd.GeoDataFrame,
    log_fn,
    *,
    role: str = "Grid CRS",
    threshold: float = 0.02,
):
    """:func:`select_grid_crs` + the standard "warn if > 2 %" emit pattern.

    GVI and NDVI both call ``select_grid_crs`` and emit the same WARN/INFO
    pair depending on the measured distortion. ``log_fn`` is the engine's
    ``_log`` callable so each engine's messages still appear under its own
    logger name (e.g. ``[GVI]`` vs ``[NDVI]``); only the boilerplate is shared.
    """
    crs, distortion, choice_name = select_grid_crs(gdf)
    if distortion > threshold:
        log_fn(
            "WARN",
            f"{role} distortion ~ {distortion * 100:.2f}% across the extent "
            f"({choice_name}). Pixel scale will drift across widely-spaced tiles.",
        )
    else:
        log_fn(
            "INFO",
            f"{role}: {choice_name} (max planar distortion ~ "
            f"{distortion * 100:.3f}%).",
        )
    return crs, float(distortion), choice_name


def default_geotiff_creation_options(dtype, *, sparse: bool = False) -> dict:
    """Compression / tiling / BIGTIFF defaults for our GeoTIFF outputs.

    Applied by every helper that writes a GeoTIFF (``reproject_raster_to_wgs84``,
    ``stream_mosaic_to_geotiff``) and by the GVI per-cluster writer in
    ``runners.py``. The combination — DEFLATE with a dtype-appropriate
    predictor, 256×256 internal tiling, BIGTIFF support — cuts national-scale
    NDVI outputs by ~5–10× and unlocks fast random reads for Folium overlays
    and downstream raster sampling.

    Predictor choice follows the libtiff convention:
      * ``3`` (floating-point predictor) for ``float32`` / ``float64`` rasters
        — handles NDVI's continuous values well.
      * ``2`` (horizontal differencing) for integer rasters.

    Pass ``sparse=True`` to enable ``SPARSE_OK``, which lets libtiff omit
    blocks whose pixels are all zero or all the nodata value. Useful for
    buffered-AOI outputs where most of the bounding box is empty.
    """
    predictor = 3 if np.issubdtype(np.dtype(dtype), np.floating) else 2
    opts = {
        "compress": "DEFLATE",
        "predictor": predictor,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "YES",
    }
    if sparse:
        opts["SPARSE_OK"] = "TRUE"
    return opts


def raster_geographic_bounds(
    raster_path: str,
) -> tuple[float, float, float, float] | None:
    """Return the raster's bounding box in EPSG:4326 as ``(left, bottom, right, top)``.

    Returns ``None`` if the file cannot be opened or has no declared CRS.
    Used by sidecar / tile-index writers that want a geographic locator
    alongside the raster's native-CRS extent.
    """
    try:
        with rasterio.open(raster_path) as src:
            if src.crs is None:
                return None
            left, bottom, right, top = transform_bounds(
                src.crs, WGS84_EPSG, *src.bounds
            )
            return float(left), float(bottom), float(right), float(top)
    except Exception:
        return None


def build_internal_overviews(
    path: str,
    factors: tuple[int, ...] = (2, 4, 8),
    resampling: Resampling = Resampling.average,
) -> None:
    """Add internal overview pyramids to an existing GeoTIFF in place.

    Three levels (2×, 4×, 8×) give Folium / QGIS previews snappy zoom-out
    without doubling write time on national-scale rasters. Default
    resampling is ``average``.

    GDAL drops factors that would shrink the raster below 1 px, so the
    default tuple is safe for small tiles too.
    """
    with rasterio.open(path, "r+") as dst:
        dst.build_overviews(list(factors), resampling)
        dst.update_tags(ns="rio_overview", resampling=resampling.name)


def stream_mosaic_to_geotiff(
    tile_paths: list[str],
    dst_path: str,
    *,
    nodata: float = -9999,
    resampling: Resampling = Resampling.bilinear,
    progress_cb: Callable[[int, int], None] | None = None,
    build_overviews: bool = True,
    compress: bool = True,
) -> int:
    """Stream-mosaic same-CRS GeoTIFF tiles into one output via windowed writes.

    Never holds more than one tile's worth of pixels in memory at a time, so
    national-scale outputs don't OOM (vs. ``rasterio.merge.merge`` which
    materialises the entire mosaic up front). All tiles must share the same
    CRS and band count; minor pixel-size differences between tiles — e.g.
    WGS84 tiles reprojected at different centroid latitudes — are resampled
    into the unified grid via ``resampling``. The first tile defines the
    output's pixel size, dtype, and band count.

    ``progress_cb(k, n)`` fires after each tile lands; pass it to mirror
    download-phase progress into the mosaic phase. Returns the number of
    tiles written.

    Set ``compress=False`` for ephemeral intermediate mosaics that are read
    once and deleted (e.g. the planar mosaic feeding a WGS84 reproject).
    Skipping DEFLATE saves write time and lets the downstream reader scan
    the file without per-block decompression.
    """
    if not tile_paths:
        raise ValueError("stream_mosaic_to_geotiff: tile_paths is empty.")

    tile_profiles: list[dict] = []
    for path in tile_paths:
        with rasterio.open(path) as src:
            tile_profiles.append(
                {
                    "path": path,
                    "bounds": src.bounds,
                    "transform": src.transform,
                    "crs": src.crs,
                    "dtype": src.dtypes[0],
                    "count": src.count,
                }
            )

    ref = tile_profiles[0]
    union_left = min(p["bounds"].left for p in tile_profiles)
    union_bottom = min(p["bounds"].bottom for p in tile_profiles)
    union_right = max(p["bounds"].right for p in tile_profiles)
    union_top = max(p["bounds"].top for p in tile_profiles)

    pixel_w = abs(ref["transform"].a)
    pixel_h = abs(ref["transform"].e)
    out_width = max(1, int(round((union_right - union_left) / pixel_w)))
    out_height = max(1, int(round((union_top - union_bottom) / pixel_h)))
    out_transform = from_bounds(
        union_left, union_bottom, union_right, union_top, out_width, out_height
    )

    dst_profile = {
        "driver": "GTiff",
        "height": out_height,
        "width": out_width,
        "count": ref["count"],
        "dtype": ref["dtype"],
        "crs": ref["crs"],
        "transform": out_transform,
        "nodata": nodata,
    }
    if compress:
        dst_profile.update(default_geotiff_creation_options(ref["dtype"], sparse=True))
    else:
        dst_profile.update(
            {
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "BIGTIFF": "YES",
                "SPARSE_OK": "TRUE",
            }
        )

    n = len(tile_profiles)
    written = 0
    with rasterio.open(dst_path, "w", **dst_profile) as dst:
        for tp in tile_profiles:
            win = (
                window_from_bounds(*tp["bounds"], transform=out_transform)
                .round_offsets()
                .round_lengths()
            )
            if win.width <= 0 or win.height <= 0:
                if progress_cb is not None:
                    progress_cb(written, n)
                continue
            win_transform = window_transform(win, out_transform)
            with rasterio.open(tp["path"]) as src:
                for band in range(1, src.count + 1):
                    dst_arr = np.full(
                        (win.height, win.width),
                        fill_value=nodata,
                        dtype=src.dtypes[band - 1],
                    )
                    reproject(
                        source=rasterio.band(src, band),
                        destination=dst_arr,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=win_transform,
                        dst_crs=ref["crs"],
                        resampling=resampling,
                        src_nodata=nodata,
                        dst_nodata=nodata,
                    )
                    dst.write(dst_arr, indexes=band, window=win)
            written += 1
            if progress_cb is not None:
                progress_cb(written, n)

    if build_overviews and written > 0:
        # Overviews are added *after* the dataset is closed so GDAL flushes
        # the base raster first. Float NDVI uses ``average`` resampling —
        # nearest would alias on smooth gradients.
        try:
            build_internal_overviews(dst_path)
        except Exception:
            # Overview build is non-fatal — the base raster is still valid.
            pass
    return written


def reproject_raster_to_wgs84(
    src_path: str,
    dst_path: str,
    *,
    target_resolution_m: float,
    resampling: Resampling = Resampling.bilinear,
    src_nodata: float | None = None,
    dst_nodata: float | None = None,
    num_threads: int | None = None,
    block_size: int = 512,
    warp_mem_limit_mb: int = 2048,
    gdal_cachemax_mb: int = 2048,
) -> None:
    """Reproject a planar-CRS GeoTIFF to EPSG:4326 in a single GDAL warp call.

    Drives the same warp kernel that ``gdalwarp`` / QGIS use, in-process
    and end-to-end: one :func:`rasterio.warp.reproject` call per band with
    ``source=rasterio.band(src, b)`` and ``destination=rasterio.band(dst, b)``.
    GDAL streams blocks internally on ``num_threads`` worker threads,
    ``SPARSE_OK=TRUE`` keeps fully-nodata blocks off disk, and
    ``src_nodata`` / ``dst_nodata`` propagate through the resampler so
    bilinear does not bleed the sentinel into edge pixels.

    No overview pyramids are built. For float-nodata sparse rasters
    GDAL's overview builder reads sparse blocks as nodata, which the
    `average` resampler then mixes with valid pixels to produce
    misleading level-0 values, and the pass adds tens of minutes to
    national-scale runs. Call :func:`build_internal_overviews` on the
    finished file if a specific consumer needs them.

    Defaults to bilinear resampling — appropriate for continuous bands
    like NDVI. Pass ``resampling=Resampling.nearest`` for QA /
    classification bands.

    Geometry note: pixels end up roughly square in **ground metres** at
    the centroid latitude (``res_*_deg = target_resolution_m /
    m_per_deg_*``). EPSG:4326 is a geographic CRS so the corresponding
    **degree** resolution is rectangular — one degree of longitude is
    ``cos(lat)`` shorter than one degree of latitude, so the lon/lat
    degree-per-pixel ratio sits at ~1.3-1.7 at mid-latitudes. QGIS
    displays pixel size as ``degrees × 111139`` without the cos-lat
    correction, which reads as "wider in EW than NS" even though the
    on-ground spacing is square at the centroid.
    """
    if num_threads is None:
        num_threads = max(1, os.cpu_count() or 1)

    with rasterio.Env(GDAL_CACHEMAX=int(gdal_cachemax_mb)):
        with rasterio.open(src_path) as src:
            left, bottom, right, top = array_bounds(
                src.height, src.width, src.transform
            )
            cx, cy = (left + right) / 2, (bottom + top) / 2
            transformed = rio_warp_transform(src.crs, WGS84_EPSG, [cx], [cy])
            avg_lat = transformed[1][0]

            m_per_deg_lon, m_per_deg_lat = metres_per_degree_at_lat(avg_lat)
            res_x_deg = target_resolution_m / m_per_deg_lon
            res_y_deg = target_resolution_m / m_per_deg_lat

            dst_transform, width, height = calculate_default_transform(
                src.crs,
                WGS84_EPSG,
                src.width,
                src.height,
                *src.bounds,
                resolution=(res_x_deg, res_y_deg),
            )
            assert width is not None and height is not None
            width = int(width)
            height = int(height)

            dst_kwargs = src.meta.copy()
            dst_kwargs.update(
                {
                    "crs": WGS84_EPSG,
                    "transform": dst_transform,
                    "width": width,
                    "height": height,
                }
            )
            if dst_nodata is not None:
                dst_kwargs["nodata"] = dst_nodata
            dst_kwargs.update(
                default_geotiff_creation_options(src.dtypes[0], sparse=True)
            )
            dst_kwargs["blockxsize"] = block_size
            dst_kwargs["blockysize"] = block_size
            dst_kwargs["NUM_THREADS"] = str(num_threads)

            with rasterio.open(dst_path, "w", **dst_kwargs) as dst:
                for b in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, b),
                        destination=rasterio.band(dst, b),
                        resampling=resampling,
                        src_nodata=src_nodata,
                        dst_nodata=dst_nodata,
                        num_threads=num_threads,
                        warp_mem_limit=warp_mem_limit_mb,
                    )


def _raise_if_geographic_coords_outside_degree_range(gdf: gpd.GeoDataFrame) -> None:
    """Detect invalid GeoJSON: geographic CRS metadata but projected metre coordinates."""
    if gdf.empty or gdf.crs is None or not gdf.crs.is_geographic:
        return
    try:
        w, s, e, n = gdf.total_bounds
    except Exception:
        return
    if not np.isfinite((w, s, e, n)).all():
        return
    tol = 1e-3
    if (
        abs(w) > 180.0 + tol
        or abs(e) > 180.0 + tol
        or abs(s) > 90.0 + tol
        or abs(n) > 90.0 + tol
    ):
        raise ValueError(
            "This file declares a geographic CRS (e.g. OGC:CRS84 / EPSG:4326), but "
            "its coordinates are outside valid degree ranges (longitude ±180°, "
            "latitude ±90°). The numbers look like **projected metres** (e.g. "
            "EPSG:3978) while the GeoJSON `crs` field was set to a lon/lat CRS "
            "without transforming the geometry — the file is internally inconsistent.\n\n"
            "Fix when exporting: either keep a CRS-aware format (GeoPackage), or "
            "reproject to WGS84 before GeoJSON:\n"
            '  gdf.to_crs("EPSG:4326").to_file(..., driver="GeoJSON")\n'
            "If coordinates are EPSG:3978 but CRS was lost, assign then reproject:\n"
            '  gdf.set_crs("EPSG:3978").to_crs("EPSG:4326").to_file(...)'
        )


def _geographic_to_epsg4326_always_xy(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject geographic CRS (incl. OGC:CRS84) to EPSG:4326 with lon/lat as x/y."""
    if gdf.empty:
        return gdf.copy()
    if gdf.crs is None:
        return gdf.set_crs(WGS84_EPSG)
    transformer = Transformer.from_crs(gdf.crs, WGS84_EPSG, always_xy=True)
    geom_col = gdf.geometry.name

    if len(gdf) > 0 and (gdf.geometry.geom_type == "Point").all():
        x = gdf.geometry.x.to_numpy(dtype=float, copy=False)
        y = gdf.geometry.y.to_numpy(dtype=float, copy=False)
        nx, ny = transformer.transform(x, y)
        return gpd.GeoDataFrame(
            gdf.drop(columns=geom_col),
            geometry=gpd.points_from_xy(nx, ny),
            crs=WGS84_EPSG,
        )

    def _tf(xx: float, yy: float, zz: float | None = None) -> tuple[float, float]:
        rx, ry = transformer.transform(xx, yy)
        return (rx, ry)

    new_geom = gdf.geometry.map(
        lambda g: (
            shapely_xy_transform(_tf, g) if g is not None and not g.is_empty else g
        )
    )
    return gpd.GeoDataFrame(
        gdf.drop(columns=geom_col),
        geometry=new_geom,
        crs=WGS84_EPSG,
    )


def normalize_geographic_gdf_to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Normalize **geographic** CRS to EPSG:4326; leave **projected** CRS unchanged.

    Use for metric GeoJSON that may be UTM (keep) or CRS84/4326 (normalize).
    """
    if gdf.empty:
        return gdf.copy()
    if gdf.crs is None:
        out = gdf.set_crs(WGS84_EPSG)
        _raise_if_geographic_coords_outside_degree_range(out)
        return out
    if not gdf.crs.is_geographic:
        return gdf
    _raise_if_geographic_coords_outside_degree_range(gdf)
    return _geographic_to_epsg4326_always_xy(gdf)


def crs_uses_metre_axes(crs) -> bool:
    """True if ``crs`` is projected and planar axes use the metre unit."""
    if crs is None:
        return False
    try:
        c = PyProjCRS.from_user_input(crs)
    except Exception:
        return False
    if not c.is_projected:
        return False
    axes = c.axis_info
    if not axes:
        return False
    for ax in axes:
        unit = getattr(ax, "unit_name", None) or ""
        if unit != "metre":
            return False
    return True


def _estimate_utm_epsg_from_wgs84_centroid(gdf_wgs84: gpd.GeoDataFrame) -> str:
    """Fallback UTM EPSG code from geometry centroid in lon/lat degrees."""
    u = gdf_wgs84.geometry.union_all()
    if u is None or u.is_empty:
        raise ValueError("Cannot estimate UTM for empty geometry.")
    lon, lat = float(u.centroid.x), float(u.centroid.y)
    zone = int((lon + 180.0) / 6.0) + 1
    zone = min(60, max(1, zone))
    epsg = 32600 + zone if lat >= 0.0 else 32700 + zone
    return f"EPSG:{epsg}"


def estimate_metre_projected_crs_for_gdf(gdf: gpd.GeoDataFrame) -> str | PyProjCRS:
    """Pick a projected CRS for metre-accurate planar ops (buffer, length).

    Prefer GeoPandas ``estimate_utm_crs`` from the extent; fall back to UTM from the
    WGS84 centroid if estimation fails.
    """
    try:
        est = gdf.estimate_utm_crs()
        if est is not None:
            return est
    except Exception:
        pass
    g_ll = gdf if gdf.crs and gdf.crs.is_geographic else gdf.to_crs(WGS84_EPSG)
    if g_ll.crs is None:
        g_ll = g_ll.set_crs(WGS84_EPSG)
    return _estimate_utm_epsg_from_wgs84_centroid(g_ll)


def _build_lcc(minx: float, maxx: float, miny: float, maxy: float) -> PyProjCRS:
    """Two-parallel Lambert Conformal Conic, parallels by Kavraisky's rule."""
    h = maxy - miny
    lat_1 = miny + h / 6.0
    lat_2 = maxy - h / 6.0
    lat_0 = (miny + maxy) / 2.0
    lon_0 = (minx + maxx) / 2.0
    return PyProjCRS.from_proj4(
        f"+proj=lcc +lat_1={lat_1} +lat_2={lat_2} +lat_0={lat_0} +lon_0={lon_0} "
        f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )


def _measure_planar_distortion(
    crs: PyProjCRS, minx: float, maxx: float, miny: float, maxy: float
) -> float:
    """Max relative deviation from 1.0 of a 1000 m planar step vs. true geodesic
    distance, sampled on a 3x3 grid across the WGS84 extent."""
    fwd = Transformer.from_crs(WGS84_EPSG, crs, always_xy=True)
    inv = Transformer.from_crs(crs, WGS84_EPSG, always_xy=True)
    geod = Geod(ellps="WGS84")
    lons = np.linspace(minx, maxx, 3)
    lats = np.linspace(miny, maxy, 3)
    max_dev = 0.0
    for slon in lons:
        for slat in lats:
            x0, y0 = fwd.transform(slon, slat)
            x1, y1 = x0 + 1000.0, y0
            lon1, lat1 = inv.transform(x1, y1)
            _, _, dist = geod.inv(slon, slat, lon1, lat1)
            dev = abs(dist / 1000.0 - 1.0)
            if dev > max_dev:
                max_dev = dev
    return float(max_dev)


def select_grid_crs(gdf: gpd.GeoDataFrame) -> tuple[PyProjCRS, float, str]:
    """Pick a projected metre CRS for intermediate grid math from data extent.

    Decision tree (on WGS84 bbox):
      * Centroid |lat| > 75 deg          -> polar stereographic at the relevant pole.
      * Lon span <= 6 deg and lat span <= 8 deg -> single UTM zone via ``estimate_utm_crs``.
      * Otherwise                        -> two-parallel LCC centred on the data
        (Kavraisky standard parallels: lat_min + h/6, lat_max - h/6).

    Returns ``(crs, max_distortion, choice_name)``. ``max_distortion`` is the
    largest relative error of a 1000 m planar step vs. true geodesic distance,
    sampled across the extent; callers should warn when this exceeds ~0.02.
    """
    if gdf.empty:
        raise ValueError("Cannot select grid CRS for empty GeoDataFrame.")

    if gdf.crs is None:
        gdf_ll = gdf.set_crs(WGS84_EPSG)
    elif gdf.crs.is_geographic:
        gdf_ll = gdf
    else:
        gdf_ll = gdf.to_crs(WGS84_EPSG)

    minx, miny, maxx, maxy = gdf_ll.total_bounds
    if not np.isfinite([minx, miny, maxx, maxy]).all():
        raise ValueError("Cannot select grid CRS: non-finite bounds.")
    cent_lat = (miny + maxy) / 2.0
    span_lon = maxx - minx
    span_lat = maxy - miny

    if abs(cent_lat) > 75.0:
        if cent_lat >= 0:
            crs = PyProjCRS.from_proj4(
                "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=0 +x_0=0 +y_0=0 "
                "+datum=WGS84 +units=m +no_defs"
            )
            name = "Polar Stereographic (North)"
        else:
            crs = PyProjCRS.from_proj4(
                "+proj=stere +lat_0=-90 +lat_ts=-70 +lon_0=0 +x_0=0 +y_0=0 "
                "+datum=WGS84 +units=m +no_defs"
            )
            name = "Polar Stereographic (South)"
    elif span_lon <= 6.0 and span_lat <= 8.0:
        try:
            crs = gdf_ll.estimate_utm_crs()
            name = f"UTM ({getattr(crs, 'name', str(crs))})"
        except Exception:
            crs = _build_lcc(minx, maxx, miny, maxy)
            name = "Lambert Conformal Conic (UTM fallback)"
    else:
        crs = _build_lcc(minx, maxx, miny, maxy)
        name = "Lambert Conformal Conic"

    distortion = _measure_planar_distortion(crs, minx, maxx, miny, maxy)
    return crs, distortion, name


def buffer_gdf_union_metres(
    gdf: gpd.GeoDataFrame,
    buffer_m: float,
) -> gpd.GeoDataFrame:
    """Single-feature GeoDataFrame: ``union_all(gdf).buffer(buffer_m)`` in metres.

    Distance is never applied in geographic degrees: buffering uses either the
    dataset CRS when it is already projected with metre axes, or an estimated UTM
    CRS, then results are reprojected back to ``gdf.crs`` (or EPSG:4326 if unset).
    """
    if buffer_m <= 0:
        return gdf
    if gdf.empty:
        return gdf.copy()

    orig_crs = gdf.crs
    if orig_crs is not None and crs_uses_metre_axes(orig_crs):
        merged = gdf.geometry.union_all().buffer(buffer_m)
        return gpd.GeoDataFrame({"geometry": [merged]}, crs=orig_crs)

    gdf_work = gdf.set_crs(WGS84_EPSG) if orig_crs is None else gdf
    utm = estimate_metre_projected_crs_for_gdf(gdf_work)
    gdf_utm = gdf_work.to_crs(utm)
    merged = gdf_utm.geometry.union_all().buffer(buffer_m)
    out = gpd.GeoDataFrame({"geometry": [merged]}, crs=gdf_utm.crs)
    if orig_crs is None:
        return out.to_crs(WGS84_EPSG)
    return out.to_crs(orig_crs)


def reproject_geodataframe_to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Put every feature in EPSG:4326 (x=lon°, y=lat°) for engines and map previews."""
    if gdf.empty:
        return gdf.copy()
    if gdf.crs is None:
        out = gdf.set_crs(WGS84_EPSG)
        _raise_if_geographic_coords_outside_degree_range(out)
        return out
    if gdf.crs.is_geographic:
        _raise_if_geographic_coords_outside_degree_range(gdf)
        return _geographic_to_epsg4326_always_xy(gdf)
    return gdf.to_crs(WGS84_EPSG)
