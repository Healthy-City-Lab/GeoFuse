"""Mercator-aware raster overlays for ``folium.raster_layers.ImageOverlay``.

Leaflet's ``ImageOverlay`` places an image between two lat/lon corner points
and stretches it **linearly in Mercator screen space**.  When the source
pixels are uniformly spaced in latitude (i.e. an EPSG:4326 raster), that
linear stretch displaces every row N-S as latitudes deviate from the
equator — visible as the input data drifting away from the basemap.

The fix is to warp the source into EPSG:3857 with ``Resampling.nearest`` so
the pixels are already Mercator-uniform; the linear stretch then lands each
row at its true geographic position.  ``Resampling.nearest`` is essential
here: bilinear/cubic would average the nodata sentinel with valid values and
produce a fringe of mid-ramp pixels that aren't equal to the sentinel
anymore, leaving a coloured halo around the polygon.

This module is shared by the NDVI and GVI result inspectors.
"""

from __future__ import annotations

import base64
import io

import folium
import numpy as np
import rasterio
from PIL import Image as PILImage
from rasterio.transform import array_bounds
from rasterio.warp import (
    Resampling,
    calculate_default_transform,
    reproject,
    transform_bounds,
)

# Internal sentinel used to round-trip the nodata mask through the
# reprojection step.  Picked far outside any sane NDVI/GVI value range so
# nothing in real data collides with it.
_REPROJECT_NODATA = -9999.0


def add_mercator_image_overlay(
    m: folium.Map,
    arr: np.ndarray,
    *,
    src_transform,
    src_crs,
    cmap,
    vmin: float,
    vmax: float,
    nodata_mask: np.ndarray,
    opacity: float,
    interactive: bool = False,
) -> tuple[float, float, float, float]:
    """Warp ``arr`` to EPSG:3857, colourise, and attach an ImageOverlay.

    Parameters
    ----------
    m : folium.Map
        Map to add the overlay to.
    arr : np.ndarray
        2-D source raster in the CRS described by ``src_crs`` /
        ``src_transform``.  Any dtype castable to float64.
    src_transform : affine.Affine
        Source affine transform (``rasterio.transform.Affine``).
    src_crs : str | rasterio.crs.CRS
        Source CRS (e.g. ``"EPSG:4326"``).
    cmap : matplotlib.colors.Colormap
        Already-resolved matplotlib colormap (e.g. ``plt.get_cmap("RdYlGn")``).
    vmin, vmax : float
        Value range used to normalise ``arr`` to ``[0, 1]`` for the colormap.
        Values outside the range are clipped (not made transparent — that's
        what ``nodata_mask`` is for).
    nodata_mask : np.ndarray of bool
        Same shape as ``arr``; True marks pixels that should be fully
        transparent in the output (raster nodata, NaN, fill sentinels, etc).
    opacity : float
        Alpha applied to **valid** pixels (0-1).  Masked pixels are always
        alpha=0.
    interactive : bool
        Forwarded to ``folium.raster_layers.ImageOverlay`` so the layer can
        be hit-tested (NDVI inspector relies on this for click-to-inspect).

    Returns
    -------
    (west, south, east, north) : tuple of float
        Lat/lon bounding rectangle of the rendered overlay.  Useful for
        ``fit_bounds`` / extent tracking by the caller.
    """
    # 1. Bake the mask into the data array so reproject can carry it through
    #    as a single value.  We copy because callers reuse `arr` afterwards.
    src_arr = np.asarray(arr, dtype=np.float64).copy()
    src_arr[nodata_mask] = _REPROJECT_NODATA

    # 2. Compute the destination grid in EPSG:3857 that matches the source's
    #    native resolution as closely as possible.
    src_h, src_w = src_arr.shape
    src_bounds = array_bounds(src_h, src_w, src_transform)
    dst_crs = "EPSG:3857"
    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, dst_crs, src_w, src_h, *src_bounds
    )
    # rasterio's stub types width/height as Optional but they're always int
    # at runtime.
    dst_w = int(dst_w)  # type: ignore[arg-type]
    dst_h = int(dst_h)  # type: ignore[arg-type]

    # 3. Warp with nearest so the sentinel survives bit-exact (no fringe).
    dst_arr = np.full((dst_h, dst_w), _REPROJECT_NODATA, dtype=np.float64)
    reproject(
        source=src_arr,
        destination=dst_arr,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=_REPROJECT_NODATA,
        dst_nodata=_REPROJECT_NODATA,
        resampling=Resampling.nearest,
    )

    # 4. Mercator extent of the reprojected grid → lat/lon corners for
    #    folium.  ``transform_bounds`` returns the encompassing rectangle in
    #    4326 — exactly the corners Leaflet will project back to the same
    #    Mercator extent we just rendered into.
    x_min = dst_transform.c
    y_max = dst_transform.f
    x_max = x_min + dst_transform.a * dst_w
    y_min = y_max + dst_transform.e * dst_h
    w_lon, s_lat, e_lon, n_lat = transform_bounds(
        dst_crs, "EPSG:4326", x_min, y_min, x_max, y_max
    )

    # 5. Colormap + alpha.  Clip into [vmin, vmax] then normalise to [0,1].
    span = vmax - vmin if vmax > vmin else 1.0
    norm = np.clip((dst_arr - vmin) / span, 0, 1)
    colored = cmap(norm)
    dst_mask = (dst_arr == _REPROJECT_NODATA) | np.isnan(dst_arr)
    colored[..., 3] = np.where(dst_mask, 0, opacity)

    # 6. PNG → data URL → ImageOverlay.
    img_bytes = (colored * 255).astype(np.uint8)
    buff = io.BytesIO()
    PILImage.fromarray(img_bytes).save(buff, format="PNG")
    img_url = (
        f"data:image/png;base64,{base64.b64encode(buff.getvalue()).decode()}"
    )

    folium.raster_layers.ImageOverlay(
        image=img_url,
        bounds=[[s_lat, w_lon], [n_lat, e_lon]],
        opacity=opacity,
        interactive=interactive,
    ).add_to(m)

    return float(w_lon), float(s_lat), float(e_lon), float(n_lat)


def add_mercator_image_overlay_from_file(
    m: folium.Map,
    tif_path: str,
    *,
    band: int = 1,
    cmap,
    vmin: float,
    vmax: float,
    extra_nodata_values: tuple[float, ...] = (),
    opacity: float,
    interactive: bool = False,
) -> tuple[float, float, float, float]:
    """Convenience wrapper that opens a GeoTIFF, reads one band, and builds
    the nodata mask from the file's declared nodata plus any extra
    sentinel values listed in ``extra_nodata_values``.

    NDVI inspector uses this with ``extra_nodata_values=(0,)`` because the
    pipeline writes ``0`` as a second, undeclared sentinel at the polygon
    edge that the file's metadata doesn't flag.
    """
    with rasterio.open(tif_path) as src:
        arr = src.read(band).astype(np.float64)
        src_transform = src.transform
        src_crs = src.crs
        file_nodata = src.nodata

    mask = np.isnan(arr)
    if file_nodata is not None:
        mask |= arr == file_nodata
    for v in extra_nodata_values:
        mask |= arr == v

    return add_mercator_image_overlay(
        m,
        arr,
        src_transform=src_transform,
        src_crs=src_crs,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        nodata_mask=mask,
        opacity=opacity,
        interactive=interactive,
    )
