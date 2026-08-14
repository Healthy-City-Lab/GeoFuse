import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from rasterio import features
from rasterio.transform import from_bounds, from_origin, xy
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import transform as shapely_transform
from shapely.strtree import STRtree

from .crs_utils import (
    WGS84_EPSG,
    buffer_union_planar,
    metres_per_degree_at_lat,
    select_grid_crs_with_warning,
)
from .logger import get_logger

_log_core = get_logger("GVI")

# Inside-buffer vs outside-buffer in the raster mask (any value other than *fill* works).
_RASTER_INSIDE = 1
_RASTER_OUTSIDE = 0

# Candidate cell centres tested per ``contains_xy`` call. A cluster's bbox can
# hold tens of millions of cells (a metro at 20 m step already exceeds 4M), and
# testing them in one shot means holding two float64 coordinate arrays plus the
# boolean result for the whole bbox at once. Blocking by grid rows caps that
# working set at a few tens of MB no matter how large the cluster is.
_GRID_TEST_BLOCK_CELLS = 1_000_000


def _geometry_union_all(geoms: gpd.GeoSeries):
    if hasattr(geoms, "union_all"):
        return geoms.union_all()
    return geoms.unary_union


def generate_raster_grid(gdf_4326, spacing_meters):
    """Build a regular lon/lat grid over the union of *gdf_4326*, then keep cell centres inside it.

    Steps: axis-aligned bbox of the union of geometries (callers typically pass buffered
    features), raster at *spacing_meters* with value ``_RASTER_INSIDE`` under the union and
    ``_RASTER_OUTSIDE`` elsewhere, then emit points at pixel centres for inside cells.

    Returns ``(GeoDataFrame with row, col, x, y, geometry``, metadata dict with
    ``transform``, ``width``, ``height``, ``crs``).
    """
    if gdf_4326 is None or gdf_4326.empty:
        empty = gpd.GeoDataFrame(
            columns=["row", "col", "x", "y"], geometry=[], crs="EPSG:4326"
        )
        ident = from_bounds(0, 0, 1, 1, 1, 1)
        return empty, {
            "transform": ident,
            "width": 1,
            "height": 1,
            "crs": "EPSG:4326",
        }

    geom_union = _geometry_union_all(gdf_4326.geometry)
    if geom_union is None or geom_union.is_empty:
        empty = gpd.GeoDataFrame(
            columns=["row", "col", "x", "y"], geometry=[], crs="EPSG:4326"
        )
        ident = from_bounds(0, 0, 1, 1, 1, 1)
        return empty, {
            "transform": ident,
            "width": 1,
            "height": 1,
            "crs": "EPSG:4326",
        }

    minx, miny, maxx, maxy = geom_union.bounds
    center_lat = (miny + maxy) / 2.0
    m_per_deg_lon, m_per_deg_lat = metres_per_degree_at_lat(center_lat)
    res_x = spacing_meters / m_per_deg_lon
    res_y = spacing_meters / m_per_deg_lat
    width = max(1, int(np.ceil(float(maxx - minx) / float(res_x))))
    height = max(1, int(np.ceil(float(maxy - miny) / float(res_y))))

    transform = from_bounds(
        minx, miny, minx + (width * res_x), miny + (height * res_y), width, height
    )

    mask = features.rasterize(
        [(geom_union, _RASTER_INSIDE)],
        out_shape=(height, width),
        transform=transform,
        fill=_RASTER_OUTSIDE,
        dtype=np.uint8,
        all_touched=True,
    )

    rows, cols = np.nonzero(mask == _RASTER_INSIDE)
    if rows.size == 0:
        gdf_pts = gpd.GeoDataFrame(
            columns=["row", "col", "x", "y"], geometry=[], crs="EPSG:4326"
        )
    else:
        xs, ys = xy(transform, rows, cols, offset="center")
        x_arr = np.asarray(xs, dtype=np.float64).ravel()
        y_arr = np.asarray(ys, dtype=np.float64).ravel()
        df = pd.DataFrame(
            {
                "row": rows.astype(np.int64),
                "col": cols.astype(np.int64),
                "x": x_arr,
                "y": y_arr,
            }
        )
        gdf_pts = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:4326"
        )

    return gdf_pts, {
        "transform": transform,
        "width": width,
        "height": height,
        "crs": "EPSG:4326",
    }


def _anchored_cells_inside(
    poly,
    x0: float,
    y0: float,
    step_m: float,
    col_min: int,
    col_max: int,
    row_min: int,
    row_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cells of the anchored grid whose centre falls inside ``poly``.

    Returns ``(rows, cols, xs, ys)`` for the surviving cells. Membership is
    decided with ``shapely.contains_xy`` against a prepared polygon: it reads
    raw coordinate arrays, so no Python-level ``Point`` is ever built for a
    candidate — the previous ``MultiPoint`` + ``STRtree`` formulation
    materialised one shapely object *and* one index entry per candidate cell,
    which is where a metro-scale grid spent 250 s and 1.8 GB.

    Candidates are produced one row-block at a time so peak memory tracks the
    block size rather than the cluster's bbox.
    """
    shapely.prepare(poly)
    cols_range = np.arange(col_min, col_max, dtype=np.int64)
    xs_row = x0 + (cols_range + 0.5) * step_m
    rows_per_block = max(1, _GRID_TEST_BLOCK_CELLS // max(1, cols_range.size))

    rows_out: list[np.ndarray] = []
    cols_out: list[np.ndarray] = []
    xs_out: list[np.ndarray] = []
    ys_out: list[np.ndarray] = []
    for block_start in range(row_min, row_max, rows_per_block):
        block_rows = np.arange(
            block_start, min(block_start + rows_per_block, row_max), dtype=np.int64
        )
        ys_col = y0 - (block_rows + 0.5) * step_m
        x_flat = np.broadcast_to(xs_row, (block_rows.size, xs_row.size)).ravel()
        y_flat = np.repeat(ys_col, xs_row.size)
        hit = shapely.contains_xy(poly, x_flat, y_flat)
        if not hit.any():
            continue
        rows_out.append(np.repeat(block_rows, cols_range.size)[hit])
        cols_out.append(np.tile(cols_range, block_rows.size)[hit])
        xs_out.append(x_flat[hit])
        ys_out.append(y_flat[hit])

    empty_i = np.empty(0, dtype=np.int64)
    empty_f = np.empty(0, dtype=np.float64)
    if not rows_out:
        return empty_i, empty_i, empty_f, empty_f
    return (
        np.concatenate(rows_out),
        np.concatenate(cols_out),
        np.concatenate(xs_out),
        np.concatenate(ys_out),
    )


def generate_clustered_grid(
    gdf_4326: gpd.GeoDataFrame,
    buffer_m: float,
    step_m: float,
    anchor: tuple[float, float] = (0.0, 0.0),
    grid_crs=None,
):
    """Cluster-aware anchored sampling grid for nation-scale point inputs.

    Buffers + unions input geometries in an auto-selected planar CRS, splits
    the result into connected components, and generates a ``step_m``-spaced
    grid per cluster snapped to a single global anchor in the planar CRS.
    Sample points across all clusters land on one unified grid so no
    resampling is needed if downstream code rasterizes the output.

    Pass ``grid_crs`` to skip CRS auto-selection and force a specific planar
    CRS — used when several subsets (e.g. per-year slices of one dataset)
    must share one CRS so their grids and rasters align pixel-for-pixel.

    Returns ``(GeoDataFrame, meta)``:
      * GeoDataFrame columns ``row, col, x, y, cluster_id`` and Point geometry
        in EPSG:4326. ``row`` / ``col`` index the *global* anchored grid in the
        planar CRS — they are stable across clusters and runs given the same
        anchor + step.
      * ``meta`` dict with ``grid_crs_wkt``, ``anchor_x``, ``anchor_y``,
        ``step_m``, ``distortion``, ``choice_name``, and ``clusters`` (one
        entry per cluster: bounds, height, width, transform — used by the
        per-cluster GeoTIFF writer).
    """
    if grid_crs is not None:
        distortion, choice_name = 0.0, "forced"
    else:
        grid_crs, distortion, choice_name = select_grid_crs_with_warning(
            gdf_4326, _log_core, role="Grid CRS"
        )
    gdf_m = gdf_4326.to_crs(grid_crs)
    buffered = buffer_union_planar(gdf_m.geometry.values, buffer_m)

    x0, y0 = float(anchor[0]), float(anchor[1])
    base_meta: dict = {
        "grid_crs_wkt": grid_crs.to_wkt(),
        "anchor_x": x0,
        "anchor_y": y0,
        "step_m": float(step_m),
        "distortion": float(distortion),
        "choice_name": choice_name,
        "clusters": [],
    }

    if buffered.is_empty:
        empty = gpd.GeoDataFrame(
            columns=["row", "col", "x", "y", "cluster_id"],
            geometry=[],
            crs=WGS84_EPSG,
        )
        return empty, base_meta

    if isinstance(buffered, MultiPolygon):
        cluster_polys = list(buffered.geoms)
    elif isinstance(buffered, Polygon):
        cluster_polys = [buffered]
    else:
        cluster_polys = [
            g for g in getattr(buffered, "geoms", []) if isinstance(g, Polygon)
        ]

    all_rows: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_cid: list[np.ndarray] = []
    cluster_meta: list[dict] = []

    for cid, poly in enumerate(cluster_polys):
        if poly.is_empty:
            continue
        cmin_x, cmin_y, cmax_x, cmax_y = poly.bounds
        col_min = int(np.floor((cmin_x - x0) / step_m))
        col_max = int(np.ceil((cmax_x - x0) / step_m))
        row_min = int(np.floor((y0 - cmax_y) / step_m))
        row_max = int(np.ceil((y0 - cmin_y) / step_m))

        local_width = max(0, col_max - col_min)
        local_height = max(0, row_max - row_min)
        if local_width == 0 or local_height == 0:
            continue

        hit_rows, hit_cols, hit_x, hit_y = _anchored_cells_inside(
            poly, x0, y0, step_m, col_min, col_max, row_min, row_max
        )

        snapped_left = x0 + col_min * step_m
        snapped_top = y0 - row_min * step_m
        local_transform = from_origin(snapped_left, snapped_top, step_m, step_m)
        cluster_meta.append(
            {
                "cluster_id": cid,
                "row_min": int(row_min),
                "row_max": int(row_max),
                "col_min": int(col_min),
                "col_max": int(col_max),
                "height": int(local_height),
                "width": int(local_width),
                "transform": local_transform,
                "bbox_grid_crs": (
                    float(snapped_left),
                    float(snapped_top - local_height * step_m),
                    float(snapped_left + local_width * step_m),
                    float(snapped_top),
                ),
            }
        )

        if hit_rows.size == 0:
            continue

        all_rows.append(hit_rows)
        all_cols.append(hit_cols)
        all_x.append(hit_x)
        all_y.append(hit_y)
        all_cid.append(np.full(hit_rows.size, cid, dtype=np.int64))

    base_meta["clusters"] = cluster_meta

    if not all_x:
        empty = gpd.GeoDataFrame(
            columns=["row", "col", "x", "y", "cluster_id"],
            geometry=[],
            crs=WGS84_EPSG,
        )
        return empty, base_meta

    rows_arr = np.concatenate(all_rows).astype(np.int64)
    cols_arr = np.concatenate(all_cols).astype(np.int64)
    x_arr = np.concatenate(all_x)
    y_arr = np.concatenate(all_y)
    cid_arr = np.concatenate(all_cid).astype(np.int64)

    # Transform the coordinate arrays, not a GeoDataFrame: building points in
    # the grid CRS only to reproject them allocates a second geometry per sample,
    # and a metro-scale grid holds millions.
    to_wgs84 = Transformer.from_crs(grid_crs, WGS84_EPSG, always_xy=True)
    lon_arr, lat_arr = to_wgs84.transform(x_arr, y_arr)

    pts_4326 = gpd.GeoDataFrame(
        {
            "row": rows_arr,
            "col": cols_arr,
            "cluster_id": cid_arr,
            "x": lon_arr,
            "y": lat_arr,
        },
        geometry=gpd.points_from_xy(lon_arr, lat_arr),
        crs=WGS84_EPSG,
    )

    return pts_4326, base_meta


def build_planar_tiles(
    geom_wgs84_gdf: gpd.GeoDataFrame,
    grid_crs,
    max_tile_size_km: float,
) -> list[dict]:
    """Cluster-aware export tile specs in true planar metres.

    The polygon-input counterpart to :func:`generate_clustered_grid`. Where
    that builds a per-cluster *point* grid for sampling, this builds a
    per-cluster *tile rectangle* grid for raster exports (Earth Engine, in
    practice — but the helper itself has no EE coupling).

    Decomposes the geometry into connected polygon components in
    ``grid_crs``, tiles each component's bbox at ``max_tile_size_km`` step
    in true metres, and STRtree-culls candidate tiles that don't intersect
    their cluster polygon. Scattered national-scale inputs (e.g. one feature
    per province) no longer spend export quota on tiles that fall over ocean
    or empty bbox regions. Non-polygon inputs (raw points / lines) degrade
    gracefully to one envelope cluster so the rest of the pipeline still has
    a polygon to tile.

    Returns a list of ``{cluster_id, tile_idx, tile_geom_4326}`` dicts;
    ``tile_geom_4326`` is a shapely polygon (reprojected from ``grid_crs``
    to EPSG:4326) ready to feed into ``ee.Geometry`` or any other consumer.
    """
    tile_size_m = float(max_tile_size_km) * 1000.0
    if tile_size_m <= 0:
        return []
    gdf_planar = geom_wgs84_gdf.to_crs(grid_crs)
    merged = gdf_planar.geometry.union_all()
    if merged is None or merged.is_empty:
        return []

    if isinstance(merged, MultiPolygon):
        cluster_polys: list = list(merged.geoms)
    elif isinstance(merged, Polygon):
        cluster_polys = [merged]
    else:
        env = merged.envelope
        cluster_polys = [env] if not env.is_empty else []

    fwd = Transformer.from_crs(grid_crs, WGS84_EPSG, always_xy=True)

    def _to_4326(box_geom):
        return shapely_transform(lambda x, y, z=None: fwd.transform(x, y), box_geom)

    tiles: list[dict] = []
    tile_idx_global = 0
    for cid, poly in enumerate(cluster_polys):
        if poly.is_empty:
            continue
        cmin_x, cmin_y, cmax_x, cmax_y = poly.bounds
        candidate: list = []
        x = cmin_x
        while x < cmax_x:
            x_end = min(x + tile_size_m, cmax_x)
            y = cmin_y
            while y < cmax_y:
                y_end = min(y + tile_size_m, cmax_y)
                candidate.append(box(x, y, x_end, y_end))
                y = y_end
            x = x_end
        if not candidate:
            continue
        tree = STRtree(candidate)
        keep_idx = tree.query(poly, predicate="intersects")
        for ki in sorted(int(i) for i in keep_idx):
            tiles.append(
                {
                    "cluster_id": cid,
                    "tile_idx": tile_idx_global,
                    "tile_geom_4326": _to_4326(candidate[ki]),
                }
            )
            tile_idx_global += 1

    return tiles
