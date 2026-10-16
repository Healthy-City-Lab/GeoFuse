"""Exposure weights of target geometries on the CGI grid.

An entity is exposed to the CGI cells its geometry touches, and optionally to
the cells within a buffer around it. Every element of a geometry (each part of
a Multi* geometry, each member of a collection) contributes on its own, and the
contributions add up, so a cell touched twice counts twice:

- **points** count once in every cell whose closed square holds the point, so a
  point on a cell edge counts in both cells and one on a corner in all four;
- **lines** count once per *pass*: each separate stretch of the line inside a
  cell's closed square, so a line that leaves a cell and comes back counts
  twice, and a line running along an edge counts in both cells;
- **polygons** count once per part in every cell they share area with; contact
  along an edge or at a corner alone does not count.

A buffer widens the reach to the cells within ``radius_m`` of an element, each
contribution weighted by a kernel of the distance from the cell's square to the
element (0 when it touches): ``uniform`` (1 up to the radius), ``linear`` (1 up
to ``plateau_m``, then falling to 0 at the radius) or ``gaussian``
(``exp(−½ (d / sigma_m)²)``, cut at the radius). ``none`` is the touch rule
alone.

The grid is the global lattice anchored at the CRS origin that the engine's
pre-aggregation cache keys on: cell ``(row, col)`` is the square
``[col·s, (col + 1)·s] × [−(row + 1)·s, −row·s]`` for cell size ``s``, rows
increasing southwards. Entities never take cells from each other; a cell shared
by several entities is part of each one's exposure in full.
"""

from __future__ import annotations

import numpy as np
import shapely

KERNELS: tuple[str, ...] = ("none", "uniform", "linear", "gaussian")

# Candidate cells evaluated per vectorised chunk of points.
_POINT_CHUNK_CELLS = 2_000_000


# ────────────────────────────────────────────────────────────────────
# Kernel
# ────────────────────────────────────────────────────────────────────


def check_kernel(
    kernel: str, radius_m: float, plateau_m: float, sigma_m: float
) -> None:
    """Raise ``ValueError`` naming the setting a buffer definition gets wrong."""
    if kernel not in KERNELS:
        raise ValueError(f"exposure_kernel must be one of {KERNELS}; got {kernel!r}.")
    if kernel == "none":
        return
    if not radius_m > 0:
        raise ValueError(f"exposure_radius_m must be positive; got {radius_m}.")
    if kernel == "linear" and not 0 <= plateau_m <= radius_m:
        raise ValueError(
            f"exposure_plateau_m must lie in [0, exposure_radius_m]; got {plateau_m}."
        )
    if kernel == "gaussian" and not sigma_m > 0:
        raise ValueError(f"exposure_sigma_m must be positive; got {sigma_m}.")


def kernel_weights(
    d: np.ndarray, kernel: str, radius_m: float, plateau_m: float, sigma_m: float
) -> np.ndarray:
    """Weight of a contribution at distance ``d`` (metres) from the cell."""
    d = np.asarray(d, dtype=np.float64)
    if kernel in ("none", "uniform"):
        return np.ones_like(d)
    if kernel == "linear":
        span = radius_m - plateau_m
        if span <= 0:
            return np.ones_like(d)
        return np.clip((radius_m - d) / span, 0.0, 1.0)
    return np.exp(-0.5 * (d / sigma_m) ** 2)


# ────────────────────────────────────────────────────────────────────
# Per-element cell sets
# ────────────────────────────────────────────────────────────────────


def _cell_boxes(rows: np.ndarray, cols: np.ndarray, s: float) -> np.ndarray:
    return shapely.box(cols * s, -(rows + 1) * s, (cols + 1) * s, -rows * s)


def _bbox_cells(bounds, s: float, reach: float) -> tuple[np.ndarray, np.ndarray]:
    """Every cell whose square can lie within ``reach`` of ``bounds``."""
    minx, miny, maxx, maxy = bounds
    c = np.arange(
        int(np.floor((minx - reach) / s)) - 1, int(np.floor((maxx + reach) / s)) + 2
    )
    r = np.arange(
        int(np.floor(-(maxy + reach) / s)) - 1, int(np.floor(-(miny - reach) / s)) + 2
    )
    cc, rr = np.meshgrid(c, r, indexing="xy")
    return rr.ravel().astype(np.int64), cc.ravel().astype(np.int64)


def _point_cells(x, y, owner, s, reach):
    """``(owner, row, col, distance)`` for every point/cell pair within reach."""
    k = int(np.ceil(reach / s)) + 1
    offsets = np.arange(-k, k + 1, dtype=np.int64)
    c0 = np.floor(x / s).astype(np.int64)
    r0 = np.floor(-y / s).astype(np.int64)
    per_point = len(offsets) ** 2
    step = max(1, _POINT_CHUNK_CELLS // per_point)
    out = []
    for a in range(0, len(x), step):
        b = a + step
        cc = (c0[a:b, None, None] + offsets[None, None, :]).repeat(len(offsets), 1)
        rr = (r0[a:b, None, None] + offsets[None, :, None]).repeat(len(offsets), 2)
        px = x[a:b, None, None]
        py = y[a:b, None, None]
        dx = np.maximum(np.maximum(cc * s - px, px - (cc + 1) * s), 0.0)
        dy = np.maximum(np.maximum(-(rr + 1) * s - py, py + rr * s), 0.0)
        d = np.hypot(dx, dy)
        keep = d <= reach
        own = np.broadcast_to(owner[a:b, None, None], d.shape)
        out.append((own[keep], rr[keep], cc[keep], d[keep]))
    return out


def _line_cells(part, owner, s, reach):
    """``(owner, row, col, distance)`` for every pass of a line near a cell."""
    rows, cols = _bbox_cells(part.bounds, s, reach)
    boxes = _cell_boxes(rows, cols, s)
    zone = part if reach == 0 else shapely.buffer(part, reach)
    shapely.prepare(zone)
    near = shapely.intersects(zone, boxes)
    if not near.any():
        return None
    rows, cols, boxes = rows[near], cols[near], boxes[near]
    zones = boxes if reach == 0 else shapely.buffer(boxes, reach)
    inter = shapely.intersection(part, zones)
    pieces, idx = shapely.get_parts(inter, return_index=True)
    kind = shapely.get_type_id(pieces)
    # Line pieces that meet end to end are one pass; a touch with no length
    # (a corner, or a point on an edge) is a pass of its own unless it lies
    # on a line piece of the same cell.
    lines = np.full(len(boxes), None, dtype=object)
    is_line = kind == 1
    if is_line.any():
        used, dense = np.unique(idx[is_line], return_inverse=True)
        lines[used] = shapely.multilinestrings(pieces[is_line], indices=dense)
    merged = shapely.line_merge(lines)
    comps, cidx = shapely.get_parts(merged, return_index=True)
    is_point = kind == 0
    lone = is_point.copy()
    lone[is_point] = ~shapely.intersects(pieces[is_point], merged[idx[is_point]])
    geoms = np.concatenate([comps, pieces[lone]])
    cell = np.concatenate([cidx, idx[lone]])
    if not len(geoms):
        return None
    d = shapely.distance(geoms, boxes[cell])
    own = np.full(len(cell), owner, dtype=np.int64)
    return own, rows[cell], cols[cell], d


def _polygon_cells(part, owner, s, reach):
    """``(owner, row, col, distance)`` for every cell a polygon part reaches."""
    rows, cols = _bbox_cells(part.bounds, s, reach)
    boxes = _cell_boxes(rows, cols, s)
    shapely.prepare(part)
    if reach == 0:
        cx = (cols + 0.5) * s
        cy = -(rows + 0.5) * s
        hit = shapely.contains_xy(part, cx, cy)
        edge = ~hit & shapely.intersects(part, boxes)
        if edge.any():
            hit[edge] = shapely.relate_pattern(boxes[edge], part, "2********")
        d = np.zeros(int(hit.sum()))
    else:
        d_all = shapely.distance(boxes, part)
        hit = d_all <= reach
        d = d_all[hit]
    if not hit.any():
        return None
    return np.full(int(hit.sum()), owner, dtype=np.int64), rows[hit], cols[hit], d


# ────────────────────────────────────────────────────────────────────
# Public entry point
# ────────────────────────────────────────────────────────────────────


def _elements(geoms) -> tuple[np.ndarray, np.ndarray]:
    """Single-part elements of ``geoms`` and the entity each belongs to."""
    parts, owner = shapely.get_parts(np.asarray(geoms, dtype=object), return_index=True)
    # A collection may hold Multi* members; flatten until every element is
    # a single point, line or polygon.
    while True:
        multi = shapely.get_type_id(parts) >= 4
        if not multi.any():
            break
        sub, sub_idx = shapely.get_parts(parts[multi], return_index=True)
        parts = np.concatenate([parts[~multi], sub])
        owner = np.concatenate([owner[~multi], owner[multi][sub_idx]])
    keep = ~shapely.is_empty(parts)
    return parts[keep], owner[keep].astype(np.int64)


def cell_weights(
    geoms,
    cell_size_m: float,
    *,
    kernel: str = "none",
    radius_m: float = 100.0,
    plateau_m: float = 0.0,
    sigma_m: float = 35.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(entity, row, col, weight)`` of every entity–cell exposure.

    ``geoms`` are the entities' geometries in a metre CRS, one per entity in
    order; ``entity`` indexes them. Each ``(entity, row, col)`` appears once,
    its weight summed over every element and pass that reaches the cell. An
    entity whose geometry is empty has no rows.
    """
    check_kernel(kernel, radius_m, plateau_m, sigma_m)
    s = float(cell_size_m)
    reach = 0.0 if kernel == "none" else float(radius_m)
    parts, owner = _elements(geoms)
    kind = shapely.get_type_id(parts)

    chunks = []
    pts = kind == 0
    if pts.any():
        chunks.extend(
            _point_cells(
                shapely.get_x(parts[pts]),
                shapely.get_y(parts[pts]),
                owner[pts],
                s,
                reach,
            )
        )
    for part, own, k in zip(parts[~pts], owner[~pts], kind[~pts]):
        handler = _polygon_cells if k == 3 else _line_cells
        got = handler(part, own, s, reach)
        if got is not None:
            chunks.append(got)
    if not chunks:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty, np.zeros(0, dtype=np.float32)

    ent, rows, cols, d = (np.concatenate(c) for c in zip(*chunks))
    w = kernel_weights(d, kernel, radius_m, plateau_m, sigma_m)
    order = np.lexsort((cols, rows, ent))
    ent, rows, cols, w = ent[order], rows[order], cols[order], w[order]
    new = np.r_[True, (np.diff(ent) != 0) | (np.diff(rows) != 0) | (np.diff(cols) != 0)]
    starts = np.flatnonzero(new)
    return (
        ent[starts],
        rows[starts],
        cols[starts],
        np.add.reduceat(w, starts).astype(np.float32),
    )
