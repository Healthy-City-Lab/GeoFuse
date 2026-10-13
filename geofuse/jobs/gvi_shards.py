"""GVI runs split over several machines: shards and their merge.

A GVI step with ``shards = N`` becomes N shard jobs and one merge job. A shard
samples only its blocks of the sampling grid (:func:`geofuse.core.shard_mask`)
and keeps its raw results — ``<name>_gvi.shard<i>of<N>.gpkg`` — beside a small
manifest. The merge waits for every manifest, rebuilds the grid (the same input
gives the same grid on any machine), writes the outputs an unsharded run
writes, and removes the shard files.
"""

from __future__ import annotations

import json
import os

import geopandas as gpd
import pandas as pd

from ..logger import get_logger

SHARD_LAYER = "gvi_shard"

_log_gvi = get_logger("GVI")


class ShardsMissing(RuntimeError):
    """The merge found shards that have not finished."""


def shard_label(shard) -> str:
    """``shard<i>of<N>`` for the 0-based ``(i, N)``."""
    index, count = shard
    return f"shard{int(index) + 1}of{int(count)}"


def _manifest(folder: str, stem: str, label: str) -> str:
    return os.path.join(folder, f"{stem}.{label}.json")


def write_shard(
    results: dict[str, list],
    crs,
    folder: str,
    stem: str,
    shard,
    points: int,
    throughput: list | None = None,
) -> list[str]:
    """Keep one shard's raw results: ``{name}.{label}.gpkg`` for each result
    set that has rows, and the manifest ``{stem}.{label}.json`` (with the
    engine's stage counts of each run, ``throughput``)."""
    label = shard_label(shard)
    os.makedirs(folder, exist_ok=True)
    written, counts = [], {}
    for name, records in results.items():
        counts[name] = len(records)
        if not records:
            continue
        path = os.path.join(folder, f"{name}.{label}.gpkg")
        frame = gpd.GeoDataFrame(records, crs=crs)
        frame.to_file(path, driver="GPKG", layer=SHARD_LAYER)
        written.append(path)
    manifest = _manifest(folder, stem, label)
    with open(manifest, "w", encoding="utf-8") as f:
        record = {
            "shard": int(shard[0]) + 1,
            "shards": int(shard[1]),
            "points": int(points),
            "results": counts,
            "throughput": [t for t in throughput or [] if t],
        }
        json.dump(record, f, indent=2)
    return [*written, manifest]


def _records(frame: gpd.GeoDataFrame) -> list[dict]:
    """Result records as the engine produced them (missing values ``None``)."""
    data = pd.DataFrame(frame.drop(columns="geometry"))
    data = data.astype(object).where(data.notna(), None)
    rows = data.to_dict("records")
    for row, geometry in zip(rows, frame.geometry, strict=True):
        row["geometry"] = geometry
    return rows


def read_shards(folder: str, stem: str, shards: int) -> dict[str, tuple[list, object]]:
    """Every shard's results by result set: ``{name: (records, crs)}``.

    Raises :class:`ShardsMissing` naming the shards without a manifest.
    """
    labels = [shard_label((i, shards)) for i in range(shards)]
    missing = [
        label for label in labels if not os.path.isfile(_manifest(folder, stem, label))
    ]
    if missing:
        raise ShardsMissing(
            f"{len(missing)} of {shards} shards of {stem} have not finished: "
            + ", ".join(missing)
        )
    merged: dict[str, tuple[list, object]] = {}
    for label in labels:
        with open(_manifest(folder, stem, label), encoding="utf-8") as f:
            counts = json.load(f)["results"]
        for name, count in counts.items():
            if not count:
                merged.setdefault(name, ([], None))
                continue
            frame = gpd.read_file(
                os.path.join(folder, f"{name}.{label}.gpkg"), layer=SHARD_LAYER
            )
            records, crs = merged.get(name, ([], None))
            merged[name] = (records + _records(frame), crs or frame.crs)
    return merged


def shard_throughput(folder: str, stem: str, shards: int) -> list[dict]:
    """Every shard's recorded stage counts, shard by shard."""
    found = []
    for i in range(shards):
        path = _manifest(folder, stem, shard_label((i, shards)))
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        for stats in record.get("throughput") or []:
            found.append({"shard": record["shard"], **stats})
    return found


def _throughput(folder: str, stem: str, shards: int) -> dict | None:
    found = shard_throughput(folder, stem, shards)
    return {"shards": found} if found else None


def remove_shards(folder: str, stem: str, names, shards: int) -> None:
    for i in range(shards):
        label = shard_label((i, shards))
        for path in [_manifest(folder, stem, label)] + [
            os.path.join(folder, f"{name}.{label}.gpkg") for name in names
        ]:
            if os.path.exists(path):
                os.remove(path)


def run_gvi_merge(
    ctx,
    *,
    fname: str,
    dataset_data: dict,
    run_args: dict,
    output_dir: str,
    shards: int,
    date_column: str | None = None,
    save_gpkg: bool = True,
    save_geotiff: bool = False,
    save_geojson: bool = False,
) -> dict:
    """Merge a sharded GVI run into the outputs an unsharded run writes."""
    from geofuse.core import gvi_sampling_points

    from . import runners

    shards = int(shards)
    step = float(run_args["step"])
    buffer_m = float(run_args.get("buffer") or 0)
    raw = dataset_data["raw"]
    write = {
        "save_gpkg": save_gpkg,
        "save_geotiff": save_geotiff,
        "save_geojson": save_geojson,
    }
    ctx.progress(status_text=f"Merging {shards} shards...")

    if date_column:
        base = os.path.splitext(fname)[0]
        folder = os.path.join(output_dir, f"{base}_temporal_gvi")
        merged = read_shards(folder, f"{base}_gvi", shards)
        throughput = _throughput(folder, f"{base}_gvi", shards)
        gdf_4326, years, grid_crs = runners.column_years(raw, date_column)
        outputs: list[str] = [folder]
        for year in years:
            name = f"{base}_{year}_gvi"
            records, crs = merged.get(name, ([], None))
            if not records:
                _log_gvi("WARN", f"Year {year}: no results produced.")
                continue
            _pts, meta = runners.year_grid(gdf_4326, year, buffer_m, step, grid_crs)
            paths, _ = runners._write_gvi_outputs(
                accumulated=records,
                processed_crs=crs,
                meta=meta or {},
                out_name=f"{base}_{year}",
                output_dir=folder,
                capture=runners._capture_settings("column", year, run_args),
                throughput=throughput,
                **write,
            )
            outputs.extend(paths)
        remove_shards(folder, f"{base}_gvi", merged, shards)
    else:
        out_name = runners.gvi_output_name(fname, run_args)
        stem = f"{out_name}_gvi"
        merged = read_shards(output_dir, stem, shards)
        throughput = _throughput(output_dir, stem, shards)
        records, crs = merged.get(stem, ([], None))
        outputs = []
        if not records:
            _log_gvi("WARN", "No panorama found in any shard; no outputs written.")
        else:
            processed, meta = gvi_sampling_points(
                raw, dataset_data.get("type") == "poly", buffer_m, step
            )
            target_year = run_args.get("target_year")
            outputs, _ = runners._write_gvi_outputs(
                accumulated=records,
                processed_crs=crs or processed.crs,
                meta=meta or {},
                out_name=out_name,
                output_dir=output_dir,
                capture=runners._capture_settings(
                    "recent" if target_year is None else "year", target_year, run_args
                ),
                throughput=throughput,
                **write,
            )
        remove_shards(output_dir, stem, merged, shards)
    ctx.progress(value=1.0, status_text="Merged")
    return {"output_paths": outputs}
