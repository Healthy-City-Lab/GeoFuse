"""In-RAM view of a complete pre-aggregation cache.

The on-disk SQLite cache (:mod:`geofuse.preaggregation`) is durable and
resumable, but per-trial lookups against it read whole columns from disk —
seconds each once the working set exceeds the LRU. Since a study revisits the
same ``(channel, wave, radius, stat)`` cells thousands of times across trials
and bootstraps, the whole cache is loaded **once** into a compact float32
structure at engine start, and every trial then gathers from RAM.

Layout (per channel): the values for one wave are stored as a single dense
``float32`` array ``[n_pixels, n_radii, n_stats]`` plus the sorted pixel ids,
so pixel ids are held once per wave (not once per radius) and a lookup is a
``searchsorted`` + gather. The structure is **read-only after loading**, so
concurrent trial threads share it without locks.
"""

from __future__ import annotations

import gc

import numpy as np

from .preaggregation import (
    CHANNELS,
    DEFAULT_WAVE_INDEX,
    STAT_COLUMNS,
    PreAggregationCache,
)

_STAT_INDEX = {c: i for i, c in enumerate(STAT_COLUMNS)}


class InMemoryPreAggregation:
    """Read-only, RAM-resident mirror of a complete ``PreAggregationCache``."""

    def __init__(
        self,
        *,
        gvi_radii: tuple[int, ...],
        ndvi_radii: tuple[int, ...],
        wave_labels: tuple[str, ...],
        wave_alias: dict[tuple[str, int], int],
    ) -> None:
        self.gvi_radii = tuple(int(r) for r in gvi_radii)
        self.ndvi_radii = tuple(int(r) for r in ndvi_radii)
        self.wave_labels = tuple(wave_labels)
        self._wave_alias = dict(wave_alias)
        # (channel, wave) -> (sorted_ids int64[n], values float32[n, n_radii, n_stats])
        self._data: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        self._radius_pos: dict[str, dict[int, int]] = {}
        self.n_bytes: int = 0

    # ------------------------------------------------------------------ build

    @classmethod
    def from_cache(cls, cache: PreAggregationCache) -> "InMemoryPreAggregation":
        """Load every stored ``(channel, wave, radius)`` cell into RAM.

        One query per ``(channel, wave)`` pulls all radii at once; the rows are
        packed into the dense per-wave array and the transient buffers are freed
        before the next wave, so peak overhead stays near one wave's worth.
        """
        mem = cls(
            gvi_radii=cache.gvi_radii,
            ndvi_radii=cache.ndvi_radii,
            wave_labels=cache.wave_labels,
            wave_alias=dict(cache._wave_alias),
        )
        for ch in CHANNELS:
            radii = mem.radii_for(ch)
            mem._radius_pos[ch] = {int(r): i for i, r in enumerate(radii)}
            # Representative wave indices actually stored for this channel
            # (aliased waves share storage; load each representative once).
            reps = sorted(
                {mem.resolve_wave(ch, i) for i in range(len(mem.wave_labels))}
            )
            for rep in reps:
                dense = cache.read_dense(ch, rep)
                if dense is None:
                    continue
                ids, values = dense
                # The dense array is in id order; the mirror keeps ids sorted so
                # a lookup is a searchsorted + gather. Copy the memmap into RAM
                # (sorted) so trials never touch disk.
                order = np.argsort(ids, kind="stable")
                sorted_ids = np.ascontiguousarray(ids[order].astype(np.int64))
                out = np.ascontiguousarray(
                    np.asarray(values, dtype=np.float32)[order]
                )
                mem._data[(ch, int(rep))] = (sorted_ids, out)
                mem.n_bytes += sorted_ids.nbytes + out.nbytes
            gc.collect()
        return mem

    # -------------------------------------------------------------- accessors

    def radii_for(self, channel: str) -> tuple[int, ...]:
        return self.ndvi_radii if channel == "ndvi" else self.gvi_radii

    def resolve_wave(self, channel: str, wave_index: int) -> int:
        return self._wave_alias.get((channel, int(wave_index)), int(wave_index))

    def lookup(
        self,
        entity_ids: np.ndarray,
        channel: str,
        radius: int,
        column: str,
        *,
        wave_index: int = DEFAULT_WAVE_INDEX,
    ) -> np.ndarray | None:
        """float32 values for ``entity_ids`` at one cell, or ``None`` if the
        cell is not stored (caller falls back). Missing ids come back NaN.
        Matches :meth:`PreAggregationCache.lookup` so callers are unchanged."""
        col_idx = _STAT_INDEX.get(column)
        if col_idx is None:
            return None
        radius = int(radius)
        rpos = self._radius_pos.get(channel)
        if rpos is None or radius not in rpos:
            return None
        rep = self.resolve_wave(channel, wave_index)
        entry = self._data.get((channel, rep))
        if entry is None:
            return None
        sorted_ids, values = entry
        req = np.asarray(entity_ids).astype(np.int64, copy=False)
        out = np.full(req.shape[0], np.nan, dtype=np.float32)
        if not sorted_ids.size:
            return out
        pos = np.searchsorted(sorted_ids, req)
        pos = np.clip(pos, 0, sorted_ids.size - 1)
        valid = sorted_ids[pos] == req
        out[valid] = values[pos[valid], rpos[radius], col_idx]
        return out

    def close(self) -> None:
        """Release the RAM held by the structure."""
        self._data.clear()
        self._radius_pos.clear()
        self.n_bytes = 0
        gc.collect()
