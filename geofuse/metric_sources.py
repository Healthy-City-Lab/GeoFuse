"""On-demand access to the per-temporal-key greenery metric files.

A longitudinal study has one greenery file per channel per temporal key (the
measurement year, or the wave label when waves are not date-derived). Holding
every one of them open at once costs hundreds of megabytes per file; a study
with a decade of waves does not fit in memory that way.

:class:`LongitudinalMetricSources` resolves ``(channel, key) -> source`` by
loading from disk on demand and keeps only what the caller currently holds, so
a pass over the keys never has more than one key's files resident. Callers load
the sources for one key, use them, then :meth:`release` before moving on.

File identity (:meth:`identity`) is derived from the path plus its size and
mtime, so grouping keys that share a file — and fingerprinting the cache —
never require reading the file itself.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from typing import Any


class LongitudinalMetricSources:
    """Lazily-loaded metric sources keyed by ``(channel, temporal key)``."""

    def __init__(
        self,
        paths: Mapping[str, Mapping[str, str]],
        loader: Callable[[str, str], Any],
    ) -> None:
        self._paths: dict[str, dict[str, str]] = {
            str(ch): {str(k): str(p) for k, p in per_key.items()}
            for ch, per_key in paths.items()
        }
        self._loader = loader
        # (channel, path) -> loaded source. Keyed by path rather than temporal
        # key so two keys sharing a file share one load.
        self._open: dict[tuple[str, str], Any] = {}
        self._identity_cache: dict[str, str] = {}

    # ------------------------------------------------------------------ info

    @property
    def channels(self) -> list[str]:
        return list(self._paths)

    def keys_for(self, channel: str) -> list[str]:
        return list(self._paths.get(channel, {}))

    def has_channel(self, channel: str) -> bool:
        return channel in self._paths

    def path(self, channel: str, key: str) -> str | None:
        return self._paths.get(channel, {}).get(key)

    def identity(self, channel: str, key: str) -> str:
        """Cheap content-independent identity for the file behind a key.

        Path plus size plus mtime: enough to tell two keys apart, and to
        invalidate a cache when a file is replaced, without opening it.
        """
        path = self.path(channel, key)
        if path is None:
            return f"{channel}@{key}:missing"
        cached = self._identity_cache.get(path)
        if cached is None:
            try:
                st = os.stat(path)
                cached = f"{os.path.abspath(path)}:{st.st_size}:{st.st_mtime_ns}"
            except OSError:
                cached = f"{os.path.abspath(path)}:unstatable"
            self._identity_cache[path] = cached
        return cached

    def group_keys_by_identity(
        self, channel: str, keys: Iterable[str]
    ) -> dict[str, list[str]]:
        """Temporal keys of *channel* bucketed by the file they resolve to.

        Keys in one bucket read the same file, so whatever is computed from it
        holds for all of them.
        """
        groups: dict[str, list[str]] = {}
        for key in keys:
            groups.setdefault(self.identity(channel, key), []).append(str(key))
        return groups

    # ------------------------------------------------------------- retrieval

    def get(self, channel: str, key: str) -> Any:
        """The source for ``(channel, key)``, loading it if not already held."""
        path = self.path(channel, key)
        if path is None:
            return None
        cache_key = (channel, path)
        src = self._open.get(cache_key)
        if src is None:
            src = self._loader(path, channel)
            self._open[cache_key] = src
        return src

    def get_all(self, key: str, channels: Iterable[str]) -> dict[str, Any]:
        """Sources for one temporal key across *channels*."""
        return {ch: self.get(ch, key) for ch in channels}

    # --------------------------------------------------------------- release

    def release(self) -> None:
        """Drop every source currently held, closing raster file handles.

        Windows keeps a file locked until its handle is closed, so raster
        sources are closed explicitly rather than left to collection.
        """
        for src in self._open.values():
            _close_source(src)
        self._open.clear()

    def __enter__(self) -> LongitudinalMetricSources:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


class PreloadedMetricSources(LongitudinalMetricSources):
    """Adapter for callers that already hold every source in memory.

    Keeps the same interface so the engine has one code path, but never loads
    or frees anything: :meth:`release` is a no-op because the caller, not this
    object, owns the sources.
    """

    def __init__(self, data: Mapping[str, Mapping[str, Any]]) -> None:
        super().__init__({}, lambda _p, _c: None)
        self._data: dict[str, dict[str, Any]] = {
            str(ch): dict(per_key) for ch, per_key in data.items()
        }
        self._paths = {ch: {} for ch in self._data}

    @property
    def channels(self) -> list[str]:
        return list(self._data)

    def keys_for(self, channel: str) -> list[str]:
        return list(self._data.get(channel, {}))

    def has_channel(self, channel: str) -> bool:
        return channel in self._data

    def path(self, channel: str, key: str) -> str | None:
        return None

    def identity(self, channel: str, key: str) -> str:
        src = self._data.get(channel, {}).get(key)
        return f"{channel}@obj:{id(src)}"

    def get(self, channel: str, key: str) -> Any:
        return self._data.get(channel, {}).get(key)

    def release(self) -> None:
        return None


def _close_source(src: Any) -> None:
    if isinstance(src, dict):
        data = src.get("data")
        closer = getattr(data, "close_all", None)
        if callable(closer):
            closer()
