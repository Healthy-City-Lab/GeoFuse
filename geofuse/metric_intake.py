"""Where a metric value comes from: which column, and which file.

Two questions the fusion engine asks of every greenery metric, answered here
because they are the same question at two scales.

**Which column holds a channel.** Resolution is rigid: a channel is recognised
only by the names listed for it, and no name is shared between channels. There
is no generic fallback and no "first numeric column" guess. The strictness is
what makes a combined file safe to read — a GVI file carries vegetation and
terrain together, so a generically named column (``value``, or a bare ``gvi``)
cannot say which channel it holds, and accepting one would let a terrain column
be read as vegetation. A file must name its columns explicitly to be read.

**Which file serves a temporal key.** A longitudinal study has one greenery
file per channel per temporal key (the measurement year, or the wave label when
waves are not date-derived). Holding every one open at once costs hundreds of
megabytes per file; a decade of waves does not fit that way.
:class:`LongitudinalMetricSources` resolves ``(channel, key) -> source`` by
loading on demand and keeps only what the caller currently holds: callers load
one key's sources, use them, then ``release()`` before moving on. File identity
(``identity``) is path plus size plus mtime, so fingerprinting never requires
reading the file.
"""

from __future__ import annotations

from .vector_io import match_column_alias
from collections.abc import Callable, Iterable, Mapping
from collections.abc import Iterable, Mapping
from typing import Any
import os


VEG_COLUMNS: tuple[str, ...] = ("gvi_veg", "veg")

# Terrain-only GVI, the independent Cityscapes class written alongside ``veg``.
TERRAIN_COLUMNS: tuple[str, ...] = ("gvi_ter", "terrain")

# NDVI, as NDVIEngine names its vector chunks and raster band.
NDVI_COLUMNS: tuple[str, ...] = ("ndvi",)

CHANNEL_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "veg": VEG_COLUMNS,
    "terrain": TERRAIN_COLUMNS,
    "ndvi": NDVI_COLUMNS,
}


def _assert_channels_disjoint() -> None:
    """Guard the invariant the rest of this module's strictness rests on."""
    seen: dict[str, str] = {}
    for channel, names in CHANNEL_COLUMNS.items():
        for name in names:
            key = name.lower()
            if key in seen:
                raise RuntimeError(
                    f"Greenery column alias {name!r} is claimed by both "
                    f"{seen[key]!r} and {channel!r}; channels must stay "
                    "distinguishable."
                )
            seen[key] = channel


_assert_channels_disjoint()


def channel_columns(channel: str) -> tuple[str, ...]:
    """Accepted column names for *channel*, in resolution order."""
    try:
        return CHANNEL_COLUMNS[channel]
    except KeyError:
        raise ValueError(
            f"Unknown greenery channel {channel!r}; expected one of "
            f"{list(CHANNEL_COLUMNS)}."
        ) from None


def resolve_channel_column(columns: Iterable[str], channel: str) -> str:
    """The column in *columns* holding *channel*'s values.

    Matches case-insensitively against :func:`channel_columns`. Raises
    ``ValueError`` naming the accepted spellings when none is present, rather
    than falling back to a column that may belong to another channel.
    """
    names = channel_columns(channel)
    available = [str(c) for c in columns]
    col = match_column_alias(available, names)
    if col is None:
        raise ValueError(
            f"No {channel!r} column found. Expected one of {list(names)}; got "
            f"{available}."
        )
    return col


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


def _close_source(src: Any) -> None:
    if isinstance(src, dict):
        data = src.get("data")
        closer = getattr(data, "close_all", None)
        if callable(closer):
            closer()
