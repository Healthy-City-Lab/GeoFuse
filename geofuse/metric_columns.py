"""Column names that identify each greenery channel in a vector metric file.

Resolution is rigid: a channel is recognised only by the names listed for it,
and no name is shared between channels. There is no generic fallback and no
"first numeric column" guess.

The strictness is what makes a combined file safe to read. A GVI file always
carries vegetation and terrain together, so a generically named column
(``value``, or a bare ``gvi``) cannot say which channel it holds; accepting one
would let a terrain column be read as vegetation. A file must name its columns
explicitly to be read at all.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .vector_io import match_column_alias

# Vegetation-only GVI. ``gvi_veg`` is what GVIEngine writes; ``veg`` is what
# the fusion engine names it when it splits a combined file into per-channel
# caches.
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
