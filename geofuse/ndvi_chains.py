"""NDVI aggregation chains: the grammar the engine, settings and tabs share.

A chain is an ordered list of operations, each a short string, with exactly
one temporal step. Operations before it act on every scene, those after it
on the composite:

* ``focal:<mean|max|min|median>:<radius_m>``: a statistic over a circular
  neighbourhood of ``radius_m`` metres (above 0, at most 5,000), as Earth
  Engine's ``reduceNeighborhood`` computes it by default: a normalised
  kernel, and a masked centre pixel stays masked;
* ``temporal:<median|mean|max>``: the statistic over the window's scenes;
* ``water_mask``: mask pixels open water covers at least half the time.

Each chain has a name of letters, digits and ``_``, which its files carry.
Without chains a date mode writes one set of files per temporal statistic,
each the chain ``["temporal:<stat>"]`` (after a per-scene ``water_mask`` when
the water mask is on). Pure Python, so settings, run files and the tabs can
check chains without loading Earth Engine.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

#: Temporal statistics (``geofuse.ndvi.TEMPORAL_REDUCERS``).
TEMPORAL = ("median", "mean", "max")
#: Statistics a ``focal`` step can take.
FOCAL = ("mean", "max", "min", "median")
#: Largest neighbourhood radius, in metres.
RADIUS_MAX_M = 5000
WATER_MASK = "water_mask"

_NAME = re.compile(r"^[A-Za-z0-9_]+$")
_FORMS = "focal:<stat>:<radius_m>, temporal:<stat> or water_mask"


class ChainError(ValueError):
    """A chain that cannot be used, with the chain named."""


# ────────────────────────────────────────────────────────────────────
# Checking
# ────────────────────────────────────────────────────────────────────


def _radius_text(radius: float) -> str:
    return str(int(radius)) if radius.is_integer() else repr(radius)


def _step(text: Any) -> str:
    """One operation in its canonical spelling, or ``ValueError`` saying why."""
    if not isinstance(text, str):
        raise ValueError(f"an operation is text, not {text!r}")
    shown = text.strip()
    parts = [p.strip().lower() for p in shown.split(":")]
    if parts == [WATER_MASK]:
        return WATER_MASK
    if parts[0] == "temporal" and len(parts) == 2:
        if parts[1] not in TEMPORAL:
            raise ValueError(
                f"'{shown}': the temporal statistic is one of {list(TEMPORAL)}"
            )
        return f"temporal:{parts[1]}"
    if parts[0] == "focal" and len(parts) == 3:
        if parts[1] not in FOCAL:
            raise ValueError(
                f"'{shown}': the neighbourhood statistic is one of {list(FOCAL)}"
            )
        try:
            radius = float(parts[2])
        except ValueError:
            raise ValueError(f"'{shown}': the radius is a number of metres") from None
        if not 0 < radius <= RADIUS_MAX_M:
            raise ValueError(
                f"'{shown}': the radius must be above 0 and at most "
                f"{RADIUS_MAX_M:,} m"
            )
        return f"focal:{parts[1]}:{_radius_text(radius)}"
    raise ValueError(f"unknown operation '{shown}' (use {_FORMS})")


def _chain(item: Any, index: int) -> dict:
    """``{"name", "steps"}`` from a ``{name, steps}`` table or a
    ``"name: op, op"`` line."""
    if isinstance(item, str):
        name, sep, rest = item.partition(":")
        if not sep:
            raise ChainError(
                f"Chain {index + 1} ({item.strip()!r}): write it as "
                "'name: operation, operation, ...'."
            )
        name = name.strip()
        steps: Any = [s.strip() for s in rest.split(",") if s.strip()]
    elif isinstance(item, Mapping):
        extra = sorted(set(item) - {"name", "steps"})
        if extra:
            raise ChainError(
                f"Chain {index + 1}: unknown key(s) {extra}; a chain has "
                "'name' and 'steps'."
            )
        name, steps = item.get("name"), item.get("steps")
    else:
        raise ChainError(
            f"Chain {index + 1} must be a {{name, steps}} table or "
            f"'name: operation, ...' text, not {item!r}."
        )
    named = isinstance(name, str) and name
    label = f"Chain {name!r}" if named else f"Chain {index + 1}"
    if not isinstance(name, str) or not _NAME.match(name):
        raise ChainError(f"{label}: a name uses letters, digits and '_' only.")
    if not isinstance(steps, list) or not steps:
        raise ChainError(f"{label}: needs a list of operations ({_FORMS}).")
    try:
        canonical = [_step(s) for s in steps]
    except ValueError as exc:
        raise ChainError(f"{label}: {exc}.") from None
    n_temporal = sum(s.startswith("temporal:") for s in canonical)
    if n_temporal != 1:
        found = "none" if not n_temporal else f"{n_temporal}"
        raise ChainError(
            f"{label}: needs exactly one temporal step (temporal:median, "
            f"temporal:mean or temporal:max); it has {found}."
        )
    return {"name": name, "steps": canonical}


def check_chains(value: Any) -> list[dict]:
    """``value`` as checked chains in canonical spelling.

    ``value`` lists ``{name, steps}`` tables or ``"name: op, op"`` lines;
    ``None`` or an empty list means no chains. Names must differ even
    ignoring case, since they end up in file names. Raises
    :class:`ChainError` naming the chain.
    """
    if value is None:
        return []
    if not isinstance(value, list | tuple):
        raise ChainError(f"Aggregation chains are a list, not {value!r}.")
    chains = [_chain(item, i) for i, item in enumerate(value)]
    seen: dict[str, str] = {}
    for c in chains:
        key = c["name"].lower()
        if key in seen:
            raise ChainError(
                f"Chain {c['name']!r}: another chain is named {seen[key]!r}; "
                "names must differ, ignoring case."
            )
        seen[key] = c["name"]
    return chains


def parse_lines(text: str) -> list[dict]:
    """Chains from text, one ``name: op, op`` per line; blank lines and
    lines starting with ``#`` are skipped."""
    lines = (line.strip() for line in str(text or "").splitlines())
    return check_chains([line for line in lines if line and not line.startswith("#")])


def format_lines(chains: Iterable[Mapping]) -> str:
    """The text :func:`parse_lines` reads back as ``chains``."""
    return "\n".join(f"{c['name']}: {', '.join(c['steps'])}" for c in chains)


# ────────────────────────────────────────────────────────────────────
# Reading checked chains
# ────────────────────────────────────────────────────────────────────


def statistic_chain(statistic: str, water_mask: bool = False) -> list[str]:
    """The chain a temporal statistic computes on its own."""
    return ([WATER_MASK] if water_mask else []) + [f"temporal:{statistic}"]


def as_statistic(steps: Sequence[str]) -> tuple[str, bool] | None:
    """``(statistic, water_mask)`` when ``steps`` is a statistic's own chain
    (:func:`statistic_chain`), else ``None``."""
    *before, last = steps
    if last.startswith("temporal:") and before in ([], [WATER_MASK]):
        return last.split(":")[1], bool(before)
    return None


def temporal_statistic(steps: Sequence[str]) -> str:
    """The statistic of the chain's temporal step."""
    return next(s.split(":")[1] for s in steps if s.startswith("temporal:"))


def focal_radius(chains: Iterable[Sequence[str]]) -> float:
    """The largest neighbourhood radius (m) among the chains' steps; 0 if none."""
    radii = [
        float(s.split(":")[2])
        for steps in chains
        for s in steps
        if s.startswith("focal:")
    ]
    return max(radii, default=0.0)


def output_stems(
    output_name: str,
    statistics: Iterable[str] = ("median",),
    chains: Iterable[Mapping] = (),
) -> dict[str, str]:
    """Each output's base name, keyed by chain name or, without chains, by
    statistic. A chain appends its name; a statistic other than the median
    appends its own (``median`` keeps ``output_name``)."""
    chains = list(chains)
    if chains:
        return {c["name"]: f"{output_name}_{c['name']}" for c in chains}
    return {
        s: output_name if s == "median" else f"{output_name}_{s}" for s in statistics
    }
