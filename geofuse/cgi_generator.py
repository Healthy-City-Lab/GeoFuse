"""Planned: a full-area CGI map from a finished study's definition.

A study fixes everything a CGI value needs: the channels, each channel's
radius and statistic, the formula and its weights. The map a study writes
covers only the cells its targets reach. The generator will lay the same CGI
grid over a whole area, compute every cell's channel values with the study's
radii and statistics, and apply its formula, so the index can be mapped where
no target sits and reused by other studies of the same area.

Not available yet: :func:`generate` raises :class:`NotImplementedError`.
"""

from __future__ import annotations

#: Whether :func:`generate` is implemented.
AVAILABLE = False


def generate(
    params_path: str,
    area_path: str,
    output_path: str,
    *,
    cgi_grid_spacing_m: float | None = None,
) -> str:
    """Write the CGI map of ``area_path`` for the study in ``params_path``.

    ``params_path`` is a study's ``*_params.json``; ``cgi_grid_spacing_m``
    defaults to the study's own grid. Returns ``output_path``.
    """
    raise NotImplementedError(
        "Generating a CGI map from a study's definition is not available yet."
    )
