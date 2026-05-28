"""Composite Greenery Index (CGI) formula registry.

Single source of truth for how an Optuna trial's per-trial parameters become a
numeric greenery value at each sample. The fusion engine's objective, test
evaluation, ``apply_fusion``, and composite-raster generation all route
through :func:`compute_cgi` so the four call sites can never drift apart.

Formulas
--------

``weighted_average`` (legacy default)
    Three weights on min-max-normalized channel values. Backward-compatible
    with all pre-overhaul trials, which recorded ``ndvi_weight`` /
    ``veg_weight`` / ``terrain_weight`` as integers summing to 100. The
    formula renormalizes by their total so older trials replay identically.

``synergy``
    Three-metric generalisation of Wang et al. 2026 (doi:10.3390/rs18010009),
    paper-faithful in pattern: powers on the **main** terms only, plain
    products elsewhere::

        CGI = w_N · NDVI^pN + w_V · Veg^pV + w_T · Ter^pT
            + w_NV · NDVI · Veg
            + w_NT · NDVI · Ter
            + w_TV · Ter  · Veg
            + w_NVT · NDVI · Veg · Ter

    Seven weights ``w_*`` constrained to sum to 1, plus three powers
    ``pN`` / ``pV`` / ``pT``. AHP-set in the paper; optimizer-driven here.

Weight sampling
---------------

Every formula's weight suggestion uses **sequential conditional allocation on
the simplex**: each weight is suggested with bounds that depend on the
previously-suggested weights, and the *last* weight is pinned to the remainder
via ``trial.suggest_int(rem, rem)`` / ``trial.suggest_float(rem, rem)`` so it
is recorded as a real trial parameter. This guarantees **recorded params ==
used params** — the user does post-hoc weight-vs-association analyses on the
recorded trial parameters, so a "sample raw then normalize" gap would silently
break the analysis. *Caveat:* sequential allocation is order-dependent (a mild
prior bias toward larger early weights); ordering is fixed (main terms first,
then pairwise, then triple) and documented per formula.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import optuna

# ---------------------------------------------------------------------------
# Public formula names + parameter spaces
# ---------------------------------------------------------------------------

WEIGHTED_AVERAGE = "weighted_average"
SYNERGY = "synergy"

ALL_CHANNELS: tuple[str, ...] = ("veg", "terrain", "ndvi")

# Power range for the synergy formula's main-term exponents. The paper used
# 0.4–0.7 with AHP-fixed values; widening the upper bound to 1.0 lets the
# optimizer fall back to the "no transform" case when the data prefers a plain
# weighted sum of channels + interactions. Adjustable from one place.
SYNERGY_POWER_LOW: float = 0.4
SYNERGY_POWER_HIGH: float = 1.0

# Numeric floor used when clamping channel values before raising to a
# fractional power so float roundoff in the min-max normalisation can't
# produce ``NaN`` (e.g. ``(-1e-12)**0.5``).
_CLAMP_LO = 0.0
_CLAMP_HI = 1.0
_FLOAT_EPS = 1e-12


# ---------------------------------------------------------------------------
# Formula descriptor
# ---------------------------------------------------------------------------

# Mapping from canonical channel name to its 1-D values array for one batch of
# samples. The engine builds these from its per-fold MinMaxScaler-normalised
# arrays before calling :func:`compute_cgi`.
ComponentDict = dict[str, np.ndarray]


@dataclass(frozen=True)
class CGIFormula:
    """Static descriptor for one CGI formula.

    ``suggest_params`` reads / writes a trial; ``compute`` is pure and used by
    the engine both during optimization and at apply/export time. The
    ``*_keys`` tuples let the UI and reporting code introspect a formula's
    parameter shape without hard-coding any names.
    """

    name: str
    main_weight_keys: tuple[str, ...]
    interaction_weight_keys: tuple[str, ...]
    power_keys: tuple[str, ...]
    suggest_params: Callable[[optuna.Trial], dict]
    compute: Callable[[dict, ComponentDict], np.ndarray]
    channel_active: Callable[[dict], dict[str, bool]]

    @property
    def weight_keys(self) -> tuple[str, ...]:
        return self.main_weight_keys + self.interaction_weight_keys


# ---------------------------------------------------------------------------
# Sequential conditional simplex sampling
# ---------------------------------------------------------------------------


def _suggest_simplex_weights_int(
    trial: optuna.Trial, keys: tuple[str, ...], total: int = 100
) -> dict[str, int]:
    """Suggest ``keys`` as integers in ``[0, remaining]`` summing exactly to ``total``.

    The last key is pinned via ``suggest_int(rem, rem)`` so it is recorded as
    a real parameter rather than implicit — keeps "recorded == used" intact
    for downstream weight-vs-association analyses on the study DB.
    """
    out: dict[str, int] = {}
    remaining = int(total)
    last = len(keys) - 1
    for i, k in enumerate(keys):
        if i == last:
            out[k] = trial.suggest_int(k, remaining, remaining)
        else:
            out[k] = trial.suggest_int(k, 0, remaining)
            remaining -= out[k]
    return out


def _suggest_simplex_weights_float(
    trial: optuna.Trial, keys: tuple[str, ...], total: float = 1.0
) -> dict[str, float]:
    """Float analogue of :func:`_suggest_simplex_weights_int`.

    The last key is pinned via ``suggest_float(rem, rem)``. A tiny epsilon
    bumps the upper bound on intermediate weights to avoid Optuna's empty-
    range error when the running remainder hits exactly 0 partway through.
    """
    out: dict[str, float] = {}
    remaining = float(total)
    last = len(keys) - 1
    for i, k in enumerate(keys):
        if i == last:
            out[k] = trial.suggest_float(k, remaining, remaining)
        else:
            hi = max(_FLOAT_EPS, remaining)
            out[k] = trial.suggest_float(k, 0.0, hi)
            remaining = max(0.0, remaining - out[k])
    return out


# ---------------------------------------------------------------------------
# weighted_average formula
# ---------------------------------------------------------------------------

_WA_WEIGHT_KEYS: tuple[str, ...] = ("ndvi_weight", "veg_weight", "terrain_weight")


def _suggest_weighted_average(trial: optuna.Trial) -> dict:
    return _suggest_simplex_weights_int(trial, _WA_WEIGHT_KEYS, total=100)


def _compute_weighted_average(params: dict, components: ComponentDict) -> np.ndarray:
    veg = np.asarray(components["veg"], dtype=np.float64)
    ter = np.asarray(components["terrain"], dtype=np.float64)
    ndvi = np.asarray(components["ndvi"], dtype=np.float64)
    wv = float(params.get("veg_weight", 0))
    wt = float(params.get("terrain_weight", 0))
    wn = float(params.get("ndvi_weight", 0))
    total = wv + wt + wn
    if total <= 0:
        return np.full_like(veg, np.nan, dtype=np.float64)
    return (wv / total) * veg + (wt / total) * ter + (wn / total) * ndvi


def _wa_channel_active(params: dict) -> dict[str, bool]:
    return {
        "veg": float(params.get("veg_weight", 0)) > 0,
        "terrain": float(params.get("terrain_weight", 0)) > 0,
        "ndvi": float(params.get("ndvi_weight", 0)) > 0,
    }


# ---------------------------------------------------------------------------
# synergy formula
# ---------------------------------------------------------------------------

# Order is fixed: main NDVI / Veg / Ter first, then pairwise (NV / NT / TV),
# then the triple. Sequential allocation biases toward larger early weights, so
# the main terms get the most prior mass — matches the paper, which gives the
# main terms the dominant share of the AHP-set weights.
_SYN_MAIN_KEYS: tuple[str, ...] = ("w_ndvi", "w_veg", "w_ter")
_SYN_INTER_KEYS: tuple[str, ...] = (
    "w_ndvi_veg",
    "w_ndvi_ter",
    "w_ter_veg",
    "w_ndvi_veg_ter",
)
_SYN_WEIGHT_KEYS: tuple[str, ...] = _SYN_MAIN_KEYS + _SYN_INTER_KEYS
_SYN_POWER_KEYS: tuple[str, ...] = ("ndvi_power", "veg_power", "terrain_power")


def _suggest_synergy(trial: optuna.Trial) -> dict:
    params: dict = {}
    # Powers attach only to the main (standalone) terms — paper-faithful
    # pattern generalised to three metrics. Range tuned to include the
    # "no transform" endpoint (1.0) so the optimizer can recover the linear
    # weighted-sum-plus-interactions case if the data prefers it.
    for k in _SYN_POWER_KEYS:
        params[k] = trial.suggest_float(k, SYNERGY_POWER_LOW, SYNERGY_POWER_HIGH)
    params.update(_suggest_simplex_weights_float(trial, _SYN_WEIGHT_KEYS, total=1.0))
    return params


def _compute_synergy(params: dict, components: ComponentDict) -> np.ndarray:
    veg = np.asarray(components["veg"], dtype=np.float64)
    ter = np.asarray(components["terrain"], dtype=np.float64)
    ndvi = np.asarray(components["ndvi"], dtype=np.float64)

    # Channels arrive min-max normalised to [0, 1]; clamp away any float
    # roundoff so a 0.4 power doesn't produce NaN on a slightly-negative input.
    veg = np.clip(veg, _CLAMP_LO, _CLAMP_HI)
    ter = np.clip(ter, _CLAMP_LO, _CLAMP_HI)
    ndvi = np.clip(ndvi, _CLAMP_LO, _CLAMP_HI)

    pN = float(params.get("ndvi_power", 1.0))
    pV = float(params.get("veg_power", 1.0))
    pT = float(params.get("terrain_power", 1.0))

    main_n = ndvi**pN
    main_v = veg**pV
    main_t = ter**pT

    wN = float(params["w_ndvi"])
    wV = float(params["w_veg"])
    wT = float(params["w_ter"])
    wNV = float(params["w_ndvi_veg"])
    wNT = float(params["w_ndvi_ter"])
    wTV = float(params["w_ter_veg"])
    wNVT = float(params["w_ndvi_veg_ter"])

    return (
        wN * main_n
        + wV * main_v
        + wT * main_t
        + wNV * ndvi * veg
        + wNT * ndvi * ter
        + wTV * ter * veg
        + wNVT * ndvi * veg * ter
    )


def _synergy_channel_active(params: dict) -> dict[str, bool]:
    """A channel is active iff any term that involves it has non-zero weight.

    Mostly informational for the engine's "skip aggregation when channel
    contributes nothing" optimisation. With seven weights summing to 1 in
    continuous floats the all-zero case is vanishingly rare for synergy, so
    the engine generally aggregates every channel anyway.
    """
    n = (
        float(params.get("w_ndvi", 0)),
        float(params.get("w_ndvi_veg", 0)),
        float(params.get("w_ndvi_ter", 0)),
        float(params.get("w_ndvi_veg_ter", 0)),
    )
    v = (
        float(params.get("w_veg", 0)),
        float(params.get("w_ndvi_veg", 0)),
        float(params.get("w_ter_veg", 0)),
        float(params.get("w_ndvi_veg_ter", 0)),
    )
    t = (
        float(params.get("w_ter", 0)),
        float(params.get("w_ndvi_ter", 0)),
        float(params.get("w_ter_veg", 0)),
        float(params.get("w_ndvi_veg_ter", 0)),
    )
    return {
        "veg": any(x > 0 for x in v),
        "terrain": any(x > 0 for x in t),
        "ndvi": any(x > 0 for x in n),
    }


# ---------------------------------------------------------------------------
# Registry + public API
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, CGIFormula] = {
    WEIGHTED_AVERAGE: CGIFormula(
        name=WEIGHTED_AVERAGE,
        main_weight_keys=_WA_WEIGHT_KEYS,
        interaction_weight_keys=(),
        power_keys=(),
        suggest_params=_suggest_weighted_average,
        compute=_compute_weighted_average,
        channel_active=_wa_channel_active,
    ),
    SYNERGY: CGIFormula(
        name=SYNERGY,
        main_weight_keys=_SYN_MAIN_KEYS,
        interaction_weight_keys=_SYN_INTER_KEYS,
        power_keys=_SYN_POWER_KEYS,
        suggest_params=_suggest_synergy,
        compute=_compute_synergy,
        channel_active=_synergy_channel_active,
    ),
}


def available_formulas() -> list[str]:
    """Stable-ordered list of registered formula names."""
    return sorted(_REGISTRY)


def get_formula(name: str) -> CGIFormula:
    """Return the formula descriptor for ``name`` or raise ``ValueError``."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown CGI formula '{name}'. Expected one of {available_formulas()}."
        ) from exc


def compute_cgi(formula: str, params: dict, components: ComponentDict) -> np.ndarray:
    """Evaluate ``formula`` at ``components`` using the trial-recorded ``params``.

    ``components`` must contain keys ``veg`` / ``terrain`` / ``ndvi`` mapped to
    equally-shaped 1-D float arrays. ``params`` is the dict of trial parameters
    as Optuna recorded them (the engine reads them straight off ``trial.params``
    after optimization), and the formula's :func:`compute` does any
    renormalization or clamping it needs.
    """
    return get_formula(formula).compute(params, components)


def channel_active(formula: str, params: dict) -> dict[str, bool]:
    """Which channels contribute under ``formula`` for the given ``params``."""
    return get_formula(formula).channel_active(params)
