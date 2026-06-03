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

    Seven weights ``w_*`` constrained to sum to 100 (same integer 0–100 scale
    as ``weighted_average`` for consistency in the trial DB — the user's
    post-hoc weight-vs-association analysis pools both formulas without a
    scale conversion), plus three powers ``pN`` / ``pV`` / ``pT``. The
    ``compute`` function divides each weight by their total so the final
    composite is on a [0, 1]-ish range. AHP-set in the paper; optimizer-driven
    here.

Weight sampling
---------------

Every formula's weight suggestion draws from a **symmetric Dirichlet(1, …, 1)
prior** — i.e. uniform on the simplex, no per-channel ordering bias. The
implementation uses the Gamma-normalize trick: an ``Exp(1)`` sample (computed
as ``-log(U(0, 1))``) is drawn per weight key, then the values are normalised
to sum to ``total``. By symmetry every weight has the same marginal
distribution.

Optuna integration: the raw uniform draws live in ``trial.params`` under
``<key>_raw`` so TPE has flat per-dimension axes to model; the normalised
**integer** weight is then *pinned* via ``trial.suggest_int(value, value)`` so
``trial.params`` still carries the canonical ``*_weight`` keys that downstream
Optuna plots, robust-trial reports, and weight-vs-association CSV analyses
expect. TPE searches over the ``*_raw`` axes; the pinned weights are
bookkeeping.

Edge cases: the raw float is clamped to ``[1e-9, 1 - 1e-9]`` so ``-log(u)``
never hits ``±inf`` or ``0``. Largest-remainder rounding keeps the integer
weights summing to exactly ``total``.
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
# 0.4–0.7 with AHP-fixed values; widened to 0.2–1.0 so the optimizer can pick
# anything from a strongly concave transform down to the "no transform"
# endpoint (1.0) if the data prefers a plain weighted sum + interactions. Step
# of 0.1 makes the grid match the pre-aggregation cache's stat resolution
# (percentiles in 10 % steps) and gives TPE clean ordinal bins for the
# weight-vs-association post-hoc analysis. Adjustable from one place.
SYNERGY_POWER_LOW: float = 0.2
SYNERGY_POWER_HIGH: float = 1.0
SYNERGY_POWER_STEP: float = 0.1

# Numeric floor used when clamping channel values before raising to a
# fractional power so float roundoff in the min-max normalisation can't
# produce ``NaN`` (e.g. ``(-1e-12)**0.5``).
_CLAMP_LO = 0.0
_CLAMP_HI = 1.0


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
# Symmetric Dirichlet(1,…,1) sampling (uniform on the simplex)
# ---------------------------------------------------------------------------

# Floor for the raw uniform draws so ``-log(u)`` stays finite and the
# normalising sum is never zero. The symmetric upper clamp keeps every key's
# marginal exactly identical.
_DIRICHLET_EPS: float = 1e-9


def _suggest_simplex_weights_dirichlet(
    trial: optuna.Trial, keys: tuple[str, ...], total: int = 100
) -> dict[str, int]:
    """Sample ``keys`` from symmetric Dirichlet(1, …, 1), rounded to integers
    summing exactly to ``total``.

    Each key gets the SAME marginal distribution (Beta(1, K-1)) by the
    Gamma-normalize construction's symmetry, so no key is favoured by the
    prior. TPE optimises against the per-key ``<key>_raw`` axes recorded by
    ``trial.suggest_float`` (flat U(eps, 1-eps), which TPE can model cleanly);
    the integer weight is then pinned via ``trial.suggest_int(w, w)`` so the
    canonical ``*_weight`` keys still appear in ``trial.params`` for Optuna
    plots, robust-trial reports, and weight-vs-association CSV readers.
    """
    import math as _math

    raw_xs: list[float] = []
    for k in keys:
        u = trial.suggest_float(
            f"{k}_raw", _DIRICHLET_EPS, 1.0 - _DIRICHLET_EPS
        )
        raw_xs.append(-_math.log(u))
    sum_x = sum(raw_xs)
    if sum_x <= 0.0:
        # Defensive: all u_i hit the upper clamp. Fall back to equal weights.
        scaled = [float(total) / len(keys)] * len(keys)
    else:
        scaled = [x / sum_x * total for x in raw_xs]

    # Largest-remainder rounding so the integer weights sum to ``total`` exactly.
    floors = [int(_math.floor(s)) for s in scaled]
    remainder = int(total) - sum(floors)
    if remainder > 0:
        order = sorted(
            range(len(keys)), key=lambda i: scaled[i] - floors[i], reverse=True
        )
        for i in order[:remainder]:
            floors[i] += 1

    out: dict[str, int] = {}
    for k, w in zip(keys, floors):
        out[k] = trial.suggest_int(k, int(w), int(w))
    return out


# ---------------------------------------------------------------------------
# weighted_average formula
# ---------------------------------------------------------------------------

_WA_WEIGHT_KEYS: tuple[str, ...] = ("ndvi_weight", "veg_weight", "terrain_weight")


def _suggest_weighted_average(trial: optuna.Trial) -> dict:
    return _suggest_simplex_weights_dirichlet(trial, _WA_WEIGHT_KEYS, total=100)


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

# Key ordering is informational only — the symmetric Dirichlet sampler treats
# every key identically, so the (main NDVI / Veg / Ter, then pairwise NV / NT
# / TV, then triple) order doesn't bias which terms get larger prior mass.
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
    # pattern generalised to three metrics. Stepped grid keeps the parameter
    # ordinal (TPE exploits the order; categorical wouldn't) and reflects the
    # coarse-grained nature of the synergy concavity choice.
    for k in _SYN_POWER_KEYS:
        params[k] = trial.suggest_float(
            k, SYNERGY_POWER_LOW, SYNERGY_POWER_HIGH, step=SYNERGY_POWER_STEP
        )
    # Same int 0–100 scale as ``weighted_average`` so the two formulas share
    # one downstream trial-DB convention; ``_compute_synergy`` self-renormalizes
    # by their actual total before evaluating the composite.
    params.update(
        _suggest_simplex_weights_dirichlet(trial, _SYN_WEIGHT_KEYS, total=100)
    )
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

    # Self-renormalize by the actual weight sum (100 for trial-recorded ints,
    # 1.0 for hand-built float params) so the composite is on a comparable
    # scale regardless of which convention the caller used — same pattern as
    # ``_compute_weighted_average``.
    total = wN + wV + wT + wNV + wNT + wTV + wNVT
    if total <= 0:
        return np.full_like(veg, np.nan, dtype=np.float64)
    inv = 1.0 / total

    return inv * (
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
    contributes nothing" optimisation. With seven int weights summing to 100
    the all-zero case is vanishingly rare for synergy, so the engine generally
    aggregates every channel anyway.
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
