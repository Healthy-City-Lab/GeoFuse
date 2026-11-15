"""Engine keywords of a basis adjustment named by its rule.

The reporting tests compare the two df rules of the basis adjustment: the
outcome model picking the df (``"outcome"``) and the exposure model picking
it (``"exposure"``). A scale-mode engine shares one basis per row set, which
the engine's ``fixed_scale`` selection gives.
"""

SELECT = {"outcome": "df_outcome_ic", "exposure": "df_exposure_ic"}


def spatial_kwargs(rule: str) -> dict:
    """``spatial_adjust_method`` / ``spatial_adjust_select`` for ``rule``
    (``"outcome"``, ``"exposure"`` or ``"none"``)."""
    if rule == "none":
        return {"spatial_adjust_method": "none"}
    return {"spatial_adjust_method": "basis", "spatial_adjust_select": SELECT[rule]}


def stand_in(rule: str) -> dict:
    """:func:`spatial_kwargs` plus the attributes the engine derives from
    them, for the ``SimpleNamespace`` stand-ins the reporting tests bind
    engine methods to."""
    return {
        **spatial_kwargs(rule),
        "spatial_adjust_select": SELECT.get(rule),
        "_basis": rule,
        "_scoring_spatial_method": "none" if rule == "none" else "basis",
        "spatial_select_criterion": "bic",
        "_in_ladder": False,
        "_control_scales": {},
    }
