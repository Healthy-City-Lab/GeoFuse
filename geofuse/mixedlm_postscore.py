"""Post-hoc multi-metric reporting for mixed-effects fusion runs.

A longitudinal run optimizes one ``mixedlm_*`` metric, but the other three are
just as informative about the same fit — a coefficient that looks strong on
``mixedlm_tstat`` may carry a negligible marginal R². This re-scores the final
composite on the held-out test set with all four scorer metrics and writes
them to ``mixedlm_metrics.csv``.

Output schema (``<study_results>/mixedlm_metrics.csv``)::

    pool, trial_id, mixedlm_tstat, mixedlm_marginal_r2, mixedlm_lr, mixedlm_coef
"""

from __future__ import annotations

import csv
import os
from typing import Any

from . import mixed_effects_scoring

_CSV_BASENAME: str = "mixedlm_metrics.csv"


def compute_post_metrics(
    engine,
    output_dir: str,
    *,
    winning_params: dict,
    log: Any = None,
) -> str | None:
    """Score the discovered composite on the held-out test with every
    ``mixedlm_*`` metric and write ``mixedlm_metrics.csv``.

    No-op (returns ``None``) for cross-sectional runs, when there's no test
    split, or when the params carry nothing to score. ``log`` is an optional
    ``_log``-style callable from the runner so progress routes into the job log.
    """
    if not engine.is_longitudinal or engine.test_data is None:
        return None
    params = {
        k: v for k, v in (winning_params or {}).items() if not str(k).startswith("__")
    }
    if not params:
        return None

    def _say(level: str, msg: str) -> None:
        if log is not None:
            log(level, msg)

    try:
        out = engine.evaluate_on_test(
            params=params,
            metric=engine.longitudinal_spec.scoring_metric,
            return_all_mixedlm=True,
        )
        metrics = out.get("mixedlm_metrics") or {}
    except Exception as exc:
        _say("WARN", f"Post-hoc MixedLM metrics failed: {exc}")
        return None
    if not metrics:
        _say("WARN", "Post-hoc MixedLM metrics: nothing scored; skipping.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, _CSV_BASENAME)
    fields = ["pool", "trial_id"] + sorted(mixed_effects_scoring.MIXEDLM_METRICS)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(
            {
                "pool": "final",
                "trial_id": "final_composite",
                **{k: metrics.get(k, "") for k in fields[2:]},
            }
        )

    _say("OK", f"Post-hoc MixedLM metrics written: {csv_path}")
    return csv_path
