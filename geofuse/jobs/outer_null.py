"""Null calibration of the repeated-split decision rule.

A repeated study decides on the corrected resampled t of each quantity across
its K held-out splits (:mod:`geofuse.jobs.outer_splits`), the CGI's held-out
slope first. Its false-positive rate is measured here: ``n`` null outcomes
(:mod:`geofuse.null_outcomes`, free or spatially structured) each go through
the same search and the same K splits and seeds as the real outcome, in the
``decision`` report, and the share of nulls whose corrected p falls below 5 %
is the rate of that decision, with its exact interval. The real outcome's
corrected p is also placed among the nulls' (``calibrated_p``).

Only the target column changes: the input preparation, the exposure tensor,
the pre-aggregation and the smooth bases are the real run's. The nulls run in
a process pool, each worker holding one copy of the engine; one worker runs
them in the job's own process. Every finished null is appended to a resume
file keyed by the setup, so a cancelled or rerun job continues from the next
null. A sequential rule (``stop``) can end a kind after ``after`` nulls when
the hits are clearly few or clearly many; the rate is then over those nulls.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd

from .. import null_outcomes as _no
from ..logger import muted
from . import outer_splits as _outer

#: The decision the sequential rule and the log headline read.
HEADLINE = ("held_out_beta", "cgi", None)

#: The label entities without a group value share.
MISSING_GROUP = "(missing)"

#: Group balance below which the between-group permutation has little to
#: permute: fewer groups than this, or one group above this share.
MIN_GROUPS = 5
MAX_GROUP_SHARE = 0.25

#: Memory one null worker reserves next to its copy of the engine.
_WORKER_BASE_BYTES = 1_500_000_000

#: The most null workers sized automatically: each samples its posteriors on
#: one thread per chain and holds its own JAX runtime (about 2.3 GB measured on
#: a 3,000-entity study), so a wider pool contends for cores and memory.
AUTO_WORKERS_MAX = 4

#: Engine state a search or a smooth selection updates: snapshotted before
#: the nulls and restored after them, so the real outcome's reports keep
#: their own. Every null starts from the snapshot's warm start.
_ENGINE_STATE = (
    "_posterior_warm",
    "_spatial_selections",
    "_spatial_run_summary",
    "_spatial_adjust_summary",
    "_spatial_scale_summaries",
)


def _snapshot(engine: Any) -> dict:
    return {
        k: copy.deepcopy(getattr(engine, k))
        for k in _ENGINE_STATE
        if hasattr(engine, k)
    }


def _restore(engine: Any, snapshot: dict) -> None:
    for k, v in snapshot.items():
        setattr(engine, k, copy.deepcopy(v))


# ────────────────────────────────────────────────────────────────────
# Entities of the real outcome
# ────────────────────────────────────────────────────────────────────


class Entities:
    """One row per entity of the fusion frame's kept rows.

    ``codes`` maps every kept frame row to its entity; ``y`` is the real
    outcome, ``xy`` the entity's location in metres (its ``_cx`` / ``_cy``,
    else the exposure-weighted centre of its cells), ``group`` the values of
    the grouping column (a missing value labelled :data:`MISSING_GROUP`,
    counted in ``n_missing_group``) and ``stratum`` the wave, each ``None``
    when absent.
    """

    def __init__(self, engine: Any, fusion_df: pd.DataFrame, group_column=None):
        keep = getattr(engine, "_coverage_keep_index", None)
        df = fusion_df if keep is None else fusion_df.loc[fusion_df.index.isin(keep)]
        self.index = df.index
        ids = df["polygon_id"].to_numpy() if "polygon_id" in df.columns else df.index
        self.codes, _ = pd.factorize(np.asarray(ids))
        n = int(self.codes.max()) + 1 if len(self.codes) else 0
        first = np.unique(self.codes, return_index=True)[1]
        self.n = n
        self.y = df["target"].to_numpy(dtype=np.float64)[first]
        if {"_cx", "_cy"} <= set(df.columns):
            self.xy = df[["_cx", "_cy"]].to_numpy(dtype=np.float64)[first]
        else:
            geom = engine.target_gdf.geometry.loc[df.index]
            w = (
                df["_weight"].to_numpy(dtype=np.float64)
                if "_weight" in df.columns
                else np.ones(len(df))
            )
            total = np.bincount(self.codes, weights=w, minlength=n)
            self.xy = np.column_stack(
                [
                    np.bincount(self.codes, weights=w * geom.x.to_numpy(), minlength=n)
                    / total,
                    np.bincount(self.codes, weights=w * geom.y.to_numpy(), minlength=n)
                    / total,
                ]
            )
        self.group, self.n_missing_group = None, 0
        if group_column and group_column in df.columns:
            labels = df[group_column].to_numpy(dtype=object)[first]
            missing = pd.isna(labels)
            labels[missing] = MISSING_GROUP
            self.group, self.n_missing_group = labels, int(missing.sum())
        self.stratum = df["wave"].to_numpy()[first] if "wave" in df.columns else None

    def frame(self, fusion_df: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
        """``fusion_df`` with the kept rows' target replaced by ``values``
        (one per entity)."""
        out = fusion_df.copy()
        out["target"] = out["target"].astype(np.float64)
        out.loc[self.index, "target"] = np.asarray(values, dtype=np.float64)[
            self.codes
        ]
        return out


def gap_groups(engine: Any, xy: np.ndarray) -> tuple[np.ndarray, str]:
    """The spatial clusters of the smooth's gap rule, and the rule as text: a
    gap of ``spatial_adjust_eps_m``, else twice the primary pattern scale, else
    the nearest-neighbour graph's components; clusters below
    ``spatial_adjust_min_component`` joined to their nearest larger one. The
    gap settings reach the engine whenever the null calibration runs, with or
    without spatial adjustment."""
    from ..spatial_basis import connected_components, merge_small_components

    eps = getattr(engine, "spatial_adjust_eps_m", None)
    scales = getattr(engine, "spatial_adjust_scales", ()) or ()
    if eps is None and scales:
        eps = 2.0 * float(scales[0])
    k = int(getattr(engine, "spatial_adjust_neighbours", 10))
    labels, _ = connected_components(xy, eps=eps, k=k)
    rule = (
        f"gap clusters at {float(eps):g} m"
        if eps is not None
        else f"{k}-nearest-neighbour components"
    )
    small = getattr(engine, "spatial_adjust_min_component", None)
    if small:
        labels = merge_small_components(xy, labels, int(small))[0]
        rule += f", below {int(small)} joined to the nearest"
    return labels, rule


def group_record(labels: np.ndarray, source: str, n_missing: int = 0) -> dict:
    """``source``, ``n_groups``, the ``smallest`` group, the ``largest_share``
    of the entities in one group and ``n_missing`` (entities without a group
    value, grouped together)."""
    _, counts = np.unique(np.asarray(labels).astype(str), return_counts=True)
    return {
        "source": source,
        "n_groups": int(len(counts)),
        "smallest": int(counts.min()) if len(counts) else 0,
        "largest_share": float(counts.max() / counts.sum()) if len(counts) else None,
        "n_missing": int(n_missing),
    }


def group_warning(record: dict) -> str | None:
    """Why the between-group permutation has little to permute, or ``None``."""
    reasons = []
    if record["n_groups"] < MIN_GROUPS:
        reasons.append(f"only {record['n_groups']} groups")
    share = record.get("largest_share")
    if share is not None and share > MAX_GROUP_SHARE:
        reasons.append(f"the largest holds {share:.0%} of the entities")
    if not reasons:
        return None
    return (
        f"Spatial null groups ({record['source']}): {' and '.join(reasons)}, so "
        "permuting the group means changes little; a column of comparable areas "
        "(outer_null_group) or a gap distance (spatial_adjust_eps_m) gives more."
    )


# ────────────────────────────────────────────────────────────────────
# One null outcome through the decision rule
# ────────────────────────────────────────────────────────────────────

_WORKER: dict = {}


def _worker_setup(state: "dict | str") -> None:
    """A null worker's state: the live engine in the job's process, else the
    pickled copy at ``state``, its own searches kept to one process."""
    _WORKER.clear()
    if isinstance(state, dict):
        _WORKER.update(state)
        return
    os.environ["GEOFUSE_WORKERS"] = "1"
    with open(state, "rb") as f:
        _WORKER.update(pickle.load(f))


def _run_null(kind: str, index: int, values: np.ndarray) -> dict:
    """The decision report of one null outcome over every split.

    A null whose search fails returns ``{"error": …}``; one cut short by a
    cancel has fewer than ``k_splits`` splits. Neither counts as finished.
    """
    try:
        return _null_report(kind, index, values)
    except Exception as exc:
        return {"kind": kind, "index": int(index), "error": str(exc)}


def _null_report(kind: str, index: int, values: np.ndarray) -> dict:
    st = _WORKER
    engine = st["engine"]
    rule = st["rule"]
    engine._posterior_warm = copy.deepcopy(st["snapshot"].get("_posterior_warm", {}))
    split_kwargs = {
        **st["split_kwargs"],
        "fusion_df": st["entities"].frame(st["fusion_df"], values),
    }
    with muted():
        rows = _outer.repeat(
            engine,
            k_splits=rule["k_splits"],
            split_seed=rule["split_seed"],
            split_kwargs=split_kwargs,
            fit_kwargs=st["fit_kwargs"],
            metric=rule["metric"],
            standalones=rule["standalones"],
            comparators=rule["comparators"],
            comparators_primary=rule["comparators_primary"],
            outer_comparators=rule["outer_comparators"],
            n_effects_boot=0,
            n_perm=0,
            n_paired=0,
            log=lambda level, msg: None,
            cancelled=st.get("cancelled"),
            specificity=rule["specificity"],
            report="decision",
            first=0,
        )
    entries = [
        {
            "quantity": e["quantity"],
            "exposure": e["exposure"],
            "control": e.get("control"),
            "median": e["median"],
            "corrected_t": e["corrected_t"],
            "corrected_p": e["corrected_p"],
        }
        for e in _outer.summarise(rows)
        if e.get("alternative")
    ]
    return {
        "kind": kind,
        "index": int(index),
        "n_splits": len({r["split"] for r in rows}),
        "entries": entries,
    }


# ────────────────────────────────────────────────────────────────────
# Resume file
# ────────────────────────────────────────────────────────────────────


def fingerprint(**parts: Any) -> str:
    """A short hash of the null setup: nulls from another setup never mix."""
    text = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def read_finished(path: str) -> dict[tuple[str, int], dict]:
    """Finished nulls of a resume file, by ``(kind, index)``."""
    out: dict[tuple[str, int], dict] = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[(rec["kind"], int(rec["index"]))] = rec
    return out


def _append(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=_json_default) + "\n")


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


# ────────────────────────────────────────────────────────────────────
# Rates
# ────────────────────────────────────────────────────────────────────


def _hit(entry: dict) -> bool:
    p = entry.get("corrected_p")
    return p is not None and p < _no.NULL_ALPHA


def _headline(record: dict) -> "dict | None":
    return next(
        (
            e
            for e in record["entries"]
            if (e["quantity"], e["exposure"], e.get("control")) == HEADLINE
        ),
        None,
    )


def null_rows(records: list[dict]) -> list[dict]:
    """``null_outer.csv``: one row per null and tested quantity."""
    rows = []
    for rec in sorted(records, key=lambda r: (r["kind"], r["index"])):
        for e in rec["entries"]:
            rows.append(
                {
                    "null_kind": rec["kind"],
                    "null": int(rec["index"]) + 1,
                    "quantity": e["quantity"],
                    "exposure": e["exposure"],
                    "control": e.get("control"),
                    "median": e["median"],
                    "corrected_t": e["corrected_t"],
                    "corrected_p": e["corrected_p"],
                    "hit": _hit(e),
                    "n_splits": rec["n_splits"],
                    "surrogate_k": rec.get("k"),
                }
            )
    return rows


def null_summary(
    records: list[dict],
    observed: list[dict],
    decisions: dict[str, str],
) -> list[dict]:
    """``null_summary.csv``: per kind and tested quantity, the rate of
    corrected p below 5 % with its Clopper–Pearson interval, the share of
    nulls with a positive median, the real outcome's corrected p and its
    place among the nulls' (``calibrated_p`` = (1 + nulls at or below it) /
    (1 + nulls)), and on the headline row the sequential rule's decision."""
    real = {
        (e["quantity"], e["exposure"], e.get("control")): e.get("corrected_p")
        for e in observed
        if e.get("alternative") and e.get("spatial_scale_m") is None
    }
    groups: dict[tuple, list[dict]] = {}
    for rec in records:
        for e in rec["entries"]:
            key = (rec["kind"], e["quantity"], e["exposure"], e.get("control"))
            groups.setdefault(key, []).append(e)
    order = list(_outer.QUANTITIES)
    out = []
    for (kind, quantity, exposure, control), entries in sorted(
        groups.items(),
        key=lambda kv: (kv[0][0], order.index(kv[0][1]), kv[0][2], kv[0][3] or ""),
    ):
        n = len(entries)
        hits = sum(_hit(e) for e in entries)
        lo, hi = _no.clopper_pearson(hits, n)
        medians = [e["median"] for e in entries if e["median"] is not None]
        p_obs = real.get((quantity, exposure, control))
        p_null = [e["corrected_p"] for e in entries if e["corrected_p"] is not None]
        row = {
            "null_kind": kind,
            "quantity": quantity,
            "exposure": exposure,
            "control": control,
            "finished": n,
            "hits": hits,
            "rate": hits / n if n else None,
            "ci_low": lo,
            "ci_high": hi,
            "share_positive": (
                float(np.mean([m > 0 for m in medians])) if medians else None
            ),
            "observed_corrected_p": p_obs,
            "calibrated_p": (
                (1 + sum(p <= p_obs for p in p_null)) / (1 + len(p_null))
                if p_obs is not None and p_null
                else None
            ),
            "decision": None,
        }
        if (quantity, exposure, control) == HEADLINE:
            row["decision"] = decisions.get(kind)
        out.append(row)
    return out


# ────────────────────────────────────────────────────────────────────
# The run
# ────────────────────────────────────────────────────────────────────


def run(
    engine: Any,
    *,
    fusion_df: pd.DataFrame,
    n: int,
    kind: str,
    group_column: str | None,
    stop: dict | None,
    seed: int,
    rule: dict,
    split_kwargs: dict,
    fit_kwargs: dict,
    observed: list[dict],
    resume_dir: str,
    setup: dict,
    workers: int | None = None,
    log: Any = None,
    progress: Any = None,
    cancelled: Any = None,
) -> dict:
    """``n`` nulls of each kind of ``kind`` through the decision rule.

    ``rule`` holds the repeat's settings (``k_splits``, ``split_seed``,
    ``metric``, ``standalones``, ``comparators``, ``comparators_primary``,
    ``outer_comparators``, ``specificity``); ``split_kwargs`` and
    ``fit_kwargs`` are the real repeats' (``fusion_df`` is replaced per null).
    ``observed`` is the real outcome's across-split summary. ``setup`` names
    what the nulls depend on beyond these (the run's settings), so the resume
    file under ``resume_dir`` is only reused for the same study. ``workers``
    ``None`` sizes the pool on cores and memory. ``progress(done, total)``
    and ``cancelled()`` as for the repeats. Returns ``{kinds, runs, workers,
    groups, records, rows, summary, variogram, decisions, complete}``.
    """
    log = log or (lambda level, msg: None)
    stop = stop or {}
    entities = Entities(engine, fusion_df, group_column)
    groups, group_info = None, None
    if "spatial" in _no.kinds(kind):
        if group_column:
            groups = entities.group
            group_info = group_record(
                groups, f"column {group_column}", entities.n_missing_group
            )
        else:
            groups, gap_rule = gap_groups(engine, entities.xy)
            group_info = group_record(groups, gap_rule)
        warning = group_warning(group_info)
        if warning:
            log("WARN", warning)
    spatial = (
        _no.SpatialNull(entities.y, entities.xy, groups, seed=int(seed))
        if groups is not None
        else None
    )

    key = fingerprint(
        setup=setup,
        rule=rule,
        seed=int(seed),
        group=group_column,
        n_entities=entities.n,
        target=hashlib.sha256(entities.y.tobytes()).hexdigest(),
    )
    os.makedirs(resume_dir, exist_ok=True)
    resume_path = os.path.join(resume_dir, f"{key}.jsonl")
    finished = read_finished(resume_path)
    if finished:
        log("INFO", f"Null calibration: {len(finished)} finished nulls reused.")

    def draw(k: str, i: int) -> tuple[np.ndarray, dict]:
        rng = _no.null_seed(int(seed), k, i)
        if k == "free":
            return _no.free_null(entities.y, rng, entities.stratum), {}
        d = spatial.draw(rng)
        return d["y"], {"k": d["k"], "variogram": d["variogram"].tolist()}

    state = {
        "engine": engine,
        "rule": rule,
        "split_kwargs": {k: v for k, v in split_kwargs.items() if k != "fusion_df"},
        "fit_kwargs": fit_kwargs,
        "fusion_df": fusion_df,
        "entities": entities,
        "cancelled": cancelled,
        "snapshot": _snapshot(engine),
    }
    pool = 1 if workers is None and n <= 1 else workers
    blob_path = None
    if pool is None or pool > 1:
        blob_path = os.path.join(resume_dir, f"engine_{key}_{os.getpid()}.pkl")
        blob_path = _write_state(state, blob_path)
        size = os.path.getsize(blob_path)
        if pool is None:
            from .. import parallel

            pool = parallel.process_worker_count(
                n,
                cap=AUTO_WORKERS_MAX,
                bytes_per_worker=3 * size + _WORKER_BASE_BYTES,
            )
        log(
            "INFO",
            f"Null calibration: {pool} worker(s), each with a "
            f"{size / 2**20:,.0f} MB copy of the engine.",
        )
    total = n * len(_no.kinds(kind))
    records: list[dict] = []
    decisions: dict[str, str] = {}
    complete = True
    try:
        for k in _no.kinds(kind):
            mine = [r for (kk, _), r in finished.items() if kk == k and r["index"] < n]
            todo = [i for i in range(n) if (k, i) not in finished]
            extra: dict[int, dict] = {}
            tasks = []
            for i in todo:
                values, meta = draw(k, i)
                extra[i] = meta
                tasks.append((k, i, values))
            stopped = {"decision": _decide(mine, stop)}

            def collect(result: dict, k=k, extra=extra, mine=mine, stopped=stopped):
                if result.get("error") or result["n_splits"] < rule["k_splits"]:
                    if result.get("error"):
                        log(
                            "WARN",
                            f"Null {k} {result['index'] + 1}/{n} failed: "
                            f"{result['error']}",
                        )
                    return
                result.update(extra.get(result["index"], {}))
                _append(resume_path, result)
                mine.append(result)
                head = _headline(result) or {}
                log(
                    "OK",
                    f"Null {k} {result['index'] + 1}/{n}: corrected "
                    f"p={_fmt(head.get('corrected_p'))}"
                    + (f" (surrogate k {result['k']})" if result.get("k") else "")
                    + ".",
                )
                if progress is not None:
                    progress(len(records) + len(mine), total)
                if stopped["decision"] is None:
                    stopped["decision"] = _decide(mine, stop)

            def halt(stopped=stopped) -> bool:
                if cancelled is not None and cancelled():
                    return True
                return stopped["decision"] is not None

            if tasks and stopped["decision"] is None:
                from .. import parallel

                done = parallel.map_batches(
                    _run_null,
                    tasks,
                    workers=max(1, min(int(pool or 1), len(tasks))),
                    initializer=_worker_setup,
                    initargs=(state if blob_path is None else blob_path,),
                    on_result=collect,
                    cancel_check=halt,
                )
                if not done and stopped["decision"] is None:
                    complete = False
            decision = stopped["decision"]
            if decision is not None:
                after = int(stop["after"])
                mine = [r for r in mine if r["index"] < after]
                decisions[k] = f"stopped at {after}: {decision}"
            elif stop.get("after"):
                decisions[k] = "ran all"
            records += mine
            if cancelled is not None and cancelled():
                complete = False
                break
    finally:
        _restore(engine, state["snapshot"])
        if blob_path is not None and os.path.exists(blob_path):
            os.remove(blob_path)

    variogram = None
    if spatial is not None:
        gs = [r["variogram"] for r in records if r.get("variogram") is not None]
        variogram = spatial.match_table(gs)
    for k in _no.kinds(kind):
        mine = [r for r in records if r["kind"] == k]
        hits = sum(_hit(_headline(r) or {}) for r in mine)
        lo, hi = _no.clopper_pearson(hits, len(mine))
        if mine:
            log(
                "OK",
                f"Null calibration ({k}): the held-out decision rejects in "
                f"{hits}/{len(mine)} null outcomes ({hits / len(mine):.1%}, 95 % "
                f"CI {lo:.1%}–{hi:.1%})"
                + (f"; {decisions[k]}" if k in decisions else "")
                + ".",
            )
    return {
        "kinds": _no.kinds(kind),
        "runs": int(n),
        "workers": int(pool or 1),
        "groups": group_info,
        "stop": dict(stop) or None,
        "records": records,
        "rows": null_rows(records),
        "summary": null_summary(records, observed, decisions),
        "variogram": variogram,
        "decisions": decisions,
        "complete": complete and all(
            len([r for r in records if r["kind"] == k]) >= n or k in decisions
            for k in _no.kinds(kind)
        ),
    }


def _decide(records: list[dict], stop: dict) -> str | None:
    """The sequential rule once the first ``after`` nulls are all done."""
    after = stop.get("after")
    if not after:
        return None
    first = [r for r in records if r["index"] < int(after)]
    if len(first) < int(after):
        return None
    hits = sum(_hit(_headline(r) or {}) for r in first)
    return _no.stop_decision(
        hits, int(after), after=int(after), low=stop.get("low"), high=stop.get("high")
    )


def _write_state(state: dict, path: str) -> str:
    """Pickle the worker state without what only the job's process can hold."""
    engine = state["engine"]
    cancel = getattr(engine, "_cancel_callback", None)
    engine._cancel_callback = None
    try:
        payload = {
            **state,
            "fit_kwargs": {
                k: v for k, v in state["fit_kwargs"].items() if k != "cancel_callback"
            },
            "cancelled": None,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        engine._cancel_callback = cancel
    return path


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3g}"
