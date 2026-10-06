"""Ordered stage ledger for resumable multi-step jobs.

A lightweight, serializable record of a job's progress through named stages.
The fusion runner drives one of these so the job monitor can show per-stage
progress and a re-run can report where a stopped job left off. The ledger is
persisted in the ``JobStore`` record (SQLite) so it survives process death.

Resume correctness does **not** depend on the ledger. The underlying compute
stages are content-addressed and idempotent — the metric-download cache, the
per-``(entity, radius)`` pre-aggregation cache keyed by fingerprint, and the
Optuna study SQLite keyed by study name. Re-running a job with the same target
and parameters reuses or resumes each on-disk artifact regardless of the
ledger. The ledger is the *visibility* layer over that durability: it tells the
UI (and the user) which stages finished and which remain, and lets later phases
register their own stages without re-plumbing.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
SKIPPED = "skipped"

# Statuses that mean "no work left for this stage" — both a completed stage and
# one deliberately skipped (e.g. an unused standalone metric) count as finished.
_FINISHED = frozenset({DONE, SKIPPED})


@dataclass
class Stage:
    """One named step with a status, fractional progress, and a short message.

    ``started_at`` / ``ended_at`` are epoch seconds, filled in as the stage is
    marked, so a finished job carries its own wall-clock breakdown and nobody
    has to time a run by hand to find where it spent itself.
    """

    key: str
    label: str
    status: str = PENDING
    progress: float = 0.0
    message: str = ""
    started_at: float | None = None
    ended_at: float | None = None

    @property
    def duration_s(self) -> float | None:
        """Wall seconds the stage took, or has been running for so far."""
        if self.started_at is None:
            return None
        end = self.ended_at if self.ended_at is not None else time.time()
        return max(0.0, float(end) - float(self.started_at))

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Stage:
        def _ts(name: str) -> float | None:
            v = d.get(name)
            return None if v is None else float(v)

        return cls(
            key=str(d["key"]),
            label=str(d.get("label", d["key"])),
            status=str(d.get("status", PENDING)),
            progress=float(d.get("progress", 0.0)),
            message=str(d.get("message", "")),
            started_at=_ts("started_at"),
            ended_at=_ts("ended_at"),
        )


class StageLedger:
    """An ordered list of :class:`Stage` with mark/query helpers.

    Serializes to/from a plain dict (``{"stages": [...]}``) so the ``JobStore``
    can persist it as JSON without importing this module — the store treats the
    ledger as opaque, the runner and UI use this class.
    """

    def __init__(self, stages: Iterable[Stage]):
        self._stages: list[Stage] = list(stages)
        self._by_key: dict[str, Stage] = {s.key: s for s in self._stages}

    @classmethod
    def from_steps(cls, steps: Iterable[tuple[str, str]]) -> StageLedger:
        """Build a fresh ledger from ``(key, label)`` pairs (all pending)."""
        return cls(Stage(key=k, label=lbl) for k, lbl in steps)

    @classmethod
    def from_dict(cls, d: dict | None) -> StageLedger:
        if not d:
            return cls([])
        return cls(Stage.from_dict(s) for s in d.get("stages", []))

    def to_dict(self) -> dict:
        return {"stages": [s.to_dict() for s in self._stages]}

    # ────────────────────────────────────────────────────────────
    # Queries
    # ────────────────────────────────────────────────────────────

    @property
    def stages(self) -> list[Stage]:
        return list(self._stages)

    def get(self, key: str) -> Stage | None:
        return self._by_key.get(key)

    def all_finished(self) -> bool:
        return all(s.status in _FINISHED for s in self._stages)

    # ────────────────────────────────────────────────────────────
    # Mutations
    # ────────────────────────────────────────────────────────────

    def set_status(
        self,
        key: str,
        status: str,
        *,
        progress: float | None = None,
        message: str | None = None,
    ) -> None:
        s = self._by_key.get(key)
        if s is None:
            return
        # Timing lives here rather than in the mark_* helpers because callers
        # also set status directly, and a stage that misses its clock is
        # invisible in the breakdown.
        if status == RUNNING:
            if s.started_at is None:
                s.started_at = time.time()
            s.ended_at = None
        elif status in (DONE, FAILED):
            if s.started_at is None:
                s.started_at = time.time()
            s.ended_at = time.time()
        s.status = status
        if progress is not None:
            s.progress = max(0.0, min(1.0, float(progress)))
        if message is not None:
            s.message = message

    def mark_running(self, key: str, message: str = "") -> None:
        self.set_status(key, RUNNING, progress=0.0, message=message)

    def mark_progress(self, key: str, progress: float, message: str = "") -> None:
        self.set_status(key, RUNNING, progress=progress, message=message or None)

    def mark_done(self, key: str, message: str = "") -> None:
        self.set_status(key, DONE, progress=1.0, message=message)

    def mark_skipped(self, key: str, message: str = "") -> None:
        # A skipped stage did no work, so it never gets a duration.
        self.set_status(key, SKIPPED, progress=1.0, message=message)

    def mark_failed(self, key: str, message: str = "") -> None:
        self.set_status(key, FAILED, message=message)

    def stop_running(self, status: str = FAILED, message: str = "") -> str | None:
        """Close the clock on whatever stage was still running, and return its key.

        A stage's duration is measured against the wall clock until its
        ``ended_at`` is filled in, so a run that ends anywhere other than a
        stage boundary leaves the monitor counting a stage that stopped long
        ago. Ending the run is what stops the stage.
        """
        stopped = None
        for s in self._stages:
            if s.status == RUNNING:
                self.set_status(s.key, status, message=message or None)
                stopped = s.key
        return stopped

    def timing_report(self) -> list[tuple[str, str, float, float]]:
        """``(key, label, seconds, share)`` per timed stage, longest first.

        ``share`` is of the summed stage time, which is what a breakdown can
        actually account for — job wall-clock also covers setup between stages.
        """
        rows = [
            (s.key, s.label, float(s.duration_s))
            for s in self._stages
            if s.duration_s is not None and s.duration_s > 0
        ]
        total = sum(r[2] for r in rows)
        if total <= 0:
            return []
        return sorted(
            ((k, lbl, secs, secs / total) for k, lbl, secs in rows),
            key=lambda r: r[2],
            reverse=True,
        )

    def reset_unfinished(self) -> None:
        """Flip ``running``/``failed`` stages back to ``pending`` for a re-run.

        Finished stages (``done``/``skipped``) keep their state so a resubmitted
        job picks up after them; an interrupted or failed stage re-executes from
        the start (its underlying cache resumes mid-way on its own).
        """
        for s in self._stages:
            if s.status in (RUNNING, FAILED):
                s.status = PENDING
                # The abandoned attempt's clock would otherwise be charged to
                # the retry, which restarts the stage from the top.
                s.started_at = None
                s.ended_at = None
                s.progress = 0.0
