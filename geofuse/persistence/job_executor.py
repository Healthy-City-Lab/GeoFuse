"""Thread-pool executor that runs ``geofuse.jobs.runners`` callables.

The executor owns a single process-level GPU lock (replaces the
``@st.cache_resource`` lock that previously lived in ``ui/tabs/gvi.py``).
``submit_runner`` wraps a runner callable so the surrounding
``JobStore.transition(...)`` calls happen automatically — runners only call
``ctx.progress(...)``, ``ctx.heartbeat()``, ``ctx.is_cancelled()``.
"""

from __future__ import annotations

import logging
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from geofuse.persistence.job_store import JobRecord, JobStore

logger = logging.getLogger(__name__)


@dataclass
class JobContext:
    """Handle a runner uses to publish progress and check for cancellation.

    Runners never touch the ``JobStore`` directly — everything flows through
    this object so the executor can stay in control of state transitions.
    """

    job_id: str
    store: JobStore
    cancel_event: threading.Event

    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def progress(
        self,
        value: float | None = None,
        status_text: str | None = None,
        **extra,
    ) -> None:
        self.store.update_progress(
            self.job_id, progress=value, status_text=status_text, **extra
        )

    def heartbeat(self) -> None:
        self.store.heartbeat(self.job_id)

    def set_extra(self, **kwargs) -> None:
        """Stash live, non-persisted values (engine handles, GeoDataFrames) on the record."""
        self.store.update_progress(self.job_id, **kwargs)


class JobExecutor:
    """ThreadPoolExecutor that runs runners with a shared GPU lock."""

    def __init__(self, store: JobStore, max_workers: int = 4):
        self._store = store
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="geofuse-job"
        )
        self._gpu_lock = threading.Lock()
        self._lock = threading.Lock()
        self._futures: dict[str, Future] = {}

    @property
    def gpu_lock(self) -> threading.Lock:
        """Module-level GPU serialization. Workers must acquire before GPU work."""
        return self._gpu_lock

    @property
    def store(self) -> JobStore:
        return self._store

    def submit_runner(
        self,
        record: JobRecord,
        runner_fn: Callable[..., Any],
        *args,
        **kwargs,
    ) -> Future:
        """Submit ``runner_fn(ctx, *args, **kwargs)`` to the pool.

        The executor transitions the record to ``running`` on entry and to
        ``completed`` / ``error`` / ``cancelled`` on exit. Runners only need
        to use ``ctx`` for progress and cancellation.
        """
        ctx = JobContext(
            job_id=record.id,
            store=self._store,
            cancel_event=record.cancel_event,
        )

        def _wrapped():
            self._store.transition(record.id, "running")
            try:
                result = runner_fn(ctx, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — workers surface any error
                logger.exception("Job %s (%s) failed", record.id, record.type)
                self._store.transition(
                    record.id,
                    "error",
                    error=f"{type(exc).__name__}: {exc}",
                )
                rec = self._store.get(record.id)
                if rec is not None:
                    rec.extra["error_detail"] = traceback.format_exc()
                return
            finally:
                with self._lock:
                    self._futures.pop(record.id, None)

            # Runner returned normally — decide terminal status.
            if ctx.is_cancelled():
                self._store.transition(record.id, "cancelled")
                return

            outputs = None
            if isinstance(result, dict):
                outputs = result.get("output_paths")
            self._store.transition(
                record.id,
                "completed",
                output_paths=outputs,
            )

        future = self._pool.submit(_wrapped)
        with self._lock:
            self._futures[record.id] = future
        return future

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=True)
