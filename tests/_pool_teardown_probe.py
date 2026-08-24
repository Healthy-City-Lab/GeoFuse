"""Driver for the "killing a child reaps its pool" check. Run as its own script.

A spawn worker re-executes its parent's ``__main__`` before it runs anything
else, so this scenario cannot live inside a test method: whichever module the
runner has left in ``sys.modules["__main__"]`` is what every worker would try
to import, and the result then depends on which tests ran first. As a standalone
script the bootstrap has a real file to re-execute and the outcome is the
process behaviour alone.

The worker pids are collected *before* the kill and each is then polled by pid.
Re-walking the tree afterwards cannot work: Windows does not reparent orphans,
so once the middle process is gone its children point at a pid that no longer
exists and a walk from the top reports an empty tree whether or not they are
still running — which is exactly the failure this checks for.

Pass ``--bare-pool`` to build the pool without :func:`geofuse.parallel.map_batches`,
which is the arrangement that leaks; it turns this script into a negative control
that must fail. Exits ``0`` when the child's workers went with it, non-zero
otherwise, printing one line either way. Not named ``test_*`` so discovery leaves
it alone; :mod:`tests.test_job_cancellation` runs it as a subprocess.
"""

import multiprocessing as mp
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

WORKERS = 3


def _slow(i):
    time.sleep(120)
    return i


def _child_owning_a_pool():
    """Stands in for a job child: owns a pool and answers nothing."""
    from geofuse import parallel
    from geofuse.jobs.subprocess_runner import exit_with_parent

    exit_with_parent()
    parallel.map_batches(_slow, [(i,) for i in range(8)], workers=WORKERS)


def _child_owning_a_bare_pool():
    """Negative control: the arrangement with no parent guard in the workers."""
    from concurrent.futures import ProcessPoolExecutor

    from geofuse.jobs.subprocess_runner import exit_with_parent

    exit_with_parent()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(_slow, range(8), chunksize=1))


def _process_table() -> dict[int, tuple[int, str]]:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process | ForEach-Object { "
         "$_.ProcessId.ToString() + ',' + $_.ParentProcessId.ToString() + "
         "',' + $_.Name }"],
        capture_output=True, text=True,
    ).stdout
    rows: dict[int, tuple[int, str]] = {}
    for line in out.splitlines():
        parts = line.strip().split(",")
        if len(parts) == 3 and parts[0].isdigit():
            rows[int(parts[0])] = (int(parts[1]), parts[2])
    return rows


def python_descendants(pid: int) -> list[int]:
    """Every ``python.exe`` whose ancestry reaches ``pid``, chain intact."""
    rows = _process_table()
    found: set[int] = set()
    frontier = {pid}
    while frontier:
        nxt = {p for p, (par, _n) in rows.items()
               if par in frontier and p != pid and p not in found}
        if not nxt:
            break
        found |= nxt
        frontier = nxt
    return sorted(p for p in found if "python" in rows[p][1].lower())


def still_running(pids: list[int]) -> list[int]:
    """Which of ``pids`` are still live processes, by pid — no tree walk."""
    rows = _process_table()
    return [p for p in pids if p in rows]


def _reap(pids: list[int]) -> None:
    """Kill leftover workers, so a run that proves the leak does not cause one."""
    for pid in still_running(pids):
        subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                       capture_output=True, text=True)


def main(bare: bool) -> int:
    from geofuse.persistence.job_executor import JobExecutor

    target = _child_owning_a_bare_pool if bare else _child_owning_a_pool
    proc = mp.get_context("spawn").Process(target=target, name="pool-owner")
    proc.start()
    workers: list[int] = []
    try:
        want = WORKERS + 1          # the child plus its workers
        deadline = time.time() + 90
        tree = python_descendants(os.getpid())
        while time.time() < deadline and len(tree) < want:
            time.sleep(1.0)
            tree = python_descendants(os.getpid())
        if len(tree) < want:
            print(f"FAIL: pool never came up (saw {tree})")
            return 2
        workers = [p for p in tree if p != proc.pid]

        JobExecutor._stop_process(proc, grace_s=0.0)
        if proc.is_alive():
            print("FAIL: child survived _stop_process")
            return 3

        deadline = time.time() + 30
        left = still_running(workers)
        while time.time() < deadline and left:
            time.sleep(1.0)
            left = still_running(workers)
        if left:
            print(f"FAIL: {len(left)}/{len(workers)} workers outlived the "
                  f"child: {left}")
            return 4
        print(f"OK: killed the child, all {len(workers)} workers went with it")
        return 0
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
        _reap(workers)


if __name__ == "__main__":
    sys.exit(main(bare="--bare-pool" in sys.argv))
