"""What the ASE-driven benchmarks share to run sharded over ranks and worker processes.

The cluster scripts launch one process per GPU (RANK, WORLD_SIZE, LOCAL_RANK); a
benchmark may spawn several workers per process. Each run is guarded by a
wall-clock alarm, and a CUDA device fault stops the process, because its CUDA
context cannot be used again; the unfinished item is picked up by a rerun.
"""

from __future__ import annotations

import os
import signal
from contextlib import contextmanager


def placement(device: str) -> tuple[int, int, str]:
    """(rank, world size, device) of this process; ``"cuda"`` becomes this rank's own GPU."""
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if device == "cuda":
        import torch

        device = f"cuda:{local % max(torch.cuda.device_count(), 1)}"
    return rank, world, device


class Timeout(Exception):
    pass


@contextmanager
def wall_timeout(seconds: float, what: str = "run"):
    """SIGALRM guard: raises ``Timeout`` in the main thread after ``seconds``."""

    def handler(signum, frame):
        raise Timeout(f"{what} timed out after {seconds:g} s")

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def failure(tb: str, what: str) -> str:
    """Classify a failed attempt from its traceback: "oom" or "crash".

    A CUDA device fault (e.g. an illegal memory access) says nothing about the model
    but leaves this process's CUDA context unusable, so every later item here would
    fail at once: the process stops instead and the item stays open for a rerun.
    """
    if "CUDA error" in tb:
        raise SystemExit(f"CUDA device error in {what}; process stopped, rerun needed")
    return "oom" if "out of memory" in tb.lower() else "crash"


def spawn(target, jobs: list[tuple], check: bool = True) -> None:
    """Run ``target(*job)`` for each job in its own spawned process and wait for all."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=target, args=job) for job in jobs]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    bad = [p.exitcode for p in procs if p.exitcode != 0]
    if check and bad:
        raise SystemExit(f"{len(bad)} of {len(procs)} workers failed (exit codes {bad})")
