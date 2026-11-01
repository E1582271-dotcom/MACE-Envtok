"""The helpers the ASE-driven benchmarks share for running sharded."""

import time

import pytest

from envtok_mace.bench import _runner


def test_placement_reads_the_launcher_variables(monkeypatch):
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert _runner.placement("cpu") == (5, 8, "cpu")


def test_placement_defaults_to_a_single_process(monkeypatch):
    for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(var, raising=False)
    assert _runner.placement("cpu") == (0, 1, "cpu")


def test_wall_timeout_interrupts_and_disarms():
    with pytest.raises(_runner.Timeout, match="hop timed out after 0.05 s"):
        with _runner.wall_timeout(0.05, "hop"):
            time.sleep(1)
    with _runner.wall_timeout(0.05):  # finishes in time; the alarm must not fire later
        pass
    time.sleep(0.1)


def test_failures_are_classified_from_the_traceback():
    assert _runner.failure("RuntimeError: CUDA out of memory. Tried to allocate", "x") == "oom"
    assert _runner.failure("ValueError: bad cell", "x") == "crash"
    with pytest.raises(SystemExit, match="CUDA device error in run_7"):
        _runner.failure("RuntimeError: CUDA error: an illegal memory access was encountered", "run_7")


def _exit_with(code):
    raise SystemExit(code)


def test_spawn_reports_failed_workers():
    _runner.spawn(_exit_with, [(0,), (0,)])
    with pytest.raises(SystemExit, match="1 of 2 workers failed"):
        _runner.spawn(_exit_with, [(0,), (3,)])
    _runner.spawn(_exit_with, [(3,)], check=False)
