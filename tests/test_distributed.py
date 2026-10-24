"""The same global batch gives the same training trajectory however it is split
over processes and gradient-accumulation steps, and resuming is exact."""
import torch
from _train_helpers import run_training

from envtok_mace.train.checkpoint import load_checkpoint


def _final_state(out_dir):
    return load_checkpoint(out_dir / "last.pt")


def _assert_same(a, b, atol):
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=atol, msg=key)
    for s, t in zip(a["ema"]["shadow"], b["ema"]["shadow"]):
        torch.testing.assert_close(s, t, rtol=0, atol=atol)


def test_gradient_accumulation_and_ranks_are_equivalent(tmp_path):
    # global batch of 8 structures split three ways; 6 optimiser steps
    splits = [(1, 4, 2), (2, 2, 2), (4, 1, 2)]  # (world, accumulation, per-rank batch)
    states = []
    for world, accum, bs in splits:
        out = tmp_path / f"w{world}_a{accum}"
        run_training(world, out_dir=out, accumulation=accum, batch_size=bs, max_steps=6)
        states.append(_final_state(out))
    for other in states[1:]:
        assert other["global_step"] == states[0]["global_step"] == 6
        _assert_same(states[0], other, atol=1e-11)


def test_resume_is_bitwise_identical(tmp_path):
    whole, split = tmp_path / "whole", tmp_path / "split"
    run_training(1, out_dir=whole, accumulation=2, max_steps=8)
    # first job is "pre-empted" after 3 steps (mid-epoch), the second resumes
    assert run_training(1, out_dir=split, accumulation=2, max_steps=8, interrupt_after_steps=3) == "walltime"
    assert not (split / "FINISHED").exists()
    run_training(1, out_dir=split, accumulation=2, max_steps=8)
    a, b = _final_state(whole), _final_state(split)
    assert a["global_step"] == b["global_step"] == 8
    _assert_same(a, b, atol=0)
    for key in a["optimizer"]["state"]:
        for name, value in a["optimizer"]["state"][key].items():
            if torch.is_tensor(value):
                assert torch.equal(value, b["optimizer"]["state"][key][name])
    assert a["best_val"] == b["best_val"]


def test_finished_marker_stops_a_chained_job(tmp_path):
    run_training(1, out_dir=tmp_path, max_steps=2)
    assert (tmp_path / "FINISHED").exists()
    assert run_training(1, out_dir=tmp_path, max_steps=4) == "finished"


def test_resume_on_a_different_number_of_processes(tmp_path):
    """With a fixed global batch, a run interrupted on 2 processes and resumed on 1
    (accumulation 2) follows the same trajectory as an uninterrupted 1-process run."""
    kw = dict(global_batch=8, batch_size=4, max_steps=6)
    whole, elastic = tmp_path / "whole", tmp_path / "elastic"
    run_training(1, out_dir=whole, **kw)
    assert run_training(2, out_dir=elastic, interrupt_after_steps=3, **kw) is None  # spawn returns None
    assert not (elastic / "FINISHED").exists()
    run_training(1, out_dir=elastic, **kw)
    a, b = _final_state(whole), _final_state(elastic)
    assert a["global_step"] == b["global_step"] == 6
    _assert_same(a, b, atol=1e-11)
