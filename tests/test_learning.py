"""Learning-curve tables from the trainer's step log and the archived checkpoints' evals."""

import csv
import json

import numpy as np
import pytest

from envtok_mace.analysis import learning


def _steps(path, rows):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["epoch", "step", "global_step", "lr", "loss"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_step_losses_are_reduced_per_epoch(tmp_path):
    rows = [{"epoch": e, "step": s, "global_step": 0, "lr": 1e-3, "loss": x} for e, s, x in
            [(0, 50, 0.4), (0, 100, 0.2), (0, 150, 0.3), (1, 50, 0.1), (1, 100, float("nan"))]]  # fmt: skip
    _steps(tmp_path / "steps.csv", rows)
    out = learning.epoch_loss(tmp_path / "steps.csv")
    assert [r["epoch"] for r in out] == [0, 1]
    assert out[0]["n_logged"] == 3 and out[0]["loss_mean"] == pytest.approx(0.3)
    assert out[0]["loss_median"] == pytest.approx(0.3)
    assert out[1]["n_logged"] == 1  # the non-finite step is skipped


def _eval_json(path, results):
    path.write_text(json.dumps({"frames": "x.h5", "n_frames": 10, "dtype": "float32", "results": results}))


def _metrics(f):
    return {k: f for k in learning.METRICS} | {"n_structures": 10, "model": "m.pt"}


def test_checkpoint_evals_line_up_with_the_epoch_log(tmp_path):
    _eval_json(tmp_path / "train.json", {"epoch_0050": _metrics(2.0), "epoch_0025": _metrics(3.0)})
    _eval_json(tmp_path / "test.json", {"epoch_0025": _metrics(4.0)})
    rows = learning.checkpoint_rows({"train": tmp_path / "train.json", "test": tmp_path / "test.json"})
    # the archive written after 25 epochs is the state at epochs.csv's epoch 24
    assert [(r["set"], r["epoch"], r["force_mae_meV_per_A"]) for r in rows] == [
        ("test", 24, 4.0),
        ("train", 24, 3.0),
        ("train", 49, 2.0),
    ]


def test_a_failed_checkpoint_eval_is_an_error(tmp_path):
    _eval_json(tmp_path / "t.json", {"epoch_0025": {"model": "m.pt", "error": "OOM"}})
    with pytest.raises(ValueError, match="failed"):
        learning.checkpoint_rows({"train": tmp_path / "t.json"})
    with pytest.raises(ValueError, match="not an archived checkpoint"):
        learning.epochs_completed("best")


def test_tables_are_written(tmp_path):
    _steps(tmp_path / "steps.csv", [{"epoch": 0, "step": 50, "global_step": 50, "lr": 1e-3, "loss": 0.5}])
    _eval_json(tmp_path / "train.json", {"epoch_0025": _metrics(3.0)})
    args = type("A", (), {})()
    args.steps, args.checkpoint_evals, args.out = (
        str(tmp_path / "steps.csv"), [f"train={tmp_path / 'train.json'}"], str(tmp_path / "out"),
    )  # fmt: skip
    learning.run(args)
    loss = list(csv.DictReader((tmp_path / "out" / "train_loss.csv").open()))
    ckpt = list(csv.DictReader((tmp_path / "out" / "checkpoints.csv").open()))
    assert np.isclose(float(loss[0]["loss_mean"]), 0.5) and ckpt[0]["set"] == "train"
