"""Learning curves: training against validation loss, and errors on the training,
validation and held-out test sets against epoch.

The trainer logs two things:
- every ``log_every`` steps, the loss of one optimiser step (``steps.csv``: raw weights
  on that step's global batch);
- once per epoch, the validation metrics (``epochs.csv``: EMA weights).

Both use the same loss (weights and Huber delta), so they share an axis.

``--steps`` reduces the step log to one row per epoch. ``--checkpoint-evals`` collects
``envtok-mace eval`` results for the archived checkpoints (``epoch_XXXX.pt``, scored
with their EMA weights by the validation metrics) on fixed subsets of the training and
held-out sets. The best epoch is chosen on the validation set; the test curve is shown
for transparency and selects nothing.

    envtok-mace learning-curves --steps run/steps.csv \\
        --checkpoint-evals train=ckpt_train.json test=ckpt_test.json --out results/runs/envtok_8gpu
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = (
    "loss",
    "energy_mae_meV_per_atom",
    "energy_rmse_meV_per_atom",
    "force_mae_meV_per_A",
    "force_rmse_meV_per_A",
    "stress_mae_meV_per_A3",
)


def epoch_loss(steps_csv: str | Path) -> list[dict]:
    """Per epoch: how many steps were logged, and the mean, median and 10/90 % quantiles
    of their loss. Epochs are numbered like ``epochs.csv`` (0 = the first epoch)."""
    losses = defaultdict(list)
    with open(steps_csv) as fh:
        for row in csv.DictReader(fh):
            loss = float(row["loss"])
            if np.isfinite(loss):
                losses[int(row["epoch"])].append(loss)
    rows = []
    for epoch in sorted(losses):
        v = np.asarray(losses[epoch])
        p10, p50, p90 = np.quantile(v, [0.1, 0.5, 0.9])
        rows.append(
            {
                "epoch": epoch,
                "n_logged": len(v),
                "loss_mean": float(v.mean()),
                "loss_median": float(p50),
                "loss_p10": float(p10),
                "loss_p90": float(p90),
            }
        )
    return rows


def epochs_completed(label: str) -> int:
    """``epoch_0025`` (the archive after 25 epochs) -> 25."""
    m = re.fullmatch(r"epoch_(\d+)", label)
    if m is None:
        raise ValueError(f"not an archived checkpoint label: {label!r}")
    return int(m.group(1))


def checkpoint_rows(evals: dict[str, str | Path]) -> list[dict]:
    """One row per (set, checkpoint) from ``eval`` JSON files keyed by set name."""
    rows = []
    for name, path in evals.items():
        d = json.loads(Path(path).read_text())
        for label, r in d["results"].items():
            if "error" in r:
                raise ValueError(f"{name}/{label} failed: {r['error']}")
            rows.append(
                {
                    "set": name,
                    # epochs.csv counts from 0, so the archive after n epochs sits at epoch n - 1
                    "epoch": epochs_completed(label) - 1,
                    "n_structures": r["n_structures"],
                    **{k: r[k] for k in METRICS},
                }
            )
    return sorted(rows, key=lambda r: (r["set"], r["epoch"]))


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"{len(rows)} rows -> {path}", flush=True)


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--steps", default=None, help="the trainer's steps.csv")
    p.add_argument(
        "--checkpoint-evals",
        nargs="+",
        default=[],
        metavar="SET=EVAL_JSON",
        help="eval results of the archived checkpoints, e.g. train=a.json test=b.json",
    )
    p.add_argument("--out", required=True, help="run directory in results/, e.g. results/runs/envtok_8gpu")


def run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    if args.steps:
        write_csv(epoch_loss(args.steps), out / "train_loss.csv")
    if args.checkpoint_evals:
        evals = dict(item.split("=", 1) for item in args.checkpoint_evals)
        write_csv(checkpoint_rows(evals), out / "checkpoints.csv")
