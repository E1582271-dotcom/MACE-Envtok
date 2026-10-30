"""Energy/force/stress errors of a model on a set of structures, summed across ranks."""

from __future__ import annotations

import math

import numpy as np
import torch

from ..data.batch import collate
from ..data.frames import Frames
from ..train.dist import all_reduce_sum
from ..train.loss import TERMS, LossConfig, combine, loss_sums

_FIELDS = (
    "n_graphs",
    "n_force",
    "n_stress",
    "e_abs",
    "e_sq",
    "f_abs",
    "f_sq",
    "s_abs",
    "loss_energy",
    "loss_forces",
    "loss_stress",
)


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def evaluate(
    model,
    frames: Frames,
    batches: list[np.ndarray],
    device: torch.device,
    dtype: torch.dtype,
    loss_cfg: LossConfig | None = None,
    collect=None,
) -> dict[str, float]:
    """``batches`` are this rank's index arrays; results are global (all-reduced).

    ``collect(idx, out, batch)``, if given, sees every batch's model output, e.g. an
    ``eval.predictions.Collector`` that keeps the per-structure predictions.
    """
    model = unwrap(model)
    loss_cfg = loss_cfg or LossConfig()
    acc = torch.zeros(len(_FIELDS), dtype=torch.float64, device=device)
    for idx in batches:
        batch = collate(frames, idx).to(device=device, dtype=dtype)
        out = model(batch, compute_force=True, compute_stress=True, training=False)
        if collect is not None:
            collect(idx, out, batch)
        e_err = ((out["energy"].detach() - batch.energy) / batch.natoms).double()
        f_err = (out["forces"].detach() - batch.forces).double()
        s_err = (out["stress"].detach() - batch.stress).double()
        sums = loss_sums(
            {k: v.detach() for k, v in out.items() if k in ("energy", "forces", "stress")}, batch, loss_cfg
        )
        acc += torch.stack(
            [
                torch.tensor(float(len(idx)), device=device, dtype=torch.float64),
                torch.tensor(float(f_err.numel()), device=device, dtype=torch.float64),
                torch.tensor(float(s_err.numel()), device=device, dtype=torch.float64),
                e_err.abs().sum(),
                (e_err**2).sum(),
                f_err.abs().sum(),
                (f_err**2).sum(),
                s_err.abs().sum(),
                sums["energy"].double(),
                sums["forces"].double(),
                sums["stress"].double(),
            ]
        )
    acc = all_reduce_sum(acc)
    v = dict(zip(_FIELDS, acc.tolist()))
    counts = {"energy": v["n_graphs"], "forces": v["n_force"], "stress": v["n_stress"]}
    loss = combine({t: torch.tensor(v[f"loss_{t}"]) for t in TERMS}, counts, loss_cfg).item()
    return {
        "loss": loss,
        "energy_mae_meV_per_atom": 1e3 * v["e_abs"] / v["n_graphs"],
        "energy_rmse_meV_per_atom": 1e3 * math.sqrt(v["e_sq"] / v["n_graphs"]),
        "force_mae_meV_per_A": 1e3 * v["f_abs"] / v["n_force"],
        "force_rmse_meV_per_A": 1e3 * math.sqrt(v["f_sq"] / v["n_force"]),
        "stress_mae_meV_per_A3": 1e3 * v["s_abs"] / v["n_stress"],
        "n_structures": int(v["n_graphs"]),
    }
