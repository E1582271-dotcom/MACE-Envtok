"""Energy/force/stress loss ("universal" loss of the paper's recipe).

Each term is a Huber loss averaged over its elements:
  energy: per-atom energy of each structure, one element per structure
  forces: every force component, with a smaller Huber delta for atoms under very
          large reference forces (|F| >= 100, 200, 300 eV/A -> delta x 0.7, 0.4, 0.1)
  stress: all 9 stress components of each structure

The terms are returned as *sums* together with their element counts. The trainer
divides by counts summed over every rank and every gradient-accumulation
micro-batch, so the gradient is that of the mean over the whole global batch,
no matter how it is split across GPUs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

TERMS = ("energy", "forces", "stress")


@dataclass
class LossConfig:
    energy_weight: float = 1.0
    forces_weight: float = 10.0
    stress_weight: float = 10.0
    huber_delta: float = 0.01

    def weights(self) -> dict[str, float]:
        return {"energy": self.energy_weight, "forces": self.forces_weight, "stress": self.stress_weight}


def huber(err: torch.Tensor, delta: torch.Tensor | float) -> torch.Tensor:
    """Elementwise Huber loss (same convention as torch.nn.functional.huber_loss)."""
    a = err.abs()
    return torch.where(a < delta, 0.5 * err**2, delta * (a - 0.5 * delta))


def force_deltas(ref_forces: torch.Tensor, delta: float) -> torch.Tensor:
    """Per-atom Huber delta, reduced for atoms under very large reference forces."""
    norm = torch.linalg.norm(ref_forces, dim=-1)
    factor = torch.ones_like(norm)
    factor = torch.where(norm >= 100.0, torch.full_like(norm, 0.7), factor)
    factor = torch.where(norm >= 200.0, torch.full_like(norm, 0.4), factor)
    factor = torch.where(norm >= 300.0, torch.full_like(norm, 0.1), factor)
    return (delta * factor).unsqueeze(-1)


def loss_sums(pred: dict[str, torch.Tensor], batch, cfg: LossConfig) -> dict[str, torch.Tensor]:
    """Sum of the elementwise loss of each term over this (micro-)batch."""
    natoms = batch.natoms.to(pred["energy"].dtype)
    e_err = (pred["energy"] - batch.energy) / natoms
    f_err = pred["forces"] - batch.forces
    s_err = pred["stress"] - batch.stress
    return {
        "energy": huber(e_err, cfg.huber_delta).sum(),
        "forces": huber(f_err, force_deltas(batch.forces, cfg.huber_delta)).sum(),
        "stress": huber(s_err, cfg.huber_delta).sum(),
    }


def loss_counts(natoms) -> dict[str, int]:
    """Number of elements each term averages over, for structures with ``natoms`` atoms."""
    g, a = len(natoms), int(sum(int(n) for n in natoms))
    return {"energy": g, "forces": 3 * a, "stress": 9 * g}


def combine(sums: dict[str, torch.Tensor], counts: dict[str, float], cfg: LossConfig) -> torch.Tensor:
    w = cfg.weights()
    return sum(w[t] * sums[t] / counts[t] for t in TERMS)
