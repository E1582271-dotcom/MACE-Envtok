"""Optimiser, parameter groups and exponential moving average of the weights."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class OptimConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-6
    amsgrad: bool = True
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    clip_grad: float = 100.0
    ema_decay: float = 0.999
    lr_factor: float = 0.8  # ReduceLROnPlateau
    scheduler_patience: int = 5


def decays(name: str) -> bool:
    """Weight decay (L2) only on the product-basis weights and on the linear layer
    after the message aggregation; embeddings, radial MLPs, skip connections and
    readouts are not decayed. Same split as the paper's recipe."""
    return name.startswith("products.") or (
        name.startswith("interactions.") and name.endswith(".linear.weight")
    )


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (decay if decays(name) else no_decay).append(p)
    return [
        {"name": "decay", "params": decay, "weight_decay": weight_decay},
        {"name": "no_decay", "params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    # Plain Adam: weight decay is added to the gradient (L2), not decoupled as in AdamW.
    return torch.optim.Adam(
        parameter_groups(model, cfg.weight_decay),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        amsgrad=cfg.amsgrad,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: OptimConfig):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.scheduler_patience
    )


class EMA:
    """Exponential moving average of the parameters.

    After each optimiser step: shadow <- d * shadow + (1 - d) * param, with
    d = min(decay, (1 + n) / (10 + n)) so early updates are not dominated by the
    random initialisation. Validation and checkpoints use the averaged weights.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.num_updates = 0
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.shadow = [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def update(self) -> None:
        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for s, p in zip(self.shadow, self.params):
            s.mul_(d).add_(p.detach(), alpha=1.0 - d)

    @contextlib.contextmanager
    def average_parameters(self):
        """Temporarily load the averaged weights into the model."""
        backup = [p.detach().clone() for p in self.params]
        with torch.no_grad():
            for p, s in zip(self.params, self.shadow):
                p.copy_(s)
        try:
            yield
        finally:
            with torch.no_grad():
                for p, b in zip(self.params, backup):
                    p.copy_(b)

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": [s.clone() for s in self.shadow],
        }

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.num_updates = state["num_updates"]
        with torch.no_grad():
            for s, saved in zip(self.shadow, state["shadow"]):
                s.copy_(saved)
