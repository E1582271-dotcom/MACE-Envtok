"""Radial parts of the model: distance embedding, smooth cutoff, and ZBL repulsion.

All functions act on edge lengths ``r`` of shape [E, 1].
"""

from __future__ import annotations

import math

import ase.data
import torch
from torch import nn


def polynomial_cutoff(r: torch.Tensor, r_max: torch.Tensor | float, p: int = 5) -> torch.Tensor:
    """Smooth envelope going from 1 at r=0 to 0 at r=r_max, with p-1 vanishing derivatives."""
    x = r / r_max
    env = (
        1.0
        - ((p + 1.0) * (p + 2.0) / 2.0) * x**p
        + p * (p + 2.0) * x ** (p + 1)
        - (p * (p + 1.0) / 2.0) * x ** (p + 2)
    )
    return env * (r < r_max)


class RadialEmbedding(nn.Module):
    """Bessel basis on an Agnesi-transformed distance, multiplied by a polynomial cutoff.

    The Agnesi transform rescales each distance by the pair's covalent-radius sum,
    so the basis sees chemically comparable distances for every element pair:
        x = r / r0,   r0 = (R_cov(Z_s) + R_cov(Z_r)) / 2
        t = 1 / (1 + a x^q / (1 + x^(q - p)))
    The cutoff is applied to the raw distance r.
    """

    def __init__(self, r_max: float = 6.0, num_bessel: int = 10, cutoff_p: int = 5,
                 agnesi: bool = True, a: float = 1.0805, q: float = 0.9183, p: float = 4.5791):
        super().__init__()
        dtype = torch.get_default_dtype()
        self.num_bessel = num_bessel
        self.cutoff_p = cutoff_p
        self.agnesi = agnesi
        self.register_buffer("r_max", torch.tensor(r_max, dtype=dtype))
        self.register_buffer("frequencies", math.pi / r_max * torch.arange(1, num_bessel + 1, dtype=dtype))
        self.register_buffer("prefactor", torch.tensor(math.sqrt(2.0 / r_max), dtype=dtype))
        self.register_buffer("agnesi_a", torch.tensor(a, dtype=dtype))
        self.register_buffer("agnesi_q", torch.tensor(q, dtype=dtype))
        self.register_buffer("agnesi_p", torch.tensor(p, dtype=dtype))
        self.register_buffer("covalent_radii", torch.tensor(ase.data.covalent_radii, dtype=dtype))

    def agnesi_transform(self, r: torch.Tensor, z_sender: torch.Tensor, z_receiver: torch.Tensor) -> torch.Tensor:
        r0 = 0.5 * (self.covalent_radii[z_sender] + self.covalent_radii[z_receiver]).unsqueeze(-1)
        x = r / r0
        q, p = self.agnesi_q, self.agnesi_p
        return 1.0 / (1.0 + self.agnesi_a * x**q / (1.0 + x ** (q - p)))

    def bessel(self, r: torch.Tensor) -> torch.Tensor:
        return self.prefactor * torch.sin(self.frequencies * r) / r

    def forward(self, r: torch.Tensor, z_sender: torch.Tensor, z_receiver: torch.Tensor) -> torch.Tensor:
        cutoff = polynomial_cutoff(r, self.r_max, self.cutoff_p)
        t = self.agnesi_transform(r, z_sender, z_receiver) if self.agnesi else r
        return self.bessel(t) * cutoff  # [E, num_bessel]


class ZBLRepulsion(nn.Module):
    """Ziegler-Biersack-Littmark screened nuclear repulsion, faded out by the
    covalent-radius sum of each pair. Keeps atoms from collapsing onto each other
    in regions with no training data. Returns one energy per atom (eV)."""

    def __init__(self, cutoff_p: int = 5):
        super().__init__()
        dtype = torch.get_default_dtype()
        self.cutoff_p = cutoff_p
        self.register_buffer("c", torch.tensor([0.1818, 0.5099, 0.2802, 0.02817], dtype=dtype))
        self.register_buffer("b", torch.tensor([3.2, 0.9423, 0.4028, 0.2016], dtype=dtype))
        self.register_buffer("covalent_radii", torch.tensor(ase.data.covalent_radii, dtype=dtype))
        self.register_buffer("a_exp", torch.tensor(0.300, dtype=dtype))
        self.register_buffer("a_prefactor", torch.tensor(0.4543, dtype=dtype))

    def forward(self, r: torch.Tensor, z_sender: torch.Tensor, z_receiver: torch.Tensor,
                receiver: torch.Tensor, num_nodes: int) -> torch.Tensor:
        # atomic numbers as floats of the input precision, independent of torch's default dtype
        zs = z_sender.to(r.dtype).unsqueeze(-1)
        zr = z_receiver.to(r.dtype).unsqueeze(-1)
        # screening length (Angstrom); 0.529 converts Bohr to Angstrom
        a = self.a_prefactor * 0.529 / (zs**self.a_exp + zr**self.a_exp)
        phi = (self.c * torch.exp(-self.b * (r / a))).sum(-1, keepdim=True)
        v = 14.3996 * zs * zr / r * phi  # e^2 / (4 pi eps0) = 14.3996 eV A
        r_fade = (self.covalent_radii[z_sender] + self.covalent_radii[z_receiver]).unsqueeze(-1)
        v = 0.5 * v * polynomial_cutoff(r, r_fade, self.cutoff_p)  # each pair appears twice
        out = torch.zeros(num_nodes, 1, dtype=r.dtype, device=r.device)
        out.index_add_(0, receiver, v)
        return out.squeeze(-1)
