"""The MACE interatomic potential: energy as a sum of atomic energies, with
forces and stress obtained by differentiating that energy.

Energy of a structure:
    E = sum_i E0[z_i] + sum_i [ ZBL_i + sum_layers readout(h_i^(t)) ]
Forces F = -dE/dr and stress sigma = (1/V) dE/d(strain), both by autograd, so
the forces are exactly conservative.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from e3nn import o3
from torch import nn

from ..data.constants import E0S, Z_TABLE
from .blocks import DensityInteraction, MLPReadout, ProductBasis, scatter_sum
from .radial import RadialEmbedding, ZBLRepulsion


@dataclass
class MACEConfig:
    """Defaults reproduce the MACE-OMAT-0 "medium" architecture used in the paper."""

    r_max: float = 6.0
    num_bessel: int = 10
    cutoff_p: int = 5
    max_ell: int = 3            # highest spherical-harmonic degree on edges
    num_channels: int = 128
    max_L: int = 1              # highest irrep kept in hidden node features
    correlation: int = 3        # body order of the product basis (up to 4-body per layer)
    num_interactions: int = 2
    radial_mlp: tuple[int, ...] = (64, 64, 64)
    readout_hidden: int = 16
    agnesi: bool = True
    pair_repulsion: bool = True
    z_table: tuple[int, ...] = Z_TABLE
    e0s: dict[int, float] = field(default_factory=lambda: dict(E0S))
    scale: float = 1.0
    shift: float = 0.0
    backend: str = "e3nn"       # "e3nn" (reference) or "cueq" (cuEquivariance kernels)
    cueq_fallback: bool | None = None  # cueq only: True forces the pure-PyTorch implementation

    def to_dict(self) -> dict:
        d = asdict(self)
        d["e0s"] = {int(k): float(v) for k, v in self.e0s.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MACEConfig":
        d = dict(d)
        d["radial_mlp"] = tuple(d["radial_mlp"])
        d["z_table"] = tuple(d["z_table"])
        d["e0s"] = {int(k): float(v) for k, v in d["e0s"].items()}
        return cls(**d)


class MACE(nn.Module):
    def __init__(self, config: MACEConfig | None = None):
        super().__init__()
        cfg = config or MACEConfig()
        self.config = cfg
        C, Z = cfg.num_channels, len(cfg.z_table)
        dtype = torch.get_default_dtype()

        self.register_buffer("z_table", torch.tensor(cfg.z_table, dtype=torch.long))
        self.register_buffer("e0", torch.tensor([cfg.e0s[z] for z in cfg.z_table], dtype=dtype))
        self.register_buffer("scale", torch.tensor(cfg.scale, dtype=dtype))
        self.register_buffer("shift", torch.tensor(cfg.shift, dtype=dtype))

        attrs = o3.Irreps(f"{Z}x0e")
        hidden = o3.Irreps([(C, (l, (-1) ** l)) for l in range(cfg.max_L + 1)])
        scalars = o3.Irreps(f"{C}x0e")
        sh = o3.Irreps.spherical_harmonics(cfg.max_ell)
        inter = (sh * C).sort().irreps.simplify()  # C x (0e + 1o + 2e + 3o)

        self.node_embedding = o3.Linear(attrs, scalars)
        self.radial = RadialEmbedding(cfg.r_max, cfg.num_bessel, cfg.cutoff_p, agnesi=cfg.agnesi)
        self.zbl = ZBLRepulsion(cfg.cutoff_p) if cfg.pair_repulsion else None
        self.spherical_harmonics = o3.SphericalHarmonics(sh, normalize=True, normalization="component")

        self.interactions = nn.ModuleList()
        self.products = nn.ModuleList()
        self.readouts = nn.ModuleList()
        for t in range(cfg.num_interactions):
            first, last = t == 0, t == cfg.num_interactions - 1
            node_in = scalars if first else hidden
            out = scalars if last else hidden
            self.interactions.append(DensityInteraction(
                node_in, sh, cfg.num_bessel, inter, Z, residual=not first,
                skip_irreps=None if first else out, radial_mlp=cfg.radial_mlp, backend=cfg.backend,
                use_fallback=cfg.cueq_fallback))
            self.products.append(ProductBasis(inter, out, cfg.correlation, Z, backend=cfg.backend,
                                              use_fallback=cfg.cueq_fallback))
            self.readouts.append(MLPReadout(out, cfg.readout_hidden) if last else o3.Linear(out, o3.Irreps("0e")))

    # ------------------------------------------------------------------ helpers
    def species_index(self, numbers: torch.Tensor) -> torch.Tensor:
        """Map atomic numbers to rows of the element table.

        The lookup covers every element (Z <= 118), so an atomic number outside the
        table maps to -1 and raises below. A table only as long as the largest
        supported Z would let Z >= 84 index past its end, which on a GPU is a
        device-side assertion that leaves the process unable to use CUDA.
        """
        lookup = torch.full((119,), -1, dtype=torch.long, device=numbers.device)
        lookup[self.z_table] = torch.arange(len(self.z_table), device=numbers.device)
        idx = lookup[numbers]
        if (idx < 0).any():
            raise ValueError("structure contains an element outside the model's element table")
        return idx

    # ------------------------------------------------------------------ forward
    def forward(self, batch, compute_force: bool = True, compute_stress: bool = False,
                training: bool = False, return_features: bool = False) -> dict[str, torch.Tensor]:
        """``batch`` provides positions [N,3], numbers [N], edge_index [2,E],
        unit_shifts [E,3], cell [G,3,3], batch [N] (graph of each atom), num_graphs."""
        numbers = batch.numbers
        sender, receiver = batch.edge_index
        num_graphs = batch.num_graphs
        n = numbers.shape[0]
        graph = batch.batch

        positions = batch.positions
        if compute_force or compute_stress:
            positions = positions.detach().requires_grad_(True)
        cell = batch.cell
        displacement = None
        if compute_stress:
            # Differentiate w.r.t. a symmetric strain applied to positions and cell.
            displacement = torch.zeros((num_graphs, 3, 3), dtype=positions.dtype, device=positions.device,
                                       requires_grad=True)
            strain = 0.5 * (displacement + displacement.transpose(-1, -2))
            pos = positions + torch.einsum("ni,nij->nj", positions, strain[graph])
            cell = cell + torch.matmul(cell, strain)
        else:
            pos = positions
        shifts = torch.einsum("ei,eij->ej", batch.unit_shifts.to(pos.dtype), cell[graph[sender]])
        vectors = pos[receiver] - pos[sender] + shifts
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)

        species = self.species_index(numbers)
        node_attrs = torch.nn.functional.one_hot(species, len(self.z_table)).to(pos.dtype)
        z_s, z_r = numbers[sender], numbers[receiver]

        node_e0 = self.e0[species]
        edge_sh = self.spherical_harmonics(vectors)
        edge_radial = self.radial(lengths, z_s, z_r)
        node_energy = (self.zbl(lengths, z_s, z_r, receiver, n) if self.zbl is not None
                       else torch.zeros(n, dtype=pos.dtype, device=pos.device))

        h = self.node_embedding(node_attrs)
        features = []
        for interaction, product, readout in zip(self.interactions, self.products, self.readouts):
            x, sc = interaction(h, node_attrs, edge_sh, edge_radial, batch.edge_index)
            h = product(x, node_attrs, sc)
            features.append(h)
            node_energy = node_energy + readout(h).squeeze(-1)

        node_energy = self.scale * node_energy + self.shift
        inter_e = scatter_sum(node_energy, graph, num_graphs)
        energy = scatter_sum(node_e0, graph, num_graphs) + inter_e
        out = {"energy": energy, "node_energy": node_e0 + node_energy}

        if compute_force or compute_stress:
            inputs = [positions] + ([displacement] if compute_stress else [])
            grads = torch.autograd.grad([inter_e.sum()], inputs, create_graph=training,
                                        retain_graph=training, allow_unused=True)
            g_pos = grads[0] if grads[0] is not None else torch.zeros_like(positions)
            out["forces"] = -g_pos
            if compute_stress:
                volume = torch.linalg.det(batch.cell).abs().view(-1, 1, 1)
                out["stress"] = grads[1] / volume
                out["virials"] = -grads[1]
        if return_features:
            # invariant (scalar) channels of every layer, e.g. for latent-space analysis
            C = self.config.num_channels
            out["node_features"] = torch.cat([f[:, :C] for f in features], dim=-1)
        return out
