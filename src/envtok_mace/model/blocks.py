"""Building blocks of the MACE network.

Shapes: N nodes (atoms), E edges, C channels, Z elements, and node features
laid out e3nn-style ("mul_ir": channel-major within each irrep block).
"""

from __future__ import annotations

import torch
from e3nn import nn as e3nn_nn
from e3nn import o3
from torch import nn

from . import backends
from .cg import u_matrix


def scatter_sum(src: torch.Tensor, index: torch.Tensor, num: int) -> torch.Tensor:
    out = torch.zeros((num,) + src.shape[1:], dtype=src.dtype, device=src.device)
    return out.index_add_(0, index, src)


def conv_tp_instructions(irreps_node: o3.Irreps, irreps_sh: o3.Irreps, target: o3.Irreps):
    """Output irreps and "uvu" instructions of the edge tensor product.

    Every (node irrep) x (spherical harmonic) product that lands in ``target``
    becomes one path with its own per-channel weight. Outputs are sorted by irrep
    but not merged, which fixes the layout of the following linear layer.
    """
    outs, instructions = [], []
    for i, (mul, ir_in) in enumerate(irreps_node):
        for j, (_, ir_sh) in enumerate(irreps_sh):
            for ir_out in ir_in * ir_sh:
                if ir_out in target:
                    instructions.append((i, j, len(outs), "uvu", True))
                    outs.append((mul, ir_out))
    irreps_out, perm, _ = o3.Irreps(outs).sort()
    instructions = sorted(
        ((i, j, perm[k], mode, train) for i, j, k, mode, train in instructions), key=lambda ins: ins[2]
    )
    return irreps_out, instructions


class SymmetricContraction(nn.Module):
    """Element-dependent symmetric tensor product of a node's features with itself,
    up to order ``correlation`` (the MACE "product basis", eqs. 10-11).

    For each output irrep L and each order nu, the coupling tensor U_nu (fixed,
    from Clebsch-Gordan coefficients) maps nu copies of the node features to L.
    Learnable weights W_nu[z, k, c] mix the K coupling paths per element z and
    channel c. Evaluated in nested (Horner) form, highest order first:
        A_3 = U_3 . w_3,  B_3 = A_3 . x
        A_2 = U_2 . w_2 + B_3,  B_2 = A_2 . x
        A_1 = U_1 . w_1 + B_2,  B_1 = A_1 . x   -> output
    where ". x" contracts the last input axis with the node features.
    """

    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps, correlation: int, num_elements: int):
        super().__init__()
        irreps_in, irreps_out = o3.Irreps(irreps_in), o3.Irreps(irreps_out)
        self.num_channels = irreps_in.count((0, 1))
        assert all(mul == self.num_channels for mul, _ in irreps_in), "needs equal multiplicities"
        coupling = o3.Irreps([ir for _, ir in irreps_in])  # one copy of each irrep, dim D
        self.correlation = correlation
        self.out_irreps = [ir for _, ir in irreps_out]
        dtype = torch.get_default_dtype()
        self.weights = nn.ParameterList()
        for k, ir in enumerate(self.out_irreps):
            for nu in range(1, correlation + 1):
                U = u_matrix(coupling, ir, nu, dtype=dtype)
                self.register_buffer(f"U_{k}_{nu}", U)
                num_paths = U.shape[-1]
                self.weights.append(
                    nn.Parameter(torch.randn(num_elements, num_paths, self.num_channels) / num_paths)
                )

    def U(self, k: int, nu: int) -> torch.Tensor:
        return getattr(self, f"U_{k}_{nu}")

    def weight(self, k: int, nu: int) -> torch.Tensor:
        return self.weights[k * self.correlation + nu - 1]

    def forward(self, x: torch.Tensor, node_attrs: torch.Tensor) -> torch.Tensor:
        # x: [N, C, D]; node_attrs: one-hot [N, Z]
        outs = []
        for k in range(len(self.out_irreps)):
            out = None
            for nu in range(self.correlation, 0, -1):
                w = torch.einsum("nz,zkc->nkc", node_attrs, self.weight(k, nu))
                if out is None:
                    out = torch.einsum("m...ik,nkc,nci->ncm...", self.U(k, nu), w, x)
                else:
                    a = torch.einsum("m...k,nkc->ncm...", self.U(k, nu), w) + out
                    out = torch.einsum("ncm...i,nci->ncm...", a, x)
            outs.append(out.reshape(out.shape[0], -1))  # [N, C * (2L+1)]
        return torch.cat(outs, dim=-1)


class DensityInteraction(nn.Module):
    """Equivariant message passing with density normalisation.

    For node i with neighbours j:
        m_i = Linear( sum_j R(r_ij) * (h_j (x) Y(r_ij)) ) / (1 + rho_i),
        rho_i = sum_j tanh(w . R_ij)^2
    where R are radial weights from an MLP on the edge embedding, Y are spherical
    harmonics and (x) is the channel-wise tensor product. Dividing by a learned
    local density keeps messages on the same scale in sparse and dense
    environments instead of using a fixed average neighbour count.

    The first layer (``residual=False``) applies an element-dependent linear map to
    the message itself. Later layers (``residual=True``) instead return a skip
    connection: an element-dependent linear map of the incoming node features.
    """

    def __init__(
        self,
        irreps_node: o3.Irreps,
        irreps_sh: o3.Irreps,
        num_radial: int,
        target_irreps: o3.Irreps,
        num_elements: int,
        residual: bool,
        skip_irreps: o3.Irreps | None = None,
        radial_mlp: tuple[int, ...] = (64, 64, 64),
        backend: str = "e3nn",
        use_fallback: bool | None = None,
    ):
        super().__init__()
        irreps_node, target_irreps = o3.Irreps(irreps_node), o3.Irreps(target_irreps)
        attrs = o3.Irreps(f"{num_elements}x0e")
        self.residual = residual
        self.target_irreps = target_irreps
        fb = use_fallback
        self.linear_up = backends.linear(irreps_node, irreps_node, backend, fb)
        irreps_mid, instructions = conv_tp_instructions(irreps_node, irreps_sh, target_irreps)
        self.conv_tp = backends.MessageTP(
            irreps_node, irreps_sh, irreps_mid, instructions, target_irreps, backend, fb
        )
        self.conv_tp_weights = e3nn_nn.FullyConnectedNet(
            [num_radial, *radial_mlp, self.conv_tp.weight_numel], torch.nn.functional.silu
        )
        self.linear = backends.linear(irreps_mid, target_irreps, backend, fb)
        self.density_fn = e3nn_nn.FullyConnectedNet([num_radial, 1], torch.nn.functional.silu)
        if residual:
            self.skip_tp = backends.fully_connected_tp(
                irreps_node, attrs, o3.Irreps(skip_irreps), backend, fb
            )
        else:
            self.skip_tp = backends.fully_connected_tp(target_irreps, attrs, target_irreps, backend, fb)
        self._dims = [(mul, ir.dim) for mul, ir in target_irreps]

    def to_channels(self, x: torch.Tensor) -> torch.Tensor:
        """[N, irreps] -> [N, C, D] with all irreps of a channel side by side."""
        blocks, start = [], 0
        for mul, d in self._dims:
            blocks.append(x[:, start : start + mul * d].reshape(x.shape[0], mul, d))
            start += mul * d
        return torch.cat(blocks, dim=-1)

    def forward(self, node_feats, node_attrs, edge_sh, edge_radial, edge_index):
        sender, receiver = edge_index
        n = node_feats.shape[0]
        sc = self.skip_tp(node_feats, node_attrs) if self.residual else None
        h = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_radial)
        density = scatter_sum(torch.tanh(self.density_fn(edge_radial) ** 2), receiver, n)
        aggregated = self.conv_tp(h, edge_sh, tp_weights, sender, receiver, n)
        m = self.linear(aggregated) / (density + 1.0)
        if not self.residual:
            m = self.skip_tp(m, node_attrs)
        return self.to_channels(m), sc


class ProductBasis(nn.Module):
    """Symmetric contraction followed by a linear mix, plus the skip connection."""

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: int,
        num_elements: int,
        backend: str = "e3nn",
        use_fallback: bool | None = None,
    ):
        super().__init__()
        self.symmetric_contraction = backends.symmetric_contraction(
            irreps_in, irreps_out, correlation, num_elements, backend, use_fallback
        )
        self.linear = backends.linear(irreps_out, irreps_out, backend, use_fallback)

    def forward(self, x: torch.Tensor, node_attrs: torch.Tensor, sc: torch.Tensor | None) -> torch.Tensor:
        out = self.linear(self.symmetric_contraction(x, node_attrs))
        return out + sc if sc is not None else out


class MLPReadout(nn.Module):
    def __init__(self, irreps_in: o3.Irreps, hidden: int = 16):
        super().__init__()
        hidden_irreps = o3.Irreps(f"{hidden}x0e")
        self.linear_1 = o3.Linear(irreps_in, hidden_irreps)
        self.act = e3nn_nn.Activation(hidden_irreps, [torch.nn.functional.silu])
        self.linear_2 = o3.Linear(hidden_irreps, o3.Irreps("0e"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(x)))
