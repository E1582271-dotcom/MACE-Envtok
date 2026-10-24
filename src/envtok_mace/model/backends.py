"""Two interchangeable implementations of the heavy equivariant operations.

"e3nn":  reference implementation (e3nn modules + our own symmetric contraction).
"cueq":  NVIDIA cuEquivariance kernels (fused gather/tensor-product/scatter for the
         messages, fused symmetric contraction). Same weights and same results,
         up to floating-point rounding; tested against "e3nn" in tests/test_backends.py.

Every factory returns a module with the same call signature for both backends,
and activations always use the e3nn "mul_ir" layout outside these modules.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
from e3nn import o3
from torch import nn

BACKENDS = ("e3nn", "cueq")


def _cue():
    try:
        import cuequivariance as cue
        import cuequivariance_torch as cuet
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "backend 'cueq' needs cuequivariance-torch "
            "(pip install cuequivariance-torch cuequivariance-ops-torch-cu12)"
        ) from exc
    return cue, cuet


_O3_E3NN = None


def o3_e3nn_group():
    """cuEquivariance's O(3) group with e3nn's conventions (Clebsch-Gordan
    coefficients from e3nn's Wigner 3j symbols, irreps ordered as in e3nn), so
    that weights and results match the e3nn backend. Same adapter as mace-torch."""
    global _O3_E3NN
    if _O3_E3NN is not None:
        return _O3_E3NN
    cue, _ = _cue()

    class O3_e3nn(cue.O3):
        def __mul__(rep1, rep2):  # noqa: N805
            return [O3_e3nn(l=ir.l, p=ir.p) for ir in cue.O3.__mul__(rep1, rep2)]

        @classmethod
        def clebsch_gordan(cls, rep1, rep2, rep3):
            rep1, rep2, rep3 = cls._from(rep1), cls._from(rep2), cls._from(rep3)
            if rep1.p * rep2.p == rep3.p:
                return o3.wigner_3j(rep1.l, rep2.l, rep3.l).numpy()[None] * np.sqrt(rep3.dim)
            return np.zeros((0, rep1.dim, rep2.dim, rep3.dim))

        def __lt__(rep1, rep2):  # noqa: N805
            rep2 = rep1._from(rep2)
            return (rep1.l, rep1.p) < (rep2.l, rep2.p)

        @classmethod
        def iterator(cls):
            for ell in itertools.count(0):
                yield O3_e3nn(l=ell, p=1 * (-1) ** ell)
                yield O3_e3nn(l=ell, p=-1 * (-1) ** ell)

    _O3_E3NN = O3_e3nn
    return _O3_E3NN


def _irreps(irreps):
    cue, _ = _cue()
    return cue.Irreps(o3_e3nn_group(), str(o3.Irreps(irreps)))


# ---------------------------------------------------------------- factories
# use_fallback (cueq only): None = CUDA kernels when the kernel package is installed,
# True = pure-PyTorch reference implementation (e.g. tests on a machine without GPU).
#
# Linear layers and the element-dependent skip (a tensor product with one-hot
# element attributes) always use cuEquivariance's PyTorch implementation, which is
# a few matrix multiplications: the generic fused kernel is many times slower for
# them, especially in the double backward pass of force training. Only the message
# tensor product and the symmetric contraction run as fused CUDA kernels.
def linear(irreps_in, irreps_out, backend: str, use_fallback: bool | None = None) -> nn.Module:
    if backend == "e3nn":
        return o3.Linear(irreps_in, irreps_out)
    cue, cuet = _cue()
    return cuet.Linear(_irreps(irreps_in), _irreps(irreps_out), layout=cue.mul_ir, use_fallback=True)


def fully_connected_tp(
    irreps1, irreps2, irreps_out, backend: str, use_fallback: bool | None = None
) -> nn.Module:
    if backend == "e3nn":
        return o3.FullyConnectedTensorProduct(irreps1, irreps2, irreps_out)
    cue, cuet = _cue()
    return cuet.FullyConnectedTensorProduct(
        _irreps(irreps1), _irreps(irreps2), _irreps(irreps_out), layout=cue.mul_ir, use_fallback=True
    )


class MessageTP(nn.Module):
    """sum over edges j->i of TP(h_j, Y_ij; w_ij), with per-edge weights w."""

    def __init__(
        self,
        irreps_node,
        irreps_sh,
        irreps_mid,
        instructions,
        target_irreps,
        backend: str,
        use_fallback: bool | None = None,
    ):
        super().__init__()
        self.backend = backend
        if backend == "e3nn":
            self.tp = o3.TensorProduct(
                irreps_node,
                irreps_sh,
                irreps_mid,
                instructions=instructions,
                shared_weights=False,
                internal_weights=False,
            )
        else:
            cue, cuet = _cue()
            self.tp = cuet.ChannelWiseTensorProduct(
                _irreps(irreps_node),
                _irreps(irreps_sh),
                filter_irreps_out=_irreps(target_irreps),
                layout=cue.mul_ir,
                shared_weights=False,
                internal_weights=False,
                use_fallback=use_fallback,
            )
            if str(self.tp.irreps_out) != str(_irreps(irreps_mid)):
                raise RuntimeError(f"cueq output irreps {self.tp.irreps_out} != {irreps_mid}")
        self.weight_numel = self.tp.weight_numel

    def forward(self, h, edge_sh, weights, sender, receiver, num_nodes):
        if self.backend == "e3nn":
            messages = self.tp(h[sender], edge_sh, weights)
            out = torch.zeros(num_nodes, messages.shape[1], dtype=messages.dtype, device=messages.device)
            return out.index_add_(0, receiver, messages)
        return self.tp(h, edge_sh, weights, indices_1=sender, indices_out=receiver, size_out=num_nodes)


class CueqSymmetricContraction(nn.Module):
    """cuEquivariance symmetric contraction in the original MACE coupling basis."""

    def __init__(
        self, irreps_in, irreps_out, correlation: int, num_elements: int, use_fallback: bool | None = None
    ):
        super().__init__()
        cue, cuet = _cue()
        self.correlation = correlation
        self.out_irreps = [ir for _, ir in o3.Irreps(irreps_out)]
        self.sc = cuet.SymmetricContraction(
            _irreps(irreps_in),
            _irreps(irreps_out),
            contraction_degree=correlation,
            num_elements=num_elements,
            layout_in=cue.ir_mul,
            layout_out=cue.mul_ir,
            original_mace=True,
            # the kernel computes in math_dtype; it must match the model precision
            dtype=torch.get_default_dtype(),
            math_dtype=torch.get_default_dtype(),
            use_fallback=use_fallback,
        )
        from .cg import u_matrix

        coupling = o3.Irreps([ir for _, ir in o3.Irreps(irreps_in)])
        self.path_counts = {
            (k, nu): u_matrix(coupling, ir, nu).shape[-1]
            for k, ir in enumerate(self.out_irreps)
            for nu in range(1, correlation + 1)
        }

    def forward(self, x: torch.Tensor, node_attrs: torch.Tensor) -> torch.Tensor:
        # x: [N, C, D] -> "ir_mul" layout expected by the kernel
        return self.sc(x.transpose(1, 2).flatten(1), node_attrs.argmax(-1).int())


def symmetric_contraction(
    irreps_in, irreps_out, correlation: int, num_elements: int, backend: str, use_fallback: bool | None = None
) -> nn.Module:
    if backend == "e3nn":
        from .blocks import SymmetricContraction

        return SymmetricContraction(irreps_in, irreps_out, correlation, num_elements)
    return CueqSymmetricContraction(irreps_in, irreps_out, correlation, num_elements, use_fallback)


# --------------------------------------------------------- weight conversion
def convert_state_dict(state: dict, src_model, dst_model) -> dict:
    """Map a state dict of ``src_model`` onto ``dst_model`` (same architecture,
    possibly different backend)."""
    dst = dst_model.state_dict()
    out = {}
    for t, (p_src, p_dst) in enumerate(zip(src_model.products, dst_model.products)):
        src_sc, dst_sc = p_src.symmetric_contraction, p_dst.symmetric_contraction
        prefix = f"products.{t}.symmetric_contraction."
        corr, n_out = src_sc.correlation, len(src_sc.out_irreps)
        # cuEquivariance stores one tensor: per output irrep, orders corr..1 along dim 1
        order = [(k, nu) for k in range(n_out) for nu in range(corr, 0, -1)]
        if isinstance(src_sc, CueqSymmetricContraction):
            w = state[prefix + "sc.weight"].reshape(
                -1, sum(src_sc.path_counts.values()), state[prefix + "sc.weight"].shape[-1]
            )
            parts = dict(zip(order, torch.split(w, [src_sc.path_counts[kn] for kn in order], dim=1)))
        else:
            parts = {(k, nu): state[prefix + f"weights.{k * corr + nu - 1}"] for k, nu in order}
        if isinstance(dst_sc, CueqSymmetricContraction):
            w = torch.cat([parts[kn] for kn in order], dim=1)
            out[prefix + "sc.weight"] = w.reshape(dst[prefix + "sc.weight"].shape)
        else:
            for k, nu in order:
                out[prefix + f"weights.{k * corr + nu - 1}"] = parts[(k, nu)]
        for key in dst:  # constant buffers of the e3nn contraction (U matrices)
            if key.startswith(prefix) and key not in out:
                out[key] = state.get(key, dst[key])
    params = {k for k, _ in dst_model.named_parameters()}
    for key, value in dst.items():
        if key in out:
            continue
        if key not in state:
            if key in params:
                raise KeyError(f"no source tensor for parameter {key}")
            out[key] = value  # backend-specific constant buffer
            continue
        src = state[key]
        if src.numel() != value.numel():
            raise ValueError(f"{key}: {tuple(src.shape)} cannot become {tuple(value.shape)}")
        out[key] = src.reshape(value.shape)
    return out
