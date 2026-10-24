"""Generalised Clebsch-Gordan coefficients for coupling several copies of one
irreps into a single output irrep.

``coupling_basis(irreps, n)`` returns every way of coupling n copies of
``irreps`` into an irrep, built by chaining ordinary Wigner 3j symbols left to
right: ((x1 (x) x2) -> L12) (x) x3 -> L123 ... Each path is a tensor of shape
(2L+1, D, ..., D) with n input axes of size D = irreps.dim. The recursion and
the ordering of paths follow mace-torch's ``_wigner_nj``, so trained mace weights
can be loaded; the Wigner 3j symbols themselves come from e3nn.
"""

from __future__ import annotations

import torch
from e3nn import o3


def coupling_basis(irreps: o3.Irreps, n: int, dtype: torch.dtype | None = None):
    """List of (output irrep, coupling tensor) pairs, sorted by output irrep."""
    irreps = o3.Irreps(irreps)
    dim = irreps.dim
    if n == 1:
        eye = torch.eye(dim, dtype=dtype)
        paths, start = [], 0
        for mul, ir in irreps:
            for _ in range(mul):
                paths.append((ir, eye[start : start + ir.dim]))
                start += ir.dim
        return paths

    paths = []
    for ir_left, c_left in coupling_basis(irreps, n - 1, dtype):
        start = 0
        for mul, ir in irreps:
            for ir_out in ir_left * ir:
                # component normalisation: scale by sqrt(2L+1)
                w3j = o3.wigner_3j(ir_out.l, ir_left.l, ir.l, dtype=dtype) * ir_out.dim**0.5
                c = torch.einsum("jk,ijl->ikl", c_left.flatten(1), w3j)
                c = c.reshape(ir_out.dim, *([dim] * (n - 1)), ir.dim)
                for u in range(mul):
                    full = torch.zeros(ir_out.dim, *([dim] * n), dtype=dtype)
                    sl = slice(start + u * ir.dim, start + (u + 1) * ir.dim)
                    full[..., sl] = c
                    paths.append((ir_out, full))
            start += mul * ir.dim
    return sorted(paths, key=lambda item: item[0])


def u_matrix(irreps: o3.Irreps, ir_out: o3.Irrep, n: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Stack of the coupling paths that end in ``ir_out``.

    Shape (2L+1, D, ..., D, K) with n input axes and K paths. If no path exists,
    a single all-zero path is returned (its weights are then unused).
    """
    ir_out = o3.Irrep(ir_out)
    dim = o3.Irreps(irreps).dim
    paths = [c for ir, c in coupling_basis(irreps, n, dtype) if ir == ir_out]
    if not paths:
        return torch.zeros(ir_out.dim, *([dim] * n), 1, dtype=dtype)
    return torch.stack(paths, dim=-1)
