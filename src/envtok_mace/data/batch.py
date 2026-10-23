"""Turn a set of frames into one disconnected graph for the model.

Atoms of all structures are concatenated; ``batch[i]`` tells which structure
atom i belongs to, and edge indices are shifted from per-structure to global
atom numbering.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import torch

from .frames import Frames

# ASE Voigt order xx, yy, zz, yz, xz, xy -> full 3x3 matrix
_VOIGT_TO_FULL = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])


@dataclass
class Batch:
    positions: torch.Tensor    # [N, 3]
    numbers: torch.Tensor      # [N] atomic numbers
    batch: torch.Tensor        # [N] structure index of each atom
    edge_index: torch.Tensor   # [2, E] global atom indices (sender, receiver)
    unit_shifts: torch.Tensor  # [E, 3]
    cell: torch.Tensor         # [G, 3, 3]
    natoms: torch.Tensor       # [G]
    energy: torch.Tensor       # [G]
    forces: torch.Tensor       # [N, 3]
    stress: torch.Tensor       # [G, 3, 3]

    @property
    def num_graphs(self) -> int:
        return int(self.natoms.shape[0])

    def to(self, device=None, dtype=None) -> "Batch":
        out = {}
        for f in fields(self):
            t = getattr(self, f.name)
            if dtype is not None and t.is_floating_point():
                t = t.to(dtype)
            out[f.name] = t.to(device, non_blocking=True) if device is not None else t
        return Batch(**out)

    def pin_memory(self) -> "Batch":
        return Batch(**{f.name: getattr(self, f.name).pin_memory() for f in fields(self)})


def voigt_to_full(stress: np.ndarray) -> np.ndarray:
    return stress[..., _VOIGT_TO_FULL]


def collate(frames: Frames, indices) -> Batch:
    """Build a Batch from the frames at ``indices`` (requires neighbour lists)."""
    if not frames.has_edges:
        raise ValueError("frames have no neighbour lists; run `envtok-mace preprocess` first")
    sub = frames.subset(np.asarray(indices))
    natoms = sub.natoms
    nedges = sub.nedges
    graph = np.repeat(np.arange(len(sub)), natoms)
    offsets = np.repeat(sub.atom_ptr[:-1], nedges)
    edge_index = sub.edge_index.astype(np.int64) + offsets[None, :]
    return Batch(
        positions=torch.from_numpy(sub.positions),
        numbers=torch.from_numpy(sub.numbers.astype(np.int64)),
        batch=torch.from_numpy(graph),
        edge_index=torch.from_numpy(edge_index),
        unit_shifts=torch.from_numpy(sub.unit_shifts.astype(np.float64)),
        cell=torch.from_numpy(sub.cell),
        natoms=torch.from_numpy(natoms),
        energy=torch.from_numpy(sub.energy),
        forces=torch.from_numpy(sub.forces),
        stress=torch.from_numpy(voigt_to_full(sub.stress)),
    )
