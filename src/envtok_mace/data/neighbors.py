"""Radius-graph construction with periodic images.

Convention (same as mace-torch, so edge lists are interchangeable):
  edge_index[0] = sender, edge_index[1] = receiver,
  r = positions[receiver] - positions[sender] + unit_shifts @ cell
An atom may neighbour its own periodic image; only the zero-shift self pair is removed.
"""

from __future__ import annotations

import numpy as np
from matscipy.neighbours import neighbour_list


def radius_graph(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``edge_index`` (2, E) int32 and ``unit_shifts`` (E, 3) int8."""
    positions = np.asarray(positions, dtype=np.float64)
    cell = np.array(cell, dtype=np.float64)
    pbc = tuple(bool(p) for p in pbc)
    if not np.any(cell):
        cell = np.eye(3)
    # A non-periodic direction gets a cell vector long enough that no image is in range.
    span = np.abs(positions).max() + 1.0
    for d in range(3):
        if not pbc[d]:
            cell[d] = 0.0
            cell[d, d] = span * 5.0 * cutoff

    i, j, shifts = neighbour_list("ijS", pbc=pbc, cell=cell, positions=positions, cutoff=cutoff)
    keep = ~((i == j) & np.all(shifts == 0, axis=1))
    i, j, shifts = i[keep], j[keep], shifts[keep]
    if shifts.size and np.abs(shifts).max() > 127:
        raise ValueError("cell shift does not fit in int8; cell is far smaller than the cutoff")
    # matscipy returns D = pos[j] - pos[i] + S @ cell, i.e. a vector from i to j.
    # With sender = i and receiver = j that is r = pos[receiver] - pos[sender] + S @ cell.
    edge_index = np.stack((i, j)).astype(np.int32)
    return edge_index, shifts.astype(np.int8)


def edge_vectors(
    positions: np.ndarray, cell: np.ndarray, edge_index: np.ndarray, unit_shifts: np.ndarray
) -> np.ndarray:
    """Vectors from sender to receiver for an edge list built by ``radius_graph``."""
    sender, receiver = edge_index
    return positions[receiver] - positions[sender] + unit_shifts.astype(np.float64) @ cell
