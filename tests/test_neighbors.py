import numpy as np
import pytest
from ase.build import bulk
from ase.neighborlist import primitive_neighbor_list

from envtok_mace.data.neighbors import edge_vectors, radius_graph

R_MAX = 6.0


def _edge_set(edge_index, unit_shifts):
    return {(int(s), int(r), *map(int, sh)) for (s, r), sh in zip(edge_index.T, unit_shifts)}


def _ase_edge_set(positions, cell, pbc, cutoff):
    i, j, S = primitive_neighbor_list("ijS", pbc, cell, positions, cutoff, self_interaction=False)
    return {(int(a), int(b), *map(int, s)) for a, b, s in zip(i, j, S)}


def test_matches_ase_on_fixture(val40):
    for k in range(len(val40)):
        f = val40.get(k)
        ei, us = radius_graph(f["positions"], f["cell"], f["pbc"], R_MAX)
        assert _edge_set(ei, us) == _ase_edge_set(f["positions"], f["cell"], f["pbc"], R_MAX)


def test_stored_graphs_match_recomputed(val40):
    for k in range(len(val40)):
        f = val40.get(k)
        ei, us = radius_graph(f["positions"], f["cell"], f["pbc"], R_MAX)
        assert _edge_set(ei, us) == _edge_set(f["edge_index"], f["unit_shifts"])


def test_matches_mace_get_neighborhood(val40):
    neighborhood = pytest.importorskip("mace.data.neighborhood")
    for k in range(0, len(val40), 5):
        f = val40.get(k)
        ei, us = radius_graph(f["positions"], f["cell"], f["pbc"], R_MAX)
        ref_ei, _, ref_us, _ = neighborhood.get_neighborhood(
            f["positions"], R_MAX, pbc=tuple(bool(p) for p in f["pbc"]), cell=f["cell"].copy()
        )
        assert _edge_set(ei, us) == _edge_set(ref_ei, ref_us)


def test_tiny_cell_has_self_images_and_correct_lengths():
    atoms = bulk("Cu", "fcc", a=3.6)  # one atom, many periodic images inside 6 A
    ei, us = radius_graph(atoms.positions, atoms.cell[:], atoms.pbc, R_MAX)
    assert ei.shape[1] > 0 and np.all(ei[0] == 0) and np.all(ei[1] == 0)
    assert not np.any(np.all(us == 0, axis=1)), "zero-shift self edge must be removed"
    lengths = np.linalg.norm(edge_vectors(atoms.positions, atoms.cell[:], ei, us), axis=1)
    assert lengths.max() < R_MAX and lengths.min() > 2.0
    # every edge has its reverse (same pair, opposite shift)
    s = _edge_set(ei, us)
    assert all((r, snd, -a, -b, -c) in s for snd, r, a, b, c in s)


def test_non_periodic_molecule():
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.1], [5.0, 5.0, 5.0], [20.0, 0.0, 0.0]])
    ei, us = radius_graph(pos, np.zeros((3, 3)), np.array([False] * 3), R_MAX)
    assert np.all(us == 0)
    pairs = {(int(a), int(b)) for a, b in ei.T}
    assert (0, 1) in pairs and (1, 0) in pairs
    assert not any(3 in p for p in pairs)
