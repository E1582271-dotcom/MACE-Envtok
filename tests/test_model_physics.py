"""Symmetries and derivative consistency of the model (small random model, float64)."""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from envtok_mace.data.batch import Batch, collate
from envtok_mace.data.neighbors import radius_graph
from envtok_mace.model.mace import MACE, MACEConfig

R_MAX = 6.0


@pytest.fixture(scope="module")
def model():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1)
    m = MACE(MACEConfig(num_channels=16, radial_mlp=(16, 16)))
    # scale up the interaction part so derivatives are not tiny
    m.scale.fill_(10.0)
    yield m
    torch.set_default_dtype(old)


def _single(positions, numbers, cell, pbc=(True, True, True), energy=0.0):
    ei, us = radius_graph(positions, cell, np.array(pbc), R_MAX)
    n = len(numbers)
    return Batch(
        positions=torch.tensor(positions),
        numbers=torch.tensor(numbers, dtype=torch.long),
        batch=torch.zeros(n, dtype=torch.long),
        edge_index=torch.tensor(ei, dtype=torch.long),
        unit_shifts=torch.tensor(us, dtype=torch.float64),
        cell=torch.tensor(cell)[None],
        natoms=torch.tensor([n]),
        energy=torch.tensor([energy]),
        forces=torch.zeros(n, 3),
        stress=torch.zeros(1, 3, 3),
    )


def _frame(val40, k):
    f = val40.get(k)
    return np.array(f["positions"]), np.array(f["numbers"], dtype=np.int64), np.array(f["cell"])


def test_rotation_equivariance(model, val40):
    pos, z, cell = _frame(val40, 3)
    R = Rotation.random(random_state=7).as_matrix()
    a = model(_single(pos, z, cell), compute_stress=True)
    b = model(_single(pos @ R.T, z, cell @ R.T), compute_stress=True)
    Rt = torch.tensor(R)
    torch.testing.assert_close(b["energy"], a["energy"], rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(b["forces"], a["forces"] @ Rt.T, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(b["stress"][0], Rt @ a["stress"][0] @ Rt.T, rtol=1e-9, atol=1e-11)


def test_translation_invariance(model, val40):
    pos, z, cell = _frame(val40, 5)
    a = model(_single(pos, z, cell))
    b = model(_single(pos + np.array([0.37, -1.2, 2.5]), z, cell))
    torch.testing.assert_close(b["energy"], a["energy"], rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(b["forces"], a["forces"], rtol=1e-9, atol=1e-10)


def test_permutation_equivariance(model, val40):
    pos, z, cell = _frame(val40, 11)
    perm = np.random.default_rng(0).permutation(len(z))
    a = model(_single(pos, z, cell))
    b = model(_single(pos[perm], z[perm], cell))
    torch.testing.assert_close(b["energy"], a["energy"], rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(b["forces"], a["forces"][perm], rtol=1e-9, atol=1e-10)


def test_forces_match_finite_differences(model, val40):
    pos, z, cell = _frame(val40, 2)
    forces = model(_single(pos, z, cell))["forces"].detach().numpy()
    h = 1e-5
    for atom, axis in [(0, 0), (3, 1), (7, 2)]:
        plus, minus = pos.copy(), pos.copy()
        plus[atom, axis] += h
        minus[atom, axis] -= h
        # same edge list for both points: only positions move
        e_p = model(_single(plus, z, cell), compute_force=False)["energy"].item()
        e_m = model(_single(minus, z, cell), compute_force=False)["energy"].item()
        assert forces[atom, axis] == pytest.approx(-(e_p - e_m) / (2 * h), rel=1e-5, abs=1e-8)


def test_stress_matches_finite_strain(model, val40):
    pos, z, cell = _frame(val40, 4)
    stress = model(_single(pos, z, cell), compute_stress=True)["stress"][0].detach().numpy()
    volume = abs(np.linalg.det(cell))
    h = 1e-6
    for i, j in [(0, 0), (1, 2), (2, 2)]:
        eps = np.zeros((3, 3))
        eps[i, j] = eps[j, i] = h / (2 if i != j else 1)

        def energy(sign, eps=eps):
            d = np.eye(3) + sign * eps
            return model(_single(pos @ d, z, cell @ d), compute_force=False)["energy"].item()

        de = (energy(+1) - energy(-1)) / (2 * h)
        assert stress[i, j] == pytest.approx(de / volume, rel=1e-5, abs=1e-9)


def test_batched_equals_single(model, val40):
    batch = collate(val40, np.arange(6))
    out = model(batch.to(dtype=torch.float64), compute_stress=True)
    for k in range(6):
        single = model(collate(val40, [k]).to(dtype=torch.float64), compute_stress=True)
        torch.testing.assert_close(out["energy"][k], single["energy"][0], rtol=1e-12, atol=1e-10)
        torch.testing.assert_close(out["stress"][k], single["stress"][0], rtol=1e-10, atol=1e-12)
    a0, a1 = batch.natoms[:2].tolist()
    single1 = model(collate(val40, [1]).to(dtype=torch.float64))
    torch.testing.assert_close(out["forces"][a0 : a0 + a1], single1["forces"], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("z", [57, 84, 92, 118])
def test_elements_outside_the_table_raise_cleanly(model, z):
    """Lanthanides map to -1 inside the table; Z >= 84 lies past the largest supported
    element and must not index out of bounds (a device assertion on a GPU)."""
    numbers = torch.tensor([14, z])
    with pytest.raises(ValueError, match="outside the model's element table"):
        model.species_index(numbers)
