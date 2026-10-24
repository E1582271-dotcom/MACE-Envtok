"""The ASE calculator must agree with the batched pipeline and with finite differences."""

import numpy as np
import pytest
import torch

from envtok_mace.data.batch import collate
from envtok_mace.eval.ase_calc import EnvTokCalculator


@pytest.fixture(scope="module")
def calc(envtok_model_path):
    return EnvTokCalculator(envtok_model_path, device="cpu", dtype=torch.float64)


def test_matches_the_batched_pipeline(val40, calc):
    """Same edges, same energies, forces and Voigt stress as training-time batching."""
    idx = np.arange(6)
    batch = collate(val40, idx).to(dtype=torch.float64)
    ref = calc.model(batch, compute_force=True, compute_stress=True, training=False)
    a0 = 0
    for g, i in enumerate(idx):
        atoms = val40.to_atoms(int(i))
        atoms.calc = calc
        n = len(atoms)
        assert atoms.get_potential_energy() == pytest.approx(ref["energy"][g].item(), rel=1e-12)
        assert np.allclose(
            atoms.get_forces(), ref["forces"][a0 : a0 + n].detach().numpy(), rtol=1e-10, atol=1e-12
        )
        s = ref["stress"][g].detach().numpy()
        voigt = np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])
        assert np.allclose(atoms.get_stress(), voigt, rtol=1e-10, atol=1e-13)
        a0 += n


def test_forces_and_stress_are_derivatives_of_the_energy(val40, calc):
    """Finite differences fix the sign and the Voigt order independently of our code."""
    from ase.calculators.fd import calculate_numerical_forces, calculate_numerical_stress

    atoms = val40.to_atoms(3)
    atoms.calc = calc
    assert np.allclose(atoms.get_forces(), calculate_numerical_forces(atoms, eps=1e-5), atol=1e-6)
    assert np.allclose(atoms.get_stress(), calculate_numerical_stress(atoms, eps=1e-6), atol=1e-7)


def test_rebuilds_the_graph_after_a_move(val40, calc):
    atoms = val40.to_atoms(0)
    atoms.calc = calc
    e0 = atoms.get_potential_energy()
    atoms.positions[0] += 0.05
    assert atoms.get_potential_energy() != pytest.approx(e0, rel=1e-9)
    atoms.set_cell(atoms.cell * 1.02, scale_atoms=True)
    assert np.isfinite(atoms.get_stress()).all()
