"""Phonons and thermal conductivity, with ASE's EMT potential in place of a model."""

import json

import numpy as np
import pytest

pytest.importorskip("phono3py")

from ase.build import bulk  # noqa: E402
from ase.calculators.emt import EMT  # noqa: E402

from envtok_mace.bench import kappa  # noqa: E402

R_J_PER_K_MOL = kappa.KB_J_PER_K * kappa.AVOGADRO


def _copper():
    atoms = bulk("Cu", cubic=True)
    atoms.info.update(
        material_id="cu",
        name="Cu",
        fc3_supercell=2 * np.eye(3, dtype=int),
        fc2_supercell=2 * np.eye(3, dtype=int),
        q_point_mesh=np.array([4, 4, 4]),
    )
    return atoms


def test_heat_capacity_and_entropy_of_an_einstein_solid():
    high = kappa.harmonic(np.full((2, 3), 0.01 + 1e-6), t=3000.0)
    assert high["cv_J_per_K_mol"] == pytest.approx(3 * R_J_PER_K_MOL, rel=1e-4)
    # at a low temperature the energy is frozen in, leaving the zero-point energy per atom
    frozen = kappa.harmonic(np.full((2, 3), 10.0), t=5.0)
    assert frozen["cv_J_per_K_mol"] < 1e-6 and frozen["s_J_per_K_mol"] < 1e-6
    zero_point = 3 * 0.5 * kappa.H_J_S * 10e12 * kappa.AVOGADRO / 1e3
    assert frozen["f_kJ_per_mol"] == pytest.approx(zero_point, rel=1e-6)


def test_weights_and_atoms_per_cell_do_not_change_the_per_atom_values():
    f = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    weighted = kappa.harmonic(f, weights=[1, 3])
    repeated = kappa.harmonic(np.vstack([f[0], f[1], f[1], f[1]]))
    assert weighted == pytest.approx(repeated)
    two_atoms = kappa.harmonic(np.hstack([f, f]))
    assert two_atoms["cv_J_per_K_mol"] == pytest.approx(repeated["cv_J_per_K_mol"], rel=0.5)
    assert kappa.harmonic(f)["omega_max_THz"] == 6.0


def test_imaginary_modes_allow_only_noise_in_the_acoustic_modes_at_gamma():
    stable = np.array([[-0.005, 0.0, 0.0, 5.0], [1.0, 1.0, 2.0, 5.0]])
    assert not kappa.has_imaginary_modes(stable)
    assert kappa.has_imaginary_modes(np.array([[-0.5, 0.0, 0.0, 5.0], [1.0, 1.0, 2.0, 5.0]]))
    assert kappa.has_imaginary_modes(np.array([[0.0, 0.0, 0.0, -1.0], [1.0, 1.0, 2.0, 5.0]]))
    assert kappa.has_imaginary_modes(np.array([[0.0, 0.0, 0.0, 5.0], [-0.1, 1.0, 2.0, 5.0]]))


def test_the_score_of_a_perfect_a_wrong_and_a_missing_prediction():
    mode = np.array([[1.0, 2.0], [3.0, 4.0]])
    w = [1, 1]
    ref_kappa = float(mode.sum() / 2)
    perfect = kappa.score(ref_kappa, mode, w, False, ref_kappa, mode)
    assert perfect == {"srd": 0.0, "sre": 0.0, "srme": 0.0, "censored": False}
    double = kappa.score(2 * ref_kappa, 2 * mode, w, False, ref_kappa, mode)
    assert double["srd"] == pytest.approx(2 / 3)
    assert double["srme"] == pytest.approx(2 / 3)
    missing = kappa.score(None, None, None, False, ref_kappa, mode)
    assert missing == {"srd": -2.0, "sre": 2.0, "srme": 2.0, "censored": True}
    imaginary = kappa.score(ref_kappa, mode, w, True, ref_kappa, mode)
    assert imaginary["srme"] == 2.0 and imaginary["censored"] and imaginary["srd"] == 0.0


@pytest.fixture(scope="module")
def copper_run():
    atoms = _copper()
    rec, arrays = kappa.forces_for_structure(atoms, EMT())
    return atoms, rec, arrays


def test_the_force_stage_on_copper(copper_run):
    _, rec, arrays = copper_run
    assert rec["init_spg"] == rec["final_spg"] == 225 and not rec["broken_symmetry"]
    assert not rec["imaginary"] and not rec["conductivity_skipped"]
    assert 5.0 < rec["omega_max_THz"] < 12.0
    assert rec["cv_J_per_K_mol"] == pytest.approx(3 * R_J_PER_K_MOL, rel=0.1)
    assert arrays["fc3_forces"].shape == (rec["fc3_supercells"], 32, 3)
    assert arrays["fc2_forces"].shape == (rec["fc2_supercells"], 32, 3)
    json.dumps(rec)  # the record goes to a JSON file


def test_the_conductivity_stage_on_copper(copper_run):
    atoms, _, arrays = copper_run
    result, modes = kappa.conductivity(atoms.info, arrays)
    assert result["kappa_W_per_mK"] > 0
    total = result["kappa_p_W_per_mK"] + result["kappa_c_W_per_mK"]
    assert total == pytest.approx(result["kappa_W_per_mK"], rel=1e-6)
    assert modes["mode_kappa"].sum() / modes["weights"].sum() == pytest.approx(
        result["kappa_W_per_mK"], rel=1e-6
    )
    assert modes["weights"].sum() == 4**3


def test_a_failed_structure_is_kept_and_scored_as_the_worst(tmp_path):
    reference = {
        "a": {
            "formula": "A",
            "kappa": 2.0,
            "mode_kappa": np.ones((1, 3)),
            **{f"dft_{k}": 1.0 for k in kappa.HARMONIC},
        }
    }
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "a.json").write_text(json.dumps({"id": "a", "error": "crash"}))
    rows = kappa.tables_for(tmp_path, "m", reference)
    summary = kappa.aggregate(rows)["m"]
    assert summary["kappa_srme"] == 2.0 and summary["n_failed"] == 1 and summary["n_stable"] == 0
    assert summary["omega_max_THz_mae"] is None
