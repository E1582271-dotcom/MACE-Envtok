"""A model written as a mace-torch file gives the same results through mace-torch's own tools."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from envtok_mace.data.batch import collate
from envtok_mace.eval.ase_calc import EnvTokCalculator
from envtok_mace.eval.run_eval import load_model
from envtok_mace.model.mace import MACE, MACEConfig

mace_compat = pytest.importorskip("envtok_mace.model.mace_compat")
pytest.importorskip("mace")
export_mace = pytest.importorskip("envtok_mace.eval.export_mace").export_mace

KEYS = ("energy", "forces", "stress")
RELEASED = Path(__file__).resolve().parents[1] / "local" / "release" / "v1.0.0" / "mace-envtok-1.0.0.pt"


def _compare(ours, ref, batch, dtype):
    out = ours(batch.to(dtype=dtype), compute_force=True, compute_stress=True)
    exp = ref(mace_compat.mace_input_dict(batch, ref), compute_force=True, compute_stress=True)
    return {k: (out[k].detach() - exp[k].detach()).abs().max().item() for k in KEYS}


@pytest.fixture
def weights(tmp_path):
    """A randomly initialised model as a weights file."""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    model = MACE(MACEConfig())
    path = tmp_path / "weights.pt"
    torch.save({"model_config": model.config.to_dict(), "model": model.state_dict()}, path)
    return path


def test_to_mace_gives_the_same_results(val40, weights):
    model = load_model(weights, torch.device("cpu"), torch.float64, "e3nn")
    ref = mace_compat.to_mace(model)
    assert sum(p.numel() for p in ref.parameters()) == sum(p.numel() for p in model.parameters())
    diff = _compare(model, ref, collate(val40, np.arange(8)), torch.float64)
    assert max(diff.values()) < 1e-12, diff


def test_to_mace_needs_the_e3nn_backend():
    cueq = SimpleNamespace(config=SimpleNamespace(backend="cueq"))
    with pytest.raises(ValueError, match="e3nn"):
        mace_compat.to_mace(cueq)


def test_the_file_works_with_mace_calculator(val40, weights, tmp_path):
    from mace.calculators import MACECalculator

    dst = tmp_path / "exported.model"
    report = export_mace(weights, dst)
    assert report["bytes"] == dst.stat().st_size
    theirs = MACECalculator(model_paths=str(dst), device="cpu", default_dtype="float64")
    ours = EnvTokCalculator(weights, device="cpu", dtype="float64", backend="e3nn")
    for i in range(4):
        atoms = val40.to_atoms(i)
        atoms.calc = ours
        energy, forces, stress = atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()
        atoms.calc = theirs
        assert atoms.get_potential_energy() == pytest.approx(energy, abs=1e-12)
        assert np.abs(atoms.get_forces() - forces).max() < 1e-12
        assert np.abs(atoms.get_stress() - stress).max() < 1e-12


def test_the_file_compiles_for_lammps(val40, weights, tmp_path, monkeypatch):
    """mace-torch's TorchScript for LAMMPS, run without LAMMPS on every atom as a local one."""
    from mace.cli import create_lammps_model

    dst = tmp_path / "exported.model"
    export_mace(weights, dst)
    monkeypatch.setattr(sys, "argv", ["create_lammps_model", str(dst)])
    create_lammps_model.main()
    compiled = torch.jit.load(f"{dst}-lammps.pt", map_location="cpu")

    model = load_model(weights, torch.device("cpu"), torch.float64, "e3nn")
    ref = mace_compat.to_mace(model)
    batch = collate(val40, np.arange(1))
    out = model(batch.to(dtype=torch.float64), compute_force=True, compute_stress=False)
    data = mace_compat.mace_input_dict(batch, ref)
    natoms = data["positions"].shape[0]
    result = compiled(data, torch.ones(natoms, dtype=torch.float64), compute_virials=False)
    assert result["total_energy_local"].item() == pytest.approx(out["energy"].item(), abs=1e-10)
    assert (result["forces"].detach() - out["forces"].detach()).abs().max().item() < 1e-10


@pytest.mark.skipif(not RELEASED.exists(), reason="the released weights are not in local/")
def test_released_weights_differ_only_by_the_rounding_of_zbl_exponents(val40):
    batch = collate(val40, np.arange(len(val40)))
    model = load_model(RELEASED, torch.device("cpu"), torch.float64, "e3nn")
    ref = mace_compat.to_mace(model)
    as_released = _compare(model, ref, batch, torch.float64)
    assert as_released["forces"] < 1e-6 and as_released["energy"] < 1e-6, as_released

    exact = MACE(MACEConfig.from_dict(model.config.to_dict())).to(torch.float64).zbl.b
    model.zbl.b.copy_(exact)
    aligned = _compare(model, ref, batch, torch.float64)
    assert max(aligned.values()) < 1e-12, aligned


@pytest.mark.skipif(not RELEASED.exists(), reason="the released weights are not in local/")
def test_released_weights_agree_in_float32(val40):
    """The precision of training and of the OMat24 and MatPES errors: only summation order differs."""
    batch = collate(val40, np.arange(len(val40)))
    model = load_model(RELEASED, torch.device("cpu"), torch.float32, "e3nn")
    ref = mace_compat.to_mace(model)
    diff = _compare(model, ref, batch, torch.float32)
    assert diff["forces"] < 2e-4 and diff["stress"] < 1e-6, diff
    assert diff["energy"] / batch.natoms.min().item() < 1e-5, diff
