"""Li diffusion: the MSD analysis on random walks of known D, the Arrhenius fit, and the
MD driver with ASE's EMT potential in place of a model."""

import json

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from envtok_mace.bench import diffusion


def _walk(n_frames=1001, n_mobile=100, n_framework=50, frame_ps=0.1, d_a2_ps=0.1, drift=0.02, seed=0):
    """Mobile atoms on a random walk, framework atoms vibrating on their sites, all sharing
    one drift, wrapped into a cubic box as an MD code may leave them."""
    rng = np.random.default_rng(seed)
    box = 20.0
    start = rng.uniform(0, box, size=(n_mobile + n_framework, 3))
    steps = rng.normal(0, np.sqrt(2 * d_a2_ps * frame_ps), size=(n_frames - 1, n_mobile, 3))
    walk = np.concatenate([np.zeros((1, n_mobile, 3)), np.cumsum(steps, axis=0)])
    shake = rng.normal(0, 0.1, size=(n_frames, n_framework, 3))
    pos = start + np.concatenate([walk, shake], axis=1)
    pos[..., 0] += drift * np.arange(n_frames)[:, None]
    numbers = np.array([3] * n_mobile + [16] * n_framework)
    return pos % box, numbers, np.eye(3) * box, frame_ps


def test_unwrapping_follows_an_atom_across_the_boundary():
    cell = np.eye(3) * 10.0
    x = np.linspace(8.0, 13.0, 11)
    pos = np.stack([x % 10, np.zeros_like(x), np.zeros_like(x)], axis=1)[:, None, :]
    assert np.allclose(diffusion.unwrap(pos, cell)[:, 0, 0], x)


def test_a_random_walk_gives_back_its_diffusion_coefficient():
    pos, numbers, cell, frame_ps = _walk()
    s = diffusion.score_run(pos, numbers, cell, frame_ps)
    assert s["D_cm2_s"] == pytest.approx(0.1 / 1e4, rel=0.1)  # 0.1 A^2/ps
    assert s["D_err_cm2_s"] < 0.2 * s["D_cm2_s"]
    assert s["msd_end_A2"] == pytest.approx(6 * 0.1 * 50, rel=0.15)
    assert s["framework_msd_end_A2"] == pytest.approx(2 * 3 * 0.1**2, rel=0.2)  # the drift is gone
    assert s["diffusive"] and not s["framework_flows"]
    json.dumps(s)


def test_atoms_that_only_vibrate_are_not_diffusive():
    pos, numbers, cell, frame_ps = _walk(d_a2_ps=1e-4)
    s = diffusion.score_run(pos, numbers, cell, frame_ps)
    assert not s["diffusive"]


def test_the_arrhenius_fit_returns_the_activation_energy():
    t = np.array([600.0, 700.0, 800.0, 900.0])
    d = 1e-3 * np.exp(-0.25 / (diffusion.KB_EV * t))
    fit = diffusion.arrhenius(t, d)
    assert fit["Ea_eV"] == pytest.approx(0.25, rel=1e-9)
    assert fit["Ea_err_eV"] < 1e-9
    assert fit["D300_cm2_s"] == pytest.approx(1e-3 * np.exp(-0.25 / (diffusion.KB_EV * 300)), rel=1e-9)


def test_nernst_einstein_conductivity():
    # 160 Li in 4000 A^3 with D = 1e-6 cm^2/s at 300 K
    n = 160 / 4000e-30
    expected = n * (1.602176634e-19) ** 2 * 1e-10 / (1.380649e-23 * 300) * 10
    assert diffusion.conductivity_mS_cm(160, 4000.0, 1e-6, 300.0) == pytest.approx(expected)
    assert expected == pytest.approx(247.8, rel=1e-3)


def test_only_usable_runs_enter_the_fit():
    base = {"model": "m", "id": "x", "n_mobile": 10, "volume_A3": 1000.0, "framework_flows": False}
    rows = [
        {
            **base,
            "temperature_K": t,
            "D_cm2_s": 1e-3 * np.exp(-0.3 / (diffusion.KB_EV * t)),
            "diffusive": True,
        }
        for t in (600.0, 700.0, 800.0)
    ]
    rows.append({**base, "temperature_K": 900.0, "D_cm2_s": 1.0, "diffusive": True, "framework_flows": True})
    (summary,) = diffusion.aggregate(rows)
    assert summary["Ea_eV"] == pytest.approx(0.3)
    assert summary["temperatures_fitted_K"] == [600.0, 700.0, 800.0]
    assert summary["n_framework_flows"] == 1
    failed = diffusion.aggregate(rows[:2] + [{**base, "temperature_K": 800.0, "error": "diverged"}])[0]
    assert failed["Ea_eV"] is None and failed["n_failed"] == 1


def test_structures_are_repeated_to_at_least_200_atoms(tmp_path):
    from ase.io import write

    cells = []
    for name, symbol in (("a", "Cu"), ("b", "Al")):
        atoms = bulk(symbol, cubic=True)
        atoms.info.update(name=name, material_id=f"mp-{name}")
        cells.append(atoms)
    write(tmp_path / "s.extxyz", cells, format="extxyz")
    (b,) = diffusion.read_structures(str(tmp_path / "s.extxyz"), ["b"])
    assert len(b) >= diffusion.MIN_ATOMS and b.info["name"] == "b" and b.info["material_id"] == "mp-b"


def test_the_md_keeps_its_temperature_and_saves_every_frame():
    atoms = bulk("Cu", cubic=True).repeat(3)
    frames = diffusion.molecular_dynamics(atoms, EMT(), 300.0, equilibrate_ps=0.2, production_ps=0.5)
    n = round(0.5e3 / diffusion.DT_FS / diffusion.SAVE_EVERY) + 1
    assert frames["positions"].shape == (n, len(atoms), 3)
    assert frames["positions"].dtype == np.float32
    assert 150 < frames["temperature_K"].mean() < 450


def test_a_run_that_overheats_or_collapses_is_stopped():
    atoms = bulk("Cu", cubic=True).repeat(2)
    atoms.calc = EMT()
    atoms.set_velocities(np.full((len(atoms), 3), 0.5))  # far above any target
    with pytest.raises(diffusion.Diverged, match="temperature"):
        diffusion._check(atoms, 300.0)
    atoms.set_velocities(np.zeros((len(atoms), 3)))
    atoms.positions[1] = atoms.positions[0] + [0.3, 0.0, 0.0]
    with pytest.raises(diffusion.Diverged, match="closer"):
        diffusion._check(atoms, 300.0)
