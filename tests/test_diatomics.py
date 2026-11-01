"""Scoring of diatomic curves, on synthetic Morse curves written like the arena's task."""

import json

import numpy as np
import pytest

pytest.importorskip("mlip_arena")

from ase import Atoms  # noqa: E402
from ase.calculators.singlepoint import SinglePointCalculator  # noqa: E402
from ase.io import write  # noqa: E402

from envtok_mace.bench import diatomics  # noqa: E402

DEPTH, R0, A, E_ATOM = 4.5, 1.2, 1.8, -3.0


def _morse(r):
    x = np.exp(-A * (r - R0))
    e = DEPTH * (x**2 - 2 * x)
    f = 2 * A * DEPTH * (x**2 - x)  # -dE/dr, along the bond: positive = repulsive
    return e, f


def _write_curve(d, symbol, rs, e_atom=E_ATOM):
    d.mkdir(parents=True, exist_ok=True)
    for r in rs:
        e, f = _morse(r)
        atoms = Atoms(symbol * 2, positions=[[0, 0, 0], [r, 0, 0]], cell=[30, 30, 30], pbc=False)
        forces = np.array([[-f, 0, 0], [f, 0, 0]])
        atoms.calc = SinglePointCalculator(atoms, energy=e + 2 * e_atom, forces=forces)
        write(d / f"{symbol}{symbol}.extxyz", atoms, append=True)
    (d / f"{symbol}.json").write_text(json.dumps({"symbol": symbol, "e_atom": e_atom}))


def test_a_morse_curve_scores_as_one_smooth_well(tmp_path):
    _write_curve(tmp_path / "m", "O", np.arange(0.6, 6.0, 0.01))
    rows, curves = diatomics.summarize(tmp_path, "m")
    (row,) = rows
    assert row["well_depth_eV"] == pytest.approx(-DEPTH, abs=1e-3)
    assert row["r_at_min_A"] == pytest.approx(R0, abs=0.01)
    assert row["e_close_eV"] > 0
    assert row["e_far_eV"] == pytest.approx(_morse(row["r_hi_A"])[0], abs=1e-6)  # against 2 atoms
    assert row["energy-diff-flip-times"] == 1
    assert row["conservation-deviation"] < 1e-2
    assert "O" in curves and len(curves["O"]["r"]) == len(range(0, row["n_points"], diatomics.CURVE_STRIDE))
    s = diatomics.aggregate(rows)["m"]
    assert s["n_elements"] == 1 and s["single_well"] == 1 and s["repulsive_at_close_range"] == 1
    assert s["too_deep"] == []


def test_a_collapsing_curve_is_counted_as_not_repulsive(tmp_path):
    d = tmp_path / "bad"
    _write_curve(d, "H", np.arange(0.6, 4.0, 0.01))
    # overwrite with a curve that falls without bound at short range
    (d / "HH.extxyz").unlink()
    for r in np.arange(0.3, 4.0, 0.01):
        atoms = Atoms("H2", positions=[[0, 0, 0], [r, 0, 0]], cell=[30, 30, 30], pbc=False)
        e = -50.0 / r
        atoms.calc = SinglePointCalculator(atoms, energy=e + 2 * E_ATOM, forces=np.zeros((2, 3)))
        write(d / "HH.extxyz", atoms, append=True)
    s = diatomics.aggregate(diatomics.summarize(tmp_path, "bad")[0])["bad"]
    assert s["repulsive_at_close_range"] == 0
    assert s["too_deep"] == ["H"]


def test_unfinished_curves_are_left_out(tmp_path):
    _write_curve(tmp_path / "m", "O", np.arange(0.6, 3.0, 0.01))
    (tmp_path / "m" / "O.json").unlink()  # the curve was interrupted before the atom energy
    assert diatomics.summarize(tmp_path, "m")[0] == []


def test_every_element_of_the_model_is_run():
    from ase.data import atomic_numbers

    syms = diatomics.symbols()
    assert len(syms) == 68 and all(atomic_numbers[s] < 57 or 72 <= atomic_numbers[s] <= 83 for s in syms)
    assert set(diatomics.CURVE_ELEMENTS) <= set(syms)
