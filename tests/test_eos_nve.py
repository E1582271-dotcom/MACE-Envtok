"""Equation of state and NVE drift, with ASE's EMT potential in place of a model."""

import json

import numpy as np
import pytest

pytest.importorskip("mlip_arena")

from ase.build import bulk  # noqa: E402
from ase.calculators.emt import EMT  # noqa: E402
from ase.db import connect  # noqa: E402
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution  # noqa: E402
from mlip_arena.tasks.md import run as md  # noqa: E402

from envtok_mace.bench import eos, nve  # noqa: E402


def test_structures_outside_the_element_table_are_skipped(tmp_path):
    db_path = tmp_path / "set.db"
    with connect(db_path) as db:
        db.write(bulk("Cu"), wbm_id="WBM-2")
        db.write(bulk("Al"), wbm_id="WBM-1")
        db.write(bulk("Ce"), wbm_id="WBM-3")  # a lanthanide
    assert [a.info["id"] for a in eos.structures(str(db_path))] == ["WBM-1", "WBM-2"]


@pytest.fixture(scope="module")
def copper():
    atoms = bulk("Cu", cubic=True)
    atoms.rattle(0.02, seed=0)
    return {"id": "Cu", "formula": "Cu4", "natoms": 4, "seconds": 1.0, **eos.eos_curve(atoms, EMT())}


def test_the_curve_of_a_metal_is_one_smooth_well(copper):
    assert len(copper["volumes_A3"]) == eos.N_VOLUMES
    assert copper["converged"]
    assert copper["K_GPa"] == pytest.approx(134, abs=10)  # EMT's copper
    assert copper["v0_A3"] / 4 == pytest.approx(11.6, abs=0.3)
    row = eos.score(copper, "emt")
    assert row["energy-diff-flip-times"] == 1
    assert row["tortuosity"] == pytest.approx(1.0, abs=1e-6)
    assert row["spearman-compression-energy"] == pytest.approx(-1.0)
    assert row["spearman-tension-energy"] == pytest.approx(1.0)
    assert row["error"] == ""


def test_the_volumes_span_twenty_percent_around_the_relaxed_cell(copper):
    v = np.array(copper["volumes_A3"])
    assert v[-1] / v[len(v) // 2] == pytest.approx(1.2)
    assert v[0] / v[len(v) // 2] == pytest.approx(0.8)


def test_a_broken_curve_is_counted_and_failures_are_kept(copper):
    broken = np.array(copper["energies_eV"])
    broken[-3] -= 1.0  # a dip on the stretched side adds sign changes of the slope
    failed = {"id": "x", "formula": "Cu", "natoms": 1, "seconds": 1.0, "error": "timeout"}
    rows = [
        eos.score({**copper, "id": "a"}, "mace_omat0"),
        eos.score({**copper, "id": "a"}, "ours"),
        eos.score(
            {**copper, "id": "b", "energies_eV": broken.tolist(), "K_GPa": None, "v0_A3": None}, "ours"
        ),
        eos.score(failed, "ours"),
    ]
    s = eos.aggregate(rows)["ours"]
    assert s["n_structures"] == 3
    assert (s["n_failed"], s["n_without_fit"], s["one_well"]) == (1, 2, 1)
    assert s["K_against_mace_omat0"] == {"n": 1, "median_abs_GPa": 0.0, "mean_abs_GPa": 0.0}


def test_tables_are_written(copper, tmp_path):
    for label in ("ours", "mace_omat0"):
        (tmp_path / label).mkdir()
        (tmp_path / label / "Cu.json").write_text(json.dumps(copper))
    eos.write_tables(tmp_path, ["ours", "mace_omat0"], tmp_path / "t")
    summary = json.loads((tmp_path / "t" / "eos_summary.json").read_text())
    assert len(summary["volume_ratios"]) == eos.N_VOLUMES
    assert summary["bands"]["ours"]["p50_meV"][eos.N_VOLUMES // 2] == 0.0
    assert (tmp_path / "t" / "eos_structures.csv").read_text().count("\n") == 3


# ---------------------------------------------------------------------------- NVE
def test_a_supercell_reaches_the_size_and_is_roughly_cubic():
    atoms = bulk("Cu", cubic=True)
    atoms.info["id"] = "x"
    big = nve.supercell(atoms)
    assert len(big) >= nve.MIN_ATOMS and big.info["id"] == "x"
    lengths = big.cell.lengths()
    assert lengths.max() / lengths.min() == pytest.approx(1.0)
    lengths = nve.supercell(atoms.repeat((3, 1, 1))).cell.lengths()
    assert lengths.max() / lengths.min() < 1.5


def test_a_skewed_primitive_cell_still_reaches_the_size():
    assert len(nve.supercell(bulk("Cu"))) >= nve.MIN_ATOMS


def test_drift_recovers_a_known_slope():
    time_ps = np.linspace(0, 10, 501)
    noise = 1e-4 * np.sin(40 * time_ps)
    d = nve.drift(time_ps, -3.0 + 2e-3 * time_ps + noise)
    assert d["drift_meV_per_atom_ps"] == pytest.approx(2.0, abs=0.01)
    assert d["scatter_meV_per_atom"] == pytest.approx(0.1 / np.sqrt(2), abs=0.01)


def test_a_conservative_potential_has_almost_no_drift(tmp_path):
    atoms = bulk("Cu", cubic=True).repeat(2)
    MaxwellBoltzmannDistribution(atoms, temperature_K=600.0, rng=np.random.default_rng(0))
    traj = tmp_path / "run.traj"
    md.fn(
        atoms=atoms,
        calculator=EMT(),
        ensemble="nve",
        dynamics="velocityverlet",
        time_step=2.0,
        total_time=400.0,
        traj_file=traj,
        traj_interval=nve.TRAJ_INTERVAL,
        restart=False,
    )
    s = nve.score_trajectory(traj, n_steps=200, dt_fs=2.0)
    assert s["n_frames"] == 200 // nve.TRAJ_INTERVAL + 1
    assert abs(s["drift_meV_per_atom_ps"]) < 1.0
    assert 150 < s["T_mean_K"] < 450  # about half of the initial 600 K


def test_failed_runs_are_counted_and_left_out():
    common = dict(model="m", dtype="float32", temperature_K=300.0)
    rows = [
        {**common, "drift_meV_per_atom_ps": -2.0, "T_mean_K": 300.0},
        {**common, "drift_meV_per_atom_ps": 4.0, "T_mean_K": 310.0},
        {**common, "error": "crash"},
    ]
    (s,) = nve.aggregate(rows)
    assert (s["n_runs"], s["n_failed"]) == (3, 1)
    assert s["median_abs_drift_meV_per_atom_ps"] == pytest.approx(3.0)
    assert s["max_abs_drift_meV_per_atom_ps"] == pytest.approx(4.0)
