"""WBM relaxations: the set is drawn reproducibly from structures the model covers, a
relaxation follows Matbench Discovery's settings, and the RMSD is scored as upstream."""

import io
import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import write

from envtok_mace.bench import wbm

pytest.importorskip("pymatgen")


def _zip(path, frames):
    with zipfile.ZipFile(path, "w") as z:
        for atoms in frames:
            buf = io.StringIO()
            write(buf, atoms, format="extxyz")
            z.writestr(f"{atoms.info['material_id']}.extxyz", buf.getvalue())


STRAIN = 1.01


def _set(tmp_path, n_ok=12):
    """Rock-salt cells (covered) and a cerium cell (Z = 58, not covered). The 'relaxed'
    copies are the same cells strained, with the atoms listed in reverse, as the DFT
    copies often list them in another order."""
    initial, relaxed = [], []
    for i in range(n_ok + 1):
        atoms = (
            bulk("Ce", "fcc", a=5.2, cubic=True) if i == n_ok else bulk("NaCl", "rocksalt", a=5.6, cubic=True)
        )
        atoms.info["material_id"] = f"wbm-1-{i}"
        initial.append(atoms)
        done = atoms[::-1]
        done.set_cell(atoms.cell * STRAIN, scale_atoms=True)
        done.info["material_id"] = atoms.info["material_id"]
        relaxed.append(done)
    _zip(tmp_path / "initial.zip", initial)
    _zip(tmp_path / "relaxed.zip", relaxed)
    return tmp_path / "initial.zip", tmp_path / "relaxed.zip"


def _record(atoms):
    """A relaxation that ends on the DFT geometry, in the initial atom order."""
    return {
        "id": atoms.info["material_id"],
        "natoms": len(atoms),
        "converged": True,
        "steps": 1,
        "numbers": atoms.numbers.tolist(),
        "cell": (np.asarray(atoms.cell) * STRAIN).tolist(),
        "positions": (atoms.positions * STRAIN).tolist(),
    }


def test_the_set_is_drawn_from_covered_structures_and_is_reproducible(tmp_path):
    zi, zr = _set(tmp_path)
    wbm.prepare(zi, zr, tmp_path / "a", n=5)
    wbm.prepare(zi, zr, tmp_path / "b", n=5)
    a = wbm.read_structures(tmp_path / "a")
    b = wbm.read_structures(tmp_path / "b")
    ref = wbm.read_structures(tmp_path / "a", "relaxed")
    assert [x.info["material_id"] for x in a] == [x.info["material_id"] for x in b]
    assert [x.info["material_id"] for x in a] == [x.info["material_id"] for x in ref]
    assert all(set(x.get_chemical_symbols()) == {"Na", "Cl"} for x in a)
    meta = json.loads((tmp_path / "a/subset.json").read_text())
    assert (meta["n_wbm"], meta["n_candidates"], meta["n_structures"]) == (13, 12, 5)


def test_a_relaxation_converges_and_keeps_the_atom_order():
    atoms = bulk("Cu", "fcc", a=3.8, cubic=True)
    atoms.rattle(0.05, seed=1)
    rec = wbm.relax(atoms, EMT())
    assert rec["converged"] and 0 < rec["steps"] <= wbm.MAX_STEPS
    assert np.array(rec["positions"]).shape == (len(atoms), 3)
    assert np.linalg.det(rec["cell"]) < atoms.get_volume()  # EMT copper is smaller than a = 3.8
    json.dumps(rec)


def test_rmsd_is_zero_for_the_same_structure_and_none_without_a_match():
    from pymatgen.io.ase import AseAtomsAdaptor

    a = AseAtomsAdaptor.get_structure(bulk("NaCl", "rocksalt", a=5.6))
    assert wbm.rmsd(a, a)[0] == pytest.approx(0, abs=1e-8)
    moved = a.copy()
    moved.translate_sites([0], [0.1, 0, 0], frac_coords=False)  # 0.1 A
    assert 0 < wbm.rmsd(moved, a)[0] < 0.1
    assert wbm.rmsd(AseAtomsAdaptor.get_structure(bulk("Cu", "bcc", a=2.9, cubic=True)), a) == (None, None)


def test_scoring_counts_failed_and_unmatched_structures_as_stol(tmp_path):
    zi, zr = _set(tmp_path)
    wbm.prepare(zi, zr, tmp_path / "set", n=4)
    refs = wbm.read_structures(tmp_path / "set", "relaxed")
    ids = [a.info["material_id"] for a in refs]
    records = {
        ids[0]: _record(wbm.read_structures(tmp_path / "set")[0]),
        ids[1]: {"id": ids[1], "natoms": 8, "error": "crash"},
    }  # ids[2:] have no record
    rows = wbm.score(records, refs, "m")
    (s,) = wbm.aggregate(rows).values()
    assert rows[0]["rmsd"] == pytest.approx(0, abs=1e-8)
    assert s["n_failed"] == 3 and s["failures"] == {"crash": 1, "missing": 2}
    assert s["rmsd_mean"] == pytest.approx(3 * wbm.STOL / 4)
    assert s["rmsd_median_matched"] == pytest.approx(0, abs=1e-8)


def test_a_rerun_skips_finished_structures_and_keeps_the_last_record(tmp_path):
    d = tmp_path / "ours"
    d.mkdir()
    (d / "shard_000of002.jsonl").write_text(
        json.dumps({"id": "a", "error": "oom"}) + "\n" + json.dumps({"id": "b", "converged": True}) + "\n"
    )
    (d / "shard_001of003.jsonl").write_text(
        json.dumps({"id": "a", "converged": True}) + '\n{"id": "c", "conv'
    )
    records = wbm.read_records(d)
    assert records["a"] == {"id": "a", "converged": True} and set(records) == {"a", "b"}


def test_the_summary_compares_with_published_rmsds(tmp_path):
    zi, zr = _set(tmp_path)
    wbm.prepare(zi, zr, tmp_path / "set", n=3)
    refs = wbm.read_structures(tmp_path / "set", "relaxed")
    ids = [a.info["material_id"] for a in refs]
    (tmp_path / "runs/m").mkdir(parents=True)
    with (tmp_path / "runs/m/shard_000of001.jsonl").open("w") as fh:
        for a in wbm.read_structures(tmp_path / "set"):
            fh.write(json.dumps(_record(a)) + "\n")
    (tmp_path / "pub.csv").write_text(
        "material_id,structure_rmsd_vs_dft\n" + f"{ids[0]},0.01\n{ids[1]},\n{ids[2]},0.03\nwbm-9-9,0.5\n"
    )
    args = SimpleNamespace(
        structures=str(tmp_path / "set"),
        limit=None,
        labels=["m"],
        out=str(tmp_path / "runs"),
        workers=1,
        published=[f"m={tmp_path / 'pub.csv'}"],
        tables=str(tmp_path / "tables"),
    )
    wbm.write_tables(args)
    s = json.loads((tmp_path / "tables/wbm_summary.json").read_text())
    assert s["models"]["m"]["rmsd_mean"] == pytest.approx(0, abs=1e-8)
    table = (tmp_path / "tables/wbm_structures.csv").read_text().splitlines()
    assert table[0] == "id,formula,natoms,m_rmsd,m_steps,m_error" and len(table) == 1 + len(ids)
    assert table[1].startswith(f"{ids[0]},Cl4Na4,8,")
    p = s["published"]["m"]
    assert p["rmsd_mean"] == pytest.approx((0.01 + wbm.STOL + 0.03) / 3)
    assert (p["n_unmatched"], p["n_both_matched"]) == (1, 2)
    assert p["median_abs_difference"] == pytest.approx(0.02)
