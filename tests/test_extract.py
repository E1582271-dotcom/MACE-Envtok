import zlib

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from envtok_mace.data.extract_subset import extract
from envtok_mace.data.frames import Frames

aselmdb = pytest.importorskip("ase_db_backends.aselmdb")


def _write_db(path, structures):
    with aselmdb.LMDBDatabase(str(path), readonly=False) as db:
        for sid, numbers in structures:
            n = len(numbers)
            rng = np.random.default_rng(zlib.crc32(sid.encode()))
            atoms = Atoms(numbers=numbers, positions=rng.random((n, 3)) * 4, cell=np.eye(3) * 6, pbc=True)
            atoms.calc = SinglePointCalculator(
                atoms, energy=-1.0 * n, forces=rng.normal(size=(n, 3)), stress=rng.normal(size=6)
            )
            db.write(atoms, data={"sid": sid})


@pytest.fixture
def omat_dir(tmp_path):
    root = tmp_path / "omat"
    (root / "rattled-300").mkdir(parents=True)
    (root / "rattled-relax").mkdir(parents=True)
    _write_db(root / "rattled-300" / "data.aselmdb", [
        ("s3", [8, 14]), ("s1", [1, 1, 8]), ("re", [57, 8]),  # La: rare earth, dropped
    ])
    _write_db(root / "rattled-relax" / "data.aselmdb", [("s2", [26, 26]), ("s4", [29])])
    return root


def test_keep_all_drops_rare_earths_and_sorts(omat_dir, tmp_path):
    out = tmp_path / "all.h5"
    frames = extract(omat_dir, out, workers=1, chunk_rows=2)
    assert frames.sid.tolist() == ["s1", "s2", "s3", "s4"]
    assert frames.attrs["dropped_rare_earth"] == 1
    back = Frames.load(out)
    assert back.source.tolist() == ["rattled-300", "rattled-relax", "rattled-300", "rattled-relax"]
    np.testing.assert_array_equal(back.natoms, [3, 2, 2, 1])


def test_id_filter_and_expect(omat_dir, tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("s4\ns1\n")
    frames = extract(omat_dir, tmp_path / "sub.h5", ids_path=ids, expect=2, workers=2, chunk_rows=1)
    assert frames.sid.tolist() == ["s1", "s4"]


def test_missing_id_is_an_error(omat_dir, tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("s1\nnot-there\n")
    with pytest.raises(RuntimeError, match="not found"):
        extract(omat_dir, tmp_path / "x.h5", ids_path=ids)
