"""Scoring and merging of MatCalc-Bench results (no relaxations run here)."""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from envtok_mace.bench import matcalc as mc  # noqa: E402


def test_values_follow_the_papers_definitions():
    df = pd.DataFrame({"K_vrh_DFT": [100.0, 50.0, 80.0, np.nan], "K_vrh_m": [110.0, -5.0, np.nan, 10.0]})
    v = mc.values(df, "kvrh", unphysical="drop")
    # row 1 is unphysical (negative modulus), row 2 failed, row 3 has no reference
    assert v["values"] == pytest.approx([10.0])
    assert (v["n_failed"], v["n_unphysical"]) == (1, 1)
    kept = mc.values(df, "kvrh", unphysical="keep")
    assert kept["values"] == pytest.approx([10.0, -110.0])

    eform = pd.DataFrame({"Eform_DFT": [-1.0, 0.5], "Eform_m": [-0.9, 0.3]})
    assert mc.values(eform, "eform")["values"] == pytest.approx([0.1, 0.2])
    soft = pd.DataFrame({"softening_scale_m": [0.8, 1.0]})
    assert mc.values(soft, "softening")["values"] == pytest.approx([20.0, 0.0])


def test_summary_pools_random_seeds(tmp_path):
    for label, err in (("rdm1", 0.1), ("rdm2", 0.3), ("ours", 0.2)):
        d = tmp_path / label
        d.mkdir()
        pd.DataFrame(
            {
                "material_id": ["a", "b"],
                "formula": ["SiO2", "MgO"],
                "Eform_DFT": [0.0, 0.0],
                f"Eform_{label}": [err, -err],
            }
        ).to_csv(d / f"Equilibrium_{label}_FULL.csv", index=False)
    res = mc.summary([str(tmp_path / L) for L in ("rdm1", "rdm2", "ours")], ["rdm1", "rdm2", "ours"])
    models = res["quantities"]["eform"]["models"]
    assert models["random"]["n"] == 4
    assert models["random"]["mean"] == pytest.approx(0.2)
    assert models["ours"]["median"] == pytest.approx(0.2)


def test_merge_refuses_incomplete_shards_and_cuts_elements(tmp_path):
    pytest.importorskip("pymatgen")
    for i, (mid, formula) in enumerate((("a", "SiO2"), ("b", "CeO2"))):
        pd.DataFrame({"material_id": [mid], "formula": [formula], "softening_scale_m": [0.9]}).to_csv(
            tmp_path / f"Softening_m_shard{i}of2.csv", index=False
        )
    pd.DataFrame({"mp_id": ["x"], "formula": ["MgO"]}).to_csv(
        tmp_path / "Phonon_m_shard0of3.csv", index=False
    )  # 1 of 3 shards present
    mc.merge(tmp_path, "m")
    merged = pd.read_csv(tmp_path / "Softening_m_FULL.csv")
    assert list(merged["material_id"]) == ["a"]  # Ce is outside the model's table
    assert not (tmp_path / "Phonon_m_FULL.csv").exists()
