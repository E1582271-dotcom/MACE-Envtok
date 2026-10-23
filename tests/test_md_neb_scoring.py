"""Outcome classification of the MD-stability runs and scoring of the NEB barriers."""

import json

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from envtok_mace.bench import nebdft2k, stability  # noqa: E402


def _frames(source, nfs, temps, natoms=100, sps=0.1):
    return pd.DataFrame({"source": source, "natoms": natoms, "normalized_final_step": nfs,
                         "temperatures": temps, "seconds_per_step": sps, "target_steps": 5000})


def test_stability_outcomes_follow_the_papers_precedence(tmp_path):
    rows = [
        _frames("m_random_A_0_npt", 0.9996, [300.0, 3000.0]),        # finished
        _frames("m_random_B_1_npt", 0.4, [300.0, 900.0]),            # singular cell
        _frames("m_random_C_2_npt", 0.3, [300.0, 2e5]),              # temperature blow-up
        _frames("m_random_D_3_npt", 0.5, [300.0, 1200.0]),           # OOM
        _frames("m_random_E_4_npt", 0.6, [300.0, 1500.0]),           # out of wall time
        _frames("m_random_F_5_npt", 0.9996, [300.0, np.nan]),        # finished despite a NaN
    ]
    pd.concat(rows, ignore_index=True).to_parquet(tmp_path / "m-compression.parquet")
    (tmp_path / ".failed_m_random_B_1_npt.txt").write_text("LinAlgError: Singular matrix")
    (tmp_path / ".failed_m_random_D_3_npt.txt").write_text("torch.OutOfMemoryError: CUDA out of memory")
    (tmp_path / ".oom_m_random_D_3_npt.txt").write_text("cuda-oom\n")

    per = stability.classify(tmp_path, n_npt=7)
    assert list(per["outcome"]) == ["survived", "diverged", "diverged", "oom", "truncated",
                                    "survived", "missing"]
    r = stability.rates(per)
    assert (r["n_survived"], r["n_diverged"], r["n_missing"]) == (2, 2, 1)
    expected = (0.9996 + 0.4 + 0.3 + 0.5 + 0.6 + 0.9996 + 0.0) / 7
    assert r["mean_completion_pct"] == pytest.approx(100 * expected)
    assert r["first_div_ps"] == pytest.approx(3.0)  # the earliest divergence, at 0.3 of 10 ps


def test_speed_matching_truncates_where_the_slower_setup_ran_out_of_time():
    per = pd.DataFrame({"idx": [0, 1, 2], "nfs": [0.9996, 0.9, 0.2],
                        "outcome": ["survived", "diverged", "diverged"]})
    # the reference needed 0.45 s per step: 1800 s cover 4000 of 5000 steps (0.8)
    ref = pd.DataFrame({"idx": [0, 1, 2], "seconds_per_step": [0.45, 0.45, 0.1],
                        "target_steps": [5000, 5000, 5000]})
    out = stability.speed_matched(per, ref)
    assert list(out["outcome"]) == ["truncated", "truncated", "diverged"]
    assert list(out["nfs"]) == pytest.approx([0.8, 0.8, 0.2])


def test_neb_scores_only_converged_supported_hops(tmp_path):
    hops = [
        dict(edge_id="a", chemsys="Li-O-Si", em_pred=0.50, em_dft=0.40, converged=True,
             has_specific_TM=False),
        dict(edge_id="b", chemsys="Li-O-P", em_pred=0.30, em_dft=0.40, converged=True,
             has_specific_TM=False),
        dict(edge_id="c", chemsys="Li-O-S", em_pred=9.00, em_dft=0.40, converged=False,
             has_specific_TM=False),                            # not converged: not scored
        dict(edge_id="d", chemsys="Fe-Li-O", em_pred=0.70, em_dft=0.40, converged=True,
             has_specific_TM=True),
    ]
    for h in hops:
        (tmp_path / f"m_{h['edge_id']}.json").write_text(json.dumps(h))
    res = nebdft2k.summary([str(tmp_path)], ["m"])["m"]
    assert res["without_U"]["n_kept"] == 2
    assert res["without_U"]["mae"] == pytest.approx(0.1)
    assert res["without_U"]["median_signed"] == pytest.approx(0.0)
    assert res["without_U"]["n_not_converged"] == 1
    assert res["with_U"]["mae"] == pytest.approx(0.3)
