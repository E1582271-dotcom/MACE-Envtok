"""MatPES ingestion, the per-structure/per-atom scoring and the estimator tables."""

import json

import numpy as np
import pytest
import torch

from envtok_mace.bench import matpes as bench
from envtok_mace.data import matpes as ingest
from envtok_mace.data.samplers import eval_shards

# A record in the shape of the MatPES release: a pymatgen structure dict plus
# energy / forces / stress, the stress a 6-vector of VASP's kbar line.
RAW_STRESS = [1.0, 2.0, 3.0, 40.0, 50.0, 60.0]  # XX YY ZZ XY YZ ZX


def _record(sid, symbols, cell=((4.0, 0, 0), (0, 4.0, 0), (0, 0, 4.0))):
    cell = np.asarray(cell, dtype=float)
    frac = np.linspace(0.1, 0.6, len(symbols))
    return {
        "matpes_id": sid,
        "structure": {
            "lattice": {"matrix": cell.tolist(), "pbc": [True, True, True]},
            "sites": [
                {"species": [{"element": s, "occu": 1}], "xyz": (np.array([f, f, f]) @ cell).tolist()}
                for s, f in zip(symbols, frac)
            ],
        },
        "energy": -1.5 * len(symbols),
        "forces": [[0.1 * (i + 1), -0.2, 0.3] for i in range(len(symbols))],
        "stress": RAW_STRESS,
    }


@pytest.fixture
def jsonl(tmp_path):
    path = tmp_path / "matpes_sample.jsonl"
    records = [
        _record("matpes-a_1", ["Si", "O"]),
        _record("matpes-b_2", ["Fe", "Fe", "O"]),
        _record("matpes-c_3", ["Ce", "O"]),
    ]  # Ce is outside the model's table
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def test_element_filter_and_order(jsonl):
    frames = ingest.read_matpes(jsonl)
    assert list(frames.sid) == ["matpes-a_1", "matpes-b_2"]  # the Ce structure is dropped
    assert list(frames.natoms) == [2, 3]
    assert list(frames.numbers) == [14, 8, 26, 26, 8]
    assert frames.energy[1] == pytest.approx(-4.5)


def test_sid_selection_sets_the_order(jsonl):
    frames = ingest.read_matpes(jsonl, keep_sids=["matpes-b_2", "matpes-a_1"])
    assert list(frames.sid) == ["matpes-b_2", "matpes-a_1"]
    with pytest.raises(ValueError, match="not in"):
        ingest.read_matpes(jsonl, keep_sids=["matpes-a_1", "matpes-nope"])


def test_stress_units_sign_and_voigt_order(jsonl):
    """kbar -> eV/A^3 with the sign flipped; the release's order is already Voigt."""
    k = ingest.KBAR_TO_EV_PER_A3
    assert ingest.read_matpes(jsonl).stress[0] == pytest.approx(-k * np.array(RAW_STRESS))
    # reading the same numbers as VASP's XX YY ZZ XY YZ ZX line moves the shears
    as_vasp = ingest.read_matpes(jsonl, stress_order="vasp").stress[0]
    assert as_vasp == pytest.approx(-k * np.array([1.0, 2.0, 3.0, 50.0, 60.0, 40.0]))


def test_restress_moves_between_conventions(jsonl):
    frames = ingest.read_matpes(jsonl)
    stored = (frames.attrs["stress_sign"], frames.attrs["stress_order"])
    assert stored == (-1.0, "voigt")
    assert bench._restress(frames.stress, stored, stored) == pytest.approx(frames.stress)
    flipped = bench._restress(frames.stress, stored, (1.0, "vasp"))
    assert flipped[0] == pytest.approx(ingest.KBAR_TO_EV_PER_A3 * np.array([1.0, 2.0, 3.0, 50.0, 60.0, 40.0]))


# ------------------------------------------------------------------- estimators
def test_estimators_match_plain_numpy_with_equal_weights():
    v = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    w = np.ones_like(v)
    assert bench.wmean(v, w) == pytest.approx(v.mean())
    assert bench.wquantile(v, w, 0.5) == 3.0
    assert bench.wgeomean(v, w) == pytest.approx(np.exp(np.log(v).mean()))
    # 20% of the weight trimmed off the upper tail drops the 100
    assert bench.wtrimmed_mean(v, w, 0.2) == pytest.approx(v[:4].mean())
    assert bench.wwinsor_mean(v, w, 4.0) == pytest.approx(np.clip(v, None, 4.0).mean())


def test_weights_and_cap_are_shared_across_models():
    v, w = np.array([1.0, 10.0]), np.array([9.0, 1.0])
    assert bench.wmean(v, w) == pytest.approx((9 * 1 + 10) / 10)
    cap = bench.pooled_cap([np.array([1.0, 2.0]), np.array([1e6, 3.0])], [np.ones(2), np.ones(2)], q=0.75)
    assert cap == 3.0  # the outlier is above the 0.75 quantile of the pool


def test_worst_fill_replaces_blowups_with_the_worst_finite_value():
    out = bench.worst_fill(np.array([1.0, np.inf, 5.0, np.nan]))
    assert list(out) == [1.0, 5.0, 5.0, 5.0]


def test_tables_use_the_reference_domains_and_atom_weights():
    sid = np.array(["a", "b", "c"])
    natoms = np.array([1, 1, 98])
    ref = {
        "sid": sid,
        "natoms": natoms,
        "domain": np.array(["bulk", "bulk", "non_bulk"]),
        "f_mae_comp": np.array([1.0, 3.0, 10.0]),
    }
    ours = {
        "sid": sid[::-1],
        "natoms": natoms[::-1],
        "f_mae_comp": np.array([10.0, 3.0, 1.0]),
    }  # same values, reversed order
    runs, domain = bench.align({"theirs": ref, "ours": ours}, domain_from="theirs", rule="stored")
    assert list(runs["ours"]["f_mae_comp"]) == [1.0, 3.0, 10.0]
    out = bench.tables(runs, domain)
    bulk = out["metrics"]["f_mae_comp"]["domains"]["bulk"]["estimators"]
    assert bulk["pooled_mean_MAE"]["ours"] == pytest.approx(2.0)
    allrows = out["metrics"]["f_mae_comp"]["domains"]["all"]["estimators"]
    # atom-weighted: the 98-atom structure dominates the pooled mean
    assert allrows["pooled_mean_MAE"]["ours"] == pytest.approx((1 + 3 + 98 * 10) / 100)
    assert out["metrics"]["f_mae_comp"]["domains"]["non_bulk"]["n"] == 1


def test_table_units_name_the_stored_values(tmp_path):
    runs = {"ours": {"sid": np.array(["a"]), "natoms": np.array([2]), "f_mae_comp": np.array([0.11])}}
    out = bench.tables(runs, np.array(["bulk"]))
    bench.write_tables(out, tmp_path)
    saved = json.loads((tmp_path / "matpes_tables.json").read_text())["metrics"]["f_mae_comp"]
    assert saved["unit"] == "eV/A"
    assert saved["domains"]["bulk"]["estimators"]["median"]["ours"] == pytest.approx(0.11)
    lines = (tmp_path / "f_mae_comp_bulk.csv").read_text().splitlines()
    assert "in meV/A" in lines[0]
    assert lines[-1].endswith("110")  # the CSVs are scaled to meV


def test_paper_domains_relabel_by_dimensionality_only():
    """Fragmented but 3D-periodic cells count as bulk; overcompressed keeps its label."""
    run = {
        "domain": np.array(["bulk", "non_bulk", "non_bulk", "overcompressed"]),
        "dim": np.array([3, 3, 2, 1]),
    }
    assert list(bench.paper_domains(run)) == ["bulk", "bulk", "non_bulk", "overcompressed"]


# ---------------------------------------------------------------------- scoring
def test_scoring_identities_against_the_pooled_metrics(val40, envtok_model_path):
    """The atom-weighted pooled mean of the per-structure force MAE must equal the
    plain force MAE over every component, and the per-atom file must agree with it."""
    from envtok_mace.eval.metrics import evaluate
    from envtok_mace.eval.run_eval import load_model

    device, dtype = torch.device("cpu"), torch.float64
    model = load_model(envtok_model_path, device, dtype)
    idx = np.arange(len(val40))
    batches = eval_shards(idx, val40.nedges, 1, 0, 8)
    ps, pa = bench.score(model, val40, batches, device, dtype)
    metrics = evaluate(model, val40, batches, device, dtype)

    w = ps["natoms"].astype(float)
    assert 1e3 * bench.wmean(ps["f_mae_comp"], w) == pytest.approx(metrics["force_mae_meV_per_A"], rel=1e-9)
    assert 1e3 * bench.wmean(ps["abs_d_energy_per_atom"], np.ones_like(w)) == pytest.approx(
        metrics["energy_mae_meV_per_atom"], rel=1e-9
    )
    # The pooled stress metric averages all nine components of the 3x3, the paper's
    # quantity the six Voigt ones, so they differ; their ratio is bounded by
    # [0.75, 1.5] (both extremes mean a purely diagonal or purely shear error).
    ratio = bench.wmean(ps["s_mae_voigt6"], np.ones_like(w)) / (1e-3 * metrics["stress_mae_meV_per_A3"])
    assert 0.75 <= ratio <= 1.5

    # Independent path for the Voigt-6 stress error: compare the model's 3x3 output
    # against the Voigt-6 reference as stored, without going through the batcher.
    from envtok_mace.data.batch import collate

    batch = collate(val40, idx).to(device=device, dtype=dtype)
    pred = model(batch, compute_force=True, compute_stress=True, training=False)["stress"]
    voigt = pred.detach().double()[:, [0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]].numpy()
    assert np.allclose(np.abs(voigt - val40.stress).mean(axis=1), ps["s_mae_voigt6"], rtol=1e-9)

    assert pa["err_norm"].shape == (int(val40.atom_ptr[-1]),)
    assert np.array_equal(pa["struct_offset"], val40.atom_ptr)
    assert np.allclose(pa["ref_norm"], np.linalg.norm(val40.forces, axis=1), rtol=1e-6)
    per_struct_max = np.array(
        [pa["err_norm"][a:b].max() for a, b in zip(pa["struct_offset"][:-1], pa["struct_offset"][1:])]
    )
    assert np.allclose(per_struct_max, ps["f_max_norm"], rtol=1e-5)
    assert ps["f_finite"].all() and ps["e_finite"].all()


def test_compare_per_atom_round_trip(tmp_path, val40, envtok_model_path):
    """Scoring the same model twice, in a different structure order, must agree."""
    from envtok_mace.eval.run_eval import load_model

    device, dtype = torch.device("cpu"), torch.float64
    model = load_model(envtok_model_path, device, dtype)
    a, b = tmp_path / "a", tmp_path / "b"
    for out, order in ((a, np.arange(len(val40))), (b, np.arange(len(val40))[::-1])):
        frames = val40.subset(order)
        ps, pa = bench.score(
            model, frames, eval_shards(np.arange(len(frames)), frames.nedges, 1, 0, 8), device, dtype
        )
        out.mkdir()
        np.savez(out / "per_structure.npz", **ps)
        np.savez(out / "per_atom_forces.npz", **pa)
    report = bench.compare_per_atom(a, b)
    assert report["atom_order_matches"]
    assert report["n_atoms"] == int(val40.atom_ptr[-1])
    assert report["max_abs_d_err_norm_eV_per_A"] < 1e-9


def test_exceedance_is_the_atom_weighted_tail():
    runs = {
        "m": {
            "f_mae_comp": np.array([0.010, 0.100, 1.000, np.inf]),  # eV/A; inf = blow-up
            "natoms": np.array([1, 1, 2, 4]),
        }
    }
    domain = np.array(["bulk", "bulk", "bulk", "non_bulk"])
    rows = bench.exceedance(runs, domain, points_per_decade=1, lo_meV=1.0, hi_meV=1e4)
    bulk = {r["threshold_meV_per_A"]: r["m"] for r in rows if r["domain"] == "bulk"}
    assert bulk[1.0] == pytest.approx(1.0)  # every atom above 1 meV/A
    assert bulk[10.0] == pytest.approx(0.75)  # the 10 meV/A structure is not above 10
    assert bulk[100.0] == pytest.approx(0.5)
    assert bulk[1e4] == pytest.approx(0.0)
    non_bulk = [r["m"] for r in rows if r["domain"] == "non_bulk"]
    assert non_bulk == pytest.approx([1.0] * len(non_bulk))  # a lone blow-up stays the worst
