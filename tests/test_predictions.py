"""Per-structure predictions: saving them changes no metric, and they reproduce the metrics."""

import argparse
from pathlib import Path

import numpy as np
import pytest
import torch

from envtok_mace.analysis import accuracy
from envtok_mace.bench import matpes
from envtok_mace.data.batch import voigt_to_full
from envtok_mace.data.samplers import eval_shards
from envtok_mace.eval import predictions, run_eval
from envtok_mace.eval.metrics import evaluate
from envtok_mace.model.mace import MACE, MACEConfig

FIXTURE = str(Path(__file__).parent / "fixtures" / "omat24_val_40.h5")
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    m = MACE(MACEConfig(num_channels=8, radial_mlp=(8, 8)))
    m.scale.fill_(10.0)  # so the errors are not dominated by the E0s
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    torch.save({"model_config": m.config.to_dict(), "model": m.state_dict()}, path)
    torch.set_default_dtype(old)
    return str(path)


def _eval_args(*argv):
    p = argparse.ArgumentParser()
    run_eval.add_arguments(p)
    return p.parse_args(["--frames", FIXTURE, "--device", "cpu", "--dtype", "float64", *argv])


def _scored(checkpoint, val40):
    model = run_eval.load_model(checkpoint, CPU, torch.float64)
    # reversed, so the collector has to put the structures back in order
    batches = eval_shards(np.arange(len(val40)), val40.nedges, 1, 0, 8, 60000)[::-1]
    collect = predictions.Collector()
    metrics = evaluate(model, val40, batches, CPU, torch.float64, collect=collect)
    return metrics, collect


@pytest.fixture
def one_thread():
    """Multi-threaded CPU reductions differ in the last bit from run to run."""
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_saving_predictions_leaves_the_results_unchanged(checkpoint, tmp_path, one_thread):
    plain, saved = tmp_path / "plain.json", tmp_path / "saved.json"
    run_eval.run(_eval_args("--models", checkpoint, "--labels", "tiny", "--out", str(plain)))
    run_eval.run(
        _eval_args(
            "--models", checkpoint, "--labels", "tiny", "--out", str(saved),
            "--save-predictions", str(tmp_path / "pred"),
        )
    )  # fmt: skip
    assert plain.read_bytes() == saved.read_bytes()
    p = predictions.load(tmp_path / "pred" / "tiny.h5")
    assert p["attrs"]["model_sha256"] == predictions.sha256(checkpoint)
    assert p["attrs"]["dtype"] == "float64"


def test_predictions_reproduce_the_metrics(checkpoint, val40, tmp_path):
    metrics, collect = _scored(checkpoint, val40)
    p = predictions.load(collect.save(tmp_path / "p.h5", val40))

    assert np.array_equal(p["sid"], val40.sid)
    assert np.array_equal(p["atom_ptr"], val40.atom_ptr)
    assert np.array_equal(p["forces_dft"], val40.forces)
    e = (p["energy_pred"] - p["energy_dft"]) / p["natoms"]
    f = p["forces_pred"] - p["forces_dft"]
    s = voigt_to_full(p["stress_pred"] - p["stress_dft"])  # the metrics average all 9 components
    assert 1e3 * np.abs(e).mean() == pytest.approx(metrics["energy_mae_meV_per_atom"], rel=1e-12)
    assert 1e3 * np.sqrt((e**2).mean()) == pytest.approx(metrics["energy_rmse_meV_per_atom"], rel=1e-12)
    assert 1e3 * np.abs(f).mean() == pytest.approx(metrics["force_mae_meV_per_A"], rel=1e-12)
    assert 1e3 * np.abs(s).mean() == pytest.approx(metrics["stress_mae_meV_per_A3"], rel=1e-12)

    s_all = accuracy.reduce({"tiny": p}, {"all": np.ones(len(p["sid"]), bool)})["summary"]
    got = s_all["models"]["tiny"]["domains"]["all"]
    assert got["energy_mae_meV_per_atom"] == pytest.approx(metrics["energy_mae_meV_per_atom"], rel=1e-12)
    assert got["force_mae_meV_per_A"] == pytest.approx(metrics["force_mae_meV_per_A"], rel=1e-12)


def test_a_subset_is_stored_in_frame_order(checkpoint, val40):
    model = run_eval.load_model(checkpoint, CPU, torch.float64)
    collect = predictions.Collector()
    evaluate(model, val40, [np.array([30, 5]), np.array([12])], CPU, torch.float64, collect=collect)
    a = collect.arrays(val40)
    assert a["index"].tolist() == [5, 12, 30]
    assert np.array_equal(a["sid"], val40.sid[[5, 12, 30]])
    sub = val40.subset(np.array([5, 12, 30]))
    assert np.array_equal(a["numbers"], sub.numbers) and np.array_equal(a["forces_dft"], sub.forces)
    with pytest.raises(ValueError, match="more than once"):
        collect(np.array([5]), *_one_output(model, val40, 5))
        collect.arrays(val40)


def _one_output(model, frames, i):
    from envtok_mace.data.batch import collate

    batch = collate(frames, np.array([i]))
    return model(batch, compute_force=True, compute_stress=True, training=False), batch


def test_matpes_scores_are_unchanged_by_collecting(checkpoint, val40):
    model = run_eval.load_model(checkpoint, CPU, torch.float64)
    batches = eval_shards(np.arange(len(val40)), val40.nedges, 1, 0, 8, 60000)
    plain, _ = matpes.score(model, val40, batches, CPU, torch.float64)
    collect = predictions.Collector()
    with_collect, _ = matpes.score(model, val40, batches, CPU, torch.float64, collect=collect)
    for k, v in plain.items():
        assert np.array_equal(v, with_collect[k]), k
    a = collect.arrays(val40)
    de = np.abs(a["energy_pred"] - a["energy_dft"]) / a["natoms"]
    np.testing.assert_allclose(de, plain["abs_d_energy_per_atom"], rtol=1e-12)


# ------------------------------------------------------------ reduction tables
def _synthetic(val40, e_err, f_err, s_err):
    """Predictions off by exact amounts: e_err eV/atom, f_err eV/A, s_err eV/A^3."""
    natoms = val40.natoms.astype(np.int64)
    return {
        "index": np.arange(len(val40)),
        "sid": val40.sid.astype(str),
        "source": val40.source.astype(str),
        "natoms": natoms,
        "atom_ptr": val40.atom_ptr,
        "numbers": val40.numbers,
        "energy_dft": val40.energy,
        "energy_pred": val40.energy + e_err * natoms,
        "forces_dft": val40.forces,
        "forces_pred": val40.forces + f_err,
        "stress_dft": val40.stress,
        "stress_pred": val40.stress + s_err,
    }


def test_reduction_tables(val40, tmp_path):
    paths = []
    for label, k in (("one", 1.0), ("two", 2.0)):
        arrays = _synthetic(val40, 0.01 * k, 0.05 * k, 0.001 * k)
        paths.append(str(predictions.save(tmp_path / f"{label}.h5", arrays, model=label)))
    runs = accuracy.load_runs(paths, ["one", "two"])
    out = accuracy.reduce(runs, accuracy.domains(runs["one"]["sid"], None))
    n, a = len(val40), int(val40.atom_ptr[-1])

    for label, k in (("one", 1.0), ("two", 2.0)):
        s = out["summary"]["models"][label]["domains"]["all"]
        assert s["energy_mae_meV_per_atom"] == pytest.approx(10 * k)
        assert s["force_mae_meV_per_A"] == pytest.approx(50 * k)
        assert s["stress_mae_meV_per_A3"] == pytest.approx(1 * k)
        assert s["n_structures"] == n and s["n_atoms"] == a
        for q, total in (("energy", n), ("forces", 3 * a), ("stress", 6 * n)):
            c = out["parity"]["counts"][label]["all"][q]
            assert c["n"] == total
            assert sum(x[2] for x in c["ijc"]) + c["n_outside"] == total

    one = [r for r in out["per_element"] if r["model"] == "one"]
    assert sum(r["n_atoms"] for r in one) == a
    assert all(r["force_mae_meV_per_A"] == pytest.approx(50) for r in one)
    src = [r for r in out["per_source"] if r["model"] == "one"]
    assert {r["source"] for r in src} == set(val40.source)
    assert sum(r["n_atoms"] for r in out["force_magnitude"] if r["model"] == "one") == a

    accuracy.write(out, tmp_path / "tables")
    assert {p.name for p in (tmp_path / "tables").iterdir()} == {
        "summary.json", "parity.json", "per_element.csv", "per_source.csv", "force_magnitude.csv",
    }  # fmt: skip


def test_nonfinite_structures_are_left_out_and_counted(val40, tmp_path):
    arrays = _synthetic(val40, 0.01, 0.05, 0.001)
    arrays["forces_pred"][val40.atom_ptr[3]] = np.nan  # first atom of structure 3
    run = predictions.load(predictions.save(tmp_path / "x.h5", arrays))
    out = accuracy.reduce({"x": run}, {"all": np.ones(len(val40), bool)})
    m = out["summary"]["models"]["x"]
    assert m["n_nonfinite"] == 1
    assert m["domains"]["all"]["n_structures"] == len(val40) - 1
    assert m["domains"]["all"]["force_mae_meV_per_A"] == pytest.approx(50)


def test_runs_on_different_structures_are_refused(val40, tmp_path):
    a = _synthetic(val40, 0.01, 0.05, 0.001)
    b = {k: (v[::-1] if k == "sid" else v) for k, v in a.items()}
    pa = predictions.save(tmp_path / "a.h5", a)
    pb = predictions.save(tmp_path / "b.h5", b)
    with pytest.raises(ValueError, match="same structures"):
        accuracy.load_runs([str(pa), str(pb)], ["a", "b"])


def test_matpes_domains_follow_the_paper_labels(val40, tmp_path):
    sid = val40.sid.astype(str)
    dim = np.where(np.arange(len(sid)) % 3 == 0, 2, 3)
    stored = np.array(["bulk"] * len(sid), dtype="<U14")
    stored[1] = "overcompressed"
    np.savez(tmp_path / "per_structure.npz", sid=sid[::-1], dim=dim[::-1], domain=stored[::-1])
    d = accuracy.domains(sid, str(tmp_path / "per_structure.npz"))
    assert d["overcompressed"].tolist() == [i == 1 for i in range(len(sid))]
    assert np.array_equal(d["non_bulk"], (dim < 3) & ~d["overcompressed"])
    assert np.array_equal(d["bulk"], (dim == 3) & ~d["overcompressed"])
