"""Latent-space distance analysis: PCA, kNN distances, binning and feature extraction."""

import numpy as np
import pytest
import torch

from envtok_mace.analysis import latent


def test_knn_mean_distance_matches_brute_force():
    rng = np.random.default_rng(0)
    ref, q = rng.normal(size=(300, 7)), rng.normal(size=(50, 7))
    d = np.linalg.norm(q[:, None] - ref[None], axis=-1)
    expected = np.sort(d, axis=1)[:, :5].mean(axis=1)
    got = latent.knn_mean_distance(torch.tensor(q), torch.tensor(ref), k=5, chunk=16).numpy()
    assert np.allclose(got, expected, rtol=1e-10)


def test_pca_matches_scikit_learn():
    sklearn = pytest.importorskip("sklearn.decomposition")
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2000, 12)) @ rng.normal(size=(12, 12))
    mean, comps, ratio = latent.fit_pca(torch.tensor(x), var_keep=0.9, max_dims=50)
    ref = sklearn.PCA(n_components=comps.shape[1]).fit(x)
    assert np.allclose(ratio.numpy(), ref.explained_variance_ratio_, rtol=1e-8)
    # components agree up to sign
    dots = np.abs(np.sum(comps.numpy() * ref.components_.T, axis=0))
    assert np.allclose(dots, 1.0, atol=1e-8)
    assert ratio.sum() >= 0.9 and ratio[:-1].sum() < 0.9  # the smallest k reaching 90 %


def test_error_by_distance_bins_every_atom_once():
    dist = np.array([0.1, 0.2, 0.6, 0.9, 1.5])
    err = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rows = latent.error_by_distance(dist, err, np.array([0.0, 0.5, 1.0, 2.0]))
    assert [r["n"] for r in rows] == [2, 2, 1]
    assert rows[1]["median_err"] == pytest.approx(3.5)


def test_sample_structures_reaches_the_atom_budget(val40):
    idx = latent.sample_structures(val40, n_atoms=200, seed=0)
    assert val40.natoms[idx].sum() >= 200
    assert val40.natoms[idx].sum() - val40.natoms[idx].max() < 200
    assert np.all(np.diff(idx) > 0)


def test_extract_gives_256_invariant_features_and_force_errors(val40, envtok_model_path):
    from envtok_mace.eval.run_eval import load_model

    model = load_model(envtok_model_path, torch.device("cpu"), torch.float64)
    idx = np.arange(6)
    res = latent.extract(model, val40, idx, torch.device("cpu"), torch.float64, batch_size=4)
    n = int(val40.natoms[idx].sum())
    assert res["features"].shape == (n, 256)
    assert res["force_err"].shape == (n,) and np.isfinite(res["force_err"]).all()
    assert list(np.unique(res["sid"])) == sorted(set(val40.sid[idx].astype(str)))
    # rotating a structure leaves the invariant features unchanged
    from scipy.spatial.transform import Rotation

    rot = Rotation.from_euler("xyz", [0.3, -1.1, 2.0]).as_matrix()
    turned = val40.subset(np.array([0]))
    turned.positions = turned.positions @ rot.T
    turned.cell = turned.cell @ rot.T
    turned.forces = turned.forces @ rot.T
    a = latent.extract(model, val40, np.array([0]), torch.device("cpu"), torch.float64)
    b = latent.extract(model, turned, np.array([0]), torch.device("cpu"), torch.float64)
    assert np.allclose(a["features"], b["features"], atol=1e-6)
    assert np.allclose(a["force_err"], b["force_err"], atol=1e-6)


def test_id_ood_table_covers_every_benchmark():
    from pathlib import Path

    from envtok_mace.analysis import id_ood

    results = Path(__file__).resolve().parents[1] / "results"
    rows = id_ood.rows_from_results(results)
    assert {r["category"] for r in rows} == {"ID", "shift", "OOD"}
    tests = {r["test"].split(" (")[0] for r in rows}
    assert {"OMat24 validation", "MatPES bulk", "MatPES non-bulk", "nebDFT2k Li-ion migration"} <= tests
    assert all(np.isfinite(r[c]) for r in rows for c in id_ood.COLUMNS)
