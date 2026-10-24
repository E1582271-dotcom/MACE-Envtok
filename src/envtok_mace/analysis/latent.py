"""Distance from the training data in the model's latent space, against force error.

Each atom is described by the model's invariant node features: the 128 scalar
channels of both interaction layers, 256 numbers, the descriptor the authors use
for their latent-space analysis. A reference set of training atoms defines a PCA
(95 % of the variance, at most 50 components, as in their analysis), and every
test atom gets the mean Euclidean distance to its k = 5 nearest reference atoms
in that space. The question is whether atoms far from anything seen in training
are the ones with large force errors, on in-distribution data (OMat24
validation) and out of distribution (MatPES bulk and non-bulk).

``--extract`` runs a model over a Frames file and stores, per atom, the features,
the force error norm ||F_pred - F_ref||, the element and the structure id.
``--analyze`` fits the PCA on a reference extract and scores test extracts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..data.batch import collate
from ..data.frames import Frames
from ..data.samplers import eval_shards

K_NEIGHBOURS = 5
VAR_KEEP = 0.95
MAX_DIMS = 50


# ------------------------------------------------------------------ extraction
def sample_structures(frames: Frames, n_atoms: int | None, seed: int = 0) -> np.ndarray:
    """Random structures, in random order, until ``n_atoms`` atoms are covered."""
    order = np.random.default_rng(seed).permutation(len(frames))
    if n_atoms is None:
        return np.sort(order)
    take = np.searchsorted(np.cumsum(frames.natoms[order]), n_atoms) + 1
    return np.sort(order[:take])


def extract(model, frames: Frames, idx: np.ndarray, device: torch.device, dtype: torch.dtype,
            batch_size: int = 64, max_edges: int = 60000) -> dict[str, np.ndarray]:
    """Per-atom features and force-error norms for the structures ``idx``."""
    feats, errs, zs, sids = [], [], [], []
    for b in eval_shards(idx, frames.nedges, 1, 0, batch_size, max_edges):
        b = np.sort(b)
        batch = collate(frames, b).to(device=device, dtype=dtype)
        out = model(batch, compute_force=True, compute_stress=False, training=False,
                    return_features=True)
        feats.append(out["node_features"].detach().float().cpu().numpy())
        errs.append((out["forces"].detach() - batch.forces).norm(dim=1).float().cpu().numpy())
        zs.append(batch.numbers.cpu().numpy().astype(np.int16))
        sids.append(np.repeat(frames.sid[b].astype(str), frames.natoms[b]))
    return {"features": np.concatenate(feats), "force_err": np.concatenate(errs),
            "Z": np.concatenate(zs), "sid": np.concatenate(sids)}


# ---------------------------------------------------------------- PCA and kNN
def fit_pca(x: torch.Tensor, var_keep: float = VAR_KEEP, max_dims: int = MAX_DIMS):
    """Mean, components (D x k) and explained-variance ratios of the kept components."""
    mean = x.mean(dim=0)
    xc = x - mean
    cov = xc.T @ xc / (len(x) - 1)
    evals, evecs = torch.linalg.eigh(cov.double())
    evals, evecs = evals.flip(0).clamp(min=0), evecs.flip(1)
    ratio = evals / evals.sum()
    k = int(min(max_dims, int(torch.searchsorted(ratio.cumsum(0), torch.tensor(
        var_keep, dtype=ratio.dtype, device=ratio.device))) + 1))
    return mean, evecs[:, :k].to(x.dtype), ratio[:k]


def project(x: torch.Tensor, mean: torch.Tensor, comps: torch.Tensor) -> torch.Tensor:
    return (x - mean) @ comps


def knn_mean_distance(query: torch.Tensor, ref: torch.Tensor, k: int = K_NEIGHBOURS,
                      chunk: int = 512) -> torch.Tensor:
    """Mean Euclidean distance from each query row to its k nearest reference rows."""
    out = torch.empty(len(query), dtype=query.dtype, device=query.device)
    for s in range(0, len(query), chunk):
        d = torch.cdist(query[s:s + chunk], ref)
        out[s:s + chunk] = d.topk(k, dim=1, largest=False).values.mean(dim=1)
    return out


# -------------------------------------------------------------------- analysis
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def error_by_distance(dist: np.ndarray, err: np.ndarray, edges: np.ndarray) -> list[dict]:
    """Force-error statistics of the atoms in each distance bin."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dist >= lo) & (dist < hi)
        if m.sum() == 0:
            rows.append({"lo": float(lo), "hi": float(hi), "n": 0})
            continue
        e = err[m]
        rows.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                     "median_err": float(np.median(e)), "mean_err": float(e.mean()),
                     "p90_err": float(np.quantile(e, 0.9))})
    return rows


def analyze(ref: dict, tests: dict[str, dict], device: torch.device,
            k: int = K_NEIGHBOURS, n_bins: int = 10) -> dict:
    """PCA on the reference atoms, kNN distances and error/distance statistics per set."""
    x_ref = torch.as_tensor(ref["features"], device=device)
    mean, comps, ratio = fit_pca(x_ref)
    z_ref = project(x_ref, mean, comps)
    dists = {}
    for name, t in tests.items():
        z = project(torch.as_tensor(t["features"], device=device), mean, comps)
        dists[name] = knn_mean_distance(z, z_ref, k).cpu().numpy()
    # distance bins shared by all sets: deciles of the pooled distances
    pooled = np.concatenate(list(dists.values()))
    edges = np.quantile(pooled, np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    out = {"pca_dims": int(comps.shape[1]), "explained_variance": float(ratio.sum()),
           "k": k, "n_reference_atoms": int(len(x_ref)), "bin_edges": edges.tolist(), "sets": {}}
    for name, t in tests.items():
        d, e = dists[name], t["force_err"].astype(np.float64)
        ok = np.isfinite(e)
        out["sets"][name] = {
            "n_atoms": int(len(d)),
            "distance": {q: float(np.quantile(d, p)) for q, p in
                         (("median", .5), ("p10", .1), ("p90", .9), ("p99", .99))},
            "median_force_err": float(np.median(e[ok])),
            "spearman_distance_vs_error": spearman(d[ok], e[ok]),
            "error_by_distance": error_by_distance(d[ok], e[ok], edges),
        }
    return out, dists


# -------------------------------------------------------------------------- CLI
def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--extract", action="store_true")
    p.add_argument("--model", default=None)
    p.add_argument("--frames", default=None)
    p.add_argument("--n-atoms", type=int, default=None, help="random structures up to this many atoms")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--ref", default=None, help="extract of training atoms (reference)")
    p.add_argument("--tests", nargs="+", default=None, help="extracts to score")
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--split-by", default=None, metavar="PER_STRUCTURE_NPZ",
                   help="split test extracts into MatPES bulk / non-bulk with the authors' file")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-edges", type=int, default=60000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--backend", default=None, choices=[None, "e3nn", "cueq"])


def _load(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def _split_domains(t: dict, per_structure: str) -> dict[str, dict]:
    from ..bench.matpes import paper_domains

    ps = _load(per_structure)
    dom_of = dict(zip(ps["sid"].astype(str), paper_domains(ps)))
    dom = np.array([dom_of.get(s, "unknown") for s in t["sid"].astype(str)])
    return {d: {k: v[dom == d] for k, v in t.items()} for d in ("bulk", "non_bulk")}


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if args.extract:
        from ..eval.run_eval import load_model

        dtype = getattr(torch, args.dtype)
        frames = Frames.load(args.frames)
        idx = sample_structures(frames, args.n_atoms, args.seed)
        model = load_model(args.model, device, dtype, args.backend)
        res = extract(model, frames, idx, device, dtype, args.batch_size, args.max_edges)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, **res)
        print(f"{len(idx):,} structures, {len(res['Z']):,} atoms -> {args.out}", flush=True)
        return
    if args.analyze:
        labels = args.labels or [Path(t).stem for t in args.tests]
        tests = {}
        for label, path in zip(labels, args.tests):
            t = _load(path)
            if args.split_by and label.startswith("matpes"):
                for dom, part in _split_domains(t, args.split_by).items():
                    tests[f"{label}_{dom}"] = part
            else:
                tests[label] = t
        res, _ = analyze(_load(args.ref), tests, device)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
        print(f"PCA: {res['pca_dims']} components, {100 * res['explained_variance']:.1f} % of the "
              f"variance; reference {res['n_reference_atoms']:,} atoms", flush=True)
        for name, s in res["sets"].items():
            print(f"  {name:22s} n={s['n_atoms']:>9,}  median distance {s['distance']['median']:.3f}  "
                  f"p90 {s['distance']['p90']:.3f}  median |dF| {1e3 * s['median_force_err']:.1f} meV/A  "
                  f"Spearman {s['spearman_distance_vs_error']:+.3f}", flush=True)
