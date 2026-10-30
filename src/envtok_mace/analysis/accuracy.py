"""Reduce per-structure prediction files to the small tables the accuracy figures draw from.

Input: one predictions file per model (``eval/predictions.py``), all made on the same
frames. Output, in ``--out``, small enough for ``results/``:

- ``summary.json``: per model and domain, MAE / RMSE / R^2 and error quantiles of the
  energy per atom, the force components and the six Voigt stress components (the
  stress MAE of ``eval`` averages all nine tensor components, so it is smaller);
- ``parity.json``: 2-D histograms of predicted against DFT values on bins shared by
  every model, stored sparse (forces on asinh-spaced bins, because their range spans
  five decades);
- ``per_element.csv``: force error of the atoms of each element, and energy error of
  the structures that contain it;
- ``per_source.csv``: errors per OMat24 sub-dataset (the frames' ``source``);
- ``force_magnitude.csv``: per-atom force error against the size of the DFT force;
- ``training_sources.csv`` (with ``--training``): structures and atoms of each
  sub-dataset in the training set, to set the per-source errors against.

Domains are "all", plus MatPES bulk / non_bulk / overcompressed when ``--domains``
points at the authors' per-structure file (labelled as in ``bench.matpes.paper_domains``).
A structure whose prediction is not finite is left out of every number and counted.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..eval import predictions

EV_PER_A3_TO_GPA = 160.21766208
QUANTILES = (0.5, 0.9, 0.99)
N_BINS = 100
FORCE_BIN_SCALE = 0.1  # eV/A; bins are uniform in asinh(F / FORCE_BIN_SCALE)
FORCE_MAGNITUDE_EDGES = np.concatenate([[0.0], np.logspace(-2, 2, 17)])  # eV/A


def load_runs(paths: list[str], labels: list[str]) -> dict[str, dict]:
    runs = {label: predictions.load(p) for label, p in zip(labels, paths)}
    first = runs[labels[0]]
    for label in labels[1:]:
        if not np.array_equal(runs[label]["sid"], first["sid"]):
            raise ValueError(f"{label} was not evaluated on the same structures as {labels[0]}")
    return runs


def domains(sid: np.ndarray, per_structure_npz: str | None) -> dict[str, np.ndarray]:
    out = {"all": np.ones(len(sid), bool)}
    if per_structure_npz is None:
        return out
    from ..bench.matpes import DOMAIN_ORDER, paper_domains

    z = np.load(per_structure_npz, allow_pickle=False)
    pos = {s: i for i, s in enumerate(z["sid"].astype(str))}
    missing = [s for s in sid if s not in pos]
    if missing:
        raise ValueError(f"{len(missing)} sids have no domain, e.g. {missing[:3]}")
    take = np.array([pos[s] for s in sid])
    label = paper_domains({"domain": z["domain"][take], "dim": z["dim"][take]})
    for k in DOMAIN_ORDER[1:]:
        if (label == k).any():
            out[k] = label == k
    return out


def finite(run: dict) -> np.ndarray:
    """Structures whose predicted energy, forces and stress are all finite."""
    f_ok = np.isfinite(run["forces_pred"]).all(axis=1)
    f_ok = np.logical_and.reduceat(f_ok, run["atom_ptr"][:-1])
    return np.isfinite(run["energy_pred"]) & f_ok & np.isfinite(run["stress_pred"]).all(axis=1)


def _errors(run: dict) -> dict[str, np.ndarray]:
    natoms = run["natoms"].astype(np.float64)
    return {
        "e_ref": run["energy_dft"] / natoms,  # eV/atom
        "e_err": (run["energy_pred"] - run["energy_dft"]) / natoms,
        "f_ref": run["forces_dft"].astype(np.float64),  # eV/A
        "f_err": run["forces_pred"].astype(np.float64) - run["forces_dft"].astype(np.float64),
        "s_ref": run["stress_dft"].astype(np.float64),  # eV/A^3
        "s_err": run["stress_pred"].astype(np.float64) - run["stress_dft"].astype(np.float64),
    }


def _r2(ref: np.ndarray, err: np.ndarray) -> float:
    return float(1.0 - (err**2).sum() / ((ref - ref.mean()) ** 2).sum())


def _quantiles(v: np.ndarray, scale: float) -> dict[str, float]:
    q = np.quantile(v, QUANTILES) * scale
    return {f"p{round(100 * k)}": float(x) for k, x in zip(QUANTILES, q)} | {"max": float(v.max() * scale)}


def summary(err: dict, smask: np.ndarray, amask: np.ndarray) -> dict:
    e_ref, e_err = err["e_ref"][smask], err["e_err"][smask]
    f_ref, f_err = err["f_ref"][amask], err["f_err"][amask]
    s_ref, s_err = err["s_ref"][smask], err["s_err"][smask]
    return {
        "n_structures": int(smask.sum()),
        "n_atoms": int(amask.sum()),
        "energy_mae_meV_per_atom": 1e3 * float(np.abs(e_err).mean()),
        "energy_rmse_meV_per_atom": 1e3 * float(np.sqrt((e_err**2).mean())),
        "energy_r2": _r2(e_ref, e_err),
        "energy_abs_err_meV_per_atom": _quantiles(np.abs(e_err), 1e3),
        "force_mae_meV_per_A": 1e3 * float(np.abs(f_err).mean()),
        "force_rmse_meV_per_A": 1e3 * float(np.sqrt((f_err**2).mean())),
        "force_r2": _r2(f_ref, f_err),
        "force_err_norm_meV_per_A": _quantiles(np.linalg.norm(f_err, axis=1), 1e3),
        "stress_mae_voigt6_meV_per_A3": 1e3 * float(np.abs(s_err).mean()),
        "stress_r2": _r2(s_ref, s_err),
        "stress_struct_mae_voigt6_meV_per_A3": _quantiles(np.abs(s_err).mean(axis=1), 1e3),
    }


def bin_edges(err: dict, ok: np.ndarray, aok: np.ndarray) -> dict[str, np.ndarray]:
    """Bins shared by all models and domains, from the DFT values of the first model."""

    def linear(v):
        lo, hi = np.quantile(v, [5e-4, 1 - 5e-4])
        pad = 0.05 * (hi - lo)
        return np.linspace(lo - pad, hi + pad, N_BINS + 1)

    fmax = np.arcsinh(np.quantile(np.abs(err["f_ref"][aok]), 1 - 1e-4) / FORCE_BIN_SCALE)
    return {
        "energy": linear(err["e_ref"][ok]),
        "forces": FORCE_BIN_SCALE * np.sinh(np.linspace(-fmax, fmax, N_BINS + 1)),
        "stress": linear(err["s_ref"][ok].ravel() * EV_PER_A3_TO_GPA),
    }


def parity_counts(err: dict, smask, amask, edges: dict) -> dict[str, dict]:
    pairs = {
        "energy": (err["e_ref"][smask], err["e_ref"][smask] + err["e_err"][smask]),
        "forces": (err["f_ref"][amask].ravel(), (err["f_ref"][amask] + err["f_err"][amask]).ravel()),
        "stress": (
            err["s_ref"][smask].ravel() * EV_PER_A3_TO_GPA,
            (err["s_ref"][smask] + err["s_err"][smask]).ravel() * EV_PER_A3_TO_GPA,
        ),
    }
    out = {}
    for q, (x, y) in pairs.items():
        h, _, _ = np.histogram2d(x, y, bins=[edges[q], edges[q]])
        i, j = np.nonzero(h)
        out[q] = {
            "n": int(len(x)),
            "n_outside": int(len(x) - h.sum()),
            "ijc": np.stack([i, j, h[i, j].astype(np.int64)], axis=1).tolist(),
        }
    return out


def per_element(run: dict, err: dict, smask, amask) -> list[dict]:
    from ase.data import chemical_symbols

    z = run["numbers"].astype(np.int64)
    frame = np.repeat(np.arange(len(run["natoms"])), run["natoms"])
    # (structure, element) pairs, each once
    pairs = np.unique(frame * 256 + z)
    p_frame, p_z = pairs // 256, pairs % 256
    rows = []
    for el in np.unique(z[amask]):
        a = amask & (z == el)
        s = p_frame[(p_z == el) & smask[p_frame]]
        rows.append(
            {
                "Z": int(el),
                "symbol": chemical_symbols[el],
                "n_atoms": int(a.sum()),
                "force_mae_meV_per_A": 1e3 * float(np.abs(err["f_err"][a]).mean()),
                "n_structures": int(len(s)),
                "energy_mae_meV_per_atom": 1e3 * float(np.abs(err["e_err"][s]).mean()),
            }
        )
    return rows


def per_source(run: dict, err: dict, smask, amask) -> list[dict]:
    src_atom = np.repeat(run["source"], run["natoms"])
    rows = []
    for src in np.unique(run["source"][smask]):
        s, a = smask & (run["source"] == src), amask & (src_atom == src)
        rows.append(
            {
                "source": str(src),
                "n_structures": int(s.sum()),
                "n_atoms": int(a.sum()),
                "energy_mae_meV_per_atom": 1e3 * float(np.abs(err["e_err"][s]).mean()),
                "force_mae_meV_per_A": 1e3 * float(np.abs(err["f_err"][a]).mean()),
                "stress_mae_voigt6_meV_per_A3": 1e3 * float(np.abs(err["s_err"][s]).mean()),
            }
        )
    return rows


def force_magnitude(err: dict, amask) -> list[dict]:
    ref = np.linalg.norm(err["f_ref"][amask], axis=1)
    e = np.linalg.norm(err["f_err"][amask], axis=1)
    which = np.digitize(ref, FORCE_MAGNITUDE_EDGES[1:-1])  # values past the last edge join the top bin
    rows = []
    for b in range(len(FORCE_MAGNITUDE_EDGES) - 1):
        m = which == b
        if not m.any():
            continue
        rel = e[m] / np.maximum(ref[m], 1e-12)
        rows.append(
            {
                "ref_lo_eV_per_A": float(FORCE_MAGNITUDE_EDGES[b]),
                "ref_hi_eV_per_A": float(FORCE_MAGNITUDE_EDGES[b + 1]),
                "n_atoms": int(m.sum()),
                "mean_err_norm_meV_per_A": 1e3 * float(e[m].mean()),
                "median_err_norm_meV_per_A": 1e3 * float(np.median(e[m])),
                "median_relative_err": float(np.median(rel)),
            }
        )
    return rows


def composition(frames_path: str) -> list[dict]:
    """Structures and atoms of each sub-dataset in a Frames file, read without loading the rest."""
    import h5py

    with h5py.File(frames_path, "r") as h:
        source = h["source"].asstr()[()]
        natoms = np.diff(h["atom_ptr"][()])
    return [
        {
            "source": str(s),
            "n_structures": int((source == s).sum()),
            "n_atoms": int(natoms[source == s].sum()),
        }
        for s in np.unique(source)
    ]


def reduce(runs: dict[str, dict], doms: dict[str, np.ndarray]) -> dict:
    labels = list(runs)
    first = runs[labels[0]]
    errs = {label: _errors(run) for label, run in runs.items()}
    oks = {label: finite(run) for label, run in runs.items()}
    a_oks = {label: np.repeat(ok, runs[label]["natoms"]) for label, ok in oks.items()}
    edges = bin_edges(errs[labels[0]], oks[labels[0]], a_oks[labels[0]])
    out = {
        "summary": {"labels": labels, "n_structures": int(len(first["sid"])), "models": {}},
        "parity": {
            "x": "DFT",
            "y": "predicted",
            "units": {"energy": "eV/atom", "forces": "eV/A", "stress": "GPa"},
            "edges": {q: e.tolist() for q, e in edges.items()},
            "counts": {},
        },
        "per_element": [],
        "per_source": [],
        "force_magnitude": [],
    }
    for label in labels:
        run, err = runs[label], errs[label]
        out["summary"]["models"][label] = {
            "model": Path(str(run["attrs"].get("model", ""))).name,  # no local paths in results/
            "model_sha256": str(run["attrs"].get("model_sha256", "")),
            "dtype": str(run["attrs"].get("dtype", "")),
            "n_nonfinite": int((~oks[label]).sum()),
            "domains": {},
        }
        out["parity"]["counts"][label] = {}
        for dom, dmask in doms.items():
            smask = dmask & oks[label]
            amask = np.repeat(smask, run["natoms"])
            out["summary"]["models"][label]["domains"][dom] = summary(err, smask, amask)
            out["parity"]["counts"][label][dom] = parity_counts(err, smask, amask, edges)
            tag = {"model": label, "domain": dom}
            out["per_element"] += [tag | r for r in per_element(run, err, smask, amask)]
            out["per_source"] += [tag | r for r in per_source(run, err, smask, amask)]
            out["force_magnitude"] += [tag | r for r in force_magnitude(err, amask)]
    return out


def _write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def write(reduced: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(reduced["summary"], indent=1) + "\n")
    (out / "parity.json").write_text(json.dumps(reduced["parity"], separators=(",", ":")) + "\n")
    for name in ("per_element", "per_source", "force_magnitude"):
        _write_csv(reduced[name], out / f"{name}.csv")
    size = sum(p.stat().st_size for p in out.iterdir())
    print(f"{len(reduced['summary']['labels'])} models -> {out} ({size / 1e3:.0f} kB)", flush=True)


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--predictions", nargs="+", required=True, help="predictions files, one per model")
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument(
        "--domains",
        default=None,
        metavar="PER_STRUCTURE_NPZ",
        help="MatPES: the authors' per_structure.npz, for the bulk / non-bulk split",
    )
    p.add_argument(
        "--training",
        default=None,
        metavar="FRAMES",
        help="the training Frames file: also write its structures and atoms per sub-dataset",
    )
    p.add_argument("--out", required=True, help="e.g. results/accuracy/omat24_val90")


def run(args: argparse.Namespace) -> None:
    labels = args.labels or [Path(p).stem for p in args.predictions]
    runs = load_runs(args.predictions, labels)
    write(reduce(runs, domains(runs[labels[0]]["sid"], args.domains)), Path(args.out))
    if args.training:
        _write_csv(composition(args.training), Path(args.out) / "training_sources.csv")
