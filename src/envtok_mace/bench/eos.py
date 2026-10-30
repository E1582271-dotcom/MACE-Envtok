"""Equation of state of WBM structures (MLIP Arena's ``eos_bulk`` task).

Each structure is relaxed, first the positions (FIRE, fmax 0.1 eV/A) and then the
cell as well (FrechetCellFilter, fmax 0.05 eV/A). It is then scaled to 21 volumes
within +-20 % and its positions are relaxed at each one. A Birch-Murnaghan fit gives
the bulk modulus. The steps and settings are the arena's; they are called here
without prefect, which would run every relaxation as a task of its own.

The curves are scored with the arena's metrics: how often the energy slope changes
sign (once for a smooth well), tortuosity (1 for a smooth well), and rank correlations
of energy with volume on the compressed and stretched sides.

The arena's set has 1,000 structures; those with an element outside the model's table
are skipped, so all models see the same set.

    envtok-mace eos --models a.pt b.model --labels ours authors_envtok --out runs/eos
    envtok-mace eos --summary --labels ours authors_envtok --out runs/eos --tables results/eos
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

from ..data.constants import Z_TABLE
from ._runner import Timeout, failure, placement, spawn, wall_timeout

DATASET, DATASET_FILE = "atomind/mlip-arena", "wbm_subset.db"
N_VOLUMES = 21
MAX_VOLUME_STRAIN = 0.2
REFERENCE = "mace_omat0"  # the set has no DFT moduli, so the models are compared with this one
METRICS = (
    "energy-diff-flip-times",
    "tortuosity",
    "spearman-compression-energy",
    "spearman-compression-derivative",
    "spearman-tension-energy",
)


def structures(db_path: str | None = None) -> list:
    """The arena's WBM structures with every element in the model's table, by id."""
    from ase.db import connect

    if db_path is None:
        from huggingface_hub import hf_hub_download

        db_path = hf_hub_download(repo_id=DATASET, filename=DATASET_FILE, repo_type="dataset")
    allowed = set(int(z) for z in Z_TABLE)
    out = []
    with connect(db_path) as db:
        for row in db.select():
            atoms = row.toatoms(add_additional_information=True)
            if set(atoms.numbers.tolist()) <= allowed:
                atoms.info["id"] = atoms.info["key_value_pairs"]["wbm_id"]
                out.append(atoms)
    return sorted(out, key=lambda a: a.info["id"])


def eos_curve(atoms, calc) -> dict:
    """E-V curve and Birch-Murnaghan fit of one structure."""
    from mlip_arena.tasks.optimize import run as opt
    from pymatgen.analysis.eos import BirchMurnaghan, EOSError

    quiet = dict(logfile=None)
    relaxed = opt.fn(atoms, calc, optimizer="FIRE", optimizer_kwargs=quiet, criterion=dict(fmax=0.1))["atoms"]
    first = opt.fn(relaxed, calc, optimizer="FIRE", optimizer_kwargs=quiet, filter="FrechetCell")
    relaxed, steps, converged = first["atoms"], first["steps"], bool(first["converged"])
    cell = relaxed.get_cell()
    volumes, energies = [], []
    for factor in np.linspace(1 - MAX_VOLUME_STRAIN, 1 + MAX_VOLUME_STRAIN, N_VOLUMES) ** (1 / 3):
        scaled = relaxed.copy()
        scaled.set_cell(cell * factor, scale_atoms=True)
        point = opt.fn(scaled, calc, optimizer="FIRE", optimizer_kwargs=quiet)
        volumes.append(float(point["atoms"].get_volume()))
        energies.append(float(point["atoms"].get_potential_energy()))
        steps += point["steps"]
        converged = converged and bool(point["converged"])
    fit = {"K_GPa": None, "b1": None, "e0_eV": None, "v0_A3": None}
    try:
        bm = BirchMurnaghan(volumes=volumes, energies=energies)
        bm.fit()
        fit = {"K_GPa": float(bm.b0_GPa), "b1": float(bm.b1), "e0_eV": float(bm.e0), "v0_A3": float(bm.v0)}
    except EOSError:  # the curve has no minimum a fit can settle on
        pass
    return {
        "volumes_A3": volumes,
        "energies_eV": energies,
        "v_input_A3": float(atoms.get_volume()),
        "steps": int(steps),
        "converged": converged,
        **fit,
    }


def _run_shard(model, label, out, index, count, device, dtype, backend, threads, timeout, limit) -> None:
    os.environ.setdefault("TQDM_DISABLE", "1")
    import torch

    from ..eval.ase_calc import EnvTokCalculator

    torch.set_num_threads(threads)
    calc = EnvTokCalculator(model, device=device, dtype=dtype, backend=backend)
    out = Path(out) / label
    todo = structures()
    for atoms in (todo[:limit] if limit else todo)[index::count]:
        path = out / f"{atoms.info['id']}.json"
        if path.exists():
            continue
        t0 = time.time()
        rec = {"id": atoms.info["id"], "formula": atoms.get_chemical_formula(), "natoms": len(atoms)}
        try:
            with wall_timeout(timeout, atoms.info["id"]):
                rec.update(eos_curve(atoms, calc))
        except Timeout:
            rec["error"] = "timeout"
        except Exception:
            tb = traceback.format_exc()
            rec["error"] = failure(tb, atoms.info["id"])
            rec["detail"] = tb.strip().splitlines()[-1]
        rec["seconds"] = round(time.time() - t0, 1)
        path.write_text(json.dumps(rec))
        print(f"[{label} {index}/{count}] {rec['id']} ({len(atoms)} atoms): {rec['seconds']} s", flush=True)


def run_curves(args: argparse.Namespace) -> None:
    rank, world, device = placement(args.device)
    labels = args.labels or [Path(m).stem for m in args.models]
    k = args.workers
    for model, label in zip(args.models, labels):
        (Path(args.out) / label).mkdir(parents=True, exist_ok=True)
        jobs = [
            (
                model,
                label,
                args.out,
                rank * k + i,
                world * k,
                device,
                args.dtype,
                args.backend,
                args.threads,
                args.timeout,
                args.limit,
            )
            for i in range(k)
        ]
        if k == 1:
            _run_shard(*jobs[0])
        else:
            spawn(_run_shard, jobs)


# ------------------------------------------------------------------ summary
def score(rec: dict, label: str) -> dict:
    """One row per structure: the fit, and the arena's metrics of the curve."""
    from ase import Atoms
    from mlip_arena.flows.eos_bulk import calculate_metrics

    row = {
        "model": label,
        **{
            k: rec.get(k) for k in ("id", "formula", "natoms", "K_GPa", "b1", "steps", "converged", "seconds")
        },
        "error": rec.get("error", ""),
    }
    if rec.get("v0_A3") is not None:
        row["v0_A3_per_atom"] = rec["v0_A3"] / rec["natoms"]
        row["v0_over_input"] = rec["v0_A3"] / rec["v_input_A3"]
    if "energies_eV" in rec and np.isfinite(rec["energies_eV"]).all():
        curve = {"volumes": rec["volumes_A3"], "energies": rec["energies_eV"]}
        metrics = calculate_metrics(curve, rec.get("K_GPa"), Atoms(rec["formula"]), label, rec["id"])
        row.update({m: float(metrics[m]) for m in METRICS})
    return row


def aggregate(rows: list[dict]) -> dict:
    """Per model: how many curves are smooth, the mean and median of each metric (the means
    are dominated by the few curves that break), and bulk moduli against ``REFERENCE``."""
    reference = {r["id"]: r["K_GPa"] for r in rows if r["model"] == REFERENCE and r["K_GPa"] is not None}
    out = {}
    for label in dict.fromkeys(r["model"] for r in rows):
        mine = [r for r in rows if r["model"] == label]
        scored = [r for r in mine if "tortuosity" in r]
        pairs = [
            (r["K_GPa"], reference[r["id"]]) for r in mine if r["K_GPa"] is not None and r["id"] in reference
        ]
        s = {
            "n_structures": len(mine),
            "n_failed": int(sum(bool(r["error"]) for r in mine)),
            "n_without_fit": int(sum(r["K_GPa"] is None for r in mine)),
            "n_unconverged": int(sum(r["converged"] is False for r in mine)),
            "one_well": int(sum(r["energy-diff-flip-times"] == 1 for r in scored)),
            "median_K_GPa": float(np.median([r["K_GPa"] for r in mine if r["K_GPa"] is not None])),
            "mean": {m: float(np.nanmean([r[m] for r in scored])) for m in METRICS},
            "median": {m: float(np.nanmedian([r[m] for r in scored])) for m in METRICS},
        }
        if label != REFERENCE and pairs:
            k, ref = np.array(pairs).T
            s[f"K_against_{REFERENCE}"] = {
                "n": len(k),
                "median_abs_GPa": float(np.median(np.abs(k - ref))),
                "mean_abs_GPa": float(np.mean(np.abs(k - ref))),
            }
        out[label] = s
    return out


def curve_bands(records: list[dict]) -> dict:
    """10th, 50th and 90th percentile of the energy per atom against the middle volume, at
    each of the volumes, over the structures with a finished curve."""
    curves = [
        (np.array(r["energies_eV"]) - r["energies_eV"][N_VOLUMES // 2]) / r["natoms"]
        for r in records
        if "energies_eV" in r and np.isfinite(r["energies_eV"]).all()
    ]
    lo, mid, hi = np.percentile(np.array(curves) * 1e3, [10, 50, 90], axis=0)
    return {
        "p10_meV": lo.round(3).tolist(),
        "p50_meV": mid.round(3).tolist(),
        "p90_meV": hi.round(3).tolist(),
    }


def write_tables(run_dir: Path, labels: list[str], out: Path) -> None:
    rows, bands = [], {}
    for label in labels:
        records = [json.loads(p.read_text()) for p in sorted((run_dir / label).glob("*.json"))]
        rows += [score(r, label) for r in records]
        bands[label] = curve_bands(records)
    out.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "eos_structures.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    summary = {
        "labels": labels,
        "volume_ratios": np.linspace(1 - MAX_VOLUME_STRAIN, 1 + MAX_VOLUME_STRAIN, N_VOLUMES)
        .round(4)
        .tolist(),
        "models": aggregate(rows),
        "bands": bands,
    }
    (out / "eos_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for label, s in summary["models"].items():
        print(
            f"{label}: {s['n_structures']} structures, {s['one_well']} with one well, "
            f"{s['n_failed']} failed, {s['n_without_fit']} without a fit, "
            f"median tortuosity {s['median']['tortuosity']:.4f}",
            flush=True,
        )


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--models", nargs="+", default=[])
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--out", required=True, help="run directory (one folder per label)")
    p.add_argument("--summary", action="store_true", help="score finished curves instead of running")
    p.add_argument("--tables", default="results/eos", help="--summary: where to write the tables")
    p.add_argument("--limit", type=int, default=None, help="only the first N structures (a pilot)")
    p.add_argument("--timeout", type=float, default=3600.0, help="seconds per structure")
    p.add_argument(
        "--workers", type=int, default=1, help="processes per rank, each a share of the structures"
    )
    p.add_argument("--threads", type=int, default=1, help="torch threads per worker")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float64")
    p.add_argument("--backend", default=None, choices=[None, "e3nn", "cueq"])


def run(args: argparse.Namespace) -> None:
    if args.summary:
        write_tables(Path(args.out), args.labels, Path(args.tables))
    else:
        run_curves(args)
