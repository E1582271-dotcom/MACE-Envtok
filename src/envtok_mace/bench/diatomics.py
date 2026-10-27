"""Homonuclear diatomic energy curves for every element of the model (MLIP Arena task).

For each of the 68 elements, two atoms in vacuum are pulled apart from 0.9 x the
covalent radius to 3.1 x the van der Waals radius in 0.01 A steps. This is the
arena's own task, called without prefect. The arena's smoothness metrics are then
computed from each curve: sign flips and jumps of force and energy, total variation,
tortuosity, and the deviation of F from -dE/dr. A physically sensible pair curve has
one well, repulsion at short range, and no flips or jumps.

The arena puts the energy zero at the largest distance, which for light elements
lies inside the 6 A cutoff. Each curve is therefore also reported relative to two
isolated atoms (E - 2 E_atom), computed with the same model.

    envtok-mace diatomics --models a.pt b.model --labels ours authors_envtok --out runs/diatomics
    envtok-mace diatomics --summary --labels ours authors_envtok --out runs/diatomics \\
        --tables results/diatomics
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

from ..data.constants import Z_TABLE
from ._runner import placement, spawn

# curves kept in results/ for the figure: elements that make up battery materials
CURVE_ELEMENTS = ("H", "Li", "C", "N", "O", "F", "Na", "Mg", "Si", "P", "S", "Cl", "Mn", "Fe", "Co", "Ni")
CURVE_STRIDE = 5  # keep every 5th point (0.05 A) of a curve
# No homonuclear bond is deeper than N2's 9.8 eV; a deeper well is an artefact.
MAX_PHYSICAL_WELL_EV = 12.0
METRICS = (
    "force-flip-times",
    "energy-diff-flip-times",
    "force-jump",
    "energy-jump",
    "tortuosity",
    "conservation-deviation",
    "spearman-repulsion-energy",
    "spearman-attraction-energy",
)


def symbols() -> list[str]:
    from ase.data import chemical_symbols

    return [chemical_symbols[z] for z in sorted(Z_TABLE)]


def _run_shard(model, label, out, index, count, device, dtype, backend, threads) -> None:
    os.environ.setdefault("TQDM_DISABLE", "1")
    import torch
    from ase import Atoms
    from mlip_arena.flows.diatomics import homonuclear_diatomic

    from ..eval.ase_calc import EnvTokCalculator

    torch.set_num_threads(threads)
    calc = EnvTokCalculator(model, device=device, dtype=dtype, backend=backend)
    out = Path(out) / label
    for symbol in symbols()[index::count]:
        if (out / f"{symbol}.json").exists():
            continue
        t0 = time.time()
        homonuclear_diatomic.fn(symbol, calc, out)  # resumes from a partial curve
        atom = Atoms(symbol, positions=[[0.0, 0.0, 0.0]], cell=[20.0, 20.0, 20.0], pbc=False)
        atom.calc = calc
        e_atom = float(atom.get_potential_energy())
        (out / f"{symbol}.json").write_text(json.dumps({"symbol": symbol, "e_atom": e_atom}))
        print(f"[{label} {index}/{count}] {symbol}: {time.time() - t0:.0f} s", flush=True)


def run_curves(args: argparse.Namespace) -> None:
    rank, world, device = placement(args.device)
    labels = args.labels or [Path(m).stem for m in args.models]
    k = args.workers
    for model, label in zip(args.models, labels):
        (Path(args.out) / label).mkdir(parents=True, exist_ok=True)
        jobs = [
            (model, label, args.out, rank * k + i, world * k, device, args.dtype, args.backend, args.threads)
            for i in range(k)
        ]
        if k == 1:
            _run_shard(*jobs[0])
        else:
            spawn(_run_shard, jobs)


# ------------------------------------------------------------------ summary
def summarize(run_dir: Path, label: str) -> tuple[list[dict], dict]:
    """Per-element rows (arena metrics plus the curve against two isolated atoms) and
    the curves of ``CURVE_ELEMENTS``."""
    from mlip_arena.flows.diatomics import analyze

    d = Path(run_dir) / label
    df = analyze(d)
    rows, curves = [], {}
    for rec in df.to_dict("records"):
        symbol = rec["name"][: len(rec["name"]) // 2]
        meta = d / f"{symbol}.json"
        if not meta.exists():  # curve not finished
            continue
        e_atom = json.loads(meta.read_text())["e_atom"]
        r, e, f = (np.asarray(rec[k], dtype=float) for k in ("R", "E", "F"))
        order = np.argsort(r)
        r, e, f = r[order], e[order] - 2 * e_atom, f[order]
        i_min = int(np.argmin(e))
        rows.append(
            {
                "model": label,
                "symbol": symbol,
                "n_points": int(len(r)),
                "r_lo_A": float(r[0]),
                "r_hi_A": float(r[-1]),
                "well_depth_eV": float(e[i_min]),
                "r_at_min_A": float(r[i_min]),
                "e_far_eV": float(e[-1]),  # 0 when the far end is beyond the cutoff
                "e_close_eV": float(e[0]),
                **{m: float(rec[m]) for m in METRICS},
            }
        )
        if symbol in CURVE_ELEMENTS:
            curves[symbol] = {
                "r": np.round(r[::CURVE_STRIDE], 4).tolist(),
                "e": np.round(e[::CURVE_STRIDE], 5).tolist(),
                "f": np.round(f[::CURVE_STRIDE], 5).tolist(),
            }
    return rows, curves


def aggregate(rows: list[dict]) -> dict:
    """Per model: how many curves have one well, are repulsive at close range or have an
    unphysically deep well, and the mean and median of each arena metric over elements
    (the means are dominated by the few curves that blow up)."""
    out = {}
    for label in dict.fromkeys(r["model"] for r in rows):
        mine = [r for r in rows if r["model"] == label]
        out[label] = {
            "n_elements": len(mine),
            "single_well": int(sum(r["energy-diff-flip-times"] == 1 for r in mine)),
            "repulsive_at_close_range": int(sum(r["e_close_eV"] > 0 for r in mine)),
            "too_deep": [r["symbol"] for r in mine if r["well_depth_eV"] < -MAX_PHYSICAL_WELL_EV],
            "mean": {m: float(np.nanmean([r[m] for r in mine])) for m in METRICS},
            "median": {m: float(np.nanmedian([r[m] for r in mine])) for m in METRICS},
        }
    return out


def write_tables(run_dir: Path, labels: list[str], out: Path) -> None:
    rows, curves = [], {}
    for label in labels:
        r, c = summarize(run_dir, label)
        rows += r
        curves[label] = c
    out.mkdir(parents=True, exist_ok=True)
    with (out / "diatomics_elements.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    summary = {"labels": labels, "models": aggregate(rows)}
    (out / "diatomics_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out / "diatomics_curves.json").write_text(json.dumps(curves, separators=(",", ":")) + "\n")
    for label, s in summary["models"].items():
        print(
            f"{label}: {s['n_elements']} elements, {s['single_well']} with one well, "
            f"{s['repulsive_at_close_range']} repulsive at 0.9 x covalent radius, "
            f"wells deeper than {MAX_PHYSICAL_WELL_EV:g} eV: {', '.join(s['too_deep']) or 'none'}",
            flush=True,
        )


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--models", nargs="+", default=[])
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--out", required=True, help="run directory (one folder per label)")
    p.add_argument("--summary", action="store_true", help="score finished curves instead of running")
    p.add_argument("--tables", default="results/diatomics", help="--summary: where to write the tables")
    p.add_argument("--workers", type=int, default=1, help="processes per rank, each a share of the elements")
    p.add_argument("--threads", type=int, default=1, help="torch threads per worker")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float64")
    p.add_argument("--backend", default=None, choices=[None, "e3nn", "cueq"])


def run(args: argparse.Namespace) -> None:
    if args.summary:
        write_tables(Path(args.out), args.labels, Path(args.tables))
    else:
        run_curves(args)
