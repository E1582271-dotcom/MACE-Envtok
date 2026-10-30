"""Energy conservation: the drift of the total energy in NVE molecular dynamics.

A potential whose forces are the exact gradient of its energy conserves the total
energy, up to the integrator's error. A drift therefore shows forces that are not
smooth enough for the time step, or single-precision noise, so every run is done in
float32 and float64.

A fixed random sample of the WBM structures of ``bench.eos`` is repeated to at least
64 atoms. Velocities are drawn at twice the target temperature, because about half of
the kinetic energy goes into potential energy in the first steps. The MD is the arena's
task (velocity Verlet, 2 fs, or 0.5 fs with hydrogen). The drift is the slope of a line
through the total energy per atom over the whole run.

    envtok-mace nve --models a.pt --labels ours --out runs/nve --temperatures 300 1000
    envtok-mace nve --summary --out runs/nve --labels ours --tables results/nve
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

from ._runner import Timeout, failure, placement, spawn, wall_timeout
from .eos import structures

N_STRUCTURES = 20
SEED = 0
MIN_ATOMS = 64
TRAJ_INTERVAL = 10  # MD steps between saved frames


def sample(n: int = N_STRUCTURES, seed: int = SEED) -> list:
    """A fixed random sample of the WBM structures, each repeated to at least MIN_ATOMS."""
    all_structures = structures()
    picked = np.random.default_rng(seed).choice(len(all_structures), size=n, replace=False)
    return [supercell(all_structures[i]) for i in sorted(picked)]


def supercell(atoms, min_atoms: int = MIN_ATOMS):
    """The smallest repeat with at least ``min_atoms`` atoms and a roughly cubic shape."""
    widths = 1 / np.linalg.norm(
        atoms.cell.reciprocal(), axis=1
    )  # perpendicular, so skewed cells are big enough
    edge = (atoms.get_volume() / len(atoms) * min_atoms) ** (1 / 3)
    big = atoms.repeat([max(1, int(np.ceil(edge / w))) for w in widths])
    big.info.update(atoms.info)
    return big


def drift(time_ps: np.ndarray, e_per_atom_eV: np.ndarray) -> dict:
    """Slope of the total energy per atom in meV/atom/ps, and the scatter around the line."""
    slope, intercept = np.polyfit(time_ps, e_per_atom_eV, 1)
    residual = e_per_atom_eV - (slope * time_ps + intercept)
    return {
        "drift_meV_per_atom_ps": float(slope * 1e3),
        "scatter_meV_per_atom": float(np.sqrt(np.mean(residual**2)) * 1e3),
    }


def score_trajectory(path: Path, n_steps: int, dt_fs: float) -> dict:
    """Total-energy drift and mean temperature of a finished trajectory."""
    from ase.io import read

    # the arena's task may append one more frame for the very last step
    frames = read(path, index=":")[: n_steps // TRAJ_INTERVAL + 1]
    natoms = len(frames[0])
    e_total = np.array([a.get_potential_energy() + a.get_kinetic_energy() for a in frames]) / natoms
    time_ps = np.arange(len(frames)) * TRAJ_INTERVAL * dt_fs * 1e-3
    second_half = frames[len(frames) // 2 :]
    return {
        "n_frames": len(frames),
        "T_mean_K": float(np.mean([a.get_temperature() for a in second_half])),
        **drift(time_ps, e_total),
    }


def _run_shard(model, label, out, index, count, device, dtype, backend, threads, args) -> None:
    os.environ.setdefault("TQDM_DISABLE", "1")
    import torch
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from mlip_arena.tasks.md import run as md

    from ..eval.ase_calc import EnvTokCalculator

    torch.set_num_threads(threads)
    calc = EnvTokCalculator(model, device=device, dtype=dtype, backend=backend)
    out = Path(out) / label / dtype
    out.mkdir(parents=True, exist_ok=True)
    runs = [(a, t) for a in sample(args["n_structures"]) for t in args["temperatures"]]
    for atoms, temperature in runs[index::count]:
        stem = f"{atoms.info['id']}_{temperature:g}K"
        path = out / f"{stem}.json"
        if path.exists():
            continue
        dt_fs = 0.5 if "H" in atoms.get_chemical_symbols() else 2.0
        n_steps = int(args["total_fs"] / dt_fs)
        rec = {
            "id": atoms.info["id"],
            "formula": atoms.get_chemical_formula(),
            "natoms": len(atoms),
            "temperature_K": temperature,
            "dt_fs": dt_fs,
            "total_fs": args["total_fs"],
            "model": label,
            "dtype": dtype,
        }
        traj = out / f"{stem}.traj"
        moving = atoms.copy()  # the arena's task draws no velocities for NVE
        MaxwellBoltzmannDistribution(moving, temperature_K=2.0 * temperature, rng=np.random.default_rng(SEED))
        t0 = time.time()
        try:
            with wall_timeout(args["timeout"], stem):
                md.fn(
                    atoms=moving,
                    calculator=calc,
                    ensemble="nve",
                    dynamics="velocityverlet",
                    time_step=dt_fs,
                    total_time=args["total_fs"],
                    traj_file=traj,
                    traj_interval=TRAJ_INTERVAL,
                    restart=False,
                )
            rec.update(score_trajectory(traj, n_steps, dt_fs))
        except Timeout:
            rec["error"] = "timeout"
        except Exception:
            tb = traceback.format_exc()
            rec["error"] = failure(tb, stem)
            rec["detail"] = tb.strip().splitlines()[-1]
        rec["seconds"] = round(time.time() - t0, 1)
        rec["ms_per_step"] = round(rec["seconds"] / n_steps * 1e3, 2)
        path.write_text(json.dumps(rec))
        traj.unlink(missing_ok=True)
        print(
            f"[{label} {dtype} {index}/{count}] {stem} ({len(atoms)} atoms): {rec['seconds']} s", flush=True
        )


def run_md(args: argparse.Namespace) -> None:
    rank, world, device = placement(args.device)
    labels = args.labels or [Path(m).stem for m in args.models]
    k = args.workers
    opts = {
        "temperatures": args.temperatures,
        "total_fs": args.total_ps * 1e3,
        "n_structures": args.n_structures,
        "timeout": args.timeout,
    }
    for model, label in zip(args.models, labels):
        for dtype in args.dtypes:
            jobs = [
                (
                    model,
                    label,
                    args.out,
                    rank * k + i,
                    world * k,
                    device,
                    dtype,
                    args.backend,
                    args.threads,
                    opts,
                )
                for i in range(k)
            ]
            if k == 1:
                _run_shard(*jobs[0])
            else:
                spawn(_run_shard, jobs)


# ------------------------------------------------------------------ summary
def aggregate(rows: list[dict]) -> list[dict]:
    """One row per model, precision and temperature. Magnitudes, because a drift of either
    sign is a failure to conserve; failed runs are counted and left out of the statistics."""
    out = []
    keys = dict.fromkeys((r["model"], r["dtype"], r["temperature_K"]) for r in rows)
    for model, dtype, temperature in keys:
        mine = [
            r for r in rows if (r["model"], r["dtype"], r["temperature_K"]) == (model, dtype, temperature)
        ]
        ok = [r for r in mine if "error" not in r]
        size = np.abs([r["drift_meV_per_atom_ps"] for r in ok]) if ok else np.array([np.nan])
        out.append(
            {
                "model": model,
                "dtype": dtype,
                "temperature_K": temperature,
                "n_runs": len(mine),
                "n_failed": len(mine) - len(ok),
                "median_abs_drift_meV_per_atom_ps": float(np.median(size)),
                "max_abs_drift_meV_per_atom_ps": float(np.max(size)),
                "median_T_K": float(np.median([r["T_mean_K"] for r in ok])) if ok else float("nan"),
            }
        )
    return out


def write_tables(run_dir: Path, labels: list[str], out: Path) -> None:
    rows = [json.loads(p.read_text()) for label in labels for p in sorted((run_dir / label).glob("*/*.json"))]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "nve_runs.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    summary = aggregate(rows)
    (out / "nve_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for s in summary:
        print(
            f"{s['model']} {s['dtype']} {s['temperature_K']:g} K: median |drift| "
            f"{s['median_abs_drift_meV_per_atom_ps']:.3g} meV/atom/ps, {s['n_failed']}/{s['n_runs']} failed",
            flush=True,
        )


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--models", nargs="+", default=[])
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--out", required=True, help="run directory (one folder per label)")
    p.add_argument("--summary", action="store_true", help="score finished runs instead of running")
    p.add_argument("--tables", default="results/nve", help="--summary: where to write the tables")
    p.add_argument("--temperatures", nargs="+", type=float, default=[300.0, 1000.0])
    p.add_argument("--dtypes", nargs="+", default=["float32", "float64"])
    p.add_argument("--total-ps", type=float, default=10.0)
    p.add_argument("--n-structures", type=int, default=N_STRUCTURES)
    p.add_argument("--timeout", type=float, default=7200.0, help="seconds per run")
    p.add_argument("--workers", type=int, default=1, help="processes per rank, each a share of the runs")
    p.add_argument("--threads", type=int, default=1, help="torch threads per worker")
    p.add_argument("--device", default="cpu")
    p.add_argument("--backend", default=None, choices=[None, "e3nn", "cueq"])


def run(args: argparse.Namespace) -> None:
    if args.summary:
        write_tables(Path(args.out), args.labels, Path(args.tables))
    else:
        run_md(args)
