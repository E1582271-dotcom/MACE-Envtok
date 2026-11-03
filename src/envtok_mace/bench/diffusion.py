"""Li-ion diffusion in solid electrolytes: NVT molecular dynamics and an Arrhenius fit.

Five electrolytes, each in the ordered structure the Materials Project computed:
LGPS, Li7P3S11, beta-Li3PS4, the argyrodite Li6PS5Cl and Li3InCl6. Li3YCl6 is not
among them: its Y sites are partly occupied, so neither the Materials Project nor
Alexandria has it in the experimental structure. Each cell is repeated to at least
200 atoms and kept at its PBE volume, the same for every model, as in AIMD studies.

The MD runs a Nose-Hoover chain at 600, 700, 800 and 900 K with a 2 fs step, in
float32: 10 ps to equilibrate, then 100 ps whose positions are saved every 50 fs.
D is the slope of the Li mean squared displacement, averaged over time origins,
from 5 to 50 ps of lag, after removing the drift of the framework's centre; its
spread comes from four 25 ps blocks. A line through ln D against 1/T gives the
activation energy and D at 300 K, and the Nernst-Einstein relation with a Haven
ratio of 1 (assumed) gives the conductivity.

A run is flagged rather than fitted when Li barely moves (MSD below 3 A^2 at the
end of the fit window) or when the framework itself flows (its MSD above 1.5 A^2).
A run stops as diverged when two atoms come closer than 0.6 A or the temperature
passes twice the target.

    envtok-mace diffusion --fetch structures.extxyz        # needs MP_API_KEY
    envtok-mace diffusion --models a.pt --labels ours --structures structures.extxyz \\
        --out runs/diffusion --device cuda
    envtok-mace diffusion --summary --labels ours --structures structures.extxyz \\
        --out runs/diffusion --tables results/diffusion
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
from .nve import supercell

# name -> Materials Project id and the space group of that ordered structure
MATERIALS = {
    "LGPS": ("mp-aaabnpue", "P4_2mc"),
    "Li7P3S11": ("mp-aaabkngx", "P-1"),
    "beta-Li3PS4": ("mp-aaacebzb", "Pnma"),
    "Li6PS5Cl": ("mp-aaacebzk", "F-43m"),
    "Li3InCl6": ("mp-aaabmmef", "C2"),
}
MOBILE = "Li"
MIN_ATOMS = 200
TEMPERATURES_K = (600.0, 700.0, 800.0, 900.0)
DT_FS = 2.0
EQUILIBRATE_PS = 10.0
PRODUCTION_PS = 100.0
SAVE_EVERY = 25  # MD steps between saved frames: 50 fs
TDAMP_FS = 100.0
SEED = 0
FIT = (0.05, 0.5)  # lag window of the MSD fit, as fractions of the production run
N_BLOCKS = 4
MIN_MSD_A2 = 3.0
FRAMEWORK_MSD_A2 = 1.5
CLOSE_A = 0.6
HOT = 2.0
ROOM_K = 300.0
KB_EV = 8.617333262e-5
KB_J = 1.380649e-23
E_C = 1.602176634e-19
EV_A3_TO_GPA = 160.21766208


# --------------------------------------------------------------------- structures
def fetch(path: Path) -> None:
    """Write the five structures from the Materials Project API (needs MP_API_KEY)."""
    import urllib.parse
    import urllib.request

    from ase.io import write
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    def get(endpoint: str, **query) -> dict:
        req = urllib.request.Request(
            f"https://api.materialsproject.org/{endpoint}?{urllib.parse.urlencode(query)}",
            headers={"X-API-KEY": os.environ["MP_API_KEY"], "User-Agent": "envtok-mace"},
        )
        with urllib.request.urlopen(req, timeout=60) as fh:
            return json.load(fh)

    version = get("heartbeat").get("db_version")
    out = []
    for name, (material_id, spacegroup) in MATERIALS.items():
        (doc,) = get(
            "materials/summary/",
            material_ids=material_id,
            _fields="material_id,symmetry,energy_above_hull,structure",
        )["data"]
        if doc["symmetry"]["symbol"] != spacegroup:
            raise ValueError(f"{name}: {material_id} is {doc['symmetry']['symbol']}, not {spacegroup}")
        atoms = AseAtomsAdaptor.get_atoms(Structure.from_dict(doc["structure"]))
        atoms.info.update(
            name=name,
            material_id=material_id,
            spacegroup=spacegroup,
            e_above_hull_meV=round(doc["energy_above_hull"] * 1e3, 1),
            mp_db_version=version,
        )
        out.append(atoms)
    write(path, out, format="extxyz")


def read_structures(path: str, names: list[str] | None = None) -> list:
    """The structures, each repeated to at least MIN_ATOMS atoms."""
    from ase.io import read

    return [
        supercell(a, MIN_ATOMS)
        for a in read(path, format="extxyz", index=":")
        if names is None or a.info["name"] in names
    ]


# ------------------------------------------------------------------------- MD
class Diverged(Exception):
    pass


def _check(atoms, target_K: float) -> None:
    from matscipy.neighbours import neighbour_list

    t = atoms.get_temperature()
    if not np.isfinite(atoms.get_potential_energy()) or t > HOT * target_K:
        raise Diverged(f"temperature {t:.0f} K")
    if len(neighbour_list("i", atoms, CLOSE_A)):
        raise Diverged(f"atoms closer than {CLOSE_A} A")


def molecular_dynamics(atoms, calc, temperature_K: float, equilibrate_ps: float, production_ps: float):
    """NVT run; returns the production frames: positions, temperature and potential energy."""
    from ase import units
    from ase.md.nose_hoover_chain import NoseHooverChainNVT
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    atoms = atoms.copy()
    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=np.random.default_rng(SEED))
    Stationary(atoms)
    dyn = NoseHooverChainNVT(
        atoms, timestep=DT_FS * units.fs, temperature_K=temperature_K, tdamp=TDAMP_FS * units.fs
    )
    dyn.attach(lambda: _check(atoms, temperature_K), interval=SAVE_EVERY)
    dyn.run(_steps(equilibrate_ps))
    frames = {"positions": [], "temperature_K": [], "epot_eV": []}

    def save():
        frames["positions"].append(atoms.get_positions().astype(np.float32))
        frames["temperature_K"].append(atoms.get_temperature())
        frames["epot_eV"].append(atoms.get_potential_energy())

    # ASE counts steps over both runs and calls observers only at multiples of the interval,
    # so the first frame of the production run is saved here
    save()
    dyn.attach(save, interval=SAVE_EVERY)
    dyn.run(_steps(production_ps))
    return {k: np.asarray(v) for k, v in frames.items()}


def _steps(ps: float) -> int:
    """MD steps in ``ps``, a whole number of saving intervals."""
    return SAVE_EVERY * round(ps * 1e3 / DT_FS / SAVE_EVERY)


# ------------------------------------------------------------------- analysis
def unwrap(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Continuous trajectories: each step taken as its shortest periodic image."""
    frac = np.asarray(positions, dtype=np.float64) @ np.linalg.inv(cell)
    step = np.diff(frac, axis=0)
    step -= np.round(step)
    frac = np.concatenate([frac[:1], frac[:1] + np.cumsum(step, axis=0)])
    return frac @ cell


def msd(unwrapped: np.ndarray, mobile: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean squared displacement of the mobile and of the framework atoms against lag in
    frames, averaged over time origins, after removing the framework's mean drift."""
    shifted = unwrapped - unwrapped[0]
    if (~mobile).any():
        shifted = shifted - shifted[:, ~mobile].mean(axis=1, keepdims=True)
    out = np.zeros((2, max_lag + 1))
    for lag in range(1, max_lag + 1):
        d2 = ((shifted[lag:] - shifted[:-lag]) ** 2).sum(axis=-1).mean(axis=0)
        out[0, lag] = d2[mobile].mean()
        out[1, lag] = d2[~mobile].mean() if (~mobile).any() else 0.0
    return out[0], out[1]


def diffusion_coefficient(unwrapped: np.ndarray, mobile: np.ndarray, frame_ps: float) -> dict:
    """D in cm^2/s from the slope of the MSD in the fit window."""
    n = len(unwrapped) - 1
    lo, hi = round(FIT[0] * n), round(FIT[1] * n)
    m, f = msd(unwrapped, mobile, hi)
    lags = np.arange(lo, hi + 1) * frame_ps
    slope = np.polyfit(lags, m[lo : hi + 1], 1)[0]  # A^2/ps
    return {"D_cm2_s": slope / 6 * 1e-4, "msd_end_A2": float(m[hi]), "framework_msd_end_A2": float(f[hi])}


def score_run(positions: np.ndarray, numbers: np.ndarray, cell: np.ndarray, frame_ps: float) -> dict:
    from ase.data import atomic_numbers

    mobile = np.asarray(numbers) == atomic_numbers[MOBILE]
    u = unwrap(positions, cell)
    whole = diffusion_coefficient(u, mobile, frame_ps)
    size = (len(u) - 1) // N_BLOCKS
    blocks = [
        diffusion_coefficient(u[i * size : (i + 1) * size + 1], mobile, frame_ps) for i in range(N_BLOCKS)
    ]
    d_blocks = np.array([b["D_cm2_s"] for b in blocks])
    return {
        **whole,
        "D_err_cm2_s": float(d_blocks.std(ddof=1) / np.sqrt(N_BLOCKS)),
        "diffusive": whole["msd_end_A2"] >= MIN_MSD_A2,
        "framework_flows": whole["framework_msd_end_A2"] > FRAMEWORK_MSD_A2,
    }


def arrhenius(temperatures_K, d_cm2_s) -> dict:
    """Activation energy (eV) and its standard error from ln D = ln D0 - Ea / kT, and D at 300 K."""
    x = 1 / (KB_EV * np.asarray(temperatures_K, dtype=float))
    y = np.log(np.asarray(d_cm2_s, dtype=float))
    a = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(a, y, rcond=None)
    dof = len(x) - 2
    err = np.nan
    if dof > 0:
        s2 = ((y - a @ (slope, intercept)) ** 2).sum() / dof
        err = float(np.sqrt(s2 * np.linalg.inv(a.T @ a)[0, 0]))
    return {
        "Ea_eV": float(-slope),
        "Ea_err_eV": err,
        "D300_cm2_s": float(np.exp(intercept + slope / (KB_EV * ROOM_K))),
    }


def conductivity_mS_cm(n_mobile: int, volume_A3: float, d_cm2_s: float, temperature_K: float) -> float:
    """Nernst-Einstein with a Haven ratio of 1."""
    n = n_mobile / (volume_A3 * 1e-30)
    return n * E_C**2 * (d_cm2_s * 1e-4) / (KB_J * temperature_K) * 10  # S/m -> mS/cm


# ------------------------------------------------------------------------ runs
def _pressure_GPa(atoms, calc) -> float:
    """Static pressure of the starting cell, which every model is held at."""
    calc.stress = True
    probe = atoms.copy()
    probe.calc = calc
    p = -probe.get_stress()[:3].mean() * EV_A3_TO_GPA
    calc.stress = False
    calc.reset()
    return float(p)


def _run_shard(models, labels, out, structures, names, index, count, device, dtype, backend, threads, opts):
    """This worker's share of every model, material and temperature, so that the long runs
    spread evenly; each model is loaded when the worker first needs it."""
    os.environ.setdefault("TQDM_DISABLE", "1")
    import torch

    from ..eval.ase_calc import EnvTokCalculator

    torch.set_num_threads(threads)
    calcs = {}
    runs = [
        (model, label, atoms, t)
        for atoms in read_structures(structures, names)
        for t in opts["temperatures"]
        for model, label in zip(models, labels)
    ]
    for model, label, atoms, temperature in runs[index::count]:
        stem = f"{atoms.info['name']}_{temperature:g}K"
        path = Path(out) / label / f"{stem}.json"
        # running out of GPU memory depends on how many workers share the GPU, so it is retried
        if path.exists() and json.loads(path.read_text()).get("error") != "oom":
            continue
        if label not in calcs:
            calcs[label] = EnvTokCalculator(model, device=device, dtype=dtype, backend=backend, stress=False)
        calc = calcs[label]
        rec = {
            "id": atoms.info["name"],
            "material_id": atoms.info["material_id"],
            "formula": atoms.get_chemical_formula(),
            "model": label,
            "dtype": dtype,
            "temperature_K": temperature,
            "natoms": len(atoms),
            "n_mobile": atoms.get_chemical_symbols().count(MOBILE),
            "volume_A3": float(atoms.get_volume()),
            "equilibrate_ps": opts["equilibrate_ps"],
            "production_ps": opts["production_ps"],
        }
        t0 = time.time()
        try:
            with wall_timeout(opts["timeout"], stem):
                rec["pressure_GPa"] = _pressure_GPa(atoms, calc)
                t1 = time.time()
                frames = molecular_dynamics(
                    atoms, calc, temperature, opts["equilibrate_ps"], opts["production_ps"]
                )
                n_steps = _steps(opts["equilibrate_ps"]) + _steps(opts["production_ps"])
                rec["ms_per_step"] = round((time.time() - t1) / n_steps * 1e3, 2)
            frame_ps = SAVE_EVERY * DT_FS * 1e-3
            cell = np.asarray(atoms.cell)
            np.savez_compressed(
                path.with_suffix(".npz"), numbers=atoms.numbers, cell=cell, frame_ps=frame_ps, **frames
            )
            rec["T_mean_K"] = float(frames["temperature_K"].mean())
            rec.update(score_run(frames["positions"], atoms.numbers, cell, frame_ps))
        except Timeout:
            rec["error"] = "timeout"
        except Diverged as e:
            rec["error"] = "diverged"
            rec["detail"] = str(e)
        except Exception:
            tb = traceback.format_exc()
            rec["error"] = failure(tb, stem)
            rec["detail"] = tb.strip().splitlines()[-1]
        rec["seconds"] = round(time.time() - t0, 1)
        path.write_text(json.dumps(rec))
        print(f"[{label} {index}/{count}] {stem} ({len(atoms)} atoms): {rec['seconds']} s", flush=True)


def run_md(args: argparse.Namespace) -> None:
    rank, world, device = placement(args.device)
    labels = args.labels or [Path(m).stem for m in args.models]
    k = args.workers
    opts = {
        "temperatures": args.temperatures,
        "equilibrate_ps": args.equilibrate_ps,
        "production_ps": args.production_ps,
        "timeout": args.timeout,
    }
    for label in labels:
        (Path(args.out) / label).mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            args.models,
            labels,
            args.out,
            args.structures,
            args.ids,
            rank * k + i,
            world * k,
            device,
            args.dtype,
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


# --------------------------------------------------------------------- summary
def aggregate(rows: list[dict]) -> list[dict]:
    """One row per model and material: the Arrhenius fit over the usable temperatures."""
    out = []
    for model, name in dict.fromkeys((r["model"], r["id"]) for r in rows):
        mine = sorted(
            (r for r in rows if (r["model"], r["id"]) == (model, name)), key=lambda r: r["temperature_K"]
        )
        usable = [r for r in mine if "error" not in r and r["diffusive"] and not r["framework_flows"]]
        row = {
            "model": model,
            "id": name,
            "n_runs": len(mine),
            "n_failed": sum("error" in r for r in mine),
            "n_not_diffusive": sum("error" not in r and not r["diffusive"] for r in mine),
            "n_framework_flows": sum("error" not in r and r["framework_flows"] for r in mine),
            "temperatures_fitted_K": [r["temperature_K"] for r in usable],
            "Ea_eV": None,
            "Ea_err_eV": None,
            "D300_cm2_s": None,
            "sigma300_mS_cm": None,
        }
        if len(usable) >= 3:
            fit = arrhenius([r["temperature_K"] for r in usable], [r["D_cm2_s"] for r in usable])
            row.update(fit)
            row["sigma300_mS_cm"] = conductivity_mS_cm(
                usable[0]["n_mobile"], usable[0]["volume_A3"], fit["D300_cm2_s"], ROOM_K
            )
        out.append(row)
    return out


def write_tables(run_dir: Path, labels: list[str], out: Path) -> None:
    rows = [json.loads(p.read_text()) for label in labels for p in sorted((run_dir / label).glob("*.json"))]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "diffusion_runs.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    summary = aggregate(rows)
    (out / "diffusion_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for s in summary:
        ea = "n/a" if s["Ea_eV"] is None else f"{s['Ea_eV']:.3f} eV"
        sigma = "n/a" if s["sigma300_mS_cm"] is None else f"{s['sigma300_mS_cm']:.3g} mS/cm"
        print(f"{s['model']} {s['id']}: Ea {ea}, sigma(300 K) {sigma}, {s['n_failed']} failed", flush=True)


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--fetch", default=None, help="write the structures from the Materials Project here")
    p.add_argument("--models", nargs="+", default=[])
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--structures", default=None, help="extxyz written by --fetch")
    p.add_argument("--out", default=None, help="run directory (one folder per label)")
    p.add_argument("--summary", action="store_true", help="tabulate finished runs instead of running")
    p.add_argument("--tables", default="results/diffusion", help="--summary: where to write the tables")
    p.add_argument("--ids", nargs="+", default=None, choices=sorted(MATERIALS), help="only these materials")
    p.add_argument("--temperatures", nargs="+", type=float, default=list(TEMPERATURES_K))
    p.add_argument("--equilibrate-ps", type=float, default=EQUILIBRATE_PS)
    p.add_argument("--production-ps", type=float, default=PRODUCTION_PS)
    p.add_argument("--timeout", type=float, default=6 * 3600.0, help="seconds per run")
    p.add_argument("--workers", type=int, default=1, help="processes per rank, each a share of the runs")
    p.add_argument("--threads", type=int, default=1, help="torch threads per worker")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--backend", default=None, choices=[None, "e3nn", "cueq"])


def run(args: argparse.Namespace) -> None:
    if args.fetch:
        fetch(Path(args.fetch))
    elif args.summary:
        write_tables(Path(args.out), args.labels, Path(args.tables))
    else:
        run_md(args)
