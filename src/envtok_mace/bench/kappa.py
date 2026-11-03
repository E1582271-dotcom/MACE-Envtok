"""Phonons and lattice thermal conductivity of the 103 phononDB-PBE binaries.

The protocol and the DFT reference are Matbench Discovery's ("phonondb-v1"):

- relax with FIRE and a FrechetCellFilter without tilt, keeping the symmetry, to 1e-4 eV/A
  in at most 300 steps;
- finite displacements of 0.01 A for the second- and third-order force constants;
- the Wigner transport equation at 300 K, on the reference's q mesh.

A structure whose relaxation changes its space group, or that has imaginary modes, gets no
conductivity and the maximum error of 2, as upstream.

The reference holds frequencies as well, so the same run also gives the harmonic properties
of each structure: the highest frequency, and the heat capacity, free energy and entropy.

The work runs in two stages because they want different hardware. ``--stage forces`` needs
the model, so it runs on a GPU. ``--stage conductivity`` needs only those forces and many
CPU threads, so the GPU is not held while phono3py solves the transport equation.

    envtok-mace kappa --stage forces --models a.pt b.model --labels ours x --out runs/kappa \
        --structures phonondb.extxyz --device cuda --workers 4
    envtok-mace kappa --stage conductivity --labels ours x --out runs/kappa \
        --structures phonondb.extxyz --workers 4 --threads 16
    envtok-mace kappa --summary --labels ours x --out runs/kappa \
        --reference kappas.json.gz --tables results/kappa
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

from ._runner import Timeout, failure, placement, spawn, wall_timeout

DISPLACEMENT_A = 0.01
TEMPERATURE_K = 300.0
SYMPREC = 1e-5
RELAX_SYMPREC = 0.01
FORCE_MAX = 1e-4
MAX_STEPS = 300
NO_TILT = [True, True, True, False, False, False]
GAMMA_ACOUSTIC_THZ = -0.01  # numerical noise in the acoustic modes at q = 0 is not an instability
MAX_ERROR = 2.0

KB_J_PER_K = 1.380649e-23
H_J_S = 6.62607015e-34
AVOGADRO = 6.02214076e23


def read_structures(path: str, ids: list[str] | None = None) -> list:
    from ase.io import read

    atoms = read(path, format="extxyz", index=":")
    return [a for a in atoms if ids is None or a.info["material_id"] in ids]


# --------------------------------------------------------------------- harmonic
def harmonic(
    frequencies_thz: np.ndarray, weights: np.ndarray | None = None, t: float = TEMPERATURE_K
) -> dict:
    """Highest frequency, and per-atom heat capacity (J/K/mol), free energy (kJ/mol) and entropy
    (J/K/mol) at ``t``, from frequencies of shape (q points, bands). Modes at or below zero
    carry no thermal weight, as in phonopy; a dynamically stable structure has none but the
    acoustic modes at q = 0."""
    f = np.asarray(frequencies_thz, dtype=float)
    w = np.ones(len(f)) if weights is None else np.asarray(weights, dtype=float)
    w = np.broadcast_to(w[:, None], f.shape) / (w.sum() * f.shape[1] / 3)  # sums to 3: per atom
    real = f > 1e-2
    x = H_J_S * f[real] * 1e12 / (KB_J_PER_K * t)
    w = w[real]
    r = KB_J_PER_K * AVOGADRO
    return {
        "omega_max_THz": float(f.max()),
        "cv_J_per_K_mol": float(np.sum(w * r * x**2 * np.exp(x) / np.expm1(x) ** 2)),
        "f_kJ_per_mol": float(np.sum(w * r * t * (x / 2 + np.log1p(-np.exp(-x)))) / 1e3),
        "s_J_per_K_mol": float(np.sum(w * r * (x / np.expm1(x) - np.log1p(-np.exp(-x))))),
    }


def has_imaginary_modes(frequencies_thz: np.ndarray) -> bool:
    """Any negative mode, allowing numerical noise in the acoustic modes at q = 0, which comes first."""
    f = np.asarray(frequencies_thz)
    return bool(np.any(f[0, :3] < GAMMA_ACOUSTIC_THZ) or np.any(f[0, 3:] < 0) or np.any(f[1:] < 0))


# ----------------------------------------------------------------------- forces
def space_group(atoms, symprec: float = SYMPREC) -> int:
    import spglib

    cell = (np.asarray(atoms.cell), atoms.get_scaled_positions(), atoms.numbers)
    return int(spglib.get_symmetry_dataset(cell, symprec=symprec).number)


def relax(atoms, calc):
    """The relaxed structure and the number of steps, keeping the symmetry and the cell's angles."""
    from ase.constraints import FixSymmetry
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE

    atoms = atoms.copy()
    atoms.calc = calc
    atoms.set_constraint(FixSymmetry(atoms, symprec=RELAX_SYMPREC))
    optimizer = FIRE(FrechetCellFilter(atoms, mask=NO_TILT), logfile=None)
    optimizer.run(fmax=FORCE_MAX, steps=MAX_STEPS)
    steps = int(optimizer.get_number_of_steps())
    atoms.calc = None
    atoms.constraints = None
    return atoms, steps


def init_phono3py(atoms, info: dict):
    from phono3py import Phono3py
    from phonopy.structure.atoms import PhonopyAtoms

    unitcell = PhonopyAtoms(
        atoms.get_chemical_symbols(), cell=np.asarray(atoms.cell), positions=atoms.positions
    )
    ph3 = Phono3py(
        unitcell=unitcell,
        supercell_matrix=np.asarray(info["fc3_supercell"]),
        phonon_supercell_matrix=np.asarray(info["fc2_supercell"]),
        primitive_matrix="auto",
        symprec=SYMPREC,
    )
    ph3.mesh_numbers = tuple(int(n) for n in np.asarray(info["q_point_mesh"]))
    ph3.generate_displacements(distance=DISPLACEMENT_A, is_plusminus="auto")
    return ph3


def _force_set(supercells, calc) -> np.ndarray:
    """Forces on each displaced supercell; phonopy leaves None for displacements it does not need."""
    from ase import Atoms

    forces = []
    for cell in supercells:
        if cell is None:
            forces.append(None)
            continue
        atoms = Atoms(cell.symbols, cell=cell.cell, scaled_positions=cell.scaled_positions, pbc=True)
        atoms.calc = calc
        forces.append(atoms.get_forces())
    shape = next(f.shape for f in forces if f is not None)
    return np.array([np.zeros(shape) if f is None else f for f in forces])


def forces_for_structure(atoms, calc) -> tuple[dict, dict]:
    """Relax, then the FC2 forces and harmonic properties, then the FC3 forces unless the
    structure has already failed. Returns the record and the arrays to save for the next stage."""
    rec = {"init_spg": space_group(atoms)}
    t0 = time.time()
    relaxed, rec["relax_steps"] = relax(atoms, calc)
    rec["relax_s"] = round(time.time() - t0, 1)
    rec["final_spg"] = space_group(relaxed)
    rec["broken_symmetry"] = rec["final_spg"] != rec["init_spg"]
    rec["volume_A3_per_atom"] = relaxed.get_volume() / len(relaxed)

    ph3 = init_phono3py(relaxed, atoms.info)
    t0 = time.time()
    fc2_forces = _force_set(ph3.phonon_supercells_with_displacements, calc)
    rec["fc2_s"] = round(time.time() - t0, 1)
    rec["fc2_supercells"], rec["fc2_atoms"] = len(fc2_forces), fc2_forces.shape[1]
    ph3.phonon_forces = fc2_forces
    ph3.produce_fc2(symmetrize_fc2=True)
    ph3.init_phph_interaction(symmetrize_fc3q=False)
    ph3.run_phonon_solver()
    frequencies = ph3.get_phonon_data()[0][ph3.grid.grg2bzg]  # every point of the grid once, q = 0 first
    rec["imaginary"] = has_imaginary_modes(frequencies)
    rec.update(harmonic(frequencies))

    arrays = {
        "cell": np.asarray(relaxed.cell),
        "positions": relaxed.positions,
        "numbers": relaxed.numbers,
        "fc2_forces": fc2_forces,
    }
    rec["conductivity_skipped"] = rec["broken_symmetry"] or rec["imaginary"]
    if not rec["conductivity_skipped"]:
        t0 = time.time()
        arrays["fc3_forces"] = _force_set(ph3.supercells_with_displacements, calc)
        rec["fc3_s"] = round(time.time() - t0, 1)
        rec["fc3_supercells"], rec["fc3_atoms"] = len(arrays["fc3_forces"]), arrays["fc3_forces"].shape[1]
    return rec, arrays


def _forces_shard(model, label, out, structures, ids, index, count, device, dtype, backend, threads, timeout):
    os.environ.setdefault("TQDM_DISABLE", "1")
    import torch

    from ..eval.ase_calc import EnvTokCalculator

    torch.set_num_threads(threads)
    calc = EnvTokCalculator(model, device=device, dtype=dtype, backend=backend)
    out = Path(out) / label
    for atoms in read_structures(structures, ids)[index::count]:
        name = atoms.info["material_id"]
        path = out / f"{name}.json"
        # running out of GPU memory depends on how many workers share the GPU, so it is retried
        if path.exists() and json.loads(path.read_text()).get("error") != "oom":
            continue
        rec = {"id": name, "formula": atoms.info.get("name") or atoms.get_chemical_formula()}
        t0 = time.time()
        try:
            with wall_timeout(timeout, name):
                result, arrays = forces_for_structure(atoms, calc)
            rec.update(result)
            np.savez_compressed(out / f"{name}.forces.npz", **arrays)
        except Timeout:
            rec["error"] = "timeout"
        except Exception:
            tb = traceback.format_exc()
            rec["error"] = failure(tb, name)
            rec["detail"] = tb.strip().splitlines()[-1]
        rec["forces_s"] = round(time.time() - t0, 1)
        path.write_text(json.dumps(rec))
        print(f"[{label} {index}/{count}] {name}: forces in {rec['forces_s']} s", flush=True)


# ----------------------------------------------------------------- conductivity
def conductivity(atoms_info: dict, arrays) -> tuple[dict, dict]:
    """The conductivity from saved forces: the total, and per mode for the error against DFT."""
    from ase import Atoms

    relaxed = Atoms(numbers=arrays["numbers"], cell=arrays["cell"], positions=arrays["positions"], pbc=True)
    ph3 = init_phono3py(relaxed, atoms_info)
    ph3.phonon_forces = arrays["fc2_forces"]
    ph3.forces = arrays["fc3_forces"]
    ph3.produce_fc2(symmetrize_fc2=True)
    ph3.produce_fc3(symmetrize_fc3r=True)
    ph3.init_phph_interaction(symmetrize_fc3q=False)
    ph3.run_thermal_conductivity(
        temperatures=(TEMPERATURE_K,),
        is_isotope=True,
        transport_type="SMM19",  # particle-like and wave-like parts: the Wigner transport equation
        boundary_mfp=1e6,
    )
    result = ph3.thermal_conductivity
    mode = _mode_kappa_total(result.mode_kappa_matrix[0, 0], result.mode_heat_capacities[0])
    return (
        {
            "kappa_W_per_mK": float(result.kappa[0, 0, :3].mean()),
            "kappa_p_W_per_mK": float(result.kappa_intra[0, 0, :3].mean()),
            "kappa_c_W_per_mK": float(result.kappa_inter[0, 0, :3].mean()),
        },
        {
            "mode_kappa": mode[..., :3].mean(axis=-1),
            "weights": np.asarray(result.grid_weights),
            "q_points": np.asarray(result.qpoints),
            "frequencies": np.asarray(result.frequencies),
        },
    )


def _mode_kappa_total(matrix: np.ndarray, heat_capacity: np.ndarray) -> np.ndarray:
    """Conductivity per mode, with Matbench Discovery's rule: the diagonal of phono3py's matrix
    (the particle-like part), plus the coherence between each pair of modes, shared by heat
    capacity. Shapes (q, band, band, 6) and (q, band); returns (q, band, 6)."""
    bands = np.arange(matrix.shape[1])
    particle = matrix[:, bands, bands]
    coherence = matrix.copy()
    coherence[:, bands, bands] = 0
    c_a, c_b = heat_capacity[:, :, None, None], heat_capacity[:, None, :, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        per_mode = 2 * (coherence * c_a / (c_a + c_b)).sum(axis=1)
    per_mode[np.isnan(per_mode)] = 0
    return per_mode + particle


def _conductivity_shard(label, out, structures, ids, index, count, threads, timeout):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    out = Path(out) / label
    for atoms in read_structures(structures, ids)[index::count]:
        name = atoms.info["material_id"]
        path = out / f"{name}.json"
        if not path.exists():
            continue
        rec = json.loads(path.read_text())
        if (
            rec.get("error")
            or rec.get("conductivity_skipped")
            or "kappa_W_per_mK" in rec
            or "kappa_error" in rec
        ):
            continue
        t0 = time.time()
        try:
            with wall_timeout(timeout, name), np.load(out / f"{name}.forces.npz") as arrays:
                result, modes = conductivity(atoms.info, arrays)
            rec.update(result)
            np.savez_compressed(out / f"{name}.modes.npz", **modes)
        except Timeout:
            rec["kappa_error"] = "timeout"
        except Exception:
            rec["kappa_error"] = traceback.format_exc().strip().splitlines()[-1]
        rec["conductivity_s"] = round(time.time() - t0, 1)
        rec["conductivity_threads"] = threads
        path.write_text(json.dumps(rec))
        print(f"[{label} {index}/{count}] {name}: conductivity in {rec['conductivity_s']} s", flush=True)


def run_stage(args: argparse.Namespace) -> None:
    rank, world, device = placement(args.device)
    labels = args.labels or [Path(m).stem for m in args.models]
    k = args.workers
    if args.stage == "forces":
        for model, label in zip(args.models, labels):
            (Path(args.out) / label).mkdir(parents=True, exist_ok=True)
            jobs = [
                (
                    model,
                    label,
                    args.out,
                    args.structures,
                    args.ids,
                    rank * k + i,
                    world * k,
                    device,
                    args.dtype,
                    args.backend,
                    args.threads,
                    args.timeout,
                )
                for i in range(k)
            ]
            spawn(_forces_shard, jobs)
    else:
        for label in labels:
            jobs = [
                (
                    label,
                    args.out,
                    args.structures,
                    args.ids,
                    rank * k + i,
                    world * k,
                    args.threads,
                    args.timeout,
                )
                for i in range(k)
            ]
            spawn(_conductivity_shard, jobs)


# ---------------------------------------------------------------------- summary
def score(kappa: float | None, mode_kappa, weights, censored: bool, ref_kappa: float, ref_mode) -> dict:
    """Symmetric relative difference and error of the total, and the mode-resolved error
    (SRME), for one material. A missing conductivity counts as the largest error, and
    ``censored`` (imaginary modes, or a broken symmetry) sets SRME to it."""
    if kappa is None or not np.isfinite(kappa) or kappa < 0:
        srd = -MAX_ERROR
    else:
        total = kappa + ref_kappa
        srd = 0.0 if total == 0 else MAX_ERROR * (kappa - ref_kappa) / total
    srme, failed = MAX_ERROR, True
    if not censored and kappa is not None and mode_kappa is not None and np.isfinite(mode_kappa).all():
        if np.shape(mode_kappa) == np.shape(ref_mode) and kappa >= 0 and kappa + ref_kappa > 0:
            error = np.abs(np.asarray(mode_kappa) - np.asarray(ref_mode)).sum() / np.sum(weights)
            srme, failed = float(MAX_ERROR * error / (kappa + ref_kappa)), False
    return {"srd": float(srd), "sre": float(abs(srd)), "srme": srme, "censored": failed}


def read_reference(path: str) -> dict[str, dict]:
    """Conductivities and frequencies per material, from Matbench Discovery's file."""
    with gzip.open(path, "rt") as fh:
        rows = json.load(fh)
    out = {}
    for r in rows:
        out[r["material_id"]] = {
            "formula": r["name"],
            "kappa": float(np.ravel(r["kappa_tot_avg"])[0]),
            "mode_kappa": np.asarray(r["mode_kappa_tot_avg"][0], dtype=float),
            **{f"dft_{k}": v for k, v in harmonic(r["ph_freqs"], r["weights"]).items()},
        }
    return out


HARMONIC = ("omega_max_THz", "cv_J_per_K_mol", "f_kJ_per_mol", "s_J_per_K_mol")


def tables_for(run_dir: Path, label: str, reference: dict[str, dict]) -> list[dict]:
    rows = []
    for name, ref in reference.items():
        path = run_dir / label / f"{name}.json"
        rec = json.loads(path.read_text()) if path.exists() else {"error": "missing"}
        modes_path = run_dir / label / f"{name}.modes.npz"
        mode, weights = None, None
        if modes_path.exists():
            with np.load(modes_path) as m:
                mode, weights = m["mode_kappa"], m["weights"]
        censored = bool(rec.get("imaginary") or rec.get("broken_symmetry"))
        s = score(rec.get("kappa_W_per_mK"), mode, weights, censored, ref["kappa"], ref["mode_kappa"])
        row = {"model": label, "id": name, "formula": ref["formula"], "kappa_dft": ref["kappa"], **s}
        row.update({k: rec.get(k) for k in ("kappa_W_per_mK", "imaginary", "broken_symmetry", "relax_steps")})
        row.update({k: rec.get(k) for k in ("forces_s", "conductivity_s", "error", "kappa_error")})
        for key in HARMONIC:
            row[key], row[f"dft_{key}"] = rec.get(key), ref[f"dft_{key}"]
        rows.append(row)
    return rows


def aggregate(rows: list[dict]) -> dict:
    out = {}
    for label in dict.fromkeys(r["model"] for r in rows):
        mine = [r for r in rows if r["model"] == label]
        stable = [r for r in mine if r["omega_max_THz"] is not None and not r["imaginary"]]
        s = {
            "n_structures": len(mine),
            "kappa_srme": float(np.mean([r["srme"] for r in mine])),
            "kappa_sre": float(np.mean([r["sre"] for r in mine])),
            "kappa_srd": float(np.mean([r["srd"] for r in mine])),
            "kappa_failure_rate": float(np.mean([r["censored"] for r in mine])),
            "imaginary_rate": float(np.mean([bool(r["imaginary"]) for r in mine])),
            "broken_symmetry": int(sum(bool(r["broken_symmetry"]) for r in mine)),
            "n_failed": int(sum(bool(r["error"] or r["kappa_error"]) for r in mine)),
            "n_stable": len(stable),
        }
        for key in HARMONIC:
            s[f"{key}_mae"] = (
                float(np.mean([abs(r[key] - r[f"dft_{key}"]) for r in stable])) if stable else None
            )
        out[label] = s
    return out


def write_tables(run_dir: Path, labels: list[str], reference_path: str, out: Path) -> None:
    reference = read_reference(reference_path)
    rows = [row for label in labels for row in tables_for(run_dir, label, reference)]
    out.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "kappa_structures.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    summary = {"labels": labels, "models": aggregate(rows)}
    (out / "kappa_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for label, s in summary["models"].items():
        print(
            f"{label}: SRME {s['kappa_srme']:.4f}, SRE {s['kappa_sre']:.4f}, "
            f"failure rate {s['kappa_failure_rate']:.3f}, imaginary {s['imaginary_rate']:.3f}",
            flush=True,
        )


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--stage", default="forces", choices=["forces", "conductivity"])
    p.add_argument("--summary", action="store_true", help="score finished runs instead of running")
    p.add_argument("--models", nargs="+", default=[])
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--out", required=True, help="run directory (one folder per label)")
    p.add_argument("--structures", help="Matbench Discovery's phononDB-PBE extxyz")
    p.add_argument("--reference", help="--summary: Matbench Discovery's phononDB-PBE kappas (json.gz)")
    p.add_argument("--tables", default="results/kappa", help="--summary: where to write the tables")
    p.add_argument("--ids", nargs="+", default=None, help="only these material ids (a pilot)")
    p.add_argument("--timeout", type=float, default=14400.0, help="seconds per structure and stage")
    p.add_argument(
        "--workers", type=int, default=1, help="processes per rank, each a share of the structures"
    )
    p.add_argument("--threads", type=int, default=1, help="threads per worker")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float64")
    p.add_argument("--backend", default="e3nn", choices=["e3nn", "cueq"])


def run(args: argparse.Namespace) -> None:
    if args.summary:
        write_tables(Path(args.out), args.labels, args.reference, Path(args.tables))
    else:
        run_stage(args)
