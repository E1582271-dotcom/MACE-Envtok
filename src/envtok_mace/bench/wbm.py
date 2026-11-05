"""Relaxation of WBM structures against their DFT-relaxed geometry (the RMSD of
Matbench Discovery's geometry-optimisation task).

Each structure is relaxed from its initial WBM geometry with FIRE on a
FrechetCellFilter, until every force is below 0.05 eV/A or for at most 500 steps:
Matbench Discovery's settings. The result is compared with the DFT-relaxed structure
by pymatgen's StructureMatcher with stol 1 and no volume scaling, as upstream does. The
RMSD is normalised by (V/atom)^(1/3), and a pair that does not match counts as 1 in
the mean.

The set is 10,000 WBM structures, drawn with a fixed seed from those whose elements
are all in the model's table, so every model sees the same structures. They are
written once from Matbench Discovery's two extxyz archives:

    envtok-mace wbm --prepare DIR --initial initial-atoms.extxyz.zip --relaxed relaxed-atoms.extxyz.zip
    envtok-mace wbm --models a.pt b.model --labels ours authors --structures DIR --out runs/wbm
    envtok-mace wbm --summary --labels ours authors --structures DIR --out runs/wbm --tables results/wbm
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np

from ..data.constants import Z_TABLE
from ._runner import Timeout, failure, placement, spawn, wall_timeout

N_STRUCTURES = 10_000
SEED = 0
FMAX = 0.05  # eV/A
MAX_STEPS = 500
STOL = 1.0  # StructureMatcher site tolerance; also the RMSD given to a pair that does not match
PER_MODEL = ("rmsd", "steps", "error")  # columns of each model in wbm_structures.csv


# ------------------------------------------------------------------ structures
def _symbols(text: str) -> list[str]:
    """Element symbols of one extxyz frame whose first column is the species."""
    lines = text.splitlines()
    return [line.split()[0] for line in lines[2 : 2 + int(lines[0])]]


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare(initial_zip: Path, relaxed_zip: Path, out: Path, n: int = N_STRUCTURES, seed: int = SEED) -> None:
    """Draw ``n`` structures with every element in the model's table and write their
    initial and DFT-relaxed geometries, in the same order. The DFT copy often lists the
    atoms in another order, which StructureMatcher does not mind."""
    from ase.data import chemical_symbols
    from ase.io import read, write

    allowed = {chemical_symbols[z] for z in Z_TABLE}
    with zipfile.ZipFile(initial_zip) as zi, zipfile.ZipFile(relaxed_zip) as zr:
        names = sorted(set(zi.namelist()) & set(zr.namelist()))
        candidates = [m for m in names if set(_symbols(zi.read(m).decode())) <= allowed]
        rng = np.random.default_rng(seed)
        chosen = [candidates[i] for i in sorted(rng.choice(len(candidates), size=n, replace=False))]
        frames = {}
        for kind, z in (("initial", zi), ("relaxed", zr)):
            frames[kind] = [read(io.StringIO(z.read(m).decode()), format="extxyz") for m in chosen]
    for a, b in zip(frames["initial"], frames["relaxed"]):
        assert a.info["material_id"] == b.info["material_id"] and sorted(a.numbers) == sorted(b.numbers)
    out.mkdir(parents=True, exist_ok=True)
    for kind, atoms in frames.items():
        write(out / f"{kind}.extxyz", atoms, format="extxyz")
    meta = {
        "n_wbm": len(names),
        "n_candidates": len(candidates),
        "n_structures": n,
        "seed": seed,
        "initial_md5": _md5(initial_zip),
        "relaxed_md5": _md5(relaxed_zip),
    }
    (out / "subset.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{n} of {len(candidates)} WBM structures with every element in the table -> {out}", flush=True)


def read_structures(directory: str, kind: str = "initial", limit: int | None = None) -> list:
    from ase.io import read

    atoms = read(Path(directory) / f"{kind}.extxyz", index=":", format="extxyz")
    return atoms[:limit] if limit else atoms


# ------------------------------------------------------------------ relaxation
def relax(atoms, calc) -> dict:
    """Matbench Discovery's relaxation: FIRE on a FrechetCellFilter."""
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE

    atoms = atoms.copy()
    atoms.calc = calc
    opt = FIRE(FrechetCellFilter(atoms), logfile=None)
    converged = bool(opt.run(fmax=FMAX, steps=MAX_STEPS))
    energy = float(atoms.get_potential_energy())
    if not np.isfinite(energy):
        raise ValueError(f"non-finite energy {energy}")
    return {
        "converged": converged,
        "steps": int(opt.get_number_of_steps()),
        "energy_eV": energy,
        "numbers": atoms.numbers.tolist(),
        "cell": np.asarray(atoms.cell).round(6).tolist(),
        "positions": atoms.positions.round(6).tolist(),
    }


def read_records(directory: Path) -> dict[str, dict]:
    """Every structure's last record in a model's shard files; a rerun's line wins."""
    out = {}
    for path in sorted(directory.glob("shard_*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # a line cut short when a job was stopped
                continue
            out[rec["id"]] = rec
    return out


def _run_shard(model, label, out, structures, limit, index, count, device, dtype, backend, threads, timeout):
    os.environ.setdefault("TQDM_DISABLE", "1")
    import torch

    from ..eval.ase_calc import EnvTokCalculator

    torch.set_num_threads(threads)
    out = Path(out) / label
    done = read_records(out)
    calc = None
    with (out / f"shard_{index:03d}of{count:03d}.jsonl").open("a") as fh:
        for atoms in read_structures(structures, limit=limit)[index::count]:
            sid = atoms.info["material_id"]
            # running out of GPU memory depends on how many workers share the GPU, so it is retried
            if sid in done and done[sid].get("error") != "oom":
                continue
            if calc is None:
                calc = EnvTokCalculator(model, device=device, dtype=dtype, backend=backend)
            t0 = time.time()
            rec = {"id": sid, "formula": atoms.get_chemical_formula(), "natoms": len(atoms)}
            try:
                with wall_timeout(timeout, sid):
                    rec.update(relax(atoms, calc))
            except Timeout:
                rec["error"] = "timeout"
            except Exception:
                tb = traceback.format_exc()
                rec["error"] = failure(tb, sid)
                rec["detail"] = tb.strip().splitlines()[-1]
            rec["seconds"] = round(time.time() - t0, 2)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()


def run_relaxations(args: argparse.Namespace) -> None:
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
                args.structures,
                args.limit,
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
        if k == 1:
            _run_shard(*jobs[0])
        else:
            spawn(_run_shard, jobs)


# ------------------------------------------------------------------ summary
def rmsd(pred, ref) -> tuple[float | None, float | None]:
    """Normalised RMSD and largest distance of two structures, or (None, None) if
    StructureMatcher does not match them."""
    from pymatgen.analysis.structure_matcher import StructureMatcher

    return StructureMatcher(stol=STOL, scale=False).get_rms_dist(pred, ref) or (None, None)


def _score(job: tuple) -> dict:
    from pymatgen.core import Lattice, Structure

    rec, ref = job
    row = {k: rec.get(k) for k in ("id", "formula", "natoms", "converged", "steps", "seconds")}
    row["error"] = rec.get("error", "")
    row["rmsd"] = row["max_dist_A"] = None
    if not row["error"]:
        pred = Structure(Lattice(rec["cell"]), rec["numbers"], rec["positions"], coords_are_cartesian=True)
        row["rmsd"], row["max_dist_A"] = rmsd(pred, ref)
    return row


def score(records: dict[str, dict], references: list, label: str, workers: int = 1) -> list[dict]:
    """One row per structure of the set, in its order; a structure without a record
    counts as failed."""
    from pymatgen.io.ase import AseAtomsAdaptor

    jobs = []
    for atoms in references:
        sid = atoms.info["material_id"]
        rec = records.get(sid) or {"id": sid, "natoms": len(atoms), "error": "missing"}
        jobs.append((rec, AseAtomsAdaptor.get_structure(atoms)))
    if workers > 1:
        from multiprocessing import get_context

        with get_context("spawn").Pool(workers) as pool:
            rows = pool.map(_score, jobs, chunksize=64)
    else:
        rows = [_score(j) for j in jobs]
    return [{"model": label, **r} for r in rows]


def wide(rows: list[dict], references: list) -> list[dict]:
    """One row per structure, with every model's RMSD, steps and error side by side:
    four models in long form would be too large a file to keep in the repository."""
    table = {
        a.info["material_id"]: {
            "id": a.info["material_id"],
            "formula": a.get_chemical_formula(),
            "natoms": len(a),
        }
        for a in references
    }
    for r in rows:
        table[r["id"]].update({f"{r['model']}_{k}": r[k] for k in PER_MODEL})
    return list(table.values())


def aggregate(rows: list[dict]) -> dict:
    """Per model, Matbench Discovery's mean RMSD (a failed or unmatched structure counts
    as ``STOL``), and the distribution over the matched structures."""
    from collections import Counter

    out = {}
    for label in dict.fromkeys(r["model"] for r in rows):
        mine = [r for r in rows if r["model"] == label]
        matched = [r["rmsd"] for r in mine if r["rmsd"] is not None]
        steps = [r["steps"] for r in mine if r["steps"] is not None]
        out[label] = {
            "n_structures": len(mine),
            "n_failed": int(sum(bool(r["error"]) for r in mine)),
            "failures": dict(sorted(Counter(r["error"] for r in mine if r["error"]).items())),
            "n_unconverged": int(sum(r["converged"] is False for r in mine)),
            "n_unmatched": int(sum(not r["error"] and r["rmsd"] is None for r in mine)),
            "rmsd_mean": float(np.mean([STOL if r["rmsd"] is None else r["rmsd"] for r in mine])),
            "rmsd_median_matched": float(np.median(matched)) if matched else None,
            "rmsd_p90_matched": float(np.percentile(matched, 90)) if matched else None,
            "steps_median": float(np.median(steps)) if steps else None,
        }
    return out


def published(path: str, ids: list[str]) -> dict:
    """A model's RMSDs from Matbench Discovery's analysis file, on the same structures."""
    import pandas as pd

    df = pd.read_csv(path, usecols=["material_id", "structure_rmsd_vs_dft"], index_col="material_id")
    values = pd.to_numeric(df["structure_rmsd_vs_dft"], errors="coerce").reindex(ids)
    return {sid: (None if np.isnan(v) else float(v)) for sid, v in values.items()}


def write_tables(args: argparse.Namespace) -> None:
    references = read_structures(args.structures, "relaxed", args.limit)
    ids = [a.info["material_id"] for a in references]
    rows = []
    for label in args.labels:
        rows += score(read_records(Path(args.out) / label), references, label, args.workers)
    summary = {"models": aggregate(rows)}
    for spec in args.published or []:
        label, path = spec.split("=", 1)
        theirs = published(path, ids)
        mine = {r["id"]: r["rmsd"] for r in rows if r["model"] == label}
        pairs = np.array(
            [(mine[i], theirs[i]) for i in ids if mine.get(i) is not None and theirs[i] is not None]
        )
        summary.setdefault("published", {})[label] = {
            "rmsd_mean": float(np.mean([STOL if theirs[i] is None else theirs[i] for i in ids])),
            "n_unmatched": int(sum(theirs[i] is None for i in ids)),
            "n_both_matched": len(pairs),
            "median_abs_difference": float(np.median(np.abs(pairs[:, 0] - pairs[:, 1])))
            if len(pairs)
            else None,
        }
    out = Path(args.tables)
    out.mkdir(parents=True, exist_ok=True)
    table = wide(rows, references)
    with (out / "wbm_structures.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]), lineterminator="\n")
        w.writeheader()
        for r in table:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    (out / "wbm_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for label, s in summary["models"].items():
        print(
            f"{label}: mean RMSD {s['rmsd_mean']:.4f} over {s['n_structures']}, {s['n_failed']} failed, "
            f"{s['n_unmatched']} unmatched, {s['n_unconverged']} unconverged",
            flush=True,
        )
    for label, p in summary.get("published", {}).items():
        print(f"{label} published: mean RMSD {p['rmsd_mean']:.4f} on the same structures", flush=True)


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--prepare", default=None, help="write the structure set to this directory")
    p.add_argument("--initial", default=None, help="--prepare: wbm-initial-atoms.extxyz.zip")
    p.add_argument("--relaxed", default=None, help="--prepare: wbm-relaxed-atoms.extxyz.zip")
    p.add_argument("--models", nargs="+", default=[])
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--structures", default=None, help="directory written by --prepare")
    p.add_argument("--out", default=None, help="run directory (one folder per label)")
    p.add_argument("--summary", action="store_true", help="score finished relaxations instead of running")
    p.add_argument("--tables", default="results/wbm", help="--summary: where to write the tables")
    p.add_argument(
        "--published",
        nargs="+",
        default=None,
        help="--summary: LABEL=analysis.csv.gz, a model's published RMSDs to compare with",
    )
    p.add_argument("--limit", type=int, default=None, help="only the first N structures (a pilot)")
    p.add_argument("--timeout", type=float, default=1800.0, help="seconds per structure")
    p.add_argument(
        "--workers", type=int, default=1, help="processes per rank (or CPU processes for --summary)"
    )
    p.add_argument("--threads", type=int, default=1, help="torch threads per worker")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--backend", default=None, choices=[None, "e3nn", "cueq"])


def run(args: argparse.Namespace) -> None:
    if args.prepare:
        prepare(Path(args.initial), Path(args.relaxed), Path(args.prepare))
    elif args.summary:
        write_tables(args)
    else:
        run_relaxations(args)
