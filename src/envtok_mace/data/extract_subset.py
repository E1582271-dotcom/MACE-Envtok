"""Pull structures out of the OMat24 ASE-LMDB files into one Frames HDF5 file.

Training set: pass the authors' ID list (``best_subset_*_topup.txt``); every
row whose ``sid`` is in the list is kept. Validation set: pass no ID list and
every row is kept. In both cases structures containing rare earths (or any
element outside ``Z_TABLE``) are dropped.

The databases are read directly, so the 100-200 GB extxyz conversion of the
authors' stage 00 is not needed. Work is split into (file, row range) chunks
and spread over a process pool; the output is sorted by sid, so it does not
depend on the number of workers.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .constants import RARE_EARTH_Z, Z_TABLE
from .frames import Frames, _ptr

_ALLOWED = np.zeros(256, dtype=bool)
_ALLOWED[list(Z_TABLE)] = True
_RARE = np.zeros(256, dtype=bool)
_RARE[list(RARE_EARTH_Z)] = True

_WANTED: frozenset[str] | None = None


def read_ids(path: str | Path) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def find_databases(root: str | Path) -> list[Path]:
    dbs = sorted(Path(root).rglob("*.aselmdb"))
    if not dbs:
        raise FileNotFoundError(f"no .aselmdb files under {root}")
    return dbs


def _count_rows(path: Path) -> int:
    from ase_db_backends.aselmdb import LMDBDatabase

    with LMDBDatabase(str(path), readonly=True) as db:
        return db.count()


def _init_worker(ids_path: str | None) -> None:
    global _WANTED
    _WANTED = frozenset(read_ids(ids_path)) if ids_path else None


def _scan_chunk(task: tuple[str, str, int, int]) -> tuple[Frames | None, Counter]:
    """Read rows [start, stop) of one database and keep the wanted ones."""
    from ase_db_backends.aselmdb import LMDBDatabase

    path, source, start, stop = task
    stats: Counter = Counter()
    sids, natoms, numbers, positions, forces = [], [], [], [], []
    cells, pbcs, energies, stresses = [], [], [], []
    with LMDBDatabase(path, readonly=True) as db:
        for k in range(start, stop):
            row = db._get_row_by_index(k, include_data=True)
            stats["scanned"] += 1
            sid = row.data["sid"]
            if _WANTED is not None and sid not in _WANTED:
                continue
            z = np.asarray(row.numbers)
            if _RARE[z].any():
                stats["dropped_rare_earth"] += 1
                continue
            if not _ALLOWED[z].all():
                stats["dropped_other_element"] += 1
                continue
            atoms = row.toatoms()
            res = atoms.calc.results
            sids.append(sid)
            natoms.append(len(z))
            numbers.append(z.astype(np.uint8))
            positions.append(atoms.positions)
            forces.append(np.asarray(res["forces"], dtype=np.float64))
            cells.append(np.asarray(atoms.cell[:], dtype=np.float64))
            pbcs.append(np.asarray(atoms.pbc, dtype=bool))
            energies.append(float(res["energy"]))
            stresses.append(_voigt(np.asarray(res["stress"], dtype=np.float64)))
            stats["kept"] += 1
    if not sids:
        return None, stats
    frames = Frames(
        sid=np.array(sids, dtype=object),
        source=np.array([source] * len(sids), dtype=object),
        atom_ptr=_ptr(np.array(natoms)),
        numbers=np.concatenate(numbers),
        positions=np.concatenate(positions),
        forces=np.concatenate(forces),
        cell=np.stack(cells),
        pbc=np.stack(pbcs),
        energy=np.array(energies),
        stress=np.stack(stresses),
    )
    return frames, stats


def _voigt(stress: np.ndarray) -> np.ndarray:
    if stress.shape == (6,):
        return stress
    if stress.shape == (3, 3):
        return stress[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
    raise ValueError(f"unexpected stress shape {stress.shape}")


def extract(
    omat_dir: str | Path,
    out: str | Path,
    ids_path: str | Path | None = None,
    expect: int | None = None,
    workers: int = 1,
    chunk_rows: int = 50_000,
) -> Frames:
    t0 = time.time()
    dbs = find_databases(omat_dir)
    tasks = []
    for db in dbs:
        n = _count_rows(db)
        source = db.parent.name
        tasks += [(str(db), source, s, min(s + chunk_rows, n)) for s in range(0, n, chunk_rows)]
    total_rows = sum(t[3] - t[2] for t in tasks)
    print(f"{len(dbs)} databases, {total_rows:,} rows, {len(tasks)} chunks, {workers} workers", flush=True)

    ids_arg = str(ids_path) if ids_path else None
    parts, stats = [], Counter()
    if workers > 1:
        ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
        with ctx.Pool(workers, initializer=_init_worker, initargs=(ids_arg,)) as pool:
            for i, (frames, st) in enumerate(pool.imap_unordered(_scan_chunk, tasks), 1):
                stats.update(st)
                if frames is not None:
                    parts.append(frames)
                if i % max(1, len(tasks) // 50) == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)} chunks, scanned {stats['scanned']:,}, "
                          f"kept {stats['kept']:,} ({time.time() - t0:.0f} s)", flush=True)
    else:
        _init_worker(ids_arg)
        for frames, st in map(_scan_chunk, tasks):
            stats.update(st)
            if frames is not None:
                parts.append(frames)

    frames = Frames.concatenate(parts)
    order = np.argsort(frames.sid.astype(str), kind="stable")
    frames = frames.subset(order)

    # Boundary checks: one frame per sid, and (for an ID list) every id found.
    n_unique = len(np.unique(frames.sid))
    if n_unique != len(frames):
        raise RuntimeError(f"{len(frames) - n_unique} duplicate sids in the output")
    if ids_path:
        wanted = set(read_ids(ids_path))
        missing = wanted - set(frames.sid.tolist())
        print(f"ids matched: {len(wanted) - len(missing):,} / {len(wanted):,}", flush=True)
        if missing:
            example = sorted(missing)[:5]
            raise RuntimeError(f"{len(missing):,} requested ids not found, e.g. {example}")
    if expect is not None and len(frames) != expect:
        raise RuntimeError(f"expected {expect:,} frames, got {len(frames):,}")

    frames.attrs.update(
        omat_dir=str(omat_dir),
        ids_file=str(ids_path or ""),
        rows_scanned=int(stats["scanned"]),
        dropped_rare_earth=int(stats["dropped_rare_earth"]),
        dropped_other_element=int(stats["dropped_other_element"]),
    )
    frames.save(out)
    print(f"wrote {len(frames):,} frames, {int(frames.atom_ptr[-1]):,} atoms -> {out}  "
          f"(dropped {stats['dropped_rare_earth']:,} rare-earth, "
          f"{stats['dropped_other_element']:,} other; {time.time() - t0:.0f} s)", flush=True)
    return frames


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--omat-dir", required=True, help="directory searched recursively for *.aselmdb")
    p.add_argument("--out", required=True, help="output Frames HDF5 file")
    p.add_argument("--ids", default=None, help="text file of sids to keep (default: keep all)")
    p.add_argument("--expect", type=int, default=None, help="fail unless exactly this many frames are written")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--chunk-rows", type=int, default=50_000)


def run(args: argparse.Namespace) -> None:
    extract(args.omat_dir, args.out, args.ids, args.expect, args.workers, args.chunk_rows)
