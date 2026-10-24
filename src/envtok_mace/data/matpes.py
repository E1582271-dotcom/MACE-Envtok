"""Build the MatPES test set from the published JSONL release.

MatPES PBE 2025.2 ships as one JSON record per line (433,189 structures). The
paper's test set is what is left after dropping every structure that contains an
element outside the model's table, which is 323,096 structures / 3,011,678 atoms
-- the count the authors' benchmark asserts. Passing ``--sids`` (the sid column
of their published per-structure results) selects exactly their set instead of
relying on the filter reproducing it.

Units and conventions in the release are undocumented. Energies are eV and
forces eV/A, as in the rest of the paper's data. The stress is VASP's kbar with
the sign opposite to dE/de, in pymatgen's Voigt order (xx, yy, zz, yz, xz, xy)
rather than the order of VASP's own "in kB" line (XX YY ZZ XY YZ ZX). That is
measured, not assumed: ``envtok-mace matpes --calibrate-stress`` scores a trained
model against all four combinations, and on a sample of the release the one used
here gives 6.4 meV/A^3 against 20.0 for the other order and 44-52 with the sign
flipped.
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

from .constants import Z_TABLE
from .frames import Frames, _ptr

SOURCE = "matpes-pbe-2025.2"

# VASP prints the stress in kbar; 1 kbar = 0.1 GPa and 1 eV/A^3 = 160.21766208 GPa.
KBAR_TO_EV_PER_A3 = 0.1 / 160.21766208

# Permutation from the release's 6-vector to ASE's Voigt order (xx, yy, zz, yz, xz, xy).
STRESS_ORDERS: dict[str, tuple[int, ...]] = {
    "voigt": (0, 1, 2, 3, 4, 5),  # pymatgen's Voigt order: already xx yy zz yz xz xy
    "vasp": (0, 1, 2, 4, 5, 3),  # the order of VASP's "in kB" line: XX YY ZZ XY YZ ZX
}

# The paper's boundary check on the test set.
EXPECTED = {"structures": 323_096, "atoms": 3_011_678}

_SYMBOL_Z: dict[str, int] | None = None


def _symbol_to_z() -> dict[str, int]:
    global _SYMBOL_Z
    if _SYMBOL_Z is None:
        from ase.data import atomic_numbers

        _SYMBOL_Z = dict(atomic_numbers)
    return _SYMBOL_Z


class _Filter:
    """Which structures to keep, and how to read their stress."""

    def __init__(
        self,
        allowed_z: frozenset[int],
        keep_sids: frozenset[str] | None,
        stress_sign: float,
        stress_order: str,
    ):
        self.allowed_z = allowed_z
        self.keep_sids = keep_sids
        self.stress_scale = stress_sign * KBAR_TO_EV_PER_A3
        self.perm = np.array(STRESS_ORDERS[stress_order], dtype=np.int64)


_FILTER: _Filter | None = None


def _init_worker(flt: _Filter) -> None:
    global _FILTER
    _FILTER = flt


def _parse_record(line: str, flt: _Filter) -> dict | None:
    rec = json.loads(line)
    sid = rec["matpes_id"]
    if flt.keep_sids is not None and sid not in flt.keep_sids:
        return None
    sites = rec["structure"]["sites"]
    z = np.empty(len(sites), dtype=np.uint8)
    pos = np.empty((len(sites), 3), dtype=np.float64)
    table = _symbol_to_z()
    for i, site in enumerate(sites):
        species = site["species"]
        if len(species) != 1 or abs(species[0].get("occu", 1.0) - 1.0) > 1e-12:
            raise ValueError(f"{sid}: partial occupancy is not supported")
        z[i] = table[species[0]["element"]]
        pos[i] = site["xyz"]
    if not flt.allowed_z.issuperset(z.tolist()):
        return None
    stress = flt.stress_scale * np.asarray(rec["stress"], dtype=np.float64)[flt.perm]
    return {
        "sid": sid,
        "numbers": z,
        "positions": pos,
        "forces": np.asarray(rec["forces"], dtype=np.float64),
        "cell": np.asarray(rec["structure"]["lattice"]["matrix"], dtype=np.float64),
        "pbc": np.asarray(rec["structure"]["lattice"].get("pbc", (True, True, True)), dtype=bool),
        "energy": float(rec["energy"]),
        "stress": stress,
    }


def _parse_chunk(lines: list[str]) -> list[dict]:
    out = [_parse_record(line, _FILTER) for line in lines]
    return [r for r in out if r is not None]


def _open(path: str | Path):
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def _chunks(path: str | Path, size: int):
    with _open(path) as fh:
        batch: list[str] = []
        for line in fh:
            if not line.strip():
                continue
            batch.append(line)
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch


def read_matpes(
    path: str | Path,
    keep_sids: list[str] | None = None,
    allowed_z: frozenset[int] = frozenset(Z_TABLE),
    stress_sign: float = -1.0,
    stress_order: str = "voigt",
    workers: int = 1,
    chunk: int = 2000,
) -> Frames:
    """Parse the MatPES JSONL into Frames, in file order, without neighbour lists."""
    t0 = time.time()
    flt = _Filter(
        allowed_z, frozenset(keep_sids) if keep_sids is not None else None, stress_sign, stress_order
    )
    kept: list[dict] = []
    if workers > 1:
        ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
        with ctx.Pool(workers, initializer=_init_worker, initargs=(flt,)) as pool:
            for i, part in enumerate(pool.imap(_parse_chunk, _chunks(path, chunk)), 1):
                kept.extend(part)
                if i % 20 == 0:
                    print(
                        f"  {i * chunk:,} records read, {len(kept):,} kept ({time.time() - t0:.0f} s)",
                        flush=True,
                    )
    else:
        _init_worker(flt)
        for part in _chunks(path, chunk):
            kept.extend(_parse_chunk(part))

    if not kept:
        raise ValueError(f"no structures kept from {path}")
    if keep_sids is not None:  # the caller's order wins, and every sid must be there
        found = {r["sid"]: r for r in kept}
        missing = [s for s in keep_sids if s not in found]
        if missing:
            raise ValueError(f"{len(missing)} requested sids are not in {path}, e.g. {missing[:3]}")
        kept = [found[s] for s in keep_sids]

    frames = Frames(
        sid=np.array([r["sid"] for r in kept]),
        source=np.array([SOURCE] * len(kept)),
        atom_ptr=_ptr(np.array([len(r["numbers"]) for r in kept], dtype=np.int64)),
        numbers=np.concatenate([r["numbers"] for r in kept]),
        positions=np.concatenate([r["positions"] for r in kept]),
        forces=np.concatenate([r["forces"] for r in kept]),
        cell=np.stack([r["cell"] for r in kept]),
        pbc=np.stack([r["pbc"] for r in kept]),
        energy=np.array([r["energy"] for r in kept]),
        stress=np.stack([r["stress"] for r in kept]),
        attrs={"source": SOURCE, "stress_sign": float(stress_sign), "stress_order": stress_order},
    )
    frames.validate()
    print(
        f"kept {len(frames):,} structures, {int(frames.atom_ptr[-1]):,} atoms ({time.time() - t0:.0f} s)",
        flush=True,
    )
    return frames


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--src", required=True, help="MatPES-PBE-2025.2.jsonl (optionally .gz)")
    p.add_argument("--dst", required=True, help="output Frames HDF5 (no neighbour lists yet)")
    p.add_argument(
        "--sids", default=None, help="text file of sids to keep, one per line (overrides the element filter)"
    )
    p.add_argument("--stress-sign", type=float, default=-1.0)
    p.add_argument("--stress-order", default="voigt", choices=sorted(STRESS_ORDERS))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--chunk", type=int, default=2000)
    p.add_argument(
        "--expect", action="store_true", help="assert the paper's 323,096 structures / 3,011,678 atoms"
    )


def run(args: argparse.Namespace) -> None:
    sids = None
    if args.sids:
        sids = [s.strip() for s in Path(args.sids).read_text().split() if s.strip()]
    frames = read_matpes(
        args.src,
        sids,
        workers=args.workers,
        chunk=args.chunk,
        stress_sign=args.stress_sign,
        stress_order=args.stress_order,
    )
    if args.expect:
        assert len(frames) == EXPECTED["structures"], f"{len(frames)} structures"
        assert int(frames.atom_ptr[-1]) == EXPECTED["atoms"], f"{int(frames.atom_ptr[-1])} atoms"
        print("boundary check passed", flush=True)
    frames.save(args.dst)
    print(f"wrote {args.dst}", flush=True)
