"""Flat, array-based storage for many atomic structures ("frames").

All per-atom arrays are concatenated; ``atom_ptr[i]:atom_ptr[i+1]`` selects the
atoms of frame ``i``. The same layout is used on disk (one HDF5 file) and in
memory, so a whole training set loads with a handful of array reads.

Optional neighbour lists use the same idea: ``edge_ptr`` slices ``edge_index``
(local atom indices, sender row then receiver row) and ``unit_shifts`` (integer
cell shifts). Integer shifts are stored instead of Cartesian ones because
straining the cell for the stress calculation changes the Cartesian shifts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import h5py
import numpy as np

FORMAT_VERSION = 1

# Stress is stored as 6-component Voigt in ASE order: xx, yy, zz, yz, xz, xy (eV/A^3).
VOIGT_ORDER = ("xx", "yy", "zz", "yz", "xz", "xy")


@dataclass
class Frames:
    sid: np.ndarray  # (N,) str
    source: np.ndarray  # (N,) str, OMat24 sub-dataset or other origin
    atom_ptr: np.ndarray  # (N+1,) int64
    numbers: np.ndarray  # (A,) uint8
    positions: np.ndarray  # (A, 3) float64, Angstrom
    forces: np.ndarray  # (A, 3) float64, eV/Angstrom
    cell: np.ndarray  # (N, 3, 3) float64, rows are lattice vectors
    pbc: np.ndarray  # (N, 3) bool
    energy: np.ndarray  # (N,) float64, eV
    stress: np.ndarray  # (N, 6) float64, eV/Angstrom^3, ASE Voigt order
    edge_ptr: np.ndarray | None = None  # (N+1,) int64
    edge_index: np.ndarray | None = None  # (2, E) int32, local indices
    unit_shifts: np.ndarray | None = None  # (E, 3) int8
    attrs: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return len(self.sid)

    @property
    def natoms(self) -> np.ndarray:
        return np.diff(self.atom_ptr)

    @property
    def nedges(self) -> np.ndarray:
        if self.edge_ptr is None:
            raise ValueError("frames have no neighbour lists")
        return np.diff(self.edge_ptr)

    @property
    def has_edges(self) -> bool:
        return self.edge_ptr is not None

    def validate(self) -> None:
        n, a = len(self.sid), int(self.atom_ptr[-1])
        assert self.atom_ptr[0] == 0 and np.all(np.diff(self.atom_ptr) > 0)
        assert self.source.shape == (n,)
        assert self.numbers.shape == (a,)
        assert self.positions.shape == (a, 3) and self.forces.shape == (a, 3)
        assert self.cell.shape == (n, 3, 3) and self.pbc.shape == (n, 3)
        assert self.energy.shape == (n,) and self.stress.shape == (n, 6)
        if self.has_edges:
            e = int(self.edge_ptr[-1])
            assert self.edge_ptr.shape == (n + 1,) and self.edge_ptr[0] == 0
            assert self.edge_index.shape == (2, e) and self.unit_shifts.shape == (e, 3)

    # ---------------------------------------------------------------- indexing
    def subset(self, idx: np.ndarray) -> Frames:
        """Return the frames at positions ``idx`` (in that order) as a new object."""
        idx = np.asarray(idx, dtype=np.int64)
        a0, a1 = self.atom_ptr[idx], self.atom_ptr[idx + 1]
        atom_sel = _ranges(a0, a1)
        out = dict(
            sid=self.sid[idx],
            source=self.source[idx],
            atom_ptr=_ptr(a1 - a0),
            numbers=self.numbers[atom_sel],
            positions=self.positions[atom_sel],
            forces=self.forces[atom_sel],
            cell=self.cell[idx],
            pbc=self.pbc[idx],
            energy=self.energy[idx],
            stress=self.stress[idx],
            attrs=dict(self.attrs),
        )
        if self.has_edges:
            e0, e1 = self.edge_ptr[idx], self.edge_ptr[idx + 1]
            edge_sel = _ranges(e0, e1)
            out.update(
                edge_ptr=_ptr(e1 - e0),
                edge_index=self.edge_index[:, edge_sel],
                unit_shifts=self.unit_shifts[edge_sel],
            )
        return Frames(**out)

    @staticmethod
    def concatenate(parts: list[Frames]) -> Frames:
        parts = [p for p in parts if len(p)]
        if not parts:
            raise ValueError("nothing to concatenate")
        with_edges = parts[0].has_edges
        assert all(p.has_edges == with_edges for p in parts)
        out = dict(
            sid=np.concatenate([p.sid for p in parts]),
            source=np.concatenate([p.source for p in parts]),
            atom_ptr=_ptr(np.concatenate([p.natoms for p in parts])),
            numbers=np.concatenate([p.numbers for p in parts]),
            positions=np.concatenate([p.positions for p in parts]),
            forces=np.concatenate([p.forces for p in parts]),
            cell=np.concatenate([p.cell for p in parts]),
            pbc=np.concatenate([p.pbc for p in parts]),
            energy=np.concatenate([p.energy for p in parts]),
            stress=np.concatenate([p.stress for p in parts]),
            attrs=dict(parts[0].attrs),
        )
        if with_edges:
            out.update(
                edge_ptr=_ptr(np.concatenate([p.nedges for p in parts])),
                edge_index=np.concatenate([p.edge_index for p in parts], axis=1),
                unit_shifts=np.concatenate([p.unit_shifts for p in parts]),
            )
        return Frames(**out)

    def get(self, i: int) -> dict:
        """Arrays of frame ``i`` as a dict (views, no copies)."""
        a0, a1 = self.atom_ptr[i], self.atom_ptr[i + 1]
        d = dict(
            sid=self.sid[i],
            numbers=self.numbers[a0:a1],
            positions=self.positions[a0:a1],
            forces=self.forces[a0:a1],
            cell=self.cell[i],
            pbc=self.pbc[i],
            energy=self.energy[i],
            stress=self.stress[i],
        )
        if self.has_edges:
            e0, e1 = self.edge_ptr[i], self.edge_ptr[i + 1]
            d.update(edge_index=self.edge_index[:, e0:e1], unit_shifts=self.unit_shifts[e0:e1])
        return d

    def to_atoms(self, i: int):
        """Frame ``i`` as an ``ase.Atoms`` with a single-point calculator attached."""
        from ase import Atoms
        from ase.calculators.singlepoint import SinglePointCalculator

        f = self.get(i)
        atoms = Atoms(numbers=f["numbers"], positions=f["positions"], cell=f["cell"], pbc=f["pbc"])
        atoms.info["sid"] = str(f["sid"])
        atoms.calc = SinglePointCalculator(
            atoms, energy=float(f["energy"]), forces=np.array(f["forces"]), stress=np.array(f["stress"])
        )
        return atoms

    # --------------------------------------------------------------------- I/O
    def save(self, path: str | Path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        str_dt = h5py.string_dtype("utf-8")
        with h5py.File(tmp, "w") as h:
            h.attrs["format_version"] = FORMAT_VERSION
            for k, v in self.attrs.items():
                h.attrs[k] = v
            for f in fields(self):
                if f.name == "attrs":
                    continue
                arr = getattr(self, f.name)
                if arr is None:
                    continue
                if f.name in ("sid", "source"):
                    h.create_dataset(f.name, data=arr.astype(object), dtype=str_dt)
                else:
                    h.create_dataset(f.name, data=arr)
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> Frames:
        with h5py.File(path, "r") as h:
            if int(h.attrs.get("format_version", -1)) != FORMAT_VERSION:
                raise ValueError(f"{path}: unsupported format version")
            kw = {}
            for f in fields(cls):
                if f.name == "attrs":
                    continue
                if f.name in h:
                    ds = h[f.name]
                    kw[f.name] = ds.asstr()[()] if f.name in ("sid", "source") else ds[()]
            attrs = {k: v for k, v in h.attrs.items() if k != "format_version"}
        frames = cls(**kw, attrs=attrs)
        frames.validate()
        return frames


def _ptr(counts: np.ndarray) -> np.ndarray:
    ptr = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=ptr[1:])
    return ptr


def _ranges(starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Concatenation of ``arange(s, e)`` for every pair, without a Python loop."""
    lengths = stops - starts
    if len(lengths) == 0:
        return np.zeros(0, dtype=np.int64)
    offsets = np.repeat(starts - _ptr(lengths)[:-1], lengths)
    return np.arange(int(lengths.sum()), dtype=np.int64) + offsets
