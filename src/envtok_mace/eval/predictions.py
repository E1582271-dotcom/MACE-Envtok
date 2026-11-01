"""Per-structure predictions next to their DFT labels, for parity plots and error breakdowns.

``metrics.evaluate`` keeps only running sums. A ``Collector`` passed to it as
``collect`` keeps what the model predicted for every structure, and ``save``
writes that, in the frames' order, to one HDF5 file per model:

- per structure: sid, source, natoms, energy_dft / energy_pred (eV),
  stress_dft / stress_pred (eV/A^3, ASE Voigt order);
- per atom: atom_ptr, numbers, forces_dft / forces_pred (eV/A).

Forces and stresses are stored in the evaluation dtype (the DFT labels cast the
way the model saw them), so errors recomputed from the file are the ones the
metrics summed. Energies are float64; the DFT energy is the uncast label.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np

from ..data.batch import full_to_voigt
from ..data.frames import Frames, _ptr, _ranges

FORMAT_VERSION = 1
PER_STRUCTURE = ("sid", "source", "natoms", "energy_dft", "energy_pred", "stress_dft", "stress_pred")
PER_ATOM = ("numbers", "forces_dft", "forces_pred")


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Collector:
    """Keeps the model output of every batch it is called with."""

    def __init__(self):
        self._idx, self._energy, self._forces, self._stress = [], [], [], []

    def __call__(self, idx, out, batch) -> None:
        self._idx.append(np.asarray(idx, dtype=np.int64))
        self._energy.append(out["energy"].detach().double().cpu().numpy())
        self._forces.append(out["forces"].detach().cpu().numpy())
        self._stress.append(full_to_voigt(out["stress"].detach()).cpu().numpy())

    def arrays(self, frames: Frames) -> dict[str, np.ndarray]:
        """Everything collected, sorted by frame index, next to the frames' labels."""
        idx = np.concatenate(self._idx)
        if len(np.unique(idx)) != len(idx):
            raise ValueError("a frame was evaluated more than once")
        order = np.argsort(idx, kind="stable")
        ptr = _ptr(frames.natoms[idx])  # atom offsets in evaluation order
        forces = np.concatenate(self._forces)[_ranges(ptr[order], ptr[order + 1])]
        stress = np.concatenate(self._stress)[order]
        sub = frames.subset(idx[order])
        return {
            "index": idx[order],
            "sid": sub.sid.astype(str),
            "source": sub.source.astype(str),
            "natoms": sub.natoms.astype(np.int64),
            "atom_ptr": sub.atom_ptr,
            "numbers": sub.numbers,
            "energy_dft": sub.energy.astype(np.float64),
            "energy_pred": np.concatenate(self._energy)[order],
            "forces_dft": sub.forces.astype(forces.dtype),
            "forces_pred": forces,
            "stress_dft": sub.stress.astype(stress.dtype),
            "stress_pred": stress,
        }

    def save(self, path: str | Path, frames: Frames, **attrs) -> Path:
        return save(path, self.arrays(frames), **attrs)


def save(path: str | Path, arrays: dict[str, np.ndarray], **attrs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    str_dt = h5py.string_dtype("utf-8")
    with h5py.File(tmp, "w") as h:
        h.attrs["format_version"] = FORMAT_VERSION
        for k, v in attrs.items():
            h.attrs[k] = "" if v is None else v
        for k, v in arrays.items():
            if k in ("sid", "source"):
                h.create_dataset(k, data=v.astype(object), dtype=str_dt)
            else:
                h.create_dataset(k, data=v)
    tmp.replace(path)
    return path


def load(path: str | Path) -> dict:
    """Arrays of a predictions file, plus its attributes under ``"attrs"``."""
    with h5py.File(path, "r") as h:
        if int(h.attrs.get("format_version", -1)) != FORMAT_VERSION:
            raise ValueError(f"{path}: unsupported predictions format")
        out = {k: (h[k].asstr()[()] if k in ("sid", "source") else h[k][()]) for k in h}
        out["attrs"] = {k: v for k, v in h.attrs.items() if k != "format_version"}
    return out
