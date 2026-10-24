"""Evaluate a reference mace-torch model on a Frames file.

Used for the data handshake (does a published model reproduce sensible errors on
our stored energies, forces and stresses?) and later to check that our
evaluation code reproduces the authors' published numbers.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from ..data.frames import Frames


def load_mace_model(path: str, device: str = "cpu", dtype: torch.dtype = torch.float64):
    """Load a published mace-torch model file (a pickled module; only for trusted,
    checksum-verified files such as the authors' Zenodo release)."""
    model = torch.load(path, map_location=device, weights_only=False)
    return model.to(dtype).eval()


def predict_with_mace(
    model_path: str,
    frames: Frames,
    indices: np.ndarray,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    from ..data.batch import collate
    from ..model.mace_compat import mace_input_dict

    torch.set_default_dtype(dtype)
    ref = load_mace_model(model_path, device, dtype)
    energy, forces, stress = [], [], []
    for start in range(0, len(indices), batch_size):
        batch = collate(frames, indices[start : start + batch_size]).to(device=device, dtype=dtype)
        out = ref(mace_input_dict(batch, ref), compute_force=True, compute_stress=True)
        energy.append(out["energy"].detach().cpu().numpy())
        forces.append(out["forces"].detach().cpu().numpy())
        s = out["stress"].detach().cpu().numpy()
        stress.append(s[:, [0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]])  # 3x3 -> ASE Voigt
    return dict(energy=np.concatenate(energy), forces=np.concatenate(forces), stress=np.concatenate(stress))


def error_summary(frames: Frames, indices: np.ndarray, pred: dict[str, np.ndarray]) -> dict[str, float]:
    sub = frames.subset(indices)
    n = sub.natoms
    e_err = (pred["energy"] - sub.energy) / n
    f_err = pred["forces"] - sub.forces
    s_err = pred["stress"] - sub.stress
    return {
        "n_frames": int(len(indices)),
        "energy_mae_meV_per_atom": float(np.abs(e_err).mean() * 1e3),
        "force_mae_meV_per_A": float(np.abs(f_err).mean() * 1e3),
        "stress_mae_meV_per_A3": float(np.abs(s_err).mean() * 1e3),
        # Same sign convention gives a strongly positive correlation.
        "stress_sign_corr": float(np.corrcoef(pred["stress"].ravel(), sub.stress.ravel())[0, 1]),
    }


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="mace-torch .model file")
    p.add_argument("--frames", required=True, help="Frames HDF5 file with neighbour lists")
    p.add_argument("--n", type=int, default=500, help="number of frames, sampled with a fixed seed")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)


def run(args: argparse.Namespace) -> None:
    frames = Frames.load(args.frames)
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(frames), size=min(args.n, len(frames)), replace=False))
    t0 = time.time()
    pred = predict_with_mace(args.model, frames, idx, device=args.device)
    summary = error_summary(frames, idx, pred)
    summary["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(summary, indent=2))
