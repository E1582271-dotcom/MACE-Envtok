"""On a GPU: same weights through the e3nn and the cuEquivariance backends."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from ..data.batch import collate
from ..data.frames import Frames
from ..model.backends import convert_state_dict
from ..model.mace import MACE, MACEConfig


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--frames", required=True)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--device", default="cuda")


def run(args: argparse.Namespace) -> None:
    frames = Frames.load(args.frames)
    idx = np.random.default_rng(0).choice(len(frames), args.n, replace=False)
    for dtype in (torch.float32, torch.float64):
        torch.set_default_dtype(dtype)
        torch.manual_seed(0)
        ref = MACE(MACEConfig(backend="e3nn"))
        cueq = MACE(MACEConfig(backend="cueq", cueq_fallback=False))  # kernels or error
        cueq.load_state_dict(convert_state_dict(ref.state_dict(), ref, cueq))
        ref, cueq = ref.to(args.device), cueq.to(args.device)
        batch = collate(frames, np.sort(idx)).to(device=args.device, dtype=dtype)
        a = ref(batch, compute_stress=True)
        b = cueq(batch, compute_stress=True)
        row = {"dtype": str(dtype)}
        for k in ("energy", "forces", "stress"):
            d = (a[k] - b[k]).abs().max().item()
            row[f"{k}_max_abs_diff"] = d
            row[f"{k}_rel"] = d / max(a[k].abs().max().item(), 1e-30)
        print(json.dumps(row), flush=True)
