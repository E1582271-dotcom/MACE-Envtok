"""Evaluate one or more models on a Frames file with the same metrics as training.

Models are either our checkpoints (best.pt / last.pt, EMA weights for last.pt) or
published mace-torch .model files, loaded into our implementation through the
parity-tested bridge, so every model is scored by exactly the same code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..data.frames import Frames
from ..data.samplers import eval_shards
from ..model.backends import load_weights
from ..model.mace import MACE, MACEConfig
from .metrics import evaluate


def load_model(path: str, device: torch.device, dtype: torch.dtype, backend: str | None = None) -> MACE:
    """``backend``: None keeps the backend the model was trained with."""
    path = str(path)
    torch.set_default_dtype(dtype)
    if path.endswith(".model"):  # published mace-torch model (trusted, checksum-verified file)
        from ..model.mace_compat import from_mace

        ref = torch.load(path, map_location="cpu", weights_only=False).to(dtype)
        model = from_mace(ref)  # e3nn backend
        if backend == "cueq":
            from ..model.backends import convert_state_dict

            target = MACE(MACEConfig(**{**model.config.to_dict(), "backend": "cueq"}))
            target.load_state_dict(convert_state_dict(model.state_dict(), model, target))
            model = target
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)
        cfg = MACEConfig.from_dict(state["model_config"])
        trained_backend = cfg.backend
        cfg.backend, cfg.cueq_fallback = trained_backend, None
        model = MACE(cfg)
        weights = state["model"]
        if "ema" in state:  # a full training checkpoint: evaluate the EMA weights
            names = [n for n, p in model.named_parameters() if p.requires_grad]
            weights = dict(weights)
            weights.update(dict(zip(names, state["ema"]["shadow"])))
        load_weights(model, weights)
        if backend is not None and backend != trained_backend:
            from ..model.backends import convert_state_dict

            target = MACE(MACEConfig(**{**cfg.to_dict(), "backend": backend, "cueq_fallback": None}))
            target.load_state_dict(convert_state_dict(model.state_dict(), model, target))
            model = target
    return model.to(device=device, dtype=dtype).eval()


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--models", nargs="+", required=True, help="checkpoints (.pt) or mace-torch files (.model)"
    )
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--frames", required=True, help="Frames HDF5 with neighbour lists")
    p.add_argument("--n", type=int, default=None, help="evaluate a fixed random subset of this size")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-edges", type=int, default=60000, help="edge budget per batch")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32")
    p.add_argument(
        "--backend",
        default=None,
        choices=[None, "e3nn", "cueq"],
        help="evaluate with this backend (default: the one the model was trained with)",
    )
    p.add_argument("--out", default=None, help="write the results as JSON")


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    frames = Frames.load(args.frames)
    idx = np.arange(len(frames))
    if args.n is not None and args.n < len(frames):
        idx = np.sort(np.random.default_rng(0).choice(len(frames), args.n, replace=False))
    batches = eval_shards(idx, frames.nedges, 1, 0, args.batch_size, args.max_edges)
    labels = args.labels or [Path(m).stem for m in args.models]
    results = {}
    for label, path in zip(labels, args.models):
        try:
            model = load_model(path, device, dtype, args.backend)
            metrics = evaluate(model, frames, batches, device, dtype)
            results[label] = {"model": str(path), **metrics}
            print(json.dumps({"label": label, **metrics}), flush=True)
        except Exception as exc:  # one failing model must not lose the others
            results[label] = {"model": str(path), "error": repr(exc)}
            print(json.dumps({"label": label, "error": repr(exc)}), flush=True)
        finally:
            torch.cuda.empty_cache() if device.type == "cuda" else None
        if args.out:  # written after every model, so partial results survive
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(
                json.dumps(
                    {
                        "frames": args.frames,
                        "n_frames": int(len(idx)),
                        "dtype": args.dtype,
                        "results": results,
                    },
                    indent=2,
                )
            )
