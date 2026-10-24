"""Write a checkpoint as a portable weights file.

Training on GPUs uses the cuEquivariance backend. The released weights are
converted to the e3nn backend, which needs only the base dependencies and runs on
CPU; ``load_model(path, backend="cueq")`` converts them back for the GPU kernels.
The file holds the model configuration, the weights the reported results were
computed with (for a training checkpoint, its EMA weights), the epoch, the
validation metrics stored at save time, and provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .run_eval import load_model


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def export_weights(src: str, dst: str, backend: str = "e3nn", meta: dict | None = None) -> dict:
    state = torch.load(src, map_location="cpu", weights_only=False)
    # Built in float64 so the constants derived from the irreps (the coupling
    # matrices) are exact at either precision; the learned parameters are stored
    # in the float32 they were trained in, which float64 represents exactly.
    model = load_model(src, torch.device("cpu"), torch.float64, backend)
    params = {name for name, _ in model.named_parameters()}
    weights = {
        k: (v.detach().float() if k in params else v.detach()).clone() for k, v in model.state_dict().items()
    }
    out = {
        "model_config": model.config.to_dict(),
        "model": weights,
        "epoch": state.get("epoch"),
        "val": state.get("val"),
        "meta": {"source_sha256": sha256(src), "backend": backend, **(meta or {})},
    }
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    return {
        "file": str(dst),
        "sha256": sha256(dst),
        "bytes": Path(dst).stat().st_size,
        "epoch": out["epoch"],
        "meta": out["meta"],
    }


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--src", required=True, help="training checkpoint (best.pt / last.pt)")
    p.add_argument("--dst", required=True)
    p.add_argument("--backend", default="e3nn", choices=["e3nn", "cueq"])
    p.add_argument("--meta", default=None, help="JSON object of provenance fields to store")


def run(args: argparse.Namespace) -> None:
    meta = json.loads(args.meta) if args.meta else None
    print(json.dumps(export_weights(args.src, args.dst, args.backend, meta), indent=1))
